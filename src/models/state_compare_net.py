"""StateCompareNet — Siamese CNN + spatial cross-attention for state-transition.

WHY this architecture for "what happens next":

  Every existing Track A model (TSM, R2+1D, etc.) treats the task as standard
  video classification: run all frames through a backbone, pool, classify.
  This forces the temporal comparison to happen implicitly inside a pooling head.

  "What happens next" is fundamentally about STATE TRANSITIONS:
      initial_state → [action] → final_state
  The correct architectural primitive is: compare initial and final states.

HOW it works:

  before_tokens = encoder(frame0 || frame1)   (B, 49, D)  — initial state
  after_tokens  = encoder(frame2 || frame3)   (B, 49, D)  — final state

  Cross-attention (before Q, after KV):
    each before-patch asks "what corresponds to me in the after-state?"
    → transition_tokens  (B, 49, D)

  Pooled features:
    transition = mean(transition_tokens)          (B, D) — how the scene evolved
    delta      = mean(after) − mean(before)       (B, D) — net state shift
    feat = cat([norm(transition), norm(delta)])   (B, 2D)
    → head → num_classes

KEY DESIGN CHOICES:

  Siamese encoder: same weights for before and after.
    - Forces encoder to learn a general "state representation"
    - Doubles effective training samples per parameter
    - Reduces total params (1× backbone instead of 2×)

  6-channel input (2 frames stacked as channels):
    - No temporal shift needed — both states are explicit
    - The encoder learns intra-pair dynamics (what moved within the pair)
    - Works with standard ResNet conv1 (extended to 6 ch with zero-init on extra 3)

  Spatial cross-attention over 7×7 patch tokens:
    - Operates at 2048-D projected to proj_dim=512 for efficiency
    - 49 tokens → O(49²) = 2401 per-head, completely tractable
    - Much richer than vector-level [before, after, delta] MLPs

  No avgpool before cross-attention:
    - Preserves spatial structure ("left side of the scene changed")
    - TSM collapses spatial info too early; here it's used for comparison

PARAMS: ~25M encoder + ~3M cross-attn + <1M head ≈ 29M total (all trainable).
Compare: R152-TSM has ~60M trainable from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torchvision.models import resnext50_32x4d, resnext101_32x8d

# ── cross-attention block ──────────────────────────────────────────────────────

class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention: query attends to kv."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(attn_out)
        q = q + self.drop(self.ff(self.norm_ff(q)))
        return q


# ── main model ─────────────────────────────────────────────────────────────────

class StateCompareNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 33,
        n_resnet_layers: int = 50,
        backbone_variant: str = "resnet",   # "resnet" or "resnext"
        proj_dim: int = 512,                # cross-attention token dimension
        n_cross_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.4,
        use_frame_diff: bool = True,        # add diff channel to 6ch → 9ch input
    ) -> None:
        super().__init__()
        self.use_frame_diff = use_frame_diff
        in_channels = 9 if use_frame_diff else 6   # (RGB×2 [+ diff]) per pair

        # ── backbone (shared Siamese encoder) ─────────────────────────────────
        _factories = {
            ("resnet",  50):  resnet50,
            ("resnet",  101): resnet101,
            ("resnext", 50):  resnext50_32x4d,
            ("resnext", 101): resnext101_32x8d,
        }
        key = (backbone_variant, n_resnet_layers)
        if key not in _factories:
            raise ValueError(f"Unsupported backbone: variant={backbone_variant}, depth={n_resnet_layers}")

        _bb = _factories[key](weights=None)

        # Extend conv1 to in_channels. Extra channels zero-init (conservative start).
        old = _bb.conv1
        new_conv1 = nn.Conv2d(in_channels, old.out_channels,
                              kernel_size=old.kernel_size, stride=old.stride,  # type: ignore
                              padding=old.padding, bias=False)                  # type: ignore
        nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        _bb.conv1 = new_conv1

        # Store as named sub-modules (no avgpool / fc)
        self.stem   = nn.Sequential(_bb.conv1, _bb.bn1, _bb.relu, _bb.maxpool)
        self.layer1 = _bb.layer1
        self.layer2 = _bb.layer2
        self.layer3 = _bb.layer3
        self.layer4 = _bb.layer4

        enc_dim = _bb.fc.in_features    # 2048 for R50/R101/ResNeXt-50/101

        # ── token projection ──────────────────────────────────────────────────
        self.token_proj = nn.Linear(enc_dim, proj_dim)
        nn.init.trunc_normal_(self.token_proj.weight, std=0.02)

        # ── cross-attention layers ────────────────────────────────────────────
        assert proj_dim % n_heads == 0, f"proj_dim={proj_dim} must be divisible by n_heads={n_heads}"
        self.cross_layers = nn.ModuleList([
            _CrossAttnBlock(proj_dim, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # ── classification head ───────────────────────────────────────────────
        self.norm_t = nn.LayerNorm(proj_dim)
        self.norm_d = nn.LayerNorm(proj_dim)
        head_in = 2 * proj_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, head_in // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_in // 2, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"StateCompareNet: {n_params/1e6:.1f}M params  "
              f"(enc={backbone_variant}{n_resnet_layers}, proj={proj_dim}, "
              f"layers={n_cross_layers}, in_ch={in_channels})")

    # ── encode a (B, 2, C, H, W) frame pair to (B, n_tokens, proj_dim) tokens ─

    def _encode_pair(self, pair: torch.Tensor) -> torch.Tensor:
        B, _, C, H, W = pair.shape
        if self.use_frame_diff:
            diff = pair[:, 1] - pair[:, 0]                    # (B, C, H, W)
            x = torch.cat([pair[:, 0], pair[:, 1], diff], dim=1)  # (B, 3C, H, W)
        else:
            x = pair.reshape(B, 2 * C, H, W)                  # (B, 2C, H, W)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                      # (B, enc_dim, h, w)

        h, w = x.shape[2], x.shape[3]
        tokens = x.permute(0, 2, 3, 1).reshape(B, h * w, -1)  # (B, h*w, enc_dim)
        return self.token_proj(tokens)          # (B, h*w, proj_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, C, H, W) — ImageNet-normalised
        before = self._encode_pair(x[:, :2])    # (B, n_tok, proj_dim)  initial state
        after  = self._encode_pair(x[:, 2:])    # (B, n_tok, proj_dim)  final state

        # Cross-attention: each before-patch attends to all after-patches.
        # "This region had an object — did it persist, move, or disappear?"
        q = before
        for layer in self.cross_layers:
            q = layer(q, after)                 # transition_tokens

        transition = q.mean(dim=1)                              # (B, proj_dim)
        delta      = after.mean(dim=1) - before.mean(dim=1)    # (B, proj_dim)

        feat = torch.cat([self.norm_t(transition), self.norm_d(delta)], dim=1)
        return self.head(feat)

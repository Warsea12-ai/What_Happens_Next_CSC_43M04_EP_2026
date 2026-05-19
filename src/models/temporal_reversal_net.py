"""TemporalReversalNet — Symmetric + Asymmetric Temporal Encoding.

WHY this architecture:

  Many action pairs in "what happens next" are temporal reverses of each other:
    pick_up  ↔  put_down     (upward vs downward trajectory)
    open     ↔  close        (outward vs inward motion)
    pour     ↔  fill         (container-tilting direction)
    stir/shake               (self-similar under reversal — symmetric)

  A standard temporal model sees f0→f1→f2→f3 and must learn implicitly
  that "first frame = grip at floor level, last frame = raised object" = pick_up.
  It has to discover the arrow of time from scratch.

  TemporalReversalNet makes the arrow of time an EXPLICIT architectural
  feature by processing the 4-frame sequence both FORWARD and REVERSED
  through the same shared temporal transformer:

    fwd_feat  = encode(f0, f1, f2, f3)
    rev_feat  = encode(f3, f2, f1, f0)

  These two descriptors naturally decompose into:
    symmetric  = (fwd + rev) / 2   —  "what kind of action", direction-agnostic
    asymmetric = (fwd − rev) / 2   —  "which temporal direction it goes"

  The head sees both and can exploit whichever is informative per class.
  For symmetric actions (stir, shake), asym ≈ 0 and the model ignores it.
  For asymmetric actions (pick_up, pour, open), asym is large and is the
  primary discriminator between reversed-pair classes.

HOW it works:

  1. Shared R50 encoder WITH avgpool:
       (B×4, 3, H, W)  →  (B×4, 2048)  →  (B, 4, 2048)
     All frames processed in one batched CNN pass.

  2. Project CNN output to transformer dim: (B, 4, proj_dim)

  3. Add learned temporal positional encoding of shape (1, 4, proj_dim).
     The SAME PE is applied to both passes — position 0 always means
     "first token in this particular pass", giving the transformer a
     consistent notion of "beginning" and "end" for both directions.

  4. Forward pass: 2-layer self-attention over [projected f0, f1, f2, f3]
       → mean-pool over T → fwd_feat  (B, proj_dim)

  5. Reverse pass: 2-layer self-attention over [projected f3, f2, f1, f0]
       (identical weights, identical PE, flipped feature order)
       → mean-pool over T → rev_feat  (B, proj_dim)

  6. Head: cat([fwd_feat, rev_feat]) → 2-layer MLP → num_classes

COMPLEMENTARY TO EXISTING ARCHITECTURES:
  StateCompareNet:      WHERE did things change? (spatial comparison)
  FactorSTNet:          WHEN did things change? (per-patch trajectory)
  MotionPyramidNet:     AT WHAT TEMPORAL SCALE? (frequency decomposition)
  ActionletNet:         WHICH discriminative pattern? (learned detectors)
  TemporalReversalNet:  IN WHICH DIRECTION? (forward vs reversed encoding)

SIGNAL PROCESSING VIEW:
  fwd_feat + rev_feat = 2 × symmetric content  (even part of the temporal signal)
  fwd_feat − rev_feat = 2 × asymmetric content (odd part = the "arrow of time")
  Concatenating [fwd, rev] is equivalent to concatenating [even, odd] and is a
  complete representation — no information is discarded.

PARAMS: ~26M (R50) + ~6M (2 × temporal blocks at D=512) + 0.5M (head) ≈ 33M,
         all from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torchvision.models import resnext50_32x4d, resnext101_32x8d


class _TemporalBlock(nn.Module):
    """Pre-LN self-attention block over a sequence of T temporal tokens."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TemporalReversalNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 33,
        n_resnet_layers: int = 50,
        backbone_variant: str = "resnet",
        proj_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        _factories = {
            ("resnet",  50):  resnet50,
            ("resnet",  101): resnet101,
            ("resnext", 50):  resnext50_32x4d,
            ("resnext", 101): resnext101_32x8d,
        }
        key = (backbone_variant, n_resnet_layers)
        if key not in _factories:
            raise ValueError(f"Unsupported backbone: {key}")
        _bb = _factories[key](weights=None)

        # ── shared CNN encoder WITH avgpool (global vector per frame) ─────────
        self.encoder = nn.Sequential(
            _bb.conv1, _bb.bn1, _bb.relu, _bb.maxpool,
            _bb.layer1, _bb.layer2, _bb.layer3, _bb.layer4,
            _bb.avgpool,
            nn.Flatten(1),     # (B, enc_dim)
        )
        enc_dim = _bb.fc.in_features   # 2048 for R50/R101/ResNeXt

        # ── project CNN output into transformer width ──────────────────────────
        assert proj_dim % n_heads == 0
        self.proj = nn.Linear(enc_dim, proj_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

        # ── learned temporal positional encoding ──────────────────────────────
        # Shape (1, max_T, proj_dim); same PE used for forward and reversed pass.
        self.temporal_pe = nn.Parameter(torch.zeros(1, 32, proj_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        # ── shared temporal transformer ───────────────────────────────────────
        # The SAME weights encode both the forward and the reversed sequence.
        # Any asymmetry in output comes purely from the reversed feature order,
        # not from different weights — keeping the parameter count low.
        self.blocks = nn.ModuleList([
            _TemporalBlock(proj_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(proj_dim)

        # ── head: receives [fwd_feat ‖ rev_feat] ──────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TemporalReversalNet: {n_params/1e6:.1f}M params  "
              f"(enc={backbone_variant}{n_resnet_layers}, "
              f"proj={proj_dim}, layers={n_layers})")

    # ── temporal encoding pass ────────────────────────────────────────────────

    def _encode_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T, proj_dim) with PE already added → (B, proj_dim)."""
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).mean(dim=1)  # (B, proj_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, 3, H, W) — ImageNet-normalised
        B, T, C, H, W = x.shape

        # Encode all T frames in a single batched CNN pass
        frames = x.reshape(B * T, C, H, W)
        feats  = self.encoder(frames)          # (B*T, enc_dim)
        feats  = self.proj(feats)              # (B*T, proj_dim)
        feats  = feats.reshape(B, T, -1)       # (B, T, proj_dim)

        pe = self.temporal_pe[:, :T, :]        # (1, T, proj_dim)

        # Forward pass: f0 → f1 → f2 → f3
        fwd_tokens = feats + pe                       # (B, T, proj_dim)
        fwd_feat   = self._encode_sequence(fwd_tokens)  # (B, proj_dim)

        # Reverse pass: f3 → f2 → f1 → f0 (same PE, reversed feature order)
        rev_tokens = feats.flip(dims=[1]) + pe        # (B, T, proj_dim)
        rev_feat   = self._encode_sequence(rev_tokens)  # (B, proj_dim)

        # Concatenate: [symmetric + asymmetric information]
        feat = torch.cat([fwd_feat, rev_feat], dim=-1)  # (B, 2*proj_dim)
        return self.head(feat)

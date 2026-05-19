"""SoftFlowNet — Feature-Space Soft Correspondence Tokens.

WHY this architecture:

  All existing Track A models operate on either pixel-level temporal differences
  (MotionPyramidNet) or raw per-frame feature vectors (FactorSTNet, StateCompareNet,
  ActionletNet, TemporalReversalNet).  None of them explicitly models WHERE each
  spatial patch moved in feature space between frames.

  SoftFlowNet computes a differentiable soft optical flow entirely in CNN feature space:

    corr(t→t+1)_ij  = softmax( feat_{t,i} · feat_{t+1,j} / √d )   # (B, 49, 49)
    flow(t→t+1)_i   = Σ_j  corr_ij · feat_{t+1,j}  −  feat_{t,i} # (B, 49, d)

  Interpretation:
    corr_ij ≈ 1   if patch i in frame t "moved to" patch j in frame t+1
    flow_i  ≈ 0   if the patch at position i stayed still (static background)
    flow_i  large if something arrived at position i that wasn't there before

  This is strictly a feature-level operation — the correspondence is semantic, not
  pixel-level.  A cup moving to the left produces a large flow_i at its departure
  position and at its arrival position, even if the pixel colors are slightly different.

HOW it works (for 4 frames):

  Step 1 — Shared R50 encoder (no avgpool):
    (B×4, 3, H, W) → (B×4, 2048, 7, 7) → project to (B, 4, 49, flow_dim)

  Step 2 — Compute 4 soft-flow tokens via spatial average:
    flow_01 = avg_patches( corr(f0→f1) @ f1 − f0 )   (B, flow_dim)
    flow_12 = avg_patches( corr(f1→f2) @ f2 − f1 )
    flow_23 = avg_patches( corr(f2→f3) @ f3 − f2 )
    flow_03 = avg_patches( corr(f0→f3) @ f3 − f0 )   global displacement

  Step 3 — Appearance token:
    mean over all frames and patches                  (B, flow_dim)

  Step 4 — 5 learned token-type embeddings injected before attention
    (appearance / flow_01 / flow_12 / flow_23 / flow_03) — 5 distinct roles.

  Step 5 — 2-layer self-attention over the 5 tokens.

  Step 6 — Mean-pool → LayerNorm → MLP head → 33 classes.

COMPLEMENTARY TO EXISTING ARCHITECTURES:
  StateCompareNet:    WHERE did things change? (patch cross-attention, pixel features)
  FactorSTNet:        WHEN did things change? (per-patch self-attention over time)
  MotionPyramidNet:   AT WHAT TEMPORAL SCALE? (pixel-level temporal differences)
  ActionletNet:       WHICH discriminative pattern? (learned query detectors)
  TemporalReversalNet: IN WHICH DIRECTION? (forward vs reversed encoding)
  SoftFlowNet:        WHERE DID EACH PATCH GO? (feature-space soft correspondences)

SIGNAL PROCESSING VIEW:
  Soft-flow is a first-order Taylor expansion of optical flow in feature space.
  The correlation matrix is the attention map of a cross-attention operation
  where frame t queries and frame t+1 provides keys and values — the residual
  of this cross-attention w.r.t. the query is the flow signal.

PARAMS: ~26M (R50) + ~1M (flow head + attn) ≈ 27M, all from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models import resnext50_32x4d, resnext101_32x8d

# Token-type indices
_TOKEN_APPEARANCE = 0
_TOKEN_FLOW_01    = 1
_TOKEN_FLOW_12    = 2
_TOKEN_FLOW_23    = 3
_TOKEN_FLOW_03    = 4   # global displacement


class _AttnBlock(nn.Module):
    """Pre-LN self-attention block for the 5-token flow sequence."""

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
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class SoftFlowNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 33,
        n_resnet_layers: int = 50,
        backbone_variant: str = "resnet",
        flow_dim: int = 256,
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

        # ── CNN backbone (no avgpool — we need spatial tokens) ────────────────
        self.stem   = nn.Sequential(_bb.conv1, _bb.bn1, _bb.relu, _bb.maxpool)
        self.layer1 = _bb.layer1
        self.layer2 = _bb.layer2
        self.layer3 = _bb.layer3
        self.layer4 = _bb.layer4
        enc_dim = _bb.fc.in_features   # 2048

        # ── project CNN tokens into flow_dim ──────────────────────────────────
        assert flow_dim % n_heads == 0
        self.token_proj = nn.Linear(enc_dim, flow_dim)
        nn.init.trunc_normal_(self.token_proj.weight, std=0.02)

        # ── token-type embeddings: 5 roles ────────────────────────────────────
        self.token_type_embed = nn.Embedding(5, flow_dim)
        self.register_buffer(
            "token_types",
            torch.tensor(
                [_TOKEN_APPEARANCE, _TOKEN_FLOW_01, _TOKEN_FLOW_12,
                 _TOKEN_FLOW_23, _TOKEN_FLOW_03],
                dtype=torch.long,
            ),
        )

        # ── self-attention over the 5 flow tokens ─────────────────────────────
        self.attn_blocks = nn.ModuleList([
            _AttnBlock(flow_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(flow_dim)

        # ── classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flow_dim, flow_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(flow_dim * 2, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SoftFlowNet: {n_params/1e6:.1f}M params  "
              f"(enc={backbone_variant}{n_resnet_layers}, "
              f"flow_dim={flow_dim}, layers={n_layers})")

    # ── per-frame spatial feature extraction ─────────────────────────────────

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) → (B, T, P, flow_dim)  where P = h×w patches."""
        B, T, C, H, W = x.shape
        frames = x.reshape(B * T, C, H, W)
        feat = self.stem(frames)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)                        # (B*T, enc_dim, h, w)
        _, D, h, w = feat.shape
        P = h * w
        tokens = feat.permute(0, 2, 3, 1).reshape(B * T, P, D)
        tokens = self.token_proj(tokens)                # (B*T, P, flow_dim)
        return tokens.reshape(B, T, P, -1)             # (B, T, P, flow_dim)

    # ── soft flow between a source and destination frame ─────────────────────

    @staticmethod
    def _soft_flow(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Compute spatially-averaged soft flow from src to dst.

        src, dst: (B, P, d) — spatial patch tokens of two adjacent frames.
        Returns:  (B, d)    — mean flow vector across all P patch positions.

        The correlation matrix C_ij is the cross-attention weight from patch i
        in src to patch j in dst.  The flow residual at position i is the
        weighted combination of dst features minus the src feature at i.
        """
        d = src.shape[-1]
        # Cross-frame correlation: (B, P_src, P_dst)
        corr = torch.bmm(src, dst.transpose(1, 2)) / (d ** 0.5)
        corr = F.softmax(corr, dim=-1)
        # Soft-propagated dst features for each src position: (B, P, d)
        propagated = torch.bmm(corr, dst)
        # Flow residual: what arrived minus what left
        flow = propagated - src                         # (B, P, d)
        return flow.mean(dim=1)                         # (B, d)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, 3, H, W)
        B = x.shape[0]

        tokens = self._encode_frames(x)                # (B, 4, P, flow_dim)
        f0, f1, f2, f3 = tokens[:, 0], tokens[:, 1], tokens[:, 2], tokens[:, 3]

        # Appearance token: global mean over all frames and spatial positions
        appearance = tokens.mean(dim=(1, 2))           # (B, flow_dim)

        # 4 soft-flow tokens
        flow_01 = self._soft_flow(f0, f1)              # (B, flow_dim)
        flow_12 = self._soft_flow(f1, f2)
        flow_23 = self._soft_flow(f2, f3)
        flow_03 = self._soft_flow(f0, f3)              # global displacement

        # Stack into 5-token sequence
        seq = torch.stack(
            [appearance, flow_01, flow_12, flow_23, flow_03], dim=1
        )  # (B, 5, flow_dim)

        # Inject token-type embeddings so the attention knows each token's role
        type_embeds = self.token_type_embed(self.token_types)  # (5, flow_dim)
        seq = seq + type_embeds.unsqueeze(0)

        # Self-attention: let the 5 flow tokens exchange information
        for block in self.attn_blocks:
            seq = block(seq)                           # (B, 5, flow_dim)

        feat = self.norm(seq).mean(dim=1)              # (B, flow_dim)
        return self.head(feat)

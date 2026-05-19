"""ActionletNet — Learned Actionlet Detectors via Cross-Attention.

WHY this architecture:

  StateCompareNet, FactorSTNet, and MotionPyramidNet all extract features
  from the video and then pool them into a global descriptor (mean or attention-
  weighted mean).  This pooling can wash out highly localised discriminative
  patterns that occur in only a small sub-region or sub-interval of the video.

  Actions are characterised by SPECIFIC spatial-temporal patterns — "actionlets"
  — that are sparse and localised:
    pour / fill:   a small pouring region in the upper-middle of the scene
    cut / chop:    a repetitive wrist-level motion in one corner
    stir:          a circular motion confined to the bowl area
    pick_up:       a gripper region that moves upward

  ActionletNet learns K detector queries that each specialise in detecting one
  type of actionlet pattern.  Classification comes from the PATTERN of which
  detectors fire — not from a blurred global average over all tokens.

HOW it works (for 4 frames, 7×7 = 49 patches per frame):

  Step 1 — CNN encoder (R50, no avgpool):
    All 4 frames processed in one batched CNN pass:
      (B×4, 3, H, W) → (B×4, 2048, 7, 7)
    Flatten spatial and project to 256-D:
      (B, 4, 49, 2048) → (B, 4, 49, 256)
    Add (additive) temporal PE + spatial PE.
    Flatten time and space:
      (B, 4×49, 256) = (B, 196, 256)  — the "video token grid"

  Step 2 — K=64 learned actionlet detector queries:
    Each query is a learnable 256-D vector, initialised with truncated normal.
    Two cross-attention layers: queries attend to the 196 video tokens.
    "Where in the video does my template pattern appear?"

  Step 3 — Actionlet interaction (1 self-attention layer):
    Queries exchange information among themselves.
    "Do detectors A and B co-fire?  That combination might mean 'pick_up'."

  Step 4 — Classify:
    Mean-pool over K=64 detector responses → LayerNorm → 2-layer MLP → 33 classes.

COMPLEMENTARY TO EXISTING ARCHITECTURES:
  StateCompareNet:    WHERE did things change? (spatial cross-attention before↔after)
  FactorSTNet:        WHEN did things change? (per-patch temporal trajectory)
  MotionPyramidNet:   AT WHAT TEMPORAL SCALE? (frequency decomposition)
  ActionletNet:       WHICH discriminative patterns are present? (learned detectors)

SIGNAL PROCESSING VIEW:
  The actionlet queries act like a learned filter bank: each filter fires when
  its template pattern appears anywhere in the spatiotemporal volume.  The
  classifier then operates on the filter-response vector rather than raw pixels
  or pooled features.

PARAMS: ~26M (R50) + ~3M (cross/self-attn at D=256) + 0.1M (head) ≈ 29M,
         all from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torchvision.models import resnext50_32x4d, resnext101_32x8d


# ── attention blocks ──────────────────────────────────────────────────────────

class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention: queries attend to context (key-value) tokens."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn    = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2   = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 4, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q:  (B, K, D)  — actionlet queries
        # kv: (B, N, D)  — video token context
        q_norm  = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        q = q + self.drop(attn_out)
        q = q + self.drop(self.ff(self.norm2(q)))
        return q  # (B, K, D)


class _SelfAttnBlock(nn.Module):
    """Pre-LN self-attention block for the K actionlet responses."""

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


# ── main model ────────────────────────────────────────────────────────────────

class ActionletNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 33,
        n_resnet_layers: int = 50,
        backbone_variant: str = "resnet",
        proj_dim: int = 256,
        n_actionlets: int = 64,
        n_cross_layers: int = 2,
        n_self_layers: int = 1,
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

        # ── CNN backbone (shared across frames, no avgpool/fc) ────────────────
        self.stem   = nn.Sequential(_bb.conv1, _bb.bn1, _bb.relu, _bb.maxpool)
        self.layer1 = _bb.layer1
        self.layer2 = _bb.layer2
        self.layer3 = _bb.layer3
        self.layer4 = _bb.layer4
        enc_dim = _bb.fc.in_features   # 2048 for R50/R101/ResNeXt

        # ── token projection: from CNN feature depth to transformer width ─────
        assert proj_dim % n_heads == 0
        self.token_proj = nn.Linear(enc_dim, proj_dim)
        nn.init.trunc_normal_(self.token_proj.weight, std=0.02)

        # ── additive positional encodings (temporal + spatial, separate) ──────
        # Temporal: one vector per frame position (buffer up to 32 frames)
        # Spatial:  one vector per patch (buffer up to 14×14=196 patches)
        self.temporal_pe = nn.Parameter(torch.zeros(1, 32, 1, proj_dim))
        self.spatial_pe  = nn.Parameter(torch.zeros(1, 1, 14 * 14, proj_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        nn.init.trunc_normal_(self.spatial_pe,  std=0.02)

        # ── learned actionlet queries ─────────────────────────────────────────
        # K templates, each a proj_dim vector; shared across the batch.
        self.actionlets = nn.Parameter(torch.empty(1, n_actionlets, proj_dim))
        nn.init.trunc_normal_(self.actionlets, std=0.02)
        self.n_actionlets = n_actionlets

        # ── cross-attention: queries → video tokens ───────────────────────────
        self.cross_layers = nn.ModuleList([
            _CrossAttnBlock(proj_dim, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # ── self-attention: among activated actionlet responses ───────────────
        self.self_layers = nn.ModuleList([
            _SelfAttnBlock(proj_dim, n_heads, dropout)
            for _ in range(n_self_layers)
        ])

        self.norm = nn.LayerNorm(proj_dim)

        # ── classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim * 2, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ActionletNet: {n_params/1e6:.1f}M params  "
              f"(enc={backbone_variant}{n_resnet_layers}, "
              f"proj={proj_dim}, K={n_actionlets}, "
              f"cross={n_cross_layers}×CA+{n_self_layers}×SA)")

    # ── per-frame encoding ────────────────────────────────────────────────────

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) → (B, T, P, proj_dim)  where P = h×w patches."""
        B, T, C, H, W = x.shape
        frames = x.reshape(B * T, C, H, W)
        feat = self.stem(frames)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)                       # (B*T, enc_dim, h, w)
        _, D, h, w = feat.shape
        P = h * w
        tokens = feat.permute(0, 2, 3, 1).reshape(B * T, P, D)  # (B*T, P, enc_dim)
        tokens = self.token_proj(tokens)               # (B*T, P, proj_dim)
        return tokens.reshape(B, T, P, -1)            # (B, T, P, proj_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, 3, H, W)
        B, T, C, H, W = x.shape

        tokens = self._encode_frames(x)               # (B, T, P, proj_dim)
        P = tokens.shape[2]

        # Additive positional encodings
        tokens = tokens + self.temporal_pe[:, :T, :, :]     # broadcast over P
        tokens = tokens + self.spatial_pe[:, :, :P, :]      # broadcast over T

        # Flatten temporal and spatial dimensions → video token grid
        video_tokens = tokens.reshape(B, T * P, -1)   # (B, 196, proj_dim)

        # Expand learned actionlet queries to batch dimension
        queries = self.actionlets.expand(B, -1, -1)   # (B, K, proj_dim)

        # Cross-attention: each query probes the full spatiotemporal token grid
        for layer in self.cross_layers:
            queries = layer(queries, video_tokens)     # (B, K, proj_dim)

        # Self-attention: let activated actionlets communicate
        for layer in self.self_layers:
            queries = layer(queries)                   # (B, K, proj_dim)

        # Aggregate: mean over K actionlet responses
        feat = self.norm(queries).mean(dim=1)          # (B, proj_dim)
        return self.head(feat)

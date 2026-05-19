"""FactorSTNet — CNN + Divided Space-Time Transformer, trained from scratch.

WHY this architecture:

  StateCompareNet compares initial state vs final state — it asks "what changed
  between frame 0 and frame 3?" This is powerful but discards the TRAJECTORY:
  the intermediate dynamics of how the scene evolved across all 4 frames.

  Two actions can have identical start/end states but different intermediate
  paths:  picking_up vs putting_down,  opening vs closing,  pouring vs filling.
  StateCompareNet cannot distinguish these without trajectory information.

  FactorSTNet models the full 4D trajectory: each spatial patch has an
  independent temporal view of how IT evolved, then patches communicate
  spatially to build a global representation.

HOW it works:

  shared CNN (per-frame):  4 × (B, 3, H, W) → 4 × (B, D_enc, 7, 7)
  token projection:        (B, 4, 49, D_enc) → (B, 4, 49, proj_dim=256)
  + temporal PE + spatial PE

  4 × FactoredSTBlock:
    ① Temporal attention: reshape to (B×49, 4, D), attend over T=4
       each patch tracks ITS OWN temporal evolution independently
    ② Spatial  attention: reshape to (B×4, 49, D), attend over P=49
       patches exchange context within each time step

  global average pool (over T and P) → (B, D)
  head → num_classes

DIVIDED VS JOINT ATTENTION:
  Joint space-time: 196×196=38k pairs per head (tractable but expensive, may overfit)
  Divided (this):   temporal 4×4=16 + spatial 49×49=2401 per layer — much cheaper,
                    better inductive bias (temporal axis and spatial axis are different)

COMPLEMENTARY TO StateCompareNet:
  StateCompareNet: "what is the net spatial change between initial and final state?"
  FactorSTNet:     "how did each spatial location evolve step-by-step?"

Params: ~25M (CNN) + ~5M (factored transformer) ≈ 30M total, all from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torchvision.models import resnext50_32x4d, resnext101_32x8d


# ── factored ST attention blocks ───────────────────────────────────────────────

class _TemporalAttnBlock(nn.Module):
    """Pre-LN self-attention over the T time-step dimension.

    Input shape: (B, T, P, D) — batch, time, patches, features.
    Temporal attention is applied independently for each spatial patch:
    reshapes to (B×P, T, D), applies MHSA, reshapes back.
    """

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
        B, T, P, D = x.shape
        # Temporal attention: each patch attends over its own T-step trajectory
        h = x.permute(0, 2, 1, 3).reshape(B * P, T, D)
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)
        h = h + self.drop(attn_out)
        h = h + self.drop(self.ff(self.norm2(h)))
        return h.reshape(B, P, T, D).permute(0, 2, 1, 3)  # (B, T, P, D)


class _SpatialAttnBlock(nn.Module):
    """Pre-LN self-attention over the P spatial-patch dimension.

    Input shape: (B, T, P, D).
    Spatial attention applied independently for each time step:
    reshapes to (B×T, P, D), applies MHSA, reshapes back.
    """

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
        B, T, P, D = x.shape
        # Spatial attention: each time step, all patches attend to each other
        h = x.reshape(B * T, P, D)
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)
        h = h + self.drop(attn_out)
        h = h + self.drop(self.ff(self.norm2(h)))
        return h.reshape(B, T, P, D)


class _FactoredSTBlock(nn.Module):
    """One factored space-time block: temporal attention then spatial attention."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.temporal = _TemporalAttnBlock(d, n_heads, dropout)
        self.spatial  = _SpatialAttnBlock(d, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        return x


# ── main model ─────────────────────────────────────────────────────────────────

class FactorSTNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 33,
        n_resnet_layers: int = 50,
        backbone_variant: str = "resnet",
        proj_dim: int = 256,
        n_blocks: int = 4,
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

        # ── CNN backbone (shared across all frames, no avgpool/fc) ───────────
        self.stem   = nn.Sequential(_bb.conv1, _bb.bn1, _bb.relu, _bb.maxpool)
        self.layer1 = _bb.layer1
        self.layer2 = _bb.layer2
        self.layer3 = _bb.layer3
        self.layer4 = _bb.layer4
        enc_dim = _bb.fc.in_features   # 2048 for R50/R101/ResNeXt variants

        # ── token projection ──────────────────────────────────────────────────
        assert proj_dim % n_heads == 0
        self.token_proj = nn.Linear(enc_dim, proj_dim)
        nn.init.trunc_normal_(self.token_proj.weight, std=0.02)

        # ── positional encodings (temporal + spatial, separate and additive) ──
        # Temporal: one vector per frame position (up to 32 frames)
        # Spatial:  one vector per patch position (up to 14×14=196 patches)
        self.temporal_pe = nn.Parameter(torch.zeros(1, 32, 1, proj_dim))
        self.spatial_pe  = nn.Parameter(torch.zeros(1, 1, 14 * 14, proj_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        nn.init.trunc_normal_(self.spatial_pe,  std=0.02)

        # ── factored ST blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            _FactoredSTBlock(proj_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(proj_dim)

        # ── head ──────────────────────────────────────────────────────────────
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
        print(f"FactorSTNet: {n_params/1e6:.1f}M params  "
              f"(enc={backbone_variant}{n_resnet_layers}, "
              f"proj={proj_dim}, blocks={n_blocks}×[T+S])")

    # ── per-frame encoding ────────────────────────────────────────────────────

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W) → (B, T, P, proj_dim)  where P = h×w spatial patches."""
        B, T, C, H, W = x.shape
        # Process all T frames in one batch through the shared CNN
        frames = x.reshape(B * T, C, H, W)
        feat = self.stem(frames)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)               # (B*T, enc_dim, h, w)
        _, D, h, w = feat.shape
        P = h * w
        tokens = feat.permute(0, 2, 3, 1).reshape(B * T, P, D)  # (B*T, P, enc_dim)
        tokens = self.token_proj(tokens)       # (B*T, P, proj_dim)
        return tokens.reshape(B, T, P, -1)    # (B, T, P, proj_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, 3, H, W)
        B, T, C, H, W = x.shape

        tokens = self._encode_frames(x)        # (B, T, P, proj_dim)
        P = tokens.shape[2]

        # Additive positional encodings: where in time + where in space
        tokens = tokens + self.temporal_pe[:, :T, :, :]
        tokens = tokens + self.spatial_pe[:, :, :P, :]

        for block in self.blocks:
            tokens = block(tokens)             # (B, T, P, proj_dim)

        # Global average pool over both temporal and spatial dimensions
        feat = self.norm(tokens).mean(dim=(1, 2))  # (B, proj_dim)
        return self.head(feat)

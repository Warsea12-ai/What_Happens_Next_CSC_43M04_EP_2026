"""MotionPyramidNet — Temporal difference pyramid + shared CNN + scale attention.

WHY this architecture:

  StateCompareNet and FactorSTNet both take raw RGB frames and derive motion
  implicitly through learned representations.  Neither explicitly decomposes
  the temporal signal into multiple frequency bands.

  For "what happens next", different action classes are best identified at
  different temporal scales:
    pick_up / put_down:  the coarse first-to-last diff captures net displacement
    stir / shake:        local consecutive diffs capture oscillatory motion
    pour / fill:         medium-scale diffs capture continuous directional flow
    open / close:        direction of the global diff (positive vs negative)

  A network that sees only one temporal scale must learn to ignore irrelevant
  scales for each class.  A pyramid network sees all scales simultaneously and
  learns to select the right one.

HOW it works (for 4 frames f₀ f₁ f₂ f₃):

  Temporal pyramid — 8 signals computed from the 4 input frames:

    Fine   (consecutive):  f₁−f₀,  f₂−f₁,  f₃−f₂        3 signals
    Medium (stride-2):     f₂−f₀,  f₃−f₁                  2 signals
    Coarse (global):       f₃−f₀                           1 signal
    Context (appearance):  f₀,     f₃                      2 signals
    ─────────────────────────────────────────────────────  ─────────
    Total                                                   8 signals

  All 8 signals are 3-channel images processed by a SHARED R50 encoder in
  one batched forward pass: (B×8, 3, H, W) → (B×8, D) → (B, 8, D).

  Scale-type embeddings: each signal gets a learned 4-class type tag
  (fine / medium / coarse / context) added before the attention — the model
  knows what each token represents.

  Scale-selection transformer (2 layers, 8 tokens):
    Self-attention over the 8 scale tokens; learns to up-weight the temporal
    scales that are most informative for each class.
    8 × 8 = 64 attention pairs per head — completely trivial.

  Head: mean-pool over 8 attended tokens → LayerNorm → classify.

COMPLEMENTARY TO EXISTING ARCHITECTURES:
  StateCompareNet: WHERE did things change? (spatial cross-attention before↔after)
  FactorSTNet:     WHEN did things change? (per-patch 4-step trajectory)
  MotionPyramidNet: AT WHAT TEMPORAL SCALE? (pyramid frequency selection)

SIGNAL PROCESSING VIEW:
  The 4-frame sequence is a temporal signal sampled at 4 points.
  Fine diffs  = high temporal frequency (local velocity)
  Medium diffs = mid temporal frequency (integrated motion)
  Coarse diff  = low temporal frequency (global displacement)
  This is a discrete, learned approximation of a multi-resolution temporal analysis.

PARAMS: ~25M (R50 encoder) + <1M (scale attn + head) ≈ 26M, all from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
from torchvision.models import resnext50_32x4d, resnext101_32x8d

# Signal type indices for the scale-type embedding
_FINE    = 0   # consecutive diffs
_MEDIUM  = 1   # stride-2 diffs
_COARSE  = 2   # global diff
_CONTEXT = 3   # appearance frames

_SIGNAL_TYPES = [
    _FINE, _FINE, _FINE,        # f1-f0, f2-f1, f3-f2
    _MEDIUM, _MEDIUM,           # f2-f0, f3-f1
    _COARSE,                    # f3-f0
    _CONTEXT, _CONTEXT,         # f0, f3
]  # 8 entries


class _ScaleAttnBlock(nn.Module):
    """Pre-LN self-attention block for the 8-token scale sequence."""

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


class MotionPyramidNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 33,
        n_resnet_layers: int = 50,
        backbone_variant: str = "resnet",
        n_scale_layers: int = 2,    # transformer layers over 8 scale tokens
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

        # ── shared CNN encoder (avgpool kept — we want a global vector per signal) ─
        self.encoder = nn.Sequential(
            _bb.conv1, _bb.bn1, _bb.relu, _bb.maxpool,
            _bb.layer1, _bb.layer2, _bb.layer3, _bb.layer4,
            _bb.avgpool,       # AdaptiveAvgPool2d(1)
            nn.Flatten(1),     # (B, D)
        )
        enc_dim = _bb.fc.in_features   # 2048 for R50/R101/ResNeXt

        # ── scale-type embeddings (4 types: fine, medium, coarse, context) ──────
        # Added to token features before the transformer so the model knows
        # what kind of signal each token represents.
        self.scale_type_embed = nn.Embedding(4, enc_dim)
        # Register signal-type index tensor as a buffer (moves with .to(device))
        self.register_buffer(
            "signal_types",
            torch.tensor(_SIGNAL_TYPES, dtype=torch.long),  # (8,)
        )

        # ── scale-selection transformer over 8 tokens ─────────────────────────
        assert enc_dim % n_heads == 0
        self.scale_layers = nn.ModuleList([
            _ScaleAttnBlock(enc_dim, n_heads, dropout)
            for _ in range(n_scale_layers)
        ])
        self.norm = nn.LayerNorm(enc_dim)

        # ── classification head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim, enc_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(enc_dim // 2, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"MotionPyramidNet: {n_params/1e6:.1f}M params  "
              f"(enc={backbone_variant}{n_resnet_layers}, "
              f"scale_layers={n_scale_layers})")

    # ── pyramid decomposition ─────────────────────────────────────────────────

    @staticmethod
    def _build_pyramid(x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4, 3, H, W) → (B, 8, 3, H, W) temporal pyramid signals."""
        f0, f1, f2, f3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        signals = torch.stack([
            f1 - f0,            # fine:    local velocity t=0→1
            f2 - f1,            # fine:    local velocity t=1→2
            f3 - f2,            # fine:    local velocity t=2→3
            f2 - f0,            # medium:  integrated motion 0→2
            f3 - f1,            # medium:  integrated motion 1→3
            f3 - f0,            # coarse:  global displacement 0→3
            f0,                 # context: initial appearance
            f3,                 # context: final appearance
        ], dim=1)               # (B, 8, 3, H, W)

        return signals

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 3, H, W) — ImageNet-normalised
        B = x.shape[0]

        # Build 8 temporal pyramid signals
        signals = self._build_pyramid(x)            # (B, 8, 3, H, W)

        # Encode all 8 signals in one batched CNN pass
        signals_flat = signals.reshape(B * 8, *signals.shape[2:])  # (B*8, 3, H, W)
        features = self.encoder(signals_flat)       # (B*8, enc_dim)
        tokens = features.reshape(B, 8, -1)         # (B, 8, enc_dim)

        # Add scale-type embeddings — tell the model what each signal is
        type_embeds = self.scale_type_embed(self.signal_types)  # (8, enc_dim)
        tokens = tokens + type_embeds.unsqueeze(0)              # (B, 8, enc_dim)

        # Scale-selection transformer: learns which temporal scale matters per class
        for layer in self.scale_layers:
            tokens = layer(tokens)                  # (B, 8, enc_dim)

        # Aggregate: mean over scale dimension
        feat = self.norm(tokens).mean(dim=1)        # (B, enc_dim)
        return self.head(feat)

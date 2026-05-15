"""
VideoTransformer — CNN per-frame features + Transformer temporal reasoning.

CNNTemporal uses a handcrafted pooling (first_half | second_half | diff).
This model replaces that with learned self-attention: the T=4 frame tokens
attend to each other, and a CLS token aggregates the result.

With only 5 tokens (1 CLS + 4 frames), attention is near-free. The gain is
that the model can learn arbitrary temporal patterns — not just direction —
e.g. "first frame X, last frame Y regardless of middle frames" or
"frames 2 and 3 are the discriminative ones".

Architecture:
    1. Shared ResNet backbone  →  (B, T, D) per-frame features
    2. Linear projection       →  (B, T, d_model)
    3. Prepend CLS token       →  (B, T+1, d_model)
    4. Add learnable pos embed →  preserve temporal order
    5. Transformer encoder     →  (B, T+1, d_model)   [pre-norm, stable from scratch]
    6. CLS token               →  Dropout → Linear → num_classes

Input : (B, T, C, H, W)
Output: (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class VideoTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "resnet34",
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_frames: int = 32,
    ) -> None:
        super().__init__()

        backbone_specs = {
            "resnet18": (models.resnet18, 512),
            "resnet34": (models.resnet34, 512),
            "resnet50": (models.resnet50, 2048),
        }
        if backbone not in backbone_specs:
            raise ValueError(f"backbone must be one of {list(backbone_specs)}")

        fn, feat_dim = backbone_specs[backbone]
        net = fn(weights=None)
        net.fc = nn.Identity()  # type: ignore
        self.backbone = net
        self.feat_dim = feat_dim

        # Project backbone features down to transformer dimension
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # CLS token — aggregates temporal information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embedding (position 0 = CLS, 1..T = frames)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer — pre-norm (norm_first=True) is more stable from scratch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape

        # Per-frame features via shared backbone
        feats = self.backbone(video_batch.reshape(B * T, C, H, W))
        feats = feats.view(B, T, self.feat_dim)      # (B, T, D)

        # Project to transformer dimension
        feats = self.proj(feats)                     # (B, T, d_model)

        # Prepend CLS token and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, d_model)
        tokens = torch.cat([cls, feats], dim=1)      # (B, T+1, d_model)
        tokens = tokens + self.pos_embed[:, : T + 1]

        # Temporal self-attention
        out = self.transformer(tokens)               # (B, T+1, d_model)

        # CLS token → class prediction
        return self.head(out[:, 0])                  # (B, num_classes)

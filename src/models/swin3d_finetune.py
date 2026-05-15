"""
VideoSwinFinetune — Video Swin-3D-S fine-tuned for the 33-class challenge.

Track B (Open World): pretrained models are allowed.

Video Swin-3D is pretrained on Kinetics400 (video classification), so it
already captures motion, object-action interactions, and temporal patterns.
We replace the 400-class Kinetics head with a 33-class head and fine-tune
with a small LR on the backbone (to preserve pretrained knowledge) and a
larger LR on the new head.

Key properties:
  - Works with any T (tested T=4, 8, 16)
  - Uses ImageNet normalization (same as our pipeline: mean/std 0.485/0.229)
  - ~50M params total; head has 768→33 = ~25k params

Input : (B, T, C, H, W)   — standard dataset format
Output: (B, num_classes)

Differential LR example (in your optimizer):
    optimizer = AdamW([
        {"params": model.backbone_parameters(), "lr": 1e-5},
        {"params": model.head_parameters(),     "lr": 1e-4},
    ])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_s, Swin3D_S_Weights


class VideoSwinFinetune(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        model = swin3d_s(weights=weights)

        # Replace the Kinetics-400 head with our 33-class head
        model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )
        self.model = model

        self._init_head()

    def _init_head(self) -> None:
        nn.init.trunc_normal_(self.model.head[-1].weight, std=0.01)
        nn.init.zeros_(self.model.head[-1].bias)

    def backbone_parameters(self):
        """All parameters except the classification head."""
        return [p for n, p in self.named_parameters() if "model.head" not in n]

    def head_parameters(self):
        """Classification head parameters only."""
        return list(self.model.head.parameters())

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # Our dataset: (B, T, C, H, W) → Swin3D expects (B, C, T, H, W)
        return self.model(clips.permute(0, 2, 1, 3, 4).contiguous())

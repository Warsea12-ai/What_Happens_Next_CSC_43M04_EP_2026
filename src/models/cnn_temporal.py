"""
CNNTemporal — CNN baseline + explicit temporal direction encoding.

The CNN baseline (ResNet18 + avg pool) ignores the ORDER of frames.
For Something-Something style datasets, order is everything:
"moving left" ≠ "moving right", "picking up" ≠ "putting down".

This model keeps the same stable backbone but adds a direction signal:
    first_half_avg = avg(frames 0..T//2-1)   — where did we start?
    second_half_avg = avg(frames T//2..T-1)  — where did we end?
    direction = second_half - first_half     — what changed?

Concatenating all three gives the classifier an explicit motion vector
while retaining the full spatial context from each half.

Input : (B, T, C, H, W)
Output: (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNTemporal(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "resnet34",
        dropout: float = 0.5,
        head_dim: int = 512,
    ) -> None:
        super().__init__()

        backbone_fn = {
            "resnet18": (models.resnet18, 512),
            "resnet34": (models.resnet34, 512),
            "resnet50": (models.resnet50, 2048),
        }
        if backbone not in backbone_fn:
            raise ValueError(f"backbone must be one of {list(backbone_fn)}")

        fn, feature_dim = backbone_fn[backbone]
        net = fn(weights=None)
        net.fc = nn.Identity()  # type: ignore
        self.backbone = net
        self.feature_dim = feature_dim

        # Two-layer head: 3×D → head_dim → num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(3 * feature_dim),
            nn.Dropout(dropout),
            nn.Linear(3 * feature_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Small init on classifier so logits start near uniform
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape

        # Shared backbone over all frames
        feats = self.backbone(video_batch.reshape(B * T, C, H, W))  # (B*T, D)
        feats = feats.view(B, T, self.feature_dim)                  # (B, T, D)

        mid = T // 2 or 1  # guard against T=1
        first_half  = feats[:, :mid].mean(dim=1)        # (B, D) — beginning
        second_half = feats[:, mid:].mean(dim=1)        # (B, D) — end
        direction   = second_half - first_half          # (B, D) — motion vector

        combined = torch.cat([first_half, second_half, direction], dim=1)  # (B, 3D)
        return self.head(combined)

"""
CNNTemporal — CNN backbone + temporal direction encoding.

Three signals fed to the classifier:
    global_avg   = mean over all T frames         — spatial content
    avg_diff     = mean of adjacent frame diffs   — frame-by-frame motion
    net_dir      = last_frame - first_frame       — overall temporal direction

Adjacent differences (avg_diff) give T-1 independent gradient paths into the
backbone from the first epoch, unlike the previous first/second half split which
provided only one coarse difference signal after the backbone had converged.

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
        pretrained: bool = False,
        dropout: float = 0.5,
        head_dim: int = 512,
    ) -> None:
        super().__init__()

        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
        backbone_fn = {
            "resnet18": (models.resnet18, 512, ResNet18_Weights.IMAGENET1K_V1),
            "resnet34": (models.resnet34, 512, ResNet34_Weights.IMAGENET1K_V1),
            "resnet50": (models.resnet50, 2048, ResNet50_Weights.IMAGENET1K_V1),
        }
        if backbone not in backbone_fn:
            raise ValueError(f"backbone must be one of {list(backbone_fn)}")

        fn, feature_dim, weights = backbone_fn[backbone]
        net = fn(weights=weights if pretrained else None)
        net.fc = nn.Identity()  # type: ignore
        self.backbone = net
        self.feature_dim = feature_dim

        # Separate LayerNorms per signal so direction components (which are small
        # in magnitude for similar frames) are not drowned out by content components
        # when feeding into the linear head.
        self.ln_content   = nn.LayerNorm(feature_dim)
        self.ln_direction = nn.LayerNorm(feature_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * feature_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape

        feats = self.backbone(video_batch.reshape(B * T, C, H, W))  # (B*T, D)
        feats = feats.view(B, T, self.feature_dim)                  # (B, T, D)

        global_avg = feats.mean(dim=1)                  # (B, D) — content
        avg_diff   = (feats[:, 1:] - feats[:, :-1]).mean(dim=1)  # (B, D) — motion
        net_dir    = feats[:, -1] - feats[:, 0]         # (B, D) — overall direction

        combined = torch.cat([
            self.ln_content(global_avg),
            self.ln_direction(avg_diff),
            self.ln_direction(net_dir),
        ], dim=1)  # (B, 3D)
        return self.head(combined)

"""
CNNTemporal — CNN backbone + multi-scale temporal direction encoding.

Four signals fed to the classifier:
    global_avg      = mean over all T frames              — spatial content
    avg_diff_dt1    = mean of adjacent diffs (dt=1)       — local motion (fast)
    avg_diff_dt2    = mean of skip-one diffs (dt=2)       — mid-range motion (slow)
    net_dir         = last_frame - first_frame            — overall temporal direction

For T=4: dt=1 gives 3 diffs, dt=2 gives 2 diffs, dt=3 (net_dir) gives 1 diff.
Multi-scale captures both fast local changes and slower global dynamics, which
are complementary for distinguishing action classes that differ in pace or rhythm.

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

        self.ln_content   = nn.LayerNorm(feature_dim)
        self.ln_direction = nn.LayerNorm(feature_dim)  # shared across all diff scales

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * feature_dim, head_dim),  # global + dt1 + dt2 + net_dir
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

        global_avg   = feats.mean(dim=1)                        # (B, D)
        avg_diff_dt1 = (feats[:, 1:] - feats[:, :-1]).mean(dim=1)  # (B, D) dt=1
        avg_diff_dt2 = (feats[:, 2:] - feats[:, :-2]).mean(dim=1)  # (B, D) dt=2
        net_dir      = feats[:, -1] - feats[:, 0]               # (B, D) dt=T-1

        combined = torch.cat([
            self.ln_content(global_avg),
            self.ln_direction(avg_diff_dt1),
            self.ln_direction(avg_diff_dt2),
            self.ln_direction(net_dir),
        ], dim=1)  # (B, 4D)
        return self.head(combined)

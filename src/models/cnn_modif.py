from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class TemporalAttentionPool(nn.Module):
    """Pool (B, T, D) -> (B, D) en utilisant une query apprise."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, D)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)         # (B, 1, D)
        out, _ = self.attn(q, x, x)              # (B, 1, D)
        return self.norm(out.squeeze(1))         # (B, D)


class cnn_modif(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        feature_dim = backbone.fc.in_features    # 512
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.temporal_pool = TemporalAttentionPool(dim=feature_dim, num_heads=4)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape

        frames = video_batch.reshape(B * T, C, H, W)
        frame_features = self.backbone(frames)            # (B*T, 512)
        frame_features = torch.flatten(frame_features, 1)
        sequence = frame_features.view(B, T, -1)          # (B, T, 512)

        pooled = self.temporal_pool(sequence)             # (B, 512)
        return self.classifier(pooled)
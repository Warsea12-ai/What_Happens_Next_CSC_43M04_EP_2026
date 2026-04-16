"""
Track A / Track B: same architecture, `pretrained` flag toggles ImageNet weights.

Forward (conceptually):
    Input:  (batch, time, C, H, W)
    Reshape: (batch * time, C, H, W)  # each frame is an independent image
    Backbone: ResNet18 up to global average pool -> (batch * time, 512, 1, 1)
    Flatten: (batch * time, 512)
    Reshape: (batch, time, 512)
    Mean over time: (batch, 512)
    Linear classifier: (batch, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace the original 1000-way ImageNet head with identity; we add our own layer.
        feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape

        # Merge batch and time so the CNN runs frame-wise: (B*T, C, H, W)
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # (B*T, 512, 1, 1) -> (B*T, 512)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # Restore temporal structure: (B, T, 512)
        sequence_features = frame_features.view(batch_size, num_frames, -1)

        # Simple temporal pooling: average over frames -> (B, 512)
        pooled_features = sequence_features.mean(dim=1)

        # Class scores: (B, num_classes)
        logits = self.classifier(pooled_features)
        return logits

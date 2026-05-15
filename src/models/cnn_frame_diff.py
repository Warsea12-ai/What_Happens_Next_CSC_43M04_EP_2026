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
import math

# =============================================================================
# Positional encoding temporel
# =============================================================================
class TemporalPositionalEncoding(nn.Module):
    """
    PE temporel appliqué AVANT le backbone, sur des images (B, T, C, H, W).
    Le PE a shape (1, T, C, 1, 1) et broadcast sur H, W.
    """
    def __init__(self, num_channels: int, max_len: int = 32, mode: str = "learned"):
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels

        if mode == "sinusoidal":
            pe = torch.zeros(max_len, num_channels)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, num_channels, 2).float()
                * (-math.log(10000.0) / max(num_channels, 2))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            # gère le cas num_channels impair (ex. C=3)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
            # (1, T, C, 1, 1) pour broadcast sur (B, T, C, H, W)
            self.register_buffer("pe", pe.view(1, max_len, num_channels, 1, 1))

        elif mode == "learned":
            self.pe = nn.Parameter(torch.zeros(1, max_len, num_channels, 1, 1))
            nn.init.trunc_normal_(self.pe, std=0.02)

        else:
            raise ValueError(f"Unknown PE mode: {mode}")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, C, H, W)
        returns: (B, T, C, H, W) avec biais temporel additionné
        """
        T = video.size(1)
        return video + self.pe[:, :T]


class CNNBaseline(nn.Module):
    def __init__(
        self, num_classes: int, 
        pretrained: bool = False, 
        use_frame_diff=True, use_positional_encoding = True, 
        pe_mode: str = "sinusoidal") -> None:

        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace the original 1000-way ImageNet head with identity; we add our own layer.
        feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.use_frame_diff = use_frame_diff
        self.use_positional_encoding = use_positional_encoding

        self.pos_encoding = TemporalPositionalEncoding(
                num_channels=3,
                max_len=32,
                mode=pe_mode,
            )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape

        if self.use_positional_encoding:
            video_batch = self.pos_encoding(video_batch)

        # Merge batch and time so the CNN runs frame-wise: (B*T, C, H, W)
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        if self.use_frame_diff:
            diff = frames[:, 1:] - frames[:, :-1]
            frames = torch.cat([torch.zeros_like(frames[:, :1]), diff], dim=1)
            frames = torch.cat([frames, diff], dim=2)        # (B, T, 6, H, W)
            channels = 6

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

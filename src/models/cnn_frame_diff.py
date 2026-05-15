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


class cnn_frame_diff(nn.Module):
    def __init__(
        self, num_classes: int,
        pretrained: bool = False,
        use_frame_diff: bool = True,
        use_positional_encoding: bool = True,
        pe_mode: str = "sinusoidal",
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.use_frame_diff = use_frame_diff
        self.use_positional_encoding = use_positional_encoding

        # --- Adapter conv1 pour accepter 6 canaux si frame_diff ---
        if use_frame_diff:
            old_conv = backbone.conv1  # Conv2d(3, 64, 7, 2, 3, bias=False)
            new_conv = nn.Conv2d(
                in_channels=6,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    # Canaux 0-2 : poids ImageNet pour la frame brute
                    new_conv.weight[:, :3] = old_conv.weight
                    # Canaux 3-5 : on réutilise les mêmes poids pour le diff
                    # (alternative : .zero_() pour un démarrage neutre)
                    new_conv.weight[:, 3:] = old_conv.weight
            backbone.conv1 = new_conv

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.pos_encoding = TemporalPositionalEncoding(
            num_channels=3, max_len=32, mode=pe_mode,
        )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape

        if self.use_positional_encoding:
            video_batch = self.pos_encoding(video_batch)   # (B, T, 3, H, W)

        if self.use_frame_diff:
            # diff[t] = frame[t] - frame[t-1], diff[0] = 0
            diff = torch.zeros_like(video_batch)
            diff[:, 1:] = video_batch[:, 1:] - video_batch[:, :-1]
            # concat sur l'axe canal -> (B, T, 6, H, W)
            video_batch = torch.cat([video_batch, diff], dim=2)
            C = video_batch.size(2)  # 6

        # Frame-wise: (B*T, C, H, W)
        frames = video_batch.reshape(B * T, C, H, W)

        # (B*T, 512)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # (B, T, 512) -> mean -> (B, 512)
        sequence_features = frame_features.view(B, T, -1)
        pooled_features = sequence_features.mean(dim=1)

        return self.classifier(pooled_features)
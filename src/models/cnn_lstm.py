"""
CNN + LSTM: ResNet18 per frame, then an LSTM reads the frame feature sequence.

Forward:
    Input: (B, T, C, H, W)
    Frame CNN: (B*T, C, H, W) -> (B*T, 512)
    Sequence: (B, T, 512)
    LSTM: (B, T, hidden) -> take last timestep -> (B, hidden)
    Linear: (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        lstm_hidden_size: int = 512,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (batch_size, T, C, H, W)
        returns logits: (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = video_batch.shape
        frames = video_batch.reshape(batch_size * num_frames, channels, height, width)

        # (B*T, 512)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)

        # (B, T, 512)
        sequence = frame_features.view(batch_size, num_frames, -1)

        # lstm_out: (B, T, hidden), h_n: (1, B, hidden)
        lstm_out, (h_n, _) = self.lstm(sequence)

        # Last timestep output: (B, hidden)
        last_hidden = lstm_out[:, -1, :]

        logits = self.classifier(last_hidden)
        return logits

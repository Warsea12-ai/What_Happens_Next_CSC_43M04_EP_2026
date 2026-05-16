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
    """
    pool="avg"       — original: mean over all T frames (order-agnostic)
    pool="firstlast" — [first, last, last-first] each LayerNorm'd then linear
                       gives the head an explicit motion direction signal
    pool="attention" — lightweight learned attention (512→1) weights the T frames;
                       learns which frames are most discriminative per class
    """
    def __init__(self, num_classes: int, pretrained: bool = False, pool: str = "avg") -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.pool = pool

        if pool == "firstlast":
            self.ln = nn.LayerNorm(feature_dim)
            self.classifier = nn.Linear(3 * feature_dim, num_classes)
        elif pool == "attention":
            self.attn = nn.Linear(feature_dim, 1, bias=False)
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:  # "avg" — original, backward-compatible
            self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape
        feats = self.backbone(video_batch.reshape(B * T, C, H, W))
        feats = feats.view(B, T, self.feature_dim)                  # (B, T, 512)

        if self.pool == "firstlast":
            first     = feats[:, 0]                                  # (B, 512)
            last      = feats[:, -1]                                 # (B, 512)
            direction = last - first                                  # (B, 512)
            # Separate LayerNorm per signal so direction (small magnitude)
            # contributes equally to the linear layer alongside content signals.
            combined = torch.cat([self.ln(first), self.ln(last), self.ln(direction)], dim=1)
            return self.classifier(combined)

        elif self.pool == "attention":
            scores  = self.attn(feats)                               # (B, T, 1)
            weights = scores.softmax(dim=1)                          # (B, T, 1)
            pooled  = (weights * feats).sum(dim=1)                   # (B, 512)
            return self.classifier(pooled)

        else:  # avg
            return self.classifier(feats.mean(dim=1))

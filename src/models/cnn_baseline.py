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

import torch
import torch.nn as nn
from supplement/.blocks import R2Plus1DBlock, SEBlock3D
from supplement/.temporal_transformer import TemporalTransformer

class HybridR2Plus1DTransformer(nn.Module):
    """
    Modèle hybride pour Track A :
    - Stem : conv (2+1)D initiale
    - 4 stages de blocs R(2+1)D (comme un ResNet-18 video)
    - SE attention
    - Transformer temporel
    - Classification
    """
    def __init__(self, num_classes=33, num_frames=16, dropout=0.5):
        super().__init__()
        
        # Stem : capture les basses fréquences spatio-temporelles
        self.stem = nn.Sequential(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), 
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), 
                      padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # 4 stages à la ResNet
        self.stage1 = self._make_stage(64, 64, blocks=2, stride=1)
        self.stage2 = self._make_stage(64, 128, blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, blocks=2, stride=2)
        
        # Attention canal
        self.se = SEBlock3D(512, reduction=16)
        
        # Pooling spatial -> on garde la dimension temporelle
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Transformer temporel
        self.temporal_transformer = TemporalTransformer(
            d_model=512,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _make_stage(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        
        layers = [R2Plus1DBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(R2Plus1DBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.se(x)
        
        # Pool spatial uniquement : (B, 512, T, 1, 1) -> (B, T, 512)
        x = self.spatial_pool(x)
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        
        # Transformer temporel
        x = self.temporal_transformer(x)  # (B, 512)
        
        # Classification
        return self.classifier(x)
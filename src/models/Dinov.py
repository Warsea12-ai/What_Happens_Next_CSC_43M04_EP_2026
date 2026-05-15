import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class Dinov(nn.Module):
    def __init__(self, num_classes=33, backbone="facebook/dinov2-giant"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        d = self.backbone.config.hidden_size  # 1536 pour ViT-g
        # Positional embeddings temporels pour 4 frames
        self.temporal_pe = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        # Petit transformer temporel
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=4*d,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))

    def forward(self, x):  # x: (B, T=4, 3, H, W)
        B, T = x.shape[:2]
        with torch.no_grad():
            feats = self.backbone(pixel_values=x.flatten(0, 1)).last_hidden_state[:, 0]
        feats = feats.view(B, T, -1) + self.temporal_pe
        feats = self.temporal(feats).mean(dim=1)
        return self.head(feats)
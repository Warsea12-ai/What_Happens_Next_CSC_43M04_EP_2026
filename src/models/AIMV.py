import torch 
import torch.nn as nn 
from transformers import AutoModel, AutoImageProcessor

class AIMV(nn.Module):
    def __init__(self, num_classes=33, num_frames=4,
                 backbone="apple/aimv2-1B-patch14-448"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone,
            trust_remote_code=True,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        d = self.backbone.config.hidden_size
        self.num_frames = num_frames
        self.temporal_pe = nn.Parameter(torch.randn(1, num_frames, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=4*d,
                                            batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))

    def forward(self, x):
        B, T = x.shape[:2]
        with torch.no_grad():
            out = self.backbone(pixel_values=x.flatten(0, 1))
            # AIMv2 expose les patch tokens dans last_hidden_state ; on pool en moyenne
            feats = out.last_hidden_state.mean(dim=1)
        feats = feats.view(B, T, -1) + self.temporal_pe
        feats = self.temporal(feats).mean(dim=1)
        return self.head(feats)
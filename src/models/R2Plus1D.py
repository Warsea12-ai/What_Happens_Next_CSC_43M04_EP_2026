"""
R(2+1)D-18 — Wrapper for video classification, from-scratch (no pretraining).

Tran et al., CVPR 2018
"A Closer Look at Spatiotemporal Convolutions for Action Recognition"
arXiv:1711.11248

Format d'entrée : (B, T, C, H, W)  — format standard de ton dataloader
Format de sortie : (B, num_classes) — logits, compatible CrossEntropyLoss
"""
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18


class R2Plus1D(nn.Module):
    """
    R(2+1)D-18 pour la classification vidéo, entraîné from-scratch.

    L'architecture R(2+1)D décompose chaque convolution 3D classique en :
      - une convolution 2D spatiale (1 × 3 × 3)
      - suivie d'une convolution 1D temporelle (3 × 1 × 1)
    Cela donne plus de non-linéarités et facilite l'optimisation.

    Args
    ----
    num_classes : int
        Nombre de classes (ici 33 pour "What Happens Next").
    dropout : float
        Dropout appliqué avant la couche de classification finale.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        # weights=None → poids initialisés aléatoirement (from-scratch)
        self.backbone = r2plus1d_18(weights=None)

        # La tête originale est un Linear(512, 400) pour Kinetics-400.
        # On la remplace par notre tête à num_classes (33 ici).
        in_features = self.backbone.fc.in_features    # 512
        self.backbone.fc = nn.Identity()   # type: ignore[assignment]
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips : (B, T, C, H, W)  — T frames par vidéo
        return : (B, num_classes)
        """
        # torchvision attend (B, C, T, H, W), on permute
        x = clips.permute(0, 2, 1, 3, 4).contiguous()
        x = self.backbone(x)
        return self.head(x)
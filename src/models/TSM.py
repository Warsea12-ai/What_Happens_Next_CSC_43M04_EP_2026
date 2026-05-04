"""
TSM — Temporal Shift Module
Lin, Gan, Han, ICCV 2019
"A Closer Look at Spatial-Temporal Convolutions for Action Recognition"
arXiv:1811.08383

Idée : on prend un ResNet-18 standard 2D, et on insère un module qui
décale 1/8 des canaux de chaque feature map dans le temps :
    - 1/8 des canaux décalés vers le futur (shift left)
    - 1/8 décalés vers le passé (shift right)
    - 6/8 inchangés
Cela permet à chaque frame d'accéder à de l'information temporelle
voisine, sans aucun paramètre supplémentaire.

Format d'entrée : (B, T, C, H, W)
Format de sortie : (B, num_classes)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import BasicBlock


class TemporalShift(nn.Module):
    """
    Décale 1/fold_div des canaux le long de l'axe temporel.

    Args
    ----
    n_segment : int
        Nombre de frames T par vidéo (ici 4).
    fold_div : int
        Fraction de canaux décalés (papier : 8 → 1/8 des canaux).
    """
    def __init__(self, n_segment: int = 4, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B*T, C, H, W) — T frames empilées dans la dim batch
        nt, c, h, w = x.shape
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)

        # Canaux 0 .. fold : décalage vers le passé (chaque frame voit la suivante)
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # Canaux fold .. 2*fold : décalage vers le futur (chaque frame voit la précédente)
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]

        # Canaux restants : inchangés
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)


class TSMBasicBlock(nn.Module):
    """
    Wrapper qui applique TemporalShift avant le BasicBlock de ResNet.
    On garde le BasicBlock original tel quel et on shift juste avant.
    """
    def __init__(self, basic_block: BasicBlock, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.shift = TemporalShift(n_segment=n_segment, fold_div=fold_div)
        self.block = basic_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shift(x)
        return self.block(x)


def make_temporal_shift(net: nn.Module, n_segment: int, fold_div: int = 8) -> None:
    """
    Insère un TemporalShift avant chaque BasicBlock de chaque layer du ResNet.
    Modifie `net` en place.
    """
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(net, layer_name)
        new_blocks = nn.Sequential(*[
            TSMBasicBlock(block, n_segment=n_segment, fold_div=fold_div)
            for block in layer
        ])
        setattr(net, layer_name, new_blocks)


class TSM(nn.Module):
    """
    TSM-ResNet18, from-scratch, pour la classification vidéo.

    Args
    ----
    num_classes : int
        Nombre de classes (33 pour ton challenge).
    n_segment : int
        Nombre de frames par vidéo (4).
    fold_div : int
        Fraction de canaux décalés (papier : 8).
    dropout : float
        Dropout avant la couche de classification finale.
    """
    def __init__(
        self,
        num_classes: int,
        n_segment: int = 4,
        fold_div: int = 8,
        dropout: float = 0.5,
        n_resnet_layers: int = 18,
    ):
        super().__init__()
        self.n_segment = n_segment

        # Charger un ResNet-18 pré-entraîné sur ImageNet
        if n_resnet_layers == 18:
            backbone = resnet18(weights=None)  # pretrained=False est déprécié, utiliser weights=None
        elif n_resnet_layers == 34:       
            backbone = resnet34(weights=None)
        elif n_resnet_layers == 50:      
            backbone = resnet50(weights=None)                
        elif n_resnet_layers == 101:      
            backbone = resnet101(weights=None)                
        elif n_resnet_layers == 152:  
            backbone = resnet152(weights=None)
        else:
            raise ValueError(f"Unsupported n_resnet_layers: {n_resnet_layers}")

        # Insérer les TemporalShift dans chaque BasicBlock
        make_temporal_shift(backbone, n_segment=n_segment, fold_div=fold_div)

        # Remplacer la tête (1000 classes ImageNet → num_classes)
        in_features = backbone.fc.in_features      # 512
        backbone.fc = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clips.shape
        
        # Mettre à jour n_segment de tous les TemporalShift à la volée
        if T != self.n_segment:
            self.n_segment = T
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                for block in getattr(self.backbone, layer_name):
                    block.shift.n_segment = T
        
        x = clips.reshape(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, T, -1).mean(dim=1)
        return self.head(feats)                    # (B, num_classes) 
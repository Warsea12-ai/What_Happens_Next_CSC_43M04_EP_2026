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
from torchvision.models.resnet import BasicBlock, Bottleneck


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

class ResidualTSMBasicBlock(nn.Module):
    """
    Variante "in-place residual shift" du papier TSM.
    
    Le shift est appliqué uniquement à la branche convolutive du résidu :
    la connexion identité reste intacte. Cela évite que les corruptions
    successives du shift s'accumulent à travers les blocs, tout en
    laissant l'info temporelle se propager via la branche conv.
    """
    def __init__(self, basic_block: BasicBlock, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.shift = TemporalShift(n_segment=n_segment, fold_div=fold_div)
        self.block = basic_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x                              # <-- on garde x intact
        
        # Branche convolutive : on shift, puis conv1/bn1/relu/conv2/bn2
        out = self.shift(x)
        out = self.block.conv1(out)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        
        # Le downsample s'applique sur x (pas sur out) pour ajuster la
        # résolution/canaux quand nécessaire (changement de stage)
        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        
        out += identity
        return self.block.relu(out)

from torchvision.models.resnet import BasicBlock, Bottleneck


class ResidualTSMBottleneck(nn.Module):
    """Residual shift pour Bottleneck (ResNet-50/101/152).
    
    Différence avec BasicBlock : 3 convs (1x1 -> 3x3 -> 1x1) avec expansion x4,
    donc il faut appeler conv3 et bn3 en plus.
    """
    def __init__(self, bottleneck: Bottleneck, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.shift = TemporalShift(n_segment=n_segment, fold_div=fold_div)
        self.block = bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.shift(x)
        out = self.block.conv1(out)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)
        out = self.block.conv3(out)        # <-- la conv supplémentaire
        out = self.block.bn3(out)
        if self.block.downsample is not None:
            identity = self.block.downsample(x)
        out += identity
        return self.block.relu(out)

def make_temporal_shift(net: nn.Module, n_segment: int, fold_div: int = 8,
                        residual: bool = True) -> None:
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(net, layer_name)
        new_blocks = []
        for block in layer:
            if isinstance(block, Bottleneck):
                wrapped = ResidualTSMBottleneck(block, n_segment, fold_div)
            elif isinstance(block, BasicBlock):
                wrapped = ResidualTSMBasicBlock(block, n_segment, fold_div)
            else:
                raise TypeError(f"Bloc inattendu : {type(block).__name__}")
            new_blocks.append(wrapped)
        setattr(net, layer_name, nn.Sequential(*new_blocks))

class TSM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_segment: int = 4,
        fold_div: int = 8,
        dropout: float = 0.5,
        n_resnet_layers: int = 50,
        residual_shift: bool = True,        # <-- nouveau
    ):
        super().__init__()
        self.n_segment = n_segment

        # Charger un ResNet-18 pré-entraîné sur ImageNet
        if n_resnet_layers == 18:
            backbone = resnet18(pretrained=False)
        elif n_resnet_layers == 34:       
            backbone = resnet34(pretrained=False)
        elif n_resnet_layers == 50:      
            backbone = resnet50(pretrained=False)                
        elif n_resnet_layers == 101:      
            backbone = resnet101(pretrained=False)                
        elif n_resnet_layers == 152:   
            backbone = resnet152(pretrained=False)
        else:
            raise ValueError(f"Unsupported n_resnet_layers: {n_resnet_layers}")

        make_temporal_shift(
            backbone,
            n_segment=n_segment,
            fold_div=fold_div,
            residual=residual_shift,        # <-- propagation
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity() #type:ignore 
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    
    # forward inchangé
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
    

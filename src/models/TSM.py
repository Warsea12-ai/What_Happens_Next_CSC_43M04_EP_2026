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

class NonLocalBlock(nn.Module):
    """Non-local block (Wang et al. 2018) en version embedded Gaussian."""
    def __init__(self, in_channels, n_segment, reduction=2):
        super().__init__()
        inter = in_channels // reduction
        self.n_segment = n_segment
        self.theta = nn.Conv3d(in_channels, inter, 1)
        self.phi   = nn.Conv3d(in_channels, inter, 1)
        self.g     = nn.Conv3d(in_channels, inter, 1)
        self.W = nn.Sequential(
            nn.Conv3d(inter, in_channels, 1),
            nn.BatchNorm3d(in_channels),
        )
        nn.init.zeros_(self.W[0].weight)   #type:ignore init à 0 = identité au départ

    def forward(self, x):
        # x : (B*T, C, H, W) -> (B, C, T, H, W) pour Conv3d
        BT, C, H, W = x.shape
        T = self.n_segment
        B = BT // T
        x_reshaped = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        theta = self.theta(x_reshaped).flatten(2)              # (B, c, THW)
        phi   = self.phi(x_reshaped).flatten(2)
        g     = self.g(x_reshaped).flatten(2)

        attn = torch.softmax(theta.transpose(1, 2) @ phi, dim=-1)
        y = (g @ attn.transpose(1, 2)).view(B, -1, T, H, W)

        y = self.W(y) + x_reshaped
        return y.permute(0, 2, 1, 3, 4).reshape(BT, C, H, W)


def insert_nonlocal(net, n_segment):
    """Insère un NL block après le 2e bloc du layer3 (pratique standard TSM+NL)."""
    layer3 = net.layer3
    # Récupère les canaux de sortie
    sample_block = layer3[0].block if hasattr(layer3[0], 'block') else layer3[0]
    if hasattr(sample_block, 'conv3'):       # Bottleneck
        out_ch = sample_block.conv3.out_channels
    else:                                     # BasicBlock
        out_ch = sample_block.conv2.out_channels

    nl = NonLocalBlock(out_ch, n_segment)
    new_layer3 = nn.Sequential(
        layer3[0], layer3[1], nl, *layer3[2:]
    )
    net.layer3 = new_layer3

class TSM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_segment: int = 4,
        fold_div: int = 4,
        dropout: float = 0.5,
        n_resnet_layers: int = 50,
        residual_shift: bool = True,     
        temporal_pool: str = "attention",
        use_nonlocal: bool = True,
        use_frame_diff=False,             
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

        if use_frame_diff :
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=6,                              # <-- 6 au lieu de 3
                out_channels=old_conv.out_channels,         # 64
                kernel_size=old_conv.kernel_size,           # type:ignore 
                stride=old_conv.stride,                     # type:ignore 
                padding=old_conv.padding,                   # type:ignore 
                bias=False,
            )

        make_temporal_shift(
            backbone,
            n_segment=n_segment,
            fold_div=fold_div,
            residual=residual_shift,        # <-- propagation
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity() #type:ignore 
        self.backbone = backbone
        self.temporal_pool = temporal_pool

        if temporal_pool == "attention":
            self.attn = nn.Linear(in_features, 1)  # pour calculer les poids d'attention

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        if use_nonlocal:
            insert_nonlocal(self.backbone, n_segment)
        
        self.n_segment = n_segment  # Stocker n_segment pour les shifts dynamiques
        self.use_frame_diff = use_frame_diff
    

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clips.shape
        
        # Mettre à jour n_segment de tous les TemporalShift à la volée
        if T != self.n_segment:
            self.n_segment = T
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                for block in getattr(self.backbone, layer_name):
                    block.shift.n_segment = T
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                for block in getattr(self.backbone, layer_name):
                    if isinstance(block, NonLocalBlock):
                        block.n_segment = T
        
        if self.use_frame_diff:
            diff = clips[:, 1:] - clips[:, :-1]
            diff = torch.cat([torch.zeros_like(clips[:, :1]), diff], dim=1)
            clips = torch.cat([clips, diff], dim=2)              # (B, T, 6, H, W)
            C=6

        x = clips.reshape(B * T, C, H, W)
        feats = self.backbone(x)
        feats = self.backbone(x).view(B, T, -1)

        if self.temporal_pool == "mean":
            feats = feats.mean(dim=1)
        elif self.temporal_pool == "attention":
            w = torch.softmax(self.attn(feats), dim=1)  # (B, T, 1)
            feats = (feats * w).sum(dim=1)              # (B, D)
        elif self.temporal_pool == "last":
            feats = feats[:, -1]  

        return self.head(feats)                    # (B, num_classes) 
    

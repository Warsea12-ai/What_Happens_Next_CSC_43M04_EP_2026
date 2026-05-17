"""
TSM — Temporal Shift Module
Lin, Gan, Han, ICCV 2019
arXiv:1811.08383

Version adaptée pour from-scratch sur ~50k clips de 4 frames :
- Backbone léger par défaut (R18/R34)
- Stochastic depth dans les blocs résiduels
- Tête simple à 1 couche
- Régularisation forte par défaut

Format d'entrée : (B, T, C, H, W)
Format de sortie : (B, num_classes)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.nn.functional as F


# =============================================================================
# Temporal Shift
# =============================================================================
class TemporalShift(nn.Module):
    """
    Décale 1/fold_div des canaux le long de l'axe temporel.
    """
    def __init__(self, n_segment: int = 4, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                  # shift past
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]      # shift future
        out[:, :, 2*fold:] = x[:, :, 2*fold:]                 # unchanged

        return out.view(nt, c, h, w)


# =============================================================================
# Residual TSM blocks avec stochastic depth
# =============================================================================
class ResidualTSMBasicBlock(nn.Module):
    def __init__(self, basic_block: BasicBlock, n_segment: int,
                 fold_div: int = 8, drop_prob: float = 0.0):
        super().__init__()
        self.shift = TemporalShift(n_segment=n_segment, fold_div=fold_div)
        self.block = basic_block
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x if self.block.downsample is None else self.block.downsample(x)

        out = self.shift(x)
        out = self.block.conv1(out); out = self.block.bn1(out); out = self.block.relu(out)
        out = self.block.conv2(out); out = self.block.bn2(out)

        if self.training and self.drop_prob > 0.0:
            # Bernoulli par échantillon, asynchrone, avec rescaling
            keep = 1.0 - self.drop_prob
            B = out.shape[0]  # ici B = n_batch * n_segment, OK pour la régul
            mask = torch.empty(B, 1, 1, 1, device=out.device, dtype=out.dtype).bernoulli_(keep)
            out = out * mask / keep

        return F.relu(out + identity)


class ResidualTSMBottleneck(nn.Module):
    def __init__(self, bottleneck: Bottleneck, n_segment: int,
                 fold_div: int = 8, drop_prob: float = 0.0):
        super().__init__()
        self.shift = TemporalShift(n_segment=n_segment, fold_div=fold_div)
        self.block = bottleneck
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        if self.training and self.drop_prob > 0.0:
            if torch.rand(1, device=x.device).item() < self.drop_prob:
                return F.relu(identity)

        out = self.shift(x)
        out = self.block.conv1(out)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)
        out = self.block.conv3(out)
        out = self.block.bn3(out)

        out = out + identity                   # <-- pas de +=
        return F.relu(out)                     # <-- relu non-inplace


def make_temporal_shift(net: nn.Module, n_segment: int, fold_div: int = 8,
                        max_drop_prob: float = 0.0) -> None:
    """
    Wrappe tous les blocs résiduels du backbone. Le drop_prob augmente
    linéairement de 0 à max_drop_prob avec la profondeur (linear scaling rule
    de stochastic depth, Huang et al. 2016).
    """
    # Compter le nombre total de blocs pour répartir le drop_prob
    all_blocks = []
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        for block in getattr(net, layer_name):
            all_blocks.append((layer_name, block))
    total = len(all_blocks)

    idx = 0
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(net, layer_name)
        new_blocks = []
        for block in layer:
            # drop_prob linéaire : 0 au début, max_drop_prob à la fin
            dp = max_drop_prob * idx / max(total - 1, 1)
            if isinstance(block, Bottleneck):
                wrapped = ResidualTSMBottleneck(block, n_segment, fold_div, dp)
            elif isinstance(block, BasicBlock):
                wrapped = ResidualTSMBasicBlock(block, n_segment, fold_div, dp)
            else:
                raise TypeError(f"Bloc inattendu : {type(block).__name__}")
            new_blocks.append(wrapped)
            idx += 1
        setattr(net, layer_name, nn.Sequential(*new_blocks))


# =============================================================================
# Non-Local block (optionnel)
# =============================================================================
class NonLocalBlock(nn.Module):
    """Non-local block (Wang et al. 2018), embedded Gaussian."""
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
        nn.init.zeros_(self.W[0].weight)  # type: ignore # init = identité

    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.n_segment
        B = BT // T
        x_reshaped = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        theta = self.theta(x_reshaped).flatten(2)
        phi   = self.phi(x_reshaped).flatten(2)
        g     = self.g(x_reshaped).flatten(2)

        attn = torch.softmax(theta.transpose(1, 2) @ phi, dim=-1)
        y = (g @ attn.transpose(1, 2)).view(B, -1, T, H, W)

        y = self.W(y) + x_reshaped
        return y.permute(0, 2, 1, 3, 4).reshape(BT, C, H, W)


def insert_nonlocal(net, n_segment):
    """Insère un NL block après le 2e bloc du layer3."""
    layer3 = net.layer3
    sample_block = layer3[0].block if hasattr(layer3[0], 'block') else layer3[0]
    if hasattr(sample_block, 'conv3'):       # Bottleneck
        out_ch = sample_block.conv3.out_channels
    else:                                     # BasicBlock
        out_ch = sample_block.conv2.out_channels

    nl = NonLocalBlock(out_ch, n_segment)
    new_layer3 = nn.Sequential(layer3[0], layer3[1], nl, *layer3[2:])
    net.layer3 = new_layer3


# =============================================================================
# Positional encoding temporel
# =============================================================================
class TemporalPositionalEncoding(nn.Module):
    """PE 1D sur l'axe temporel des features (B, T, D)."""
    def __init__(self, d_model: int, max_len: int = 32, mode: str = "learned"):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        if mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        elif mode == "learned":
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            raise ValueError(f"Unknown PE mode: {mode}")

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        T = feats.size(1)
        return feats + self.pe[:, :T]


# =============================================================================
# Modèle principal
# =============================================================================
class TSM_s(nn.Module):
    """
    TSM avec backbone ResNet, conçu pour from-scratch sur petits datasets.
    
    Defaults adaptés à ~50k clips de 4 frames :
    - R18 (léger, évite l'overfit)
    - fold_div=8 (préserve la majorité des canaux pour features spatiales)
    - stochastic depth = 0.1 (régularise sans casser l'apprentissage)
    - tête simple (1 Linear)
    - Non-Local désactivé (à réactiver si la baseline marche)
    """
    def __init__(
        self,
        num_classes: int,
        n_segment: int = 4,
        fold_div: int = 8,
        dropout: float = 0.5,
        n_resnet_layers: int = 34,
        temporal_pool: str = "attention",
        use_nonlocal: bool = False,
        use_frame_diff: bool = True,
        use_positional_encoding: bool = True,
        pe_mode: str = "learned",
        stochastic_depth: float = 0.1,
        head_hidden: bool = False,
        pretrained_backbone_path: Optional[str] = None,
    ):
        super().__init__()
        self.n_segment = n_segment
        self.use_frame_diff = use_frame_diff
        self.use_positional_encoding = use_positional_encoding
        self.temporal_pool = temporal_pool

        backbone_factory = {
            18: resnet18, 34: resnet34, 50: resnet50,
            101: resnet101, 152: resnet152,
        }

        if n_resnet_layers not in backbone_factory:
            raise ValueError(f"Unsupported n_resnet_layers: {n_resnet_layers}")
            
        backbone = backbone_factory[n_resnet_layers](weights=None)

        # ---- Conv1 modifiée si frame-diff ---------------------------------
        if use_frame_diff:
            old_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels=6,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,           # type: ignore
                stride=old_conv.stride,                     # type: ignore
                padding=old_conv.padding,                   # type: ignore
                bias=False,
            )

        # ---- Insertion des temporal shifts + stochastic depth -------------
        make_temporal_shift(
            backbone,
            n_segment=n_segment,
            fold_div=fold_div,
            max_drop_prob=stochastic_depth,
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # type: ignore
        self.backbone = backbone

        # ---- Load SSL-pretrained backbone (optional) -----------------------
        if pretrained_backbone_path is not None:
            import os, torch as _torch
            if not os.path.exists(pretrained_backbone_path):
                raise FileNotFoundError(
                    f"SSL backbone not found: {pretrained_backbone_path}\n"
                    "Run pretraining first:\n"
                    "  python pretrain_ssl_trackA.py --arch resnet18 --use_tsm "
                    "--epochs 30 --out ssl_tsm_backbone.pt"
                )
            ckpt = _torch.load(pretrained_backbone_path, map_location="cpu",
                               weights_only=False)
            self.backbone.load_state_dict(ckpt["backbone_state_dict"])
            print(f"Loaded TSM SSL backbone from {pretrained_backbone_path} "
                  f"(val_acc={ckpt.get('val_acc', '?'):.4f})")

        # ---- Non-Local (optionnel) ----------------------------------------
        if use_nonlocal:
            insert_nonlocal(self.backbone, n_segment)

        # ---- Positional encoding temporel ---------------------------------
        if use_positional_encoding:
            self.pos_encoding = TemporalPositionalEncoding(
                d_model=in_features,
                max_len=32,
                mode=pe_mode,
            )

        # ---- Pooling temporel ---------------------------------------------
        if temporal_pool == "attention":
            self.attn = nn.Linear(in_features, 1)

        # ---- Tête de classification ---------------------------------------
        if temporal_pool == "motion":
            # Motion-aware head: [global, first, last, delta] → 4*D → 512 → num_classes
            # Separate LayerNorms keep each feature type well-scaled before concat.
            self.norm_global = nn.LayerNorm(in_features)
            self.norm_first  = nn.LayerNorm(in_features)
            self.norm_last   = nn.LayerNorm(in_features)
            self.norm_delta  = nn.LayerNorm(in_features)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(4 * in_features, 512),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(512, num_classes),
            )
        elif head_hidden:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(in_features // 2, num_classes),
            )
        else:
            # Tête simple recommandée from-scratch sur petit dataset
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

    # -----------------------------------------------------------------------
    def _update_n_segment(self, T: int) -> None:
        """Met à jour n_segment dans tous les modules concernés (TSM, NL)."""
        self.n_segment = T
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            for block in getattr(self.backbone, layer_name):
                if hasattr(block, "shift"):
                    block.shift.n_segment = T
                if isinstance(block, NonLocalBlock):
                    block.n_segment = T

    # -----------------------------------------------------------------------
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clips.shape

        if T != self.n_segment:
            self._update_n_segment(T)

        # ---- Frame difference --------------------------------------------
        if self.use_frame_diff:
            diff = clips[:, 1:] - clips[:, :-1]
            diff = torch.cat([torch.zeros_like(clips[:, :1]), diff], dim=1)
            clips = torch.cat([clips, diff], dim=2)        # (B, T, 6, H, W)
            C = 6

        # ---- Backbone -----------------------------------------------------
        x = clips.reshape(B * T, C, H, W)
        feats = self.backbone(x).view(B, T, -1)            # (B, T, D)

        # ---- PE temporel --------------------------------------------------
        if self.use_positional_encoding:
            feats = self.pos_encoding(feats)

        # ---- Pooling temporel --------------------------------------------
        if self.temporal_pool == "mean":
            feats = feats.mean(dim=1)
        elif self.temporal_pool == "attention":
            w = torch.softmax(self.attn(feats), dim=1)     # (B, T, 1)
            feats = (feats * w).sum(dim=1)                 # (B, D)
        elif self.temporal_pool == "last":
            feats = feats[:, -1]
        elif self.temporal_pool == "motion":
            global_f = feats.mean(dim=1)                   # (B, D) — scene context
            first    = feats[:, 0]                         # (B, D) — initial state
            last     = feats[:, -1]                        # (B, D) — final state
            delta    = last - first                        # (B, D) — net motion
            feats = torch.cat([
                self.norm_global(global_f),
                self.norm_first(first),
                self.norm_last(last),
                self.norm_delta(delta),
            ], dim=1)                                      # (B, 4*D)
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")

        return self.head(feats)                            # (B, num_classes)
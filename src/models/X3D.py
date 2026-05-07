"""
X3D — Expanding Architectures for Efficient Video Recognition
Christoph Feichtenhofer, CVPR 2020
arXiv:2004.04730

Version améliorée pour le challenge "What happens next?" sur SSv2 :
    - Variante XS spécialement conçue pour 4 frames
    - Train-from-scratch friendly (pas de pretrain requis)
    - Améliorations ajoutées par rapport au wrapper de base :
        1. Channel attention (SE-like) en sortie pour pondérer les features
        2. Temporal attention pooling apprenable au lieu du global avg
        3. Dual-head : prédiction principale + auxiliary head sur la moyenne
           temporelle (régularise et stabilise la convergence from scratch)
        4. Frame difference en entrée optionnelle (capture le mouvement
           explicitement, important pour SSv2)
        5. DropPath (stochastic depth) pour la régularisation
        6. Initialization adaptée from-scratch (zero-init des dernières BN)

Format d'entrée : (B, T, C, H, W)
Format de sortie : logits (B, num_classes)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.x3d import create_x3d


# ======================================================================
#   MODULES AUXILIAIRES
# ======================================================================
class SqueezeExcite3D(nn.Module):
    """Channel attention type Squeeze-and-Excitation, en 3D.
    
    Apprend à pondérer chaque canal selon le contenu spatio-temporel
    global de la feature map. Très efficace pour les modèles vidéo
    et léger en paramètres.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, T, H, W) ou (B, C) selon où on l'insère
        if x.dim() == 5:
            squeeze = x.mean(dim=(2, 3, 4))      # (B, C)
        else:
            squeeze = x
        excite = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze), inplace=True)))
        if x.dim() == 5:
            return x * excite.view(x.size(0), -1, 1, 1, 1)
        return x * excite


class DropPath(nn.Module):
    """Stochastic depth : drop le chemin résiduel avec probabilité drop_prob."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask


# ======================================================================
#   X3D AMÉLIORÉ
# ======================================================================
class X3D(nn.Module):
    """
    X3D-XS adapté et amélioré pour le challenge SSv2 (4 frames, 33 classes).
    
    Args
    ----
    num_classes : int
        Nombre de classes (33 pour le challenge).
    variant : str
        "xs" (4 frames, ~3M params), "s" (13), "m" (16).
    input_clip_length : int
        Nombre de frames T en entrée. Doit matcher le variant choisi.
    input_crop_size : int
        Taille spatiale H=W (160 pour XS, 224 pour M).
    dropout : float
        Dropout dans la tête de classification.
    use_se : bool
        Active le Squeeze-Excite channel attention en sortie.
    use_temporal_attention : bool
        Remplace le global avg pool par un pooling temporel apprenable.
    use_aux_head : bool
        Ajoute une 2e tête de classification (régularisation).
        Renvoie alors un tuple (logits_main, logits_aux) en mode training.
    use_frame_diff : bool
        Concatène la différence de frames consécutives aux 3 canaux RGB.
        Très efficace sur SSv2 (focus mouvement).
    drop_path_rate : float
        Stochastic depth pour régulariser. 0.1 raisonnable from-scratch.
    """
    def __init__(
        self,
        num_classes: int,
        variant: str = "xs",
        input_clip_length: int = 4,
        input_crop_size: int = 160,
        dropout: float = 0.5,
        use_se: bool = True,
        use_temporal_attention: bool = True,
        use_aux_head: bool = True,
        use_frame_diff: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_frames = input_clip_length
        self.use_se = use_se
        self.use_temporal_attention = use_temporal_attention
        self.use_aux_head = use_aux_head
        self.use_frame_diff = use_frame_diff

        # ---- Configs officielles X3D du paper ----
        configs = {
            "xs": dict(input_clip_length=4,  input_crop_size=160,
                       depth_factor=2.2, width_factor=2.0, bottleneck_factor=2.25),
            "s":  dict(input_clip_length=13, input_crop_size=160,
                       depth_factor=2.2, width_factor=2.0, bottleneck_factor=2.25),
            "m":  dict(input_clip_length=16, input_crop_size=224,
                       depth_factor=2.2, width_factor=2.0, bottleneck_factor=2.25),
        }
        cfg = configs[variant]

        # ---- Channels d'entrée : 3 (RGB) ou 6 (RGB + frame diff) ----
        in_channels = 6 if use_frame_diff else 3

        # ---- Backbone X3D ----
        self.backbone = create_x3d(
            input_channel=in_channels,
            input_clip_length=input_clip_length,
            input_crop_size=input_crop_size,
            model_num_class=num_classes,
            dropout_rate=dropout,
            width_factor=cfg["width_factor"],
            depth_factor=cfg["depth_factor"],
            bottleneck_factor=cfg["bottleneck_factor"],
        )

        # On va remplacer la tête finale du backbone par notre propre pipeline
        # pour pouvoir insérer SE / temporal attention / aux head.
        # Mais d'abord, on doit récupérer la dimension des features pré-tête.
        # Dans X3D créé par PyTorchVideo, la dernière partie est un "head"
        # qui contient pool -> dropout -> proj. On la remplace par Identity
        # et on prend la sortie juste avant.
        head = self.backbone.blocks[-1] #type:ignore 
        # On veut : la projection finale (proj.weight.shape[1]) = embed_dim
        feature_dim = head.proj.weight.shape[1] #type:ignore 
        self.feature_dim = feature_dim

        # On désactive la tête originale : on travaille à partir des feature maps
        # pré-pooling. Pour ça, on bypasse le head dans forward.
        self.backbone_head = head        # gardé pour debug, pas utilisé directement

        # ---- Squeeze-Excite channel attention (optionnel) ----
        if use_se:
            self.se = SqueezeExcite3D(feature_dim, reduction=4)

        # ---- Temporal attention pour le pooling apprenable ----
        if use_temporal_attention:
            self.temporal_attn = nn.Linear(feature_dim, 1)

        # ---- DropPath ----
        self.drop_path = DropPath(drop_path_rate)

        # ---- Tête principale ----
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # ---- Tête auxiliaire (régularisation from-scratch) ----
        if use_aux_head:
            self.aux_classifier = nn.Linear(feature_dim, num_classes)

        self._init_new_params()

    def _init_new_params(self):
        """Init adaptée from-scratch : zero-init des têtes pour démarrer
        avec des prédictions uniformes (cross-entropy initiale = log(N))."""
        nn.init.trunc_normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        if self.use_aux_head:
            nn.init.trunc_normal_(self.aux_classifier.weight, std=0.01)
            nn.init.zeros_(self.aux_classifier.bias)
        if self.use_temporal_attention:
            nn.init.zeros_(self.temporal_attn.weight)
            nn.init.zeros_(self.temporal_attn.bias)

    # ------------------------------------------------------------------
    def _add_frame_diff(self, clips: torch.Tensor) -> torch.Tensor:
        """Ajoute 3 canaux de différence de frames consécutives.
        
        clips : (B, T, 3, H, W) -> (B, T, 6, H, W)
        Frame 0 : diff = 0 (pas de frame précédente).
        """
        diff = clips[:, 1:] - clips[:, :-1]
        diff = torch.cat([torch.zeros_like(clips[:, :1]), diff], dim=1)
        return torch.cat([clips, diff], dim=2)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward dans le backbone jusqu'à la feature map pré-tête.
        
        x : (B, C, T, H, W)
        retour : (B, feature_dim, T', H', W')
        """
        # On itère sur tous les blocs SAUF le dernier (le head)
        for block in self.backbone.blocks[:-1]: #type:ignore 
            x = block(x) #type:ignore 
        return x

    # ------------------------------------------------------------------
    def forward(self, clips: torch.Tensor):
        """
        clips : (B, T, C, H, W)
        retour :
            - mode eval : logits (B, num_classes)
            - mode train avec aux_head : (logits_main, logits_aux)
        """
        B, T, C, H, W = clips.shape
        assert T == self.num_frames, (
            f"X3D attend T={self.num_frames} frames, reçu T={T}"
        )

        # 1. Frame difference (optionnel)
        if self.use_frame_diff:
            clips = self._add_frame_diff(clips)            # (B, T, 6, H, W)

        # 2. Permute pour PyTorchVideo : (B, C, T, H, W)
        x = clips.permute(0, 2, 1, 3, 4).contiguous()

        # 3. Backbone -> feature map (B, D, T', H', W')
        feat_map = self._extract_features(x)

        # 4. Squeeze-Excite channel attention
        if self.use_se:
            feat_map = self.se(feat_map)

        # 5. Spatial global pooling
        # (B, D, T', H', W') -> (B, D, T')
        feat_temporal = feat_map.mean(dim=(3, 4))          # spatial avg

        # 6. Temporal pooling : attention apprenable ou simple moyenne
        if self.use_temporal_attention:
            # (B, D, T') -> (B, T', D)
            feat_t = feat_temporal.transpose(1, 2)
            scores = self.temporal_attn(feat_t)             # (B, T', 1)
            weights = torch.softmax(scores, dim=1)
            features = (feat_t * weights).sum(dim=1)        # (B, D)
        else:
            features = feat_temporal.mean(dim=2)             # (B, D)

        # 7. DropPath sur la représentation finale (régularisation)
        features = self.drop_path(features)

        # 8. Tête principale
        features_dropped = self.dropout(features)
        logits_main = self.classifier(features_dropped)     # (B, num_classes)

        # 9. Tête auxiliaire (sur la moyenne temporelle pure, pas l'attention)
        if self.use_aux_head and self.training:
            features_aux = feat_temporal.mean(dim=2)         # (B, D)
            logits_aux = self.aux_classifier(self.dropout(features_aux))
            return logits_main, logits_aux

        return logits_main


# ======================================================================
#   LOSS UTILITAIRE POUR LE DUAL-HEAD
# ======================================================================
def x3d_loss(outputs, labels: torch.Tensor,
             loss_fn: nn.Module, aux_weight: float = 0.3) -> torch.Tensor:
    """
    Combine main loss et auxiliary loss pour la régularisation.
    
    Si outputs est un tuple, on calcule :
        L_total = L_main + aux_weight * L_aux
    Sinon (eval mode), on calcule juste L_main.
    
    Usage dans la boucle d'entraînement :
        outputs = model(clips)
        loss = x3d_loss(outputs, labels, loss_fn, aux_weight=0.3)
    """
    if isinstance(outputs, tuple):
        logits_main, logits_aux = outputs
        loss_main = loss_fn(logits_main, labels)
        loss_aux  = loss_fn(logits_aux, labels)
        return loss_main + aux_weight * loss_aux
    return loss_fn(outputs, labels)


def x3d_get_logits(outputs):
    """Extrait les logits principaux d'une sortie X3D (qui peut être tuple
    en training avec aux head)."""
    return outputs[0] if isinstance(outputs, tuple) else outputs


# ======================================================================
#   TEST RAPIDE
# ======================================================================
if __name__ == "__main__":
    print("=== X3D-XS amélioré ===")
    model = X3D(
        num_classes=33,
        variant="xs",
        input_clip_length=4,
        input_crop_size=160,
        use_se=True,
        use_temporal_attention=True,
        use_aux_head=True,
        use_frame_diff=False,
        drop_path_rate=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_params / 1e6:.2f}M")

    x = torch.randn(2, 4, 3, 160, 160)

    # Mode training : tuple
    model.train()
    out_train = model(x)
    print(f"Train mode : main={out_train[0].shape}, aux={out_train[1].shape}")

    # Mode eval : tensor seul
    model.eval()
    out_eval = model(x)
    print(f"Eval mode  : {out_eval.shape}")

    # Test avec frame diff
    print("\n=== X3D-XS avec frame diff ===")
    model_fd = X3D(
        num_classes=33, variant="xs",
        input_clip_length=4, input_crop_size=160,
        use_frame_diff=True,
    )
    n_params_fd = sum(p.numel() for p in model_fd.parameters() if p.requires_grad)
    print(f"Params : {n_params_fd / 1e6:.2f}M")
    out_fd = model_fd(x)
    if isinstance(out_fd, tuple):
        print(f"Output  : main={out_fd[0].shape}")
    else:
        print(f"Output  : {out_fd.shape}")

    # Test du loss combiné
    print("\n=== Loss combiné ===")
    labels = torch.randint(0, 33, (2,))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    outputs = model(x)
    loss = x3d_loss(outputs, labels, loss_fn, aux_weight=0.3)
    print(f"Loss combiné : {loss.item():.4f}")
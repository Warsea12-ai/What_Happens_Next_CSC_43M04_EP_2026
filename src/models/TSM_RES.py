"""
TSM — Temporal Shift Module for Efficient Video Understanding
Ji Lin, Chuang Gan, Song Han, ICCV 2019
arXiv:1811.08383

Implémentation from-scratch friendly pour le challenge SSv2 :
- Backbone ResNet-50 (torchvision, init from scratch)
- Temporal Shift Module inséré dans chaque bottleneck (1/8 des canaux shifted)
- Format d'entrée : (B, T, C, H, W)
- Format de sortie : logits (B, num_classes)

Le shift n'a aucun paramètre : il déplace simplement 1/8 des canaux
d'une frame vers la suivante (forward shift) et 1/8 vers la précédente
(backward shift). Cela force le modèle à apprendre des features
spatio-temporelles sans coût additionnel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ======================================================================
#   TEMPORAL SHIFT MODULE
# ======================================================================
class TemporalShift(nn.Module):
    """Shift 1/n_div des canaux dans le temps (sans paramètres).
    
    Args
    ----
    num_segments : int
        Nombre T de frames par clip (ici 4).
    n_div : int
        Fraction des canaux à shifter. 8 = 1/8 forward + 1/8 backward
        + 6/8 inchangés. Valeur par défaut du papier.
    inplace : bool
        Shift in-place pour économiser la mémoire. False par sécurité.
    """
    def __init__(self, num_segments: int = 4, n_div: int = 8, inplace: bool = False):
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = n_div
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrive en (B*T, C, H, W) — on doit reshape en (B, T, C, H, W)
        nt, c, h, w = x.size()
        n_batch = nt // self.num_segments
        x = x.view(n_batch, self.num_segments, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)

        # Shift forward : canaux [0:fold] -> frame suivante
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # Shift backward : canaux [fold:2*fold] -> frame précédente
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        # Reste inchangé
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)


# ======================================================================
#   PATCH ResNet : insertion du TSM dans chaque bottleneck
# ======================================================================
def make_temporal_shift(net: nn.Module, num_segments: int, n_div: int = 8,
                        place: str = "blockres") -> None:
    """Insère un TemporalShift au début de chaque bottleneck du ResNet.
    
    Stratégie 'blockres' (recommandée par le papier) : on insère le shift
    dans la branche résiduelle (avant conv1 du bottleneck), donc le shift
    est dans le residual path et non sur l'identity. Ça évite de dégrader
    les features spatiales pures.
    """
    def make_block_temporal(stage: nn.Module, this_segment: int):
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            # Wrap conv1 avec un TSM en amont
            blocks[i].conv1 = nn.Sequential(
                TemporalShift(num_segments=this_segment, n_div=n_div),
                b.conv1,
            )
        return nn.Sequential(*blocks)

    net.layer1 = make_block_temporal(net.layer1, num_segments)
    net.layer2 = make_block_temporal(net.layer2, num_segments)
    net.layer3 = make_block_temporal(net.layer3, num_segments)
    net.layer4 = make_block_temporal(net.layer4, num_segments)


# ======================================================================
#   TSM-ResNet50
# ======================================================================
class TSMResNet50(nn.Module):
    """TSM avec backbone ResNet-50, adapté pour le challenge SSv2.
    
    Args
    ----
    num_classes : int
        Nombre de classes (33 pour le challenge).
    num_segments : int
        Nombre T de frames par clip (4 pour le challenge).
    n_div : int
        Fraction des canaux shifted (8 = 1/8, recommandé).
    dropout : float
        Dropout dans la tête de classification.
    pretrained : bool
        Init ImageNet pour le backbone 2D. Track A = from scratch -> False.
        En pratique, même pour le track A, le pretrain ImageNet 2D est
        souvent toléré et donne un gain massif. À vérifier dans tes règles.
    consensus : str
        Comment agréger les T prédictions par frame. "avg" (moyenne) est
        standard. "linear" apprend une combinaison linéaire pondérée.
    partial_bn : bool
        Freeze les BN sauf la première (utile en fine-tuning ImageNet,
        moins en from-scratch — laisse à False).
    """
    def __init__(
        self,
        num_classes: int = 33,
        num_segments: int = 4,
        n_div: int = 8,
        dropout: float = 0.5,
        pretrained: bool = False,
        consensus: str = "avg",
        partial_bn: bool = False,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.consensus_type = consensus
        self.partial_bn = partial_bn

        # ---- Backbone ResNet-50 ----
        weights = "IMAGENET1K_V2" if pretrained else None
        backbone = resnet50(weights=weights)

        # ---- Insertion du TSM dans chaque bottleneck ----
        make_temporal_shift(backbone, num_segments=num_segments, n_div=n_div)

        # ---- Récupération de la dim feature avant classif ----
        feature_dim = backbone.fc.in_features  # 2048 pour ResNet-50

        # ---- Remplacement de la tête FC ----
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.backbone = backbone

        # ---- Consensus apprenable (optionnel) ----
        if consensus == "linear":
            self.consensus_weights = nn.Parameter(torch.ones(num_segments) / num_segments)

        self._init_new_params()

    def _init_new_params(self):
        """Init des têtes : trunc_normal sur le linear, biais à zéro."""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def train(self, mode: bool = True):
        """Override train() pour gérer le partial BN si activé."""
        super().train(mode)
        if mode and self.partial_bn:
            count = 0
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= 2:
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        return self

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips : (B, T, C, H, W)
        retour : logits (B, num_classes)
        """
        B, T, C, H, W = clips.shape
        assert T == self.num_segments, (
            f"TSM attend T={self.num_segments} frames, reçu T={T}"
        )

        # Reshape en (B*T, C, H, W) pour passer dans le ResNet 2D
        # Le TSM s'occupe du shift temporel à chaque bottleneck
        x = clips.view(B * T, C, H, W)

        # Forward dans le ResNet (avec shifts temporels insérés)
        per_frame_logits = self.backbone(x)            # (B*T, num_classes)

        # Reshape en (B, T, num_classes)
        per_frame_logits = per_frame_logits.view(B, T, -1)

        # Consensus temporel : moyenne ou combinaison apprenable
        if self.consensus_type == "avg":
            logits = per_frame_logits.mean(dim=1)      # (B, num_classes)
        elif self.consensus_type == "linear":
            weights = F.softmax(self.consensus_weights, dim=0)
            logits = (per_frame_logits * weights.view(1, -1, 1)).sum(dim=1)
        else:
            raise ValueError(f"consensus inconnu : {self.consensus_type}")

        return logits


# ======================================================================
#   TEST RAPIDE
# ======================================================================
if __name__ == "__main__":
    print("=== TSM-ResNet50 from scratch ===")
    model = TSMResNet50(
        num_classes=33,
        num_segments=4,
        n_div=8,
        dropout=0.5,
        pretrained=False,
        consensus="avg",
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_params / 1e6:.2f}M")

    x = torch.randn(2, 4, 3, 224, 224)
    out = model(x)
    print(f"Output shape : {out.shape}")              # (2, 33)

    # Test loss
    labels = torch.randint(0, 33, (2,))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss = loss_fn(out, labels)
    print(f"Loss : {loss.item():.4f}")

    # Test backward
    loss.backward()
    print("Backward OK")
"""
Padding Module — Learning the Padding in Deep Neural Networks
Alrasheedi, Zhong, Huang, IEEE Access 2023
arXiv:2301.04608

Architecture du module fidèle au papier :
    - Ground truth T : stack des 4 bords de l'image (top, bottom, left, right)
      dans une matrice (4, max_dim).
    - Predictor input P : stack des 4 "neighbors" (les pixels juste à l'intérieur
      des bords) avec du padding latéral (reflection à gauche, zero à droite).
    - Un petit CNN apprend P -> T (self-supervised, L1 loss).

Pour le cas d'usage RandomRotation, on l'applique itérativement pour grossir
l'image de N pixels, puis on rotate et on center-crop.

Fichier : src/models/padding_module.py
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
#   PADDING MODULE (paper-faithful)
# =====================================================================
class PaddingModule(nn.Module):
    """
    Prédit un ring de 1 pixel autour de l'image, à partir des bords existants.

    Args
    ----
    channels : int
        Nombre de canaux (3 pour RGB).
    hidden : int
        Nombre de canaux internes du CNN (64 par défaut, ~6k paramètres).
    context : int
        Largeur du contexte latéral ajouté à chaque "neighbor" avant la conv.
        Le paper utilise 8 (reflection padding à gauche, zero à droite).
    """
    def __init__(self, channels: int = 3, hidden: int = 64, context: int = 8):
        super().__init__()
        self.channels = channels
        self.context = context

        # Petit CNN 1D appliqué le long de la dimension "longueur du bord".
        # Le canal d'entrée combine : 1 bord (channels canaux) + 1 contexte.
        # Note : on traite chaque ligne du stack indépendamment, donc l'input
        # est (B*4, C, max_dim + 2*context) en passant 4 fois en parallèle.
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=3, padding=1),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_borders(img: torch.Tensor) -> torch.Tensor:
        """Extrait les 4 bords. img : (B, C, H, W) -> bords (B, 4, C, max_dim).

        Convention :
            row 0 = top   (longueur W)
            row 1 = bottom (longueur W)
            row 2 = left  (longueur H, retourné en 1D)
            row 3 = right (longueur H)
        Les bords plus courts que max_dim sont zero-padded à droite.
        """
        B, C, H, W = img.shape
        max_dim = max(H, W)

        borders = img.new_zeros(B, 4, C, max_dim)
        borders[:, 0, :, :W] = img[:, :,  0, :]                  # top
        borders[:, 1, :, :W] = img[:, :, -1, :]                  # bottom
        borders[:, 2, :, :H] = img[:, :, :,  0]                  # left
        borders[:, 3, :, :H] = img[:, :, :, -1]                  # right
        return borders                                            # (B, 4, C, max_dim)

    @staticmethod
    def _extract_neighbors(img: torch.Tensor) -> torch.Tensor:
        """Extrait les 4 'voisins' (pixels juste à l'intérieur des bords).
        img : (B, C, H, W) -> voisins (B, 4, C, max_dim).
        """
        B, C, H, W = img.shape
        max_dim = max(H, W)

        neighbors = img.new_zeros(B, 4, C, max_dim)
        neighbors[:, 0, :, :W] = img[:, :,  1,  :]                # voisin du top
        neighbors[:, 1, :, :W] = img[:, :, -2,  :]                # voisin du bottom
        neighbors[:, 2, :, :H] = img[:, :,  :,  1]                # voisin du left
        neighbors[:, 3, :, :H] = img[:, :,  :, -2]                # voisin du right
        return neighbors

    # ------------------------------------------------------------------
    def _predict_borders(self, neighbors: torch.Tensor) -> torch.Tensor:
        """neighbors : (B, 4, C, L) -> predicted borders (B, 4, C, L).

        On replie les 4 bords dans la dim batch, ajoute le contexte latéral
        (reflection à gauche, zero à droite), passe dans la conv1d, puis
        on enlève le contexte et on déplie.
        """
        B, four, C, L = neighbors.shape
        assert four == 4

        # (B, 4, C, L) -> (B*4, C, L)
        x = neighbors.reshape(B * 4, C, L)

        # Contexte latéral : reflection à gauche, zero à droite (cf. paper)
        ctx = self.context
        x = F.pad(x, (ctx, 0), mode="reflect")
        x = F.pad(x, (0, ctx), mode="constant", value=0.0)
        # x : (B*4, C, L + 2*ctx)

        pred = self.net(x)                                        # (B*4, C, L + 2*ctx)
        pred = pred[:, :, ctx:ctx + L]                            # enlève le contexte
        return pred.reshape(B, 4, C, L)                           # (B, 4, C, L)

    # ------------------------------------------------------------------
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img : (B, C, H, W)
        retour : (B, C, H+2, W+2) — image élargie d'un ring de 1 pixel.

        Étapes :
            1. Extrait les 4 bords (T) et leurs voisins (P).
            2. Le predictor estime T̂ à partir de P.
            3. On colle T̂ tout autour de l'image originale.
        """
        B, C, H, W = img.shape

        # 1. Extraire les voisins (pour prédire la couche externe)
        neighbors = self._extract_neighbors(img)                  # (B, 4, C, max(H,W))

        # 2. Prédire les bords élargis
        pred_borders = self._predict_borders(neighbors)           # (B, 4, C, max(H,W))

        # 3. Constituer l'image élargie
        # pred_borders[:, 0] -> nouveau top    (longueur W)
        # pred_borders[:, 1] -> nouveau bottom
        # pred_borders[:, 2] -> nouveau left   (longueur H, transposé)
        # pred_borders[:, 3] -> nouveau right
        new_top    = pred_borders[:, 0, :, :W]                    # (B, C, W)
        new_bottom = pred_borders[:, 1, :, :W]
        new_left   = pred_borders[:, 2, :, :H]                    # (B, C, H)
        new_right  = pred_borders[:, 3, :, :H]

        # Image élargie : (B, C, H+2, W+2)
        out = img.new_zeros(B, C, H + 2, W + 2)
        out[:, :, 1:H + 1, 1:W + 1] = img
        out[:, :, 0, 1:W + 1] = new_top
        out[:, :, H + 1, 1:W + 1] = new_bottom
        out[:, :, 1:H + 1, 0] = new_left
        out[:, :, 1:H + 1, W + 1] = new_right

        # Coins : moyenne des deux bords adjacents (le paper ne traite pas
        # explicitement les coins, c'est une approximation raisonnable)
        out[:, :, 0,   0]   = (new_top[:, :, 0]    + new_left[:, :, 0])    / 2
        out[:, :, 0,   W+1] = (new_top[:, :, -1]   + new_right[:, :, 0])   / 2
        out[:, :, H+1, 0]   = (new_bottom[:, :, 0] + new_left[:, :, -1])   / 2
        out[:, :, H+1, W+1] = (new_bottom[:, :, -1] + new_right[:, :, -1]) / 2

        return out

    # ------------------------------------------------------------------
    def self_supervised_loss(self, img: torch.Tensor) -> torch.Tensor:
        """
        Loss self-supervised (sans labels) sur une image (B, C, H, W).

        Principe : on considère les BORDS de l'image comme ground truth, et
        leurs VOISINS comme prédicteurs. Le module apprend à prédire les
        bords à partir des voisins → en inférence, il pourra prédire encore
        plus loin (un pixel au-dessus du bord) par extrapolation.

        Retour : scalar loss (L1).
        """
        borders   = self._extract_borders(img)                    # (B, 4, C, L)
        neighbors = self._extract_neighbors(img)                  # (B, 4, C, L)
        predicted = self._predict_borders(neighbors)              # (B, 4, C, L)

        return F.l1_loss(predicted, borders)


# =====================================================================
#   BORDER INPAINTER — applique le module N fois pour grossir l'image
# =====================================================================
class BorderInpainter(nn.Module):
    """
    Applique itérativement PaddingModule pour ajouter `pad_size` pixels
    de chaque côté de l'image.

    Pour le cas RandomRotation ±10° sur une image 224×224, le rayon est
    sqrt(2)/2 × 224 ≈ 158, et le décalage angulaire est 158 × sin(10°) ≈ 27.
    Donc pad_size ≈ 28 suffit pour couvrir le pire cas.
    """
    def __init__(self, padding_module: PaddingModule, pad_size: int = 28):
        super().__init__()
        self.module = padding_module
        self.pad_size = pad_size

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img : (B, C, H, W) -> (B, C, H + 2*pad_size, W + 2*pad_size)"""
        out = img
        for _ in range(self.pad_size):
            out = self.module(out)
        return out


# =====================================================================
#   TRANSFORM : ROTATION AVEC LEARNED PADDING
# =====================================================================
class LearnedRotation(nn.Module):
    """
    Transform vidéo qui :
        1. Étend l'image avec le PaddingModule pré-entraîné
        2. Applique une rotation aléatoire dans [-deg, +deg]
        3. Center-crop pour revenir à la taille originale

    Important : la MÊME rotation est appliquée aux T frames d'une même vidéo
    (cohérence temporelle). Le PaddingModule est gelé (eval mode) ici.

    Args
    ----
    padding_module : PaddingModule
        Module pré-entraîné qui prédit l'extension d'image.
    pad_size : int
        Nombre de pixels d'extension de chaque côté avant la rotation.
        Pour ±10°, 28 est confortable (pire cas ≈ 27).
    degrees : float
        Amplitude de rotation, en degrés. La rotation tirée est uniforme
        dans [-degrees, +degrees].
    """
    def __init__(self, padding_module: PaddingModule,
                 pad_size: int = 28, degrees: float = 10.0):
        super().__init__()
        self.padder = BorderInpainter(padding_module, pad_size)
        self.pad_size = pad_size
        self.degrees = degrees

        # On garde le padding_module en eval (ses BN sont gelées)
        padding_module.eval()
        for p in padding_module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        clips : (B, T, C, H, W)
        retour : (B, T, C, H, W) — même taille, contenu rotaté avec
                                    learned padding aux coins.
        """
        B, T, C, H, W = clips.shape
        device = clips.device

        # 1. Tirer un angle PAR VIDÉO (partagé sur les T frames)
        angles = (torch.rand(B, device=device) * 2 - 1) * self.degrees  # (B,)

        # 2. Replier les T frames dans le batch pour le padder (B*T, C, H, W)
        frames = clips.reshape(B * T, C, H, W)
        padded = self.padder(frames)                              # (B*T, C, H', W')
        Hp, Wp = padded.shape[-2:]

        # 3. Construire les matrices de rotation 2D (B,) -> (B, 2, 3) puis
        #    dupliquer sur les T frames
        cos = torch.cos(angles * math.pi / 180)
        sin = torch.sin(angles * math.pi / 180)

        # Affine matrix pour grid_sample (convention -theta vs theta)
        # On veut faire tourner l'image de +theta, donc on échantillonne
        # avec la matrice inverse (-theta).
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] =  cos
        theta[:, 0, 1] =  sin
        theta[:, 1, 0] = -sin
        theta[:, 1, 1] =  cos

        # Réplique sur les T frames : (B, 2, 3) -> (B*T, 2, 3)
        theta = theta.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, 2, 3)

        # 4. Échantillonner avec grid_sample
        grid = F.affine_grid(theta, size=padded.shape, align_corners=False)
        rotated = F.grid_sample(padded, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)

        # 5. Center-crop pour revenir à (H, W)
        top  = (Hp - H) // 2
        left = (Wp - W) // 2
        cropped = rotated[:, :, top:top + H, left:left + W]

        return cropped.reshape(B, T, C, H, W)


# =====================================================================
#   TEST RAPIDE
# =====================================================================
if __name__ == "__main__":
    # 1. Test du module
    pm = PaddingModule(channels=3, hidden=64)
    img = torch.rand(2, 3, 64, 64)
    out = pm(img)
    print(f"Input  : {tuple(img.shape)}")
    print(f"Output : {tuple(out.shape)}  (devrait être H+2, W+2)")

    # 2. Test de la loss self-supervised
    loss = pm.self_supervised_loss(img)
    print(f"SSL loss : {loss.item():.4f}")

    # 3. Test du BorderInpainter (28 pixels)
    inpainter = BorderInpainter(pm, pad_size=28)
    extended = inpainter(img)
    print(f"Padded image : {tuple(extended.shape)}  (devrait être H+56, W+56)")

    # 4. Test de la transform LearnedRotation
    transform = LearnedRotation(pm, pad_size=28, degrees=10.0)
    clips = torch.rand(2, 4, 3, 64, 64)
    rotated = transform(clips)
    print(f"Rotated clips : {tuple(rotated.shape)}  (devrait être identique)")
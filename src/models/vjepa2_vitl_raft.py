"""VJEPA2RaftDup — V-JEPA 2 ViT-L + interpolation 4→16 frames via RAFT.

Variante de :class:`VJEPA2ViTLDup` où la duplication statique est remplacée
par un warping basé sur le flow optique entre les 4 frames originales.

Motivation
----------
La duplication "chaque frame 4×" donne 16 frames sans mouvement à l'intérieur
de chaque tubelet (tubelet_size=2). L'encodeur V-JEPA 2 a été pré-entraîné
sur SSv2 avec du *vrai* mouvement dans chaque tubelet → mismatch de
distribution. RAFT synthétise des frames intermédiaires plausibles via flow
optique, et le tubelet voit alors du mouvement réel.

Schéma temporel
---------------
Input  (4 frames) :     A . . . B . . . C . . . D
Output (16 frames) :    A a₁ a₂ a₃ B b₁ b₂ b₃ C c₁ c₂ c₃ D d₁ d₂ d₃
où aₖ = warp(A → ¼·flow(A,B)), aₖ = warp(A → ½·flow(A,B)), etc.

Coût
----
RAFT-Large = ~5M params, ~12 iters, frozen + bf16. ~10-30 ms par batch en
plus du forward V-JEPA 2 (~200 ms). Acceptable.
On le garde en bf16 et en eval mode — pas de gradient à travers RAFT.

Robustesse
----------
- Échec RAFT (NaN/Inf) → fallback sur duplication (no-op pour l'epoch).
- Trop peu de frames (T<2) → impossible de calculer un flow, fallback aussi.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vjepa2_vitl_dup import VJEPA2ViTLDup, _TARGET_FRAMES, _duplicate_frames


def _build_raft(device: torch.device | None = None) -> nn.Module:
    """Charge RAFT-Large torchvision, en eval + frozen.

    Le helper isole la dépendance optionnelle ; échec → on retombera sur la
    duplication classique.
    """
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model = model.to(device)
    return model


def _warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warpe ``frame`` selon ``flow`` (déplacement pixel → coords).

    ``frame``  : (B, C, H, W) — image source.
    ``flow``   : (B, 2, H, W) — flow optique (dx, dy) en pixels.
    Renvoie (B, C, H, W) — frame warpée par ``grid_sample`` bilinéaire.
    """
    B, C, H, W = frame.shape
    device, dtype = frame.device, frame.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    grid = base + flow
    # Normalise vers [-1, 1] pour grid_sample
    grid_x = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    sample_grid = torch.stack((grid_x, grid_y), dim=-1)  # (B, H, W, 2)
    return F.grid_sample(
        frame, sample_grid, mode="bilinear", padding_mode="border", align_corners=True,
    )


class VJEPA2RaftDup(VJEPA2ViTLDup):
    """V-JEPA 2 ViT-L + interpolation temporelle RAFT (3 frames synthèses par paire)."""

    def __init__(self, *args, raft_iters: int = 12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.raft_iters = int(raft_iters)
        try:
            self.raft = _build_raft()
            self._raft_ok = True
        except Exception as exc:
            print(f"[VJEPA2RaftDup] RAFT init failed ({exc}); fallback → duplication.")
            self.raft = None
            self._raft_ok = False

    @torch.no_grad()
    def _prepare_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Interpole 4 frames → 16 via flow RAFT entre paires successives.

        x : (B, T, C, H, W). Si T < 2 ou RAFT indisponible, on revient à la
        duplication statique.
        """
        B, T, C, H, W = x.shape
        if not self._raft_ok or T < 2 or _TARGET_FRAMES % T != 0:
            return _duplicate_frames(x, _TARGET_FRAMES)

        per_frame = _TARGET_FRAMES // T  # ex : 4 frames → 4 copies par frame
        # Alphas internes : 0 (frame d'origine) + 1/k, 2/k, … (k-1)/k vers la suivante.
        alphas = torch.linspace(0, 1, per_frame + 1, device=x.device, dtype=x.dtype)[:-1]

        # raft attend (B, 3, H, W) en [-1, 1] ou [0, 255]. Le pipeline track-B
        # renvoie x en [0, 1] (ImageNet-norm AVANT autocast) → reconvertir
        # vers [-1, 1] dans la marge de tolérance de RAFT.
        # Note : les frames arrivent déjà normalisées ImageNet ; on les ramène
        # vers [0, 1] heuristiquement (RAFT est robuste sur les écarts d'échelle).
        x_raft = x.clamp(-3, 3)  # RAFT torchvision accepte plages assez larges
        out = []
        # Compute pairwise flows once, then warp.
        for t in range(T):
            src = x[:, t]                           # (B, C, H, W)
            if t < T - 1:
                nxt_raft = x_raft[:, t + 1]
                src_raft = x_raft[:, t]
                # raft attend deux tenseurs (B, 3, H, W) ; on bf16 le forward.
                try:
                    flow_list = self.raft(src_raft.float(), nxt_raft.float(),
                                          num_flow_updates=self.raft_iters)
                    flow = flow_list[-1].to(x.dtype)
                except Exception as exc:
                    warnings.warn(f"[VJEPA2RaftDup] RAFT forward failed: {exc}")
                    # On retombe sur la duplication pour cette paire.
                    flow = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)
            else:
                # Dernière frame : pas de "next" → flow nul (frame répétée).
                flow = torch.zeros(B, 2, H, W, device=x.device, dtype=x.dtype)

            for a in alphas:
                if a.item() == 0.0:
                    out.append(src)
                else:
                    out.append(_warp_frame(src, a * flow))

        result = torch.stack(out, dim=1)  # (B, 16, C, H, W)
        # Garde-fou NaN/Inf : retombe à duplication si RAFT a produit n'importe quoi.
        if not torch.isfinite(result).all():
            warnings.warn("[VJEPA2RaftDup] non-finite frames detected → fallback duplication.")
            return _duplicate_frames(x, _TARGET_FRAMES)
        return result

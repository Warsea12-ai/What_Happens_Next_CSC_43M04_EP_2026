"""temporal_interp.py — Stratégies d'upsampling temporel 4-frames → 16-frames
pour les backbones vidéo (VideoMAE, V-JEPA 2, etc.) entraînés sur des clips
de 16 frames.

Trois stratégies disponibles, sélectionnables via `mode`:

  - "aligned" (défaut historique, équivalent au `_linear_upsample` dupliqué dans
    19 modèles) : les 4 frames d'entrée sont placées exactement aux positions
    [0, 5, 10, 15] dans la sortie 16-frames ; les 12 positions restantes sont
    des combinaisons linéaires des frames adjacentes. Inconvénient : les frames
    F0 et F3 sont privilégiées (présentes telles quelles aux bords), ce qui
    biaise le gradient temporel apporté par le backbone vers les extrémités.

  - "centered" (NOUVEAU) : on découpe les 16 positions en 4 segments contigus de
    4 positions ([0..3], [4..7], [8..11], [12..15]) et on place chaque frame
    d'entrée au CENTRE de son segment (positions [1.5, 5.5, 9.5, 13.5]). Toutes
    les positions de sortie sont alors des interpolations linéaires symétriques
    des frames d'entrée adjacentes — aucune frame n'est privilégiée. Pour T_in=4,
    target_T=16 :
        F0 → pos 1.5 ; F1 → pos 5.5 ; F2 → pos 9.5 ; F3 → pos 13.5
        chaque tubelet (paire de 2 positions consécutives) reçoit deux blends
        cohérents de la même région temporelle.

  - "repeat" : répétition simple (chaque frame d'entrée est dupliquée
    `target_T // T_in` fois) — fourni pour l'exhaustivité, équivalent à ce que
    `videomae.py` faisait quand `use_linear_interp=False`.

Interface stable : toutes les fonctions prennent un tenseur (B, T, C, H, W) et
renvoient un tenseur (B, target_T, C, H, W) du même dtype et device.

Usage côté modèle :

    from temporal_interp import interp_temporal
    x = interp_temporal(x, target_T=16, mode=cfg.model.get("interp_mode", "aligned"))
"""
from __future__ import annotations

import torch


def linear_interp_aligned(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linear interpolation along the time axis, input frames placed at the EDGES
    (positions 0 and target_T-1 are exact copies of F0 and F[T-1]).

    Matches the existing `_linear_upsample` duplicated across the codebase —
    use this as the canonical version going forward.
    """
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


def linear_interp_centered(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """NEW : split the target_T positions into T equal segments, place each
    input frame at the CENTER of its segment, then linearly interpolate.

    For T_in=4 → target_T=16:
      - 4 segments of width 4: [0..3], [4..7], [8..11], [12..15]
      - Input frames placed at segment centers: [1.5, 5.5, 9.5, 13.5]
      - All 16 output positions are symmetric linear blends of two adjacent
        input frames (with edge positions extrapolated by clamping, identical
        to the closest input frame).

    Differs from `linear_interp_aligned` in two ways:
      1. Symmetric (no frame is privileged at the edges)
      2. Aligns naturally with VideoMAE's tubelet structure (each input frame
         occupies an entire tubelet pair without contamination from its
         neighbour mid-tubelet).
    """
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype

    seg = target_T / T                                   # 4.0 for T=4, target=16
    # Segment centers in output coordinate (positions where the input frames live):
    #   k * seg + seg/2 - 0.5  for k=0..T-1
    # The -0.5 shifts to a 0-indexed position centre (positions 0..target_T-1).
    # For T=4, seg=4: centres at [1.5, 5.5, 9.5, 13.5].
    # For each output index j, compute the fractional input index:
    #   idx_f = (j + 0.5) / seg - 0.5   ∈ [-0.5, T-0.5]
    out_pos = torch.arange(target_T, device=device, dtype=dtype)
    idx_f = (out_pos + 0.5) / seg - 0.5                   # fractional input index
    idx_f = idx_f.clamp(0, T - 1)                          # clamp at edges
    t_low  = idx_f.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (idx_f - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


def frame_repeat(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Repeat each input frame target_T // T times (no interpolation).
    Requires target_T to be a multiple of T."""
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    if target_T % T != 0:
        raise ValueError(f"frame_repeat needs target_T ({target_T}) divisible by T ({T})")
    k = target_T // T
    return x.unsqueeze(2).expand(B, T, k, C, H, W).reshape(B, target_T, C, H, W)


_DISPATCH = {
    "aligned":  linear_interp_aligned,
    "centered": linear_interp_centered,
    "repeat":   frame_repeat,
}


def interp_temporal(x: torch.Tensor, target_T: int = 16,
                    mode: str = "aligned") -> torch.Tensor:
    """Single entry point. `mode` is one of: 'aligned', 'centered', 'repeat'."""
    fn = _DISPATCH.get(mode)
    if fn is None:
        raise ValueError(f"Unknown interp mode {mode!r}; expected one of {list(_DISPATCH)}")
    return fn(x, target_T)


__all__ = [
    "linear_interp_aligned",
    "linear_interp_centered",
    "frame_repeat",
    "interp_temporal",
]

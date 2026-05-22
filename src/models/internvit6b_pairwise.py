"""InternViT6BPairwise — InternViT-6B + comparaison temporelle par paires.

Pourquoi les paires ordonnées (pas un Set Transformer)
-------------------------------------------------------
Des actions comme "Opening something" et "Closing something" ont le MÊME ensemble
de frames (mêmes états visuels) mais dans un ORDRE INVERSE. Un Set Transformer
ne peut pas les distinguer car il ignore l'ordre.

La solution : modéliser explicitement toutes les paires ordonnées (i→j).
Pour 4 frames, 4×3 = 12 paires ordonnées (i≠j).
La cross-attention frame_j-attend-à-frame_i capture "qu'est-ce qui a changé
de la frame i à la frame j ?", ce qui encode la DIRECTION temporelle.

Exemple :
  "Opening" : paire (fermé→ouvert) a une attention spécifique
  "Closing" : paire (ouvert→fermé) a la même paire mais INVERSÉE → attention différente

Architecture complète
---------------------
Input   : (B, T=4, C=3, H=224, W=224) — ImageNet-normalised
1. Resize 224→448 par frame
2. InternViT-6B frozen [no_grad] → CLS token par frame → (B, 4, 3200)
3. Projection linéaire 3200→proj_dim (512)  → (B, 4, 512) [trainable]
4. Génération des 12 paires ordonnées (i,j) avec i≠j :
   Pour chaque paire : frame_j fait une cross-attention sur frame_i
   → relation(i→j) de forme (B, 512)
5. Stack des 12 relations → (B, 12, 512)
6. Self-attention sur les 12 relations (raisonnement global sur l'ordre)
7. Mean pool → (B, 512)
8. Dropout + Linear(512, 33) → (B, 33)

Pourquoi c'est adapté à ce challenge
--------------------------------------
- Toutes les paires ordonnées sont vues explicitement → invariant à l'ordre
  de présentation des frames dans l'input (robustesse au shuffle),
  mais SENSIBLE à la direction temporelle (frame i avant frame j ≠ après)
- Cross-attention asymétrique : attend(i→j) ≠ attend(j→i)
- 6B params vision → features suffisamment riches pour les états transitoires
  des actions (main qui s'approche vs s'éloigne, objet qui s'ouvre vs se ferme)

Mémoire sur RTX A5000 / RTX 3090 (24 GB) :
  InternViT-6B bfloat16 frozen : ~12 GB
  Proj + pairwise + head        :  ~0.2 GB
  Activations batch=12, 448px   :  ~6 GB
  Total                         : ~18 GB
"""
from __future__ import annotations

from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


_TARGET_RES = 448


def _resize(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == (_TARGET_RES, _TARGET_RES):
        return x
    return F.interpolate(x, size=(_TARGET_RES, _TARGET_RES),
                         mode="bilinear", align_corners=False)


class _CrossFrameAttention(nn.Module):
    """frame_j cross-attends frame_i → relation directionnelle i→j."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.drop  = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # query = frame_j (B, 1, d), key_value = frame_i (B, 1, d)
        h, _ = self.attn(query, key_value, key_value)
        x = self.norm1(query + self.drop(h))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x.squeeze(1)   # (B, d)


class InternViT6BPairwise(nn.Module):
    """InternViT-6B frozen + cross-attention pairwise pour 33-class classification."""

    # Toutes les paires ordonnées (i→j) pour T=4 frames
    _PAIRS = [(i, j) for i in range(4) for j in range(4) if i != j]  # 12 paires

    def __init__(
        self,
        num_classes:  int   = 33,
        backbone:     str   = "OpenGVLab/InternViT-6B-448px-V2_5",
        proj_dim:     int   = 512,
        n_heads:      int   = 8,
        dropout:      float = 0.25,
    ) -> None:
        super().__init__()

        # ── Vision encoder (frozen) ─────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(
            backbone,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

        vit_dim = self.encoder.config.hidden_size  # 3200

        # ── Projection frame-level (trainable) ─────────────────────────────
        self.proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, proj_dim),
            nn.GELU(),
        )

        # ── Cross-attention pairwise (une couche par paire, partagée) ───────
        # Poids partagés entre les 12 paires → généralisation + économie mémoire
        self.pair_attn = _CrossFrameAttention(proj_dim, n_heads, dropout)

        # ── Self-attention sur les 12 relations ────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads, dim_feedforward=proj_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.relation_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # ── Tête de classification ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"InternViT6BPairwise: {n_frozen/1e6:.0f}M frozen (InternViT-6B), "
              f"{n_train/1e6:.1f}M trainable | {len(self._PAIRS)} paires ordonnées")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        return (list(self.proj.parameters()) +
                list(self.pair_attn.parameters()) +
                list(self.relation_transformer.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T=4, C, H, W) — ImageNet-normalised
        Returns : (B, 33)
        """
        B, T, C, H, W = x.shape
        enc_dtype = next(self.encoder.parameters()).dtype

        # 1. Encoder (frozen) — CLS token par frame
        frames = x.view(B * T, C, H, W).to(enc_dtype)
        frames = _resize(frames)  # (B*T, 3, 448, 448)

        with torch.no_grad():
            out = self.encoder(pixel_values=frames)
            cls = out.last_hidden_state[:, 0, :]   # (B*T, 3200)

        # 2. Projection → (B, T, proj_dim)
        feats = self.proj(cls.float())             # (B*T, proj_dim) float32
        feats = feats.view(B, T, -1)               # (B, 4, proj_dim)

        # 3. Cross-attention pour chaque paire ordonnée (i→j)
        #    frame_j cross-attends frame_i → capte le changement i→j
        relations = []
        for i, j in self._PAIRS:
            fi = feats[:, i:i+1, :]    # (B, 1, proj_dim) — frame source
            fj = feats[:, j:j+1, :]    # (B, 1, proj_dim) — frame query
            r = self.pair_attn(fj, fi) # (B, proj_dim) — relation i→j
            relations.append(r)

        rel_seq = torch.stack(relations, dim=1)    # (B, 12, proj_dim)

        # 4. Self-attention sur les 12 relations → raisonnement global
        rel_ctx = self.relation_transformer(rel_seq)  # (B, 12, proj_dim)

        # 5. Mean pool + classification
        pooled = rel_ctx.mean(dim=1)               # (B, proj_dim)
        return self.head(pooled)                   # (B, 33)

"""InternViT6BTemporal — InternViT-6B (vision encoder d'InternVL2-26B) +
Set Transformer permutation-invariant pour la classification de 33 actions.

Pourquoi c'est plus intelligent qu'InternVL2-8B complet
---------------------------------------------------------
InternVL2-8B charge :
  - InternViT-300M (vision)   → 300 M params utiles
  - InternLM2-7.7B (LLM)     → 7 700 M params pour de la génération de texte
                                 qu'on n'utilise PAS (on remplace par une tête linéaire)

InternViT-6B-448px-V2_5 charge UNIQUEMENT :
  - InternViT-6B (vision)     → 6 000 M params utiles
  - Rien d'autre
  Résultat : +20× de capacité vision pour un coût GPU similaire.

Pourquoi le Set Transformer est meilleur que mean-pool
-------------------------------------------------------
La tâche est fondamentalement invariante à la permutation des 4 frames d'entrée :
f(π(x)) = f(x) pour toute permutation π.
Un Set Transformer est invariant à l'ordre par construction — ses Self-Attention
layers n'ont pas de positional encoding entre frames.

Mean-pool est aussi invariant mais aveugle ; le Set Transformer apprend à
pondérer chaque frame selon sa pertinence pour l'action (main saisit l'objet
vs fond statique).

Architecture complète
---------------------
Input   : (B, T=4, C=3, H=224, W=224) — ImageNet-normalised
1. Resize 224 → 448 px par frame
2. InternViT-6B frozen [no_grad] → (B*4, 1025, 3200)
   - 1025 = 1 CLS + 1024 patches (patch=14, res=448)
3. CLS token seulement → (B*4, 3200)  [optimal pour la sémantique globale]
4. Linear 3200 → proj_dim [trainable, proj_dim=512 par défaut]
5. Reshape → (B, 4, proj_dim) — les 4 frames comme un SET
6. Set Transformer (n_layers couches MHSA sans PE entre frames) → (B, 4, proj_dim)
7. Mean pool sur les frames → (B, proj_dim)
8. Dropout → Linear(proj_dim, num_classes) → (B, 33)

Mémoire sur RTX A5000 / RTX 3090 (24 GB) :
  InternViT-6B bfloat16 : ~12 GB (frozen, no_grad → pas de backward)
  Proj + Set Transformer  :  ~60 M params × 2 octets = ~0.12 GB
  Activations batch=8     :  ~6 GB (448px, bfloat16)
  Total                   : ~18 GB → confortable sur 24 GB
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


_TARGET_RES = 448   # résolution native d'InternViT-6B-448px


def _resize(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == (_TARGET_RES, _TARGET_RES):
        return x
    return F.interpolate(x, size=(_TARGET_RES, _TARGET_RES),
                         mode="bilinear", align_corners=False)


class _SetTransformerLayer(nn.Module):
    """Une couche de Self-Attention sans positional encoding entre frames.
    Invariant à la permutation par construction.
    """
    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(h))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class InternViT6BTemporal(nn.Module):
    """InternViT-6B frozen + Set Transformer pour classification 33 classes."""

    def __init__(
        self,
        num_classes:    int   = 33,
        backbone:       str   = "OpenGVLab/InternViT-6B-448px-V2_5",
        proj_dim:       int   = 512,
        n_set_layers:   int   = 3,
        n_heads:        int   = 8,
        dropout:        float = 0.25,
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

        vit_dim = self.encoder.config.hidden_size  # 3200 for InternViT-6B

        # ── Projection trainable ────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, proj_dim),
            nn.GELU(),
        )

        # ── Set Transformer (invariant à la permutation) ────────────────────
        self.set_transformer = nn.Sequential(*[
            _SetTransformerLayer(proj_dim, n_heads, dropout)
            for _ in range(n_set_layers)
        ])

        # ── Tête de classification ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"InternViT6BTemporal: {n_frozen/1e6:.0f}M frozen (InternViT-6B), "
              f"{n_train/1e6:.1f}M trainable (proj+SetTransformer+head)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()   # toujours en eval — poids gelés
        return self

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        return (list(self.proj.parameters()) +
                list(self.set_transformer.parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T=4, C, H, W) — ImageNet-normalised float32
        Returns: (B, 33)
        """
        B, T, C, H, W = x.shape
        enc_dtype = next(self.encoder.parameters()).dtype

        # 1. Resize chaque frame 224→448
        frames = x.view(B * T, C, H, W).to(enc_dtype)
        frames = _resize(frames)                    # (B*T, 3, 448, 448)

        # 2. InternViT-6B (frozen, no_grad)
        with torch.no_grad():
            out = self.encoder(pixel_values=frames)
            # Prend le CLS token : représentation sémantique globale de la frame
            cls_tokens = out.last_hidden_state[:, 0, :]  # (B*T, 3200)

        # 3. Projection → (B, T, proj_dim)
        feats = self.proj(cls_tokens.float())        # (B*T, proj_dim) float32
        feats = feats.view(B, T, -1)                 # (B, 4, proj_dim)

        # 4. Set Transformer (invariant à l'ordre des frames)
        feats = self.set_transformer(feats)          # (B, 4, proj_dim)

        # 5. Mean pool sur les frames + classification
        pooled = feats.mean(dim=1)                   # (B, proj_dim)
        return self.head(pooled)                     # (B, 33)

"""VJEPA2ViTLDup — V-JEPA 2 ViT-L (~300M, SSv2-finetuned) + frames dupliquées + dégel doux.

Architecture (cf. slide « V-JEPA 2 × ViT-Large ») :
    Vidéo d'entrée (B, 4, 3, H, W)
      → Duplication 4 → 16 frames (chaque frame répétée 4 fois)
      → Tubelet Embedding              → 2048 tokens × 1024
      → Encoder Transformer ViT-L       (24 couches, hidden=1024)
      → Attentive Pooler                (2048 tokens → 1 × 1024)
      → Tête MLP  1024 → 512 → 33

Méthode (« Frames dupliquées + dégel doux ») :
  1. 16 frames (4 dupliquées) — au lieu de restreindre à 4 frames en entrée,
     on reproduit chaque frame 4 fois pour exposer l'encodeur à sa résolution
     temporelle d'entraînement SSv2 (tubelet_size=2 → 8 tubelets).
  2. Dégel doux : tous les blocs entraînables, LR backbone ×0.03 (sinon overfit).
     Le scaling backbone passe par layerwise_lr_groups → un seul groupe backbone
     uniforme (pas de LLRD inter-blocs ici, le dégel doux suffit).
  3. Tête compacte (1024→512→33) volontairement plus petite que VJEPA2Head
     (1408→2048→33) parce que ViT-L sort déjà des features bien séparables
     post-fine-tuning SSv2.

Note résolution : la slide indique 256×256 mais le pipeline produit 224×224.
V-JEPA 2 interpole dynamiquement les position embeddings, donc on charge avec
crop_size=224 pour aligner d'entrée les paramètres.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VJEPA2Model

_TARGET_FRAMES = 16


def _duplicate_frames(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Répète chaque frame ``target_T // T`` fois (et tronque si nécessaire).

    Ex. T=4 → target_T=16 : [A,B,C,D] → [A,A,A,A, B,B,B,B, C,C,C,C, D,D,D,D].
    """
    T = x.shape[1]
    if T == target_T:
        return x
    if target_T % T == 0:
        return x.repeat_interleave(target_T // T, dim=1)
    # Fallback générique : repeat puis crop (rare car target_T=16, T∈{1,2,4,8,16})
    reps = (target_T + T - 1) // T
    return x.repeat_interleave(reps, dim=1)[:, :target_T]


class _AttentionPool(nn.Module):
    """Pooling par une requête apprise (2048 tokens → 1 vecteur de dim ``d``)."""

    def __init__(self, d: int, n_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class VJEPA2ViTLDup(nn.Module):
    """V-JEPA 2 ViT-L (SSv2-finetuned) + attentive pooler + tête MLP 1024→512→33."""

    def __init__(
        self,
        num_classes:       int   = 33,
        backbone:          str   = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        cache_dir:         str   = "/Data/vianney.gauthier/hub",
        num_frozen_blocks: int   = 0,
        crop_size:         int   = 256,
        dropout:           float = 0.1,
        head_hidden:       int   = 512,
        pool_heads:        int   = 8,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks

        full = VJEPA2Model.from_pretrained(
            backbone,
            cache_dir=cache_dir,
            frames_per_clip=_TARGET_FRAMES,
            crop_size=crop_size,
            use_safetensors=True,
        )
        self.encoder = full.encoder
        del full

        # Dégel doux : par défaut tout est entraînable. Si num_frozen_blocks > 0,
        # on gèle les N premiers blocs (option d'expérimentation, pas utilisée
        # dans la config principale).
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = (num_frozen_blocks <= 0)
        for i, block in enumerate(self.encoder.layer):
            for p in block.parameters():
                p.requires_grad = (i >= num_frozen_blocks)

        hidden = 1024  # ViT-L hidden dimension

        self.pool = _AttentionPool(hidden, n_heads=pool_heads)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_blocks = len(self.encoder.layer)
        print(f"VJEPA2ViTLDup: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}/{n_blocks})")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.num_frozen_blocks > 0:
            self.encoder.embeddings.eval()
            for i, block in enumerate(self.encoder.layer):
                if i < self.num_frozen_blocks:
                    block.eval()
        return self

    def head_parameters(self):
        return list(self.pool.parameters()) + list(self.head.parameters())

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 1.0):
        """Deux groupes : tête à ``head_lr``, backbone à ``backbone_lr``.

        Ignore ``llrd`` (dégel doux uniforme : un seul LR pour tout le backbone).
        """
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        bb = self.backbone_parameters()
        if bb:
            groups.append({"params": bb, "lr": backbone_lr})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Duplication temporelle 4 → 16 (cf. slide « 16 frames (4 dupliquées) »)
        x = _duplicate_frames(x, _TARGET_FRAMES)

        # Frozen part (si num_frozen_blocks > 0) en no_grad pour économiser
        # les activations ; sinon tout est en autograd dès le premier bloc.
        if self.num_frozen_blocks > 0:
            with torch.no_grad():
                h = self.encoder.embeddings(pixel_values_videos=x)
                for block in self.encoder.layer[:self.num_frozen_blocks]:
                    h = block(h)[0]
            h = h.detach().to(x.dtype)
            blocks = self.encoder.layer[self.num_frozen_blocks:]
        else:
            h = self.encoder.embeddings(pixel_values_videos=x)
            blocks = self.encoder.layer

        for block in blocks:
            h = block(h)[0]

        feat = self.pool(h)              # (B, 1024)
        return self.head(feat)           # (B, num_classes)

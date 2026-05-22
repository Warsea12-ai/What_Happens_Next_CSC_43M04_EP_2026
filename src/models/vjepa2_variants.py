"""vjepa2_variants — Extensions architecturales de VJEPA2Head pour la recherche.

Quatre variantes explorées en 20 epochs pour identifier les pistes prometteuses :

  lora       : LoRA rank 16 sur TOUS les 40 blocs (vs gel/dégel partiel)
               → plus de paramètres backbone actifs pour le même coût mémoire
  attn_pool  : Pooling par attention apprise (vs mean-pool aveugle)
               → sélectionne les tokens pertinents pour l'action
  deep_head  : Tête MLP à 2 couches cachées 1408→2048→1024→33
               → plus de capacité dans la tête de classification
  multiscale : Features des blocs 20, 28, 34, 39 concaténées → proj → 33
               → combine sémantique fine (blocs bas) et globale (blocs hauts)

Toutes les variantes partagent le même backbone V-JEPA 2 ViT-G (SSv2-finetuned)
et le même forward jusqu'au bloc trainable, seule la tête diffère.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VJEPA2Model
from peft import get_peft_model, LoraConfig

_TARGET_FRAMES = 16


def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


class _AttentionPool(nn.Module):
    """Pooling par une requête apprise qui sélectionne les tokens pertinents."""
    def __init__(self, d: int, n_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class VJEPA2Variant(nn.Module):
    """V-JEPA 2 ViT-G + tête interchangeable.

    variant : "lora" | "attn_pool" | "deep_head" | "multiscale"
    """

    _MULTISCALE_BLOCKS = [20, 28, 34, 39]  # 4 niveaux de représentation

    def __init__(
        self,
        num_classes:       int   = 33,
        backbone:          str   = "facebook/vjepa2-vitg-fpc64-384-ssv2",
        cache_dir:         str   = "/Data/vianney.gauthier/hub",
        num_frozen_blocks: int   = 32,
        dropout:           float = 0.25,
        head_hidden:       int   = 2048,
        variant:           str   = "lora",   # lora | attn_pool | deep_head | multiscale
        lora_rank:         int   = 16,
        lora_alpha:        float = 32.0,
    ) -> None:
        super().__init__()
        self.variant           = variant
        self.num_frozen_blocks = num_frozen_blocks

        full = VJEPA2Model.from_pretrained(
            backbone, cache_dir=cache_dir,
            frames_per_clip=_TARGET_FRAMES, crop_size=224, use_safetensors=True,
        )
        self.encoder = full.encoder
        del full

        hidden = 1408   # ViT-G hidden dim

        # ── Backbone setup selon la variante ────────────────────────────────
        if variant == "lora":
            # LoRA sur TOUS les 40 blocs — base weights gelés, adapteurs entraînables
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False   # tout gelé d'abord

            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                # Noms des projections dans les blocs VJEPA2 (ViT-style)
                target_modules=["query", "key", "value", "dense"],
                bias="none",
                lora_dropout=0.05,
            )
            # Apply LoRA — les adapteurs auront requires_grad=True automatiquement
            self.encoder = get_peft_model(self.encoder, lora_cfg)

        else:
            # Comportement standard : geler les embeddings + N premiers blocs
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for i, block in enumerate(self.encoder.layer):
                for p in block.parameters():
                    p.requires_grad = (i >= num_frozen_blocks)

        # ── Tête selon la variante ───────────────────────────────────────────
        if variant == "multiscale":
            # Concatène 4 niveaux → projection → classe
            ms_dim = hidden * len(self._MULTISCALE_BLOCKS)
            self.ms_proj = nn.Sequential(
                nn.LayerNorm(ms_dim),
                nn.Linear(ms_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(head_hidden, num_classes)
        elif variant == "attn_pool":
            self.pool = _AttentionPool(hidden, n_heads=8)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_hidden, num_classes),
            )
        elif variant == "deep_head":
            # 2 couches cachées : 1408 → 2048 → 1024 → 33
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_hidden // 2, num_classes),
            )
        else:  # lora — même tête simple que vjepa2_head
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_hidden, num_classes),
            )

        # Init tête
        last_linear = [m for m in self.head.modules() if isinstance(m, nn.Linear)][-1] \
            if hasattr(self, 'head') and not isinstance(self.head, nn.Linear) \
            else (self.head if isinstance(self.head, nn.Linear) else None)
        if last_linear is not None:
            nn.init.trunc_normal_(last_linear.weight, std=0.01)
            nn.init.zeros_(last_linear.bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VJEPA2Variant[{variant}]: {n_frozen/1e6:.0f}M frozen, "
              f"{n_train/1e6:.1f}M trainable (frozen_blocks={num_frozen_blocks}/40)")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.variant != "lora":
            self.encoder.embeddings.eval()
            for i, block in enumerate(self.encoder.layer):
                if i < self.num_frozen_blocks:
                    block.eval()
        return self

    def head_parameters(self):
        params = []
        for name in ("head", "pool", "ms_proj"):
            if hasattr(self, name):
                params += list(getattr(self, name).parameters())
        return params

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.80):
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        if self.variant == "lora":
            # LoRA : tous les adapteurs au même LR backbone (pas de LLRD bloc par bloc)
            groups.append({"params": self.backbone_parameters(), "lr": backbone_lr})
            return groups
        n = len(self.encoder.layer)
        for depth, idx in enumerate(range(n - 1, self.num_frozen_blocks - 1, -1)):
            groups.append({
                "params": list(self.encoder.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = _linear_upsample(x, _TARGET_FRAMES)

        if self.variant == "multiscale":
            return self._forward_multiscale(x, dtype)

        # Forward standard (frozen + trainable)
        with torch.no_grad():
            h = self.encoder.embeddings(pixel_values_videos=x)
            frozen_end = 0 if self.variant == "lora" else self.num_frozen_blocks
            for block in self.encoder.layer[:frozen_end]:
                h = block(h)[0]
        h = h.detach().to(dtype)

        if self.variant == "lora":
            for block in self.encoder.layer:
                h = block(h)[0]
        else:
            for block in self.encoder.layer[self.num_frozen_blocks:]:
                h = block(h)[0]

        if self.variant == "attn_pool":
            feat = self.pool(h)
        else:
            feat = h.mean(dim=1)

        return self.head(feat)

    def _forward_multiscale(self, x: torch.Tensor, dtype) -> torch.Tensor:
        """Collecte des features aux blocs 20, 28, 34, 39 et les concatène."""
        collect_at = set(self._MULTISCALE_BLOCKS)
        collected  = []

        with torch.no_grad():
            h = self.encoder.embeddings(pixel_values_videos=x)
            for i, block in enumerate(self.encoder.layer[:self.num_frozen_blocks]):
                h = block(h)[0]
                if i in collect_at:
                    collected.append(h.mean(dim=1).detach().to(dtype))

        h = h.detach().to(dtype)
        for i, block in enumerate(
            self.encoder.layer[self.num_frozen_blocks:],
            start=self.num_frozen_blocks
        ):
            h = block(h)[0]
            if i in collect_at:
                collected.append(h.mean(dim=1))

        # Assure qu'on a bien 4 features (peut manquer si num_frozen_blocks > max collect)
        while len(collected) < len(self._MULTISCALE_BLOCKS):
            collected.append(h.mean(dim=1))

        feat = torch.cat(collected[-len(self._MULTISCALE_BLOCKS):], dim=-1)  # (B, 4×1408)
        return self.head(self.ms_proj(feat))

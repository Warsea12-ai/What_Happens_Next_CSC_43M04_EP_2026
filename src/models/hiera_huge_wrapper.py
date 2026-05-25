"""HieraHuge — Hiera-Huge 16x224 K400-pretrained (facebookresearch/hiera).

NOTE : les checkpoints SSv2 officiels de Hiera ne sont pas publiquement disponibles
sur fbaipublicfiles. On utilise le checkpoint K400 finetuned (reported 76.0% top-1
sur K400) et on finetune sur les 33 classes du dataset curaté.

Reference :
  Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles (ICML 2023)
  https://github.com/facebookresearch/hiera

Trois variantes selon `variant` :
  frozen     : backbone gelé, tête 400→33 trainable seulement
  attn_pool  : ~60% des stages gelés + attention pool sur features + head
  lora       : LoRA rank=16 sur QKV de tous les blocs

Hiera est hiérarchique avec 4 stages — différent d'un ViT plat. On accède
aux features via .head_drop input (pre-classifier) ou via forward_features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import hiera

_TARGET_FRAMES = 16
_IMG_SIZE      = 224

# K400 pretrained Hiera utilise sa propre normalisation (souvent ImageNet)
_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3, 1, 1)
_STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3, 1, 1)


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
    def __init__(self, d: int, n_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(d)
    def forward(self, x):
        # x peut être (B, N, D) — déjà des tokens — ou (B, D) — déjà pooled.
        if x.dim() == 2:
            return self.norm(x)
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


def _apply_lora_to_attention(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> None:
    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            in_dim, out_dim = base.in_features, base.out_features
            self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * (1.0 / rank))
            self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
            self.scale = alpha / rank
        def forward(self, x):
            return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    # Hiera utilise des Linears nommés qkv dans MaskUnitAttention
    n_lora = 0
    for module in model.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, rank, alpha)
            n_lora += 1
    print(f"[HieraHuge] LoRA appliqué sur {n_lora} couches d'attention", flush=True)


class HieraHuge(nn.Module):
    """Hiera-Huge 16x224 K400-pretrained + tête 33-classes.

    Args:
        variant : "frozen" | "attn_pool" | "lora"
        num_classes : 33 par défaut
        pretrained : True charge les poids K400 finetuned
        head_hidden : MLP intermédiaire (0 = linear direct)
        dropout
    """

    _VARIANTS = ("frozen", "attn_pool", "lora")

    def __init__(
        self,
        num_classes: int = 33,
        variant: str = "frozen",
        pretrained: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        head_hidden: int = 0,
        dropout: float = 0.3,
        num_frozen_stages: int = 3,  # Hiera a 4 stages ; 3 frozen = dernière trainable
    ) -> None:
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"variant doit être dans {self._VARIANTS}, reçu {variant!r}")
        self.variant = variant

        # 1. Backbone Hiera-Huge K400-pretrained (174M params, K400 76.0%)
        self.backbone = hiera.hiera_huge_16x224(
            pretrained=pretrained,
            checkpoint="mae_k400_ft_k400",
            strict=False,
        )
        # Dimension du token de classification — pour Huge c'est 1280 typiquement
        hidden = self.backbone.head.projection.in_features if hasattr(self.backbone.head, "projection") \
                 else self.backbone.head.in_features
        # Remplace la tête 400 → Identity (on met notre head)
        self.backbone.head = nn.Identity()

        # 2. Gel selon variant
        if variant == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.pool = None
        elif variant == "attn_pool":
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Hiera a self.backbone.blocks (liste de Block) répartis en 4 stages.
            # On dégèle les blocs du dernier stage.
            blocks = list(self.backbone.blocks)
            n_blocks = len(blocks)
            n_train = max(1, n_blocks // 4)  # ~25% des blocs trainables
            for blk in blocks[-n_train:]:
                for p in blk.parameters():
                    p.requires_grad = True
            self.pool = _AttentionPool(hidden, n_heads=8)
        elif variant == "lora":
            for p in self.backbone.parameters():
                p.requires_grad = False
            _apply_lora_to_attention(self.backbone, rank=lora_rank, alpha=lora_alpha)
            self.pool = None

        # 3. Tête 33-classes
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[HieraHuge] variant={variant} : {n_train/1e6:.1f}M trainable / "
              f"{n_total/1e6:.1f}M total ({(n_total-n_train)/1e6:.1f}M frozen)",
              flush=True)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) en [0,1] → (B, C, T, H, W) normalisé pour Hiera."""
        x = _linear_upsample(x, _TARGET_FRAMES)
        mean = _MEAN.to(x.device, x.dtype)
        std  = _STD.to(x.device, x.dtype)
        x = (x - mean) / std
        # Hiera video : (B, C, T, H, W)
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        features = self.backbone(x)  # (B, hidden) — head=Identity donc features
        if self.pool is not None:
            features = self.pool(features)
        return self.head(features)

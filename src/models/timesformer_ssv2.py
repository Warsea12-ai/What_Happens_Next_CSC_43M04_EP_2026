"""TimesFormerSSv2 — TimesFormer SSv2-finetuned (HF native).

Architecture : divided space-time attention (Bertasius et al., ICML 2021).
Différent du masked autoencoder typique : pas de MAE pretrain, juste
ImageNet → SSv2 finetuning avec attention factorisée temps/espace.

Variants:
  Base : facebook/timesformer-base-finetuned-ssv2  (88M params, 224x224, 59.5% SSv2)
  HR   : facebook/timesformer-hr-finetuned-ssv2    (121M params, 448x448, 63.5% SSv2)

Trois config-variants selon `variant`:
  frozen     : backbone gelé, tête 174→33 trainable
  attn_pool  : derniers 25% des blocs + attention pool
  lora       : LoRA rank=16 sur Q/V

Note : 224x224 (base) fit 130W facilement ; 448x448 (HR) nécessite plus de VRAM
mais reste largement OK sur RTX 4000 Ada 20GB avec batch_size raisonnable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers import TimesformerForVideoClassification

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


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

    n_lora = 0
    for name, module in model.named_modules():
        # TimesFormer HF utilise des Linear nommés query, value, key dans attention
        for attr in ("query", "value"):
            if hasattr(module, attr) and isinstance(getattr(module, attr), nn.Linear):
                setattr(module, attr, LoRALinear(getattr(module, attr), rank, alpha))
                n_lora += 1
    print(f"[TimesFormer] LoRA appliqué sur {n_lora} couches", flush=True)


class TimesFormerSSv2(nn.Module):
    _VARIANTS = ("frozen", "attn_pool", "lora")

    def __init__(
        self,
        num_classes: int = 33,
        variant: str = "frozen",
        backbone: str = "facebook/timesformer-base-finetuned-ssv2",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        head_hidden: int = 0,
        dropout: float = 0.3,
        num_frames: int = 8,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"variant doit être dans {self._VARIANTS}, reçu {variant!r}")
        self.variant = variant
        self.num_frames = num_frames
        self.image_size = image_size

        # Charge le modèle complet, garde l'encoder, drop la tête 174 classes
        full = TimesformerForVideoClassification.from_pretrained(
            backbone, use_safetensors=False, ignore_mismatched_sizes=True,
        )
        self.encoder = full.timesformer
        del full
        hidden = self.encoder.config.hidden_size  # base=768, HR=768 aussi

        # Gel selon variant
        if variant == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.pool = None
        elif variant == "attn_pool":
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Dégèle les derniers 25% des blocs (TimesFormer base a 12 layers, HR aussi)
            n_layers = self.encoder.config.num_hidden_layers
            n_train = max(1, n_layers // 4)
            for blk in list(self.encoder.encoder.layer)[-n_train:]:
                for p in blk.parameters():
                    p.requires_grad = True
            self.pool = _AttentionPool(hidden, n_heads=8)
        elif variant == "lora":
            for p in self.encoder.parameters():
                p.requires_grad = False
            _apply_lora_to_attention(self.encoder, rank=lora_rank, alpha=lora_alpha)
            self.pool = None

        # Tête 33-classes
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
        print(f"[TimesFormer] {backbone.split('/')[-1]} variant={variant} : "
              f"{n_train/1e6:.1f}M trainable / {n_total/1e6:.1f}M total", flush=True)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = _linear_upsample(x, self.num_frames)
        # Resize spatial à image_size si nécessaire (utile pour HR=448)
        if x.shape[-1] != self.image_size:
            x = nn.functional.interpolate(
                x.reshape(-1, *x.shape[2:]),
                size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).view(x.shape[0], x.shape[1], 3, self.image_size, self.image_size)
        mean = _IN_MEAN.to(x.device, x.dtype)
        std  = _IN_STD.to(x.device, x.dtype)
        x = (x - mean) / std
        return x  # TimesFormer HF prend (B, T, C, H, W) directement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        outputs = self.encoder(pixel_values=x)
        # last_hidden_state shape: (B, 1 + T*N_patches, hidden)
        # On prend le CLS token (index 0) ou on pool sur les patches
        if self.pool is not None:
            features = self.pool(outputs.last_hidden_state)
        else:
            features = outputs.last_hidden_state[:, 0]  # CLS
        return self.head(features)

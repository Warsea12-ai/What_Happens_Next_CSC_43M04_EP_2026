"""VideoMAELoRA — VideoMAE-Large with LoRA adapters on attention layers.

Instead of unfreezing full blocks (expensive) or fully freezing (limited),
LoRA inserts tiny rank-r adapters on the Q and V projections of every
attention block. This gives gradient flow through all 24 blocks with only
~3M additional parameters instead of unfreezeing 307M.

Adapter: LoRALinear wraps each frozen Linear with
    out = W_frozen @ x + (alpha/r) * B @ A @ x
    A ~ N(0, 1/r), B = 0  → starts as the frozen model, adapts over training.

Target modules: query, value (standard VideoMAE HF naming)
rank=8, alpha=16 → scale=2.0, ~3M params across 24 blocks.

Same motion-aware head as FrozenVideoMAELarge:
  [global, first, mid, last, delta] = 5×1024 → Linear(5120, head_hidden) → head
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

from models.lora_utils import apply_lora, get_lora_parameters, count_lora_params

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_VM_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16
_TUBELET_SIZE  = 2
_PATCH_SIZE    = 16
_IMG_SIZE      = 224
_N_TEMPORAL    = _TARGET_FRAMES // _TUBELET_SIZE   # 8
_N_SPATIAL     = (_IMG_SIZE // _PATCH_SIZE) ** 2   # 196


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


class VideoMAELoRA(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-large-finetuned-kinetics",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_target_modules: list[str] | None = None,
        dropout: float = 0.4,
        head_hidden: int = 512,
    ) -> None:
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["query", "value"]

        _full = VideoMAEForVideoClassification.from_pretrained(
            backbone, use_safetensors=True,
        )
        self.encoder = _full.videomae
        del _full

        # Freeze all params first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Apply LoRA adapters (they start unfrozen by default)
        apply_lora(self.encoder, lora_target_modules, rank=lora_rank, alpha=lora_alpha)

        hidden = self.encoder.config.hidden_size  # 1024

        self.norm_global = nn.LayerNorm(hidden)
        self.norm_first  = nn.LayerNorm(hidden)
        self.norm_mid    = nn.LayerNorm(hidden)
        self.norm_last   = nn.LayerNorm(hidden)
        self.norm_delta  = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(5 * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        n_lora   = count_lora_params(self.encoder)
        n_head   = sum(p.numel() for p in self.head_parameters())
        print(f"VideoMAELoRA: {n_frozen/1e6:.1f}M frozen, "
              f"{n_lora/1e6:.2f}M LoRA, {n_head/1e6:.2f}M head")

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep non-LoRA encoder params in eval (BN/LN stats stable)
        self.encoder.eval()
        # LoRA adapter modules follow training mode — they're just Linear layers
        from models.lora_utils import LoRALinear
        for m in self.encoder.modules():
            if isinstance(m, LoRALinear):
                m.train(mode)
        return self

    def head_parameters(self):
        return (
            list(self.norm_global.parameters())
            + list(self.norm_first.parameters())
            + list(self.norm_mid.parameters())
            + list(self.norm_last.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )

    def lora_parameters(self):
        return get_lora_parameters(self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm
        x = _linear_upsample(x, _TARGET_FRAMES)

        # LoRA adapters need gradients — run with grad enabled
        tokens = self.encoder(pixel_values=x).last_hidden_state  # (B, 1568, 1024)
        tokens = tokens.to(dtype)
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, -1)

        global_feat = tokens.mean(dim=(1, 2))
        first       = tokens[:, 0].mean(dim=1)
        mid         = tokens[:, 3:5].mean(dim=(1, 2))
        last        = tokens[:, -1].mean(dim=1)
        delta       = last - first

        feat = torch.cat([
            self.norm_global(global_feat),
            self.norm_first(first),
            self.norm_mid(mid),
            self.norm_last(last),
            self.norm_delta(delta),
        ], dim=1)

        return self.head(feat)

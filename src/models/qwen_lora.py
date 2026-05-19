"""QwenVLLoRA — Qwen2-VL visual encoder with LoRA on attention projections.

The 2B visual encoder has 1536-dim hidden size and 32 transformer blocks.
The 7B visual encoder has 3584-dim hidden size and 32 blocks.

LoRA inserts rank-r adapters on q_proj and v_proj of every attention block:
  W_adapted = W_frozen + (alpha/r) * B @ A
  ~3.5M additional params for 7B encoder (vs 3.5B frozen).

The motion-aware head is identical to QwenVLVideo:
  [global, first, last, delta] → cat → head

This is the key difference from QwenVLVideo: the visual encoder can adapt
to this specific domain (first-person actions) instead of remaining
frozen in its general VLM feature space.
"""
from __future__ import annotations

import gc
import torch
import torch.nn as nn

from models.qwen_vl_video import _imagenet_to_clip, _frames_to_patches
from models.lora_utils import apply_lora, get_lora_parameters, count_lora_params, LoRALinear


class QwenVLLoRA(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "Qwen/Qwen2-VL-7B-Instruct",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_target_modules: list[str] | None = None,
        dropout: float = 0.3,
        head_hidden: int = 768,
    ) -> None:
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        print(f"Loading {backbone} to CPU…")
        if "2.5" in backbone:
            from transformers import Qwen2_5_VLForConditionalGeneration
            _full = Qwen2_5_VLForConditionalGeneration.from_pretrained(backbone, dtype=torch.bfloat16)
        else:
            from transformers import Qwen2VLForConditionalGeneration
            _full = Qwen2VLForConditionalGeneration.from_pretrained(backbone, dtype=torch.bfloat16)
        self.visual = _full.model.visual
        del _full
        gc.collect()
        print("LLM deleted — keeping visual encoder only.")

        # Freeze all visual params first
        for p in self.visual.parameters():
            p.requires_grad = False

        # Apply LoRA adapters to attention projections
        apply_lora(self.visual, lora_target_modules, rank=lora_rank, alpha=lora_alpha)

        vcfg = self.visual.config
        self.patch_size          = vcfg.patch_size
        self.temporal_patch_size = vcfg.temporal_patch_size
        self.spatial_merge_size  = getattr(vcfg, "spatial_merge_size", 2)

        # Probe actual output dim
        ps, tps = vcfg.patch_size, vcfg.temporal_patch_size
        _H, _W = 224, 224
        _tg = 4 // tps
        _hp, _wp = _H // ps, _W // ps
        _dummy_pv = torch.zeros(_tg * _hp * _wp, 3 * tps * ps * ps, dtype=torch.bfloat16)
        _dummy_grid = torch.tensor([[_tg, _hp, _wp]], dtype=torch.long)
        with torch.no_grad():
            _out = self.visual(_dummy_pv, grid_thw=_dummy_grid)
            _raw = _out.last_hidden_state if hasattr(_out, "last_hidden_state") else _out
        hidden = int(_raw.shape[-1])

        self.norm_global = nn.LayerNorm(hidden)
        self.norm_first  = nn.LayerNorm(hidden)
        self.norm_last   = nn.LayerNorm(hidden)
        self.norm_delta  = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.visual.parameters() if not p.requires_grad)
        n_lora   = count_lora_params(self.visual)
        n_head   = sum(p.numel() for p in self.head_parameters())
        print(f"QwenVLLoRA: {n_frozen/1e6:.0f}M frozen, "
              f"{n_lora/1e6:.2f}M LoRA, {n_head/1e6:.2f}M head (hidden={hidden})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        for m in self.visual.modules():
            if isinstance(m, LoRALinear):
                m.train(mode)
        return self

    def head_parameters(self):
        return (
            list(self.norm_global.parameters())
            + list(self.norm_first.parameters())
            + list(self.norm_last.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )

    def lora_parameters(self):
        return get_lora_parameters(self.visual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        ps   = self.patch_size
        tps  = self.temporal_patch_size
        sms  = self.spatial_merge_size
        hp   = H // ps
        wp   = W // ps
        tg   = T // tps
        hpo  = hp // sms
        wpo  = wp // sms

        x_clip = _imagenet_to_clip(x)
        pixel_values = _frames_to_patches(x_clip, ps, tps).to(torch.bfloat16)
        grid_thw = torch.tensor([[tg, hp, wp]], dtype=torch.long, device=device).expand(B, 3)

        # LoRA adapters need grad — no torch.no_grad() here
        out = self.visual(pixel_values, grid_thw=grid_thw)
        raw = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        tokens = raw.to(dtype).reshape(B, tg, hpo * wpo, -1)  # (B, 2, 64, hidden)

        global_feat = tokens.mean(dim=(1, 2))
        first       = tokens[:, 0].mean(dim=1)
        last        = tokens[:, -1].mean(dim=1)
        delta       = last - first

        feat = torch.cat([
            self.norm_global(global_feat),
            self.norm_first(first),
            self.norm_last(last),
            self.norm_delta(delta),
        ], dim=1)

        return self.head(feat)

"""FrozenVideoMAELarge — VideoMAE-Large (Kinetics) with fully frozen backbone.

Largest VideoMAE checkpoint compatible with HuggingFace transformers:
  MCG-NJU/videomae-large-finetuned-kinetics — 307M params, 1024-dim, 24 blocks.

Strategy: freeze the entire backbone and train only the motion-aware head.
This lets us:
  - Use batch_size=64 (no gradient tape through 307M params → ~5 GB peak vs ~20 GB)
  - Run a higher head LR (head is small, random-init, no risk of over-training backbone)
  - Converge in fewer wall-clock seconds despite the larger backbone

Head: [global, first, last, delta] — four 1024-dim features concatenated:
  global : mean over all 1568 patch tokens     — scene/object context
  first  : mean of temporal-position-0 patches — where the action started
  last   : mean of temporal-position-7 patches — where it ended up
  delta  = last − first                        — net motion direction

Temporal upsampling: 4 input frames → 16 via linear interpolation (not repetition)
so VideoMAE's temporal attention sees a smooth motion gradient.

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

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


class FrozenVideoMAELarge(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-large-finetuned-kinetics",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Load full classification model, then keep only the backbone
        _full = VideoMAEForVideoClassification.from_pretrained(
            backbone, use_safetensors=True,
        )
        self.encoder = _full.videomae   # VideoMAEModel, 307M params
        del _full                       # free the 400-class head immediately

        # Freeze every backbone parameter — no gradients through encoder ever
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden = self.encoder.config.hidden_size  # 1024

        # Independent LayerNorms so each feature type is properly scaled
        self.norm_global = nn.LayerNorm(hidden)
        self.norm_first  = nn.LayerNorm(hidden)
        self.norm_last   = nn.LayerNorm(hidden)
        self.norm_delta  = nn.LayerNorm(hidden)

        # Compact MLP head: 4×1024=4096 → 512 → num_classes (~2M trainable params)
        # A larger head (4096→2048) overfits with only 50k training videos.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.encoder.parameters())
        n_train  = sum(p.numel() for p in self.head_parameters())
        print(f"FrozenVideoMAELarge: {n_frozen/1e6:.1f}M frozen backbone, "
              f"{n_train/1e6:.3f}M trainable head")

    def train(self, mode: bool = True):
        # Keep the frozen encoder in eval mode at all times so its dropout/batchnorm
        # layers behave deterministically — calling model.train() must not undo this.
        super().train(mode)
        self.encoder.eval()
        return self

    # ── parameter groups ──────────────────────────────────────────────────────

    def head_parameters(self):
        return (
            list(self.norm_global.parameters())
            + list(self.norm_first.parameters())
            + list(self.norm_last.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, C, H, W) — ImageNet-normalised
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE [0.5, 0.5, 0.5] normalisation
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # 4 frames → 16 via linear interpolation
        x = _linear_upsample(x, _TARGET_FRAMES)

        # Frozen backbone — no gradient tape, activations freed immediately
        with torch.no_grad():
            tokens = self.encoder(pixel_values=x).last_hidden_state  # (B, 1568, 1024)

        # Cast to head dtype (bfloat16 backbone → float32 head is fine)
        tokens = tokens.to(dtype)

        # Temporal × spatial grid
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, -1)  # (B, 8, 196, 1024)

        global_feat = tokens.mean(dim=(1, 2))       # scene context
        first       = tokens[:, 0].mean(dim=1)      # start of action
        last        = tokens[:, -1].mean(dim=1)     # end of action
        delta       = last - first                  # net motion direction

        feat = torch.cat([
            self.norm_global(global_feat),
            self.norm_first(first),
            self.norm_last(last),
            self.norm_delta(delta),
        ], dim=1)  # (B, 4096)

        return self.head(feat)

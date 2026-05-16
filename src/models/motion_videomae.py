"""MotionVideoMAE — VideoMAE-Base (SSv2-pretrained) with explicit motion encoding.

Designed specifically for the "What Happens Next?" challenge:
  - 4 frames from the START of an action → predict which of 33 classes

Key adaptations vs vanilla VideoMAE:
  1. Linear temporal interpolation 4→16 instead of frame repetition.
     Repetition makes adjacent temporal patches identical, killing temporal attention.
     Interpolation gives a smooth motion gradient → attention sees real change.

  2. Motion-aware head: [global_pool, delta]
     - global_pool: what scene / object context is present
     - delta = last_temporal_feat − first_temporal_feat: where the action is heading
     This is directly adapted to the temporal-reverse pairs in the dataset
     (classes that differ only by direction require exactly this signal).

  3. VideoMAE backbone pretrained on SSv2 (same domain as challenge data).

Architecture:
    VideoMAE-Base encoder (partial fine-tune, last 4 of 12 blocks)
       ↓  patch tokens (B, 8 temporal × 196 spatial, 768)
    global_pool = mean(all tokens)           — scene context
    delta = mean(tokens[T=7]) − mean(tokens[T=0])  — net motion direction
       ↓
    cat([LayerNorm(global_pool), LayerNorm(delta)])  → (B, 1536)
    Dropout → Linear(1536, 512) → GELU → Dropout → Linear(512, 33)

Input : (B, T=4, C, H, W)  ImageNet-normalized (standard training pipeline)
Output: (B, num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEModel

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_VM_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16   # VideoMAE-Base expects 16 frames
_PATCH_SIZE    = 16   # spatial patch size
_IMG_SIZE      = 224
_TUBELET_SIZE  = 2    # VideoMAE temporal patch depth
_N_TEMPORAL    = _TARGET_FRAMES // _TUBELET_SIZE   # 8
_N_SPATIAL     = (_IMG_SIZE // _PATCH_SIZE) ** 2   # 196


def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linearly interpolate (B, T, C, H, W) along the time axis to target_T."""
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


class MotionVideoMAE(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frozen_blocks: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze patch/temporal embeddings
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        # Freeze first num_frozen_blocks encoder blocks, fine-tune the rest
        for i, block in enumerate(self.encoder.encoder.layer):
            trainable = i >= num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = trainable

        # Keep the output LayerNorm trainable (normalizes final patch representations)
        if hasattr(self.encoder, "layernorm"):
            for p in self.encoder.layernorm.parameters():
                p.requires_grad = True

        hidden = self.encoder.config.hidden_size  # 768 for base

        self.norm_global = nn.LayerNorm(hidden)
        self.norm_delta  = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MotionVideoMAE: {n_frozen/1e6:.1f}M frozen, {n_train/1e6:.1f}M trainable")

    # ── parameter groups for differential LR ─────────────────────────────────

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
        return (
            list(self.norm_global.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, C, H, W) — ImageNet-normalized
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE normalization (fused, no extra allocation)
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # 4 frames → 16 via linear interpolation (smooth motion gradient)
        x = _linear_upsample(x, _TARGET_FRAMES)  # (B, 16, C, H, W)

        # VideoMAE encoder — returns all patch tokens, no CLS
        tokens = self.encoder(pixel_values=x).last_hidden_state  # (B, 1568, 768)

        # Reshape to temporal × spatial grid
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, -1)  # (B, 8, 196, 768)

        # Scene context: mean over all patches
        global_feat = tokens.mean(dim=(1, 2))  # (B, 768)

        # Motion direction: where did the scene end up vs where it started
        first = tokens[:, 0].mean(dim=1)   # (B, 768) — temporal position 0
        last  = tokens[:, -1].mean(dim=1)  # (B, 768) — temporal position 7
        delta = last - first               # (B, 768) — net motion direction

        feat = torch.cat([
            self.norm_global(global_feat),
            self.norm_delta(delta),
        ], dim=1)  # (B, 1536)

        return self.head(feat)

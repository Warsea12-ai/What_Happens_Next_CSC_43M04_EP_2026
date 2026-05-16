"""MotionVideoMAE — VideoMAE-Base (SSv2-pretrained) with explicit motion encoding.

Designed specifically for the "What Happens Next?" challenge:
  4 frames from the START of an action → predict which of 33 classes.

Key adaptations:
  1. Linear temporal interpolation 4→16 frames (not frame repetition).
     Repetition makes adjacent temporal patches identical, eliminating the
     motion gradient that VideoMAE's temporal attention is built to exploit.

  2. Four-feature head: [global, first, last, delta]
     - global : mean over all 8×196 patch tokens  — scene/object context
     - first  : mean of temporal position 0 tokens — where the action started
     - last   : mean of temporal position 7 tokens — where it ended up
     - delta  = last − first                       — net motion direction
     delta is the critical signal for temporal-reverse class pairs
     (e.g., "moving left" ↔ "moving right") that differ only in direction.

  3. Layer-wise LR decay via `layerwise_lr_groups()`.
     Standard flat backbone LR over-trains later blocks and corrupts early
     generalisation. LLRD decays LR geometrically toward earlier blocks.

Input : (B, T=4, C, H, W)  ImageNet-normalised
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

_TARGET_FRAMES = 16
_TUBELET_SIZE  = 2
_PATCH_SIZE    = 16
_IMG_SIZE      = 224
_N_TEMPORAL    = _TARGET_FRAMES // _TUBELET_SIZE   # 8
_N_SPATIAL     = (_IMG_SIZE // _PATCH_SIZE) ** 2   # 196


def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linearly interpolate (B, T, C, H, W) along the time axis."""
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
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
        self.num_frozen_blocks = num_frozen_blocks

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze patch embedding (Conv3d tubelet projection)
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        # Freeze first num_frozen_blocks, fine-tune the rest
        for i, block in enumerate(self.encoder.encoder.layer):
            trainable = i >= num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = trainable

        hidden = self.encoder.config.hidden_size  # 768

        # Separate LayerNorms so each feature type is properly scaled before concat
        self.norm_global = nn.LayerNorm(hidden)
        self.norm_first  = nn.LayerNorm(hidden)
        self.norm_last   = nn.LayerNorm(hidden)
        self.norm_delta  = nn.LayerNorm(hidden)

        # Head: 4×768 → bottleneck → num_classes
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MotionVideoMAE: {n_frozen/1e6:.1f}M frozen, {n_train/1e6:.1f}M trainable")

    # ── parameter groups ──────────────────────────────────────────────────────

    def _head_params(self):
        return (
            list(self.norm_global.parameters())
            + list(self.norm_first.parameters())
            + list(self.norm_last.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
        return self._head_params()

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.75):
        """Parameter groups with layer-wise LR decay for AdamW.

        Trainable blocks get geometrically decaying LR from the top:
            block 11 → backbone_lr
            block 10 → backbone_lr × llrd
            block  9 → backbone_lr × llrd²
            ...
        Head and norms always use head_lr.
        """
        groups = [{"params": self._head_params(), "lr": head_lr}]
        n_layers = len(self.encoder.encoder.layer)
        # Iterate from last block down to first trainable block
        for depth, idx in enumerate(range(n_layers - 1, self.num_frozen_blocks - 1, -1)):
            block_lr = backbone_lr * (llrd ** depth)
            groups.append({
                "params": list(self.encoder.encoder.layer[idx].parameters()),
                "lr": block_lr,
            })
        return groups

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE [0.5, 0.5, 0.5] normalisation (fused)
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # 4 → 16 frames via linear interpolation
        x = _linear_upsample(x, _TARGET_FRAMES)

        # Encode: (B, 1568, 768) — no CLS token in VideoMAEModel
        tokens = self.encoder(pixel_values=x).last_hidden_state
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, -1)  # (B, 8, 196, 768)

        global_feat = tokens.mean(dim=(1, 2))       # (B, 768) — scene context
        first       = tokens[:, 0].mean(dim=1)      # (B, 768) — start of action
        last        = tokens[:, -1].mean(dim=1)     # (B, 768) — end of action
        delta       = last - first                  # (B, 768) — net motion direction

        feat = torch.cat([
            self.norm_global(global_feat),
            self.norm_first(first),
            self.norm_last(last),
            self.norm_delta(delta),
        ], dim=1)  # (B, 3072)

        return self.head(feat)

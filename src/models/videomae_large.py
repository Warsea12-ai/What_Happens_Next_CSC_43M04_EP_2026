"""VideoMAELarge — VideoMAE-Large-Kinetics backbone + espagne-style MLP head.

Motivation
----------
VideoMAE-Base-SSv2 achieves 49.71% on evaluate.py with the 2frozen+LLRD recipe.
VideoMAE-Large-Kinetics (MCG-NJU/videomae-large-finetuned-kinetics) is 3.5× larger:
  - 24 transformer blocks (vs 12)
  - 1024-dim hidden (vs 768)
  - 307M parameters (vs 86M)

Hypothesis: more capacity → richer representations → better temporal ordering,
even though pre-trained on Kinetics rather than SSv2. The LLRD recipe prevents
catastrophic forgetting while adapting to SSv2-style tasks.

Architecture
------------
Same recipe as espagne (track_B_videomae_ssv2_2frozen):
  - num_frozen_blocks=4 (analogous to 2/12 ≈ 4/24, same relative fraction)
  - LLRD 0.80 across 20 trainable blocks
  - MLP head: Dropout → Linear(1024, 2048) → GELU → Dropout → Linear(2048, 33)
  - All augmentations: RandomResizedCrop, flip, color jitter, mixup

Input normalisation: ImageNet → VideoMAE (same conversion as other VideoMAE models)

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, 33)
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


class VideoMAELarge(nn.Module):
    """VideoMAE-Large-Kinetics + espagne-style head (2frozen → 4frozen for Large)."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-large-finetuned-kinetics",
        num_frozen_blocks: int = 4,
        dropout: float = 0.25,
        head_hidden: int = 2048,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze embeddings and first num_frozen_blocks transformer blocks
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.encoder.encoder.layer):
            for p in block.parameters():
                p.requires_grad = (i >= num_frozen_blocks)

        hidden = self.encoder.config.hidden_size  # 1024

        # Classification head: same design as espagne but wider for 1024-dim input
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VideoMAELarge: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self.num_frozen_blocks < len(self.encoder.encoder.layer):
            for block in self.encoder.encoder.layer[self.num_frozen_blocks:]:
                block.train(mode)
        return self

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.80):
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        n_layers = len(self.encoder.encoder.layer)
        for depth, idx in enumerate(range(n_layers - 1, self.num_frozen_blocks - 1, -1)):
            groups.append({
                "params": list(self.encoder.encoder.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE normalisation
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        x = _linear_upsample(x, _TARGET_FRAMES)

        tokens = self.encoder(pixel_values=x).last_hidden_state.to(dtype)  # (B, 1568, 1024)
        feat = tokens.mean(1)  # (B, 1024) — global mean pool (espagne-style)
        return self.head(feat)  # (B, 33)

"""MViT-V2 wrapper for SthSth-finetuned `facebook/mvit-base-finetuned-ssv2`.

Multiscale Vision Transformer V2 (Li et al. CVPR 2022, ICCV 2021 v1) :
designed specifically for temporal reasoning. Strong baseline on
Something-Something (~67% top-1 reported on SSv2 dataset).

We discard the SSv2 classifier (174 classes) and replace it with our 33-class
head. Backbone weights stay frozen-or-trainable via num_frozen_blocks.

Input  : (B, T=16, C, H=224, W=224) — MViT-V2 expects 16 frames natively.
Output : (B, 33) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import MvitModel

from temporal_interp import interp_temporal

_TARGET_FRAMES = 16


class MViTV2SSv2(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "facebook/mvit-base-finetuned-ssv2",
        num_frozen_blocks: int = 0,
        dropout: float = 0.25,
        head_hidden: int = 1024,
        interp_mode: str = "aligned",
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks
        if interp_mode not in ("aligned", "centered", "repeat"):
            raise ValueError(f"interp_mode must be aligned|centered|repeat, got {interp_mode!r}")
        self.interp_mode = interp_mode

        self.encoder = MvitModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze patch embeddings + first N transformer blocks.
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        blocks = self.encoder.encoder.layer
        for i, block in enumerate(blocks):
            for p in block.parameters():
                p.requires_grad = (i >= num_frozen_blocks)

        hidden = self.encoder.config.hidden_size  # 768 for base
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
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
        print(f"MViTV2SSv2: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}/{len(blocks)}, interp={interp_mode})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.embeddings.eval()
        for i, block in enumerate(self.encoder.encoder.layer):
            block.train(mode if i >= self.num_frozen_blocks else False)
        return self

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.80):
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        layers = self.encoder.encoder.layer
        n = len(layers)
        for depth, idx in enumerate(range(n - 1, self.num_frozen_blocks - 1, -1)):
            params = list(layers[idx].parameters())
            if params:
                groups.append({"params": params, "lr": backbone_lr * (llrd ** depth)})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample 4 -> 16 frames according to interp_mode
        x = interp_temporal(x, target_T=_TARGET_FRAMES, mode=self.interp_mode)
        # MViT expects (B, T, C, H, W) as pixel_values
        out = self.encoder(pixel_values=x)
        # MViT pools to (B, hidden) via mean pool on last_hidden_state
        cls = out.last_hidden_state.mean(dim=1)  # (B, 768)
        return self.head(cls)

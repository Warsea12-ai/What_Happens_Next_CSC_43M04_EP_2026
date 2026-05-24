"""TimeSformerHead — TimeSformer (divided space-time attention) + MLP head.

Architecture
------------
HuggingFace `TimesformerModel` : Pure divided space-time attention transformer
(Bertasius et al. 2021), 12 blocks, hidden 768, 121M params.
- backbone : `facebook/timesformer-base-finetuned-ssv2` (~62% top-1 SSv2)
- num_frames input must match the pre-training (8 by default)
- output : pooled CLS token of shape (B, 768)

Wrapper :
- charge le backbone + optionnellement gèle N blocs (frozen → LLRD via train_trackB)
- jette le classifier d'origine (pas exposé via TimesformerModel)
- ajoute la tête MLP style espagne : LN -> Dropout -> Linear(hidden, head_hidden)
  -> GELU -> Dropout -> Linear(head_hidden, num_classes)

Input  : (B, T, C, H, W), ImageNet-normalised (will be re-arranged to (B, T, C, H, W)
         which is the format TimesformerModel expects via pixel_values).
Output : (B, num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import TimesformerModel


class TimesformerHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "facebook/timesformer-base-finetuned-ssv2",
        num_frozen_blocks: int = 0,
        dropout: float = 0.25,
        head_hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks
        self.encoder = TimesformerModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze patch embeddings and first num_frozen_blocks transformer blocks
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        layers = self.encoder.encoder.layer  # ModuleList[TimesformerLayer]
        for i, block in enumerate(layers):
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
        print(f"TimesformerHead: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}/{len(layers)})")

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
        """LLRD across the trainable backbone blocks (top → bottom)."""
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        layers = self.encoder.encoder.layer
        n = len(layers)
        for depth, idx in enumerate(range(n - 1, self.num_frozen_blocks - 1, -1)):
            params = list(layers[idx].parameters())
            if params:
                groups.append({"params": params, "lr": backbone_lr * (llrd ** depth)})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TimesformerModel expects (B, T, C, H, W) → returns last_hidden_state
        # (B, num_tokens, hidden). CLS token is index 0.
        out = self.encoder(pixel_values=x)
        cls = out.last_hidden_state[:, 0]  # (B, 768)
        return self.head(cls)

"""X-CLIP wrapper for `microsoft/xclip-base-patch32-16-frames`.

X-CLIP (Ni et al. ECCV 2022) extends CLIP to video. Uses video adapter
modules to add cross-frame attention on top of CLIP image encoder. The
pretraining is contrastive (video <-> text), giving very different features
from purely supervised SSv2-finetuned models — useful for ensemble diversity.

We use only the vision tower (`XCLIPModel.vision_model`) + an MLP head;
the text tower stays unused since we have explicit labels.

Input  : (B, T=4, C, H=224, W=224) — upsampled to 16 internally.
Output : (B, 33) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import XCLIPModel

from temporal_interp import interp_temporal

_TARGET_FRAMES = 16


class XCLIPVideo(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "microsoft/xclip-base-patch32-16-frames",
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

        full = XCLIPModel.from_pretrained(backbone, use_safetensors=True)
        # Keep ONLY the video-aware vision encoder (drop text + projection layers)
        self.vision_model     = full.vision_model       # CLIP ViT-B/32
        self.mit              = full.mit                # multiframe integration transformer
        self.prompts_generator = None  # discard
        del full

        # Freeze patch embed + first N CLIP blocks.
        for p in self.vision_model.embeddings.parameters():
            p.requires_grad = False
        blocks = self.vision_model.encoder.layers
        for i, block in enumerate(blocks):
            for p in block.parameters():
                p.requires_grad = (i >= num_frozen_blocks)

        hidden = self.vision_model.config.hidden_size  # 768
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
        print(f"XCLIPVideo: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}/{len(blocks)}, interp={interp_mode})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.embeddings.eval()
        for i, block in enumerate(self.vision_model.encoder.layers):
            block.train(mode if i >= self.num_frozen_blocks else False)
        return self

    def head_parameters(self):
        return list(self.head.parameters()) + list(self.mit.parameters())

    def backbone_parameters(self):
        return [p for p in self.vision_model.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.80):
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        layers = self.vision_model.encoder.layers
        n = len(layers)
        for depth, idx in enumerate(range(n - 1, self.num_frozen_blocks - 1, -1)):
            params = list(layers[idx].parameters())
            if params:
                groups.append({"params": params, "lr": backbone_lr * (llrd ** depth)})
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample 4 -> 16 frames (X-CLIP base expects 16)
        x = interp_temporal(x, target_T=_TARGET_FRAMES, mode=self.interp_mode)
        B, T, C, H, W = x.shape
        # Reshape (B, T, C, H, W) -> (B*T, C, H, W) for the vision encoder
        pix = x.reshape(B * T, C, H, W)
        vout = self.vision_model(pixel_values=pix)
        # CLIP gives (B*T, num_patches+1, hidden); take CLS token
        feats = vout.last_hidden_state[:, 0]              # (B*T, hidden)
        feats = feats.reshape(B, T, -1)                    # (B, T, hidden)
        # Multiframe integration transformer (X-CLIP's temporal aggregator)
        agg = self.mit(feats).last_hidden_state.mean(dim=1)  # (B, hidden)
        return self.head(agg)

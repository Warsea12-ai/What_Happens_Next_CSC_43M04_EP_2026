"""VJEPA2Head — V-JEPA 2 ViT-G (1.01B, SSv2-finetuned) + espagne-style MLP head.

Why this architecture can exceed 60% on evaluate.py
-----------------------------------------------------
V-JEPA 2 ViT-G is a qualitatively different scale from every model tested so far:

  Model                  Params   Pre-train domain  Status
  VideoMAE-Base-SSv2       86M    SSv2 (exact)      49.71% evaluate.py (best)
  VideoMAE-Large-Kinetics 307M    Kinetics          running
  DINOv3-ViT-H (frozen)   841M    1.7B images       running
  V-JEPA 2 ViT-G (this)  1012M   SSv2 (exact)      ← largest, same domain

V-JEPA 2 ViT-G is 12× larger than VideoMAE-Base-SSv2 AND was trained on SSv2
with video joint-embedding objectives — the same dataset and temporal domain as
our task. Its representations already encode rich temporal structure at exactly
the right granularity for hand-object interaction ordering.

Key design choices that limit the train/eval gap
-------------------------------------------------
1. Dynamic positional encoding: no parameter mismatch at any resolution or
   frame count — all 1012M pretrained weights load perfectly at 224×224/16f.
2. Freeze 32/40 blocks: run frozen part in no_grad (saves activation memory),
   only gradient-compute through last 8 blocks (~202M trainable backbone params).
3. LLRD 0.80: backbone_lr × 0.80^depth_from_output per trainable block,
   protecting the rich SSv2 representations from large updates.

Architecture
------------
Input   : (B, T=4, C=3, H=224, W=224)  ImageNet-normalised
1. Linear upsample 4 → 16 frames (same interpolation as all VideoMAE models)
2. V-JEPA 2 embeddings: 3D-conv tubelets → (B, 1568, 1408)
3. Frozen transformer blocks 0…31: run in torch.no_grad (no activation storage)
4. Trainable transformer blocks 32…39: run with autograd
5. Global mean pool over 1568 tokens → (B, 1408)
6. MLP: Dropout(0.25) → Linear(1408, 2048) → GELU → Dropout(0.125) → Linear(2048, 33)
Output  : (B, 33)

No normalisation conversion needed: V-JEPA 2 uses ImageNet normalisation
(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), same as our pipeline.

Cache: model lives at /Data/vianney.gauthier/hub — readable by all hosts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VJEPA2Model

_TARGET_FRAMES = 16


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


class VJEPA2Head(nn.Module):
    """V-JEPA 2 ViT-G (SSv2-finetuned) + espagne-style 33-class head."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "facebook/vjepa2-vitg-fpc64-384-ssv2",
        cache_dir: str = "/Data/vianney.gauthier/hub",
        num_frozen_blocks: int = 32,
        dropout: float = 0.25,
        head_hidden: int = 2048,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks

        # Load full VJEPA2Model then keep only the encoder (discard the JEPA predictor)
        full = VJEPA2Model.from_pretrained(
            backbone,
            cache_dir=cache_dir,
            frames_per_clip=_TARGET_FRAMES,
            crop_size=224,
            use_safetensors=True,
        )
        self.encoder = full.encoder  # VJEPA2Encoder: .embeddings + .layer[0..39]
        del full  # release predictor weights immediately

        # Freeze patch embeddings and first num_frozen_blocks transformer blocks
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.encoder.layer):
            for p in block.parameters():
                p.requires_grad = (i >= num_frozen_blocks)

        hidden = 1408  # ViT-G hidden dimension

        # Classification head: same design as espagne, scaled for 1408-dim input
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
        print(f"VJEPA2Head: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}/40)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.embeddings.eval()
        for i, block in enumerate(self.encoder.layer):
            if i < self.num_frozen_blocks:
                block.eval()
            else:
                block.train(mode)
        return self

    def unfreeze_to(self, n_frozen: int) -> None:
        """Reduce frozen encoder blocks to n_frozen; self.train() already uses self.num_frozen_blocks."""
        self.num_frozen_blocks = n_frozen
        for i, block in enumerate(self.encoder.layer):
            for p in block.parameters():
                p.requires_grad = (i >= n_frozen)
        n_f = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  VJEPA2Head thawed → {n_frozen} frozen blocks "
              f"({n_f/1e6:.0f}M frozen, {n_t/1e6:.1f}M trainable)")

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.80):
        """LLRD groups: head at head_lr, trainable blocks decay geometrically."""
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        n = len(self.encoder.layer)
        for depth, idx in enumerate(range(n - 1, self.num_frozen_blocks - 1, -1)):
            groups.append({
                "params": list(self.encoder.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = _linear_upsample(x, _TARGET_FRAMES)

        # ── Frozen part: embeddings + blocks 0…(num_frozen-1) in no_grad ──────
        # torch.no_grad() prevents activation graph storage for frozen layers,
        # saving ~(num_frozen_blocks/40) × activation memory without losing
        # any gradient information (these blocks have requires_grad=False anyway).
        with torch.no_grad():
            h = self.encoder.embeddings(pixel_values_videos=x)  # (B, 1568, 1408)
            for block in self.encoder.layer[:self.num_frozen_blocks]:
                h = block(h)[0]  # each block returns (hidden_states, *extras)

        # Detach from no_grad context then cast to training dtype
        h = h.detach().to(dtype)

        # ── Trainable part: blocks num_frozen…39 with full autograd ────────────
        for block in self.encoder.layer[self.num_frozen_blocks:]:
            h = block(h)[0]

        # Global mean pool → classification head
        feat = h.mean(dim=1)          # (B, 1408)
        return self.head(feat)        # (B, 33)

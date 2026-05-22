"""VideoMAEMultiScale — VideoMAE-SSv2 + Multi-scale intermediate features.

Key insight over espagne (2frozen + simple MLP head):
  espagne pools only the FINAL layer of VideoMAE, which is optimised for
  action recognition (label = order-invariant). Intermediate layers encode
  LOCAL motion direction and spatial change — exactly what temporal ordering
  requires.

Architecture:
  1. VideoMAE backbone (2frozen, LLRD 0.80), output_hidden_states=True
  2. Extract features at blocks 3, 7, 11 (early / mid / final scales)
  3. Per-frame spatial pool at each scale -> 3 × (B, 4, D)
  4. Multi-scale fusion: concat -> (B, 4, 3D) -> linear project -> (B, 4, D)
  5. Temporal transformer WITHOUT positional encodings
     (the model must infer ordering from frame CONTENT, not input position)
  6. CLS output -> MLP head -> 33 classes

Why no temporal PE:
  In the ordering task, the position of a frame in the input sequence is
  meaningless — we don't know if "frame 0" is first or last temporally.
  A PE would give the model a spurious positional cue. Without PE the
  temporal transformer is a SET transformer: it discovers ordering purely
  from self-attention over frame content.

Same 2frozen + LLRD 0.80 training recipe as espagne.

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import math
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
_N_FRAMES      = 4
_ORIG_GROUPS   = (0, 2, 5, 7)

# Indices into hidden_states tuple (embedding=0, block_k = k+1)
# We pick after blocks 3, 7, 11  ->  indices 4, 8, 12
_SCALE_INDICES = (4, 8, 12)


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


class _TemporalBlock(nn.Module):
    """Transformer block WITHOUT positional encoding — pure set attention."""
    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ff(self.norm2(x))
        return x


class VideoMAEMultiScale(nn.Module):
    """VideoMAE + multi-scale intermediate features + set-based temporal transformer."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frozen_blocks: int = 2,
        scale_indices: tuple = _SCALE_INDICES,
        n_temporal_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.25,
        head_hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks
        self.scale_indices = scale_indices

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        for i, block in enumerate(self.encoder.encoder.layer):
            trainable = i >= num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = trainable

        hidden = self.encoder.config.hidden_size  # 768 for base

        # Fuse multi-scale features: len(scale_indices) * hidden -> hidden
        n_scales = len(scale_indices)
        self.scale_proj = nn.Sequential(
            nn.Linear(n_scales * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # CLS token (no positional encoding — set transformer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(hidden, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden)

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
        print(f"VideoMAEMultiScale: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}, scales={scale_indices})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self.num_frozen_blocks < len(self.encoder.encoder.layer):
            for block in self.encoder.encoder.layer[self.num_frozen_blocks:]:
                block.train(mode)
        return self

    def head_parameters(self):
        return (
            list(self.scale_proj.parameters())
            + [self.cls_token]
            + list(self.temporal_blocks.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.head.parameters())
        )

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.85):
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        n_layers = len(self.encoder.encoder.layer)
        for depth, idx in enumerate(range(n_layers - 1, self.num_frozen_blocks - 1, -1)):
            groups.append({
                "params": list(self.encoder.encoder.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        x = _linear_upsample(x, _TARGET_FRAMES)

        # Extract multi-scale intermediate features
        outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        # hidden_states: tuple of (embedding + 12 block outputs), each (B, 1568, D)

        scale_feats = []
        for idx in self.scale_indices:
            h = outputs.hidden_states[idx].to(dtype)  # (B, 1568, D)
            D = h.shape[-1]
            h_3d = h.view(B, _N_TEMPORAL, _N_SPATIAL, D)          # (B, 8, 196, D)
            frame_h = h_3d[:, list(_ORIG_GROUPS)].mean(2)          # (B, 4, D) — spatial pool
            scale_feats.append(frame_h)

        # Multi-scale fusion: concat along feature dim, project back to D
        multi = torch.cat(scale_feats, dim=-1)          # (B, 4, n_scales*D)
        frame_feats = self.scale_proj(multi)             # (B, 4, D)

        # Set-based temporal transformer: CLS + 4 frame tokens (no PE)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        seq = torch.cat([cls, frame_feats], dim=1)        # (B, 5, D)
        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])           # (B, D)

        return self.head(cls_out)

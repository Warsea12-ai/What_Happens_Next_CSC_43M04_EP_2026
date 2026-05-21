"""VideoMAEPairAttn — VideoMAE-SSv2 + All-Pairs Temporal Comparison Head.

Key innovation over VideoMAETemporalHead:
  - Operates on 4 real frame features (groups 0,2,5,7 — original frame positions)
  - Computes all C(4,2)=6 pairwise differences (TemporalHead only uses 3)
  - Temporal self-attention on 4 real frames + CLS (5 tokens, not 9 interpolated)
  - Classification feature: cat([CLS, 6 deltas]) = 7×D

All 6 pairwise comparisons for 4 frames:
  consecutive: (f1-f0), (f2-f1), (f3-f2)
  skip-one:    (f2-f0), (f3-f1)
  full-span:   (f3-f0)
This covers the complete temporal relationship space, critical for 33-class ordering.

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformers import VideoMAEModel

from models.lora_utils import apply_lora, get_lora_parameters, count_lora_params

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
_N_FRAMES      = 4  # number of original input frames

# Temporal groups corresponding to original frames 0-3 after 4->16 linear interp.
# Frames at positions 0,5,10,15 in the 16-frame sequence map to groups 0,2,5,7.
_ORIG_GROUPS = (0, 2, 5, 7)

# All C(4,2)=6 ordered pairs (i,j) with i<j
_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


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
    """Pre-LN self-attention block for the temporal transformer."""

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


class VideoMAEPairAttn(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frozen_blocks: int = 6,
        n_temporal_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.3,
        head_hidden: int = 1024,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self._use_lora = lora_rank > 0
        self.num_frozen_blocks = num_frozen_blocks

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Always freeze patch embeddings
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        if self._use_lora:
            # Freeze all encoder params, inject LoRA on Q,V of every block
            for p in self.encoder.parameters():
                p.requires_grad = False
            apply_lora(self.encoder, ["query", "value"], rank=lora_rank, alpha=lora_alpha)
        else:
            # Partial unfreezing: blocks [num_frozen_blocks:] are trainable
            for i, block in enumerate(self.encoder.encoder.layer):
                trainable = i >= num_frozen_blocks
                for p in block.parameters():
                    p.requires_grad = trainable

        hidden = self.encoder.config.hidden_size  # 768 for base, 1024 for large

        # Spatial attention pooling — one learned query, shared across all frame groups.
        # Different frame groups produce different attention distributions, so spatial
        # weighting is still frame-specific despite the shared query vector.
        self.spatial_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        # Temporal self-attention over [CLS] + 4 real frames = 5 tokens.
        # Using only the 4 original frames (not all 8 interpolated groups) gives
        # a cleaner temporal signal without pseudo-frame noise.
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Position IDs: 0=CLS, 1..4=frames 0-3
        self.temporal_pe = nn.Embedding(_N_FRAMES + 1, hidden)

        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(hidden, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden)

        # One LayerNorm per pairwise delta (6 total)
        self.pair_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in _PAIRS])

        # Head: [CLS (1xD) + 6 pairwise deltas (6xD)] = 7xD -> num_classes
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear((1 + len(_PAIRS)) * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_n   = count_lora_params(self.encoder) if self._use_lora else 0
        print(f"VideoMAEPairAttn: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(lora={self._use_lora} rank={lora_rank}, lora_params={lora_n/1e6:.2f}M)")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self._use_lora:
            from models.lora_utils import LoRALinear
            for m in self.encoder.modules():
                if isinstance(m, LoRALinear):
                    m.train(mode)
        elif self.num_frozen_blocks < len(self.encoder.encoder.layer):
            for block in self.encoder.encoder.layer[self.num_frozen_blocks:]:
                block.train(mode)
        return self

    def head_parameters(self):
        return (
            [self.spatial_query, self.cls_token]
            + list(self.temporal_pe.parameters())
            + list(self.temporal_blocks.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.pair_norms.parameters())
            + list(self.head.parameters())
        )

    def backbone_parameters(self):
        if self._use_lora:
            return []
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def lora_parameters(self):
        if not self._use_lora:
            return []
        return get_lora_parameters(self.encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet -> VideoMAE [0.5, 0.5, 0.5] renormalization
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # 4 -> 16 frames via linear interpolation
        x = _linear_upsample(x, _TARGET_FRAMES)

        # Backbone encoding: (B, 1568, D)
        tokens = self.encoder(pixel_values=x).last_hidden_state
        tokens = tokens.to(dtype)
        D = tokens.shape[-1]
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, D)  # (B, 8, 196, D)

        # Extract the 4 original frame temporal groups: (B, 4, 196, D)
        frame_tokens = tokens[:, list(_ORIG_GROUPS)]

        # Spatial attention pooling: one query attends to 196 spatial patches per frame
        ft_flat = frame_tokens.reshape(B * _N_FRAMES, _N_SPATIAL, D)  # (B*4, 196, D)
        q = self.spatial_query.expand(B * _N_FRAMES, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q, ft_flat.transpose(1, 2)) / math.sqrt(D),     # (B*4, 1, 196)
            dim=-1,
        )
        frame_feats = torch.bmm(attn_w, ft_flat).squeeze(1)            # (B*4, D)
        frame_feats = frame_feats.reshape(B, _N_FRAMES, D)             # (B, 4, D)

        # All 6 pairwise differences: f_j - f_i for all (i,j) pairs with i<j
        pair_feats = []
        for k, (i, j) in enumerate(_PAIRS):
            delta = frame_feats[:, j] - frame_feats[:, i]   # (B, D)
            pair_feats.append(self.pair_norms[k](delta))    # normalized

        # Temporal self-attention on 4 real frames + CLS (5 tokens total)
        pos_ids = torch.arange(1, _N_FRAMES + 1, device=device)
        pe = self.temporal_pe(pos_ids).unsqueeze(0)              # (1, 4, D)
        feats_pe = frame_feats + pe                               # (B, 4, D)

        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls = self.cls_token.expand(B, -1, -1) + cls_pe          # (B, 1, D)
        seq = torch.cat([cls, feats_pe], dim=1)                   # (B, 5, D)

        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])                   # (B, D)

        # Classify from [CLS + 6 pairwise deltas] = 7xD
        feat = torch.cat([cls_out] + pair_feats, dim=1)           # (B, 7D)
        return self.head(feat)

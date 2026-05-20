"""VideoMAETemporalHead — VideoMAE-Large/SSv2 + spatial attention + temporal transformer head.

Two key improvements over DynamicVideoMAE (which uses simple mean-pooling):

1. SSv2-domain backbone by default (MCG-NJU/videomae-base-finetuned-ssv2).
   SSv2 = hand-object manipulation, same visual domain as our 33 classes.
   vs. Kinetics = outdoor sports/scenes. Expected +2-4% from better transfer.

2. Spatial attention pooling (per temporal group):
   A learned query Q attends to all 196 spatial patch tokens per temporal group
   via scaled dot-product attention. Weights differ by group: the model learns
   "where the action is" rather than blending background uniformly.

3. Temporal self-attention (3 layers, [CLS] + 8 tokens):
   After spatial pooling we get (B, 8, D) temporal features. A lightweight
   3-layer transformer + prepended [CLS] captures long-range temporal
   dependencies (acceleration, deceleration, reversal, etc.) that simple
   per-frame differences miss.

4. Explicit delta features at original frame positions (inductive bias):
   Temporal groups 0, 2, 5, 7 align with original frames 0-3 after 4→16
   linear interpolation. delta_full, delta_first, delta_last give a strong
   motion direction signal that complements the CLS summary.

5. Optional LoRA (lora_rank > 0): applies rank-r adapters to Q,V projections
   of every encoder block. num_frozen_blocks is then ignored (all blocks adapt
   through LoRA). Good when GPU memory limits num_frozen_blocks.

Architecture:
  VideoMAE-Large → (B, 1568, D)
  → reshape (B, 8, 196, D)
  → SpatialAttnPool per group → (B, 8, D)
  → TemporalTransformer ([CLS]+8 tokens, 3 layers) → cls_out: (B, D)
  → cat([cls_out, Δfull, Δfirst, Δlast]) → (B, 4D)
  → Dropout → Linear(4D, head_hidden) → GELU → Dropout → Linear → 33

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

# Temporal groups that correspond to original frames 0-3 after 4→16 linear interpolation.
# linspace(0,3,16) → [0,0.2,0.4,...,3]; frame i maps to tubelet group round(i/3*7).
_ORIG_GROUPS = (0, 2, 5, 7)


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


class VideoMAETemporalHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frozen_blocks: int = 12,
        n_temporal_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.3,
        head_hidden: int = 1024,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks
        self._use_lora = lora_rank > 0

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Always freeze patch embeddings (Conv3d tubelet projector)
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        if self._use_lora:
            # Freeze all encoder params, then inject LoRA adapters on Q,V
            for p in self.encoder.parameters():
                p.requires_grad = False
            apply_lora(self.encoder, ["query", "value"], rank=lora_rank, alpha=lora_alpha)
        else:
            # Partial unfreezing: blocks [num_frozen_blocks:] are trainable
            for i, block in enumerate(self.encoder.encoder.layer):
                trainable = i >= num_frozen_blocks
                for p in block.parameters():
                    p.requires_grad = trainable

        hidden = self.encoder.config.hidden_size  # 1024 for Large

        # ── Spatial attention pooling ──────────────────────────────────────────
        # One learned query vector, shared across temporal groups.
        # The attention distribution over 196 patches DIFFERS per group (temporal context),
        # so this still learns group-specific spatial weighting.
        self.spatial_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        # ── Temporal transformer ───────────────────────────────────────────────
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Positions: 0=CLS, 1..N_TEMPORAL=temporal groups 0..7
        self.temporal_pe = nn.Embedding(_N_TEMPORAL + 1, hidden)

        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(hidden, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden)

        # ── Delta feature norms ────────────────────────────────────────────────
        self.norm_delta_full  = nn.LayerNorm(hidden)
        self.norm_delta_first = nn.LayerNorm(hidden)
        self.norm_delta_last  = nn.LayerNorm(hidden)

        # ── Classification head: [CLS + 3 deltas] = 4×hidden → num_classes ───
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VideoMAETemporalHead: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(lora={self._use_lora}, frozen_blocks={num_frozen_blocks})")

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep encoder in eval for stable BatchNorm/LayerNorm stats
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

    # ── parameter groups ──────────────────────────────────────────────────────

    def _head_params(self):
        return (
            [self.spatial_query, self.cls_token]
            + list(self.temporal_pe.parameters())
            + list(self.temporal_blocks.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.norm_delta_full.parameters())
            + list(self.norm_delta_first.parameters())
            + list(self.norm_delta_last.parameters())
            + list(self.head.parameters())
        )

    def head_parameters(self):
        return self._head_params()

    def backbone_parameters(self):
        if self._use_lora:
            return []  # LoRA handled separately via lora_parameters()
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def lora_parameters(self):
        if not self._use_lora:
            return []
        return get_lora_parameters(self.encoder)

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.85):
        groups = [{"params": self._head_params(), "lr": head_lr}]
        if self._use_lora:
            groups.append({"params": self.lora_parameters(), "lr": backbone_lr})
            return groups
        n_layers = len(self.encoder.encoder.layer)
        for depth, idx in enumerate(range(n_layers - 1, self.num_frozen_blocks - 1, -1)):
            groups.append({
                "params": list(self.encoder.encoder.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE [0.5, 0.5, 0.5] normalisation
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # 4 → 16 frames via linear interpolation
        x = _linear_upsample(x, _TARGET_FRAMES)

        # VideoMAE encoding: (B, N_TEMPORAL × N_SPATIAL, hidden) — no CLS token
        tokens = self.encoder(pixel_values=x).last_hidden_state  # (B, 1568, D)
        tokens = tokens.to(dtype)
        D = tokens.shape[-1]
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, D)       # (B, 8, 196, D)

        # ── Spatial attention pooling (per temporal group) ─────────────────────
        # Batch all temporal groups together for efficiency
        tokens_flat = tokens.reshape(B * _N_TEMPORAL, _N_SPATIAL, D)      # (B*8, 196, D)
        q = self.spatial_query.expand(B * _N_TEMPORAL, 1, D)              # (B*8, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q, tokens_flat.transpose(1, 2)) / math.sqrt(D),     # (B*8, 1, 196)
            dim=-1,
        )
        tfeats = torch.bmm(attn_w, tokens_flat).squeeze(1)                # (B*8, D)
        tfeats = tfeats.reshape(B, _N_TEMPORAL, D)                        # (B, 8, D)

        # ── Delta features at original frame positions ─────────────────────────
        g0, g1, g2, g3 = _ORIG_GROUPS
        f0, f1, f2, f3 = tfeats[:, g0], tfeats[:, g1], tfeats[:, g2], tfeats[:, g3]
        delta_full  = f3 - f0   # net motion: start → end
        delta_first = f1 - f0   # first-half motion
        delta_last  = f3 - f2   # second-half motion

        # ── Temporal self-attention with [CLS] ────────────────────────────────
        pos_ids = torch.arange(1, _N_TEMPORAL + 1, device=device)
        pe      = self.temporal_pe(pos_ids).unsqueeze(0)                  # (1, 8, D)
        tfeats  = tfeats + pe                                              # (B, 8, D)

        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls    = self.cls_token.expand(B, -1, -1) + cls_pe                # (B, 1, D)
        seq    = torch.cat([cls, tfeats], dim=1)                          # (B, 9, D)

        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])                           # (B, D)

        # ── Head ──────────────────────────────────────────────────────────────
        feat = torch.cat([
            cls_out,
            self.norm_delta_full(delta_full),
            self.norm_delta_first(delta_first),
            self.norm_delta_last(delta_last),
        ], dim=1)   # (B, 4D)

        return self.head(feat)

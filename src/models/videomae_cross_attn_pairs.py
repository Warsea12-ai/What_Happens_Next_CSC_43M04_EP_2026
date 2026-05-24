"""VideoMAECrossAttnPairs — VideoMAE-SSv2 + Spatial Cross-Attention Pairs.

Key innovation over VideoMAEPairAttn:
  Instead of computing F_j - F_i (difference of globally-pooled frame features),
  we compute SPATIAL CROSS-ATTENTION: frame_j's summary query attends to frame_i's
  full token map (196 tokens). This captures WHICH spatial regions of frame_i
  are most relevant to understanding frame_j — "where did the change happen?"

  For temporal ordering, spatial correspondence matters: "the hand moved from
  region X (frame_i) to region Y (frame_j)." Cross-attention finds region X,
  while the global difference only captures the average change.

  Pairwise feature = LayerNorm(cross_delta - f_i) where
    cross_delta = cross_attn(query=pool(frame_j), keys/values=tokens_i)

  This is:
    1. Asymmetric (j attends to i != i attends to j) — encodes temporal direction
    2. Spatially-resolved (which patch of i is most relevant to j)
    3. Efficient (1 query x 196 key-value pairs, not 196x196)

Training recipe: same as espagne (2frozen + LLRD 0.80) — the recipe that gives
the smallest train->evaluate gap.

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformers import VideoMAEModel

from temporal_interp import interp_temporal

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
_PAIRS         = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


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


class VideoMAECrossAttnPairs(nn.Module):
    """VideoMAE + Spatial Cross-Attention Pairs.

    For each pair (i,j) with i<j:
      1. Pool frame_i and frame_j via shared spatial attention query -> f_i, f_j
      2. Cross-attention: f_j (projected query) attends to tokens_i (keys/values)
         -> cross_feat = weighted combination of frame_i patches most relevant to frame_j
      3. Pair feature = LayerNorm(cross_feat - f_i)
         (what frame_i contributes when seen through frame_j, minus the frame_i baseline)
    """

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frozen_blocks: int = 2,
        n_temporal_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.25,
        head_hidden: int = 1024,
        lora_rank: int = 0,
        lora_alpha: float = 0.0,
        interp_mode: str = "aligned",
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks
        # "aligned"  : input frames at output positions [0, 5, 10, 15] (historical).
        # "centered" : input frames centered in their segments → all 16 positions
        #              are symmetric linear blends (no edge bias).
        # "repeat"   : duplicate each input frame 4× (no interpolation).
        # See src/temporal_interp.py.
        if interp_mode not in ("aligned", "centered", "repeat"):
            raise ValueError(f"interp_mode must be aligned|centered|repeat, got {interp_mode!r}")
        self.interp_mode = interp_mode

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        for i, block in enumerate(self.encoder.encoder.layer):
            trainable = i >= num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = trainable

        # Optional LoRA on backbone attention (q/v projections, like videomae_lora).
        # rank=0 → no-op, preserves the existing ablation behavior.
        if lora_rank > 0:
            from models.lora_utils import apply_lora
            _alpha = lora_alpha if lora_alpha > 0 else float(2 * lora_rank)
            apply_lora(self.encoder, ["query", "value"], rank=lora_rank, alpha=_alpha)

        hidden = self.encoder.config.hidden_size  # 768 for base

        self.spatial_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        self.cross_q_proj = nn.Linear(hidden, hidden, bias=False)
        self.cross_k_proj = nn.Linear(hidden, hidden, bias=False)
        self.cross_v_proj = nn.Linear(hidden, hidden, bias=False)

        self.pair_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in _PAIRS])

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.temporal_pe = nn.Embedding(_N_FRAMES + 1, hidden)

        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(hidden, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden)

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
        print(f"VideoMAECrossAttnPairs: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self.num_frozen_blocks < len(self.encoder.encoder.layer):
            for block in self.encoder.encoder.layer[self.num_frozen_blocks:]:
                block.train(mode)
        return self

    def head_parameters(self):
        return (
            [self.spatial_query, self.cls_token,
             self.cross_q_proj.weight, self.cross_k_proj.weight, self.cross_v_proj.weight]
            + list(self.temporal_pe.parameters())
            + list(self.temporal_blocks.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.pair_norms.parameters())
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

        # Upsample 4 → 16 frames according to the configured interp_mode.
        # NB: _ORIG_GROUPS = (0, 2, 5, 7) was tuned for aligned mode (input
        # frames at positions [0, 5, 10, 15] → tubelets 0, 2, 5, 7). For
        # centered/repeat modes the mapping is slightly different but the
        # 4 chosen tubelets still capture the most "F_k-dominant" region.
        x = interp_temporal(x, target_T=_TARGET_FRAMES, mode=self.interp_mode)

        tokens = self.encoder(pixel_values=x).last_hidden_state.to(dtype)
        D = tokens.shape[-1]
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, D)

        frame_tokens = tokens[:, list(_ORIG_GROUPS)]               # (B, 4, 196, D)

        ft_flat = frame_tokens.reshape(B * _N_FRAMES, _N_SPATIAL, D)
        q = self.spatial_query.expand(B * _N_FRAMES, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q, ft_flat.transpose(1, 2)) / math.sqrt(D), dim=-1)
        frame_feats = torch.bmm(attn_w, ft_flat).squeeze(1).reshape(B, _N_FRAMES, D)

        scale = math.sqrt(D)
        pair_feats = []
        for k, (i, j) in enumerate(_PAIRS):
            q_j = self.cross_q_proj(frame_feats[:, j]).unsqueeze(1)   # (B, 1, D)
            k_i = self.cross_k_proj(frame_tokens[:, i])               # (B, 196, D)
            v_i = self.cross_v_proj(frame_tokens[:, i])               # (B, 196, D)
            cross_w = torch.softmax(
                torch.bmm(q_j, k_i.transpose(1, 2)) / scale, dim=-1)  # (B, 1, 196)
            cross_feat = torch.bmm(cross_w, v_i).squeeze(1)           # (B, D)
            # Subtract frame_i baseline: "what's specific to j->i, not just i"
            delta = cross_feat - frame_feats[:, i]
            pair_feats.append(self.pair_norms[k](delta))

        pos_ids = torch.arange(1, _N_FRAMES + 1, device=device)
        pe = self.temporal_pe(pos_ids).unsqueeze(0)
        feats_pe = frame_feats + pe
        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls = self.cls_token.expand(B, -1, -1) + cls_pe
        seq = torch.cat([cls, feats_pe], dim=1)
        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])

        feat = torch.cat([cls_out] + pair_feats, dim=1)
        return self.head(feat)

"""BidirectionalFramePairNet — symmetric cross-frame attention.

Extends FramePairNet with bidirectional cross-attention:
  before → after  (what in the initial state maps to the final state?)
  after  → before (what appeared in the final state that wasn't there before?)

Both streams run in parallel and are fused at the head:
  feat = cat([norm(fwd_pool), norm(bwd_pool), norm(delta)])  → head

This captures both "what left the scene" and "what entered the scene",
which the unidirectional version misses (it only asks before-queries about after).

Visual encoder: Qwen2-VL-2B (frozen, 1536-dim) or 7B (frozen, 3584-dim).
Architecture:
  before → n_cross_layers of CrossAttn(Q=before, KV=after) → fwd_tokens
  after  → n_cross_layers of CrossAttn(Q=after,  KV=before) → bwd_tokens
  fwd_pool = mean(fwd_tokens)   (B, hidden)
  bwd_pool = mean(bwd_tokens)   (B, hidden)
  delta    = mean(after) − mean(before)
  feat = cat([norm_fwd(fwd_pool), norm_bwd(bwd_pool), norm_delta(delta)])
  → head → num_classes

Trainable: ~46M (2 × n_cross_layers cross-attn) + head. Frozen: 665M.
"""
from __future__ import annotations

import gc
import torch
import torch.nn as nn

from models.qwen_vl_video import _imagenet_to_clip, _frames_to_patches

_PS  = 14
_TPS = 2
_SMS = 2
_HPO = (224 // _PS) // _SMS   # 8
_WPO = _HPO
_N_TOKENS = _HPO * _WPO       # 64


class _CrossAttnBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k)
        query = query + attn_out
        query = query + self.ff(self.norm_ff(query))
        return query


class BidirectionalFramePairNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "Qwen/Qwen2-VL-2B-Instruct",
        n_cross_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.3,
        head_hidden: int = 512,
    ) -> None:
        super().__init__()
        print(f"Loading {backbone} to CPU…")
        if "2.5" in backbone:
            from transformers import Qwen2_5_VLForConditionalGeneration
            _full = Qwen2_5_VLForConditionalGeneration.from_pretrained(backbone, dtype=torch.bfloat16)
        else:
            from transformers import Qwen2VLForConditionalGeneration
            _full = Qwen2VLForConditionalGeneration.from_pretrained(backbone, dtype=torch.bfloat16)
        self.visual = _full.model.visual
        del _full
        gc.collect()
        print("LLM deleted — keeping visual encoder only.")

        for p in self.visual.parameters():
            p.requires_grad = False

        vcfg = self.visual.config
        self.spatial_merge_size = getattr(vcfg, "spatial_merge_size", 2)

        _ps, _tps = vcfg.patch_size, vcfg.temporal_patch_size
        _H, _W = 224, 224
        _tg = 2 // _tps if 2 >= _tps else 1
        _hp, _wp = _H // _ps, _W // _ps
        _dummy_pv = torch.zeros(_tg * _hp * _wp, 3 * _tps * _ps * _ps, dtype=torch.bfloat16)
        _dummy_grid = torch.tensor([[_tg, _hp, _wp]], dtype=torch.long)
        with torch.no_grad():
            _out = self.visual(_dummy_pv, grid_thw=_dummy_grid)
            _raw = _out.last_hidden_state if hasattr(_out, "last_hidden_state") else _out
        hidden = int(_raw.reshape(_N_TOKENS, -1).shape[-1])

        # Forward stream: before attends to after
        self.fwd_layers = nn.ModuleList([
            _CrossAttnBlock(hidden, n_heads, dropout) for _ in range(n_cross_layers)
        ])
        # Backward stream: after attends to before
        self.bwd_layers = nn.ModuleList([
            _CrossAttnBlock(hidden, n_heads, dropout) for _ in range(n_cross_layers)
        ])

        self.norm_fwd   = nn.LayerNorm(hidden)
        self.norm_bwd   = nn.LayerNorm(hidden)
        self.norm_delta = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.visual.parameters())
        n_train  = sum(p.numel() for p in self.head_parameters())
        print(f"BidirectionalFramePairNet: {n_frozen/1e6:.0f}M frozen, "
              f"{n_train/1e6:.1f}M trainable (hidden={hidden})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        return self

    def head_parameters(self):
        params = (
            list(self.norm_fwd.parameters())
            + list(self.norm_bwd.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )
        for layer in self.fwd_layers:
            params += list(layer.parameters())
        for layer in self.bwd_layers:
            params += list(layer.parameters())
        return params

    def _encode_pair(self, pair: torch.Tensor) -> torch.Tensor:
        B, _, C, H, W = pair.shape
        device = pair.device
        tg = 1
        hp = H // _PS
        wp = W // _PS
        pair_clip = _imagenet_to_clip(pair)
        pixel_values = _frames_to_patches(pair_clip, _PS, _TPS).to(torch.bfloat16)
        grid_thw = torch.tensor([[tg, hp, wp]], dtype=torch.long, device=device).expand(B, 3)
        with torch.no_grad():
            out = self.visual(pixel_values, grid_thw=grid_thw)
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        return tokens.reshape(B, _N_TOKENS, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        before = self._encode_pair(x[:, :2]).to(dtype)   # (B, 64, hidden)
        after  = self._encode_pair(x[:, 2:]).to(dtype)   # (B, 64, hidden)

        # Forward: before queries about after
        fwd = before
        for layer in self.fwd_layers:
            fwd = layer(fwd, after)

        # Backward: after queries about before
        bwd = after
        for layer in self.bwd_layers:
            bwd = layer(bwd, before)

        fwd_pool = fwd.mean(dim=1)                               # (B, hidden)
        bwd_pool = bwd.mean(dim=1)                               # (B, hidden)
        delta    = after.mean(dim=1) - before.mean(dim=1)       # (B, hidden)

        feat = torch.cat([
            self.norm_fwd(fwd_pool),
            self.norm_bwd(bwd_pool),
            self.norm_delta(delta),
        ], dim=1)   # (B, 3*hidden)

        return self.head(feat)

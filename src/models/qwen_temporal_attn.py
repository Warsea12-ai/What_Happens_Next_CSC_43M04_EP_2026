"""QwenTemporalAttn — Qwen2-VL encoder + learned temporal self-attention head.

Replaces the mean-pooling aggregation of QwenVLVideo with a learned
self-attention transformer over all 128 output tokens (2 temporal groups × 64).

The key insight: mean-pooling discards the spatial token identity. Here,
a learned [CLS] token attends to all 128 patch tokens via self-attention,
plus we keep the motion-aware delta features. The [CLS] output is a
learned aggregation that weights informative regions dynamically.

Architecture:
  128 patch tokens (B, 128, hidden)
  + prepend [CLS] token → (B, 129, hidden)
  + temporal position encoding (group 0 vs 1)
  → 2 layers of Multi-Head Self-Attention
  → CLS output = (B, hidden)
  delta = mean(last_64) − mean(first_64)
  feat = cat([norm_cls(cls), norm_delta(delta)])  (B, 2*hidden)
  → head → num_classes

Trainable: cls_token + pos_enc + 2×MHSA + head ≈ 12M
Frozen: Qwen2-VL visual encoder (665M for 2B, 3.5B for 7B)
"""
from __future__ import annotations

import gc
import torch
import torch.nn as nn

from models.qwen_vl_video import _imagenet_to_clip, _frames_to_patches


class _SelfAttnBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class QwenTemporalAttn(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "Qwen/Qwen2-VL-2B-Instruct",
        n_attn_layers: int = 2,
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
        self.patch_size          = vcfg.patch_size
        self.temporal_patch_size = vcfg.temporal_patch_size
        self.spatial_merge_size  = getattr(vcfg, "spatial_merge_size", 2)

        ps, tps = vcfg.patch_size, vcfg.temporal_patch_size
        _H, _W = 224, 224
        _tg = 4 // tps
        _hp, _wp = _H // ps, _W // ps
        _dummy_pv = torch.zeros(_tg * _hp * _wp, 3 * tps * ps * ps, dtype=torch.bfloat16)
        _dummy_grid = torch.tensor([[_tg, _hp, _wp]], dtype=torch.long)
        with torch.no_grad():
            _out = self.visual(_dummy_pv, grid_thw=_dummy_grid)
            _raw = _out.last_hidden_state if hasattr(_out, "last_hidden_state") else _out
        hidden = int(_raw.shape[-1])

        # [CLS] token and temporal group position encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 3 positions: 0=CLS, 1=temporal group 0, 2=temporal group 1
        self.pos_embed = nn.Embedding(3, hidden)

        self.attn_layers = nn.ModuleList([
            _SelfAttnBlock(hidden, n_heads, dropout) for _ in range(n_attn_layers)
        ])

        self.norm_cls   = nn.LayerNorm(hidden)
        self.norm_delta = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.visual.parameters())
        n_train  = sum(p.numel() for p in self.head_parameters())
        print(f"QwenTemporalAttn: {n_frozen/1e6:.0f}M frozen, "
              f"{n_train/1e6:.1f}M trainable (hidden={hidden})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        return self

    def head_parameters(self):
        params = (
            [self.cls_token]
            + list(self.pos_embed.parameters())
            + list(self.norm_cls.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )
        for layer in self.attn_layers:
            params += list(layer.parameters())
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        ps   = self.patch_size
        tps  = self.temporal_patch_size
        sms  = self.spatial_merge_size
        hp   = H // ps
        wp   = W // ps
        tg   = T // tps
        hpo  = hp // sms
        wpo  = wp // sms

        x_clip = _imagenet_to_clip(x)
        pixel_values = _frames_to_patches(x_clip, ps, tps).to(torch.bfloat16)
        grid_thw = torch.tensor([[tg, hp, wp]], dtype=torch.long, device=device).expand(B, 3)

        with torch.no_grad():
            out = self.visual(pixel_values, grid_thw=grid_thw)
        raw = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        tokens = raw.to(dtype).reshape(B, tg, hpo * wpo, -1)  # (B, 2, 64, hidden)

        first_tokens = tokens[:, 0]   # (B, 64, hidden)
        last_tokens  = tokens[:, 1]   # (B, 64, hidden)

        # Add temporal group position embeddings
        pos_0 = self.pos_embed(torch.tensor(1, device=device))   # group 0
        pos_1 = self.pos_embed(torch.tensor(2, device=device))   # group 1
        first_tokens = first_tokens + pos_0
        last_tokens  = last_tokens  + pos_1

        # Prepend [CLS] with position 0
        cls = self.cls_token.expand(B, -1, -1)
        pos_cls = self.pos_embed(torch.tensor(0, device=device))
        cls = cls + pos_cls

        # Concatenate: [CLS, group0_tokens, group1_tokens] → (B, 129, hidden)
        seq = torch.cat([cls, first_tokens, last_tokens], dim=1)

        for layer in self.attn_layers:
            seq = layer(seq)

        cls_out = seq[:, 0]                                           # (B, hidden)
        delta   = last_tokens.mean(dim=1) - first_tokens.mean(dim=1) # (B, hidden)

        feat = torch.cat([self.norm_cls(cls_out), self.norm_delta(delta)], dim=1)
        return self.head(feat)

"""FramePairNet — Action recognition as spatial state-transition detection.

Paradigm shift from all existing Track B models:

  EXISTING (VideoMAE / QwenVL):
    all 4 frames → video backbone → temporal pool [global, first, last, delta] → head
    Limitation: temporal pooling discards spatial structure of the change.

  THIS MODEL:
    frames 0-1 → visual encoder → before_tokens  (B, 64, 1536)  -- initial state
    frames 2-3 → visual encoder → after_tokens   (B, 64, 1536)  -- final state
    CrossAttention(query=before, kv=after) → transition_tokens   -- what changed?
    pool(transition) + delta → head → class

The cross-attention is the key: each spatial patch in the before-state attends
to all patches in the after-state. This lets the model ask, per region:
  "this area had an object — is it still there? did it move? disappear?"

That patch-level spatial comparison captures changes that CLS-level delta misses:
  - Object held vs dropped: hand patch attends to object region
  - Contact point changed: spatial attention highlights different patches
  - Direction of movement: asymmetric cross-attention weights encode direction

Visual encoder: Qwen2-VL-2B (665M, frozen).
  Processes frame pairs (temporal_patch_size=2) in a single forward pass.
  before_tokens = visual(frame0, frame1) → (B, 64, 1536)
  after_tokens  = visual(frame2, frame3) → (B, 64, 1536)

Cross-Frame Attention (2 layers, ~23M learned params):
  Each layer: MultiheadAttention(query=before, kv=after) + FFN
  query ← norm(query + cross_attn(query, after, after))
  query ← norm(query + ffn(query))

Classification head:
  transition = mean(transition_tokens)          (B, 1536) — global change signal
  delta      = mean(after) − mean(before)       (B, 1536) — coarse state shift
  feat = cat([norm(transition), norm(delta)])    (B, 3072)
  → Dropout(0.4) → Linear(3072, 512) → GELU → Dropout(0.2) → Linear(512, 33)

Trainable params: ~23M (cross-attention) + ~1.6M (head) = ~25M total.
Frozen params: 665M (visual encoder).

Input : (B, T=4, C, H, W) ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import gc
import torch
import torch.nn as nn

from models.qwen_vl_video import _imagenet_to_clip, _frames_to_patches

_PS  = 14   # patch size
_TPS = 2    # temporal patch size (frames per group)
_SMS = 2    # spatial merge size
_HPO = (224 // _PS) // _SMS   # 8 — spatial tokens per axis after merge
_WPO = _HPO                   # 8
_N_TOKENS = _HPO * _WPO       # 64 tokens per frame-pair


class _CrossAttnBlock(nn.Module):
    """Standard Pre-LN cross-attention block (query attends to kv)."""

    def __init__(self, hidden: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden),   # expansion = 1 (keeps param count low)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Pre-LN cross-attention
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k)
        query = query + attn_out
        # Pre-LN FFN
        query = query + self.ff(self.norm_ff(query))
        return query


class FramePairNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "Qwen/Qwen2-VL-2B-Instruct",
        n_cross_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from transformers import Qwen2VLForConditionalGeneration

        print(f"Loading {backbone} to CPU…")
        _full = Qwen2VLForConditionalGeneration.from_pretrained(
            backbone, dtype=torch.bfloat16,
        )
        self.visual = _full.model.visual
        del _full
        gc.collect()
        print("LLM deleted — keeping visual encoder (665M).")

        for p in self.visual.parameters():
            p.requires_grad = False

        vcfg = self.visual.config
        hidden = vcfg.hidden_size   # 1536

        # Cross-frame attention: before-state queries attend to after-state
        self.cross_layers = nn.ModuleList([
            _CrossAttnBlock(hidden, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # Per-feature norms before concat
        self.norm_transition = nn.LayerNorm(hidden)
        self.norm_delta      = nn.LayerNorm(hidden)

        # Head: [transition, delta] = 2 × 1536 = 3072 → 512 → num_classes
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.visual.parameters())
        n_train  = sum(p.numel() for p in self.head_parameters())
        print(f"FramePairNet: {n_frozen/1e6:.0f}M frozen encoder, "
              f"{n_train/1e6:.1f}M trainable cross-attn + head")

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        return self

    # ── parameter helpers ─────────────────────────────────────────────────────

    def head_parameters(self):
        params = (
            list(self.norm_transition.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )
        for layer in self.cross_layers:
            params += list(layer.parameters())
        return params

    # ── frame-pair encoder ────────────────────────────────────────────────────

    def _encode_pair(
        self, pair: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a (B, 2, C, H, W) frame pair → (B, 64, 1536) spatial tokens."""
        B, _, C, H, W = pair.shape
        device = pair.device

        tg = 1   # T=2 frames, tps=2 → 1 temporal group
        hp = H // _PS   # 16
        wp = W // _PS   # 16

        pair_clip = _imagenet_to_clip(pair)
        pixel_values = _frames_to_patches(pair_clip, _PS, _TPS).to(torch.bfloat16)
        grid_thw = torch.tensor(
            [[tg, hp, wp]], dtype=torch.long, device=device,
        ).expand(B, 3)

        with torch.no_grad():
            tokens = self.visual(pixel_values, grid_thw=grid_thw)  # (B*64, 1536)

        return tokens.reshape(B, _N_TOKENS, -1)   # (B, 64, 1536)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, C, H, W) — ImageNet-normalised
        dtype = x.dtype

        # Encode before (frames 0-1) and after (frames 2-3) independently
        before = self._encode_pair(x[:, :2]).to(dtype)   # (B, 64, 1536)
        after  = self._encode_pair(x[:, 2:]).to(dtype)   # (B, 64, 1536)

        # Cross-attention: before patches attend to after patches
        # Each layer: "what in the before state corresponds to the after state?"
        q = before
        for layer in self.cross_layers:
            q = layer(q, after)   # transition_tokens: (B, 64, 1536)

        # Global features
        transition = q.mean(dim=1)                              # (B, 1536)
        delta      = after.mean(dim=1) - before.mean(dim=1)    # (B, 1536)

        feat = torch.cat([
            self.norm_transition(transition),
            self.norm_delta(delta),
        ], dim=1)   # (B, 3072)

        return self.head(feat)

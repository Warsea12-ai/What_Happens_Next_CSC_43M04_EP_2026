"""DINOv3 + Spatial Cross-Attention Pairs head.

Pourquoi cette refonte
-----------------------
La version existante (`dinov3_temporal`) ne pioche QUE le CLS token de chaque
frame (1×1280 par frame, 4 features au total) puis applique un transformer
temporel sur 5 tokens. Avec si peu d'info spatiale, la head ne peut pas
apprendre OÙ les choses bougent entre frames → score plafonné à 12.5% top-1
(vs 50% pour VideoMAE+cross_attn_pairs+best_combo_v2).

Ce module conserve EXACTEMENT le même backbone (DINOv3-ViT-H+ frozen) mais
applique le winning pattern :
  1. Extract per-frame PATCH tokens (196 patches/frame à 224x224, patch=16).
  2. Spatial attention pool par frame (1 query → 4 frame summaries f_0..f_3).
  3. Cross-attention pairs : pour chaque (i, j) avec i<j,
       q = projection(f_j) attend aux 196 patches de frame i,
       pair_feat = LayerNorm(cross_attn_output - f_i).
  4. Temporal CLS over 4 frame summaries → cls_out.
  5. Head : cat([cls_out, 6 pair deltas]) → MLP → 33 classes.

Recette best_combo_v2 (winner à 50%) : Prodigy(lr=1.0) + head_hidden=2048,
backbone frozen donc 0 trainable blocks (head-only training).

Input  : (B, T=4, C=3, 224, 224)  ImageNet-normalised
Output : (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformers import AutoModel


_N_FRAMES = 4
_PAIRS    = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
# DINOv3 has CLS + (typically 4) register tokens before patch tokens.
# Index where patches start is read from model config (num_register_tokens + 1).


class _TemporalBlock(nn.Module):
    """Pre-LN self-attention block (mêmes hyperparamètres que les autres _pairs)."""

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


class DINOv3Pairs(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        backbone_dtype: str = "bfloat16",
        n_temporal_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.25,
        head_hidden: int = 2048,
    ) -> None:
        super().__init__()

        _DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        bb_dtype = _DTYPES.get(backbone_dtype, torch.bfloat16)

        self.encoder = AutoModel.from_pretrained(
            backbone, use_safetensors=True, torch_dtype=bb_dtype,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        D = int(self.encoder.config.hidden_size)         # 1280 for ViT-H+, 4096 for 7B
        self.hidden_size = D
        # Number of [CLS] + register tokens before patches. DINOv3 config exposes
        # num_register_tokens; +1 for the CLS itself.
        self._n_prefix = 1 + int(getattr(self.encoder.config, "num_register_tokens", 4))
        print(f"  DINOv3Pairs config: hidden={D}, prefix_tokens={self._n_prefix} "
              f"(CLS + {self._n_prefix - 1} register tokens)")

        # ── Spatial attention pool per frame (1 shared learnable query) ──────
        self.spatial_query = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        # ── Cross-attention pairs ─────────────────────────────────────────────
        self.cross_q_proj = nn.Linear(D, D, bias=False)
        self.cross_k_proj = nn.Linear(D, D, bias=False)
        self.cross_v_proj = nn.Linear(D, D, bias=False)
        self.pair_norms = nn.ModuleList([nn.LayerNorm(D) for _ in _PAIRS])

        # ── Temporal CLS path over 4 frame summaries ──────────────────────────
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.temporal_pe = nn.Embedding(_N_FRAMES + 1, D)
        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(D, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(D)

        # ── Classification head : [CLS, 6 pair deltas] = 7 × D → MLP → num_classes ─
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear((1 + len(_PAIRS)) * D, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DINOv3Pairs: backbone={backbone}  hidden={D}  "
              f"{n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()           # never train the backbone
        return self

    def head_parameters(self):
        return [p for n, p in self.named_parameters()
                if not n.startswith("encoder.") and p.requires_grad]

    def backbone_parameters(self):
        return []     # fully frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape          # (B, 4, 3, 224, 224)
        dtype = x.dtype

        # ── Backbone forward per frame (frozen, no_grad) ─────────────────────
        x_flat = x.reshape(B * T, C, H, W)
        enc_dtype = next(self.encoder.parameters()).dtype
        with torch.no_grad():
            out = self.encoder(pixel_values=x_flat.to(enc_dtype))
        all_tokens = out.last_hidden_state.to(dtype)        # (B*T, 1+R+N, D)
        # Drop CLS + register tokens — keep spatial patches only
        patches = all_tokens[:, self._n_prefix:]              # (B*T, N, D)
        N = patches.shape[1]
        D = patches.shape[-1]
        frame_tokens = patches.reshape(B, T, N, D)            # (B, 4, N, D)

        # ── Spatial attention pool per frame (shared query) ───────────────────
        ft_flat = frame_tokens.reshape(B * T, N, D)
        q = self.spatial_query.expand(B * T, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q, ft_flat.transpose(1, 2)) / math.sqrt(D), dim=-1)
        frame_feats = torch.bmm(attn_w, ft_flat).squeeze(1).reshape(B, T, D)

        # ── Cross-attention pairs ─────────────────────────────────────────────
        scale = math.sqrt(D)
        pair_feats = []
        for k, (i, j) in enumerate(_PAIRS):
            q_j = self.cross_q_proj(frame_feats[:, j]).unsqueeze(1)    # (B, 1, D)
            k_i = self.cross_k_proj(frame_tokens[:, i])                # (B, N, D)
            v_i = self.cross_v_proj(frame_tokens[:, i])                # (B, N, D)
            cross_w = torch.softmax(
                torch.bmm(q_j, k_i.transpose(1, 2)) / scale, dim=-1)
            cross_feat = torch.bmm(cross_w, v_i).squeeze(1)
            delta = cross_feat - frame_feats[:, i]
            pair_feats.append(self.pair_norms[k](delta))

        # ── Temporal CLS path ─────────────────────────────────────────────────
        device = x.device
        pos_ids = torch.arange(1, _N_FRAMES + 1, device=device)
        pe = self.temporal_pe(pos_ids).unsqueeze(0)
        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls = self.cls_token.expand(B, -1, -1) + cls_pe
        seq = torch.cat([cls, frame_feats + pe], dim=1)
        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])

        feat = torch.cat([cls_out] + pair_feats, dim=1)
        return self.head(feat)

"""SigLIP-2 Giant + Spatial Cross-Attention Pairs head.

Why this backbone hasn't been tested yet
-----------------------------------------
- Released March 2025 by Google as the successor to SigLIP-1.
- Sigmoid contrastive loss on billions of (image, alt-text) pairs — a totally
  different pretraining objective than VideoMAE (MAE), V-JEPA2 (joint embedding
  prediction), DINOv3 (self-distillation) ALREADY in the ensemble.
- ~1B params (vision tower of siglip2-giant), large enough to qualify as "gros".
- Strict image encoder → forces temporal reasoning to live in the head only,
  which complements the strongly-temporal backbones already tested.
- Architectural diversity = useful for the final ensemble (uncorrelated errors).

Architecture
------------
1. 4 input frames → resize 224 → 384, ImageNet→SigLIP norm renorm.
2. Per-frame forward through SigLIP-2 (frozen) → (B*T, 576, 1536) patch tokens.
3. Per-frame spatial attention pool (1 learnable query) → 4 frame summaries.
4. Cross-attention pairs (winning pattern): for each (i, j) with i<j,
     q = f_j (projected) attends to spatial tokens of frame i,
     pair_feat = LayerNorm(cross_attn_output - f_i).
5. Temporal CLS path over 4 frame summaries → cls_out.
6. Head: cat([cls_out, 6 pair deltas]) → MLP → 33 classes.

Input  : (B, T=4, C=3, 224, 224)  ImageNet-normalised
Output : (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
# SigLIP-2 follows SigLIP-1 normalisation : mean=0.5, std=0.5 (centered [-1, 1])
_SG_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_SG_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_N_FRAMES = 4
_PAIRS    = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _imagenet_to_siglip(x: torch.Tensor) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    m_in = _IN_MEAN.to(device, dtype)
    s_in = _IN_STD.to(device, dtype)
    m_sg = _SG_MEAN.to(device, dtype)
    s_sg = _SG_STD.to(device, dtype)
    return (x * s_in + m_in - m_sg) / s_sg


def _extract_patch_tokens(out) -> torch.Tensor:
    """Defensive extraction of (B, N, D) patch tokens from a Siglip2 forward output."""
    feats = getattr(out, "last_hidden_state", None)
    if feats is None:
        feats = out[0] if isinstance(out, (tuple, list)) else out
    # SigLIP-2 may or may not include a CLS token depending on the variant.
    # If shape is (B, N+1, D) and the first token looks like a CLS (no spatial
    # structure), we keep all tokens — the spatial pool attends to whatever
    # we hand it. Cleaner to keep all tokens than to misalign.
    return feats


def _unwrap_vision(model: nn.Module) -> nn.Module:
    """Get the vision sub-module from a Siglip2 / SigLIP2VisionModel / Auto-loaded model."""
    for attr in ("vision_model", "vision_tower", "visual"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module):
            return sub
    return model


class _TemporalBlock(nn.Module):
    """Pre-LN self-attention block."""

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


class SigLIP2GiantPairs(nn.Module):
    """SigLIP-2 Giant (frozen) + cross-attention pairs head."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "google/siglip2-giant-opt-patch16-384",
        backbone_dtype: str = "bfloat16",
        n_temporal_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.25,
        head_hidden: int = 2048,
    ) -> None:
        super().__init__()

        _DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        bb_dtype = _DTYPES.get(backbone_dtype, torch.bfloat16)

        raw = AutoModel.from_pretrained(backbone, torch_dtype=bb_dtype, trust_remote_code=True)
        self.encoder = _unwrap_vision(raw)
        if self.encoder is not raw:
            del raw

        # Frozen — only the head trains.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # Probe hidden size (siglip2-giant ≈ 1536) and image_size (384)
        cfg = self.encoder.config
        D = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "embed_dim", None) or 1536)
        self.hidden_size = D
        self.image_size = int(getattr(cfg, "image_size", 384))
        self.patch_size = int(getattr(cfg, "patch_size", 16))

        # Spatial attention pool per frame (shared query)
        self.spatial_query = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        # Cross-attention pairs projections
        self.cross_q_proj = nn.Linear(D, D, bias=False)
        self.cross_k_proj = nn.Linear(D, D, bias=False)
        self.cross_v_proj = nn.Linear(D, D, bias=False)
        self.pair_norms = nn.ModuleList([nn.LayerNorm(D) for _ in _PAIRS])

        # Temporal CLS path
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.temporal_pe = nn.Embedding(_N_FRAMES + 1, D)
        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(D, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(D)

        # Head : [CLS, 6 pair deltas] = 7 × D → MLP → num_classes
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
        print(f"SigLIP2GiantPairs: backbone={backbone}  hidden={D}  image_size={self.image_size}  "
              f"{n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()    # never train the backbone
        return self

    def head_parameters(self):
        return [p for n, p in self.named_parameters()
                if not n.startswith("encoder.") and p.requires_grad]

    def backbone_parameters(self):
        return []  # fully frozen, no LoRA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet → SigLIP renorm
        x = _imagenet_to_siglip(x)

        # Resize to backbone's native (224 → 384 for giant)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(
                x.view(B * _N_FRAMES, 3, x.shape[-2], x.shape[-1]),
                size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).view(B, _N_FRAMES, 3, self.image_size, self.image_size)

        # Run frozen vision encoder per frame (concatenated batch).
        enc_dtype = next(self.encoder.parameters()).dtype
        x_flat = x.view(B * _N_FRAMES, 3, self.image_size, self.image_size).to(enc_dtype)
        with torch.no_grad():
            out = self.encoder(pixel_values=x_flat)
        tokens = _extract_patch_tokens(out).to(dtype)         # (B*T, N, D)
        N = tokens.shape[1]
        D = tokens.shape[-1]
        # (B, T, N, D)
        frame_tokens = tokens.view(B, _N_FRAMES, N, D)

        # Spatial attention pool per frame
        ft_flat = frame_tokens.reshape(B * _N_FRAMES, N, D)
        q = self.spatial_query.expand(B * _N_FRAMES, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q, ft_flat.transpose(1, 2)) / math.sqrt(D), dim=-1)
        frame_feats = torch.bmm(attn_w, ft_flat).squeeze(1).reshape(B, _N_FRAMES, D)

        # Cross-attention pairs (j attends to spatial tokens of frame i, baseline f_i)
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

        # Temporal CLS path
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

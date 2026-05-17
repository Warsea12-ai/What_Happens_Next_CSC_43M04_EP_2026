"""
MobileNetV3-Small + Temporal Shift Module for Track A (from scratch).

Architecturally distinct from ResNet-based TSM:
- Depthwise separable convolutions (DW-PW decomposition)
- Squeeze-Excitation channel attention in every InvertedResidual block
- Hard-swish activation
- ~2.5M params vs ~11M for ResNet-18 — stronger implicit regularization

TSM is applied before each InvertedResidual block; the block's own
use_res_connect handles skip connections naturally.

Usage (from src/):
    uv run python train.py experiment=track_A_mobilenet
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import InvertedResidual


class TemporalShift(nn.Module):
    def __init__(self, n_segment: int = 4, fold_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = max(1, c // self.fold_div)
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class MobileTSMBlock(nn.Module):
    """InvertedResidual block with temporal shift applied before the block."""
    def __init__(self, block: nn.Module, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.shift = TemporalShift(n_segment, fold_div)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.shift(x))


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32, mode: str = "sinusoidal"):
        super().__init__()
        self.mode = mode
        if mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).float().unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        elif mode == "learned":
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            raise ValueError(f"Unknown PE mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MobileNetTSM(nn.Module):
    """
    MobileNetV3-Small with Temporal Shift Module, trained from scratch.

    ~2.5M params. Depthwise-separable convs + SE blocks provide different
    inductive bias than ResNet — useful as an ensemble component with TSM_s.
    """

    def __init__(
        self,
        num_classes: int,
        n_segment: int = 4,
        fold_div: int = 8,
        dropout: float = 0.5,
        temporal_pool: str = "attention",
        use_positional_encoding: bool = True,
        pe_mode: str = "sinusoidal",
    ):
        super().__init__()
        self.n_segment = n_segment
        self.temporal_pool = temporal_pool
        self.use_pe = use_positional_encoding

        net = mobilenet_v3_small(weights=None)

        # Wrap every InvertedResidual block with a temporal shift
        features = list(net.features)
        for i, blk in enumerate(features):
            if isinstance(blk, InvertedResidual):
                features[i] = MobileTSMBlock(blk, n_segment, fold_div)
        net.features = nn.Sequential(*features)

        # feat_dim is the output of the last Conv in features (before avgpool)
        feat_dim: int = net.classifier[0].in_features  # 576 for MobileNetV3-Small

        self.backbone = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

        if use_positional_encoding:
            self.pos_enc = TemporalPositionalEncoding(feat_dim, mode=pe_mode)

        if temporal_pool == "attention":
            self.attn = nn.Linear(feat_dim, 1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def _update_n_segment(self, T: int) -> None:
        self.n_segment = T
        for blk in self.backbone:
            if isinstance(blk, MobileTSMBlock):
                blk.shift.n_segment = T

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clips.shape
        if T != self.n_segment:
            self._update_n_segment(T)

        x = clips.reshape(B * T, C, H, W)
        x = self.backbone(x)            # (B*T, 576, h, w)
        x = self.pool(x).flatten(1)     # (B*T, 576)
        feats = x.view(B, T, -1)        # (B, T, 576)

        if self.use_pe:
            feats = self.pos_enc(feats)

        if self.temporal_pool == "mean":
            feats = feats.mean(1)
        elif self.temporal_pool == "attention":
            w = torch.softmax(self.attn(feats), dim=1)  # (B, T, 1)
            feats = (feats * w).sum(1)                   # (B, 576)
        elif self.temporal_pool == "last":
            feats = feats[:, -1]
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")

        return self.head(feats)

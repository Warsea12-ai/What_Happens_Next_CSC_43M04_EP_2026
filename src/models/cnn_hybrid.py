"""
CNNHybrid — explicit temporal direction + learned temporal self-attention.

Two complementary temporal paths share a single ResNet34 backbone:

  Direction path (CNNTemporal-style):
    global_avg = mean over T frames          — scene content
    avg_diff   = mean of adjacent diffs      — frame-by-frame motion magnitude
    net_dir    = last_frame − first_frame    — overall temporal direction
    Each normalized with its own LayerNorm → 3×D features

  Attention path (VideoTransformer-style):
    T frame tokens + CLS + positional embed
    → 2-layer pre-norm Transformer
    → CLS output → d_model features

  Combined: [3×D, d_model] → Dropout → Linear → GELU → Dropout → Linear → num_classes

Why combine: direction signals give stable gradient flow from epoch 1 (reliable
handcrafted features); attention path learns complex multi-frame interactions that
the sum-based direction signals cannot capture (e.g. "frames 2–3 are discriminative,
frames 1 and 4 are not"). Neither alone extracts all temporal information.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNHybrid(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "resnet34",
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_dim: int = 512,
        max_frames: int = 16,
    ) -> None:
        super().__init__()

        backbone_specs = {
            "resnet18": (models.resnet18, 512),
            "resnet34": (models.resnet34, 512),
            "resnet50": (models.resnet50, 2048),
        }
        if backbone not in backbone_specs:
            raise ValueError(f"backbone must be one of {list(backbone_specs)}")

        fn, feat_dim = backbone_specs[backbone]
        net = fn(weights=None)
        net.fc = nn.Identity()  # type: ignore
        self.backbone = net
        self.feat_dim = feat_dim

        # ── Direction path ────────────────────────────────────────────────────
        self.ln_content   = nn.LayerNorm(feat_dim)
        self.ln_direction = nn.LayerNorm(feat_dim)

        # ── Attention path ────────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, max_frames + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
        )

        # ── Shared head ───────────────────────────────────────────────────────
        combined_dim = 3 * feat_dim + d_model
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, num_classes),
        )

        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape

        feats = self.backbone(video_batch.reshape(B * T, C, H, W))
        feats = feats.view(B, T, self.feat_dim)                 # (B, T, D)

        # Direction path
        global_avg = feats.mean(dim=1)
        avg_diff   = (feats[:, 1:] - feats[:, :-1]).mean(dim=1)
        net_dir    = feats[:, -1] - feats[:, 0]
        dir_feat = torch.cat([
            self.ln_content(global_avg),
            self.ln_direction(avg_diff),
            self.ln_direction(net_dir),
        ], dim=1)                                               # (B, 3D)

        # Attention path
        tokens = self.proj(feats)                               # (B, T, d_model)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)                # (B, T+1, d_model)
        tokens = tokens + self.pos_embed[:, : T + 1]
        out    = self.transformer(tokens)                       # (B, T+1, d_model)
        attn_feat = out[:, 0]                                   # (B, d_model) — CLS

        combined = torch.cat([dir_feat, attn_feat], dim=1)     # (B, 3D+d_model)
        return self.head(combined)

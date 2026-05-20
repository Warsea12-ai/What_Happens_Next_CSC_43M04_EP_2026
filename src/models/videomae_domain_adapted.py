"""VideoMAEDomainAdapted — VideoMAE-Large fine-tuned from a domain-adapted backbone.

Loads a VideoMAEModel encoder from a local checkpoint saved by pretrain_videomae.py,
then attaches the same temporal-attention head as VideoMAETemporalHead.

This decouples the pretraining from fine-tuning:
  1. pretrain_videomae.py   → outputs/pretrained_videomae/backbone_best.pt
  2. train_trackB.py experiment=track_B_videomae_domain_adapted → loads that backbone

If backbone_path is empty or the file does not exist, falls back to loading the
HuggingFace checkpoint directly (same behaviour as VideoMAETemporalHead).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig

from models.videomae_temporal_head import (
    _linear_upsample, _TemporalBlock,
    _IN_MEAN, _IN_STD, _VM_MEAN, _VM_STD,
    _TARGET_FRAMES, _N_TEMPORAL, _N_SPATIAL, _ORIG_GROUPS,
)


class VideoMAEDomainAdapted(nn.Module):
    """VideoMAETemporalHead that can load a locally-pretrained backbone checkpoint."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        backbone_path: str = "",
        num_frozen_blocks: int = 12,
        n_temporal_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.3,
        head_hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks

        # Load backbone: local checkpoint takes priority
        local = Path(backbone_path) if backbone_path else None
        if local and local.exists():
            print(f"Loading domain-adapted backbone from {local}…")
            cfg = VideoMAEConfig.from_pretrained(backbone)
            self.encoder = VideoMAEModel(cfg)
            state = torch.load(local, map_location="cpu", weights_only=True)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)} (likely fine)")
        else:
            if backbone_path:
                print(f"Warning: backbone_path={backbone_path!r} not found, "
                      f"falling back to HuggingFace.")
            print(f"Loading {backbone} from HuggingFace…")
            self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze embeddings always
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        # Partially freeze transformer blocks
        for i, block in enumerate(self.encoder.encoder.layer):
            trainable = i >= num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = trainable

        hidden = self.encoder.config.hidden_size

        self.spatial_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.temporal_pe = nn.Embedding(_N_TEMPORAL + 1, hidden)

        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(hidden, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden)

        self.norm_delta_full  = nn.LayerNorm(hidden)
        self.norm_delta_first = nn.LayerNorm(hidden)
        self.norm_delta_last  = nn.LayerNorm(hidden)

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
        print(f"VideoMAEDomainAdapted: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self.num_frozen_blocks < len(self.encoder.encoder.layer):
            for block in self.encoder.encoder.layer[self.num_frozen_blocks:]:
                block.train(mode)
        return self

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
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
        x = _linear_upsample(x, _TARGET_FRAMES)

        tokens = self.encoder(pixel_values=x).last_hidden_state
        tokens = tokens.to(dtype)
        D = tokens.shape[-1]
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, D)

        tokens_flat = tokens.reshape(B * _N_TEMPORAL, _N_SPATIAL, D)
        q = self.spatial_query.expand(B * _N_TEMPORAL, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q, tokens_flat.transpose(1, 2)) / math.sqrt(D), dim=-1,
        )
        tfeats = torch.bmm(attn_w, tokens_flat).squeeze(1).reshape(B, _N_TEMPORAL, D)

        g0, g1, g2, g3 = _ORIG_GROUPS
        f0, f1, f2, f3 = tfeats[:, g0], tfeats[:, g1], tfeats[:, g2], tfeats[:, g3]
        delta_full  = f3 - f0
        delta_first = f1 - f0
        delta_last  = f3 - f2

        pos_ids = torch.arange(1, _N_TEMPORAL + 1, device=device)
        pe      = self.temporal_pe(pos_ids).unsqueeze(0)
        tfeats  = tfeats + pe

        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls    = self.cls_token.expand(B, -1, -1) + cls_pe
        seq    = torch.cat([cls, tfeats], dim=1)

        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])

        feat = torch.cat([
            cls_out,
            self.norm_delta_full(delta_full),
            self.norm_delta_first(delta_first),
            self.norm_delta_last(delta_last),
        ], dim=1)
        return self.head(feat)

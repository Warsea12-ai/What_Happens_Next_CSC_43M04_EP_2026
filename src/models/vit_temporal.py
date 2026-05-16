"""
ViTTemporal — ViT-B/16 (ImageNet pretrained) + temporal transformer head.

Why this beats VideoSwinFinetune (0.35) on this task:
  - Swin3D collapses T=4 → T=1 within the first two 3D stages (temporal stride=2
    at each stage), so it effectively sees one merged frame and ignores order.
  - ViT-B/16 processes each frame independently → four 768-dim CLS tokens that
    preserve full temporal structure → temporal transformer can reason over them.
  - ImageNet pretraining >> Kinetics for this task: the challenge is about hand-
    object interactions and motion direction, not sport recognition.

Architecture:
    ViT-B/16 (first 8 blocks frozen, last 4 fine-tuned)
       ↓  per-frame CLS tokens  (B, T, 768)
    Learnable temporal positional encoding
       ↓
    2-layer pre-norm transformer with CLS token  (B, T+1, 768)
       ↓  CLS token
    LayerNorm → Dropout → Linear(768, num_classes)

Training (differential LR via backbone_parameters / head_parameters):
    frozen blocks         lr = 0       (requires_grad=False)
    last 4 ViT blocks     lr = 1e-5    (gentle fine-tune)
    temporal head         lr = 1e-4    (train from scratch)

Input : (B, T, C, H, W)   H=W=224
Output: (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


_VIT_DIM = 768          # hidden dim of ViT-B
_NUM_BLOCKS = 12        # total transformer blocks in ViT-B/16


class ViTTemporal(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        num_frozen_blocks: int = 8,   # freeze first N blocks (0 = fully fine-tune)
        dropout: float = 0.3,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        max_frames: int = 32,
    ) -> None:
        super().__init__()

        # ── ViT-B/16 backbone ────────────────────────────────────────────────
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()   # output CLS token (768-dim) instead of logits
        self.vit = vit

        # Freeze patch embedding + positional embedding + first N blocks
        for param in self.vit.conv_proj.parameters():
            param.requires_grad = False
        self.vit.class_token.requires_grad_(False)
        # Note: encoder.pos_embedding is a Parameter, not a module
        self.vit.encoder.pos_embedding.requires_grad_(False)
        for i, block in enumerate(self.vit.encoder.layers):
            requires_grad = (i >= num_frozen_blocks)
            for param in block.parameters():
                param.requires_grad = requires_grad
        # Always train the final LayerNorm
        for param in self.vit.encoder.ln.parameters():
            param.requires_grad = True

        n_frozen = sum(1 for p in self.vit.parameters() if not p.requires_grad)
        n_tuned  = sum(1 for p in self.vit.parameters() if p.requires_grad)
        print(f"ViTTemporal: {n_frozen} frozen param tensors, {n_tuned} fine-tuned param tensors")

        # ── Temporal reasoning ───────────────────────────────────────────────
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_frames, _VIT_DIM))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, _VIT_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=_VIT_DIM,
            nhead=temporal_heads,
            dim_feedforward=_VIT_DIM * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm: more stable for small datasets
        )
        self.temporal = nn.TransformerEncoder(
            enc_layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(_VIT_DIM),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(_VIT_DIM),
            nn.Dropout(dropout),
            nn.Linear(_VIT_DIM, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    # ── parameter groups for differential LR ─────────────────────────────────

    def backbone_parameters(self):
        """Trainable ViT backbone parameters (last few blocks + LayerNorm)."""
        return [p for p in self.vit.parameters() if p.requires_grad]

    def head_parameters(self):
        """Temporal head parameters (temporal PE, CLS, transformer, classifier)."""
        return (
            [self.temporal_pe, self.cls_token]
            + list(self.temporal.parameters())
            + list(self.head.parameters())
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video_batch.shape
        x = video_batch.reshape(B * T, C, H, W)

        # Run the ViT encoder directly to get patch tokens.
        # The CLS token (optimised for ImageNet object recognition) produces nearly
        # identical vectors for "moving left" vs "moving right" because scene
        # composition is the same in both.  Patch tokens are position-specific:
        # the patch on the left side of frame 4 carries different content for
        # "moving right" vs "moving left", so their spatial average gives the
        # temporal transformer a direct signal about where the object ended up.
        x = self.vit._process_input(x)                              # (B*T, N, 768)
        n = x.shape[0]
        batch_cls = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_cls, x], dim=1)                        # (B*T, N+1, 768)
        x = self.vit.encoder(x)                                     # (B*T, N+1, 768)

        # Average pool patch tokens (index 0 is CLS, skip it)
        frame_feats = x[:, 1:].mean(dim=1).view(B, T, _VIT_DIM)    # (B, T, 768)

        # Add temporal positional encoding
        frame_feats = frame_feats + self.temporal_pe[:, :T]

        # Prepend CLS token and apply temporal transformer
        cls = self.cls_token.expand(B, -1, -1)                      # (B, 1, 768)
        tokens = torch.cat([cls, frame_feats], dim=1)               # (B, T+1, 768)
        out = self.temporal(tokens)                                  # (B, T+1, 768)

        return self.head(out[:, 0])                                  # (B, num_classes)

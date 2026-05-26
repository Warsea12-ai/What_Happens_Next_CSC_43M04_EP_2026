"""MViT wrapper using pytorchvideo (transformers.MvitModel removed in 5.x).

Original plan was MViT-V2 SSv2-finetuned via `transformers.MvitModel`, but
that class no longer exists in transformers 5.8+. Fallback : MViT-V1 base
from pytorchvideo, K400-finetuned (78.9% top-1 on K400). Architecturally
similar (hierarchical multi-scale) — sweep hyperparams still meaningful.

We keep the wrapper class name `MViTV2SSv2` and config name `mvit_v2_ssv2`
to avoid breaking the 21 existing configs. Docstring + log message make
the actual K400/V1 reality clear.

Input  : (B, T=4..16, C, H=224, W=224) — interpolated to 16 internally.
Output : (B, 33) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pytorchvideo.models.hub import mvit_base_16x4

from temporal_interp import interp_temporal

_TARGET_FRAMES = 16


class MViTV2SSv2(nn.Module):
    """MViT-V1-Base K400 (architectural cousin of MViT-V2-SSv2).

    The `backbone` argument is IGNORED — pytorchvideo loads its own weights.
    Kept in signature for config compatibility.

    Note : the original SSv2-finetuned MViT-V2 weights are not loadable here
    (different architecture from V1, and `transformers.MvitModel` removed).
    For SSv2-finetuned alternatives we have Hiera-Base/Huge, V-JEPA2 ViT-g,
    VideoMAE V1/V2 (already covered by other wrappers).
    """

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "facebook/mvit-base-finetuned-ssv2",  # ignored
        num_frozen_blocks: int = 0,
        dropout: float = 0.25,
        head_hidden: int = 1024,
        interp_mode: str = "aligned",
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks
        if interp_mode not in ("aligned", "centered", "repeat"):
            raise ValueError(f"interp_mode must be aligned|centered|repeat, got {interp_mode!r}")
        self.interp_mode = interp_mode

        # Load pytorchvideo MViT-Base-16x4 (K400-pretrained, MViT-V1).
        # The model is a pytorchvideo `Net` ; its `head` is a pooled MLP outputting 400 classes.
        # We KEEP the K400 head and treat its 400 logits as features for our 33-class head.
        # Surgical replacement of the inner Linear didn't take (the head module has internal
        # reshapes that resist `.proj = Identity()`). Using the 400 logits as features is
        # functionally equivalent — the 768-dim features are linearly projected to 400, and
        # we re-project 400 -> head_hidden -> 33. A bit less expressive than 768 -> 33 but
        # still gives the K400-pretrained MViT representation, and works reliably.
        self.net = mvit_base_16x4(pretrained=True)
        feat_dim = 400  # K400 logits

        # Freeze patch_embed (block 0) + first num_frozen_blocks transformer blocks
        # (excluding the head, which we keep trainable since it's already projecting
        # to features we'll re-use).
        n_blocks = len(self.net.blocks)
        for p in self.net.blocks[0].parameters():
            p.requires_grad = False
        for i in range(1, min(1 + num_frozen_blocks, n_blocks - 1)):
            for p in self.net.blocks[i].parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MViTV2SSv2 (pytorchvideo MViT-V1-Base K400) : "
              f"{n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}, feat_dim={feat_dim}, interp={interp_mode})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample to 16 frames (MViT-Base-16x4 expects 16).
        x = interp_temporal(x, target_T=_TARGET_FRAMES, mode=self.interp_mode)
        # pytorchvideo MViT expects (B, C, T, H, W) — permute from (B, T, C, H, W).
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feats = self.net(x)  # (B, feat_dim) after pooled head
        return self.head(feats)

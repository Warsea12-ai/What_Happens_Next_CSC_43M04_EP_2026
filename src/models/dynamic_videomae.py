"""DynamicVideoMAE — VideoMAE-Large (SSv2) with Temporal Pyramid head.

Key advances over MotionVideoMAE (Base + endpoints only):

1. VideoMAE-Large backbone: 307M params, 24 blocks, 1024-dim hidden.
   vs Base: 86M, 12 blocks, 768-dim.

2. SSv2-domain pretraining: same distribution as our 33 challenge classes.

3. Temporal Pyramid head — 8 complementary features, not 4:

   After spatial pooling → (B, 8_temporal, 1024) per-group features.
   The 4 original frame positions map to specific temporal groups:
     f0 → group 0  (interpolated frames 0-1  ≈ original frame 0)
     f1 → group 2  (interpolated frames 4-5  ≈ original frame 1)
     f2 → group 5  (interpolated frames 10-11 ≈ original frame 2)
     f3 → group 7  (interpolated frames 14-15 ≈ original frame 3)

   Features extracted:
     global      = mean(all groups)   — overall scene/object context
     f0, f1, f2, f3                   — per-frame state
     delta_full  = f3 - f0            — net motion (start → end)
     delta_first = f1 - f0            — first-half motion
     delta_last  = f3 - f2            — second-half motion

   delta_first vs delta_last distinguishes trajectory shapes:
     Uniform:       delta_first ≈ delta_last
     Acceleration:  |delta_last| > |delta_first|
     Deceleration:  |delta_first| > |delta_last|
     U-shaped:      delta_first ≈ -delta_last, delta_full ≈ 0
                    (e.g. "pick up and put back" — delta_full would miss this)

   → cat([global, f0, f1, f2, f3, delta_full, delta_first, delta_last])
   → shape (B, 8×1024 = 8192)
   → Dropout(0.3) → Linear(8192, 2048) → GELU → Dropout(0.15) → Linear(2048, 33)

4. Fine-tune last 12 of 24 blocks with LLRD.
   Blocks 0-11: frozen (preserve SSv2 low-level features)
   Blocks 12-23: trainable (adapt to 33-class task)

Input : (B, T=4, C, H, W) ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEModel

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_VM_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16
_TUBELET_SIZE  = 2
_PATCH_SIZE    = 16
_IMG_SIZE      = 224
_N_TEMPORAL    = _TARGET_FRAMES // _TUBELET_SIZE   # 8
_N_SPATIAL     = (_IMG_SIZE // _PATCH_SIZE) ** 2   # 196

# Temporal groups that correspond to the 4 original input frames after
# linear interpolation of 4→16 frames (linspace(0, 3, 16)):
#   frame index  5 → position 1.0 = original frame 1 → tubelet group 2
#   frame index 10 → position 2.0 = original frame 2 → tubelet group 5
#   frame index 15 → position 3.0 = original frame 3 → tubelet group 7
_ORIG_GROUPS = (0, 2, 5, 7)   # groups for original frames [0, 1, 2, 3]


def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


class DynamicVideoMAE(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-large-finetuned-ssv2",
        num_frozen_blocks: int = 12,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze patch embedding (Conv3d tubelet projection)
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False

        # Freeze first num_frozen_blocks transformer blocks
        for i, block in enumerate(self.encoder.encoder.layer):
            trainable = i >= num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = trainable

        hidden = self.encoder.config.hidden_size  # 1024 for Large

        # Per-feature LayerNorms — scale each stream independently before concat
        _mk_norm = lambda: nn.LayerNorm(hidden)
        self.norm_global      = _mk_norm()
        self.norm_f0          = _mk_norm()
        self.norm_f1          = _mk_norm()
        self.norm_f2          = _mk_norm()
        self.norm_f3          = _mk_norm()
        self.norm_delta_full  = _mk_norm()
        self.norm_delta_first = _mk_norm()
        self.norm_delta_last  = _mk_norm()

        # Temporal Pyramid head: 8 × hidden → 2048 → num_classes
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(8 * hidden, 2048),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(2048, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DynamicVideoMAE: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.0f}M trainable "
              f"(backbone frozen {num_frozen_blocks}/{len(self.encoder.encoder.layer)} blocks)")

    # ── parameter groups ──────────────────────────────────────────────────────

    def _head_params(self):
        norms = [self.norm_global, self.norm_f0, self.norm_f1, self.norm_f2, self.norm_f3,
                 self.norm_delta_full, self.norm_delta_first, self.norm_delta_last]
        return sum((list(n.parameters()) for n in norms), []) + list(self.head.parameters())

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
        return self._head_params()

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.85):
        """LLRD: each block gets backbone_lr × llrd^depth (depth=0 at last block)."""
        groups = [{"params": self._head_params(), "lr": head_lr}]
        n_layers = len(self.encoder.encoder.layer)
        for depth, idx in enumerate(range(n_layers - 1, self.num_frozen_blocks - 1, -1)):
            groups.append({
                "params": list(self.encoder.encoder.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE [0.5, 0.5, 0.5] normalisation
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # 4 → 16 frames via linear interpolation
        x = _linear_upsample(x, _TARGET_FRAMES)

        # Encode: (B, N_TEMPORAL × N_SPATIAL, hidden) — no CLS token
        tokens = self.encoder(pixel_values=x).last_hidden_state
        # Reshape to (B, 8, 196, hidden)
        tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, -1)
        # Spatial pool → (B, 8, hidden)
        tfeats = tokens.mean(dim=2)

        # Extract features at the 4 original-frame temporal positions
        g0, g1, g2, g3 = _ORIG_GROUPS
        f0 = tfeats[:, g0]                # (B, hidden) — original frame 0
        f1 = tfeats[:, g1]                # (B, hidden) — original frame 1
        f2 = tfeats[:, g2]                # (B, hidden) — original frame 2
        f3 = tfeats[:, g3]                # (B, hidden) — original frame 3
        global_feat  = tfeats.mean(dim=1) # (B, hidden) — all temporal groups

        delta_full  = f3 - f0             # net motion (start → end)
        delta_first = f1 - f0             # first-half motion
        delta_last  = f3 - f2             # second-half motion

        feat = torch.cat([
            self.norm_global(global_feat),
            self.norm_f0(f0),
            self.norm_f1(f1),
            self.norm_f2(f2),
            self.norm_f3(f3),
            self.norm_delta_full(delta_full),
            self.norm_delta_first(delta_first),
            self.norm_delta_last(delta_last),
        ], dim=1)  # (B, 8 × hidden)

        return self.head(feat)

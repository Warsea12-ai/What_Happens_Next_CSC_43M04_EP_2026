"""VideoMAEClassifier — VideoMAE-Base fine-tuned on Something-Something v2.

Backbone: MCG-NJU/videomae-base-finetuned-ssv2 (pretrained on SSv2, same domain).
The 174-class head is replaced with a 33-class head (ignore_mismatched_sizes).

Frame handling: VideoMAE expects 16 frames; we expand T=4 by repeating each frame
`frames_repeat` times (default 4 → 16), so temporal attention still sees 4 distinct
content states spread across 16 temporal tokens.

Normalization: the training pipeline provides ImageNet-normalized frames; we
renormalize to VideoMAE's [0.5, 0.5, 0.5] stats inside the model's forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_VM_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16


def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linearly interpolate (B, T, C, H, W) along the time axis to target_T."""
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


class VideoMAEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frozen_blocks: int = 8,
        dropout: float = 0.3,
        frames_repeat: int = 4,
        use_linear_interp: bool = True,
    ) -> None:
        super().__init__()
        self.frames_repeat = frames_repeat
        self.use_linear_interp = use_linear_interp

        self.model = VideoMAEForVideoClassification.from_pretrained(
            backbone,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )

        # Freeze patch/temporal embeddings — pretrained positional encoding should not shift
        for p in self.model.videomae.embeddings.parameters():
            p.requires_grad = False

        # Freeze first num_frozen_blocks encoder blocks
        for i, block in enumerate(self.model.videomae.encoder.layer):
            freeze = i < num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = not freeze

        # Always keep the LayerNorm (fc_norm) trainable
        for p in self.model.fc_norm.parameters():
            p.requires_grad = True

        # Replace head: fc_norm output is already layer-normed, just add dropout
        hidden = self.model.config.hidden_size
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        nn.init.trunc_normal_(self.model.classifier[-1].weight, std=0.01)
        nn.init.zeros_(self.model.classifier[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VideoMAEClassifier: {n_frozen/1e6:.1f}M frozen, {n_train/1e6:.1f}M trainable")

    def unfreeze_to(self, n_frozen: int) -> None:
        """Reduce frozen encoder blocks to n_frozen; call before rebuilding optimizer."""
        for i, block in enumerate(self.model.videomae.encoder.layer):
            for p in block.parameters():
                p.requires_grad = (i >= n_frozen)
        n_f = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  VideoMAEClassifier thawed → {n_frozen} frozen blocks "
              f"({n_f/1e6:.1f}M frozen, {n_t/1e6:.1f}M trainable)")

    def backbone_parameters(self):
        return [p for n, p in self.named_parameters()
                if "classifier" not in n and p.requires_grad]

    def head_parameters(self):
        return list(self.model.classifier.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, C, H, W) — ImageNet normalized
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE normalization in one fused step
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # Expand T=4 → 16 frames
        if self.use_linear_interp:
            x = _linear_upsample(x, _TARGET_FRAMES)
        else:
            x = x.unsqueeze(2).expand(B, T, self.frames_repeat, C, H, W)
            x = x.reshape(B, T * self.frames_repeat, C, H, W)

        return self.model(pixel_values=x).logits

"""DualEncoder — VideoMAE-Large (Kinetics) + Qwen2-VL-7B (vision-language), both frozen.

The two encoders have orthogonal pretraining objectives:
  VideoMAE: masked autoencoding on Kinetics — learns motion and spatio-temporal structure
  Qwen2-VL: vision-language alignment — learns rich semantics and object understanding

Concatenating their motion features gives a 19 456-dim vector that contains both
low-level temporal dynamics and high-level semantic context.

VideoMAE motion features (5 × 1024 = 5 120):
  global = mean(all 1568 tokens)   — full scene
  first  = mean(temporal pos 0)    — where action started
  mid    = mean(temporal pos 3-4)  — midpoint state
  last   = mean(temporal pos 7)    — where it ended
  delta  = last − first            — net motion

Qwen2-VL-7B motion features (4 × 3584 = 14 336):
  global = mean(all 128 tokens)    — semantic scene
  first  = temporal group 0        — frames 0-1 state
  last   = temporal group 1        — frames 2-3 state
  delta  = last − first            — semantic shift

Joint head: cat(5120, 14336) = 19456 → Linear(19456, 1024) → GELU → Linear(1024, 33)

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import gc
import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

from models.qwen_vl_video import _imagenet_to_clip, _frames_to_patches

# ── VideoMAE constants ────────────────────────────────────────────────────────
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_VM_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16
_TUBELET       = 2
_PATCH_VM      = 16
_IMG_SIZE      = 224
_N_TEMPORAL    = _TARGET_FRAMES // _TUBELET   # 8
_N_SPATIAL     = (_IMG_SIZE // _PATCH_VM) ** 2  # 196


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


class DualEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        videomae_backbone: str = "MCG-NJU/videomae-large-finetuned-kinetics",
        qwen_backbone: str = "Qwen/Qwen2-VL-7B-Instruct",
        dropout: float = 0.4,
        head_hidden: int = 1024,
    ) -> None:
        super().__init__()

        # ── VideoMAE-Large encoder ────────────────────────────────────────────
        print(f"Loading VideoMAE encoder: {videomae_backbone}…")
        _vm_full = VideoMAEForVideoClassification.from_pretrained(
            videomae_backbone, use_safetensors=True,
        )
        self.vm_encoder = _vm_full.videomae
        del _vm_full
        gc.collect()
        for p in self.vm_encoder.parameters():
            p.requires_grad = False
        vm_hidden = self.vm_encoder.config.hidden_size  # 1024

        # ── Qwen2-VL-7B visual encoder ────────────────────────────────────────
        print(f"Loading Qwen visual encoder: {qwen_backbone}…")
        if "2.5" in qwen_backbone:
            from transformers import Qwen2_5_VLForConditionalGeneration
            _q_full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_backbone, dtype=torch.bfloat16,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            _q_full = Qwen2VLForConditionalGeneration.from_pretrained(
                qwen_backbone, dtype=torch.bfloat16,
            )
        self.qwen_visual = _q_full.model.visual
        del _q_full
        gc.collect()
        print("LLMs deleted — keeping both visual encoders.")
        for p in self.qwen_visual.parameters():
            p.requires_grad = False

        qvcfg = self.qwen_visual.config
        q_hidden = qvcfg.hidden_size  # 3584 for Qwen2-VL-7B
        self._q_ps  = qvcfg.patch_size
        self._q_tps = qvcfg.temporal_patch_size
        self._q_sms = getattr(qvcfg, "spatial_merge_size", 2)

        # ── LayerNorms (one per feature, per encoder) ─────────────────────────
        for suffix in ("global", "first", "mid", "last", "delta"):
            setattr(self, f"vm_norm_{suffix}", nn.LayerNorm(vm_hidden))
        for suffix in ("global", "first", "last", "delta"):
            setattr(self, f"q_norm_{suffix}", nn.LayerNorm(q_hidden))

        vm_feat_dim = 5 * vm_hidden   # 5120
        q_feat_dim  = 4 * q_hidden    # 14336
        total_dim   = vm_feat_dim + q_feat_dim  # 19456

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = (sum(p.numel() for p in self.vm_encoder.parameters())
                    + sum(p.numel() for p in self.qwen_visual.parameters()))
        n_train  = sum(p.numel() for p in self.head_parameters())
        print(f"DualEncoder: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.2f}M trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        self.vm_encoder.eval()
        self.qwen_visual.eval()
        return self

    def head_parameters(self):
        params = list(self.head.parameters())
        for suffix in ("global", "first", "mid", "last", "delta"):
            params += list(getattr(self, f"vm_norm_{suffix}").parameters())
        for suffix in ("global", "first", "last", "delta"):
            params += list(getattr(self, f"q_norm_{suffix}").parameters())
        return params

    def _videomae_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, C, H, W) → (B, 5120) motion features."""
        B, dtype = x.shape[0], x.dtype
        device = x.device
        m_in = _IN_MEAN.to(device, dtype); s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype); s_vm = _VM_STD.to(device, dtype)
        x_vm = (x * s_in + m_in - m_vm) / s_vm
        x_vm = _linear_upsample(x_vm, _TARGET_FRAMES)
        with torch.no_grad():
            tokens = self.vm_encoder(pixel_values=x_vm).last_hidden_state  # (B, 1568, 1024)
        tokens = tokens.to(dtype).reshape(B, _N_TEMPORAL, _N_SPATIAL, -1)
        global_f = tokens.mean(dim=(1, 2))
        first    = tokens[:, 0].mean(1)
        mid      = tokens[:, _N_TEMPORAL // 2].mean(1)
        last     = tokens[:, -1].mean(1)
        delta    = last - first
        return torch.cat([
            self.vm_norm_global(global_f), self.vm_norm_first(first),
            self.vm_norm_mid(mid), self.vm_norm_last(last),
            self.vm_norm_delta(delta),
        ], dim=1)  # (B, 5120)

    def _qwen_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, C, H, W) → (B, 14336) motion features."""
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        ps, tps, sms = self._q_ps, self._q_tps, self._q_sms
        hp = H // ps; wp = W // ps; tg = T // tps
        hpo = hp // sms; wpo = wp // sms

        x_q = _imagenet_to_clip(x)
        pixel_values = _frames_to_patches(x_q, ps, tps).to(torch.bfloat16)
        grid_thw = torch.tensor([[tg, hp, wp]], dtype=torch.long, device=device).expand(B, 3)
        with torch.no_grad():
            out = self.qwen_visual(pixel_values, grid_thw=grid_thw)
        raw = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        tokens = raw.to(dtype).reshape(B, tg, hpo * wpo, -1)  # (B, 2, 64, hidden)

        global_f = tokens.mean(dim=(1, 2))
        first    = tokens[:, 0].mean(1)
        last     = tokens[:, -1].mean(1)
        delta    = last - first
        return torch.cat([
            self.q_norm_global(global_f), self.q_norm_first(first),
            self.q_norm_last(last), self.q_norm_delta(delta),
        ], dim=1)  # (B, 14336)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vm_feat = self._videomae_features(x)   # (B, 5120)
        q_feat  = self._qwen_features(x)       # (B, 14336)
        feat = torch.cat([vm_feat, q_feat], dim=1)  # (B, 19456)
        return self.head(feat)

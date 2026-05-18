"""QwenVLVideo — Qwen2-VL visual encoder (frozen) + motion-aware head.

Works with any Qwen2-VL checkpoint (2B or 7B). The full model is loaded to
CPU, the visual encoder is extracted, and the LLM is deleted before moving
the encoder to GPU. The head dimension auto-adapts to the encoder's hidden_size.

  2B (Qwen/Qwen2-VL-2B-Instruct):  hidden=1536, head_hidden=512  default
  7B (Qwen/Qwen2-VL-7B-Instruct):  hidden=3584, head_hidden=1024 recommended

Patch format (Qwen2-VL's Conv3d tubelet embedding):
  4 frames at 224×224, patch_size=14, temporal_patch_size=2, spatial_merge=2
  → 2 temporal groups × 16×16 spatial patches = 512 patches per video (input)
  → 2 temporal groups × 8×8 merged tokens = 128 tokens per video (output)

Head (motion-aware):
  global = mean(all 128 tokens)   — what action / what scene
  first  = mean(temporal-group-0) — frames 0-1 state
  last   = mean(temporal-group-1) — frames 2-3 state
  delta  = last − first           — net motion direction
  → cat([global, first, last, delta]) → Dropout → Linear(4×hidden, head_hidden)
  → GELU → Dropout → Linear(head_hidden, num_classes)

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import gc
import torch
import torch.nn as nn

# ImageNet normalization (what the training pipeline produces)
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
# CLIP / Qwen2-VL normalization
_CL_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 1, 3, 1, 1)
_CL_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


def _imagenet_to_clip(x: torch.Tensor) -> torch.Tensor:
    """Fused ImageNet → CLIP renormalisation, in-place friendly."""
    device, dtype = x.device, x.dtype
    m_in = _IN_MEAN.to(device, dtype)
    s_in = _IN_STD.to(device, dtype)
    m_cl = _CL_MEAN.to(device, dtype)
    s_cl = _CL_STD.to(device, dtype)
    return (x * s_in + m_in - m_cl) / s_cl


def _frames_to_patches(
    x: torch.Tensor,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
) -> torch.Tensor:
    """Convert (B, T, C, H, W) to Qwen2-VL pixel_values format.

    Qwen2-VL's PatchEmbed does view(-1, C, tps, ps, ps) then Conv3d.
    So each row must store a single (C, tps, ps, ps) patch in C-major order.

    Returns shape: (B * T//tps * H//ps * W//ps,  C * tps * ps * ps)
    """
    B, T, C, H, W = x.shape
    ps, tps = patch_size, temporal_patch_size
    hp, wp = H // ps, W // ps
    tg = T // tps

    # (B, T, C, H, W) → (B, tg, tps, C, hp, ps_h, wp, ps_w)
    x = x.view(B, tg, tps, C, hp, ps, wp, ps)
    # → (B, tg, hp, wp, C, tps, ps_h, ps_w)  — patch identity × patch content
    x = x.permute(0, 1, 4, 6, 3, 2, 5, 7).contiguous()
    # → (B * tg * hp * wp,  C * tps * ps * ps)
    return x.reshape(B * tg * hp * wp, C * tps * ps * ps)


class QwenVLVideo(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "Qwen/Qwen2-VL-2B-Instruct",
        dropout: float = 0.5,
        head_hidden: int = 512,
    ) -> None:
        super().__init__()
        from transformers import Qwen2VLForConditionalGeneration

        print(f"Loading {backbone} to CPU…")
        _full = Qwen2VLForConditionalGeneration.from_pretrained(
            backbone,
            dtype=torch.bfloat16,
        )
        self.visual = _full.model.visual
        del _full
        gc.collect()
        print("LLM deleted — keeping visual encoder only.")

        for p in self.visual.parameters():
            p.requires_grad = False

        vcfg = self.visual.config
        self.patch_size          = vcfg.patch_size           # 14
        self.temporal_patch_size = vcfg.temporal_patch_size  # 2
        self.spatial_merge_size  = vcfg.spatial_merge_size   # 2
        hidden = vcfg.hidden_size                            # 1536 (2B) or 3584 (7B)

        self.norm_global = nn.LayerNorm(hidden)
        self.norm_first  = nn.LayerNorm(hidden)
        self.norm_last   = nn.LayerNorm(hidden)
        self.norm_delta  = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.visual.parameters())
        n_train  = sum(p.numel() for p in self.head_parameters())
        print(f"QwenVLVideo: {n_frozen/1e6:.0f}M frozen, "
              f"{n_train/1e6:.2f}M trainable  "
              f"(hidden={hidden}, head_hidden={head_hidden})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()   # keep frozen encoder in eval at all times
        return self

    # ── parameter groups ──────────────────────────────────────────────────────

    def head_parameters(self):
        return (
            list(self.norm_global.parameters())
            + list(self.norm_first.parameters())
            + list(self.norm_last.parameters())
            + list(self.norm_delta.parameters())
            + list(self.head.parameters())
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        ps   = self.patch_size           # 14
        tps  = self.temporal_patch_size  # 2
        sms  = self.spatial_merge_size   # 2
        hp   = H // ps                   # 16
        wp   = W // ps                   # 16
        tg   = T // tps                  # 2  (two temporal groups from 4 frames)
        hpo  = hp // sms                 # 8  (after spatial merge)
        wpo  = wp // sms                 # 8

        # ImageNet → CLIP normalization
        x = _imagenet_to_clip(x)

        # (B, T, C, H, W) → Qwen2-VL patch format: (B*tg*hp*wp, C*tps*ps*ps)
        pixel_values = _frames_to_patches(x, ps, tps).to(torch.bfloat16)

        # grid_thw: (B, 3) — [temporal_groups, height_patches, width_patches] per video
        grid_thw = torch.tensor([[tg, hp, wp]], dtype=torch.long, device=device).expand(B, 3)

        # Frozen visual encoder — no gradient tape
        with torch.no_grad():
            out = self.visual(pixel_values, grid_thw=grid_thw)

        # visual encoder returns post-merger tokens directly: (B * tg * hpo * wpo, hidden)
        raw = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        tokens = raw.to(dtype)                          # (B*128, 1536)
        tokens = tokens.reshape(B, tg, hpo * wpo, -1)  # (B, 2, 64, 1536)

        global_feat = tokens.mean(dim=(1, 2))           # (B, 1536) — all 128 tokens
        first       = tokens[:, 0].mean(dim=1)          # (B, 1536) — frames 0–1
        last        = tokens[:, -1].mean(dim=1)         # (B, 1536) — frames 2–3
        delta       = last - first                      # (B, 1536) — motion direction

        feat = torch.cat([
            self.norm_global(global_feat),
            self.norm_first(first),
            self.norm_last(last),
            self.norm_delta(delta),
        ], dim=1)  # (B, 6144)

        return self.head(feat)

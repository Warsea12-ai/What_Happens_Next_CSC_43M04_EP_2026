"""Qwen25VL3BVideo — Qwen2.5-VL-3B-Instruct as a video classifier for SSv2.

Motivation
----------
Newest Qwen vision-language model (Alibaba, Jan 2025). Native video understanding
with dynamic FPS sampling, 3D rotary positional encoding, and a vision tower
designed for multi-image / video reasoning. Bigger than V-JEPA 2 in total
parameters (~4B vs 1B) — most of the capacity is in the Qwen2.5-2B LLM, which
brings strong semantic reasoning the V-JEPA encoder lacks.

Memory budget (bf16, no quantization)
-------------------------------------
  Qwen2.5-VL-3B weights : ~8.0 GB    (4B params × 2 bytes)
  Vision activations    : ~1.0 GB    (16 frames @ 224², 128 merged tokens)
  LLM activations       : ~1.0-1.5 GB (grad checkpointing, batch=1)
  LoRA + head + TASK_CLS:  ~0.05 GB
                          ----------
                          ~10-11 GB → fits comfortably on RTX A4000 16GB.

Token economy
-------------
Qwen2.5-VL patchifies (4 frames @ 224×224) into:
  - temporal_patch_size = 2 → 2 temporal patches
  - spatial_patch_size  = 14 → 16×16 = 256 spatial patches
  - per-frame pair = 2 × 16 × 16 = 512 vision patches
After the 2×2 spatial merger inside the vision tower (Qwen2.5-VL native):
  - 2 × 8 × 8 = 128 vision tokens per video
+ 1 [TASK_CLS] token = 129 tokens fed to the LLM. Very manageable.

Architecture
------------
Input  : (B, T=4, C=3, 224, 224)  ImageNet-normalised (matches dataloader)
1.  De-normalise ImageNet → re-normalise CLIP (Qwen2.5-VL native norm)
2.  Manual patchify (B, T, C, 224, 224) → (B*num_patches, patch_dim)
3.  Vision tower forward → (B*num_merged_tokens, llm_hidden) bf16
4.  Reshape to (B, num_merged_tokens, llm_hidden)
5.  Prepend learnable [TASK_CLS] → (B, 1+num_merged_tokens, llm_hidden)
6.  Qwen2.5 LLM forward (LoRA on q_proj/v_proj, grad-checkpointed)
7.  Read LAST hidden state (causal attention has seen all vision tokens)
8.  LayerNorm + MLP head → (B, 33)
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_BACKBONE   = "Qwen/Qwen2.5-VL-3B-Instruct"
_QWEN_TEMPORAL_PAT  = 2     # native Qwen2.5-VL temporal patch size
_QWEN_SPATIAL_PAT   = 14    # native Qwen2.5-VL spatial patch size
_QWEN_SPATIAL_MERGE = 2     # merger groups 2×2 spatially-adjacent patches
_QWEN_TARGET_H      = 224
_QWEN_TARGET_W      = 224

# ImageNet normalisation (what the dataloader hands us)
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
# Qwen2.5-VL uses OpenAI CLIP normalisation (mean=0.5, std=0.5 per channel)
# https://github.com/QwenLM/Qwen2.5-VL — confirmed in processor config.
_QW_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 1, 3, 1, 1)
_QW_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


def _imagenet_to_qwen(x: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalised clips to Qwen2.5-VL's CLIP normalisation."""
    device, dtype = x.device, x.dtype
    m_in = _IN_MEAN.to(device, dtype)
    s_in = _IN_STD.to(device, dtype)
    m_qw = _QW_MEAN.to(device, dtype)
    s_qw = _QW_STD.to(device, dtype)
    return (x * s_in + m_in - m_qw) / s_qw


def _grab(module: nn.Module, names: list[str]):
    for n in names:
        v = getattr(module, n, None)
        if v is not None:
            return v
    return None


def _qwen_patchify(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Patchify a video clip into Qwen2.5-VL's native flat-patch format.

    x: (B, T, C, H, W) in Qwen normalisation, T multiple of 2.
    Returns:
      patches: (B * num_patches, C * Tpat * P * P)
      grid:    (Tp, Hp, Wp) — same for every video in the batch

    Patch order is spatial-merge-aware: for each temporal index, patches are
    grouped so that consecutive blocks of (spatial_merge²) patches form a 2×2
    spatial neighbourhood. This matches the layout expected by the merger's
    ``view(-1, hidden_size * spatial_merge²)`` projection.
    """
    B, T, C, H, W = x.shape
    Tpat = _QWEN_TEMPORAL_PAT
    P    = _QWEN_SPATIAL_PAT
    sm   = _QWEN_SPATIAL_MERGE
    assert T % Tpat == 0, f"T={T} must be multiple of {Tpat}"
    assert H % P == 0 and W % P == 0, f"H,W must be multiples of {P}, got {H}×{W}"
    Tp, Hp, Wp = T // Tpat, H // P, W // P
    assert Hp % sm == 0 and Wp % sm == 0, \
        f"Hp×Wp ({Hp}×{Wp}) must be divisible by spatial_merge={sm}"

    # (B, T=Tp*Tpat, C, H=Hp*P, W=Wp*P)
    # → split into (B, Tp, Tpat, C, Hp/sm, sm, P, Wp/sm, sm, P)
    x = x.view(B, Tp, Tpat, C, Hp // sm, sm, P, Wp // sm, sm, P)
    # Reorder so consecutive sm×sm patches inside each (Tp, Hb, Wb) block are
    # memory-adjacent, then patch pixels follow:
    # → (B, Tp, Hp/sm, Wp/sm, sm, sm, C, Tpat, P, P)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
    # Flatten all but pixel dims:
    # (B * Tp * Hp/sm * Wp/sm * sm * sm,  C * Tpat * P * P)
    # which equals (B * Tp * Hp * Wp, C * Tpat * P * P) — same total count,
    # just ordered so consecutive 4 patches are spatial neighbours.
    patches = x.view(B * Tp * Hp * Wp, C * Tpat * P * P)
    return patches, (Tp, Hp, Wp)


class Qwen25VL3BVideo(nn.Module):
    """Qwen2.5-VL-3B-Instruct (frozen vision + LoRA LLM) → classifier head."""

    def __init__(
        self,
        num_classes:           int   = 33,
        backbone:              str   = _DEFAULT_BACKBONE,
        cache_dir:             str   = "/Data/.hf_cache",
        lora_rank:             int   = 16,
        lora_alpha:            float = 32.0,
        lora_dropout:          float = 0.05,
        lora_targets:          tuple = ("q_proj", "v_proj"),
        head_hidden:           int   = 2048,
        dropout:               float = 0.3,
        use_gradient_checkpointing: bool = True,
        target_frames:         int   = 8,    # multiple of temporal_patch_size=2
    ) -> None:
        super().__init__()
        assert target_frames % _QWEN_TEMPORAL_PAT == 0, \
            f"target_frames={target_frames} must be multiple of {_QWEN_TEMPORAL_PAT}"
        self.target_frames = target_frames

        # ── Load Qwen2.5-VL full model in bf16 ───────────────────────────────
        print(f"[Qwen25VL3BVideo] Loading {backbone} (bfloat16)…")
        from transformers import Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            backbone, cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )

        # transformers 4.x layout: .visual + .model (Qwen2_5_VLModel with .language_model)
        # transformers 5.x layout: nested under .model
        # Defensive extraction:
        host = _grab(full, ["model"]) or full
        self.visual = _grab(full, ["visual"]) or _grab(host, ["visual"])
        if self.visual is None:
            raise RuntimeError(
                f"Could not locate visual encoder. full children: "
                f"{[n for n, _ in full.named_children()]}"
            )

        # The LLM: try host.language_model first (5.x), then host itself
        llm = _grab(host, ["language_model"])
        if llm is None:
            # In 4.x, .model is the full LM model already
            llm = host

        # Drill down: bare transformer (no lm_head) is at .model on Qwen2 decoders
        inner = getattr(llm, "model", None)
        if isinstance(inner, nn.Module) and any(n == "layers" for n, _ in inner.named_children()):
            self.llm = inner
        else:
            self.llm = llm

        del full
        gc.collect()

        # ── Freeze the entire backbone ────────────────────────────────────────
        for p in self.visual.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False
        self.visual.eval()

        # ── LoRA on LLM q/v_proj ──────────────────────────────────────────────
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(lora_targets),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        if use_gradient_checkpointing:
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                if hasattr(self.llm, "enable_input_require_grads"):
                    self.llm.enable_input_require_grads()
            except Exception as e:
                print(f"[Qwen25VL3BVideo] gradient_checkpointing_enable failed: {e}")

        # Probe LLM hidden size
        try:
            self.llm_hidden = int(self.llm.config.hidden_size)
        except AttributeError:
            self.llm_hidden = int(self.llm.base_model.config.hidden_size)

        # ── Learnable [TASK_CLS] in LLM embedding space ──────────────────────
        self.task_cls = nn.Parameter(torch.zeros(1, 1, self.llm_hidden))
        nn.init.trunc_normal_(self.task_cls, std=0.02)

        # ── Classification head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(self.llm_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.llm_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[Qwen25VL3BVideo] LLM_hidden={self.llm_hidden} target_frames={target_frames} "
            f"| {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.2f}M trainable "
            f"(LoRA rank={lora_rank}, targets={list(lora_targets)})"
        )

    # ── training mode / param routing ────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()  # vision tower always in eval
        return self

    def head_parameters(self):
        return [self.task_cls] + list(self.head.parameters())

    def lora_parameters(self):
        return [p for n, p in self.llm.named_parameters()
                if p.requires_grad and ("lora_" in n)]

    def backbone_parameters(self):
        return []

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 1.0):
        return [
            {"params": self.head_parameters(), "lr": head_lr},
            {"params": self.lora_parameters(), "lr": head_lr},
        ]

    # ── forward ───────────────────────────────────────────────────────────────

    def _upsample_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Linear-interpolate T frames to self.target_frames (matches V-JEPA wrapper)."""
        B, T, C, H, W = x.shape
        if T == self.target_frames:
            return x
        device, dtype = x.device, x.dtype
        t_src  = torch.linspace(0, T - 1, self.target_frames, device=device, dtype=dtype)
        t_low  = t_src.floor().long().clamp(0, T - 2)
        t_high = (t_low + 1).clamp(0, T - 1)
        alpha  = (t_src - t_low.to(dtype)).view(1, self.target_frames, 1, 1, 1)
        return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W), ImageNet-normalised. T may be 4 (default dataloader).
        B = x.shape[0]
        out_dtype = x.dtype

        # 1. Upsample T → target_frames (multiple of 2) so temporal_patch fits
        x = self._upsample_frames(x)                        # (B, target_frames, C, H, W)

        # 2. Convert ImageNet norm → Qwen CLIP norm
        x = _imagenet_to_qwen(x)

        # 3. Resize to Qwen's expected resolution if needed (no-op for 224×224)
        B, T, C, H, W = x.shape
        if H != _QWEN_TARGET_H or W != _QWEN_TARGET_W:
            x = F.interpolate(
                x.view(B * T, C, H, W),
                size=(_QWEN_TARGET_H, _QWEN_TARGET_W),
                mode="bilinear", align_corners=False,
            ).view(B, T, C, _QWEN_TARGET_H, _QWEN_TARGET_W)

        # 4. Manual patchify into Qwen's flat-patch format
        patches, (Tp, Hp, Wp) = _qwen_patchify(x.to(torch.bfloat16))
        # patches: (B * Tp * Hp * Wp, C * Tpat * P * P)

        # 5. Vision tower forward — grid_thw is the same for every video in the batch
        grid_thw = torch.tensor([[Tp, Hp, Wp]] * B,
                                device=patches.device, dtype=torch.long)
        with torch.no_grad():
            vis_out = self.visual(patches, grid_thw=grid_thw)
        # Unwrap: tensor (older HF) or BaseModelOutputWithPooling (transformers 5.x)
        if isinstance(vis_out, torch.Tensor):
            vis_seq = vis_out
        else:
            vis_seq = getattr(vis_out, "last_hidden_state", None)
            if vis_seq is None:
                vis_seq = vis_out[0] if isinstance(vis_out, (tuple, list)) else vis_out

        # In transformers 5.x, the visual encoder returns features at
        # visual_hidden (e.g. 1280 for Qwen2.5-VL-3B), NOT at llm_hidden.
        # The merger projects (groups_of_4_at_visual_hidden) → llm_hidden, doing
        # the spatial 2×2 pool simultaneously. Patchify above already ordered
        # patches so consecutive 4 are spatial neighbours — merger reshapes
        # via .view(-1, 4 * visual_hidden) and runs an MLP.
        if vis_seq.shape[-1] != self.llm_hidden:
            if hasattr(self.visual, "merger"):
                with torch.no_grad():
                    vis_seq = self.visual.merger(vis_seq)
            else:
                raise RuntimeError(
                    f"Visual encoder emits dim {vis_seq.shape[-1]} ≠ LLM dim "
                    f"{self.llm_hidden} and no .merger module is exposed."
                )

        # After merger: (B * merged_per_vid, llm_hidden)
        # merged_per_vid = Tp * (Hp/sm) * (Wp/sm) = 4 * 8 * 8 = 256 for 8×224²
        total_merged = vis_seq.shape[0]
        merged_per_vid = total_merged // B
        vis_seq = vis_seq.view(B, merged_per_vid, self.llm_hidden).to(torch.bfloat16)

        # 6. Prepend [TASK_CLS]
        cls = self.task_cls.expand(B, -1, -1).to(torch.bfloat16)
        inputs_embeds = torch.cat([cls, vis_seq], dim=1)        # (B, 1+merged, llm_hidden)

        # 7. LLM forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
        )
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            last_hidden = outputs.hidden_states[-1]

        # 8. Causal read-out: last token has attended to every vision token
        pooled = last_hidden[:, -1].to(out_dtype)
        return self.head(pooled)

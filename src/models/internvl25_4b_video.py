"""InternVL25_4BVideo — InternVL2.5-4B as a video classifier for SSv2.

Motivation
----------
OpenGVLab InternVL2.5 family (released Dec 2024) is the 5th-gen InternVL,
combining the InternViT-300M-448px-V2_5 vision encoder with the Qwen2.5-3B-
Instruct language model. Distinct from Qwen2.5-VL in:
  - InternViT vision tower (300M, 448×448, separate pretraining)
  - Pixel-shuffle 2×2 vision-token compression (1024 → 256 per image)
  - Per-frame encoding (no native video tokenizer)
                                                                            ── 4B
Memory budget (bf16)
--------------------
  InternVL2.5-4B weights : ~8.0 GB     (4B × 2)
  Vision activations     : ~1.0-1.5 GB (4 frames × 448²)
  LLM activations        : ~1.0-1.5 GB (1024+1 tokens, grad checkpoint, batch=1)
  LoRA + head + TASK_CLS :  ~0.05 GB
                           -----------
                           ~10-11 GB → fits comfortably on RTX A4000 16GB.

Token economy
-------------
Per frame at 448×448 with patch_size=14: (448/14)² = 1024 vision patches.
After InternVL's pixel-shuffle 2×2 spatial compression: 256 tokens / frame.
For 4 frames: 4 × 256 = 1024 vision tokens fed to the LLM, + 1 [TASK_CLS]
= 1025 tokens. Comfortable LLM sequence length.

Architecture
------------
Input  : (B, T=4, C=3, H, W)  ImageNet-normalised (matches dataloader)
1.  Resize H,W → 448×448 + re-normalise to InternViT's expected range
2.  Encode each frame independently via vision_model → (B*T, P, Dv)
3.  mlp1 projector + pixel_shuffle → (B*T, P/4, llm_hidden)
4.  Reshape (B, T*P/4, llm_hidden)
5.  Prepend [TASK_CLS]
6.  Forward Qwen2.5-3B LLM with LoRA on q/v_proj
7.  Read LAST hidden state → LayerNorm + MLP head → (B, 33)
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_BACKBONE   = "OpenGVLab/InternVL2_5-4B"
_INTERN_TARGET_SIZE = 448  # InternViT-300M-448px native resolution
_INTERN_PATCH_SIZE  = 14   # ViT patch size → (448/14)² = 1024 patches per frame

# ImageNet normalisation (dataloader convention)
_IN_MEAN  = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD   = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
# InternViT normalisation: same as ImageNet (per InternVL config — IMG_MEAN/IMG_STD)
# https://huggingface.co/OpenGVLab/InternVL2_5-4B/blob/main/configuration_internvl_chat.py
# → no renormalisation needed when source is ImageNet-normalised.


def _grab(module: nn.Module, names: list[str]):
    for n in names:
        v = getattr(module, n, None)
        if v is not None:
            return v
    return None


def _pixel_shuffle(x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    """Pixel shuffle as used by InternVL: trades spatial → channel by 1/scale².

    x: (B, H*W, C). H and W are assumed equal (square spatial grid).
    Returns: (B, (H*scale)*(W*scale), C/scale²).
    """
    B, N, C = x.shape
    side = int(N ** 0.5)
    assert side * side == N, f"Non-square spatial grid: {N}"
    h, w = side, side
    x = x.view(B, h, w, C)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    inv  = int(1 / scale_factor)              # e.g. 2 for scale=0.5
    # (B, h, w, C) → (B, new_h, inv, new_w, inv, C)
    x = x.view(B, new_h, inv, new_w, inv, C)
    # → (B, new_h, new_w, inv, inv, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # collapse inv·inv into channels
    x = x.view(B, new_h * new_w, C * inv * inv)
    return x


class InternVL25_4BVideo(nn.Module):
    """InternVL2.5-4B (frozen vision + LoRA LLM) → classifier head."""

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
        pixel_shuffle_scale:   float = 0.5,   # 0.5 = 4× token compression (256/frame)
    ) -> None:
        super().__init__()
        self.pixel_shuffle_scale = pixel_shuffle_scale

        # ── Load InternVL2.5-4B (trust_remote_code) ───────────────────────────
        print(f"[InternVL25_4BVideo] Loading {backbone} (bfloat16)…")
        from transformers import AutoConfig, AutoModel
        from peft import LoraConfig, get_peft_model

        # Force eager attention everywhere — InternVL's custom code instantiates
        # Qwen2ForCausalLM(config.llm_config) directly, so the attn_implementation
        # kwarg on from_pretrained never reaches the inner LLM. We mutate the
        # sub-configs ourselves to bypass the flash_attn check that crashes with
        # "flash_attn.__spec__ is None" on hosts with a half-installed flash_attn.
        config = AutoConfig.from_pretrained(
            backbone, cache_dir=cache_dir, trust_remote_code=True,
        )
        config._attn_implementation = "eager"
        for sub in ("llm_config", "vision_config", "text_config"):
            sub_cfg = getattr(config, sub, None)
            if sub_cfg is not None:
                sub_cfg._attn_implementation = "eager"

        full = AutoModel.from_pretrained(
            backbone, cache_dir=cache_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        # InternVL chat model layout:
        #   .vision_model  : InternVisionModel
        #   .mlp1          : projector (Linear/MLP → LLM hidden)
        #   .language_model: Qwen2ForCausalLM (or LlamaForCausalLM in other variants)
        self.vision_model = _grab(full, ["vision_model", "visual"])
        if self.vision_model is None:
            raise RuntimeError(
                f"Could not locate vision_model. children: "
                f"{[n for n, _ in full.named_children()]}"
            )

        self.mlp1 = _grab(full, ["mlp1", "multi_modal_projector", "projector"])
        if self.mlp1 is None:
            raise RuntimeError("Could not locate mlp1/projector module.")

        lm = _grab(full, ["language_model"])
        if lm is None:
            raise RuntimeError("Could not locate language_model.")
        # Drill to the bare transformer (no lm_head) — .model on Qwen2/Llama
        inner = getattr(lm, "model", None)
        if isinstance(inner, nn.Module) and any(n == "layers" for n, _ in inner.named_children()):
            self.llm = inner
        else:
            self.llm = lm

        # InternVL stores its own pixel_shuffle scale in config; respect it if present.
        cfg_scale = getattr(getattr(full, "config", None), "downsample_ratio", None)
        if cfg_scale is not None:
            self.pixel_shuffle_scale = float(cfg_scale)

        del full
        gc.collect()

        # ── Freeze backbone ───────────────────────────────────────────────────
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.mlp1.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False
        self.vision_model.eval()
        self.mlp1.eval()

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
                print(f"[InternVL25_4BVideo] gradient_checkpointing_enable failed: {e}")

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
            f"[InternVL25_4BVideo] LLM_hidden={self.llm_hidden} "
            f"pixel_shuffle_scale={self.pixel_shuffle_scale} "
            f"| {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.2f}M trainable "
            f"(LoRA rank={lora_rank}, targets={list(lora_targets)})"
        )

    # ── training mode / param routing ────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        self.mlp1.eval()
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

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode (BT, C, 448, 448) → (BT, num_tokens_after_shuffle, llm_hidden)."""
        out = self.vision_model(pixel_values=frames)
        feats = getattr(out, "last_hidden_state", None)
        if feats is None:
            feats = out[0] if isinstance(out, (tuple, list)) else out
        # Drop CLS token at position 0 (InternViT outputs CLS + patches)
        if feats.dim() == 3 and feats.shape[1] > 1:
            feats = feats[:, 1:]
        # Pixel-shuffle compression
        feats = _pixel_shuffle(feats, scale_factor=self.pixel_shuffle_scale)
        # Project to LLM hidden via mlp1
        feats = self.mlp1(feats)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) — ImageNet-normalised. T = 4 by default.
        B, T, C, H, W = x.shape

        # 1. Resize each frame to 448×448 if needed
        if H != _INTERN_TARGET_SIZE or W != _INTERN_TARGET_SIZE:
            x = F.interpolate(
                x.view(B * T, C, H, W),
                size=(_INTERN_TARGET_SIZE, _INTERN_TARGET_SIZE),
                mode="bilinear", align_corners=False,
            ).view(B, T, C, _INTERN_TARGET_SIZE, _INTERN_TARGET_SIZE)

        # 2. Encode each frame independently in bf16 (vision model is frozen)
        frames = x.view(B * T, C, _INTERN_TARGET_SIZE, _INTERN_TARGET_SIZE).to(torch.bfloat16)
        vis_feats = self._encode_frames(frames)             # (B*T, P/4, llm_hidden)

        # 3. Reshape to (B, T * P/4, llm_hidden) — all frames concatenated in sequence
        Ntok = vis_feats.shape[1]
        vis_seq = vis_feats.view(B, T * Ntok, self.llm_hidden).to(torch.bfloat16)

        # 4. Prepend [TASK_CLS]
        cls = self.task_cls.expand(B, -1, -1).to(torch.bfloat16)
        inputs_embeds = torch.cat([cls, vis_seq], dim=1)    # (B, 1 + T*Ntok, llm_hidden)

        # 5. Forward LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
        )
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            last_hidden = outputs.hidden_states[-1]

        # 6. Causal read-out
        pooled = last_hidden[:, -1].to(x.dtype)
        return self.head(pooled)

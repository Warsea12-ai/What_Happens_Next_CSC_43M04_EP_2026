"""VJEPA2Llama3VLM — V-JEPA 2 ViT-G as vision tower + LLaMA-3.1-8B as LLM.

Motivation
----------
LlavaMistralVideo uses CLIP-L (304M, image-level, web-pretrained) as the
vision tower. This module swaps CLIP for V-JEPA 2 ViT-G (1.01B, *video*-native,
*SSv2-pretrained*) — the same backbone that already gives our best single-model
results — and pairs it with the bigger / more recent LLaMA-3.1-8B-Instruct.

Hypothesis: a vision tower trained on the *exact* domain (SSv2 hand-object
interactions) feeding a stronger LLM should beat the CLIP+Llama-2 combination
on temporally-fine action classification, especially for the "What Happens
Next" anticipation framing where ordering matters.

Memory budget (bf16, no quantization — bitsandbytes not in the venv)
--------------------------------------------------------------------
  V-JEPA 2 ViT-G   : 1012M × 2  =  ~2.0 GB
  LLaMA-3.1-8B     : 8030M × 2  = ~16.1 GB
  Projector+head   :              ~0.1 GB
  LoRA adapters    :              ~0.05 GB
  Activations      : ~1-2 GB (gradient checkpointing on LLaMA, batch=1)
                                 -----------
                                 ~19-21 GB → fits on A5000/3090 24GB,
                                              comfortable on A100 40GB.

Token economy
-------------
V-JEPA 2 with 16 frames @ 224x224 → 8 temporal × 14×14 spatial = 1568 tokens.
LLaMA reading 1568 tokens with grad-checkpointing is feasible but slow. We
expose `vision_token_pool` to spatially pool the V-JEPA tokens before they
hit the projector:
  - "none"        : 1568 tokens                              (most info, slowest)
  - "spatial2x"   : 8 × 7×7  = 392 tokens                    (default — 4× cheaper)
  - "spatial4x"   : 8 × 4×4  = 128 tokens                    (max compression)
  - "temporal2x"  : 4 × 14×14 = 784 tokens
  - "both"        : 4 × 7×7  = 196 tokens                    (aggressive)

Architecture
------------
Input  : (B, T=4, C=3, 224, 224)  ImageNet-normalised — same as the rest of the codebase
1.  Linear upsample 4 → 16 frames                                (same as vjepa2_head)
2.  V-JEPA 2 encoder forward in no_grad → (B, 1568, 1408)        (full freeze by default)
3.  Optional spatial/temporal token pool                         → (B, P, 1408)
4.  Projector: LayerNorm → Linear(1408, 4096) → GELU → Linear(4096, 4096)
                                                                 → (B, P, 4096)
5.  Prepend learnable [TASK_CLS] token                            → (B, 1+P, 4096)
6.  LLaMA-3.1-8B forward (LoRA on q_proj/v_proj, grad-checkpointed)
7.  Read LAST hidden state (causal: it has seen all P vision tokens)
8.  LayerNorm + MLP head → (B, 33)
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import VJEPA2Model

_VJEPA_TARGET_FRAMES = 16          # native eval clip length for ViT-G
_VJEPA_HIDDEN        = 1408        # ViT-G width
_VJEPA_SPATIAL_SIDE  = 14          # 224 / 16 patch size = 14
_VJEPA_TEMPORAL      = 8           # 16 frames / 2-frame tubelet = 8
_DEFAULT_VJEPA_CKPT  = "facebook/vjepa2-vitg-fpc64-384-ssv2"
_DEFAULT_LLAMA_CKPT  = "NousResearch/Meta-Llama-3.1-8B-Instruct"


# ── helpers ───────────────────────────────────────────────────────────────────

def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Same temporal upsampler as vjepa2_head, kept local to avoid cross-import."""
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


def _pool_vjepa_tokens(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """Spatial/temporal pool V-JEPA tokens before projecting to LLaMA.

    tokens: (B, T_v * S_v, D) with T_v=8, S_v=14*14.
    """
    if mode == "none":
        return tokens
    B, N, D = tokens.shape
    T_v, side = _VJEPA_TEMPORAL, _VJEPA_SPATIAL_SIDE
    assert N == T_v * side * side, f"Expected {T_v*side*side} tokens, got {N}"
    x = tokens.view(B, T_v, side, side, D).permute(0, 1, 4, 2, 3).contiguous()
    # x: (B, T_v, D, side, side)
    if mode in ("spatial2x", "both"):
        x = F.avg_pool2d(x.flatten(0, 1), kernel_size=2, stride=2).view(B, T_v, D, side // 2, side // 2)
        side = side // 2
    elif mode == "spatial4x":
        # 14 isn't divisible by 4 — use adaptive pool to 4x4 (factor ~3.5x)
        x = F.adaptive_avg_pool2d(x.flatten(0, 1), output_size=(4, 4)).view(B, T_v, D, 4, 4)
        side = 4
    if mode in ("temporal2x", "both"):
        # avg pool over time by 2: (B, T_v, D, s, s) → (B, T_v/2, D, s, s)
        Tn = T_v // 2
        x = x.view(B, Tn, 2, D, side, side).mean(dim=2)
        T_v = Tn
    return x.permute(0, 1, 3, 4, 2).reshape(B, T_v * side * side, D).contiguous()


# ── main model ────────────────────────────────────────────────────────────────

class VJEPA2Llama3VLM(nn.Module):
    """V-JEPA 2 ViT-G (frozen) → projector → LLaMA-3.1-8B (LoRA) → classifier head."""

    def __init__(
        self,
        num_classes:           int   = 33,
        vjepa_backbone:        str   = _DEFAULT_VJEPA_CKPT,
        llama_backbone:        str   = _DEFAULT_LLAMA_CKPT,
        cache_dir:             str   = "/Data/.hf_cache/hub",
        num_frozen_vjepa_blocks: int = 40,    # 40/40 = fully frozen V-JEPA
        vision_token_pool:     str   = "spatial2x",
        lora_rank:             int   = 16,
        lora_alpha:            float = 32.0,
        lora_dropout:          float = 0.05,
        lora_targets:          tuple = ("q_proj", "v_proj"),
        head_hidden:           int   = 2048,
        dropout:               float = 0.3,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        assert vision_token_pool in ("none", "spatial2x", "spatial4x", "temporal2x", "both"), \
            f"Unknown vision_token_pool: {vision_token_pool}"
        self.vision_token_pool        = vision_token_pool
        self.num_frozen_vjepa_blocks  = num_frozen_vjepa_blocks

        # ── 1. V-JEPA 2 vision tower ─────────────────────────────────────────
        print(f"[VJEPA2Llama3VLM] Loading V-JEPA 2 vision tower: {vjepa_backbone}")
        full = VJEPA2Model.from_pretrained(
            vjepa_backbone, cache_dir=cache_dir,
            frames_per_clip=_VJEPA_TARGET_FRAMES, crop_size=224,
            use_safetensors=True,
        )
        self.vjepa = full.encoder  # discard the JEPA predictor
        del full
        gc.collect()

        for p in self.vjepa.embeddings.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.vjepa.layer):
            for p in block.parameters():
                p.requires_grad = (i >= num_frozen_vjepa_blocks)

        # ── 2. LLaMA-3.1-8B language model ───────────────────────────────────
        # Lazy import so this file is importable without LLaMA deps installed.
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        print(f"[VJEPA2Llama3VLM] Loading LLM: {llama_backbone} (bfloat16)")
        llama_full = AutoModelForCausalLM.from_pretrained(
            llama_backbone,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # We want the bare transformer (.model on LlamaForCausalLM): its forward
        # accepts inputs_embeds and returns BaseModelOutputWithPast — no lm_head.
        self.llama = llama_full.model
        del llama_full
        gc.collect()

        for p in self.llama.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(lora_targets),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.llama = get_peft_model(self.llama, lora_cfg)

        if use_gradient_checkpointing:
            try:
                self.llama.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                if hasattr(self.llama, "enable_input_require_grads"):
                    self.llama.enable_input_require_grads()
            except Exception as e:
                print(f"[VJEPA2Llama3VLM] gradient_checkpointing_enable failed: {e}")

        # ── 3. Probe dims, build projector + head ────────────────────────────
        try:
            self.llama_hidden = int(self.llama.config.hidden_size)
        except AttributeError:
            self.llama_hidden = int(self.llama.base_model.config.hidden_size)

        # Projector: V-JEPA hidden (1408) → LLaMA hidden (4096)
        # 2-layer MLP mirrors LLaVA's mm_projector design.
        self.projector = nn.Sequential(
            nn.LayerNorm(_VJEPA_HIDDEN),
            nn.Linear(_VJEPA_HIDDEN, self.llama_hidden),
            nn.GELU(),
            nn.Linear(self.llama_hidden, self.llama_hidden),
        )

        # Learnable [TASK_CLS] token, lives in LLaMA embedding space
        self.task_cls = nn.Parameter(torch.zeros(1, 1, self.llama_hidden))
        nn.init.trunc_normal_(self.task_cls, std=0.02)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.llama_hidden),
            nn.Dropout(dropout),
            nn.Linear(self.llama_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[VJEPA2Llama3VLM] V-JEPA={_VJEPA_HIDDEN} LLaMA={self.llama_hidden} "
            f"pool={vision_token_pool} | {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.2f}M trainable "
            f"(LoRA rank={lora_rank}, targets={list(lora_targets)}, "
            f"vjepa_frozen={num_frozen_vjepa_blocks}/40)"
        )

    # ── training mode / param routing ────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen vision parts stay in eval to disable dropout / running-stats updates.
        self.vjepa.embeddings.eval()
        for i, block in enumerate(self.vjepa.layer):
            if i < self.num_frozen_vjepa_blocks:
                block.eval()
        return self

    def head_parameters(self):
        """Non-LoRA trainable params: projector + [TASK_CLS] + classification head."""
        return [self.task_cls] + list(self.projector.parameters()) + list(self.head.parameters())

    def lora_parameters(self):
        return [p for n, p in self.llama.named_parameters()
                if p.requires_grad and ("lora_" in n)]

    def backbone_parameters(self):
        # Only relevant if num_frozen_vjepa_blocks < 40
        return [p for p in self.vjepa.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 1.0):
        """Param groups for AdamW.

        - head_lr   → projector + [TASK_CLS] + classification head + LoRA adapters
        - backbone_lr → unfrozen V-JEPA blocks (if any), with optional LLRD
        """
        groups = [
            {"params": self.head_parameters(), "lr": head_lr},
            {"params": self.lora_parameters(), "lr": head_lr},
        ]
        bb_params = self.backbone_parameters()
        if not bb_params:
            return groups
        if llrd >= 1.0:
            groups.append({"params": bb_params, "lr": backbone_lr})
            return groups
        # Per-block LLRD on the unfrozen V-JEPA blocks
        n = len(self.vjepa.layer)
        for depth, idx in enumerate(range(n - 1, self.num_frozen_vjepa_blocks - 1, -1)):
            groups.append({
                "params": list(self.vjepa.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    # ── forward ───────────────────────────────────────────────────────────────

    def _encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """Run V-JEPA encoder. Returns (B, N, _VJEPA_HIDDEN) in input dtype."""
        out_dtype = x.dtype
        x = _linear_upsample(x, _VJEPA_TARGET_FRAMES)

        # Frozen part in no_grad → no activation graph stored
        with torch.no_grad():
            h = self.vjepa.embeddings(pixel_values_videos=x)
            for block in self.vjepa.layer[:self.num_frozen_vjepa_blocks]:
                h = block(h)[0]
        h = h.detach().to(out_dtype)

        # Unfrozen V-JEPA blocks (only if num_frozen_vjepa_blocks < 40)
        for block in self.vjepa.layer[self.num_frozen_vjepa_blocks:]:
            h = block(h)[0]
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W), ImageNet-normalised. T is whatever the dataset gives (usually 4).
        B = x.shape[0]

        vis_feats = self._encode_video(x)                       # (B, 1568, 1408)
        vis_feats = _pool_vjepa_tokens(vis_feats, self.vision_token_pool)  # (B, P, 1408)

        # Project to LLaMA embedding space — keep bf16 for memory parity with LLaMA
        proj = self.projector(vis_feats).to(torch.bfloat16)     # (B, P, llama_hidden)

        # Prepend [TASK_CLS]
        cls = self.task_cls.expand(B, -1, -1).to(torch.bfloat16)
        inputs_embeds = torch.cat([cls, proj], dim=1)           # (B, 1+P, llama_hidden)

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
        )
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            last_hidden = outputs.hidden_states[-1]

        # Causal read-out: last token has attended to every vision token.
        pooled = last_hidden[:, -1].to(x.dtype)                 # (B, llama_hidden)
        return self.head(pooled)                                # (B, num_classes)

"""LlavaMistralVideo — biggest pure-bf16 LLM (Mistral-7B) adapted to the video task.

Backbone: llava-hf/LLaVA-NeXT-Video-7B-hf
  - LLM            : Mistral-7B-Instruct-v0.2 (32 layers, hidden=4096) — LoRA-tuned
  - Vision tower   : CLIP-ViT-L-336 (304 M)                            — frozen
  - MM projector   : 2-layer MLP (CLIP → LLM)                          — frozen

Why this is "the biggest LLM trainable on an A5000"
---------------------------------------------------
A5000 has 24 GB. Without bitsandbytes we cannot quantize, so bf16 is the floor:
  7 B params × 2 bytes = 14 GB just for weights. With LoRA-only training the
  optimiser states are negligible (~50 MB on a few-rank A/B matrices). Add
  CLIP-L (0.6 GB), the rich head (50 MB), and activations under gradient
  checkpointing (~1 GB at batch 1, ~2 GB at batch 2), and we land at 17-18 GB.

Anything bigger (Pixtral-12B, Mistral-Small-3.1-24B) requires 4-bit QLoRA,
which needs bitsandbytes — not currently installed on the shared venv.

How the task is adapted
-----------------------
The model is a chat VLM, not a classifier. To turn it into one without losing
the multi-modal pretraining:

  1. 4 input frames → resize 336×336 → renormalise ImageNet → CLIP.
  2. CLIP-L encodes each frame → 576 patch tokens × 4 frames = 2304 vision
     tokens (we drop CLIP's own CLS).
  3. Pretrained multi-modal projector maps each vision token into Mistral's
     4096-dim space — this is the projector that LLaVA already trained on
     hundreds of millions of (image, caption) pairs; we reuse it verbatim.
  4. Prepend one learnable [TASK_CLS] embedding (also in Mistral space) so
     the LLM has a dedicated read-out position. Total sequence = 1 + 2304.
  5. Forward through Mistral (LoRA q_proj/v_proj, gradient checkpointing).
  6. Use the [TASK_CLS] output as the pooled feature → MLP head → 33 classes.

No tokenizer, no chat template, no text prompt. The LLM acts as a deep
spatiotemporal mixer over the projected vision tokens, with one extra learned
slot that aggregates everything into a single classification vector.

Input : (B, T=4, C=3, 224, 224)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet normalisation (what the dataloader hands us)
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
# CLIP normalisation (what the LLaVA vision tower wants)
_CL_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 1, 3, 1, 1)
_CL_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)

# LLaVA-NeXT-Video-7B's CLIP-L expects 336×336
_CLIP_IMAGE_SIZE = 336
_N_FRAMES = 4


def _imagenet_to_clip(x: torch.Tensor) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    m_in = _IN_MEAN.to(device, dtype)
    s_in = _IN_STD.to(device, dtype)
    m_cl = _CL_MEAN.to(device, dtype)
    s_cl = _CL_STD.to(device, dtype)
    return (x * s_in + m_in - m_cl) / s_cl


def _grab(module: nn.Module, names: list[str]):
    """Return the first attribute among `names` that exists and is non-None."""
    for n in names:
        v = getattr(module, n, None)
        if v is not None:
            return v
    return None


class LlavaMistralVideo(nn.Module):
    """LLaVA-NeXT-Video-7B (Mistral-7B + CLIP-L) adapted for 33-class classification."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "llava-hf/LLaVA-NeXT-Video-7B-hf",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        head_hidden: int = 2048,
        dropout: float = 0.3,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        print(f"[LlavaMistralVideo] Loading {backbone} (bfloat16)…")
        # Lazy imports so the file is at least importable when the heavy deps
        # are missing (we only crash when the model is actually instantiated).
        from transformers import LlavaNextVideoForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        full = LlavaNextVideoForConditionalGeneration.from_pretrained(
            backbone, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )

        # ── Extract the three components defensively ─────────────────────────
        self.vision_tower = _grab(full, ["vision_tower", "vision_model", "visual"])
        if self.vision_tower is None:
            raise RuntimeError("Could not locate vision_tower in LlavaNextVideo model.")

        self.multi_modal_projector = _grab(
            full, ["multi_modal_projector", "mm_projector", "vision_resampler",
                   "projector"],
        )
        if self.multi_modal_projector is None:
            raise RuntimeError("Could not locate multi_modal_projector in LlavaNextVideo model.")

        # The language model is a MistralForCausalLM-like wrapper. We want its
        # *transformer* (returns hidden states), not the lm_head.
        lm = _grab(full, ["language_model"])
        if lm is None:
            raise RuntimeError("Could not locate language_model in LlavaNextVideo model.")
        # MistralForCausalLM exposes the bare transformer as `.model`. If not,
        # fall back to the LM wrapper itself (its forward returns logits we'd
        # have to ignore — but output_hidden_states still works).
        inner = getattr(lm, "model", None)
        self.language_model = inner if isinstance(inner, nn.Module) else lm

        # Free the wrapper (we only kept submodules)
        del full
        gc.collect()

        # ── Freeze everything that should stay frozen ────────────────────────
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.multi_modal_projector.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False
        self.vision_tower.eval()
        self.multi_modal_projector.eval()

        # ── Apply LoRA to the Mistral transformer ────────────────────────────
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        # get_peft_model wraps the language model; the wrapper exposes the same
        # forward interface plus trainable LoRA parameters.
        self.language_model = get_peft_model(self.language_model, lora_cfg)

        if use_gradient_checkpointing:
            try:
                # PEFT proxies gradient_checkpointing_enable() to the base model
                self.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                # required when inputs come from non-grad sources (vision tokens)
                if hasattr(self.language_model, "enable_input_require_grads"):
                    self.language_model.enable_input_require_grads()
            except Exception as e:
                print(f"[LlavaMistralVideo] gradient_checkpointing_enable failed: {e}")

        # ── Probe hidden sizes ───────────────────────────────────────────────
        # Mistral hidden size = LLM embedding dim
        try:
            D_llm = self.language_model.config.hidden_size
        except AttributeError:
            D_llm = self.language_model.base_model.config.hidden_size
        self.hidden_size = int(D_llm)

        # CLIP hidden size — only used to sanity-check, not stored
        try:
            D_vis = self.vision_tower.config.hidden_size
        except AttributeError:
            D_vis = None

        # ── Learnable [TASK_CLS] token in Mistral embedding space ────────────
        self.task_cls = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.trunc_normal_(self.task_cls, std=0.02)

        # ── Classification head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[LlavaMistralVideo] hidden_llm={self.hidden_size} hidden_vis={D_vis} "
            f"  {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.2f}M trainable "
            f"(LoRA rank={lora_rank}, targets={list(target_modules)})"
        )

    # ── training / param routing ──────────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen submodules in eval no matter what
        self.vision_tower.eval()
        self.multi_modal_projector.eval()
        return self

    def head_parameters(self):
        """All non-LoRA trainable params: [TASK_CLS] + classification head."""
        return [self.task_cls] + list(self.head.parameters())

    def lora_parameters(self):
        """LoRA adapters injected into the Mistral transformer."""
        return [p for n, p in self.language_model.named_parameters()
                if p.requires_grad and ("lora_" in n)]

    def backbone_parameters(self):
        # No fully-unfrozen backbone params — LoRA owns the backbone gradient flow.
        return []

    # ── forward ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_frames(self, x_clip: torch.Tensor) -> torch.Tensor:
        """Frozen vision tower forward, returning per-frame patch tokens (without CLS).

        x_clip: (B*T, C, 336, 336) in CLIP normalisation, bf16.
        Returns: (B*T, num_patches, vis_hidden) in bf16.
        """
        out = self.vision_tower(x_clip, output_hidden_states=False)
        feats = getattr(out, "last_hidden_state", None)
        if feats is None:
            feats = out[0] if isinstance(out, (tuple, list)) else out
        # Drop the CLIP [CLS] token at position 0 — we want spatial patches only.
        if feats.dim() == 3 and feats.shape[1] > 1:
            feats = feats[:, 1:]
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, C=3, H, W) — ImageNet-normalised
        B, T, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # ImageNet → CLIP renorm + resize to 336×336 if needed
        x = _imagenet_to_clip(x)
        if H != _CLIP_IMAGE_SIZE or W != _CLIP_IMAGE_SIZE:
            x = F.interpolate(
                x.view(B * T, C, H, W), size=(_CLIP_IMAGE_SIZE, _CLIP_IMAGE_SIZE),
                mode="bilinear", align_corners=False,
            ).view(B, T, C, _CLIP_IMAGE_SIZE, _CLIP_IMAGE_SIZE)

        # Vision encoding in bf16 under no_grad (vision tower is frozen).
        x_bf16 = x.view(B * T, C, _CLIP_IMAGE_SIZE, _CLIP_IMAGE_SIZE).to(torch.bfloat16)
        vis_feats = self._encode_frames(x_bf16)                 # (B*T, P, Dv)
        P = vis_feats.shape[1]

        # Multi-modal projector: frozen, but we still need gradient *through* it
        # because the [TASK_CLS] token and LoRA gradients flow through Mistral
        # which reads the projector's outputs. Projector weights have
        # requires_grad=False so no gradient is *stored* on them.
        proj_tokens = self.multi_modal_projector(vis_feats)     # (B*T, P, D_llm)
        D_llm = proj_tokens.shape[-1]

        # Reshape to (B, T*P, D_llm) — all vision tokens in a single LLM sequence
        vision_seq = proj_tokens.view(B, T * P, D_llm).to(torch.bfloat16)

        # Prepend [TASK_CLS] (learnable, lives in Mistral embedding space)
        cls = self.task_cls.expand(B, -1, -1).to(torch.bfloat16)
        inputs_embeds = torch.cat([cls, vision_seq], dim=1)     # (B, 1 + T*P, D_llm)

        # Mistral forward (LoRA + gradient checkpointing handle memory).
        # We don't pass an attention_mask: all positions are valid, and Mistral
        # uses a causal mask internally so [TASK_CLS] at position 0 can only
        # attend to itself initially — but the loss is read from the LAST
        # position, not the first, to give the LLM full context.
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=False,
        )
        # MistralModel returns BaseModelOutputWithPast: last_hidden_state available.
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            # If wrapped class returned logits, fall back to hidden_states
            last_hidden = outputs.hidden_states[-1]

        # Pool: read out the FINAL token's hidden state. In a causal LM that
        # token has attended to every preceding vision token, giving us a
        # full-context summary (mirrors how Llama/Mistral classifiers do
        # sequence classification with attn_implementation="eager").
        pooled = last_hidden[:, -1].to(dtype)                   # (B, D_llm)

        return self.head(pooled)

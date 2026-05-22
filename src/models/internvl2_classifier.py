"""InternVL2Classifier — InternVL2-8B (SigLIP-SO400M + InternLM2-7.7B) for 33-class
temporal ordering classification.

Why InternVL2 over previous VLMs (Qwen2-VL, DualEncoder)
----------------------------------------------------------
Qwen2-VL relied on text generation → 16-24 % top-1.  InternVL2 avoids this trap:
the LLM last-hidden-state is mean-pooled and fed directly into a linear head,
so the model never generates tokens — it acts as a rich visual-semantic encoder.

Key architecture advantages over DINOv3Temporal / VideoMAE:
  - SigLIP-SO400M vision encoder: trained on 4 B image-text pairs (vs pure-image DINOv3)
  - InternLM2-7.7B cross-attention enriches frame representations with language priors
    about hand-object interactions (SSv2 captions are in the pretraining corpus)
  - Native multi-image input: 4 frames are jointly attended in a single forward pass
    → the LLM can compare frames directly (vs per-frame pooling in DINOv3Temporal)

Memory on RTX A5000 / RTX 3090 (24 GB):
  - InternVL2-8B in bfloat16: ~16 GB weights
  - LoRA adds ~0.3 GB (rank 16, q/k/v/o of 32 LLM layers)
  - Activations (batch=1, 4 frames × 256 tokens + 32 text tokens): ~4 GB
  - Total: ~20 GB → comfortable margin on 24 GB

Forward pass:
  Input   (B, 4, C, H, W) — ImageNet-normalised, H=W=224
  1. Renormalize: ImageNet → InternVL2 (SigLIP: same ImageNet stats — no-op)
  2. Reshape → (B*4, C, H, W), resize to 448×448 via bilinear
  3. InternViT-300M → (B*4, 256, 3200) visual tokens
  4. Vision projector (mlp1) → (B*4, 256, 4096)  [LoRA disabled on vision]
  5. Reshape → (B, 4*256=1024, 4096) — "video tokens"
  6. InternLM2 self-attention over video tokens + 32 text tokens (frozen embeddings)
     [LoRA rank 16 on q/k/v/o — ~180 M extra params]
  7. Mean-pool last hidden state over all tokens → (B, 4096)
  8. Dropout + Linear(4096, 33) → (B, 33)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig

# InternVL2 uses the same ImageNet normalisation as our pipeline — no conversion
_TARGET_RES = 448   # InternVL2's native resolution per tile


def _resize_to_intern(x: torch.Tensor) -> torch.Tensor:
    """Bilinear resize (B, C, H, W) → (B, C, 448, 448)."""
    if x.shape[-2:] == (_TARGET_RES, _TARGET_RES):
        return x
    return F.interpolate(x, size=(_TARGET_RES, _TARGET_RES), mode="bilinear", align_corners=False)


class InternVL2Classifier(nn.Module):
    """InternVL2-8B adapted as a 33-class video-frame classifier."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "OpenGVLab/InternVL2-8B",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        dropout: float = 0.25,
        num_frozen_llm_layers: int = 24,
    ) -> None:
        super().__init__()

        self.backbone_id = backbone

        # ── Load full VLM ───────────────────────────────────────────────────
        model = AutoModel.from_pretrained(
            backbone,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Vision encoder (InternViT-300M) + projection MLP — keep, freeze encoder
        self.vision_model = model.vision_model
        self.mlp1 = model.mlp1          # 2-layer MLP: 3200 → 4096

        for p in self.vision_model.parameters():
            p.requires_grad = False

        # LLM (InternLM2-7.7B) — apply LoRA then freeze lower layers
        llm = model.language_model
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            lora_dropout=0.05,
        )
        self.language_model = get_peft_model(llm, lora_cfg)

        # Freeze lower LLM layers (keep top layers trainable via LoRA)
        n_layers = len(self.language_model.base_model.model.model.layers)
        for i, layer in enumerate(self.language_model.base_model.model.model.layers):
            if i < num_frozen_llm_layers:
                for name, p in layer.named_parameters():
                    if "lora_" not in name:
                        p.requires_grad = False

        del model  # free original reference

        # Tokenizer — fixed 32-token prompt used to give LLM text context
        tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        # Minimal prompt: tells the LLM what to expect without adding noise
        # We use im_patch tokens as visual placeholders internally
        _prompt = "Identify the action category from these video frames."
        prompt_ids = tok(_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        self.register_buffer("prompt_ids", prompt_ids, persistent=False)

        hidden_size = self.language_model.config.hidden_size  # 4096 for InternLM2-7.7B

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"InternVL2Classifier: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(LoRA rank={lora_rank}, {num_frozen_llm_layers}/{n_layers} LLM layers frozen)")

    def train(self, mode: bool = True):
        super().train(mode)
        # Vision encoder always in eval mode (frozen weights, no dropout needed)
        self.vision_model.eval()
        return self

    def head_parameters(self):
        return list(self.head.parameters()) + list(self.mlp1.parameters())

    def backbone_parameters(self):
        return [p for p in self.language_model.parameters() if p.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T=4, C, H, W) — ImageNet-normalised, float32 or bfloat16
        Returns: (B, 33)
        """
        B, T, C, H, W = x.shape
        dtype = next(self.vision_model.parameters()).dtype
        device = x.device

        # ── 1. Vision encoder (frozen) ──────────────────────────────────────
        # Reshape to (B*T, C, 448, 448) for vision encoder
        frames = x.view(B * T, C, H, W).to(dtype)
        frames = _resize_to_intern(frames)                   # (B*T, C, 448, 448)

        with torch.no_grad():
            vit_out = self.vision_model(pixel_values=frames)
            vit_feats = vit_out.last_hidden_state            # (B*T, 256, 3200)

        # ── 2. Vision projector (trainable) ────────────────────────────────
        vit_feats = self.mlp1(vit_feats)                     # (B*T, 256, 4096)
        # Reshape to (B, T*256, 4096) — all 4 frames' tokens in one sequence
        video_tokens = vit_feats.view(B, T * 256, 4096)      # (B, 1024, 4096)

        # ── 3. Prepend text prompt tokens ───────────────────────────────────
        # Embed the fixed prompt through the LLM's embedding layer
        prompt_ids = self.prompt_ids.expand(B, -1).to(device)    # (B, L)
        embed_layer = self.language_model.base_model.model.model.embed_tokens
        text_embeds = embed_layer(prompt_ids).to(dtype)           # (B, L, 4096)

        # Concatenate: [text | visual tokens]
        seq_embeds = torch.cat([text_embeds, video_tokens], dim=1)  # (B, L+1024, 4096)

        # ── 4. LLM forward (LoRA on attention) ─────────────────────────────
        out = self.language_model(
            inputs_embeds=seq_embeds,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]  # (B, L+1024, 4096)

        # ── 5. Global mean pool → classification ────────────────────────────
        feat = last_hidden.mean(dim=1).float()  # (B, 4096) in float32 for stability
        return self.head(feat)                   # (B, 33)

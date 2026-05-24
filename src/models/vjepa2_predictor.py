"""VJEPA2PredictorAnticipator — exploite le `predictor` de V-JEPA 2.

Pourquoi
========
Tous les autres wrappers vjepa2_* du repo font ``del full`` après avoir gardé
l'encoder, jetant le predictor — alors qu'il a été entraîné précisément à
``prédire l'embedding de tokens masqués depuis le contexte``. Pour "what
happens next" c'est l'objectif natif du predictor : on lui donne le présent
encodé + des query tokens représentant le futur, et il populate ces queries
avec des embeddings prédits du futur. On classifie ensuite sur la
concaténation [représentation présente, futur prédit].

Choix d'implémentation
======================
La signature exacte de ``VJEPA2Predictor.forward`` (position_ids, masques,
RoPE-3D…) varie entre versions de ``transformers``. Plutôt que de la deviner,
on traite le predictor comme une **pile de blocs Transformer** appliquée à la
séquence ``[ctx_tokens; future_query_tokens]`` :

  - `ctx_tokens` = sortie de l'encoder (frames observées)
  - `future_query_tokens` = paramètres learnable, joués comme des
    placeholders que le predictor doit "remplir" pendant l'entraînement
  - on accède directement à ``predictor.layer`` (la stack de blocs), ce qui
    est stable côté API

Les poids pré-entraînés du predictor restent utiles : ils savent déjà
mélanger contexte + queries, ce qui se voit dès les premières epochs
(convergence plus rapide qu'un transformer init from scratch).

Architecture
------------
Input : (B, T=4, C=3, H=224, W=224)

1.  Linear-upsample 4 → 16 frames (cohérent avec les autres vjepa2_*).
2.  Encoder V-JEPA 2 ViT-G (gelé en no_grad)  → (B, N_ctx, D_enc=1408)
3.  Projection D_enc → D_pred (LayerNorm + Linear) car le predictor a un
    hidden_size plus petit (ex: 384) que l'encoder.
4.  Concat : [ctx_proj ; future_queries] où future_queries est
    nn.Parameter(N_fut, D_pred), init trunc_normal.
5.  predictor.layer (k blocks) appliqué sur toute la séquence.
6.  Split: present_h (premier morceau) / future_h (queries refined).
7.  Pool mean sur present_h et future_h → concat → tête MLP → 33 classes.

LoRA optionnel sur les blocs du predictor pour fine-tuner sans casser ses
poids pré-entraînés.

Memory budget
=============
  V-JEPA 2 ViT-G encoder (frozen, no_grad)  : ~2.0 GB bf16
  Predictor (~22M, LoRA-fine-tuned)         : ~0.05 GB
  Future queries (128 × 384)               : ~0.2 MB
  Activations @ batch=4, T=16              : ~3-5 GB
  Head + LoRA + AdamW                       : ~0.1 GB
                                            -----------
                                            ~5-7 GB → fits 20 GB easily.
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn

from transformers import VJEPA2Model

_TARGET_FRAMES = 16
_VJEPA_ENC_HIDDEN = 1408   # ViT-G hidden dim — used as fallback if probe fails


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


def _probe_predictor_hidden(predictor: nn.Module, fallback: int = 384) -> int:
    """Find the predictor's hidden dimension defensively.

    Order of preference (most reliable first):
      1. First LayerNorm in the first block — that's the dim the predictor
         actually operates on internally. The HF VJEPA2 config inherits
         hidden_size from the encoder (1408 for ViT-G), which is WRONG
         for the predictor stack.
      2. Explicit `predictor_hidden_size` on the config.
      3. Fallback.
    """
    blocks = getattr(predictor, "layer", None) or getattr(predictor, "blocks", None)
    if blocks is not None and len(blocks) > 0:
        for m in blocks[0].modules():
            if isinstance(m, nn.LayerNorm):
                shape = m.normalized_shape
                if isinstance(shape, (tuple, list)) and len(shape) > 0:
                    return int(shape[0])
                if isinstance(shape, int):
                    return int(shape)
    cfg = getattr(predictor, "config", None)
    if cfg is not None and hasattr(cfg, "predictor_hidden_size"):
        val = getattr(cfg, "predictor_hidden_size")
        if isinstance(val, int) and val > 0:
            return val
    return fallback


def _get_predictor_blocks(predictor: nn.Module) -> nn.ModuleList:
    """Get the stack of transformer blocks from the predictor across HF versions."""
    for attr in ("layer", "blocks", "layers"):
        blocks = getattr(predictor, attr, None)
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            return blocks
    raise RuntimeError(
        f"Cannot locate the predictor's block stack. Children: "
        f"{[n for n, _ in predictor.named_children()]}"
    )


class VJEPA2PredictorAnticipator(nn.Module):
    """V-JEPA 2 encoder (frozen) + predictor (LoRA) + future-query head."""

    def __init__(
        self,
        num_classes:        int   = 33,
        backbone:           str   = "facebook/vjepa2-vitg-fpc64-384-ssv2",
        cache_dir:          Optional[str] = None,
        n_future_queries:   int   = 128,
        head_hidden:        int   = 2048,
        dropout:            float = 0.25,
        use_lora:           bool  = True,
        lora_rank:          int   = 16,
        lora_alpha:         float = 32.0,
        lora_dropout:       float = 0.05,
        lora_targets:       tuple = ("query", "key", "value", "dense"),
        train_predictor_full: bool = False,
    ) -> None:
        super().__init__()
        self.n_future_queries = n_future_queries

        # ── Load V-JEPA 2 with BOTH encoder and predictor ────────────────────
        print(f"[VJEPA2PredictorAnticipator] Loading {backbone}")
        load_kwargs = dict(
            frames_per_clip=_TARGET_FRAMES, crop_size=224, use_safetensors=True,
        )
        if cache_dir is not None and cache_dir != "":
            load_kwargs["cache_dir"] = cache_dir
        full = VJEPA2Model.from_pretrained(backbone, **load_kwargs)
        self.encoder = full.encoder

        predictor = getattr(full, "predictor", None)
        if predictor is None:
            raise RuntimeError(
                "V-JEPA 2 model has no .predictor attribute. The checkpoint "
                "or transformers version may not ship it."
            )
        self.predictor = predictor
        del full
        gc.collect()

        enc_hidden = getattr(self.encoder.config, "hidden_size", _VJEPA_ENC_HIDDEN)
        pred_hidden = _probe_predictor_hidden(self.predictor)
        self.enc_hidden = int(enc_hidden)
        self.pred_hidden = int(pred_hidden)

        # ── Freeze encoder fully (it's the heavy part) ───────────────────────
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ── Predictor: either LoRA fine-tune or full freeze + train scratch ──
        for p in self.predictor.parameters():
            p.requires_grad = train_predictor_full
        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                target_modules=list(lora_targets), lora_dropout=lora_dropout,
                bias="none", task_type="FEATURE_EXTRACTION",
            )
            self.predictor = get_peft_model(self.predictor, lora_cfg)

        # ── Encoder → Predictor dim projection (with LayerNorm) ──────────────
        if self.enc_hidden != self.pred_hidden:
            self.enc_to_pred = nn.Sequential(
                nn.LayerNorm(self.enc_hidden),
                nn.Linear(self.enc_hidden, self.pred_hidden),
            )
        else:
            self.enc_to_pred = nn.LayerNorm(self.enc_hidden)

        # ── Learnable future queries (the model populates these via predictor)
        self.future_queries = nn.Parameter(torch.zeros(1, n_future_queries, self.pred_hidden))
        nn.init.trunc_normal_(self.future_queries, std=0.02)

        # ── Classification head : concat[present_mean, future_mean] → MLP ────
        self.head = nn.Sequential(
            nn.LayerNorm(self.pred_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(self.pred_hidden * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VJEPA2PredictorAnticipator] enc={self.enc_hidden} pred={self.pred_hidden} "
              f"future_queries={n_future_queries} | "
              f"{n_frozen/1e6:.0f}M frozen, {n_train/1e6:.2f}M trainable "
              f"(lora={use_lora}, rank={lora_rank if use_lora else 0})")

    # ── training / param routing ─────────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # always frozen
        return self

    def head_parameters(self):
        return [self.future_queries] + list(self.enc_to_pred.parameters()) + list(self.head.parameters())

    def lora_parameters(self):
        return [p for n, p in self.predictor.named_parameters()
                if p.requires_grad and "lora_" in n]

    def backbone_parameters(self):
        # Encoder is frozen; predictor may have non-LoRA params if train_predictor_full
        bb = [p for p in self.predictor.parameters() if p.requires_grad]
        # subtract LoRA params already listed
        lora_ids = {id(p) for p in self.lora_parameters()}
        return [p for p in bb if id(p) not in lora_ids]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 1.0):
        groups = [
            {"params": self.head_parameters(), "lr": head_lr},
            {"params": self.lora_parameters(), "lr": head_lr},
        ]
        bb = self.backbone_parameters()
        if bb:
            groups.append({"params": bb, "lr": backbone_lr})
        return groups

    # ── forward ───────────────────────────────────────────────────────────────

    def _run_predictor(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply the predictor's transformer stack to a (B, N, D_pred) sequence.

        We deliberately bypass the predictor's wrapper forward (which expects
        position_ids / target masks whose semantics differ across HF versions)
        and call its block stack directly. Pretrained weights still apply.
        """
        # PEFT-wrapped predictor exposes the original under .base_model.model
        # or .base_model — defensively navigate down to the block stack.
        base = self.predictor
        for attr in ("base_model", "model"):
            inner = getattr(base, attr, None)
            if inner is not None and hasattr(inner, "layer") or hasattr(inner, "blocks"):
                base = inner
        try:
            blocks = _get_predictor_blocks(base)
        except RuntimeError:
            blocks = _get_predictor_blocks(self.predictor)

        h = seq
        for block in blocks:
            out = block(h)
            h = out[0] if isinstance(out, (tuple, list)) else out
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W), ImageNet-normalised
        B = x.shape[0]
        out_dtype = x.dtype

        # 1. Upsample temporal to V-JEPA's native window
        x = _linear_upsample(x, _TARGET_FRAMES)

        # 2. Encode (frozen, no grad)
        with torch.no_grad():
            enc_out = self.encoder(pixel_values_videos=x)
        # Handle the various return types
        if hasattr(enc_out, "last_hidden_state"):
            ctx = enc_out.last_hidden_state
        elif isinstance(enc_out, (tuple, list)):
            ctx = enc_out[0]
        else:
            ctx = enc_out
        ctx = ctx.detach()                                # (B, N_ctx, D_enc)

        # 3. Project encoder → predictor dim
        ctx_p = self.enc_to_pred(ctx.to(out_dtype))       # (B, N_ctx, D_pred)

        # 4. Concatenate with learnable future queries
        fut_q = self.future_queries.expand(B, -1, -1).to(ctx_p.dtype)
        seq = torch.cat([ctx_p, fut_q], dim=1)            # (B, N_ctx + N_fut, D_pred)

        # 5. Run predictor blocks
        h = self._run_predictor(seq)                       # (B, N_ctx + N_fut, D_pred)

        # 6. Split present vs predicted future
        N_ctx = ctx_p.shape[1]
        present_h = h[:, :N_ctx]
        future_h  = h[:, N_ctx:]

        # 7. Pool & classify
        present_feat = present_h.mean(dim=1)              # (B, D_pred)
        future_feat  = future_h.mean(dim=1)               # (B, D_pred)
        feat = torch.cat([present_feat, future_feat], dim=-1).to(out_dtype)  # (B, 2*D_pred)

        return self.head(feat)                            # (B, num_classes)

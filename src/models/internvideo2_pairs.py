"""InternVideo2Pairs — InternVideo2-Stage2-1B + Perceiver pool + Cross-Attention Pairs.

Why this architecture
---------------------
VideoMAE saturates around 46% top-1 on this task. The winning pattern is
spatial cross-attention pairs (videomae_cross_attn_pairs, 46.39%): for each
ordered pair (i, j) of input frames, query=pool(j) attends to spatial tokens
of frame i. This captures *where* in frame i the change relevant to frame j
happened — exactly what "what happens next" rewards.

We keep the winning head pattern but swap the backbone to a fundamentally
heavier and newer video foundation model:

  Model                       Params   Pretraining
  VideoMAE-Base-SSv2          86M      SSv2 (only)
  VideoMAE-Large-Kinetics     307M     Kinetics-400
  InternVideo2-Stage2-1B      1.0B     75M videos, multi-objective
                                       (masked + contrastive + UMT distill)
                                       finetuned on K710/SSv2

InternVideo2 Stage-2 was pretrained with the explicit goal of producing rich
spatiotemporal features for downstream tasks — closer to what we need than any
single-objective MAE.

Memory strategy (single RTX 3090, 24 GB)
----------------------------------------
- Backbone frozen in bfloat16 (~2 GB weights, no optimizer state)
- LoRA on attention projections (defensive: tries `qkv` then `q_proj/v_proj`
  then `query/value` to match BEiT, OPT, BERT-style naming)
- Rich head trainable in fp32 (~80M params)
- Effective batch size 32 via batch_size=4 + grad_accum_steps=8

Robustness to backbone internals
--------------------------------
InternVideo2 ships via `trust_remote_code` and the exact attribute layout can
shift between Stage 1, Stage 2, and InternVideo2.5 checkpoints. The model is
written to NEVER touch internal layers directly. It only:
  1. Calls `encoder(pixel_values)` and reads the output defensively
     (`last_hidden_state` or `pooler_output` or bare tensor).
  2. Reshapes the token sequence to (B, M, D) via flatten, so any output rank
     ≥ 2 works.
  3. Builds frame-level summaries via a Perceiver pool (4 learnable queries
     cross-attend to all M tokens) — no assumption about M's structure.
  4. Does spatial-style cross-attention pairs against the full token sequence
     (still asymmetric & spatially-resolved, since M tokens carry spatial info).

The only fragile point is LoRA target matching. If no LoRA adapters are
inserted (no matching module names), the model degrades cleanly to a frozen
backbone + rich head — still strictly more expressive than the existing
InternVideo2 (mean-pool + 2-layer MLP).

Input : (B, T=4, C=3, 224, 224) ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import math
import sys
import types

# ── Stub heavy optional imports that the trust_remote_code modelling file
#    references at top-level. Without these, `transformers.check_imports`
#    rejects the file before we even reach the forward pass.
class _LooseStub(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _LooseStub(f"{self.__name__}.{name}")
        setattr(self, name, sub)
        sys.modules[sub.__name__] = sub
        return sub

    def __call__(self, *args, **kwargs):
        return None


for _stub_name in ("cv2", "flash_attn", "flash_attn.flash_attn_interface",
                   "flash_attn.bert_padding"):
    try:
        __import__(_stub_name)
    except ImportError:
        sys.modules[_stub_name] = _LooseStub(_stub_name)


import torch
import torch.nn as nn
from transformers import AutoModel

from models.lora_utils import apply_lora, get_lora_parameters, count_lora_params


# InternVideo2 expects 8 frames per clip (default tubelet_size=1 → 8 temporal tokens
# stacks, or tubelet_size=2 → 4 temporal stacks). We linearly upsample 4 → 8 so we
# still feed 8 distinct frames but our 4 inputs occupy the even slots, mirroring
# the choice in VideoMAECrossAttnPairs (4 → 16 with the 0/5/10/15 mapping).
_TARGET_FRAMES = 8
_N_FRAMES      = 4
_PAIRS         = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# InternVideo2 uses ImageNet normalisation, same as our input pipeline. No
# renormalisation needed (unlike VideoMAE which uses [0.5, 0.5, 0.5]).


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


def _extract_features(out) -> torch.Tensor:
    """Defensive extraction of (B, M, D) token features from heterogeneous outputs."""
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        feats = out.last_hidden_state
    elif hasattr(out, "hidden_states") and out.hidden_states is not None:
        feats = out.hidden_states[-1]
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        feats = out.pooler_output
    elif isinstance(out, torch.Tensor):
        feats = out
    elif isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        raise RuntimeError(f"Unexpected encoder output type: {type(out)}")

    # Collapse anything between batch and feature dim into a single token axis.
    if feats.dim() == 2:           # (B, D) — already pooled; add fake token axis
        feats = feats.unsqueeze(1)
    elif feats.dim() == 3:         # (B, M, D) — canonical case
        pass
    elif feats.dim() == 4:         # (B, T, N, D) → (B, T*N, D)
        feats = feats.flatten(1, 2)
    elif feats.dim() == 5:         # (B, T, H, W, D) → (B, T*H*W, D)
        feats = feats.flatten(1, 3)
    else:
        raise RuntimeError(f"Cannot flatten features of shape {tuple(feats.shape)}")
    return feats


def _infer_hidden_size(encoder: nn.Module) -> int:
    """Find the hidden dim from the encoder config under any of several names."""
    cfg = getattr(encoder, "config", None)
    if cfg is None:
        raise RuntimeError("Encoder has no .config attribute")
    for attr in ("hidden_size", "embed_dim", "vision_embed_dim", "d_model"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    # Nested config (e.g. vision_config inside a multi-modal config)
    for attr in ("vision_config", "vision_cfg"):
        sub = getattr(cfg, attr, None)
        if sub is not None:
            for sub_attr in ("hidden_size", "embed_dim", "vision_embed_dim", "d_model"):
                v = getattr(sub, sub_attr, None)
                if isinstance(v, int) and v > 0:
                    return v
    raise RuntimeError(
        f"Cannot determine hidden size from {type(encoder).__name__} — "
        f"config attrs: {sorted(vars(cfg).keys())[:20]}"
    )


def _unwrap_vision_encoder(model: nn.Module) -> nn.Module:
    """If the loaded model is multi-modal, return the vision-only sub-module."""
    for attr in ("vision_encoder", "vision_model", "visual", "encoder_vision"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module):
            return sub
    return model


class _TemporalBlock(nn.Module):
    """Pre-LN self-attention block."""

    def __init__(self, d: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ff(self.norm2(x))
        return x


class InternVideo2Pairs(nn.Module):
    """InternVideo2-Stage2-1B + Perceiver pool + Cross-Attention Pairs head."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "OpenGVLab/InternVideo2-Stage2-1B-224p",
        backbone_dtype: str = "bfloat16",
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        perceiver_heads: int = 8,
        n_temporal_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.25,
        head_hidden: int = 2048,
    ) -> None:
        super().__init__()

        _DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        bb_dtype = _DTYPES.get(backbone_dtype, torch.bfloat16)
        self._bb_dtype = bb_dtype

        raw = AutoModel.from_pretrained(
            backbone, trust_remote_code=True, torch_dtype=bb_dtype,
        )
        self.encoder = _unwrap_vision_encoder(raw)
        # Drop reference to any text/predictor head we discarded so it can be GC'd
        if self.encoder is not raw:
            del raw

        # Freeze the entire backbone — only LoRA adapters (if any) and the head train.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # LoRA injection: try BEiT-style `qkv`, then OPT/LLaMA-style `q_proj`/`v_proj`,
        # then BERT-style `query`/`value`. Stop at the first naming that matches.
        self._use_lora = False
        if use_lora and lora_rank > 0:
            for targets in (["qkv"], ["q_proj", "v_proj"], ["query", "value"]):
                n_lora = apply_lora(self.encoder, targets, rank=lora_rank, alpha=lora_alpha)
                if n_lora > 0:
                    self._use_lora = True
                    self._lora_targets = targets
                    break
            if not self._use_lora:
                print("InternVideo2Pairs: WARNING — no LoRA target matched; backbone "
                      "stays fully frozen. Head capacity must carry the task.")

        D = _infer_hidden_size(self.encoder)
        self.hidden_size = D

        # Perceiver pool: 4 learnable per-frame queries cross-attend to all M backbone
        # tokens. Output is a clean (B, 4, D) frame summary regardless of how
        # InternVideo2 organises its tokens internally.
        self.perceiver_queries = nn.Parameter(torch.zeros(1, _N_FRAMES, D))
        nn.init.trunc_normal_(self.perceiver_queries, std=0.02)
        self.perceiver_q_norm = nn.LayerNorm(D)
        self.perceiver_kv_norm = nn.LayerNorm(D)
        self.perceiver_attn = nn.MultiheadAttention(
            D, perceiver_heads, dropout=dropout, batch_first=True,
        )
        self.perceiver_out_norm = nn.LayerNorm(D)
        self.perceiver_ff = nn.Sequential(
            nn.Linear(D, 4 * D), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * D, D),
        )

        # Spatial cross-attention pairs — j's projected query attends to the FULL
        # backbone feature sequence (which carries spatial *and* temporal info),
        # then subtracts f_i baseline. Same intuition as VideoMAECrossAttnPairs but
        # generalised to backbones that don't expose clean (T, N, D) layouts.
        self.cross_q_proj = nn.Linear(D, D, bias=False)
        self.cross_k_proj = nn.Linear(D, D, bias=False)
        self.cross_v_proj = nn.Linear(D, D, bias=False)
        self.pair_norms = nn.ModuleList([nn.LayerNorm(D) for _ in _PAIRS])

        # Temporal CLS path over the 4 frame summaries
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.temporal_pe = nn.Embedding(_N_FRAMES + 1, D)
        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(D, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(D)

        # Classification head: [CLS, 6 pair deltas] = 7 × D → MLP → num_classes
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear((1 + len(_PAIRS)) * D, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_n   = count_lora_params(self.encoder) if self._use_lora else 0
        print(
            f"InternVideo2Pairs: backbone={backbone}  hidden={D}  "
            f"{n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
            f"(lora={self._use_lora}, lora_params={lora_n/1e6:.2f}M)"
        )

    # ── training / param routing ──────────────────────────────────────────────

    def train(self, mode: bool = True):
        super().train(mode)
        # Backbone always in eval mode (frozen body / LoRA-only training).
        self.encoder.eval()
        if self._use_lora:
            from models.lora_utils import LoRALinear
            for m in self.encoder.modules():
                if isinstance(m, LoRALinear):
                    m.train(mode)
        return self

    def head_parameters(self):
        """Everything outside the backbone (incl. perceiver, pairs, temporal, head)."""
        return [p for n, p in self.named_parameters()
                if not n.startswith("encoder.") and p.requires_grad]

    def backbone_parameters(self):
        """Non-LoRA backbone params that require grad (empty when fully frozen)."""
        if self._use_lora:
            return []
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def lora_parameters(self):
        if not self._use_lora:
            return []
        return get_lora_parameters(self.encoder)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=4, C, H, W) — already ImageNet-normalised by the dataloader
        B = x.shape[0]
        dtype = x.dtype

        # 4 → 8 frames via linear interpolation (matches InternVideo2's expected clip)
        x = _linear_upsample(x, _TARGET_FRAMES)

        # Backbone forward. The frozen body runs in its loaded dtype (bf16) for
        # speed; LoRA adapters wrap individual Linear layers so autograd still
        # flows through them. No no_grad() wrapper here because we need grads
        # through LoRA — but if no LoRA was inserted, autograd produces no leaf
        # gradients for the encoder anyway (all encoder params have requires_grad=False).
        enc_dtype = next(self.encoder.parameters()).dtype
        out = self.encoder(x.to(enc_dtype))
        feats = _extract_features(out).to(dtype)           # (B, M, D)
        M = feats.shape[1]

        # Perceiver pool: 4 learnable frame queries cross-attend to all M tokens,
        # with pre-LayerNorm on Q and K/V (standard Pre-LN block), residual on Q,
        # then FF. This gives clean per-frame summaries regardless of how
        # InternVideo2 organises its M tokens.
        Q = self.perceiver_queries.expand(B, -1, -1)        # (B, 4, D)
        Qn = self.perceiver_q_norm(Q)
        Kn = self.perceiver_kv_norm(feats)
        attn_out, _ = self.perceiver_attn(Qn, Kn, Kn)        # (B, 4, D)
        Q = Q + attn_out
        Q = Q + self.perceiver_ff(self.perceiver_out_norm(Q))
        frame_feats = Q                                       # (B, 4, D)

        # Spatial-style cross-attention pairs: q=f_j attends to ALL backbone tokens
        # (carrying spatial+temporal info from InternVideo2), result minus f_i.
        D = feats.shape[-1]
        scale = math.sqrt(D)
        k_all = self.cross_k_proj(feats)                     # (B, M, D)
        v_all = self.cross_v_proj(feats)                     # (B, M, D)
        pair_feats = []
        for idx, (i, j) in enumerate(_PAIRS):
            q_j = self.cross_q_proj(frame_feats[:, j]).unsqueeze(1)         # (B, 1, D)
            cross_w = torch.softmax(
                torch.bmm(q_j, k_all.transpose(1, 2)) / scale, dim=-1)       # (B, 1, M)
            cross_feat = torch.bmm(cross_w, v_all).squeeze(1)                # (B, D)
            delta = cross_feat - frame_feats[:, i]                            # (B, D)
            pair_feats.append(self.pair_norms[idx](delta))

        # Temporal CLS path with positional encoding
        device = x.device
        pos_ids = torch.arange(1, _N_FRAMES + 1, device=device)
        pe = self.temporal_pe(pos_ids).unsqueeze(0)                           # (1, 4, D)
        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls = self.cls_token.expand(B, -1, -1) + cls_pe                       # (B, 1, D)
        seq = torch.cat([cls, frame_feats + pe], dim=1)                       # (B, 5, D)
        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])                               # (B, D)

        feat = torch.cat([cls_out] + pair_feats, dim=1)                       # (B, 7D)
        return self.head(feat)

"""VideoMAEPairwiseRanking — VideoMAE-SSv2 + Pairwise Temporal Ranking Branch.

Motivation
----------
espagne (2frozen + LLRD + MLP on mean-pool) achieves 49.71% and beats all
complex heads tested so far — because VideoMAE-SSv2 already encodes temporal
ordering in its tokens; a simple head lets the backbone exploit this freely.

The missing signal: DIRECTION of change between specific frame pairs.
VideoMAE's bidirectional attention knows "frame_i and frame_j differ" but
does not systematically encode "frame_i comes before frame_j."

Architecture
------------
Two parallel branches share the same backbone (2frozen, LLRD 0.80):

1. Content branch (= espagne):
   mean_pool(1568 tokens) -> MLP -> logits_content  ∈ R^33

2. Pairwise ranking branch:
   For each of the 6 frame pairs (i,j) with i < j:
     - Spatial cross-attention: query = f_j (pooled), keys/values = tokens_i (196)
       -> cross_feat = spatially-weighted combination of frame_i tokens
     - delta_ij = LayerNorm(cross_feat - f_i)   [asymmetric: j attending i]
     - s_ij = sigmoid(MLP(delta_ij))  ∈ [0, 1]  [pairwise preference score]
   scores = [s_01, s_02, s_03, s_12, s_13, s_23]  ∈ R^6
   logits_rank = scores @ class_pair_weights^T     ∈ R^33

   class_pair_weights (33, 6) is a *learnable* matrix.
   Each row k is a "pairwise preference template" for class k:
   it discovers which pattern of 6 scores corresponds to which class.
   No external knowledge of class→permutation mapping is needed.

Final logits = logits_content + logits_rank

Key properties
--------------
- Performance floor = espagne (49.71%): if ranking_branch contributes
  nothing, model degenerates to content branch only.
- class_pair_weights provides implicit regularisation: classes with similar
  ordering patterns share similar weight rows -> generalises better.
- The 6-score bottleneck forces the model to capture ordering in a compact,
  structured representation (vs 768-dim unstructured).
- Computationally light: 6 pair scorers (D->64->1) + one 33x6 matrix.

Training recipe: identical to espagne — 2frozen + LLRD 0.80.

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformers import VideoMAEModel

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_VM_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16
_TUBELET_SIZE  = 2
_PATCH_SIZE    = 16
_IMG_SIZE      = 224
_N_TEMPORAL    = _TARGET_FRAMES // _TUBELET_SIZE   # 8
_N_SPATIAL     = (_IMG_SIZE // _PATCH_SIZE) ** 2   # 196
_N_FRAMES      = 4
_ORIG_GROUPS   = (0, 2, 5, 7)
_PAIRS         = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


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


class VideoMAEPairwiseRanking(nn.Module):
    """VideoMAE + dual-branch: content (espagne-style) + pairwise ranking."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_frozen_blocks: int = 2,
        dropout: float = 0.25,
        head_hidden: int = 1024,
        rank_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.num_frozen_blocks = num_frozen_blocks
        n_pairs = len(_PAIRS)

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.encoder.encoder.layer):
            for p in block.parameters():
                p.requires_grad = (i >= num_frozen_blocks)

        hidden = self.encoder.config.hidden_size  # 768

        # ── Content branch (identical to espagne) ──────────────────────────
        self.content_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.content_head[-1].weight, std=0.01)
        nn.init.zeros_(self.content_head[-1].bias)

        # ── Pairwise ranking branch ─────────────────────────────────────────
        # Shared spatial attention query to pool per-frame features
        self.spatial_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        # Cross-attention projections (shared across all pairs)
        self.cross_q_proj = nn.Linear(hidden, hidden, bias=False)
        self.cross_k_proj = nn.Linear(hidden, hidden, bias=False)
        self.cross_v_proj = nn.Linear(hidden, hidden, bias=False)

        # Per-pair LayerNorm for the delta features
        self.pair_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in _PAIRS])

        # Shared lightweight scorer: delta -> scalar preference score
        # All 6 pairs share weights (same temporal ordering principle applies)
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden, rank_hidden),
            nn.GELU(),
            nn.Linear(rank_hidden, 1),
        )

        # Learnable class-pair affinity matrix (33, 6):
        # row k = pairwise preference template for class k
        self.class_pair_weights = nn.Parameter(
            torch.zeros(num_classes, n_pairs)
        )
        nn.init.trunc_normal_(self.class_pair_weights, std=0.01)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VideoMAEPairwiseRanking: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        if self.num_frozen_blocks < len(self.encoder.encoder.layer):
            for block in self.encoder.encoder.layer[self.num_frozen_blocks:]:
                block.train(mode)
        return self

    def head_parameters(self):
        return (
            list(self.content_head.parameters())
            + [self.spatial_query,
               self.cross_q_proj.weight,
               self.cross_k_proj.weight,
               self.cross_v_proj.weight,
               self.class_pair_weights]
            + list(self.pair_norms.parameters())
            + list(self.pair_scorer.parameters())
        )

    def backbone_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.85):
        groups = [{"params": self.head_parameters(), "lr": head_lr}]
        n_layers = len(self.encoder.encoder.layer)
        for depth, idx in enumerate(range(n_layers - 1, self.num_frozen_blocks - 1, -1)):
            groups.append({
                "params": list(self.encoder.encoder.layer[idx].parameters()),
                "lr": backbone_lr * (llrd ** depth),
            })
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet -> VideoMAE normalisation
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        x = _linear_upsample(x, _TARGET_FRAMES)

        tokens_all = self.encoder(pixel_values=x).last_hidden_state.to(dtype)  # (B,1568,D)
        D = tokens_all.shape[-1]

        # ── Content branch ─────────────────────────────────────────────────
        content_feat = tokens_all.mean(1)              # (B, D)  — espagne-style
        logits_content = self.content_head(content_feat)  # (B, 33)

        # ── Pairwise ranking branch ─────────────────────────────────────────
        tokens_3d = tokens_all.view(B, _N_TEMPORAL, _N_SPATIAL, D)
        frame_tokens = tokens_3d[:, list(_ORIG_GROUPS)]  # (B, 4, 196, D)

        # Spatial attention pool: single learnable query per frame
        ft_flat = frame_tokens.reshape(B * _N_FRAMES, _N_SPATIAL, D)
        q_pool = self.spatial_query.expand(B * _N_FRAMES, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q_pool, ft_flat.transpose(1, 2)) / math.sqrt(D), dim=-1)
        frame_feats = torch.bmm(attn_w, ft_flat).squeeze(1).reshape(B, _N_FRAMES, D)
        # frame_feats: (B, 4, D)

        scale = math.sqrt(D)
        pair_scores = []
        for k, (i, j) in enumerate(_PAIRS):
            # Cross-attention: frame_j (query) attends to 196 tokens of frame_i
            q_j = self.cross_q_proj(frame_feats[:, j]).unsqueeze(1)  # (B, 1, D)
            k_i = self.cross_k_proj(frame_tokens[:, i])              # (B, 196, D)
            v_i = self.cross_v_proj(frame_tokens[:, i])              # (B, 196, D)
            cross_w = torch.softmax(
                torch.bmm(q_j, k_i.transpose(1, 2)) / scale, dim=-1)  # (B, 1, 196)
            cross_feat = torch.bmm(cross_w, v_i).squeeze(1)            # (B, D)

            delta = self.pair_norms[k](cross_feat - frame_feats[:, i])  # (B, D)
            s = self.pair_scorer(delta)                                  # (B, 1)
            pair_scores.append(s)

        scores = torch.cat(pair_scores, dim=1)                           # (B, 6)

        # Class ranking logits via affinity matrix
        logits_rank = scores @ self.class_pair_weights.t()               # (B, 33)

        return logits_content + logits_rank

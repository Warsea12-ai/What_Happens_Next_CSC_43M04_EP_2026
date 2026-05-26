"""TimeSformerHR — TimeSformer-HR-finetuned-SSv2 + tête `best_combo_v2`.

Pourquoi ce modèle vaut le coup
--------------------------------
- Préentraîné sur **Something-Something v2** : domaine EXACT de "what happens next"
  (action recognition temporellement orientée). Aucun autre backbone testé dans ce
  repo n'a cette propriété (VideoMAE-base est aussi SSv2 mais a déjà été essoré).
- **Divided space-time attention** : à chaque bloc, alterne attention temporelle
  (parmi les T positions, par patch spatial fixe) puis attention spatiale (parmi
  les N patches, par frame fixe). Biais inductif distinct de VideoMAE/V-JEPA2
  (joint attention) et de DINOv3 (pure spatial).
- **Variant HR (high-resolution)** : 16 frames à 448×448 (vs 224×224 pour
  videomae-base) → 32×32=1024 patches par frame (vs 14×14=196), 4× plus
  d'info visuelle.
- 430M params, fit largement 24GB en bf16 + best_combo_v2 (Prodigy).

Architecture (sortie)
---------------------
1. Upsample 4 → 16 frames (interp_mode configurable, défaut "aligned").
2. Renormalise ImageNet → CLIP-style si nécessaire (TimeSformer attend
   mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225]).
3. Resize 224×224 → 448×448 par interpolation bilinéaire.
4. TimeSformer body → tokens (B, 1+T*N, 768).
5. Drop CLS, extract per-frame summary via attention pool (1 learnable query
   par frame, applied to spatial tokens of that frame).
6. Cross-attention pairs head (winning pattern de cross_attn_pairs).
7. Head MLP → 33 classes.

Input  : (B, T=4, C=3, 224, 224)  ImageNet-normalised
Output : (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimesformerModel

from temporal_interp import interp_temporal


_IN_MEAN  = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD   = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_TSF_MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 3, 1, 1)
_TSF_STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16
# image_size + patch_size are read from the loaded model config now, so this
# class works for both timesformer-hr (448×448 → 784 patches) and
# timesformer-base (224×224 → 196 patches).
_N_FRAMES      = 4
_PAIRS         = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


class _TemporalBlock(nn.Module):
    """Pre-LN self-attention block (identique aux versions cross_attn_pairs/etc.)."""

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


class TimeSformerHR(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "facebook/timesformer-hr-finetuned-ssv2",
        num_frozen_blocks: int = 0,
        n_temporal_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.25,
        head_hidden: int = 2048,
        interp_mode: str = "aligned",
    ) -> None:
        super().__init__()
        if interp_mode not in ("aligned", "centered", "repeat"):
            raise ValueError(f"interp_mode must be aligned|centered|repeat, got {interp_mode!r}")
        self.interp_mode = interp_mode
        self.num_frozen_blocks = num_frozen_blocks

        # Body uniquement (pas la head 174-classes pré-entraînée SSv2)
        self.encoder = TimesformerModel.from_pretrained(backbone, use_safetensors=True)

        # Adapte spatial layout au backbone (base=224/16=14² ; HR=448/16=28²)
        self.image_size = int(getattr(self.encoder.config, "image_size", 224))
        self.patch_size = int(getattr(self.encoder.config, "patch_size", 16))
        self.n_spatial  = (self.image_size // self.patch_size) ** 2
        print(f"  TimeSformer config: image_size={self.image_size} "
              f"patch_size={self.patch_size} n_spatial={self.n_spatial}")

        # Freeze patch embeddings, num_classes head non chargée car on prend juste le body.
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.encoder.encoder.layer):
            trainable = i >= num_frozen_blocks
            for p in block.parameters():
                p.requires_grad = trainable

        hidden = self.encoder.config.hidden_size  # 768 pour timesformer-hr base
        self.hidden_size = hidden

        # Spatial attention pool (1 query, applied to each frame's spatial tokens)
        self.spatial_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        # Cross-attention pairs : f_j attends to spatial tokens of frame_i
        self.cross_q_proj = nn.Linear(hidden, hidden, bias=False)
        self.cross_k_proj = nn.Linear(hidden, hidden, bias=False)
        self.cross_v_proj = nn.Linear(hidden, hidden, bias=False)
        self.pair_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in _PAIRS])

        # Temporal CLS path
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.temporal_pe = nn.Embedding(_N_FRAMES + 1, hidden)
        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(hidden, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(hidden)

        # Head : [CLS, 6 pair deltas] = 7 × hidden → MLP → num_classes
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear((1 + len(_PAIRS)) * hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TimeSformerHR: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable "
              f"(num_frozen_blocks={num_frozen_blocks}, interp={interp_mode}, hidden={hidden})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.embeddings.eval()
        if self.num_frozen_blocks < len(self.encoder.encoder.layer):
            for block in self.encoder.encoder.layer[self.num_frozen_blocks:]:
                block.train(mode)
        return self

    def head_parameters(self):
        return (
            [self.spatial_query, self.cls_token,
             self.cross_q_proj.weight, self.cross_k_proj.weight, self.cross_v_proj.weight]
            + list(self.temporal_pe.parameters())
            + list(self.temporal_blocks.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.pair_norms.parameters())
            + list(self.head.parameters())
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

        # Renorm ImageNet → TimeSformer norms
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_tf = _TSF_MEAN.to(device, dtype)
        s_tf = _TSF_STD.to(device, dtype)
        x = (x * s_in + m_in - m_tf) / s_tf

        # Upsample 4 → 16 frames (interp_mode configurable)
        x = interp_temporal(x, target_T=_TARGET_FRAMES, mode=self.interp_mode)
        # x : (B, 16, C, H, W)

        # Resize to model's native resolution (224 for base, 448 for HR)
        target_size = self.image_size
        B_, T_, C_, H_, W_ = x.shape
        if H_ != target_size or W_ != target_size:
            x = F.interpolate(
                x.view(B_ * T_, C_, H_, W_),
                size=(target_size, target_size),
                mode="bilinear", align_corners=False,
            ).view(B_, T_, C_, target_size, target_size)

        # TimeSformer attend (B, T, C, H, W) → output last_hidden_state (B, 1+T*N, hidden)
        out = self.encoder(pixel_values=x)
        tokens = out.last_hidden_state.to(dtype)
        # Drop CLS
        tokens = tokens[:, 1:]                    # (B, T*N, hidden)
        D = tokens.shape[-1]
        tokens = tokens.view(B, _TARGET_FRAMES, self.n_spatial, D)

        # Pick 4 "frame anchor" positions out of 16 — same positions that interp_temporal
        # places the input frames at (aligned mode: [0,5,10,15]).
        # For other modes we keep the simple [0,5,10,15] mapping as a robust default.
        anchor_idx = [0, 5, 10, 15]
        frame_tokens = tokens[:, anchor_idx]      # (B, 4, n_spatial, hidden)

        # Spatial attention pool : 1 query per frame
        ft_flat = frame_tokens.reshape(B * _N_FRAMES, self.n_spatial, D)
        q = self.spatial_query.expand(B * _N_FRAMES, 1, D)
        attn_w = torch.softmax(
            torch.bmm(q, ft_flat.transpose(1, 2)) / math.sqrt(D), dim=-1)
        frame_feats = torch.bmm(attn_w, ft_flat).squeeze(1).reshape(B, _N_FRAMES, D)

        # Cross-attention pairs (j attends to spatial tokens of frame i, minus f_i baseline)
        scale = math.sqrt(D)
        pair_feats = []
        for k, (i, j) in enumerate(_PAIRS):
            q_j = self.cross_q_proj(frame_feats[:, j]).unsqueeze(1)
            k_i = self.cross_k_proj(frame_tokens[:, i])
            v_i = self.cross_v_proj(frame_tokens[:, i])
            cross_w = torch.softmax(
                torch.bmm(q_j, k_i.transpose(1, 2)) / scale, dim=-1)
            cross_feat = torch.bmm(cross_w, v_i).squeeze(1)
            delta = cross_feat - frame_feats[:, i]
            pair_feats.append(self.pair_norms[k](delta))

        # Temporal CLS path
        pos_ids = torch.arange(1, _N_FRAMES + 1, device=device)
        pe = self.temporal_pe(pos_ids).unsqueeze(0)
        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls = self.cls_token.expand(B, -1, -1) + cls_pe
        seq = torch.cat([cls, frame_feats + pe], dim=1)
        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])

        feat = torch.cat([cls_out] + pair_feats, dim=1)
        return self.head(feat)

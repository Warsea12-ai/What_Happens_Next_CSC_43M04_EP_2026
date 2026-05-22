"""VideoMAEMultiScaleLoRATemporal — VideoMAE-base SSv2 + multi-scale features + LoRA + temporal head.

Hybride de deux architectures Track B qui ont chacune un point fort distinct :

- `videomae_temporal_head` (top actuel 0.428% top1) :
    spatial attention pool + 3-layer temporal transformer + delta features + LoRA.
    Capte bien le « comment la scène se transforme » sur le dernier hidden state.

- `videomae_multiscale` :
    fusion multi-scales (blocs intermédiaires) car les couches early-mid encodent
    le **mouvement local** (direction, vitesse), info atténuée par la sémantique
    des dernières couches.

Ce modèle combine les deux : on extrait 3 scales (blocs 5, 8, 11) du même backbone
VideoMAE-base SSv2, on fait une attention spatiale par scale, on fusionne avant
le temporal transformer, puis on ajoute les deltas extraits du **scale final**
(plus discriminatifs sémantiquement) à la classification.

Architecture :
  VideoMAE-base SSv2 (LoRA Q,V, all blocks frozen + adapted)
    → hidden_states[6], [9], [12]                           # blocs 5, 8, 11
  pour chaque scale :
    → reshape (B, 8 temporal groups, 196 spatial, D)
    → spatial-attn pool per group (query partagé entre scales)  → (B, 8, D)
  fusion : concat scales sur D → (B, 8, 3D) → Linear → (B, 8, D)
  CLS + 8 tokens + positional encoding → 3-layer Transformer
    → cls_out (B, D)
  deltas du scale FINAL aux groupes (0, 2, 5, 7) :
    Δfull = f7-f0, Δfirst = f2-f0, Δlast = f7-f5
  head MLP sur cat([cls_out, Δfull, Δfirst, Δlast]) → 33

Input : (B, T=4, C, H, W)  ImageNet-normalised
Output: (B, num_classes)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformers import VideoMAEModel

from models.lora_utils import apply_lora, get_lora_parameters

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

# Indices dans le tuple hidden_states de VideoMAE-base (12 blocs) :
#   embedding=0, block_k=k+1  →  block 5/8/11 = indices 6/9/12
_SCALE_INDICES = (6, 9, 12)

# Groupes temporels qui correspondent aux frames originales 0..3 après 4→16 interp.
_ORIG_GROUPS = (0, 2, 5, 7)


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


class VideoMAEMultiScaleLoRATemporal(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        n_temporal_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.3,
        head_hidden: int = 1024,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        scale_indices: tuple[int, ...] = _SCALE_INDICES,
    ) -> None:
        super().__init__()
        assert lora_rank > 0, "ce modèle suppose LoRA activée (rank > 0)"
        self.scale_indices = tuple(scale_indices)

        self.encoder = VideoMAEModel.from_pretrained(backbone, use_safetensors=True)

        # Freeze tout le backbone (LoRA s'en chargera)
        for p in self.encoder.parameters():
            p.requires_grad = False
        apply_lora(self.encoder, ["query", "value"], rank=lora_rank, alpha=lora_alpha)

        hidden = self.encoder.config.hidden_size   # 768 pour base
        D = hidden

        # ── Spatial attention pooling (partagé entre scales) ──────────────────
        # 1 query, applique par (scale, temporal group). L'attention diffère par
        # scale et par groupe (contexte spatial différent).
        self.spatial_query = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.spatial_query, std=0.02)

        # ── Fusion multi-scale : concat (3·D) → Linear → D ────────────────────
        self.scale_norms = nn.ModuleList([nn.LayerNorm(D) for _ in self.scale_indices])
        self.fusion = nn.Sequential(
            nn.LayerNorm(len(self.scale_indices) * D),
            nn.Linear(len(self.scale_indices) * D, D),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Temporal transformer ──────────────────────────────────────────────
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Positions : 0=CLS, 1..N_TEMPORAL=groupes 0..7
        self.temporal_pe = nn.Embedding(_N_TEMPORAL + 1, D)
        self.temporal_blocks = nn.ModuleList([
            _TemporalBlock(D, n_heads, dropout) for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(D)

        # ── Delta feature norms (sur le scale final, plus discriminatif) ──────
        self.norm_delta_full  = nn.LayerNorm(D)
        self.norm_delta_first = nn.LayerNorm(D)
        self.norm_delta_last  = nn.LayerNorm(D)

        # ── Head : [CLS + 3 deltas] = 4·D → num_classes ───────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * D, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VideoMAEMultiScaleLoRATemporal: {n_frozen/1e6:.0f}M frozen, "
              f"{n_train/1e6:.1f}M trainable (lora rank={lora_rank}, scales={self.scale_indices})")

    def train(self, mode: bool = True):
        super().train(mode)
        # Encoder en eval pour stats LayerNorm/Dropout stables ; LoRA seul s'entraîne.
        self.encoder.eval()
        from models.lora_utils import LoRALinear
        for m in self.encoder.modules():
            if isinstance(m, LoRALinear):
                m.train(mode)
        return self

    # ── parameter groups (compat train_trackB) ────────────────────────────────

    def _head_params(self):
        return (
            [self.spatial_query, self.cls_token]
            + list(self.scale_norms.parameters())
            + list(self.fusion.parameters())
            + list(self.temporal_pe.parameters())
            + list(self.temporal_blocks.parameters())
            + list(self.temporal_norm.parameters())
            + list(self.norm_delta_full.parameters())
            + list(self.norm_delta_first.parameters())
            + list(self.norm_delta_last.parameters())
            + list(self.head.parameters())
        )

    def head_parameters(self):
        return self._head_params()

    def backbone_parameters(self):
        return []  # LoRA-only — pas de blocs unfrozen

    def lora_parameters(self):
        return get_lora_parameters(self.encoder)

    def layerwise_lr_groups(self, head_lr: float, backbone_lr: float, llrd: float = 0.85):
        # Pas de LLRD sensé sur LoRA (les adaptateurs ne suivent pas la hiérarchie habituelle).
        return [
            {"params": self._head_params(),   "lr": head_lr},
            {"params": self.lora_parameters(), "lr": backbone_lr},
        ]

    # ── forward ───────────────────────────────────────────────────────────────

    def _spatial_pool(self, tokens_BNT_SD: torch.Tensor) -> torch.Tensor:
        """tokens_BNT_SD: (B*N_TEMPORAL, N_SPATIAL, D) → (B*N_TEMPORAL, D)."""
        BNT, S, D = tokens_BNT_SD.shape
        q = self.spatial_query.expand(BNT, 1, D)
        attn = torch.softmax(
            torch.bmm(q, tokens_BNT_SD.transpose(1, 2)) / math.sqrt(D),
            dim=-1,
        )
        return torch.bmm(attn, tokens_BNT_SD).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # ImageNet → VideoMAE [0.5, 0.5, 0.5] normalisation
        m_in = _IN_MEAN.to(device, dtype)
        s_in = _IN_STD.to(device, dtype)
        m_vm = _VM_MEAN.to(device, dtype)
        s_vm = _VM_STD.to(device, dtype)
        x = (x * s_in + m_in - m_vm) / s_vm

        # 4 → 16 frames via interpolation linéaire
        x = _linear_upsample(x, _TARGET_FRAMES)

        # Encode AVEC hidden states intermédiaires (3 scales)
        out = self.encoder(pixel_values=x, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple : embedding + 12 blocs

        # Pour chaque scale demandé : pool spatial par groupe temporel → (B, 8, D)
        scale_feats: list[torch.Tensor] = []
        for idx, norm in zip(self.scale_indices, self.scale_norms):
            tokens = hidden_states[idx].to(dtype)                                # (B, 1568, D)
            D = tokens.shape[-1]
            tokens = norm(tokens)
            tokens = tokens.view(B, _N_TEMPORAL, _N_SPATIAL, D)                  # (B, 8, 196, D)
            tokens_flat = tokens.reshape(B * _N_TEMPORAL, _N_SPATIAL, D)
            pooled = self._spatial_pool(tokens_flat).reshape(B, _N_TEMPORAL, D)  # (B, 8, D)
            scale_feats.append(pooled)

        # Fusion multi-scale : concat sur D, puis Linear → (B, 8, D)
        fused = torch.cat(scale_feats, dim=-1)            # (B, 8, 3D)
        tfeats = self.fusion(fused)                       # (B, 8, D)

        # Delta features depuis le SCALE FINAL (plus sémantique)
        final = scale_feats[-1]
        g0, g1, g2, g3 = _ORIG_GROUPS
        delta_full  = final[:, g3] - final[:, g0]
        delta_first = final[:, g1] - final[:, g0]
        delta_last  = final[:, g3] - final[:, g2]

        # Temporal transformer avec [CLS] et PE
        pos_ids = torch.arange(1, _N_TEMPORAL + 1, device=device)
        pe = self.temporal_pe(pos_ids).unsqueeze(0)                # (1, 8, D)
        tfeats = tfeats + pe

        cls_pe = self.temporal_pe(torch.zeros(1, dtype=torch.long, device=device))
        cls = self.cls_token.expand(B, -1, -1) + cls_pe            # (B, 1, D)
        seq = torch.cat([cls, tfeats], dim=1)                      # (B, 9, D)

        for block in self.temporal_blocks:
            seq = block(seq)
        cls_out = self.temporal_norm(seq[:, 0])                    # (B, D)

        # Head
        feat = torch.cat([
            cls_out,
            self.norm_delta_full(delta_full),
            self.norm_delta_first(delta_first),
            self.norm_delta_last(delta_last),
        ], dim=1)
        return self.head(feat)

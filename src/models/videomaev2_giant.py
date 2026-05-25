"""VideoMAEv2Giant — ViT-g (1B params), SSv2-finetuned (OpenGVLab/VideoMAEv2).

Source : https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/mae-g
        vit_g_hybrid_pt_1200e_ssv2_ft.pth (2.02 GB)

Architecture : VisionTransformer du modeling_finetune.py upstream
  patch=14, embed=1408, depth=40, heads=16, mlp_ratio=48/11
  all_frames=16 (8 temporal tokens après tubelet_size=2)
  num_classes=174 dans le checkpoint → on remplace par 33

Reporté : 77.0% top-1 sur SSv2 (vs 70.6% pour VideoMAE V1 base).
Comme notre dataset est un sous-ensemble curaté de SSv2, le backbone a
déjà vu nos données de train pendant son pretrain SSv2 → transfert direct.

Trois variantes selon l'arg `variant` :
  frozen     : tout le backbone gelé, seule la nouvelle tête est entraînée
  attn_pool  : 32/40 blocs gelés + AttentionPool + head linéaire
  lora       : LoRA rank=16 sur Q/V de tous les 40 blocs
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from models.videomaev2_modeling import vit_giant_patch14_224

# Constants
_TARGET_FRAMES = 16        # vit_giant attend 16 frames RGB
_TUBELET_SIZE  = 2          # tokens temporels : 16/2 = 8
_PATCH_SIZE    = 14
_IMG_SIZE      = 224
_EMBED_DIM     = 1408
_DEPTH         = 40

# Normalisation : VideoMAE utilise ImageNet mean/std (différent de V1 base qui était 0.5)
_VM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Upsample temporel par interpolation linéaire pour atteindre target_T frames."""
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


class _AttentionPool(nn.Module):
    """Pooling par requête apprise — sélectionne les tokens utiles pour l'action."""
    def __init__(self, d: int, n_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


def _apply_lora_to_attention(model: nn.Module, rank: int = 16, alpha: float = 32.0) -> None:
    """Applique LoRA sur les projections QKV de tous les blocs ViT.

    Le modèle upstream utilise nn.Linear `qkv` (concat Q,K,V). On wrap chacune avec
    une LoRALinear qui ajoute le delta low-rank. Approche minimaliste — pas peft."""
    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            in_dim, out_dim = base.in_features, base.out_features
            self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * (1.0 / rank))
            self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
            self.scale = alpha / rank
        def forward(self, x):
            return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    for block in model.blocks:
        if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)


class VideoMAEv2Giant(nn.Module):
    """ViT-g VideoMAE V2 pretrained SSv2 → adapted to num_classes (default 33).

    Args:
        variant : "frozen" | "attn_pool" | "lora"
        num_classes : 33 pour SSv2 curaté
        checkpoint_path : path absolu vers vit_g_hybrid_pt_1200e_ssv2_ft.pth
        num_frozen_blocks : pour "attn_pool", nb de blocs gelés (32 par défaut)
        lora_rank, lora_alpha : pour "lora"
        head_hidden : taille du MLP head intermédiaire (0 = linear direct)
        dropout : sur le head
    """

    _VARIANTS = ("frozen", "attn_pool", "lora")

    def __init__(
        self,
        num_classes: int = 33,
        variant: str = "frozen",
        checkpoint_path: str = "/Data/What_Happens_Next_CSC_43M04_EP_2026/checkpoints/videomaev2_vitg_ssv2_ft.pth",
        num_frozen_blocks: int = 32,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        head_hidden: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if variant not in self._VARIANTS:
            raise ValueError(f"variant doit être dans {self._VARIANTS}, reçu {variant!r}")
        self.variant = variant

        # 1. Architecture (sans poids)
        self.backbone = vit_giant_patch14_224(
            pretrained=False,
            num_classes=174,      # match du checkpoint
            all_frames=_TARGET_FRAMES,
            tubelet_size=_TUBELET_SIZE,
            use_mean_pooling=True,
            drop_path_rate=0.2,
        )

        # 2. Chargement des poids SSv2-finetuned. Le ckpt fait 2 GB et n'est pas
        # forcément déjà présent sur le host (chaque host a son propre /Data,
        # pas de NFS partagé). On utilise hf_hub_download qui gère un cache
        # par-host transparent et résume les downloads interrompus.
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            from huggingface_hub import hf_hub_download
            print(f"[VideoMAEv2Giant] ckpt absent ({ckpt_p}) — download depuis HF...",
                  flush=True)
            ckpt_p = Path(hf_hub_download(
                repo_id="OpenGVLab/VideoMAE2",
                filename="mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
            ))
            print(f"[VideoMAEv2Giant] download OK : {ckpt_p}", flush=True)
        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        sd = ckpt.get("module", ckpt.get("model", ckpt))
        msg = self.backbone.load_state_dict(sd, strict=False)
        n_missing  = len(msg.missing_keys)
        n_unexpect = len(msg.unexpected_keys)
        # Seuils larges : la tête head/fc_norm peut différer ; mais blocs/patch_embed/pos_embed
        # doivent matcher.
        critical_missing = [k for k in msg.missing_keys
                            if k.startswith(("blocks.", "patch_embed.", "pos_embed"))]
        if critical_missing:
            raise RuntimeError(
                f"Chargement VideoMAE V2 ViT-g raté — {len(critical_missing)} clés critiques "
                f"manquantes (ex : {critical_missing[:3]}). Le checkpoint est-il intact ?"
            )
        print(f"[VideoMAEv2Giant] chargé : {n_missing} missing, {n_unexpect} unexpected "
              f"(têtes/normes attendues si != 174 classes).", flush=True)

        # 3. Variante : gel + tête
        hidden = _EMBED_DIM  # 1408
        # On supprime la tête 174-classes upstream pour récupérer les features
        self.backbone.head = nn.Identity()

        if variant == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.pool = None  # backbone.forward_features → mean-pooled features
        elif variant == "attn_pool":
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Dégèle les derniers (depth - num_frozen_blocks) blocs
            n_train = max(0, _DEPTH - num_frozen_blocks)
            for blk in self.backbone.blocks[-n_train:]:
                for p in blk.parameters():
                    p.requires_grad = True
            self.pool = _AttentionPool(hidden, n_heads=8)
            # fc_norm aussi trainable pour la queue de réseau
            if self.backbone.fc_norm is not None:
                for p in self.backbone.fc_norm.parameters():
                    p.requires_grad = True
        elif variant == "lora":
            for p in self.backbone.parameters():
                p.requires_grad = False
            _apply_lora_to_attention(self.backbone, rank=lora_rank, alpha=lora_alpha)
            self.pool = None

        # 4. Tête de classification 33-classes (toujours entraînée)
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

        n_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[VideoMAEv2Giant] variant={variant} : {n_train_params/1e6:.1f}M trainable / "
              f"{n_total/1e6:.1f}M total ({(n_total-n_train_params)/1e6:.1f}M frozen)",
              flush=True)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) en [0,1] → (B, C, T, H, W) normalisé pour le backbone."""
        x = _linear_upsample(x, _TARGET_FRAMES)
        mean = _VM_MEAN.to(x.device, x.dtype)
        std  = _VM_STD.to(x.device, x.dtype)
        x = (x - mean) / std
        # (B, T, C, H, W) → (B, C, T, H, W) attendu par PatchEmbed
        return x.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x)
        if self.variant == "attn_pool" and self.pool is not None:
            # On a besoin des tokens, pas du mean-pool — appel manuel
            x = self.backbone.patch_embed(x)
            x = x + self.backbone.pos_embed.type_as(x).to(x.device).clone().detach()
            x = self.backbone.pos_drop(x)
            for blk in self.backbone.blocks:
                x = blk(x)
            if self.backbone.fc_norm is not None:
                x = self.backbone.fc_norm(x)
            features = self.pool(x)
        else:
            # Mean-pool path standard
            features = self.backbone.forward_features(x)
        return self.head(features)

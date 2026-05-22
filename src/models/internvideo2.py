"""InternVideo2-Stage2-6B (OpenGVLab) — frozen video encoder + MLP head.

Risque
------
Le checkpoint est sur HF mais l'API n'est pas garantie via AutoModel standard.
Si crash au chargement, voir le log : on adapte (trust_remote_code, dépendance
flash_attn manquante, format d'entrée différent, etc.).

Stratégie
---------
1. Charger le backbone frozen en BF16 (~12 GB).
2. Forward sur la séquence vidéo complète (B, T, C, H, W).
3. Pool sur les tokens → embedding vidéo.
4. MLP head pour 33 classes.

Le wrapper est défensif : il tolère plusieurs conventions de sortie
(`last_hidden_state`, tenseur brut, tuple) et plusieurs noms de hidden_size.
"""
from __future__ import annotations

import sys
import types


# Le modeling file InternVideo2 sur HF déclare `import cv2` et `import flash_attn`
# au top-level. transformers.check_imports refuse de charger le fichier si l'un
# n'est pas importable. On stub-out les deux pour que check_imports passe ; les
# vraies fonctions doivent être installées séparément si le forward s'en sert.
class _LooseStub(types.ModuleType):
    """Module factice permissif : tout attribut renvoie un sous-stub.
    Tolère `from X import Y`, `import X.Y.Z`, etc. Si le modèle APPELLE
    réellement les fonctions, ça plantera proprement avec un TypeError."""
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _LooseStub(f"{self.__name__}.{name}")
        setattr(self, name, sub)
        sys.modules[sub.__name__] = sub
        return sub

    def __call__(self, *args, **kwargs):
        return None


for _stub_name in ("cv2", "flash_attn"):
    try:
        __import__(_stub_name)
    except ImportError:
        sys.modules[_stub_name] = _LooseStub(_stub_name)


import torch
import torch.nn as nn
from transformers import AutoModel


class InternVideo2(nn.Module):
    """InternVideo2-Stage2-6B frozen + MLP classification head."""

    def __init__(
        self,
        num_classes: int = 33,
        backbone: str = "OpenGVLab/InternVideo2-Stage2_6B",
        dropout: float = 0.3,
        head_hidden: int = 2048,
        backbone_dtype: str = "bfloat16",
    ) -> None:
        super().__init__()

        _DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        _bb_dtype = _DTYPES.get(backbone_dtype, torch.bfloat16)

        self.encoder = AutoModel.from_pretrained(
            backbone,
            trust_remote_code=True,
            torch_dtype=_bb_dtype,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # InternVideo2 expose hidden_size sous différents noms selon la version
        cfg = self.encoder.config
        hidden = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "embed_dim", None)
            or getattr(cfg, "vision_embed_dim", None)
            or getattr(cfg, "d_model", None)
        )
        if hidden is None:
            raise RuntimeError(
                f"Impossible de déterminer hidden_size pour {backbone} — "
                f"config attrs: {list(vars(cfg).keys())[:20]}"
            )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        n_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"InternVideo2: {n_frozen/1e6:.0f}M frozen, {n_train/1e6:.1f}M trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        return []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W), ImageNet-normalised
        dtype = x.dtype
        enc_dtype = next(self.encoder.parameters()).dtype

        with torch.no_grad():
            out = self.encoder(x.to(enc_dtype))

        # Extraction défensive du tenseur de features
        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif isinstance(out, torch.Tensor):
            feats = out
        elif isinstance(out, (tuple, list)):
            feats = out[0]
        else:
            raise RuntimeError(f"Type de sortie inattendu de l'encoder : {type(out)}")

        # Pool sur les tokens si tenseur 3D (B, N, H) → (B, H)
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        elif feats.dim() == 4:
            # (B, T, N, H) ou (B, C, T, N) — on aplatit tout sauf B et la dernière dim
            feats = feats.flatten(1, -2).mean(dim=1)

        return self.head(feats.to(dtype))

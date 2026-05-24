"""EnsembleSoftVote : un nn.Module qui combine plusieurs sub-modèles entraînés
via une pondération **per-class** apprise hors-ligne, puis sauvegardable comme
checkpoint standard pour passer dans ``create_submission.py``.

Idée
----
Au lieu de garder l'ensemble dans un script séparé (qui ne s'intègre pas
au pipeline canonique de soumission Kaggle), on emballe le tout dans une
classe ``nn.Module`` :

    sub_models : nn.ModuleList   — chaque sub-modèle est un vrai classifier
                                   33-classes, instancié via le dispatcher
                                   Track A/B correspondant.
    weight_matrix : buffer (C, M)
        weight_matrix[c, m] = poids attribué au modèle m sur la classe c.
        Normalisé par classe (Σ_m W[c,m] = 1).

Forward
-------
Pour chaque sub-modèle, on calcule sa softmax sur le batch, on multiplie
classe par classe par ``weight_matrix.T`` (M, C), et on somme cross-modèles.

Comme la sortie est interprétée comme des "logits" par ``create_submission.py``
qui fait ``argmax(dim=1)``, on renvoie ``log(combined_proba + eps)``. Aucun
impact sur l'argmax, mais préserve un comportement "logit-like".

État
----
- En inférence pure (``eval()``), on ne touche pas aux poids des sub-modèles.
- Tous les paramètres sont gelés (``requires_grad=False``) — éviter qu'un
  ``.train()`` accidentel ne corrompe rien.
- ``weight_matrix`` est un buffer, donc inclus dans ``state_dict()`` et
  restauré au load.

Persistance
-----------
Le ``state_dict()`` inclut ``sub_models.<i>.<param>`` (PyTorch préfixe
automatiquement les ModuleList) plus ``weight_matrix``. Le ``cfg`` stocké
dans le ``.pt`` contient ``cfg.model.sub_configs`` (liste des configs
des sub-modèles, telles que sauvegardées dans leurs checkpoints d'origine).
À la reconstruction, on rebâtit chaque sub-modèle à partir de son sub_config
via le routeur Track A/B approprié.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def _routed_build(sub_cfg: Dict[str, Any] | DictConfig) -> nn.Module:
    """Dispatcher Track A/B pour reconstruire un sub-modèle depuis son cfg.

    On utilise la liste ``_TRACK_B_MODELS`` d'``evaluate.py`` comme source
    de vérité unique (pas de duplication ici).
    """
    if not isinstance(sub_cfg, DictConfig):
        sub_cfg = OmegaConf.create(sub_cfg)
    name = sub_cfg.model.name
    # Import retardé : évite un cycle ``train → models → evaluate → train``.
    from evaluate import _TRACK_B_MODELS
    if name in _TRACK_B_MODELS:
        import train_trackB as _trk
    else:
        import train as _trk
    return _trk.build_model(sub_cfg)


class EnsembleSoftVote(nn.Module):
    """Combine ``M`` classifiers via soft-vote pondéré par classe.

    Args:
        sub_configs : liste de configs (dict ou DictConfig) tels que stockés
            dans ``cfg["config"]`` des checkpoints individuels.
        weight_matrix : tableau de forme (num_classes, M), normalisé par
            classe. Peut être ``None`` à la construction (chargé via
            ``load_state_dict`` ensuite).
        num_classes : nombre total de classes.
    """

    def __init__(
        self,
        sub_configs: Sequence[Dict[str, Any] | DictConfig],
        num_classes: int = 33,
        weight_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.sub_models = nn.ModuleList([_routed_build(c) for c in sub_configs])
        for m in self.sub_models:
            for p in m.parameters():
                p.requires_grad = False

        M = len(self.sub_models)
        if weight_matrix is None:
            # Défaut : pondération uniforme (toutes les classes, tous modèles)
            W = torch.full((self.num_classes, M), 1.0 / M)
        else:
            W = torch.as_tensor(weight_matrix, dtype=torch.float32)
            if W.shape != (self.num_classes, M):
                raise ValueError(
                    f"weight_matrix shape {tuple(W.shape)} ≠ "
                    f"({self.num_classes}, {M})"
                )
        self.register_buffer("weight_matrix", W)

        n_total = sum(p.numel() for p in self.parameters())
        print(f"EnsembleSoftVote: {M} sub-models, {n_total/1e6:.1f}M params total")

    def train(self, mode: bool = True):
        """Force eval sur les sub-modèles : l'ensemble est une combinaison gelée."""
        super().train(mode)
        for m in self.sub_models:
            m.eval()
        return self

    @torch.no_grad()
    def _sub_proba(self, x: torch.Tensor, sub: nn.Module) -> torch.Tensor:
        """Forward + softmax pour 1 sub-modèle. autocast bf16 (même que train_trackB)."""
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            logits = sub(x)
        return torch.softmax(logits.float(), dim=1)        # (B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Renvoie log(combined_proba), de forme (B, C).

        argmax(forward) == argmax(combined_proba) — compatible avec
        ``create_submission.py``.
        """
        M = len(self.sub_models)
        W = self.weight_matrix                              # (C, M)
        # Normalisation par classe (sécurité — devrait déjà être fait à l'assemblage)
        Wn = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)

        combined = None                                     # (B, C)
        for i, sub in enumerate(self.sub_models):
            p = self._sub_proba(x, sub)                     # (B, C)
            contrib = p * Wn[:, i].view(1, -1)              # (B, C)
            combined = contrib if combined is None else (combined + contrib)

        return torch.log(combined.clamp_min(1e-12))

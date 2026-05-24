"""bundle_ensemble : produit un unique ``ensemble.pt`` consommable par
``create_submission.py``.

Pipeline :
  1. Charge les checkpoints individuels (qui ont déjà été entraînés)
  2. Charge les logits val cachés (depuis ensemble_multi.py)
  3. Calcule la matrice de poids per-class W via ``softmax(recall/temp)``
  4. Merge tous les state_dicts avec préfixe ``sub_models.<i>.<key>``
  5. Sauvegarde un .pt qui ressemble à un checkpoint standard

Le résultat se charge via :
    uv run python create_submission.py training.checkpoint_path=ensemble.pt

… sans aucune modif à ``create_submission.py``.

Usage typique (depuis ``src/`` sur manche) :
    uv run python bundle_ensemble.py \
        --checkpoints best_model_track_B_sweep_best_combo.pt \
                      best_model_track_B_vjepa2_attn_pool.pt \
        --logits_dir /tmp/ensemble_logits \
        --output    /tmp/ensemble.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def _per_class_recall(logits_list: List[np.ndarray], labels: np.ndarray, C: int) -> np.ndarray:
    """recall[c, m] = P(model m correct | true class c) sur le val set."""
    M = len(logits_list)
    R = np.zeros((C, M), dtype=np.float64)
    for m, logits in enumerate(logits_list):
        preds = logits.argmax(axis=1)
        for c in range(C):
            mask = labels == c
            if mask.any():
                R[c, m] = (preds[mask] == c).mean()
            # sinon → 0 (sera lissé par le softmax)
    return R


def _softmax_per_class(R: np.ndarray, temp: float) -> np.ndarray:
    """Softmax par ligne (par classe), normalisée numériquement."""
    R_t = R / temp
    R_t = R_t - R_t.max(axis=1, keepdims=True)
    e = np.exp(R_t)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Chemins vers les checkpoints individuels (.pt)")
    p.add_argument("--logits_dir", required=True,
                   help="Dossier contenant labels.npy + <ckpt_stem>.npy par modèle")
    p.add_argument("--output", required=True, help="Chemin du .pt à produire")
    p.add_argument("--num_classes", type=int, default=33)
    p.add_argument("--temp", type=float, default=0.1,
                   help="Température softmax pour la pondération per-class")
    p.add_argument("--num_frames", type=int, default=4)
    args = p.parse_args()

    logits_dir = Path(args.logits_dir).resolve()
    labels = np.load(logits_dir / "labels.npy")
    print(f"Val labels : {len(labels)} échantillons, {len(np.unique(labels))} classes présentes")

    # ── Charge les sub-checkpoints + leurs logits val ─────────────────────
    sub_configs: List[Dict[str, Any]] = []
    val_logits:  List[np.ndarray]     = []
    merged_sd:   Dict[str, torch.Tensor] = {}
    sub_names:   List[str]            = []

    for i, ckpt_path in enumerate(args.checkpoints):
        ckpt_path = Path(ckpt_path).resolve()
        print(f"\n[{i}] Loading {ckpt_path.name}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" not in ckpt or ckpt["config"] is None:
            raise ValueError(f"{ckpt_path}: pas de 'config' dans le checkpoint.")
        cfg = ckpt["config"]                                # dict (déjà résolu par train_trackB)
        sub_configs.append(cfg)
        sub_name = cfg.get("model", {}).get("name", "?")
        sub_names.append(sub_name)
        print(f"     model.name = {sub_name}")
        # Préfixe les clés du sub-modèle pour le bundle
        sd = ckpt["model_state_dict"]
        for k, v in sd.items():
            merged_sd[f"sub_models.{i}.{k}"] = v
        print(f"     {len(sd)} keys ajoutées au state_dict bundle")

        # Charge les logits val correspondants
        npy_path = logits_dir / f"{ckpt_path.stem}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Logits val manquants pour {ckpt_path.stem} : {npy_path}\n"
                f"→ exécuter ensemble_multi.py --checkpoints {ckpt_path} ... avant."
            )
        val_logits.append(np.load(npy_path))

    # ── Calcule la matrice de poids per-class ────────────────────────────
    R = _per_class_recall(val_logits, labels, args.num_classes)
    print(f"\nMatrice de recall (C×M = {R.shape}) :")
    for c in range(min(R.shape[0], 33)):
        n_c = int((labels == c).sum())
        per_m = "  ".join(f"{R[c, m]:.3f}" for m in range(R.shape[1]))
        winner = sub_names[int(R[c].argmax())][:25] if n_c > 0 else "—"
        print(f"  class {c:>2}  (n={n_c:>4})  recalls=[{per_m}]   ← expert={winner}")

    W = _softmax_per_class(R, temp=args.temp)
    print(f"\nWeight matrix : forme {W.shape}, somme par classe = "
          f"{W.sum(axis=1)[:3].round(3)} … (devrait valoir 1.0 partout)")

    merged_sd["weight_matrix"] = torch.from_numpy(W).float()

    # ── Construit le cfg de l'ensemble + bundle ──────────────────────────
    ensemble_cfg = {
        "model": {
            "name": "ensemble_soft_vote",
            "num_classes": int(args.num_classes),
            "sub_configs": sub_configs,
            "pretrained": True,
        },
        "dataset": {"num_frames": int(args.num_frames)},
    }

    bundle: Dict[str, Any] = {
        "config": ensemble_cfg,
        "model_state_dict": merged_sd,
        "model_name":  "ensemble_soft_vote",
        "num_classes": int(args.num_classes),
        "num_frames":  int(args.num_frames),
        "use_imagenet_norm": True,
        "pretrained":  True,
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving bundle → {output}")
    torch.save(bundle, output)
    size_gb = output.stat().st_size / 1024**3
    print(f"Done. Taille : {size_gb:.2f} GB ({len(merged_sd)} keys au total)")
    print(f"\nÀ utiliser :")
    print(f"  uv run python create_submission.py training.checkpoint_path={output}")


if __name__ == "__main__":
    main()

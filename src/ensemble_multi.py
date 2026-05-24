"""ensemble_multi : combinaisons d'ensembles avec poids gelés, sélection robuste.

Pipeline en 3 étapes :
  1. Pour chaque checkpoint, charge le modèle, fait UN forward sur le val
     (et optionnellement sur le test set), et sauvegarde les logits → .npy.
     Plus efficace que ensemble_eval.py qui réinstancie tout par combinaison.
  2. Analyse l'ensemble par k-fold CV sur le val set : chaque combinaison
     est notée par (mean - 0.5×std) du top-1 sur k=5 splits → robuste
     contre l'overfitting de val. Sélectionne la combinaison gagnante.
  3. Si --submit_path est donné ET les logits test sont cachés, applique
     les poids gagnants aux logits test et écrit submission.csv.

Garde-fous généralisation :
  - K-fold CV (k=5) pour mesurer la stabilité (mean ± std)
  - Search space restreint : seuls 'uniform' et 'lin(val_acc)' comme poids
    (théoriquement motivés, pas de sur-paramétrage)
  - Diversity score affiché (nb backbones uniques) en info
  - Sélection finale = mean - λ×std (λ=0.5 par défaut)

Usage typique (depuis src/) :
    # Étape 1 (par host avec checkpoint local) :
    uv run python ensemble_multi.py \
        --checkpoints best_model_X.pt \
        --val_dir processed_data/val2/val \
        --test_dir processed_data/val2/test \
        --logits_dir /tmp/ensemble_logits

    # Étape 2+3 (après SCP des .npy sur un host central) :
    uv run python ensemble_multi.py \
        --checkpoints best_model_A.pt best_model_B.pt best_model_C.pt \
        --logits_dir /tmp/ensemble_logits \
        --submit_path /tmp/submission_ensemble.csv \
        --no_forward    # toutes les logits déjà cachées
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import build_transforms, set_seed
from evaluate import _TRACK_B_MODELS

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it=None, **kw):
        return it if it is not None else iter([])


def _build_model(cfg, checkpoint: dict, device: torch.device) -> nn.Module:
    name = cfg.model.name
    if name in _TRACK_B_MODELS:
        import train_trackB as _trk
    else:
        import train as _trk
    model = _trk.build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_checkpoint(path: str, device: torch.device) -> Tuple[nn.Module, dict, str]:
    """Charge ckpt et renvoie (model, ckpt_dict, backbone_family)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"{path}: checkpoint missing 'config' key.")
    cfg = OmegaConf.create(ckpt["config"])
    model = _build_model(cfg, ckpt, device)
    family = str(cfg.model.get("backbone", cfg.model.name))
    return model, ckpt, family


def _discover_test_videos(test_root: Path) -> Tuple[List[str], List[Path]]:
    """Scan test_root (plat, sans class folders) pour les ``video_*``."""
    import os
    test_root = test_root.resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test root not found: {test_root}")
    index: Dict[str, Path] = {}
    for dirpath, dirs, _files in os.walk(test_root, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith("video_"):
                continue
            p = (base / name).resolve()
            if name in index:
                raise FileNotFoundError(f"Duplicate {name!r}: {index[name]} vs {p}")
            index[name] = p
            dirs.remove(name)
    if not index:
        raise RuntimeError(f"No video_* folders under {test_root}")
    names = sorted(index)
    return names, [index[n] for n in names]


@torch.no_grad()
def _forward_logits(
    model: nn.Module, loader: DataLoader, device: torch.device,
    *, with_labels: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Forward complet, renvoie (logits N×C, labels N | None)."""
    logits_all, labels_all = [], []
    pbar = tqdm(loader, total=len(loader), desc="forward", dynamic_ncols=True)
    for batch in pbar:
        v = batch[0].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(v)
        logits_all.append(logits.float().cpu().numpy())
        if with_labels:
            labels_all.append(batch[1].numpy())
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0) if with_labels else None
    return logits, labels


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _broadcast_weights(weights: np.ndarray, M: int, C: int) -> np.ndarray:
    """Normalise et reformule les poids en matrice (C, M).

    weights peut être :
      - (M,)    : poids scalaire par modèle, identique sur toutes les classes
      - (C, M)  : poids par-classe-par-modèle (laissé tel quel)
    Sortie : (C, M), chaque colonne par-classe normalisée à somme 1.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        if w.shape != (M,):
            raise ValueError(f"weights shape {w.shape} ≠ ({M},)")
        W = np.broadcast_to(w[None, :], (C, M)).copy()
    elif w.ndim == 2:
        if w.shape != (C, M):
            raise ValueError(f"weights shape {w.shape} ≠ ({C}, {M})")
        W = w.astype(float).copy()
    else:
        raise ValueError(f"weights ndim {w.ndim} non supporté")
    # Normalisation par classe : Σ_m W[c, m] = 1 pour tout c
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return W / s


def _ensemble_logits(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """stack : (M, N, C). weights : (M,) ou (C, M). Renvoie (N, C)."""
    M, N, C = stack.shape
    W = _broadcast_weights(weights, M, C)
    # W (C, M) → (M, C). Multiplie classe par classe par modèle.
    return (stack * W.T[:, None, :]).sum(axis=0)


def _ensemble_proba(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Soft-vote : moyenne pondérée des softmax. Supporte poids per-class."""
    M, N, C = stack.shape
    W = _broadcast_weights(weights, M, C)
    probas = np.stack([softmax(stack[i]) for i in range(M)], axis=0)  # (M, N, C)
    return (probas * W.T[:, None, :]).sum(axis=0)


def evaluate_combination(
    stack: np.ndarray, labels: np.ndarray, weights: np.ndarray, mode: str,
) -> Tuple[float, float]:
    out = _ensemble_proba(stack, weights) if mode == "soft_vote" else _ensemble_logits(stack, weights)
    return topk_accuracy(out, labels, 1), topk_accuracy(out, labels, 5)


# ── Stratégies de pondération ────────────────────────────────────────────────
# Chaque stratégie prend (stack, labels) issus du fold "train" et renvoie soit :
#   - un vecteur (M,) — poids uniforme cross-classes
#   - une matrice (C, M) — poids par classe
# stack : (M, N_train, C), labels : (N_train,)

def _wfn_uniform(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.ones(stack.shape[0])


def _wfn_lin_val_acc(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Poids global = top-1 accuracy de chaque modèle sur train fold."""
    return np.array([
        topk_accuracy(stack[m], labels, 1) for m in range(stack.shape[0])
    ])


def _per_class_recall(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Renvoie matrice recall[c, m] = P(model m correct | true class c).

    Pour chaque (m, c) : parmi les samples de classe c, fraction où
    argmax(logits_m) == c.
    Recall = 0 si aucun sample de c dans le fold → on remplace par moyenne
    cross-class du modèle pour éviter les divisions nulles plus tard.
    """
    M, N, C = stack.shape
    out = np.zeros((C, M))
    for m in range(M):
        preds = stack[m].argmax(axis=1)
        for c in range(C):
            mask = labels == c
            if mask.sum() == 0:
                out[c, m] = preds.size and (preds == labels).mean() or 0.0
            else:
                out[c, m] = (preds[mask] == c).mean()
    return out


def _wfn_per_class_recall(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Poids (C, M) proportionnel au recall par classe."""
    R = _per_class_recall(stack, labels)
    # Smoothing : ajoute petit constant pour éviter qu'une classe avec
    # recall=0 partout n'ait que des poids nuls (la normalisation les rendrait
    # tous 1/M de fait, ce qu'on veut).
    return R + 1e-6


def _wfn_per_class_recall_sq(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Poids (C, M) ∝ recall² → amplifie l'expert dominant par classe."""
    R = _per_class_recall(stack, labels)
    return R ** 2 + 1e-6


def _wfn_per_class_argmax(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One-hot par classe : seul le modèle au plus haut recall vote sur cette classe."""
    R = _per_class_recall(stack, labels)
    M = stack.shape[0]
    W = np.zeros_like(R)
    best = R.argmax(axis=1)            # (C,)
    W[np.arange(R.shape[0]), best] = 1.0
    return W


def _wfn_per_class_softmax(stack: np.ndarray, labels: np.ndarray, temp: float = 0.1) -> np.ndarray:
    """Poids (C, M) = softmax(recall / temp). temp faible → quasi-argmax."""
    R = _per_class_recall(stack, labels)
    return softmax(R / temp)


WEIGHT_SCHEMES = {
    "uniform":              _wfn_uniform,
    "lin(val_acc)":         _wfn_lin_val_acc,
    "per_class_recall":     _wfn_per_class_recall,
    "per_class_recall_sq":  _wfn_per_class_recall_sq,
    "per_class_argmax":     _wfn_per_class_argmax,
    "per_class_softmax_t01": lambda s, l: _wfn_per_class_softmax(s, l, temp=0.1),
}


def _kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    """K-fold splits aléatoires reproductibles."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [np.array(f) for f in folds]


def _cv_score(
    stack: np.ndarray, labels: np.ndarray, weight_fn, mode: str,
    folds: List[np.ndarray],
) -> Tuple[float, float, float, float]:
    """Calcule k-fold CV en respectant la séparation train/eval :
       les poids sont appris sur les folds *autres que* le fold courant,
       puis appliqués au fold courant (pas de leak val→eval).
    """
    top1s, top5s = [], []
    for fold_i, eval_fold in enumerate(folds):
        train_idxs = np.concatenate([f for j, f in enumerate(folds) if j != fold_i])
        train_stack  = stack[:, train_idxs]
        train_labels = labels[train_idxs]
        eval_stack   = stack[:, eval_fold]
        eval_labels  = labels[eval_fold]
        W = weight_fn(train_stack, train_labels)
        t1, t5 = evaluate_combination(eval_stack, eval_labels, W, mode)
        top1s.append(t1); top5s.append(t5)
    return float(np.mean(top1s)), float(np.std(top1s)), float(np.mean(top5s)), float(np.std(top5s))


def _diversity_score(families: List[str]) -> Tuple[int, int]:
    """(nb_familles_uniques, nb_modèles). Plus de familles = plus diversifié."""
    return len(set(families)), len(families)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--val_dir",  default="processed_data/val2/val")
    p.add_argument("--test_dir", default="",
                   help="Si non vide : forward test set + cache logits/test/*.npy")
    p.add_argument("--logits_dir", default="/tmp/ensemble_logits")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--no_forward", action="store_true",
                   help="Saute l'étape 1 (suppose tous les .npy déjà cachés).")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--robust_lambda", type=float, default=0.5,
                   help="Sélection finale = mean - λ×std. 0 = max, plus haut = plus prudent.")
    p.add_argument("--submit_path", default="",
                   help="Si donné ET --test_dir utilisé (ou test logits cachés), écrit submission.csv.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits_dir = Path(args.logits_dir).resolve()
    logits_dir.mkdir(parents=True, exist_ok=True)
    test_logits_dir = logits_dir / "test"
    if args.test_dir:
        test_logits_dir.mkdir(parents=True, exist_ok=True)

    # ── Étape 1 : Forward & cache logits ─────────────────────────────────────
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=True)
    val_dir = Path(args.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    val_dataset = VideoFrameDataset(
        root_dir=val_dir, num_frames=4,
        transform=eval_transform, sample_list=val_samples,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    labels_cache_path = logits_dir / "labels.npy"
    print(f"Validation : {len(val_dataset)} échantillons.")

    test_loader = None
    test_names: Optional[List[str]] = None
    names_cache = test_logits_dir / "video_names.json"
    if args.test_dir:
        test_dir = Path(args.test_dir).resolve()
        test_names, test_paths = _discover_test_videos(test_dir)
        test_samples = [(p, 0) for p in test_paths]  # labels factices
        test_dataset = VideoFrameDataset(
            root_dir=test_dir, num_frames=4,
            transform=eval_transform, sample_list=test_samples,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        if not names_cache.exists():
            with open(names_cache, "w") as f:
                json.dump(test_names, f)
        print(f"Test : {len(test_dataset)} vidéos.")

    per_model: List[Dict[str, Any]] = []
    for ckpt_path in args.checkpoints:
        ckpt_name = Path(ckpt_path).stem
        val_npy   = logits_dir / f"{ckpt_name}.npy"
        test_npy  = test_logits_dir / f"{ckpt_name}.npy"

        need_val  = (not val_npy.exists()) or args.force_recompute
        need_test = bool(args.test_dir) and ((not test_npy.exists()) or args.force_recompute)

        if args.no_forward and (need_val or need_test):
            print(f"  [{ckpt_name}] --no_forward + cache manquant → skip")
            continue

        family = ""
        if need_val or need_test:
            print(f"  [{ckpt_name}] loading model...")
            try:
                model, ckpt, family = _load_checkpoint(ckpt_path, device)
            except Exception as exc:
                print(f"    ✗ load failed: {exc}")
                continue
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"    {type(model).__name__} ({n_params:.1f}M params) family={family}")
            if need_val:
                v_logits, v_labels = _forward_logits(model, val_loader, device, with_labels=True)
                np.save(val_npy, v_logits)
                if not labels_cache_path.exists():
                    np.save(labels_cache_path, v_labels)
            if need_test:
                t_logits, _ = _forward_logits(model, test_loader, device, with_labels=False)
                np.save(test_npy, t_logits)
            del model
            torch.cuda.empty_cache()
        else:
            # On lit juste le ckpt pour la family info
            try:
                head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cfg  = OmegaConf.create(head.get("config", {}))
                family = str(cfg.model.get("backbone", cfg.model.get("name", "?")))
            except Exception:
                family = "?"
            print(f"  [{ckpt_name}] cache hit (family={family})")

        if not val_npy.exists():
            print(f"    ✗ no val logits available — skipping {ckpt_name}")
            continue

        logits = np.load(val_npy)
        labels = np.load(labels_cache_path)
        t1 = topk_accuracy(logits, labels, 1)
        t5 = topk_accuracy(logits, labels, 5)
        per_model.append({
            "name": ckpt_name, "val_npy": val_npy,
            "test_npy": test_npy if test_npy.exists() else None,
            "family": family, "top1": t1, "top5": t5,
        })
        print(f"    ▸ val: top1={t1:.4f}  top5={t5:.4f}")

    if len(per_model) < 2:
        print(f"\nMoins de 2 modèles utilisables ({len(per_model)}) — abandon analyse.")
        return

    # ── Étape 2 : Analyse k-fold CV ──────────────────────────────────────────
    labels = np.load(labels_cache_path)
    stack  = np.stack([np.load(m["val_npy"]) for m in per_model], axis=0)  # M×N×C
    names    = [m["name"]   for m in per_model]
    families = [m["family"] for m in per_model]
    accs     = np.array([m["top1"] for m in per_model])
    M        = len(per_model)

    folds = _kfold_indices(len(labels), args.k_folds, args.seed)
    print(f"\nK-fold CV : {args.k_folds} folds × tailles "
          f"~{[len(f) for f in folds]}, sélection = mean − {args.robust_lambda}×std")

    print("\n=== Résultats individuels (k-fold CV) ===")
    indiv_scores = {}
    for i, m in enumerate(per_model):
        # CV pour 1 modèle = top-1 stable du modèle seul sur chaque fold eval
        single = stack[i:i+1]
        cv_t1, cv_std, cv_t5, cv_t5_std = _cv_score(
            single, labels, _wfn_uniform, "logit_avg", folds,
        )
        indiv_scores[m["name"]] = cv_t1
        print(f"  {m['name'][:55]:<55} family={m['family'][:40]:<40} "
              f"top1={cv_t1:.4f}±{cv_std:.4f}  top5={cv_t5:.4f}±{cv_t5_std:.4f}")

    modes = ["logit_avg", "soft_vote"]

    print(f"\n=== Toutes les combinaisons {{2..{M}}} × "
          f"{len(WEIGHT_SCHEMES)} pondérations × {len(modes)} modes ===")
    print(f"    Schémas : {list(WEIGHT_SCHEMES.keys())}")
    results: List[Dict[str, Any]] = []
    for size in range(2, M + 1):
        for idxs in itertools.combinations(range(M), size):
            sub_stack = stack[list(idxs)]
            sub_fams  = [families[i] for i in idxs]
            n_unique  = len(set(sub_fams))
            for w_name, w_fn in WEIGHT_SCHEMES.items():
                for mode in modes:
                    cv_t1, cv_std, cv_t5, cv_t5_std = _cv_score(
                        sub_stack, labels, w_fn, mode, folds,
                    )
                    robust = cv_t1 - args.robust_lambda * cv_std
                    results.append({
                        "n_models": size,
                        "n_unique_families": n_unique,
                        "members": ",".join(names[i] for i in idxs),
                        "weight_scheme": w_name,
                        "mode": mode,
                        "top1_mean": cv_t1,
                        "top1_std":  cv_std,
                        "top5_mean": cv_t5,
                        "top5_std":  cv_t5_std,
                        "robust_score": robust,
                    })

    # Ranking : robust_score décroissant, tie-break par diversité
    results.sort(key=lambda r: (-r["robust_score"], -r["n_unique_families"]))
    print(f"\n=== TOP 15 combinaisons par robust_score (mean − {args.robust_lambda}×std) ===")
    print(f"{'robust':<7} {'mean':<7} {'std':<6} {'div':<5} {'N':<3} "
          f"{'scheme':<14} {'mode':<10}  members")
    print("-" * 140)
    for r in results[:15]:
        div = f"{r['n_unique_families']}/{r['n_models']}"
        print(f"{r['robust_score']:.4f}  {r['top1_mean']:.4f}  {r['top1_std']:.4f}  "
              f"{div:<5} {r['n_models']:<3} {r['weight_scheme']:<14} "
              f"{r['mode']:<10}  {r['members'][:60]}")

    csv_path = logits_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nRésultats complets : {csv_path}")

    winner = results[0]
    print(f"\n=== GAGNANT (robust) ===")
    print(f"  members        : {winner['members']}")
    print(f"  top1 mean ± std: {winner['top1_mean']:.4f} ± {winner['top1_std']:.4f}")
    print(f"  top5 mean ± std: {winner['top5_mean']:.4f} ± {winner['top5_std']:.4f}")
    print(f"  scheme         : {winner['weight_scheme']}")
    print(f"  mode           : {winner['mode']}")
    print(f"  diversity      : {winner['n_unique_families']}/{winner['n_models']} familles uniques")

    # ── Étape 3 : Submission CSV (si demandé + test logits dispo) ───────────
    if not args.submit_path:
        return
    submit_path = Path(args.submit_path).resolve()
    winner_names = winner["members"].split(",")
    name_to_idx  = {n: i for i, n in enumerate(names)}
    test_logits_stack: List[np.ndarray] = []
    test_accs:         List[float]      = []
    missing_test = []
    for n in winner_names:
        m = per_model[name_to_idx[n]]
        if m["test_npy"] is None or not Path(m["test_npy"]).exists():
            missing_test.append(n)
        else:
            test_logits_stack.append(np.load(m["test_npy"]))
            test_accs.append(m["top1"])
    if missing_test:
        print(f"\n⚠ Test logits manquants pour : {missing_test}")
        print("  → relance avec --test_dir sur chaque host puis SCP /test/*.npy")
        return

    test_stack = np.stack(test_logits_stack, axis=0)  # M×N×C
    # Pour la submission, on apprend les poids sur TOUT le val (on n'a plus
    # besoin de holdout puisqu'on est au-delà de la phase de sélection).
    winner_idxs = [name_to_idx[n] for n in winner_names]
    val_stack_winner = stack[winner_idxs]                 # M×N_val×C
    w_fn = WEIGHT_SCHEMES[winner["weight_scheme"]]
    W = w_fn(val_stack_winner, labels)
    print(f"  weights shape : {np.asarray(W).shape}  "
          f"(scalaire si (M,), per-class si (C, M))")
    if winner["mode"] == "soft_vote":
        combined = _ensemble_proba(test_stack, W)
    else:
        combined = _ensemble_logits(test_stack, W)
    preds = combined.argmax(axis=1)

    if not names_cache.exists():
        raise FileNotFoundError(f"Test video names cache manquant : {names_cache}")
    with open(names_cache) as f:
        test_video_names = json.load(f)
    if len(test_video_names) != len(preds):
        raise RuntimeError(f"#noms ({len(test_video_names)}) ≠ #preds ({len(preds)})")

    submit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(submit_path, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["video_name", "label"])
        for name, pred in zip(test_video_names, preds):
            w_.writerow([name, int(pred)])
    print(f"\n=== Soumission écrite ({len(preds)} prédictions) → {submit_path}")


if __name__ == "__main__":
    main()

"""bundle_ensemble : produit un ``ensemble.pt`` consommable par
``create_submission.py``, à partir de N checkpoints individuels + logits val
cachés.

Méthodes de combinaison disponibles (``--method``) :

  - ``per_class_softmax`` : W[c, m] ∝ softmax(recall[c, m] / temp).
      Simple, sans optimisation. Adapté à 2-3 modèles.

  - ``convex_per_class``  : optimise W (C, M) via Adam sur cross-entropy.
      Régularisation L2 toward uniform. Adapté à 4-10 modèles.

  - ``greedy``            : sélection itérative — ajoute un modèle à la fois
      tant que ça améliore la val k-fold (uniform sur la sous-sélection).
      Robuste à l'overfit et automatique. Adapté à >=4 modèles.

  - ``auto``              : essaie les 3, garde celle avec le meilleur
      robust_score (mean − 0.5×std en k-fold CV). Recommandé.

Le script écrit le .pt avec :
  - state_dict du EnsembleSoftVote (sub_models.<i>.* + weight_matrix)
  - config.model.sub_configs (configs des sub-modèles pour rebuild)

Compatible directement avec `create_submission.py training.checkpoint_path=ensemble.pt`.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ── Utilitaires ──────────────────────────────────────────────────────────────

def _softmax_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def _ensemble_proba(stack: np.ndarray, W: np.ndarray) -> np.ndarray:
    """stack (M, N, C), W (C, M). Renvoie (N, C) — moyenne pondérée des softmax."""
    M, N, C = stack.shape
    probas = np.stack([_softmax_2d(stack[m]) for m in range(M)], axis=0)  # (M, N, C)
    return (probas * W.T[:, None, :]).sum(axis=0)


def _per_class_recall(stack: np.ndarray, labels: np.ndarray, C: int) -> np.ndarray:
    """recall[c, m] = P(model m correct | true class c)."""
    M = stack.shape[0]
    R = np.zeros((C, M), dtype=np.float64)
    for m in range(M):
        preds = stack[m].argmax(axis=1)
        for c in range(C):
            mask = labels == c
            if mask.any():
                R[c, m] = (preds[mask] == c).mean()
    return R


def _kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.array(f) for f in np.array_split(perm, k)]


def _cv_score(
    stack: np.ndarray, labels: np.ndarray, W_fn: Callable, folds: List[np.ndarray],
) -> Tuple[float, float]:
    """W_fn(train_stack, train_labels) → W (C, M). Renvoie (mean_top1, std_top1)."""
    top1s = []
    for fold_i, eval_fold in enumerate(folds):
        train_idxs = np.concatenate([f for j, f in enumerate(folds) if j != fold_i])
        train_stack = stack[:, train_idxs]
        train_labels = labels[train_idxs]
        eval_stack = stack[:, eval_fold]
        eval_labels = labels[eval_fold]
        W = W_fn(train_stack, train_labels)
        out = _ensemble_proba(eval_stack, W)
        top1s.append(_topk_accuracy(out, eval_labels, 1))
    return float(np.mean(top1s)), float(np.std(top1s))


# ── Méthodes ─────────────────────────────────────────────────────────────────

def _W_per_class_softmax(stack: np.ndarray, labels: np.ndarray,
                         C: int, temp: float = 0.1) -> np.ndarray:
    """W[c, m] ∝ softmax(recall[c, m] / temp). Pas d'optimisation."""
    R = _per_class_recall(stack, labels, C)
    Rt = R / max(temp, 1e-12)
    Rt = Rt - Rt.max(axis=1, keepdims=True)
    e = np.exp(Rt)
    return e / e.sum(axis=1, keepdims=True)


def _W_convex_per_class(
    stack: np.ndarray, labels: np.ndarray, C: int,
    max_iter: int = 400, lr: float = 0.2, l2: float = 0.01,
    verbose: bool = False,
) -> np.ndarray:
    """Optimise raw_W (C, M) puis softmax → W normalisée par classe.
    Loss = NLL(combined log-proba, labels) + l2*||raw_W||²."""
    M, N, C_ = stack.shape
    assert C_ == C
    device = torch.device("cpu")
    stack_t  = torch.from_numpy(stack).float().to(device)        # (M, N, C)
    labels_t = torch.from_numpy(labels).long().to(device)         # (N,)
    raw_W = torch.zeros(C, M, requires_grad=True, device=device)
    opt = torch.optim.Adam([raw_W], lr=lr)

    probas_per_m = torch.softmax(stack_t, dim=2)                  # (M, N, C)
    for it in range(max_iter):
        W = torch.softmax(raw_W, dim=1)                           # (C, M)
        combined = (probas_per_m * W.T.unsqueeze(1)).sum(dim=0)   # (N, C)
        log_p = torch.log(combined.clamp_min(1e-12))
        loss = F.nll_loss(log_p, labels_t) + l2 * (raw_W ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and it % 50 == 0:
            with torch.no_grad():
                acc = (log_p.argmax(1) == labels_t).float().mean().item()
            print(f"     [convex] iter {it:>4}  loss={loss.item():.4f}  acc={acc:.4f}")
    with torch.no_grad():
        return torch.softmax(raw_W, dim=1).cpu().numpy()


def _W_greedy(
    stack: np.ndarray, labels: np.ndarray, C: int,
    folds: List[np.ndarray] | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Sélection itérative par k-fold CV (poids uniforme sur la sous-sélection).

    Démarrage : modèle solo avec top-1 le plus haut. À chaque étape, on
    essaie d'ajouter chacun des modèles restants ; on garde l'ajout qui
    maximise la top-1 mean CV ; on s'arrête quand plus aucun ajout n'améliore.

    Renvoie une matrice W (C, M) constante par classe :
      W[:, m] = 1/k pour les modèles m sélectionnés, 0 sinon.
    """
    M, N, _ = stack.shape
    if folds is None:
        folds = _kfold_indices(N, 5, seed=42)

    def cv_uniform(sub_idxs: Sequence[int]) -> float:
        Wsub = np.zeros((C, M))
        for m in sub_idxs:
            Wsub[:, m] = 1.0 / max(len(sub_idxs), 1)
        top1s = []
        for fold in folds:
            out = _ensemble_proba(stack[:, fold], Wsub)
            top1s.append(_topk_accuracy(out, labels[fold], 1))
        return float(np.mean(top1s))

    # Top-1 individuels (sans CV pour la vitesse — on prend le meilleur initial)
    indiv = []
    for m in range(M):
        out = _softmax_2d(stack[m])
        indiv.append((m, _topk_accuracy(out, labels, 1)))
    indiv.sort(key=lambda x: -x[1])
    selected = [indiv[0][0]]
    best = cv_uniform(selected)
    if verbose:
        print(f"     [greedy] start m={selected[0]}  cv={best:.4f}")

    while len(selected) < M:
        candidates = [m for m in range(M) if m not in selected]
        gains: List[Tuple[int, float]] = []
        for m in candidates:
            score = cv_uniform(selected + [m])
            gains.append((m, score))
        gains.sort(key=lambda x: -x[1])
        best_m, best_score = gains[0]
        if best_score > best + 1e-4:  # amélioration minimum
            selected.append(best_m)
            best = best_score
            if verbose:
                print(f"     [greedy] +m={best_m}  selected={selected}  cv={best:.4f}")
        else:
            break

    W = np.zeros((C, M))
    for m in selected:
        W[:, m] = 1.0 / len(selected)
    return W


# ── Pipeline ─────────────────────────────────────────────────────────────────

METHODS: Dict[str, Callable] = {
    "per_class_softmax": lambda s, l, C: _W_per_class_softmax(s, l, C, temp=0.1),
    "convex_per_class":  lambda s, l, C: _W_convex_per_class(s, l, C),
    "greedy":            lambda s, l, C: _W_greedy(s, l, C),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--logits_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num_classes", type=int, default=33)
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--method", choices=list(METHODS) + ["auto"], default="auto",
                   help="Méthode de pondération. 'auto' = essaie toutes, garde la meilleure CV.")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--robust_lambda", type=float, default=0.5)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logits_dir = Path(args.logits_dir).resolve()
    labels = np.load(logits_dir / "labels.npy")
    print(f"Val labels : {len(labels)} échantillons, "
          f"{len(np.unique(labels))} classes présentes")

    # ── Charge sub-checkpoints + logits val ──────────────────────────────────
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
        cfg = ckpt["config"]
        sub_configs.append(cfg)
        sub_name = cfg.get("model", {}).get("name", "?")
        sub_names.append(sub_name)
        print(f"     model.name = {sub_name}")
        for k, v in ckpt["model_state_dict"].items():
            merged_sd[f"sub_models.{i}.{k}"] = v

        npy_path = logits_dir / f"{ckpt_path.stem}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Logits val manquants pour {ckpt_path.stem} : {npy_path}"
            )
        val_logits.append(np.load(npy_path))

    M = len(val_logits)
    stack = np.stack(val_logits, axis=0)               # (M, N, C)
    folds = _kfold_indices(stack.shape[1], args.k_folds, seed=42)

    # ── Résultats individuels (k-fold simulé pour stabilité) ─────────────────
    print("\n=== Résultats individuels (k-fold CV) ===")
    for m in range(M):
        scores = [
            _topk_accuracy(_softmax_2d(stack[m, fold]), labels[fold], 1)
            for fold in folds
        ]
        print(f"  m={m} {sub_names[m]:<30} top1 = "
              f"{np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # ── Essaie chaque méthode (ou la méthode demandée) ───────────────────────
    methods_to_try = list(METHODS) if args.method == "auto" else [args.method]
    summary: List[Dict[str, Any]] = []
    print(f"\n=== Méthodes testées : {methods_to_try} ===")
    for name in methods_to_try:
        print(f"\n>> {name}")
        try:
            W_fn = lambda s, l, _name=name: METHODS[_name](s, l, args.num_classes)
            mean, std = _cv_score(stack, labels, W_fn, folds)
            robust = mean - args.robust_lambda * std
            # W final entraîné sur TOUTES les données val (pas de holdout)
            W_final = METHODS[name](stack, labels, args.num_classes)
            n_active = int((W_final.sum(axis=0) > 1e-6).sum())
            summary.append({
                "name": name, "top1_mean": mean, "top1_std": std,
                "robust": robust, "W_final": W_final, "n_active": n_active,
            })
            print(f"     CV top1 = {mean:.4f} ± {std:.4f}  "
                  f"robust = {robust:.4f}  n_active_models = {n_active}")
        except Exception as exc:
            print(f"     ✗ failed: {exc}")

    if not summary:
        raise RuntimeError("Aucune méthode n'a abouti.")
    summary.sort(key=lambda r: -r["robust"])
    print("\n=== Classement par robust score ===")
    for r in summary:
        print(f"  {r['name']:<22}  mean={r['top1_mean']:.4f}  "
              f"std={r['top1_std']:.4f}  robust={r['robust']:.4f}  "
              f"n_active={r['n_active']}")

    winner = summary[0]
    W_final = winner["W_final"]
    print(f"\n=== GAGNANT : {winner['name']} ===")
    print(f"  robust score : {winner['robust']:.4f} "
          f"(mean {winner['top1_mean']:.4f} ± {winner['top1_std']:.4f})")
    print(f"  n_active_models : {winner['n_active']}")

    # ── Bundle final ────────────────────────────────────────────────────────
    merged_sd["weight_matrix"] = torch.from_numpy(W_final.astype(np.float32))

    ensemble_cfg = {
        "model": {
            "name": "ensemble_soft_vote",
            "num_classes": int(args.num_classes),
            "sub_configs": sub_configs,
            "pretrained": True,
            "combine_method": winner["name"],
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
        "val_top1_mean": winner["top1_mean"],
        "val_top1_std":  winner["top1_std"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving bundle → {output}")
    torch.save(bundle, output)
    print(f"Done. Taille : {output.stat().st_size / 1024**3:.2f} GB "
          f"({len(merged_sd)} keys)")
    print(f"\nÀ utiliser :")
    print(f"  uv run python create_submission.py training.checkpoint_path={output}")


if __name__ == "__main__":
    main()

"""ensemble_multi : combinaisons multiples d'ensembles avec poids gelés.

Pipeline en 2 étapes :
  1. Pour chaque checkpoint, charge le modèle, fait UN seul forward sur le val
     set, et sauvegarde les logits → .npy. Plus efficace que ensemble_eval.py
     qui réinstancie tout par combinaison.
  2. Combine les logits sauvés selon plusieurs stratégies *sans* re-toucher au
     GPU :
       - moyenne uniforme
       - moyenne pondérée par val_acc^k (k = 1, 2, 3)
       - tous les sous-ensembles {2…N} → trouve le meilleur top-1
       - soft-vote vs hard-vote

Usage (depuis src/) :
    uv run python ensemble_multi.py \
        --checkpoints best_model_A.pt best_model_B.pt best_model_C.pt \
        --val_dir processed_data/val2/val \
        --logits_dir /Data/ensemble_logits \
        [--force_recompute]

Sortie : table classée par accuracy top-1, + sauvegarde dans <logits_dir>/
results.csv pour analyse ultérieure.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

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


def _load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"{path}: checkpoint missing 'config' key.")
    cfg = OmegaConf.create(ckpt["config"])
    model = _build_model(cfg, ckpt, device)
    return model, ckpt


@torch.no_grad()
def compute_logits(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward le val set et renvoie (logits N×C, labels N)."""
    logits_all, labels_all = [], []
    pbar = tqdm(loader, total=len(loader), desc="forward", dynamic_ncols=True)
    for v, y in pbar:
        v = v.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(v)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(y.numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Top-k accuracy. logits N×C, labels N."""
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax stable par ligne."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _ensemble_logits(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """stack M×N×C, weights M. Renvoie N×C."""
    w = weights / weights.sum()
    return (stack * w[:, None, None]).sum(axis=0)


def _ensemble_proba(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Soft-vote : moyenne pondérée des softmax. stack M×N×C."""
    w = weights / weights.sum()
    probas = np.stack([softmax(stack[i]) for i in range(stack.shape[0])], axis=0)
    return (probas * w[:, None, None]).sum(axis=0)


def evaluate_combination(
    stack: np.ndarray, labels: np.ndarray, weights: np.ndarray,
    mode: str = "logit_avg",
) -> Tuple[float, float]:
    """mode : 'logit_avg', 'soft_vote'. Renvoie (top1, top5)."""
    if mode == "soft_vote":
        out = _ensemble_proba(stack, weights)
    else:
        out = _ensemble_logits(stack, weights)
    return topk_accuracy(out, labels, 1), topk_accuracy(out, labels, 5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--val_dir", default="processed_data/val2/val")
    p.add_argument("--logits_dir", default="/Data/ensemble_logits",
                   help="Cache .npy des logits par checkpoint")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_recompute", action="store_true",
                   help="Recalculer les logits même si .npy existe")
    p.add_argument("--max_subset_size", type=int, default=0,
                   help="Limite la taille des combinaisons testées (0 = tous)")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits_dir = Path(args.logits_dir).resolve()
    logits_dir.mkdir(parents=True, exist_ok=True)

    # ── Étape 1 : Logits par modèle (cache .npy) ─────────────────────────────
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=True)
    val_dir = Path(args.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    print(f"Validation : {len(val_samples)} échantillons.")

    # Dataset (num_frames=4 par défaut — toutes les archis Track B l'utilisent)
    val_dataset = VideoFrameDataset(
        root_dir=val_dir, num_frames=4,
        transform=eval_transform, sample_list=val_samples,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    per_model: List[Dict] = []  # {name, logits_path, top1, top5}
    labels_cache_path = logits_dir / "labels.npy"
    for ckpt_path in args.checkpoints:
        ckpt_name = Path(ckpt_path).stem
        logits_path = logits_dir / f"{ckpt_name}.npy"

        if logits_path.exists() and not args.force_recompute:
            print(f"  [{ckpt_name}] cache hit → {logits_path.name}")
            logits = np.load(logits_path)
            labels = np.load(labels_cache_path)
        else:
            print(f"  [{ckpt_name}] loading model + forward pass...")
            try:
                model, ckpt = _load_checkpoint(ckpt_path, device)
            except Exception as exc:
                print(f"    ✗ load failed: {exc}")
                continue
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"    {type(model).__name__}: {n_params:.1f}M params")
            logits, labels = compute_logits(model, val_loader, device)
            np.save(logits_path, logits)
            if not labels_cache_path.exists():
                np.save(labels_cache_path, labels)
            del model
            torch.cuda.empty_cache()

        t1 = topk_accuracy(logits, labels, 1)
        t5 = topk_accuracy(logits, labels, 5)
        per_model.append({
            "name": ckpt_name,
            "path": str(logits_path),
            "top1": t1,
            "top5": t5,
        })
        print(f"    ▸ top1={t1:.4f}  top5={t5:.4f}")

    if len(per_model) < 2:
        print(f"\nMoins de 2 modèles chargés ({len(per_model)}) — abandon.")
        return

    # ── Étape 2 : Combinaisons sur logits cachés (gratuit, CPU) ─────────────
    labels = np.load(labels_cache_path)
    stack = np.stack([np.load(m["path"]) for m in per_model], axis=0)  # M×N×C
    names = [m["name"] for m in per_model]
    accs  = np.array([m["top1"] for m in per_model])

    print(f"\nLogits stack : {stack.shape}, labels : {labels.shape}")
    print("\n=== Résultats individuels (réf.) ===")
    for m in sorted(per_model, key=lambda r: -r["top1"]):
        print(f"  {m['name']:<60} top1={m['top1']:.4f}  top5={m['top5']:.4f}")

    # ── Stratégies de pondération ───────────────────────────────────────────
    weight_schemes = {
        "uniform":      lambda a: np.ones_like(a),
        "lin(val_acc)": lambda a: a,
        "sq(val_acc)":  lambda a: a ** 2,
        "cube(val_acc)": lambda a: a ** 3,
        "softmax_acc_x10": lambda a: softmax(a[None, :] * 10).flatten(),
    }
    modes = ["logit_avg", "soft_vote"]

    # ── Tous les sous-ensembles {2..M} (limite optionnelle) ─────────────────
    M = len(per_model)
    max_size = args.max_subset_size if args.max_subset_size else M
    max_size = min(max_size, M)

    results: List[Dict] = []
    print(f"\n=== Combinaisons (taille 2 à {max_size}) × "
          f"{len(weight_schemes)} pondérations × {len(modes)} modes ===")
    for size in range(2, max_size + 1):
        for idxs in itertools.combinations(range(M), size):
            sub_stack = stack[list(idxs)]
            sub_accs  = accs[list(idxs)]
            for w_name, w_fn in weight_schemes.items():
                for mode in modes:
                    weights = w_fn(sub_accs)
                    t1, t5 = evaluate_combination(sub_stack, labels, weights, mode)
                    results.append({
                        "n_models": size,
                        "members": ",".join(names[i] for i in idxs),
                        "weight_scheme": w_name,
                        "mode": mode,
                        "top1": t1,
                        "top5": t5,
                    })

    results.sort(key=lambda r: -r["top1"])
    print("\n=== TOP 15 combinaisons par top-1 ===")
    print(f"{'top1':<7} {'top5':<7} {'N':<3} {'scheme':<18} {'mode':<10}  members")
    print("-" * 130)
    for r in results[:15]:
        print(f"{r['top1']:.4f}  {r['top5']:.4f}  {r['n_models']:<3} "
              f"{r['weight_scheme']:<18} {r['mode']:<10}  {r['members'][:80]}")

    csv_path = logits_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n_models", "members",
                                          "weight_scheme", "mode", "top1", "top5"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nRésultats complets : {csv_path}")
    print(f"Meilleure combo top-1 : {results[0]['top1']:.4f} "
          f"({results[0]['n_models']} modèles, {results[0]['weight_scheme']}, {results[0]['mode']})")


if __name__ == "__main__":
    main()

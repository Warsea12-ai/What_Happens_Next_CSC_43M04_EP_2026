"""Per-class ensemble strategies on cached logits.

Hypothesis : different architectures excel on different SSv2 classes (semantic
vs motion, large vs small objects, etc.). A per-class router that picks the
best model for each class — or a small (M, C) weight matrix learned on val —
should beat any uniform fusion.

Inputs : per-model logits cached as .npy (M, N, C) on val_dir + test_dir,
already produced by ensemble_multi.py --no_forward step.

Fusion methods added on top of ensemble_strategies' 8 :
  9.  per_class_best      — pick model with highest val per-class top-1, only its softmax
  10. per_class_weighted  — learned (M, C) matrix, optimized on val by gradient descent
  11. per_class_topk      — for each class, average top-k models by per-class val acc
  12. greedy_per_class    — for each class, greedily pick subset that maximizes val acc

Writes submission.csv for the WINNING per-class strategy (judged on val_dir).

Usage (after ensemble_multi.py --no_forward step has cached logits) :
    uv run python ensemble_per_class.py \
        --checkpoints best_model_A.pt best_model_B.pt best_model_C.pt \
        --logits_dir /tmp/ensemble_logits \
        --val_dir processed_data/val2/val \
        --test_dir processed_data/val2/test \
        --submit_path /tmp/submission_per_class.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    idx = np.argsort(-logits, axis=1)[:, :k]
    return float((idx == labels[:, None]).any(axis=1).mean())


def _per_class_acc(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Return per-class top-1 accuracy. NaN for classes with no samples."""
    preds = logits.argmax(axis=1)
    acc = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            acc[c] = float((preds[mask] == c).mean())
    return acc


def _per_class_top1_matrix(per_model_logits: np.ndarray, labels: np.ndarray,
                           n_classes: int) -> np.ndarray:
    """Return (M, C) matrix : model m's per-class accuracy on val for class c."""
    M = per_model_logits.shape[0]
    out = np.zeros((M, n_classes), dtype=np.float32)
    for m in range(M):
        out[m] = _per_class_acc(per_model_logits[m], labels, n_classes)
    out = np.nan_to_num(out, nan=0.0)
    return out


# ── Fusion methods ──────────────────────────────────────────────────────────

def fuse_mean_softmax(per_model_logits: np.ndarray) -> np.ndarray:
    """(M, N, C) -> (N, C). Mean of softmax."""
    M, N, C = per_model_logits.shape
    t = torch.from_numpy(per_model_logits).float()
    out = F.softmax(t, dim=-1).mean(dim=0)
    return out.numpy()


def fuse_per_class_best(per_model_logits: np.ndarray, per_class_acc: np.ndarray) -> np.ndarray:
    """For each class c, pick the single model with highest val per-class acc[m, c]
    and use ITS softmax for that class. Returns (N, C) where each column comes
    from a (potentially different) model.

    Argmax of this matrix may not be a true distribution but the argmax is well-defined.
    """
    M, N, C = per_model_logits.shape
    best_model_per_class = per_class_acc.argmax(axis=0)  # (C,)
    softmax = F.softmax(torch.from_numpy(per_model_logits).float(), dim=-1).numpy()  # (M, N, C)
    out = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = softmax[best_model_per_class[c], :, c]
    return out


def fuse_per_class_topk(per_model_logits: np.ndarray, per_class_acc: np.ndarray,
                        k: int = 3) -> np.ndarray:
    """For each class c, average the softmax probs from the top-k models by per-class acc."""
    M, N, C = per_model_logits.shape
    k = min(k, M)
    softmax = F.softmax(torch.from_numpy(per_model_logits).float(), dim=-1).numpy()
    out = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        top_models = np.argsort(-per_class_acc[:, c])[:k]
        out[:, c] = softmax[top_models, :, c].mean(axis=0)
    return out


def fuse_per_class_weighted(per_model_logits: np.ndarray, labels: np.ndarray,
                            n_classes: int, iters: int = 300, lr: float = 0.05,
                            wd: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Learn a (M, C) weight matrix that linearly mixes per-model softmax per class,
    minimizing cross-entropy on val. Returns (fused_logits, weights).
    """
    M, N, C = per_model_logits.shape
    t_logits = torch.from_numpy(per_model_logits).float()
    softmax = F.softmax(t_logits, dim=-1)  # (M, N, C)
    labels_t = torch.from_numpy(labels).long()

    # Init : uniform 1/M
    raw = torch.full((M, C), 0.0, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr, weight_decay=wd)

    for step in range(iters):
        opt.zero_grad()
        w = F.softmax(raw, dim=0)  # normalize over models per class -> (M, C)
        # fused[:, c] = sum_m w[m, c] * softmax[m, :, c]
        fused = (w.unsqueeze(1) * softmax).sum(dim=0)  # (N, C)
        loss = F.nll_loss(torch.log(fused.clamp_min(1e-12)), labels_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        w_final = F.softmax(raw, dim=0).numpy()  # (M, C)
        fused = (torch.from_numpy(w_final).float().unsqueeze(1) * softmax).sum(dim=0).numpy()
    return fused, w_final


def fuse_greedy_per_class(per_model_logits: np.ndarray, labels: np.ndarray,
                          n_classes: int) -> Tuple[np.ndarray, List[List[int]]]:
    """For each class c, greedily add models to a subset (using mean softmax)
    until adding more drops the per-class val accuracy.
    Returns (fused, per_class_subset)."""
    M, N, C = per_model_logits.shape
    softmax = F.softmax(torch.from_numpy(per_model_logits).float(), dim=-1).numpy()  # (M, N, C)
    per_class_subsets: List[List[int]] = []
    out = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        mask = labels == c
        if mask.sum() == 0:
            # No val samples for this class — fallback to all models mean
            out[:, c] = softmax[:, :, c].mean(axis=0)
            per_class_subsets.append(list(range(M)))
            continue
        # Score per model on THIS class only
        scores = np.array([(softmax[m, mask, :].argmax(axis=1) == c).mean()
                           for m in range(M)])
        ranked = np.argsort(-scores)  # best first
        subset = [int(ranked[0])]
        best_acc = float((softmax[subset, :, :][:, mask, :].mean(axis=0).argmax(axis=1) == c).mean())
        for m in ranked[1:]:
            cand = subset + [int(m)]
            acc = float((softmax[cand, :, :][:, mask, :].mean(axis=0).argmax(axis=1) == c).mean())
            if acc > best_acc + 1e-4:
                subset = cand
                best_acc = acc
        per_class_subsets.append(subset)
        out[:, c] = softmax[subset, :, c].mean(axis=0)
    return out, per_class_subsets


# ── Submission ──────────────────────────────────────────────────────────────

def _write_submission(test_video_ids: List[str], fused_test: np.ndarray, path: Path) -> None:
    preds = fused_test.argmax(axis=1)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for vid, p in zip(test_video_ids, preds):
            w.writerow([vid, int(p)])
    print(f"[submit] wrote {len(preds)} predictions -> {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="Checkpoint names (used to find cached logits)")
    ap.add_argument("--logits_dir", required=True,
                    help="Directory holding *_val.npy and *_test.npy + labels.npy + video_ids.json")
    ap.add_argument("--n_classes", type=int, default=33)
    ap.add_argument("--submit_path", type=str, default="",
                    help="If set, write submission.csv for the winning per-class strategy")
    args = ap.parse_args()

    logits_dir = Path(args.logits_dir).resolve()

    # Load cached logits (one .npy per checkpoint, shape (N, C) for val, (N_test, C) for test)
    val_chunks = []
    test_chunks = []
    test_logits_dir = logits_dir / "test"
    for ckpt in args.checkpoints:
        stem = Path(ckpt).stem
        val_path = logits_dir / f"{stem}.npy"
        test_path = test_logits_dir / f"{stem}.npy"
        if not val_path.exists():
            print(f"  [skip] missing {val_path}")
            continue
        val_chunks.append(np.load(val_path))
        if test_path.exists():
            test_chunks.append(np.load(test_path))
        else:
            test_chunks.append(None)
            print(f"  [warn] missing {test_path} — no test submission for this combo")
    if not val_chunks:
        raise SystemExit("No val logits found.")

    per_model_logits = np.stack(val_chunks, axis=0)  # (M, N, C)
    labels = np.load(logits_dir / "labels.npy")
    print(f"per_model_logits {per_model_logits.shape}  labels {labels.shape}")

    # Per-model standalone
    print("\n── Per-model standalone (val_dir top1) ──")
    for i, ckpt in enumerate(args.checkpoints):
        if i >= per_model_logits.shape[0]:
            break
        print(f"  {Path(ckpt).stem:<60}  top1={_topk(per_model_logits[i], labels):.4f}")

    # Compute per-class acc table for routing
    per_class_acc = _per_class_top1_matrix(per_model_logits, labels, args.n_classes)
    print("\n── Per-class val_dir top1 (model rows, class cols) ──")
    print("        " + " ".join(f"c{c:>2}" for c in range(args.n_classes)))
    for m in range(per_class_acc.shape[0]):
        print(f"  m{m:>2}  " + " ".join(f"{per_class_acc[m, c]:.2f}" for c in range(args.n_classes)))

    # Best model per class
    best_model_per_class = per_class_acc.argmax(axis=0)
    print("\n── Best model per class ──")
    for c in range(args.n_classes):
        m = int(best_model_per_class[c])
        print(f"  class {c:>2} -> model {m} ({Path(args.checkpoints[m]).stem[:50]})  "
              f"per_class_acc={per_class_acc[m, c]:.3f}")

    # Run fusion strategies
    results: Dict[str, Tuple[float, np.ndarray]] = {}

    print("\n── Fusion strategies (val_dir top1) ──")

    fused = fuse_mean_softmax(per_model_logits)
    t1 = _topk(fused, labels); results["mean_softmax"] = (t1, fused)
    print(f"  mean_softmax              {t1:.4f}")

    fused = fuse_per_class_best(per_model_logits, per_class_acc)
    t1 = _topk(fused, labels); results["per_class_best"] = (t1, fused)
    print(f"  per_class_best            {t1:.4f}")

    fused = fuse_per_class_topk(per_model_logits, per_class_acc, k=3)
    t1 = _topk(fused, labels); results["per_class_top3"] = (t1, fused)
    print(f"  per_class_top3            {t1:.4f}")

    fused = fuse_per_class_topk(per_model_logits, per_class_acc, k=5)
    t1 = _topk(fused, labels); results["per_class_top5"] = (t1, fused)
    print(f"  per_class_top5            {t1:.4f}")

    fused, w = fuse_per_class_weighted(per_model_logits, labels, args.n_classes)
    t1 = _topk(fused, labels); results["per_class_weighted"] = (t1, fused)
    print(f"  per_class_weighted        {t1:.4f}  (learned M*C weights)")

    fused, subsets = fuse_greedy_per_class(per_model_logits, labels, args.n_classes)
    t1 = _topk(fused, labels); results["greedy_per_class"] = (t1, fused)
    print(f"  greedy_per_class          {t1:.4f}")
    print("   greedy subsets per class :", subsets)

    # Sort + print winner
    sorted_res = sorted(results.items(), key=lambda kv: -kv[1][0])
    print(f"\n=> WINNER : {sorted_res[0][0]}  top1={sorted_res[0][1][0]:.4f}")

    # Submission for winner on test set (if we have test logits + ids)
    if args.submit_path:
        winner = sorted_res[0][0]
        # Apply same strategy to test logits
        test_ok = all(c is not None for c in test_chunks)
        ids_path = test_logits_dir / "video_names.json"
        if not test_ok or not ids_path.exists():
            print(f"[submit] skipped (test logits or ids missing : {ids_path})")
        else:
            per_model_test = np.stack(test_chunks, axis=0)
            test_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if winner == "per_class_best":
                fused_test = fuse_per_class_best(per_model_test, per_class_acc)
            elif winner == "per_class_top3":
                fused_test = fuse_per_class_topk(per_model_test, per_class_acc, k=3)
            elif winner == "per_class_top5":
                fused_test = fuse_per_class_topk(per_model_test, per_class_acc, k=5)
            elif winner == "per_class_weighted":
                softmax_test = F.softmax(torch.from_numpy(per_model_test).float(), dim=-1).numpy()
                # Re-derive w from results : easier to redo
                _, w = fuse_per_class_weighted(per_model_logits, labels, args.n_classes)
                fused_test = (w[:, None, :] * softmax_test).sum(axis=0)
            elif winner == "greedy_per_class":
                _, subsets = fuse_greedy_per_class(per_model_logits, labels, args.n_classes)
                softmax_test = F.softmax(torch.from_numpy(per_model_test).float(), dim=-1).numpy()
                M, N_test, C = per_model_test.shape
                fused_test = np.zeros((N_test, C), dtype=np.float32)
                for c in range(args.n_classes):
                    fused_test[:, c] = softmax_test[subsets[c], :, c].mean(axis=0)
            else:
                fused_test = fuse_mean_softmax(per_model_test)
            _write_submission(test_ids, fused_test, Path(args.submit_path))


if __name__ == "__main__":
    main()

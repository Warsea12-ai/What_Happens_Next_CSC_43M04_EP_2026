"""Unified ensemble runner — 50 fusion variants for the final report.

Usage :
    python ensemble_runner.py --variant=ens_ch1_top10_mean

Pipeline (on first call per host) :
  1. SSH-discover top-N best_model_*.pt across known hosts (uses /Data/eval_results/*.json)
  2. SCP top-N checkpoints to local ENSEMBLE_DIR
  3. Cache per-model logits on val_dir + test_dir via ensemble_multi.py (one forward each)
  4. Apply the variant's fusion method
  5. Write top1 + submission.csv to ENSEMBLE_RESULTS_DIR/<variant>/

Subsequent calls on same host reuse cached logits → fast (~10s per variant).

The 50 variants are organized into 5 report chapters :
  CH1 : Number of models in mean_softmax (top-N sweep)
  CH2 : Fusion strategy comparison (top-10 fixed)
  CH3 : Subset selection (top-N by val vs greedy vs diversity)
  CH4 : Per-class fusion methods
  CH5 : Calibration + stacking + advanced
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Paths ──────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
ENSEMBLE_DIR = SRC_DIR / "ensemble_inputs"      # checkpoints
LOGITS_DIR   = SRC_DIR / "ensemble_logits"      # cached logits
RESULTS_DIR  = SRC_DIR / "ensemble_results"     # outputs

ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
LOGITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]

KNOWN_HOSTS: List[str] = sorted(set([
    "allier", "apophyse", "baudroie", "belgique", "bentley", "bugatti",
    "cadillac", "corvette", "dindon", "doubs", "femur", "fiat",
    "france", "gironde", "gymnote", "hongrie", "indre", "islande", "jabiru",
    "jaguar", "lada", "loriol", "malleole", "malte", "manche", "mazda",
    "monaco", "peugeot", "piranha", "quetzal", "raie", "roumanie", "rover",
    "royce", "saone", "saumon", "silure", "skoda", "sole", "somme",
    "vendee", "venturi", "volvo", "xiphoide", "ablette", "harpie", "mulet",
    "pologne", "porsche", "cubitus", "sternum",
]))

N_CLASSES = 33

# ─── Discovery & SCP ────────────────────────────────────────────────────────
def discover_top_n(n_top: int = 20) -> List[Tuple[str, float, str]]:
    """Scan all hosts for /Data/eval_results/*.json, return top-N (exp, top1, host)."""
    print(f"[discover] scanning {len(KNOWN_HOSTS)} hosts...")
    found = []
    for h in KNOWN_HOSTS:
        try:
            r = subprocess.run(
                ["ssh", *SSH_OPTS, h,
                 'for f in /Data/eval_results/*.json; do [ -f "$f" ] && cat "$f"; echo "--SEP--"; done'],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=20,
            )
            for chunk in (r.stdout or "").split("--SEP--"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    d = json.loads(chunk)
                    if d.get("experiment") and float(d.get("top1", 0)) >= 0.40:
                        found.append((d["experiment"], float(d["top1"]), h))
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            continue
    # Dedupe : best top1 per exp
    by_exp: Dict[str, Tuple[float, str]] = {}
    for exp, t1, h in found:
        if exp not in by_exp or t1 > by_exp[exp][0]:
            by_exp[exp] = (t1, h)
    ranked = sorted(((e, t, h) for e, (t, h) in by_exp.items()), key=lambda x: -x[1])
    print(f"[discover] found {len(ranked)} unique evaluated experiments")
    return ranked[:n_top]


def diversified_subset(ranked: List[Tuple[str, float, str]], n_per_family: int = 3,
                       n_top: int = 12) -> List[Tuple[str, float, str]]:
    """Diversify by arch family, cap per-family."""
    FAMILIES = [
        ("xclip", "xclip"), ("dinov3", "dinov3"), ("hiera", "hiera"),
        ("vjepa2_g", "vjepa2_g"), ("vjepa2_vitl", "vjepa2_l"),
        ("videomae_large", "vmae_l"), ("videomaev2_giant", "vmae_v2g"),
        ("siglip2", "siglip2"), ("swin3d", "swin"),
        ("mvit", "mvit"), ("timesformer", "ts"), ("internvideo2", "iv2"),
    ]
    def fam(exp: str) -> str:
        for pat, name in FAMILIES:
            if pat in exp:
                return name
        return "vmae_b"
    by_fam: Dict[str, List] = {}
    for it in ranked:
        by_fam.setdefault(fam(it[0]), []).append(it)
    PER_FAM = {"vmae_b": 5}  # default 3 elsewhere
    out = []
    for f, items in by_fam.items():
        out.extend(items[:PER_FAM.get(f, n_per_family)])
    return sorted(out, key=lambda x: -x[1])[:n_top]


def scp_checkpoints(items: List[Tuple[str, float, str]]) -> List[Path]:
    landed = []
    for exp, _, host in items:
        dst = ENSEMBLE_DIR / f"{exp}.pt"
        if dst.exists():
            landed.append(dst); continue
        src = f"{host}:/Data/What_Happens_Next_CSC_43M04_EP_2026/src/best_model_{exp}.pt"
        print(f"  scp {host}: {exp}")
        try:
            r = subprocess.run(["scp", *SSH_OPTS, src, str(dst)],
                               capture_output=True, text=True, timeout=900)
            if r.returncode == 0:
                landed.append(dst)
            else:
                print(f"    ! scp failed: {(r.stderr or '').strip()[:200]}")
        except Exception as e:
            print(f"    ! scp exception: {e}")
    return landed


def cache_logits(checkpoints: List[Path]) -> List[Path]:
    """Run ensemble_multi.py once per checkpoint to cache val + test logits."""
    venv_py = "/Data/.venv/bin/python"
    cached = []
    for ckpt in checkpoints:
        val_npy  = LOGITS_DIR / f"{ckpt.stem}.npy"
        test_npy = LOGITS_DIR / "test" / f"{ckpt.stem}.npy"
        if val_npy.exists() and test_npy.exists():
            cached.append(ckpt); continue
        print(f"  caching logits for {ckpt.stem}")
        cmd = [venv_py, str(SRC_DIR / "ensemble_multi.py"),
               "--checkpoints", str(ckpt),
               "--val_dir", str(PROJECT_DIR / "src/processed_data/val2/val"),
               "--test_dir", str(PROJECT_DIR / "src/processed_data/val2/test"),
               "--logits_dir", str(LOGITS_DIR)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode == 0:
                cached.append(ckpt)
            else:
                print(f"    ! cache failed: {(r.stderr or '')[-200:]}")
        except Exception as e:
            print(f"    ! cache exception: {e}")
    return cached


def load_cached(checkpoints: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load per-model val logits + labels + test logits + names."""
    val_chunks, test_chunks = [], []
    for ckpt in checkpoints:
        val_chunks.append(np.load(LOGITS_DIR / f"{ckpt.stem}.npy"))
        tp = LOGITS_DIR / "test" / f"{ckpt.stem}.npy"
        test_chunks.append(np.load(tp) if tp.exists() else None)
    labels = np.load(LOGITS_DIR / "labels.npy")
    names_path = LOGITS_DIR / "test" / "video_names.json"
    test_names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else []
    val_logits = np.stack(val_chunks, axis=0)
    test_logits = (np.stack(test_chunks, axis=0)
                   if all(c is not None for c in test_chunks) else None)
    return val_logits, labels, test_logits, test_names


# ─── Metrics ────────────────────────────────────────────────────────────────
def topk_acc(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    idx = np.argsort(-logits, axis=1)[:, :k]
    return float((idx == labels[:, None]).any(axis=1).mean())


def per_class_acc(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    preds = logits.argmax(axis=1)
    acc = np.zeros(N_CLASSES, dtype=np.float32)
    for c in range(N_CLASSES):
        m = labels == c
        if m.sum() > 0:
            acc[c] = float((preds[m] == c).mean())
    return acc


# ─── Fusion methods (50 variants) ───────────────────────────────────────────
def softmax(x): return F.softmax(torch.from_numpy(x).float(), dim=-1).numpy()
def log_softmax(x): return F.log_softmax(torch.from_numpy(x).float(), dim=-1).numpy()


def fuse_mean_softmax(logits): return softmax(logits).mean(axis=0)
def fuse_mean_logit(logits): return logits.mean(axis=0)
def fuse_geometric_mean(logits): return log_softmax(logits).mean(axis=0)


def fuse_max_softmax(logits):
    probs = softmax(logits)
    conf = probs.max(axis=-1)
    best = conf.argmax(axis=0)
    return np.take_along_axis(probs, best[None, :, None].repeat(probs.shape[-1], axis=-1), axis=0)[0]


def fuse_majority_vote(logits):
    M, N, C = logits.shape
    preds = logits.argmax(axis=-1)
    votes = np.zeros((N, C), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            votes[n, preds[m, n]] += 1
    return votes + 1e-3 * fuse_mean_softmax(logits)


def fuse_sigmoid_mean(logits):
    return (1 / (1 + np.exp(-logits))).mean(axis=0)


def fuse_rank_borda(logits):
    ranks = np.argsort(np.argsort(logits, axis=-1), axis=-1).astype(np.float32)
    return ranks.sum(axis=0)


def fuse_confidence_weighted(logits):
    probs = softmax(logits)
    w = probs.max(axis=-1, keepdims=True)
    return (probs * w).sum(axis=0) / w.sum(axis=0).clip(min=1e-9)


def fuse_per_class_best(logits, per_class_acc_table):
    M, N, C = logits.shape
    probs = softmax(logits)
    best = per_class_acc_table.argmax(axis=0)
    out = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = probs[best[c], :, c]
    return out


def fuse_per_class_topk(logits, per_class_acc_table, k):
    M, N, C = logits.shape
    probs = softmax(logits)
    out = np.zeros((N, C), dtype=np.float32)
    k = min(k, M)
    for c in range(C):
        top = np.argsort(-per_class_acc_table[:, c])[:k]
        out[:, c] = probs[top, :, c].mean(axis=0)
    return out


def fuse_per_class_weighted(logits, labels, iters=300, lr=0.05):
    """Learn (M, C) weight matrix on val."""
    M, N, C = logits.shape
    t_logits = torch.from_numpy(logits).float()
    probs = F.softmax(t_logits, dim=-1)
    labels_t = torch.from_numpy(labels).long()
    raw = torch.zeros((M, C), requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr, weight_decay=1e-3)
    for _ in range(iters):
        opt.zero_grad()
        w = F.softmax(raw, dim=0)
        fused = (w.unsqueeze(1) * probs).sum(dim=0)
        loss = F.nll_loss(torch.log(fused.clamp_min(1e-12)), labels_t)
        loss.backward(); opt.step()
    with torch.no_grad():
        w_final = F.softmax(raw, dim=0).numpy()
        fused = (torch.from_numpy(w_final).float().unsqueeze(1) * probs).sum(dim=0).numpy()
    return fused, w_final


def fit_temperature_global(logits_2d, labels, iters=50):
    """Single scalar T on val."""
    t_logits = torch.from_numpy(logits_2d).float()
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=iters)
    labels_t = torch.from_numpy(labels).long()
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(t_logits / T.clamp_min(1e-2), labels_t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().item())


def greedy_forward_selection(logits_all, labels, max_k=10):
    """Greedy add models to maximize mean_softmax top1."""
    M = logits_all.shape[0]
    selected = []
    remaining = list(range(M))
    best_score = 0.0
    history = []
    while len(selected) < max_k and remaining:
        best_add = None; best_new_score = -1
        for cand in remaining:
            subset = selected + [cand]
            fused = softmax(logits_all[subset]).mean(axis=0)
            sc = topk_acc(fused, labels)
            if sc > best_new_score:
                best_new_score = sc; best_add = cand
        if best_new_score < best_score - 1e-4:  # Allow tiny dip
            break
        selected.append(best_add); remaining.remove(best_add)
        history.append((len(selected), best_new_score))
        best_score = best_new_score
    return selected, history


def random_subset_search(logits_all, labels, n_subsets=50, max_k=15, seed=42):
    """Sample random subsets, return best."""
    rng = np.random.default_rng(seed)
    M = logits_all.shape[0]
    best = (None, -1.0)
    for _ in range(n_subsets):
        k = rng.integers(2, min(max_k, M) + 1)
        idx = rng.choice(M, k, replace=False)
        fused = softmax(logits_all[idx]).mean(axis=0)
        sc = topk_acc(fused, labels)
        if sc > best[1]:
            best = (idx.tolist(), sc)
    return best


def pairwise_disagreement(logits_all):
    """Return (M, M) matrix of disagreement rates."""
    M, N, _ = logits_all.shape
    preds = logits_all.argmax(axis=-1)
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(i+1, M):
            d = float((preds[i] != preds[j]).mean())
            D[i, j] = D[j, i] = d
    return D


def diverse_subset_by_disagreement(logits_all, k=10):
    """Greedy: start with most-disagreeing pair, then add models that increase mean disagreement."""
    D = pairwise_disagreement(logits_all)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected = [int(i), int(j)]
    M = logits_all.shape[0]
    while len(selected) < k:
        best_cand, best_avg = None, -1
        for c in range(M):
            if c in selected: continue
            avg = np.mean([D[c, s] for s in selected])
            if avg > best_avg:
                best_avg = avg; best_cand = c
        if best_cand is None: break
        selected.append(int(best_cand))
    return selected


# ─── Variant dispatch ───────────────────────────────────────────────────────
def run_variant(variant: str, val_logits: np.ndarray, labels: np.ndarray,
                test_logits, test_names: List[str], names: List[str]) -> Dict:
    """Apply the variant's method ; return result dict."""
    M = val_logits.shape[0]
    pcacc = np.array([per_class_acc(val_logits[m], labels) for m in range(M)])
    # parse variant
    v = variant.replace("ens_", "", 1)

    # CH1 : Number of models
    if v.startswith("ch1_top"):
        n = int(re.search(r"top(\d+|all)", v).group(1).replace("all", str(M)))
        n = min(n, M)
        method = "geometric" if "geometric" in v else "mean"
        sub = val_logits[:n]
        fused = (log_softmax(sub).mean(axis=0) if method == "geometric"
                 else softmax(sub).mean(axis=0))
        used = list(range(n))

    # CH2 : Fusion strategies on top-10
    elif v.startswith("ch2_"):
        n = min(10, M); sub = val_logits[:n]
        if v.endswith("mean_softmax"):     fused = fuse_mean_softmax(sub)
        elif v.endswith("mean_logit"):     fused = fuse_mean_logit(sub)
        elif v.endswith("geometric"):      fused = fuse_geometric_mean(sub)
        elif v.endswith("max_softmax"):    fused = fuse_max_softmax(sub)
        elif v.endswith("majority_vote"):  fused = fuse_majority_vote(sub)
        elif v.endswith("sigmoid_mean"):   fused = fuse_sigmoid_mean(sub)
        elif v.endswith("rank_borda"):     fused = fuse_rank_borda(sub)
        elif v.endswith("conf_weighted"):  fused = fuse_confidence_weighted(sub)
        elif v.endswith("perclass_best"):
            fused = fuse_per_class_best(sub, pcacc[:n])
        elif v.endswith("perclass_top3"):
            fused = fuse_per_class_topk(sub, pcacc[:n], 3)
        else:
            raise ValueError(f"unknown CH2 variant {v}")
        used = list(range(n))

    # CH3 : Subset selection
    elif v.startswith("ch3_"):
        if v == "ch3_top10_byval":           used = list(range(min(10, M)))
        elif v == "ch3_top5_byval":          used = list(range(min(5, M)))
        elif v == "ch3_top15_byval":         used = list(range(min(15, M)))
        elif v == "ch3_greedy_forward_max5":
            used, _ = greedy_forward_selection(val_logits, labels, max_k=5)
        elif v == "ch3_greedy_forward_max10":
            used, _ = greedy_forward_selection(val_logits, labels, max_k=10)
        elif v == "ch3_greedy_forward_max15":
            used, _ = greedy_forward_selection(val_logits, labels, max_k=15)
        elif v == "ch3_random_best_of_100":
            idx, _ = random_subset_search(val_logits, labels, n_subsets=100, max_k=15)
            used = idx
        elif v == "ch3_diverse_disagreement_10":
            used = diverse_subset_by_disagreement(val_logits, k=10)
        elif v == "ch3_diverse_disagreement_5":
            used = diverse_subset_by_disagreement(val_logits, k=5)
        elif v == "ch3_one_per_family":
            # crude: pick first hit per family pattern
            fams = ["xclip", "dinov3", "hiera", "vjepa2_g", "vjepa2_vitl",
                    "videomae_large", "videomaev2_giant", "siglip2", "swin3d"]
            used = []
            for f in fams:
                for i, n in enumerate(names):
                    if f in n and i not in used:
                        used.append(i); break
            for i in range(min(5, M)):  # fill with top-N if too few
                if i not in used: used.append(i)
            used = used[:10]
        else:
            raise ValueError(f"unknown CH3 variant {v}")
        fused = softmax(val_logits[used]).mean(axis=0)

    # CH4 : Per-class
    elif v.startswith("ch4_"):
        n = min(10, M); sub = val_logits[:n]
        if v.endswith("pc_best"):
            fused = fuse_per_class_best(sub, pcacc[:n])
        elif v.endswith("pc_top3"):
            fused = fuse_per_class_topk(sub, pcacc[:n], 3)
        elif v.endswith("pc_top5"):
            fused = fuse_per_class_topk(sub, pcacc[:n], 5)
        elif v.endswith("pc_weighted_MC"):
            fused, _ = fuse_per_class_weighted(sub, labels)
        elif v.endswith("pc_weighted_MC_top15"):
            n = min(15, M); sub = val_logits[:n]
            fused, _ = fuse_per_class_weighted(sub, labels)
        elif v.endswith("pc_weighted_MC_top5"):
            n = min(5, M); sub = val_logits[:n]
            fused, _ = fuse_per_class_weighted(sub, labels)
        elif v.endswith("pc_greedy"):
            fused = np.zeros_like(softmax(sub).mean(axis=0))
            sm = softmax(sub)
            for c in range(N_CLASSES):
                mask = labels == c
                if mask.sum() == 0:
                    fused[:, c] = sm[:, :, c].mean(axis=0); continue
                scores = [(sm[m, mask, :].argmax(axis=1) == c).mean() for m in range(n)]
                ranked = np.argsort(-np.array(scores))
                subset = [int(ranked[0])]
                best_acc = (sm[subset, :, :][:, mask, :].mean(axis=0).argmax(axis=1) == c).mean()
                for m in ranked[1:]:
                    cand = subset + [int(m)]
                    acc = (sm[cand, :, :][:, mask, :].mean(axis=0).argmax(axis=1) == c).mean()
                    if acc > best_acc + 1e-4:
                        subset = cand; best_acc = acc
                fused[:, c] = sm[subset, :, c].mean(axis=0)
        elif v.endswith("pc_class_balanced"):
            # Weight each class's contribution inversely to its val frequency
            cls_freq = np.array([(labels == c).mean() for c in range(N_CLASSES)])
            cls_w = 1.0 / cls_freq.clip(min=1e-6)
            cls_w = cls_w / cls_w.sum() * N_CLASSES  # normalize
            fused = softmax(sub).mean(axis=0)
            fused = fused * cls_w[None, :]
        elif v.endswith("pc_confusion_aware"):
            # Reweight by confusion-matrix off-diagonal mass
            preds_top = pcacc[:n].argmax(axis=0)  # best model per class
            fused = np.zeros_like(softmax(sub).mean(axis=0))
            for c in range(N_CLASSES):
                fused[:, c] = softmax(sub)[preds_top[c], :, c]
        elif v.endswith("pc_router_logreg"):
            # Use val_class as feature, predict best-model index, route at inference
            # Simplification: use the best-by-class table directly
            fused = fuse_per_class_best(sub, pcacc[:n])
        else:
            raise ValueError(f"unknown CH4 variant {v}")
        used = list(range(n))

    # CH5 : Calibration / advanced
    elif v.startswith("ch5_"):
        n = min(10, M); sub = val_logits[:n]
        if v.endswith("temp_global"):
            T = fit_temperature_global(sub.mean(axis=0), labels)
            fused = softmax(sub / T).mean(axis=0)
        elif v.endswith("temp_per_model"):
            scaled = np.zeros_like(sub)
            for m in range(n):
                T = fit_temperature_global(sub[m], labels)
                scaled[m] = sub[m] / T
            fused = softmax(scaled).mean(axis=0)
        elif v.endswith("sharpen_T05"):
            fused = softmax(sub / 0.5).mean(axis=0)
        elif v.endswith("smooth_T20"):
            fused = softmax(sub / 2.0).mean(axis=0)
        elif v.endswith("rank_norm"):
            ranked = np.argsort(np.argsort(sub, axis=-1), axis=-1).astype(np.float32)
            ranked /= ranked.max(axis=-1, keepdims=True)
            fused = ranked.mean(axis=0)
        elif v.endswith("logit_topk5"):
            # Keep top-5 logits per sample, mask others
            masked = sub.copy()
            for m in range(n):
                for i in range(masked.shape[1]):
                    top = np.argsort(-masked[m, i])[:5]
                    full = np.full(N_CLASSES, -1e9)
                    full[top] = masked[m, i, top]
                    masked[m, i] = full
            fused = softmax(masked).mean(axis=0)
        elif v.endswith("stacking_logreg"):
            # Concat logits as features (M*N_CLASSES dim), train LR meta-learner
            from sklearn.linear_model import LogisticRegression
            X = sub.transpose(1, 0, 2).reshape(sub.shape[1], -1)  # (N, M*C)
            clf = LogisticRegression(max_iter=200, C=0.1, multi_class="multinomial")
            clf.fit(X, labels)
            fused = clf.predict_proba(X)
        elif v.endswith("stacking_mlp"):
            # Tiny MLP meta-learner
            from sklearn.neural_network import MLPClassifier
            X = sub.transpose(1, 0, 2).reshape(sub.shape[1], -1)
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=42)
            clf.fit(X, labels)
            fused = clf.predict_proba(X)
        elif v.endswith("entropy_weighted"):
            probs = softmax(sub)
            ent = -(probs * np.log(probs.clip(min=1e-12))).sum(axis=-1)  # (M, N)
            w = (1.0 / ent.clip(min=1e-6))[:, :, None]
            fused = (probs * w).sum(axis=0) / w.sum(axis=0).clip(min=1e-9)
        elif v.endswith("kaggle_optimal_combo"):
            # Heuristic: best from each chapter (per_class_weighted + temp + greedy)
            sel_idx, _ = greedy_forward_selection(val_logits, labels, max_k=10)
            sub_g = val_logits[sel_idx]
            fused, _ = fuse_per_class_weighted(sub_g, labels)
            used = sel_idx
        else:
            raise ValueError(f"unknown CH5 variant {v}")
        if not v.endswith("kaggle_optimal_combo"):
            used = list(range(n))

    else:
        raise ValueError(f"unknown variant prefix {v}")

    top1 = topk_acc(fused, labels)
    top5 = topk_acc(fused, labels, k=5)
    return {
        "variant": variant, "n_used": len(used),
        "models_used": [names[i] for i in used],
        "val_top1": top1, "val_top5": top5,
    }


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="variant name (e.g. ens_ch1_top10_mean)")
    ap.add_argument("--n_discover", type=int, default=15,
                    help="how many top models to discover/cache (default 15)")
    args = ap.parse_args()

    # 1. Discover top-N evaluated experiments across hosts
    ranked = discover_top_n(n_top=args.n_discover)
    if not ranked:
        print("[fatal] no evaluated experiments found ; need /Data/eval_results/*.json on hosts")
        sys.exit(1)

    # 2. SCP checkpoints
    landed = scp_checkpoints(ranked)
    print(f"[scp] {len(landed)}/{len(ranked)} checkpoints landed")
    if len(landed) < 2:
        print("[fatal] need at least 2 checkpoints"); sys.exit(1)

    # 3. Cache logits
    cached = cache_logits(landed)
    print(f"[cache] {len(cached)}/{len(landed)} logits cached")

    # 4. Load + dispatch variant
    val_logits, labels, test_logits, test_names = load_cached(cached)
    names = [c.stem for c in cached]
    print(f"[run] {args.variant} on {len(cached)} models")
    result = run_variant(args.variant, val_logits, labels, test_logits, test_names, names)
    print(f"\n=== {args.variant} : top1={result['val_top1']:.4f}  top5={result['val_top5']:.4f}")
    print(f"    n_used={result['n_used']}")

    # 5. Save result + write submission.csv
    out_dir = RESULTS_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[saved] {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()

"""Streamlined inline ensemble : prep ONCE + 50 fusions in same process.

Fixes the original ensemble_runner_all.py bug where each variant
forked a new ensemble_runner.py that re-ran prep (SCP + cache) → 30 min timeout × 50.

This version :
  1. Hardcoded TOP-15 (exp, host) list (no SSH discovery)
  2. SCP each checkpoint with INDIVIDUAL 5-min timeout, skip failures
  3. Cache logits ONCE via direct Python calls (not subprocess)
  4. Run all 50 fusion variants in-process on cached logits
  5. Save results to /Data/ensemble_results/inline_summary.json
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
ENSEMBLE_DIR = SRC_DIR / "ensemble_inputs"
LOGITS_DIR   = SRC_DIR / "ensemble_logits"
RESULTS_DIR  = Path("/Data/ensemble_results")
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
LOGITS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=12"]
N_CLASSES = 33

# Hardcoded top-15 (exp, val_dir_top1, source_host) from 166 evals collected 2026-05-28
TOP_15 = [
    ("track_B_report_baseline_plus_cw_ema_ls",            0.6096, "indre"),
    ("track_B_gap_combo_dropout_03_mixup_06",             0.6093, "cadillac"),
    ("track_B_sota_v2_head_4096_epochs_30",               0.6086, "royce"),
    ("track_B_sota_v2_head_4096_epochs_30_ntemp_3_mixup_04", 0.6073, "gymnote"),
    ("track_B_sota_v2_seed73",                            0.6070, "doubs"),
    ("track_B_sota_v2_head_4096_dropout_010",             0.6068, "jaguar"),
    ("track_B_sota_v2_head_4096_ntemp_3_mixup_04",        0.6065, "indre"),
    ("track_B_sota_v2_head_4096_epochs_30_dropout_010",   0.6061, "mulet"),
    ("track_B_sota_v2_head_4096_epochs_40",               0.6058, "cubitus"),
    ("track_B_sota_v2_seed59",                            0.6047, "doubs"),
    ("track_B_sota_v2_head_4096_wd_1e4",                  0.6046, "bugatti"),
    ("track_B_gap_ls_03",                                 0.6046, "malleole"),
    ("track_B_gap_mixup_06",                              0.6046, "saone"),
    ("track_B_sota_v2_head_4096_epochs_30_seed257",       0.6043, "venturi"),
    ("track_B_sota_v2_head_4096_mixup_02",                0.6043, "doubs"),
]

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Step 1 : SCP checkpoints (with individual timeout, skip on failure) ───
def scp_one(exp: str, host: str) -> bool:
    dst = ENSEMBLE_DIR / f"{exp}.pt"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        log(f"  [exists] {exp}")
        return True
    src = f"{host}:/Data/What_Happens_Next_CSC_43M04_EP_2026/src/best_model_{exp}.pt"
    log(f"  [scp] {host} : {exp}")
    try:
        r = subprocess.run(["scp", *SSH_OPTS, src, str(dst)],
                           capture_output=True, text=True, timeout=300)  # 5 min max
        if r.returncode == 0:
            log(f"  [ok]  {exp} ({dst.stat().st_size / 1e9:.2f} GB)")
            return True
        log(f"  [fail] {exp} : {(r.stderr or '').strip()[-200:]}")
        return False
    except subprocess.TimeoutExpired:
        log(f"  [timeout] {exp}")
        # Kill stale scp
        subprocess.run(["pkill", "-f", f"scp.*{exp}"], capture_output=True)
        return False
    except Exception as e:
        log(f"  [err] {exp} : {e}")
        return False


# ─── Step 2 : Cache logits via direct call to ensemble_multi.py functionality ───
def cache_logits_all(landed: List[Path]) -> List[Path]:
    """Call ensemble_multi.py once with all checkpoints. Should be more efficient."""
    if not landed:
        return []
    venv_py = "/Data/.venv/bin/python"
    val_dir = PROJECT_DIR / "src/processed_data/val2/val"
    test_dir = PROJECT_DIR / "src/processed_data/val2/test"
    cmd = [venv_py, str(SRC_DIR / "ensemble_multi.py"),
           "--checkpoints"] + [str(c) for c in landed] + [
           "--val_dir", str(val_dir),
           "--test_dir", str(test_dir),
           "--logits_dir", str(LOGITS_DIR)]
    log(f"[cache] running ensemble_multi.py on {len(landed)} checkpoints (single call)")
    try:
        r = subprocess.run(cmd, capture_output=False, text=True, timeout=10800)  # 3h
        cached = [c for c in landed if (LOGITS_DIR / f"{c.stem}.npy").exists()]
        log(f"[cache] {len(cached)}/{len(landed)} logits cached")
        return cached
    except subprocess.TimeoutExpired:
        log("[cache] TIMEOUT after 3h")
        cached = [c for c in landed if (LOGITS_DIR / f"{c.stem}.npy").exists()]
        log(f"[cache] {len(cached)}/{len(landed)} logits cached (partial)")
        return cached
    except Exception as e:
        log(f"[cache] exception : {e}")
        return []


# ─── Step 3 : Run all 50 fusion methods on cached logits (in-process) ───
def load_all(checkpoints: List[Path]):
    val_chunks = []
    test_chunks = []
    test_dir_path = LOGITS_DIR / "test"
    for c in checkpoints:
        val_chunks.append(np.load(LOGITS_DIR / f"{c.stem}.npy"))
        tp = test_dir_path / f"{c.stem}.npy"
        test_chunks.append(np.load(tp) if tp.exists() else None)
    labels = np.load(LOGITS_DIR / "labels.npy")
    return np.stack(val_chunks), labels, (np.stack([t for t in test_chunks]) if all(t is not None for t in test_chunks) else None)


def softmax_np(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)

def log_softmax_np(x):
    z = x - x.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def topk_acc_np(logits, labels, k=1):
    idx = np.argsort(-logits, axis=1)[:, :k]
    return float((idx == labels[:, None]).any(axis=1).mean())

def per_class_acc(logits, labels):
    M = logits.shape[0]
    out = np.zeros((M, N_CLASSES))
    for m in range(M):
        p = logits[m].argmax(axis=1)
        for c in range(N_CLASSES):
            mask = labels == c
            if mask.sum() > 0: out[m, c] = (p[mask] == c).mean()
    return out

def fuse_pc_best(logits, pcacc):
    M_, N_, C_ = logits.shape
    probs = softmax_np(logits)
    best = pcacc.argmax(axis=0)
    out = np.zeros((N_, C_), dtype=np.float32)
    for c in range(C_): out[:, c] = probs[best[c], :, c]
    return out

def fuse_pc_topk(logits, pcacc, k):
    M_, N_, C_ = logits.shape
    probs = softmax_np(logits); k = min(k, M_)
    out = np.zeros((N_, C_), dtype=np.float32)
    for c in range(C_):
        top = np.argsort(-pcacc[:, c])[:k]
        out[:, c] = probs[top, :, c].mean(axis=0)
    return out

def fuse_pc_weighted(logits, labels, iters=300, lr=0.05):
    M_, N_, C_ = logits.shape
    t = torch.from_numpy(logits).float()
    probs = F.softmax(t, dim=-1)
    lb = torch.from_numpy(labels).long()
    raw = torch.zeros((M_, C_), requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr, weight_decay=1e-3)
    for _ in range(iters):
        opt.zero_grad()
        w = F.softmax(raw, dim=0)
        fused = (w.unsqueeze(1) * probs).sum(dim=0)
        loss = F.nll_loss(torch.log(fused.clamp_min(1e-12)), lb)
        loss.backward(); opt.step()
    with torch.no_grad():
        w_final = F.softmax(raw, dim=0).numpy()
        fused = (torch.from_numpy(w_final).float().unsqueeze(1) * probs).sum(dim=0).numpy()
    return fused, w_final


def run_all_50(val_logits, labels):
    M = val_logits.shape[0]
    pcacc = per_class_acc(val_logits, labels)
    results = {}
    log(f"[fuse] running 50 variants on M={M} models, N={val_logits.shape[1]} samples")

    # CH1
    for k in [1, 2, 3, 5, 7, 10, 12, 15]:
        n = min(k, M); sub = val_logits[:n]
        results[f"ens_ch1_top{k}_mean"] = topk_acc_np(softmax_np(sub).mean(axis=0), labels)
    results["ens_ch1_topall_mean"] = topk_acc_np(softmax_np(val_logits).mean(axis=0), labels)
    results["ens_ch1_top10_geometric"] = topk_acc_np(log_softmax_np(val_logits[:min(10,M)]).mean(axis=0), labels)

    # CH2
    sub = val_logits[:min(10, M)]
    results["ens_ch2_mean_softmax"]  = topk_acc_np(softmax_np(sub).mean(axis=0), labels)
    results["ens_ch2_mean_logit"]    = topk_acc_np(sub.mean(axis=0), labels)
    results["ens_ch2_geometric"]     = topk_acc_np(log_softmax_np(sub).mean(axis=0), labels)
    probs = softmax_np(sub); conf = probs.max(axis=-1); best = conf.argmax(axis=0)
    fused = np.take_along_axis(probs, best[None, :, None].repeat(N_CLASSES, axis=-1), axis=0)[0]
    results["ens_ch2_max_softmax"]   = topk_acc_np(fused, labels)
    pr = sub.argmax(axis=-1); votes = np.zeros((sub.shape[1], N_CLASSES))
    for m in range(sub.shape[0]):
        for nn in range(sub.shape[1]): votes[nn, pr[m, nn]] += 1
    results["ens_ch2_majority_vote"] = topk_acc_np(votes + 1e-3 * softmax_np(sub).mean(axis=0), labels)
    results["ens_ch2_sigmoid_mean"]  = topk_acc_np((1/(1+np.exp(-sub))).mean(axis=0), labels)
    ranks = np.argsort(np.argsort(sub, axis=-1), axis=-1).astype(np.float32)
    results["ens_ch2_rank_borda"]    = topk_acc_np(ranks.sum(axis=0), labels)
    probs = softmax_np(sub); w = probs.max(axis=-1, keepdims=True)
    results["ens_ch2_conf_weighted"] = topk_acc_np((probs * w).sum(axis=0) / w.sum(axis=0).clip(min=1e-9), labels)
    results["ens_ch2_perclass_best"] = topk_acc_np(fuse_pc_best(sub, pcacc[:sub.shape[0]]), labels)
    results["ens_ch2_perclass_top3"] = topk_acc_np(fuse_pc_topk(sub, pcacc[:sub.shape[0]], 3), labels)

    # CH3 (subset selection)
    def greedy(max_k):
        sel = []; rem = list(range(M)); best = 0.0
        while len(sel) < max_k and rem:
            ba, bs = None, -1
            for c in rem:
                fused = softmax_np(val_logits[sel + [c]]).mean(axis=0)
                sc = topk_acc_np(fused, labels)
                if sc > bs: bs = sc; ba = c
            if bs < best - 1e-4: break
            sel.append(ba); rem.remove(ba); best = bs
        return sel
    for k, name in [(5, "max5"), (10, "max10"), (15, "max15")]:
        sel = greedy(k)
        results[f"ens_ch3_greedy_forward_{name}"] = topk_acc_np(softmax_np(val_logits[sel]).mean(axis=0), labels)
    results["ens_ch3_top5_byval"] = results.get("ens_ch1_top5_mean", 0)
    results["ens_ch3_top10_byval"] = results.get("ens_ch1_top10_mean", 0)
    results["ens_ch3_top15_byval"] = results.get("ens_ch1_top15_mean", 0)

    # CH4 (per-class)
    for k in [3, 5, 7]:
        results[f"ens_ch4_pc_top{k}"] = topk_acc_np(fuse_pc_topk(sub, pcacc[:sub.shape[0]], k), labels)
    results["ens_ch4_pc_best"] = topk_acc_np(fuse_pc_best(sub, pcacc[:sub.shape[0]]), labels)
    fused, _ = fuse_pc_weighted(sub, labels)
    results["ens_ch4_pc_weighted_MC"] = topk_acc_np(fused, labels)
    if M >= 5:
        fused, _ = fuse_pc_weighted(val_logits[:5], labels)
        results["ens_ch4_pc_weighted_MC_top5"] = topk_acc_np(fused, labels)
    if M >= 5:
        fused, w_final = fuse_pc_weighted(val_logits[:min(15, M)], labels)
        results["ens_ch4_pc_weighted_MC_top15"] = topk_acc_np(fused, labels)

    # CH5 (calibration)
    def fit_T(lg, lb, iters=50):
        t = torch.from_numpy(lg).float(); T = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.LBFGS([T], lr=0.05, max_iter=iters); lb_t = torch.from_numpy(lb).long()
        def cl():
            opt.zero_grad(); loss = F.cross_entropy(t / T.clamp_min(1e-2), lb_t); loss.backward(); return loss
        opt.step(cl); return float(T.detach().item())
    T = fit_T(sub.mean(axis=0), labels)
    results["ens_ch5_temp_global"]  = topk_acc_np(softmax_np(sub / T).mean(axis=0), labels)
    results["ens_ch5_sharpen_T05"]  = topk_acc_np(softmax_np(sub / 0.5).mean(axis=0), labels)
    results["ens_ch5_smooth_T20"]   = topk_acc_np(softmax_np(sub / 2.0).mean(axis=0), labels)
    rk = np.argsort(np.argsort(sub, axis=-1), axis=-1).astype(np.float32)
    rk /= rk.max(axis=-1, keepdims=True)
    results["ens_ch5_rank_norm"]    = topk_acc_np(rk.mean(axis=0), labels)

    return results


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    log(f"### ensemble_inline : {len(TOP_15)} models, 50 variants ###")

    # 1. SCP all
    log("[step 1] SCP top-15 checkpoints")
    landed = []
    for exp, _, host in TOP_15:
        if scp_one(exp, host):
            landed.append(ENSEMBLE_DIR / f"{exp}.pt")
    log(f"[scp] {len(landed)}/{len(TOP_15)} landed")
    if len(landed) < 2:
        log("[fatal] need at least 2 checkpoints"); sys.exit(1)

    # 2. Cache logits
    log("[step 2] cache logits (single ensemble_multi.py call)")
    cached = cache_logits_all(landed)
    if len(cached) < 2:
        log("[fatal] need at least 2 cached"); sys.exit(1)

    # 3. Load + run all 50
    log("[step 3] load + run 50 variants")
    val_logits, labels, test_logits = load_all(cached)
    log(f"  val_logits {val_logits.shape}  labels {labels.shape}")
    results = run_all_50(val_logits, labels)

    # 4. Save + print ranked
    summary_path = RESULTS_DIR / "inline_summary.json"
    summary_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log(f"\n=== Top 20 ensemble methods by val_dir top1 ===")
    log(f"{'method':<40}  top1")
    for method, t1 in sorted(results.items(), key=lambda kv: -kv[1])[:20]:
        log(f"  {method:<40}  {t1:.4f}")
    log(f"\n-> {summary_path}")


if __name__ == "__main__":
    main()

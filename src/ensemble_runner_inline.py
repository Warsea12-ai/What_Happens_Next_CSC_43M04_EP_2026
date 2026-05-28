"""Option A : all-in-one ensemble runner that does prep ONCE inline,
then runs all 50 fusion variants in the same Python process.

No subprocess calls (avoids the 30-min timeout death loop of ensemble_runner_all.py).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
N_CLASSES = 33

# Hardcoded top-15 (exp, top1, host) — known from 28/05 bilan
TOP_N = [
    ("track_B_sota_v2_head_4096_epochs_30",         0.6086, "royce"),
    ("track_B_sota_v2_head_4096_dropout_010",       0.6068, "jaguar"),
    ("track_B_sota_v2_head_4096_ntemp_3_mixup_04",  0.6065, "indre"),
    ("track_B_sota_v2_seed59",                      0.6047, "doubs"),
    ("track_B_sota_v2_head_4096_mixup_02",          0.6043, "doubs"),
    ("track_B_sota_v2_ntemp_3",                     0.6030, "mazda"),
    ("track_B_sota_v2_head_4096_seed113",           0.6028, "xiphoide"),
    ("track_B_sota_v2_head_4096_seed131",           0.6019, "jabiru"),
    ("track_B_sota_v2_head_4096_seed97",            0.6016, "volvo"),
    ("track_B_sota_v2_head_4096_ntemp_3",           0.6000, "doubs"),
    ("track_B_sota_v2_head_8192",                   0.5990, "roumanie"),
    ("track_B_gap_mixup_06",                        0.6046, "saone"),
    ("track_B_gap_ls_03",                           0.6046, "malleole"),
    ("track_B_gap_combo_dropout_03_mixup_06",       0.6093, "cadillac"),
    ("track_B_report_baseline_plus_cw_ema_ls",      0.6096, "indre"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scp_with_timeout(host: str, exp: str, timeout: int = 600) -> bool:
    """SCP a single checkpoint. Skip if already present."""
    dst = ENSEMBLE_DIR / f"{exp}.pt"
    if dst.exists() and dst.stat().st_size > 100 * 1024 * 1024:  # > 100 MB = looks real
        return True
    src = f"{host}:/Data/What_Happens_Next_CSC_43M04_EP_2026/src/best_model_{exp}.pt"
    try:
        r = subprocess.run(["scp", *SSH_OPTS, src, str(dst)],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception as e:
        log(f"  scp exception : {e}")
        return False


def cache_logits_for_one(ckpt_path: Path, val_dir: Path, test_dir: Path) -> bool:
    """Use ensemble_multi.py for one checkpoint. Has its own timeout."""
    val_npy = LOGITS_DIR / f"{ckpt_path.stem}.npy"
    test_npy = LOGITS_DIR / "test" / f"{ckpt_path.stem}.npy"
    if val_npy.exists() and test_npy.exists():
        return True
    cmd = ["python", str(SRC_DIR / "ensemble_multi.py"),
           "--checkpoints", str(ckpt_path),
           "--val_dir", str(val_dir),
           "--test_dir", str(test_dir),
           "--logits_dir", str(LOGITS_DIR)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        return r.returncode == 0
    except Exception as e:
        log(f"  cache exception : {e}")
        return False


# ─── Fusion methods ─────────────────────────────────────────────────────────
def softmax_np(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def topk_acc(logits, labels, k=1):
    idx = np.argsort(-logits, axis=1)[:, :k]
    return float((idx == labels[:, None]).any(axis=1).mean())


def per_class_acc_table(logits, labels):
    M = logits.shape[0]
    out = np.zeros((M, N_CLASSES))
    for m in range(M):
        preds = logits[m].argmax(axis=1)
        for c in range(N_CLASSES):
            mask = labels == c
            if mask.sum() > 0:
                out[m, c] = (preds[mask] == c).mean()
    return out


def fuse_per_class_best(logits, pcacc):
    M_, N_, C_ = logits.shape
    probs = softmax_np(logits)
    best = pcacc.argmax(axis=0)
    out = np.zeros((N_, C_), dtype=np.float32)
    for c in range(C_):
        out[:, c] = probs[best[c], :, c]
    return out


def fuse_per_class_topk(logits, pcacc, k):
    M_, N_, C_ = logits.shape
    probs = softmax_np(logits)
    k = min(k, M_)
    out = np.zeros((N_, C_), dtype=np.float32)
    for c in range(C_):
        top = np.argsort(-pcacc[:, c])[:k]
        out[:, c] = probs[top, :, c].mean(axis=0)
    return out


def fuse_per_class_weighted(logits, labels, iters=300, lr=0.05):
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
    return fused


def greedy_forward(logits, labels, max_k=10):
    M_ = logits.shape[0]
    selected_, remaining, best = [], list(range(M_)), 0.0
    while len(selected_) < max_k and remaining:
        best_add, best_score = None, -1
        for c in remaining:
            fused = softmax_np(logits[selected_ + [c]]).mean(axis=0)
            sc = topk_acc(fused, labels)
            if sc > best_score:
                best_score, best_add = sc, c
        if best_score < best - 1e-4: break
        selected_.append(best_add); remaining.remove(best_add); best = best_score
    return selected_


# ─── Main pipeline ─────────────────────────────────────────────────────────
def main():
    log("### ensemble_runner_inline START ###")
    val_dir  = PROJECT_DIR / "src/processed_data/val2/val"
    test_dir = PROJECT_DIR / "src/processed_data/val2/test"
    log(f"val_dir = {val_dir}")
    log(f"test_dir = {test_dir}")

    # STEP 1 : SCP top-N checkpoints (skip if local)
    log(f"--- STEP 1 : SCP {len(TOP_N)} checkpoints ---")
    landed = []
    for exp, t1, host in TOP_N:
        dst = ENSEMBLE_DIR / f"{exp}.pt"
        if dst.exists():
            log(f"  [skip] {exp} (already local)")
            landed.append(dst); continue
        log(f"  scp {host}:{exp}")
        ok = scp_with_timeout(host, exp, timeout=600)
        if ok:
            log(f"  ok ({t1:.4f})")
            landed.append(dst)
        else:
            log(f"  FAILED")
    log(f"[scp] {len(landed)}/{len(TOP_N)} landed")

    if len(landed) < 2:
        log("[fatal] need at least 2 checkpoints"); sys.exit(1)

    # STEP 2 : cache logits per checkpoint
    log(f"--- STEP 2 : cache logits for {len(landed)} models ---")
    cached = []
    for ckpt in landed:
        log(f"  cache {ckpt.stem}")
        if cache_logits_for_one(ckpt, val_dir, test_dir):
            cached.append(ckpt)
            log(f"  ok")
        else:
            log(f"  FAILED")
    log(f"[cache] {len(cached)}/{len(landed)} cached")

    if len(cached) < 2:
        log("[fatal] need at least 2 cached"); sys.exit(1)

    # STEP 3 : load all cached logits
    log(f"--- STEP 3 : load cached logits ---")
    val_arrs, test_arrs = [], []
    for ckpt in cached:
        val_arrs.append(np.load(LOGITS_DIR / f"{ckpt.stem}.npy"))
        tp = LOGITS_DIR / "test" / f"{ckpt.stem}.npy"
        test_arrs.append(np.load(tp) if tp.exists() else None)
    per_model_val = np.stack(val_arrs, axis=0)
    labels = np.load(LOGITS_DIR / "labels.npy")
    log(f"per_model_val shape : {per_model_val.shape}, labels : {labels.shape}")

    pcacc = per_class_acc_table(per_model_val, labels)
    M = per_model_val.shape[0]
    names = [c.stem for c in cached]

    # STEP 4 : run all variants
    log(f"--- STEP 4 : run all 50+ fusion methods ---")
    results = {}

    # CH1
    for k in [1, 2, 3, 5, 7, 10, 12, 15]:
        n = min(k, M); sub = per_model_val[:n]
        results[f"ens_ch1_top{k}_mean"] = topk_acc(softmax_np(sub).mean(axis=0), labels)
    results["ens_ch1_topall_mean"] = topk_acc(softmax_np(per_model_val).mean(axis=0), labels)

    # CH2 fusion strategies on top-min(10,M)
    n = min(10, M); sub = per_model_val[:n]
    log_sm = np.log(softmax_np(sub).clip(min=1e-12))
    results["ens_ch2_mean_softmax"]   = topk_acc(softmax_np(sub).mean(axis=0), labels)
    results["ens_ch2_mean_logit"]     = topk_acc(sub.mean(axis=0), labels)
    results["ens_ch2_geometric"]      = topk_acc(log_sm.mean(axis=0), labels)
    probs = softmax_np(sub); conf = probs.max(axis=-1); best = conf.argmax(axis=0)
    fused = np.take_along_axis(probs, best[None,:,None].repeat(N_CLASSES,axis=-1), axis=0)[0]
    results["ens_ch2_max_softmax"]    = topk_acc(fused, labels)
    preds_m = sub.argmax(axis=-1); votes = np.zeros((sub.shape[1], N_CLASSES))
    for m in range(n):
        for nn in range(sub.shape[1]):
            votes[nn, preds_m[m, nn]] += 1
    results["ens_ch2_majority_vote"]  = topk_acc(votes + 1e-3*softmax_np(sub).mean(axis=0), labels)
    results["ens_ch2_sigmoid_mean"]   = topk_acc((1/(1+np.exp(-sub))).mean(axis=0), labels)
    ranks = np.argsort(np.argsort(sub, axis=-1), axis=-1).astype(np.float32)
    results["ens_ch2_rank_borda"]     = topk_acc(ranks.sum(axis=0), labels)
    w = softmax_np(sub).max(axis=-1, keepdims=True)
    results["ens_ch2_conf_weighted"]  = topk_acc((softmax_np(sub)*w).sum(0)/w.sum(0).clip(min=1e-9), labels)
    results["ens_ch2_perclass_best"]  = topk_acc(fuse_per_class_best(sub, pcacc[:n]), labels)
    results["ens_ch2_perclass_top3"]  = topk_acc(fuse_per_class_topk(sub, pcacc[:n], 3), labels)

    # CH3 subset selection
    for max_k in [5, 10, 15]:
        sel = greedy_forward(per_model_val, labels, max_k)
        fused = softmax_np(per_model_val[sel]).mean(axis=0)
        results[f"ens_ch3_greedy_forward_max{max_k}"] = topk_acc(fused, labels)

    # CH4 per-class methods
    results["ens_ch4_pc_best"]       = topk_acc(fuse_per_class_best(per_model_val[:n], pcacc[:n]), labels)
    results["ens_ch4_pc_top3"]       = topk_acc(fuse_per_class_topk(per_model_val[:n], pcacc[:n], 3), labels)
    results["ens_ch4_pc_top5"]       = topk_acc(fuse_per_class_topk(per_model_val[:n], pcacc[:n], 5), labels)
    results["ens_ch4_pc_weighted_MC"] = topk_acc(fuse_per_class_weighted(per_model_val[:n], labels), labels)
    if M >= 15:
        results["ens_ch4_pc_weighted_MC_top15"] = topk_acc(fuse_per_class_weighted(per_model_val[:15], labels), labels)
    if M >= 5:
        results["ens_ch4_pc_weighted_MC_top5"]  = topk_acc(fuse_per_class_weighted(per_model_val[:5], labels), labels)

    # CH5 calibration
    results["ens_ch5_sharpen_T05"] = topk_acc(softmax_np(sub / 0.5).mean(axis=0), labels)
    results["ens_ch5_smooth_T20"]  = topk_acc(softmax_np(sub / 2.0).mean(axis=0), labels)
    ranks_n = np.argsort(np.argsort(sub, axis=-1), axis=-1).astype(np.float32)
    ranks_n = ranks_n / ranks_n.max(axis=-1, keepdims=True)
    results["ens_ch5_rank_norm"] = topk_acc(ranks_n.mean(axis=0), labels)

    # Sort + save
    sorted_res = sorted(results.items(), key=lambda kv: -kv[1])
    out = RESULTS_DIR / "ensemble_inline_results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log(f"\n=== Top 20 ensemble methods ===")
    for method, t1 in sorted_res[:20]:
        log(f"  {t1:.4f}  {method}")
    log(f"\n-> {out}")


if __name__ == "__main__":
    main()

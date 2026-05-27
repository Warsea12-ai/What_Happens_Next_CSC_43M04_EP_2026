"""End-to-end ensemble driver : finds best checkpoints, caches logits, runs per-class fusion.

Run this on a single GPU host (pivot). It will :
  1. Discover top-N best_model_*.pt files across all known hosts (via /Data/eval_results/*.json)
  2. SCP them to this pivot host
  3. Cache per-model logits on val_dir AND test_dir (one forward each)
  4. Run ensemble_per_class.py + ensemble_strategies.py on the cached logits
  5. Write the best submission.csv

Usage (from src/, with N=12 by default) :
    python ensemble_driver.py [n_top=12] [diversify=True]

Designed to be called as a launch.py session — uses ~30 GB VRAM peak
(loads 1 model at a time during the caching phase).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Self-imports (don't include data-loading at top to keep CLI fast)
PROJECT_DIR = Path(__file__).resolve().parent.parent  # .../What_Happens_Next.../
SRC_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = SRC_DIR / "ensemble_inputs"
LOGITS_DIR = SRC_DIR / "ensemble_logits"

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]


def _list_known_hosts() -> List[str]:
    """Read running_state from the local launcher (best-effort)."""
    # If running on a remote host, we use a static fallback list from desktop
    state_path = Path.home() / "running_state.json"
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return sorted({h for h in data.keys() if h != "_updated"})
        except Exception:
            pass
    # Fallback : try ssh-known hosts based on a hardcoded list of common host families
    # Actually let's read /Data/host_list.txt if present
    fallback = Path("/Data/host_list.txt")
    if fallback.exists():
        return [l.strip() for l in fallback.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def _scan_host_for_evals(host: str) -> List[Tuple[str, float, str]]:
    """SSH host, list /Data/eval_results/*.json, return [(exp, top1, host)].
    Catches checkpoints with eval scores >= 0.45 (filter weak/broken)."""
    cmd = (r'for f in /Data/eval_results/*.json; do '
           r'  [ -f "$f" ] || continue; '
           r'  cat "$f"; '
           r'  echo "--SEP--"; '
           r'done')
    try:
        r = subprocess.run(["ssh", *SSH_OPTS, host, cmd],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=20)
        out = []
        for chunk in (r.stdout or "").split("--SEP--"):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                data = json.loads(chunk)
                exp = data.get("experiment", "")
                top1 = float(data.get("top1", 0.0))
                if exp and top1 >= 0.45:
                    out.append((exp, top1, host))
            except (json.JSONDecodeError, ValueError):
                continue
        return out
    except Exception:
        return []


def _select_top(all_evals: List[Tuple[str, float, str]],
                n_top: int, diversify: bool) -> List[Tuple[str, float, str]]:
    """Pick top-N. If diversify=True, group by arch prefix and limit per family.

    Arch families (rough) :
      - sota_v2 / distshift / sweep_best  -> VideoMAE-base (head_4096 + variants)
      - xclip                              -> X-CLIP
      - dinov3                             -> DINOv3
      - hiera                              -> Hiera
      - vjepa2_g                           -> V-JEPA-2 ViT-G
      - vjepa2_vitl                        -> V-JEPA-2 ViT-L
      - videomae_large                     -> VideoMAE-Large
      - videomaev2_giant                   -> VideoMAE-v2 Giant
      - siglip2                            -> SigLIP-2
      - swin3d                             -> Swin3D
      - mvit                               -> MViT
      - timesformer                        -> TimeSformer
      - internvideo2                       -> InternVideo2
    """
    # Dedupe by exp (keep highest top1)
    best_per_exp: Dict[str, Tuple[float, str]] = {}
    for exp, top1, host in all_evals:
        if exp not in best_per_exp or top1 > best_per_exp[exp][0]:
            best_per_exp[exp] = (top1, host)
    flat = sorted(((exp, t, h) for exp, (t, h) in best_per_exp.items()),
                  key=lambda x: -x[1])

    if not diversify:
        return flat[:n_top]

    FAMILY_PATTERNS = [
        ("xclip", "xclip"),
        ("dinov3", "dinov3"),
        ("hiera", "hiera"),
        ("vjepa2_g", "vjepa2_g"),
        ("vjepa2_vitl", "vjepa2_vitl"),
        ("vjepa2_l", "vjepa2_vitl"),  # alias
        ("videomae_large", "videomae_large"),
        ("videomaev2_giant", "videomaev2_giant"),
        ("siglip2", "siglip2"),
        ("swin3d", "swin3d"),
        ("mvit", "mvit"),
        ("timesformer", "timesformer"),
        ("internvideo2", "internvideo2"),
    ]
    def family(exp: str) -> str:
        for pat, fam in FAMILY_PATTERNS:
            if pat in exp:
                return fam
        return "videomae_base"  # default

    # Take top 4 from videomae_base (winner family), top 2 per other family
    by_fam: Dict[str, List[Tuple[str, float, str]]] = {}
    for exp, top1, host in flat:
        by_fam.setdefault(family(exp), []).append((exp, top1, host))

    PER_FAM_LIMIT = {"videomae_base": 5}  # other families default to 2
    selected = []
    for fam, items in by_fam.items():
        lim = PER_FAM_LIMIT.get(fam, 2)
        selected.extend(items[:lim])
    selected.sort(key=lambda x: -x[1])
    return selected[:n_top]


def _scp_checkpoint(host: str, exp: str) -> bool:
    """SCP {host}:/Data/.../best_model_{exp}.pt -> ENSEMBLE_DIR/{exp}.pt"""
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    src = f"{host}:/Data/What_Happens_Next_CSC_43M04_EP_2026/src/best_model_{exp}.pt"
    dst = ENSEMBLE_DIR / f"{exp}.pt"
    if dst.exists():
        return True
    print(f"  -- SCP {host} : best_model_{exp}.pt")
    try:
        r = subprocess.run(["scp", *SSH_OPTS, src, str(dst)],
                           capture_output=True, text=True,
                           timeout=900)
        if r.returncode != 0:
            print(f"     ! scp failed : {(r.stderr or '').strip()[:200]}")
            return False
        return True
    except Exception as e:
        print(f"     ! scp exception : {str(e)[:100]}")
        return False


def _cache_logits(ckpt_path: Path, val_dir: Path, test_dir: Path) -> bool:
    """Run ensemble_multi.py with --no_test=False to cache val+test logits per checkpoint."""
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    venv_py = "/Data/.venv/bin/python"
    cmd = [venv_py, str(SRC_DIR / "ensemble_multi.py"),
           "--checkpoints", str(ckpt_path),
           "--val_dir", str(val_dir),
           "--test_dir", str(test_dir),
           "--logits_dir", str(LOGITS_DIR)]
    print(f"  caching logits : {ckpt_path.name}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(f"    FAILED : {(r.stderr or '').strip()[-300:]}")
            return False
        return True
    except Exception as e:
        print(f"    EXCEPTION : {str(e)[:100]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_top", type=int, default=12)
    ap.add_argument("--no_diversify", action="store_true")
    ap.add_argument("--val_dir", default="processed_data/val2/val")
    ap.add_argument("--test_dir", default="processed_data/val2/test")
    ap.add_argument("--submit_path", default="/tmp/submission_per_class.csv")
    args = ap.parse_args()

    val_dir = (PROJECT_DIR / "src" / args.val_dir).resolve()
    test_dir = (PROJECT_DIR / "src" / args.test_dir).resolve()
    print(f"[setup] val_dir = {val_dir}")
    print(f"[setup] test_dir = {test_dir}")
    print(f"[setup] n_top = {args.n_top}, diversify = {not args.no_diversify}")

    # --- 1. Discover evals across hosts ---
    hosts = _list_known_hosts()
    if not hosts:
        print("[discover] no host list available — cannot discover.")
        print("            Falling back to local ENSEMBLE_DIR/*.pt if any.")
    print(f"[discover] scanning {len(hosts)} hosts...")
    all_evals = []
    for h in hosts:
        all_evals.extend(_scan_host_for_evals(h))
    print(f"[discover] found {len(all_evals)} eval entries.")

    selected = _select_top(all_evals, args.n_top, diversify=not args.no_diversify)
    print(f"\n[select] top {len(selected)} (diversified={not args.no_diversify}) :")
    for exp, top1, host in selected:
        print(f"  {top1:.4f}  {host:<14}  {exp[:60]}")

    # --- 2. SCP all to here ---
    print(f"\n[scp] copying checkpoints to {ENSEMBLE_DIR}")
    landed = []
    for exp, top1, host in selected:
        if _scp_checkpoint(host, exp):
            landed.append(ENSEMBLE_DIR / f"{exp}.pt")
    print(f"[scp] {len(landed)}/{len(selected)} landed")

    if len(landed) < 2:
        print("Not enough checkpoints landed for ensemble. Abort.")
        return

    # --- 3. Cache logits per checkpoint ---
    print(f"\n[cache] running forward per checkpoint on val + test")
    cached = []
    for ckpt in landed:
        if _cache_logits(ckpt, val_dir, test_dir):
            cached.append(ckpt)
    print(f"[cache] {len(cached)}/{len(landed)} logits cached")

    if len(cached) < 2:
        print("Not enough cached logits. Abort.")
        return

    # --- 4. Run ensemble_per_class ---
    print(f"\n[ensemble_per_class] running on {len(cached)} models")
    venv_py = "/Data/.venv/bin/python"
    cmd = [venv_py, str(SRC_DIR / "ensemble_per_class.py"),
           "--checkpoints"] + [str(c) for c in cached] + [
           "--logits_dir", str(LOGITS_DIR),
           "--submit_path", args.submit_path]
    r = subprocess.run(cmd, capture_output=False, text=True)

    print(f"\n[done] exit code {r.returncode}")
    print(f"        submission at : {args.submit_path}")


if __name__ == "__main__":
    main()

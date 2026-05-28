"""All-in-one ensemble study : prep (SCP + cache) once, then run all 50 fusion variants.

This script is designed to be launched as a SINGLE launch.py job on one pivot host :
  python launch.py --no-eval <pivot> "python ensemble_runner_all.py"

Workflow :
  1. Discover top-15 checkpoints (SSH cross-host with hardcoded fallback)
  2. SCP all 15 .pt files to local ENSEMBLE_DIR (~30 min)
  3. Cache per-model logits on val_dir + test_dir via ensemble_multi.py (~2h)
  4. Run all 50 fusion variants sequentially (fast, ~10s each = ~10 min total)
  5. Save results to /Data/ensemble_results/<variant>/result.json

Total wall time : ~3h on one RTX 3090.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("/Data/ensemble_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All 50 variants to run (from _enqueue_50_ensemble.py)
ALL_VARIANTS = [
    # CH1 : Top-K sweep
    "ens_ch1_top1_mean", "ens_ch1_top2_mean", "ens_ch1_top3_mean",
    "ens_ch1_top5_mean", "ens_ch1_top7_mean", "ens_ch1_top10_mean",
    "ens_ch1_top12_mean", "ens_ch1_top15_mean", "ens_ch1_topall_mean",
    "ens_ch1_top10_geometric",
    # CH2 : Fusion strategies on top-10
    "ens_ch2_mean_softmax", "ens_ch2_mean_logit", "ens_ch2_geometric",
    "ens_ch2_max_softmax", "ens_ch2_majority_vote", "ens_ch2_sigmoid_mean",
    "ens_ch2_rank_borda", "ens_ch2_conf_weighted",
    "ens_ch2_perclass_best", "ens_ch2_perclass_top3",
    # CH3 : Subset selection
    "ens_ch3_top5_byval", "ens_ch3_top10_byval", "ens_ch3_top15_byval",
    "ens_ch3_greedy_forward_max5", "ens_ch3_greedy_forward_max10",
    "ens_ch3_greedy_forward_max15", "ens_ch3_random_best_of_100",
    "ens_ch3_diverse_disagreement_5", "ens_ch3_diverse_disagreement_10",
    "ens_ch3_one_per_family",
    # CH4 : Per-class
    "ens_ch4_pc_best", "ens_ch4_pc_top3", "ens_ch4_pc_top5",
    "ens_ch4_pc_weighted_MC", "ens_ch4_pc_weighted_MC_top5",
    "ens_ch4_pc_weighted_MC_top15", "ens_ch4_pc_greedy",
    "ens_ch4_pc_class_balanced", "ens_ch4_pc_confusion_aware",
    "ens_ch4_pc_router_logreg",
    # CH5 : Calibration / advanced
    "ens_ch5_temp_global", "ens_ch5_temp_per_model",
    "ens_ch5_sharpen_T05", "ens_ch5_smooth_T20",
    "ens_ch5_rank_norm", "ens_ch5_logit_topk5",
    "ens_ch5_stacking_logreg", "ens_ch5_stacking_mlp",
    "ens_ch5_entropy_weighted", "ens_ch5_kaggle_optimal_combo",
]
assert len(ALL_VARIANTS) == 50

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Run all 50 variants ───────────────────────────────────────────────────
def run_one(variant: str) -> dict:
    """Run one variant via ensemble_runner.py CLI, return result dict."""
    cmd = ["python", str(SRC_DIR / "ensemble_runner.py"), "--variant", variant]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        if r.returncode != 0:
            return {"variant": variant, "status": "failed",
                    "stderr": (r.stderr or "")[-300:]}
        # Result already saved by ensemble_runner.py
        result_path = SRC_DIR / "ensemble_results" / variant / "result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            data["status"] = "ok"
            return data
        return {"variant": variant, "status": "no_result_file"}
    except Exception as e:
        return {"variant": variant, "status": "exception", "error": str(e)}


def main():
    log(f"### ensemble_runner_all : {len(ALL_VARIANTS)} variants ###")
    log("Step 1 : prep (discover + SCP + cache) — runs as side-effect of first variant")
    log("Step 2 : 50 fusion methods (fast after cache)")

    all_results = []
    for i, variant in enumerate(ALL_VARIANTS, 1):
        log(f"--- [{i}/{len(ALL_VARIANTS)}] {variant} ---")
        result = run_one(variant)
        all_results.append(result)
        if result.get("status") == "ok":
            log(f"    top1={result.get('val_top1', '?'):.4f}  "
                f"top5={result.get('val_top5', '?'):.4f}")
        else:
            log(f"    {result['status']} : {result.get('error', result.get('stderr', ''))[:120]}")

    # Save aggregated results
    summary_path = RESULTS_DIR / "all_50_variants_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=1), encoding="utf-8")
    log(f"\n=== Summary saved : {summary_path} ===")

    # Print ranked table
    log("\n=== Ranked by val_top1 ===")
    ok = [r for r in all_results if r.get("status") == "ok"]
    ok.sort(key=lambda r: -r.get("val_top1", 0))
    for r in ok[:20]:
        log(f"  {r['val_top1']:.4f}  {r['variant']}")
    if len(all_results) - len(ok) > 0:
        log(f"\n{len(all_results) - len(ok)} variants failed")


if __name__ == "__main__":
    main()

"""Compare 8 ensembling strategies on a set of checkpoints in a single pass.

Computes per-model logits on the val set ONCE, then evaluates every fusion
method on top of the cached predictions. Lets us answer "which combination
rule wins on this checkpoint mix?" without re-running inference per method.

Usage (from src/):
    python ensemble_strategies.py \
        training.checkpoints=ckpt_A.pt,ckpt_B.pt,ckpt_C.pt

Fusion methods reported:
  1. mean_softmax        — average of softmax probabilities (the classical default)
  2. mean_logit          — average of raw logits (equivalent if temperatures match)
  3. geometric_mean      — geometric mean of softmax probabilities (= mean log-prob)
  4. max_softmax         — pick the most confident model's prediction per sample
  5. majority_vote       — hard top-1 vote across models, ties broken by mean_softmax
  6. sigmoid_mean        — treat each class independently (mean of per-class sigmoids)
  7. rank_borda          — Borda count: sum per-class ranks across models
  8. confidence_weighted — softmax avg weighted per-sample by each model's top-prob

Optional :
  - training.use_tta=true → adds horizontal-flip TTA per model before fusion
  - training.temperature_scale=true → fits a per-model temperature on val first
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from evaluate import load_model_from_checkpoint
from utils import build_transforms, set_seed


def _topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    _, idx = logits.topk(k, dim=1)
    return float(idx.eq(labels.view(-1, 1)).any(dim=1).sum().item()) / labels.size(0)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                     iters: int = 50, lr: float = 0.01) -> float:
    """Fit a single temperature T on a held-out set by NLL minimisation.
    Returns the optimal T (>0)."""
    T = torch.nn.Parameter(torch.ones(1, device=logits.device))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=iters)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-2), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().item())


def collect_predictions(models: List[torch.nn.Module], loader: DataLoader,
                        device: torch.device, use_tta: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward all models on the loader, return:
       per_model_logits: (M, N, C)
       labels:           (N,)
    """
    all_logits = []  # M lists of (n_batches × (B, C))
    all_labels = []
    for m in models:
        m.eval()
    with torch.no_grad():
        per_model_chunks: list[list[torch.Tensor]] = [[] for _ in models]
        for batch_idx, (videos, labels) in enumerate(loader):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if batch_idx == 0:
                all_labels = [labels]
            else:
                all_labels.append(labels)
            for mi, m in enumerate(models):
                l = m(videos)
                if use_tta:
                    l = (l + m(torch.flip(videos, dims=[-1]))) * 0.5
                per_model_chunks[mi].append(l.detach())
        per_model_logits = torch.stack(
            [torch.cat(chunks, dim=0) for chunks in per_model_chunks],
            dim=0,
        )  # (M, N, C)
        labels_full = torch.cat(all_labels, dim=0)  # (N,)
    return per_model_logits, labels_full


# ── 8 fusion strategies ──────────────────────────────────────────────────────
# All take logits of shape (M, N, C) and return aggregated logits/probs (N, C).

def fuse_mean_softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1).mean(dim=0)

def fuse_mean_logit(logits: torch.Tensor) -> torch.Tensor:
    return logits.mean(dim=0)

def fuse_geometric_mean(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.mean(dim=0)  # treat as logits; argmax is the same as on probs

def fuse_max_softmax(logits: torch.Tensor) -> torch.Tensor:
    """For each sample, pick the model whose max softmax confidence is highest,
    and use ITS softmax as the fused output."""
    probs = F.softmax(logits, dim=-1)                # (M, N, C)
    top_conf = probs.max(dim=-1).values              # (M, N) — per model per sample
    best_model = top_conf.argmax(dim=0)              # (N,)
    N, C = probs.shape[1], probs.shape[2]
    idx = best_model.view(1, N, 1).expand(1, N, C)
    return probs.gather(0, idx).squeeze(0)

def fuse_majority_vote(logits: torch.Tensor) -> torch.Tensor:
    """Hard top-1 vote per sample. Ties broken by mean_softmax."""
    M, N, C = logits.shape
    preds = logits.argmax(dim=-1)                    # (M, N)
    votes = torch.zeros(N, C, device=logits.device)
    votes.scatter_add_(1, preds.t(), torch.ones_like(preds.t(), dtype=votes.dtype))
    # Tie-break: blend with mean_softmax at a tiny weight
    return votes + 1e-3 * fuse_mean_softmax(logits)

def fuse_sigmoid_mean(logits: torch.Tensor) -> torch.Tensor:
    """Treat each class independently — mean of per-class sigmoids."""
    return torch.sigmoid(logits).mean(dim=0)

def fuse_rank_borda(logits: torch.Tensor) -> torch.Tensor:
    """Borda count: each model ranks all classes; sum ranks per class.
    Higher Borda score = better."""
    M, N, C = logits.shape
    ranks = logits.argsort(dim=-1).argsort(dim=-1).float()  # 0..C-1, higher = better
    return ranks.sum(dim=0)

def fuse_confidence_weighted(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample, weight each model by its own max softmax confidence."""
    probs = F.softmax(logits, dim=-1)                # (M, N, C)
    w = probs.max(dim=-1, keepdim=True).values       # (M, N, 1) — confidence as weight
    weighted = (probs * w).sum(dim=0) / w.sum(dim=0).clamp_min(1e-9)
    return weighted


FUSION_METHODS = {
    "mean_softmax":         fuse_mean_softmax,
    "mean_logit":           fuse_mean_logit,
    "geometric_mean":       fuse_geometric_mean,
    "max_softmax":          fuse_max_softmax,
    "majority_vote":        fuse_majority_vote,
    "sigmoid_mean":         fuse_sigmoid_mean,
    "rank_borda":           fuse_rank_borda,
    "confidence_weighted":  fuse_confidence_weighted,
}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths_raw = cfg.training.get("checkpoints", "")
    if not ckpt_paths_raw:
        raise ValueError("Pass training.checkpoints=a.pt,b.pt,... on the command line")
    ckpt_paths: List[Path] = [Path(p.strip()).resolve() for p in str(ckpt_paths_raw).split(",")]
    if len(ckpt_paths) < 2:
        raise ValueError("Need at least 2 checkpoints for ensembling.")

    # Load all models
    models = []
    raws   = []
    for path in ckpt_paths:
        print(f"Loading {path.name}…")
        raw: Dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
        m = load_model_from_checkpoint(raw, device)
        models.append(m)
        raws.append(raw)
    print(f"Ensemble of {len(models)} models on device {device}.")

    use_imagenet_norm = any(
        bool(r.get("use_imagenet_norm", r.get("pretrained", False))) for r in raws
    )
    use_tta = bool(cfg.training.get("use_tta", False))
    temperature_scale = bool(cfg.training.get("temperature_scale", False))

    eval_transform = build_transforms(image_size=224, is_training=False,
                                      use_imagenet_norm=use_imagenet_norm)
    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    val_dataset = VideoFrameDataset(val_dir, num_frames=int(cfg.dataset.num_frames),
                                    transform=eval_transform, sample_list=val_samples)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg.training.batch_size),
                            shuffle=False, num_workers=int(cfg.training.num_workers),
                            pin_memory=(device.type == "cuda"))

    print(f"\nCollecting per-model logits on val ({len(val_dataset)} samples, "
          f"TTA={use_tta}, temp_scale={temperature_scale})...")
    per_model_logits, labels = collect_predictions(models, val_loader, device, use_tta)
    print(f"  logits shape : {tuple(per_model_logits.shape)}")
    print(f"  labels shape : {tuple(labels.shape)}")

    # Per-model standalone metrics
    print("\n── Per-model standalone ──")
    for i, path in enumerate(ckpt_paths):
        l = per_model_logits[i]
        print(f"  {path.name:<55}  top1={_topk_acc(l, labels, 1):.4f}  top5={_topk_acc(l, labels, 5):.4f}")

    # Optional temperature calibration BEFORE fusion (per model on val itself —
    # this is in-sample so it's optimistic; use a held-out split for honest TS).
    if temperature_scale:
        print("\n── Temperature calibration (fit on val itself — optimistic) ──")
        Ts = []
        for i in range(per_model_logits.shape[0]):
            T = _fit_temperature(per_model_logits[i], labels)
            Ts.append(T)
            print(f"  model[{i}] T = {T:.3f}")
        per_model_logits = per_model_logits / torch.tensor(Ts, device=device).view(-1, 1, 1)

    # Evaluate every fusion method
    print("\n── Fusion strategies (sorted by top-1 desc) ──")
    results = []
    for name, fn in FUSION_METHODS.items():
        out = fn(per_model_logits)
        results.append((name, _topk_acc(out, labels, 1), _topk_acc(out, labels, 5)))
    results.sort(key=lambda r: r[1], reverse=True)
    width = max(len(n) for n, _, _ in results)
    for name, t1, t5 in results:
        print(f"  {name:<{width}}  top1={t1:.4f}  top5={t5:.4f}")

    # Cross-model agreement (for diagnostics)
    print("\n── Cross-model agreement (top-1) ──")
    M = per_model_logits.shape[0]
    preds = per_model_logits.argmax(dim=-1)            # (M, N)
    for i in range(M):
        for j in range(i + 1, M):
            agree = float((preds[i] == preds[j]).float().mean().item())
            print(f"  {ckpt_paths[i].name[:30]:<30}  ↔  "
                  f"{ckpt_paths[j].name[:30]:<30}  agree={agree:.3f}")


if __name__ == "__main__":
    main()

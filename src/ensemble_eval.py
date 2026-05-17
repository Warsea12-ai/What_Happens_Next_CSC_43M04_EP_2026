"""
Ensemble evaluation: averages logits from multiple Track A checkpoints.

Each checkpoint must have been saved by train.py (which embeds the full
Hydra config).  Models can have different architectures — only the output
logit dimension must match (33 classes).

Usage (from src/):
    uv run python ensemble_eval.py \\
        --checkpoints best_model_TSM_s_18.yaml.pt best_model_r2plus1d.pt \\
        --val_dir processed_data/val2/val

    # With per-model weights (higher = more influence):
    uv run python ensemble_eval.py \\
        --checkpoints a.pt b.pt c.pt \\
        --weights 2 1 1 \\
        --val_dir processed_data/val2/val
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import build_transforms, set_seed
import train as _train_a

_TRACK_B_MODELS = {
    "videomae", "motion_videomae", "frozen_videomae",
    "swin3d_finetune", "vit_temporal", "qwen_vl_video",
}


def _build_model(cfg, checkpoint: dict, device: torch.device) -> nn.Module:
    name = cfg.model.name
    if name in _TRACK_B_MODELS:
        import train_trackB as _train_b
        model = _train_b.build_model(cfg)
    else:
        model = _train_a.build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"{path}: checkpoint missing 'config' key. "
                         "Retrain with current train.py to embed the config.")
    cfg = OmegaConf.create(ckpt["config"])
    model = _build_model(cfg, ckpt, device)
    use_imagenet_norm = bool(ckpt.get("use_imagenet_norm", True))
    num_frames = int(ckpt.get("num_frames", 4))
    return model, use_imagenet_norm, num_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Paths to checkpoint .pt files",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Per-model weights (default: uniform). Must match --checkpoints length.",
    )
    parser.add_argument(
        "--val_dir", default="processed_data/val2/val",
        help="Directory with validation videos",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tta", action="store_true",
                        help="Average over original + horizontal flip")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = args.weights
    if weights is None:
        weights = [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints):
        raise ValueError("--weights must have the same length as --checkpoints")
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    print(f"Loading {len(args.checkpoints)} models...")
    models = []
    for path, w in zip(args.checkpoints, weights):
        model, use_imagenet_norm, num_frames = load_checkpoint(path, device)
        models.append((model, w, num_frames))
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {Path(path).name}: {type(model).__name__} "
              f"({n_params:.1f}M params, weight={w:.3f})")

    # Use the most common num_frames (should always be 4)
    num_frames_all = [m[2] for m in models]
    if len(set(num_frames_all)) > 1:
        print(f"Warning: models use different num_frames {num_frames_all}. "
              "Using the most common value.")
    num_frames = max(set(num_frames_all), key=num_frames_all.count)

    # All Track A models use imagenet norm (always True now)
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=True)

    val_dir = Path(args.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"\nEvaluating on {len(val_dataset)} samples (num_frames={num_frames})...")

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for video_batch, labels in val_loader:
            video_batch = video_batch.to(device)
            labels = labels.to(device)

            # Weighted average of logits across all models
            ensemble_logits: Optional[torch.Tensor] = None
            for model, w, _ in models:
                if args.use_tta:
                    logits = (model(video_batch) + model(torch.flip(video_batch, dims=[-1]))) * 0.5
                else:
                    logits = model(video_batch)

                if ensemble_logits is None:
                    ensemble_logits = w * logits
                else:
                    ensemble_logits = ensemble_logits + w * logits

            # Top-1
            pred_top1 = ensemble_logits.argmax(dim=1)
            correct_top1 += int((pred_top1 == labels).sum().item())

            # Top-5
            _, pred_top5 = ensemble_logits.topk(5, dim=1)
            correct_top5 += int(pred_top5.eq(labels.view(-1, 1)).any(dim=1).sum().item())

            total += labels.size(0)

    print(f"\nEnsemble results ({len(models)} models, TTA={args.use_tta}):")
    print(f"  Top-1 accuracy: {correct_top1 / total:.4f}")
    print(f"  Top-5 accuracy: {correct_top5 / total:.4f}")

    # Also print individual model accuracies for reference
    print("\nIndividual model accuracies:")
    for path, (model, w, _) in zip(args.checkpoints, models):
        c1, c5, n = 0, 0, 0
        with torch.no_grad():
            for video_batch, labels in val_loader:
                video_batch = video_batch.to(device)
                labels = labels.to(device)
                if args.use_tta:
                    logits = (model(video_batch) + model(torch.flip(video_batch, dims=[-1]))) * 0.5
                else:
                    logits = model(video_batch)
                c1 += int((logits.argmax(1) == labels).sum().item())
                _, p5 = logits.topk(5, dim=1)
                c5 += int(p5.eq(labels.view(-1, 1)).any(dim=1).sum().item())
                n += labels.size(0)
        print(f"  {Path(path).name}: top-1={c1/n:.4f}  top-5={c5/n:.4f}  (weight={w:.3f})")


if __name__ == "__main__":
    main()

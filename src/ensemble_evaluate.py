"""Ensemble evaluation: average logits from multiple checkpoints.

Usage (from src/):
    python ensemble_evaluate.py \
        checkpoints=best_model_A.pt,best_model_B.pt,best_model_C.pt \
        dataset.val_dir=processed_data/val2/val
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from evaluate import load_model_from_checkpoint
from utils import build_transforms, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accept comma-separated list via: checkpoints=a.pt,b.pt,c.pt
    ckpt_paths_raw = cfg.training.get("checkpoints", "")
    if not ckpt_paths_raw:
        raise ValueError("Pass checkpoints=a.pt,b.pt,... on the command line")
    ckpt_paths: List[Path] = [Path(p.strip()).resolve() for p in str(ckpt_paths_raw).split(",")]

    # Load all models
    models = []
    for path in ckpt_paths:
        print(f"Loading {path.name}…")
        raw: Dict[str, Any] = torch.load(path, map_location=device)
        model = load_model_from_checkpoint(raw, device)
        model.eval()
        models.append((model, raw))
    print(f"Ensemble of {len(models)} models.")

    # Use imagenet norm if any model uses it
    use_imagenet_norm = any(
        bool(raw.get("use_imagenet_norm", raw.get("pretrained", False)))
        for _, raw in models
    )
    use_tta = bool(cfg.training.get("use_tta", False))

    eval_transform = build_transforms(image_size=224, is_training=False,
                                      use_imagenet_norm=use_imagenet_norm)
    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    val_dataset = VideoFrameDataset(val_dir, num_frames=int(cfg.dataset.num_frames),
                                    transform=eval_transform, sample_list=val_samples)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg.training.batch_size),
                            shuffle=False, num_workers=int(cfg.training.num_workers),
                            pin_memory=(device.type == "cuda"))

    correct_top1 = correct_top5 = total = 0

    with torch.no_grad():
        for video_batch, labels in val_loader:
            video_batch = video_batch.to(device)
            labels      = labels.to(device)

            # Average logits across all models (and TTA if enabled)
            logits_sum = None
            n = 0
            for model, _ in models:
                l = model(video_batch)
                if use_tta:
                    l = (l + model(torch.flip(video_batch, dims=[-1]))) * 0.5
                logits_sum = l if logits_sum is None else logits_sum + l
                n += 1
            logits = logits_sum / n

            correct_top1 += int((logits.argmax(1) == labels).sum())
            _, top5 = logits.topk(5, dim=1)
            correct_top5 += int(top5.eq(labels.view(-1, 1)).any(dim=1).sum())
            total += labels.size(0)

    print(f"\nEnsemble top-1: {correct_top1/total:.4f}")
    print(f"Ensemble top-5: {correct_top5/total:.4f}")


if __name__ == "__main__":
    main()

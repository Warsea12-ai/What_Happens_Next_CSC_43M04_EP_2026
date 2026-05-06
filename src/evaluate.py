"""
Evaluate a saved checkpoint on the **full** validation split: reports top-1 and top-5 accuracy.

Uses ``dataset.val_dir`` (entire folder; no ``split_train_val``).

Example (from ``src/``)::

    python evaluate.py training.checkpoint_path=best_model.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, set_seed


def load_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Rebuild the model from the Hydra config stored in the checkpoint (same as training).

    Checkpoints must include ``config`` (saved by ``train.py``). No duplicate
    architecture list here—``build_model`` is the single construction site.
    """
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise ValueError(
            "Checkpoint has no 'config' entry. Train with the current train.py so the "
            "full Hydra config is saved with the weights."
        )
    cfg = OmegaConf.create(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    raw: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model = load_model_from_checkpoint(raw, device)

    # Normalization must match how the checkpoint was trained (ImageNet stats if pretrained).
    pretrained_used = bool(raw.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for video_batch, labels in val_loader:
            video_batch = video_batch.to(device)
            labels = labels.to(device)
            logits = model(video_batch)  # (B, num_classes)

            # Top-1: argmax class matches label
            predictions_top1 = logits.argmax(dim=1)
            correct_top1 += int((predictions_top1 == labels).sum().item())

            # Top-5: label appears in the five largest logits per row
            _, predictions_top5 = logits.topk(5, dim=1, largest=True, sorted=True)
            # (B, 5) compared with (B, 1) -> (B, 5) boolean, True if label in top-5
            matches_top5 = predictions_top5.eq(labels.view(-1, 1)).any(dim=1)
            correct_top5 += int(matches_top5.sum().item())

            total += labels.size(0)

    top1_accuracy = correct_top1 / max(total, 1)
    top5_accuracy = correct_top5 / max(total, 1)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")


if __name__ == "__main__":
    main()

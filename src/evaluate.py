"""
Evaluate a saved checkpoint on the **full** validation split: reports top-1 and top-5 accuracy.

Uses ``dataset.val_dir`` (entire folder; no ``split_train_val``).

Example (from ``src/``)::

    uv run python evaluate.py training.checkpoint_path=best_model.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
import train as _train_a
import train_trackB as _train_b
from utils import build_transforms, set_seed

_TRACK_B_MODELS = {
    "videomae", "motion_videomae", "frozen_videomae", "swin3d_finetune",
    "vit_temporal", "qwen_vl_video", "dynamic_videomae", "frame_pair_net",
    "qwen2_vl_7b", "qwen25_7b", "dual_encoder", "cnn_temporal",
    # models added in later experiments
    "bidirectional_frame_pair", "qwen_temporal_attn",
    "videomae_lora", "qwen_lora",
    "videomae_temporal_head", "videomae_domain_adapted",
    "videomae_pair_attn",
    "videomae_cross_attn_pairs",
    "videomae_multiscale",
}


def build_model(cfg):
    name = cfg.model.name
    if name in _TRACK_B_MODELS:
        return _train_b.build_model(cfg)
    return _train_a.build_model(cfg)


def _three_crop(video: torch.Tensor, crop_size: int = 224) -> list:
    """Return top-left, center, bottom-right crops from a (B, T, C, H, W) tensor."""
    H, W = video.shape[-2], video.shape[-1]
    r_offsets = [0, (H - crop_size) // 2, H - crop_size]
    c_offsets = [0, (W - crop_size) // 2, W - crop_size]
    return [video[:, :, :, r:r + crop_size, c:c + crop_size] for r, c in zip(r_offsets, c_offsets)]


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
    raw_cfg = checkpoint["config"]
    # Old checkpoints (pre-Hydra refactor) store a flat model dict with no 'model' subkey.
    # Wrap it into the shape build_model() expects: cfg.model.* and cfg.dataset.num_frames.
    if isinstance(raw_cfg, dict) and "model" not in raw_cfg:
        num_frames = raw_cfg.get("n_segment") or checkpoint.get("num_frames", 4)
        raw_cfg = {"model": raw_cfg, "dataset": {"num_frames": int(num_frames)}}
    cfg = OmegaConf.create(raw_cfg)
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

    # Normalization must exactly match what was used during training.
    # Checkpoints from train_trackA/B.py store use_imagenet_norm explicitly.
    # Older checkpoints fall back to the pretrained flag (original heuristic).
    use_imagenet_norm = bool(
        raw["use_imagenet_norm"] if "use_imagenet_norm" in raw
        else raw["pretrained"] if "pretrained" in raw
        else cfg.model.get("pretrained", True)
    )
    use_multi_crop = bool(cfg.training.get("use_multi_crop", False))
    use_tta = bool(cfg.training.get("use_tta", False))

    # Multi-crop: load at 256×256, crop 3×224×224 at inference; otherwise 224×224.
    eval_size = 256 if use_multi_crop else 224
    eval_transform = build_transforms(image_size=eval_size, is_training=False, use_imagenet_norm=use_imagenet_norm)

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
            labels      = labels.to(device)

            if use_multi_crop:
                crops = _three_crop(video_batch, crop_size=224)
                logits = sum(model(c) for c in crops) / len(crops)
            elif use_tta:
                # Average logits over: original + horizontal flip
                logits_orig = model(video_batch)
                logits_flip = model(torch.flip(video_batch, dims=[-1]))
                logits = (logits_orig + logits_flip) * 0.5
            else:
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

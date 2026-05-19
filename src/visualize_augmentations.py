"""
Visualize data augmentation transformations applied during training.

Two outputs:
  - augmentation_single_video.png: one video, one row per augmentation type
  - augmentation_batch.png: N videos x batch-level augmentations

Usage (from src/):
    uv run python visualize_augmentations.py
    uv run python visualize_augmentations.py --train_dir processed_data/val2/train
    uv run python visualize_augmentations.py --n_videos 4 --output_dir .
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset.video_dataset import _list_frame_paths, _pick_frame_indices, collect_video_samples
from train import (
    color_jitter_video,
    gaussian_blur_video,
    label_aware_horizontal_flip,
    label_aware_temporal_reverse,
    mixup_batch,
)
from utils import build_transforms

NUM_FRAMES = 4
USE_IMAGENET_NORM = False  # matches Track A (pretrained=false)

_train_transform = build_transforms(is_training=True,  use_imagenet_norm=USE_IMAGENET_NORM,
                                    use_random_resized_crop=True)
_eval_transform  = build_transforms(is_training=False, use_imagenet_norm=USE_IMAGENET_NORM)

# Denorm constants matching build_transforms normalization
if USE_IMAGENET_NORM:
    _MEAN = torch.tensor([0.485, 0.456, 0.406])
    _STD  = torch.tensor([0.229, 0.224, 0.225])
else:
    _MEAN = torch.tensor([0.5, 0.5, 0.5])
    _STD  = torch.tensor([0.5, 0.5, 0.5])


def denorm(x: torch.Tensor) -> torch.Tensor:
    """Undo build_transforms normalization. x: (..., C, H, W) -> [..., C, H, W] in [0, 1]."""
    m = _MEAN.view(*([1] * (x.dim() - 3)), 3, 1, 1)
    s = _STD.view(*([1] * (x.dim() - 3)), 3, 1, 1)
    return (x * s + m).clamp(0, 1)


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [0, 1] tensor -> (H, W, 3) uint8 numpy array."""
    return (tensor.float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def load_raw_frames(video_dir: Path, num_frames: int = NUM_FRAMES) -> List[Image.Image]:
    frame_paths = _list_frame_paths(video_dir)
    indices = _pick_frame_indices(len(frame_paths), num_frames)
    frames = []
    for i in indices:
        with Image.open(frame_paths[i]) as img:
            frames.append(img.convert("RGB").copy())
    return frames


def frames_to_tensor(frames: List[Image.Image], transform) -> torch.Tensor:
    """Returns (T, C, H, W) normalized tensor."""
    return torch.stack([transform(f) for f in frames])


def video_to_grid(video: torch.Tensor) -> List[np.ndarray]:
    """(T, C, H, W) [0, 1] tensor -> list of T uint8 HWC arrays."""
    return [to_uint8(video[t]) for t in range(video.shape[0])]


def build_figure(
    rows: List[Tuple[str, List[np.ndarray]]],
    title: str,
    class_name: str,
) -> plt.Figure:
    n_rows = len(rows)
    label_col_w = 2.2
    frame_col_w = 2.2
    n_cols = NUM_FRAMES + 1

    fig = plt.figure(figsize=(label_col_w + NUM_FRAMES * frame_col_w, n_rows * frame_col_w))
    fig.suptitle(f"{title}\nClass: {class_name}", fontsize=10, fontweight="bold", y=1.01)

    gs = fig.add_gridspec(
        n_rows, n_cols,
        width_ratios=[label_col_w] + [frame_col_w] * NUM_FRAMES,
        hspace=0.08,
        wspace=0.04,
    )

    for row_idx, (label, frames) in enumerate(rows):
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.text(
            0.95, 0.5, label,
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.axis("off")

        for t, frame in enumerate(frames):
            ax = fig.add_subplot(gs[row_idx, t + 1])
            ax.imshow(frame)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"Frame {t}", fontsize=8, pad=3)

    return fig


def visualize_one_video(
    video_dir: Path,
    class_name: str,
    output_path: Path,
) -> None:
    raw_frames = load_raw_frames(video_dir)

    original    = frames_to_tensor(raw_frames, _eval_transform)   # (T,C,H,W) normalized
    transformed = frames_to_tensor(raw_frames, _train_transform)  # (T,C,H,W) normalized

    # Denorm to [0,1] for augmentation functions
    batch_01 = denorm(transformed.unsqueeze(0))  # (1,T,C,H,W)

    labels    = torch.tensor([0])
    forbidden = torch.tensor([], dtype=torch.long)

    jittered = color_jitter_video(batch_01.clone(), p=1.0, strength=0.8)[0]
    blurred  = gaussian_blur_video(batch_01.clone(), p=1.0, sigma_range=(1.0, 2.5))[0]

    flipped, _ = label_aware_horizontal_flip(batch_01.clone(), labels, forbidden, p=1.0)

    reverse_lookup = torch.full((33,), -1, dtype=torch.long)
    reverse_lookup[0] = 0
    reversed_, _ = label_aware_temporal_reverse(batch_01.clone(), labels, reverse_lookup, p=1.0)

    rows: List[Tuple[str, List[np.ndarray]]] = [
        ("Original\n(resize only)",                                video_to_grid(denorm(original))),
        ("Train transform\n(crop + h-flip)",                       video_to_grid(denorm(transformed))),
        ("Color jitter\n(brightness/contrast/\nsaturation, s=0.8)", video_to_grid(jittered)),
        ("Gaussian blur\n(sigma in [1.0, 2.5])",                   video_to_grid(blurred)),
        ("Horizontal flip",                                        video_to_grid(flipped[0])),
        ("Temporal reverse\n(frame order reversed)",               video_to_grid(reversed_[0])),
    ]

    fig = build_figure(rows, "Per-video augmentation pipeline", class_name)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_batch_augments(
    samples: List[Tuple[Path, int]],
    class_names: List[str],
    output_path: Path,
    n_videos: int = 4,
) -> None:
    chosen = random.sample(samples, min(n_videos, len(samples)))
    clips, labels_list, names = [], [], []

    for video_dir, label in chosen:
        raw = load_raw_frames(video_dir)
        clips.append(frames_to_tensor(raw, _train_transform))
        labels_list.append(label)
        names.append(class_names[label] if label < len(class_names) else str(label))

    batch_norm = torch.stack(clips)          # (B,T,C,H,W) normalized
    batch_01   = denorm(batch_norm)          # (B,T,C,H,W) in [0,1] for augmentation functions

    # MixUp done locally so we can track the permutation and show both class names
    B = len(chosen)
    mixup_lam  = float(torch.distributions.Beta(0.5, 0.5).sample())
    mixup_perm = torch.randperm(B).tolist()
    mixup_batch_01 = mixup_lam * batch_01 + (1 - mixup_lam) * batch_01[mixup_perm]
    # Label for each MixUp row: "A (λ) × B (1-λ)"
    mixup_row_labels = [
        f"MixUp (λ={mixup_lam:.2f})\n{names[i]}\n× {names[mixup_perm[i]]}"
        for i in range(B)
    ]

    # Non-MixUp augments share the same per-video label format
    simple_augments = {
        "Original":               (batch_01.clone(),  [f"Original\n[{names[i]}]"         for i in range(B)]),
        "Color jitter\n(p=0.8)":  (color_jitter_video(batch_01.clone(), p=0.8, strength=0.8),
                                   [f"Color jitter\n[{names[i]}]"      for i in range(B)]),
        "Gaussian blur\n(p=1.0)": (gaussian_blur_video(batch_01.clone(), p=1.0, sigma_range=(1.0, 2.5)),
                                   [f"Gaussian blur\n[{names[i]}]"     for i in range(B)]),
    }
    mixup_entry = (mixup_batch_01, mixup_row_labels)

    all_groups = list(simple_augments.values()) + [mixup_entry]

    n_aug       = len(all_groups)
    total_rows  = n_videos * n_aug
    label_col_w = 2.4
    frame_col_w = 2.2
    n_cols      = NUM_FRAMES + 1

    fig = plt.figure(figsize=(label_col_w + NUM_FRAMES * frame_col_w, total_rows * 2.0))
    fig.suptitle("Batch-level augmentations", fontsize=11, fontweight="bold", y=1.005)

    gs = fig.add_gridspec(
        total_rows, n_cols,
        width_ratios=[label_col_w] + [frame_col_w] * NUM_FRAMES,
        hspace=0.06,
        wspace=0.04,
    )

    for aug_idx, (aug_batch, row_labels) in enumerate(all_groups):
        for vid_idx in range(n_videos):
            row_idx = aug_idx * n_videos + vid_idx
            frames  = video_to_grid(aug_batch[vid_idx])

            ax_label = fig.add_subplot(gs[row_idx, 0])
            ax_label.text(
                0.95, 0.5, row_labels[vid_idx],
                ha="right", va="center",
                fontsize=7,
                fontweight="bold" if vid_idx == 0 else "normal",
                transform=ax_label.transAxes,
            )
            ax_label.axis("off")

            if vid_idx == 0 and aug_idx > 0:
                ax_label.plot([0, 1], [1, 1], color="gray", linewidth=0.8,
                              transform=ax_label.transAxes, clip_on=False)

            for t, frame in enumerate(frames):
                ax = fig.add_subplot(gs[row_idx, t + 1])
                ax.imshow(frame)
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"Frame {t}", fontsize=8, pad=3)
                if vid_idx == 0 and aug_idx > 0:
                    ax.plot([0, 1], [1, 1], color="gray", linewidth=0.8,
                            transform=ax.transAxes, clip_on=False)

    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize train augmentations")
    parser.add_argument(
        "--train_dir", default="processed_data/val2/train",
        help="Path to the train split root (class folders with video subfolders)",
    )
    parser.add_argument("--n_videos", type=int, default=4, help="Videos shown in batch view")
    parser.add_argument("--output_dir", default=".", help="Directory where PNG files are saved")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir = Path(args.train_dir)
    if not train_dir.is_absolute():
        train_dir = Path(__file__).parent / train_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_video_samples(train_dir)
    class_dirs  = sorted(p for p in train_dir.iterdir() if p.is_dir())
    class_names = [p.name.split("_", 1)[1] if "_" in p.name else p.name for p in class_dirs]

    first_video_dir, first_label = samples[0]
    visualize_one_video(
        first_video_dir,
        class_names[first_label] if first_label < len(class_names) else str(first_label),
        output_dir / "augmentation_single_video.png",
    )

    visualize_batch_augments(
        samples,
        class_names,
        output_dir / "augmentation_batch.png",
        n_videos=args.n_videos,
    )


if __name__ == "__main__":
    main()

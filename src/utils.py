"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
    use_random_resized_crop: bool = False,
) -> transforms.Compose:
    """
    Standard torchvision pipeline for single RGB frames.

    use_imagenet_norm:
        True  -> mean/std from ImageNet (usual when pretrained=True)
        False -> still scale to [0,1]; you can swap norms if you prefer
    use_random_resized_crop:
        True  -> RandomResizedCrop(scale=(0.6, 1.0)) instead of plain Resize
    """
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        spatial = (
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0))
            if use_random_resized_crop
            else transforms.Resize((image_size, image_size))
        )
        return transforms.Compose(
            [
                spatial,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """
    Compute top-k correctness for each k in topk.

    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class indices
    Returns a tuple of tensors, each shape (1,) with accuracy in [0, 1].
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    # (batch_size, max_k) indices of top predictions
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()  # (max_k, batch_size)
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        # Any hit in the top-k row slice counts
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


def measure_grad_norm(parameters) -> float:
    """Total L2 norm of all gradients combined, without clipping.

    Call after ``loss.backward()`` and before ``optimizer.step()``.
    Uses ``clip_grad_norm_`` with ``max_norm=inf`` so gradients are inspected
    but not modified — convenient single source for the actual gradient norm
    even when no clipping is configured.
    """
    return float(
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf")).item()
    )


@torch.no_grad()
def log_wandb_diagnostics(
    model,
    val_loader,
    device,
    num_classes: int,
    prefix: str = "val",
    class_names: List[str] | None = None,
) -> None:
    """Run one validation pass, log confusion matrix + per-class accuracy to wandb.

    Call after restoring the best checkpoint at the end of training. Silently
    no-ops when wandb is not installed or no active run.
    """
    try:
        import wandb  # noqa: WPS433
    except Exception:
        return
    if getattr(wandb, "run", None) is None:
        return

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for video_batch, labels in val_loader:
        video_batch = video_batch.to(device)
        logits = model(video_batch)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    names = class_names or [str(i) for i in range(num_classes)]
    try:
        wandb.log({
            f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels, preds=all_preds, class_names=names,
            ),
        })
    except Exception as exc:
        print(f"[wandb] confusion_matrix skipped ({exc})")

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    per_class = {}
    for c in range(num_classes):
        mask = labels_arr == c
        n = int(mask.sum())
        if n > 0:
            per_class[f"{prefix}_per_class_acc/class_{c}"] = (
                float((preds_arr[mask] == c).sum()) / n
            )
    if per_class:
        wandb.log(per_class)


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Shuffle then split a list of (video_path, label) into train and validation portions.

    Mirrors a standard random hold-out so train.py and evaluate.py stay consistent.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples

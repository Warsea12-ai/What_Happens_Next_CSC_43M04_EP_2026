"""
Track B training script — pretrained models, fine-tuning with differential LR.

Key differences from train_trackA.py:
  - Differential learning rate: backbone gets lr * backbone_lr_scale (e.g. 0.1)
    so pretrained weights are only gently updated while the new head learns faster
  - Longer warmup (10 epochs) to avoid corrupting pretrained weights early on
  - Same augmentation pipeline (flip, color jitter, RandomResizedCrop)

Run from src/::

    python train_trackB.py experiment=track_B_swin
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.swin3d_finetune import VideoSwinFinetune
from models.vit_temporal import ViTTemporal
from models.videomae import VideoMAEClassifier
from models.motion_videomae import MotionVideoMAE
from utils import build_transforms, set_seed, split_train_val


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# ── augmentations (same as train_trackA) ──────────────────────────────────────

@torch.no_grad()
def color_jitter_video(clips: torch.Tensor, p: float = 0.8, strength: float = 0.4) -> torch.Tensor:
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype

    def _factor():
        return 1 + (torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength

    brightness, contrast, saturation = _factor(), _factor(), _factor()
    v = clips * brightness
    mean_per_video = v.mean(dim=(1, 3, 4), keepdim=True)
    v = (v - mean_per_video) * contrast + mean_per_video
    if C == 3:
        gray = (0.299 * v[:, :, 0] + 0.587 * v[:, :, 1] + 0.114 * v[:, :, 2]).unsqueeze(2)
        v = (v - gray) * saturation + gray
    v = v.clamp(0, 1)
    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, v, clips)


@torch.no_grad()
def gaussian_blur_video(
    clips: torch.Tensor, p: float = 0.3,
    sigma_range: Tuple[float, float] = (0.1, 1.5), kernel_size: int = 5,
) -> torch.Tensor:
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype
    half = kernel_size // 2
    sigmas = torch.empty(B, device=device, dtype=dtype).uniform_(*sigma_range)
    x = torch.arange(kernel_size, device=device, dtype=dtype) - half
    g1 = torch.exp(-(x[None, :] ** 2) / (2 * sigmas[:, None] ** 2))
    g1 = g1 / g1.sum(dim=1, keepdim=True)
    kernel2d = g1[:, :, None] * g1[:, None, :]
    kernel = kernel2d[:, None].expand(B, C, kernel_size, kernel_size).reshape(B * C, 1, kernel_size, kernel_size)
    x_in = clips.permute(1, 0, 2, 3, 4).reshape(T, B * C, H, W)
    blurred = F.conv2d(x_in, kernel, padding=half, groups=B * C)
    blurred = blurred.reshape(T, B, C, H, W).permute(1, 0, 2, 3, 4).contiguous()
    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, blurred, clips)


@torch.no_grad()
def label_aware_horizontal_flip(
    clips: torch.Tensor, labels: torch.Tensor,
    forbidden_labels: torch.Tensor, p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = clips.shape[0]
    device = clips.device
    is_forbidden = torch.isin(labels, forbidden_labels)
    should_flip = (torch.rand(B, device=device) < p) & (~is_forbidden)
    flipped = torch.flip(clips, dims=[-1])
    new_clips = torch.where(should_flip.view(B, 1, 1, 1, 1), flipped, clips)
    return new_clips, labels


@torch.no_grad()
def label_aware_temporal_reverse(
    clips: torch.Tensor, labels: torch.Tensor,
    reverse_lookup: torch.Tensor, p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = clips.shape[0]
    device = clips.device
    mapped_labels = reverse_lookup[labels]
    has_mapping = mapped_labels >= 0
    should_rev = (torch.rand(B, device=device) < p) & has_mapping
    reversed_ = torch.flip(clips, dims=[1])
    new_clips = torch.where(should_rev.view(B, 1, 1, 1, 1), reversed_, clips)
    new_labels = torch.where(should_rev, mapped_labels, labels)
    return new_clips, new_labels


def _build_label_lookup(
    label_map: Dict[int, int], num_classes: int, device: torch.device,
) -> torch.Tensor:
    lookup = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    for src, dst in label_map.items():
        lookup[src] = dst
    return lookup


@torch.no_grad()
def video_augment(
    clips: torch.Tensor, *,
    color_prob: float = 0.8, color_strength: float = 0.4,
    blur_prob: float = 0.1,
    input_normalized: bool = True, output_normalized: bool = True,
    mean: torch.Tensor | None = None, std: torch.Tensor | None = None,
) -> torch.Tensor:
    device, dtype = clips.device, clips.dtype
    m = (mean if mean is not None else _IMAGENET_MEAN).to(device, dtype)
    s = (std  if std  is not None else _IMAGENET_STD).to(device, dtype)
    if input_normalized:
        clips = clips * s + m
        clips = clips.clamp(0, 1)
    clips = color_jitter_video(clips, p=color_prob, strength=color_strength)
    if blur_prob > 0:
        clips = gaussian_blur_video(clips, p=blur_prob)
    if output_normalized:
        clips = (clips - m) / s
    return clips


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name
    if name == "swin3d_finetune":
        return VideoSwinFinetune(
            num_classes=int(cfg.model.num_classes),
            pretrained=bool(cfg.model.pretrained),
            dropout=float(cfg.model.get("dropout", 0.5)),
        )
    if name == "vit_temporal":
        return ViTTemporal(
            num_classes=int(cfg.model.num_classes),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            temporal_layers=int(cfg.model.get("temporal_layers", 2)),
            temporal_heads=int(cfg.model.get("temporal_heads", 8)),
        )
    if name == "videomae":
        return VideoMAEClassifier(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            frames_repeat=int(cfg.model.get("frames_repeat", 4)),
        )
    if name == "motion_videomae":
        return MotionVideoMAE(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )
    raise ValueError(f"Unknown model.name for Track B: {name!r}")


# ── optimizer with differential LR ────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    base_lr = float(cfg.training.lr)
    backbone_scale = float(cfg.training.get("backbone_lr_scale", 0.1))
    weight_decay = float(cfg.training.get("weight_decay", 1e-4))

    if hasattr(model, "backbone_parameters") and hasattr(model, "head_parameters"):
        param_groups = [
            {"params": model.backbone_parameters(), "lr": base_lr * backbone_scale},
            {"params": model.head_parameters(),     "lr": base_lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_augment: bool = True,
    aug_kwargs: Dict[str, Any] | None = None,
    flip_forbidden: torch.Tensor | None = None,
    flip_prob: float = 0.5,
    reverse_lookup: torch.Tensor | None = None,
    reverse_prob: float = 0.5,
    use_temporal_map: bool = False,
    clip_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    aug_kwargs = aug_kwargs or {}

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)

        if flip_prob > 0 and flip_forbidden is not None:
            video_batch, labels = label_aware_horizontal_flip(
                video_batch, labels, flip_forbidden, p=flip_prob,
            )
        if use_temporal_map and reverse_lookup is not None:
            video_batch, labels = label_aware_temporal_reverse(
                video_batch, labels, reverse_lookup, p=reverse_prob,
            )
        if use_augment:
            video_batch = video_augment(video_batch, **aug_kwargs)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        loss.backward()
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    print(f"  Train loss: {avg_loss:.4f}, accuracy: {acc:.4f}")
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module, data_loader: DataLoader,
    loss_fn: nn.Module, device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)
        logits = model(video_batch)
        loss = loss_fn(logits, labels)
        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


# ── main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)
    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples, val_ratio=float(cfg.dataset.val_ratio), seed=int(cfg.dataset.seed),
    )

    use_rrc = bool(cfg.training.get("use_random_resized_crop", True))
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=True, use_random_resized_crop=use_rrc,
    )
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=True)

    train_dataset = VideoFrameDataset(
        root_dir=train_dir, num_frames=int(cfg.dataset.num_frames),
        transform=train_transform, sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir, num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform, sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg.training.batch_size), shuffle=True,
        num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    model = build_model(cfg).to(device)

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.training.get("label_smoothing", 0.1))
    )
    optimizer = build_optimizer(model, cfg)

    # Print LR groups
    for i, pg in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in pg["params"])
        print(f"  Param group {i}: {n/1e6:.1f}M params, lr={pg['lr']:.2e}")

    total_epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 10))
    min_lr = float(cfg.training.get("min_lr", 1e-6))
    warmup_sched = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    use_augment = bool(cfg.training.get("use_augment", True))
    aug_kwargs: Dict[str, Any] = {
        "color_prob":       float(cfg.training.get("color_prob",     0.8)),
        "color_strength":   float(cfg.training.get("color_strength", 0.3)),
        "blur_prob":        float(cfg.training.get("blur_prob",      0.1)),
        "input_normalized": True,
        "output_normalized": True,
    }

    forbidden = list(cfg.dataset.get("flip_forbidden_labels", []))
    flip_forbidden = torch.tensor([int(x) for x in forbidden], dtype=torch.long, device=device)
    flip_prob = float(cfg.training.get("flip_prob", 0.5))

    num_classes = int(cfg.model.num_classes)
    reverse_lookup = None
    if cfg.dataset.get("temporal_reverse_map"):
        rev_map = {int(k): int(v) for k, v in cfg.dataset.temporal_reverse_map.items()}
        reverse_lookup = _build_label_lookup(rev_map, num_classes, device)
        print(f"  Temporal reverse equivariance: {len(rev_map)} class pairs")
    reverse_prob = float(cfg.training.get("equivariant_reverse_prob", 0.5))
    use_temporal_map = bool(cfg.training.get("use_temporal_map", True))

    clip_grad_norm = float(cfg.training.get("clip_grad_norm", 1.0))

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[-1]["lr"]  # head LR
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            use_augment=use_augment, aug_kwargs=aug_kwargs,
            flip_forbidden=flip_forbidden, flip_prob=flip_prob,
            reverse_lookup=reverse_lookup, reverse_prob=reverse_prob,
            use_temporal_map=use_temporal_map,
            clip_grad_norm=clip_grad_norm,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{total_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"head_lr {current_lr:.2e}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": num_classes,
                "pretrained": bool(cfg.model.pretrained),
                "use_imagenet_norm": True,  # always True in this script — used by evaluate.py
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path} (val acc={val_acc:.4f})")

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()

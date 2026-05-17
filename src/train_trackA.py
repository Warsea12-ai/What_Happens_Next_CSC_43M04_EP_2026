"""
Track A training script — from scratch only, no pretrained weights.

Key additions over train_augment_3.py:
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - RandomResizedCrop support (via utils.build_transforms)

Run from the ``src/`` directory::

    uv run python train_trackA.py experiment=track_A_best

Or from repo root::

    uv run python src/train_trackA.py experiment=track_A_best
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_temporal import CNNTemporal
from models.video_transformer import VideoTransformer
from models.R2Plus1D import R2Plus1D
from models.TSM import TSM
from models.TSM_RES import TSMResNet50
from models.X3D import X3D
from utils import build_transforms, set_seed, split_train_val


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# ── augmentations ─────────────────────────────────────────────────────────────

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
def motion_blur_video(clips: torch.Tensor, p: float = 0.3, kernel_size: int = 9) -> torch.Tensor:
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype
    K, half = kernel_size, kernel_size // 2
    angles = torch.rand(B, device=device, dtype=dtype) * math.pi
    cos, sin = angles.cos(), angles.sin()
    coords = torch.arange(K, device=device, dtype=dtype) - half
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist = xx[None] * sin[:, None, None] - yy[None] * cos[:, None, None]
    kernel2d = torch.exp(-(dist ** 2) / 1.5)
    kernel2d = kernel2d / kernel2d.sum(dim=(1, 2), keepdim=True)
    kernel = kernel2d[:, None].expand(B, C, K, K).reshape(B * C, 1, K, K)
    x_in = clips.permute(1, 0, 2, 3, 4).reshape(T, B * C, H, W)
    blurred = F.conv2d(x_in, kernel, padding=half, groups=B * C)
    blurred = blurred.reshape(T, B, C, H, W).permute(1, 0, 2, 3, 4).contiguous()
    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, blurred, clips)


@torch.no_grad()
def pixelation_video(
    clips: torch.Tensor, p: float = 0.3, block_range: Tuple[int, int] = (3, 10),
) -> torch.Tensor:
    B, T, C, H, W = clips.shape
    apply = torch.rand(B, device=clips.device) < p
    if not apply.any():
        return clips
    out = clips.clone()
    blocks = torch.randint(block_range[0], block_range[1] + 1, (B,)).tolist()
    for b in range(B):
        if not apply[b]:
            continue
        k = blocks[b]
        small_h, small_w = max(1, H // k), max(1, W // k)
        v = clips[b].reshape(T * C, 1, H, W)
        v = F.adaptive_avg_pool2d(v, (small_h, small_w))
        v = F.interpolate(v, size=(H, W), mode="nearest")
        out[b] = v.reshape(T, C, H, W)
    return out


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
    clips: torch.Tensor,
    labels: torch.Tensor,
    reverse_lookup: torch.Tensor,
    p: float = 0.5,
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
    blur_prob: float = 0.15, motion_blur_prob: float = 0.15,
    pixelation_prob: float = 0.15,
    input_normalized: bool = False, output_normalized: bool = True,
    mean: torch.Tensor | None = None, std: torch.Tensor | None = None,
) -> torch.Tensor:
    device, dtype = clips.device, clips.dtype
    m = (mean if mean is not None else _IMAGENET_MEAN).to(device, dtype)
    s = (std  if std  is not None else _IMAGENET_STD).to(device, dtype)
    if input_normalized:
        clips = clips * s + m
        clips = clips.clamp(0, 1)
    clips = color_jitter_video(clips, p=color_prob, strength=color_strength)
    total = blur_prob + motion_blur_prob + pixelation_prob
    if total > 0:
        B = clips.shape[0]
        r = torch.rand(B, device=device)
        m_blur   = r < blur_prob
        m_motion = (r >= blur_prob) & (r < blur_prob + motion_blur_prob)
        m_pixel  = (r >= blur_prob + motion_blur_prob) & (r < total)
        if m_blur.any():
            transformed = gaussian_blur_video(clips, p=1.0)
            clips = torch.where(m_blur.view(B, 1, 1, 1, 1), transformed, clips)
        if m_motion.any():
            transformed = motion_blur_video(clips, p=1.0)
            clips = torch.where(m_motion.view(B, 1, 1, 1, 1), transformed, clips)
        if m_pixel.any():
            transformed = pixelation_video(clips, p=1.0)
            clips = torch.where(m_pixel.view(B, 1, 1, 1, 1), transformed, clips)
    if output_normalized:
        clips = (clips - m) / s
    return clips


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)

    if name == "video_transformer":
        return VideoTransformer(
            num_classes=num_classes,
            backbone=cfg.model.get("backbone", "resnet34"),
            d_model=int(cfg.model.get("d_model", 256)),
            nhead=int(cfg.model.get("nhead", 4)),
            num_layers=int(cfg.model.get("num_layers", 2)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )

    if name == "cnn_temporal":
        return CNNTemporal(
            num_classes=num_classes,
            backbone=cfg.model.get("backbone", "resnet34"),
            dropout=float(cfg.model.get("dropout", 0.5)),
            head_dim=int(cfg.model.get("head_dim", 512)),
        )

    if name == "cnn_lstm":
        return CNNLSTM(
            num_classes=num_classes, pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 512)),
        )

    if name == "R2Plus1D":
        return R2Plus1D(num_classes=num_classes)

    if name == "TSM":
        return TSM(
            num_classes=num_classes,
            n_segment=cfg.dataset.num_frames,
            n_resnet_layers=cfg.model.get("n_resnet_layers", 34),
            fold_div=cfg.model.get("fold_div", 8),
            residual_shift=cfg.model.get("residual_shift", True),
            use_frame_diff=cfg.model.get("use_frame_diff", True),
            temporal_pool=cfg.model.get("temporal_pool", "attention"),
            use_positional_encoding=cfg.model.get("use_positional_encoding", True),
            pe_mode=cfg.model.get("pe_mode", "sinusoidal"),
        )

    if name == "TSM_RES":
        return TSMResNet50(
            num_classes=num_classes, num_segments=cfg.dataset.num_frames,
            n_div=8, dropout=0.5, pretrained=pretrained,
            consensus=cfg.model.get("consensus", "avg"),
        )

    if name == "X3D":
        return X3D(
            num_classes=cfg.model.get("num_classes", 33),
            variant=cfg.model.get("variant", "xs"),
            input_clip_length=int(cfg.dataset.num_frames),
            input_crop_size=cfg.model.get("input_crop_size", 160),
            use_se=cfg.model.get("use_se", True),
            use_temporal_attention=cfg.model.get("use_temporal_attention", True),
            use_aux_head=cfg.model.get("use_aux_head", True),
            use_frame_diff=cfg.model.get("use_frame_diff", False),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.1),
        )

    raise ValueError(f"Unknown model.name: {name!r}")


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

        if use_temporal_map and reverse_lookup is not None and reverse_prob > 0:
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
        print("CUDA not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)
    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    # Always use ImageNet norm statistics — valid for from-scratch training and
    # required for the video_augment pipeline (which denormalizes to [0,1] first).
    use_rrc = bool(cfg.training.get("use_random_resized_crop", False))
    train_transform = build_transforms(
        is_training=True,
        use_imagenet_norm=True,
        use_random_resized_crop=use_rrc,
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.get("weight_decay", 5e-4)),
    )

    # Cosine LR with linear warmup
    total_epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 5))
    min_lr = float(cfg.training.get("min_lr", 1e-5))
    warmup_sched = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    # Augmentation config
    use_augment = bool(cfg.training.get("use_augment", True))
    aug_kwargs: Dict[str, Any] = {
        "color_prob":        float(cfg.training.get("color_prob",       0.8)),
        "color_strength":    float(cfg.training.get("color_strength",   0.4)),
        "blur_prob":         float(cfg.training.get("blur_prob",        0.15)),
        "motion_blur_prob":  float(cfg.training.get("motion_blur_prob", 0.15)),
        "pixelation_prob":   float(cfg.training.get("pixelation_prob",  0.15)),
        "input_normalized":  True,   # data arrives as ImageNet-normalized [-2,2]
        "output_normalized": True,   # return ImageNet-normalized for the model
    }

    forbidden = list(cfg.dataset.get("flip_forbidden_labels", []))
    flip_forbidden = torch.tensor(
        [int(x) for x in forbidden], dtype=torch.long, device=device,
    )
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
        current_lr = optimizer.param_groups[0]["lr"]
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
            f"lr {current_lr:.2e}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": num_classes,
                "pretrained": bool(cfg.model.pretrained),
                "use_imagenet_norm": True,  # always True in this script — used by evaluate.py
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(payload, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path} (val acc={val_acc:.4f})")

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()

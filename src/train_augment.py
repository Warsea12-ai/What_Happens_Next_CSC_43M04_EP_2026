"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.EarlyVit import EarlyVit
from models.R2Plus1D import R2Plus1D
from models.TSM import TSM
from utils import build_transforms, set_seed, split_train_val
from torchvision.models.video import mvit_v1_b
from models.X3D import X3D 


# =====================================================================
#   AUGMENTATIONS — COLOR JITTER + BLUR (cohérents sur les T frames)
# =====================================================================
@torch.no_grad()
def color_jitter_video(clips: torch.Tensor, strength: float = 0.4) -> torch.Tensor:
    """
    Applique brightness, contrast, saturation avec les MÊMES facteurs sur
    les T frames d'une même vidéo (cohérence temporelle).

    clips : (B, T, C, H, W) en [0, 1]
    """
    B, T, C, H, W = clips.shape
    out = clips.clone()

    for b in range(B):
        # Un facteur tiré par vidéo, partagé sur les T frames
        brightness = 1 + random.uniform(-strength, strength)
        contrast   = 1 + random.uniform(-strength, strength)
        saturation = 1 + random.uniform(-strength, strength)

        v = out[b]                                          # (T, C, H, W)

        # 1. Brightness
        v = v * brightness

        # 2. Contrast : (x - mean) * factor + mean
        mean_per_video = v.mean(dim=(0, 2, 3), keepdim=True)
        v = (v - mean_per_video) * contrast + mean_per_video

        # 3. Saturation : (x - gray) * factor + gray
        gray = (0.299 * v[:, 0] + 0.587 * v[:, 1] + 0.114 * v[:, 2]).unsqueeze(1)
        v = (v - gray) * saturation + gray

        out[b] = v.clamp(0, 1)

    return out


@torch.no_grad()
def gaussian_blur_video(clips: torch.Tensor,
                        sigma_range: Tuple[float, float] = (0.1, 1.5),
                        kernel_size: int = 5) -> torch.Tensor:
    """
    Applique un flou gaussien léger avec le MÊME sigma sur les T frames
    d'une même vidéo. Le sigma diffère entre vidéos.

    clips : (B, T, C, H, W)
    """
    B, T, C, H, W = clips.shape
    out = clips.clone()
    half = kernel_size // 2

    for b in range(B):
        sigma = random.uniform(*sigma_range)
        x = torch.arange(kernel_size, dtype=clips.dtype,
                         device=clips.device) - half
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)

        # On flou les T frames d'un coup en repliant T dans le batch
        v = out[b].reshape(T, C, H, W)
        v = F.conv2d(v, kernel, padding=half, groups=C)
        out[b] = v

    return out


@torch.no_grad()
def video_augment(clips: torch.Tensor,
                  color_prob: float = 0.8, color_strength: float = 0.4,
                  blur_prob: float = 0.3) -> torch.Tensor:
    """
    Pipeline d'augmentation très léger : color jitter + blur seulement.
    À appeler sur GPU dans la boucle d'entraînement.

    clips : (B, T, C, H, W) en [0, 1]
    retour : (B, T, C, H, W) augmenté, toujours en [0, 1]
    """
    if random.random() < color_prob:
        clips = color_jitter_video(clips, strength=color_strength)
    if random.random() < blur_prob:
        clips = gaussian_blur_video(clips)
    return clips


# =====================================================================
#   MODEL FACTORY (inchangé)
# =====================================================================
def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )

    if name == "EarlyVit":
        return EarlyVit(num_classes=cfg.model.num_classes)

    if name == "R2Plus1D":
        return R2Plus1D(num_classes=num_classes)

    if name == "TSM":
        print(cfg.model.get("n_resnet_layers"))
        return TSM(
            num_classes=num_classes,
            n_segment=cfg.dataset.num_frames,
            n_resnet_layers=cfg.model.get("n_resnet_layers", 50),
            fold_div=cfg.model.get("fold_div", 8),
            residual_shift=cfg.model.get("residual_shift", True),
            use_frame_diff=cfg.model.get("use_frame_diff", False),
            temporal_pool=cfg.model.get("temporal_pool", "attention"),
            use_positional_encoding=cfg.model.get("use_positional_encoding", True),
            pe_mode=cfg.model.get("pe_mode", "sinusoidal"),
        )

    if name == "X3D":
        return X3D(
            num_classes=num_classes,
            variant=cfg.model.get("variant", "xs"),
            input_clip_length=int(cfg.dataset.num_frames),
            input_crop_size=cfg.model.get("input_crop_size", 160),
            use_se=cfg.model.get("use_se", True),
            use_temporal_attention=cfg.model.get("use_temporal_attention", True),
            use_aux_head=cfg.model.get("use_aux_head", True),
            use_frame_diff=cfg.model.get("use_frame_diff", False),
            drop_path_rate=cfg.model.get("drop_path_rate", 0.1),
        )
    
    raise ValueError(f"Unknown model.name: {name}")


# =====================================================================
#   TRAIN LOOP — appelle video_augment sur chaque batch
# =====================================================================
def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_augment: bool = False,
    color_strength: float = 0.4,
    blur_prob: float = 0.3,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        # ============ AUGMENTATION : color jitter + blur ============
        if use_augment:
            video_batch = video_augment(
                video_batch,
                color_prob=0.8,
                color_strength=color_strength,
                blur_prob=blur_prob,
            )
        # =============================================================

        optimizer.zero_grad()
        logits = model(video_batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    print(f"  Train loss: {average_loss:.4f}, accuracy: {accuracy:.4f}")
    return average_loss, accuracy


# =====================================================================
#   EVAL LOOP (inchangé — pas d'augmentation à l'évaluation)
# =====================================================================
@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


# =====================================================================
#   MAIN
# =====================================================================
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
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

    use_imagenet_norm = bool(cfg.model.pretrained)
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=use_imagenet_norm
    )
    eval_transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    # ============ Lecture des hyperparamètres d'augmentation ============
    use_augment    = bool(cfg.training.get("use_augment", False))
    color_strength = float(cfg.training.get("color_strength", 0.4))
    blur_prob      = float(cfg.training.get("blur_prob", 0.3))
    if use_augment:
        print(f"  Augmentations : color_jitter (strength={color_strength}) "
              f"+ blur (prob={blur_prob})")
    # =====================================================================

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            use_augment=use_augment,
            color_strength=color_strength,
            blur_prob=blur_prob,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
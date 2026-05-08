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
from models.TSM_s import TSM_s
from utils import build_transforms, set_seed, split_train_val
from torchvision.models.video import mvit_v1_b
from models.X3D import X3D


# =====================================================================
#   AUGMENTATIONS SPATIALES (cohérentes temporellement)
# =====================================================================
@torch.no_grad()
def random_horizontal_flip_video(clips: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Flip horizontal aléatoire, MÊME décision pour les T frames d'un clip.
    Décision indépendante par vidéo dans le batch.

    clips : (B, T, C, H, W)
    """
    B = clips.shape[0]
    out = clips.clone()
    for b in range(B):
        if random.random() < p:
            out[b] = torch.flip(out[b], dims=[-1])  # flip W
    return out


@torch.no_grad()
def random_resized_crop_video(
    clips: torch.Tensor,
    scale: Tuple[float, float] = (0.7, 1.0),
    ratio: Tuple[float, float] = (0.85, 1.15),
) -> torch.Tensor:
    """
    Crop aléatoire (zoom) puis resize à la taille originale.
    MÊME crop pour les T frames d'un clip, crops indépendants entre vidéos.

    clips : (B, T, C, H, W)
    """
    B, T, C, H, W = clips.shape
    out = torch.empty_like(clips)

    for b in range(B):
        # Tirage des paramètres du crop
        for _ in range(10):  # 10 essais pour trouver un crop valide
            target_area = random.uniform(*scale) * H * W
            aspect_ratio = random.uniform(*ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w <= W and 0 < h <= H:
                break
        else:
            # Fallback : center crop si rien ne marche
            w, h = min(W, H), min(W, H)

        i = random.randint(0, H - h)
        j = random.randint(0, W - w)

        # Appliqué identiquement aux T frames
        cropped = clips[b, :, :, i:i + h, j:j + w]              # (T, C, h, w)
        # Resize à (H, W) — interpolate attend (N, C, H, W)
        resized = F.interpolate(
            cropped, size=(H, W), mode="bilinear", align_corners=False
        )
        out[b] = resized

    return out


@torch.no_grad()
def random_temporal_reverse(clips: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Inverse l'ordre temporel des frames avec proba p.
    ATTENTION : ne pas utiliser si les classes dépendent du sens du temps
    (ex. "ouvrir" vs "fermer").

    clips : (B, T, C, H, W)
    """
    B = clips.shape[0]
    out = clips.clone()
    for b in range(B):
        if random.random() < p:
            out[b] = torch.flip(out[b], dims=[0])  # flip T
    return out


# =====================================================================
#   AUGMENTATIONS PHOTOMETRIQUES (cohérentes temporellement)
# =====================================================================
@torch.no_grad()
def color_jitter_video(clips: torch.Tensor, strength: float = 0.4) -> torch.Tensor:
    """
    Brightness, contrast, saturation : MÊMES facteurs sur les T frames.
    clips : (B, T, C, H, W) en [0, 1]
    """
    B, T, C, H, W = clips.shape
    out = clips.clone()

    for b in range(B):
        brightness = 1 + random.uniform(-strength, strength)
        contrast   = 1 + random.uniform(-strength, strength)
        saturation = 1 + random.uniform(-strength, strength)

        v = out[b]                                          # (T, C, H, W)
        v = v * brightness

        mean_per_video = v.mean(dim=(0, 2, 3), keepdim=True)
        v = (v - mean_per_video) * contrast + mean_per_video

        gray = (0.299 * v[:, 0] + 0.587 * v[:, 1] + 0.114 * v[:, 2]).unsqueeze(1)
        v = (v - gray) * saturation + gray

        out[b] = v.clamp(0, 1)

    return out


@torch.no_grad()
def gaussian_blur_video(
    clips: torch.Tensor,
    sigma_range: Tuple[float, float] = (0.1, 1.5),
    kernel_size: int = 5,
) -> torch.Tensor:
    """Flou gaussien, MÊME sigma sur les T frames d'un clip."""
    B, T, C, H, W = clips.shape
    out = clips.clone()
    half = kernel_size // 2

    for b in range(B):
        sigma = random.uniform(*sigma_range)
        x = torch.arange(kernel_size, dtype=clips.dtype, device=clips.device) - half
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d[:, None] * gauss_1d[None, :]
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)

        v = out[b].reshape(T, C, H, W)
        v = F.conv2d(v, kernel, padding=half, groups=C)
        out[b] = v

    return out


@torch.no_grad()
def random_erasing_video(
    clips: torch.Tensor,
    p: float = 0.25,
    scale: Tuple[float, float] = (0.02, 0.15),
    ratio: Tuple[float, float] = (0.3, 3.3),
) -> torch.Tensor:
    """
    Random erasing : efface une zone rectangulaire avec du bruit.
    MÊME zone effacée sur les T frames d'un clip (cohérence spatiale).
    """
    B, T, C, H, W = clips.shape
    out = clips.clone()

    for b in range(B):
        if random.random() >= p:
            continue
        for _ in range(10):
            target_area = random.uniform(*scale) * H * W
            aspect_ratio = random.uniform(*ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < w < W and 0 < h < H:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                noise = torch.rand(T, C, h, w, device=clips.device, dtype=clips.dtype)
                out[b, :, :, i:i + h, j:j + w] = noise
                break

    return out


# =====================================================================
#   MIXUP ENTRE CLIPS (mélange deux vidéos entières)
# =====================================================================
@torch.no_grad()
def mixup_video(
    clips: torch.Tensor, labels: torch.Tensor, num_classes: int, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp entre paires de clips. Retourne (clips_mixés, labels_soft).
    À utiliser avec une cross-entropy soft (cf. mixup_cross_entropy ci-dessous).
    """
    B = clips.shape[0]
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(B, device=clips.device)

    mixed_clips = lam * clips + (1 - lam) * clips[perm]

    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    mixed_labels = lam * one_hot + (1 - lam) * one_hot[perm]

    return mixed_clips, mixed_labels


def mixup_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy avec targets soft (one-hot mixé). Compatible label smoothing."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


# =====================================================================
#   PIPELINE D'AUGMENTATION COMPLET
# =====================================================================
@torch.no_grad()
def video_augment(
    clips: torch.Tensor,
    flip_prob: float = 0.5,
    crop_prob: float = 0.8,
    crop_scale: Tuple[float, float] = (0.7, 1.0),
    temporal_reverse_prob: float = 0.0,
    color_prob: float = 0.8,
    color_strength: float = 0.4,
    blur_prob: float = 0.3,
    erase_prob: float = 0.25,
) -> torch.Tensor:
    """
    Pipeline d'augmentation cohérent temporellement.
    Toutes les opérations spatiales appliquent les MÊMES paramètres aux T frames
    d'un même clip ; les paramètres sont indépendants entre clips du batch.

    Ordre : géométrie spatiale → temporel → photométrie → erasing
    
    clips : (B, T, C, H, W) en [0, 1]
    """
    # 1. Géométrie spatiale
    if crop_prob > 0 and random.random() < crop_prob:
        clips = random_resized_crop_video(clips, scale=crop_scale)
    clips = random_horizontal_flip_video(clips, p=flip_prob)

    # 2. Temporel (désactivé par défaut — n'active que si tes classes sont
    #    invariantes au sens du temps)
    if temporal_reverse_prob > 0:
        clips = random_temporal_reverse(clips, p=temporal_reverse_prob)

    # 3. Photométrie
    if random.random() < color_prob:
        clips = color_jitter_video(clips, strength=color_strength)
    if random.random() < blur_prob:
        clips = gaussian_blur_video(clips)

    # 4. Erasing (en dernier, sur l'image finale)
    if erase_prob > 0:
        clips = random_erasing_video(clips, p=erase_prob)

    return clips


# =====================================================================
#   MODEL FACTORY (inchangé)
# =====================================================================
def build_model(cfg: DictConfig) -> nn.Module:
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
            use_frame_diff=cfg.model.get("use_frame_diff", False),
            temporal_pool=cfg.model.get("temporal_pool", "attention"),
            use_positional_encoding=cfg.model.get("use_positional_encoding", True),
            pe_mode=cfg.model.get("pe_mode", "sinusoidal"),
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
    
    if name == "TSM_s":
        return TSM_s(
            num_classes=num_classes,
            n_segment=4,
            fold_div=8,
            dropout=0.5,
            n_resnet_layers=18,           # commence ici
            temporal_pool="attention",
            use_nonlocal=False,           # à activer plus tard
            use_frame_diff=True,
            use_positional_encoding=True,
            pe_mode="learned",
            stochastic_depth=0.1,
            head_hidden=False,
        )
    raise ValueError(f"Unknown model.name: {name}")


# =====================================================================
#   TRAIN LOOP
# =====================================================================
def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    use_augment: bool = True,
    aug_kwargs: Dict[str, Any] | None = None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    aug_kwargs = aug_kwargs or {}

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ============ Augmentations spatiales + photométriques ============
        if use_augment:
            video_batch = video_augment(video_batch, **aug_kwargs)

        # ============ MixUp (mélange entre clips) =========================
        if use_mixup:
            video_batch, soft_labels = mixup_video(
                video_batch, labels, num_classes=num_classes, alpha=mixup_alpha
            )

        optimizer.zero_grad()
        logits = model(video_batch)

        if use_mixup:
            loss = mixup_cross_entropy(logits, soft_labels) #type: ignore
        else:
            loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())  # comparaison aux labels durs originaux
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    print(f"  Train loss: {average_loss:.4f}, accuracy: {accuracy:.4f}")
    return average_loss, accuracy


# =====================================================================
#   EVAL LOOP (inchangé)
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
    train_transform = build_transforms(is_training=True, use_imagenet_norm=use_imagenet_norm)
    eval_transform  = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)

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

    # Label smoothing : excellent from-scratch, recommandé
    label_smoothing = float(cfg.training.get("label_smoothing", 0.1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Weight decay : utiliser AdamW plutôt que Adam from-scratch
    weight_decay = float(cfg.training.get("weight_decay", 5e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=weight_decay,
    )

    # ============ Lecture des hyperparamètres d'augmentation ============
    use_augment = bool(cfg.training.get("use_augment", False))
    aug_kwargs = {
        "flip_prob":             float(cfg.training.get("flip_prob", 0.5)),
        "crop_prob":             float(cfg.training.get("crop_prob", 0.8)),
        "crop_scale":            tuple(cfg.training.get("crop_scale", [0.7, 1.0])),
        "temporal_reverse_prob": float(cfg.training.get("temporal_reverse_prob", 0.0)),
        "color_prob":            float(cfg.training.get("color_prob", 0.8)),
        "color_strength":        float(cfg.training.get("color_strength", 0.4)),
        "blur_prob":             float(cfg.training.get("blur_prob", 0.3)),
        "erase_prob":            float(cfg.training.get("erase_prob", 0.25)),
    }
    use_mixup   = bool(cfg.training.get("use_mixup", False))
    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.2))

    if use_augment:
        print(f"  Augmentations actives :")
        for k, v in aug_kwargs.items():
            print(f"    - {k}: {v}")
        if use_mixup:
            print(f"    - mixup (alpha={mixup_alpha})")
    # =====================================================================

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            num_classes=int(cfg.model.num_classes),
            use_augment=use_augment,
            aug_kwargs=aug_kwargs,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
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
                payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))

            torch.save(payload, checkpoint_path)
            print(f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})")

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
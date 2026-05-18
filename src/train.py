"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    uv run python train.py
    uv run python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
from models.EarlyVit import EarlyVit
from models.R2Plus1D import R2Plus1D 
from models.TSM import TSM
from models.TSM_s import TSM_s
from models.TSM_RES import TSMResNet50
from models.cnn_modif import cnn_modif
from models.cnn_CLS import cnn_CLS 
from models.Dinov import Dinov
from models.AIMV import AIMV 
from models.cnn_frame_diff import TemporalPositionalEncoding, cnn_frame_diff
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_temporal import CNNTemporal
from models.cnn_hybrid import CNNHybrid
from models.mobilenet_tsm import MobileNetTSM
from models.video_transformer import VideoTransformer
from models.R2Plus1D import R2Plus1D
from models.TSM import TSM
from models.TSM_RES import TSMResNet50
from models.X3D import X3D
from utils import build_transforms, set_seed, split_train_val
from models.swin3d_finetune import VideoSwinFinetune
from models.vit_temporal import ViTTemporal
from torchvision.models.video import mvit_v1_b


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


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
    input_normalized: bool = False, output_normalized: bool = False,
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


@torch.no_grad()
def mixup_batch(
    clips: torch.Tensor, labels: torch.Tensor,
    alpha: float = 0.2, num_classes: int = 33,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MixUp: blends pairs of samples and returns soft labels."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    B = clips.shape[0]
    idx = torch.randperm(B, device=clips.device)
    mixed_clips = lam * clips + (1 - lam) * clips[idx]
    y = torch.zeros(B, num_classes, device=clips.device)
    y.scatter_(1, labels.unsqueeze(1), 1.0)
    mixed_labels = lam * y + (1 - lam) * y[idx]
    return mixed_clips, mixed_labels


@torch.no_grad()
def cutmix_batch(
    clips: torch.Tensor, labels: torch.Tensor,
    alpha: float = 0.4, num_classes: int = 33,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CutMix: paste a random box from a shuffled sample; returns soft labels."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    B, T, C, H, W = clips.shape
    idx = torch.randperm(B, device=clips.device)

    cut_h = int(H * (1 - lam) ** 0.5)
    cut_w = int(W * (1 - lam) ** 0.5)
    cy = int(torch.randint(H, (1,)))
    cx = int(torch.randint(W, (1,)))
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)

    mixed = clips.clone()
    mixed[:, :, :, y1:y2, x1:x2] = clips[idx, :, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)  # actual lambda after clipping

    y = torch.zeros(B, num_classes, device=clips.device)
    y.scatter_(1, labels.unsqueeze(1), 1.0)
    mixed_labels = lam * y + (1 - lam) * y[idx]
    return mixed, mixed_labels


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(
            num_classes=num_classes,
            pretrained=pretrained,
            pool=cfg.model.get("pool", "avg"),
        )
    
    if name == "video_transformer":
        return VideoTransformer(
            num_classes=num_classes,
            backbone=cfg.model.get("backbone", "resnet34"),
            d_model=int(cfg.model.get("d_model", 256)),
            nhead=int(cfg.model.get("nhead", 4)),
            num_layers=int(cfg.model.get("num_layers", 2)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )

    if name == "cnn_hybrid":
        return CNNHybrid(
            num_classes=num_classes,
            backbone=cfg.model.get("backbone", "resnet34"),
            d_model=int(cfg.model.get("d_model", 256)),
            nhead=int(cfg.model.get("nhead", 4)),
            num_layers=int(cfg.model.get("num_layers", 2)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_dim=int(cfg.model.get("head_dim", 512)),
        )

    if name == "cnn_temporal":
        return CNNTemporal(
            num_classes=num_classes,
            backbone=cfg.model.get("backbone", "resnet34"),
            pretrained=pretrained,
            dropout=float(cfg.model.get("dropout", 0.5)),
            head_dim=int(cfg.model.get("head_dim", 512)),
            pretrained_backbone_path=cfg.model.get("pretrained_backbone_path", None),
        )

    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
   
    if name == "cnn_modif":
        return cnn_modif(num_classes=cfg.model.num_classes,)

    if name == "cnn_CLS":
        return cnn_CLS(num_classes=cfg.model.num_classes,)
   
    if name == "EarlyVit":
        return EarlyVit(
            num_classes=cfg.model.num_classes,
        )
    
    if name == "R2Plus1D":
        return R2Plus1D(
            num_classes=num_classes,
            dropout=float(cfg.model.get("dropout", 0.5)),
        )
    
    if name == "TSM":
        print(cfg.model.get("n_resnet_layers") )
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

    if name == "mobilenet_tsm":
        return MobileNetTSM(
            num_classes=num_classes,
            n_segment=int(cfg.dataset.num_frames),
            fold_div=int(cfg.model.get("fold_div", 8)),
            dropout=float(cfg.model.get("dropout", 0.5)),
            temporal_pool=str(cfg.model.get("temporal_pool", "attention")),
            use_positional_encoding=bool(cfg.model.get("use_positional_encoding", True)),
            pe_mode=str(cfg.model.get("pe_mode", "sinusoidal")),
        )

    if name == "TSM_s":
        return TSM_s(
            num_classes=num_classes,
            n_segment=int(cfg.dataset.num_frames),
            fold_div=int(cfg.model.get("fold_div", 8)),
            dropout=float(cfg.model.get("dropout", 0.5)),
            n_resnet_layers=int(cfg.model.get("n_resnet_layers", 18)),
            temporal_pool=str(cfg.model.get("temporal_pool", "attention")),
            use_nonlocal=bool(cfg.model.get("use_nonlocal", False)),
            use_frame_diff=bool(cfg.model.get("use_frame_diff", True)),
            use_positional_encoding=bool(cfg.model.get("use_positional_encoding", True)),
            pe_mode=str(cfg.model.get("pe_mode", "sinusoidal")),
            stochastic_depth=float(cfg.model.get("stochastic_depth", 0.1)),
            head_hidden=bool(cfg.model.get("head_hidden", False)),
            pretrained_backbone_path=cfg.model.get("pretrained_backbone_path", None),
        )

    if name == "TSM_RES":
        return TSMResNet50(
            num_classes=num_classes,
            num_segments=cfg.dataset.num_frames,
            n_div=8,
            dropout=0.5,
            pretrained=pretrained,
            consensus=cfg.model.get("consensus", "avg"),
        )
    
    if name == "Dinov":
        return Dinov(num_classes=num_classes, backbone=cfg.model.get("backbone", "facebook/dinov2-giant"))

    if name =="AIMV":
        return AIMV(num_classes=num_classes,)

    if name == "cnn_frame_diff":
        return cnn_frame_diff(num_classes=num_classes, pretrained=pretrained, pe_mode=cfg.model.get("pe_mode", "sinusoidal"), 
        use_frame_diff=cfg.model.get("use_frame_diff", True), use_positional_encoding=cfg.model.get("use_positional_encoding", True))

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

    raise ValueError(f"Unknown model.name: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float = 0.0,
    use_augment: bool = False,
    aug_kwargs: Dict[str, Any] | None = None,
    flip_forbidden: torch.Tensor | None = None,
    flip_prob: float = 0.5,
    reverse_lookup: torch.Tensor | None = None,
    reverse_prob: float = 0.5,
    use_temporal_map: bool = False,
    use_imagenet_norm: bool = False,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    use_cutmix: bool = False,
    cutmix_alpha: float = 0.4,
    num_classes: int = 33,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    aug_kwargs = aug_kwargs or {}

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        if flip_prob > 0 and flip_forbidden is not None:
            video_batch, labels = label_aware_horizontal_flip(
                video_batch, labels, flip_forbidden, p=flip_prob,
            )
        if use_temporal_map and reverse_lookup is not None:
            video_batch, labels = label_aware_temporal_reverse(
                video_batch, labels, reverse_lookup, p=reverse_prob,
            )
        if use_augment:
            video_batch = video_augment(
                video_batch,
                input_normalized=use_imagenet_norm,
                output_normalized=use_imagenet_norm,
                **aug_kwargs,
            )

        soft_labels = None
        if use_mixup and use_cutmix:
            # Alternate randomly between MixUp and CutMix
            if torch.rand(1).item() < 0.5:
                video_batch, soft_labels = mixup_batch(
                    video_batch, labels, alpha=mixup_alpha, num_classes=num_classes,
                )
            else:
                video_batch, soft_labels = cutmix_batch(
                    video_batch, labels, alpha=cutmix_alpha, num_classes=num_classes,
                )
        elif use_mixup:
            video_batch, soft_labels = mixup_batch(
                video_batch, labels, alpha=mixup_alpha, num_classes=num_classes,
            )
        elif use_cutmix:
            video_batch, soft_labels = cutmix_batch(
                video_batch, labels, alpha=cutmix_alpha, num_classes=num_classes,
            )

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(video_batch)
            if soft_labels is not None:
                # Soft cross-entropy: -sum(soft_labels * log_softmax(logits))
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=1)
                loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:
                loss = loss_fn(logits, labels)

        loss.backward()
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        hard_labels = soft_labels.argmax(dim=1) if soft_labels is not None else labels
        correct += int((predictions == hard_labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    print(f"  Train loss: {average_loss:.4f}, accuracy: {accuracy:.4f}")
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
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

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


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

    # Always use ImageNet normalisation — even from-scratch models benefit from
    # zero-mean unit-variance inputs, and it avoids the clamp(0,1) corruption
    # in video_augment when clips are in the [-1,1] range produced by 0.5/0.5 norm.
    use_imagenet_norm = True
    use_rrc = bool(cfg.training.get("use_random_resized_crop", False))
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=use_imagenet_norm,
        use_random_resized_crop=use_rrc,
    )
    eval_transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm,
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

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    use_class_weights = bool(cfg.training.get("use_class_weights", False))
    if use_class_weights:
        n_cls = int(cfg.model.num_classes)
        counts = torch.zeros(n_cls)
        for _, cls_idx in train_samples:
            counts[cls_idx] += 1
        weights = 1.0 / counts.clamp(min=1)
        weights = (weights / weights.sum() * n_cls).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        print(f"  Class-balanced loss weights — min: {weights.min():.3f}, max: {weights.max():.3f}")
        print(f"  Class counts — min: {int(counts.min())}, max: {int(counts.max())}, mean: {counts.mean():.0f}")
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=weight_decay,
    )

    # Optional cosine LR with linear warmup — disabled when warmup_epochs=0 (default).
    total_epochs  = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    scheduler: Optional[object] = None
    if warmup_epochs > 0:
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

    clip_grad_norm = float(cfg.training.get("clip_grad_norm", 0.0))

    use_augment = bool(cfg.training.get("use_augment", False))
    aug_kwargs: Dict[str, Any] = {
        "color_prob":     float(cfg.training.get("color_prob",     0.8)),
        "color_strength": float(cfg.training.get("color_strength", 0.4)),
        "blur_prob":      float(cfg.training.get("blur_prob",      0.0)),
    }

    num_classes = int(cfg.model.num_classes)
    forbidden = list(cfg.dataset.get("flip_forbidden_labels", []))
    flip_forbidden = torch.tensor([int(x) for x in forbidden], dtype=torch.long, device=device)
    flip_prob = float(cfg.training.get("flip_prob", 0.0))

    reverse_lookup = None
    if cfg.dataset.get("temporal_reverse_map"):
        rev_map = {int(k): int(v) for k, v in cfg.dataset.temporal_reverse_map.items()}
        reverse_lookup = _build_label_lookup(rev_map, num_classes, device)
        print(f"  Temporal reverse equivariance: {len(rev_map)} class pairs")
    reverse_prob = float(cfg.training.get("equivariant_reverse_prob", 0.5))
    use_temporal_map = bool(cfg.training.get("use_temporal_map", False))

    use_mixup    = bool(cfg.training.get("use_mixup",    False))
    mixup_alpha  = float(cfg.training.get("mixup_alpha",  0.2))
    use_cutmix   = bool(cfg.training.get("use_cutmix",   False))
    cutmix_alpha = float(cfg.training.get("cutmix_alpha", 0.4))

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(total_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            clip_grad_norm=clip_grad_norm,
            use_augment=use_augment,
            aug_kwargs=aug_kwargs,
            flip_forbidden=flip_forbidden,
            flip_prob=flip_prob,
            reverse_lookup=reverse_lookup,
            reverse_prob=reverse_prob,
            use_temporal_map=use_temporal_map,
            use_imagenet_norm=use_imagenet_norm,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            use_cutmix=use_cutmix,
            cutmix_alpha=cutmix_alpha,
            num_classes=num_classes,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        if scheduler is not None:
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
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "use_imagenet_norm": use_imagenet_norm,
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

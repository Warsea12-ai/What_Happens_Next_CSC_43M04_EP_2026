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

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
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
from models.video_transformer import VideoTransformer
from models.R2Plus1D import R2Plus1D
from models.TSM import TSM
from models.TSM_RES import TSMResNet50
from models.X3D import X3D
from utils import build_transforms, set_seed, split_train_val
from models.swin3d_finetune import VideoSwinFinetune
from models.vit_temporal import ViTTemporal
from torchvision.models.video import mvit_v1_b


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

    if name == "cnn_temporal":
        return CNNTemporal(
            num_classes=num_classes,
            backbone=cfg.model.get("backbone", "resnet34"),
            pretrained=pretrained,
            dropout=float(cfg.model.get("dropout", 0.5)),
            head_dim=int(cfg.model.get("head_dim", 512)),
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
            num_classes=num_classes
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
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(video_batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
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

    # Match normalization to pretrained flag (ImageNet stats when using pretrained weights).
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

    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
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
    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(total_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            clip_grad_norm=clip_grad_norm,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
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

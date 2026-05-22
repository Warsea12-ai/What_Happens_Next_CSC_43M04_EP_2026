"""
Track B training script — pretrained models, fine-tuning with differential LR.

Key differences from train_trackA.py:
  - Differential learning rate: backbone gets lr * backbone_lr_scale (e.g. 0.1)
    so pretrained weights are only gently updated while the new head learns faster
  - Longer warmup (10 epochs) to avoid corrupting pretrained weights early on
  - Same augmentation pipeline (flip, color jitter, RandomResizedCrop)

Run from src/::

    uv run python train_trackB.py experiment=track_B_swin
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
from models.cnn_temporal import CNNTemporal
from models.swin3d_finetune import VideoSwinFinetune
from models.vit_temporal import ViTTemporal
from models.videomae import VideoMAEClassifier
from models.motion_videomae import MotionVideoMAE
from models.frozen_videomae import FrozenVideoMAELarge
from models.qwen_vl_video import QwenVLVideo
from models.dynamic_videomae import DynamicVideoMAE
from models.frame_pair_net import FramePairNet
from models.dual_encoder import DualEncoder
from models.bidirectional_frame_pair import BidirectionalFramePairNet
from models.qwen_temporal_attn import QwenTemporalAttn
from models.videomae_lora import VideoMAELoRA
from models.qwen_lora import QwenVLLoRA
from models.videomae_temporal_head import VideoMAETemporalHead
from models.videomae_domain_adapted import VideoMAEDomainAdapted
from models.videomae_pair_attn import VideoMAEPairAttn
from models.videomae_cross_attn_pairs import VideoMAECrossAttnPairs
from models.videomae_multiscale import VideoMAEMultiScale
from models.videomae_pairwise_ranking import VideoMAEPairwiseRanking
from models.dinov3_temporal import DINOv3Temporal
from models.videomae_large import VideoMAELarge
from models.vjepa2_head import VJEPA2Head
from models.internvl2_classifier import InternVL2Classifier
from models.internvit6b_temporal import InternViT6BTemporal
from models.internvit6b_pairwise import InternViT6BPairwise
from models.vjepa2_variants import VJEPA2Variant
from models.internvideo2 import InternVideo2
from utils import build_transforms, set_seed, split_train_val

import os
try:
    import wandb
    _WANDB_OK = True
except Exception as _wandb_err:
    print(f"[wandb] import failed ({_wandb_err}) — logging désactivé pour ce run")
    _WANDB_OK = False
    class _WandbStub:
        run = type("R", (), {"summary": {}})()
        def __getattr__(self, _name):
            return lambda *a, **kw: None
    wandb = _WandbStub()


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
    if name == "cnn_temporal":
        return CNNTemporal(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "resnet34")),
            pretrained=bool(cfg.model.get("pretrained", True)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_dim=int(cfg.model.get("head_dim", 512)),
        )
    if name == "swin3d_finetune":
        return VideoSwinFinetune(
            num_classes=int(cfg.model.num_classes),
            pretrained=bool(cfg.model.get("pretrained", True)),
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
            use_linear_interp=bool(cfg.model.get("use_linear_interp", True)),
        )
    if name == "motion_videomae":
        return MotionVideoMAE(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )
    if name == "frozen_videomae":
        return FrozenVideoMAELarge(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            dropout=float(cfg.model.get("dropout", 0.5)),
            num_unfrozen_blocks=int(cfg.model.get("num_unfrozen_blocks", 0)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen_vl_video":
        return QwenVLVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.5)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen2_vl_7b":
        return QwenVLVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-7B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "qwen25_7b":
        return QwenVLVideo(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2.5-VL-7B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 768)),
        )
    if name == "dynamic_videomae":
        return DynamicVideoMAE(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 12)),
            dropout=float(cfg.model.get("dropout", 0.3)),
        )
    if name == "frame_pair_net":
        return FramePairNet(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            n_cross_layers=int(cfg.model.get("n_cross_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "dual_encoder":
        return DualEncoder(
            num_classes=int(cfg.model.num_classes),
            videomae_backbone=str(cfg.model.get("videomae_backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            qwen_backbone=str(cfg.model.get("qwen_backbone", "Qwen/Qwen2-VL-7B-Instruct")),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "bidirectional_frame_pair":
        return BidirectionalFramePairNet(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            n_cross_layers=int(cfg.model.get("n_cross_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen_temporal_attn":
        return QwenTemporalAttn(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-2B-Instruct")),
            n_attn_layers=int(cfg.model.get("n_attn_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "videomae_lora":
        return VideoMAELoRA(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            lora_rank=int(cfg.model.get("lora_rank", 8)),
            lora_alpha=float(cfg.model.get("lora_alpha", 16.0)),
            dropout=float(cfg.model.get("dropout", 0.4)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
        )
    if name == "qwen_lora":
        return QwenVLLoRA(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "Qwen/Qwen2-VL-7B-Instruct")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 768)),
        )
    if name == "videomae_temporal_head":
        return VideoMAETemporalHead(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 12)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            lora_rank=int(cfg.model.get("lora_rank", 0)),
            lora_alpha=float(cfg.model.get("lora_alpha", 16.0)),
        )
    if name == "videomae_domain_adapted":
        return VideoMAEDomainAdapted(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            backbone_path=str(cfg.model.get("backbone_path", "")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 12)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "videomae_pair_attn":
        return VideoMAEPairAttn(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 6)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
        )
    if name == "videomae_cross_attn_pairs":
        return VideoMAECrossAttnPairs(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 2)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "videomae_multiscale":
        return VideoMAEMultiScale(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 2)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 2)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
        )
    if name == "videomae_pairwise_ranking":
        return VideoMAEPairwiseRanking(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 2)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 1024)),
            rank_hidden=int(cfg.model.get("rank_hidden", 64)),
        )
    if name == "dinov3_temporal":
        return DINOv3Temporal(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/dinov3-vith16plus-pretrain-lvd1689m")),
            proj_dim=int(cfg.model.get("proj_dim", 512)),
            n_temporal_layers=int(cfg.model.get("n_temporal_layers", 4)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 512)),
            backbone_dtype=str(cfg.model.get("backbone_dtype", "float32")),
        )
    if name == "videomae_large":
        return VideoMAELarge(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "MCG-NJU/videomae-large-finetuned-kinetics")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 4)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
        )
    if name == "vjepa2_head":
        return VJEPA2Head(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/vjepa2-vitg-fpc64-384-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 32)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
        )
    if name == "internvideo2":
        return InternVideo2(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternVideo2-Stage2_6B")),
            dropout=float(cfg.model.get("dropout", 0.3)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            backbone_dtype=str(cfg.model.get("backbone_dtype", "bfloat16")),
        )
    if name == "internvl2_classifier":
        return InternVL2Classifier(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternVL2-8B")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            num_frozen_llm_layers=int(cfg.model.get("num_frozen_llm_layers", 24)),
        )
    if name == "internvit6b_temporal":
        return InternViT6BTemporal(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternViT-6B-448px-V2_5")),
            proj_dim=int(cfg.model.get("proj_dim", 512)),
            n_set_layers=int(cfg.model.get("n_set_layers", 3)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
        )
    if name == "vjepa2_variants":
        return VJEPA2Variant(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "facebook/vjepa2-vitg-fpc64-384-ssv2")),
            num_frozen_blocks=int(cfg.model.get("num_frozen_blocks", 32)),
            dropout=float(cfg.model.get("dropout", 0.25)),
            head_hidden=int(cfg.model.get("head_hidden", 2048)),
            variant=str(cfg.model.get("variant", "lora")),
            lora_rank=int(cfg.model.get("lora_rank", 16)),
            lora_alpha=float(cfg.model.get("lora_alpha", 32.0)),
        )
    if name == "internvit6b_pairwise":
        return InternViT6BPairwise(
            num_classes=int(cfg.model.num_classes),
            backbone=str(cfg.model.get("backbone", "OpenGVLab/InternViT-6B-448px-V2_5")),
            proj_dim=int(cfg.model.get("proj_dim", 512)),
            n_heads=int(cfg.model.get("n_heads", 8)),
            dropout=float(cfg.model.get("dropout", 0.25)),
        )
    raise ValueError(f"Unknown model.name for Track B: {name!r}")


# ── optimizer with differential LR ────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    base_lr = float(cfg.training.lr)
    backbone_scale = float(cfg.training.get("backbone_lr_scale", 0.1))
    weight_decay = float(cfg.training.get("weight_decay", 1e-4))
    llrd = float(cfg.training.get("llrd_factor", 0.0))  # 0 = disabled

    # Layer-wise LR decay: each transformer block gets a geometrically smaller LR
    # moving from the top of the network toward the bottom. This prevents early
    # (more general) layers from being over-trained while adapting later layers.
    if llrd > 0 and hasattr(model, "layerwise_lr_groups"):
        param_groups = model.layerwise_lr_groups(
            head_lr=base_lr,
            backbone_lr=base_lr * backbone_scale,
            llrd=llrd,
        )
    elif hasattr(model, "head_parameters"):
        # head_parameters() always present; backbone_parameters() may return [] (frozen model)
        backbone_params = model.backbone_parameters() if hasattr(model, "backbone_parameters") else []
        lora_params = model.lora_parameters() if hasattr(model, "lora_parameters") else []
        param_groups = [{"params": model.head_parameters(), "lr": base_lr}]
        if lora_params:
            # LoRA adapters train at backbone_lr_scale (gentler than head, faster than 0)
            param_groups.append({"params": lora_params, "lr": base_lr * backbone_scale})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * backbone_scale})
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
    grad_accum_steps: int = 1,
) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    aug_kwargs = aug_kwargs or {}
    n_batches = len(data_loader)

    optimizer.zero_grad()
    for step, (video_batch, labels) in enumerate(data_loader):
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

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(video_batch)
            loss = loss_fn(logits, labels) / grad_accum_steps

        loss.backward()

        is_last_batch = (step + 1 == n_batches)
        if (step + 1) % grad_accum_steps == 0 or is_last_batch:
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += float(loss.item()) * grad_accum_steps * labels.size(0)
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

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "What_Happens_Next_CSC_43M04_EP_2026"),
        name=os.environ.get("WANDB_RUN_NAME"),
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )

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
        temporal_jitter=int(cfg.training.get("temporal_jitter", 0)),
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
    grad_accum_steps = int(cfg.training.get("grad_accum_steps", 1))
    if grad_accum_steps > 1:
        print(f"  Gradient accumulation: {grad_accum_steps} steps "
              f"(effective batch size = {int(cfg.training.batch_size) * grad_accum_steps})")

    # Gradual unfreezing schedule: list of [epoch, n_frozen_blocks] pairs
    unfreeze_sched: List[Tuple[int, int]] = []
    if cfg.training.get("unfreeze_schedule"):
        unfreeze_sched = sorted(
            [(int(p[0]), int(p[1])) for p in cfg.training.unfreeze_schedule],
            key=lambda x: x[0],
        )

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    recent_save_accs: list = []
    epochs_since_save = 0
    start_epoch = 0

    if checkpoint_path.exists():
        print(f"  Checkpoint trouvé : {checkpoint_path} — reprise de l'entraînement...")
        _ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model_state_dict"])
        if "optimizer_state_dict" in _ckpt:
            optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
        if _ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(_ckpt["scheduler_state_dict"])
        start_epoch       = _ckpt.get("epoch", 0)
        best_val_accuracy = _ckpt.get("best_val_accuracy", _ckpt.get("val_accuracy", 0.0))
        recent_save_accs  = _ckpt.get("recent_save_accs", [])
        epochs_since_save = _ckpt.get("epochs_since_save", 0)
        print(f"  Reprise depuis epoch {start_epoch}/{total_epochs} (best val acc={best_val_accuracy:.4f})")

    for epoch in range(start_epoch, total_epochs):
        # Apply gradual unfreezing at scheduled epochs
        for sched_epoch, n_frozen in unfreeze_sched:
            if epoch == sched_epoch and hasattr(model, "unfreeze_to"):
                model.unfreeze_to(n_frozen)
                optimizer = build_optimizer(model, cfg)
                remaining = max(1, total_epochs - epoch)
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=min_lr)
                print(f"  Epoch {epoch+1}: encoder thawed to {n_frozen} frozen blocks, optimizer rebuilt")
                break

        current_lr = optimizer.param_groups[-1]["lr"]  # head LR
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            use_augment=use_augment, aug_kwargs=aug_kwargs,
            flip_forbidden=flip_forbidden, flip_prob=flip_prob,
            reverse_lookup=reverse_lookup, reverse_prob=reverse_prob,
            use_temporal_map=use_temporal_map,
            clip_grad_norm=clip_grad_norm,
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{total_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"head_lr {current_lr:.2e}"
        )
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "head_lr": current_lr,
        }, step=epoch + 1)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            recent_save_accs.append(val_acc)
            epochs_since_save = 0
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":                epoch + 1,
                "best_val_accuracy":    best_val_accuracy,
                "recent_save_accs":     recent_save_accs,
                "epochs_since_save":    0,
                "model_name":           cfg.model.name,
                "num_classes":          num_classes,
                "pretrained":           bool(cfg.model.get("pretrained", True)),
                "use_imagenet_norm":    True,
                "num_frames":           int(cfg.dataset.num_frames),
                "val_accuracy":         val_acc,
                "config":               OmegaConf.to_container(cfg, resolve=True),
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path} (epoch={epoch + 1}, val acc={val_acc:.4f})")
            wandb.run.summary["best_val_acc"] = val_acc
        else:
            epochs_since_save += 1

        stop = False
        if epochs_since_save >= 6:
            print("  Early stop: no model saved in the last 6 epochs.")
            stop = True
        if len(recent_save_accs) >= 3 and recent_save_accs[-1] - recent_save_accs[-3] < 0.01:
            print("  Early stop: val accuracy gain over last 3 saves < 1%.")
            stop = True
        if stop:
            break

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()

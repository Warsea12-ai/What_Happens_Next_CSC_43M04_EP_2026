"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pipeline d'augmentation, dans l'ordre par batch :

    1. label_aware_horizontal_flip   (change le contenu, labels inchangés
       pour les classes autorisées ; pas appliqué aux labels interdits)
    2. video_augment                 (color jitter + UNE distorsion de netteté
       parmi blur / motion blur / pixelation, par vidéo)
    3. mixup                         (mélange entre clips, produit labels soft)
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
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.EarlyVit import EarlyVit
from models.R2Plus1D import R2Plus1D
from models.TSM import TSM
from models.TSM_s import TSM_s
from models.TSM_RES import TSMResNet50
from models.X3D import X3D
from utils import build_transforms, set_seed, split_train_val


# Stats ImageNet, utilisées si on dénormalise en début de pipeline
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# =====================================================================
#   AUGMENTATIONS PHOTOMÉTRIQUES (label-preserving, sur [0, 1])
# =====================================================================
@torch.no_grad()
def color_jitter_video(clips: torch.Tensor, p: float = 0.8, strength: float = 0.4) -> torch.Tensor:
    """Brightness / contrast / saturation, MEMES facteurs sur les T frames."""
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
    """Flou gaussien, sigma par vidéo, une seule conv groupée pour tout le batch."""
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
    """Flou directionnel, angle par vidéo (effet 'caméra qui bouge')."""
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
    """Effet pixelisé : downsample agressif puis nearest-upsample, par vidéo."""
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


# =====================================================================
#   AUGMENTATION ÉQUIVARIANTE (touche aux labels)
# =====================================================================
@torch.no_grad()
def label_aware_horizontal_flip(
    clips: torch.Tensor, labels: torch.Tensor,
    forbidden_labels: torch.Tensor, p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symétrie miroir gauche/droite (flip de W). Appliqué à toutes les vidéos
    SAUF celles dont le label figure dans `forbidden_labels`. Labels inchangés."""
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
    """Inversion temporelle équivariante. Même logique que le flip mais sur la
    dimension T au lieu de W."""
    B = clips.shape[0]
    device = clips.device

    mapped_labels = reverse_lookup[labels]
    has_mapping   = mapped_labels >= 0
    should_rev    = (torch.rand(B, device=device) < p) & has_mapping

    reversed_ = torch.flip(clips, dims=[1])                      # flip de T
    new_clips  = torch.where(should_rev.view(B, 1, 1, 1, 1), reversed_, clips)
    new_labels = torch.where(should_rev, mapped_labels, labels)

    return new_clips, new_labels

def _build_label_lookup(
    label_map: Dict[int, int], num_classes: int, device: torch.device,
) -> torch.Tensor:
    """Dict → tenseur de lookup (num_classes,). Indice non mappé → -1."""
    lookup = torch.full((num_classes,), -1, dtype=torch.long, device=device)
    for src, dst in label_map.items():
        if not (0 <= src < num_classes) or not (0 <= dst < num_classes):
            raise ValueError(f"Mapping {src}->{dst} hors plage [0, {num_classes})")
        lookup[src] = dst
    return lookup


# =====================================================================
#   PIPELINE LABEL-PRESERVING
# =====================================================================
@torch.no_grad()
def video_augment(
    clips: torch.Tensor, *,
    color_prob: float = 0.8,
    color_strength: float = 0.4,
    blur_prob: float = 0.15,
    motion_blur_prob: float = 0.15,
    pixelation_prob: float = 0.15,
    input_normalized: bool = False,
    output_normalized: bool = True,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pipeline d'augmentation label-preserving. Travaille en [0, 1] en interne,
    quel que soit l'état d'entrée (normalisé ou pas).

    Ordre :
        1. Color jitter (toujours appliqué avec proba color_prob par vidéo)
        2. UNE distorsion de netteté par vidéo, parmi :
              - gaussian blur (proba blur_prob)
              - motion blur   (proba motion_blur_prob)
              - pixelation    (proba pixelation_prob)
              - rien          (proba 1 - somme)
           Les 3 probabilités sont MUTUELLEMENT EXCLUSIVES par vidéo.
        3. Normalisation finale si output_normalized=True.

    Args:
        clips:             (B, T, C, H, W)
        input_normalized:  True si le dataset normalise déjà les frames (cas
                           build_transforms(use_imagenet_norm=True)). On
                           dénormalise alors avant d'augmenter.
        output_normalized: True pour renvoyer du tensor normalisé au modèle.
    """
    device, dtype = clips.device, clips.dtype
    m = (mean if mean is not None else _IMAGENET_MEAN).to(device, dtype)
    s = (std  if std  is not None else _IMAGENET_STD ).to(device, dtype)

    # 0. Ramener en [0, 1] pour que color_jitter / pixelation / blurs aient du sens
    if input_normalized:
        clips = clips * s + m
        clips = clips.clamp(0, 1)

    # 1. Couleur
    clips = color_jitter_video(clips, p=color_prob, strength=color_strength)

    # 2. Choix EXCLUSIF d'une distorsion de netteté par vidéo
    total = blur_prob + motion_blur_prob + pixelation_prob
    if total > 1 + 1e-6:
        raise ValueError(
            f"blur_prob + motion_blur_prob + pixelation_prob doit être ≤ 1, "
            f"reçu {total:.3f}"
        )
    if total > 0:
        B = clips.shape[0]
        r = torch.rand(B, device=device)

        # Construire les masques mutuellement exclusifs sans appliquer les
        # transfos sur les vidéos qui ne les reçoivent pas
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

    # 3. Normalisation finale
    if output_normalized:
        clips = (clips - m) / s

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
        return CNNLSTM(
            num_classes=num_classes, pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 512)),
        )
    if name == "EarlyVit":
        return EarlyVit(num_classes=num_classes)
    if name == "R2Plus1D":
        return R2Plus1D(num_classes=num_classes)
    if name == "TSM":
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
            num_classes=num_classes, n_segment=4, fold_div=8, dropout=0.5,
            n_resnet_layers=18, temporal_pool="attention", use_nonlocal=False,
            use_frame_diff=True, use_positional_encoding=True, pe_mode="learned",
            stochastic_depth=0.1, head_hidden=False,
        )
    if name == "TSM_RES":
        return TSMResNet50(
            num_classes=num_classes, num_segments=cfg.dataset.num_frames,
            n_div=8, dropout=0.5, pretrained=pretrained,
            consensus=cfg.model.get("consensus", "avg"),
        )
    raise ValueError(f"Unknown model.name: {name}")


# =====================================================================
#   TRAIN LOOP — intègre flip label-aware + video_augment + mixup
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
    mixup_alpha: float = 0.2,
    flip_forbidden: torch.Tensor | None = None,
    flip_prob: float = 0.0,
    reverse_lookup = torch.Tensor | None = None, 
    reverse_prob: float = 0.5,

) -> Tuple[float, float]:

    """Une epoch d'entraînement. Ordre strict des augmentations :
       flip label-aware → video_augment → mixup.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    aug_kwargs = aug_kwargs or {}

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels      = labels.to(device, non_blocking=True)

        # 1. FLIP LABEL-AWARE (avant toute autre transfo : il opère sur la
        #    structure géométrique, le reste opère sur les pixels)
        if flip_prob > 0:
            video_batch, labels = label_aware_horizontal_flip(
                video_batch, labels, flip_forbidden, p=flip_prob,
            )
        
        if reverse_lookup is not None and reverse_prob > 0:
            video_batch, labels = label_aware_temporal_reverse(
                video_batch, labels, reverse_lookup, p=reverse_prob
            )

        # 2. AUGMENTATIONS PHOTOMÉTRIQUES (label-preserving, normalise en sortie)
        if use_augment:
            video_batch = video_augment(video_batch, **aug_kwargs)

        optimizer.zero_grad()
        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    print(f"  Train loss: {avg_loss:.4f}, accuracy: {acc:.4f}")
    return avg_loss, acc


# =====================================================================
#   EVAL LOOP (inchangé)
# =====================================================================
@torch.no_grad()
def evaluate_epoch(model, data_loader, loss_fn, device) -> Tuple[float, float]:
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
    train_transform = build_transforms(is_training=True,  use_imagenet_norm=use_imagenet_norm)
    eval_transform  = build_transforms(is_training=False, use_imagenet_norm=use_imagenet_norm)

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
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)

    label_smoothing = float(cfg.training.get("label_smoothing", 0.1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    weight_decay = float(cfg.training.get("weight_decay", 5e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=weight_decay,
    )

    # ===== Lecture des hyperparamètres d'augmentation =====
    use_augment = bool(cfg.training.get("use_augment", False))
    aug_kwargs: Dict[str, Any] = {
        "color_prob":        float(cfg.training.get("color_prob",       0.8)),
        "color_strength":    float(cfg.training.get("color_strength",   0.4)),
        "blur_prob":         float(cfg.training.get("blur_prob",        0.15)),
        "motion_blur_prob":  float(cfg.training.get("motion_blur_prob", 0.15)),
        "pixelation_prob":   float(cfg.training.get("pixelation_prob",  0.15)),
        "input_normalized":  use_imagenet_norm,   # le dataset normalise déjà si pretrained
        "output_normalized": use_imagenet_norm,   # ce qu'attend le modèle
    }
    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.2))

    # ===== Flip label-aware =====
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
        print(f"  Reverse équivariant : {len(rev_map)} labels mappés")
    reverse_prob = float(cfg.training.get("equivariant_reverse_prob", 0.5))

    if use_augment:
        print("  Augmentations actives :")
        for k, v in aug_kwargs.items():
            print(f"    - {k}: {v}")
    if flip_prob > 0:
        if len(forbidden) > 0:
            print(f"  Flip horizontal (gauche↔droite) : p={flip_prob}, "
                  f"désactivé pour labels {forbidden}")
        else:
            print(f"  Flip horizontal (gauche↔droite) : p={flip_prob}, "
                  f"toutes classes autorisées")

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            num_classes=int(cfg.model.num_classes),
            use_augment=use_augment, aug_kwargs=aug_kwargs,
            flip_forbidden=flip_forbidden, flip_prob=flip_prob,
            reverse_lookup = reverse_lookup, reverse_prob = reverse_prob,
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
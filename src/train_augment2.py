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

from typing import Tuple

import torch
import torch.nn.functional as F

@torch.no_grad()
def color_jitter_video(
    clips: torch.Tensor, p: float = 0.8, strength: float = 0.4,
) -> torch.Tensor:
    """Brightness / contrast / saturation, MEMES facteurs sur les T frames
    d'un clip. Suppose clips ∈ [0, 1]."""
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype

    def _factor():
        return 1 + (torch.rand(B, 1, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * strength

    brightness = _factor()
    contrast   = _factor()
    saturation = _factor()

    v = clips * brightness

    # Contrast : autour de la moyenne par vidéo
    mean_per_video = v.mean(dim=(1, 3, 4), keepdim=True)         # (B, 1, C, 1, 1)
    v = (v - mean_per_video) * contrast + mean_per_video

    # Saturation : autour du gris (suppose RGB)
    if C == 3:
        gray = (
            0.299 * v[:, :, 0] + 0.587 * v[:, :, 1] + 0.114 * v[:, :, 2]
        ).unsqueeze(2)                                            # (B, T, 1, H, W)
        v = (v - gray) * saturation + gray

    v = v.clamp(0, 1)

    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, v, clips)


@torch.no_grad()
def gaussian_blur_video(
    clips: torch.Tensor,
    p: float = 0.3,
    sigma_range: Tuple[float, float] = (0.1, 1.5),
    kernel_size: int = 5,
) -> torch.Tensor:
    """Flou gaussien avec sigma différent par vidéo, en une seule conv groupée."""
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype
    half = kernel_size // 2

    # Sigma par vidéo, noyau 1D puis 2D séparable
    sigmas = torch.empty(B, device=device, dtype=dtype).uniform_(*sigma_range)
    x = (torch.arange(kernel_size, device=device, dtype=dtype) - half)         # (K,)
    g1 = torch.exp(-(x[None, :] ** 2) / (2 * sigmas[:, None] ** 2))             # (B, K)
    g1 = g1 / g1.sum(dim=1, keepdim=True)
    kernel2d = g1[:, :, None] * g1[:, None, :]                                  # (B, K, K)

    # Un noyau par (vidéo, canal) — convolution groupée
    kernel = (
        kernel2d[:, None, :, :].expand(B, C, kernel_size, kernel_size)
        .reshape(B * C, 1, kernel_size, kernel_size)
    )

    # (B, T, C, H, W) → (T, B*C, H, W) pour passer en une seule conv2d
    x_in = clips.permute(1, 0, 2, 3, 4).reshape(T, B * C, H, W)
    blurred = F.conv2d(x_in, kernel, padding=half, groups=B * C)
    blurred = blurred.reshape(T, B, C, H, W).permute(1, 0, 2, 3, 4).contiguous()

    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, blurred, clips)


@torch.no_grad()
def pixelation_video(
    clips: torch.Tensor,
    p: float = 0.3,
    block_range: Tuple[int, int] = (3, 10),
) -> torch.Tensor:
    """Effet pixelisé : downsample agressif puis nearest-upsample.
    Taille de bloc différente par vidéo, MEME bloc sur les T frames."""
    B, T, C, H, W = clips.shape
    out = clips.clone()

    # Choix d'appliquer ou non, et facteur, par vidéo
    apply = torch.rand(B, device=clips.device) < p
    blocks = torch.randint(block_range[0], block_range[1] + 1, (B,)).tolist()

    for b in range(B):
        if not apply[b]:
            continue
        k = blocks[b]
        small_h, small_w = max(1, H // k), max(1, W // k)
        v = clips[b].reshape(T * C, 1, H, W)
        v = F.adaptive_avg_pool2d(v, (small_h, small_w))      # downsample
        v = F.interpolate(v, size=(H, W), mode="nearest")     # upsample nearest
        out[b] = v.reshape(T, C, H, W)

    return out

@torch.no_grad()
def motion_blur_video(
    clips: torch.Tensor,
    p: float = 0.3,
    kernel_size: int = 9,
) -> torch.Tensor:
    """Flou directionnel (effet 'mouvement de caméra').
    Angle et intensité différents par vidéo."""
    B, T, C, H, W = clips.shape
    device, dtype = clips.device, clips.dtype
    K = kernel_size
    half = K // 2

    # Angle uniforme dans [0, π) par vidéo (π suffit, symétrie de la ligne)
    angles = torch.rand(B, device=device, dtype=dtype) * math.pi
    cos, sin = angles.cos(), angles.sin()

    # Coordonnées centrées sur le noyau
    coords = torch.arange(K, device=device, dtype=dtype) - half        # (K,)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")             # (K, K)

    # Distance signée à la ligne passant par le centre, d'angle theta
    # ligne : x*sin - y*cos = 0  ; on prend exp(-(distance**2)/sigma**2)
    dist = xx[None] * sin[:, None, None] - yy[None] * cos[:, None, None]  # (B, K, K)
    kernel2d = torch.exp(-(dist ** 2) / 1.5)
    kernel2d = kernel2d / kernel2d.sum(dim=(1, 2), keepdim=True)

    # Kernel par (vidéo, canal), conv groupée
    kernel = kernel2d[:, None, :, :].expand(B, C, K, K).reshape(B * C, 1, K, K)
    x_in = clips.permute(1, 0, 2, 3, 4).reshape(T, B * C, H, W)
    blurred = F.conv2d(x_in, kernel, padding=half, groups=B * C)
    blurred = blurred.reshape(T, B, C, H, W).permute(1, 0, 2, 3, 4).contiguous()

    apply_mask = torch.rand(B, 1, 1, 1, 1, device=device) < p
    return torch.where(apply_mask, blurred, clips)

@torch.no_grad()
def label_aware_horizontal_flip(
    clips: torch.Tensor,
    labels: torch.Tensor,
    forbidden_labels: torch.Tensor,
    p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flip horizontal appliqué à toutes les vidéos SAUF celles dont le label
    figure dans `forbidden_labels`. Le label reste inchangé pour les vidéos
    flippées (= classes considérées comme invariantes par symétrie miroir).

    Args:
        clips:            (B, T, C, H, W)
        labels:           (B,) entiers
        forbidden_labels: 1D tensor des labels interdits au flip (ex: [18, 19])
        p:                proba d'appliquer le flip parmi les vidéos autorisées
    """
    B = clips.shape[0]
    device = clips.device

    # Masque des vidéos autorisées au flip
    is_forbidden = torch.isin(labels, forbidden_labels)
    allowed = ~is_forbidden                                       # (B,)

    # Décision aléatoire par vidéo, conditionnée
    should_flip = (torch.rand(B, device=device) < p) & allowed

    flipped = torch.flip(clips, dims=[-1])                        # flip de W
    new_clips = torch.where(should_flip.view(B, 1, 1, 1, 1), flipped, clips)

    # Labels inchangés (le flip est invariant pour ces classes)
    return new_clips, labels




# Buffers ImageNet (calculés une fois, réutilisés)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@torch.no_grad()
def video_augment(
    clips: torch.Tensor,
    *,
    flip_prob: float = 0.5,
    crop_prob: float = 0.8,
    crop_scale: Tuple[float, float] = (0.7, 1.0),
    temporal_reverse_prob: float = 0.0,
    color_prob: float = 0.8,
    color_strength: float = 0.4,
    blur_prob: float = 0.3,
    erase_prob: float = 0.25,
    normalize: bool = True,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pipeline d'augmentation vidéo cohérent temporellement.

    Args:
        clips : (B, T, C, H, W) en [0, 1] — NON normalisé.
        normalize : si True, applique mean/std à la fin (par défaut ImageNet).

    Ordre : géométrie spatiale → temporel → photométrie → erasing → normalisation.
    """

    # 3. Photométrie (en espace [0, 1])
    clips = color_jitter_video(clips, p=color_prob, strength=color_strength)
    clips = gaussian_blur_video(clips, p=blur_prob)

    # 5. Normalisation finale, en sortie de pipeline
    if normalize:
        m = (mean if mean is not None else _IMAGENET_MEAN).to(clips.device, clips.dtype)
        s = (std  if std  is not None else _IMAGENET_STD ).to(clips.device, clips.dtype)
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
    if name == "TSM_RES":
        return TSMResNet50(
            num_classes=num_classes,
            num_segments=cfg.dataset.num_frames,
            n_div=8,
            dropout=0.5,
            pretrained=pretrained,
            consensus=cfg.model.get("consensus", "avg"),
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
    use_temporal = True
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

        if reverse_lookup is not None and reverse_prob > 0:
            video_batch, labels = label_aware_temporal_reverse(
                video_batch, labels, reverse_lookup, p=reverse_prob
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
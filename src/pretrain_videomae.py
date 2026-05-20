"""Domain-adaptive masked video pretraining on competition data.

VideoMAE pretraining objective: mask 90% of space-time tubes (same spatial position
across all temporal groups), encode the 10% visible tubes with the ViT backbone,
decode the masked patches with a lightweight 8-block decoder, compute MSE loss on
the raw pixel values of the masked patches.

Why this helps:
  The competition videos are first-person hand-object manipulation clips. Starting
  from MCG-NJU/videomae-large (Kinetics-pretrained) and continuing masked pretraining
  on our 44k videos adapts the low-level and mid-level features to this specific domain
  BEFORE task-specific fine-tuning. Expected gain: +1-3% vs. directly fine-tuning from
  the Kinetics checkpoint.

Tube masking strategy (temporal coherence):
  For each video, randomly sample mask_ratio × 196 spatial positions to mask.
  ALL temporal groups of those positions are masked (= "tubes"). The encoder must
  infer missing patches from context, not from temporal neighbors.

After pretraining, fine-tune with:
    python train_trackB.py experiment=track_B_videomae_domain_adapted
  which loads the saved backbone from outputs/pretrained_videomae/backbone_final.pt

Usage:
    cd src
    uv run python pretrain_videomae.py
    # or with custom args:
    uv run python pretrain_videomae.py \\
        --backbone MCG-NJU/videomae-large \\
        --init_from MCG-NJU/videomae-large-finetuned-ssv2 \\
        --data_dir ../data \\
        --output_dir ../outputs/pretrained_videomae \\
        --epochs 10 \\
        --batch_size 4 \\
        --mask_ratio 0.9
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import VideoMAEForPreTraining, VideoMAEModel

sys.path.insert(0, str(Path(__file__).parent))

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import build_transforms, set_seed, split_train_val

_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
_VM_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
_VM_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)

_TARGET_FRAMES = 16
_TUBELET_SIZE  = 2
_PATCH_SIZE    = 16
_IMG_SIZE      = 224
_N_TEMPORAL    = _TARGET_FRAMES // _TUBELET_SIZE   # 8
_N_SPATIAL     = (_IMG_SIZE // _PATCH_SIZE) ** 2   # 196
_N_PATCHES     = _N_TEMPORAL * _N_SPATIAL           # 1568


def _linear_upsample(x: torch.Tensor, target_T: int) -> torch.Tensor:
    B, T, C, H, W = x.shape
    if T == target_T:
        return x
    device, dtype = x.device, x.dtype
    t_src  = torch.linspace(0, T - 1, target_T, device=device, dtype=dtype)
    t_low  = t_src.floor().long().clamp(0, T - 2)
    t_high = (t_low + 1).clamp(0, T - 1)
    alpha  = (t_src - t_low.to(dtype)).view(1, target_T, 1, 1, 1)
    return x[:, t_low] + alpha * (x[:, t_high] - x[:, t_low])


def make_tube_mask(
    batch_size: int,
    mask_ratio: float = 0.9,
    n_spatial: int = _N_SPATIAL,
    n_temporal: int = _N_TEMPORAL,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Tube masking: same spatial positions masked across all temporal groups.

    Returns bool_masked_pos of shape (B, N_TEMPORAL * N_SPATIAL) = (B, 1568).
    True = masked (not shown to encoder).
    """
    n_mask = round(mask_ratio * n_spatial)
    bool_masked_pos = torch.zeros(batch_size, n_temporal * n_spatial, dtype=torch.bool, device=device)
    for b in range(batch_size):
        spatial_mask = torch.randperm(n_spatial, device=device)[:n_mask]
        for t in range(n_temporal):
            bool_masked_pos[b, t * n_spatial + spatial_mask] = True
    return bool_masked_pos


def load_pretrain_model(
    backbone: str = "MCG-NJU/videomae-large",
    init_from: str | None = None,
) -> VideoMAEForPreTraining:
    """Load VideoMAEForPreTraining, optionally initialising the encoder from a fine-tuned checkpoint.

    `backbone`: base MAE checkpoint that has the decoder (e.g. videomae-large pretrained).
    `init_from`: (optional) fine-tuned checkpoint whose encoder weights replace the base.
                 This gives us the SSv2-adapted encoder + the Kinetics MAE decoder.
    """
    print(f"Loading VideoMAEForPreTraining from {backbone}…")
    model = VideoMAEForPreTraining.from_pretrained(backbone, use_safetensors=True)

    if init_from is not None and init_from != backbone:
        print(f"Initialising encoder from fine-tuned checkpoint: {init_from}…")
        ft_encoder = VideoMAEModel.from_pretrained(init_from, use_safetensors=True)
        # VideoMAEForPreTraining.videomae = the VideoMAEModel backbone
        missing, unexpected = model.videomae.load_state_dict(
            ft_encoder.state_dict(), strict=False
        )
        if missing:
            print(f"  Missing keys (expected for decoder-only keys): {missing[:5]}…")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}…")
        del ft_encoder
    return model


def pretrain_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float = 0.9,
    clip_grad_norm: float = 1.0,
    grad_accum_steps: int = 4,
    epoch: int = 0,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad()

    for step, (frames, _labels) in enumerate(loader):
        # frames: (B, T=4, C, H, W)  ImageNet-normalised
        frames = frames.to(device, non_blocking=True)
        B = frames.shape[0]

        # ImageNet → VideoMAE [0.5, 0.5, 0.5] normalisation
        m_in = _IN_MEAN.to(device, frames.dtype)
        s_in = _IN_STD.to(device, frames.dtype)
        m_vm = _VM_MEAN.to(device, frames.dtype)
        s_vm = _VM_STD.to(device, frames.dtype)
        frames = (frames * s_in + m_in - m_vm) / s_vm

        # 4 → 16 frames via linear interpolation
        frames = _linear_upsample(frames, _TARGET_FRAMES)

        # Tube masking: (B, 1568) bool tensor
        bool_masked_pos = make_tube_mask(B, mask_ratio=mask_ratio, device=device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(pixel_values=frames, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss / grad_accum_steps

        loss.backward()

        is_last = (step + 1 == len(loader))
        if (step + 1) % grad_accum_steps == 0 or is_last:
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += float(loss.item()) * grad_accum_steps
        n_batches += 1

        if step % 50 == 0:
            avg = total_loss / n_batches
            print(f"  Epoch {epoch} | step {step}/{len(loader)} | loss {avg:.4f}")

    return total_loss / max(n_batches, 1)


def save_backbone(model: VideoMAEForPreTraining, output_dir: Path, name: str) -> None:
    """Save only the VideoMAE encoder backbone (VideoMAEModel) state dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_state = model.videomae.state_dict()
    path = output_dir / name
    torch.save(backbone_state, path)
    print(f"Backbone saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoMAE domain-adaptive pretraining")
    parser.add_argument("--backbone",   default="MCG-NJU/videomae-large",
                        help="Base VideoMAEForPreTraining checkpoint (must include decoder)")
    parser.add_argument("--init_from",  default="MCG-NJU/videomae-large-finetuned-ssv2",
                        help="Optional fine-tuned checkpoint to init encoder from")
    parser.add_argument("--data_dir",   default="../data",
                        help="Root directory with training videos")
    parser.add_argument("--output_dir", default="../outputs/pretrained_videomae",
                        help="Where to save backbone checkpoints")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser.add_argument("--grad_accum", type=int,   default=8,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--val_ratio",     type=float, default=0.05,
                        help="Fraction of data held out for validation (loss only)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir).resolve()
    all_samples = collect_video_samples(data_dir)
    train_samples, val_samples = split_train_val(all_samples, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Dataset: {len(train_samples)} train, {len(val_samples)} val videos")

    transform = build_transforms(is_training=True, use_imagenet_norm=True, use_random_resized_crop=True)
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=True)

    train_ds = VideoFrameDataset(root_dir=data_dir, num_frames=4, transform=transform,
                                 sample_list=train_samples)
    val_ds   = VideoFrameDataset(root_dir=data_dir, num_frames=4, transform=eval_transform,
                                 sample_list=val_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_pretrain_model(backbone=args.backbone, init_from=args.init_from)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {n_params:.0f}M")

    # ── Optimizer + cosine schedule ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Pretraining loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = pretrain_one_epoch(
            model, train_loader, optimizer, device,
            mask_ratio=args.mask_ratio,
            grad_accum_steps=args.grad_accum,
            epoch=epoch,
        )
        scheduler.step()

        # Validation loss
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for frames, _ in val_loader:
                frames = frames.to(device)
                B = frames.shape[0]
                m_in = _IN_MEAN.to(device, frames.dtype)
                s_in = _IN_STD.to(device, frames.dtype)
                m_vm = _VM_MEAN.to(device, frames.dtype)
                s_vm = _VM_STD.to(device, frames.dtype)
                frames = (frames * s_in + m_in - m_vm) / s_vm
                frames = _linear_upsample(frames, _TARGET_FRAMES)
                bool_masked_pos = make_tube_mask(B, mask_ratio=args.mask_ratio, device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(pixel_values=frames, bool_masked_pos=bool_masked_pos)
                val_loss += float(outputs.loss.item())
                val_steps += 1
        val_loss /= max(val_steps, 1)

        print(f"  Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Save backbone every epoch (overwrite)
        save_backbone(model, output_dir, "backbone_latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_backbone(model, output_dir, "backbone_best.pt")
            print(f"  *** New best val_loss: {best_val_loss:.4f} — backbone_best.pt saved ***")

    save_backbone(model, output_dir, "backbone_final.pt")
    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Use backbone_best.pt (or backbone_final.pt) with experiment=track_B_videomae_domain_adapted")


if __name__ == "__main__":
    main()

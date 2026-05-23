"""
Evaluate a saved checkpoint on the **full** validation split: reports top-1 and top-5 accuracy.

Uses ``dataset.val_dir`` (entire folder; no ``split_train_val``).

Example (from ``src/``)::

    uv run python evaluate.py training.checkpoint_path=best_model.pt
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
import train as _train_a
import train_trackB as _train_b
from utils import build_transforms, set_seed

try:
    import wandb
    _WANDB_OK = True
except Exception as _wandb_err:
    print(f"[wandb] import failed ({_wandb_err}) — eval logging désactivé")
    _WANDB_OK = False

_TRACK_B_MODELS = {
    "videomae", "motion_videomae", "frozen_videomae", "swin3d_finetune",
    "vit_temporal", "qwen_vl_video", "dynamic_videomae", "frame_pair_net",
    "qwen2_vl_7b", "qwen25_7b", "dual_encoder", "cnn_temporal",
    # models added in later experiments
    "bidirectional_frame_pair", "qwen_temporal_attn",
    "videomae_lora", "qwen_lora",
    "videomae_temporal_head", "videomae_domain_adapted",
    "videomae_pair_attn",
    "videomae_cross_attn_pairs",
    "videomae_multiscale",
    "videomae_pairwise_ranking",
    "dinov3_temporal",
    "videomae_large",
    "vjepa2_head",
    "internvl2_classifier",
    "internvit6b_temporal",
    "internvit6b_pairwise",
    "vjepa2_variants",
}


def build_model(cfg):
    name = cfg.model.name
    if name in _TRACK_B_MODELS:
        return _train_b.build_model(cfg)
    return _train_a.build_model(cfg)


def _three_crop(video: torch.Tensor, crop_size: int = 224) -> list:
    """Return top-left, center, bottom-right crops from a (B, T, C, H, W) tensor."""
    H, W = video.shape[-2], video.shape[-1]
    r_offsets = [0, (H - crop_size) // 2, H - crop_size]
    c_offsets = [0, (W - crop_size) // 2, W - crop_size]
    return [video[:, :, :, r:r + crop_size, c:c + crop_size] for r, c in zip(r_offsets, c_offsets)]


def load_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Rebuild the model from the Hydra config stored in the checkpoint (same as training).

    Checkpoints must include ``config`` (saved by ``train.py``). No duplicate
    architecture list here—``build_model`` is the single construction site.
    """
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise ValueError(
            "Checkpoint has no 'config' entry. Train with the current train.py so the "
            "full Hydra config is saved with the weights."
        )
    raw_cfg = checkpoint["config"]
    # Old checkpoints (pre-Hydra refactor) store a flat model dict with no 'model' subkey.
    # Wrap it into the shape build_model() expects: cfg.model.* and cfg.dataset.num_frames.
    if isinstance(raw_cfg, dict) and "model" not in raw_cfg:
        num_frames = raw_cfg.get("n_segment") or checkpoint.get("num_frames", 4)
        raw_cfg = {"model": raw_cfg, "dataset": {"num_frames": int(num_frames)}}
    cfg = OmegaConf.create(raw_cfg)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    raw: Dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model = load_model_from_checkpoint(raw, device)

    # Normalization must exactly match what was used during training.
    # Checkpoints from train_trackA/B.py store use_imagenet_norm explicitly.
    # Older checkpoints fall back to the pretrained flag (original heuristic).
    use_imagenet_norm = bool(
        raw["use_imagenet_norm"] if "use_imagenet_norm" in raw
        else raw["pretrained"] if "pretrained" in raw
        else cfg.model.get("pretrained", True)
    )
    use_multi_crop = bool(cfg.training.get("use_multi_crop", False))
    use_tta = bool(cfg.training.get("use_tta", False))

    # Multi-crop: load at 256×256, crop 3×224×224 at inference; otherwise 224×224.
    eval_size = 256 if use_multi_crop else 224
    eval_transform = build_transforms(image_size=eval_size, is_training=False, use_imagenet_norm=use_imagenet_norm)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    num_classes = int(cfg.model.num_classes) if hasattr(cfg, "model") else 33
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for video_batch, labels in val_loader:
            video_batch = video_batch.to(device)
            labels      = labels.to(device)

            if use_multi_crop:
                crops = _three_crop(video_batch, crop_size=224)
                logits = sum(model(c) for c in crops) / len(crops)
            elif use_tta:
                # Average logits over: original + horizontal flip
                logits_orig = model(video_batch)
                logits_flip = model(torch.flip(video_batch, dims=[-1]))
                logits = (logits_orig + logits_flip) * 0.5
            else:
                logits = model(video_batch)  # (B, num_classes)

            # Top-1: argmax class matches label
            predictions_top1 = logits.argmax(dim=1)
            correct_top1 += int((predictions_top1 == labels).sum().item())

            # Top-5: label appears in the five largest logits per row
            _, predictions_top5 = logits.topk(5, dim=1, largest=True, sorted=True)
            # (B, 5) compared with (B, 1) -> (B, 5) boolean, True if label in top-5
            matches_top5 = predictions_top5.eq(labels.view(-1, 1)).any(dim=1)
            correct_top5 += int(matches_top5.sum().item())

            total += labels.size(0)

            # Accumulate confusion matrix (CPU)
            preds_cpu  = predictions_top1.cpu()
            labels_cpu = labels.cpu()
            for p, l in zip(preds_cpu, labels_cpu):
                conf_matrix[l.item(), p.item()] += 1
            all_preds.extend(preds_cpu.tolist())
            all_labels.extend(labels_cpu.tolist())

    top1_accuracy = correct_top1 / max(total, 1)
    top5_accuracy = correct_top5 / max(total, 1)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    per_class_correct = conf_matrix.diagonal()
    per_class_total   = conf_matrix.sum(dim=1)
    per_class_acc = per_class_correct.float() / per_class_total.float().clamp(min=1)
    print("\nPer-class top-1 accuracy:")
    for cls_idx in range(num_classes):
        bar = "#" * int(per_class_acc[cls_idx].item() * 20)
        print(f"  {cls_idx:2d}: {per_class_acc[cls_idx]:.3f} [{bar:<20}]"
              f"  ({per_class_correct[cls_idx]}/{per_class_total[cls_idx]})")

    # ── Most confused pairs ───────────────────────────────────────────────────
    conf_offdiag = conf_matrix.clone()
    conf_offdiag.fill_diagonal_(0)
    flat = conf_offdiag.view(-1)
    top_k = min(10, (flat > 0).sum().item())
    if top_k > 0:
        top_vals, top_idx = flat.topk(top_k)
        print("\nTop confused pairs (true → predicted: count):")
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            true_cls = idx // num_classes
            pred_cls = idx % num_classes
            print(f"  {true_cls:2d} → {pred_cls:2d}: {val}")

    # ── Save confusion matrix ─────────────────────────────────────────────────
    import numpy as np
    cm_path = checkpoint_path.with_suffix("").with_name(
        checkpoint_path.stem + "_confusion_matrix.npy"
    )
    np.save(cm_path, conf_matrix.numpy())
    print(f"\nConfusion matrix saved → {cm_path}")

    # ── wandb logging ─────────────────────────────────────────────────────────
    # New run named "<train_run>_eval" (or checkpoint stem if no env var set).
    # Logs top-1/top-5, confusion matrix, per-class accuracy. Silently no-ops
    # when WANDB_API_KEY is absent — wandb's offline mode would clutter logs.
    if _WANDB_OK and os.environ.get("WANDB_API_KEY"):
        try:
            base_name = os.environ.get("WANDB_RUN_NAME") or checkpoint_path.stem
            wandb.init(
                project=os.environ.get(
                    "WANDB_PROJECT", "What_Happens_Next_CSC_43M04_EP_2026"
                ),
                name=f"{base_name}_eval",
                job_type="eval",
                config={
                    "checkpoint_path": str(checkpoint_path),
                    "val_dir":         str(val_dir),
                    "num_samples":     total,
                    "num_classes":     num_classes,
                    "use_multi_crop":  use_multi_crop,
                    "use_tta":         use_tta,
                    "num_frames":      num_frames,
                },
                reinit=True,
            )
            class_names = [str(i) for i in range(num_classes)]
            wandb.log({
                "eval/top1": top1_accuracy,
                "eval/top5": top5_accuracy,
                "eval/num_samples": total,
                "eval/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=all_labels, preds=all_preds, class_names=class_names,
                ),
            })
            per_class_dict = {
                f"eval_per_class_acc/class_{c}": float(per_class_acc[c].item())
                for c in range(num_classes) if int(per_class_total[c]) > 0
            }
            wandb.log(per_class_dict)
            wandb.run.summary["eval_top1"] = top1_accuracy
            wandb.run.summary["eval_top5"] = top5_accuracy
            wandb.finish()
        except Exception as exc:
            print(f"[wandb] eval logging skipped ({exc})")
    else:
        print("[wandb] WANDB_API_KEY non défini — eval non envoyé à wandb")


if __name__ == "__main__":
    main()

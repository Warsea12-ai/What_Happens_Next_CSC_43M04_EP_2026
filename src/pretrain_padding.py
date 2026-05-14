"""
Pré-entraînement du PaddingModule sur les frames vidéo du challenge.

Usage (depuis src/) :
    python pretrain_padding.py

Sauvegarde les poids dans `padding_module.pt`.

Self-supervised : aucun label nécessaire. La loss compare les bords
réels de chaque frame avec ceux prédits à partir des voisins.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.padding_module import PaddingModule
from utils import build_transforms, set_seed, split_train_val


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.dataset.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams (override possibles depuis la CLI)
    epochs        = int(cfg.training.get("padding_epochs", 5))
    batch_size    = int(cfg.training.get("padding_batch_size", 16))
    lr            = float(cfg.training.get("padding_lr", 1e-3))
    hidden        = int(cfg.training.get("padding_hidden", 64))
    context       = int(cfg.training.get("padding_context", 8))
    weights_path  = Path(cfg.training.get(
        "padding_weights_path", "padding_module.pt"
    )).resolve()
    max_samples   = cfg.training.get("padding_max_samples", 2000)

    # --- Dataset (mêmes images que pour le training principal) ---
    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]
    train_samples, _ = split_train_val(
        all_samples, val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    # On désactive la normalisation ImageNet : on travaille en [0, 1]
    transform = build_transforms(is_training=False, use_imagenet_norm=False)

    dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=transform,
        sample_list=train_samples,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(cfg.training.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
    )

    # --- Module ---
    module = PaddingModule(channels=3, hidden=hidden, context=context).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-4)
    print(f"Padding Module : {sum(p.numel() for p in module.parameters())} params")

    # --- Entraînement ---
    for epoch in range(epochs):
        module.train()
        running, total = 0.0, 0
        for video_batch, _ in loader:
            # video_batch : (B, T, C, H, W)
            # On replie T dans la dim batch : chaque frame est un sample indépendant
            B, T, C, H, W = video_batch.shape
            frames = video_batch.reshape(B * T, C, H, W).to(device, non_blocking=True)

            loss = module.self_supervised_loss(frames)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * frames.size(0)
            total   += frames.size(0)

        print(f"[Epoch {epoch + 1}/{epochs}] L1 loss = {running / max(total, 1):.4f}")

    # --- Sauvegarde ---
    torch.save(module.state_dict(), weights_path)
    print(f"Saved padding module weights to {weights_path}")


if __name__ == "__main__":
    main()
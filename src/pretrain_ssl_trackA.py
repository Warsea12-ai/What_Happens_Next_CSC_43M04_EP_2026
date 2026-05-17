"""
Self-supervised temporal pre-training for Track A (no external pretrained weights).

Two complementary self-supervised tasks:

  Task 1 — Temporal order (original, binary):
    Given a 4-frame clip, predict forward (0) or reversed (1) order.
    Forces the backbone to detect motion direction.

  Task 2 — Temporal position (new, 4-class per frame, --multi_task):
    Shuffle the 4 frames into a random permutation; predict each frame's
    original temporal position (0 / 1 / 2 / 3).
    Forces the backbone to encode absolute temporal location, not just
    relative direction — richer signal for distinguishing action stages.

Combined loss (multi_task): L = order_CE + 0.5 × position_CE
100k order pairs + 200k (clip × permutation) position pairs → backbone
learns both "which direction" and "where in time".

Usage (from src/):
    uv run python pretrain_ssl_trackA.py --multi_task
    # saves: ssl_backbone_resnet18.pt

Fine-tune with saved backbone:
    uv run python train.py experiment=track_A_ssl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import set_seed, split_train_val

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class TemporalOrderDataset(Dataset):
    """Wraps VideoFrameDataset: returns (clip, order_label, shuffled_clip, position_labels).

    order_label      : 0 = forward, 1 = reversed
    shuffled_clip    : frames in a random permutation (for multi-task position task)
    position_labels  : (T,) original position of each frame in shuffled_clip
    """

    def __init__(self, base_dataset: VideoFrameDataset) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base) * 2  # forward + reversed for every video

    def __getitem__(self, idx: int):
        is_reversed = idx >= len(self.base)
        clip, _ = self.base[idx % len(self.base)]  # (T, C, H, W)
        if is_reversed:
            order_clip = clip.flip(0)
        else:
            order_clip = clip
        order_label = torch.tensor(int(is_reversed), dtype=torch.long)

        # Random permutation of original clip for position task
        T = clip.shape[0]
        perm = torch.randperm(T)
        shuffled = clip[perm]           # shuffled[i] came from position perm[i]
        pos_labels = perm.long()        # pos_labels[i] = original position of frame i

        return order_clip, order_label, shuffled, pos_labels


def build_backbone(arch: str = "resnet18", use_tsm: bool = False, n_segment: int = 4, fold_div: int = 8) -> tuple[nn.Module, int]:
    if arch == "resnet18":
        net = models.resnet18(weights=None)
        dim = 512
    elif arch == "resnet34":
        net = models.resnet34(weights=None)
        dim = 512
    else:
        raise ValueError(arch)
    net.fc = nn.Identity()
    if use_tsm:
        # Wrap with residual TSM (same structure as TSM_s, stochastic_depth=0 for SSL stability)
        from models.TSM_s import make_temporal_shift as _make_tsm
        _make_tsm(net, n_segment=n_segment, fold_div=fold_div, max_drop_prob=0.0)
    return net, dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       default="resnet18")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--train_dir",  default="processed_data/val2/train")
    parser.add_argument("--out",        default="ssl_backbone_resnet18.pt")
    parser.add_argument("--use_tsm",    action="store_true",
                        help="Wrap backbone with TSM (saves TSM-compatible weights)")
    parser.add_argument("--fold_div",   type=int,   default=8)
    parser.add_argument("--multi_task", action="store_true",
                        help="Add temporal position prediction task (richer SSL signal)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([
        T.Resize((112, 112)),          # smaller resolution → faster pre-training
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    train_dir = Path(args.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)
    train_samples, val_samples = split_train_val(all_samples, val_ratio=0.1, seed=args.seed)

    base_train = VideoFrameDataset(train_dir, num_frames=4, transform=transform,
                                   sample_list=train_samples)
    base_val   = VideoFrameDataset(train_dir, num_frames=4, transform=transform,
                                   sample_list=val_samples)

    train_ds = TemporalOrderDataset(base_train)
    val_ds   = TemporalOrderDataset(base_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    backbone, feat_dim = build_backbone(args.arch, use_tsm=args.use_tsm,
                                        n_segment=4, fold_div=args.fold_div)
    if args.use_tsm:
        print(f"Using TSM-wrapped backbone (fold_div={args.fold_div})")

    class SSLModel(nn.Module):
        def __init__(self, bb: nn.Module, dim: int, multi_task: bool = False) -> None:
            super().__init__()
            self.bb = bb
            self.multi_task = multi_task
            self.ln_c = nn.LayerNorm(dim)
            self.ln_d = nn.LayerNorm(dim)
            # Task 1: temporal order (forward vs reversed)
            self.order_head = nn.Linear(3 * dim, 2)
            if multi_task:
                # Task 2: temporal position (which of T positions is this frame?)
                self.pos_head = nn.Linear(dim, 4)

        def forward(self, order_clip: torch.Tensor, shuffled: torch.Tensor | None = None):
            B, T, C, H, W = order_clip.shape
            f = self.bb(order_clip.reshape(B * T, C, H, W)).view(B, T, -1)
            avg  = f.mean(1)
            diff = (f[:, 1:] - f[:, :-1]).mean(1)
            net  = f[:, -1] - f[:, 0]
            order_logits = self.order_head(torch.cat([self.ln_c(avg), self.ln_d(diff), self.ln_d(net)], 1))

            if self.multi_task and shuffled is not None:
                g = self.bb(shuffled.reshape(B * T, C, H, W)).view(B * T, -1)
                pos_logits = self.pos_head(g).view(B, T, 4)   # (B, T, 4)
                return order_logits, pos_logits

            return order_logits

    ssl = SSLModel(backbone, feat_dim, multi_task=args.multi_task).to(device)
    optimizer = torch.optim.AdamW(ssl.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    warmup = 3
    sched = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
        CosineAnnealingLR(optimizer, T_max=args.epochs - warmup, eta_min=1e-5),
    ], milestones=[warmup])

    if args.multi_task:
        print("Multi-task SSL: temporal order + temporal position prediction")

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        ssl.train()
        order_correct, order_total = 0, 0
        for order_clips, order_labels, shuffled_clips, pos_labels in train_loader:
            order_clips   = order_clips.to(device)
            order_labels  = order_labels.to(device)
            shuffled_clips = shuffled_clips.to(device)
            pos_labels    = pos_labels.to(device)   # (B, T)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                if args.multi_task:
                    order_logits, pos_logits = ssl(order_clips, shuffled_clips)
                    B, T = pos_labels.shape
                    order_loss = loss_fn(order_logits, order_labels)
                    pos_loss   = loss_fn(pos_logits.reshape(B * T, 4), pos_labels.reshape(B * T))
                    loss = order_loss + 0.5 * pos_loss
                else:
                    order_logits = ssl(order_clips)
                    loss = loss_fn(order_logits, order_labels)

            loss.backward()
            nn.utils.clip_grad_norm_(ssl.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                order_correct += (order_logits.argmax(1) == order_labels).sum().item()
                order_total   += order_labels.size(0)
        sched.step()

        ssl.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for order_clips, order_labels, _, _ in val_loader:
                order_clips  = order_clips.to(device)
                order_labels = order_labels.to(device)
                logits = ssl(order_clips)
                if isinstance(logits, tuple):
                    logits = logits[0]
                val_correct += (logits.argmax(1) == order_labels).sum().item()
                val_total   += order_labels.size(0)
        val_acc = val_correct / max(val_total, 1)
        train_acc = order_correct / max(order_total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | train_order={train_acc:.3f} val_order={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"backbone_state_dict": backbone.state_dict(),
                        "arch": args.arch, "val_acc": val_acc}, args.out)
            print(f"  Saved backbone → {args.out} (val_acc={val_acc:.4f})")

    print(f"Done. Best val acc: {best_val_acc:.4f}")
    if args.use_tsm:
        print(f"Load with: uv run python train.py experiment=track_A_tsm_ssl")
        print(f"       or: uv run python train.py experiment=track_A_tsm_ssl_motion  (motion head)")
    else:
        print(f"Load with: uv run python train.py experiment=track_A_ssl")


if __name__ == "__main__":
    main()

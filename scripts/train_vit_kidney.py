"""
Train ViTSeg3D to segment kidney_left on TotalSegmentator.

Usage
-----
  python scripts/train_vit_kidney.py [--data /path/to/totalseg] [--epochs 50]

The script uses DiceCE loss, AdamW with cosine LR, and logs per-epoch
train/val Dice to stdout and W&B.  Best checkpoint is saved to results/vit_kidney_best.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.totalseg_dataset import TotalSegDataset
from src.vit_seg import ViTSeg3D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_SIZE = (64, 64, 64)
PATCH_SIZE = (8, 8, 8)        # → 8×8×8 = 512 tokens
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-5
CLASSES = ["kidney_left"]
NUM_CLASSES = 2               # background + kidney_left

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, D, H, W),  targets: (B, D, H, W) long
        probs = F.softmax(logits, dim=1)
        n_cls = logits.shape[1]
        targets_oh = F.one_hot(targets, n_cls).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.ce(logits, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, cls: int = 1) -> float:
    pred = logits.argmax(1) == cls
    gt   = targets == cls
    inter = (pred & gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    return (2 * inter + 1) / (union + 1)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    total_loss, total_dice, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_dice += dice_score(logits, labels, cls=1)
            n += 1
    return total_loss / n, total_dice / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/work/dlclarge2/ndirt-SegFM3D/data/totalseg")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_train", type=int, default=None,
                        help="limit training subjects (useful for quick smoke-tests)")
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--wandb_project", default="vit-kidney-seg")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=dict(
            image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
            batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
            epochs=args.epochs, classes=CLASSES,
        ),
    )

    train_ds = TotalSegDataset(
        args.data, classes=CLASSES, image_size=IMAGE_SIZE,
        split="train", max_subjects=args.max_train,
    )
    val_ds = TotalSegDataset(
        args.data, classes=CLASSES, image_size=IMAGE_SIZE,
        split="val", max_subjects=args.max_val,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = ViTSeg3D(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=1,
        num_classes=NUM_CLASSES,
        embed_dim=256,
        depth=6,
        num_heads=8,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = DiceCELoss()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    best_ckpt = results_dir / "vit_kidney_best.pt"

    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        va_loss, va_dice = run_epoch(model, val_loader,   optimizer, loss_fn, device, train=False)
        scheduler.step()

        tag = " *" if va_dice > best_val_dice else ""
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"train loss={tr_loss:.4f} dice={tr_dice:.4f}  "
            f"val loss={va_loss:.4f} dice={va_dice:.4f}{tag}"
        )
        wandb.log({
            "train/loss": tr_loss, "train/dice": tr_dice,
            "val/loss": va_loss,   "val/dice": va_dice,
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch)

    wandb.summary["best_val_dice"] = best_val_dice
    wandb.finish()
    print(f"\nBest val Dice: {best_val_dice:.4f}  →  {best_ckpt}")


if __name__ == "__main__":
    main()

"""
Train ViTInContext3D for in-context 3-D segmentation on TotalSegmentator.

Config is managed by Hydra. All hyperparameters live in configs/config.yaml.
Any value can be overridden on the command line using dot-notation:

  python scripts/train_vit_in_context.py train.epochs=10
  python scripts/train_vit_in_context.py train.run_name=debug data.max_train_subjects=50
  python scripts/train_vit_in_context.py model.depth_stage1=6 model.depth_stage2=6
"""

import sys
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset,
    incontext_collate_fn,
)
from src.vit_in_context import ViTInContext3D


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = F.softmax(logits, dim=1)
        n_cls  = logits.shape[1]
        tgt_oh = F.one_hot(targets, n_cls).permute(0, 4, 1, 2, 3).float()
        dims   = (0, 2, 3, 4)
        inter  = (probs * tgt_oh).sum(dims)
        union  = probs.sum(dims) + tgt_oh.sum(dims)
        return 1 - ((2 * inter + self.smooth) / (union + self.smooth)).mean()


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
def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred  = logits.argmax(1) == 1
    gt    = targets == 1
    inter = (pred & gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    return (2 * inter + 1) / (union + 1)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, loss_fn, scaler, device, train: bool):
    model.train(train)
    total_loss = total_dice = n = 0
    per_class_dice: dict[str, list[float]] = defaultdict(list)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            images      = batch["image"].to(device, non_blocking=True)
            labels      = batch["label"].to(device, non_blocking=True)
            context_in  = batch["context_in"].to(device, non_blocking=True)
            context_out = batch["context_out"].to(device, non_blocking=True)
            label_names = batch["label_names"]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images, context_in, context_out)
                loss   = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_dice += dice_score(logits, labels)
            n += 1

            with torch.no_grad():
                for i, lname in enumerate(label_names):
                    per_class_dice[lname].append(dice_score(logits[i:i+1], labels[i:i+1]))

    mean_per_class = {cls: sum(v) / len(v) for cls, v in per_class_dice.items()}
    return total_loss / n, total_dice / n, mean_per_class


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Resolve to plain Python types where needed
    image_size = tuple(cfg.data.image_size)
    patch_size = tuple(cfg.model.patch_size)
    classes    = list(cfg.data.classes)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.train.wandb_project,
        name=cfg.train.run_name or None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    loader_kw = dict(
        num_workers=cfg.train.workers,
        pin_memory=True,
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=4 if cfg.train.workers > 0 else None,
        collate_fn=incontext_collate_fn,
    )

    train_ds = TotalSegInContextDataset(
        root=cfg.paths.totalseg, classes=classes,
        image_size=image_size, split="train",
        context_size=cfg.data.context_size,
        max_subjects=cfg.data.max_train_subjects,
    )
    val_ds = TotalSegInContextDataset(
        root=cfg.paths.totalseg, classes=classes,
        image_size=image_size, split="val",
        context_size=cfg.data.context_size,
        max_subjects=cfg.data.max_val_subjects,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False, **loader_kw)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ViTInContext3D(
        image_size=image_size,    patch_size=patch_size,
        in_channels=1,            num_classes=2,
        embed_dim=cfg.model.embed_dim,
        depth_stage1=cfg.model.depth_stage1,
        depth_stage2=cfg.model.depth_stage2,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
    ).to(device)

    print("Compiling model...", flush=True)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )
    loss_fn = DiceCELoss()
    scaler  = torch.amp.GradScaler("cuda")

    results_dir = Path(cfg.paths.results)
    results_dir.mkdir(exist_ok=True)
    best_ckpt = results_dir / "vit_incontext_best.pt"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_dice = 0.0

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_dice, tr_cls = run_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, train=True
        )
        va_loss, va_dice, va_cls = run_epoch(
            model, val_loader, optimizer, loss_fn, scaler, device, train=False
        )
        scheduler.step()

        tag = " *" if va_dice > best_val_dice else ""
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"[{epoch:3d}/{cfg.train.epochs}] "
            f"train loss={tr_loss:.4f} dice={tr_dice:.4f}  "
            f"val loss={va_loss:.4f} dice={va_dice:.4f}{tag}"
        )
        cls_str = "  ".join(f"{c}={v:.3f}" for c, v in sorted(va_cls.items()))
        print(f"  val per-class: {cls_str}")

        mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        # Per-class table: one row per class, train + val dice side by side
        table = wandb.Table(
            columns=["class", "train_dice", "val_dice"],
            data=[
                [cls, tr_cls.get(cls, float("nan")), va_cls.get(cls, float("nan"))]
                for cls in sorted(classes)
            ],
        )

        log = {
            "train/loss": tr_loss, "train/dice": tr_dice,
            "val/loss":   va_loss, "val/dice":   va_dice,
            "lr": scheduler.get_last_lr()[0],
            "gpu/vram_peak_gb": mem,
            "dice_by_class": table,
        }
        # Slash-separated keys → W&B groups them under val/dice/ and train/dice/
        log.update({f"val/dice/{c}":   v for c, v in va_cls.items()})
        log.update({f"train/dice/{c}": v for c, v in tr_cls.items()})
        wandb.log(log, step=epoch)

    wandb.summary["best_val_dice"] = best_val_dice
    wandb.finish()
    print(f"\nBest val Dice: {best_val_dice:.4f}  →  {best_ckpt}")


if __name__ == "__main__":
    main()

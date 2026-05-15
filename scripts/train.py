"""
General training script for in-context 3-D segmentation.

Config is managed by Hydra.  All hyperparameters live in configs/config.yaml.
Select the architecture with model.name:

  python scripts/train.py model.name=vit_in_context
  python scripts/train.py model.name=resenc_in_context

Any value can be overridden on the command line using dot-notation:

  python scripts/train.py train.epochs=10
  python scripts/train.py train.run_name=debug data.max_train_subjects=50
  python scripts/train.py model.depth_stage1=6 model.depth_stage2=6
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.totalseg_classes import resolve_classes
from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset,
    incontext_collate_fn,
)
from src.vit_in_context import ViTInContext3D
from src.models.resenc_in_context import ResEncInContext3D


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig, image_size: tuple[int, int, int]) -> nn.Module:
    name = cfg.model.name
    shared = dict(
        image_size=image_size,
        in_channels=1,
        num_classes=2,
        depth_stage1=cfg.model.depth_stage1,
        depth_stage2=cfg.model.depth_stage2,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
    )
    if name == "vit_in_context":
        return ViTInContext3D(
            **shared,
            patch_size=tuple(cfg.model.patch_size),
            embed_dim=cfg.model.embed_dim,
        )
    if name == "resenc_in_context":
        return ResEncInContext3D(
            **shared,
            features_per_stage=tuple(cfg.model.features_per_stage),
            rope_theta=cfg.model.rope_theta,
        )
    raise ValueError(f"Unknown model.name: {name!r}. Choose vit_in_context or resenc_in_context.")


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
# Visualisation
# ---------------------------------------------------------------------------

def _best_z(mask: torch.Tensor) -> int:
    """Axial slice index with the most foreground pixels."""
    return int(mask.float().sum(dim=(-2, -1)).argmax())


def _overlay(ax, img_slice, mask_slice, cmap, title):
    ax.imshow(img_slice, cmap="gray", interpolation="nearest")
    ax.imshow(mask_slice, cmap=cmap, alpha=0.45, vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def collect_viz_samples(dataset, classes: list[str]) -> dict[str, dict]:
    """
    Call dataset[idx] directly in the main process — no DataLoader workers.
    Avoids abandoning mid-flight iterators (which deadlocks persistent workers).

    When synth is active:
      - p_synth < 1: one synth item + one real item per class
      - p_synth = 1: one synth item only
    """
    def _extract(item: dict) -> dict:
        return {k: item[k] for k in ("image", "label", "context_in", "context_out")}

    samples: dict[str, dict] = {}

    if dataset.synth_method is not None:
        item = dataset._get_synth_item()
        samples[item["label_name"]] = _extract(item)

    if dataset.synth_method is None or dataset.p_synth < 1.0:
        saved_p,  dataset.p_synth        = dataset.p_synth,        0.0
        saved_cb, dataset.class_balanced = dataset.class_balanced, False
        class_to_idx: dict[str, int] = {}
        for i, (_, cls) in enumerate(dataset.samples):
            if cls not in class_to_idx:
                class_to_idx[cls] = i
            if len(class_to_idx) == len(classes):
                break
        for cls, idx in class_to_idx.items():
            samples[cls] = _extract(dataset[idx])
        dataset.p_synth        = saved_p
        dataset.class_balanced = saved_cb

    return samples


@torch.no_grad()
def log_predictions(model, model_module, samples: dict[str, dict], device,
                    phase: str, epoch: int,
                    save_dir: Path | None = None,
                    use_wandb: bool = True):
    """
    Run the model on pre-collected samples, save one figure per class to
    save_dir (always) and optionally log to W&B.
    Each figure: tgt+GT  |  ctx1+GT  |  …  |  ctxK+GT  |  tgt+Pred
    """
    if not use_wandb and save_dir is None:
        return
    model_module.eval()

    if save_dir is not None:
        plot_dir = save_dir / "plots" / phase
        plot_dir.mkdir(parents=True, exist_ok=True)

    wandb_images: dict[str, wandb.Image] = {}

    for cls, s in samples.items():
        img   = s["image"].squeeze(0)   # (D, H, W)
        label = s["label"]              # (D, H, W)
        K     = s["context_in"].shape[0]

        amp_dtype = torch.bfloat16 if device.type == "xla" else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(
                s["image"].unsqueeze(0).to(device),
                s["context_in"].unsqueeze(0).to(device),
                s["context_out"].unsqueeze(0).to(device),
            )
        pred = logits.argmax(1).squeeze(0).cpu()   # (D, H, W)

        z_tgt  = _best_z(label)
        n_cols = 2 + K
        fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 3))

        _overlay(axes[0], img[z_tgt], label[z_tgt].float(), "Reds", "tgt + GT")
        for k in range(K):
            ctx_img  = s["context_in"][k].squeeze(0)
            ctx_mask = s["context_out"][k].float()
            z_ctx = _best_z(ctx_mask)
            _overlay(axes[1 + k], ctx_img[z_ctx], ctx_mask[z_ctx], "Reds", f"ctx {k+1} + GT")
        _overlay(axes[-1], img[z_tgt], pred[z_tgt].float(), "Blues", "tgt + Pred")

        fig.suptitle(f"{phase} | {cls}", fontsize=9, y=1.02)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(plot_dir / f"epoch{epoch:03d}_{cls}.png", dpi=150, bbox_inches="tight")
        if use_wandb:
            wandb_images[f"{phase}/pred/{cls}"] = wandb.Image(fig)
        plt.close(fig)

    if use_wandb:
        wandb.log(wandb_images, step=epoch)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, model_module, loader, optimizer, loss_fn, scaler, device, train: bool):
    model_module.train(train)
    total_loss = total_dice = n = 0
    per_class_dice: dict[str, list[float]] = defaultdict(list)

    phase = "train" if train else "val"
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=phase, leave=False, unit="batch")
        for batch in pbar:
            images      = batch["image"].to(device, non_blocking=True)
            labels      = batch["label"].to(device, non_blocking=True)
            context_in  = batch["context_in"].to(device, non_blocking=True)
            context_out = batch["context_out"].to(device, non_blocking=True)
            label_names = batch["label_names"]

            use_xla   = device.type == "xla"
            amp_dtype = torch.bfloat16 if use_xla else torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(images, context_in, context_out)
                loss   = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    if use_xla:
                        import torch_xla
                        torch_xla.sync()

            total_loss += loss.item()
            total_dice += dice_score(logits, labels)
            n += 1
            pbar.set_postfix(loss=f"{total_loss/n:.4f}", dice=f"{total_dice/n:.4f}")

            with torch.no_grad():
                for i, lname in enumerate(label_names):
                    key = "synth" if lname.startswith("sv_") else lname
                    per_class_dice[key].append(dice_score(logits[i:i+1], labels[i:i+1]))

    mean_per_class = {cls: sum(v) / len(v) for cls, v in per_class_dice.items()}
    if n == 0:
        return float("nan"), float("nan"), mean_per_class
    return total_loss / n, total_dice / n, mean_per_class


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    image_size    = tuple(cfg.data.image_size)
    train_classes = resolve_classes(cfg.data.train_classes, cfg.paths.totalseg)
    val_classes   = resolve_classes(cfg.data.val_classes,   cfg.paths.totalseg)

    if cfg.train.tpu:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            import subprocess, sys
            gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                      capture_output=True, text=True).stdout.strip()
            print(f"WARNING: torch.cuda.is_available()=False. nvidia-smi sees: {gpu_info or 'nothing'}")
            print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
            print(f"  torch version: {torch.__version__}  |  CUDA build: {torch.version.cuda}")
            device = torch.device("cpu")

    use_wandb = bool(cfg.train.wandb_project)
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(cfg))

    if use_wandb:
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
        prefetch_factor=2 if cfg.train.workers > 0 else None,
        collate_fn=incontext_collate_fn,
    )

    synth_method = cfg.data.synth_method or None
    train_ds = TotalSegInContextDataset(
        root=cfg.paths.totalseg, classes=train_classes,
        image_size=image_size, split="train",
        context_size=cfg.data.context_size,
        max_subjects=cfg.data.max_train_subjects,
        aug_cfg=cfg.augmentations,
        synth_method=synth_method,
        synth_unions=cfg.data.synth_unions,
        p_synth=cfg.data.p_synth,
        class_balanced=cfg.data.class_balanced,
        use_crop=cfg.data.use_crop,
    )
    val_ds = TotalSegInContextDataset(
        root=cfg.paths.totalseg, classes=val_classes,
        image_size=image_size, split="val",
        context_size=cfg.data.context_size,
        max_subjects=cfg.data.max_val_subjects,
        use_crop=cfg.data.use_crop,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False, **loader_kw)

    val_viz = collect_viz_samples(val_ds, val_classes)
    print(f"Collected val viz samples: {len(val_viz)} classes", flush=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg, image_size).to(device)

    if cfg.train.checkpoint:
        ckpt_path = Path(cfg.train.checkpoint)
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location=device)
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model.name}  |  Parameters: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )

    print("Compiling model...", flush=True)
    model_module = model
    if cfg.train.tpu:
        import torch_xla
        model = torch_xla.compile(model)
    else:
        model = torch.compile(model)

    loss_fn = DiceCELoss()
    scaler  = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    results_dir = Path(cfg.paths.results)
    results_dir.mkdir(exist_ok=True)
    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{cfg.model.name}_best.pt"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_dice = 0.0
    epoch_bar = tqdm(range(1, cfg.train.epochs + 1), desc="epochs", unit="ep")

    for epoch in epoch_bar:
        tr_loss, tr_dice, tr_cls = run_epoch(
            model, model_module, train_loader, optimizer, loss_fn, scaler, device, train=True
        )
        va_loss, va_dice, va_cls = run_epoch(
            model, model_module, val_loader, optimizer, loss_fn, scaler, device, train=False
        )
        scheduler.step()

        # Re-collect after both loaders are fully consumed — workers are idle,
        # no mid-flight iterator risk.
        train_viz = collect_viz_samples(train_ds, train_classes)

        is_best = va_dice > best_val_dice
        if is_best:
            best_val_dice = va_dice
            if cfg.train.tpu:
                import torch_xla
                torch_xla.sync()
            torch.save(model_module.state_dict(), best_ckpt)

        epoch_bar.set_postfix(
            tr_loss=f"{tr_loss:.4f}", tr_dice=f"{tr_dice:.4f}",
            va_loss=f"{va_loss:.4f}", va_dice=f"{va_dice:.4f}",
            best=f"{best_val_dice:.4f}",
        )
        cls_str = "  ".join(f"{c}={v:.3f}" for c, v in sorted(va_cls.items()))
        tqdm.write(f"  [{epoch:3d}] val per-class: {cls_str}" + (" *" if is_best else ""))

        log_predictions(model, model_module, train_viz, device, "train", epoch,
                        save_dir=ckpt_dir, use_wandb=use_wandb)
        log_predictions(model, model_module, val_viz,   device, "val",   epoch,
                        save_dir=ckpt_dir, use_wandb=use_wandb)

        if cfg.train.tpu:
            mem = 0.0
        else:
            mem = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        if use_wandb:
            log = {
                "train/loss": tr_loss, "train/dice": tr_dice,
                "val/loss":   va_loss, "val/dice":   va_dice,
                "lr": scheduler.get_last_lr()[0],
                "gpu/vram_peak_gb": mem,
            }
            log.update({f"val/dice/{c}":   v for c, v in va_cls.items()})
            log.update({f"train/dice/{c}": v for c, v in tr_cls.items()})
            wandb.log(log, step=epoch)

    if use_wandb:
        wandb.summary["best_val_dice"] = best_val_dice
        wandb.finish()
    print(f"\nBest val Dice: {best_val_dice:.4f}  →  {best_ckpt}")


if __name__ == "__main__":
    main()

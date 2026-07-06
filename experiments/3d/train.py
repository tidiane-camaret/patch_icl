"""
Fine-tune Medverse for 3D in-context segmentation — the harness twin of
experiments/3d/eval.py, mirroring experiments/2d/train.py. Shares the train loader
(common.train_loader) and per-class val loop (evaluate.evaluate_classes).

Loss / optimizer / scheduler are config-driven (train.*), defaulting to the
Medverse/Neuroverse3D recipe: Adam(3e-5) + ReduceLROnPlateau + 50·SmoothL3-L1.
Focused on 128³ inputs (Medverse runs level=1, no AR); AR teacher forcing is
deferred (see docs/superpowers/specs/2026-07-06-3d-medverse-eval-harness-design.md).

Best checkpoint (by mean val Dice) is saved so experiments/3d/eval.py can reload it.

    python experiments/3d/train.py experiment=3d/medverse
    python experiments/3d/train.py experiment=3d/medverse train.loss=bce_dice train.optimizer=adamw
"""

import datetime
import math
import random
import sys
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling common/evaluate (dir '3d')

from data.totalseg_classes import resolve_classes
from common import DEVICE, _source_root, train_loader
from evaluate import evaluate_classes


def _autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                          enabled=DEVICE.type == "cuda")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SmoothL3L1(nn.Module):
    """Modified smooth-L1 (Neuroverse3D / Hu et al. 2025): cubic below beta, linear above."""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        n = torch.abs(pred - target)
        b = self.beta
        loss = torch.where(n < b, 0.333 * n ** 3 / b ** 2, n + 0.333 * b ** 3 - b)
        return loss.mean()


def _soft_dice(prob, target, eps: float = 1e-6):
    p, g = prob.flatten(1), target.flatten(1)
    inter = (p * g).sum(1)
    den = p.sum(1) + g.sum(1)
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def build_loss(cfg):
    """Return loss_fn(logits, target) -> scalar, selected by cfg.train.loss."""
    name = cfg.train.get("loss", "smooth_l1")
    if name == "smooth_l1":
        crit = SmoothL3L1(beta=float(cfg.train.get("smooth_l1_beta", 1.0)))
        scale = float(cfg.train.get("loss_scale", 50.0))
        return lambda logits, target: scale * crit(torch.sigmoid(logits.float()), target)
    if name == "bce_dice":
        w = float(cfg.train.get("dice_weight", 1.0))
        return lambda logits, target: (
            F.binary_cross_entropy_with_logits(logits.float(), target)
            + w * _soft_dice(torch.sigmoid(logits.float()), target))
    raise ValueError(f"unknown train.loss {name!r} (smooth_l1 | bce_dice)")


@torch.no_grad()
def _hard_dice(logits, target):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return (2 * inter + 1) / (den + 1)


# ---------------------------------------------------------------------------
# Optimizer / scheduler (config-driven)
# ---------------------------------------------------------------------------

def build_optimizer(cfg, params):
    name = cfg.train.get("optimizer", "adam")
    lr, wd = float(cfg.train.lr), float(cfg.train.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown train.optimizer {name!r} (adam | adamw)")


def build_scheduler(cfg, optimizer, total_steps, steps_per_epoch):
    """Return (scheduler, step_per_batch). Plateau steps on val Dice in main()."""
    name = cfg.train.get("scheduler", "plateau")
    lr = float(cfg.train.lr)
    if name == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=float(cfg.train.get("lr_factor", 0.5)),
            patience=int(cfg.train.get("lr_patience", 5)),
            min_lr=lr * float(cfg.train.get("lr_min_factor", 0.01)))
        return sch, False
    if name == "cosine":
        warmup = int(cfg.train.get("warmup_epochs", 1) * steps_per_epoch)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), True
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0), True
    raise ValueError(f"unknown train.scheduler {name!r} (plateau | cosine | constant)")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig):
    """Instantiate the trainable model (medverse only for now; extensible)."""
    name = cfg.get("model", "medverse")
    if name == "medverse":
        from src.benchmark_models.medverse import MedverseModel
        mk = {"sw_roi_size": tuple(cfg.data.image_size)}  # val predict = single ROI
        if cfg.train.get("base_ckpt"):
            mk["ckpt_path"] = cfg.train.base_ckpt
        return MedverseModel(device=DEVICE, **mk), name
    raise ValueError(f"unknown model {name!r} (medverse)")


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, step_per_batch, loss_fn, cfg, epoch):
    net = model.model
    net.train()
    total, dice_sum, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False)
    for batch in pbar:
        lbl  = batch["label"].to(DEVICE, non_blocking=True).float()   # (B,D,H,W)
        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            logits = model.train_forward(batch["image"], batch["context_in"],
                                         batch["context_out"])          # (B,1,D,H,W)
            target = lbl.unsqueeze(1)
            loss = loss_fn(logits, target)
        loss.backward()
        if cfg.train.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
        optimizer.step()
        if step_per_batch:
            scheduler.step()

        total += loss.item()
        dice_sum += _hard_dice(logits.float(), target)
        n += 1
        pbar.set_postfix(loss=f"{total/n:.4f}", dice=f"{dice_sum/n:.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.1e}")
    return total / max(n, 1), dice_sum / max(n, 1)


@torch.no_grad()
def validate_mean(model, cfg, classes):
    """Mean val Dice via the shared per-class eval loop (uses model.predict)."""
    model.model.eval()
    rows, _ = evaluate_classes(model, cfg, classes, split="val")
    valid = [r for r in rows if "mean_dice" in r]
    mean_dice = sum(r["mean_dice"] for r in valid) / len(valid) if valid else float("nan")
    return mean_dice, rows


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _, root, is_mri = _source_root(cfg)
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    image_size = tuple(cfg.data.image_size)
    print(f"Device: {DEVICE} | model={cfg.get('model','medverse')} | size={image_size} "
          f"| K={cfg.data.context_size} | loss={cfg.train.get('loss','smooth_l1')} "
          f"| opt={cfg.train.get('optimizer','adam')} lr={cfg.train.lr} "
          f"| sched={cfg.train.get('scheduler','plateau')} | val classes={len(val_classes)}")

    loader = train_loader(cfg)
    model, model_name = build_model(cfg)
    net = model.model
    print(f"Trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.1f}M")

    if cfg.train.get("checkpoint"):
        ckpt = torch.load(cfg.train.checkpoint, map_location=DEVICE, weights_only=False)
        model.load_finetuned(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"Resumed weights from {cfg.train.checkpoint}")

    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, net.parameters())
    steps = max(1, len(loader))
    scheduler, step_per_batch = build_scheduler(cfg, optimizer, cfg.train.epochs * steps, steps)

    wb_on = bool(cfg.wandb.get("project"))
    run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name,
                     mode="online" if wb_on else "disabled",
                     config=OmegaConf.to_container(cfg, resolve=True))
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or model_name
    out_dir = Path(cfg.train.out_dir) / f"{datetime.date.today():%Y-%m-%d}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    print(f"Checkpoints -> {ckpt_path}")

    best = -1.0
    for epoch in range(cfg.train.epochs):
        t0 = time.perf_counter()
        loss, tr_dice = train_epoch(model, loader, optimizer, scheduler, step_per_batch,
                                    loss_fn, cfg, epoch)
        log = {"epoch": epoch, "train/loss": loss, "train/dice": tr_dice,
               "train/lr": optimizer.param_groups[0]["lr"], "time/epoch_s": time.perf_counter() - t0}

        if epoch % cfg.train.get("eval_every", 1) == 0 or epoch == cfg.train.epochs - 1:
            val_dice, rows = validate_mean(model, cfg, val_classes)
            log["val/dice"] = val_dice
            log.update({f"val/dice/{r['class']}": r["mean_dice"] for r in rows if "mean_dice" in r})
            if not step_per_batch:  # plateau: step on the val metric
                scheduler.step(val_dice)
            tqdm.write(f"  [e{epoch}] loss={loss:.4f} train_dice={tr_dice:.4f} "
                       f"val_dice={val_dice:.4f} (best {max(best, val_dice):.4f})")
            if val_dice > best:
                best = val_dice
                torch.save({
                    "model": net.state_dict(), "model_name": model_name,
                    "image_size": list(image_size), "context_size": cfg.data.context_size,
                    "best_val_dice": best, "epoch": epoch,
                    "data": OmegaConf.to_container(cfg.data, resolve=True),
                }, ckpt_path)
                log["val/best_dice"] = best
        wandb.log(log)

    print(f"Done. Best val Dice={best:.4f} -> {ckpt_path}")
    run.finish()


if __name__ == "__main__":
    main()

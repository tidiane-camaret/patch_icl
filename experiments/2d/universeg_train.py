"""Fine-tune (or train from scratch) UniverSeg on a 2D in-context dataset.

No trainer existed for UniverSeg: eval.py / universeg.py only evaluate it, and
pfn_seg.py is ImagePFN-specific (Muon + transformer signature). This is a minimal
AdamW trainer — BCE + soft-Dice on the native-resolution logit, per-epoch val Dice,
best-checkpoint saving, wandb. UniverSeg is a small (~1.2M-param) conv model, so
full fine-tuning is cheap.

The saved best.pt is loadable by eval.py's universeg branch (eval.checkpoint=...),
so the trained model can be swept across target_modes with the existing harness.

Usage:
    # fine-tune pretrained UniverSeg on omniSynth, mixed target_mode, 4x4 @ 128px
    python experiments/2d/universeg_train.py --config-name universeg_train \
        synth=omniglot synth.scene.target_mode=mix synth.scene.grid=4
"""

import datetime
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, build_dataset, hard_dice, log_summary, make_loader
from pfn_train import soft_dice_loss


def _autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                          enabled=DEVICE.type == "cuda")


def train_epoch(model, loader, optimizer, scheduler, cfg, epoch) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train e{epoch}")
    for batch in pbar:
        if batch is None:
            continue
        img = batch["image"].to(DEVICE, non_blocking=True)        # (B,1,H,W)
        lbl = batch["label"].to(DEVICE, non_blocking=True).float()
        cin = batch["context_in"].to(DEVICE, non_blocking=True)   # (B,K,1,H,W)
        cout = batch["context_out"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            out = model(img, context_in=cin, context_out=cout, mode="train")
            logit = out["final_logit"].float()                    # (B,1,H,W)
        bce = F.binary_cross_entropy_with_logits(logit, lbl)
        dice = soft_dice_loss(torch.sigmoid(logit), lbl)
        loss = bce + cfg.train.dice_weight * dice

        loss.backward()
        if cfg.train.get("grad_clip", None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()

        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total / n:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    per_ds: dict[str, list[float]] = defaultdict(list)
    per_label: dict[str, list[float]] = defaultdict(list)
    for batch in tqdm(loader, desc="val", leave=False):
        if batch is None:
            continue
        img = batch["image"].to(DEVICE, non_blocking=True)
        cin = batch["context_in"].to(DEVICE, non_blocking=True)
        cout = batch["context_out"].to(DEVICE, non_blocking=True)
        labels = batch["label"]
        with _autocast():
            out = model(img, context_in=cin, context_out=cout, mode="val")
        preds = (out["final_logit"] > 0).float().cpu()
        for b in range(len(batch["dataset"])):
            d = hard_dice(preds[b, 0], labels[b, 0])
            per_ds[batch["dataset"][b]].append(d)
            per_label[f"{batch['dataset'][b]}/label_{int(batch['label_value'][b])}"].append(d)
    return per_ds, per_label


@hydra.main(config_path="../../configs/experiment/2d", config_name="universeg_train",
            version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader = make_loader(build_dataset(cfg, "train"), cfg, "train", shuffle=True)
    val_loader = make_loader(build_dataset(cfg, "val"), cfg, "val", shuffle=False)

    from src.models.universeg_baseline import UniverSegBaseline
    pretrained = bool(cfg.train.get("pretrained", True))
    print(f"Building UniverSeg (pretrained={pretrained}, size={cfg.data.image_size})...")
    model = UniverSegBaseline(pretrained=pretrained, input_size=cfg.data.image_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.get("adam_wd", 0.01))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = int(cfg.train.get("warmup_epochs", 1) * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── wandb / output dir ──────────────────────────────────────────────────────
    wandb_enabled = bool(cfg.wandb.get("enabled", True))
    run_name = cfg.wandb.name or (f"useg-train-{cfg.data.source}-s{cfg.data.image_size}"
                                  f"-{cfg.synth.scene.target_mode}")
    run = wandb.init(
        project=cfg.wandb.project, name=run_name,
        mode="online" if wandb_enabled else "disabled",
        config={
            "model": "universeg", "pretrained": pretrained,
            "source": cfg.data.source, "image_size": cfg.data.image_size,
            "context_size": cfg.data.context_size,
            "grid": cfg.synth.scene.grid, "target_mode": cfg.synth.scene.target_mode,
            "epochs": cfg.train.epochs, "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr, "epoch_length": cfg.synth.sampling.epoch_length,
        },
    )
    run_name = (wandb.run.name if wandb.run is not None else None) or run_name
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    out_dir = Path(cfg.eval.out_dir) / f"{date_str}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    print(f"Checkpoints -> {ckpt_path}")

    # ── train loop ──────────────────────────────────────────────────────────────
    best_dice = -1.0
    for epoch in range(cfg.train.epochs):
        t0 = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, scheduler, cfg, epoch)
        log = {"epoch": epoch, "train/loss": loss,
               "train/lr": scheduler.get_last_lr()[0],
               "time/epoch_s": time.perf_counter() - t0}

        if epoch % cfg.train.get("eval_every", 1) == 0 or epoch == cfg.train.epochs - 1:
            per_ds, per_label = validate(model, val_loader)
            summary = log_summary(per_ds, per_label)
            mean_dice = summary.get("dice/mean", float("nan"))
            log.update(summary)
            tqdm.write(f"  [e{epoch}] train loss={loss:.4f}  val Dice={mean_dice:.4f}"
                       f"  (best={max(best_dice, mean_dice):.4f})")
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save({
                    "model": model.state_dict(),
                    "model_name": "universeg",
                    "pretrained": pretrained,
                    "image_size": cfg.data.image_size,
                    "context_size": cfg.data.context_size,
                    "best_val_dice": best_dice, "epoch": epoch,
                    "data": OmegaConf.to_container(cfg.data, resolve=True),
                    "synth": OmegaConf.to_container(cfg.synth, resolve=True),
                }, ckpt_path)
                log["val/best_dice"] = best_dice
        wandb.log(log)

    print(f"Done. Best val Dice={best_dice:.4f}. Checkpoint: {ckpt_path}")
    run.finish()


if __name__ == "__main__":
    main()

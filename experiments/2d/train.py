"""Unified AdamW trainer for 2D in-context segmentation models.

Fuses the former universeg_train.py with the PatchSetCNN trainer: one loop trains
either model, selected by `cfg.model`:

  - universeg     : UniverSegBaseline, native-resolution (H×W) logit.
  - patchset_cnn  : PatchSetCNN, low-resolution (R×R) logit (no decoder/upsampling).

The loop is model-agnostic. Both models are called as
    model(img, context_in=cin, context_out=cout, mode=...)  -> {"final_logit": ...}
and the GT mask is avg-pooled to the logit's spatial size — a no-op for UniverSeg
(H×W) and an R×R downsample for PatchSetCNN. So the BCE + soft-Dice loss, train-Dice
monitoring, per-epoch val Dice, best-checkpoint saving and wandb logging are shared.

The saved best.pt records model_name + arch metadata so eval.py can reload it.

Usage:
    # UniverSeg on omniSynth (default config)
    python experiments/2d/train.py synth=omniglot

    # PatchSetCNN on omniSynth, low-res 16×16 prediction
    python experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot
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
from common import DEVICE, build_dataset, hard_dice, log_summary, make_loader, soft_dice
from pfn_train import soft_dice_loss


def _autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                          enabled=DEVICE.type == "cuda")


def _upsample_to(x: torch.Tensor, size) -> torch.Tensor:
    """Bilinear-resize (B,1,h,w) → (B,1,*size); no-op when already at `size`."""
    return (x if x.shape[-2:] == tuple(size)
            else F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False))


def _soft_sum(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Threshold-free Dice SUM + valid-row count (prob & target at the same res)."""
    p = prob.detach().flatten(1).float()
    g = target.detach().flatten(1).float()
    den = p.sum(1) + g.sum(1)
    ok = den > eps
    s = torch.where(ok, 2 * (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return s.sum(), ok.sum()


def _hard_sum(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Hard Dice SUM + valid-row count: pred≥0.5 vs GT>0 (prob & gt at the same res)."""
    p = (prob.detach().flatten(1).float() >= 0.5).float()
    g = (gt.detach().flatten(1).float() > 0).float()
    den = p.sum(1) + g.sum(1)
    ok = den > eps
    h = torch.where(ok, 2 * (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return h.sum(), ok.sum()


def build_model(cfg) -> tuple[torch.nn.Module, str, dict]:
    """Construct the model selected by cfg.model; return (model, name, ckpt_meta)."""
    name = cfg.get("model", "universeg")
    if name == "universeg":
        from src.models.universeg_baseline import UniverSegBaseline
        pretrained = bool(cfg.train.get("pretrained", True))
        print(f"Building UniverSeg (pretrained={pretrained}, size={cfg.data.image_size})...")
        model = UniverSegBaseline(pretrained=pretrained, input_size=cfg.data.image_size)
        return model, name, {"pretrained": pretrained}
    if name == "patchset_cnn":
        from src.models.patchset_cnn import PatchSetCNN
        a = cfg.arch
        print(f"Building PatchSetCNN (size={cfg.data.image_size}, resolution={a.resolution})...")
        model = PatchSetCNN(
            image_size=cfg.data.image_size, resolution=a.resolution,
            enc_dims=tuple(a.enc_dims), e=a.e, h=a.h, l=a.l, a=a.a,
            thinking_rows=a.thinking_rows, residual_decay=a.residual_decay,
            query_self_attn=a.get("query_self_attn", False),
        )
        return model, name, {"resolution": a.resolution, "enc_dims": list(a.enc_dims),
                             "query_self_attn": a.get("query_self_attn", False)}
    raise ValueError(f"unknown model {name!r} (universeg | patchset_cnn)")


def _target_like(lbl: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    """Avg-pool the (B,1,H,W) GT to the logit's spatial size (no-op when equal).

    UniverSeg's logit is H×W → identity; PatchSetCNN's is R×R → soft occupancy GT.
    """
    if lbl.shape[-2:] == logit.shape[-2:]:
        return lbl
    return F.adaptive_avg_pool2d(lbl, logit.shape[-2:])


def train_epoch(model, loader, optimizer, scheduler, cfg, epoch) -> tuple[float, float, float]:
    model.train()
    total, n = 0.0, 0
    soft_sum = soft_cnt = hard_sum = hard_cnt = 0.0   # on-device running sums (synced once)
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
            logit = out["final_logit"].float()                    # (B,1,h,w)
        target = _target_like(lbl, logit)                         # pooled to logit res
        bce = F.binary_cross_entropy_with_logits(logit, target)
        dice = soft_dice_loss(torch.sigmoid(logit), target)
        loss = bce + cfg.train.dice_weight * dice

        loss.backward()
        if cfg.train.get("grad_clip", None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()

        # Monitoring (not the loss): downsampled soft Dice (prob vs occupancy at logit
        # res) + native-resolution hard Dice (preds upscaled to original res vs GT).
        with torch.no_grad():
            prob = torch.sigmoid(logit)
            ss, sc = _soft_sum(prob, target)
            hs, hc = _hard_sum(_upsample_to(prob, lbl.shape[-2:]), lbl)
            soft_sum += ss; soft_cnt += sc; hard_sum += hs; hard_cnt += hc

        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total / n:.4f}",
                         dice=f"{float(soft_sum) / max(float(soft_cnt), 1):.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.1e}")
    return (total / max(n, 1),
            float(soft_sum) / max(float(soft_cnt), 1),
            float(hard_sum) / max(float(hard_cnt), 1))


@torch.no_grad()
def validate(model, loader):
    """Returns (soft_ds, soft_label), (hard_ds, hard_label) Dice score dicts.

    dice_ds_soft = threshold-free Dice of the prob map vs the pooled occupancy GT, at
    the model's (downsampled) logit res. dice = hard Dice (pred≥0.5 vs GT>0) at the
    ORIGINAL resolution — preds upscaled to H×W vs the full-res GT, so it is directly
    comparable across models/resolutions (a no-op upscale for native-res UniverSeg).
    """
    model.eval()
    soft_ds: dict[str, list[float]] = defaultdict(list)
    soft_label: dict[str, list[float]] = defaultdict(list)
    hard_ds: dict[str, list[float]] = defaultdict(list)
    hard_label: dict[str, list[float]] = defaultdict(list)
    for batch in tqdm(loader, desc="val", leave=False):
        if batch is None:
            continue
        img = batch["image"].to(DEVICE, non_blocking=True)
        lbl = batch["label"].to(DEVICE, non_blocking=True).float()
        cin = batch["context_in"].to(DEVICE, non_blocking=True)
        cout = batch["context_out"].to(DEVICE, non_blocking=True)
        with _autocast():
            out = model(img, context_in=cin, context_out=cout, mode="val")
        logit = out["final_logit"].float()
        target = _target_like(lbl, logit)               # GT pooled to logit res (soft)
        prob = torch.sigmoid(logit)
        prob_nat = _upsample_to(prob, lbl.shape[-2:])    # preds upscaled to original res (hard)
        for b in range(len(batch["dataset"])):
            key = f"{batch['dataset'][b]}/label_{int(batch['label_value'][b])}"
            s = soft_dice(prob[b, 0], target[b, 0])      # dice_ds_soft: downsampled soft
            h = hard_dice(prob_nat[b, 0], lbl[b, 0])     # dice: native-resolution hard
            soft_ds[batch["dataset"][b]].append(s); soft_label[key].append(s)
            hard_ds[batch["dataset"][b]].append(h); hard_label[key].append(h)
    return (soft_ds, soft_label), (hard_ds, hard_label)


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

    model, model_name, ckpt_meta = build_model(cfg)
    model = model.to(DEVICE)
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
    scene = cfg.synth.get("scene", None) if cfg.get("synth", None) else None
    target_mode = scene.get("target_mode", "na") if scene else "na"
    # name=None → wandb auto-generates the run name; the checkpoint dir uses it (cf. pfn_seg.py).
    run_name = cfg.wandb.name
    wandb_config = {
        "model": model_name, "source": cfg.data.source,
        "image_size": cfg.data.image_size, "context_size": cfg.data.context_size,
        "epochs": cfg.train.epochs, "batch_size": cfg.train.batch_size, "lr": cfg.train.lr,
        **ckpt_meta,
    }
    if scene is not None:
        wandb_config.update({"grid": scene.get("grid", None), "target_mode": target_mode,
                             "epoch_length": cfg.synth.sampling.epoch_length})
    run = wandb.init(
        project=cfg.wandb.project, name=run_name,
        mode="online" if wandb_enabled else "disabled", config=wandb_config,
    )
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or model_name
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    out_dir = Path(cfg.eval.out_dir) / f"{date_str}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    print(f"Checkpoints -> {ckpt_path}")

    # ── train loop ──────────────────────────────────────────────────────────────
    best_dice = -1.0
    for epoch in range(cfg.train.epochs):
        t0 = time.perf_counter()
        loss, train_soft, train_hard = train_epoch(model, train_loader, optimizer,
                                                    scheduler, cfg, epoch)
        log = {"epoch": epoch, "train/loss": loss,
               "train/dice_ds_soft": train_soft, "train/dice": train_hard,
               "train/lr": scheduler.get_last_lr()[0],
               "time/epoch_s": time.perf_counter() - t0}

        if epoch % cfg.train.get("eval_every", 1) == 0 or epoch == cfg.train.epochs - 1:
            (soft_ds, soft_label), (hard_ds, hard_label) = validate(model, val_loader)
            summary = log_summary(soft_ds, soft_label, prefix="dice_ds_soft", metric_label="ds soft")
            summary.update(log_summary(hard_ds, hard_label, prefix="dice", metric_label="native"))
            mean_dice = summary.get("dice_ds_soft/mean", float("nan"))   # checkpoint on ds soft
            log.update(summary)
            tqdm.write(f"  [e{epoch}] train loss={loss:.4f}  train ds_soft={train_soft:.4f}"
                       f"  val ds_soft={mean_dice:.4f} dice={summary.get('dice/mean', float('nan')):.4f}"
                       f"  (best ds_soft={max(best_dice, mean_dice):.4f})")
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save({
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "image_size": cfg.data.image_size,
                    "context_size": cfg.data.context_size,
                    "best_val_dice": best_dice, "epoch": epoch,
                    "data": OmegaConf.to_container(cfg.data, resolve=True),
                    "synth": OmegaConf.to_container(cfg.synth, resolve=True) if cfg.get("synth", None) else None,
                    **ckpt_meta,
                }, ckpt_path)
                log["val/best_dice"] = best_dice
        wandb.log(log)

    print(f"Done. Best val Dice={best_dice:.4f}. Checkpoint: {ckpt_path}")
    run.finish()


if __name__ == "__main__":
    main()

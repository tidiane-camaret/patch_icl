"""
Fine-tune Medverse for 3D in-context segmentation — the harness twin of
experiments/3d/eval.py, mirroring experiments/2d/train.py. Shares the train loader
(common.train_loader) and per-class val loop (evaluate.evaluate_classes).

Loss / optimizer / scheduler are config-driven (train.*), defaulting to the
Medverse/Neuroverse3D recipe: Adam(3e-5) + ReduceLROnPlateau + 50·SmoothL3-L1.
Focused on 128³ inputs (Medverse runs level=1, no AR); AR teacher forcing is
deferred (see docs/superpowers/specs/2026-07-06-3d-medverse-eval-harness-design.md).

Best checkpoint (by mean val Dice) is saved so experiments/3d/eval.py can reload it.

    python experiments/3d/train.py                       # medverse on totalseg (default)
    python experiments/3d/train.py dataset=omnisynth3d   # train on omniSynth-3D
    python experiments/3d/train.py train.loss=bce_dice train.optimizer=adamw
"""

import collections
import datetime
import math
import os
import random
import socket
import sys
import tempfile
import time
from pathlib import Path

# Node-local compile caches: ~/.triton and ~/.cache live on shared NFS, so a cuda_utils.so
# compiled on a node with a newer GLIBC poisons the cache for nodes with an older GLIBC
# ("GLIBC_2.34 not found"). Key the cache by hostname on local /tmp so each node compiles
# its own artifacts. Must be set BEFORE torch is imported (cf. experiments/2d/train.py).
_cache_root = os.path.join(tempfile.gettempdir(), f"{os.environ.get('USER', 'user')}_compile_{socket.gethostname()}")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "inductor"))

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
sys.path.append(str(ROOT / "experiments" / "2d"))         # reuse Muon/LAWA from the 2D trainer

from data.totalseg_classes import resolve_classes
from common import DEVICE, _source_root, train_loader, make_eval_loader
from evaluate import evaluate_classes, build_sample_table
from grid_metrics import target_like, soft_sum, hard_sum, cos_sum
from pfn_train import Muon, lawa_average   # noqa: E402  (2D trainer shared utilities)


def _autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                          enabled=DEVICE.type == "cuda")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SmoothL3L1(nn.Module):
    """Cubic smooth-L1 segmentation loss (Neuroverse3D / Hu et al. 2025), knee fixed at 1:

        L(n) = (1/3) n^3        if n < 1
             = n - 2/3          otherwise        (n = |pred - target|)

    C1-continuous at n=1 (both branches -> 1/3, slope -> 1). The knee is hardcoded per
    the paper, which has no beta parameter; a general-beta form was dropped as its linear
    branch was only continuous at beta=1 (the sole value ever used)."""
    def forward(self, pred, target):
        n = torch.abs(pred - target)
        loss = torch.where(n < 1.0, n ** 3 / 3.0, n - 2.0 / 3.0)
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
        crit = SmoothL3L1()
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
        # Weight source is driven entirely by train.checkpoint (see main()'s loader):
        #   "orig_weights" -> released Medverse.ckpt, "random" -> from scratch,
        #   <path> -> our finetuned best.pt (built with released weights here, then
        #   overridden by load_finetuned in main()).
        if cfg.train.get("checkpoint") == "random":
            mk["random_init"] = True  # train from scratch (ignores pretrained weights)
        return MedverseModel(device=DEVICE, **mk), name
    if name == "patchset3d":
        from src.models.patchset3d import PatchSet3D
        a = cfg.arch
        arch = {
            "resolution": a.resolution, "enc_dims": list(a.enc_dims),
            "e": a.e, "h": a.h, "l": a.l, "a": a.a,
            "thinking_rows": a.thinking_rows, "residual_decay": a.residual_decay,
            "fourier_bands": a.get("fourier_bands", 8),
            "mask_patch_size": a.get("mask_patch_size", 1),
            "mask_patch_decode_size": a.get("mask_patch_decode_size", 1),
            "context_id_embed": a.get("context_id_embed", False),
            "max_context": a.get("max_context", 16),
            "full_attn": a.get("full_attn", False),
            "query_self_attn": a.get("query_self_attn", False),
            "image_size": list(cfg.data.image_size),
        }
        return PatchSet3D(**arch), name
    raise ValueError(f"unknown model {name!r} (medverse | patchset3d)")


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizers, scheduler, step_per_batch, loss_fn, cfg, epoch,
                is_patchset=False):
    """optimizers is a list: [AdamW] for medverse, [AdamW, Muon] for patchset3d (the
    scheduler drives AdamW = optimizers[0]; Muon is unscheduled, cf. experiments/2d/train.py)."""
    net = getattr(model, "model", model)
    net.train()
    total, dice_sum, soft_run, n = 0.0, 0.0, 0.0, 0
    gh = ghc = gs = gsc = gc = gcc = 0.0          # grid-metric running sums
    rd = None
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False)
    for batch in pbar:
        lbl = batch["label"].to(DEVICE, non_blocking=True).float()     # (B,D,H,W)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with _autocast():
            if is_patchset:
                out = model(batch["image"].to(DEVICE, non_blocking=True),
                            context_in=batch["context_in"].to(DEVICE, non_blocking=True),
                            context_out=batch["context_out"].to(DEVICE, non_blocking=True),
                            mode="train")
                logits = out["final_logit"].float()                    # (B,1,Rd,Rd,Rd)
                target = target_like(lbl.unsqueeze(1), logits)         # GT pooled to grid
            else:
                logits = model.train_forward(batch["image"], batch["context_in"],
                                             batch["context_out"])      # (B,1,D,H,W)
                target = lbl.unsqueeze(1)
            loss = loss_fn(logits, target)
        loss.backward()
        if cfg.train.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()
        if step_per_batch:
            scheduler.step()

        total += loss.item()
        dice_sum += _hard_dice(logits.float(), target)
        soft_run += 1.0 - _soft_dice(torch.sigmoid(logits.float()), target).item()
        n += 1
        if is_patchset:
            rd = logits.shape[-1]
            prob = torch.sigmoid(logits.float())
            h, hc = hard_sum(prob, target); gh += h; ghc += hc
            s, sc = soft_sum(prob, target); gs += s; gsc += sc
            c, cc = cos_sum(prob, target);  gc += c; gcc += cc
        pbar.set_postfix(loss=f"{total/n:.4f}", dice=f"{dice_sum/n:.4f}",
                         soft=f"{soft_run/n:.4f}", lr=f"{optimizers[0].param_groups[0]['lr']:.1e}")
    grid = {}
    if is_patchset and rd is not None:
        grid[f"dice_ds@{rd}"] = float(gh) / max(float(ghc), 1)
        grid[f"dice_ds_soft@{rd}"] = float(gs) / max(float(gsc), 1)
        grid[f"cossim@{rd}"] = float(gc) / max(float(gcc), 1)
    return total / max(n, 1), dice_sum / max(n, 1), soft_run / max(n, 1), grid


@torch.no_grad()
def validate_mean(model, cfg, classes, loader=None, loss_fn=None):
    """Mean val Dice via the shared per-class eval loop (uses model.predict).

    Reuses `loader` (built once in main) so the val dataset isn't rebuilt each epoch.
    Passing `logits_fn`/`loss_fn` adds val soft-Dice + val loss (from single-ROI logits,
    the training criterion on val). Returns (mean_dice, mean_soft_dice, mean_loss, rows, cases).
    """
    net = getattr(model, "model", model)
    net.eval()
    # NB: for patchset3d, train_forward returns NATIVE-res logits, so val/loss and
    # val/dice_soft are computed at native res (comparable to Medverse) while train/loss
    # is at grid res — the two loss scales are intentionally not directly comparable.
    rows, cases = evaluate_classes(model, cfg, classes, split="val", loader=loader,
                                   logits_fn=model.train_forward, loss_fn=loss_fn,
                                   grid_res=getattr(model, "grid_size", None))
    valid = [r for r in rows if "mean_dice" in r]
    mean_dice = sum(r["mean_dice"] for r in valid) / len(valid) if valid else float("nan")
    soft = [r["mean_soft_dice"] for r in valid if "mean_soft_dice" in r]
    mean_soft = sum(soft) / len(soft) if soft else float("nan")
    losses = [c["loss"] for c in cases if "loss" in c]
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    return mean_dice, mean_soft, mean_loss, rows, cases


@hydra.main(config_path="../../configs/experiment/3d", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 tensor cores for fp32 matmuls

    if cfg.data.get("source") == "anchor_synth3d":
        # anchor_synth3d groups val by object shape (each item's label_name = its shape).
        from common import anchor_shapes
        val_classes = anchor_shapes(cfg)
    elif cfg.data.get("source", "totalseg") == "omnisynth3d":
        # omniSynth3D val classes come from the tile-cache pool (the label_names the
        # dataset emits), not label_stats.csv — mirrors eval.py's resolution.
        from src.datasets.omniSynth.bank_totalseg import get_or_build_totalseg_bank
        s3 = cfg.synth3d
        root = s3.get("tiles_root") or cfg.paths.totalseg
        classes = resolve_classes(s3.get("classes") or (),
                                  totalseg_root=cfg.paths.get("totalseg"))
        bank = get_or_build_totalseg_bank(
            root, tuple(s3.get("size", cfg.data.image_size)),
            "val", tuple(classes))
        val_classes = [bank.alphabet(c) for c in bank.task_ids()]
    else:
        _, root, is_mri = _source_root(cfg)
        val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    image_size = tuple(cfg.data.image_size)
    print(f"Device: {DEVICE} | model={cfg.get('model','medverse')} | size={image_size} "
          f"| K={cfg.data.context_size} | loss={cfg.train.get('loss','smooth_l1')} "
          f"| opt={cfg.train.get('optimizer','adam')} lr={cfg.train.lr} "
          f"| sched={cfg.train.get('scheduler','plateau')} | val classes={len(val_classes)}")

    loader = train_loader(cfg)
    val_loader = make_eval_loader(cfg, val_classes, split="val")  # built once, reused every eval
    model, model_name = build_model(cfg)
    is_patchset = model_name == "patchset3d"
    net = getattr(model, "model", model)
    if is_patchset:
        net.to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6:.1f}M")

    # train.checkpoint is the single weight-source knob. Sentinels ("orig_weights",
    # "random") are handled at model construction (build_model); only an actual path
    # loads weights here — our finetuned best.pt for medverse, a raw resume for patchset3d.
    checkpoint = cfg.train.get("checkpoint")
    if checkpoint and checkpoint not in ("orig_weights", "random"):
        ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        if is_patchset:
            # Strip the `_orig_mod.` prefix torch.compile may have left on saved keys.
            net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        else:
            model.load_finetuned(sd)
        print(f"Resumed weights from {checkpoint}")

    # Compile only the transformer submodule (like experiments/2d/train.py): it is pure
    # tensor ops so it graph-compiles cleanly, whereas the conv encoder's adaptive_avg_pool3d
    # / trilinear resamples with data-dependent windows would graph-break. Encoder stays eager.
    # Compiled AFTER the checkpoint load so warm-starting a raw state_dict isn't blocked by the
    # `_orig_mod.` prefix; the prefix is stripped again when this run saves (see below).
    # Muon's Newton–Schulz orthogonalization is pure tensor ops → compile it too (cf. 2D).
    if is_patchset and cfg.arch.get("compile", False) and hasattr(net, "transformer"):
        net.transformer = torch.compile(net.transformer, dynamic=True)
        import pfn_train
        pfn_train._newtonschulz5_batched = torch.compile(pfn_train._newtonschulz5_batched)
        print("Compiled net.transformer + Newton–Schulz (dynamic=True); conv encoder runs eager")

    loss_fn = build_loss(cfg)
    # Optimizers (cf. experiments/2d/train.py): patchset3d trains its transformer 2D weight
    # matrices with Muon (Newton–Schulz orthogonalized grads) + AdamW on everything else +
    # LAWA checkpoint averaging. Other models use the single config-driven optimizer. The
    # cosine/plateau scheduler drives AdamW only (= optimizers[0]); Muon is unscheduled.
    use_muon = is_patchset and cfg.train.get("muon", True)
    if use_muon:
        muon_params = [p for n, p in net.named_parameters()
                       if p.requires_grad and p.ndim == 2 and "transformer" in n]
        adam_params = [p for n, p in net.named_parameters()
                       if p.requires_grad and not (p.ndim == 2 and "transformer" in n)]
        optimizer = torch.optim.AdamW(adam_params, lr=float(cfg.train.lr),
                                      weight_decay=float(cfg.train.get("weight_decay", 0.01)))
        optimizers = [optimizer]
        if muon_params:
            optimizers.append(Muon(
                muon_params,
                lr=cfg.train.get("muon_lr_scale", 0.1) * float(cfg.train.lr),
                momentum=cfg.train.get("muon_momentum", 0.96),
                weight_decay=cfg.train.get("muon_wd", 0.1)))
        print(f"Muon on {len(muon_params)} transformer matrices + AdamW on {len(adam_params)} "
              f"other tensors, LAWA k={cfg.train.get('lawa_k', 10)}")
    else:
        optimizer = build_optimizer(cfg, net.parameters())
        optimizers = [optimizer]
    steps = max(1, len(loader))
    scheduler, step_per_batch = build_scheduler(cfg, optimizer, cfg.train.epochs * steps, steps)

    # LAWA checkpoint-averaging buffer (patchset3d + Muon only): a CPU state_dict is pushed
    # each epoch; at eval the queue is averaged into the model, evaluated + saved, then the raw
    # training weights are restored so optimization continues from them (cf. 2D trainer).
    lawa_queue = collections.deque(maxlen=cfg.train.get("lawa_k", 10)) if use_muon else None

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
        loss, tr_dice, tr_soft, tr_grid = train_epoch(
            model, loader, optimizers, scheduler, step_per_batch, loss_fn, cfg, epoch,
            is_patchset=is_patchset)
        log = {"epoch": epoch, "train/loss": loss, "train/dice": tr_dice,
               "train/dice_soft": tr_soft,
               "train/lr": optimizer.param_groups[0]["lr"], "time/epoch_s": time.perf_counter() - t0}
        log.update({f"train/{k}": v for k, v in tr_grid.items()})

        if lawa_queue is not None:   # push this epoch's raw weights to the LAWA buffer
            lawa_queue.append({k: v.cpu().clone() for k, v in net.state_dict().items()})

        if epoch % cfg.train.get("eval_every", 1) == 0 or epoch == cfg.train.epochs - 1:
            # Eval (and any checkpoint saved below) uses LAWA-averaged weights; the raw
            # training weights are restored after so optimization continues from them.
            saved = lawa_average(lawa_queue, net, DEVICE) if lawa_queue is not None else None
            val_dice, val_soft, val_loss, rows, cases = validate_mean(
                model, cfg, val_classes, loader=val_loader, loss_fn=loss_fn)
            log["val/dice"] = val_dice
            log["val/dice_soft"] = val_soft
            log["val/loss"] = val_loss
            log.update({f"val/dice/{r['class']}": r["mean_dice"] for r in rows if "mean_dice" in r})
            if is_patchset:
                rd = net.grid_size
                for mkey, label in (("mean_dice_ds", "dice_ds"),
                                    ("mean_dice_ds_soft", "dice_ds_soft"),
                                    ("mean_cossim", "cossim")):
                    vals = [r[mkey] for r in rows if mkey in r]
                    if vals:
                        log[f"val/{label}@{rd}"] = sum(vals) / len(vals)
            if wb_on:  # per-sample detail table (mirrors experiments/2d train.py's val/samples)
                log["val/samples"] = build_sample_table(cases, epoch=epoch)
            if not step_per_batch:  # plateau: step on the val metric
                scheduler.step(val_dice)
            tqdm.write(f"  [e{epoch}] loss={loss:.4f} train_dice={tr_dice:.4f} "
                       f"val_dice={val_dice:.4f} val_soft={val_soft:.4f} val_loss={val_loss:.4f} "
                       f"(best {max(best, val_dice):.4f})")
            if val_dice > best:
                best = val_dice
                # Strip the `_orig_mod.` prefix torch.compile leaves on transformer keys so the
                # checkpoint is compile-agnostic (eval/resume load into an un-compiled model).
                sd = {k.replace("_orig_mod.", ""): v for k, v in net.state_dict().items()}
                torch.save({
                    "model": sd, "model_name": model_name,
                    "image_size": list(image_size), "context_size": cfg.data.context_size,
                    "best_val_dice": best, "epoch": epoch,
                    "data": OmegaConf.to_container(cfg.data, resolve=True),
                    # arch (patchset3d only): lets eval.py rebuild the exact architecture
                    # from the checkpoint instead of re-supplying arch.* overrides.
                    "arch": (OmegaConf.to_container(cfg.arch, resolve=True)
                             if "arch" in cfg else None),
                }, ckpt_path)
                log["val/best_dice"] = best
            if saved is not None:   # restore raw training weights after LAWA-averaged eval
                net.load_state_dict(saved)
        wandb.log(log)

    print(f"Done. Best val Dice={best:.4f} -> {ckpt_path}")
    run.finish()


if __name__ == "__main__":
    main()

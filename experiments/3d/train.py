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

# Broken-usr-merge nodes (thor, odin): /bin is a REAL dir (not a symlink to /usr/bin) and
# comes first on PATH, so bare `gcc`/`g++` resolve to /bin/*, derive prefix `/`, and fail —
# g++ omits its C++ headers (torch.compile/inductor dies on `#include <algorithm>`), gcc
# can't find cc1 (Triton dies). Point CC/CXX at the absolute /usr/bin paths (prefix `/usr`)
# so both work. No-op on usr-merged nodes. Must precede `import torch` (Triton reads CC).
import shutil as _shutil
if not os.path.islink("/bin"):
    for _var, _tool in (("CC", "gcc"), ("CXX", "g++")):
        _abs = f"/usr/bin/{_tool}"
        _found = _shutil.which(_tool)
        if _var not in os.environ and _found and _found.startswith("/bin/") and os.path.exists(_abs):
            os.environ[_var] = _abs

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
from evaluate import evaluate_classes, build_sample_table, measure_flops
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


def model_output_is_prob(cfg) -> bool:
    """True when the model output is already a [0,1] probability map, so it must NOT be
    sigmoided before loss/metrics. Medverse regresses the mask straight into [0,1] — its
    released checkpoint has loss_seg='smoothl3_l1' and no output activation (verified: raw
    output ranges ~[0,1], and its own predict() thresholds the raw output at 0.5). patchset3d
    emits real logits, which do need a sigmoid. Sigmoiding Medverse's [0,1] output pins every
    voxel to foreground (sigmoid(x)>=0.5 for x>=0), collapsing dice to ~0.1."""
    return cfg.get("model", "medverse") == "medverse"


def _to_prob(logits, is_prob: bool):
    """Map raw model output to a probability map. For is_prob (medverse) the output is already
    a probability but the plain-conv head can dip slightly outside [0,1]; clamp so it's a valid
    probability — without it, tiny-negative background summed over ~262k voxels drives the
    soft-Dice denominator negative and the soft metric below 0. Clamp is a no-op for the >=0.5
    hard threshold. Non-prob models get a sigmoid."""
    return logits.clamp(0.0, 1.0) if is_prob else torch.sigmoid(logits)


def _st_clamp(x, lo, hi):
    """Straight-through clamp: forward value is clamp(x, lo, hi) but the gradient passes
    through as identity (d/dx == 1 everywhere). A plain torch.clamp has ZERO gradient outside
    [lo, hi]; used on the soft-Dice term so it keeps a bounded gradient if the output briefly
    leaves [0,1]. NB the clamp's real danger for Medverse is not the dead zone but the opposite:
    clamp makes bce+dice BLIND to how far outside [0,1] the unbounded conv output goes, so
    nothing bounds the raw output and it random-walks to ~1e24 and overflows to NaN. That is
    fixed by the out-of-bounds anchor in build_loss, not here. See docs/logs.md 2026-07-29."""
    return x + (x.clamp(lo, hi) - x).detach()


def build_loss(cfg):
    """Return loss_fn(logits, target) -> scalar, selected by cfg.train.loss.

    Losses operate on the probability map via _to_prob — no sigmoid for models whose output
    is already a probability (medverse; see model_output_is_prob)."""
    name = cfg.train.get("loss", "smooth_l1")
    is_prob = model_output_is_prob(cfg)
    if name == "smooth_l1":
        crit = SmoothL3L1()
        scale = float(cfg.train.get("loss_scale", 50.0))
        return lambda logits, target: scale * crit(_to_prob(logits.float(), is_prob), target)
    if name == "bce_dice":
        w = float(cfg.train.get("dice_weight", 1.0))
        if is_prob:   # output already in [0,1] -> plain BCE (with_logits would sigmoid again)
            eps = 1e-6
            oob_w = float(cfg.train.get("oob_weight", 10.0))

            def _bce_dice_prob(logits, target):
                out = logits.float()
                prob_bce = out.clamp(eps, 1 - eps)     # hard clamp: value-safe log, smooth in-range
                prob_dice = _st_clamp(out, 0.0, 1.0)
                # F.binary_cross_entropy is autocast-UNSAFE and the train forward runs under
                # bf16 autocast; compute it in fp32 with autocast disabled. Can't use the
                # autocast-safe with_logits variant here — the output is already a probability,
                # not a logit, so with_logits would sigmoid it again (the bug we fixed).
                with torch.autocast(device_type=DEVICE.type, enabled=False):
                    bce = F.binary_cross_entropy(prob_bce, target.float())
                # Out-of-bounds anchor: Medverse's plain-conv head is UNBOUNDED, and the clamps
                # above make bce+dice saturate (blind) once the output leaves [0,1] — so nothing
                # keeps the raw output in range and it random-walks to ~1e24 and overflows to NaN
                # (verified: exp31 resumed from best.pt diverged this way; docs/logs.md 2026-07-29).
                # This quadratic penalty is exactly 0 while the output stays in [0,1] (never fights
                # in-range learning) and only pulls it back when it escapes — pins |out| to ~[0,6].
                oob = ((out - out.clamp(0.0, 1.0)) ** 2).mean()
                return bce + w * _soft_dice(prob_dice, target) + oob_w * oob
            return _bce_dice_prob
        return lambda logits, target: (
            F.binary_cross_entropy_with_logits(logits.float(), target)
            + w * _soft_dice(torch.sigmoid(logits.float()), target))
    raise ValueError(f"unknown train.loss {name!r} (smooth_l1 | bce_dice)")


@torch.no_grad()
def _hard_dice(logits, target, is_prob: bool = False):
    pred = (_to_prob(logits, is_prob) >= 0.5).float()
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
            "encoder": a.get("encoder", "conv"),
            "encoder_frozen": a.get("encoder_frozen", True),
            "primus_sidecar": a.get("primus_sidecar", None),
            "img_embed_mlp": a.get("img_embed_mlp", False),
            "encoder_stage": a.get("encoder_stage", None),
            "encoder_native_grid": a.get("encoder_native_grid", False),
            "encoder_spacing_aware": a.get("encoder_spacing_aware", False),
            "encoder_precision": a.get("encoder_precision", "bf16"),
            "transformer_rope": a.get("transformer_rope", False),
            "rope_theta": a.get("rope_theta", 100.0),
            "feat_norm": a.get("feat_norm", "context"),
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
    is_prob = model_output_is_prob(cfg)           # medverse output is already a probability
    total, dice_sum, soft_run, n = 0.0, 0.0, 0.0, 0
    gh = ghc = gs = gsc = gc = gcc = 0.0          # grid-metric running sums
    rd = None
    # Optional per-phase timing: data-wait (perf_counter between steps) + image-encode and
    # attention GPU time (CUDA events on net.encoder / net.transformer, each called once per
    # forward). The loop already syncs every step (loss.item()), so reading elapsed_time is
    # cheap; OFF by default (train.profile_timing) → zero overhead. patchset3d + CUDA only.
    prof = bool(cfg.train.get("profile_timing", False)) and is_patchset and DEVICE.type == "cuda"
    tsum, hooks, prof_items = {"data": 0.0, "encode": 0.0, "attn": 0.0}, [], 0
    if prof:
        ee = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        ea = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        hooks = [net.encoder.register_forward_pre_hook(lambda m, i: ee[0].record()),
                 net.encoder.register_forward_hook(lambda m, i, o: ee[1].record()),
                 net.transformer.register_forward_pre_hook(lambda m, i: ea[0].record()),
                 net.transformer.register_forward_hook(lambda m, i, o: ea[1].record())]
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False)
    t_prev = time.perf_counter()
    for batch in pbar:
        if prof:
            tsum["data"] += (time.perf_counter() - t_prev) * 1000
            prof_items += batch["image"].shape[0]      # per-item = per-step ÷ batch size
        lbl = batch["label"].to(DEVICE, non_blocking=True).float()     # (B,D,H,W)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with _autocast():
            if is_patchset:
                # Per-batch physical spacing (the batch sampler makes it constant across the
                # batch) → one shared RoPE table in the spacing-aware frozen encoder. None
                # when not spacing-aware (encoder falls back to the train-pitch identity grid).
                spacing = (float(batch["spacing"][0, 0])
                           if getattr(net, "spacing_aware", False) and "spacing" in batch else None)
                out = model(batch["image"].to(DEVICE, non_blocking=True),
                            context_in=batch["context_in"].to(DEVICE, non_blocking=True),
                            context_out=batch["context_out"].to(DEVICE, non_blocking=True),
                            mode="train", spacing=spacing)
                logits = out["final_logit"].float()                    # (B,1,Rd,Rd,Rd)
                target = target_like(lbl.unsqueeze(1), logits)         # GT pooled to grid
            else:
                logits = model.train_forward(batch["image"], batch["context_in"],
                                             batch["context_out"])      # (B,1,D,H,W)
                target = lbl.unsqueeze(1)
            # Fail fast + legibly on a non-finite forward: a NaN here otherwise passes through
            # clamp() into F.binary_cross_entropy, tripping an async CUDA device-side assert
            # (input in [0,1]) that surfaces at a later .backward() with a useless stack. Catch
            # it at the source with the stats that say whether the MODEL diverged (NaN logits)
            # vs the loss (finite logits). One cheap reduction per step.
            lf = logits.float()
            if not torch.isfinite(lf).all():
                raise RuntimeError(
                    f"non-finite forward @ epoch {epoch} step {n}: logits "
                    f"nan={torch.isnan(lf).any().item()} inf={torch.isinf(lf).any().item()} "
                    f"min={torch.nan_to_num(lf).min().item():.3g} "
                    f"max={torch.nan_to_num(lf).max().item():.3g}; target_fg={target.float().mean().item():.3g}. "
                    f"Model diverged (see docs/logs.md 2026-07-29): bce_dice near the clamp "
                    f"boundary; don't resume from the collapse-edge best.pt, start from orig_weights.")
            loss = loss_fn(logits, target)
        loss.backward()
        if cfg.train.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()
        if step_per_batch:
            scheduler.step()

        total += loss.item()
        dice_sum += _hard_dice(logits.float(), target, is_prob)
        soft_run += 1.0 - _soft_dice(_to_prob(logits.float(), is_prob), target).item()
        n += 1
        if is_patchset:
            rd = logits.shape[-1]
            prob = torch.sigmoid(logits.float())
            h, hc = hard_sum(prob, target); gh += h; ghc += hc
            s, sc = soft_sum(prob, target); gs += s; gsc += sc
            c, cc = cos_sum(prob, target);  gc += c; gcc += cc
        pbar.set_postfix(loss=f"{total/n:.4f}", dice=f"{dice_sum/n:.4f}",
                         soft=f"{soft_run/n:.4f}", lr=f"{optimizers[0].param_groups[0]['lr']:.1e}")
        if prof:
            torch.cuda.synchronize()
            tsum["encode"] += ee[0].elapsed_time(ee[1])
            tsum["attn"]   += ea[0].elapsed_time(ea[1])
            t_prev = time.perf_counter()
    for h in hooks:
        h.remove()
    grid = {}
    if is_patchset and rd is not None:
        grid[f"dice_ds@{rd}"] = float(gh) / max(float(ghc), 1)
        grid[f"dice_ds_soft@{rd}"] = float(gs) / max(float(gsc), 1)
        grid[f"cossim@{rd}"] = float(gc) / max(float(gcc), 1)
    if prof and n:
        pi = max(prof_items, 1)                        # total tasks profiled (Σ batch sizes)
        bs = prof_items / n                            # avg batch size
        for k in ("data", "encode", "attn"):
            grid[f"time/{k}_ms"] = tsum[k] / n         # per-step (per-batch) wall time
            grid[f"time/{k}_ms_item"] = tsum[k] / pi   # per-item (÷ batch size): B-comparable
        tqdm.write(f"  [e{epoch}] per-step: data {tsum['data']/n:5.0f}ms | "
                   f"encode {tsum['encode']/n:5.0f}ms | attn {tsum['attn']/n:5.0f}ms"
                   f"  ||  per-item (÷{bs:.0f}): data {tsum['data']/pi:4.0f}ms | "
                   f"encode {tsum['encode']/pi:4.0f}ms | attn {tsum['attn']/pi:4.0f}ms")
    return total / max(n, 1), dice_sum / max(n, 1), soft_run / max(n, 1), grid


@torch.no_grad()
def _feature_sim_trace(net, loader, n_tasks):
    """Per-layer feature-matching trace on a val subsample (patchset3d only): transfer_dice
    + retrieval_at1 at the transformer INPUT (img token after the trainable img_embed+pos,
    pre-attention, tag 'transformer_input') and after each block ('L{i}'), all at res=R.
    NB: this is NOT the frozen encoder output. One hooked forward per task —
    cheap on a subsample — traces how the jointly-trained encoder and transformer co-evolve.

    Returns a list of per-task records {class, subject, fs_<name>_dice, fs_<name>_retr, ...}
    keyed so main() can both average them (epoch curves) and merge them into the per-sample
    val table by (class, subject)."""
    from feature_sim.adapters import PatchSet3DEncoderAdapter
    from feature_sim.metrics import label_transfer, retrieval_at1
    from feature_sim.labels import grid_labels
    adapter = PatchSet3DEncoderAdapter(net)
    R, records, seen = adapter.R, [], 0
    for batch in loader:
        subjects = batch.get("subjects", [None] * batch["image"].shape[0])
        names = batch["label_names"]
        # Per-batch physical spacing for the spacing-aware encoder's RoPE (constant across the
        # batch); None when not spacing-aware -> encoder uses the train-pitch identity grid.
        spacing = (float(batch["spacing"][0, 0])
                   if getattr(net, "spacing_aware", False) and "spacing" in batch else None)
        for b in range(batch["image"].shape[0]):
            if seen >= n_tasks:
                break
            image = batch["image"][b:b + 1].to(DEVICE)
            cin = batch["context_in"][b:b + 1].to(DEVICE)
            cout = batch["context_out"][b:b + 1].to(DEVICE)
            K = cin.shape[1]
            tl = grid_labels(batch["label"][b], R, threshold=None).flatten().to(DEVICE)
            cl = torch.stack([grid_labels(cout[0, k], R, threshold=None).flatten()
                              for k in range(K)]).flatten().to(DEVICE)
            rec = {"class": names[b], "subject": subjects[b]}
            for name, tf, cf in adapter.transformer_trace(image, cin, cout, spacing=spacing):
                rec[f"fs_{name}_dice"] = round(label_transfer(tf[0], tl, cf[0], cl)["transfer_dice"], 4)
                rec[f"fs_{name}_retr"] = round(retrieval_at1(tf[0], tl, cf[0], cl), 4)
            records.append(rec)
            seen += 1
        if seen >= n_tasks:
            break
    return records


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
    # patchset3d: predict == threshold(train_forward), so reuse the logits (one forward, no
    # separate predict pass) and run eval under bf16 (matches training, avoids recompiling the
    # compiled encoder/transformer between dtypes). Off for medverse to keep its val byte-identical.
    fast_eval = cfg.get("model", "medverse") == "patchset3d"
    rows, cases = evaluate_classes(model, cfg, classes, split="val", loader=loader,
                                   logits_fn=model.train_forward, loss_fn=loss_fn,
                                   grid_res=getattr(model, "grid_size", None),
                                   output_is_prob=model_output_is_prob(cfg),
                                   autocast=fast_eval, reuse_logits=fast_eval)
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
    # Set of class names the model trains on — fills the val sample table's `in_train` flag.
    _, _troot, _is_mri = _source_root(cfg)
    train_classes = set(resolve_classes(cfg.data.train_classes, _troot, is_mri=_is_mri))
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
    # Param counts + compute (GFLOPs of one predict() call, same metric eval.py logs so the
    # two are comparable). Measured before any torch.compile so FlopCounterMode isn't tripped
    # by graph breaks; FLOP count is weight-independent so pre-checkpoint-load is fine.
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in net.parameters())
    flops = measure_flops(model, image_size, cfg.data.context_size, DEVICE)
    gflops = flops["total"]
    _brk = "  ".join(f"{k}={flops[k]:.2f}" for k in ("encoder", "transformer")
                     if flops[k] is not None)
    print(f"Params: {n_trainable/1e6:.1f}M trainable / {n_total/1e6:.1f}M total "
          f"({100 * n_trainable / max(n_total, 1):.0f}%) | predict GFLOPs: {gflops:.2f}"
          f"{('  [' + _brk + ']') if _brk else ''} (K={cfg.data.context_size}, size={image_size})")

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
        msg = "Compiled net.transformer + Newton–Schulz (dynamic=True)"
        # The frozen Primus ViT is the dominant per-step cost yet runs eager by default. It is a
        # pure attention/MLP stack (compiles cleanly, unlike the conv encoder), so compile the eva
        # block stack — the interpolate/_down_to that would graph-break live OUTSIDE it (in
        # _preprocess/_encode_batch). dynamic=True so target/context batch-size differences don't
        # retrigger compilation. No-op for encoder=conv.
        enc = getattr(net, "encoder", None)
        which_enc = cfg.arch.get("encoder", "conv")
        if which_enc == "primus" and enc is not None and hasattr(enc, "primus"):
            enc.primus.eva = torch.compile(enc.primus.eva, dynamic=True)
            msg += " + frozen Primus eva stack"
        elif which_enc == "tap_ct" and enc is not None and hasattr(enc, "model"):
            # Frozen TAP ViT: the dominant per-step cost. It encodes one volume at a time at a
            # fixed T, so shapes are static -> dynamic=False. The SDPA-patched attention
            # (F.scaled_dot_product_attention) compiles cleanly. Mirrors the feature-sim path.
            enc.model = torch.compile(enc.model, dynamic=False)
            msg += " + frozen tap-ct ViT"
        else:
            msg += "; conv encoder runs eager"
        print(msg)

    # Medverse perf knobs (medverse.grad_checkpoint / .compile — see model/medverse.yaml,
    # benchmarked in experiments/3d/bench_arch.py). Applied AFTER weight load so names/keys
    # match the checkpoint; gradient checkpointing keeps param names, compile adds an
    # `_orig_mod.` prefix that the save path below already strips. At B=4 both together beat
    # the eager baseline on time and memory.
    if not is_patchset:
        mcfg = cfg.get("medverse", {})
        if mcfg.get("grad_checkpoint", False):
            n_ck = model.enable_gradient_checkpointing()
            print(f"Gradient checkpointing enabled on {n_ck} Medverse U-Net conv blocks")
        if mcfg.get("compile", False):
            model.compile_net()
            print("Compiled the Medverse net (torch.compile); first batch will be slow")

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
        optimizer = build_optimizer(cfg, (p for p in net.parameters() if p.requires_grad))
        optimizers = [optimizer]
    steps = max(1, len(loader))
    scheduler, step_per_batch = build_scheduler(cfg, optimizer, cfg.train.epochs * steps, steps)

    # LAWA checkpoint-averaging buffer (patchset3d + Muon only): a CPU state_dict is pushed
    # each epoch; at eval the queue is averaged into the model, evaluated + saved, then the raw
    # training weights are restored so optimization continues from them (cf. 2D trainer).
    lawa_queue = collections.deque(maxlen=cfg.train.get("lawa_k", 10)) if use_muon else None
    # LAWA averages only TRAINABLE weights. With a frozen Primus encoder (~145M params) the
    # rest of state_dict never changes, so copying it to CPU each epoch — and averaging it into
    # itself — is pure waste (~580 MB/epoch × lawa_k in RAM). Snapshot the trainable key set once,
    # AFTER compile so the `_orig_mod.` prefixes match state_dict keys.
    lawa_keys = ({n for n, p in net.named_parameters() if p.requires_grad}
                 if lawa_queue is not None else None)

    wb_on = bool(cfg.wandb.get("project"))
    # Config-group selections (dataset=, augmentations=, model=, cluster=, experiment=) are
    # packaged `_global_`, so they merge into data:/paths:/... and leave no key of their own in
    # cfg — only Hydra's runtime choices record which group was picked. Log them explicitly so
    # e.g. `dataset` is visible in wandb (cf. the resolved cfg, which omits it).
    from hydra.core.hydra_config import HydraConfig
    wb_config = OmegaConf.to_container(cfg, resolve=True)
    wb_config["hydra_choices"] = dict(HydraConfig.get().runtime.choices)
    wb_config["params_trainable_M"] = round(n_trainable / 1e6, 3)
    wb_config["params_total_M"] = round(n_total / 1e6, 3)
    wb_config["gflops"] = round(gflops, 2)
    if flops["encoder"] is not None:
        wb_config["gflops_encoder"] = round(flops["encoder"], 2)
    if flops["transformer"] is not None:
        wb_config["gflops_transformer"] = round(flops["transformer"], 2)
    run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name,
                     mode="online" if wb_on else "disabled",
                     config=wb_config)
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

        if lawa_queue is not None:   # push this epoch's raw TRAINABLE weights to the LAWA buffer
            lawa_queue.append({k: v.cpu().clone()
                               for k, v in net.state_dict().items() if k in lawa_keys})

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
            fs_every = int(cfg.train.get("feature_sim_every", 0) or 0)
            if is_patchset and fs_every and epoch % fs_every == 0:
                # Encoder->per-layer correspondence trace (transfer_dice/retrieval at res=R),
                # on a val subsample; see _feature_sim_trace. Log per-rep means as epoch curves
                # AND attach the per-task values onto the matching val cases (by class+subject)
                # so they land as columns in the per-sample table below.
                records = _feature_sim_trace(net, val_loader,
                                             int(cfg.train.get("feature_sim_n_tasks", 128)))
                fs_keys = [k for k in (records[0] if records else {}) if k.startswith("fs_")]
                for k in fs_keys:  # fs_<rep>_<dice|retr> -> val/<transfer_dice|retrieval>/<rep>
                    vals = [r[k] for r in records if not math.isnan(r.get(k, float("nan")))]
                    rep, metric = k[3:].rsplit("_", 1)
                    label = "transfer_dice" if metric == "dice" else "retrieval"
                    if vals:
                        log[f"val/{label}/{rep}"] = sum(vals) / len(vals)
                by_id = {(r["class"], r["subject"]): r for r in records}
                for c in cases:  # merge per-sample fs_ values onto the matching case
                    r = by_id.get((c["class"], c.get("subject")))
                    if r:
                        c.update({k: v for k, v in r.items() if k.startswith("fs_")})
            if wb_on:  # per-sample detail table (mirrors experiments/2d train.py's val/samples)
                log["val/samples"] = build_sample_table(cases, epoch=epoch,
                                                         train_classes=train_classes)
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
                net.load_state_dict(saved, strict=False)  # `saved` holds only trainable keys
        wandb.log(log)

    print(f"Done. Best val Dice={best:.4f} -> {ckpt_path}")
    run.finish()


if __name__ == "__main__":
    main()

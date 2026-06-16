"""
Stage-2 multilevel patch refinement training.

Frozen res-16 ImagePFN (stage 1) + frozen UniverSeg encoder produce coarse target
predictions and res-32 features; we sample 256 patches/image and train a PatchSetPFN
to refine the uncertain target patches. Checkpoint selection: native-resolution
`dice/mean` (the refined res-32 map upsampled to 128), aggregated exactly like pfn_seg.py.

Metric naming convention (resolutions are read from the model/config, not hardcoded —
R1 = stage-1 native res, R2 = cfg.sample.grid_res, H = cfg.data.image_size; the defaults
below are R1=16, R2=32, H=128):
  `dice_r{R1}` / `dice_r{R2}` — dice computed AT that resolution: `dice_r16/mean` =
      stage-1 pred @R1 vs GT@R1 (== pfn_seg's low-res dice); `dice_r32/mean` = refined
      map @R2 vs GT@R2.
  `_s1` / `_s2`   — stage-1 (coarse) / stage-2 (refined) compared at a SHARED resolution
      (resolution can't distinguish them there): the native-@H pair `dice_s1/mean`
      (stage-1 R1→H, == pfn_seg headline baseline) vs `dice/mean` (stage-2 R2→H), and
      the @R2 `refine/{scope}/*_s1` vs `*_s2`.
`refine/uncertain/delta_err` (|error| reduction on the sampled boundary region, s2 vs s1)
is kept as a diagnostic.

Usage:
    python experiments/2d/multilevel/train.py
    python experiments/2d/multilevel/train.py arch.mask_prior=patch train.lr=5e-4
"""

import collections
import datetime
import math
import os
import socket
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_cache_root = os.path.join(tempfile.gettempdir(), f"{os.environ.get('USER','user')}_compile_{socket.gethostname()}")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "inductor"))

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _ROOT)
# Cache patch_icl's src before common.py inserts ic_segmentation's shadowing src.
from src.datasets.medsegbench import MedSegBenchDataset   # noqa: F401
from src.models.pfn_seg_2d import ImagePFN
from src.models.patchset_pfn import PatchSetPFN
from src.models.pretrained_encoders import UniverSegFeatureEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # experiments/2d
from common import DEVICE, TaggedDataset, collate, downsample_mask, hard_dice, soft_dice
from pfn_train import Muon, augment, lawa_average, soft_dice_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))       # multilevel
from pipeline import build_patch_batch

from torch.utils.data import DataLoader, RandomSampler


def build_split_loader(cfg, split, shuffle):
    datasets = [cfg.data.dataset] if cfg.data.dataset else None
    ds = MedSegBenchDataset(split=split, context_size=cfg.data.context_size,
                            image_size=cfg.data.image_size, datasets=datasets)
    if split == "val" and cfg.eval.max_per_label:
        import random
        groups = {}
        for i, (name, _, lv) in enumerate(ds.samples):
            groups.setdefault((name, lv), []).append(i)
        keep = []
        for idxs in groups.values():
            keep.extend(random.sample(idxs, min(cfg.eval.max_per_label, len(idxs))))
        ds.samples = [ds.samples[i] for i in sorted(keep)]
    bs = cfg.train.batch_size if split == "train" else cfg.eval.batch_size
    nw = cfg.train.workers   if split == "train" else cfg.eval.workers
    max_train = cfg.data.get("max_train_samples", None)
    sampler = (RandomSampler(ds, replacement=False, num_samples=max_train)
               if split == "train" and max_train is not None else None)
    return DataLoader(TaggedDataset(ds), batch_size=bs,
                      shuffle=(shuffle and sampler is None), sampler=sampler,
                      num_workers=nw, collate_fn=collate,
                      pin_memory=DEVICE.type == "cuda",
                      persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)


def load_stage1(cfg):
    """Load the frozen res-16 ImagePFN from its checkpoint (arch read from the .pt)."""
    ckpt = torch.load(cfg.train.stage1_checkpoint, map_location="cpu", weights_only=False)
    arch, img_size = ckpt["arch"], ckpt["image_size"]
    resolution = arch.get("resolution", img_size // arch["patch_size"] if "patch_size" in arch else None)
    input_patch_size = arch.get("input_patch_size", img_size // resolution)
    image_encoder, feature_dim = None, None
    if arch.get("image_encoder", "patch") == "universeg":
        image_encoder = UniverSegFeatureEncoder(
            level=arch.get("feature_level", "all"), input_size=128,
            resize_to_input=arch.get("encoder_resize_to_input", False)).to(DEVICE)
        feature_dim = image_encoder.feature_dim
    model = ImagePFN(resolution=resolution, image_size=img_size,
                     input_patch_size=input_patch_size,
                     image_encoder=image_encoder, feature_dim=feature_dim,
                     e=arch["e"], h=arch["h"], l=arch["l"], a=arch["a"],
                     thinking_rows=arch["thinking_rows"],
                     residual_decay=arch["residual_decay"]).to(DEVICE)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Stage-1 loaded: resolution={resolution}, encoder={arch.get('image_encoder','patch')}")
    return model


def patch_loss(logits, batch, cfg):
    target = batch["qry_gt"]
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(torch.sigmoid(logits.float()), target)
    return bce + cfg.train.dice_weight * dice


def train_epoch(model, loader, stage1, encoder, optimizers, cfg, epoch):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        # Augment whole images first; coarse pred + features are computed on them.
        if cfg.aug.enabled:
            img = batch["image"].unsqueeze(1).to(DEVICE)            # (B,1,1,H,W)
            ctx = batch["context_in"].to(DEVICE)                    # (B,K,1,H,W)
            imgs = torch.cat([ctx, img], dim=1)
            cout = batch["context_out"].to(DEVICE)
            msks = torch.cat([cout, batch["label"].unsqueeze(1).to(DEVICE)], dim=1)
            K = ctx.shape[1]
            imgs, msks = augment(imgs, msks, K, cfg.aug)
            batch = {**batch, "context_in": imgs[:, :K].cpu(), "image": imgs[:, K, 0:1].cpu(),
                     "context_out": msks[:, :K].cpu(), "label": msks[:, K, 0:1].cpu()}
        pb = build_patch_batch(batch, stage1, encoder, cfg, DEVICE, cfg.sample.train,
                               stochastic=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = model(pb["sup_feat"], pb["sup_label"], pb["sup_ij"],
                           pb["qry_feat"], pb["qry_prior"], pb["qry_ij"], cfg.sample.grid_res,
                           stage1_think=pb["stage1_think"] if cfg.arch.use_stage1_thinking else None)
            loss = patch_loss(logits, pb, cfg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()
        total += loss.item(); n += 1
        pbar.set_postfix(loss=f"{total/n:.4f}")
    return total / max(n, 1)


def _accum(d, pred_v, coarse_v, gt_v):
    """Append per-sample metrics (s2 refined vs s1 coarse, against gt_v) to accumulator d.

    s1 = stage-1 coarse value, s2 = stage-2 refined prediction. Both are scored on the
    SAME res-32 cells (the s1 value is the stage-1 map upsampled to res-32), so the suffix
    is the STAGE, not a resolution. pred_v/coarse_v/gt_v are 1-D tensors over a cell set.
    delta_err > 0 = s2 beats s1. Hard Dice binarizes gt at >=0.5 (majority vote); soft
    Dice uses the raw soft fractions (shape, no thresh)."""
    gt_bin = (gt_v >= 0.5).float()
    d["derr"].append((coarse_v - gt_v).abs().mean().item() - (pred_v - gt_v).abs().mean().item())
    hd2, hd1 = hard_dice(pred_v, gt_bin), hard_dice(coarse_v, gt_bin)
    sd2, sd1 = soft_dice(pred_v, gt_v),   soft_dice(coarse_v, gt_v)
    d["hd_s2"].append(hd2);     d["hd_s1"].append(hd1)
    d["sd_s2"].append(sd2);     d["sd_s1"].append(sd1)
    d["dd"].append(hd2 - hd1);  d["sdd"].append(sd2 - sd1)   # per-sample improvement (s2 - s1, >0 = better)


@torch.no_grad()
def run_eval(model, loader, stage1, encoder, lawa_queue, cfg, epoch):
    saved = lawa_average(lawa_queue, model, DEVICE)
    model.eval()
    # Three scopes: the boundary-core (uncertain) queries, all M sampled queries, and the
    # full res-32 image (coarse map with the sampled cells overwritten by stage-2).
    scopes = ("uncertain", "sampled", "full")
    acc = {s: {k: [] for k in ("derr", "hd_s2", "hd_s1", "sd_s2", "sd_s1", "dd", "sdd")} for s in scopes}
    cert_err_s2, cert_err_s1 = [], []
    # Resolutions, all derived (not hardcoded): R1 = stage-1 native res (from the frozen
    # model, N = R1²); R2 = grid/refined res (config); H = native image size (config).
    H, R2 = cfg.data.image_size, cfg.sample.grid_res
    R1 = int(round(stage1.N ** 0.5))
    K_R1, K_R2 = f"dice_r{R1}/mean", f"dice_r{R2}/mean"   # true-resolution dice metric keys
    # Per-resolution Dice over the whole val set, each computed AT its named resolution
    # (no cross-res upsampling fudge):
    #   {K_R1} : stage-1 pred @R1 vs GT@R1          (== pfn_seg.py's low-res dice)
    #   {K_R2} : refined map @R2 vs GT@R2
    #   dice/mean   : refined upsampled to native @H vs GT@H   (pfn_seg headline)
    #   dice_s1/mean: stage-1 upsampled to native @H vs GT@H   (s1 deployed; pfn_seg baseline)
    per_ds_native:    dict[str, list[float]] = defaultdict(list)  # s2 refined @H (headline)
    per_ds_s1_native: dict[str, list[float]] = defaultdict(list)  # s1 @H (pfn_seg baseline)
    per_ds_r16:       dict[str, list[float]] = defaultdict(list)  # s1 @R1 (true low res)
    per_ds_r32:       dict[str, list[float]] = defaultdict(list)  # s2 @R2 (true grid res)

    def _to_native(flat):  # (N,) res-32 logits/probs → (H, H)
        return F.interpolate(flat.reshape(1, 1, R2, R2).float(), size=(H, H),
                             mode="bilinear", align_corners=False).reshape(H, H)

    pbar = tqdm(loader, desc=f"eval e{epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        pb = build_patch_batch(batch, stage1, encoder, cfg, DEVICE, cfg.sample.eval,
                               stochastic=not cfg.sample.eval_deterministic)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = model(pb["sup_feat"], pb["sup_label"], pb["sup_ij"],
                           pb["qry_feat"], pb["qry_prior"], pb["qry_ij"], cfg.sample.grid_res,
                           stage1_think=pb["stage1_think"] if cfg.arch.use_stage1_thinking else None)
        pred = torch.sigmoid(logits.float())
        gt, coarse, unc = pb["qry_gt"], pb["qry_coarse"], pb["qry_is_uncertain"]
        qidx, coarse_full, gt_full = pb["qry_idx"], pb["coarse_full"], pb["gt_full"]
        coarse_lr = pb["coarse_lowres"]                                   # (B, R1, R1) native stage-1 res
        B = gt.shape[0]
        for b in range(B):
            # full image: composite stage-2 predictions into the coarse map at sampled cells
            refined = coarse_full[b].clone()
            refined[qidx[b]] = pred[b]
            _accum(acc["full"], refined, coarse_full[b], gt_full[b])
            _accum(acc["sampled"], pred[b], coarse[b], gt[b])
            u = unc[b]
            if u.any():
                _accum(acc["uncertain"], pred[b][u], coarse[b][u], gt[b][u])
            c = ~u
            if c.any():
                cert_err_s2.append((pred[b][c] - gt[b][c]).abs().mean().item())
                cert_err_s1.append((coarse[b][c] - gt[b][c]).abs().mean().item())

            ds_name   = batch["dataset"][b]
            gt_native = batch["label"][b, 0]                                  # (H,W) cpu

            # ── true-resolution dices (each scored AT its own resolution) ──
            #   @R1: stage-1 pred vs GT pooled to R1 (majority) — same as pfn_seg's low-res dice
            gt_r1 = (downsample_mask(gt_native, R1) >= 0.5).float()
            per_ds_r16[ds_name].append(hard_dice(coarse_lr[b].cpu(), gt_r1))
            #   @R2: refined map vs GT@R2 (majority)
            gt_r2_bin = (gt_full[b] >= 0.5).float()
            per_ds_r32[ds_name].append(hard_dice(refined.cpu(), gt_r2_bin.cpu()))

            # ── native @128 dices (stage comparison at the deployment resolution) ──
            #   s2: refined res-32 → 128.   s1: stage-1 res-16 → 128 DIRECTLY (pfn_seg path).
            coarse_native = F.interpolate(coarse_lr[b][None, None].float(), size=(H, H),
                                          mode="bilinear", align_corners=False).reshape(H, H)
            per_ds_native[ds_name].append(hard_dice(_to_native(refined).cpu(), gt_native))
            per_ds_s1_native[ds_name].append(hard_dice(coarse_native.cpu(), gt_native))
    if saved is not None:
        model.load_state_dict(saved)

    # Robust nanmean: returns NaN (no warning) for empty / all-NaN inputs.
    def nanmean(xs):
        vals = [v for v in xs if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    # Refine scopes: s1 (stage-1 coarse) vs s2 (stage-2 refined), both on res-32 cells.
    metrics = {"epoch": epoch,
               "refine/certain_err_s2": nanmean(cert_err_s2),
               "refine/certain_err_s1": nanmean(cert_err_s1)}
    for s in scopes:
        d = acc[s]
        metrics[f"refine/{s}/delta_err"]       = nanmean(d["derr"])   # >0 = improvement (all res-32)
        metrics[f"refine/{s}/dice_s2"]         = nanmean(d["hd_s2"])
        metrics[f"refine/{s}/dice_s1"]         = nanmean(d["hd_s1"])
        metrics[f"refine/{s}/dice_delta"]      = nanmean(d["dd"])      # s2 - s1, per-sample, >0 = better
        metrics[f"refine/{s}/soft_dice_s2"]    = nanmean(d["sd_s2"])
        metrics[f"refine/{s}/soft_dice_s1"]    = nanmean(d["sd_s1"])
        metrics[f"refine/{s}/soft_dice_delta"] = nanmean(d["sdd"])     # s2 - s1, per-sample, >0 = better

    # Per-resolution mean Dice, aggregated like pfn_seg.py (mean over all samples).
    _flat = lambda d: [x for sc in d.values() for x in sc if not np.isnan(x)]
    f_native, f_s1, f_r16, f_r32 = _flat(per_ds_native), _flat(per_ds_s1_native), _flat(per_ds_r16), _flat(per_ds_r32)
    metrics["dice/mean"]    = float(np.mean(f_native)) if f_native else float("nan")  # ← s2 @H (checkpoint; compare to pfn_seg)
    metrics["dice_s1/mean"] = float(np.mean(f_s1))     if f_s1     else float("nan")  # s1 @H (pfn_seg headline baseline)
    metrics[K_R1]           = float(np.mean(f_r16))    if f_r16    else float("nan")  # s1 @R1 (true low res; pfn_seg low-res dice)
    metrics[K_R2]           = float(np.mean(f_r32))    if f_r32    else float("nan")  # s2 @R2 (true grid res)
    if not (np.isnan(metrics["dice/mean"]) or np.isnan(metrics["dice_s1/mean"])):
        metrics["dice/margin_vs_s1"] = metrics["dice/mean"] - metrics["dice_s1/mean"]  # both @128
    for k, v in per_ds_native.items():
        metrics[f"dice/dataset/{k}"] = nanmean(v)

    tqdm.write(
        f"  [e{epoch}] dice @r{R1} {metrics[K_R1]:.4f}  @r{R2} {metrics[K_R2]:.4f}  "
        f"@{H} s1 {metrics['dice_s1/mean']:.4f}→s2 {metrics['dice/mean']:.4f}  |  "
        f"Δerr unc={metrics['refine/uncertain/delta_err']:.4f} full={metrics['refine/full/delta_err']:.4f}  "
        f"soft-dice full {metrics['refine/full/soft_dice_s1']:.3f}→{metrics['refine/full/soft_dice_s2']:.3f}")
    wandb.log(metrics)
    return metrics["dice/mean"]   # ← checkpoint selection metric (native, pfn_seg-comparable)


@hydra.main(config_path="../../../configs/experiment/2d", config_name="multilevel", version_base=None)
def main(cfg: DictConfig):
    import random
    random.seed(cfg.train.seed); np.random.seed(cfg.train.seed); torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high"); torch.backends.cudnn.benchmark = True

    print("Building data loaders...")
    train_loader = build_split_loader(cfg, "train", shuffle=True)
    val_loader   = build_split_loader(cfg, "val",   shuffle=False)

    stage1  = load_stage1(cfg)
    encoder = UniverSegFeatureEncoder(level=cfg.arch.feature_level, input_size=128).to(DEVICE)
    feature_dim = encoder.feature_dim

    # Stage-1 thinking memory: dim e1 read from the frozen stage-1's thinking tokens.
    stage1_dim = stage1.thinking.tokens.shape[-1] if cfg.arch.use_stage1_thinking else None
    if cfg.arch.use_stage1_thinking:
        print(f"Stage-1 thinking memory enabled (e1={stage1_dim}, n_think={stage1.thinking.n})")

    model = PatchSetPFN(feature_dim=feature_dim, e=cfg.arch.e, h=cfg.arch.h, l=cfg.arch.l,
                        a=cfg.arch.a, thinking_rows=cfg.arch.thinking_rows,
                        residual_decay=cfg.arch.residual_decay, fourier_bands=cfg.arch.fourier_bands,
                        mask_prior=cfg.arch.mask_prior,
                        mask_patch_size=cfg.data.image_size // cfg.sample.grid_res,
                        stage1_dim=stage1_dim,
                        query_self_attn=cfg.arch.query_self_attn).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchSetPFN: {trainable:,} trainable params")

    if cfg.train.get("checkpoint", None):
        raw = torch.load(cfg.train.checkpoint, map_location="cpu", weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        msd = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(compat, strict=False)
        print(f"Warm-start PatchSetPFN: loaded {len(compat)}/{len(msd)} tensors")

    if cfg.arch.compile:
        model = torch.compile(model, dynamic=True)

    muon_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2 and "transformer" in n]
    adam_params = [p for n, p in model.named_parameters() if p.requires_grad and not (p.ndim == 2 and "transformer" in n)]
    opt_muon = Muon(muon_params, lr=cfg.train.muon_lr_scale * cfg.train.lr,
                    momentum=cfg.train.muon_momentum, weight_decay=cfg.train.muon_wd)
    opt_adam = torch.optim.AdamW(adam_params, lr=cfg.train.lr, weight_decay=cfg.train.adam_wd)
    def lr_lambda(epoch):
        if epoch < cfg.train.warmup_epochs:
            return (epoch + 1) / cfg.train.warmup_epochs
        t = (epoch - cfg.train.warmup_epochs) / max(cfg.train.epochs - cfg.train.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_adam, lr_lambda)
    optimizers = [opt_muon, opt_adam]

    lawa_queue = collections.deque(maxlen=cfg.train.lawa_k)
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name,   # name=None → wandb auto-generates
               config={"arch": dict(cfg.arch), "train": dict(cfg.train),
                       "data": dict(cfg.data), "sample": dict(cfg.sample)},
               mode="online" if cfg.wandb.enabled else "disabled")

    # Use the wandb-given run name; save under {date}_{run_name}, e.g. 2026-05-22_deft-field-72.
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or "run"
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    ckpt_dir = Path(cfg.eval.out_dir) / f"{date_str}_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = -1e9
    for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="epochs", dynamic_ncols=True):
        loss = train_epoch(model, train_loader, stage1, encoder, optimizers, cfg, epoch)
        scheduler.step()
        wandb.log({"epoch": epoch, "train/loss": loss, "train/lr": scheduler.get_last_lr()[0]})
        lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs:
            dice_mean = run_eval(model, val_loader, stage1, encoder, lawa_queue, cfg, epoch)
            if dice_mean > best:
                best = dice_mean
                saved = lawa_average(lawa_queue, model, DEVICE)
                torch.save({"model": model.state_dict(), "arch": dict(cfg.arch),
                            "sample": dict(cfg.sample), "image_size": cfg.data.image_size,
                            "context_size": cfg.data.context_size}, ckpt_dir / "best.pt")
                if saved:
                    model.load_state_dict(saved)
                tqdm.write(f"  [best] dice/mean={best:.4f} → {ckpt_dir}/best.pt")

    wandb.log({"best_dice_mean": best})
    wandb.finish()
    print(f"\nDone. Best dice/mean: {best:.4f}")


if __name__ == "__main__":
    main()

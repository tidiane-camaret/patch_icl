"""
Stage-2 multilevel patch refinement training.

A frozen res-16 ImagePFN (stage 1) + frozen UniverSeg encoder seed a coarse-to-fine
CHAIN over `cfg.sample.resolutions` (e.g. 16→32→64→128). Each hop is its own PatchSetPFN
(an nn.ModuleList): it samples patches on the previous level's (detached) composite,
refines them, and composites back. Levels train independently — each hop's loss only
touches its own weights — and each hop's "thinking" memory is chained (detached) into the
next. See pipeline.run_chain / refine_level and the spec/plan in docs/superpowers.

Metrics (resolutions read from config, not hardcoded; final hop's grid == native H):
  `dice_r{res}/mean` — hard Dice of each level's composite computed AT that resolution
      (res = resolutions[0] is the stage-1 baseline; the rest are the per-hop composites).
  `dice_soft_r{res}/mean` — continuous (shape) Dice of the same composites vs the soft
      (avg-pooled, un-binarized) GT at that resolution.
  `dice/mean` — alias of `dice_r{final}/mean` (the native-resolution hard Dice).
  `dice_soft/mean` — alias of `dice_soft_r{final}/mean`; CHECKPOINT selection metric
      (soft Dice at the last computed level, no upsample/threshold).
  `refine/hop{L}/{delta_err, dice_delta, soft_dice_delta}` — hop L's marginal improvement
      over its immediate input (the previous level upsampled), on its sampled cells.

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
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _ROOT)
# Cache patch_icl's src before common.py inserts ic_segmentation's shadowing src.
from src.datasets.medsegbench import MedSegBenchDataset   # noqa: F401
from src.models.pfn_seg_2d import ImagePFN
from src.models.patchset_pfn import PatchSetPFN
from src.models.pretrained_encoders import build_image_encoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # experiments/2d
from common import (DEVICE, batch_dice_sums, build_dataset, downsample_mask, hard_dice,
                    make_loader, soft_dice)
from pfn_train import Muon, augment, lawa_average, soft_dice_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))       # multilevel
from pipeline import run_chain
from zoom_pipeline import run_zoom_chain


def build_split_loader(cfg, split, shuffle):
    """Tagged, collated loader for `split`; source dispatch + policy live in common."""
    return make_loader(build_dataset(cfg, split), cfg, split, shuffle)


def load_stage1(cfg):
    """Load the frozen res-16 ImagePFN from its checkpoint (arch read from the .pt)."""
    ckpt = torch.load(cfg.train.stage1_checkpoint, map_location="cpu", weights_only=False)
    arch, img_size = ckpt["arch"], ckpt["image_size"]
    resolution = arch.get("resolution", img_size // arch["patch_size"] if "patch_size" in arch else None)
    input_patch_size = arch.get("input_patch_size", img_size // resolution)
    image_encoder, feature_dim = build_image_encoder(arch, DEVICE)
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


def _check_zoom_encoder(encoder):
    """Raise loudly if the encoder applies channel reduction or per-stage L2 norm.

    crop_pool_maps concatenates raw encode_maps() stage maps without running the
    encoder's _reduce_channels or stage_l2norm — for UniverSeg (reduce=none, no L2)
    these are both no-ops so the channel count equals feature_dim and results are
    correct.  For DINOv3 with encoder_reduce != 'none' or encoder_stage_l2norm=True
    the emitted features would have the wrong channel count / be unreduced, silently
    invalidating the warm-start and the experiment.  v1 zoom is scoped to UniverSeg;
    DINOv3 reduce/PCA zoom is out of scope until crop_pool_maps is extended."""
    reduce_kind = getattr(encoder, "_reduce_kind", "none")
    stage_l2 = getattr(encoder, "stage_l2norm", False)
    if reduce_kind != "none":
        raise NotImplementedError(
            f"crop_pool_maps does not apply the encoder's channel reduction "
            f"(_reduce_kind={reduce_kind!r}); the zoom path currently supports only "
            f"plain-concat features (e.g. UniverSeg feature_level=all, reduce=none). "
            f"DINOv3 reduce/PCA zoom is out of scope for v1.")
    if stage_l2:
        raise NotImplementedError(
            "crop_pool_maps does not apply the encoder's per-stage L2 norm "
            "(stage_l2norm=True); the zoom path currently supports only "
            "plain-concat features (e.g. UniverSeg feature_level=all, reduce=none). "
            "DINOv3 stage_l2norm zoom is out of scope for v1.")


def build_zoom_models(cfg, stage1, encoder, feature_dim):
    """ModuleList of ImagePFN hops (one per crop_sizes), warm-started from frozen stage-1.

    External-features mode: the encoder lives once in the chain; hops consume crop-pooled
    features. Warm-start loads stage-1's weights minus image_encoder.* (strict=False).
    Raises NotImplementedError for encoder configs not supported by crop_pool_maps."""
    _check_zoom_encoder(encoder)
    ckpt = torch.load(cfg.train.stage1_checkpoint, map_location="cpu", weights_only=False)
    arch, img = ckpt["arch"], ckpt["image_size"]
    resolution = int(round(stage1.N ** 0.5))
    n_hops = len(cfg.sample.crop_sizes)
    models = nn.ModuleList([
        ImagePFN(resolution=resolution, image_size=img,
                 input_patch_size=arch.get("input_patch_size", img // resolution),
                 use_external_features=True, feature_dim=feature_dim,
                 e=arch["e"], h=arch["h"], l=arch["l"], a=arch["a"],
                 thinking_rows=arch["thinking_rows"],
                 residual_decay=arch["residual_decay"]).to(DEVICE)
        for _ in range(n_hops)])
    s1 = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()
          if not k.removeprefix("_orig_mod.").startswith("image_encoder.")}
    for m in models:
        m.load_state_dict(s1, strict=False)
    assert stage1.image_embed.in_features == feature_dim, (
        f"chain encoder feature_dim {feature_dim} != stage-1 image_embed "
        f"{stage1.image_embed.in_features}; encoder must match the stage-1 checkpoint")
    print(f"Zoom ImagePFN chain: {n_hops} hops (crop_sizes={list(cfg.sample.crop_sizes)}), "
          f"warm-started from stage-1; "
          f"{sum(p.numel() for p in models.parameters() if p.requires_grad):,} params")
    return models


def patch_loss(logits, batch, cfg):
    target = batch["qry_gt"]
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(torch.sigmoid(logits.float()), target)
    return bce + cfg.train.dice_weight * dice


def train_epoch(model, loader, stage1, encoder, optimizers, cfg, epoch, chain_fn, hop_labels):
    """Returns (mean_loss, train_soft, train_hard); the two dicts are keyed by hop grid
    and hold the soft (shape) / hard Dice between each hop's prediction and its query GT
    — the tensors run_chain already produced, accumulated on-GPU and synced once at epoch
    end (no extra forward). NB: measured on the *sampled query patches* (what the hop is
    trained on), not the full grid, so it reads slightly higher than the val dice_soft_r*."""
    model.train()
    total, n = 0.0, 0
    hops = list(hop_labels)
    nh = len(hops)
    soft_sum = [torch.zeros((), device=DEVICE) for _ in range(nh)]
    soft_cnt = [torch.zeros((), device=DEVICE) for _ in range(nh)]
    hard_sum = [torch.zeros((), device=DEVICE) for _ in range(nh)]
    hard_cnt = [torch.zeros((), device=DEVICE) for _ in range(nh)]
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
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            outputs, _ = chain_fn(batch, stage1, encoder, model, cfg, cfg.sample.train,
                                  stochastic=True, device=DEVICE)
            weights = list(cfg.train.loss_weights)
            loss = sum(w * patch_loss(o["logits"], {"qry_gt": o["qry_gt"]}, cfg)
                       for w, o in zip(weights, outputs))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()
        total += loss.item(); n += 1
        # Per-hop train accuracy on the sampled query patches (no extra forward).
        with torch.no_grad():
            for L, o in enumerate(outputs):
                ss, sc, hs, hc = batch_dice_sums(torch.sigmoid(o["logits"].float()), o["qry_gt"])
                soft_sum[L] += ss; soft_cnt[L] += sc; hard_sum[L] += hs; hard_cnt[L] += hc
        pbar.set_postfix(loss=f"{total/n:.4f}")
    train_soft = {hops[L]: float((soft_sum[L] / soft_cnt[L].clamp_min(1)).item()) for L in range(nh)}
    train_hard = {hops[L]: float((hard_sum[L] / hard_cnt[L].clamp_min(1)).item()) for L in range(nh)}
    return total / max(n, 1), train_soft, train_hard


@torch.no_grad()
def run_eval(model, loader, stage1, encoder, lawa_queue, cfg, epoch):
    saved = lawa_average(lawa_queue, model, DEVICE)
    for m in model: m.eval()
    H = cfg.data.image_size
    resolutions = list(cfg.sample.resolutions)
    hops = resolutions[1:]
    per_ds = {r: defaultdict(list) for r in resolutions}             # hard dice
    per_ds_soft = {r: defaultdict(list) for r in resolutions}        # soft (shape) dice
    acc = {L: {k: [] for k in ("derr", "dd", "sdd")} for L in range(len(hops))}
    total_loss, nl = 0.0, 0   # val loss, accumulated exactly as in train_epoch

    for batch in loader:
        if batch is None: continue
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            outputs, coarse_lr = run_chain(batch, stage1, encoder, model, cfg,
                                           cfg.sample.eval,
                                           stochastic=not cfg.sample.eval_deterministic,
                                           device=DEVICE)
            # Val loss: same per-hop weighted objective the chain is trained on.
            weights = list(cfg.train.loss_weights)
            total_loss += sum(w * patch_loss(o["logits"], {"qry_gt": o["qry_gt"]}, cfg)
                              for w, o in zip(weights, outputs)).item()
            nl += 1
        B = coarse_lr.shape[0]
        for b in range(B):
            ds_name   = batch["dataset"][b]
            gt_native = batch["label"][b, 0]
            R0 = resolutions[0]
            gt_r0_soft = downsample_mask(gt_native, R0)
            gt_r0 = (gt_r0_soft >= 0.5).float()
            per_ds[R0][ds_name].append(hard_dice(coarse_lr[b].cpu(), gt_r0))
            per_ds_soft[R0][ds_name].append(soft_dice(coarse_lr[b].cpu(), gt_r0_soft))
            prev_grid = coarse_lr[b].reshape(1, 1, R0, R0).float()
            for L, grid in enumerate(hops):
                o = outputs[L]
                refined = o["refined_grid"][b]
                gt_g = (o["gt_grid"][b] >= 0.5).float()
                per_ds[grid][ds_name].append(hard_dice(refined.cpu(), gt_g.cpu()))
                per_ds_soft[grid][ds_name].append(soft_dice(refined.cpu(), o["gt_grid"][b].cpu()))
                up = F.interpolate(prev_grid, size=(grid, grid), mode="bilinear",
                                   align_corners=False).reshape(-1)
                qg, qi = o["qry_gt"][b], o["qidx"][b]
                pred_q = torch.sigmoid(o["logits"][b].float())
                coarse_q = up[qi]
                acc[L]["derr"].append((coarse_q - qg).abs().mean().item()
                                      - (pred_q - qg).abs().mean().item())
                acc[L]["dd"].append(hard_dice(pred_q, (qg >= 0.5).float())
                                    - hard_dice(coarse_q, (qg >= 0.5).float()))
                acc[L]["sdd"].append(soft_dice(pred_q, qg) - soft_dice(coarse_q, qg))
                prev_grid = refined.reshape(1, 1, grid, grid).float()
    if saved is not None:
        model.load_state_dict(saved)

    def nanmean(xs):
        v = [x for x in xs if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")
    flat = lambda d: [x for sc in d.values() for x in sc if not np.isnan(x)]

    metrics = {"epoch": epoch, "val/loss": total_loss / max(nl, 1)}
    for r in resolutions:
        metrics[f"dice_r{r}/mean"] = (float(np.mean(flat(per_ds[r])))
                                      if flat(per_ds[r]) else float("nan"))
        metrics[f"dice_soft_r{r}/mean"] = (float(np.mean(flat(per_ds_soft[r])))
                                           if flat(per_ds_soft[r]) else float("nan"))
    metrics["dice/mean"]      = metrics[f"dice_r{resolutions[-1]}/mean"]
    metrics["dice_soft/mean"] = metrics[f"dice_soft_r{resolutions[-1]}/mean"]
    for L in range(len(hops)):
        metrics[f"refine/hop{L}/delta_err"]       = nanmean(acc[L]["derr"])
        metrics[f"refine/hop{L}/dice_delta"]      = nanmean(acc[L]["dd"])
        metrics[f"refine/hop{L}/soft_dice_delta"] = nanmean(acc[L]["sdd"])
    for r in resolutions:
        for k, v in per_ds[r].items():
            metrics[f"dice/dataset_r{r}/{k}"] = nanmean(v)
        for k, v in per_ds_soft[r].items():
            metrics[f"dice_soft/dataset_r{r}/{k}"] = nanmean(v)

    tqdm.write(f"  [e{epoch}] val loss={metrics['val/loss']:.4f}  hard " + "  ".join(
        f"r{r}={metrics[f'dice_r{r}/mean']:.4f}" for r in resolutions)
        + "  | soft " + "  ".join(
        f"r{r}={metrics[f'dice_soft_r{r}/mean']:.4f}" for r in resolutions))
    wandb.log(metrics)
    for m in model: m.train()
    # Checkpoint selection on soft Dice at the last computed level (no upsample/threshold).
    return metrics["dice_soft/mean"]


@torch.no_grad()
def run_eval_zoom(model, loader, stage1, encoder, lawa_queue, cfg, epoch):
    """Eval for the zoom chain: composite is at native H, so metrics are full-res Dice
    after each hop, plus the in-bbox refine delta. Returns dice_soft/mean (ckpt metric)."""
    saved = lawa_average(lawa_queue, model, DEVICE)
    for m in model: m.eval()
    crop_sizes = list(cfg.sample.crop_sizes)
    nh = len(crop_sizes)
    per_ds      = defaultdict(list)                 # final hard dice
    per_ds_soft = defaultdict(list)                 # final soft dice
    after = [defaultdict(list) for _ in range(nh)]  # hard dice after each hop
    delta = [[] for _ in range(nh)]                 # in-bbox hard-dice gain vs prior
    total_loss, nl = 0.0, 0

    for batch in loader:
        if batch is None: continue
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            outputs, coarse_lr = run_zoom_chain(batch, stage1, encoder, model, cfg,
                                                cfg.sample.eval,
                                                stochastic=not cfg.sample.eval_deterministic,
                                                device=DEVICE)
            weights = list(cfg.train.loss_weights)
            total_loss += sum(w * patch_loss(o["logits"], {"qry_gt": o["qry_gt"]}, cfg)
                              for w, o in zip(weights, outputs)).item()
            nl += 1
        B = coarse_lr.shape[0]
        H = cfg.data.image_size
        # Stage-1 baseline composite at native H (for the in-bbox delta of hop 0).
        prev_full = F.interpolate(coarse_lr.unsqueeze(1), size=(H, H),
                                  mode="bilinear", align_corners=False)
        for L, o in enumerate(outputs):
            full = o["refined_full"]
            for b in range(B):
                ds = batch["dataset"][b]
                gt = batch["label"][b, 0]
                after[L][ds].append(hard_dice(full[b, 0].cpu(), gt))
                r0, c0 = int(o["origin"][b, 0]), int(o["origin"][b, 1]); s = o["crop_size"]
                box = (slice(r0, r0 + s), slice(c0, c0 + s))
                gtb = (gt[box] >= 0.5).float()
                delta[L].append(hard_dice(full[b, 0, box[0], box[1]].cpu(), gtb)
                                - hard_dice(prev_full[b, 0, box[0], box[1]].cpu(), gtb))
            prev_full = full
        for b in range(B):
            ds = batch["dataset"][b]; gt = batch["label"][b, 0]
            per_ds[ds].append(hard_dice(outputs[-1]["refined_full"][b, 0].cpu(), gt))
            per_ds_soft[ds].append(soft_dice(outputs[-1]["refined_full"][b, 0].cpu(), gt))
    if saved is not None:
        model.load_state_dict(saved)

    def nanmean(xs):
        v = [x for x in xs if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")
    flat = lambda d: [x for sc in d.values() for x in sc if not np.isnan(x)]

    metrics = {"epoch": epoch, "val/loss": total_loss / max(nl, 1)}
    metrics["dice/mean"]      = float(np.mean(flat(per_ds)))      if flat(per_ds)      else float("nan")
    metrics["dice_soft/mean"] = float(np.mean(flat(per_ds_soft))) if flat(per_ds_soft) else float("nan")
    for L in range(nh):
        metrics[f"dice_after_hop{L}/mean"]  = float(np.mean(flat(after[L]))) if flat(after[L]) else float("nan")
        metrics[f"refine/hop{L}/dice_delta"] = nanmean(delta[L])
    for k, v in per_ds.items():
        metrics[f"dice/dataset/{k}"] = nanmean(v)
    tqdm.write(f"  [e{epoch}] val loss={metrics['val/loss']:.4f}  "
               f"dice={metrics['dice/mean']:.4f}  soft={metrics['dice_soft/mean']:.4f}  "
               + "  ".join(f"d{cs}={metrics[f'refine/hop{L}/dice_delta']:+.4f}"
                           for L, cs in enumerate(crop_sizes)))
    wandb.log(metrics)
    for m in model: m.train()
    return metrics["dice_soft/mean"]


@hydra.main(config_path="../../../configs/experiment/2d", config_name="multilevel", version_base=None)
def main(cfg: DictConfig):
    import random
    # Augmentation params come from the single shared file referenced by
    # cfg.aug_preset (configs/augmentations/<preset>.yaml), not inlined here.
    # CLI field overrides use the +-prefix, e.g. +aug.enabled=false, and win.
    _aug = OmegaConf.load(Path(_ROOT) / "configs" / "augmentations" / f"{cfg.aug_preset}.yaml")
    with open_dict(cfg):
        cfg.aug = OmegaConf.merge(_aug, cfg.aug) if "aug" in cfg else _aug
    random.seed(cfg.train.seed); np.random.seed(cfg.train.seed); torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high"); torch.backends.cudnn.benchmark = True

    print("Building data loaders...")
    train_loader = build_split_loader(cfg, "train", shuffle=True)
    val_loader   = build_split_loader(cfg, "val",   shuffle=False)

    stage1  = load_stage1(cfg)
    # Chain encoder defaults to UniverSeg (back-compat); override with arch.image_encoder=dinov3.
    encoder, feature_dim = build_image_encoder(
        {"image_encoder": cfg.arch.get("image_encoder", "universeg"),
         "feature_level": cfg.arch.feature_level,
         "encoder_resize_to_input": cfg.arch.get("encoder_resize_to_input", False),
         "encoder_imagenet_norm": cfg.arch.get("encoder_imagenet_norm", True),
         "encoder_reduce": cfg.arch.get("encoder_reduce", "none"),
         "encoder_stage_l2norm": cfg.arch.get("encoder_stage_l2norm", False)}, DEVICE)
    # Fit/load the PCA reduction once if reduce='pca:…' (other reductions need no fit).
    if getattr(encoder, "needs_pca_fit", False):
        def _img_iter():
            for batch in train_loader:
                if batch is None:
                    continue
                img = batch["image"].to(DEVICE)            # (B, 1, H, W)
                ctx = batch["context_in"].to(DEVICE)       # (B, K, 1, H, W)
                yield torch.cat([ctx.flatten(0, 1), img], dim=0)
        encoder.ensure_pca(_img_iter(), fit_out_size=list(cfg.sample.resolutions)[1])

    # Stage-1 thinking memory: dim e1 read from the frozen stage-1's thinking tokens.
    if cfg.arch.get("use_stage1_thinking", False):
        stage1_dim = stage1.thinking.tokens.shape[-1]
        print(f"Stage-1 thinking memory enabled (e1={stage1_dim}, n_think={stage1.thinking.n})")
    else:
        stage1_dim = None

    is_zoom = cfg.arch.get("refine_arch", "patchset") == "imagepfn_zoom"
    chain_fn = run_zoom_chain if is_zoom else run_chain
    eval_fn  = run_eval_zoom if is_zoom else run_eval
    hop_labels = (list(cfg.sample.crop_sizes) if is_zoom
                  else list(cfg.sample.resolutions)[1:])

    if is_zoom:
        model = build_zoom_models(cfg, stage1, encoder, feature_dim)
    else:
        resolutions = list(cfg.sample.resolutions)
        assert resolutions[0] == int(round(stage1.N ** 0.5)), \
            f"resolutions[0]={resolutions[0]} must equal stage-1 res {int(round(stage1.N ** 0.5))}"
        # Chained thinking: hop L>0 receives the previous PatchSetPFN's thinking (dim e).
        model = nn.ModuleList([
            PatchSetPFN(feature_dim=feature_dim, e=cfg.arch.e, h=cfg.arch.h, l=cfg.arch.l,
                        a=cfg.arch.a, thinking_rows=cfg.arch.thinking_rows,
                        residual_decay=cfg.arch.residual_decay, fourier_bands=cfg.arch.fourier_bands,
                        mask_prior=cfg.arch.mask_prior,
                        mask_patch_size=cfg.data.image_size // grid,
                        stage1_dim=(stage1_dim if L == 0 else cfg.arch.e),
                        query_self_attn=cfg.arch.query_self_attn).to(DEVICE)
            for L, grid in enumerate(resolutions[1:])])
        print(f"PatchSetPFN chain: {len(model)} hops, "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    if cfg.train.get("checkpoint", None):
        raw = torch.load(cfg.train.checkpoint, map_location="cpu", weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        # `replace` (not `removeprefix`): a compiled ModuleList saves keys like
        # "0._orig_mod.transformer..." — the _orig_mod. prefix is mid-key, not leading.
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        msd = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(compat, strict=False)
        print(f"Warm-start PatchSetPFN: loaded {len(compat)}/{len(msd)} tensors")
        if not compat:
            print(f"  WARNING: warm-start loaded 0 tensors — checkpoint keys do not match "
                  f"the current chain (e.g. pre-chain/unprefixed or different ladder). "
                  f"Sample ckpt key: {next(iter(sd), '<empty>')!r}  vs model key: "
                  f"{next(iter(msd), '<empty>')!r}")

    if cfg.arch.compile:
        model = nn.ModuleList([torch.compile(m, dynamic=True) for m in model])

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
        loss, train_soft, train_hard = train_epoch(model, train_loader, stage1, encoder,
                                                    optimizers, cfg, epoch, chain_fn, hop_labels)
        scheduler.step()
        # Per-hop train accuracy mirrors the val dice_soft_r{grid} / dice_r{grid} naming.
        train_log = {"epoch": epoch, "train/loss": loss, "train/lr": scheduler.get_last_lr()[0]}
        for g in train_soft:
            train_log[f"train/dice_soft_r{g}/mean"] = train_soft[g]
            train_log[f"train/dice_r{g}/mean"] = train_hard[g]
        wandb.log(train_log)
        lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs:
            # Best on soft Dice at the last computed level (dice_soft/mean), not the
            # hard native-resolution dice/mean.
            dice_soft = eval_fn(model, val_loader, stage1, encoder, lawa_queue, cfg, epoch)
            if dice_soft > best:
                best = dice_soft
                saved = lawa_average(lawa_queue, model, DEVICE)
                # Embed training data provenance (full data config + synth knobs when
                # synthetic) alongside arch/sample so eval can report what the chain was
                # trained on; at eval time cfg.data reflects the *eval* dataset instead.
                torch.save({"model": model.state_dict(), "arch": dict(cfg.arch),
                            "sample": dict(cfg.sample), "image_size": cfg.data.image_size,
                            "context_size": cfg.data.context_size,
                            "stage1_checkpoint": cfg.train.stage1_checkpoint,
                            "data": OmegaConf.to_container(cfg.data, resolve=True),
                            "synth": (OmegaConf.to_container(cfg.synth, resolve=True)
                                      if cfg.data.get("source") == "synthetic" else None)},
                           ckpt_dir / "best.pt")
                if saved:
                    model.load_state_dict(saved)
                tqdm.write(f"  [best] dice_soft/mean={best:.4f} → {ckpt_dir}/best.pt")

    wandb.log({"best_dice_soft_mean": best})
    wandb.finish()
    print(f"\nDone. Best dice_soft/mean: {best:.4f}")


if __name__ == "__main__":
    main()

"""
Stage-2 multilevel patch refinement training.

Frozen res-16 ImagePFN (stage 1) + frozen UniverSeg encoder produce coarse target
predictions and res-32 features; we sample 256 patches/image and train a PatchSetPFN
to refine the uncertain target patches. Metric of interest: |error| reduction on the
sampled uncertain region vs the stage-1 coarse value.

Usage:
    python experiments/2d/multilevel/train.py
    python experiments/2d/multilevel/train.py arch.coarse_prior=false train.lr=5e-4
"""

import collections
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
from common import DEVICE, TaggedDataset, collate, hard_dice, soft_dice
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
        pb = build_patch_batch(batch, stage1, encoder, cfg, DEVICE)

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
    """Append per-sample metrics (stage-2 vs coarse, against gt_v) to accumulator dict d.

    pred_v/coarse_v/gt_v are 1-D tensors over a cell set. delta_err > 0 = stage-2
    beats coarse. Hard Dice binarizes gt at >=0.5 (majority vote); soft Dice uses the
    raw soft fractions (shape score, no threshold)."""
    gt_bin = (gt_v >= 0.5).float()
    d["derr"].append((coarse_v - gt_v).abs().mean().item() - (pred_v - gt_v).abs().mean().item())
    d["hd_s2"].append(hard_dice(pred_v, gt_bin));   d["hd_co"].append(hard_dice(coarse_v, gt_bin))
    d["sd_s2"].append(soft_dice(pred_v, gt_v));     d["sd_co"].append(soft_dice(coarse_v, gt_v))


@torch.no_grad()
def run_eval(model, loader, stage1, encoder, lawa_queue, cfg, epoch):
    saved = lawa_average(lawa_queue, model, DEVICE)
    model.eval()
    # Three scopes: the 192 uncertain queries, all 256 sampled queries, and the full
    # res-32 image (coarse map with the sampled cells overwritten by stage-2).
    scopes = ("uncertain", "sampled", "full")
    acc = {s: {k: [] for k in ("derr", "hd_s2", "hd_co", "sd_s2", "sd_co")} for s in scopes}
    cert_err_stage2, cert_err_coarse = [], []
    # Native-resolution hard Dice over the whole val set — the SAME headline metric
    # pfn_seg.py logs as dice/mean, so the two are directly comparable. The stage-2
    # "final" map = coarse map with sampled cells overwritten, upsampled to native res.
    H, R2 = cfg.data.image_size, cfg.sample.grid_res
    per_ds_native:        dict[str, list[float]] = defaultdict(list)  # stage-2 (refined)
    per_ds_native_coarse: dict[str, list[float]] = defaultdict(list)  # coarse baseline

    def _to_native(flat):  # (N,) res-32 logits/probs → (H, H)
        return F.interpolate(flat.reshape(1, 1, R2, R2).float(), size=(H, H),
                             mode="bilinear", align_corners=False).reshape(H, H)

    pbar = tqdm(loader, desc=f"eval e{epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        pb = build_patch_batch(batch, stage1, encoder, cfg, DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = model(pb["sup_feat"], pb["sup_label"], pb["sup_ij"],
                           pb["qry_feat"], pb["qry_prior"], pb["qry_ij"], cfg.sample.grid_res,
                           stage1_think=pb["stage1_think"] if cfg.arch.use_stage1_thinking else None)
        pred = torch.sigmoid(logits.float())
        gt, coarse, unc = pb["qry_gt"], pb["qry_coarse"], pb["qry_is_uncertain"]
        qidx, coarse_full, gt_full = pb["qry_idx"], pb["coarse_full"], pb["gt_full"]
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
                cert_err_stage2.append((pred[b][c] - gt[b][c]).abs().mean().item())
                cert_err_coarse.append((coarse[b][c] - gt[b][c]).abs().mean().item())
            # Native-res Dice vs the full-res GT (matches pfn_seg.py dice/mean)
            ds_name  = batch["dataset"][b]
            gt_native = batch["label"][b, 0]                                  # (H,W) cpu
            per_ds_native[ds_name].append(hard_dice(_to_native(refined).cpu(), gt_native))
            per_ds_native_coarse[ds_name].append(hard_dice(_to_native(coarse_full[b]).cpu(), gt_native))
    if saved is not None:
        model.load_state_dict(saved)

    # Robust nanmean: returns NaN (no warning) for empty / all-NaN inputs.
    def nanmean(xs):
        vals = [v for v in xs if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    metrics = {"epoch": epoch,
               "refine/certain_err_stage2": nanmean(cert_err_stage2),
               "refine/certain_err_coarse": nanmean(cert_err_coarse)}
    for s in scopes:
        d = acc[s]
        metrics[f"refine/{s}/delta_err"]       = nanmean(d["derr"])    # >0 = improvement
        metrics[f"refine/{s}/dice_stage2"]      = nanmean(d["hd_s2"])
        metrics[f"refine/{s}/dice_coarse"]      = nanmean(d["hd_co"])
        metrics[f"refine/{s}/soft_dice_stage2"] = nanmean(d["sd_s2"])
        metrics[f"refine/{s}/soft_dice_coarse"] = nanmean(d["sd_co"])

    # Native-res mean Dice, aggregated exactly like pfn_seg.py (mean over all samples).
    flat        = [s for sc in per_ds_native.values()        for s in sc if not np.isnan(s)]
    flat_coarse = [s for sc in per_ds_native_coarse.values() for s in sc if not np.isnan(s)]
    metrics["dice/mean"]        = float(np.mean(flat))        if flat        else float("nan")  # ← stage-2 final (compare to pfn_seg)
    metrics["dice_coarse/mean"] = float(np.mean(flat_coarse)) if flat_coarse else float("nan")  # coarse baseline
    for k, v in per_ds_native.items():
        metrics[f"dice/dataset/{k}"] = nanmean(v)

    tqdm.write(
        f"  [e{epoch}] dice/mean (native): coarse {metrics['dice_coarse/mean']:.4f} → stage2 {metrics['dice/mean']:.4f}  |  "
        f"Δerr unc={metrics['refine/uncertain/delta_err']:.4f} full={metrics['refine/full/delta_err']:.4f}  "
        f"soft-dice full {metrics['refine/full/soft_dice_coarse']:.3f}→{metrics['refine/full/soft_dice_stage2']:.3f}")
    wandb.log(metrics)
    return metrics["refine/uncertain/delta_err"]


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
                        coarse_prior=cfg.arch.coarse_prior, stage1_dim=stage1_dim,
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
    run_name = cfg.wandb.name or (f"multilevel_{'coarse' if cfg.arch.coarse_prior else 'scratch'}"
                                  f"_R{cfg.sample.grid_res}_k{cfg.data.context_size}_l{cfg.arch.l}")
    wandb.init(project=cfg.wandb.project, name=run_name,
               config={"arch": dict(cfg.arch), "train": dict(cfg.train),
                       "data": dict(cfg.data), "sample": dict(cfg.sample)},
               mode="online" if cfg.wandb.enabled else "disabled")

    ckpt_dir = Path(cfg.eval.out_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = -1e9
    for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="epochs", dynamic_ncols=True):
        loss = train_epoch(model, train_loader, stage1, encoder, optimizers, cfg, epoch)
        scheduler.step()
        wandb.log({"epoch": epoch, "train/loss": loss, "train/lr": scheduler.get_last_lr()[0]})
        lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs:
            delta = run_eval(model, val_loader, stage1, encoder, lawa_queue, cfg, epoch)
            if delta > best:
                best = delta
                saved = lawa_average(lawa_queue, model, DEVICE)
                torch.save({"model": model.state_dict(), "arch": dict(cfg.arch),
                            "sample": dict(cfg.sample), "image_size": cfg.data.image_size,
                            "context_size": cfg.data.context_size}, ckpt_dir / "best.pt")
                if saved:
                    model.load_state_dict(saved)
                tqdm.write(f"  [best] Δerr(unc)={best:.4f} → {ckpt_dir}/best.pt")

    wandb.log({"best_delta_err_uncertain": best})
    wandb.finish()
    print(f"\nDone. Best Δerr(uncertain): {best:.4f}")


if __name__ == "__main__":
    main()

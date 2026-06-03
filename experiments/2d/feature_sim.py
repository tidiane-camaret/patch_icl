"""
Feature-similarity experiment on MedSegBench using the UniverSeg encoder.

Each image is encoded independently through the UniverSeg encoder blocks with a
zero dummy support (no cross-conv influence). Target patch features are then
classified by a TabPFN model fit on context patch features and their labels.

Procedure (mirrors experiments/feature_similarity/run.py for 3D):
  1. Encode target and each context image independently → per-scale feature maps
  2. Pick encoder level(s) and pool to output_size × output_size
  3. Downsample GT masks to the same output_size
  4. For each target: fit TabPFN on context (feature, label) patches → predict target

Usage:
    python experiments/2d/feature_sim.py
    python experiments/2d/feature_sim.py feature.level=-1 feature.output_size=8
    python experiments/2d/feature_sim.py data.dataset=abdomenus data.context_size=5
"""

import sys
from collections import defaultdict
from pathlib import Path

import hydra
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/home/dpxuser/ic_segmentation")
sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")

from src.datasets.medsegbench import MedSegBenchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ENCODER_LEVELS = 4  # UniverSeg v1: 4 CrossBlocks


# ── UniverSeg image-only encoding ────────────────────────────────────────────

@torch.no_grad()
def encode_images(
    useg,
    images: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """
    Extract per-scale features from the UniverSeg encoder.

    Support is always a dummy (zeros image channel), but the label channel of
    the support can optionally carry the GT mask. This lets the cross-conv fuse
    mask location into the features without the self-support trick.

    Args:
        useg:   inner UniverSeg nn.Module
        images: (B, 1, H, W) float [0, 1]
        masks:  (B, 1, H, W) binary, optional. When provided, used as the
                support label channel (second channel of dummy support).

    Returns:
        list of NUM_ENCODER_LEVELS tensors, each (B, 64, H/2^i, W/2^i)
    """
    B, _, H, W = images.shape
    target  = images.unsqueeze(1)                                # (B, 1, 1, H, W)
    dev = images.device
    label_ch = masks.to(dev) if masks is not None \
               else torch.zeros(B, 1, H, W, device=dev)
    dummy_s = torch.cat([
        torch.zeros(B, 1, 1, H, W, device=dev),                 # image channel = 0
        label_ch.unsqueeze(1),                                   # label channel
    ], dim=2)                                                    # (B, 1, 2, H, W)

    feats = []
    for i, block in enumerate(useg.enc_blocks):
        target, dummy_s = block(target, dummy_s)
        feats.append(target.squeeze(1))  # (B, C, H', W')
        if i < len(useg.enc_blocks) - 1:
            target  = F.max_pool2d(target .squeeze(1), 2).unsqueeze(1)
            dummy_s = F.max_pool2d(dummy_s.squeeze(1), 2).unsqueeze(1)

    return feats  # index 0 = highest res, -1 = bottleneck


# ── Feature / mask helpers ────────────────────────────────────────────────────

def extract_features(
    feats: list[torch.Tensor],
    level: str | int,
    output_size: int,
) -> torch.Tensor:
    """
    Pick encoder level(s), pool/upsample to output_size, return (C, H', W').

    level="all" concatenates all levels on the channel dim.
    level=int (or str int) picks a single level (negative indexing supported).
    """
    size = (output_size, output_size)
    if str(level) == "all":
        maps = [_pool2d(f, size) for f in feats]
    else:
        idx = int(level) % len(feats)
        maps = [_pool2d(feats[idx], size)]
    return torch.cat(maps, dim=0)   # (C, H', W')


def _pool2d(feat: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """feat: (B, C, H, W) with B=1 → avg-pool to size → (C, H', W')."""
    return F.adaptive_avg_pool2d(feat.float(), size).squeeze(0)


def downsample_mask(mask: torch.Tensor, output_size: int, mode: str = "avg") -> torch.Tensor:
    """mask: (H, W) → (H', W') using avg or max pool."""
    x = mask.float().unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    size = (output_size, output_size)
    if mode == "max":
        return F.adaptive_max_pool2d(x, size).squeeze()
    return F.adaptive_avg_pool2d(x, size).squeeze()


# ── TabPFN prediction (same logic as 3D, 2D spatial) ─────────────────────────

def balance_context(
    ctx_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    bg_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample background patches to bg_ratio × effective_fg_count."""
    n_eff_fg = max(1, round(labels_flat.sum().item()))
    order    = labels_flat.argsort(descending=True)
    keep_fg  = order[:n_eff_fg]
    bg_cands = order[n_eff_fg:]
    if len(bg_cands) == 0:
        return ctx_flat, labels_flat
    n_bg  = min(len(bg_cands), max(1, int(n_eff_fg * bg_ratio)))
    perm  = torch.randperm(len(bg_cands), device=bg_cands.device)[:n_bg]
    keep  = torch.cat([keep_fg, bg_cands[perm]]).sort().values
    return ctx_flat[keep], labels_flat[keep]


def predict_tabpfn(
    tgt_feat:  torch.Tensor,   # (C, H', W')
    ctx_feats: torch.Tensor,   # (K, C, H', W')
    ctx_masks: torch.Tensor,   # (K, H', W') float
    clf,
    balance_ratio: float | None = None,
) -> torch.Tensor:
    """
    Fit TabPFN on context (feature, label) patches; predict soft mask for target.

    Returns (H', W') float tensor in [0, 1].
    Falls back to context positive rate when all context labels are the same class.
    """
    C, H, W = tgt_feat.shape
    K = ctx_feats.shape[0]
    N = H * W

    ctx_flat   = ctx_feats.reshape(K, C, N).permute(0, 2, 1).reshape(K * N, C)
    ctx_labels = (ctx_masks.reshape(K * N) > 0).float()

    if balance_ratio is not None:
        ctx_flat, ctx_labels = balance_context(ctx_flat, ctx_labels, balance_ratio)

    X_ctx = ctx_flat.cpu().numpy()
    y_ctx = ctx_labels.cpu().numpy().astype(int)
    X_tgt = tgt_feat.reshape(C, N).T.cpu().numpy()

    if y_ctx.sum() == 0 or y_ctx.sum() == len(y_ctx):
        fill = float(y_ctx.mean())
        return torch.full((H, W), fill, dtype=torch.float32)

    # Per-feature z-score (matches TabPFN pretraining)
    mu  = X_ctx.mean(axis=0, keepdims=True)
    sig = X_ctx.std( axis=0, keepdims=True) + 1e-8
    X_ctx = (X_ctx - mu) / sig
    X_tgt = (X_tgt - mu) / sig

    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(X_tgt)    # (N, 2)
    return torch.from_numpy(proba[:, 1]).float().reshape(H, W)


# ── Metrics ───────────────────────────────────────────────────────────────────

def hard_dice(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    p = (pred >= threshold).float()
    g = (gt    >  0       ).float()
    num = 2 * (p * g).sum()
    den = p.sum() + g.sum()
    return float(num / den) if den > 1e-6 else float("nan")


def dice_at_native(
    pred_ds: torch.Tensor, gt_native: torch.Tensor, native_size: int
) -> float:
    """Upsample patch-level prediction to native resolution and compute Dice."""
    pred_up = F.interpolate(
        pred_ds.unsqueeze(0).unsqueeze(0).float(),
        size=(native_size, native_size),
        mode="bilinear", align_corners=False,
    ).squeeze()
    return hard_dice(pred_up, gt_native)


# ── Visualisation ────────────────────────────────────────────────────────────

def _overlay_ax(ax, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Grayscale image with red mask overlay."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask,  cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _heatmap_ax(ax, arr: np.ndarray, title: str) -> None:
    ax.imshow(arr, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_figure(
    tgt_image:   np.ndarray,         # (H, W)  float [0,1]
    tgt_gt:      np.ndarray,         # (H, W)  binary
    tgt_gt_ds:   np.ndarray,         # (os, os)
    pred_ds:     np.ndarray,         # (os, os)  soft prediction
    ctx_images:  list[np.ndarray],   # K × (H, W)
    ctx_gts:     list[np.ndarray],   # K × (H, W)
    ctx_gts_ds:  list[np.ndarray],   # K × (os, os)
    out_path:    Path,
    title:       str = "",
) -> None:
    """
    Row 0: target+GT overlay | GT@output_size | prediction@output_size
    Row 1: ctx_k+GT overlay  | ctx GT@output_size   (repeated for each k)
    """
    K     = len(ctx_images)
    ncols = max(3, 2 * K)
    span  = ncols // 3

    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.05)

    # Row 0
    _overlay_ax(axes[0, 0],          tgt_image, tgt_gt,   "Target + GT")
    _heatmap_ax(axes[0, span],       tgt_gt_ds,            f"GT ↓{tgt_gt_ds.shape[0]}")
    _heatmap_ax(axes[0, 2 * span],   pred_ds,              "Prediction")
    for col in range(1, span):
        axes[0, col].axis("off")
    for col in range(span + 1, 2 * span):
        axes[0, col].axis("off")
    for col in range(2 * span + 1, ncols):
        axes[0, col].axis("off")

    # Row 1: contexts
    for k in range(K):
        _overlay_ax(axes[1, 2 * k],     ctx_images[k], ctx_gts[k], f"Ctx {k} + GT")
        _heatmap_ax(axes[1, 2 * k + 1], ctx_gts_ds[k],             f"Ctx {k} GT ↓")
    for col in range(2 * K, ncols):
        axes[1, col].axis("off")

    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Collate ───────────────────────────────────────────────────────────────────

def collate(batch):
    batch = [b for b in batch if b["context_in"].shape[0] > 0]
    if not batch:
        return None
    return {
        "image":       torch.stack([b["image"]       for b in batch]),
        "label":       torch.stack([b["label"]       for b in batch]),
        "context_in":  torch.stack([b["context_in"]  for b in batch]),
        "context_out": torch.stack([b["context_out"] for b in batch]),
        "dataset":     [b["dataset"]     for b in batch],
        "sample_idx":  [b["sample_idx"]  for b in batch],
        "label_value": [b["label_value"] for b in batch],
    }


class TaggedDataset(torch.utils.data.Dataset):
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        ds_name, sample_idx, label_value = self.inner.samples[idx]
        item["dataset"]     = ds_name
        item["sample_idx"]  = sample_idx
        item["label_value"] = label_value
        return item


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs/experiment/2d", config_name="feature_sim", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ── dataset ───────────────────────────────────────────────────────────────
    datasets = [cfg.data.dataset] if cfg.data.dataset else None
    ds = MedSegBenchDataset(
        split=cfg.data.split,
        context_size=cfg.data.context_size,
        image_size=cfg.data.image_size,
        datasets=datasets,
    )
    loader = DataLoader(
        TaggedDataset(ds),
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.workers,
        collate_fn=collate,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=cfg.eval.workers > 0,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    from src.models.universeg_baseline import UniverSegBaseline
    print("Loading UniverSeg encoder...")
    wrapper = UniverSegBaseline(pretrained=True, input_size=cfg.data.image_size)
    useg    = wrapper.model.to(DEVICE).eval()  # encoder only; decoder stays off GPU
    del wrapper
    torch.cuda.empty_cache()

    # ── flops (encoder only, 1 target + K context images) ─────────────────────
    from torch.utils.flop_counter import FlopCounterMode
    _n = 1 + cfg.data.context_size
    _dummy = torch.zeros(_n, 1, cfg.data.image_size, cfg.data.image_size, device=DEVICE)
    with FlopCounterMode(display=False) as _fc:
        with torch.no_grad():
            encode_images(useg, _dummy)
    flops = _fc.get_total_flops()
    print(f"Encoder FLOPs (1+{cfg.data.context_size} images, {cfg.data.image_size}²): {flops/1e9:.2f} GFLOPs")
    del _dummy

    # ── TabPFN + its flops ────────────────────────────────────────────────────
    from tabpfn import TabPFNClassifier
    _mem_mode: bool | str = {"auto": "auto", "true": True, "false": False}[
        str(cfg.tabpfn.memory_saving).lower()
    ]
    clf = TabPFNClassifier(
        n_estimators=cfg.tabpfn.n_estimators,
        device=str(DEVICE),
        ignore_pretraining_limits=True,
        memory_saving_mode=_mem_mode,
    )
    print(f"TabPFN ready  n_estimators={cfg.tabpfn.n_estimators}")

    # Measure TabPFN FLOPs on a dummy call with correct dimensions.
    # FlopCounterMode captures PyTorch ops dispatched inside fit+predict_proba.
    # TabPFN converts numpy→tensor internally (leaf tensors, no grad_fn), so
    # FlopCounterMode's autograd hooks fail. We skip and rely on inference_time_mean_s.
    tabpfn_flops = None
    flops_total  = flops
    print("TabPFN FLOPs: not measurable via FlopCounterMode (numpy→tensor interface)")

    # ── wandb ─────────────────────────────────────────────────────────────────
    level_tag = str(cfg.feature.level).replace("-", "m")
    run_name  = cfg.wandb.name or (
        f"{cfg.model}_lvl{level_tag}_os{cfg.feature.output_size}"
        f"_s{cfg.data.image_size}_k{cfg.data.context_size}"
    )
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config={
            "model":          cfg.model,
            "image_size":     cfg.data.image_size,
            "context_size":   cfg.data.context_size,
            "split":          cfg.data.split,
            "feature_level":  str(cfg.feature.level),
            "output_size":    cfg.feature.output_size,
            "mask_pool":      cfg.feature.mask_pool,
            "n_estimators":    cfg.tabpfn.n_estimators,
            "balance_ratio":   cfg.tabpfn.balance_ratio,
            "memory_saving":   cfg.tabpfn.memory_saving,
            "flops_encoder": flops,
        },
    )
    wandb.log({"flops_giga": flops / 1e9})
    sample_table = wandb.Table(
        columns=["dataset", "sample_idx", "label", "dice_ds", "dice_native"]
    )

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── eval loop ─────────────────────────────────────────────────────────────
    per_ds:    dict[str, list[float]] = defaultdict(list)
    per_label: dict[str, list[float]] = defaultdict(list)
    inference_times: list[float] = []
    saved_figures: set[tuple[str, int]] = set()   # (dataset, label_value) already saved

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            if batch is None:
                continue

            B = len(batch["dataset"])
            images       = batch["image"].to(DEVICE, non_blocking=True)       # (B, 1, H, W)
            labels       = batch["label"]                                      # (B, 1, H, W) on CPU
            context_ins  = batch["context_in"].to(DEVICE, non_blocking=True)  # (B, K, 1, H, W)
            context_outs = batch["context_out"]                                # (B, K, 1, H, W) on CPU
            K = context_ins.shape[1]
            H, W = images.shape[-2], images.shape[-1]

            ctx_imgs_flat = context_ins.reshape(B * K, 1, H, W)  # (B*K, 1, H, W)

            t0 = time.perf_counter()

            # ── Single GPU encoding pass (fp16 autocast) ─────────────────────
            # When context_mask=False both targets and contexts use zero dummy
            # support → combine into one forward call (mirrors the 3D script).
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                if not cfg.feature.context_mask:
                    all_feats = encode_images(useg, torch.cat([images, ctx_imgs_flat], dim=0))
                    tgt_feats_b = [f[:B]  for f in all_feats]   # list[(B,  C, H', W')]
                    ctx_feats_b = [f[B:]  for f in all_feats]   # list[(B*K,C, H', W')]
                else:
                    tgt_feats_b = encode_images(useg, images)
                    ctx_masks_enc = context_outs.reshape(B * K, 1, H, W).to(DEVICE, non_blocking=True)
                    ctx_feats_b = encode_images(useg, ctx_imgs_flat, masks=ctx_masks_enc)

            batch_encode_time = time.perf_counter() - t0

            # ── Per-sample TabPFN ─────────────────────────────────────────────
            for b in range(B):
                tgt_feat = extract_features(
                    [f[b : b + 1] for f in tgt_feats_b],
                    cfg.feature.level, cfg.feature.output_size,
                )  # (C, os, os) — on GPU

                ctx_feats = torch.stack([
                    extract_features(
                        [f[b * K + k : b * K + k + 1] for f in ctx_feats_b],
                        cfg.feature.level, cfg.feature.output_size,
                    )
                    for k in range(K)
                ])  # (K, C, os, os) — on GPU

                ctx_masks = torch.stack([
                    downsample_mask(
                        context_outs[b, k, 0], cfg.feature.output_size, cfg.feature.mask_pool
                    )
                    for k in range(K)
                ])  # (K, os, os) — on CPU

                t1 = time.perf_counter()
                pred_ds = predict_tabpfn(
                    tgt_feat, ctx_feats, ctx_masks, clf,
                    balance_ratio=cfg.tabpfn.balance_ratio,
                )  # (os, os)
                inference_times.append(batch_encode_time / B + (time.perf_counter() - t1))

                label       = labels[b, 0]
                ds_name     = batch["dataset"][b]
                sample_idx  = int(batch["sample_idx"][b])
                label_value = int(batch["label_value"][b])

                d_ds     = hard_dice(pred_ds, downsample_mask(label, cfg.feature.output_size))
                d_native = dice_at_native(pred_ds, label, cfg.data.image_size)

                per_ds[ds_name].append(d_native)
                per_label[f"{ds_name}/label_{label_value}"].append(d_native)
                sample_table.add_data(ds_name, sample_idx, label_value, d_ds, d_native)

                # ── figure (one per dataset/label) ───────────────────────────
                fig_key = (ds_name, label_value)
                if fig_key not in saved_figures:
                    saved_figures.add(fig_key)
                    gt_ds_np = downsample_mask(label, cfg.feature.output_size,
                                               cfg.feature.mask_pool).cpu().numpy()
                    ctx_gts_ds_np = [
                        downsample_mask(context_outs[b, k, 0], cfg.feature.output_size,
                                        cfg.feature.mask_pool).cpu().numpy()
                        for k in range(K)
                    ]
                    fig_path = out_dir / f"{ds_name}_l{label_value}.png"
                    save_figure(
                        tgt_image=images[b, 0].cpu().numpy(),
                        tgt_gt=label.cpu().numpy(),
                        tgt_gt_ds=gt_ds_np,
                        pred_ds=pred_ds.cpu().numpy(),
                        ctx_images=[context_ins[b, k, 0].cpu().numpy() for k in range(K)],
                        ctx_gts=  [context_outs[b, k, 0].cpu().numpy() for k in range(K)],
                        ctx_gts_ds=ctx_gts_ds_np,
                        out_path=fig_path,
                        title=f"{ds_name}  label={label_value}  sample={sample_idx}"
                              f"  dice_native={d_native:.3f}",
                    )
                    wandb.log({
                        f"figures/{ds_name}/label_{label_value}": wandb.Image(str(fig_path)),
                    })

    # ── aggregate & log ───────────────────────────────────────────────────────
    summary = {}

    print(f"\n{'Dataset':>25}  {'N':>5}  {'Dice (native)':>14}")
    print("-" * 50)
    all_scores = []
    for name in sorted(per_ds):
        scores = [s for s in per_ds[name] if not np.isnan(s)]
        mean   = float(np.mean(scores)) if scores else float("nan")
        all_scores.extend(scores)
        summary[f"dice/dataset/{name}"] = mean
        print(f"{name:>25}  {len(per_ds[name]):>5}  {mean:>14.4f}")
    print("-" * 50)
    valid = [s for s in all_scores if not np.isnan(s)]
    overall = float(np.mean(valid)) if valid else float("nan")
    summary["dice/mean"] = overall
    print(f"{'MEAN':>25}  {len(all_scores):>5}  {overall:>14.4f}")

    for key, scores in per_label.items():
        valid_cls = [s for s in scores if not np.isnan(s)]
        if valid_cls:
            summary[f"dice/class/{key}"] = float(np.mean(valid_cls))

    mean_t = float(np.mean(inference_times)) if inference_times else float("nan")
    summary["inference_time_mean_s"] = mean_t
    print(f"\n  avg inference time: {mean_t * 1000:.1f} ms/item")

    summary["samples"] = sample_table
    wandb.log(summary)
    run.finish()


if __name__ == "__main__":
    main()

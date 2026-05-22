"""
Train PatchICLAttention on TotalSegmentator train split.

The STU-Net encoder is frozen; only the attention module is trained.
Features are extracted on-the-fly for each batch via a DataLoader with
multiple workers, enabling true batching through both the encoder and the
attention model.

Config is managed by OmegaConf: base config + cluster override +
configs/experiment/feature_attention.yaml.  Any value can be overridden
from the command line using dot-notation:

Usage
-----
    python experiments/feature_attention/train.py
    python experiments/feature_attention/train.py model.num_layers=4 model.label_injection=additive
    python experiments/feature_attention/train.py model.output_head=retrieval model.pos_encoding=sinusoidal
    python experiments/feature_attention/train.py cluster=meta
    python experiments/feature_attention/train.py train.run_name=debug data.max_ds_len_train=100
"""

import random
import sys
import time
from datetime import datetime
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.models.encoders.stunet import STUNetEncoder
from experiments.feature_attention.model import PatchICLAttention


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> OmegaConf:
    """Load base + cluster + feature_attention configs, then apply CLI overrides.

    CLI overrides use dot-notation (key=value).  Select cluster with cluster=meta
    (default: nfs).
    """
    cli_overrides = [a for a in sys.argv[1:] if "=" in a]
    cli = OmegaConf.from_dotlist(cli_overrides)

    cluster = OmegaConf.select(cli, "cluster") or "nfs"

    base        = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cluster_cfg = OmegaConf.load(ROOT / "configs" / "cluster" / f"{cluster}.yaml")
    fa_cfg      = OmegaConf.load(ROOT / "configs" / "experiment" / "feature_attention.yaml")

    return OmegaConf.merge(base, cluster_cfg, fa_cfg, cli)


# ---------------------------------------------------------------------------
# Feature extraction (batched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_only(encoder: STUNetEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
    """imgs: (B, 1, D, H, W) → list of (B, C_i, d_i, h_i, w_i)."""
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


def downsample_feat(feat: torch.Tensor, size: tuple) -> torch.Tensor:
    """(B, C, d, h, w) → (B, C, D', H', W')."""
    return F.interpolate(feat, size=size, mode="trilinear", align_corners=False)


def extract_features(feats: list[torch.Tensor], level: str, out_size: tuple, num_levels: int) -> torch.Tensor:
    """Returns (B, C, D', H', W')."""
    if level == "all":
        return torch.cat([downsample_feat(f, out_size) for f in feats], dim=1)
    return downsample_feat(feats[int(level) % num_levels], out_size)


def downsample_multiclass(labels: torch.Tensor, size: tuple) -> torch.Tensor:
    """(B, D, H, W) int64 → (B, D', H', W') int64 via per-label binary max pool.

    Each output patch gets the label of whichever class had any voxel in it.
    When multiple classes share a patch, priority is randomised to avoid bias.
    """
    unique_ids = labels.unique()
    unique_ids = unique_ids[unique_ids > 0].tolist()
    random.shuffle(unique_ids)
    result = torch.zeros(labels.shape[0], *size, dtype=torch.long, device=labels.device)
    for lid in unique_ids:
        binary = (labels == lid).float().unsqueeze(1)          # (B, 1, D, H, W)
        pooled = F.adaptive_max_pool3d(binary, size).squeeze(1) > 0
        result[pooled] = int(lid)
    return result


def downsample_mask(mask: torch.Tensor, size: tuple, mode: str = "max") -> torch.Tensor:
    """mask: (B, D, H, W) → (B, D', H', W')."""
    x = mask.float().unsqueeze(1)   # (B, 1, D, H, W)
    if mode == "max":
        return F.adaptive_max_pool3d(x, output_size=size).squeeze(1)
    return F.adaptive_avg_pool3d(x, output_size=size).squeeze(1)


def norm_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Min-max normalise pred then compute soft dice. pred, gt: arbitrary shape."""
    p = pred - pred.min()
    pmax = p.max()
    if pmax < 1e-8:
        return float("nan")
    p = p / pmax
    return (2 * (p * gt).sum() / (p.sum() + gt.sum() + 1e-6)).item()


def dice_score(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.5) -> float:
    """Hard Dice at fixed threshold. pred in [0,1], gt binary. Returns NaN when gt is empty."""
    gt_bin = (gt > 0).float()
    if gt_bin.sum() < 1:
        return float("nan")
    p = (pred >= thresh).float()
    return (2 * (p * gt_bin).sum() / (p.sum() + gt_bin.sum() + 1e-6)).item()


def soft_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Soft Dice on raw predictions (no threshold, no normalization). Returns NaN when gt is empty."""
    gt_bin = (gt > 0).float()
    if gt_bin.sum() < 1:
        return float("nan")
    p = pred.float().clamp(0, 1)
    return (2 * (p * gt_bin).sum() / (p.sum() + gt_bin.sum() + 1e-6)).item()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    encoder:       STUNetEncoder,
    model:         PatchICLAttention,
    batch:         dict,
    level:         str,
    out_size:      tuple,
    num_levels:    int,
    mask_pool:     str,
    device:        torch.device,
    amp:           bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (pred, gt_loss, gt_bin).

    Binary mode  (label_dim=1): pred/gt_loss/gt_bin all (B, N).
    RGB mode     (label_dim=3): pred/gt_loss (B, N, 3); gt_bin (B, N) — L2 norm > 0.
    """
    images  = batch["image"].to(device, non_blocking=True)        # (B, 1, D, H, W)
    labels  = batch["label"].to(device, non_blocking=True)        # (B, D, H, W)
    ctx_in  = batch["context_in"].to(device, non_blocking=True)   # (B, K, 1, D, H, W)
    ctx_out = batch["context_out"].to(device, non_blocking=True)  # (B, K, D, H, W)
    B, K = ctx_in.shape[:2]

    # Encoder is frozen — use inference_mode + autocast for maximum throughput
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        tgt_feats = encode_image_only(encoder, images)
        ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
        ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

    # (B, C, D', H', W')
    tgt_feat_ds      = extract_features(tgt_feats,      level, out_size, num_levels)
    ctx_feat_ds_flat = extract_features(ctx_feats_flat, level, out_size, num_levels)
    C = ctx_feat_ds_flat.shape[1]
    D_, H_, W_ = out_size
    N = D_ * H_ * W_
    ctx_feat_ds = ctx_feat_ds_flat.reshape(B, K, C, D_, H_, W_)

    tgt_feat = tgt_feat_ds.float().reshape(B, C, N).permute(0, 2, 1)              # (B, N, C)
    ctx_feat = ctx_feat_ds.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K*N, C)  # (B, K*N, C)

    palette  = batch.get("label_palette")   # (B, L+1, 3) or None
    is_rgb   = palette is not None
    if is_rgb:
        palette = palette.to(device)  # (B, L+1, 3)
        # Downsample integer labels correctly: per-label binary max pool so each
        # patch gets exactly one label's colour — no phantom colours from blending.
        tgt_ds = downsample_multiclass(labels, out_size)                          # (B, D',H',W')
        ctx_ds = downsample_multiclass(
            ctx_out.reshape(B * K, *ctx_out.shape[2:]), out_size
        ).reshape(B, K, D_, H_, W_)

        # Colorise by palette lookup: palette[b, label_id] → RGB
        tgt_idx  = tgt_ds.reshape(B, N)           # (B, N)
        ctx_idx  = ctx_ds.reshape(B, K * N)       # (B, K*N)
        gt_loss  = palette.gather(1, tgt_idx.unsqueeze(-1).expand(-1, -1, 3))    # (B, N, 3)
        ctx_lbls = palette.gather(1, ctx_idx.unsqueeze(-1).expand(-1, -1, 3))    # (B, K*N, 3)
        gt_bin   = (gt_loss.norm(dim=-1) > 0).float()
    else:
        tgt_mask_ds = downsample_mask(labels, out_size, mask_pool)
        ctx_mask_ds = downsample_mask(
            ctx_out.reshape(B * K, *ctx_out.shape[2:]), out_size, mask_pool
        ).reshape(B, K, D_, H_, W_)
        ctx_raw  = ctx_mask_ds.reshape(B, K * N)
        ctx_lbls = ctx_raw if model.soft_labels else (ctx_raw > 0).float()
        gt_loss  = tgt_mask_ds.reshape(B, N)       # soft coverage fraction in [0,1]
        gt_bin   = (gt_loss > 0).float()

    with torch.autocast(device_type=device.type, enabled=amp):
        pred = model(tgt_feat, ctx_feat, ctx_lbls)
    return pred.float(), gt_loss, gt_bin


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _best_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def _overlay(ax, image: np.ndarray, mask: np.ndarray, idx: int, title: str) -> None:
    sl = image[idx]
    sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
    ax.imshow(sl_norm, cmap="gray")
    ax.imshow(mask[idx], cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _heatmap(ax, vol: np.ndarray, idx: int, title: str) -> None:
    ax.imshow(vol[idx], cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# Fixed colors for label IDs 0..4 (0=background black, 1=red, 2=blue, 3=green, 4=yellow).
_VIZ_COLORS = np.array([
    [0.00, 0.00, 0.00],
    [0.90, 0.20, 0.20],
    [0.20, 0.55, 0.90],
    [0.20, 0.80, 0.30],
    [0.95, 0.75, 0.10],
], dtype=np.float32)


def _viz_palette(n: int) -> np.ndarray:
    """Fixed (n+1, 3) palette for label IDs 0..n, deterministic across samples."""
    if n + 1 <= len(_VIZ_COLORS):
        return _VIZ_COLORS[:n + 1]
    rng = np.random.default_rng(42)
    extra = rng.random((n + 1 - len(_VIZ_COLORS), 3)).astype(np.float32)
    return np.concatenate([_VIZ_COLORS, extra], axis=0)


def _overlay_multiclass(ax, image: np.ndarray, label_vol: np.ndarray, idx: int, title: str) -> None:
    """Overlay each integer label in a distinct fixed color on a grayscale image."""
    sl_norm = (image[idx] - image[idx].min()) / (image[idx].max() - image[idx].min() + 1e-6)
    ax.imshow(sl_norm, cmap="gray")
    n_labels = int(label_vol.max())
    if n_labels > 0:
        pal  = _viz_palette(n_labels)
        rgba = np.zeros((*label_vol[idx].shape, 4), dtype=np.float32)
        for lid in range(1, n_labels + 1):
            m = label_vol[idx] == lid
            rgba[m, :3] = pal[lid]
            rgba[m,  3] = 0.45
        ax.imshow(rgba)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _colored_vol(ax, label_vol: np.ndarray, palette: np.ndarray, idx: int, title: str) -> None:
    """label_vol: (D,H,W) int; palette: (L+1,3) float in [0,1]."""
    rgb = palette[label_vol[idx].clip(0, len(palette) - 1)]
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _pred_colored(ax, pred_rgb: np.ndarray, palette: np.ndarray, idx: int, title: str) -> None:
    """Map each pred patch to the closest training-palette entry, render in viz colors."""
    flat      = pred_rgb.reshape(-1, 3)
    dists     = np.linalg.norm(flat[:, None] - palette[None], axis=-1)  # (N, L+1)
    class_ids = dists.argmin(axis=-1)                                    # (N,)
    viz_pal   = _viz_palette(int(class_ids.max()))
    colored   = viz_pal[class_ids].reshape(*pred_rgb.shape[:3], 3)
    ax.imshow(colored[idx])
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_val_figure(
    tgt_image:   np.ndarray,          # (D, H, W)
    tgt_gt:      np.ndarray,          # (D, H, W)  full-res binary
    tgt_gt_ds:   np.ndarray,          # (D', H', W') binary float | int labels
    pred:        np.ndarray,          # (D', H', W') scalar | (D', H', W', 3) RGB
    ctx_images:  list[np.ndarray],    # K × (D, H, W)
    ctx_gts:     list[np.ndarray],    # K × (D, H, W)
    ctx_gts_ds:  list[np.ndarray],    # K × (D', H', W') binary float | int labels
    out_path:    Path,
    title:       str = "",
    palette:     np.ndarray | None = None,  # (L+1, 3) float; enables coloured rendering
) -> None:
    K = len(ctx_images)
    ncols = max(3, 2 + K)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5))

    tgt_z    = _best_slice(tgt_gt)
    tgt_z_ds = tgt_z * tgt_gt_ds.shape[0] // tgt_gt.shape[0]

    # Full-res overlay: multiclass when label IDs > 1 are present
    if np.asarray(tgt_gt).max() > 1:
        _overlay_multiclass(axes[0, 0], tgt_image, tgt_gt.astype(int), tgt_z, "Target + GT")
    else:
        _overlay(axes[0, 0], tgt_image, tgt_gt, tgt_z, "Target + GT")

    # Downsampled GT always uses fixed viz palette so all labels are visible.
    # Pred: random_coloring mode maps RGB → class ID via training palette → viz color.
    max_lbl = int(np.asarray(tgt_gt_ds).max())
    if palette is not None:
        viz_pal = _viz_palette(max_lbl)
        _colored_vol (axes[0, 1], tgt_gt_ds.astype(int), viz_pal, tgt_z_ds, "GT ↓")
        _pred_colored(axes[0, 2], pred,                  palette, tgt_z_ds, "Prediction")
    elif max_lbl > 1:
        viz_pal = _viz_palette(max_lbl)
        _colored_vol(axes[0, 1], tgt_gt_ds.astype(int), viz_pal, tgt_z_ds, "GT ↓")
        _heatmap    (axes[0, 2], pred,                            tgt_z_ds, "Prediction")
    else:
        _heatmap(axes[0, 1], tgt_gt_ds, tgt_z_ds, "GT ↓")
        _heatmap(axes[0, 2], pred,       tgt_z_ds, "Prediction")
    for col in range(3, ncols):
        axes[0, col].set_visible(False)

    for k in range(K):
        ctx_z    = _best_slice(ctx_gts[k])
        ctx_z_ds = ctx_z * ctx_gts_ds[k].shape[0] // ctx_gts[k].shape[0]
        if 2 * k < ncols:
            if np.asarray(ctx_gts[k]).max() > 1:
                _overlay_multiclass(axes[1, 2 * k], ctx_images[k], ctx_gts[k].astype(int), ctx_z, f"Ctx {k} + GT")
            else:
                _overlay(axes[1, 2 * k], ctx_images[k], ctx_gts[k], ctx_z, f"Ctx {k} + GT")
        if 2 * k + 1 < ncols:
            ctx_max = int(np.asarray(ctx_gts_ds[k]).max())
            if ctx_max > 1:
                _colored_vol(axes[1, 2 * k + 1], ctx_gts_ds[k].astype(int),
                             _viz_palette(ctx_max), ctx_z_ds, f"Ctx {k} GT ↓")
            else:
                _heatmap(axes[1, 2 * k + 1], ctx_gts_ds[k], ctx_z_ds, f"Ctx {k} GT ↓")
    for col in range(2 * K, ncols):
        axes[1, col].set_visible(False)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Train sample figures
# ---------------------------------------------------------------------------

def save_train_figures(
    batch:     dict,
    pred:      torch.Tensor,   # (B, N) on CPU, detached
    out_size:  tuple,
    fig_dir:   Path,
    epoch:     int,
    n_samples: int = 2,
) -> dict:
    """Save figures for the first n_samples items of a training batch.

    Returns {key: Path} for W&B logging.
    """
    D_, H_, W_ = out_size
    fig_dir.mkdir(parents=True, exist_ok=True)
    wandb_images: dict = {}
    B = min(n_samples, pred.shape[0])

    for i in range(B):
        raw = batch["label_names"][i] if "label_names" in batch else f"sample{i}"
        cls = f"synth_{i}" if raw.startswith("sv_") else raw

        tgt_image = batch["image"][i].squeeze(0).cpu().numpy()   # (D, H, W)
        tgt_gt    = batch["label"][i].cpu().numpy()               # (D, H, W) int — keeps label IDs
        K = batch["context_in"].shape[1]
        ctx_images = [batch["context_in"][i, k].squeeze(0).cpu().numpy() for k in range(K)]
        ctx_gts    = [batch["context_out"][i, k].cpu().numpy() for k in range(K)]

        palette = None
        if pred.ndim == 3 and "label_palette" in batch:
            # RGB mode: integer label GT + raw RGB pred → coloured rendering
            palette    = batch["label_palette"][i].numpy()                # (L+1, 3)
            tgt_gt_ds  = downsample_multiclass(
                batch["label"][i].unsqueeze(0), out_size
            ).squeeze(0).numpy().astype(int)                              # (D', H', W') int
            pred_vol   = pred[i].numpy().reshape(D_, H_, W_, 3)          # (D', H', W', 3)
            ctx_gts_ds = [
                downsample_multiclass(
                    batch["context_out"][i, k].unsqueeze(0), out_size
                ).squeeze(0).numpy().astype(int)
                for k in range(K)
            ]
        else:
            # Keep integer label IDs so _overlay_multiclass can colour each label
            tgt_gt_ds  = downsample_multiclass(
                batch["label"][i].unsqueeze(0), out_size
            ).squeeze(0).numpy().astype(int)                              # (D', H', W') int
            pred_vol   = pred[i].numpy().reshape(D_, H_, W_)             # (D', H', W') scalar
            ctx_gts_ds = [
                (downsample_mask(batch["context_out"][i, k].unsqueeze(0), out_size, "max") > 0)
                .float().squeeze(0).numpy()
                for k in range(K)
            ]

        fig_path = fig_dir / f"epoch{epoch:03d}_train_{cls}.png"
        save_val_figure(
            tgt_image=tgt_image, tgt_gt=tgt_gt, tgt_gt_ds=tgt_gt_ds,
            pred=pred_vol,
            ctx_images=ctx_images, ctx_gts=ctx_gts, ctx_gts_ds=ctx_gts_ds,
            out_path=fig_path,
            title=f"[ep {epoch}] TRAIN  {cls}",
            palette=palette,
        )
        wandb_images[f"train/pred/{cls}"] = fig_path

    return wandb_images


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:       PatchICLAttention,
    encoder:     STUNetEncoder,
    ds_val:      TotalSegInContextDataset,
    level:       str,
    out_size:    tuple,
    num_levels:  int,
    mask_pool:   str,
    device:      torch.device,
    items_per_class: int,
    fig_dir:     Path | None = None,
    epoch:       int = 0,
) -> tuple[dict, dict]:
    """Returns (metrics, wandb_images) where wandb_images maps key → wandb.Image."""
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    model.eval()
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    cls_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (_, cls) in enumerate(ds_val.samples):
        cls_to_indices[cls].append(i)

    aurocs, norm_dices, dices, soft_dices, losses = [], [], [], [], []
    wandb_images: dict = {}

    for cls in ds_val.classes:
        collected = 0
        cls_fig_saved = False
        for idx in cls_to_indices[cls]:
            if collected >= items_per_class:
                break
            try:
                item = ds_val[idx]
            except Exception:
                continue

            image   = item["image"].unsqueeze(0).to(device)
            label   = item["label"].unsqueeze(0).to(device)
            ctx_in  = item["context_in"].unsqueeze(0).to(device)
            ctx_out = item["context_out"].unsqueeze(0).to(device)
            K = ctx_in.shape[1]

            tgt_feats      = encode_image_only(encoder, image)
            ctx_imgs_flat  = ctx_in.reshape(K, 1, *ctx_in.shape[3:])
            ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

            tgt_feat_ds      = extract_features(tgt_feats,      level, out_size, num_levels)
            ctx_feat_ds_flat = extract_features(ctx_feats_flat, level, out_size, num_levels)
            C        = ctx_feat_ds_flat.shape[1]
            D_, H_, W_ = out_size
            N        = D_ * H_ * W_
            ctx_feat_ds = ctx_feat_ds_flat.reshape(1, K, C, D_, H_, W_)

            tgt_mask_ds = downsample_mask(label, out_size, mask_pool)
            ctx_mask_ds = downsample_mask(
                ctx_out.reshape(K, *ctx_out.shape[2:]), out_size, mask_pool
            ).reshape(1, K, D_, H_, W_)

            tgt_feat = tgt_feat_ds.reshape(1, C, N).permute(0, 2, 1)
            ctx_feat = ctx_feat_ds.permute(0, 1, 3, 4, 5, 2).reshape(1, K * N, C)
            ctx_lbls = (ctx_mask_ds.reshape(1, K * N) > 0).float()
            if model.label_dim > 1:
                ctx_lbls = ctx_lbls.unsqueeze(-1).expand(-1, -1, model.label_dim)
            gt       = (tgt_mask_ds.reshape(1, N) > 0).float()

            pred = model(tgt_feat, ctx_feat, ctx_lbls).squeeze(0)   # (N,) or (N, label_dim)
            if pred.dim() == 2:
                pred = pred.norm(dim=-1) / math.sqrt(model.label_dim)  # (N,) in [0,1]
            gt   = gt.squeeze(0)                                     # (N,)

            losses.append(F.binary_cross_entropy(pred, gt).item())
            pred_np = pred.cpu().numpy()
            gt_np   = (gt.cpu().numpy() > 0).astype(int)
            if 0 < gt_np.sum() < len(gt_np):
                aurocs.append(roc_auc_score(gt_np, pred_np))

            p = pred - pred.min()
            pmax = p.max()
            nd = float("nan")
            if pmax > 1e-8:
                p = p / pmax
                nd = (2 * (p * gt).sum() / (p.sum() + gt.sum() + 1e-6)).item()
                norm_dices.append(nd)

            dc = dice_score(pred, gt)
            if dc == dc:
                dices.append(dc)
            sd = soft_dice_score(pred, gt)
            if sd == sd:
                soft_dices.append(sd)

            # Save one figure per class (first valid item)
            if not cls_fig_saved and fig_dir is not None:
                pred_vol = pred.cpu().numpy().reshape(D_, H_, W_)
                fig_path = fig_dir / f"epoch{epoch:03d}_{cls}.png"
                title = f"[ep {epoch}] {cls}  dice={dc:.3f}  norm_dice={nd:.3f}"
                save_val_figure(
                    tgt_image  = item["image"].squeeze().cpu().numpy(),
                    tgt_gt     = item["label"].cpu().numpy(),
                    tgt_gt_ds  = tgt_mask_ds.squeeze().cpu().numpy(),
                    pred       = pred_vol,
                    ctx_images = [ctx_in[0, k].squeeze(0).cpu().numpy() for k in range(K)],
                    ctx_gts    = [ctx_out[0, k].cpu().numpy() for k in range(K)],
                    ctx_gts_ds = [ctx_mask_ds[0, k].cpu().numpy() for k in range(K)],
                    out_path   = fig_path,
                    title      = title,
                )
                wandb_images[f"val/pred/{cls}"] = fig_path
                cls_fig_saved = True

            collected += 1

    model.train()
    metrics = {
        "val/loss":      float(np.mean(losses))             if losses      else float("nan"),
        "val/auroc":     float(np.nanmean(aurocs))          if aurocs      else float("nan"),
        "val/dice":      float(np.nanmean(dices))           if dices       else float("nan"),
        "val/soft_dice": float(np.nanmean(soft_dices))      if soft_dices  else float("nan"),
        "val/norm_dice": float(np.nanmean(norm_dices))      if norm_dices  else float("nan"),
    }
    return metrics, wandb_images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.train.seed)

    device_str = cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # enable TF32 on Ampere+ GPUs

    out_dir  = Path(cfg.paths.results) / "feature_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_size = tuple(cfg.data.output_size)

    # ---- Augmentation config -----------------------------------------------
    aug_cfg = None
    if cfg.train.aug:
        aug_yaml = ROOT / "configs" / "augmentations" / f"{cfg.train.aug_preset}.yaml"
        aug_cfg  = OmegaConf.load(aug_yaml).augmentations

    train_classes = list(cfg.data.train_classes)
    val_classes   = list(cfg.data.val_classes) or train_classes

    # ---- Datasets ----------------------------------------------------------
    ds_train = TotalSegInContextDataset(
        root=cfg.paths.totalseg,
        classes=train_classes,
        image_size=tuple(cfg.data.image_size),
        split="train",
        context_size=cfg.data.context_size,
        max_subjects=None,
        class_balanced=cfg.data.class_balanced,
        aug_cfg=aug_cfg,
        use_crop=cfg.data.use_crop,
        synth_method=cfg.data.synth_method or None,
        synth_unions=cfg.data.synth_unions,
        p_synth=cfg.data.p_synth,
        random_coloring=cfg.data.random_coloring,
        num_labels_per_sample=cfg.data.num_labels_per_sample,
    )
    ds_val = TotalSegInContextDataset(
        root=cfg.paths.totalseg,
        classes=val_classes,
        image_size=tuple(cfg.data.image_size),
        split="val",
        context_size=cfg.data.context_size,
        use_crop=cfg.data.use_crop,
    )

    from torch.utils.data import RandomSampler
    n_train = min(cfg.data.max_ds_len_train, len(ds_train))
    train_sampler = RandomSampler(ds_train, replacement=False, num_samples=n_train)
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        num_workers=cfg.train.workers,
        pin_memory=True,
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=2 if cfg.train.workers > 0 else None,
        collate_fn=incontext_collate_fn,
        drop_last=True,
    )
    print(f"Train: {n_train} samples  |  {len(train_loader)} batches/epoch  |  batch_size={cfg.train.batch_size}")

    # ---- Encoder (frozen) -------------------------------------------------
    encoder = STUNetEncoder(
        in_channels=1,
        variant=cfg.model.stunet_variant,
        pretrained=cfg.model.stunet_pretrained,
        freeze_encoder=True,
    ).to(device).eval()

    num_levels = len(encoder.skip_channels) + 1
    level = cfg.model.feature_level

    # Determine embed_dim from a dummy forward (before compiling)
    with torch.inference_mode():
        dummy = torch.zeros(1, 1, *cfg.data.image_size, device=device)
        dummy_feats = encode_image_only(encoder, dummy)
        dummy_feat_ds = extract_features(dummy_feats, level, out_size, num_levels)
    embed_dim = dummy_feat_ds.shape[1]
    print(f"Encoder embed_dim: {embed_dim}  |  grid: {out_size}  |  level: {level}")


    # ---- Model -------------------------------------------------------------
    label_dim = 3 if cfg.data.random_coloring else 1
    model = PatchICLAttention(
        embed_dim       = embed_dim,
        dim             = cfg.model.dim,
        num_heads       = cfg.model.num_heads,
        num_layers      = cfg.model.num_layers,
        ff_factor       = cfg.model.ff_factor,
        label_injection = cfg.model.label_injection,
        output_head     = cfg.model.output_head,
        pos_encoding    = cfg.model.pos_encoding,
        input_norm      = cfg.model.input_norm,
        grid_size       = out_size,
        dropout         = cfg.model.dropout,
        ctx_self_attn   = cfg.model.ctx_self_attn,
        log_n_scaling   = cfg.model.log_n_scaling,
        log_n_base      = cfg.model.log_n_base,
        label_dim       = label_dim,
        soft_labels     = cfg.model.soft_labels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchICLAttention  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    amp       = device.type == "cuda"

    best_auroc = -1.0
    if cfg.train.checkpoint:
        ckpt = torch.load(cfg.train.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_auroc = ckpt.get("val_auroc", -1.0)
        print(f"Loaded checkpoint: {cfg.train.checkpoint}  "
              f"(epoch {ckpt['epoch']}, val_auroc={best_auroc:.3f})")

    model_module = model

    # ---- W&B ---------------------------------------------------------------
    use_wandb = bool(cfg.train.wandb_project) and str(cfg.train.wandb_project).lower() != "null"
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.run_name or None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"embed_dim": embed_dim, "n_params": n_params, "label_dim": label_dim})

    date_str = datetime.today().strftime("%Y-%m-%d")
    run_name = (wandb.run.name if use_wandb else None) or cfg.train.run_name or "run"
    run_dir = out_dir / f"{date_str}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = run_dir / "figures"

    # ---- Training ----------------------------------------------------------
    best_auroc = -1.0
    nd_interval = cfg.train.nd_interval
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss, epoch_nd, epoch_dice, epoch_sdice = 0.0, 0.0, 0.0, 0.0
        n_batches, n_nd, n_dice, n_sdice = 0, 0, 0, 0
        last_nd, last_dice, last_sdice = float("nan"), float("nan"), float("nan")
        t0 = time.perf_counter()

        train_vis: tuple | None = None   # (batch_cpu, pred_cpu) from first batch
        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{cfg.train.epochs}", unit="batch", leave=False)
        for batch in bar:
            pred, gt_loss, gt_bin = process_batch(
                encoder, model, batch, level, out_size, num_levels,
                cfg.data.mask_pool, device, amp=amp,
            )
            if cfg.data.random_coloring:
                loss = F.smooth_l1_loss(pred, gt_loss)
            else:
                loss = F.binary_cross_entropy(pred, gt_loss)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if train_vis is None:
                train_vis = (batch, pred.detach().cpu())

            n_batches  += 1
            epoch_loss += loss.item()   # one sync per batch (unavoidable for loss display)

            # dice metrics sync the GPU — only do it every nd_interval batches
            if n_batches % nd_interval == 0:
                with torch.no_grad():
                    pred_scalar = pred.detach().norm(dim=-1) if pred.ndim == 3 else pred.detach()
                    nd = norm_dice_score(pred_scalar, gt_bin)
                    dc = dice_score(pred_scalar, gt_bin)
                    sd = soft_dice_score(pred_scalar, gt_bin)
                if nd == nd:
                    epoch_nd    += nd;  n_nd    += 1;  last_nd    = nd
                if dc == dc:
                    epoch_dice  += dc;  n_dice  += 1;  last_dice  = dc
                if sd == sd:
                    epoch_sdice += sd;  n_sdice += 1;  last_sdice = sd

            bar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}",
                            dice=f"{last_dice:.3f}", sd=f"{last_sdice:.3f}", nd=f"{last_nd:.3f}")

        bar.close()
        avg_loss  = epoch_loss  / max(n_batches, 1)
        avg_nd    = epoch_nd    / max(n_nd, 1)
        avg_dice  = epoch_dice  / max(n_dice, 1)
        avg_sdice = epoch_sdice / max(n_sdice, 1)
        elapsed   = time.perf_counter() - t0
        print(f"Epoch {epoch:3d}/{cfg.train.epochs}  loss={avg_loss:.4f}  "
              f"dice={avg_dice:.3f}  soft_dice={avg_sdice:.3f}  norm_dice={avg_nd:.3f}  "
              f"batches={n_batches}  {elapsed:.0f}s")

        # Validation
        val_metrics, val_figs = validate(
            model, encoder, ds_val, level, out_size, num_levels,
            cfg.data.mask_pool, device, cfg.train.val_items_per_class,
            fig_dir=fig_dir, epoch=epoch,
        )
        print(f"  val auroc={val_metrics['val/auroc']:.3f}  "
              f"dice={val_metrics['val/dice']:.3f}  "
              f"soft_dice={val_metrics['val/soft_dice']:.3f}  "
              f"norm_dice={val_metrics['val/norm_dice']:.3f}")

        # Save best checkpoint
        if val_metrics["val/auroc"] > best_auroc:
            best_auroc = val_metrics["val/auroc"]
            ckpt = {
                "epoch":   epoch,
                "model":   model_module.state_dict(),
                "config": {
                    "embed_dim":       embed_dim,
                    "dim":             cfg.model.dim,
                    "num_heads":       cfg.model.num_heads,
                    "num_layers":      cfg.model.num_layers,
                    "ff_factor":       cfg.model.ff_factor,
                    "label_injection": cfg.model.label_injection,
                    "output_head":     cfg.model.output_head,
                    "pos_encoding":    cfg.model.pos_encoding,
                    "input_norm":      cfg.model.input_norm,
                    "grid_size":       list(out_size),
                    "dropout":         cfg.model.dropout,
                    "label_dim":       label_dim,
                    "soft_labels":     cfg.model.soft_labels,
                },
                "feature_level": level,
                "val_auroc":     best_auroc,
            }
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  saved best checkpoint  auroc={best_auroc:.3f}")

        if use_wandb:
            import wandb
            train_figs = {}
            if train_vis is not None:
                train_figs = save_train_figures(
                    train_vis[0], train_vis[1], out_size, fig_dir, epoch,
                )
            all_figs = {k: wandb.Image(str(v)) for k, v in {**val_figs, **train_figs}.items()}
            wandb.log({"train/loss": avg_loss, "train/dice": avg_dice,
                       "train/soft_dice": avg_sdice, "train/norm_dice": avg_nd,
                       "epoch": epoch, **val_metrics, **all_figs})

    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nBest val AUROC: {best_auroc:.3f}  |  checkpoint: {run_dir}/best.pt")


if __name__ == "__main__":
    main()
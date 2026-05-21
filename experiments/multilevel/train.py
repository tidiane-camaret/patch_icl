"""
Multilevel coarse-to-fine PatchICLAttention on TotalSegmentator.

Two levels trained jointly with detached gradients between them:
  L0: dense 8³ grid  — all 512 patches, standard ICL forward.
  L1: sparse 16³ grid — NP₁ patches, sampled from:
        target  : half highest-entropy patches from upsampled pred_0,
                  half highest-certainty foreground patches.
        context : half foreground patches (GT mask > fg_thresh),
                  half border patches (dilation − erosion of GT mask).

Usage
-----
    python experiments/multilevel/train.py
    python experiments/multilevel/train.py model.num_layers=4
    python experiments/multilevel/train.py cluster=meta
    python experiments/multilevel/train.py train.run_name=debug data.max_ds_len_train=100
"""

import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.models.encoders.stunet import STUNetEncoder
from experiments.multilevel.model import MultilevelICL
from data.totalseg_classes import resolve_classes


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> OmegaConf:
    cli_overrides = [a for a in sys.argv[1:] if "=" in a]
    cli = OmegaConf.from_dotlist(cli_overrides)
    cluster = OmegaConf.select(cli, "cluster") or "nfs"
    base    = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cl_cfg  = OmegaConf.load(ROOT / "configs" / "cluster" / f"{cluster}.yaml")
    ex_cfg  = OmegaConf.load(ROOT / "configs" / "experiment" / "multilevel.yaml")
    return OmegaConf.merge(base, cl_cfg, ex_cfg, cli)


# ---------------------------------------------------------------------------
# Feature / mask helpers  (self-contained, no import from feature_attention)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_only(encoder: STUNetEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
    """imgs: (B, 1, D, H, W) → list of feature tensors, low-res to high-res."""
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


def downsample_feat(feat: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(feat, size=size, mode="trilinear", align_corners=False)


def extract_features(feats: list[torch.Tensor], level: str, out_size: tuple, num_levels: int) -> torch.Tensor:
    """Returns (B, C, D', H', W')."""
    if level == "all":
        return torch.cat([downsample_feat(f, out_size) for f in feats], dim=1)
    return downsample_feat(feats[int(level) % num_levels], out_size)


def downsample_mask(mask: torch.Tensor, size: tuple, mode: str = "avg") -> torch.Tensor:
    """(B, D, H, W) → (B, D', H', W') float."""
    x = mask.float().unsqueeze(1)
    if mode == "max":
        return F.adaptive_max_pool3d(x, output_size=size).squeeze(1)
    return F.adaptive_avg_pool3d(x, output_size=size).squeeze(1)


def border_mask_3d(fg: torch.Tensor) -> torch.Tensor:
    """Binary mask of border voxels (dilation - erosion) for a (D, H, W) bool tensor.

    Returns a bool tensor of the same shape.
    """
    x = fg.float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    dilated = F.max_pool3d(x, kernel_size=3, stride=1, padding=1).squeeze() > 0.5
    eroded  = (-F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)).squeeze() > 0.5
    return dilated & ~eroded


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def sample_target_patches(
    pred_0_up: torch.Tensor,   # (B, N1) float in [0,1], N1 = 16³
    n_entropy: int,
    n_certain: int,
    certain_thresh: float,
) -> torch.Tensor:             # (B, NP₁) long
    """Sample target patch indices at L1 from upsampled L0 prediction.

    Half from highest-entropy positions (|pred - 0.5| smallest),
    half from highest-certainty foreground positions (pred > certain_thresh, argmax).
    Falls back to random if there are fewer certain-fg patches than needed.
    """
    B, N1 = pred_0_up.shape
    NP = n_entropy + n_certain
    idx_list = []
    for b in range(B):
        p = pred_0_up[b]  # (N1,)

        # Entropy half: smallest |p - 0.5|
        uncertainty = 0.5 - (p - 0.5).abs()
        entropy_idx = uncertainty.topk(n_entropy, largest=True).indices  # (n_entropy,)

        # Certain-fg half: largest p values among p > certain_thresh
        certain_mask = p > certain_thresh
        n_avail = int(certain_mask.sum())
        if n_avail >= n_certain:
            certain_idx = p[certain_mask].topk(n_certain, largest=True).indices
            # Map back to global indices
            global_fg = certain_mask.nonzero(as_tuple=False).squeeze(1)
            certain_idx = global_fg[certain_idx]
        else:
            # Not enough certain-fg: fill remainder with most certain patches overall
            certain_idx = p.topk(n_certain, largest=True).indices

        idx_list.append(torch.cat([entropy_idx, certain_idx]))  # (NP,)

    return torch.stack(idx_list)  # (B, NP)


def sample_context_patches(
    ctx_mask_16: torch.Tensor,  # (B, K, D1, H1, W1) float avg-pooled
    n_fg: int,
    n_border: int,
    fg_thresh: float,
) -> torch.Tensor:              # (B, NP₁) long  — shared across K context volumes
    """Sample context patch indices at L1 from GT context masks.

    n_fg patches from foreground (mask > fg_thresh) of the first context volume
    (masks are assumed consistent across contexts for the same class).
    n_border patches from the morphological border.  Falls back to fg or random
    if there are not enough patches in a pool.
    """
    B, K, D1, H1, W1 = ctx_mask_16.shape
    N1 = D1 * H1 * W1
    NP = n_fg + n_border
    idx_list = []

    for b in range(B):
        # Average context masks across K contexts for stable fg/border estimate
        avg_mask = ctx_mask_16[b].mean(0).reshape(N1)  # (N1,)

        fg = avg_mask > fg_thresh
        if fg.sum() == 0:
            # Degenerate: no foreground — random sampling
            idx_list.append(torch.randint(N1, (NP,), device=ctx_mask_16.device))
            continue

        # Border in 3D spatial layout
        fg_3d = fg.reshape(D1, H1, W1)
        border_3d = border_mask_3d(fg_3d)
        border = border_3d.reshape(N1)

        def _sample_from(mask: torch.Tensor, n: int) -> torch.Tensor:
            pos = mask.nonzero(as_tuple=False).squeeze(1)
            if len(pos) >= n:
                perm = torch.randperm(len(pos), device=ctx_mask_16.device)[:n]
                return pos[perm]
            # Not enough: repeat-sample
            repeats = (n + len(pos) - 1) // len(pos)
            return pos.repeat(repeats)[:n]

        fg_idx     = _sample_from(fg,     n_fg)
        border_idx = _sample_from(border if border.any() else fg, n_border)
        idx_list.append(torch.cat([fg_idx, border_idx]))  # (NP,)

    return torch.stack(idx_list)  # (B, NP)


def gather_patches(feat_flat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feat_flat: (B, N, C), idx: (B, NP) → (B, NP, C)."""
    B, NP = idx.shape
    C = feat_flat.shape[-1]
    idx_exp = idx.unsqueeze(-1).expand(B, NP, C)
    return feat_flat.gather(1, idx_exp)


def make_coords_3d(grid_size: tuple, device: torch.device) -> torch.Tensor:
    """Normalised (d, h, w) coordinates in [0,1] for each position in a D×H×W grid.

    Returns (N, 3) float32 where N = D*H*W, in the same flattening order as
    tensor.reshape(-1) (i.e. d varies slowest, w varies fastest).
    """
    D, H, W = grid_size
    d = torch.linspace(0, 1, D, device=device)
    h = torch.linspace(0, 1, H, device=device)
    w = torch.linspace(0, 1, W, device=device)
    gd, gh, gw = torch.meshgrid(d, h, w, indexing="ij")
    return torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)  # (N, 3)


# ---------------------------------------------------------------------------
# Forward pass for one batch
# ---------------------------------------------------------------------------

def process_batch(
    encoder:   STUNetEncoder,
    model:     MultilevelICL,
    batch:     dict,
    cfg:       OmegaConf,
    device:    torch.device,
    amp:       bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (pred_0_flat, pred_1, tgt_idx_1, loss, loss_per_level).

    pred_0_flat : (B, N0) sigmoid predictions at L0 (8³)
    pred_1      : (B, NP₁) sigmoid predictions at sampled L1 positions
    tgt_idx_1   : (B, NP₁) indices of sampled positions in the 16³ grid
    loss        : weighted scalar loss
    loss_per_level : (loss_0, loss_1) for logging
    """
    images  = batch["image"].to(device, non_blocking=True)        # (B, 1, D, H, W)
    labels  = batch["label"].to(device, non_blocking=True)        # (B, D, H, W)
    ctx_in  = batch["context_in"].to(device, non_blocking=True)   # (B, K, 1, D, H, W)
    ctx_out = batch["context_out"].to(device, non_blocking=True)  # (B, K, D, H, W)
    B, K = ctx_in.shape[:2]

    res_0 = tuple(cfg.data.resolutions[0])
    res_1 = tuple(cfg.data.resolutions[1])
    level      = cfg.model.feature_level
    num_levels = len(encoder.skip_channels) + 1

    # ---- Encode (frozen) --------------------------------------------------
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        tgt_feats      = encode_image_only(encoder, images)
        ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
        ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

    # ---- Level 0 (dense 8³) -----------------------------------------------
    tgt_feat_0  = extract_features(tgt_feats,      level, res_0, num_levels)   # (B, C, 8, 8, 8)
    ctx_feat_0f = extract_features(ctx_feats_flat, level, res_0, num_levels)   # (B*K, C, 8, 8, 8)
    C = tgt_feat_0.shape[1]
    N0 = res_0[0] * res_0[1] * res_0[2]

    ctx_feat_0 = ctx_feat_0f.reshape(B, K, C, *res_0)

    tgt_f0 = tgt_feat_0.float().reshape(B, C, N0).permute(0, 2, 1)              # (B, N0, C)
    ctx_f0 = ctx_feat_0.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K * N0, C)

    # Context labels at L0
    ctx_mask_0 = downsample_mask(
        ctx_out.reshape(B * K, *ctx_out.shape[2:]), res_0, cfg.data.mask_pool
    ).reshape(B, K, *res_0)
    ctx_lbl_0 = ctx_mask_0.reshape(B, K * N0)

    # GT target at L0
    gt_0 = downsample_mask(labels, res_0, cfg.data.mask_pool).reshape(B, N0)

    with torch.autocast(device_type=device.type, enabled=amp):
        pred_0 = model[0](tgt_f0, ctx_f0, ctx_lbl_0)  # (B, N0)
    pred_0 = pred_0.float()

    loss_0 = F.binary_cross_entropy(pred_0, gt_0)

    # ---- Level 1 (sparse 16³) ---------------------------------------------
    NP  = cfg.data.n_patches_l1
    n_e = int(NP * cfg.data.frac_entropy)
    n_c = NP - n_e
    n_fg     = NP // 2
    n_border = NP - n_fg
    N1 = res_1[0] * res_1[1] * res_1[2]

    # Upsample pred_0 to 16³ using detached predictions
    with torch.no_grad():
        pred_0_up = F.interpolate(
            pred_0.detach().reshape(B, 1, *res_0), size=res_1,
            mode="trilinear", align_corners=False,
        ).reshape(B, N1)  # (B, N1)

    # Sample target and context positions
    tgt_idx  = sample_target_patches(pred_0_up, n_e, n_c, cfg.data.certain_thresh)   # (B, NP)
    ctx_mask_1 = downsample_mask(
        ctx_out.reshape(B * K, *ctx_out.shape[2:]), res_1, cfg.data.mask_pool
    ).reshape(B, K, *res_1)
    ctx_idx  = sample_context_patches(ctx_mask_1, n_fg, n_border, cfg.data.fg_thresh)  # (B, NP)

    # Extract features at L1, gather only sampled positions
    tgt_feat_1  = extract_features(tgt_feats,      level, res_1, num_levels)   # (B, C, 16, 16, 16)
    ctx_feat_1f = extract_features(ctx_feats_flat, level, res_1, num_levels)   # (B*K, C, 16, 16, 16)
    ctx_feat_1  = ctx_feat_1f.reshape(B, K, C, *res_1)

    tgt_flat_1  = tgt_feat_1.float().reshape(B, C, N1).permute(0, 2, 1)             # (B, N1, C)
    ctx_flat_1  = ctx_feat_1.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K, N1, C)

    tgt_sparse  = gather_patches(tgt_flat_1, tgt_idx)                                # (B, NP, C)

    # Build context tokens: for each context image gather ctx_idx, then concat K
    ctx_pieces = []
    ctx_lbl_pieces = []
    ctx_mask_1_flat = ctx_mask_1.reshape(B, K, N1)
    for k in range(K):
        ctx_k_sparse  = gather_patches(ctx_flat_1[:, k], ctx_idx)                   # (B, NP, C)
        ctx_lbl_k = ctx_mask_1_flat[:, k].gather(1, ctx_idx)                        # (B, NP)
        ctx_pieces.append(ctx_k_sparse)
        ctx_lbl_pieces.append(ctx_lbl_k)
    ctx_sparse  = torch.cat(ctx_pieces,     dim=1)   # (B, K*NP, C)
    ctx_lbl_1   = torch.cat(ctx_lbl_pieces, dim=1)   # (B, K*NP)

    # Inject spatial position encoding for sparse level: project gathered (d,h,w)
    # coords into embed_dim and add before the model's input_proj.
    coords_16 = make_coords_3d(res_1, device)                                       # (N1, 3)
    coord_proj = model.coord_projs[0]                                               # level 1 → index 0
    tgt_coords = coords_16[tgt_idx.reshape(-1)].reshape(B, NP, 3)                  # (B, NP, 3)
    ctx_coords = coords_16[ctx_idx.reshape(-1)].reshape(B, NP, 3)                  # (B, NP, 3)
    tgt_sparse = tgt_sparse + coord_proj(tgt_coords)                                # (B, NP, C)
    ctx_coord_rep = ctx_coords.unsqueeze(1).expand(-1, K, -1, -1).reshape(B, K * NP, 3)
    ctx_sparse = ctx_sparse + coord_proj(ctx_coord_rep)                             # (B, K*NP, C)

    with torch.autocast(device_type=device.type, enabled=amp):
        pred_1 = model[1](tgt_sparse, ctx_sparse, ctx_lbl_1)   # (B, NP)
    pred_1 = pred_1.float()

    # GT for sampled target positions at 16³
    gt_1_flat  = downsample_mask(labels, res_1, cfg.data.mask_pool).reshape(B, N1)
    gt_1_sparse = gt_1_flat.gather(1, tgt_idx)  # (B, NP)

    loss_1 = F.binary_cross_entropy(pred_1, gt_1_sparse)

    # ---- Combined loss ----------------------------------------------------
    w0, w1 = cfg.train.loss_weights
    loss   = w0 * loss_0 + w1 * loss_1

    return pred_0, pred_1, tgt_idx, loss, (loss_0.item(), loss_1.item())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def norm_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = pred - pred.min()
    pmax = p.max()
    if pmax < 1e-8:
        return float("nan")
    p = p / pmax
    return (2 * (p * gt).sum() / (p.sum() + gt.sum() + 1e-6)).item()


def dice_score(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.5) -> float:
    gt_bin = (gt > 0).float()
    if gt_bin.sum() < 1:
        return float("nan")
    p = (pred >= thresh).float()
    return (2 * (p * gt_bin).sum() / (p.sum() + gt_bin.sum() + 1e-6)).item()


def soft_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    gt_bin = (gt > 0).float()
    if gt_bin.sum() < 1:
        return float("nan")
    p = pred.float().clamp(0, 1)
    return (2 * (p * gt_bin).sum() / (p.sum() + gt_bin.sum() + 1e-6)).item()


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


def save_val_figure(
    tgt_image:    np.ndarray,          # (D, H, W)
    tgt_gt:       np.ndarray,          # (D, H, W)
    pred_l0:      np.ndarray,          # (8, 8, 8)
    pred_fused:   np.ndarray,          # (16, 16, 16)
    gt_l1:        np.ndarray,          # (16, 16, 16)
    ctx_images:   list[np.ndarray],    # K × (D, H, W)
    ctx_gts:      list[np.ndarray],    # K × (D, H, W)
    out_path:     Path,
    title:        str = "",
) -> None:
    K    = len(ctx_images)
    ncols = max(4, 3 + K)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5))

    tgt_z     = _best_slice(tgt_gt)
    z_l0      = tgt_z * pred_l0.shape[0]   // tgt_gt.shape[0]
    z_l1      = tgt_z * pred_fused.shape[0] // tgt_gt.shape[0]

    _overlay (axes[0, 0], tgt_image, tgt_gt.astype(float), tgt_z, "Target + GT")
    _heatmap (axes[0, 1], pred_l0,   z_l0,  "pred L0 (8³)")
    _heatmap (axes[0, 2], gt_l1,     z_l1,  "GT ↓ (16³)")
    _heatmap (axes[0, 3], pred_fused,z_l1,  "pred fused (16³)")
    for col in range(4, ncols):
        axes[0, col].set_visible(False)

    for k in range(K):
        ctx_z = _best_slice(ctx_gts[k])
        if 2 * k < ncols:
            _overlay(axes[1, 2 * k], ctx_images[k], ctx_gts[k].astype(float), ctx_z, f"Ctx {k} + GT")
        if 2 * k + 1 < ncols:
            axes[1, 2 * k + 1].set_visible(False)
    for col in range(2 * K, ncols):
        axes[1, col].set_visible(False)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:           MultilevelICL,
    encoder:         STUNetEncoder,
    ds_val:          TotalSegInContextDataset,
    cfg:             OmegaConf,
    device:          torch.device,
    items_per_class: int,
    fig_dir:         Path | None = None,
    epoch:           int = 0,
) -> tuple[dict, dict]:
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    model.eval()
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    res_0 = tuple(cfg.data.resolutions[0])
    res_1 = tuple(cfg.data.resolutions[1])
    level      = cfg.model.feature_level
    num_levels = len(encoder.skip_channels) + 1
    NP  = cfg.data.n_patches_l1
    n_e = int(NP * cfg.data.frac_entropy)
    n_c = NP - n_e
    n_fg     = NP // 2
    n_border = NP - n_fg
    N0 = res_0[0] * res_0[1] * res_0[2]
    N1 = res_1[0] * res_1[1] * res_1[2]

    cls_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (_, cls) in enumerate(ds_val.samples):
        cls_to_indices[cls].append(i)

    # Collect per-level metrics separately + fused
    l0_dices, l1_dices, l1_soft_dices, l1_norm_dices, l1_aurocs, l1_losses = [], [], [], [], [], []
    wandb_images: dict = {}

    for cls in ds_val.classes:
        collected   = 0
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

            # L0
            tgt_feat_0  = extract_features(tgt_feats,      level, res_0, num_levels)
            ctx_feat_0f = extract_features(ctx_feats_flat, level, res_0, num_levels)
            C = tgt_feat_0.shape[1]
            ctx_feat_0  = ctx_feat_0f.reshape(1, K, C, *res_0)
            tgt_f0      = tgt_feat_0.float().reshape(1, C, N0).permute(0, 2, 1)
            ctx_f0      = ctx_feat_0.float().permute(0, 1, 3, 4, 5, 2).reshape(1, K * N0, C)
            ctx_mask_0  = downsample_mask(
                ctx_out.reshape(K, *ctx_out.shape[2:]), res_0, cfg.data.mask_pool
            ).reshape(1, K, *res_0)
            ctx_lbl_0   = ctx_mask_0.reshape(1, K * N0)
            gt_0        = downsample_mask(label, res_0, cfg.data.mask_pool).reshape(1, N0)

            pred_0 = model[0](tgt_f0, ctx_f0, ctx_lbl_0).float()  # (1, N0)

            # L1
            pred_0_up = F.interpolate(
                pred_0.reshape(1, 1, *res_0), size=res_1, mode="trilinear", align_corners=False
            ).reshape(1, N1)

            tgt_idx = sample_target_patches(pred_0_up, n_e, n_c, cfg.data.certain_thresh)
            ctx_mask_1 = downsample_mask(
                ctx_out.reshape(K, *ctx_out.shape[2:]), res_1, cfg.data.mask_pool
            ).reshape(1, K, *res_1)
            ctx_idx = sample_context_patches(ctx_mask_1, n_fg, n_border, cfg.data.fg_thresh)

            tgt_feat_1  = extract_features(tgt_feats,      level, res_1, num_levels)
            ctx_feat_1f = extract_features(ctx_feats_flat, level, res_1, num_levels)
            ctx_feat_1  = ctx_feat_1f.reshape(1, K, C, *res_1)
            tgt_flat_1  = tgt_feat_1.float().reshape(1, C, N1).permute(0, 2, 1)
            ctx_flat_1  = ctx_feat_1.float().permute(0, 1, 3, 4, 5, 2).reshape(1, K, N1, C)

            tgt_sparse = gather_patches(tgt_flat_1, tgt_idx)
            ctx_pieces, ctx_lbl_pieces = [], []
            ctx_mask_1_flat = ctx_mask_1.reshape(1, K, N1)
            for k in range(K):
                ctx_pieces.append(gather_patches(ctx_flat_1[:, k], ctx_idx))
                ctx_lbl_pieces.append(ctx_mask_1_flat[:, k].gather(1, ctx_idx))
            ctx_sparse = torch.cat(ctx_pieces,     dim=1)
            ctx_lbl_1  = torch.cat(ctx_lbl_pieces, dim=1)

            # Inject coord PE for sparse level (same as process_batch)
            coords_16   = make_coords_3d(res_1, device)
            coord_proj  = model.coord_projs[0]
            tgt_coords  = coords_16[tgt_idx.reshape(-1)].reshape(1, NP, 3)
            ctx_coords  = coords_16[ctx_idx.reshape(-1)].reshape(1, NP, 3)
            tgt_sparse  = tgt_sparse + coord_proj(tgt_coords)
            ctx_coord_rep = ctx_coords.unsqueeze(1).expand(-1, K, -1, -1).reshape(1, K * NP, 3)
            ctx_sparse  = ctx_sparse + coord_proj(ctx_coord_rep)

            pred_1 = model[1](tgt_sparse, ctx_sparse, ctx_lbl_1).float()   # (1, NP)

            # Fused prediction at 16³
            pred_fused = pred_0_up.clone()
            pred_fused[0, tgt_idx[0]] = pred_1[0]  # overwrite sampled positions

            gt_1 = downsample_mask(label, res_1, cfg.data.mask_pool).reshape(1, N1)

            # --- Metrics ---
            # L0 dice (8³)
            d0 = dice_score(pred_0.squeeze(0), gt_0.squeeze(0))
            if d0 == d0:
                l0_dices.append(d0)

            # L1 fused metrics (16³)
            pf = pred_fused.squeeze(0)
            gf = (gt_1.squeeze(0) > 0).float()
            loss_v = F.binary_cross_entropy(pf, gf).item()
            l1_losses.append(loss_v)

            d1 = dice_score(pf, gf)
            if d1 == d1:
                l1_dices.append(d1)
            sd = soft_dice_score(pf, gf)
            if sd == sd:
                l1_soft_dices.append(sd)
            nd = norm_dice_score(pf, gf)
            if nd == nd:
                l1_norm_dices.append(nd)

            pf_np = pf.cpu().numpy()
            gf_np = gf.cpu().numpy().astype(int)
            if 0 < gf_np.sum() < len(gf_np):
                try:
                    from sklearn.metrics import roc_auc_score
                    l1_aurocs.append(roc_auc_score(gf_np, pf_np))
                except Exception:
                    pass

            if not cls_fig_saved and fig_dir is not None:
                D0, H0, W0 = res_0
                D1, H1, W1 = res_1
                fig_path  = fig_dir / f"epoch{epoch:03d}_{cls}.png"
                save_val_figure(
                    tgt_image  = item["image"].squeeze().cpu().numpy(),
                    tgt_gt     = item["label"].cpu().numpy().astype(float),
                    pred_l0    = pred_0.squeeze().cpu().numpy().reshape(D0, H0, W0),
                    pred_fused = pred_fused.squeeze().cpu().numpy().reshape(D1, H1, W1),
                    gt_l1      = gt_1.squeeze().cpu().numpy().reshape(D1, H1, W1),
                    ctx_images = [ctx_in[0, k].squeeze(0).cpu().numpy() for k in range(K)],
                    ctx_gts    = [ctx_out[0, k].cpu().numpy() for k in range(K)],
                    out_path   = fig_path,
                    title      = f"[ep {epoch}] {cls}  dice_fused={d1:.3f}  dice_l0={d0:.3f}",
                )
                wandb_images[f"val/pred/{cls}"] = fig_path
                cls_fig_saved = True

            collected += 1

    model.train()
    metrics = {
        "val/dice_l0":      float(np.nanmean(l0_dices))      if l0_dices      else float("nan"),
        "val/dice_fused":   float(np.nanmean(l1_dices))      if l1_dices      else float("nan"),
        "val/soft_dice":    float(np.nanmean(l1_soft_dices)) if l1_soft_dices else float("nan"),
        "val/norm_dice":    float(np.nanmean(l1_norm_dices)) if l1_norm_dices else float("nan"),
        "val/auroc":        float(np.nanmean(l1_aurocs))     if l1_aurocs     else float("nan"),
        "val/loss":         float(np.mean(l1_losses))        if l1_losses     else float("nan"),
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
        torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg.paths.results) / "multilevel"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Augmentation config -----------------------------------------------
    aug_cfg = None
    if cfg.train.aug:
        aug_yaml = ROOT / "configs" / "augmentations" / f"{cfg.train.aug_preset}.yaml"
        aug_cfg  = OmegaConf.load(aug_yaml).augmentations

    train_classes = resolve_classes(cfg.data.train_classes, cfg.paths.totalseg)
    val_classes   = resolve_classes(cfg.data.val_classes, cfg.paths.totalseg) if cfg.data.val_classes else []
    val_classes   = val_classes or train_classes

    # ---- Datasets ----------------------------------------------------------
    ds_train = TotalSegInContextDataset(
        root=cfg.paths.totalseg,
        classes=train_classes,
        image_size=tuple(cfg.data.image_size),
        split="train",
        context_size=cfg.data.context_size,
        class_balanced=cfg.data.class_balanced,
        aug_cfg=aug_cfg,
        use_crop=cfg.data.use_crop,
        synth_method=cfg.data.synth_method or None,
        synth_unions=cfg.data.synth_unions,
        p_synth=cfg.data.p_synth,
        random_coloring=False,
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

    n_train = min(cfg.data.max_ds_len_train, len(ds_train))
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        sampler=RandomSampler(ds_train, replacement=False, num_samples=n_train),
        num_workers=cfg.train.workers,
        pin_memory=True,
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=2 if cfg.train.workers > 0 else None,
        collate_fn=incontext_collate_fn,
        drop_last=True,
    )
    print(f"Train: {n_train} samples  |  {len(train_loader)} batches/epoch")

    # ---- Encoder (frozen) -------------------------------------------------
    encoder = STUNetEncoder(
        in_channels=1,
        variant=cfg.model.stunet_variant,
        pretrained=cfg.model.stunet_pretrained,
        freeze_encoder=True,
    ).to(device).eval()

    num_levels = len(encoder.skip_channels) + 1
    level      = cfg.model.feature_level

    # Determine embed_dim from a dummy forward
    with torch.inference_mode():
        dummy      = torch.zeros(1, 1, *cfg.data.image_size, device=device)
        dummy_feat = encode_image_only(encoder, dummy)
        dummy_ds   = extract_features(dummy_feat, level, tuple(cfg.data.resolutions[0]), num_levels)
    embed_dim = dummy_ds.shape[1]
    print(f"embed_dim={embed_dim}  |  level={level}  |  resolutions={list(cfg.data.resolutions)}")

    # ---- Model: one PatchICLAttention per level ---------------------------
    resolutions = [tuple(r) for r in cfg.data.resolutions]
    level_cfgs  = [
        {
            "grid_size":       res,
            "dim":             cfg.model.dim,
            "num_heads":       cfg.model.num_heads,
            "num_layers":      cfg.model.num_layers,
            "ff_factor":       cfg.model.ff_factor,
            "label_injection": cfg.model.label_injection,
            "output_head":     cfg.model.output_head,
            # L0 (dense) uses its configured pos_encoding; sparse levels use "none"
            # because the grid-based sinusoidal PE can't handle arbitrary subsets of
            # positions.  Spatial info is injected via MultilevelICL.coord_projs instead.
            "pos_encoding":    cfg.model.pos_encoding if i == 0 else "none",
            "input_norm":      cfg.model.input_norm,
            "dropout":         cfg.model.dropout,
            "ctx_self_attn":   cfg.model.ctx_self_attn,
            "log_n_scaling":   cfg.model.log_n_scaling,
            "log_n_base":      cfg.model.log_n_base,
            "soft_labels":     cfg.model.soft_labels,
        }
        for i, res in enumerate(resolutions)
    ]
    model = MultilevelICL(embed_dim=embed_dim, level_cfgs=level_cfgs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultilevelICL  params: {n_params:,}  ({len(model)} levels)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    amp       = device.type == "cuda"

    if cfg.train.checkpoint:
        ckpt = torch.load(cfg.train.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {cfg.train.checkpoint}  (epoch {ckpt['epoch']})")

    # ---- W&B ---------------------------------------------------------------
    use_wandb = bool(cfg.train.wandb_project) and str(cfg.train.wandb_project).lower() != "null"
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.run_name or None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({"embed_dim": embed_dim, "n_params": n_params})

    date_str = datetime.today().strftime("%Y-%m-%d")
    run_name  = (wandb.run.name if use_wandb else None) or cfg.train.run_name or "run"
    run_dir   = out_dir / f"{date_str}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir   = run_dir / "figures"

    # ---- Training ----------------------------------------------------------
    best_dice = -1.0
    nd_interval = cfg.train.nd_interval

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss, epoch_l0, epoch_l1 = 0.0, 0.0, 0.0
        epoch_nd, epoch_dice = 0.0, 0.0
        n_batches, n_nd, n_dice = 0, 0, 0
        last_nd, last_dice = float("nan"), float("nan")
        t0 = time.perf_counter()

        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{cfg.train.epochs}", unit="batch", leave=False)
        for batch in bar:
            pred_0, pred_1, tgt_idx, loss, (l0, l1) = process_batch(
                encoder, model, batch, cfg, device, amp=amp,
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            n_batches  += 1
            epoch_loss += loss.item()
            epoch_l0   += l0
            epoch_l1   += l1

            if n_batches % nd_interval == 0:
                with torch.no_grad():
                    # Quick metrics on L0 (cheap, no sampling)
                    res_0 = tuple(cfg.data.resolutions[0])
                    N0 = res_0[0] * res_0[1] * res_0[2]
                    gt_0 = downsample_mask(
                        batch["label"].to(device), res_0, cfg.data.mask_pool
                    ).reshape(pred_0.shape[0], N0)
                    gt_bin = (gt_0 > 0).float()
                    nd = norm_dice_score(pred_0.detach(), gt_bin)
                    dc = dice_score(pred_0.detach(), gt_bin)
                if nd == nd:
                    epoch_nd   += nd; n_nd   += 1; last_nd   = nd
                if dc == dc:
                    epoch_dice += dc; n_dice += 1; last_dice = dc

            bar.set_postfix(
                loss=f"{epoch_loss / n_batches:.4f}",
                l0=f"{epoch_l0 / n_batches:.4f}",
                l1=f"{epoch_l1 / n_batches:.4f}",
                dice=f"{last_dice:.3f}",
                nd=f"{last_nd:.3f}",
            )

        bar.close()
        elapsed  = time.perf_counter() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_l0   = epoch_l0   / max(n_batches, 1)
        avg_l1   = epoch_l1   / max(n_batches, 1)
        print(f"Epoch {epoch:3d}/{cfg.train.epochs}  "
              f"loss={avg_loss:.4f}  l0={avg_l0:.4f}  l1={avg_l1:.4f}  "
              f"dice_l0={epoch_dice / max(n_dice, 1):.3f}  {elapsed:.0f}s")

        # Validation
        val_metrics, val_figs = validate(
            model, encoder, ds_val, cfg, device,
            cfg.train.val_items_per_class,
            fig_dir=fig_dir, epoch=epoch,
        )
        print(f"  val dice_l0={val_metrics['val/dice_l0']:.3f}  "
              f"dice_fused={val_metrics['val/dice_fused']:.3f}  "
              f"auroc={val_metrics['val/auroc']:.3f}")

        if val_metrics["val/dice_fused"] > best_dice:
            best_dice = val_metrics["val/dice_fused"]
            ckpt = {
                "epoch":  epoch,
                "model":  model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "val_dice": best_dice,
            }
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  saved best checkpoint  dice_fused={best_dice:.3f}")

        if use_wandb:
            import wandb
            all_figs = {k: wandb.Image(str(v)) for k, v in val_figs.items()}
            wandb.log({
                "train/loss": avg_loss, "train/loss_l0": avg_l0, "train/loss_l1": avg_l1,
                "train/dice_l0": epoch_dice / max(n_dice, 1),
                "epoch": epoch, **val_metrics, **all_figs,
            })

    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nBest val dice_fused: {best_dice:.3f}  |  checkpoint: {run_dir}/best.pt")


if __name__ == "__main__":
    main()

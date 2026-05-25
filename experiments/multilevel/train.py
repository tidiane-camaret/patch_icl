"""
Multilevel coarse-to-fine PatchICLAttention on TotalSegmentator.

Two levels trained jointly with detached gradients between them:
  L0: dense 8³ grid  — all 512 patches, standard ICL forward.
  L1: sparse 16³ grid — NP₁ patches sampled via Gumbel-TopK:
        target  : gt_previous_pred_error or gt_foreground_entropy_balanced
                  (controlled by data.target_sampling).
        context : gt_foreground_entropy_balanced (0.5·GT + 0.5·H(GT)).

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


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """H(p) = -p log p - (1-p) log(1-p), clamped for numerical stability."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def _gumbel_topk(weights: torch.Tensor, n: int, temperature: float) -> torch.Tensor:
    """Stochastic top-n sampling via Gumbel noise.

    Normalises weights to [0,1] per batch item, perturbs with Gumbel noise,
    then takes the top-n scores.  Returns (B, n) unique LongTensor indices.
    """
    w_min = weights.min(dim=1, keepdim=True).values
    w_max = weights.max(dim=1, keepdim=True).values
    w = (weights - w_min) / (w_max - w_min + 1e-6)
    u = torch.rand_like(w).clamp(1e-6, 1 - 1e-6)
    gumbel = -torch.log(-torch.log(u))
    scores = w / max(temperature, 1e-6) + gumbel
    return scores.topk(n, dim=1).indices  # always unique


def sample_target_patches(
    pred_0_up: torch.Tensor,   # (B, N1) float in [0,1]
    gt_1_flat: torch.Tensor,   # (B, N1) float in [0,1]
    n_patches: int,
    temperature: float = 1.0,
    mode: str = "gt_previous_pred_error",
) -> torch.Tensor:             # (B, n_patches) long
    """Gumbel-TopK target patch sampling.

    gt_previous_pred_error         : weight = |pred_0_up - GT|       (where L0 is wrong)
    gt_foreground_entropy_balanced : weight = 0.5·GT + 0.5·H(GT)     (fg + boundary, GT-only)
    predicted_entropy              : weight = H(pred_0_up)            (where L0 is uncertain)
    """
    if mode == "gt_previous_pred_error":
        weights = (pred_0_up - gt_1_flat).abs()
    elif mode == "gt_foreground_entropy_balanced":
        weights = 0.5 * gt_1_flat + 0.5 * _binary_entropy(gt_1_flat)
    elif mode == "predicted_entropy":
        weights = _binary_entropy(pred_0_up)
    else:
        raise ValueError(f"Unknown target_sampling mode: {mode!r}")
    return _gumbel_topk(weights, n_patches, temperature)


def sample_context_patches(
    ctx_mask_16: torch.Tensor,  # (B, K, D1, H1, W1) float avg-pooled
    n_patches: int,
    temperature: float = 1.0,
) -> torch.Tensor:              # (B, n_patches) long — shared across K context volumes
    """Gumbel-TopK over gt_foreground_entropy_balanced averaged across K context masks.

    weight = 0.5 * avg_mask + 0.5 * H(avg_mask)
    High-weight positions are both foreground (avg_mask large) and boundary
    (avg_mask ≈ 0.5, where H is maximal).
    """
    B, K, D1, H1, W1 = ctx_mask_16.shape
    N1 = D1 * H1 * W1
    avg_mask = ctx_mask_16.mean(dim=1).reshape(B, N1)
    weights  = 0.5 * avg_mask + 0.5 * _binary_entropy(avg_mask)
    return _gumbel_topk(weights, n_patches, temperature)


def gather_patches(feat_flat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feat_flat: (B, N, C), idx: (B, NP) → (B, NP, C)."""
    B, NP = idx.shape
    C = feat_flat.shape[-1]
    idx_exp = idx.unsqueeze(-1).expand(B, NP, C)
    return feat_flat.gather(1, idx_exp)


def grid_coords_3d(grid_size: tuple, device: torch.device) -> torch.Tensor:
    """Integer (d, h, w) voxel coords for every position in a D×H×W grid.

    Returns (N, 3) long where N = D*H*W (d slowest, w fastest — matches
    the default C-order reshape used everywhere in this file).
    """
    D, H, W = grid_size
    d = torch.arange(D, device=device)
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    gd, gh, gw = torch.meshgrid(d, h, w, indexing="ij")
    return torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)  # (N, 3)


# ---------------------------------------------------------------------------
# Context label encoding
# ---------------------------------------------------------------------------

def _encode_ctx_labels(
    ctx_mask_i:   torch.Tensor,         # (B, K, *res) float avg-pool
    ctx_idx:      torch.Tensor | None,  # (B, NP) long | None → dense
    soft:         bool,
    mask_cnn_vol: torch.Tensor | None,  # (B*K, mask_dim, D_f, H_f, W_f) pre-encoded
    res:          tuple,                # current level resolution
) -> torch.Tensor:
    """Build context label tensor for one resolution level.

    With mask_cnn_vol: trilinearly interpolate the pre-encoded feature volume to
    `res`, then gather sparse positions when ctx_idx is given.
    Without it: scalar avg-pool values (original behaviour).
    """
    B, K = ctx_mask_i.shape[:2]
    N    = res[0] * res[1] * res[2]

    if mask_cnn_vol is not None:
        mask_dim = mask_cnn_vol.shape[1]
        feat = (
            F.interpolate(mask_cnn_vol, size=res, mode="trilinear", align_corners=False)
            if tuple(mask_cnn_vol.shape[2:]) != res
            else mask_cnn_vol
        )  # (B*K, mask_dim, *res)
        emb = feat.flatten(2).transpose(1, 2).reshape(B, K, N, mask_dim)
        if ctx_idx is None:
            return emb.reshape(B, K * N, mask_dim)
        idx_exp = ctx_idx.unsqueeze(-1).expand(-1, -1, mask_dim)
        return torch.cat([emb[:, k].gather(1, idx_exp) for k in range(K)], dim=1)

    # Scalar fallback
    flat = ctx_mask_i.reshape(B, K, N)
    if ctx_idx is None:
        lbl = flat.reshape(B, K * N)
    else:
        lbl = torch.cat([flat[:, k].gather(1, ctx_idx) for k in range(K)], dim=1)
    return lbl if soft else (lbl > 0).float()


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
) -> tuple[list, torch.Tensor, list, list, list, list]:
    """Returns (preds, loss, level_losses, grid_preds, tgt_idxs, ctx_idxs).

    preds        : per-level predictions — (B, N_i) dense for L0, (B, NP) sparse for L>0
    loss         : weighted scalar loss
    level_losses : per-level loss scalars for logging
    grid_preds   : per-level fused prediction grids — (B, N_i)
    tgt_idxs     : per-level sampled target indices — None for L0, (B, NP) for L>0
    ctx_idxs     : per-level sampled context indices — None for L0, (B, NP) for L>0
    """
    images  = batch["image"].to(device, non_blocking=True)
    labels  = batch["label"].to(device, non_blocking=True)
    ctx_in  = batch["context_in"].to(device, non_blocking=True)
    ctx_out = batch["context_out"].to(device, non_blocking=True)
    B, K = ctx_in.shape[:2]

    resolutions = [tuple(r) for r in cfg.data.resolutions]
    level       = cfg.model.feature_level
    num_levels  = len(encoder.skip_channels) + 1
    NP          = cfg.data.n_patches_l1   # used for all sparse levels (L>0)
    temp        = cfg.data.sampling_temperature

    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        tgt_feats      = encode_image_only(encoder, images)
        ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
        ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

    preds, grid_preds, raw_losses, tgt_idxs, ctx_idxs = [], [], [], [], []
    C = None
    cascade_regs = None

    # Encode context masks once at finest resolution; interpolate per level below
    mask_cnn_vol = None
    if model.mask_cnn is not None:
        finest_res = resolutions[-1]
        mask_in = downsample_mask(
            ctx_out.reshape(B * K, *ctx_out.shape[2:]), finest_res, cfg.data.mask_pool
        ).unsqueeze(1)                                    # (B*K, 1, *finest_res)
        if not cfg.model.soft_labels_train:
            mask_in = (mask_in > 0).float()
        with torch.autocast(device_type=device.type, enabled=amp):
            emb = model.mask_cnn(mask_in)                # (B*K, N_f, mask_dim)
        mask_cnn_vol = emb.transpose(1, 2).reshape(
            B * K, model.mask_cnn_dim, *finest_res
        )                                                 # (B*K, mask_dim, D_f, H_f, W_f)

    for i, res in enumerate(resolutions):
        N = res[0] * res[1] * res[2]
        tgt_feat_i  = extract_features(tgt_feats,      level, res, num_levels)
        ctx_feat_if = extract_features(ctx_feats_flat, level, res, num_levels)
        if C is None:
            C = tgt_feat_i.shape[1]
        ctx_feat_i  = ctx_feat_if.reshape(B, K, C, *res)
        ctx_mask_i  = downsample_mask(
            ctx_out.reshape(B * K, *ctx_out.shape[2:]), res, cfg.data.mask_pool
        ).reshape(B, K, *res)

        tgt_idx = ctx_idx = None
        if i == 0:
            # Dense forward over all N patches
            tgt_f   = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
            ctx_f   = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K * N, C)
            ctx_lbl = _encode_ctx_labels(ctx_mask_i, None, cfg.model.soft_labels_train,
                                         mask_cnn_vol, res)
            gt      = downsample_mask(labels, res, cfg.data.mask_pool).reshape(B, N)

            coords   = grid_coords_3d(res, device)
            tgt_crds = coords.unsqueeze(0).expand(B, -1, -1)
            ctx_crds = coords.unsqueeze(0).expand(B, -1, -1).repeat(1, K, 1)

            with torch.autocast(device_type=device.type, enabled=amp):
                result = model[i](tgt_f, ctx_f, ctx_lbl,
                                  tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                  cascade_registers=cascade_regs)
            if isinstance(result, tuple):
                regs = result[1]
                cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                pred = result[0].float()
            else:
                pred = result.float()
            grid_pred = pred

        else:
            # Sparse: Gumbel-TopK sampling guided by the previous level's fused grid
            prev_res = resolutions[i - 1]
            with torch.no_grad():
                prev_up = F.interpolate(
                    grid_preds[-1].detach().reshape(B, 1, *prev_res),
                    size=res, mode="trilinear", align_corners=False,
                ).reshape(B, N)

            gt_flat  = downsample_mask(labels, res, cfg.data.mask_pool).reshape(B, N)
            tgt_idx  = sample_target_patches(prev_up, gt_flat, NP, temp,
                                             cfg.data.target_sampling)
            ctx_idx  = sample_context_patches(ctx_mask_i, NP, temp)

            tgt_flat = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
            ctx_flat = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K, N, C)

            tgt_sparse = gather_patches(tgt_flat, tgt_idx)
            ctx_pieces = [gather_patches(ctx_flat[:, k], ctx_idx) for k in range(K)]
            ctx_sparse = torch.cat(ctx_pieces, dim=1)
            ctx_lbl    = _encode_ctx_labels(ctx_mask_i, ctx_idx, cfg.model.soft_labels_train,
                                            mask_cnn_vol, res)

            coords   = grid_coords_3d(res, device)
            tgt_crds = coords[tgt_idx.reshape(-1)].reshape(B, NP, 3)
            ctx_crds = coords[ctx_idx.reshape(-1)].reshape(B, NP, 3)
            ctx_crds = ctx_crds.unsqueeze(1).expand(-1, K, -1, -1).reshape(B, K * NP, 3)

            with torch.autocast(device_type=device.type, enabled=amp):
                result = model[i](tgt_sparse, ctx_sparse, ctx_lbl,
                                  tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                  cascade_registers=cascade_regs)
            if isinstance(result, tuple):
                regs = result[1]
                cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                pred = result[0].float()
            else:
                pred = result.float()
            grid_pred = prev_up.clone()
            grid_pred.scatter_(1, tgt_idx, pred)
            gt        = gt_flat.gather(1, tgt_idx)

        raw_losses.append(F.binary_cross_entropy(pred, gt))
        preds.append(pred)
        grid_preds.append(grid_pred)
        tgt_idxs.append(tgt_idx)
        ctx_idxs.append(ctx_idx)

    weights = list(cfg.train.loss_weights)
    while len(weights) < len(raw_losses):
        weights.append(1.0)
    loss = sum(w * l for w, l in zip(weights, raw_losses))

    return preds, loss, [l.item() for l in raw_losses], grid_preds, tgt_idxs, ctx_idxs


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


def _patch_overlay(
    ax,
    image:      np.ndarray,           # (D, H, W) full-res image
    gt:         np.ndarray,           # (D, H, W) binary GT
    z_img:      int,                  # slice index in image space
    grid_size:  tuple,                # (D', H', W') — patch grid resolution
    patch_idx:  np.ndarray | None,    # flat indices into grid (None = dense)
    color:      str = "cyan",
    title:      str = "",
) -> None:
    """Image slice + GT overlay + sampled patch positions.

    Dense levels (patch_idx=None): draw light grid lines at patch boundaries.
    Sparse levels: draw filled boxes for each sampled patch on the best slice.
    """
    D_img, H_img, W_img = image.shape
    D_grd, H_grd, W_grd = grid_size
    ph = H_img / H_grd          # patch height in image pixels
    pw = W_img / W_grd          # patch width in image pixels
    z_grd = int(z_img * D_grd / D_img)   # corresponding grid depth slice

    sl = image[z_img]
    sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
    ax.imshow(sl_norm, cmap="gray")

    gt_sl = (gt[z_img] > 0).astype(float)
    if gt_sl.max() > 0:
        ax.imshow(gt_sl, cmap="Reds", alpha=0.35, vmin=0, vmax=1)

    if patch_idx is None:
        # Dense: light grid lines at patch boundaries
        for i in range(1, H_grd):
            ax.axhline(i * ph - 0.5, color=color, linewidth=0.4, alpha=0.4)
        for j in range(1, W_grd):
            ax.axvline(j * pw - 0.5, color=color, linewidth=0.4, alpha=0.4)
    else:
        # Sparse: filled box per sampled patch that intersects this depth slice
        d_p = patch_idx // (H_grd * W_grd)
        h_p = (patch_idx % (H_grd * W_grd)) // W_grd
        w_p = patch_idx % W_grd
        for d, h, w in zip(d_p, h_p, w_p):
            if d == z_grd:
                rect = plt.Rectangle(
                    (w * pw, h * ph), pw, ph,
                    linewidth=0.8, edgecolor=color, facecolor=color, alpha=0.4,
                )
                ax.add_patch(rect)

    ax.set_title(title, fontsize=7)
    ax.axis("off")


def save_val_figure(
    tgt_image:   np.ndarray,        # (D, H, W) full-res
    tgt_gt:      np.ndarray,        # (D, H, W) binary GT full-res
    ctx_image:   np.ndarray,        # (D, H, W) first context image
    ctx_gt:      np.ndarray,        # (D, H, W) first context GT
    levels:      list[dict],        # one dict per level — see below
    out_path:    Path,
    title:       str = "",
) -> None:
    """Save a figure with one row per level.

    Each row has 5 columns:
      0  Target slice + GT + sampled patch positions (cyan boxes / grid lines)
      1  Context slice + GT + sampled patch positions (lime boxes / grid lines)
      2  Downsampled GT mask at this level's resolution
      3  Prediction at this level (L0: dense 8³; L1: sparse NP patches in 16³ grid)
      4  Fused prediction (L0: pred_0 upsampled to 16³; L1: final pred_fused)

    Each level dict must contain:
      res        : (D', H', W') grid resolution
      gt_ds      : (D', H', W') GT downsampled to res
      pred       : (D', H', W') prediction at this level (zeros outside sampled pos for L1)
      pred_fused : (D_f, H_f, W_f) cumulative fused prediction (same shape for all levels)
      tgt_idx    : np.ndarray (NP,) flat patch indices, or None for dense
      ctx_idx    : np.ndarray (NP,) flat patch indices, or None for dense
    """
    n_levels = len(levels)
    fig, axes = plt.subplots(n_levels, 5, figsize=(18, 3.8 * n_levels),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.25})
    if n_levels == 1:
        axes = axes[np.newaxis, :]

    tgt_z = _best_slice(tgt_gt.astype(float))
    ctx_z = _best_slice(ctx_gt.astype(float))

    for row, lvl in enumerate(levels):
        res        = lvl["res"]          # (D', H', W')
        gt_ds      = lvl["gt_ds"]        # (D', H', W')
        pred       = lvl["pred"]         # (D', H', W')
        pred_fused = lvl["pred_fused"]   # (D_f, H_f, W_f)
        tgt_idx    = lvl.get("tgt_idx")  # None or (NP,)
        ctx_idx    = lvl.get("ctx_idx")  # None or (NP,)

        z_ds    = int(tgt_z * res[0]          / tgt_gt.shape[0])
        z_fused = int(tgt_z * pred_fused.shape[0] / tgt_gt.shape[0])

        _patch_overlay(axes[row, 0], tgt_image, tgt_gt, tgt_z,
                       res, tgt_idx, color="cyan",
                       title=f"L{row} target  {res}")
        _patch_overlay(axes[row, 1], ctx_image, ctx_gt, ctx_z,
                       res, ctx_idx, color="lime",
                       title=f"L{row} context  {res}")
        _heatmap(axes[row, 2], gt_ds,      z_ds,    f"GT ↓ L{row}")
        _heatmap(axes[row, 3], pred,       z_ds,    f"pred L{row}")
        _heatmap(axes[row, 4], pred_fused, z_fused, f"fused L{row}")

    fig.suptitle(title, fontsize=9)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training synth visualisation
# ---------------------------------------------------------------------------

def _save_synth_train_figure(
    batch:       dict,
    preds:       list,
    grid_preds:  list,
    tgt_idxs:    list,
    ctx_idxs:    list,
    b:           int,
    epoch:       int,
    out_dir:     Path,
    resolutions: list,
    cfg:         OmegaConf,
) -> Path:
    """Save a multilevel pred figure for one synth training item (no grad needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_name = batch["label_names"][b]
    label_b    = batch["label"][b:b + 1]          # CPU tensor

    tgt_np    = batch["image"][b].squeeze().cpu().numpy()
    tgt_gt_np = batch["label"][b].cpu().numpy()
    ctx_np    = batch["context_in"][b, 0].squeeze(0).cpu().numpy()
    ctx_gt_np = batch["context_out"][b, 0].cpu().numpy()

    levels_data = []
    for i, res in enumerate(resolutions):
        D_, H_, W_ = res
        N_i      = D_ * H_ * W_
        gt_ds_np = downsample_mask(label_b, res, cfg.data.mask_pool).squeeze().cpu().numpy().reshape(D_, H_, W_)
        fused_np = grid_preds[i][b].detach().cpu().numpy().reshape(D_, H_, W_)

        if tgt_idxs[i] is None:
            pred_np = preds[i][b].detach().cpu().numpy().reshape(D_, H_, W_)
        else:
            sparse = np.zeros(N_i, dtype=np.float32)
            sparse[tgt_idxs[i][b].cpu().numpy()] = preds[i][b].detach().cpu().numpy()
            pred_np = sparse.reshape(D_, H_, W_)

        levels_data.append({
            "res":        res,
            "gt_ds":      gt_ds_np,
            "pred":       pred_np,
            "pred_fused": fused_np,
            "tgt_idx":    tgt_idxs[i][b].cpu().numpy() if tgt_idxs[i] is not None else None,
            "ctx_idx":    ctx_idxs[i][b].cpu().numpy() if ctx_idxs[i] is not None else None,
        })

    fig_path = out_dir / f"epoch{epoch:03d}_{label_name}.png"
    save_val_figure(
        tgt_image=tgt_np, tgt_gt=tgt_gt_np,
        ctx_image=ctx_np, ctx_gt=ctx_gt_np,
        levels=levels_data, out_path=fig_path,
        title=f"[ep {epoch}] train synth  {label_name}",
    )
    return fig_path


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
    amp = device.type == "cuda"
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [tuple(r) for r in cfg.data.resolutions]
    n_levels    = len(resolutions)
    level       = cfg.model.feature_level
    num_levels  = len(encoder.skip_channels) + 1
    NP          = cfg.data.n_patches_l1
    temp        = cfg.data.sampling_temperature

    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.workers,
        collate_fn=incontext_collate_fn,
        pin_memory=True,
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=2 if cfg.train.workers > 0 else None,
        drop_last=False,
    )

    # dice_l{i}       : dice at level i's resolution (dense for L0, sparse for L>0)
    # dice_fused_l{i} : dice of the fused grid at level i vs GT at that resolution
    level_dices       = [[] for _ in range(n_levels)]
    level_fused_dices = [[] for _ in range(n_levels)]
    final_dices, final_soft_dices, final_norm_dices, final_aurocs, final_losses = [], [], [], [], []
    wandb_images: dict  = {}
    cls_fig_saved: set  = set()
    class_counts: dict  = defaultdict(int)

    for batch in val_loader:
        label_names = batch["label_names"]          # list[B]
        images  = batch["image"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        ctx_in  = batch["context_in"].to(device, non_blocking=True)
        ctx_out = batch["context_out"].to(device, non_blocking=True)
        B, K = ctx_in.shape[:2]

        # Encode all B targets and B*K context images in two batched calls
        with torch.inference_mode():
            tgt_feats      = encode_image_only(encoder, images)
            ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
            ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

        preds_l, tgt_idxs_l, ctx_idxs_l, grid_preds_l = [], [], [], []
        C = None
        cascade_regs = None

        mask_cnn_vol = None
        if model.mask_cnn is not None:
            finest_res = resolutions[-1]
            mask_in = downsample_mask(
                ctx_out.reshape(B * K, *ctx_out.shape[2:]), finest_res, cfg.data.mask_pool
            ).unsqueeze(1)
            if not cfg.model.soft_labels_eval:
                mask_in = (mask_in > 0).float()
            with torch.autocast(device_type=device.type, enabled=amp):
                emb = model.mask_cnn(mask_in)
            mask_cnn_vol = emb.transpose(1, 2).reshape(
                B * K, model.mask_cnn_dim, *finest_res
            )

        for i, res in enumerate(resolutions):
            N = res[0] * res[1] * res[2]
            tgt_feat_i  = extract_features(tgt_feats,      level, res, num_levels)
            ctx_feat_if = extract_features(ctx_feats_flat, level, res, num_levels)
            if C is None:
                C = tgt_feat_i.shape[1]
            ctx_feat_i  = ctx_feat_if.reshape(B, K, C, *res)
            ctx_mask_i  = downsample_mask(
                ctx_out.reshape(B * K, *ctx_out.shape[2:]), res, cfg.data.mask_pool
            ).reshape(B, K, *res)

            if i == 0:
                tgt_f   = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
                ctx_f   = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K * N, C)
                ctx_lbl = _encode_ctx_labels(ctx_mask_i, None, cfg.model.soft_labels_eval,
                                             mask_cnn_vol, res)

                coords   = grid_coords_3d(res, device)
                tgt_crds = coords.unsqueeze(0).expand(B, -1, -1)
                ctx_crds = coords.unsqueeze(0).expand(B, -1, -1).repeat(1, K, 1)

                with torch.autocast(device_type=device.type, enabled=amp):
                    result = model[0](tgt_f, ctx_f, ctx_lbl,
                                      tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                      cascade_registers=cascade_regs)
                if isinstance(result, tuple):
                    regs = result[1]
                    cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                    pred = result[0].float()
                else:
                    pred = result.float()
                tgt_idx   = None
                ctx_idx   = None
                grid_pred = pred

            else:
                prev_res = resolutions[i - 1]
                prev_up  = F.interpolate(
                    grid_preds_l[-1].reshape(B, 1, *prev_res),
                    size=res, mode="trilinear", align_corners=False,
                ).reshape(B, N)

                gt_flat  = downsample_mask(labels, res, cfg.data.mask_pool).reshape(B, N)
                tgt_idx  = sample_target_patches(prev_up, gt_flat, NP, temp,
                                                 cfg.data.target_sampling)
                ctx_idx  = sample_context_patches(ctx_mask_i, NP, temp)

                tgt_flat_f = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
                ctx_flat_f = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K, N, C)

                tgt_sparse = gather_patches(tgt_flat_f, tgt_idx)
                ctx_pieces = [gather_patches(ctx_flat_f[:, k], ctx_idx) for k in range(K)]
                ctx_sparse = torch.cat(ctx_pieces, dim=1)
                ctx_lbl    = _encode_ctx_labels(ctx_mask_i, ctx_idx, cfg.model.soft_labels_eval,
                                                mask_cnn_vol, res)

                coords   = grid_coords_3d(res, device)
                tgt_crds = coords[tgt_idx.reshape(-1)].reshape(B, NP, 3)
                ctx_crds = coords[ctx_idx.reshape(-1)].reshape(B, NP, 3)
                ctx_crds = ctx_crds.unsqueeze(1).expand(-1, K, -1, -1).reshape(B, K * NP, 3)

                with torch.autocast(device_type=device.type, enabled=amp):
                    result = model[i](tgt_sparse, ctx_sparse, ctx_lbl,
                                      tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                      cascade_registers=cascade_regs)
                if isinstance(result, tuple):
                    regs = result[1]
                    cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                    pred = result[0].float()
                else:
                    pred = result.float()
                grid_pred = prev_up.clone()
                grid_pred.scatter_(1, tgt_idx, pred)

            preds_l.append(pred)
            tgt_idxs_l.append(tgt_idx)
            ctx_idxs_l.append(ctx_idx)
            grid_preds_l.append(grid_pred)

        # --- Per-sample metrics and figures ---
        for b in range(B):
            cls = label_names[b]
            if class_counts[cls] >= items_per_class:
                continue

            label_b = labels[b:b + 1]

            # Per-level metrics
            for i, res in enumerate(resolutions):
                N_i      = res[0] * res[1] * res[2]
                gt_i     = downsample_mask(label_b, res, cfg.data.mask_pool).reshape(N_i)
                gt_i_bin = (gt_i > 0).float()

                if tgt_idxs_l[i] is None:
                    di = dice_score(preds_l[i][b], gt_i_bin)
                else:
                    gt_at_tgt = gt_i[tgt_idxs_l[i][b]]
                    di = dice_score(preds_l[i][b], (gt_at_tgt > 0).float())
                if di == di:
                    level_dices[i].append(di)

                df = dice_score(grid_preds_l[i][b], gt_i_bin)
                if df == df:
                    level_fused_dices[i].append(df)

            # Final level full metrics
            final_res  = resolutions[-1]
            N_final    = final_res[0] * final_res[1] * final_res[2]
            gt_final   = downsample_mask(label_b, final_res, cfg.data.mask_pool).reshape(N_final)
            gt_final_b = (gt_final > 0).float()
            pf         = grid_preds_l[-1][b]

            final_losses.append(F.binary_cross_entropy(pf, gt_final_b).item())
            d = dice_score(pf, gt_final_b)
            if d == d:
                final_dices.append(d)
            sd = soft_dice_score(pf, gt_final_b)
            if sd == sd:
                final_soft_dices.append(sd)
            nd = norm_dice_score(pf, gt_final_b)
            if nd == nd:
                final_norm_dices.append(nd)
            pf_np = pf.cpu().numpy()
            gf_np = gt_final_b.cpu().numpy().astype(int)
            if 0 < gf_np.sum() < len(gf_np):
                try:
                    final_aurocs.append(roc_auc_score(gf_np, pf_np))
                except Exception:
                    pass

            # Visualisation — one figure per class, first time we see it
            if cls not in cls_fig_saved and fig_dir is not None:
                tgt_np    = batch["image"][b].squeeze().cpu().numpy()
                tgt_gt_np = batch["label"][b].cpu().numpy()
                ctx_np    = batch["context_in"][b, 0].squeeze(0).cpu().numpy()
                ctx_gt_np = batch["context_out"][b, 0].cpu().numpy()

                levels_data = []
                for i, res in enumerate(resolutions):
                    D_, H_, W_ = res
                    N_i        = D_ * H_ * W_
                    gt_ds_np   = downsample_mask(label_b, res, cfg.data.mask_pool).squeeze().cpu().numpy().reshape(D_, H_, W_)
                    fused_np   = grid_preds_l[i][b].cpu().numpy().reshape(D_, H_, W_)

                    if tgt_idxs_l[i] is None:
                        pred_np = preds_l[i][b].cpu().numpy().reshape(D_, H_, W_)
                    else:
                        sparse = np.zeros(N_i, dtype=np.float32)
                        sparse[tgt_idxs_l[i][b].cpu().numpy()] = preds_l[i][b].cpu().numpy()
                        pred_np = sparse.reshape(D_, H_, W_)

                    levels_data.append({
                        "res":        res,
                        "gt_ds":      gt_ds_np,
                        "pred":       pred_np,
                        "pred_fused": fused_np,
                        "tgt_idx":    tgt_idxs_l[i][b].cpu().numpy() if tgt_idxs_l[i] is not None else None,
                        "ctx_idx":    ctx_idxs_l[i][b].cpu().numpy() if ctx_idxs_l[i] is not None else None,
                    })

                title_parts = [f"[ep {epoch}] {cls}"]
                for i in range(n_levels):
                    v = level_fused_dices[i][-1] if level_fused_dices[i] else float("nan")
                    title_parts.append(f"fused_l{i}={v:.3f}")
                fig_path = fig_dir / f"epoch{epoch:03d}_{cls}.png"
                save_val_figure(
                    tgt_image=tgt_np, tgt_gt=tgt_gt_np,
                    ctx_image=ctx_np, ctx_gt=ctx_gt_np,
                    levels=levels_data, out_path=fig_path,
                    title="  ".join(title_parts),
                )
                wandb_images[f"val/pred/{cls}"] = fig_path
                cls_fig_saved.add(cls)

            class_counts[cls] += 1

    model.train()
    metrics: dict = {}
    for i in range(n_levels):
        metrics[f"val/dice_l{i}"]       = float(np.nanmean(level_dices[i]))       if level_dices[i]       else float("nan")
        metrics[f"val/dice_fused_l{i}"] = float(np.nanmean(level_fused_dices[i])) if level_fused_dices[i] else float("nan")
    metrics["val/dice"]      = float(np.nanmean(final_dices))      if final_dices      else float("nan")
    metrics["val/soft_dice"] = float(np.nanmean(final_soft_dices)) if final_soft_dices else float("nan")
    metrics["val/norm_dice"] = float(np.nanmean(final_norm_dices)) if final_norm_dices else float("nan")
    metrics["val/auroc"]     = float(np.nanmean(final_aurocs))     if final_aurocs     else float("nan")
    metrics["val/loss"]      = float(np.mean(final_losses))        if final_losses     else float("nan")
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
        n_synth_merge_min=cfg.data.n_synth_merge_min,
        n_synth_merge_max=cfg.data.n_synth_merge_max,
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
            # rope3d works at any level (uses explicit coords); sinusoidal/learned
            # assume a dense fixed grid so they fall back to "none" for sparse levels.
            "pos_encoding":    cfg.model.pos_encoding
                               if (i == 0 or cfg.model.pos_encoding == "rope3d")
                               else "none",
            "input_norm":      cfg.model.input_norm,
            "dropout":         cfg.model.dropout,
            "ctx_self_attn":   cfg.model.ctx_self_attn,
            "log_n_scaling":   cfg.model.log_n_scaling,
            "log_n_base":      cfg.model.log_n_base,
            "soft_labels":     cfg.model.soft_labels_train,
        }
        for i, res in enumerate(resolutions)
    ]
    mask_cnn_dim     = int(getattr(cfg.model, "mask_cnn_dim", 0) or 0)
    num_registers    = int(getattr(cfg.model, "num_registers", 0) or 0)
    append_zero_attn = bool(getattr(cfg.model, "append_zero_attn", False))
    shared_weights   = bool(getattr(cfg.model, "shared_weights", False))
    model = MultilevelICL(embed_dim=embed_dim, level_cfgs=level_cfgs,
                          mask_cnn_dim=mask_cnn_dim,
                          num_registers=num_registers,
                          append_zero_attn=append_zero_attn,
                          shared_weights=shared_weights).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultilevelICL  params: {n_params:,}  ({len(model)} levels)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    amp       = device.type == "cuda"

    if cfg.train.checkpoint:
        ckpt = torch.load(cfg.train.checkpoint, map_location=device)
        ckpt_state = ckpt["model"]

        # Remap keys when switching between shared and per-level weight layouts.
        # per-level → shared : take levels.0.* as the shared initialisation.
        # shared → per-level : broadcast shared_level.* to every levels.i.* slot.
        ckpt_has_levels = any(k.startswith("levels.") for k in ckpt_state)
        ckpt_has_shared = any(k.startswith("shared_level.") for k in ckpt_state)
        cur_has_shared  = model.shared_weights

        if ckpt_has_levels and cur_has_shared:
            remapped = {}
            for k, v in ckpt_state.items():
                if k.startswith("levels.0."):
                    remapped["shared_level." + k[len("levels.0."):]] = v
                elif not k.startswith("levels."):
                    remapped[k] = v
            ckpt_state = remapped
            print("  Checkpoint remapped: levels.0.* → shared_level.*")
        elif ckpt_has_shared and not cur_has_shared:
            level_indices = sorted({
                int(k.split(".")[1]) for k in model.state_dict() if k.startswith("levels.")
            })
            remapped = {k: v for k, v in ckpt_state.items() if not k.startswith("shared_level.")}
            for k, v in ckpt_state.items():
                if k.startswith("shared_level."):
                    suffix = k[len("shared_level."):]
                    for i in level_indices:
                        remapped[f"levels.{i}.{suffix}"] = v
            ckpt_state = remapped
            print(f"  Checkpoint remapped: shared_level.* → levels.{level_indices}.*")

        current_shapes = {k: v.shape for k, v in model.state_dict().items()}
        ckpt_model = {k: v for k, v in ckpt_state.items()
                      if k in current_shapes and v.shape == current_shapes[k]}
        shape_skipped = [k for k in ckpt_state if k in current_shapes
                         and ckpt_state[k].shape != current_shapes[k]]
        missing, unexpected = model.load_state_dict(ckpt_model, strict=False)
        if shape_skipped:
            print(f"  Shape-mismatched keys (random init): {shape_skipped}")
        if missing:
            print(f"  New parameters (random init): {missing}")
        if unexpected:
            print(f"  Ignored checkpoint keys: {unexpected}")
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
    best_dice   = -1.0
    nd_interval = cfg.train.nd_interval
    n_levels    = len(resolutions)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss         = 0.0
        epoch_level_losses = [0.0] * n_levels
        epoch_nd, epoch_dice = 0.0, 0.0
        n_batches, n_nd, n_dice = 0, 0, 0
        last_nd, last_dice = float("nan"), float("nan")
        t0 = time.perf_counter()

        synth_train_fig_path = None
        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{cfg.train.epochs}", unit="batch", leave=False)
        for batch in bar:
            preds, loss, level_losses, grid_preds_b, tgt_idxs_b, ctx_idxs_b = process_batch(
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
            for j, lv in enumerate(level_losses):
                epoch_level_losses[j] += lv

            if n_batches % nd_interval == 0:
                with torch.no_grad():
                    res_0  = tuple(cfg.data.resolutions[0])
                    N0     = res_0[0] * res_0[1] * res_0[2]
                    gt_0   = downsample_mask(
                        batch["label"].to(device), res_0, cfg.data.mask_pool
                    ).reshape(preds[0].shape[0], N0)
                    gt_bin = (gt_0 > 0).float()
                    nd = norm_dice_score(preds[0].detach(), gt_bin)
                    dc = dice_score(preds[0].detach(), gt_bin)
                if nd == nd:
                    epoch_nd   += nd; n_nd   += 1; last_nd   = nd
                if dc == dc:
                    epoch_dice += dc; n_dice += 1; last_dice = dc

            if synth_train_fig_path is None and fig_dir is not None:
                for b_idx, lname in enumerate(batch["label_names"]):
                    if lname.startswith("sv_"):
                        synth_train_fig_path = _save_synth_train_figure(
                            batch, preds, grid_preds_b, tgt_idxs_b, ctx_idxs_b,
                            b_idx, epoch, fig_dir / "train_synth", resolutions, cfg,
                        )
                        break

            bar.set_postfix(
                loss=f"{epoch_loss / n_batches:.4f}",
                **{f"l{j}": f"{epoch_level_losses[j] / n_batches:.4f}" for j in range(n_levels)},
                dice=f"{last_dice:.3f}",
                nd=f"{last_nd:.3f}",
            )

        bar.close()
        elapsed   = time.perf_counter() - t0
        avg_loss  = epoch_loss / max(n_batches, 1)
        level_str = "  ".join(
            f"l{j}={epoch_level_losses[j] / max(n_batches, 1):.4f}" for j in range(n_levels)
        )
        print(f"Epoch {epoch:3d}/{cfg.train.epochs}  loss={avg_loss:.4f}  {level_str}  "
              f"dice_l0={epoch_dice / max(n_dice, 1):.3f}  {elapsed:.0f}s")

        # Validation
        val_metrics, val_figs = validate(
            model, encoder, ds_val, cfg, device,
            cfg.train.val_items_per_class,
            fig_dir=fig_dir, epoch=epoch,
        )
        dice_str = "  ".join(
            f"{k.split('/')[-1]}={v:.3f}"
            for k, v in val_metrics.items()
            if "dice" in k or k == "val/auroc"
        )
        print(f"  val {dice_str}")

        best_key = f"val/dice_fused_l{n_levels - 1}"
        if val_metrics[best_key] > best_dice:
            best_dice = val_metrics[best_key]
            ckpt = {
                "epoch":    epoch,
                "model":    model.state_dict(),
                "config":   OmegaConf.to_container(cfg, resolve=True),
                "val_dice": best_dice,
            }
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  saved best checkpoint  {best_key.split('/')[-1]}={best_dice:.3f}")

        if use_wandb:
            import wandb
            all_figs        = {k: wandb.Image(str(v)) for k, v in val_figs.items()}
            if synth_train_fig_path is not None:
                all_figs["train/pred_synth"] = wandb.Image(str(synth_train_fig_path))
            level_loss_log  = {f"train/loss_l{j}": epoch_level_losses[j] / max(n_batches, 1)
                               for j in range(n_levels)}
            wandb.log({
                "train/loss": avg_loss,
                **level_loss_log,
                "train/dice_l0": epoch_dice / max(n_dice, 1),
                "epoch": epoch, **val_metrics, **all_figs,
            })

    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nBest val dice_fused_l1: {best_dice:.3f}  |  checkpoint: {run_dir}/best.pt")


if __name__ == "__main__":
    main()

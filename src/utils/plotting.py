"""Visualization utilities shared across experiment training scripts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def _downsample_mask(mask: torch.Tensor, size: tuple, mode: str = "avg") -> torch.Tensor:
    """(B, D, H, W) → (B, D', H', W') float. Local copy to avoid circular imports."""
    x = mask.float().unsqueeze(1)
    if mode == "max":
        return F.adaptive_max_pool3d(x, output_size=size).squeeze(1)
    return F.adaptive_avg_pool3d(x, output_size=size).squeeze(1)


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
    image:     np.ndarray,           # (D, H, W) full-res image
    gt:        np.ndarray,           # (D, H, W) binary GT
    z_img:     int,                  # slice index in image space
    grid_size: tuple,                # (D', H', W') — patch grid resolution
    patch_idx: np.ndarray | None,    # flat indices into grid (None = dense)
    color:     str = "cyan",
    title:     str = "",
    mask_cmap: str = "Reds",
) -> None:
    """Image slice + GT overlay + sampled patch positions.

    Dense levels (patch_idx=None): draw light grid lines at patch boundaries.
    Sparse levels: draw filled boxes for each sampled patch on the best slice.
    """
    D_img, H_img, W_img = image.shape
    D_grd, H_grd, W_grd = grid_size
    ph = H_img / H_grd
    pw = W_img / W_grd
    z_grd = int(z_img * D_grd / D_img)

    sl = image[z_img]
    sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
    ax.imshow(sl_norm, cmap="gray")

    gt_sl = (gt[z_img] > 0).astype(float)
    if gt_sl.max() > 0:
        ax.imshow(gt_sl, cmap=mask_cmap, alpha=0.35, vmin=0, vmax=1)

    if patch_idx is None:
        for i in range(1, H_grd):
            ax.axhline(i * ph - 0.5, color=color, linewidth=0.4, alpha=0.4)
        for j in range(1, W_grd):
            ax.axvline(j * pw - 0.5, color=color, linewidth=0.4, alpha=0.4)
    else:
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
    tgt_image: np.ndarray,        # (D, H, W) full-res
    tgt_gt:    np.ndarray,        # (D, H, W) binary GT full-res
    ctx_image: np.ndarray,        # (D, H, W) first context image
    ctx_gt:    np.ndarray,        # (D, H, W) first context GT
    levels:    list[dict],        # one dict per level — see below
    out_path:  Path,
    title:     str = "",
) -> None:
    """Save a figure with one row per level.

    Each row has 6 columns:
      0  Context slice + GT overlay (blue) + sampled patch positions (lime / grid lines)
      1  Target slice + GT overlay (red) + sampled patch positions (cyan / grid lines)
      2  Downsampled GT mask at this level's resolution
      3  Prediction at this level (L0: dense 8³; L1+: sparse NP patches)
      4  Fused prediction at grid resolution
      5  Fused prediction upsampled to full image resolution, overlaid on target image

    Each level dict must contain:
      res        : (D', H', W') grid resolution
      gt_ds      : (D', H', W') GT downsampled to res
      pred       : (D', H', W') prediction at this level (zeros outside sampled pos for L1+)
      pred_fused : (D_f, H_f, W_f) cumulative fused prediction
      tgt_idx    : np.ndarray (NP,) flat patch indices, or None for dense
      ctx_idx    : np.ndarray (NP,) flat patch indices, or None for dense
    """
    n_levels = len(levels)
    fig, axes = plt.subplots(n_levels, 6, figsize=(21, 3.8 * n_levels),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.25})
    if n_levels == 1:
        axes = axes[np.newaxis, :]

    tgt_z = _best_slice(tgt_gt.astype(float))
    ctx_z = _best_slice(ctx_gt.astype(float))
    tgt_sl_norm = (tgt_image[tgt_z] - tgt_image[tgt_z].min()) / (
        tgt_image[tgt_z].max() - tgt_image[tgt_z].min() + 1e-6
    )

    for row, lvl in enumerate(levels):
        res        = lvl["res"]
        gt_ds      = lvl["gt_ds"]
        pred       = lvl["pred"]
        pred_fused = lvl["pred_fused"]
        tgt_idx    = lvl.get("tgt_idx")
        ctx_idx    = lvl.get("ctx_idx")

        z_ds    = int(tgt_z * res[0]              / tgt_gt.shape[0])
        z_fused = int(tgt_z * pred_fused.shape[0] / tgt_gt.shape[0])

        _patch_overlay(axes[row, 0], ctx_image, ctx_gt, ctx_z,
                       res, ctx_idx, color="lime",
                       title=f"L{row} context  {res}", mask_cmap="Blues")
        _patch_overlay(axes[row, 1], tgt_image, tgt_gt, tgt_z,
                       res, tgt_idx, color="cyan",
                       title=f"L{row} target  {res}")
        _heatmap(axes[row, 2], gt_ds,      z_ds,    f"GT ↓ L{row}")
        _heatmap(axes[row, 3], pred,       z_ds,    f"pred L{row}")
        _heatmap(axes[row, 4], pred_fused, z_fused, f"fused L{row}")

        pf_up = F.interpolate(
            torch.from_numpy(pred_fused).float().reshape(1, 1, *pred_fused.shape),
            size=tgt_image.shape, mode="trilinear", align_corners=False,
        ).squeeze().numpy()
        axes[row, 5].imshow(tgt_sl_norm, cmap="gray")
        axes[row, 5].imshow(pf_up[tgt_z], cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        axes[row, 5].set_title(f"fused↑full L{row}", fontsize=7)
        axes[row, 5].axis("off")

    fig.suptitle(title, fontsize=9)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


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
    """Save a multilevel pred figure for one training item (synth or real)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_name = batch["label_names"][b]
    label_b    = batch["label"][b:b + 1]

    tgt_np    = batch["image"][b].squeeze().cpu().numpy()
    tgt_gt_np = batch["label"][b].cpu().numpy()
    ctx_np    = batch["context_in"][b, 0].squeeze(0).cpu().numpy()
    ctx_gt_np = batch["context_out"][b, 0].cpu().numpy()

    levels_data = []
    for i, res in enumerate(resolutions):
        D_, H_, W_ = res
        N_i      = D_ * H_ * W_
        gt_ds_np = _downsample_mask(label_b, res, cfg.data.mask_pool).squeeze().cpu().numpy().reshape(D_, H_, W_)
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
        title=f"[ep {epoch}] train  {label_name}",
    )
    return fig_path

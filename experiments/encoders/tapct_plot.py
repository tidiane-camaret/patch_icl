"""QC plots for the tap-ct transfer benchmark: per-sample target/context/pred overlays.

Everything is drawn in the LPS axial-first frame (axis0 = axial), so the upsampled
feature-grid prediction aligns with the reoriented CT and masks. One row per sample:
  [target + GT] [context + GT] [target + transfer-pred]
titled with task, subject ids, spacing, and metrics.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from feature_sim.metrics import l2norm  # noqa: E402
from tapct_features import ras_to_lps_axial_first  # noqa: E402


def transfer_pred_grid(tf, cf, cl, grid_dims):
    """1-NN transfer prediction per target cell (occupancy of nearest context cell),
    reshaped to grid_dims (gd,gh,gw). Mirrors label_transfer's hard argmax."""
    nn = (l2norm(tf.float()) @ l2norm(cf.float()).T).argmax(1)
    return cl[nn].reshape(grid_dims)


def pred_to_image(pred_grid, size):
    """Trilinear-upsample a (gd,gh,gw) soft prediction to full res `size` (D,H,W)."""
    return F.interpolate(pred_grid[None, None].float(), size=size,
                         mode="trilinear", align_corners=False)[0, 0].cpu().numpy()


def _reorient(vol):
    """(1,D,H,W) or (D,H,W) tensor -> LPS axial-first (D,H,W) numpy (matches features)."""
    a = vol.squeeze().cpu().numpy()
    return ras_to_lps_axial_first(a)


def _best_slice(mask3d):
    """axis0 (axial) index with the most foreground; 0 if the mask is empty."""
    areas = mask3d.reshape(mask3d.shape[0], -1).sum(1)
    return int(areas.argmax()) if areas.max() > 0 else mask3d.shape[0] // 2


def _panel(ax, img2d, mask2d, color, title, vlim):
    ax.imshow(img2d, cmap="gray", vmin=vlim[0], vmax=vlim[1])
    m = np.ma.masked_where(mask2d <= 0, mask2d)
    ax.imshow(m, cmap=matplotlib.colors.ListedColormap([color]), alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _panel_soft(ax, img2d, soft2d, title, vlim):
    """Continuous soft-prediction overlay: jet heatmap with per-pixel alpha = value."""
    ax.imshow(img2d, cmap="gray", vmin=vlim[0], vmax=vlim[1])
    rgba = matplotlib.cm.jet(np.clip(soft2d, 0, 1))
    rgba[..., 3] = np.clip(soft2d, 0, 1) * 0.75          # transparent where pred ~0
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_sample(path, cls, tgt_subj, ctx_subj, spacing, metrics,
                tgt_img, tgt_gt, ctx_img, ctx_gt, pred_full, thr=0.5):
    """Save a 1x3 QC figure. All volumes are reoriented (D,H,W) numpy in [features] frame."""
    ts, cs = _best_slice(tgt_gt), _best_slice(ctx_gt)
    tvl = np.percentile(tgt_img, [1, 99])
    cvl = np.percentile(ctx_img, [1, 99])

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
    _panel(axs[0], tgt_img[ts], tgt_gt[ts], "lime", f"target {tgt_subj}  +GT  (z={ts})", tvl)
    _panel(axs[1], ctx_img[cs], ctx_gt[cs], "cyan", f"context {ctx_subj}  +GT  (z={cs})", cvl)
    _panel(axs[2], tgt_img[ts], (pred_full[ts] >= thr).astype(float), "red",
           f"target  +pred (hard, thr={thr})  (z={ts})", tvl)
    _panel_soft(axs[3], tgt_img[ts], pred_full[ts], f"target  +pred (soft)  (z={ts})", tvl)

    fig.suptitle(
        f"{cls}   |   tgt {tgt_subj}  vs  ctx {ctx_subj}   |   spacing {spacing:.2f} mm   |   "
        f"soft: d {metrics['soft_dice']:.3f} p {metrics['soft_precision']:.3f} "
        f"r {metrics['soft_recall']:.3f}   |   hard: d {metrics['hard_dice']:.3f}   |   "
        f"r@1 {metrics['retrieval_at1']:.3f}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

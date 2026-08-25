"""
Shared 3D in-context eval loop — the single source of truth used by
experiments/3d/eval.py (now) and experiments/3d/train.py's val step (later).

`evaluate_classes(model, cfg, classes)` runs ONE multi-class loader through
`model.predict()` and groups results back per class, returning per-class summary
rows + per-case records (plus optional qualitative figures). `validate(model,
loader, cls)` remains for a single-class loader. Mirrors experiments/2d/evaluate.py.

Ported from scripts/eval.py so the 3D experiments harness is self-contained;
scripts/eval.py stays as the legacy CLI benchmark.
"""

import contextlib
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

from grid_metrics import hard_sum, soft_sum, cos_sum


# ---------------------------------------------------------------------------
# Metrics + figures
# ---------------------------------------------------------------------------

def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Smooth Dice between two binary tensors of any shape."""
    pred, target = pred.bool(), target.bool()
    inter = (pred & target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return (2 * inter + 1) / (union + 1)


def dice_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batched smooth Dice on `pred.device`. pred, target: (B,D,H,W) binary -> (B,).

    Same smooth formula as dice_binary ((2*inter+1)/(union+1)); inter/union are integer
    voxel counts so the result is bit-identical to the per-sample CPU path once rounded.
    """
    p = (pred > 0).flatten(1).float()
    g = (target > 0).flatten(1).float()
    inter = (p * g).sum(1)
    union = p.sum(1) + g.sum(1)
    return (2 * inter + 1) / (union + 1)


def soft_dice_binary(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Threshold-free Dice between a soft probability map and a binary target (any shape).

    Matches the training soft-Dice term (train.py:_soft_dice, eps=1e-6): the shape/overlap
    signal before the 0.5 threshold used by dice_binary."""
    p = prob.flatten().float()
    g = (target.flatten() > 0).float()
    inter = (p * g).sum().item()
    den = p.sum().item() + g.sum().item()
    return (2 * inter + eps) / (den + eps)


def _surface_voxels(x: torch.Tensor) -> torch.Tensor:
    """Boundary of a binary volume: foreground voxels touching background (6-conn).

    x: (B,1,D,H,W) float {0,1} -> same shape {0,1}. Erosion keeps a voxel only if it and its
    6 face-neighbours are all foreground (out-of-bounds padded with 0, so border voxels erode);
    x - erosion is the surface. 6-connectivity + zero border matches scipy's default
    binary_erosion, i.e. MONAI's get_mask_edges — validated to give NSD identical to
    monai.metrics.compute_surface_dice.
    """
    xp = F.pad(x, (1, 1, 1, 1, 1, 1), value=0.0)
    ero = xp[..., 1:-1, 1:-1, 1:-1]
    for sl in (
        (..., slice(1, -1), slice(1, -1), slice(0, -2)),   # -x / +x
        (..., slice(1, -1), slice(1, -1), slice(2, None)),
        (..., slice(1, -1), slice(0, -2), slice(1, -1)),   # -y / +y
        (..., slice(1, -1), slice(2, None), slice(1, -1)),
        (..., slice(0, -2), slice(1, -1), slice(1, -1)),   # -z / +z
        (..., slice(2, None), slice(1, -1), slice(1, -1)),
    ):
        ero = torch.minimum(ero, xp[sl])
    return x - ero


def _ball_kernel(tol_mm: float, spacing, device, dtype):
    """Boolean ball of physical radius `tol_mm` on a grid with per-axis `spacing` (mm/vox).

    Returns (kernel (1,1,kz,ky,kx), radii (rz,ry,rx)). A voxel offset is in the ball iff its
    physical displacement is <= tol_mm, so convolving a surface with this kernel and testing
    >0 answers "is any surface voxel within tol_mm?" — a thresholded distance transform
    restricted to the ball neighbourhood (no full EDT needed for a fixed tolerance).
    """
    r = [max(1, int(tol_mm / float(s) + 0.999)) for s in spacing]
    axes = [torch.arange(-ri, ri + 1, device=device, dtype=torch.float32) for ri in r]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    d2 = ((zz * float(spacing[0])) ** 2 + (yy * float(spacing[1])) ** 2
          + (xx * float(spacing[2])) ** 2)
    return (d2 <= tol_mm ** 2).to(dtype)[None, None], tuple(r)


def nsd_batch(pred: torch.Tensor, target: torch.Tensor, spacing, tol_mm: float) -> torch.Tensor:
    """Normalized Surface Dice at tolerance `tol_mm` (voxel-count / MONAI convention).

    Fully batched on `pred.device`: extracts both surfaces (6-conn erosion), dilates each by a
    physical ball of radius tol_mm (a small fixed conv, not a full distance transform), and
    scores the fraction of each surface lying within tolerance of the other. Both surfaces
    empty -> 1.0; exactly one empty -> 0.0 (falls out of the formula: an empty surface dilates
    to nothing, so the non-empty side scores 0 while its own count fills the denominator).

    This is the voxel-count NSD, validated bit-for-bit against monai.metrics.
    compute_surface_dice (not the marching-cubes area-weighted DeepMind variant).
    pred, target: (B,D,H,W) binary. spacing: (3,) mm/voxel. -> (B,) NSD in [0,1].
    """
    p = (pred > 0)[:, None].float()
    g = (target > 0)[:, None].float()
    sp, sg = _surface_voxels(p), _surface_voxels(g)
    k, r = _ball_kernel(tol_mm, spacing, p.device, p.dtype)
    dil = lambda s: (F.conv3d(s, k, padding=r) > 0).float()  # noqa: E731
    gd, pd = dil(sg), dil(sp)
    num = (sp * gd).flatten(1).sum(1) + (sg * pd).flatten(1).sum(1)
    den = sp.flatten(1).sum(1) + sg.flatten(1).sum(1)
    nsd = num / den.clamp_min(1e-8)
    nsd[den == 0] = 1.0
    return nsd


def _best_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def save_eval_figure(target_img, gt, pred, ctx_img, ctx_gt, out_path: Path, title: str = "") -> None:
    """Save a 4-panel figure: context | target | GT overlay | pred overlay."""
    def _norm(sl):
        mn, mx = sl.min(), sl.max()
        return (sl - mn) / (mx - mn + 1e-6)

    z, z_ctx = _best_slice(gt), _best_slice(ctx_gt)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"wspace": 0.04})
    axes[0].imshow(_norm(ctx_img[z_ctx]), cmap="gray")
    axes[0].imshow(ctx_gt[z_ctx].astype(float), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[0].set_title("context", fontsize=8)
    axes[1].imshow(_norm(target_img[z]), cmap="gray")
    axes[1].set_title("target", fontsize=8)
    axes[2].imshow(_norm(target_img[z]), cmap="gray")
    axes[2].imshow(gt[z].astype(float), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("GT", fontsize=8)
    axes[3].imshow(_norm(target_img[z]), cmap="gray")
    axes[3].imshow(pred[z].astype(float), cmap="Blues", alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("pred", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _norm2d(sl):
    mn, mx = float(sl.min()), float(sl.max())
    return (sl - mn) / (mx - mn + 1e-6)


def _overlay(ax, base2d, layers, box_specs=()):
    """Grayscale `base2d` with colored foreground `layers` and optional bboxes.

    layers    : list of (mask2d | None, color, alpha) — only foreground (mask>0.5) is tinted.
    box_specs : list of (cx, cy, w, h, color, label) rectangles in imshow (x=col, y=row) coords.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    ax.imshow(_norm2d(base2d), cmap="gray")
    for m, color, a in layers:
        if m is None:
            continue
        sel = np.asarray(m) > 0.5
        rgba = np.zeros((*sel.shape, 4), dtype=float)
        rgba[sel] = mcolors.to_rgba(color)
        rgba[..., 3] = np.where(sel, a, 0.0)
        ax.imshow(rgba)
    for cx, cy, w, h, color, label in box_specs:
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                               edgecolor=color, lw=1.5, label=label))
    ax.axis("off")


def _box_inplane(center, size):
    """(center(3,), size(3,)) voxel box -> (cx, cy, w, h) for an axis-0 slice (x=W, y=H)."""
    return float(center[2]), float(center[1]), float(size[2]), float(size[1])


def save_cascade_figure(out_path: Path, *,
                        tgt_coarse, gt_coarse, pred_coarse, ctx_coarse, ctx_gt_coarse,
                        tgt_fine, gt_fine, pred_fine, ctx_fine, ctx_gt_fine,
                        refit_pred_coarse, refit_gt_coarse=None,
                        fine_box=None, oracle_box=None,
                        spacings=(None, None), title="") -> None:
    """Save the 2x5 coarse->fine cascade panel (top row target, bottom row 1st context).

    All volumes are (D,H,W); a representative axis-0 slice is picked per frame. Columns:
      1. coarse (s0) img + GT overlay
      2. coarse img + coarse target pred + fine/oracle bboxes (from pred/GT centroids)
      3. fine (s1) img + GT overlay
      4. fine img + fine target pred
      5. coarse img + fine pred REFITTED back into the coarse frame + coarse GT
    Boxes are (center(3,), size(3,)) in coarse-grid voxels. GT=lime, pred=red,
    fine box=yellow, oracle box=cyan. The context row's cols 2/4 repeat the prompt (GT),
    col 5 is blank (target-only refit)."""
    s0, s1 = spacings
    GT, PR = "lime", "red"
    zc  = _best_slice(np.asarray(gt_coarse))
    zf  = _best_slice(np.asarray(gt_fine))
    zcc = _best_slice(np.asarray(ctx_gt_coarse))
    zcf = _best_slice(np.asarray(ctx_gt_fine))

    boxes = []
    if fine_box is not None:
        boxes.append((*_box_inplane(*fine_box), "yellow", "fine box"))
    if oracle_box is not None:
        boxes.append((*_box_inplane(*oracle_box), "cyan", "oracle box"))

    fig, ax = plt.subplots(2, 5, figsize=(20, 8),
                           gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    # ── target row ───────────────────────────────────────────────────────────
    _overlay(ax[0, 0], np.asarray(tgt_coarse)[zc], [(np.asarray(gt_coarse)[zc], GT, 0.45)])
    _overlay(ax[0, 1], np.asarray(tgt_coarse)[zc], [(np.asarray(pred_coarse)[zc], PR, 0.45)],
             box_specs=boxes)
    _overlay(ax[0, 2], np.asarray(tgt_fine)[zf],   [(np.asarray(gt_fine)[zf], GT, 0.45)])
    _overlay(ax[0, 3], np.asarray(tgt_fine)[zf],   [(np.asarray(pred_fine)[zf], PR, 0.45)])
    _overlay(ax[0, 4], np.asarray(tgt_coarse)[zc],
             [(refit_gt_coarse[zc] if refit_gt_coarse is not None else np.asarray(gt_coarse)[zc], GT, 0.4),
              (np.asarray(refit_pred_coarse)[zc], PR, 0.5)])
    col_titles = [
        f"1. coarse{f' @{s0:g}mm' if s0 else ''}: img+GT",
        "2. coarse: pred + fine/oracle bbox",
        f"3. fine{f' @{s1:g}mm' if s1 else ''}: img+GT",
        "4. fine: pred",
        "5. coarse: refit fine pred + GT",
    ]
    for j, t in enumerate(col_titles):
        ax[0, j].set_title(t, fontsize=8)
    # ── 1st-context row (prompt) ───────────────────────────────────────────────
    _overlay(ax[1, 0], np.asarray(ctx_coarse)[zcc], [(np.asarray(ctx_gt_coarse)[zcc], GT, 0.45)])
    _overlay(ax[1, 1], np.asarray(ctx_coarse)[zcc], [(np.asarray(ctx_gt_coarse)[zcc], GT, 0.45)])
    _overlay(ax[1, 2], np.asarray(ctx_fine)[zcf],   [(np.asarray(ctx_gt_fine)[zcf], GT, 0.45)])
    _overlay(ax[1, 3], np.asarray(ctx_fine)[zcf],   [(np.asarray(ctx_gt_fine)[zcf], GT, 0.45)])
    ax[1, 4].axis("off")
    for j, lab in enumerate(["ctx (coarse)", "ctx (coarse)", "ctx (fine)", "ctx (fine)", ""]):
        if lab:
            ax[1, j].set_title(lab, fontsize=7)
    if boxes:
        ax[0, 1].legend(loc="lower right", fontsize=6, framealpha=0.6)
    fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _sample_detail(meta: dict | None) -> str:
    """One compact per-sample string for the sample table's `detail` column, adapting
    to the data source (mirrors experiments/2d/evaluate.py:_sample_detail). omniSynth3D
    meta -> "mode=<m> class=<id> sub=<i>"; anything else / missing -> "". Keeps the
    table's columns fixed across sources (totalseg items carry no meta)."""
    if not meta:
        return ""
    if "class_id" in meta:  # omniSynth3D
        return (f"mode={meta.get('target_mode', '')} "
                f"class={meta.get('class_id', '')} "
                f"sub={meta.get('sample_index', -1)}")
    return ""


# The sample-table columns are fixed (medverse is a native-resolution model, so there is
# no coarse-grid / refine family like 2D's patchset_cnn). One row per eval case, carrying
# the per-case Dice + GT/context occupancy stats + the source-adaptive `detail` string.
_SAMPLE_TABLE_COLS = ["epoch", "class", "in_train", "subject", "ctx_cases", "self_ctx",
                      "dice", "nsd", "soft_dice", "loss", "time_ms", "tgt_size", "tgt_occ",
                      "ctx_size", "ctx_occ", "spacing", "synth_size", "detail"]


def build_sample_table(cases: list[dict], epoch: int | None = None, train_classes=None):
    """Build a wandb.Table of per-case detail from `evaluate_classes` records.

    Shared by experiments/3d/eval.py (benchmark) and train.py's val step so both log the
    same schema. `epoch` tags the training epoch (-1 for standalone eval). Cases must be
    the enriched dicts emitted by evaluate_classes (with tgt_size/ctx_occ/detail/... keys).
    `train_classes` (set of class names seen in training) fills the `in_train` column.
    """
    import wandb
    ep = -1 if epoch is None else int(epoch)
    train_set = set(train_classes) if train_classes is not None else None
    # Optional per-layer feature-sim columns (train.py attaches fs_<rep>_<dice|retr> onto a
    # subsample of cases); absent for standalone eval, so the schema stays backward-compatible.
    fs_cols = sorted({k for c in cases for k in c if k.startswith("fs_")})
    table = wandb.Table(columns=_SAMPLE_TABLE_COLS + fs_cols)
    for c in cases:
        in_train = c["class"] in train_set if train_set is not None else None
        table.add_data(ep, c["class"], in_train, c["subject"],
                       c.get("ctx_cases", ""), c.get("self_ctx", None), c["dice"],
                       c.get("nsd", float("nan")),
                       c.get("soft_dice", float("nan")), c.get("loss", float("nan")),
                       c.get("time_ms", float("nan")),
                       c.get("tgt_size", float("nan")), c.get("tgt_occ", float("nan")),
                       c.get("ctx_size", float("nan")), c.get("ctx_occ", float("nan")),
                       c.get("spacing", float("nan")),
                       c.get("synth_size", float("nan")), c.get("detail", ""),
                       *[c.get(k, float("nan")) for k in fs_cols])
    return table


def _occupancy_stats(label_i: torch.Tensor, ctx_masks_i: torch.Tensor) -> dict:
    """GT + context foreground stats for one sample (model-independent).

    label_i: (D,H,W) target GT. ctx_masks_i: (K,D,H,W) context masks. size = foreground
    voxels; occ = foreground fraction. Context stats are averaged over the K contexts.
    """
    tgt_fg = label_i > 0
    ctx_fg = ctx_masks_i > 0
    K = max(int(ctx_masks_i.shape[0]), 1)
    return {
        "tgt_size": float(tgt_fg.sum()),
        "tgt_occ":  round(float(tgt_fg.float().mean()), 6),
        "ctx_size": float(ctx_fg.float().sum()) / K,   # mean fg voxels per context
        "ctx_occ":  round(float(ctx_fg.float().mean()), 6),
    }


def measure_flops(model, image_size: tuple, K: int, device: torch.device) -> dict:
    """GFLOPs for one predict() call with a single-sample dummy input.

    Returns {"total", "encoder", "transformer"} in GFLOPs. FlopCounterMode keys its
    per-module breakdown by class name (each top-level key aggregates its subtree), so
    the encoder / transformer shares come from the submodule class names; the small
    img/mask embeds + decoder fall outside both. encoder/transformer are None for models
    lacking those submodules (e.g. medverse). All-zero total on failure.
    """
    D, H, W = image_size
    dummy_target  = torch.zeros(1, 1, D, H, W, device=device)
    dummy_ctx_img = torch.zeros(1, K, 1, D, H, W, device=device)
    dummy_ctx_msk = torch.zeros(1, K, D, H, W, dtype=torch.long, device=device)
    # FlopCounterMode disables dynamo, so patchset3d's register_routed flex kernel can't run
    # under it (uncountable eager HOP + a spurious "flex called without compile" warning). Force
    # the dense bool-mask equivalent for the count (traceable); it overstates register_routed's
    # true cost — the sparse figure is in experiments/3d/bench_attn_pattern.py. No-op otherwise.
    import src.models.pfn_seg_2d as _pfn
    _flex_prev, _pfn._FLEX_ENABLED = _pfn._FLEX_ENABLED, False
    try:
        with FlopCounterMode(display=False) as fc:
            model.predict(dummy_target, dummy_ctx_img, dummy_ctx_msk)
        counts = fc.get_flop_counts()

        def _share(attr):
            sub = getattr(model, attr, None)
            if sub is None:
                return None
            c = counts.get(type(sub).__name__)
            return sum(c.values()) / 1e9 if c else None

        return {"total": fc.get_total_flops() / 1e9,
                "encoder": _share("encoder"), "transformer": _share("transformer")}
    except Exception as exc:  # noqa: BLE001
        print(f"    [FLOPs] Could not count: {exc}")
        return {"total": 0.0, "encoder": None, "transformer": None}
    finally:
        _pfn._FLEX_ENABLED = _flex_prev


# ---------------------------------------------------------------------------
# Per-class eval loop
# ---------------------------------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _eval_autocast(enabled: bool):
    """bf16 CUDA autocast when enabled (else a no-op). Matches training's autocast dtype so
    a compiled encoder/transformer isn't recompiled between the train (bf16) and eval paths."""
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def validate(model, loader, cls: str, *, fig_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """Run inference over one single-class loader.

    Returns (summary_row, cases): summary_row aggregates mean/std Dice + mean time;
    cases is a list of {class, subject, dice, time_ms}. Saves one figure per class
    to fig_dir (first batch) when provided. Uses model.predict().
    """
    cases: list[dict] = []
    fig_saved = False

    for batch in tqdm(loader, desc="eval", leave=False):
        target_img    = batch["image"]
        context_imgs  = batch["context_in"]
        context_masks = batch["context_out"]
        label         = batch["label"]
        subjects      = batch.get("subjects", [None] * target_img.shape[0])

        _sync()
        t0 = time.perf_counter()
        pred = model.predict(target_img, context_imgs, context_masks)
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred, label = pred.cpu(), label.cpu()

        if fig_dir is not None and not fig_saved:
            subj = subjects[0] or "s0"
            save_eval_figure(
                target_img=target_img[0].squeeze(0).numpy(),
                gt=label[0].numpy(),
                pred=pred[0].numpy(),
                ctx_img=context_imgs[0, 0].squeeze(0).numpy(),
                ctx_gt=context_masks[0, 0].numpy(),
                out_path=fig_dir / f"{cls}_{subj}.png",
                title=f"{cls}  {subj}  dice={dice_binary(pred[0], label[0]):.3f}",
            )
            fig_saved = True

        for i in range(pred.shape[0]):
            cases.append({
                "class":   cls,
                "subject": subjects[i],
                "dice":    round(dice_binary(pred[i], label[i]), 4),
                "time_ms": round(elapsed_ms / pred.shape[0], 1),
            })

    return _summarize(cls, cases), cases


def _summarize(cls: str, cases: list[dict]) -> dict:
    """Aggregate per-case records into a summary row (mean/std Dice + mean time)."""
    dice_scores = [c["dice"] for c in cases]
    times = [c["time_ms"] for c in cases]
    n = len(dice_scores)
    mean_dice = sum(dice_scores) / n if n else 0.0
    std_dice  = (sum((d - mean_dice) ** 2 for d in dice_scores) / n) ** 0.5 if n > 1 else 0.0
    mean_ms   = sum(times) / len(times) if times else 0.0
    row = {
        "class":        cls,
        "n_samples":    n,
        "mean_dice":    round(mean_dice, 4),
        "std_dice":     round(std_dice, 4),
        "mean_time_ms": round(mean_ms, 1),
    }
    # NSD is present only when an nsd_tolerance_mm was configured (eval.py benchmark path).
    nsd_scores = [c["nsd"] for c in cases if "nsd" in c]
    if nsd_scores:
        m = sum(nsd_scores) / len(nsd_scores)
        row["mean_nsd"] = round(m, 4)
        row["std_nsd"] = round(
            (sum((x - m) ** 2 for x in nsd_scores) / len(nsd_scores)) ** 0.5, 4
        ) if len(nsd_scores) > 1 else 0.0
    # Soft-Dice / loss are only present when evaluate_classes gets a logits_fn/loss_fn
    # (train.py's val step); absent for the eval.py benchmark path.
    soft = [c["soft_dice"] for c in cases if "soft_dice" in c]
    if soft:
        row["mean_soft_dice"] = round(sum(soft) / len(soft), 4)
    losses = [c["loss"] for c in cases if "loss" in c]
    if losses:
        row["mean_loss"] = round(sum(losses) / len(losses), 4)
    for key in ("dice_ds", "dice_ds_soft", "cossim"):
        vals = [c[key] for c in cases if key in c]
        if vals:
            row[f"mean_{key}"] = round(sum(vals) / len(vals), 4)
    # Locator containment (only when evaluate_classes ran with locator_ratio set).
    if any("containment" in c for c in cases):
        cont = [c["containment"] for c in cases
                if "containment" in c and not np.isnan(c["containment"])]
        orc = [c["containment_oracle"] for c in cases
               if "containment_oracle" in c and not np.isnan(c["containment_oracle"])]
        err = [c["loc_err_mm"] for c in cases
               if "loc_err_mm" in c and not np.isnan(c["loc_err_mm"])]
        row["n_locator"] = len(cont)
        row["n_locator_empty"] = sum(1 for c in cases if c.get("locator_empty"))
        row["mean_containment"] = round(sum(cont) / len(cont), 4) if cont else float("nan")
        row["mean_containment_oracle"] = round(sum(orc) / len(orc), 4) if orc else float("nan")
        row["mean_loc_err_mm"] = round(sum(err) / len(err), 2) if err else float("nan")
    return row


def _locator_containment(prob, label, ratio):
    """Coarse->fine locator containment for one sample (pure geometry).

    prob : (D,H,W) tensor — soft probability or 0/1 hard mask (locator weights).
    label: (D,H,W) tensor — GT; foreground = label > 0.
    ratio: s_fine / s_coarse in (0,1). Box side per axis = max(1, round(T_a*ratio)).

    Locator center = prob-weighted centroid over ALL voxels; sum(prob) < 1e-6 -> crop
    center + locator_empty=True. The fine box (that side, clamped inside the volume) is
    placed at the locator center; the oracle box is placed at the GT-foreground centroid.
    Returns (containment, containment_oracle, locator_empty, loc_err_vox):
      containment        = |GT_fg ∩ box|        / |GT_fg|   (NaN if no GT foreground)
      containment_oracle = |GT_fg ∩ box_oracle| / |GT_fg|   (NaN if no GT foreground)
      locator_empty      = bool
      loc_err_vox        = ||center - gt_centroid|| in voxels (NaN if no GT foreground).
                           The caller scales by the coarse spacing to get loc_err_mm.
    """
    p = prob.detach().float().cpu().numpy()
    gt = (label.detach().cpu().numpy() > 0)
    T = p.shape                                    # (D, H, W)
    box = [max(1, int(round(t * ratio))) for t in T]
    idx = np.indices(T, dtype=float)               # (3, D, H, W)

    def _frac_in_box(center):
        total = float(gt.sum())
        lo = []
        for a in range(3):
            l = int(round(center[a] - box[a] / 2))
            l = max(0, min(l, T[a] - box[a]))       # clamp so the box fits in [0, T_a]
            lo.append(l)
        sub = gt[lo[0]:lo[0] + box[0], lo[1]:lo[1] + box[1], lo[2]:lo[2] + box[2]]
        return float(sub.sum()) / total

    # Locator center: prob-weighted centroid over all voxels; empty -> crop center.
    s = float(p.sum())
    if s < 1e-6:
        center = np.array([t / 2.0 for t in T])
        locator_empty = True
    else:
        center = np.array([(idx[a] * p).sum() / s for a in range(3)])
        locator_empty = False

    gt_n = float(gt.sum())
    if gt_n == 0.0:
        return float("nan"), float("nan"), locator_empty, float("nan")

    gt_centroid = np.array([(idx[a] * gt).sum() / gt_n for a in range(3)])
    containment = _frac_in_box(center)
    containment_oracle = _frac_in_box(gt_centroid)
    loc_err_vox = float(np.linalg.norm(center - gt_centroid))
    return containment, containment_oracle, locator_empty, loc_err_vox


def _predicted_native_center(prob, geom):
    """Invert a grid-space prediction back to a native crop centre for the cascade.

    prob : (D,H,W) soft probability in the coarse crop's T³ grid.
    geom : (4,3) long tensor — the crop's [starts, crop_sizes, out_sizes, pad_lo]
           (from TotalSegInContextDataset._organ_crop_arrays).
    Returns the prob-weighted centroid mapped to native voxels (d,h,w), or the string
    "volume_center" when the prediction is empty (caller crops on the volume centre).
    """
    p = prob.detach().float().cpu().numpy()
    s = float(p.sum())
    if s < 1e-6:
        return "volume_center"
    T = p.shape
    idx = np.indices(T, dtype=float)
    g = [(idx[a] * p).sum() / s for a in range(3)]                    # grid centroid
    starts, crop_sizes, out_sizes, pad_lo = (geom[r].tolist() for r in range(4))
    native = [int(round(starts[a] + (g[a] - pad_lo[a]) / max(1, out_sizes[a]) * crop_sizes[a]))
              for a in range(3)]
    return tuple(max(0, c) for c in native)


def _grid_centroid(arr):
    """Weighted centroid of a (D,H,W) array over its own grid; None if empty."""
    a = np.asarray(arr, dtype=float)
    s = float(a.sum())
    if s < 1e-6:
        return None
    idx = np.indices(a.shape, dtype=float)
    return np.array([(idx[k] * a).sum() / s for k in range(3)])


def evaluate_classes(model, cfg, classes, *, split=None, fig_dir: Path | None = None,
                     loader=None, logits_fn=None, loss_fn=None, grid_res=None,
                     output_is_prob=False, autocast=False, reuse_logits=False,
                     locator_ratio: float | None = None, pred_centers_out: dict | None = None,
                     figure_cache: dict | None = None, figure_classes=None,
                     pred_geom_out: dict | None = None, drop_self_ctx: bool = False):
    """Eval all `classes` through ONE multi-class loader; return (rows, cases).

    Builds a single dataset over every class (via common.make_eval_loader), so the
    scan/bbox caches load once rather than once per class. Each sample carries its
    own `label_name`, so results are grouped back per class after inference —
    yielding the same (rows, cases) shape as the old per-class loop. Classes with
    no samples get an "error" row. split defaults to cfg.eval.split.

    Pass a prebuilt `loader` (from common.make_eval_loader) to reuse one dataset
    across repeated calls — train.py's val step does this so the dataset isn't
    rebuilt (and caches reloaded) every eval epoch.

    `logits_fn(target, ctx_in, ctx_out) -> (B,1,D,H,W) raw logits` enables the soft
    monitoring metrics: when given, each case also gets `soft_dice` (threshold-free
    overlap of σ(logits) vs GT) and, if `loss_fn(logits, target)` is also given, a
    per-sample `loss`. By default the hard `dice` comes from model.predict (the benchmark
    inference). eval.py passes none of these + leaves autocast/reuse_logits off, so its path
    is byte-identical.

    `drop_self_ctx=True` excludes self-context samples (all K contexts == the target case)
    from the per-class summary — cross-subject-only eval. This is not just the intentional
    self_context probe: even with self_context.p.eval=0 the context sampler falls back to
    cloning the target when a class has no cross-subject candidate (a leakage-inflated
    sample it warns about), so this guard keeps those out of the reported mean. The dropped
    cases still appear in the per-sample `cases`/table (flagged self_ctx=True), only the
    aggregate rows exclude them.

    `autocast=True` runs the eval forward(s) under bf16 (matches training; ~4x faster cold
    encode and no compile recompile between train/eval dtypes). `reuse_logits=True` (requires
    logits_fn) derives the hard prediction from the SAME native logits used for the soft
    metrics — one forward instead of predict + a second logits_fn pass. Both are opt-in and
    used only by train.py's val step for patchset3d, where predict == threshold(train_forward).

    Shared by experiments/3d/eval.py (benchmark) and train.py's val step.
    """
    from collections import defaultdict

    if loader is None:
        from common import make_eval_loader  # local import: common/evaluate are siblings
        split = split or cfg.eval.split
        loader = make_eval_loader(cfg, classes, split=split)

    # NSD tolerance (mm). Read from the eval config; absent (e.g. train.py's val cfg has no
    # nsd_tolerance_mm) -> NSD is skipped and cases carry Dice only. None disables it.
    _ev = cfg.get("eval")
    nsd_tol = _ev.get("nsd_tolerance_mm") if _ev is not None else None

    # Each case dict carries the columns for build_sample_table: class, subject, dice,
    # time_ms, detail (source-adaptive), + tgt_size/tgt_occ/ctx_size/ctx_occ occupancy stats.
    cases_by_class: dict[str, list[dict]] = defaultdict(list)
    figs_saved: set[str] = set()

    for batch in tqdm(loader, desc="eval", leave=False):
        target_img    = batch["image"]
        context_imgs  = batch["context_in"]
        context_masks = batch["context_out"]
        label         = batch["label"]
        subjects      = batch.get("subjects", [None] * target_img.shape[0])
        ctx_subjects  = batch.get("context_subjects")   # (B) list[list[str]] or None
        label_names   = batch["label_names"]
        metas         = batch.get("meta")

        # Hard prediction. With reuse_logits (+ logits_fn), derive it from the SAME native
        # logits the soft metrics use — one forward — instead of a separate model.predict pass
        # (predict == threshold(train_forward) for patchset3d / single-ROI medverse). Default
        # path (reuse_logits=False, e.g. eval.py) is unchanged: predict is the timed inference.
        # Per-batch physical spacing for the spacing-aware frozen encoder (eval spacing is
        # fixed, so batch[0] represents the whole batch). Only forwarded to models that opt
        # in (PatchSet3D.spacing_aware) — medverse's predict/logits_fn take no spacing.
        sp_kw = ({"spacing": float(batch["spacing"][0, 0])}
                 if getattr(model, "spacing_aware", False) and "spacing" in batch else {})
        # Context-free models (TotalSegmentator) can't infer the target class from context, so
        # forward the per-sample class names when the model opts in (mirrors spacing_aware).
        if getattr(model, "needs_label_names", False):
            sp_kw["label_names"] = list(label_names)
        prob = None
        logits = None
        _sync()
        t0 = time.perf_counter()
        if reuse_logits and logits_fn is not None:
            with torch.no_grad(), _eval_autocast(autocast):
                logits = logits_fn(target_img, context_imgs, context_masks, **sp_kw).float()  # (B,1,D,H,W)
            pred = ((logits.clamp(0, 1) if output_is_prob else torch.sigmoid(logits)) >= 0.5
                    ).float().squeeze(1)                                              # (B,D,H,W)
        else:
            with _eval_autocast(autocast):
                pred = model.predict(target_img, context_imgs, context_masks, **sp_kw)
        _sync()
        per_sample_ms = (time.perf_counter() - t0) * 1000 / pred.shape[0]

        # Soft monitoring pass (train.py val step only): raw logits -> σ for soft Dice + the
        # training loss. Reuse the logits computed above when available, else a single-ROI
        # forward; untimed (timing stays on the hard-prediction pass above).
        if logits_fn is not None:
            if logits is None:
                with torch.no_grad(), _eval_autocast(autocast):
                    logits = logits_fn(target_img, context_imgs, context_masks, **sp_kw).float()  # (B,1,D,H,W)
            tgt = label.to(logits.device).float().unsqueeze(1)                        # (B,1,D,H,W)
            # output_is_prob (medverse): logits_fn already returns a [0,1] probability, so do
            # NOT sigmoid it again (that pins every voxel to foreground). See train.py's
            # model_output_is_prob. Clamp to [0,1] — the plain-conv head can dip slightly out of
            # range, which else drives the soft-Dice denom negative. Default False keeps eval.py's
            # logit path byte-identical.
            prob = (logits.clamp(0, 1) if output_is_prob else torch.sigmoid(logits)).cpu()
            grid_pr = grid_gt = None
            if grid_res is not None:
                grid_pr = F.adaptive_avg_pool3d(prob, (grid_res,) * 3)                 # (B,1,g,g,g)
                grid_gt = F.adaptive_avg_pool3d(label.float().unsqueeze(1).cpu(), (grid_res,) * 3)
            sample_loss = ([float(loss_fn(logits[i:i + 1], tgt[i:i + 1]).item())
                            for i in range(logits.shape[0])] if loss_fn is not None else None)

        # Overlap metrics computed batched on `pred.device`, OUTSIDE the t0 timing block above
        # (so they never inflate the reported inference time). label moves to the GPU once and
        # is shared by Dice + NSD. Spacing is fixed within a batch (crop path), so batch[0]
        # represents all samples; fall back to an isotropic crop_spacing_mm if none is emitted.
        label_dev = label.to(pred.device, non_blocking=True)
        dice_vec = dice_batch(pred, label_dev).cpu()
        nsd_vec = None
        if nsd_tol is not None:
            sp_vox = (batch["spacing"][0].tolist() if "spacing" in batch
                      else [float(cfg.data.get("crop_spacing_mm", 1.5))] * 3)
            nsd_vec = nsd_batch(pred, label_dev, sp_vox, float(nsd_tol)).cpu()

        pred, label = pred.cpu(), label.cpu()
        context_masks = context_masks.cpu()

        for i in range(pred.shape[0]):
            cls = label_names[i]
            if fig_dir is not None and cls not in figs_saved:
                subj = subjects[i] or "s0"
                save_eval_figure(
                    target_img=target_img[i].squeeze(0).cpu().numpy(),
                    gt=label[i].numpy(),
                    pred=pred[i].numpy(),
                    ctx_img=context_imgs[i, 0].squeeze(0).cpu().numpy(),
                    ctx_gt=context_masks[i, 0].cpu().numpy(),
                    out_path=fig_dir / f"{cls}_{subj}.png",
                    title=f"{cls}  {subj}  dice={float(dice_vec[i]):.3f}",
                )
                figs_saved.add(cls)
            cids = ctx_subjects[i] if ctx_subjects is not None else None
            case = {
                "class":   cls,
                "subject": subjects[i],
                # per-context case ids + self-context flag (all ctx == target case)
                "ctx_cases": ";".join(map(str, cids)) if cids else "",
                "self_ctx":  bool(cids and all(c == subjects[i] for c in cids)),
                "dice":    round(float(dice_vec[i]), 4),
                "time_ms": round(per_sample_ms, 1),
                "detail":  _sample_detail(metas[i] if metas is not None else None),
            }
            case.update(_occupancy_stats(label[i], context_masks[i]))
            if nsd_vec is not None:
                case["nsd"] = round(float(nsd_vec[i]), 4)
            # Per-sample effective spacing (mm/voxel) when the dataset reports it. Spacing is a
            # (3,) tensor; the crop path is isotropic and the spacing-aware model consumes the
            # first axis as its scalar, so log that same scalar. Absent for datasets that emit no
            # spacing (spacing key missing) -> the column stays NaN.
            if "spacing" in batch:
                case["spacing"] = round(float(batch["spacing"][i, 0]), 4)
            # Synthetic-mask samples (data.self_context.synth_masks, label_name 'synth'): write the
            # object's shape — per-axis radii (mm) + anatomical coords.npy position — into the
            # detail column. Only for synth samples; other samples keep their provenance detail.
            if "synth_radii_mm" in batch and not torch.isnan(batch["synth_radii_mm"][i, 0]).item():
                r = batch["synth_radii_mm"][i].tolist()
                shape = f"ellipse {r[0]:.0f}x{r[1]:.0f}x{r[2]:.0f}mm"
                if "synth_coord" in batch and not torch.isnan(batch["synth_coord"][i, 0]).item():
                    c = batch["synth_coord"][i].tolist()
                    shape += f" @({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})"
                case["detail"] = shape
                case["synth_size"] = round(sum(r) / 3 * 2, 1)   # mean diameter (mm), plottable
            if locator_ratio is not None:
                # Locate a fine-spacing box from the coarse prediction and measure how much
                # GT it contains. Soft prob when available (logits_fn), else the hard mask.
                lp = prob[i, 0] if prob is not None else pred[i]
                cont, cont_orc, loc_empty, loc_err_vox = _locator_containment(
                    lp, label[i], locator_ratio)
                sp_c = float(batch["spacing"][i, 0]) if "spacing" in batch else 1.0
                case["containment"] = round(float(cont), 4)              # NaN safe: round(nan)=nan
                case["containment_oracle"] = round(float(cont_orc), 4)
                case["locator_empty"] = bool(loc_empty)
                case["loc_err_mm"] = round(loc_err_vox * sp_c, 2)
            if pred_centers_out is not None and prob is not None and "crop_geom" in batch:
                # Cascade: record this coarse pass's predicted crop centre (native voxels) so
                # the next-finer pass can re-crop the target on it (see evaluate_spacing_sweep).
                pred_centers_out[(subjects[i], cls)] = _predicted_native_center(
                    prob[i, 0], batch["crop_geom"][i])
            if pred_geom_out is not None and "crop_geom" in batch:
                # Stitched-cascade metric: keep every sample's hard pred (bit-packed) + its crop
                # geometry so the coarse & fine passes can be composited into the native volume.
                pred_geom_out[(subjects[i], cls)] = (
                    np.packbits(pred[i].numpy().astype(bool)),
                    tuple(pred[i].shape), batch["crop_geom"][i].numpy())
            if prob is not None:
                case["soft_dice"] = round(soft_dice_binary(prob[i, 0], label[i]), 4)
                if sample_loss is not None:
                    case["loss"] = sample_loss[i]
                if grid_pr is not None:
                    pr, gt = grid_pr[i:i + 1], grid_gt[i:i + 1]
                    case["dice_ds"] = round(float(hard_sum(pr, gt)[0]), 4)
                    case["dice_ds_soft"] = round(float(soft_sum(pr, gt)[0]), 4)
                    case["cossim"] = round(float(cos_sum(pr, gt)[0]), 4)
            cases_by_class[cls].append(case)
            # Cascade-figure capture: stash one case's arrays per requested class (first seen,
            # keyed (subj,cls)). The coarse and cascade-fine passes walk the same sample order,
            # so both caches key the same (subj,cls) per class -> pairable in save_cascade_figure.
            if (figure_cache is not None and (figure_classes is None or cls in figure_classes)
                    and cls not in figure_cache.get("_done", set())):
                figure_cache.setdefault("_done", set()).add(cls)
                figure_cache[(subjects[i], cls)] = {
                    "img":     target_img[i, 0].cpu().numpy(),      # (D,H,W)
                    "gt":      label[i].numpy(),
                    "pred":    pred[i].numpy(),
                    "ctx_img": context_imgs[i, 0].squeeze(0).cpu().numpy(),
                    "ctx_gt":  context_masks[i, 0].numpy(),
                    "prob":    prob[i, 0].numpy() if prob is not None else None,
                    "spacing": (round(float(batch["spacing"][i, 0]), 4)
                                if "spacing" in batch else None),
                    # Crop geometry [starts, crop_sizes, out_sizes, pad_lo] for the exact
                    # fine->coarse refit (grids are padded, so a plain resize misaligns).
                    "geom":    (batch["crop_geom"][i].numpy() if "crop_geom" in batch else None),
                }

    rows, all_cases = [], []
    # Summarize the requested classes, then any EXTRA classes that showed up in the cases but
    # weren't requested. data.self_context.synth_masks relabels every target 'synth', which is
    # not a benchmark class — without this those cases (the whole val set under a pure-synth
    # eval) would be dropped and the mean val Dice would be nan (all requested classes get
    # 'no samples'). Normal eval has label_name in classes, so `extra` is empty.
    extra = [c for c in cases_by_class if c not in set(classes)]
    for cls in list(classes) + extra:
        cases = cases_by_class.get(cls, [])
        all_cases.extend(cases)                                 # table keeps every sample
        summ = [c for c in cases if not c.get("self_ctx")] if drop_self_ctx else cases
        rows.append(_summarize(cls, summ) if summ
                    else {"class": cls, "error": "no samples"})
    return rows, all_cases


def _refit_into_coarse(pred_fine, geom_c, geom_f):
    """Map a fine-spacing prediction back into the coarse T³ frame via both crop geometries.

    Each crop resamples a native slice [starts, starts+crop_sizes) into a T³ grid region
    [pad_lo, pad_lo+out_sizes) (see _organ_crop_arrays). Composing fine grid -> native ->
    coarse grid is affine per axis: g_c = A*g_f + B. Because the grids are centre-padded
    (out_sizes<T on thin axes), a plain resize+recenter misaligns — this uses the actual
    offsets. Inverse-sampled (nearest) so no gaps. Returns a (T,T,T) 0/1 array."""
    T = pred_fine.shape[0]
    gc, gf = np.asarray(geom_c, float), np.asarray(geom_f, float)
    (starts_c, crop_c, out_c, pad_c), (starts_f, crop_f, out_f, pad_f) = gc, gf
    scale_f = crop_f / np.maximum(1, out_f)          # native voxels per fine-grid voxel
    scale_c = out_c / np.maximum(1, crop_c)          # coarse-grid voxels per native voxel
    A = scale_f * scale_c
    B = pad_c + (starts_f - starts_c - pad_f * scale_f) * scale_c
    ax = np.meshgrid(*[np.arange(T) for _ in range(3)], indexing="ij")
    fidx, valid = [], np.ones((T, T, T), bool)
    for a in range(3):
        f = (ax[a] - B[a]) / A[a]                    # coarse -> fine coordinate
        valid &= (f >= 0) & (f <= T - 1)
        fidx.append(np.clip(np.round(f), 0, T - 1).astype(np.intp))
    sampled = np.asarray(pred_fine)[fidx[0], fidx[1], fidx[2]] > 0.5
    return (valid & sampled).astype(np.float32)


def _refit_into_box(pred_fine, center, side, T):
    """Fallback refit (no crop geometry): resize the fine pred to the `side³` fine box and
    centre it at `center` in the coarse grid. Correct only when the crops aren't padded."""
    side = max(1, min(side, T))
    lo = [max(0, min(int(round(center[a] - side / 2)), T - side)) for a in range(3)]
    t = torch.from_numpy(np.ascontiguousarray(pred_fine, dtype=np.float32))[None, None]
    small = F.interpolate(t, size=(side, side, side), mode="nearest")[0, 0].numpy()
    out = np.zeros((T, T, T), dtype=np.float32)
    out[lo[0]:lo[0] + side, lo[1]:lo[1] + side, lo[2]:lo[2] + side] = (small > 0.5)
    return out


def _write_native(native, pred, geom):
    """Composite a crop-grid prediction into the native volume at its crop location.

    Inverse of the crop: the grid region [pad_lo, pad_lo+out_sizes) resamples from native
    [starts, starts+crop_sizes), so extract the object sub-block, upsample it to crop_sizes
    (nearest), and write it in. Later (finer) writes overwrite earlier (coarser) ones."""
    starts, crop, out, pad = (geom[r].astype(int) for r in range(4))
    sub = pred[pad[0]:pad[0] + out[0], pad[1]:pad[1] + out[1], pad[2]:pad[2] + out[2]]
    if sub.size == 0:
        return
    t = torch.from_numpy(np.ascontiguousarray(sub, dtype=np.float32))[None, None]
    small = F.interpolate(t, size=(int(crop[0]), int(crop[1]), int(crop[2])),
                          mode="nearest")[0, 0].numpy() > 0.5
    D, H, W = native.shape
    d0, h0, w0 = int(starts[0]), int(starts[1]), int(starts[2])
    de, he, we = min(d0 + small.shape[0], D), min(h0 + small.shape[1], H), min(w0 + small.shape[2], W)
    native[d0:de, h0:he, w0:we] = small[:de - d0, :he - h0, :we - w0]


def _unpack_pred(entry):
    packed, shape, geom = entry
    return np.unpackbits(packed)[:int(np.prod(shape))].reshape(shape).astype(bool), geom


def _stitched_native_dice(base_pg, over_pg, root):
    """Dice on the full native volume of GT vs the stitched multi-scale prediction.

    Builds a native-resolution prediction from the coarse (`base_pg`) pass, then overwrites
    each sample's fine (`over_pg`) region on top (finer replaces coarser), and scores it
    against the native GT (label.npy == class index). `over_pg` empty -> coarse-only baseline.
    Returns {(subj,cls): dice} over the keys in `over_pg` (or `base_pg` when `over_pg` empty)."""
    from src.totalseg_dataloader_incontext import _ALL_CLASSES_IDX
    keys = over_pg or base_pg
    out = {}
    for key in keys:
        subj, cls = key
        idx = _ALL_CLASSES_IDX.get(cls)
        if idx is None or key not in base_pg:
            continue
        gt = np.asarray(np.load(Path(root) / subj / "label.npy", mmap_mode="r")) == idx
        native = np.zeros(gt.shape, dtype=bool)
        bp, bgeom = _unpack_pred(base_pg[key]);  _write_native(native, bp, bgeom)
        if key in over_pg:
            op, ogeom = _unpack_pred(over_pg[key]);  _write_native(native, op, ogeom)
        inter = 2.0 * np.logical_and(native, gt).sum()
        denom = int(native.sum()) + int(gt.sum())
        out[key] = inter / denom if denom > 0 else 1.0
    return out


def _save_cascade_pair(coarse_cache, fine_cache, s_coarse, s_fine, out_dir):
    """Emit one save_cascade_figure per class present in both passes' figure caches."""
    def by_cls(cache):
        return {k[1]: (k[0], v) for k, v in cache.items() if isinstance(k, tuple)}
    cc, fc = by_cls(coarse_cache), by_cls(fine_cache)
    ratio = s_fine / s_coarse
    for cls, (subj, c) in cc.items():
        if cls not in fc:
            continue
        _, f = fc[cls]
        T = c["gt"].shape[0]
        side = max(1, round(T * ratio))
        # Box centres: coarse pred centroid (soft prob if available) and the GT centroid (oracle).
        pc = _grid_centroid(c["prob"] if c["prob"] is not None else c["pred"])
        gc = _grid_centroid(c["gt"])
        if pc is None:
            pc = np.array([T / 2.0] * 3)                # empty coarse pred -> volume centre
        fine_box   = (pc, np.array([side] * 3))
        oracle_box = (gc, np.array([side] * 3)) if gc is not None else None
        # Exact fine->coarse remap when both crop geometries are available (grids are padded);
        # else the plain resize-and-centre fallback.
        if c.get("geom") is not None and f.get("geom") is not None:
            refit = _refit_into_coarse(f["pred"], c["geom"], f["geom"])
            # Geometry guardrail: refitting the fine GT should reproduce the coarse GT (high
            # Dice, capped by the coarse resolution). A low value flags a bad affine remap.
            refit_gt = _refit_into_coarse(f["gt"], c["geom"], f["geom"])
            gd = dice_binary(torch.from_numpy(refit_gt), torch.from_numpy(c["gt"].astype("float32")))
            print(f"    [cascade-fig] {cls} ({subj}): refit(fine GT) vs coarse GT dice={gd:.3f}")
        else:
            refit = _refit_into_box(f["pred"], pc, side, T)
        save_cascade_figure(
            out_dir / f"{cls}_{s_coarse:g}to{s_fine:g}mm.png",
            tgt_coarse=c["img"], gt_coarse=c["gt"], pred_coarse=c["pred"],
            ctx_coarse=c["ctx_img"], ctx_gt_coarse=c["ctx_gt"],
            tgt_fine=f["img"], gt_fine=f["gt"], pred_fine=f["pred"],
            ctx_fine=f["ctx_img"], ctx_gt_fine=f["ctx_gt"],
            refit_pred_coarse=refit, refit_gt_coarse=c["gt"],
            fine_box=fine_box, oracle_box=oracle_box, spacings=(s_coarse, s_fine),
            title=f"{cls}  cascade {s_coarse:g}->{s_fine:g}mm  (subj {subj})")


def evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None, fig_dir=None,
                           locator=False, cascade=False, cascade_figures=False):
    """Run evaluate_classes once per physical crop spacing; tag rows with their spacing.

    Builds a constant-spacing eval loader per `s` (make_eval_loader(..., spacing=s)) and
    calls the shared evaluate_classes with that prebuilt loader. `idx` is stable across
    passes, so each spacing sees the same task + context subjects — only the crop spacing
    changes. Figures are saved on the first spacing only (later passes reuse the filenames).

    When locator=True, each coarse pass that has a next-finer spacing also runs the
    coarse->fine localization metric (see _locator_containment): it forwards a soft
    probability via model.train_forward and passes locator_ratio = s_fine/s_coarse so
    evaluate_classes records per-sample containment. A model without train_forward falls
    back to the hard predicted mask centroid (a one-time warning). The finest spacing has
    no successor, so it runs the plain single-predict path with no extra forward.

    When cascade=True, each coarse pass that has a next-finer spacing additionally runs a
    REAL coarse->fine pass: the coarse soft prediction's centroid (mapped back to native
    voxels via _predicted_native_center) becomes the TARGET crop centre for a second,
    finer-spacing eval. The cascade Dice is then scored END-TO-END on the ORIGINAL native
    volume (not per-crop): the coarse prediction is composited into the native volume and the
    fine prediction overwrites its region (finer replaces coarser), then Dice'd against the
    native GT (see _stitched_native_dice). Each cascade row also carries `coarse_only_dice`
    (the same native score from the coarse pred alone) as the no-refinement baseline. Empty
    coarse predictions crop on the volume centre. Cascade rows/cases carry `cascade_from`.
    Both locator and cascade need model.train_forward (a soft prob); if absent, cascade is
    skipped with a warning.

    When cascade_figures=True (requires cascade and fig_dir), one 2x5 coarse->fine panel per
    class is saved under fig_dir/cascade/ (see save_cascade_figure): coarse img/GT/pred +
    bboxes, fine img/GT/pred, and the fine pred refitted into the coarse frame.

    Returns (rows, cases): rows are per-(class, spacing); cases are all passes concatenated.
    Cascade rows are extra rows at the finer spacing, tagged `cascade_from`.
    """
    from common import make_eval_loader  # local import: common/evaluate are siblings

    root = None
    if cascade:
        from common import _source_root  # native GT (label.npy) for the stitched-cascade dice
        _, root, _ = _source_root(cfg)

    lf = op = None
    if locator or cascade:
        lf = getattr(model, "train_forward", None)
        if lf is None:
            what = "locator/cascade" if (locator and cascade) else ("cascade" if cascade else "locator")
            print(f"  [warn] model has no train_forward; {what} needs a soft prob. "
                  "Locator falls back to the hard predicted mask centroid; cascade is skipped.")
        else:
            from train import model_output_is_prob  # local import: sibling module
            op = model_output_is_prob(cfg)

    rows, cases = [], []
    for i, s in enumerate(spacings):
        ratio = None
        if locator and i + 1 < len(spacings) and spacings[i + 1] < s:
            ratio = spacings[i + 1] / s
        # Capture predicted native centres on this pass when a finer successor exists and a
        # soft prob is available -> feeds the cascade fine pass below.
        want_centers = (cascade and lf is not None
                        and i + 1 < len(spacings) and spacings[i + 1] < s)
        centers_out = {} if want_centers else None
        # Capture coarse-pass arrays for the cascade panel only when this pass feeds a fine one.
        coarse_cache = {} if (cascade_figures and fig_dir and want_centers) else None
        # Capture every coarse pred + geom to composite into the native volume for the
        # stitched-cascade dice (finer overwrites coarser). Only when this pass feeds a fine one.
        coarse_pg = {} if want_centers else None
        loader = make_eval_loader(cfg, classes, split=split or cfg.eval.split, spacing=s)
        rows_s, cases_s = evaluate_classes(
            model, cfg, classes, loader=loader,
            fig_dir=fig_dir if i == 0 else None,
            logits_fn=(lf if (ratio is not None or want_centers) else None),
            output_is_prob=bool(op),
            locator_ratio=ratio,
            pred_centers_out=centers_out,
            figure_cache=coarse_cache,
            pred_geom_out=coarse_pg)
        for r in rows_s:
            r["spacing"] = s
            if ratio is not None:
                r["locator_to"] = spacings[i + 1]
        rows.extend(rows_s)
        cases.extend(cases_s)

        # Cascade fine pass: re-crop the finer spacing's TARGETS on this pass's predicted
        # centres (injected via ds._pred_centers), then score the STITCHED native volume.
        if want_centers and centers_out:
            fine_s = spacings[i + 1]
            fine_loader = make_eval_loader(cfg, classes, split=split or cfg.eval.split, spacing=fine_s)
            fine_loader.dataset._pred_centers = centers_out  # set before workers fork (lazy on iter)
            fine_cache = {} if coarse_cache is not None else None
            fine_pg = {}
            rows_c, cases_c = evaluate_classes(model, cfg, classes, loader=fine_loader,
                                               figure_cache=fine_cache, pred_geom_out=fine_pg)
            # End-to-end native dice: coarse composited + fine overwrite vs native GT, plus the
            # coarse-only baseline. Replaces the per-crop fine dice as the headline cascade score.
            casc = _stitched_native_dice(coarse_pg, fine_pg, root)
            base = _stitched_native_dice(coarse_pg, {}, root)
            for c in cases_c:
                key = (c["subject"], c["class"])
                c["dice_crop"] = c["dice"]                      # keep per-crop for reference
                if key in casc:
                    c["dice"] = round(casc[key], 4)            # headline = stitched native dice
                c["coarse_only_dice"] = round(base[key], 4) if key in base else float("nan")
                c["cascade_from"] = s
            # Re-summarise per class from the stitched dice (n stays, mean/std now native).
            cbcls: dict[str, list] = {}
            for c in cases_c:
                cbcls.setdefault(c["class"], []).append(c)
            rows_c = []
            for cls in classes:
                cs = cbcls.get(cls)
                if not cs:
                    continue
                r = _summarize(cls, cs)
                r["spacing"], r["cascade_from"] = fine_s, s
                bo = [c["coarse_only_dice"] for c in cs if not np.isnan(c["coarse_only_dice"])]
                r["coarse_only_dice"] = round(sum(bo) / len(bo), 4) if bo else float("nan")
                rows_c.append(r)
            rows.extend(rows_c)
            cases.extend(cases_c)
            if coarse_cache is not None and fine_cache is not None:
                _save_cascade_pair(coarse_cache, fine_cache, s, fine_s,
                                   fig_dir / "cascade")
    return rows, cases

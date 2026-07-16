"""Shared 2D in-context evaluation: a single validate() used by train.py and
eval_incontext.py, plus qualitative figures and a per-source sample-detail
formatter. Keeps training-time and eval-time metrics coherent by construction.
"""
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import (DEVICE, hard_dice, soft_dice, cosine_sim, topk_overlap,
                    downsample_mask, log_summary)
from src.models.bbox_refine import crop_resize, place_window
from src.models.scatter_sampling import gather_grid, composite_predictions, idx_to_ij


def _overlay_ax(ax, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask,  cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _heatmap_ax(ax, arr: np.ndarray, title: str) -> None:
    ax.imshow(arr, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_figure(
    tgt_image:   np.ndarray,
    tgt_gt:      np.ndarray,
    pred_native: np.ndarray,
    ctx_images:  list[np.ndarray],
    ctx_gts:     list[np.ndarray],
    out_path:    Path,
    title:       str = "",
    pred_lowres: np.ndarray | None = None,
    gt_lowres:   np.ndarray | None = None,
) -> None:
    """Backend-agnostic qualitative panel.

    Row 0: target+GT overlay | target+pred overlay | (GT↓ | pred↓ when a low-res
    grid is supplied). The two low-res heatmaps are omitted for backends without a
    coarse grid (e.g. UniverSeg, whose only output is the native-res binary mask).
    Row 1: the K context image+GT overlays.

    `pred_native` is the native-resolution prediction (soft map for pfn_seg /
    multilevel / feature_sim, binary mask for UniverSeg). `pred_lowres` / `gt_lowres`
    are the coarse-grid prediction and avg-pooled GT at the model's token resolution.
    """
    # Row-0 panels as render closures so the low-res pair is purely optional.
    row0 = [
        lambda ax: _overlay_ax(ax, tgt_image, tgt_gt,      "Target + GT"),
        lambda ax: _overlay_ax(ax, tgt_image, pred_native, "Target + Pred"),
    ]
    if pred_lowres is not None:
        g = pred_lowres.shape[0]
        row0.append(lambda ax: _heatmap_ax(ax, gt_lowres,   f"GT ↓{g}"))
        row0.append(lambda ax: _heatmap_ax(ax, pred_lowres, f"Pred ↓{g}"))

    K     = len(ctx_images)
    ncols = max(len(row0), K, 1)

    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.05)

    for col in range(ncols):
        if col < len(row0):
            row0[col](axes[0, col])
        else:
            axes[0, col].axis("off")

    for col in range(ncols):
        if col < K:
            _overlay_ax(axes[1, col], ctx_images[col], ctx_gts[col], f"Ctx {col} + GT")
        else:
            axes[1, col].axis("off")

    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _refine_overlay_ax(ax, image, title, *, gt=None, pred=None, pred_extent=None, boxes=()):
    """Gray base + optional pred heat (Reds) + optional GT contour (lime) + bbox rectangles.
    pred_extent stretches a coarse pred map over the crop; None = pixel-aligned to `image`."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if pred is not None:
        ax.imshow(pred, cmap="Reds", alpha=0.45, vmin=0, vmax=1, extent=pred_extent)
    if gt is not None and float(gt.max()) > 0:      # contour needs a level present
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.0)
    for (r0, c0, s, color) in boxes:
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), s, s, fill=False, edgecolor=color, lw=1.5))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_refine_figure(
    tgt_image, tgt_gt, ctx_image, ctx_gt,          # full-frame (H,W)
    coarse_pred, fused_pred,                       # target preds (H,W): res0, fused
    refine_pred,                                   # target refine pred (T,T)
    tgt_box, ctx_box,                              # (r0, c0, size) int px
    out_path, title="",
):
    """2×3 refine panel. Row 0 = target, row 1 = first context; col 2 row 1 is empty.
    Col 0: full frame + GT contour + (res0 pred / bbox). Col 1: bbox crop + GT contour +
    (res1 pred on target). Col 2: full frame + GT contour + fused pred (target only)."""
    tr0, tc0, tc = tgt_box
    cr0, cc0, cc = ctx_box
    tgt_crop     = tgt_image[tr0:tr0 + tc, tc0:tc0 + tc]
    tgt_crop_gt  = tgt_gt[tr0:tr0 + tc, tc0:tc0 + tc]
    ctx_crop     = ctx_image[cr0:cr0 + cc, cc0:cc0 + cc]
    ctx_crop_gt  = ctx_gt[cr0:cr0 + cc, cc0:cc0 + cc]
    # refine_pred is T×T over the tc×tc crop: stretch it across the crop's display extent
    crop_extent  = (-0.5, tc - 0.5, tc - 0.5, -0.5)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.5), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.05)

    _refine_overlay_ax(axes[0, 0], tgt_image, "Target + GT + res0 pred",
                       gt=tgt_gt, pred=coarse_pred, boxes=[(tr0, tc0, tc, "yellow")])
    _refine_overlay_ax(axes[1, 0], ctx_image, "Ctx0 + GT",
                       gt=ctx_gt, boxes=[(cr0, cc0, cc, "cyan")])
    _refine_overlay_ax(axes[0, 1], tgt_crop, "Target crop + GT + res1 pred",
                       gt=tgt_crop_gt, pred=refine_pred, pred_extent=crop_extent)
    _refine_overlay_ax(axes[1, 1], ctx_crop, "Ctx0 crop + GT", gt=ctx_crop_gt)
    _refine_overlay_ax(axes[0, 2], tgt_image, "Target + GT + fused pred",
                       gt=tgt_gt, pred=fused_pred)
    axes[1, 2].axis("off")

    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _scatter_cells_ax(ax, image, gt, ij, is_core, is_fg, grid_res, title):
    """Gray image + lime GT contour + sampled cells colored by tier (Rf grid -> image px).
    ij: (M,2) row/col on the grid_res grid; is_core/is_fg: (M,) bool. Tiers are a partition:
    fg-core (orange) subset of core; boundary-core (red) = core & ~fg; neighbor (cyan) = ~core."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if gt is not None and float(gt.max()) > 0:
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.0)
    scale = image.shape[0] / grid_res
    y = (ij[:, 0] + 0.5) * scale
    x = (ij[:, 1] + 0.5) * scale
    fg = is_fg.astype(bool)
    bcore = is_core.astype(bool) & ~fg
    neigh = ~is_core.astype(bool)
    ax.scatter(x[neigh], y[neigh], s=12, c="cyan", marker="s", edgecolors="none",
               label=f"neighbor ({int(neigh.sum())})")
    ax.scatter(x[bcore], y[bcore], s=12, c="red", marker="s", edgecolors="none",
               label=f"boundary ({int(bcore.sum())})")
    ax.scatter(x[fg], y[fg], s=12, c="orange", marker="s", edgecolors="none",
               label=f"fg-core ({int(fg.sum())})")
    ax.legend(loc="upper right", fontsize=5, framealpha=0.6)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_scatter_figure(tgt_image, tgt_gt, coarse_pred, fused_pred,
                        qry_ij, qry_is_core, qry_is_fg,
                        ctx_image, ctx_gt, sup_ij, sup_is_core, sup_is_fg,
                        grid_res, out_path, title=""):
    """2×3 scatter-refine panel. Row 0 (target): [GT + tier-colored query cells | coarse native
    pred | fused native pred]. Row 1: [ctx0 GT + tier-colored support cells | blank | blank].
    Cells live on the grid_res grid and are scaled to image pixels."""
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.5), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.05)
    _scatter_cells_ax(axes[0, 0], tgt_image, tgt_gt, qry_ij, qry_is_core, qry_is_fg,
                      grid_res, "Target + GT + sampled cells")
    _refine_overlay_ax(axes[0, 1], tgt_image, "Target + coarse pred", gt=tgt_gt, pred=coarse_pred)
    _refine_overlay_ax(axes[0, 2], tgt_image, "Target + fused pred", gt=tgt_gt, pred=fused_pred)
    _scatter_cells_ax(axes[1, 0], ctx_image, ctx_gt, sup_ij, sup_is_core, sup_is_fg,
                      grid_res, "Ctx0 + GT + support cells")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fmt_transforms(transforms) -> str:
    """One compact 'r..,s..,dx..,dy..' string per target placement (aug mode);
    empty when the mode applies no per-placement jitter (identical/class)."""
    if not transforms:
        return ""
    parts = []
    for p in transforms:
        if p is None:
            parts.append("-")
        else:
            parts.append(f"r{p['rotate']:+.0f},s{p['scale']:.2f},"
                         f"dx{p['dx']:+.2f},dy{p['dy']:+.2f}")
    return " | ".join(parts)


def _fmt_positions(positions) -> str:
    """Compact real (x, y) glyph positions, e.g. '(0.51,0.48)(0.12,0.90)'; empty when none."""
    return "".join(f"({x:.3f},{y:.3f})" for x, y in (positions or []))


def _as_res_list(ds_metric_res) -> list[int]:
    """Normalise the ds_metric_res config (None | int | list/ListConfig) to [int, ...]."""
    if ds_metric_res is None:
        return []
    if isinstance(ds_metric_res, int):
        return [ds_metric_res]
    return [int(r) for r in ds_metric_res]


# The per-sample wandb Table columns are built dynamically per run in validate(), so they
# carry the actual resolutions (dice_ds@{T}, dice@{Rf}, ...) and only the columns that apply
# to the model at hand (native / patchset / refine). See the lazy `table` creation below.


def _sample_detail(meta: dict | None) -> str:
    """One compact string describing a sample, adapting to the data source.

    omniSynth meta -> "alphabet/class mode=<m> cells=<...> ctx=<...> pos=<...> cpos=<...>
    sub=<i> tf=<...>"; controlSynth meta -> "<morphology> task=<id>"; anything else (e.g.
    medsegbench, or missing meta) -> "". Keeps the sample table's columns fixed across sources.

    ctx (context_cells) + sub (subject_index) make an omniSynth-only run self-contained.
    pos/cpos are the real post-aug (x, y) target/context positions in [0, 1] (ink centroid),
    the continuous counterpart of the discrete cells for target<->context distance analysis.
    tf stays last because it is the only free-form field (spaces + " | " between placements).
    """
    if not meta:
        return ""
    if "alphabet" in meta:  # omniSynth
        cpos = " ".join(_fmt_positions(p) for p in meta.get("context_positions", []))
        return (f"{meta.get('alphabet')}/{meta.get('class_id')} "
                f"mode={meta.get('target_mode', '')} "
                f"cells={meta.get('target_cells', [])} "
                f"ctx={meta.get('context_cells', [])} "
                f"pos={_fmt_positions(meta.get('target_positions'))} "
                f"cpos={cpos} "
                f"sub={meta.get('subject_index', -1)} "
                f"tf={_fmt_transforms(meta.get('target_transforms'))}")
    if "morphology" in meta:  # controlSynth
        return f"{meta.get('morphology')} task={int(meta.get('task_id', -1))}"
    return ""


def _target_like(lbl: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    """Avg-pool the (B,1,H,W) GT to the logit's spatial size (no-op when equal)."""
    if lbl.shape[-2:] == logit.shape[-2:]:
        return lbl
    return F.adaptive_avg_pool2d(lbl, logit.shape[-2:])


def _upsample_to(x: torch.Tensor, size) -> torch.Tensor:
    """Bilinear-resize (B,1,h,w) -> (B,1,*size); no-op when already at `size`."""
    return (x if x.shape[-2:] == tuple(size)
            else F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False))


def _refine_geometry_scatter(out: dict, lbl: torch.Tensor) -> dict:
    """Scatter-refine geometry: per-sampled-cell prob/target + fused stitch (coarse with refined
    cells scattered in). Returns the SAME keys as the bbox refine_geometry so downstream metrics
    are model-agnostic. refine_prob/target are (B,1,M) so callers' [b,0] indexing yields (M,)."""
    coarse = out["final_logit"].float()                       # (B,1,T,T)
    refine_logit = out["refine_logit"].float()                # (B,M)
    idx = out["refine_idx"]                                    # (B,M)
    Rf = int(out["refine_grid_res"])
    B, H = lbl.shape[0], lbl.shape[-1]
    Nf = Rf * Rf
    refine_prob = torch.sigmoid(refine_logit)                 # (B,M)
    gt_Rf = F.adaptive_avg_pool2d(lbl, (Rf, Rf)).reshape(B, Nf)
    refine_target = gather_grid(gt_Rf, idx)                    # (B,M)
    coarse_prob = torch.sigmoid(coarse)
    coarse_up = F.interpolate(coarse_prob, size=(H, H), mode="bilinear", align_corners=False)
    coarse_Rf = F.interpolate(coarse_prob, size=(Rf, Rf), mode="bilinear",
                              align_corners=False).reshape(B, Nf)   # bilinear (matches coarse_nat)
    # NOTE: bilinear (not adaptive_avg_pool2d) keeps the non-sampled background consistent with
    # coarse_nat so that fused differs from coarse ONLY at the scattered cells.
    fused_flat = composite_predictions(coarse_Rf, idx, refine_prob)             # (B,Nf)
    fused_R = fused_flat.reshape(B, 1, Rf, Rf)
    fused = F.interpolate(fused_R, size=(H, H), mode="bilinear", align_corners=False)
    return {"refine_prob": refine_prob.unsqueeze(1),          # (B,1,M)
            "refine_target": refine_target.unsqueeze(1),      # (B,1,M)
            "fused": fused,                                    # (B,1,H,H)
            "fused_R": fused_R, "gt_R": gt_Rf.reshape(B, 1, Rf, Rf), "Rf": Rf,
            "coarse_nat": coarse_up,                           # (B,1,H,H)
            "coarse_R": coarse_Rf.reshape(B, 1, Rf, Rf)}       # (B,1,Rf,Rf)


def refine_geometry(out: dict, lbl: torch.Tensor) -> dict | None:
    """Per-level + fused tensors for a multi-resolution refine output; None if single-level.

    out: model output; multi-level has coarse `final_logit` (B,1,T,T), `refine_logit` (B,1,T,T),
    `refine_origin` (B,2 px), `refine_crop` (int px), `resolutions` (list). lbl: (B,1,H,W) GT.
    Returns detached maps for metrics (call under no_grad):
      refine_prob   (B,1,T,T)  sigmoid(refine_logit)
      refine_target (B,1,T,T)  crop_resize(lbl, origin, c, T) — soft cropped GT
      fused_R/gt_R  (B,1,Rf,Rf) fused prob (coarse with refine placed in the crop) and GT,
                    both avg-pooled to Rf = resolutions[-1]
      coarse_nat    (B,1,H,H)  coarse-only prob upsampled to native — the refine-off counterfactual
      coarse_R      (B,1,Rf,Rf) coarse-only prob avg-pooled to Rf (compare vs fused_R at same res)
    If `out` contains `refine_idx` (scatter mode), dispatches to `_refine_geometry_scatter`, which
    returns the same keys but with `refine_prob`/`refine_target` shaped `(B,1,M)` (per sampled cell)
    instead of `(B,1,T,T)`; bbox keys (`refine_origin`, `refine_crop`) are absent in that case.
    """
    if "refine_logit" not in out:
        return None
    if out.get("refine_idx") is not None:
        return _refine_geometry_scatter(out, lbl)
    coarse = out["final_logit"].float()
    refine = out["refine_logit"].float()
    origin = out["refine_origin"]
    c = int(out["refine_crop"])
    Rf = int(out["resolutions"][-1])
    T = refine.shape[-1]
    H = lbl.shape[-1]
    refine_prob = torch.sigmoid(refine)
    refine_target = crop_resize(lbl, origin, c, T, mode="bilinear")
    coarse_up = F.interpolate(torch.sigmoid(coarse), size=(H, H),
                              mode="bilinear", align_corners=False)
    refine_up = F.interpolate(refine_prob, size=(c, c), mode="bilinear", align_corners=False)
    fused = place_window(coarse_up, refine_up, origin, c)              # (B,1,H,H) native stitch
    return {"refine_prob": refine_prob, "refine_target": refine_target,
            "fused": fused,
            "fused_R": F.adaptive_avg_pool2d(fused, (Rf, Rf)),
            "gt_R": F.adaptive_avg_pool2d(lbl, (Rf, Rf)), "Rf": Rf,
            "coarse_nat": coarse_up,                                   # refine-off counterfactual @native
            "coarse_R": F.adaptive_avg_pool2d(coarse_up, (Rf, Rf))}    # coarse-only @Rf (vs fused_R)


@torch.no_grad()
def validate(model, loader, *, topk_k=16, epoch=0, figures=None,
             patch_csv=None, synth_csv=None, compute_flops=False, ds_metric_res=None,
             per_group=True):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    hard_ds, hard_lab = defaultdict(list), defaultdict(list)   # native hard dice
    dsh_ds,  dsh_lab  = defaultdict(list), defaultdict(list)   # low-res hard dice_ds
    soft_ds, soft_lab = defaultdict(list), defaultdict(list)   # low-res soft dice_ds_soft
    cos_ds,  cos_lab  = defaultdict(list), defaultdict(list)   # populated only when not native
    topk_ds, topk_lab = defaultdict(list), defaultdict(list)
    ref_h_ds, ref_h_lab = defaultdict(list), defaultdict(list)   # refine hard dice@Rf
    ref_s_ds, ref_s_lab = defaultdict(list), defaultdict(list)   # refine soft
    fus_h_ds, fus_h_lab = defaultdict(list), defaultdict(list)   # fused hard dice_fused@Rf
    fus_s_ds, fus_s_lab = defaultdict(list), defaultdict(list)   # fused soft
    coh_ds, coh_lab = defaultdict(list), defaultdict(list)       # coarse-only hard dice_coarse@native
    coR_ds, coR_lab = defaultdict(list), defaultdict(list)       # coarse-only hard dice_coarse@Rf
    fused_res = None                                             # resolutions[-1] when refine
    # Optional fixed-resolution hard/soft Dice on avg-pooled GT vs avg-pooled pred. This is
    # ONLY for a native-res model (UniverSeg): pooling its native prediction to R×R gives a
    # coarse-grid score comparable to patchset_cnn's own dice_ds@{low_res}. Non-native models
    # (patchset_cnn / refine) already emit their native coarse grid, so `ds_metric_res` is
    # ignored for them (its pooled @R on the coarse-upsampled pred would just be confusing).
    res_list = _as_res_list(ds_metric_res)
    dsr = {R: {"hd": defaultdict(list), "hl": defaultdict(list),
               "sd": defaultdict(list), "sl": defaultdict(list)} for R in res_list}
    low_res = None   # a non-native model's coarse logit side length (e.g. 16 for patchset_cnn)
    table = None                     # lazily built once low_res / fused_res are known (below)
    inf_times, flops, saved = [], None, set()
    warned_patch = False
    patch_rows = [] if patch_csv else None
    synth_rows = [] if synth_csv else None
    max_fig = int(figures["max_figures"]) if figures else 0

    for batch in tqdm(loader, desc="val", leave=False):
        if batch is None:
            continue
        img  = batch["image"].to(DEVICE, non_blocking=True)
        lbl  = batch["label"].to(DEVICE, non_blocking=True).float()
        cin  = batch["context_in"].to(DEVICE, non_blocking=True)
        cout = batch["context_out"].to(DEVICE, non_blocking=True)
        B, _, H, W = img.shape
        K = cin.shape[1]

        ac = torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                            enabled=DEVICE.type == "cuda")
        sync = DEVICE.type == "cuda"
        if compute_flops and flops is None:
            # Measured once; its FlopCounterMode overhead must not contaminate timing.
            with FlopCounterMode(display=False) as fc, ac:
                out = model(img, context_in=cin, context_out=cout, mode="val")
            # Per-sample, so flops_giga is invariant to eval.batch_size (total-batch
            # FLOPs scaled linearly with bs — a bs=64→128 change doubled the number).
            flops = fc.get_total_flops() / img.shape[0]
        else:
            # CUDA kernels are async, so bracket the forward with syncs to time the real
            # GPU compute (not just kernel launch). The first timed batch is dropped as
            # warmup below (cudnn.benchmark autotune / allocator / lazy init).
            if sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with ac:
                out = model(img, context_in=cin, context_out=cout, mode="val")
            if sync:
                torch.cuda.synchronize()
            inf_times.append((time.perf_counter() - t0) / B)
        logit = out["final_logit"].float()
        rg = refine_geometry(out, lbl)
        if rg is not None:
            fused_res = rg["Rf"]

        target   = _target_like(lbl, logit)             # (B,1,hp,wp) soft pooled GT
        prob     = torch.sigmoid(logit)                  # (B,1,hp,wp)
        prob_nat = _upsample_to(prob, lbl.shape[-2:])   # (B,1,H,W)
        native   = logit.shape[-2:] == lbl.shape[-2:]
        # For refine models the final prediction is the fused native stitch (last level), so
        # `dice` is scored on it (it is already at H×W). dice_ds/dice_ds_soft/cossim/top-k
        # still reflect the coarse pass for now.
        pred_dice = rg["fused"] if rg is not None else prob_nat   # (B,1,H,W)
        if not native and low_res is None:
            low_res = logit.shape[-1]
        # avg-pool the full-res pred + GT to each requested resolution (batched, once).
        # Native (UniverSeg) only — see res_list note above; empty for patchset_cnn/refine.
        pooled = ({R: (F.adaptive_avg_pool2d(prob_nat, (R, R)),
                       F.adaptive_avg_pool2d(lbl, (R, R))) for R in res_list}
                  if native else {})
        if native and patch_rows is not None and not warned_patch:
            print("warning: patch_csv is set but this model is native-resolution "
                  "(no low-res grid) — no patch rows will be written.")
            warned_patch = True
        metas = batch.get("meta")

        for b in range(B):
            ds  = batch["dataset"][b]
            lv  = int(batch["label_value"][b])
            si  = int(batch["sample_idx"][b])
            key = f"{ds}/label_{lv}"

            h = hard_dice(pred_dice[b, 0], lbl[b, 0])    # native hard dice (fused for refine)
            hard_ds[ds].append(h); hard_lab[key].append(h)

            # GT size/occupancy (model-independent) — logged for every model so downstream
            # analysis can bucket by object size without regenerating the deterministic val set.
            # size = native foreground px; occ = foreground fraction. ctx_* averaged over the K
            # context masks. (cout is (B,K,1,H,W).)
            tgt_fg   = lbl[b, 0] > 0
            tgt_size = float(tgt_fg.sum())
            tgt_occ  = float(tgt_fg.float().mean())
            ctx_fg   = cout[b, :, 0] > 0                              # (K,H,W)
            ctx_size = float(ctx_fg.float().sum()) / max(K, 1)        # mean fg px per context
            ctx_occ  = float(ctx_fg.float().mean())                  # mean fg fraction over K

            if not native:
                dh = hard_dice(prob[b, 0], (target[b, 0] >= 0.5).float())
                s  = soft_dice(prob[b, 0], target[b, 0])
                c  = cosine_sim(prob[b, 0], target[b, 0])
                t  = topk_overlap(prob[b, 0], target[b, 0], topk_k)
                cos_ds[ds].append(c); cos_lab[key].append(c)
                topk_ds[ds].append(t); topk_lab[key].append(t)
                # Coarse-grid survival: how much of the target survives avg-pooling to the token
                # grid T. tgt_cells@T = # cells with pooled GT ≥0.5 (0 = object lost at pooling);
                # tgt_peak@T = best single-cell occupancy. Both explain small-object coarse misses.
                tgt_cells = float((target[b, 0] >= 0.5).sum())
                tgt_peak  = float(target[b, 0].max())
            else:
                dh = s = float("nan")                    # native: no coarse grid
            dsh_ds[ds].append(dh); dsh_lab[key].append(dh)
            soft_ds[ds].append(s); soft_lab[key].append(s)

            for R in (res_list if native else []):        # fixed-res pooled Dice (UniverSeg only)
                p_r, g_r = pooled[R]
                sr = soft_dice(p_r[b, 0], g_r[b, 0])
                hr = hard_dice(p_r[b, 0], (g_r[b, 0] >= 0.5).float())
                d = dsr[R]
                d["sd"][ds].append(sr); d["sl"][key].append(sr)
                d["hd"][ds].append(hr); d["hl"][key].append(hr)

            # refine per-sample: refine-level (@Rf, on the crop) + fused (@Rf) Dice — computed
            # once, fed to both the aggregates and the sample-table row below.
            refine_row = []
            if rg is not None:
                rdh = hard_dice(rg["refine_prob"][b, 0], (rg["refine_target"][b, 0] >= 0.5).float())
                rds = soft_dice(rg["refine_prob"][b, 0], rg["refine_target"][b, 0])
                fdh = hard_dice(rg["fused_R"][b, 0], (rg["gt_R"][b, 0] >= 0.5).float())
                fds = soft_dice(rg["fused_R"][b, 0], rg["gt_R"][b, 0])
                # Coarse-only counterfactual (refine off): coarse pred at native + at Rf. Directly
                # comparable to `h` (fused native) and `fdh` (fused@Rf) → the exact refine delta.
                coh = hard_dice(rg["coarse_nat"][b, 0], lbl[b, 0])
                coR = hard_dice(rg["coarse_R"][b, 0], (rg["gt_R"][b, 0] >= 0.5).float())
                ref_h_ds[ds].append(rdh); ref_h_lab[key].append(rdh)
                ref_s_ds[ds].append(rds); ref_s_lab[key].append(rds)
                fus_h_ds[ds].append(fdh); fus_h_lab[key].append(fdh)
                fus_s_ds[ds].append(fds); fus_s_lab[key].append(fds)
                coh_ds[ds].append(coh); coh_lab[key].append(coh)
                coR_ds[ds].append(coR); coR_lab[key].append(coR)
                refine_row = [rdh, rds, fdh, coR, coh]    # dice@Rf, dice_soft@Rf, dice_fused@Rf, dice_coarse@Rf, dice_coarse

            # Lazy, resolution-tagged sample table. Columns depend on the model (constant per
            # run): dice always; dice_ds@{T}/dice_ds_soft@{T}/cossim@{T}/top{k}@{T} for non-native
            # (coarse grid T); dice@{Rf}/dice_soft@{Rf}/dice_fused@{Rf} for refine.
            # cossim/top-k are the RANKING metrics — the coarse pred's job is to rank target cells
            # above background (for the scatter sampler), not to hit exact occupancy, so they matter
            # per-sample (esp. for small objects whose hard/soft dice collapse at the 32² grid).
            if table is None:
                cols = ["epoch", "dataset", "sample_idx", "label", "dice",
                        "tgt_size", "tgt_occ", "ctx_size", "ctx_occ"]
                if low_res is not None:
                    cols += [f"dice_ds@{low_res}", f"dice_ds_soft@{low_res}",
                             f"cossim@{low_res}", f"top{topk_k}@{low_res}",
                             f"tgt_cells@{low_res}", f"tgt_peak@{low_res}"]
                if fused_res is not None:
                    cols += [f"dice@{fused_res}", f"dice_soft@{fused_res}", f"dice_fused@{fused_res}",
                             f"dice_coarse@{fused_res}", "dice_coarse"]
                cols.append("detail")
                table = wandb.Table(columns=cols)
            detail = _sample_detail(metas[b]) if metas is not None else ""
            row = [epoch, ds, si, lv, h, tgt_size, tgt_occ, ctx_size, ctx_occ]
            if low_res is not None:
                # dice_ds@{T}, dice_ds_soft@{T}, cossim@{T}, top{k}@{T}, tgt_cells@{T}, tgt_peak@{T}
                row += [dh, s, c, t, tgt_cells, tgt_peak]
            if fused_res is not None:
                row += refine_row
            row.append(detail)
            table.add_data(*row)

            # ── gated: qualitative figure (one per dataset/label) ──
            fig_key = (ds, lv)
            if figures and fig_key not in saved and len(saved) < max_fig:
                saved.add(fig_key)
                fig_path = Path(figures["out_dir"]) / f"{ds}_l{lv}.png"
                low = None if native else prob[b, 0].cpu().numpy()
                glow = None if native else target[b, 0].cpu().numpy()
                save_figure(
                    tgt_image=img[b, 0].cpu().numpy(), tgt_gt=lbl[b, 0].cpu().numpy(),
                    pred_native=prob_nat[b, 0].cpu().numpy(),
                    ctx_images=[cin[b, k, 0].cpu().numpy() for k in range(K)],
                    ctx_gts=[cout[b, k, 0].cpu().numpy() for k in range(K)],
                    out_path=fig_path,
                    title=f"{ds} label={lv} sample={si} dice={h:.3f}",
                    pred_lowres=low, gt_lowres=glow)
                if figures.get("to_wandb"):
                    wandb.log({f"figures/{ds}/label_{lv}": wandb.Image(str(fig_path))})
                if rg is not None and out.get("refine_origin") is not None:  # bbox-refine coarse→fine panel
                    c_px = int(out["refine_crop"])
                    fig_path_refine = Path(figures["out_dir"]) / f"{ds}_l{lv}_refine.png"
                    tr0, tc0 = (int(v) for v in out["refine_origin"][b])
                    cr0, cc0 = (int(v) for v in out["refine_ctx_origin"][b, 0])
                    save_refine_figure(
                        tgt_image=img[b, 0].cpu().numpy(), tgt_gt=lbl[b, 0].cpu().numpy(),
                        ctx_image=cin[b, 0, 0].cpu().numpy(), ctx_gt=cout[b, 0, 0].cpu().numpy(),
                        coarse_pred=prob_nat[b, 0].cpu().numpy(),
                        fused_pred=rg["fused"][b, 0].cpu().numpy(),
                        refine_pred=rg["refine_prob"][b, 0].cpu().numpy(),
                        tgt_box=(tr0, tc0, c_px), ctx_box=(cr0, cc0, c_px),
                        out_path=fig_path_refine,
                        title=f"{ds} label={lv} sample={si} refine")
                    if figures.get("to_wandb"):
                        wandb.log({f"figures_refine/{ds}/label_{lv}":
                                   wandb.Image(str(fig_path_refine))})
                elif rg is not None and out.get("refine_idx") is not None:  # scatter panel
                    Rf_g = int(out["refine_grid_res"])
                    fig_path_scatter = Path(figures["out_dir"]) / f"{ds}_l{lv}_scatter.png"
                    q_ij = idx_to_ij(out["refine_idx"][b:b + 1], Rf_g)[0].cpu().numpy()      # (M,2)
                    s_ij = idx_to_ij(out["refine_sup_idx"][b, 0:1], Rf_g)[0].cpu().numpy()   # ctx0 (M,2)
                    save_scatter_figure(
                        tgt_image=img[b, 0].cpu().numpy(), tgt_gt=lbl[b, 0].cpu().numpy(),
                        coarse_pred=prob_nat[b, 0].cpu().numpy(),
                        fused_pred=rg["fused"][b, 0].cpu().numpy(),
                        qry_ij=q_ij, qry_is_core=out["refine_is_core"][b].cpu().numpy(),
                        qry_is_fg=out["refine_is_fg"][b].cpu().numpy(),
                        ctx_image=cin[b, 0, 0].cpu().numpy(), ctx_gt=cout[b, 0, 0].cpu().numpy(),
                        sup_ij=s_ij, sup_is_core=out["refine_sup_is_core"][b, 0].cpu().numpy(),
                        sup_is_fg=out["refine_sup_is_fg"][b, 0].cpu().numpy(),
                        grid_res=Rf_g, out_path=fig_path_scatter,
                        title=f"{ds} label={lv} sample={si} scatter")
                    if figures.get("to_wandb"):
                        wandb.log({f"figures_scatter/{ds}/label_{lv}":
                                   wandb.Image(str(fig_path_scatter))})

            # ── gated: per-low-res-patch CSV (only meaningful when not native) ──
            if patch_rows is not None and not native:
                pp = prob[b, 0].cpu().numpy(); gp = target[b, 0].cpu().numpy()
                gt_size = float((lbl[b, 0] > 0).sum())
                ctx_d = [hard_dice(lbl[b, 0], cout[b, k, 0]) for k in range(K)]
                cd = float(np.nanmean(ctx_d)) if ctx_d else float("nan")
                for i in range(pp.shape[0]):
                    for j in range(pp.shape[1]):
                        patch_rows.append((ds, lv, si, i, j, float(pp[i, j]),
                                           float(gp[i, j]), float(pp[i, j] - gp[i, j]),
                                           gt_size, cd))

            # ── gated: per-element controlSynth params CSV ──
            if synth_rows is not None and metas is not None and "morphology" in (metas[b] or {}):
                m = metas[b]
                row = {"dataset": ds, "sample_idx": si, "label_value": lv,
                       "dice_native": h, "dice_ds": dh,
                       "morphology": m["morphology"], "task_id": int(m["task_id"]),
                       "subject_index": int(m.get("subject_index", -1)),
                       "fg_frac": float((lbl[b, 0] > 0).float().mean())}
                row.update({k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in m.get("difficulty", {}).items()})
                synth_rows.append(row)

    # Drop the first timed batch as warmup so autotune/allocator/lazy-init cost doesn't
    # inflate the mean (real GPU forward time, thanks to the syncs above).
    timed = inf_times[1:] if len(inf_times) > 1 else inf_times
    extra = {"time/inference_ms": (float(np.mean(timed)) * 1000
                                   if timed else float("nan"))}
    if flops is not None:
        extra["flops_giga"] = flops / 1e9

    summary = {}
    summary.update(log_summary(hard_ds, hard_lab, prefix="dice",
                               metric_label="native", extra=extra, per_group=per_group))
    if low_res is not None:   # non-native model: tag its coarse-grid Dice with the resolution
        summary.update(log_summary(dsh_ds, dsh_lab, prefix=f"dice_ds@{low_res}",
                                   metric_label=f"hard@{low_res}", per_group=per_group))
        summary.update(log_summary(soft_ds, soft_lab, prefix=f"dice_ds_soft@{low_res}",
                                   metric_label=f"soft@{low_res}", per_group=per_group))
    if cos_ds:   # populated only when non-native — computed on the coarse token grid T=low_res
        summary.update(log_summary(cos_ds, cos_lab, prefix=f"cossim@{low_res}",
                                   metric_label=f"cos sim@{low_res}", per_group=per_group))
        summary.update(log_summary(topk_ds, topk_lab, prefix=f"top{topk_k}@{low_res}",
                                   metric_label=f"top{topk_k}@{low_res}", per_group=per_group))
    for R in (res_list if native else []):                # fixed-res pooled Dice (UniverSeg only)
        d = dsr[R]
        summary.update(log_summary(d["hd"], d["hl"], prefix=f"dice_ds@{R}",
                                   metric_label=f"hard@{R}", per_group=per_group))
        summary.update(log_summary(d["sd"], d["sl"], prefix=f"dice_ds_soft@{R}",
                                   metric_label=f"soft@{R}", per_group=per_group))
    if fused_res is not None:                             # refine model: per-level + fused Dice
        summary.update(log_summary(ref_h_ds, ref_h_lab, prefix=f"dice@{fused_res}",
                                   metric_label=f"refine@{fused_res}", per_group=per_group))
        summary.update(log_summary(ref_s_ds, ref_s_lab, prefix=f"dice_soft@{fused_res}",
                                   metric_label=f"refine soft@{fused_res}", per_group=per_group))
        summary.update(log_summary(fus_h_ds, fus_h_lab, prefix=f"dice_fused@{fused_res}",
                                   metric_label=f"fused@{fused_res}", per_group=per_group))
        summary.update(log_summary(fus_s_ds, fus_s_lab, prefix=f"dice_fused_soft@{fused_res}",
                                   metric_label=f"fused soft@{fused_res}", per_group=per_group))
        # Coarse-only (refine off) at the same resolutions: refine delta = fused − coarse.
        summary.update(log_summary(coR_ds, coR_lab, prefix=f"dice_coarse@{fused_res}",
                                   metric_label=f"coarse@{fused_res}", per_group=per_group))
        summary.update(log_summary(coh_ds, coh_lab, prefix="dice_coarse",
                                   metric_label="coarse@native", per_group=per_group))

    if patch_rows:
        p = Path(patch_csv); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "label_value", "sample_idx", "patch_i", "patch_j",
                        "pred", "gt", "error", "gt_size", "ctx_dice"])
            w.writerows(patch_rows)
        print(f"Wrote {len(patch_rows)} patch rows to {p}")
    if synth_rows:
        p = Path(synth_csv); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(synth_rows[0].keys()))
            w.writeheader(); w.writerows(synth_rows)
        print(f"Wrote {len(synth_rows)} synth rows to {p}")

    return summary, table, flops

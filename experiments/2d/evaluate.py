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

from common import (DEVICE, hard_dice, soft_dice, cosine_sim, topk_overlap,
                    downsample_mask, log_summary)


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


SAMPLE_COLS = ["epoch", "dataset", "sample_idx", "label",
               "dice", "dice_ds", "dice_ds_soft", "detail"]


def _sample_detail(meta: dict | None) -> str:
    """One compact string describing a sample, adapting to the data source.

    omniSynth meta -> "alphabet/class mode=<m> cells=<...> tf=<...>";
    controlSynth meta -> "<morphology> task=<id>"; anything else (e.g. medsegbench,
    or missing meta) -> "". Keeps the wandb sample table's columns fixed across sources.
    """
    if not meta:
        return ""
    if "alphabet" in meta:  # omniSynth
        return (f"{meta.get('alphabet')}/{meta.get('class_id')} "
                f"mode={meta.get('target_mode', '')} "
                f"cells={meta.get('target_cells', [])} "
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


@torch.no_grad()
def validate(model, loader, *, topk_k=16, epoch=0, figures=None,
             patch_csv=None, synth_csv=None, compute_flops=False):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    hard_ds, hard_lab = defaultdict(list), defaultdict(list)   # native hard dice
    dsh_ds,  dsh_lab  = defaultdict(list), defaultdict(list)   # low-res hard dice_ds
    soft_ds, soft_lab = defaultdict(list), defaultdict(list)   # low-res soft dice_ds_soft
    cos_ds,  cos_lab  = defaultdict(list), defaultdict(list)   # populated only when not native
    topk_ds, topk_lab = defaultdict(list), defaultdict(list)
    table = wandb.Table(columns=SAMPLE_COLS)
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
        if compute_flops and flops is None:
            # Measured once; its FlopCounterMode overhead must not contaminate timing.
            with FlopCounterMode(display=False) as fc, ac:
                out = model(img, context_in=cin, context_out=cout, mode="val")
            flops = fc.get_total_flops()
        else:
            t0 = time.perf_counter()
            with ac:
                out = model(img, context_in=cin, context_out=cout, mode="val")
            inf_times.append((time.perf_counter() - t0) / B)
        logit = out["final_logit"].float()

        target   = _target_like(lbl, logit)             # (B,1,hp,wp) soft pooled GT
        prob     = torch.sigmoid(logit)                  # (B,1,hp,wp)
        prob_nat = _upsample_to(prob, lbl.shape[-2:])   # (B,1,H,W)
        native   = logit.shape[-2:] == lbl.shape[-2:]
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

            h = hard_dice(prob_nat[b, 0], lbl[b, 0])     # native hard dice
            hard_ds[ds].append(h); hard_lab[key].append(h)
            if not native:
                dh = hard_dice(prob[b, 0], (target[b, 0] >= 0.5).float())
                s  = soft_dice(prob[b, 0], target[b, 0])
                c  = cosine_sim(prob[b, 0], target[b, 0])
                t  = topk_overlap(prob[b, 0], target[b, 0], topk_k)
                cos_ds[ds].append(c); cos_lab[key].append(c)
                topk_ds[ds].append(t); topk_lab[key].append(t)
            else:
                dh = s = float("nan")                    # native: no coarse grid
            dsh_ds[ds].append(dh); dsh_lab[key].append(dh)
            soft_ds[ds].append(s); soft_lab[key].append(s)

            detail = _sample_detail(metas[b]) if metas is not None else ""
            table.add_data(epoch, ds, si, lv, h, dh, s, detail)

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

    extra = {"time/inference_ms": (float(np.mean(inf_times)) * 1000
                                   if inf_times else float("nan"))}
    if flops is not None:
        extra["flops_giga"] = flops / 1e9

    summary = {}
    summary.update(log_summary(hard_ds, hard_lab, prefix="dice",
                               metric_label="native", extra=extra))
    summary.update(log_summary(dsh_ds, dsh_lab, prefix="dice_ds",
                               metric_label="downsampled"))
    summary.update(log_summary(soft_ds, soft_lab, prefix="dice_ds_soft",
                               metric_label="low-res soft"))
    if cos_ds:   # populated only when some batch was non-native
        summary.update(log_summary(cos_ds, cos_lab, prefix="cossim",
                                   metric_label="cos sim"))
        summary.update(log_summary(topk_ds, topk_lab, prefix=f"top{topk_k}",
                                   metric_label=f"top{topk_k}"))

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

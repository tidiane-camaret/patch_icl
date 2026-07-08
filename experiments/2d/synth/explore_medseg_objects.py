"""Explore MedSegBench as an *object* source for omniSynth.

omniSynth currently pastes Omniglot glyphs (binary bitmaps used as both ink and
mask). To reuse real medical objects instead, we crop each connected component of
a MedSegBench mask to its bbox and keep two things: the binary mask (the "ink" /
label) and the intensity patch under it (the texture pasted into the image). This
script characterises what such objects look like so we can design the bank:

  - objects/image (connected components) -> which datasets give discrete instances
    vs. one big blob vs. shattered vessel networks
  - object bbox size + area fraction     -> how they map onto a cell_size tile
  - fill = area / bbox_area              -> how "glyph-like" (thin/sparse vs solid)
  - intensity mean/std under the mask    -> the texture we carry alongside the mask
  - contrast = obj_mean - ring_mean      -> object vs local background separation

Outputs a per-object CSV, a per-dataset text report, stat histograms, and a
montage of extracted objects (intensity crop, masked crop, binary mask) so the
object set can be eyeballed before building the bank.

Usage:
    .venv/bin/python experiments/2d/synth/explore_medseg_objects.py
    .venv/bin/python experiments/2d/synth/explore_medseg_objects.py --split train --max_per_ds 200
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.datasets.medsegbench import MedSegBenchDataset

PCTL = [0, 5, 25, 50, 75, 95, 100]
# Datasets shown in the object montage — chosen to span the object regimes seen in
# the probe: single large blob / many small instances / dense nuclei.
MONTAGE_DS = ["busi", "isic2018", "kvasir", "cellnuclei", "dynamicnuclear",
              "monusac", "wbc", "pandental"]


def object_rows(img, mask, ds, idx, lv, min_area):
    """Per-connected-component stat rows for one binary mask (mask>0)."""
    cc, n = ndi.label(mask > 0)
    if n == 0:
        return []
    H, W = mask.shape
    img_f = img.astype(np.float32)
    rows = []
    for sl in ndi.find_objects(cc):
        if sl is None:
            continue
        sub_cc = cc[sl]
        # A bbox may clip a neighbour component; restrict to the component whose slice
        # this is. find_objects returns one slice per label id, so recover that id:
        comp_id = _slice_label(cc, sl)
        obj = sub_cc == comp_id
        area = int(obj.sum())
        if area < min_area:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        ints = img_f[sl][obj]
        # local background ring: dilate the bbox by 2px, take the pixels outside the object
        ring = _ring_mean(img_f, cc, sl, comp_id)
        rows.append({
            "dataset": ds, "sample_idx": idx, "label_value": lv, "comp_id": comp_id,
            "area_px": area, "area_frac": area / (H * W),
            "bbox_max_px": max(h, w), "bbox_max_frac": max(h, w) / H,
            "fill": area / (h * w), "int_mean": float(ints.mean()),
            "int_std": float(ints.std()), "contrast": float(ints.mean() - ring),
        })
    return rows


def _slice_label(cc, sl):
    """The component id whose find_objects slice is `sl` (the id equal to the bbox's
    dominant label). find_objects is index-aligned so this is exact for that id."""
    sub = cc[sl]
    vals, counts = np.unique(sub[sub > 0], return_counts=True)
    return int(vals[counts.argmax()])


def _ring_mean(img_f, cc, sl, comp_id):
    """Mean intensity of a 2px background ring around the object (bbox-local),
    excluding any labelled pixel. Falls back to the object mean if the ring is empty."""
    pad = 3
    y0, y1 = max(0, sl[0].start - pad), min(cc.shape[0], sl[0].stop + pad)
    x0, x1 = max(0, sl[1].start - pad), min(cc.shape[1], sl[1].stop + pad)
    reg_cc = cc[y0:y1, x0:x1]
    reg_im = img_f[y0:y1, x0:x1]
    bg = reg_cc == 0
    return float(reg_im[bg].mean()) if bg.any() else float(reg_im.mean())


def summarize(s):
    p = np.percentile(s, PCTL)
    return {"n": len(s), "mean": float(s.mean()), "std": float(s.std()),
            **{f"p{q}": float(v) for q, v in zip(PCTL, p)}}


def montage(ds_obj, out_path, min_area, n_per=6):
    """Extract a few objects per dataset and show intensity crop / masked / mask."""
    rows = [d for d in MONTAGE_DS if d in ds_obj.images]
    fig, axes = plt.subplots(len(rows), n_per * 3, figsize=(n_per * 3 * 1.1, len(rows) * 1.2))
    axes = np.atleast_2d(axes)
    for r, name in enumerate(rows):
        imgs, labs = ds_obj.images[name], ds_obj.labels[name]
        found = 0
        for i in range(len(labs)):
            if found >= n_per:
                break
            cc, n = ndi.label(labs[i] > 0)
            for sl in ndi.find_objects(cc):
                if sl is None or found >= n_per:
                    continue
                cid = _slice_label(cc, sl)
                obj = cc[sl] == cid
                if obj.sum() < min_area:
                    continue
                crop = imgs[i][sl].astype(np.float32) / 255.0
                masked = crop * obj
                c = found * 3
                axes[r, c].imshow(crop, cmap="gray", vmin=0, vmax=1)
                axes[r, c + 1].imshow(masked, cmap="gray", vmin=0, vmax=1)
                axes[r, c + 2].imshow(obj, cmap="gray", vmin=0, vmax=1)
                found += 1
        for c in range(n_per * 3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=8)
    for c, t in enumerate(["img", "masked", "mask"] * n_per):
        axes[0, c].set_title(t, fontsize=6)
    fig.suptitle(f"MedSegBench extracted objects (min_area={min_area})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--min_area", type=int, default=16,
                    help="drop components smaller than this many px (vessel/noise fragments)")
    ap.add_argument("--max_per_ds", type=int, default=150,
                    help="cap images scanned per dataset (speed)")
    ap.add_argument("--out_dir", type=Path, default=Path("results/2d/medseg_objects"))
    args = ap.parse_args()

    ds = MedSegBenchDataset(split=args.split, context_size=0, image_size=args.image_size)

    # one row per connected component (per image, per label value), capped per dataset
    seen = {}
    rows = []
    for name, idx, lv in ds.samples:
        if seen.get(name, 0) >= args.max_per_ds:
            continue
        seen[name] = seen.get(name, 0) + 1
        rows += object_rows(ds.images[name][idx], ds.labels[name][idx] == lv,
                            name, idx, lv, args.min_area)
    df = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / f"objects_{args.split}_{args.image_size}.csv", index=False)

    # ── per-dataset report ────────────────────────────────────────────────────
    # objects/image needs the scanned image count per dataset (n rows grouped / imgs seen)
    lines = [f"MedSegBench objects  |  split={args.split}  size={args.image_size}  "
             f"min_area={args.min_area}px  (<= {args.max_per_ds} imgs/ds)",
             f"{len(df):,} objects across {df['dataset'].nunique()} datasets", ""]
    hdr = (f"  {'dataset':>16} {'n_obj':>7} {'obj/img':>8} {'bbox_frac':>10} "
           f"{'area_frac':>10} {'fill':>6} {'int_mean':>9} {'contrast':>9}")
    lines.append(hdr); lines.append("  " + "-" * (len(hdr) - 2))
    for name, g in sorted(df.groupby("dataset"), key=lambda kv: len(kv[1]) / seen[kv[0]]):
        opi = len(g) / seen[name]
        lines.append(f"  {name:>16} {len(g):>7} {opi:>8.1f} "
                     f"{g['bbox_max_frac'].median():>10.3f} {g['area_frac'].median():>10.4f} "
                     f"{g['fill'].median():>6.2f} {g['int_mean'].median():>9.1f} "
                     f"{g['contrast'].median():>9.1f}")
    lines.append("  " + "-" * (len(hdr) - 2))
    for col in ["bbox_max_frac", "area_frac", "fill", "int_mean", "int_std", "contrast"]:
        st = summarize(df[col])
        lines.append(f"  {col:>16}: median={st['p50']:.3f}  p5={st['p5']:.3f}  "
                     f"p95={st['p95']:.3f}  mean={st['mean']:.3f}")
    report = "\n".join(lines)
    print(report)
    (args.out_dir / f"objects_{args.split}_{args.image_size}.txt").write_text(report)

    # ── stat histograms ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    specs = [("bbox_max_px", "object bbox max side (px)", False),
             ("area_frac", "object area / image", True),
             ("fill", "fill = area / bbox_area", False),
             ("int_mean", "mean intensity under mask (0-255)", False),
             ("int_std", "intensity std under mask", False),
             ("contrast", "contrast = obj_mean - ring_mean", False)]
    for ax, (col, title, logx) in zip(axes.ravel(), specs):
        d = df[col][df[col] > 0] if logx else df[col]
        bins = (np.logspace(np.log10(max(d.min(), 1e-5)), np.log10(d.max()), 50)
                if logx else 50)
        ax.hist(d, bins=bins, color="steelblue")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log"); ax.set_xlabel(title); ax.set_ylabel("count (log)")
    fig.suptitle(f"MedSegBench object stats ({len(df):,} objects, {args.split})")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"objects_{args.split}_{args.image_size}_hist.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    montage(ds, args.out_dir / f"objects_{args.split}_{args.image_size}_montage.png",
            args.min_area)
    print(f"\nWrote CSV / report / hist / montage to {args.out_dir}")


if __name__ == "__main__":
    main()

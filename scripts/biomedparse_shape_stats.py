"""
Measure real foreground mask shape/position/size statistics from BiomedParseData,
to inform the controlSynth generator (which currently produces centered,
similar-size blobs). Samples masks balanced across all datasets and reports
distributions for the properties the synth knobs should match:

  position  — normalized centroid (cx, cy) and offset from image center
  size      — foreground area fraction (log-distributed in real data)
  shape     — eccentricity (elongation), solidity (concavity), extent,
              #connected-components, and border contact (masks clipped at frame edge)

Writes a text summary + a histogram figure to results/controlsynth/.

Usage:
    python scripts/biomedparse_shape_stats.py
    python scripts/biomedparse_shape_stats.py --per-dataset 120 --res 256 --split train
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label as cc_label, regionprops

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from src.datasets.biomedparse import DATA_ROOT, _ABSENT_MASK, _discover_sources, _SPLIT_DIRS


def mask_stats(m: np.ndarray) -> dict | None:
    """Per-mask shape/position/size stats. `m` is bool [H, W]. None if empty."""
    H, W = m.shape
    area = int(m.sum())
    if area == 0:
        return None
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() / (H - 1), xs.mean() / (W - 1)            # normalized centroid
    offset = float(np.hypot(cx - 0.5, cy - 0.5))                 # distance from center
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bbox_h, bbox_w = (y1 - y0 + 1), (x1 - x0 + 1)
    extent = area / (bbox_h * bbox_w)                            # fill of bbox
    border = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())

    lbl = cc_label(m)
    n_cc = int(lbl.max())
    # eccentricity + solidity from the largest connected component (regionprops
    # needs a single region; the whole-mask union can be disconnected).
    props = max(regionprops(lbl), key=lambda p: p.area)
    return {
        "area_frac": area / (H * W),
        "cx": float(cx), "cy": float(cy), "offset": offset,
        "extent": float(extent),
        "eccentricity": float(props.eccentricity),
        "solidity": float(props.solidity),
        "n_cc": n_cc,
        "border": border,
    }


def load_mask(path: str, res: int) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != (res, res):
        im = im.resize((res, res), Image.NEAREST)
    return np.asarray(im) > 0


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--per-dataset", type=int, default=100, help="masks sampled per dataset")
    p.add_argument("--res", type=int, default=256, help="resolution masks are measured at")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    img_dir_name, mask_dir_name = _SPLIT_DIRS[args.split]
    sources = _discover_sources(DATA_ROOT, img_dir_name, mask_dir_name, None)
    print(f"{len(sources)} dataset split-dirs found ({args.split})")

    rows = []                                   # flat list of per-mask stat dicts
    per_ds = defaultdict(list)                  # ds -> [stat dict]
    for ds, _img_dir, mask_dir in sources:
        masks = [m for m in sorted(glob.glob(os.path.join(mask_dir, "*.png")))
                 if os.path.basename(m) != _ABSENT_MASK]
        if not masks:
            continue
        if len(masks) > args.per_dataset:
            masks = [masks[i] for i in rng.choice(len(masks), args.per_dataset, replace=False)]
        for mp in masks:
            try:
                s = mask_stats(load_mask(mp, args.res))
            except Exception:
                continue
            if s is None:
                continue
            s["dataset"] = ds
            rows.append(s)
            per_ds[ds].append(s)
        print(f"  {ds:>32}: {len(per_ds[ds])} masks")

    if not rows:
        print("no masks measured"); return
    arr = {k: np.array([r[k] for r in rows], dtype=float)
           for k in ("area_frac", "cx", "cy", "offset", "extent",
                     "eccentricity", "solidity", "n_cc", "border")}
    N = len(rows)

    # ── Text summary ────────────────────────────────────────────────────────────
    lines = []
    lines.append(f"BiomedParse mask shape/position/size stats  (split={args.split}, "
                 f"res={args.res}, {len(per_ds)} datasets, N={N} masks, "
                 f"~{args.per_dataset}/dataset)\n")
    def row(name, a, fmt="{:.3f}"):
        qs = [pct(a, q) for q in (5, 25, 50, 75, 95)]
        return (f"  {name:<14} mean={fmt.format(a.mean()):>8}  "
                + "  ".join(f"p{q}={fmt.format(v)}" for q, v in zip((5, 25, 50, 75, 95), qs)))
    lines.append("POSITION (normalized; 0.5,0.5 = image center)")
    lines.append(row("centroid cx", arr["cx"]))
    lines.append(row("centroid cy", arr["cy"]))
    lines.append(row("center offset", arr["offset"]))
    lines.append(f"    cx std={arr['cx'].std():.3f}  cy std={arr['cy'].std():.3f}  "
                 f"frac offset>0.2: {(arr['offset'] > 0.2).mean():.2%}")
    lines.append("\nSIZE")
    lines.append(row("area_frac", arr["area_frac"], "{:.4f}"))
    lines.append(f"    area_frac log10 span p5..p95: "
                 f"{pct(arr['area_frac'],5):.4f} .. {pct(arr['area_frac'],95):.4f} "
                 f"({pct(arr['area_frac'],95)/max(pct(arr['area_frac'],5),1e-9):.0f}x)")
    lines.append("\nSHAPE")
    lines.append(row("eccentricity", arr["eccentricity"]))
    lines.append(row("solidity", arr["solidity"]))
    lines.append(row("extent", arr["extent"]))
    lines.append(row("n_cc", arr["n_cc"], "{:.2f}"))
    lines.append(f"    frac multi-component (n_cc>1): {(arr['n_cc'] > 1).mean():.2%}")
    lines.append(f"    frac touching frame border:   {arr['border'].mean():.2%}")

    summary = "\n".join(lines)
    print("\n" + summary)

    out_dir = _ROOT / "results" / "controlsynth"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "biomedparse_shape_stats.txt").write_text(summary + "\n")
    # machine-readable percentiles for wiring into synth config
    js = {k: {f"p{q}": pct(arr[k], q) for q in (5, 25, 50, 75, 95)}
          for k in arr}
    js["frac_offset_gt_0.2"] = float((arr["offset"] > 0.2).mean())
    js["frac_multi_cc"] = float((arr["n_cc"] > 1).mean())
    js["frac_border"] = float(arr["border"].mean())
    (out_dir / "biomedparse_shape_stats.json").write_text(json.dumps(js, indent=2))

    # ── Histograms ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].hist(arr["offset"], bins=40, color="steelblue")
    axes[0, 0].set_title("center offset (dist from 0.5,0.5)")
    axes[0, 0].axvline(0, color="red", ls="--", label="synth (centered)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].hexbin(arr["cx"], arr["cy"], gridsize=30, cmap="viridis")
    axes[0, 1].set_title("centroid (cx, cy)"); axes[0, 1].set_xlim(0, 1); axes[0, 1].set_ylim(1, 0)
    axes[0, 1].plot(0.5, 0.5, "r+", ms=12)
    axes[0, 2].hist(np.log10(arr["area_frac"]), bins=40, color="seagreen")
    axes[0, 2].set_title("log10 area_frac")
    axes[1, 0].hist(arr["eccentricity"], bins=40, color="darkorange")
    axes[1, 0].set_title("eccentricity (0=circle, →1 elongated)")
    axes[1, 1].hist(arr["solidity"], bins=40, color="purple")
    axes[1, 1].set_title("solidity (1=convex, low=concave)")
    axes[1, 2].hist(np.clip(arr["n_cc"], 0, 20), bins=20, color="brown")
    axes[1, 2].set_title("n connected components (clipped@20)")
    fig.suptitle(f"BiomedParse real mask stats ({args.split}, N={N}, {len(per_ds)} datasets)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "biomedparse_shape_stats.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {out_dir}/biomedparse_shape_stats.{{txt,json,png}}")


if __name__ == "__main__":
    main()

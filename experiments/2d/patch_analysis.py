"""
Plot a GT-value → predicted-value heatmap from a patch_analysis.csv dump.

Bins both `gt` (soft GT fraction) and `pred` (sigmoid) into N bins over [0, 1]
and shows the 2D count histogram. A perfectly calibrated model puts mass on the
diagonal.

Usage:
    python experiments/2d/patch_analysis.py                       # default CSV
    python experiments/2d/patch_analysis.py --csv path/to.csv --bins 10
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("results/2d/pfn_seg/patch_analysis.csv"))
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["gt", "pred"])
    edges   = np.linspace(0.0, 1.0, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _, _ = np.histogram2d(df["gt"], df["pred"], bins=[edges, edges])

    # Marginals are extremely zero-heavy, so raw counts hide all structure.
    # Column-normalise to the conditional P(pred | gt): each GT bin sums to 1,
    # revealing where predictions land for each GT level (a calibration map).
    col_tot = counts.sum(axis=1, keepdims=True)
    cond    = np.divide(counts, col_tot, out=np.zeros_like(counts), where=col_tot > 0)

    # mean predicted value per GT bin (calibration curve)
    mean_pred = np.divide((counts * centers[None, :]).sum(axis=1), col_tot[:, 0],
                          out=np.full(args.bins, np.nan), where=col_tot[:, 0] > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        cond.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
        cmap="viridis", vmin=0, vmax=1,
    )
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.6)                   # ideal
    ax.plot(centers, mean_pred, "o-", color="red", lw=1.5, ms=4,
            label="mean pred | gt")                                   # actual
    ax.set_xlabel("GT value (soft fraction)")
    ax.set_ylabel("Predicted value (sigmoid)")
    ax.set_title(f"P(pred | gt)  ({len(df):,} patches, {args.bins} bins)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(pred | gt bin)")

    out = args.out or args.csv.with_name(args.csv.stem + "_heatmap.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")

    # marginal distributions of gt and pred
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(df["gt"],   bins=edges, alpha=0.6, label="GT (soft fraction)")
    ax2.hist(df["pred"], bins=edges, alpha=0.6, label="Pred (sigmoid)")
    ax2.set_yscale("log")
    ax2.set_xlabel("value")
    ax2.set_ylabel("patch count (log)")
    ax2.set_title(f"GT vs Pred value distribution  ({len(df):,} patches)")
    ax2.legend()
    out_hist = out.with_name(args.csv.stem + "_hist.png")
    fig2.savefig(out_hist, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_hist}")


if __name__ == "__main__":
    main()

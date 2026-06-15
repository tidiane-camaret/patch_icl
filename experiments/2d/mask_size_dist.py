"""
Mask-size distribution of MedSegBench (val split, image_size 128).

For every in-context sample (one binary mask = labels[i] == label_value), compute
the foreground size, both in pixels and as a fraction of the 128x128 image area
(16384 px). Reports per-dataset and pooled summary stats, dumps a per-sample CSV,
and saves a histogram of the fg-fraction distribution.

Usage:
    python experiments/2d/mask_size_dist.py
    python experiments/2d/mask_size_dist.py --image_size 128 --split val
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.medsegbench import MedSegBenchDataset

PCTL = [0, 5, 25, 50, 75, 95, 100]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--out_dir", type=Path, default=Path("results/2d/mask_size"))
    args = ap.parse_args()

    area = args.image_size ** 2
    ds = MedSegBenchDataset(split=args.split, context_size=0,
                            image_size=args.image_size)

    # per-sample fg pixel count (no __getitem__ — work straight off label arrays)
    rows = []
    for name, idx, lv in ds.samples:
        fg = int((ds.labels[name][idx] == lv).sum())
        rows.append((name, idx, lv, fg, fg / area))
    df = pd.DataFrame(rows, columns=["dataset", "sample_idx", "label_value",
                                     "fg_pixels", "fg_frac"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"mask_sizes_{args.split}_{args.image_size}.csv"
    df.to_csv(csv_path, index=False)

    def summarize(s: pd.Series) -> dict:
        p = np.percentile(s, PCTL)
        return {"n": len(s), "mean": s.mean(), "std": s.std(),
                **{f"p{q}": v for q, v in zip(PCTL, p)}}

    # ── report (fraction of image area) ───────────────────────────────────────
    lines = [f"MedSegBench mask-size distribution  |  split={args.split}  "
             f"size={args.image_size} (area={area} px)",
             f"{len(df):,} masks across {df['dataset'].nunique()} datasets",
             "fg_frac = foreground pixels / image area", ""]
    hdr = (f"  {'dataset':>20} {'n':>7}  {'mean':>7} {'std':>7}  "
           f"{'min':>7} {'p5':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7} {'max':>7}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    def fmt(name, st):
        return (f"  {name:>20} {st['n']:>7}  {st['mean']:>7.4f} {st['std']:>7.4f}  "
                f"{st['p0']:>7.4f} {st['p5']:>7.4f} {st['p25']:>7.4f} {st['p50']:>7.4f} "
                f"{st['p75']:>7.4f} {st['p95']:>7.4f} {st['p100']:>7.4f}")

    per = {name: summarize(g["fg_frac"]) for name, g in df.groupby("dataset")}
    for name in sorted(per, key=lambda k: per[k]["p50"]):
        lines.append(fmt(name, per[name]))
    lines.append("  " + "-" * (len(hdr) - 2))
    lines.append(fmt("ALL (pooled)", summarize(df["fg_frac"])))

    # pooled stats also in raw pixels, for reference
    st_px = summarize(df["fg_pixels"])
    lines.append("")
    lines.append(f"  pooled fg_pixels:  mean={st_px['mean']:.1f}  "
                 f"median={st_px['p50']:.0f}  p5={st_px['p5']:.0f}  "
                 f"p95={st_px['p95']:.0f}  min={st_px['p0']:.0f}  max={st_px['p100']:.0f}")

    report = "\n".join(lines)
    print(report)
    (args.out_dir / f"mask_sizes_{args.split}_{args.image_size}.txt").write_text(report)

    # ── histogram (pooled fg_frac, log-spaced bins + log y) ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(df["fg_frac"], bins=np.linspace(0, 1, 51), color="steelblue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("fg_frac = mask px / image area")
    axes[0].set_ylabel("mask count (log)")
    axes[0].set_title(f"pooled mask-size distribution ({len(df):,} masks)")

    pos = df["fg_frac"][df["fg_frac"] > 0]
    log_bins = np.logspace(np.log10(pos.min()), np.log10(1.0), 51)
    axes[1].hist(pos, bins=log_bins, color="indianred")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("fg_frac (log)")
    axes[1].set_ylabel("mask count (log)")
    axes[1].set_title("log-scale (nonzero masks)")
    fig.tight_layout()
    fig_path = args.out_dir / f"mask_sizes_{args.split}_{args.image_size}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {csv_path}\nWrote {fig_path}")


if __name__ == "__main__":
    main()

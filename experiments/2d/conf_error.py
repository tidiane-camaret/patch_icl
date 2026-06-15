"""
Focused analysis of the relationship between prediction confidence
`conf = |pred - 0.5|` and per-patch `|error|`, from a patch_analysis.csv dump.

The "|pred-0.5| inv" selection rule (used for refinement sampling) ranks patches
by *smallest* conf first — i.e. predictions nearest 0.5 are the most uncertain.
This script:
  1. plots the (conf, |error|) relationship as a 2D histogram (+ mean curve), and
  2. tabulates the % of total |error| captured by selecting the top-k% most
     uncertain patches (|pred-0.5| inv), per dataset.

Usage:
    python experiments/2d/conf_error.py --csv path/to/patch_analysis.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRACS = (0.05, 0.1, 0.2, 0.3, 0.5)


def capture_topk(conf: np.ndarray, abs_err: np.ndarray, fracs=FRACS) -> dict:
    """Fraction of total |error| captured by the top-k% lowest-conf patches."""
    order = np.argsort(conf)              # ascending conf = most uncertain first
    cum   = np.cumsum(abs_err[order])
    total = cum[-1]
    n = len(conf)
    if total <= 0:
        return {f: float("nan") for f in fracs}
    return {f: cum[min(int(f * n), n - 1)] / total for f in fracs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("results/2d/pfn_seg_low_res_loss/"
                                 "pfn_seg_P8_e256_l6_k3_think8/patch_analysis.csv"))
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["dataset", "pred", "error"])
    df["conf"]    = (df["pred"] - 0.5).abs()      # 0 = uncertain, 0.5 = confident
    df["abs_err"] = df["error"].abs()
    N = len(df)
    out_dir = args.csv.parent

    # ── 1. Histogram of the (conf, |error|) relationship ──────────────────────
    cb = args.bins
    conf_edges = np.linspace(0.0, 0.5, cb + 1)
    err_edges  = np.linspace(0.0, df["abs_err"].max(), cb + 1)
    counts, _, _ = np.histogram2d(df["conf"], df["abs_err"],
                                  bins=[conf_edges, err_edges])
    # column-normalise to P(|err| | conf): each conf bin sums to 1, so the
    # zero-heavy marginal (most patches are confident bg, ~0 error) doesn't
    # swamp the structure.
    col_tot = counts.sum(axis=1, keepdims=True)
    cond    = np.divide(counts, col_tot, out=np.zeros_like(counts), where=col_tot > 0)
    conf_c  = 0.5 * (conf_edges[:-1] + conf_edges[1:])
    err_c   = 0.5 * (err_edges[:-1]  + err_edges[1:])
    mean_err = np.divide((counts * err_c[None, :]).sum(axis=1), col_tot[:, 0],
                         out=np.full(cb, np.nan), where=col_tot[:, 0] > 0)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(cond.T, origin="lower",
                   extent=[0, 0.5, 0, float(df["abs_err"].max())],
                   aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.plot(conf_c, mean_err, "o-", color="red", lw=1.5, ms=4, label="mean |err| | conf")
    ax.set_xlabel("conf = |pred − 0.5|   (0 = uncertain, 0.5 = confident)")
    ax.set_ylabel("|error|")
    ax.set_title(f"P(|error| | conf)   ({N:,} patches)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(|err| | conf bin)")
    out_heat = out_dir / "conf_error_hist.png"
    fig.savefig(out_heat, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_heat}")

    # ── 2. Per-dataset top-k% capture (|pred-0.5| inv) ────────────────────────
    fracs = FRACS
    rows = []
    for ds, g in df.groupby("dataset"):
        cap = capture_topk(g["conf"].values, g["abs_err"].values, fracs)
        rows.append((ds, len(g), float(g["abs_err"].sum()), *[cap[f] for f in fracs]))
    cap_all = capture_topk(df["conf"].values, df["abs_err"].values, fracs)

    cols = ["dataset", "n_patches", "err_mass"] + [f"top{int(f*100)}%" for f in fracs]
    cap_df = pd.DataFrame(rows, columns=cols).sort_values("top10%", ascending=False)
    cap_csv = out_dir / "conf_error_capture_by_dataset.csv"
    cap_df.to_csv(cap_csv, index=False)

    # printed report
    lines = [f"conf=|pred-0.5| inv  → % of |error| captured in top-k% most-uncertain patches",
             f"{args.csv}   ({N:,} patches, {len(rows)} datasets)", ""]
    hdr = f"  {'dataset':>20} {'n':>9} " + " ".join(f"{c:>7}" for c in cols[3:])
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for _, r in cap_df.iterrows():
        lines.append(f"  {r['dataset']:>20} {int(r['n_patches']):>9} " +
                     " ".join(f"{r[c]:6.1%}" for c in cols[3:]))
    lines.append("  " + "-" * (len(hdr) - 2))
    lines.append(f"  {'ALL (pooled)':>20} {N:>9} " +
                 " ".join(f"{cap_all[f]:6.1%}" for f in fracs))
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "conf_error_capture_by_dataset.txt").write_text(report)
    print(f"\nWrote {cap_csv}")


if __name__ == "__main__":
    main()

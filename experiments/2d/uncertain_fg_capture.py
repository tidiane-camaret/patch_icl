"""
Final-resolution fg/bg pixel coverage when refining the top-k% most-uncertain
low-res patches (resolution 16) from a patch_analysis.csv dump.

Each low-res patch covers a fixed (native/16)^2 block of final-res pixels, so a
patch holds `gt*ppx` foreground and `(1-gt)*ppx` background pixels (ppx cancels
in the ratios below). For each 16x16 sample map we pick its top-k% most uncertain
patches (smallest |pred-0.5|) — i.e. those a refiner would re-process — and ask
what fraction of the final-res GT fg (and bg) pixels fall inside them.

  captured_fg(k) = Σ_selected gt   / Σ_all gt
  captured_bg(k) = Σ_selected(1-gt)/ Σ_all(1-gt)

Random selection captures ~k% of each; uncertain selection should over-cover fg
(boundary/object cells) and under-cover bg.

Usage:
    python experiments/2d/uncertain_fg_capture.py --csv path/to/patch_analysis.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRACS = (0.05, 0.1, 0.2, 0.3, 0.5)
KEY = ["dataset", "label_value", "sample_idx"]


def per_sample_capture(g: pd.DataFrame, fracs) -> dict:
    """Per-sample top-k% uncertain selection → captured fg & bg pixel mass.

    Returns dict frac -> (fg_captured, fg_total, bg_captured, bg_total) summed
    over the samples in `g` (so callers can pool by simple addition)."""
    conf = (g["pred"].values - 0.5).__abs__()      # 0 = uncertain
    gt   = g["gt"].values
    sid  = g["_sid"].values                         # contiguous sample id
    out = {f: np.zeros(4) for f in fracs}
    # iterate per sample (each has 256 patches)
    for s in np.unique(sid):
        m = sid == s
        c, ggt = conf[m], gt[m]
        n = len(c)
        order = np.argsort(c)                       # most uncertain first
        bgt = 1.0 - ggt
        fg_tot, bg_tot = ggt.sum(), bgt.sum()
        cum_fg = np.cumsum(ggt[order])
        cum_bg = np.cumsum(bgt[order])
        for f in fracs:
            k = min(int(round(f * n)), n) - 1
            k = max(k, 0)
            out[f] += (cum_fg[k], fg_tot, cum_bg[k], bg_tot)
    return out


def fmt_block(name, agg, fracs):
    fg = [agg[f][0] / agg[f][1] if agg[f][1] > 0 else float("nan") for f in fracs]
    bg = [agg[f][2] / agg[f][3] if agg[f][3] > 0 else float("nan") for f in fracs]
    return name, fg, bg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("results/2d/pfn_seg_low_res_loss/"
                                 "pfn_seg_P8_e256_l6_k3_think8/patch_analysis.csv"))
    args = ap.parse_args()
    fracs = FRACS

    df = pd.read_csv(args.csv, usecols=KEY + ["pred", "gt"])
    df["_sid"] = df.groupby(KEY).ngroup()
    out_dir = args.csv.parent

    # per-dataset aggregation (pool samples within dataset)
    per_ds = {}
    total = {f: np.zeros(4) for f in fracs}
    for ds, g in df.groupby("dataset"):
        agg = per_sample_capture(g, fracs)
        per_ds[ds] = agg
        for f in fracs:
            total[f] += agg[f]

    # ── report ────────────────────────────────────────────────────────────────
    ks = [f"{int(f*100)}%" for f in fracs]
    lines = [
        f"Final-res fg/bg pixel capture by top-k% uncertain patches (res 16)",
        f"{args.csv}",
        f"per-sample top-k% selection by |pred-0.5| inv, pooled.  random ≈ k%.",
        "",
        f"  {'dataset':>20} {'n':>6} | " +
        "fg captured @ " + " ".join(f"{k:>6}" for k in ks) +
        "  ||  bg captured @ " + " ".join(f"{k:>6}" for k in ks),
    ]
    hdr_len = len(lines[-1])
    lines.append("  " + "-" * (hdr_len - 2))

    nrows = df.groupby("dataset")["_sid"].nunique()
    for ds in sorted(per_ds, key=lambda d: -(per_ds[d][fracs[1]][0] / max(per_ds[d][fracs[1]][1], 1e-9))):
        _, fg, bg = fmt_block(ds, per_ds[ds], fracs)
        lines.append(f"  {ds:>20} {int(nrows[ds]):>6} | " +
                     "              " + " ".join(f"{v:6.1%}" for v in fg) +
                     "  ||                  " + " ".join(f"{v:6.1%}" for v in bg))
    lines.append("  " + "-" * (hdr_len - 2))
    _, fg, bg = fmt_block("ALL", total, fracs)
    lines.append(f"  {'ALL (pooled)':>20} {len(df['_sid'].unique()):>6} | " +
                 "              " + " ".join(f"{v:6.1%}" for v in fg) +
                 "  ||                  " + " ".join(f"{v:6.1%}" for v in bg))

    report = "\n".join(lines)
    print(report)
    (out_dir / "uncertain_fg_capture.txt").write_text(report)

    # ── figure: fg & bg capture vs k (pooled) ────────────────────────────────
    fine = np.linspace(0.02, 0.6, 30)
    tot = {f: np.zeros(4) for f in fine}
    for _, g in df.groupby("dataset"):
        agg = per_sample_capture(g, tuple(fine))
        for f in fine:
            tot[f] += agg[f]
    fg_c = [tot[f][0] / tot[f][1] for f in fine]
    bg_c = [tot[f][2] / tot[f][3] for f in fine]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fine * 100, np.array(fg_c) * 100, "o-", color="crimson", ms=3, label="fg pixels captured")
    ax.plot(fine * 100, np.array(bg_c) * 100, "o-", color="steelblue", ms=3, label="bg pixels captured")
    ax.plot([0, 60], [0, 60], "k--", lw=1, alpha=0.5, label="random (=k%)")
    ax.set_xlabel("top-k% most uncertain patches selected (per sample)")
    ax.set_ylabel("% of final-res GT pixels captured")
    ax.set_title("fg/bg pixel coverage of uncertain-patch selection (res 16)")
    ax.legend(); ax.grid(alpha=0.3)
    fig_path = out_dir / "uncertain_fg_capture.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir/'uncertain_fg_capture.txt'}\nWrote {fig_path}")


if __name__ == "__main__":
    main()

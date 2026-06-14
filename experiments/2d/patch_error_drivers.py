"""
Analyse what drives per-patch error in a patch_analysis.csv dump.

Goal context: we want to sample patches from the predicted soft map to refine
them with a second model. So we care not only about *what* drives error but
whether those drivers are **observable at inference** (pred-based, no GT) or are
**oracle** signals (need the target GT: gt value, gt_size, ctx_dice).

Produces a printed report and a few figures next to the CSV.

Usage:
    python experiments/2d/patch_error_drivers.py --csv path/to/patch_analysis.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Best-effort dataset → imaging modality map (from MedSegBench dataset identities;
# NOT read from the data — correct here if any are off).
MODALITY = {
    "abdomenus": "ultrasound", "busi": "ultrasound", "fhpsaop": "ultrasound",
    "ultrasoundnerve": "ultrasound", "usforkidney": "ultrasound",
    "bbbc010": "microscopy", "brifiseg": "microscopy", "cellnuclei": "microscopy",
    "deepbacs": "microscopy", "dynamicnuclear": "microscopy", "monusac": "microscopy",
    "nuclei": "microscopy", "nuset": "microscopy", "tnbcnuclei": "microscopy",
    "wbc": "microscopy", "yeaz": "microscopy",
    "bkai-igh": "endoscopy", "kvasir": "endoscopy", "m2caiseg": "endoscopy",
    "polypgen": "endoscopy", "robotool": "endoscopy",
    "chasedb1": "fundus", "drive": "fundus", "idrib": "fundus",
    "chuac": "xray", "covid19radio": "xray", "covidquex": "xray",
    "dca1": "xray", "pandental": "xray",
    "mosmedplus": "ct", "promise12": "mri", "cystoidfluid": "oct",
    "isic2016": "dermoscopy", "isic2018": "dermoscopy", "uwaterlooskincancer": "dermoscopy",
}


def bin_table(df, col, target, bins=10, qcut=False, label=None):
    """Mean target (+count) per bin of `col`. Returns a formatted string."""
    if qcut:
        b = pd.qcut(df[col], q=bins, duplicates="drop")
    else:
        b = pd.cut(df[col], bins=np.linspace(df[col].min(), df[col].max(), bins + 1),
                   include_lowest=True)
    g = df.groupby(b, observed=True)[target].agg(["mean", "count"])
    out = [f"  {label or col:>14} | mean|err|  count"]
    for iv, row in g.iterrows():
        out.append(f"  {str(iv):>14} | {row['mean']:.4f}   {int(row['count']):>8}")
    return "\n".join(out)


def capture_curve(score, abs_err, fracs=(0.05, 0.1, 0.2, 0.3, 0.5)):
    """Fraction of total |error| captured by selecting top-k% patches by `score`."""
    order = np.argsort(-score)
    cum = np.cumsum(abs_err[order])
    total = cum[-1]
    n = len(score)
    return {f: cum[min(int(f * n), n - 1)] / total for f in fracs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("results/2d/pfn_seg_low_res_loss/"
                                 "pfn_seg_P8_e256_l6_k3_think8/patch_analysis.csv"))
    ap.add_argument("--sample", type=int, default=500_000,
                    help="rows subsampled for the tree models")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    df = pd.read_csv(args.csv)
    df["modality"]     = df["dataset"].map(MODALITY).fillna("other")
    df["abs_err"]      = df["error"].abs()
    df["boundaryness"] = 4 * df["gt"] * (1 - df["gt"])      # oracle: 0 at pure bg/fg, 1 at gt=.5
    df["pred_uncert"]  = 4 * df["pred"] * (1 - df["pred"])  # observable proxy for difficulty
    df["log_gt_size"]  = np.log1p(df["gt_size"])
    N = len(df)
    rep = [f"patch error-driver analysis  |  {args.csv}", f"{N:,} patches", ""]

    # ── 1. Global error budget by patch type ─────────────────────────────────
    typ = np.where(df["gt"] < 0.05, "bg(0)",
          np.where(df["gt"] > 0.95, "fg(1)", "boundary"))
    rep.append("── error budget by patch type (gt<.05 / >.95 / between) ──")
    g = df.groupby(typ)["abs_err"].agg(["mean", "count"])
    g["err_mass"] = df.groupby(typ)["abs_err"].sum()
    g["mass_share"] = g["err_mass"] / g["err_mass"].sum()
    g["count_share"] = g["count"] / N
    for name, r in g.iterrows():
        rep.append(f"  {name:>9}: mean|err|={r['mean']:.4f}  "
                   f"count={int(r['count']):>9} ({r['count_share']:5.1%})  "
                   f"error-mass={r['mass_share']:5.1%}")
    rep.append("")

    # ── 2. Univariate driver tables ──────────────────────────────────────────
    rep.append("── mean |error| vs each factor (binned) ──")
    rep.append(bin_table(df, "gt",           "abs_err", 10, label="gt"))
    rep.append(bin_table(df, "pred",         "abs_err", 10, label="pred"))
    rep.append(bin_table(df, "boundaryness", "abs_err", 10, label="boundaryness"))
    rep.append(bin_table(df, "pred_uncert",  "abs_err", 10, label="pred_uncert"))
    rep.append(bin_table(df, "log_gt_size",  "abs_err", 8,  qcut=True, label="gt_size(q)"))
    rep.append(bin_table(df, "ctx_dice",     "abs_err", 8,  qcut=True, label="ctx_dice(q)"))
    rep.append("")

    # ── 3. Spearman correlations with |error| ────────────────────────────────
    rep.append("── Spearman ρ(|error|, factor)  [+ = factor raises error] ──")
    for c in ["boundaryness", "pred_uncert", "gt", "pred", "gt_size",
              "log_gt_size", "ctx_dice"]:
        rho = spearmanr(df["abs_err"], df[c]).statistic
        rep.append(f"  {c:>14}: {rho:+.3f}")
    rep.append("")

    # ── 4. Per-modality and per-dataset hardness ─────────────────────────────
    rep.append("── mean |error| by modality (sorted hardest first) ──")
    gm = df.groupby("modality")["abs_err"].agg(["mean", "count"])
    gm["err_mass_share"] = df.groupby("modality")["abs_err"].sum() / df["abs_err"].sum()
    for name, r in gm.sort_values("mean", ascending=False).iterrows():
        rep.append(f"  {name:>11}: mean|err|={r['mean']:.4f}  "
                   f"n={int(r['count']):>9}  err-mass={r['err_mass_share']:5.1%}")
    rep.append("")
    rep.append("── 10 hardest datasets by mean |error| ──")
    gd = df.groupby("dataset")["abs_err"].agg(["mean", "count"])
    for name, r in gd.sort_values("mean", ascending=False).head(10).iterrows():
        rep.append(f"  {name:>20}: mean|err|={r['mean']:.4f}  n={int(r['count']):>8}")
    rep.append("")

    # ── 5. Multivariate importance (HistGBR) ─────────────────────────────────
    sub = df.sample(min(args.sample, N), random_state=0).copy()
    for c in ["modality", "dataset"]:
        sub[c] = sub[c].astype("category")
    # NB: error ≡ pred − gt by construction, so a model given BOTH pred and gt
    # reconstructs |error| trivially (R²≈1, meaningless). We therefore fit two
    # non-circular models: (a) intrinsic difficulty from oracle factors that do
    # NOT include pred; (b) what a refiner can actually see at inference.
    intrinsic_feats  = ["gt", "boundaryness", "gt_size", "ctx_dice"]
    observable_feats = ["pred", "pred_uncert", "patch_i", "patch_j"]
    cat_feats        = ["modality"]

    def fit_report(feats, cats, tag):
        X = sub[feats + cats]
        y = sub["abs_err"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=6,
            categorical_features=cats or None, random_state=0,
        ).fit(Xtr, ytr)
        r2 = m.score(Xte, yte)
        pi = permutation_importance(m, Xte.iloc[:40_000], yte[:40_000],
                                    n_repeats=3, random_state=0, n_jobs=-1)
        imp = sorted(zip(feats + cats, pi.importances_mean), key=lambda t: -t[1])
        rep.append(f"── HistGBR predicting |error|  [{tag}]  test R²={r2:.3f} ──")
        for name, v in imp:
            rep.append(f"  {name:>14}: {v:.5f}")
        rep.append("")
        return r2, imp

    r2_obs, _ = fit_report(observable_feats, cat_feats, "OBSERVABLE-only (no GT, what a refiner sees)")
    r2_int, _ = fit_report(intrinsic_feats, cat_feats, "INTRINSIC difficulty (oracle, excludes pred)")
    rep.append(f"  → from inference-observable signals alone, |error| is predictable to "
               f"R²={r2_obs:.3f}; intrinsic oracle difficulty reaches R²={r2_int:.3f}.")
    rep.append("")

    # ── 6. Patch-selection efficiency (for refinement sampling) ──────────────
    rep.append("── error captured by selecting top-k% patches (for refinement) ──")
    rep.append("    score \\ top-k%       5%     10%     20%     30%     50%")
    ae = df["abs_err"].values
    for tag, score in [("pred_uncert (obs)", df["pred_uncert"].values),
                       ("|pred-0.5| inv   ", -np.abs(df["pred"].values - 0.5)),
                       ("boundaryness(orac)", df["boundaryness"].values),
                       ("oracle |error|    ", ae),
                       ("random            ", rng.standard_normal(N))]:
        cap = capture_curve(score, ae)
        rep.append(f"    {tag} " + "  ".join(f"{cap[f]:6.1%}" for f in (0.05, 0.1, 0.2, 0.3, 0.5)))
    rep.append("")

    report = "\n".join(rep)
    print(report)
    out_dir = args.csv.parent
    (out_dir / "patch_error_drivers.txt").write_text(report)

    # ── Figures ──────────────────────────────────────────────────────────────
    edges = np.linspace(0, 1, 11)
    # (a) 2D mean|error| over (gt, pred)
    gi = np.clip(np.digitize(df["gt"], edges) - 1, 0, 9)
    pi_ = np.clip(np.digitize(df["pred"], edges) - 1, 0, 9)
    grid = np.full((10, 10), np.nan)
    for a in range(10):
        for b in range(10):
            mask = (gi == a) & (pi_ == b)
            if mask.any():
                grid[b, a] = df["abs_err"].values[mask].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="magma")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
    ax.set_xlabel("GT (soft fraction)"); ax.set_ylabel("pred (sigmoid)")
    ax.set_title("mean |error| over (gt, pred)")
    fig.colorbar(im, ax=ax, label="mean |error|")
    fig.savefig(out_dir / "err_gt_pred_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # (b) mean|error| vs pred_uncert + per-modality bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ub = pd.cut(df["pred_uncert"], np.linspace(0, 1, 11), include_lowest=True)
    gu = df.groupby(ub, observed=True)["abs_err"].mean()
    axes[0].plot([iv.mid for iv in gu.index], gu.values, "o-")
    axes[0].set_xlabel("pred_uncert = 4·p·(1-p)  (observable)")
    axes[0].set_ylabel("mean |error|"); axes[0].set_title("observable difficulty vs error")
    gm2 = gm.sort_values("mean", ascending=False)
    axes[1].barh(gm2.index[::-1], gm2["mean"].values[::-1], color="steelblue")
    axes[1].set_xlabel("mean |error|"); axes[1].set_title("error by modality")
    fig.tight_layout()
    fig.savefig(out_dir / "err_uncert_modality.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote report + 2 figures to {out_dir}")


if __name__ == "__main__":
    main()

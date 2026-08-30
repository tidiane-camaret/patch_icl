"""
Real-data intra-cohort intensity variance analysis for TotalSegmentator CT -- step towards
replacing the synth_gmm generator's single per-class sd (src.gpu_gmm_intensity /
SynthGmmMaisiDataset.assemble: sd[c] = sqrt(U(0, var_max)), ONE number shared by every
member of a cohort AND every voxel of that class) with something that reflects how real
data actually splits variance for class c into two very different quantities:

  between_subj_std_hu : real subject-to-subject spread of a class's MEAN HU -- the true
                         "intra-cohort" (member-to-member) quantity we don't model at all
                         today (mu[c] is drawn once and shared verbatim by every member).
  within_subj_voxel_std_hu : real within-ONE-scan voxel-to-voxel texture spread -- the
                         thing sd[c] actually models today (per-voxel noise, same value
                         reused for every member).

Both columns already exist in totalseg_intensity_class_table.csv (written by
analyze_totalseg_intensity.py); this script only re-derives their RATIO (between/within,
unit-free -- survives the fact that synth intensities live on an arbitrary 0-255 scale, not
real HU) and characterizes its distribution: overall spread, whether it lines up with the
intensity-only clusters found earlier (totalseg_intensity_factors_all_debiased.npz) or with
simple anatomical families (bone/vascular/organ/lung/muscle, name-pattern only), and whether
it's an artifact of small per-class subject counts.

  .venv_blackwell/bin/python experiments/3d/synth_task_generation/analyze_intracohort_variance.py

Outputs:
  results/synth_task_gen/intracohort_variance_ratio.csv
  results/synth_task_gen/intracohort_variance_{hist,by_cluster,by_family,vs_n}.png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/synth_task_gen")
CLASS_TABLE = OUT / "totalseg_intensity_class_table.csv"
DEBIASED = OUT / "totalseg_intensity_factors_all_debiased.npz"
MIN_N = 100    # subjects present -- below this, between_subj_std is too noisy to trust


def _family(name):
    """Coarse anatomical family from TotalSeg name PATTERN only (no MAISI/registry lookup) --
    just to sanity-check whether the ratio lines up with obvious tissue-type groupings."""
    if re.match(r"(rib_|vertebrae_|sternum|costal_cartilages|sacrum|hip_|clavicula|scapula|"
                r"humerus|femur|skull)", name):
        return "bone"
    if re.search(r"(aorta|artery|vein|vena_cava|trunk|iliac_|atrial_appendage|heart)", name):
        return "vascular/cardiac"
    if re.match(r"lung_", name):
        return "lung"
    if re.match(r"(autochthon|iliopsoas|gluteus)", name):
        return "muscle"
    if re.match(r"(kidney|liver|spleen|pancreas|adrenal|gallbladder|stomach|colon|duodenum|"
                r"small_bowel|esophagus|urinary_bladder|prostate|thyroid)", name):
        return "organ"
    return "other"


def main():
    df = pd.read_csv(CLASS_TABLE)
    df = df.rename(columns={"between_subj_std_hu": "between", "within_subj_voxel_std_hu": "within"})
    df["ratio"] = df["between"] / df["within"]
    df["family"] = df["class"].map(_family)

    z = np.load(DEBIASED, allow_pickle=True)
    core = list(z["core_classes"])
    cluster_of = z["cluster_of_class"]
    cluster_labels = z["cluster_labels"]
    cls_to_cluster = {c: cluster_labels[cluster_of[i]] for i, c in enumerate(core)}
    df["intensity_cluster"] = df["class"].map(cls_to_cluster)

    good = df[(df["n_present"] >= MIN_N) & df["ratio"].notna()].copy()
    print(f"{len(good)}/{len(df)} classes with n_present >= {MIN_N}")

    good = good.sort_values("ratio", ascending=False)
    out_csv = OUT / "intracohort_variance_ratio.csv"
    good[["class", "n_present", "mean_hu", "between", "within", "ratio", "family",
         "intensity_cluster"]].to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    print(f"\n{'class':<28}{'n':>6}{'between':>9}{'within':>9}{'ratio':>7}  family / cluster")
    for _, r in pd.concat([good.head(12), good.tail(12)]).iterrows():
        print(f"{r['class']:<28}{r['n_present']:>6}{r['between']:>9.1f}{r['within']:>9.1f}"
              f"{r['ratio']:>7.2f}  {r['family']} / {r['intensity_cluster']}")

    print(f"\noverall ratio: median={good['ratio'].median():.3f} "
          f"mean={good['ratio'].mean():.3f} std={good['ratio'].std():.3f} "
          f"[{good['ratio'].min():.2f}, {good['ratio'].max():.2f}]")

    # --- by anatomical family (coarse, name-pattern sanity check) -----------------------
    print(f"\n{'family':<20}{'n_classes':>10}{'ratio_mean':>12}{'ratio_std':>11}")
    fam_stats = good.groupby("family")["ratio"].agg(["count", "mean", "std"]).sort_values("mean")
    for fam, row in fam_stats.iterrows():
        print(f"{fam:<20}{int(row['count']):>10}{row['mean']:>12.3f}{row['std']:>11.3f}")

    # --- by the EXISTING intensity-only cluster (from the correlation study) ------------
    clu_stats = (good.groupby("intensity_cluster")["ratio"].agg(["count", "mean", "std"])
                 .query("count >= 2").sort_values("mean"))
    print(f"\n{'intensity_cluster':<32}{'n_classes':>10}{'ratio_mean':>12}{'ratio_std':>11}")
    for clu, row in clu_stats.iterrows():
        print(f"{clu:<32}{int(row['count']):>10}{row['mean']:>12.3f}{row['std']:>11.3f}")
    # within-cluster std vs. overall std: does the correlation-clustering ALSO explain ratio?
    within_clu_std = np.sqrt((clu_stats["std"] ** 2 * (clu_stats["count"] - 1)).sum()
                             / (clu_stats["count"] - 1).sum())
    print(f"\npooled within-intensity-cluster ratio std = {within_clu_std:.3f} "
          f"vs. overall ratio std = {good['ratio'].std():.3f} "
          f"(smaller = clusters explain some of the ratio spread too)")

    # --- confound check: does low n_present inflate the ratio via a noisy between_std? --
    corr_n = np.corrcoef(good["n_present"], good["ratio"])[0, 1]
    print(f"\ncorr(n_present, ratio) among kept classes = {corr_n:.3f} "
          f"(near 0 = ratio spread is not just a small-n artifact)")

    _plot(good, fam_stats, clu_stats)


def _plot(good, fam_stats, clu_stats):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(good["ratio"], bins=25, color="#4C72B0", edgecolor="white")
    ax.axvline(good["ratio"].median(), color="k", ls="--", lw=1, label="median")
    ax.set_xlabel("between_subj_std / within_subj_voxel_std")
    ax.set_ylabel("# classes")
    ax.set_title("Real intra-cohort variance ratio (TotalSeg CT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "intracohort_variance_hist.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(fam_stats))))
    ax.barh(fam_stats.index, fam_stats["mean"], xerr=fam_stats["std"], color="#55A868")
    ax.set_xlabel("ratio (mean ± std)")
    ax.set_title("By anatomical family (name-pattern only)")
    fig.tight_layout()
    fig.savefig(OUT / "intracohort_variance_by_family.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(clu_stats))))
    ax.barh(clu_stats.index, clu_stats["mean"], xerr=clu_stats["std"], color="#C44E52")
    ax.set_xlabel("ratio (mean ± std)")
    ax.set_title("By intensity-only cluster (correlation study)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "intracohort_variance_by_cluster.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(good["n_present"], good["ratio"], s=10, alpha=0.6)
    ax.set_xlabel("n_present (subjects)")
    ax.set_ylabel("ratio")
    ax.set_title("Ratio vs. sample size (confound check)")
    fig.tight_layout()
    fig.savefig(OUT / "intracohort_variance_vs_n.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote {OUT}/intracohort_variance_{{hist,by_family,by_cluster,vs_n}}.png")


if __name__ == "__main__":
    main()

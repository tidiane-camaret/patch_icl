"""
Inter-subject distance structure of the GMM mask bank, per CohortSampler component
(span / fov / spacing / by_class_size). Answers "how do subjects cluster, and which axis
drives cohort selection" before we tune cohort diversity for training.

Builds the components straight from a CohortSampler, so the by_class_size term reflects the
ACTUAL sampler (air+body dropped, shared-core restricted, fraction|volume mode). Uses only
index.pkl metadata — safe against a bank whose mask FILES are mid-rewrite.

  .venv_thor/bin/python experiments/3d/synth_task_generation/analyze_cohort_distance.py \
      --by_class_size_mode fraction --tag _frac
Outputs results/synth_task_gen/cohort_distance{tag}.png + prints scale/contribution stats.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.gmm_cohort_sampler import CohortSampler

BANK = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/gmm_bank")
COMPS = ("span", "fov", "spacing", "by_class_size")
COLORS = dict(span="tab:blue", fov="tab:orange", spacing="tab:green", by_class_size="tab:red")
# CohortSampler default distance weights
WDEF = dict(span=1.0, fov=0.02, spacing=0.3, by_class_size=3.0)


def classical_mds(D, k=2):
    """Classical MDS: 2-D coords from a distance matrix (numpy eig, no sklearn dep)."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def main():
    ap = argparse.ArgumentParser()
    for k, v in WDEF.items():
        ap.add_argument(f"--w_{k}", type=float, default=v)
    ap.add_argument("--by_class_size_mode", default="fraction", choices=["fraction", "volume"])
    ap.add_argument("--by_class_size_common_frac", type=float, default=0.75)
    ap.add_argument("--tag", default="", help="output filename suffix, e.g. _frac")
    a = ap.parse_args()
    W = {k: getattr(a, f"w_{k}") for k in COMPS}
    out = Path(f"results/synth_task_gen/cohort_distance{a.tag}.png")
    print(f"by_class_size_mode={a.by_class_size_mode} common_frac={a.by_class_size_common_frac}")
    print(f"weights: {W}")

    # Components straight from the sampler -> the study matches what training would use.
    cs = CohortSampler(BANK, k=4, w_span=a.w_span, w_fov=a.w_fov, w_spacing=a.w_spacing,
                       w_by_class_size=a.w_by_class_size,
                       by_class_size_common_frac=a.by_class_size_common_frac,
                       by_class_size_mode=a.by_class_size_mode)
    E = cs.entries
    N = len(E)
    comps = {"span": cs.span, "fov": cs.fov, "spacing": cs.spacing,
             "by_class_size": cs.by_class_size_mat}
    src = np.array([e.get("src", "?") for e in E])
    top = cs.span[:, 0].astype(int)                                  # region top idx
    print(f"bank: {N} subjects | by_class_size cols kept: {cs.by_class_size_ncols}")

    # pairwise L1 per component (cityblock), condensed then square
    Draw = {k: squareform(pdist(v, metric="cityblock")) for k, v in comps.items()}
    Dw = {k: Draw[k] * W[k] for k in COMPS}                          # weighted
    Dtot = sum(Dw.values())
    iu = np.triu_indices(N, 1)                                       # unique pairs

    # ---- stats ----
    print("\ncomponent        raw(mean/med/max)         weighted(mean)   %oftotal")
    tot_mean = Dtot[iu].mean()
    for k in COMPS:
        r = Draw[k][iu]; w = Dw[k][iu]
        print(f"  {k:14s} {r.mean():7.2f}/{np.median(r):6.2f}/{r.max():7.1f}   "
              f"w={w.mean():7.3f}   {100*w.mean()/tot_mean:5.1f}%")

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (a) raw pairwise-distance densities (own scale each -> normalize to [0,1] for shape)
    for k in COMPS:
        d = Draw[k][iu]
        ax[0, 0].hist(d / (d.max() + 1e-9), bins=60, histtype="step", density=True,
                      color=COLORS[k], label=f"{k} (max {d.max():.1f})")
    ax[0, 0].set_title("(a) raw pairwise dist, per component (x normalized by max)")
    ax[0, 0].set_xlabel("distance / max"); ax[0, 0].legend(fontsize=8)

    # (b) weighted distances on a shared axis -> shows which term dominates numerically
    for k in COMPS:
        ax[0, 1].hist(Dw[k][iu], bins=60, histtype="step", density=True,
                      color=COLORS[k], label=f"{k}×{W[k]}")
    ax[0, 1].set_title("(b) WEIGHTED pairwise dist (shared axis = real influence)")
    ax[0, 1].set_xlabel("weighted distance"); ax[0, 1].legend(fontsize=8)

    # (c) per-pair contribution share (fraction of total distance) — boxplot
    shares = [Dw[k][iu] / (Dtot[iu] + 1e-9) for k in COMPS]
    ax[0, 2].boxplot(shares, tick_labels=list(COMPS), showfliers=False)
    ax[0, 2].set_title("(c) per-pair share of total weighted distance")
    ax[0, 2].set_ylabel("fraction"); ax[0, 2].axhline(0.25, ls=":", c="gray")
    ax[0, 2].tick_params(axis="x", labelrotation=20)

    # (d) are the components redundant? Spearman corr of per-pair distances
    from scipy.stats import spearmanr
    M = np.array([Draw[k][iu] for k in COMPS])
    C = spearmanr(M, axis=1).correlation
    im = ax[1, 0].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[1, 0].set_xticks(range(len(COMPS))); ax[1, 0].set_xticklabels(COMPS, rotation=20)
    ax[1, 0].set_yticks(range(len(COMPS))); ax[1, 0].set_yticklabels(COMPS)
    for i in range(len(COMPS)):
        for j in range(len(COMPS)):
            ax[1, 0].text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1, 0].set_title("(d) Spearman corr between component distances")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    # (e) MDS on the combined weighted distance, colored by region-span top idx
    xy = classical_mds(Dtot, 2)
    sc = ax[1, 1].scatter(xy[:, 0], xy[:, 1], c=top, cmap="viridis", s=14, alpha=0.8)
    ax[1, 1].set_title("(e) MDS(combined weighted d) — color = span top idx")
    fig.colorbar(sc, ax=ax[1, 1], fraction=0.046, label="top region 0=head..3=pelvis")

    # (f) same embedding colored by source dataset
    uniq = sorted(set(src)); cmap = plt.get_cmap("tab20")
    for i, s in enumerate(uniq):
        m = src == s
        ax[1, 2].scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.8, color=cmap(i % 20), label=s)
    ax[1, 2].set_title("(f) same MDS — color = source dataset")
    ax[1, 2].legend(fontsize=6, ncol=2, markerscale=1.5)

    fig.suptitle(f"cohort distance — by_class_size_mode={a.by_class_size_mode} "
                 f"(common_frac={a.by_class_size_common_frac})", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

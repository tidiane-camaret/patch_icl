"""
Does the grouped-correlation GMM draw (src.gpu_gmm_intensity.sample_grouped_uniform, wired
into SynthGmmMaisiDataset via mu_group_ids/mu_group_rho) actually reproduce the real
TotalSegmentator CT correlation structure it was calibrated from
(analyze_totalseg_intensity.py's totalseg_intensity_factors_all_debiased.npz)?

Draws many cohort-shared mu vectors the SAME WAY SynthGmmMaisiDataset.assemble() does (fresh
rng per draw -- no crop/paint, since mu doesn't depend on which mask is cropped), indexed by
real MAISI ids, and measures the ACHIEVED cross-id Pearson correlation across draws -- the
exact same "subject x class -> corr" measurement analyze_totalseg_intensity.py used, just
with "cohort draw" standing in for "subject".

HISTORY (docs/logs.md follow-up 5): a first version reshuffled group MEMBERSHIP every draw
(no persistent slot->group map). That measurably failed this check -- EVERY pair of ids
(bone-bone, bone-organ, even ids outside any named cluster) landed at the same ~0.17
"diluted co-occurrence" correlation, matching a closed-form prediction, i.e. it did NOT
reproduce the real block-diagonal structure (bone-bone high, bone-organ ~0), just a flat,
anatomy-blind bump. Fixed via FIXED group membership (CT_GROUP_MAISI_IDS/CT_GROUP_INDICES) --
this script checks that fix actually reproduces the real structure.

  .venv_blackwell/bin/python experiments/3d/synth_task_generation/analyze_synth_gmm_intensity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.gpu_gmm_intensity import CT_GROUP_INDICES, CT_GROUP_RHO, sample_grouped_uniform

N_DRAWS = 4000
N_SLOTS = 200                                  # SynthGmmMaisiDataset default maxid


def draw_many(group_indices, group_rho, n_draws):
    """(n_draws, N_SLOTS) matrix of mu[1:] draws, one fresh rng per draw (matches
    SynthGmmMaisiDataset.assemble()'s train-time entropy: np.random.default_rng() per item)."""
    out = np.empty((n_draws, N_SLOTS))
    for d in range(n_draws):
        rng = np.random.default_rng()
        if group_indices:
            out[d] = sample_grouped_uniform(N_SLOTS, 0.0, 255.0, group_indices, group_rho, rng)
        else:
            out[d] = rng.uniform(0.0, 255.0, size=N_SLOTS)
    return out


def mean_offdiag(corr, idx_a, idx_b=None):
    """Mean off-diagonal correlation within idx_a (idx_b=None) or between idx_a x idx_b."""
    if idx_b is None:
        sub = corr[np.ix_(idx_a, idx_a)]
        return sub[~np.eye(len(idx_a), dtype=bool)].mean()
    return corr[np.ix_(idx_a, idx_b)].mean()


def main():
    all_named = sorted({i for idx in CT_GROUP_INDICES for i in idx})
    leftover = [i for i in range(N_SLOTS) if i not in all_named][:60]
    print(f"4 fixed real clusters: sizes {[len(i) for i in CT_GROUP_INDICES]}, "
          f"targets {CT_GROUP_RHO}")
    print(f"Control 'leftover' ids (not in any named cluster): {len(leftover)}\n")

    for label, group_indices, group_rho in [
        ("GROUPED, FIXED membership (mu_group_ids=ct)", CT_GROUP_INDICES, CT_GROUP_RHO),
        ("BASELINE (independent, today's default)", (), ()),
    ]:
        print(f"=== {label} ===")
        draws = draw_many(group_indices, group_rho, N_DRAWS)
        corr = np.corrcoef(draws, rowvar=False)

        for i, (idx, target) in enumerate(zip(CT_GROUP_INDICES, CT_GROUP_RHO)):
            within = mean_offdiag(corr, idx)
            print(f"  within group {i:<2} (n={len(idx):2d}) achieved mean r={within:+.3f}  "
                  f"(target {target:+.2f})")

        rib_idx, organ_idx = CT_GROUP_INDICES[0], CT_GROUP_INDICES[4]      # ribs vs abdominal organs
        cross = mean_offdiag(corr, rib_idx, organ_idx)
        print(f"  CROSS ribs x abdominal-organs achieved mean r={cross:+.3f}  "
              f"(real CT: ~0 -- different tissue types)")

        within_left = mean_offdiag(corr, leftover)
        print(f"  within control/leftover (never grouped): mean r={within_left:+.3f}  "
              f"(should be exactly ~0 -- these ids are always independent)\n")


if __name__ == "__main__":
    main()

"""Per-axis object aspect ratio (elongation) for the four worst-performing OOD sources on the
`135b` headline checkpoint (ISLES22, Shifts-MS, ATLAS v2.0, GNC_705) -- the one geometry
dimension `compute_class_properties.py`'s cache doesn't cover (it only stores the single
largest bbox extent, not per-axis shape). Spacing and target-vs-surround intensity contrast are
already characterized elsewhere (dataset docs' Geometry sections; `analyze_target_surround_
contrast.py` -> `results/synth_task_gen/target_surround_contrast.csv`) -- this script only adds
the missing "how elongated/flat is the object" axis, to check whether the `p_shape` procedural
generator (blob/splatter/disk/cylinder) actually spans the aspect ratios these targets have.

    python experiments/3d/synth_task_generation/analyze_ood_shape_aspect.py --n-sample 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.providers.atlas_v2 import AtlasV2Provider
from src.providers.gnc_kidney import GncKidneyProvider
from src.providers.isles22 import Isles22Provider
from src.providers.shifts_ms import ShiftsMsProvider

DATA_ROOT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
             "ANALYSIS_20251122/data")
REGISTRY = {
    "isles22":    (Isles22Provider, f"{DATA_ROOT}/isles22/npy"),
    "shifts_ms":  (ShiftsMsProvider, f"{DATA_ROOT}/shifts_ms/npy"),
    "atlas_v2":   (AtlasV2Provider, f"{DATA_ROOT}/atlas_v2/npy"),
    "gnc_kidney": (GncKidneyProvider, f"{DATA_ROOT}/gnc_kidney/npy"),
}
OUT = Path(__file__).resolve().parents[3] / "results" / "synth_task_gen" / "ood_shape_aspect.csv"


def one_case(provider, subj, cls):
    gt = provider.native_gt(subj, cls)
    if gt is None or gt.sum() == 0:
        return None
    spacing = provider.native_meta(subj)["spacing"]
    n_comp_total = cc_label(gt)[1]
    # largest connected component only -- for scattered targets the whole-mask bbox is
    # dominated by the spread between blobs, not the shape of any one lesion.
    lbl, n = cc_label(gt)
    if n > 1:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        gt = lbl == sizes.argmax()
    zz, yy, xx = np.nonzero(gt)
    extent_vox = np.array([zz.max() - zz.min() + 1, yy.max() - yy.min() + 1, xx.max() - xx.min() + 1])
    extent_mm = extent_vox * np.array(spacing)
    return dict(
        n_components=n_comp_total,
        extent_z_mm=extent_mm[0], extent_y_mm=extent_mm[1], extent_x_mm=extent_mm[2],
        max_extent_mm=extent_mm.max(), min_extent_mm=extent_mm.min(),
        aspect_ratio=extent_mm.max() / extent_mm.min(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for source, (Cls, root) in REGISTRY.items():
        provider = Cls(root=root, classes="all")
        rng = np.random.default_rng(args.seed)
        for cls in provider.classes:
            subjects = provider.subjects_for(cls)
            sampled = subjects if len(subjects) <= args.n_sample else list(
                np.array(subjects)[rng.choice(len(subjects), args.n_sample, replace=False)])
            for s in sampled:
                r = one_case(provider, s, cls)
                if r is not None:
                    rows.append(dict(source=source, cls=cls, subject=s, **r))
        print(f"{source} done", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} rows)\n")

    print(f"{'source':<12}{'cls':<16}{'n':>4}{'largest-comp aspect (max/min mm)':>36}   "
          f"[p10,p50,p90]      frac_single")
    for (source, cls), g in df.groupby(["source", "cls"]):
        p10, p50, p90 = g.aspect_ratio.quantile([0.1, 0.5, 0.9])
        frac_single = (g.n_components == 1).mean()
        print(f"{source:<12}{cls:<16}{len(g):>4}"
              f"{'':>10}mean={g.aspect_ratio.mean():5.2f}  [{p10:4.2f},{p50:4.2f},{p90:4.2f}]"
              f"      {frac_single:.2f}")


if __name__ == "__main__":
    main()

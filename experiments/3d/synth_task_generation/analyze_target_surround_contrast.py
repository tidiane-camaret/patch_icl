"""Real target-vs-immediately-surrounding-tissue intensity contrast, measured directly on the
7 OOD eval sources -- calibrates ShapeCohortSpec.host_contrast_range (currently a GUESSED
+/-60 on the synthetic 0-255 scale, never measured against real data) for the "classes outside
the ~130 TotalSeg/MAISI ones" (tumors/lesions) the shape-mode synth curriculum is meant to help
with.

For each (subject, class): crop to the GT's local bbox + padding, dilate the GT mask by a fixed
physical ring width (mm, converted to voxels via native spacing), subtract the original mask ->
the immediately-surrounding-tissue ring (NOT a whole-scan coarse sample, unlike
compute_class_properties.py's "contrast" column -- that measures whole-scan salience, this
measures local lesion-vs-adjacent-parenchyma contrast, the thing host_contrast_range actually
needs to mimic). Reports contrast in RING-STD units (signed, scale/modality-invariant) so CT
and MRI sources are directly comparable despite living in different raw units.

  python experiments/3d/synth_task_generation/analyze_target_surround_contrast.py --n-sample 40
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.providers.atlas_v2 import AtlasV2Provider
from src.providers.gnc_kidney import GncKidneyProvider
from src.providers.hu_lwk1 import HuLwk1Provider
from src.providers.isles22 import Isles22Provider
from src.providers.msd_hippocampus import MsdHippocampusProvider
from src.providers.msd_prostate import MsdProstateProvider
from src.providers.shifts_ms import ShiftsMsProvider

DATA_ROOT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data")
REGISTRY = {
    "hu_lwk1":         (HuLwk1Provider, f"{DATA_ROOT}/hu_lwk1/npy"),
    "isles22":         (Isles22Provider, f"{DATA_ROOT}/isles22/npy"),
    "shifts_ms":       (ShiftsMsProvider, f"{DATA_ROOT}/shifts_ms/npy"),
    "msd_hippocampus": (MsdHippocampusProvider, f"{DATA_ROOT}/msd_hippocampus/npy"),
    "msd_prostate":    (MsdProstateProvider, f"{DATA_ROOT}/msd_prostate/npy"),
    "atlas_v2":        (AtlasV2Provider, f"{DATA_ROOT}/atlas_v2/npy"),
    "gnc_kidney":      (GncKidneyProvider, f"{DATA_ROOT}/gnc_kidney/npy"),
}
RING_MM = 8.0     # physical ring width around the target GT
PAD_MM = 12.0     # bbox padding so the dilation has room (> RING_MM)
OUT = Path(__file__).resolve().parents[3] / "results" / "synth_task_gen"


def one_case(root, subj, cls, provider, ring_mm=RING_MM, pad_mm=PAD_MM):
    gt = provider.native_gt(subj, cls)
    if gt is None or gt.sum() == 0:
        return None
    spacing = provider.native_meta(subj)["spacing"]
    zz, yy, xx = np.nonzero(gt)
    pad_vox = [max(1, int(round(pad_mm / sp))) for sp in spacing]
    lo = [max(0, zz.min() - pad_vox[0]), max(0, yy.min() - pad_vox[1]), max(0, xx.min() - pad_vox[2])]
    hi = [min(gt.shape[0], zz.max() + pad_vox[0] + 1),
         min(gt.shape[1], yy.max() + pad_vox[1] + 1),
         min(gt.shape[2], xx.max() + pad_vox[2] + 1)]
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    gt_crop = gt[sl]
    img = np.load(root / subj / "ct_raw.npy", mmap_mode="r")
    img_crop = np.asarray(img[sl], dtype=np.float64)

    ring_vox = [max(1, int(round(ring_mm / sp))) for sp in spacing]
    it = max(1, int(round(np.mean(ring_vox))))
    dilated = binary_dilation(gt_crop, iterations=it)
    ring = dilated & ~gt_crop
    if ring.sum() < 30 or gt_crop.sum() < 5:
        return None

    tgt_vals = img_crop[gt_crop]
    ring_vals = img_crop[ring]
    tgt_mean, ring_mean, ring_std = float(tgt_vals.mean()), float(ring_vals.mean()), float(ring_vals.std())
    return dict(tgt_mean=tgt_mean, tgt_std=float(tgt_vals.std()),
               ring_mean=ring_mean, ring_std=ring_std,
               contrast_z=(tgt_mean - ring_mean) / (ring_std + 1e-6),
               contrast_abs=tgt_mean - ring_mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for source, (Cls, root) in REGISTRY.items():
        provider = Cls(root=root, classes="all")
        rng = np.random.default_rng(args.seed)
        for cls in provider.classes:
            subjects = provider.subjects_for(cls)
            sampled = subjects if len(subjects) <= args.n_sample else list(
                np.array(subjects)[rng.choice(len(subjects), args.n_sample, replace=False)])
            for s in sampled:
                r = one_case(provider.root, s, cls, provider)
                if r is not None:
                    rows.append(dict(source=source, cls=cls, subject=s, **r))
        print(f"{source} done ({time.time()-t0:.0f}s total so far, {len(rows)} cases)", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT / "target_surround_contrast.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows)")

    print(f"\n{'source':<18}{'n':>5}{'contrast_z mean':>17}{'std':>8}{'median':>9}  "
          f"[p10,p90]")
    for source, g in df.groupby("source"):
        p10, p90 = g.contrast_z.quantile([0.1, 0.9])
        print(f"{source:<18}{len(g):>5}{g.contrast_z.mean():>17.2f}{g.contrast_z.std():>8.2f}"
              f"{g.contrast_z.median():>9.2f}  [{p10:.2f},{p90:.2f}]")
    print(f"\noverall pooled: mean={df.contrast_z.mean():.2f} std={df.contrast_z.std():.2f} "
          f"median={df.contrast_z.median():.2f}  [p10,p90]=[{df.contrast_z.quantile(0.1):.2f},"
          f"{df.contrast_z.quantile(0.9):.2f}]")


if __name__ == "__main__":
    main()

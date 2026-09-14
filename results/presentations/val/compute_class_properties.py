"""Precompute per-(source, class) task-property features used by per_dataset_analysis.py's
"what drives Dice" section: modality, mask size (voxels / mm^3), bbox extent, connected-
component count (blob vs. scattered), and a crude foreground/background intensity contrast.

Reads native-grid ground truth directly via each source's `NativeGridProvider` subclass
(`provider.native_gt(subject, cls)` -- the same hook cascade eval uses), so it stays correct
for GNC's per-class-plane storage without special-casing it here. Samples up to `N_SAMPLE`
subjects per class (seeded) to bound runtime -- gnc_kidney's 13 classes dominate the cost
(~4 min total on `thor`, vs. seconds for every other source).

Writes results/presentations/val/class_props_cache.json, committed alongside the notebook so
it doesn't have to re-scan NFS on every notebook open. Re-run this script (not the notebook)
whenever a source's classes/labels change, or a new source is added to REGISTRY below.

    python results/presentations/val/compute_class_properties.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.providers.atlas_v2 import AtlasV2Provider
from src.providers.gnc_kidney import GncKidneyProvider
from src.providers.hu_lwk1 import HuLwk1Provider
from src.providers.isles22 import Isles22Provider
from src.providers.msd_hippocampus import MsdHippocampusProvider
from src.providers.msd_prostate import MsdProstateProvider
from src.providers.shifts_ms import ShiftsMsProvider

DATA_ROOT = (
    "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
    "ANALYSIS_20251122/data"
)
REGISTRY = {
    "hu_lwk1": (HuLwk1Provider, f"{DATA_ROOT}/hu_lwk1/npy"),
    "isles22": (Isles22Provider, f"{DATA_ROOT}/isles22/npy"),
    "shifts_ms": (ShiftsMsProvider, f"{DATA_ROOT}/shifts_ms/npy"),
    "msd_hippocampus": (MsdHippocampusProvider, f"{DATA_ROOT}/msd_hippocampus/npy"),
    "msd_prostate": (MsdProstateProvider, f"{DATA_ROOT}/msd_prostate/npy"),
    "atlas_v2": (AtlasV2Provider, f"{DATA_ROOT}/atlas_v2/npy"),
    "gnc_kidney": (GncKidneyProvider, f"{DATA_ROOT}/gnc_kidney/npy"),
}
N_SAMPLE = 40
SEED = 42
OUT_PATH = Path(__file__).resolve().parent / "class_props_cache.json"


def compute_class_properties(source, provider, cls, n_sample=N_SAMPLE, seed=SEED):
    """Mean task-property features over up to n_sample subjects carrying `cls`."""
    subjects = provider.subjects_for(cls)
    rng = np.random.default_rng(seed)
    sampled = subjects if len(subjects) <= n_sample else list(
        np.array(subjects)[rng.choice(len(subjects), n_sample, replace=False)])
    voxels, vols, extents, ncomps, fg_means, contrasts = [], [], [], [], [], []
    for s in sampled:
        gt = provider.native_gt(s, cls)
        if gt is None or gt.sum() == 0:
            continue
        spacing = provider.native_meta(s)["spacing"]
        vox = int(gt.sum())
        zz, yy, xx = np.nonzero(gt)
        extent_vox = (zz.max() - zz.min() + 1, yy.max() - yy.min() + 1, xx.max() - xx.min() + 1)
        extent_mm = max(e * sp for e, sp in zip(extent_vox, spacing))
        _, ncomp = cc_label(gt)
        img = np.load(provider.root / s / "ct_raw.npy", mmap_mode="r")
        # bbox-cropped foreground read (cheap even on a big memmap) vs. a coarsely strided
        # whole-volume sample as the "background" proxy (avoids a slow full-volume argwhere).
        crop_img = np.asarray(img[zz.min():zz.max()+1, yy.min():yy.max()+1, xx.min():xx.max()+1])
        crop_mask = np.asarray(gt[zz.min():zz.max()+1, yy.min():yy.max()+1, xx.min():xx.max()+1])
        fg_mean = float(crop_img[crop_mask].mean())
        coarse = np.asarray(img[::8, ::8, ::8])
        g_mean, g_std = float(coarse.mean()), float(coarse.std())
        contrast = abs(fg_mean - g_mean) / (g_std + 1e-6)
        voxels.append(vox)
        vols.append(vox * spacing[0] * spacing[1] * spacing[2])
        extents.append(extent_mm)
        ncomps.append(int(ncomp))
        fg_means.append(fg_mean)
        contrasts.append(contrast)
    if not voxels:
        return None
    return {
        "source": source, "class": cls, "modality": provider.modality,
        "n_subjects_total": len(subjects), "n_sampled": len(voxels),
        "mean_voxels": float(np.mean(voxels)), "median_voxels": float(np.median(voxels)),
        "mean_volume_mm3": float(np.mean(vols)),
        "mean_bbox_extent_mm": float(np.mean(extents)),
        "mean_n_components": float(np.mean(ncomps)),
        # frac_single_component: 1.0 = always one blob, 0.0 = always scattered/multi-focal --
        # the "aspect" axis (blob vs. scattered) requested for the what-drives-Dice analysis.
        "frac_single_component": float(np.mean([1.0 if n == 1 else 0.0 for n in ncomps])),
        "mean_fg_intensity": float(np.mean(fg_means)),
        "mean_contrast": float(np.mean(contrasts)),
    }


def main():
    t0 = time.time()
    rows = []
    for source, (Cls, root) in REGISTRY.items():
        provider = Cls(root=root, classes="all")
        for cls in provider.classes:
            row = compute_class_properties(source, provider, cls)
            if row:
                rows.append(row)
        print(f"{source} done ({time.time()-t0:.1f}s total so far)")
    OUT_PATH.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} rows -> {OUT_PATH} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()

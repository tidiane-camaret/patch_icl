"""Per-sample OOD failure diagnostics: merges eval.py's per-case Dice with per-subject
shape/intensity/texture task-property features (mask size, bbox extent, connected-component
count, fg/bg intensity contrast, fg intensity std as a texture proxy) -- same feature set as
results/presentations/val/compute_class_properties.py, but per-SAMPLE (every eval case, not a
40-subject-per-class cap aggregated to one mean) so it can be correlated directly against
per-sample Dice rather than just a per-class mean.

Usage:
  python experiments/3d/synth_task_generation/diagnose_ood_failures.py --checkpoint <ckpt>
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from scipy.ndimage import label as cc_label

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from eval import _build_model                                   # noqa: E402
from evaluate import evaluate_classes                            # noqa: E402

from src.providers.atlas_v2 import AtlasV2Provider, resolve_atlas_v2_classes             # noqa: E402
from src.providers.gnc_kidney import GncKidneyProvider, resolve_gnc_kidney_classes       # noqa: E402
from src.providers.hu_lwk1 import HuLwk1Provider, resolve_hu_lwk1_classes                # noqa: E402
from src.providers.isles22 import Isles22Provider, resolve_isles22_classes               # noqa: E402
from src.providers.msd_hippocampus import (MsdHippocampusProvider,                       # noqa: E402
                                           resolve_msd_hippocampus_classes)
from src.providers.msd_prostate import MsdProstateProvider, resolve_msd_prostate_classes # noqa: E402
from src.providers.shifts_ms import ShiftsMsProvider, resolve_shifts_ms_classes          # noqa: E402

DATA_ROOT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data")
SOURCES = {
    "isles22":         (Isles22Provider, resolve_isles22_classes, f"{DATA_ROOT}/isles22/npy"),
    "shifts_ms":       (ShiftsMsProvider, resolve_shifts_ms_classes, f"{DATA_ROOT}/shifts_ms/npy"),
    "msd_hippocampus": (MsdHippocampusProvider, resolve_msd_hippocampus_classes,
                        f"{DATA_ROOT}/msd_hippocampus/npy"),
    "msd_prostate":    (MsdProstateProvider, resolve_msd_prostate_classes,
                        f"{DATA_ROOT}/msd_prostate/npy"),
    "atlas_v2":        (AtlasV2Provider, resolve_atlas_v2_classes, f"{DATA_ROOT}/atlas_v2/npy"),
    "gnc_kidney":      (GncKidneyProvider, resolve_gnc_kidney_classes, f"{DATA_ROOT}/gnc_kidney/npy"),
    "hu_lwk1":         (HuLwk1Provider, resolve_hu_lwk1_classes, f"{DATA_ROOT}/hu_lwk1/npy"),
}


def task_features(provider, subj, cls):
    """Per-subject shape/intensity/texture features for one (subject, class) GT mask."""
    gt = provider.native_gt(subj, cls)
    if gt is None or gt.sum() == 0:
        return None
    spacing = provider.native_meta(subj)["spacing"]
    zz, yy, xx = np.nonzero(gt)
    extent_vox = (zz.max() - zz.min() + 1, yy.max() - yy.min() + 1, xx.max() - xx.min() + 1)
    extent_mm = max(e * sp for e, sp in zip(extent_vox, spacing))
    _, ncomp = cc_label(gt)
    img = np.load(provider.root / subj / "ct_raw.npy", mmap_mode="r")
    crop_img = np.asarray(img[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1])
    crop_mask = np.asarray(gt[zz.min():zz.max() + 1, yy.min():yy.max() + 1, xx.min():xx.max() + 1])
    fg_vals = crop_img[crop_mask].astype(np.float64)
    fg_mean, fg_std = float(fg_vals.mean()), float(fg_vals.std())
    coarse = np.asarray(img[::8, ::8, ::8]).astype(np.float64)
    g_mean, g_std = float(coarse.mean()), float(coarse.std())
    contrast = abs(fg_mean - g_mean) / (g_std + 1e-6)
    vox = int(gt.sum())
    return {
        "tgt_vox": vox,
        "tgt_vol_mm3": round(vox * spacing[0] * spacing[1] * spacing[2], 2),
        "bbox_extent_mm": round(extent_mm, 2),
        "n_components": int(ncomp),
        "fg_mean": round(fg_mean, 3),
        "fg_std": round(fg_std, 3),            # texture/heterogeneity proxy
        "contrast": round(contrast, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = (Path(args.out) if args.out
                else ROOT / "results/presentations/val/ood_failure_diag.csv")

    rows_out = []
    model = None
    t0 = time.time()
    for ds, (ProvCls, resolve_fn, root) in SOURCES.items():
        with initialize_config_dir(config_dir=str(ROOT / "configs/experiment/3d"),
                                   version_base="1.3"):
            cfg = compose(config_name="eval", overrides=[
                f"dataset={ds}", "eval.model=patchset3d",
                f"eval.checkpoint={args.checkpoint}"])
        classes = resolve_fn(cfg.data.val_classes)
        if model is None:
            model = _build_model(cfg)
        rows, cases = evaluate_classes(model, cfg, classes, fig_dir=None,
                                       autocast=bool(cfg.eval.get("autocast", False)))
        provider = ProvCls(root=root, classes=classes)
        n_ds = 0
        for c in cases:
            feat = task_features(provider, c["subject"], c["class"])
            if feat is None:
                continue
            rows_out.append({"dataset": ds, "class": c["class"], "subject": c["subject"],
                             "dice": c["dice"], "nsd": c.get("nsd", ""), **feat})
            n_ds += 1
        print(f"[{time.time()-t0:.0f}s] {ds}: {n_ds}/{len(cases)} cases with features "
              f"(cumulative {len(rows_out)})", flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows -> {out_path} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()

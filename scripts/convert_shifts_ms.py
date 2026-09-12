"""Convert Shifts-MS Part 2 (openly-licensed cohorts, .nii.gz) -> per-subject .npy at NATIVE
(near-isotropic) spacing. Mirrors scripts/convert_isles22.py: native grid, MRI modality
(per-subject mri_stats, not a fixed CT frame), FLAIR only.

Two things this dataset needs that ISLES22 did not (see docs/datasets/shifts_ms.md):

1. TWO COHORTS, NON-UNIQUE SUBJECT IDS. `best/{train,dev_in,eval_in}` and
   `ljubljana/dev_out` each number their own subjects starting near 1 -- e.g. `best/train/25`
   and `best/eval_in/25` are DIFFERENT patients (verified: different shape, different image
   content) despite sharing the numeral "25". Subject dirs are namespaced
   `{cohort}_{split}_{id}` to avoid collisions.

2. TWO DIFFERENT ORIENTATIONS, per cohort not per file. `best` ships LAS, `ljubljana` ships
   LPS (verified via nib.aff2axcodes on every file, not assumed) -- `nib.as_closest_canonical`
   handles both correctly and automatically since (unlike NasalSeg's NRRD) these are plain
   NIfTI with real affines, so no per-cohort branch is needed in code.

`ljubljana`'s flair/gt affines disagree by up to ~0.45mm (verified across all 25 cases) --
a sub-voxel origin-rounding artifact from how that cohort was independently resampled to
isovoxel space (not present at all in `best`, max diff ~1e-8mm there), not real misalignment.
The affine-agreement check therefore uses a 1mm tolerance, not "best"'s effectively-exact one.

Layout written under --out:
    best_train_0025/ct_raw.npy   (D,H,W) float32, RAS, native grid, FLAIR intensity
    best_train_0025/label.npy    (D,H,W) uint8, 0=bg, 1=ms_lesion
    spacings.json                {subj: {spacing, shape, affine}}
    ct_stats.json                {subj: {clip_lo, clip_hi, mean, std}}  (mri_stats)

Usage:
    python scripts/convert_shifts_ms.py --workers 16
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataset import mri_stats  # noqa: E402

# (cohort, split) pairs to convert. best/train+dev_in+eval_in is the OFSEP-free "in-distribution"
# half; ljubljana/dev_out is the deliberately out-of-distribution cohort (see shifts_ms.md).
SPLITS = [("best", "train"), ("best", "dev_in"), ("best", "eval_in"), ("ljubljana", "dev_out")]

DEFAULT_SRC = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/shifts_ms/shifts_ms_pt2")
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/shifts_ms/npy")


def convert_one(subj: str, flair_path: Path, gt_path: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, dict | None, str | None]:
    """Convert one case. Returns (subj, spacings_meta, ct_stats_entry, error)."""
    try:
        subj_dir = out_dir / subj
        img_out, lbl_out = subj_dir / "ct_raw.npy", subj_dir / "label.npy"

        img = nib.as_closest_canonical(nib.load(str(flair_path)))
        msk = nib.as_closest_canonical(nib.load(str(gt_path)))
        affine = np.asarray(img.affine, dtype=np.float64)
        spacing = [float(x) for x in nib.affines.voxel_sizes(affine)[:3]]
        meta = {"spacing": spacing, "shape": [int(x) for x in img.shape[:3]],
                "affine": affine.tolist()}

        if img.shape[:3] != msk.shape[:3]:
            return subj, None, None, f"shape mismatch img{img.shape[:3]} msk{msk.shape[:3]}"
        # 1mm tolerance, not the tighter atol ISLES22 used: `ljubljana`'s flair/gt affines
        # disagree by up to ~0.45mm (sub-voxel origin-rounding, verified across all 25 cases,
        # not real misalignment -- see module docstring).
        if not np.allclose(affine, np.asarray(msk.affine), atol=1.0):
            return subj, None, None, "image/mask affine mismatch (> 1mm)"

        raw = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
        stats = mri_stats(raw)

        if not overwrite and img_out.exists() and lbl_out.exists():
            return subj, meta, stats, None

        lab = np.ascontiguousarray(msk.get_fdata(dtype=np.float32))
        if not np.array_equal(lab, np.round(lab)):
            return subj, None, None, "mask is not integral-valued"
        lab = np.round(lab).astype(np.int32)
        if lab.min() < 0 or lab.max() > 1:
            return subj, None, None, f"mask range [{lab.min()}, {lab.max()}] outside 0..1"

        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(img_out, raw.astype(np.float32))
        np.save(lbl_out, lab.astype(np.uint8))
        meta["classes_present"] = sorted(int(v) for v in np.unique(lab) if v != 0)
        meta["lesion_voxels"] = int((lab > 0).sum())
        return subj, meta, stats, None
    except Exception as exc:  # noqa: BLE001
        return subj, None, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="shifts_ms_pt2 root")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)

    cases = []
    for cohort, split in SPLITS:
        flair_dir = src / cohort / split / "flair"
        gt_dir = src / cohort / split / "gt"
        if not flair_dir.is_dir():
            print(f"  [skip] {cohort}/{split}: no flair/ dir")
            continue
        for fp in sorted(flair_dir.glob("*_FLAIR_isovox.nii.gz")):
            sid = fp.name.split("_")[0]
            gp = gt_dir / f"{sid}_gt_isovox.nii.gz"
            if not gp.exists():
                print(f"  [skip] {cohort}/{split}/{sid}: no matching gt")
                continue
            subj = f"{cohort}_{split}_{int(sid):04d}"
            cases.append((subj, fp, gp))
    print(f"{len(cases)} cases  {src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    spacings, ct_stats, errors = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, f, g, out, args.overwrite) for s, f, g in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            subj, meta, stats, err = fut.result()
            if err:
                errors.append(f"{subj}: {err}")
                print(f"  [FAIL] {subj}: {err}")
            else:
                spacings[subj] = meta
                ct_stats[subj] = stats
            if n % 10 == 0:
                print(f"  {n}/{len(cases)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: spacings[k] for k in sorted(spacings)}, f, indent=1)
    with open(out / "ct_stats.json", "w") as f:
        json.dump({k: ct_stats[k] for k in sorted(ct_stats)}, f, indent=1)

    n_empty = sum(1 for m in spacings.values() if not m.get("classes_present"))
    sp = np.array([m["spacing"] for m in spacings.values()])
    print(f"\nwrote {len(spacings)} cases ({n_empty} with an empty lesion mask), "
          f"{len(errors)} failed")
    if len(sp):
        print(f"  spacing: min {sp.min(0).round(2)} max {sp.max(0).round(2)}")
        print(f"  -> {out/'spacings.json'}, {out/'ct_stats.json'}")
    for e in errors:
        print(f"  {e}")


if __name__ == "__main__":
    main()

"""Convert MSD Task04_Hippocampus (nnU-Net layout, .nii.gz) -> per-subject .npy at NATIVE
(isotropic 1mm) spacing. Mirrors scripts/convert_isles22.py: native grid only, every
resampling decision deferred to the dataloader (src/providers/msd_hippocampus.py).

What makes this source different from ISLES22/Shifts-MS (see docs/datasets/msd_hippocampus.md):

1. Volumes are ALREADY ROI-cropped to a tiny box around the hippocampus (median ~35x50x36 vox,
   max ~43x59x47), not a whole organ/head/torso — the physical FOV is ~35-60mm/axis, far
   smaller than every other source in the harness. This has no effect on the converter (still
   just RAS + per-subject MRI stats, no resampling at conversion time), but it's WHY
   crop_spacing_mm needs to drop to 0.5mm at eval time (grid-occupancy sweep in the doc) instead
   of the >=0.6-1.5mm every other source uses.
2. Raw scanner T1 units (not clipped/z-scored at the source) -> per-subject `mri_stats`
   (same MODALITY="mri" mechanism ISLES22/Shifts-MS use), written to ct_stats.json.
3. Two classes in ONE label file (1=Anterior, 2=Posterior), not separate masks -- no special
   handling needed, `label.npy` just keeps both integer values directly.
4. Already RAS + isotropic 1mm (verified across all 260 cases) -- `nib.as_closest_canonical`
   is a no-op safety net here, not a real reorientation like ISLES22/Shifts-MS needed.

Layout written under --out:
    hippocampus_001/ct_raw.npy   (D,H,W) float32, RAS, native (1mm) grid, raw T1 intensity
    hippocampus_001/label.npy    (D,H,W) uint8, 0=bg, 1=Anterior, 2=Posterior
    spacings.json                 {subj: {spacing, shape, affine}}
    ct_stats.json                 {subj: {clip_lo, clip_hi, mean, std}}  (mri_stats)

Usage:
    python scripts/convert_msd_hippocampus.py --workers 16
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

DEFAULT_SRC = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/msd_hippocampus/Task04_Hippocampus")
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/msd_hippocampus/npy")


def convert_one(subj: str, img_path: Path, lbl_path: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, dict | None, str | None]:
    """Convert one case. Returns (subj, spacings_meta, ct_stats_entry, error)."""
    try:
        subj_dir = out_dir / subj
        img_out, lbl_out = subj_dir / "ct_raw.npy", subj_dir / "label.npy"

        img = nib.as_closest_canonical(nib.load(str(img_path)))
        msk = nib.as_closest_canonical(nib.load(str(lbl_path)))
        affine = np.asarray(img.affine, dtype=np.float64)
        spacing = [float(x) for x in nib.affines.voxel_sizes(affine)[:3]]
        meta = {"spacing": spacing, "shape": [int(x) for x in img.shape[:3]],
                "affine": affine.tolist()}

        if img.shape[:3] != msk.shape[:3]:
            return subj, None, None, f"shape mismatch img{img.shape[:3]} msk{msk.shape[:3]}"
        if not np.allclose(affine, np.asarray(msk.affine), atol=1e-3):
            return subj, None, None, "image/mask affine mismatch"

        raw = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
        stats = mri_stats(raw)                              # whole-volume, BEFORE overwrite check

        if not overwrite and img_out.exists() and lbl_out.exists():
            return subj, meta, stats, None

        lab = np.ascontiguousarray(msk.get_fdata(dtype=np.float32))
        if not np.array_equal(lab, np.round(lab)):
            return subj, None, None, "mask is not integral-valued"
        lab = np.round(lab).astype(np.int32)
        if lab.min() < 0 or lab.max() > 2:
            return subj, None, None, f"mask range [{lab.min()}, {lab.max()}] outside 0..2"

        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(img_out, raw.astype(np.float32))
        np.save(lbl_out, lab.astype(np.uint8))
        meta["classes_present"] = sorted(int(v) for v in np.unique(lab) if v != 0)
        return subj, meta, stats, None
    except Exception as exc:  # noqa: BLE001
        return subj, None, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="Task04_Hippocampus root")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    images_dir, labels_dir = src / "imagesTr", src / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"expected {images_dir} and {labels_dir}")

    cases = []
    for img_path in sorted(images_dir.glob("hippocampus_*.nii.gz")):
        subj = img_path.name.removesuffix(".nii.gz")
        lbl_path = labels_dir / img_path.name
        if not lbl_path.exists():
            print(f"  [skip] {subj}: no matching label")
            continue
        cases.append((subj, img_path, lbl_path))
    print(f"{len(cases)} cases  {src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    spacings, ct_stats, errors = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, i, m, out, args.overwrite) for s, i, m in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            subj, meta, stats, err = fut.result()
            if err:
                errors.append(f"{subj}: {err}")
                print(f"  [FAIL] {subj}: {err}")
            else:
                spacings[subj] = meta
                ct_stats[subj] = stats
            if n % 50 == 0:
                print(f"  {n}/{len(cases)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: spacings[k] for k in sorted(spacings)}, f, indent=1)
    with open(out / "ct_stats.json", "w") as f:
        json.dump({k: ct_stats[k] for k in sorted(ct_stats)}, f, indent=1)

    sp = np.array([m["spacing"] for m in spacings.values()])
    sh = np.array([m["shape"] for m in spacings.values()])
    print(f"\nwrote {len(spacings)} cases, {len(errors)} failed")
    if len(sp):
        print(f"  spacing: min {sp.min(0).round(2)} max {sp.max(0).round(2)}")
        print(f"  shape: min {sh.min(0)} median {np.median(sh, 0)} max {sh.max(0)}")
        print(f"  -> {out/'spacings.json'}, {out/'ct_stats.json'}")
    for e in errors:
        print(f"  {e}")


if __name__ == "__main__":
    main()

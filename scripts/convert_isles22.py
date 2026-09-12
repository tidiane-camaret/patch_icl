"""Convert ISLES 2022 (BIDS, .nii.gz) -> per-subject .npy at NATIVE (mildly anisotropic)
spacing. Mirrors scripts/convert_flare22.py: native grid only, every resampling decision
deferred to the dataloader (src/providers/isles22.py).

Two things this dataset needs that FLARE22/NasalSeg did not (see docs/datasets/isles22.md):

1. MRI, NOT CT. The DWI sequence is arbitrary scanner-unit intensity, not HU — there is no
   fixed global clip/z-score frame to store (unlike CT's `normalize_ct`). Per-subject
   whole-volume stats (`src.totalseg_dataset.mri_stats`) are computed here at convert time and
   written to a `ct_stats.json` sidecar — same file/format `TotalSegProvider`'s totalsegmri
   branch already reads, now also read by `NativeGridProvider` (native_grid.py) for any
   subclass with `MODALITY = "mri"`.

2. Only ONE of the three co-registered sequences is converted: **DWI (b=1000)**, the
   acute-stroke-sensitive one. ADC and FLAIR ship alongside in the source but are not needed
   for a single-channel in-context task and are not converted (a future multi-sequence
   provider could read them from the same source root without re-converting).

3. Plain NIfTI with a valid affine (unlike NasalSeg's NRRD headers) — orientation is fixed with
   `nib.as_closest_canonical`, no manual axis-flip math needed (source is LAS, target RAS: a
   single-axis flip, but let nibabel derive it from the affine rather than hand-rolling it).

3/250 subjects have an entirely empty lesion mask in the source release (verified, not a
conversion bug) — they convert fine (an all-zero label.npy); NativeGridProvider's per-class
centroid cache already excludes any subject with a zero-voxel class from
`subjects_for(cls)`, so no special-casing is needed here.

Layout written under --out:
    sub-strokecase0001/ct_raw.npy   (D,H,W) float32, RAS, native grid, DWI intensity
    sub-strokecase0001/label.npy    (D,H,W) uint8, 0=bg, 1=stroke_lesion
    spacings.json                   {subj: {spacing, shape, affine}}
    ct_stats.json                   {subj: {clip_lo, clip_hi, mean, std}}  (mri_stats)

Usage:
    python scripts/convert_isles22.py --workers 16
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

ISLES22_CLASSES = ["stroke_lesion"]

DEFAULT_SRC = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/isles22/ISLES-2022")
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/isles22/npy")


def convert_one(subj: str, dwi_path: Path, msk_path: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, dict | None, str | None]:
    """Convert one case. Returns (subj, spacings_meta, ct_stats_entry, error)."""
    try:
        subj_dir = out_dir / subj
        img_out, lbl_out = subj_dir / "ct_raw.npy", subj_dir / "label.npy"

        img = nib.as_closest_canonical(nib.load(str(dwi_path)))
        msk = nib.as_closest_canonical(nib.load(str(msk_path)))
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
    ap.add_argument("--src", default=DEFAULT_SRC, help="ISLES-2022 BIDS root")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not (src / "derivatives").is_dir():
        raise SystemExit(f"expected {src/'derivatives'}")

    cases = []
    for sd in sorted(src.glob("sub-strokecase*")):
        subj = sd.name
        dwi = sorted(sd.glob("ses-*/dwi/*_dwi.nii.gz"))
        msk = sorted((src / "derivatives" / subj).glob("ses-*/*_msk.nii.gz"))
        if not dwi or not msk:
            print(f"  [skip] {subj}: missing dwi ({len(dwi)}) or mask ({len(msk)})")
            continue
        cases.append((subj, dwi[0], msk[0]))
    print(f"{len(cases)} cases  {src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    spacings, ct_stats, errors = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, d, m, out, args.overwrite) for s, d, m in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            subj, meta, stats, err = fut.result()
            if err:
                errors.append(f"{subj}: {err}")
                print(f"  [FAIL] {subj}: {err}")
            else:
                spacings[subj] = meta
                ct_stats[subj] = stats
            if n % 25 == 0:
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

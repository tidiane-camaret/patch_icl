"""Convert the zanderch HU_Messung cohort (local NFS, polytrauma whole-body CT) -> per-case
.npy at NATIVE (anisotropic) spacing. Mirrors scripts/convert_flare22.py: native grid only,
every resampling decision deferred to the dataloader (src/providers/hu_lwk1.py).

Source layout (`/nfs/data/nii/data1/zanderch___HU_Messung/{subj}/{visit}/...`):
    {visit}/HU_LWK1.nii.gz          -- ground-truth label, ONLY when a radiologist placed an
                                        HU-measurement ROI at the L1 vertebral body centrum
                                        (a small ~15x15x3-15 voxel blob, NOT a vertebra
                                        segmentation -- see docs/datasets/hu_lwk1.md).
    {visit}/{series}/*.nii          -- one or more raw CT series for that visit (different
                                        reconstruction kernels / phases of the same scan, or
                                        genuinely different series); the label shares the
                                        FULL image grid + affine of exactly one of them.
    {visit}/preds/*                 -- an existing auto-segmentation pipeline's OWN outputs
                                        (HU_LWK1_pred.nii.gz, vertebrae_L1.nii.gz) -- not GT,
                                        never read here.

Only 36/687 subjects have a GT `HU_LWK1.nii.gz` (37 files found; one, `100999`, has no
matching image anywhere under its case dir and is skipped -- a broken/incomplete case, not a
conversion bug). Unlike GNC (docs/datasets/gnc_kidney_lesions.md), label and image share the
SAME shape + affine here (no ROI-crop offset) -- but a visit dir can hold MULTIPLE `.nii`
series (different kernels), so the matching image is found by shape+affine match against the
label, not positionally. Case id is `{subj}_{visit}` since 2 subjects (27441734, 21711659)
have GT at both of their visits.

Layout written under --out:
    {subj}_{visit}/ct_raw.npy   (D,H,W) int16, RAS, native grid, HU
    {subj}_{visit}/label.npy    (D,H,W) uint8, 0=bg, 1=l1_center
    spacings.json                {case: {spacing, shape, affine}}

Usage:
    python scripts/convert_hu_lwk1.py --workers 16
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_SRC = "/nfs/data/nii/data1/zanderch___HU_Messung"
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/hu_lwk1/npy")

L1_CENTER_CLASS = "l1_center"


def _find_image(visit_dir: Path, lbl: nib.Nifti1Image):
    """Pick the one candidate .nii series matching the label's shape+affine (a visit dir can
    hold several series from different kernels/phases -- see module docstring)."""
    candidates = [p for p in visit_dir.rglob("*.nii")
                  if not ({"DICOM", "preds", "tmp"} & set(p.parts))]
    shape_matches = []
    for p in candidates:
        img = nib.load(str(p))
        if img.shape[:3] != lbl.shape[:3]:
            continue
        shape_matches.append((p, img))
        if np.allclose(img.affine, lbl.affine, atol=1e-2):
            return p, img
    # fallback: shape matched but no exact affine match (none seen in practice, kept defensive)
    return shape_matches[0] if shape_matches else (None, None)


def convert_one(case: str, lbl_path: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, str | None]:
    """Convert one case. Returns (case, meta, error)."""
    try:
        subj_dir = out_dir / case
        ct_out, lbl_out = subj_dir / "ct_raw.npy", subj_dir / "label.npy"

        lbl = nib.load(str(lbl_path))
        img_path, img = _find_image(lbl_path.parent, lbl)
        if img is None:
            return case, None, "no matching image series found (shape+affine)"

        img = nib.as_closest_canonical(img)
        lbl = nib.as_closest_canonical(lbl)
        if img.shape[:3] != lbl.shape[:3]:
            return case, None, f"shape mismatch after reorient img{img.shape[:3]} lbl{lbl.shape[:3]}"
        affine = np.asarray(img.affine, dtype=np.float64)
        if not np.allclose(affine, np.asarray(lbl.affine), atol=1e-2):
            return case, None, "image/label affine mismatch after reorient"
        spacing = [float(x) for x in nib.affines.voxel_sizes(affine)[:3]]
        meta = {"spacing": spacing, "shape": [int(x) for x in img.shape[:3]],
                "affine": affine.tolist(), "src_image": str(img_path)}

        if not overwrite and ct_out.exists() and lbl_out.exists():
            return case, meta, None

        raw = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
        if not np.array_equal(raw, np.round(raw)):
            return case, None, "CT is not integral-valued; int16 would be lossy"
        if raw.min() < -32768 or raw.max() > 32767:
            return case, None, f"CT range [{raw.min()}, {raw.max()}] exceeds int16"

        lab = np.ascontiguousarray(lbl.get_fdata(dtype=np.float32))
        if not np.array_equal(lab, np.round(lab)):
            return case, None, "label is not integral-valued"
        lab = np.round(lab).astype(np.int32)
        if lab.min() < 0 or lab.max() > 1:
            return case, None, f"label range [{lab.min()}, {lab.max()}] outside 0..1"
        if lab.sum() == 0:
            return case, None, "label is all-zero"

        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(ct_out, np.round(raw).astype(np.int16))
        np.save(lbl_out, lab.astype(np.uint8))
        meta["hu_range"] = [float(raw.min()), float(raw.max())]
        meta["l1_center_voxels"] = int((lab > 0).sum())
        return case, meta, None
    except Exception as exc:  # noqa: BLE001
        return case, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="zanderch___HU_Messung root")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    lbl_paths = sorted(src.glob("*/*/HU_LWK1.nii.gz"))
    cases = []
    for lp in lbl_paths:
        visit_dir = lp.parent
        case = f"{visit_dir.parent.name}_{visit_dir.name}"
        cases.append((case, lp))
    print(f"{len(cases)} labeled cases found  {src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    spacings, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, c, lp, out, args.overwrite) for c, lp in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            case, meta, err = fut.result()
            if err:
                errors.append(f"{case}: {err}")
                print(f"  [FAIL] {case}: {err}")
            else:
                spacings[case] = meta
            if n % 10 == 0:
                print(f"  {n}/{len(cases)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: spacings[k] for k in sorted(spacings)}, f, indent=1)

    sp = np.array([m["spacing"] for m in spacings.values()])
    print(f"\nwrote {len(spacings)} cases, {len(errors)} failed")
    if len(sp):
        print(f"  spacing: min {sp.min(0).round(3)} max {sp.max(0).round(3)}")
        print(f"  -> {out/'spacings.json'}")
    for e in errors:
        print(f"  {e}")


if __name__ == "__main__":
    main()

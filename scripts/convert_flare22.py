"""Convert FLARE22 (nii.gz) -> per-subject .npy at NATIVE anisotropic spacing.

Unlike scripts/convert_to_npy.py (which also writes pre-resized isotropic grids),
this writes ONLY the native-grid arrays. Every resampling decision is deferred to
the dataloader (src/providers/flare22.py), so changing `crop_spacing_mm` is a
config change rather than a re-conversion.

Losslessness: FLARE22 CT is integral-valued with a global range of [-1024, 3071],
so int16 is bit-exact (float16 is NOT — it cannot represent odd integers > 2048).
The check is enforced per case, not assumed.

The full 4x4 affine is stored alongside the spacing: without the translation you
cannot write a prediction back into the source NIfTI frame, which native-space
scoring needs.

Layout written under --out:
    FLARE22_Tr_0001/ct_raw.npy   (D,H,W) int16, RAS, native grid
    FLARE22_Tr_0001/label.npy    (D,H,W) uint8, 0=bg, 1..13 organs
    spacings.json                {subj: {spacing, shape, affine}}

Usage:
    python scripts/convert_flare22.py --workers 16
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

# FLARE22 label index -> TotalSegmentator class name (verified on disk: laterality
# confirmed 50/50 against the RAS x-axis for both the kidney and adrenal pairs).
FLARE22_CLASSES = [
    "liver", "kidney_right", "spleen", "pancreas", "aorta",
    "inferior_vena_cava", "adrenal_gland_right", "adrenal_gland_left",
    "gallbladder", "esophagus", "stomach", "duodenum", "kidney_left",
]

DEFAULT_SRC = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/flare22/FLARE22Train")
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/flare22/npy")


def convert_one(subj: str, img_path: Path, lbl_path: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, str | None]:
    """Convert one case. Returns (subj, meta, error)."""
    try:
        subj_dir = out_dir / subj
        ct_out, lbl_out = subj_dir / "ct_raw.npy", subj_dir / "label.npy"

        img = nib.as_closest_canonical(nib.load(str(img_path)))
        lbl = nib.as_closest_canonical(nib.load(str(lbl_path)))
        affine = np.asarray(img.affine, dtype=np.float64)
        spacing = [float(x) for x in nib.affines.voxel_sizes(affine)[:3]]
        meta = {"spacing": spacing,
                "shape": [int(x) for x in img.shape[:3]],
                "affine": affine.tolist()}

        if not overwrite and ct_out.exists() and lbl_out.exists():
            return subj, meta, None

        if img.shape[:3] != lbl.shape[:3]:
            return subj, None, f"shape mismatch img{img.shape[:3]} lbl{lbl.shape[:3]}"
        if not np.allclose(affine, np.asarray(lbl.affine), atol=1e-4):
            return subj, None, "image/label affine mismatch"

        raw = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
        # int16 is only lossless if the data really are integral and in range.
        if not np.array_equal(raw, np.round(raw)):
            return subj, None, "CT is not integral-valued; int16 would be lossy"
        if raw.min() < -32768 or raw.max() > 32767:
            return subj, None, f"CT range [{raw.min()}, {raw.max()}] exceeds int16"

        lab = np.ascontiguousarray(lbl.get_fdata(dtype=np.float32))
        if not np.array_equal(lab, np.round(lab)):
            return subj, None, "label is not integral-valued"
        lab = np.round(lab).astype(np.int32)
        if lab.min() < 0 or lab.max() > len(FLARE22_CLASSES):
            return subj, None, f"label range [{lab.min()}, {lab.max()}] outside 0..13"

        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(ct_out, np.round(raw).astype(np.int16))
        np.save(lbl_out, lab.astype(np.uint8))
        meta["hu_range"] = [float(raw.min()), float(raw.max())]
        meta["classes_present"] = sorted(int(v) for v in np.unique(lab) if v != 0)
        return subj, meta, None
    except Exception as exc:  # noqa: BLE001
        return subj, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="dir containing images/ and labels/")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    img_dir, lbl_dir = src / "images", src / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SystemExit(f"expected {img_dir} and {lbl_dir}")

    cases = []
    for lp in sorted(lbl_dir.glob("*.nii.gz")):
        subj = lp.name[: -len(".nii.gz")]
        ip = img_dir / f"{subj}_0000.nii.gz"          # nnU-Net channel-0 suffix
        if not ip.exists():
            ip = img_dir / f"{subj}.nii.gz"
        if not ip.exists():
            print(f"  [skip] {subj}: no matching image")
            continue
        cases.append((subj, ip, lp))
    print(f"{len(cases)} cases  {src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    metas, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, i, l, out, args.overwrite) for s, i, l in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            subj, meta, err = fut.result()
            if err:
                errors.append(f"{subj}: {err}")
                print(f"  [FAIL] {subj}: {err}")
            else:
                metas[subj] = meta
            if n % 10 == 0:
                print(f"  {n}/{len(cases)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: metas[k] for k in sorted(metas)}, f, indent=1)

    sp = np.array([m["spacing"] for m in metas.values()])
    print(f"\nwrote {len(metas)} cases, {len(errors)} failed")
    if len(sp):
        print(f"  in-plane spacing {sp[:, 0].min():.3f}-{sp[:, 0].max():.3f} mm  "
              f"z {sorted(set(np.round(sp[:, 2], 3)))} mm")
        print(f"  spacings.json -> {out / 'spacings.json'}")
    for e in errors:
        print(f"  {e}")


if __name__ == "__main__":
    main()

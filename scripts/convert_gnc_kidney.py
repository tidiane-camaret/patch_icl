"""Convert GNC_705 kidney-lesion labels + water-channel Dixon MRI -> per-(subject,visit) .npy
at NATIVE spacing. Mirrors scripts/convert_isles22.py: native grid only, resampling deferred to
the dataloader (src/providers/gnc_kidney.py). Full design/characterization:
docs/datasets/gnc_kidney_lesions.md.

GNC_705 is a local restricted-access National Cohort MRI release, not a public download.

Two things this converter does that no prior converter needed:

1. **One binary plane per class, not one shared multi-valued array.** Each of up to 14
   canonical lesion classes lives in its own small ROI-crop `.nii.gz` under
   `data/{subj}/{visit}/`, sharing the full image's voxel spacing/orientation but a translated
   origin. The origin diff, divided by spacing, is an exact integer voxel offset (verified at
   characterization time) -- `_paint_label` places a class's 0/1 content into a full-image-sized
   canvas at that offset. Classes were checked at conversion time and DO genuinely overlap in a
   meaningful fraction of subjects (a `mask_X.cyst` sits inside its `X` superset — thousands of
   overlapping voxels on some subjects), so painting every class into ONE shared array would
   silently overwrite the superset's voxels with the subset's class value. Instead each present
   class gets its OWN plane, `{subj}_v{visit}/label_{cls}.npy` — see
   `src/providers/gnc_kidney.py`'s module docstring for why the provider needs a matching
   per-class-plane override of `NativeGridProvider`.

2. **Channel selection out of a 4-channel image.** The per-visit MRI is one already-stitched
   4D Dixon file (`links/{subj}/{visit}/3D_GRE_TRA_4/*COMPOSED*.nii`, shape (D,H,W,4)). Channel
   order was verified by correlating each channel against the single-contrast station files
   (corr > 0.99): index 0=opposed-phase, 1=in-phase, 2=fat, 3=water. Only the **water** channel
   (index 3) is converted -- clearest fluid/parenchyma contrast for cystic/complex renal
   lesions, and keeps the model's 2-channel `[image, mask]` input unchanged (no channel-split
   needed for an eval-only source, unlike MSD Prostate's T2+ADC training case).

Data-quality note (found this session, NOT excluded -- see docs #7 step 2 correction): 4 label
files have an oversized canvas (~full native XY extent) around a genuinely tiny lesion (their
own nonzero content bbox is a normal few-cm structure) -- an earlier pass mistakenly read this
as "corrupt", but `_paint_label`'s offset+shape placement handles an oversized container
correctly with no special-casing needed. 2 subjects (130579, 131197) have 6mm Z-spacing instead
of the cohort-typical 3mm -- spacing is always read from each file's own header, never assumed.

Layout written under --out:
    100025_v30/ct_raw.npy          (D,H,W) float32, RAS, native grid, water-channel intensity
    100025_v30/label_{cls}.npy     (D,H,W) uint8 0/1, one file per class PRESENT for this
                                    subject (cls in src.providers.gnc_kidney.GNC_KIDNEY_CLASSES)
    spacings.json                   {subj: {spacing, shape, affine, classes_present}}
    ct_stats.json                   {subj: {clip_lo, clip_hi, mean, std}}  (mri_stats)

Usage:
    python scripts/convert_gnc_kidney.py --workers 16
"""
import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataset import mri_stats  # noqa: E402
from src.providers.gnc_kidney import FILE_TO_CLASS  # noqa: E402

DEFAULT_SRC = "/nfs/data/nii/data0/GNC/GNC_705"
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/gnc_kidney/npy")
WATER_CHANNEL = 3  # verified via correlation against the single-contrast station-4 files


def _find_labels(data_root: Path, subj: str, visit: str) -> list[tuple[str, Path]]:
    vdir = data_root / subj / visit
    out = []
    for fname, cls in FILE_TO_CLASS.items():
        p = vdir / fname
        if p.exists():
            out.append((cls, p))
    return out


def _find_composed_image(links_root: Path, subj: str, visit: str) -> Path | None:
    hits = glob.glob(str(links_root / subj / visit / "3D_GRE_TRA_*" / "*COMPOSED*.nii"))
    return Path(hits[0]) if len(hits) == 1 else None


def _affine_voxel_offset(affine_a, affine_b, atol=0.05):
    """Integer voxel offset of grid B's origin within grid A, or None if the two affines
    don't share the same rotation/spacing (only a translation apart) within `atol` voxels."""
    lin_a, lin_b = affine_a[:3, :3], affine_b[:3, :3]
    if not np.allclose(lin_a, lin_b, atol=1e-3):
        return None
    spacing = np.abs(np.diag(lin_a))
    spacing = np.where(spacing == 0, 1.0, spacing)
    signs = np.sign(np.diag(lin_a))
    signs = np.where(signs == 0, 1.0, signs)
    raw_offset = (affine_b[:3, 3] - affine_a[:3, 3]) / (signs * spacing)
    offset = np.round(raw_offset).astype(int)
    if np.max(np.abs(raw_offset - offset)) > atol:
        return None
    return offset


def _make_plane(image_shape, image_affine, path: Path) -> tuple[np.ndarray | None, str]:
    """Build one class's own full-image-sized 0/1 plane (uint8) from its ROI-crop file.
    Returns (plane_or_None, error_or_empty)."""
    im = nib.load(str(path))
    data = np.asarray(im.dataobj)
    if not np.array_equal(data, np.round(data)):
        return None, "mask is not integral-valued"
    data = np.round(data).astype(np.int32)
    if data.min() < 0 or data.max() > 1:
        return None, f"mask range [{data.min()}, {data.max()}] outside 0..1"

    offset = _affine_voxel_offset(image_affine, np.asarray(im.affine, dtype=np.float64))
    if offset is None:
        return None, "label affine is not a pure integer-voxel translation of the image affine"

    lo = offset
    hi = offset + np.array(data.shape)
    # Clip to the canvas bounds (defensive -- not observed, but a label could in principle
    # extend past the image's own coverage at conversion time).
    src_lo = np.maximum(0, -lo)
    src_hi = np.array(data.shape) - np.maximum(0, hi - np.array(image_shape))
    dst_lo = np.maximum(lo, 0)
    dst_hi = np.minimum(hi, image_shape)
    if np.any(src_hi <= src_lo) or np.any(dst_hi <= dst_lo):
        return None, "label falls entirely outside the image canvas after clipping"

    plane = np.zeros(image_shape, dtype=np.uint8)
    sl_src = tuple(slice(a, b) for a, b in zip(src_lo, src_hi))
    sl_dst = tuple(slice(a, b) for a, b in zip(dst_lo, dst_hi))
    plane[sl_dst] = (data[sl_src] > 0).astype(np.uint8)
    if not plane.any():
        return None, "plane is empty after clipping (no content survives)"
    return plane, ""


def convert_one(subj: str, visit: str, img_path: Path, labels: list[tuple[str, Path]],
                out_dir: Path, overwrite: bool):
    """Convert one (subject, visit). Returns (key, spacings_meta, ct_stats_entry, error,
    label_errors)."""
    key = f"{subj}_v{visit}"
    try:
        subj_dir = out_dir / key
        img_out = subj_dir / "ct_raw.npy"

        raw_img = nib.load(str(img_path))
        raw_affine = np.asarray(raw_img.affine, dtype=np.float64)
        water = np.asarray(raw_img.dataobj[..., WATER_CHANNEL], dtype=np.float32)

        label_errors = []
        planes = {}   # cls -> full-image-shaped 0/1 plane, PRE-reorientation
        for cls, path in labels:
            plane, err = _make_plane(water.shape, raw_affine, path)
            if plane is None:
                label_errors.append(f"{cls}: {err}")
            else:
                planes[cls] = plane

        # Reorient image + every class plane via the SAME affine-driven canonical transform,
        # so they all land in the same final RAS space (mirrors ISLES22/Shifts-MS).
        img_can = nib.as_closest_canonical(nib.Nifti1Image(water, raw_affine))
        affine = np.asarray(img_can.affine, dtype=np.float64)
        spacing = [float(x) for x in nib.affines.voxel_sizes(affine)[:3]]
        meta = {"spacing": spacing, "shape": [int(x) for x in img_can.shape[:3]],
                "affine": affine.tolist(), "classes_present": sorted(planes)}

        raw = np.ascontiguousarray(img_can.get_fdata(dtype=np.float32))
        stats = mri_stats(raw)

        lbl_outs = {cls: subj_dir / f"label_{cls}.npy" for cls in planes}
        if not overwrite and img_out.exists() and all(p.exists() for p in lbl_outs.values()):
            return key, meta, stats, None, label_errors

        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(img_out, raw.astype(np.float32))
        for cls, plane in planes.items():
            plane_can = nib.as_closest_canonical(nib.Nifti1Image(plane, raw_affine))
            np.save(lbl_outs[cls], np.ascontiguousarray(np.asarray(plane_can.dataobj)).astype(np.uint8))
        return key, meta, stats, None, label_errors
    except Exception as exc:  # noqa: BLE001
        return key, None, None, f"{type(exc).__name__}: {exc}", []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="GNC_705 root (data/, links/)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    data_root, links_root = src / "data", src / "links"
    if not data_root.is_dir() or not links_root.is_dir():
        raise SystemExit(f"expected {data_root} and {links_root}")

    # Every (subject,visit) with >=1 canonical label file. `data_root` has 30,385 subject
    # dirs -- a serial os.listdir walk over all of them is NFS-latency-bound and takes many
    # minutes (see scripts/inspect_gnc_kidney.py); a thread pool hides that latency in ~5min.
    def _scan_subject(subj_dir):
        subj = subj_dir.name
        out_cases = []
        try:
            visit_names = [p.name for p in subj_dir.iterdir()]
        except OSError:
            return out_cases
        for visit in visit_names:
            labels = _find_labels(data_root, subj, visit)
            if labels:
                out_cases.append((subj, visit, labels))
        return out_cases

    from concurrent.futures import ThreadPoolExecutor
    subj_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    print(f"scanning {len(subj_dirs)} subjects for labeled visits...")
    labeled = []
    with ThreadPoolExecutor(max_workers=128) as ex:
        for out_cases in ex.map(_scan_subject, subj_dirs):
            labeled.extend(out_cases)

    cases = []
    for subj, visit, labels in sorted(labeled):
        img_path = _find_composed_image(links_root, subj, visit)
        if img_path is None:
            print(f"  [skip] {subj}/{visit}: no unique COMPOSED image found")
            continue
        cases.append((subj, visit, img_path, labels))
    print(f"{len(cases)} labeled (subject,visit) cases  {src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    spacings, ct_stats, errors = {}, {}, []
    all_label_errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, v, i, l, out, args.overwrite)
                for s, v, i, l in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            key, meta, stats, err, label_errors = fut.result()
            if err:
                errors.append(f"{key}: {err}")
                print(f"  [FAIL] {key}: {err}")
            else:
                spacings[key] = meta
                ct_stats[key] = stats
            for le in label_errors:
                all_label_errors.append(f"{key}: {le}")
            if n % 100 == 0:
                print(f"  {n}/{len(cases)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: spacings[k] for k in sorted(spacings)}, f, indent=1)
    with open(out / "ct_stats.json", "w") as f:
        json.dump({k: ct_stats[k] for k in sorted(ct_stats)}, f, indent=1)

    sp = np.array([m["spacing"] for m in spacings.values()])
    sh = np.array([m["shape"] for m in spacings.values()])
    print(f"\nwrote {len(spacings)} cases, {len(errors)} failed, "
          f"{len(all_label_errors)} individual label rejections")
    if len(sp):
        print(f"  spacing: min {sp.min(0).round(2)} max {sp.max(0).round(2)}")
        print(f"  shape: min {sh.min(0)} median {np.median(sh, 0)} max {sh.max(0)}")
        print(f"  -> {out/'spacings.json'}, {out/'ct_stats.json'}")
    for e in errors:
        print(f"  [FAIL] {e}")
    for le in all_label_errors:
        print(f"  [label-reject] {le}")


if __name__ == "__main__":
    main()

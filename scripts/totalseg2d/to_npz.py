"""
Export TotalSegmentator 2D cross-sections to an .npz for the 2D experiments.

Picks ONE axial cross-section per subject (densest in the most common classes), and
stores it at a FIXED physical scale with RAW intensities, leaving normalization to the
dataloader. Two deliberate choices vs scripts/convert_to_npy.py:

1. Fixed mm/pixel (not longest-axis -> cube). Each subject's chosen slice is resampled
   in-plane to a shared --mm_per_px and placed on a fixed --size grid centered on the
   label. This fixes the cross-subject scale drift that longest-axis normalization
   causes (a whole-body scan and an abdomen-only scan end up at the same anatomy scale).

2. Raw int16 HU, no normalization. The image is stored as int16 Hounsfield units
   (exact, same bytes as float16). Clip/z-score/windowing is deferred to load time for
   flexibility. (label is uint8 multi-class, TotalSeg class index 1..117, 0=bg.)

Slice selection (per subject): score each axial slice by a soft area-ramp over global
class frequencies (see build_totalseg_2d_manifest.py), but with areas measured in
PHYSICAL units (voxel count x in-plane mm^2, expressed in output pixels at --mm_per_px),
so the score and its argmax are FOV-consistent across subjects. label.npy is native and
canonical (axis 2 = axial); only ct.nii.gz is read, for raw HU.

Output: a single npz (default <totalseg2d data dir>/totalseg2d_{size}.npz) with, per split:
    {split}_images   (N, size, size)  int16  raw HU (air pad = AIR_HU)
    {split}_label    (N, size, size)  uint8  multi-class
    {split}_subjects (N,)             <U8
    {split}_z        (N,)             int16   chosen native axial index
    {split}_spacing  (N, 3)           float32 native voxel size (mm)
plus scalars: mm_per_px, size, air_hu, and class_names (ALL_CLASSES).

Usage
-----
python scripts/totalseg2d/to_npz.py                          # all subjects, size 256, 2mm/px
python scripts/totalseg2d/to_npz.py --size 128 --mm_per_px 3.0 --workers 8
python scripts/totalseg2d/to_npz.py --splits test val --max_subjects 20   # quick run
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.totalseg_classes import ALL_CLASSES  # noqa: E402
from build_totalseg_2d_manifest import (  # noqa: E402
    AXIAL_AXIS, load_splits, load_weights, resolve_totalseg_root,
)

AIR_HU = -1024  # pad value for out-of-frame image pixels (CT air)


# ── Slice selection (physical areas) ──────────────────────────────────────────

def select_slice(label_nat, spacing, weight, mm_per_px, noise_floor, area_cap):
    """argmax axial slice by soft area-ramp over global class frequency.

    Areas are physical: voxel count x (sp_in_plane mm^2), converted to output pixels at
    mm_per_px (so noise_floor / area_cap are in the SAME units as the stored grid and
    the score is comparable across subjects regardless of native spacing/FOV).
    Returns (z, score, [(class_idx, native_voxel_count), ...]).
    """
    sp = spacing
    in_plane_axes = [a for a in range(3) if a != AXIAL_AXIS]
    px_mm2 = sp[in_plane_axes[0]] * sp[in_plane_axes[1]]  # native in-plane voxel area
    to_px = px_mm2 / (mm_per_px * mm_per_px)              # native voxels -> output px

    nz = np.where(label_nat.any(axis=tuple(in_plane_axes)))[0]  # non-empty axial idxs
    best = (-1.0, 0)
    for z in nz:
        sl = np.take(label_nat, z, axis=AXIAL_AXIS)
        vals, cnts = np.unique(sl, return_counts=True)
        score = 0.0
        for v, c in zip(vals, cnts):
            if v == 0:
                continue
            px = c * to_px
            if px >= noise_floor:
                score += weight[int(v)] * min(1.0, px / area_cap)
        if score > best[0]:
            best = (score, int(z))
    score, z = best
    sl = np.take(label_nat, z, axis=AXIAL_AXIS)
    vals, cnts = np.unique(sl, return_counts=True)
    present = [(int(v), int(c)) for v, c in zip(vals, cnts)
               if v != 0 and c * to_px >= noise_floor]
    return z, score, present


# ── In-plane resample + centered placement ────────────────────────────────────

def _resample(sl, zoom_yx, order, aa):
    """Resample a 2D slice by zoom_yx (per-axis factor). Gaussian AA on downsampling."""
    if aa and order > 0:
        sigma = [max(0.0, 0.5 * (1.0 / z - 1.0)) for z in zoom_yx]
        if any(s > 0.1 for s in sigma):
            sl = ndi.gaussian_filter(sl, sigma=sigma)
    return ndi.zoom(sl, zoom_yx, order=order)


def _place_centered(arr, size, center_yx, pad_value):
    """Paste arr onto a (size,size) canvas so center_yx maps to the grid center."""
    out = np.full((size, size), pad_value, dtype=arr.dtype)
    oy = size // 2 - int(round(center_yx[0]))
    ox = size // 2 - int(round(center_yx[1]))
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    h = min(arr.shape[0] - sy0, size - dy0)
    w = min(arr.shape[1] - sx0, size - dx0)
    if h > 0 and w > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = arr[sy0:sy0 + h, sx0:sx0 + w]
    return out


def render_slice(ct_slice, lab_slice, spacing, mm_per_px, size):
    """Resample one axial (image, label) slice to mm_per_px and center on the label.

    Returns (img int16 (size,size) raw HU, lab uint8 (size,size)).
    """
    in_plane_axes = [a for a in range(3) if a != AXIAL_AXIS]
    zoom_yx = (spacing[in_plane_axes[0]] / mm_per_px, spacing[in_plane_axes[1]] / mm_per_px)

    img_r = _resample(ct_slice.astype(np.float32), zoom_yx, order=1, aa=True)
    lab_r = _resample(lab_slice, zoom_yx, order=0, aa=False)  # uint8 nearest

    ys, xs = np.nonzero(lab_r)
    center = (ys.mean(), xs.mean()) if len(ys) else (lab_r.shape[0] / 2, lab_r.shape[1] / 2)

    img_out = _place_centered(np.round(img_r).astype(np.int16), size, center, AIR_HU)
    lab_out = _place_centered(lab_r.astype(np.uint8), size, center, 0)
    return img_out, lab_out


# ── Per-subject worker ────────────────────────────────────────────────────────

def _worker(args):
    sid, subj_dir, spacing, split, weight, mm_per_px, size, noise_floor, area_cap = args
    try:
        subj_dir = Path(subj_dir)
        label_nat = np.load(subj_dir / "label.npy")  # native, canonical, axis2=axial
        spacing = np.asarray(spacing, dtype=np.float64)
        z, score, present = select_slice(
            label_nat, spacing, weight, mm_per_px, noise_floor, area_cap)

        ct_img = nib.as_closest_canonical(nib.load(str(subj_dir / "ct.nii.gz")))
        ct = ct_img.get_fdata(dtype=np.float32)  # raw HU
        if ct.shape != label_nat.shape:
            return sid, None, f"shape mismatch ct{ct.shape} vs label{label_nat.shape}"

        ct_slice = np.take(ct, z, axis=AXIAL_AXIS)
        lab_slice = np.take(label_nat, z, axis=AXIAL_AXIS)
        img_out, lab_out = render_slice(ct_slice, lab_slice, spacing, mm_per_px, size)
        rec = dict(sid=sid, split=split, z=z, spacing=spacing.astype(np.float32),
                   score=score, n_cls=len(present), image=img_out, label=lab_out)
        return sid, rec, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return sid, None, traceback.format_exc()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--totalseg_root", default=None, help="Dataset root (auto-detected if absent)")
    p.add_argument("--size", type=int, default=256, help="Output square grid (pixels)")
    p.add_argument("--mm_per_px", type=float, default=2.0, help="Fixed in-plane resolution (mm/pixel)")
    p.add_argument("--noise_floor", type=int, default=10, help="Min output-px area for a class to score")
    p.add_argument("--area_cap", type=int, default=100, help="Output-px area at which a class scores full weight")
    p.add_argument("--splits", nargs="*", default=None, help="Restrict to these subject splits")
    p.add_argument("--max_subjects", type=int, default=None, help="Cap subjects (debug)")
    p.add_argument("--workers", type=int, default=8, help="Process pool size")
    p.add_argument("--out", default=None,
                   help="Output npz (default <totalseg root>/../totalseg2d/totalseg2d_{size}.npz)")
    args = p.parse_args()

    root = Path(resolve_totalseg_root(args.totalseg_root))   # source, from configs/cluster/*.yaml
    out_root = root.parent / "totalseg2d"                    # sibling of the source dir
    weight = load_weights(root)
    splits = load_splits(root)
    import json
    spacings = json.load(open(root / "spacings.json"))

    subjects = []
    for d in sorted(root.glob("s*")):
        if not d.is_dir():
            continue
        sid = d.name
        split = splits.get(sid, "unknown")
        if args.splits and split not in args.splits:
            continue
        if (d / "label.npy").exists() and (d / "ct.nii.gz").exists() and sid in spacings:
            subjects.append((sid, d, split))
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]

    out_path = Path(args.out) if args.out else out_root / f"totalseg2d_{args.size}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fov = args.size * args.mm_per_px
    print(f"Root      : {root}")
    print(f"Output    : size={args.size}px  mm/px={args.mm_per_px}  FOV={fov:.0f}mm  raw int16 HU")
    print(f"Subjects  : {len(subjects)}"
          + (f"  (splits={args.splits})" if args.splits else ""))

    tasks = [(sid, str(d), spacings[sid]["spacing"], split, weight,
              args.mm_per_px, args.size, args.noise_floor, args.area_cap)
             for sid, d, split in subjects]

    by_split: dict[str, list] = {}
    n_done = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            sid, rec, err = fut.result()
            if err is not None:
                n_err += 1
                print(f"  [skip] {sid}: {err.splitlines()[-1]}")
                continue
            n_done += 1
            by_split.setdefault(rec["split"], []).append(rec)
            if n_done % 200 == 0:
                print(f"  ...{n_done}/{len(tasks)} subjects")

    # Pack per split. Sort by subject id for determinism.
    arrays: dict[str, np.ndarray] = {
        "mm_per_px": np.float32(args.mm_per_px),
        "size": np.int32(args.size),
        "air_hu": np.int32(AIR_HU),
        "class_names": np.array(ALL_CLASSES),
    }
    for split, recs in by_split.items():
        recs.sort(key=lambda r: r["sid"])
        arrays[f"{split}_images"]   = np.stack([r["image"] for r in recs]).astype(np.int16)
        arrays[f"{split}_label"]    = np.stack([r["label"] for r in recs]).astype(np.uint8)
        arrays[f"{split}_subjects"] = np.array([r["sid"] for r in recs])
        arrays[f"{split}_z"]        = np.array([r["z"] for r in recs], dtype=np.int16)
        arrays[f"{split}_spacing"]  = np.stack([r["spacing"] for r in recs]).astype(np.float32)

    np.savez_compressed(out_path, **arrays)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {n_done} slices ({n_err} errors) -> {out_path}  ({size_mb:.1f} MB)")
    for split, recs in sorted(by_split.items()):
        print(f"  {split:>6}: {len(recs)} slices")


if __name__ == "__main__":
    main()

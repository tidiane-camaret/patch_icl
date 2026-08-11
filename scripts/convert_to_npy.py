"""
Convert TotalSegmentator / TotalSegMRI subjects from .nii.gz to .npy for fast data loading.

Per subject written (always):
  ct.npy     — float16, normalised, native resolution (D,H,W)
  label.npy  — uint8, merged label volume using ALL_CLASSES ordering (0=bg), native resolution

With --size D H W (e.g. --size 128 128 128), also writes:
  ct_DxHxW.npy     — float16, isotropic resize (longest axis → D) + Gaussian AA + zero-pad to cube
  label_DxHxW.npy  — uint8, same isotropic resize with nearest-neighbour interpolation

Modality handling (--modality):
  ct  (default) : reads ct.nii.gz; clips HU to [CT_CLIP_MIN, CT_CLIP_MAX]; global z-score.
  mri           : reads mri.nii.gz; clips to [0, per-volume 99.5th percentile of foreground];
                  per-volume z-score (foreground mean/std).  Output still named ct.npy so the
                  dataloader needs no changes.

Usage
-----
  python scripts/convert_to_npy.py [--data DIR] [--workers N] [--overwrite]
  python scripts/convert_to_npy.py --size 128 128 128
  python scripts/convert_to_npy.py --data /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalsegmri --modality mri --size 128 128 128 --overwrite
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.totalseg_classes import ALL_CLASSES
from src.totalseg_dataset import (
    CT_CLIP_MIN, CT_CLIP_MAX, CT_MEAN, CT_STD,
    normalize_ct, mri_stats, normalize_mri,
)

ROOT = Path(__file__).resolve().parents[1]


def _iso_resize(vol: np.ndarray, target: tuple, order: int = 1, aa: bool = True,
                spacing: tuple | None = None) -> np.ndarray:
    """Isotropic resize: scale longest physical axis to target[0], zero-pad shorter axes.

    With aa=True (and order > 0), applies a per-axis Gaussian blur before
    downsampling (σ = 0.5*(s/n − 1)) to suppress aliasing.  Upsampled axes skip
    the blur.  Uses order=0 for label volumes (nearest-neighbour, no AA needed).

    spacing: voxel size in mm (D, H, W). When provided, scaling is based on
    physical extent (shape × spacing) rather than voxel count alone — critical
    for anisotropic 2-D multi-slice MRI where slice thickness >> in-plane spacing.
    """
    T = target[0]
    if spacing is not None:
        # Scale so the longest physical dimension fits in T voxels; preserve aspect ratio.
        phys = tuple(s * sp for s, sp in zip(vol.shape, spacing))
        new_shape = tuple(min(T, max(1, round(T * p / max(phys)))) for p in phys)
    else:
        scale = T / max(vol.shape)
        new_shape = tuple(min(T, max(1, round(s * scale))) for s in vol.shape)
    if aa and order > 0:
        sigma = [max(0.0, 0.5 * (s / n - 1)) for s, n in zip(vol.shape, new_shape)]
        if any(s > 0.1 for s in sigma):
            vol = ndi.gaussian_filter(vol, sigma=sigma)
    zoom = tuple(n / s for n, s in zip(new_shape, vol.shape))
    resized = ndi.zoom(vol, zoom, order=order)
    out = np.zeros(target, dtype=vol.dtype)
    pad = [(T - s) // 2 for s in new_shape]
    sl = tuple(slice(p, p + s) for p, s in zip(pad, new_shape))
    out[sl] = resized
    return out


def _default_data_dir() -> str:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config")
    return cfg.paths.totalseg

_CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(ALL_CLASSES)}  # 1-indexed

# Verify constants match the expected fingerprint values at import time (fast sanity check).
assert CT_CLIP_MIN == -1007.0 and CT_CLIP_MAX == 1573.0, "unexpected CT clip constants"


# Normalisation delegates to the shared helpers in src.totalseg_dataset so the values
# written here are byte-identical to what the raw-CT loader path produces on the fly.
_normalise_ct = normalize_ct


def convert_subject(args: tuple) -> tuple[str, str, list | None, list | None, dict | None]:
    """Convert one subject.

    Returns (subject_id, status, native_spacing, native_shape, mri_stats).
    native_spacing and native_shape are None when the NIfTI was not read
    (skipped subjects or sized-only runs where the native files already exist).
    mri_stats is the per-volume MRI normalisation stats dict (for ct_stats.json) when
    store_raw + modality=mri, else None.
    """
    subj_dir, overwrite, size, modality, store_raw = args
    subj_dir = Path(subj_dir)
    subj = subj_dir.name

    ct_out    = subj_dir / "ct.npy"
    ct_raw_out = subj_dir / "ct_raw.npy"
    label_out = subj_dir / "label.npy"

    size_str    = f"{size[0]}x{size[1]}x{size[2]}" if size else None
    ct_sized    = subj_dir / f"ct_{size_str}.npy"  if size else None
    label_sized = subj_dir / f"label_{size_str}.npy" if size else None

    need_native = overwrite or not (ct_out.exists() and label_out.exists())
    need_raw    = store_raw and (overwrite or not ct_raw_out.exists())
    need_sized  = size is not None and (
        overwrite or not (ct_sized.exists() and label_sized.exists())
    )

    if not need_native and not need_raw and not need_sized:
        return subj, "skip", None, None, None

    try:
        vol = label = None
        native_spacing = native_shape = None
        stats = None

        if need_native or need_raw:
            img_fname = "mri.nii.gz" if modality == "mri" else "ct.nii.gz"
            img_path  = subj_dir / img_fname
            ct_img    = nib.as_closest_canonical(nib.load(str(img_path)))
            native_spacing = [float(x) for x in nib.affines.voxel_sizes(ct_img.affine)[:3]]
            raw = ct_img.get_fdata(dtype=np.float32)   # raw intensities (HU for CT)
            native_shape = list(raw.shape)

            if modality == "mri":
                stats = mri_stats(raw)                  # whole-volume stats (sidecar)
                if need_raw:
                    # MRI has no canonical integer range; keep raw as float16.
                    np.save(ct_raw_out, raw.astype(np.float16))
                vol = normalize_mri(raw, stats)
            else:
                if need_raw:
                    # CT HU are integers -> int16 is lossless and same 2 B/voxel as float16.
                    np.save(ct_raw_out, np.clip(np.round(raw), -32768, 32767).astype(np.int16))
                vol = _normalise_ct(raw)

        if need_native:
            seg_dir = subj_dir / "segmentations"
            label = np.zeros(vol.shape, dtype=np.uint8)
            for cls, idx in _CLASS_TO_IDX.items():
                mask_path = seg_dir / f"{cls}.nii.gz"
                if not mask_path.exists():
                    continue
                mask = nib.as_closest_canonical(nib.load(str(mask_path))).get_fdata(dtype=np.float32) > 0
                label[mask] = idx

            np.save(ct_out, vol.astype(np.float16))
            np.save(label_out, label)

        if need_sized:
            if vol is None:
                vol   = np.load(ct_out,    mmap_mode="r").astype(np.float32)
                label = np.load(label_out, mmap_mode="r")
            if native_shape is None:
                native_shape = list(vol.shape)
            if native_spacing is None:
                # Read spacing from the NIfTI header (header-only, no data load)
                img_fname = "mri.nii.gz" if modality == "mri" else "ct.nii.gz"
                img_path  = subj_dir / img_fname
                if img_path.exists():
                    canonical = nib.as_closest_canonical(nib.load(str(img_path)))
                    native_spacing = [float(x) for x in nib.affines.voxel_sizes(canonical.affine)[:3]]

            sp = tuple(native_spacing) if native_spacing else None
            np.save(ct_sized,    _iso_resize(vol,   size, order=1, aa=True,  spacing=sp).astype(np.float16))
            np.save(label_sized, _iso_resize(label, size, order=0, aa=False, spacing=sp))

    except Exception:
        return subj, traceback.format_exc(), None, None, None

    return subj, "ok", native_spacing, native_shape, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        help="dataset root; defaults to paths.totalseg in configs/config.yaml")
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count()),
                        help="parallel worker processes (default: min(20, cpu_count))")
    parser.add_argument("--overwrite", action="store_true",
                        help="reconvert even if .npy files already exist")
    parser.add_argument("--size", nargs=3, type=int, metavar=("D", "H", "W"),
                        default=None,
                        help="also write pre-resized ct_DxHxW.npy and label_DxHxW.npy")
    parser.add_argument("--modality", choices=["ct", "mri"], default="ct",
                        help="ct (default): reads ct.nii.gz with HU normalisation; "
                             "mri: reads mri.nii.gz with per-volume percentile z-score")
    parser.add_argument("--store-raw", action="store_true",
                        help="also write native ct_raw.npy (raw intensities: int16 HU for CT, "
                             "float16 for MRI) so the loader can normalise on the fly. For MRI "
                             "also writes per-volume stats to ct_stats.json.")
    args = parser.parse_args()

    data_dir = Path(args.data) if args.data else Path(_default_data_dir())
    subjects = sorted(p for p in data_dir.iterdir() if p.is_dir())
    total = len(subjects)
    size = tuple(args.size) if args.size else None
    size_str = f"{size[0]}x{size[1]}x{size[2]}" if size else "native only"
    print(f"Found {total} subjects  |  workers={args.workers}"
          f"  |  overwrite={args.overwrite}  |  size={size_str}  |  modality={args.modality}")

    tasks = [(str(s), args.overwrite, size, args.modality, args.store_raw) for s in subjects]

    # spacings.json: {"s0000": {"spacing": [dx,dy,dz], "shape": [D,H,W]}, ...}
    # Merged with any existing entries so incremental runs stay consistent.
    spacings_path = data_dir / "spacings.json"
    spacings: dict = {}
    if spacings_path.exists():
        with open(spacings_path) as f:
            spacings = json.load(f)

    # ct_stats.json: {"s0000": {clip_lo, clip_hi, mean, std}} — per-volume MRI norm stats,
    # only written by --store-raw --modality mri. Merged with any existing entries.
    stats_path = data_dir / "ct_stats.json"
    ct_stats: dict = {}
    if stats_path.exists():
        with open(stats_path) as f:
            ct_stats = json.load(f)

    done = ok = skipped = errors = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for subj, status, native_spacing, native_shape, subj_stats in pool.imap_unordered(
            convert_subject, tasks, chunksize=1
        ):
            done += 1
            if status == "ok":
                ok += 1
                if native_spacing is not None and native_shape is not None:
                    spacings[subj] = {"spacing": native_spacing, "shape": native_shape}
                if subj_stats is not None:
                    ct_stats[subj] = subj_stats
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                print(f"\n[ERROR] {subj}:\n{status}")

            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"\r  {done}/{total}  ok={ok}  skip={skipped}  err={errors}"
                f"  {rate:.1f} subj/s  ETA {eta/60:.0f}m",
                end="", flush=True,
            )

    if spacings:
        with open(spacings_path, "w") as f:
            json.dump(spacings, f)
        print(f"\nSpacings written to {spacings_path}  ({len(spacings)} subjects)")

    if ct_stats:
        with open(stats_path, "w") as f:
            json.dump(ct_stats, f)
        print(f"MRI norm stats written to {stats_path}  ({len(ct_stats)} subjects)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  —  ok={ok}  skipped={skipped}  errors={errors}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()

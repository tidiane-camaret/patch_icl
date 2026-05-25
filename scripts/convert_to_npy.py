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
  python scripts/convert_to_npy.py --data /path/to/totalsegmri --modality mri --size 128 128 128
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
from src.totalseg_dataset import CT_CLIP_MIN, CT_CLIP_MAX, CT_MEAN, CT_STD

ROOT = Path(__file__).resolve().parents[1]


def _iso_resize(vol: np.ndarray, target: tuple, order: int = 1, aa: bool = True) -> np.ndarray:
    """Isotropic resize: scale longest axis to target[0], zero-pad shorter axes.

    With aa=True (and order > 0), applies a per-axis Gaussian blur before
    downsampling (σ = 0.5*(s/n − 1)) to suppress aliasing.  Upsampled axes skip
    the blur.  Uses order=0 for label volumes (nearest-neighbour, no AA needed).
    """
    T = target[0]
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


def _normalise_ct(vol: np.ndarray) -> np.ndarray:
    """Clip to CT HU range and global z-score normalise."""
    vol = np.clip(vol, CT_CLIP_MIN, CT_CLIP_MAX)
    return (vol - CT_MEAN) / CT_STD


def _normalise_mri(vol: np.ndarray) -> np.ndarray:
    """Per-volume MRI normalisation: clip bright outliers then z-score over foreground.

    1. Clip to [0, 99.5th percentile of foreground] — removes coil/air outliers.
    2. Z-score using foreground (>0) mean and std — accounts for field-strength /
       protocol variability across scanners.
    """
    fg = vol[vol > 0]
    if fg.size == 0:
        return vol.astype(np.float32)
    p995 = float(np.percentile(fg, 99.5))
    vol  = np.clip(vol, 0.0, p995)
    fg   = vol[vol > 0]
    mean, std = float(fg.mean()), float(fg.std())
    if std < 1e-6:
        std = 1.0
    return (vol - mean) / std


def convert_subject(args: tuple) -> tuple[str, str, list | None, list | None]:
    """Convert one subject.

    Returns (subject_id, status, native_spacing, native_shape).
    native_spacing and native_shape are None when the NIfTI was not read
    (skipped subjects or sized-only runs where the native files already exist).
    """
    subj_dir, overwrite, size, modality = args
    subj_dir = Path(subj_dir)
    subj = subj_dir.name

    ct_out    = subj_dir / "ct.npy"
    label_out = subj_dir / "label.npy"

    size_str    = f"{size[0]}x{size[1]}x{size[2]}" if size else None
    ct_sized    = subj_dir / f"ct_{size_str}.npy"  if size else None
    label_sized = subj_dir / f"label_{size_str}.npy" if size else None

    need_native = overwrite or not (ct_out.exists() and label_out.exists())
    need_sized  = size is not None and (
        overwrite or not (ct_sized.exists() and label_sized.exists())
    )

    if not need_native and not need_sized:
        return subj, "skip", None, None

    try:
        vol = label = None
        native_spacing = native_shape = None

        if need_native:
            img_fname = "mri.nii.gz" if modality == "mri" else "ct.nii.gz"
            img_path  = subj_dir / img_fname
            ct_img    = nib.as_closest_canonical(nib.load(str(img_path)))
            native_spacing = [float(x) for x in ct_img.header.get_zooms()[:3]]
            vol = ct_img.get_fdata(dtype=np.float32)
            native_shape = list(vol.shape)

            if modality == "mri":
                vol = _normalise_mri(vol)
            else:
                vol = _normalise_ct(vol)

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

            np.save(ct_sized,    _iso_resize(vol,   size, order=1, aa=True).astype(np.float16))
            np.save(label_sized, _iso_resize(label, size, order=0, aa=False))

    except Exception:
        return subj, traceback.format_exc(), None, None

    return subj, "ok", native_spacing, native_shape


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
    args = parser.parse_args()

    data_dir = Path(args.data) if args.data else Path(_default_data_dir())
    subjects = sorted(p for p in data_dir.iterdir() if p.is_dir())
    total = len(subjects)
    size = tuple(args.size) if args.size else None
    size_str = f"{size[0]}x{size[1]}x{size[2]}" if size else "native only"
    print(f"Found {total} subjects  |  workers={args.workers}"
          f"  |  overwrite={args.overwrite}  |  size={size_str}  |  modality={args.modality}")

    tasks = [(str(s), args.overwrite, size, args.modality) for s in subjects]

    # spacings.json: {"s0000": {"spacing": [dx,dy,dz], "shape": [D,H,W]}, ...}
    # Merged with any existing entries so incremental runs stay consistent.
    spacings_path = data_dir / "spacings.json"
    spacings: dict = {}
    if spacings_path.exists():
        with open(spacings_path) as f:
            spacings = json.load(f)

    done = ok = skipped = errors = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for subj, status, native_spacing, native_shape in pool.imap_unordered(
            convert_subject, tasks, chunksize=1
        ):
            done += 1
            if status == "ok":
                ok += 1
                if native_spacing is not None and native_shape is not None:
                    spacings[subj] = {"spacing": native_spacing, "shape": native_shape}
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

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  —  ok={ok}  skipped={skipped}  errors={errors}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()

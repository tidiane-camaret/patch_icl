"""
Convert TotalSegmentator subjects from .nii.gz to .npy for fast data loading.

Per subject written (always):
  ct.npy     — float16, HU-clipped & z-score normalised, native resolution (D,H,W)
  label.npy  — uint8, merged label volume using ALL_CLASSES ordering (0=bg), native resolution

With --size D H W (e.g. --size 64 64 64), also writes:
  ct_DxHxW.npy     — float16, isotropic resize (longest axis → D) + Gaussian AA + zero-pad to cube
  label_DxHxW.npy  — uint8, same isotropic resize with nearest-neighbour interpolation

float16 halves ct disk use vs float32 with negligible precision loss after normalisation.

Usage
-----
  python scripts/convert_to_npy.py [--data DIR] [--workers N] [--overwrite]
  python scripts/convert_to_npy.py --size 64 64 64
  python scripts/convert_to_npy.py --size 64 64 64 --overwrite
"""

import argparse
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


def convert_subject(args: tuple) -> tuple[str, str]:
    """Convert one subject.  Returns (subject_id, 'ok' | 'skip' | error_message)."""
    subj_dir, overwrite, size = args
    subj_dir = Path(subj_dir)
    subj = subj_dir.name

    ct_out    = subj_dir / "ct.npy"
    label_out = subj_dir / "label.npy"

    size_str    = f"{size[0]}x{size[1]}x{size[2]}" if size else None
    ct_sized    = subj_dir / f"ct_{size_str}.npy"    if size else None
    label_sized = subj_dir / f"label_{size_str}.npy" if size else None

    need_native = overwrite or not (ct_out.exists() and label_out.exists())
    need_sized  = size is not None and (
        overwrite or not (ct_sized.exists() and label_sized.exists())
    )

    if not need_native and not need_sized:
        return subj, "skip"

    try:
        vol = label = None

        if need_native:
            ct_path = subj_dir / "ct.nii.gz"
            ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
            vol = ct_img.get_fdata(dtype=np.float32)
            vol = np.clip(vol, CT_CLIP_MIN, CT_CLIP_MAX)
            vol = (vol - CT_MEAN) / CT_STD   # z-score; float16 covers [-1.66, +3.44]

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

            np.save(ct_sized,    _iso_resize(vol,   size, order=1, aa=True).astype(np.float16))
            np.save(label_sized, _iso_resize(label, size, order=0, aa=False))

    except Exception:
        return subj, traceback.format_exc()

    return subj, "ok"


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
    args = parser.parse_args()

    data_dir = Path(args.data) if args.data else Path(_default_data_dir())
    subjects = sorted(p for p in data_dir.iterdir() if p.is_dir())
    total = len(subjects)
    size = tuple(args.size) if args.size else None
    size_str = f"{size[0]}x{size[1]}x{size[2]}" if size else "native only"
    print(f"Found {total} subjects  |  workers={args.workers}"
          f"  |  overwrite={args.overwrite}  |  size={size_str}")

    tasks = [(str(s), args.overwrite, size) for s in subjects]

    done = ok = skipped = errors = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for subj, status in pool.imap_unordered(convert_subject, tasks, chunksize=1):
            done += 1
            if status == "ok":
                ok += 1
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

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed/60:.1f} min  —  ok={ok}  skipped={skipped}  errors={errors}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()

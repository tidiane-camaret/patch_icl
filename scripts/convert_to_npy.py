"""
Convert TotalSegmentator subjects from .nii.gz to .npy for fast data loading.

Per subject written (always):
  ct.npy     — float16, HU-clipped & normalised to [0,1], native resolution (D,H,W)
  label.npy  — uint8, merged label volume using ALL_CLASSES ordering (0=bg), native resolution

With --size D H W (e.g. --size 64 64 64), also writes:
  ct_DxHxW.npy     — float16, trilinear-resized to target (D,H,W)
  label_DxHxW.npy  — uint8, nearest-neighbour-resized to target (D,H,W)

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
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.totalseg_dataset import ALL_CLASSES, HU_MIN, HU_MAX

ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir() -> str:
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["paths"]["totalseg"]

_CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(ALL_CLASSES)}  # 1-indexed


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
            vol = nib.load(str(ct_path)).get_fdata(dtype=np.float32)
            vol = np.clip(vol, HU_MIN, HU_MAX)
            vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)  # float32 for resize reuse

            seg_dir = subj_dir / "segmentations"
            label = np.zeros(vol.shape, dtype=np.uint8)
            for cls, idx in _CLASS_TO_IDX.items():
                mask_path = seg_dir / f"{cls}.nii.gz"
                if not mask_path.exists():
                    continue
                mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32) > 0
                label[mask] = idx

            np.save(ct_out, vol.astype(np.float16))
            np.save(label_out, label)

        if need_sized:
            if vol is None:
                vol   = np.load(ct_out,    mmap_mode="r").astype(np.float32)
                label = np.load(label_out, mmap_mode="r")

            zoom = tuple(t / s for t, s in zip(size, vol.shape))
            np.save(ct_sized,    ndi.zoom(vol,   zoom, order=1).astype(np.float16))
            np.save(label_sized, ndi.zoom(label, zoom, order=0))

    except Exception:
        return subj, traceback.format_exc()

    return subj, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        help="dataset root; defaults to paths.totalseg in configs/config.yaml")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()),
                        help="parallel worker processes (default: min(32, cpu_count))")
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

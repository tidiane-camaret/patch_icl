"""
Convert TotalSegmentator subjects from .nii.gz to .npy for fast data loading.

Per subject written:
  ct.npy     — float16, HU-clipped & normalised to [0,1], shape (D,H,W)
  label.npy  — uint8, merged label volume using ALL_CLASSES ordering (0=bg), shape (D,H,W)

float16 halves ct disk use vs float32 with negligible precision loss after normalisation.

Usage
-----
  python scripts/convert_to_npy.py [--data DIR] [--workers N] [--overwrite]
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.totalseg_dataset import ALL_CLASSES, HU_MIN, HU_MAX

_CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(ALL_CLASSES)}  # 1-indexed


def convert_subject(args: tuple) -> tuple[str, str]:
    """Convert one subject.  Returns (subject_id, 'ok' | error_message)."""
    subj_dir, overwrite = args
    subj_dir = Path(subj_dir)
    subj = subj_dir.name

    ct_out    = subj_dir / "ct.npy"
    label_out = subj_dir / "label.npy"

    if not overwrite and ct_out.exists() and label_out.exists():
        return subj, "skip"

    try:
        # ---- CT ----
        ct_path = subj_dir / "ct.nii.gz"
        vol = nib.load(str(ct_path)).get_fdata(dtype=np.float32)
        vol = np.clip(vol, HU_MIN, HU_MAX)
        vol = ((vol - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float16)
        np.save(ct_out, vol)

        # ---- Label: merge all available classes ----
        seg_dir = subj_dir / "segmentations"
        label = np.zeros(vol.shape, dtype=np.uint8)
        for cls, idx in _CLASS_TO_IDX.items():
            mask_path = seg_dir / f"{cls}.nii.gz"
            if not mask_path.exists():
                continue
            mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32) > 0
            label[mask] = idx
        np.save(label_out, label)

    except Exception:
        return subj, traceback.format_exc()

    return subj, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/work/dlclarge2/ndirt-SegFM3D/data/totalseg")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()),
                        help="parallel worker processes (default: min(32, cpu_count))")
    parser.add_argument("--overwrite", action="store_true",
                        help="reconvert even if .npy files already exist")
    args = parser.parse_args()

    data_dir = Path(args.data)
    subjects = sorted(p for p in data_dir.iterdir() if p.is_dir())
    total = len(subjects)
    print(f"Found {total} subjects  |  workers={args.workers}  |  overwrite={args.overwrite}")

    tasks = [(str(s), args.overwrite) for s in subjects]

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

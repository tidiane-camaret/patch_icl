"""
Build the organ-centroid bbox cache used by TotalSegInContextDataset(use_crop=True).

For each subject, loads label.npy and computes the integer centroid (d, h, w) of
every class present, in a single O(D×H×W) pass using np.bincount — independent of
the number of classes.  Results are saved as:

  {data_root}/.bbox_cache_{sha256[:12]}.pkl
  → dict[subject_id, dict[class_name, tuple[int,int,int]]]

The key (sha256 of sorted subject list) matches the one produced by the dataloader,
so this cache is picked up automatically on the next run.

Usage
-----
  python scripts/build_bbox_cache.py [--data DIR] [--workers N] [--overwrite]
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.totalseg_classes import ALL_CLASSES

ROOT = Path(__file__).resolve().parents[1]

_N_LABELS = len(ALL_CLASSES) + 1          # 0 = background, 1..117 = organs
_IDX_TO_CLASS: dict[int, str] = {i + 1: cls for i, cls in enumerate(ALL_CLASSES)}


def _default_data_dir() -> str:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config")
    return cfg.paths.totalseg


def compute_subject(args: tuple) -> tuple[str, dict | str]:
    """
    Return (subject_id, {class_name: (d, h, w)} | error_string).

    Single bincount pass: O(D×H×W) regardless of class count.
    """
    subj_dir = Path(args)
    subj = subj_dir.name
    label_npy = subj_dir / "label.npy"
    if not label_npy.exists():
        return subj, f"label.npy not found in {subj_dir}"

    try:
        arr = np.load(label_npy, mmap_mode="r")          # (D, H, W) uint8
        flat = arr.ravel().astype(np.int32)

        counts = np.bincount(flat, minlength=_N_LABELS)

        d_g, h_g, w_g = np.indices(arr.shape)
        sum_d = np.bincount(flat, weights=d_g.ravel().astype(np.float64), minlength=_N_LABELS)
        sum_h = np.bincount(flat, weights=h_g.ravel().astype(np.float64), minlength=_N_LABELS)
        sum_w = np.bincount(flat, weights=w_g.ravel().astype(np.float64), minlength=_N_LABELS)

        bboxes: dict[str, tuple[int, int, int]] = {}
        for idx in range(1, _N_LABELS):
            if counts[idx] == 0 or idx not in _IDX_TO_CLASS:
                continue
            bboxes[_IDX_TO_CLASS[idx]] = (
                int(sum_d[idx] / counts[idx]),
                int(sum_h[idx] / counts[idx]),
                int(sum_w[idx] / counts[idx]),
            )
    except Exception:
        return subj, traceback.format_exc()

    return subj, bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        help="dataset root; defaults to paths.totalseg in configs/config.yaml")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()),
                        help="parallel worker processes (default: min(32, cpu_count))")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild even if the cache file already exists")
    args = parser.parse_args()

    data_dir = Path(args.data) if args.data else Path(_default_data_dir())
    subjects = sorted(p for p in data_dir.iterdir() if p.is_dir())
    subject_names = [p.name for p in subjects]
    total = len(subjects)

    key = hashlib.sha256("|".join(subject_names).encode()).hexdigest()[:12]
    cache_path = data_dir / f".bbox_cache_{key}.pkl"

    print(f"Found {total} subjects  |  workers={args.workers}  |  cache={cache_path.name}")

    if cache_path.exists() and not args.overwrite:
        print("Cache already exists. Use --overwrite to rebuild.")
        return

    tasks = [str(s) for s in subjects]
    cache: dict[str, dict] = {}
    errors = done = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for subj, result in pool.imap_unordered(compute_subject, tasks, chunksize=1):
            done += 1
            if isinstance(result, str):
                errors += 1
                print(f"\n[ERROR] {subj}:\n{result}")
            else:
                cache[subj] = result

            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"\r  {done}/{total}  ok={len(cache)}  err={errors}"
                f"  {rate:.1f} subj/s  ETA {eta/60:.0f}m",
                end="", flush=True,
            )

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed/60:.1f} min  —  {len(cache)} subjects saved  |  {errors} errors")
    print(f"Cache: {cache_path}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()

"""Augment an existing gmm_bank's index.pkl with precomputed per-class foreground-voxel
samples, so SynthGmmProvider's data.cascade_center_mode="random_fg" can draw a center in
O(1) instead of scanning the (mmap'd, uncached) native mask live -- see docs/logs.md /
memory project_synth_gmm_paint_perf: that live scan (np.argwhere on a ~512^3 file) was
measured at 1.5-3.2s per context-recrop call on GCP, ~20% of a whole training epoch.

Reads the bank's EXISTING masks/*.npy (already resampled by build_gmm_mask_bank.py) and
index.pkl -- does NOT touch the masks or need the original MAISI zip. For each entry, does
ONE np.argsort over the flat mask (not one np.argwhere per class -- bank masks average
~20-30 present classes each, so this turns O(num_classes * native_voxels) into
O(native_voxels) per mask), buckets by label value, and stores up to `--n_samples` randomly
chosen voxel coords per class as entry["fg_samples"][cls] (int16 array, shape (k,3)).

  .venv_thor/bin/python experiments/3d/synth_task_generation/add_fg_samples_to_bank.py \
    --bank /nfs/.../data/gmm_bank --n_samples 64 --workers 16
"""
import argparse
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def fg_samples_for_mask(arr, n_samples, seed):
    """One np.argsort pass over the flat mask -> {label: (k,3) int16 coords}, k<=n_samples,
    for every present nonzero label. Sampling is uniform-without-replacement within each
    label's own voxel set (not an approximation across labels)."""
    flat = arr.ravel()
    order = np.argsort(flat)
    sorted_vals = flat[order]
    change = np.flatnonzero(np.diff(sorted_vals)) + 1
    bounds = np.concatenate(([0], change, [sorted_vals.shape[0]]))
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        lbl = int(sorted_vals[lo])
        if lbl == 0:
            continue
        idxs = order[lo:hi]
        k = min(n_samples, idxs.shape[0])
        pick = idxs if k == idxs.shape[0] else rng.choice(idxs, size=k, replace=False)
        out[lbl] = np.stack(np.unravel_index(pick, arr.shape), axis=1).astype(np.int16)
    return out


def _process_one(args):
    masks_dir, fname, n_samples, seed = args
    arr = np.squeeze(np.load(masks_dir / fname))
    return fname, fg_samples_for_mask(arr, n_samples, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True, help="bank dir (has index.pkl + masks/)")
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    idx_path = a.bank / "index.pkl"
    with open(idx_path, "rb") as f:
        index = pickle.load(f)
    entries = index["entries"]
    masks_dir = a.bank / "masks"

    jobs = [(masks_dir, e["file"], a.n_samples, a.seed + i) for i, e in enumerate(entries)]
    by_file = {}
    print(f"computing fg_samples (n={a.n_samples}) for {len(jobs)} masks, "
          f"{a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        done = 0
        for fut in as_completed([ex.submit(_process_one, jb) for jb in jobs]):
            fname, samples = fut.result()
            by_file[fname] = samples
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    for e in entries:
        e["fg_samples"] = by_file[e["file"]]

    backup = idx_path.with_suffix(".pkl.bak")
    shutil.copy2(idx_path, backup)
    with open(idx_path, "wb") as f:
        pickle.dump(index, f)
    print(f"done: {len(entries)} entries augmented. backup at {backup}, "
          f"index.pkl updated in place.", flush=True)


if __name__ == "__main__":
    main()

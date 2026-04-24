"""
Generate synthetic supervoxel labels for N subjects using 4 algorithms.

Outputs (per subject) in experiments/synth_labels/<subj>/:
  synth_grid.npy       — GridVoronoi (pure numpy, ~200 ms)
  synth_watershed.npy  — Watershed3D (gradient-aware, ~10-40 s)
  synth_slic.npy       — SLIC3D (compact, ~30-60 s)
  synth_seeds.npy      — 3D-SEEDS (requires python_3d_seeds)

Timing log: experiments/synth_labels/log.jsonl (one JSON line per run)

n_segments is drawn from U[50, 500] per subject (MultiverSeg protocol).

Usage
-----
  uv run scripts/generate_synth_labels.py
  uv run scripts/generate_synth_labels.py --n-subjects 10 --methods grid watershed slic
  uv run scripts/generate_synth_labels.py --data /path/to/totalseg --n-segments 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from skimage.segmentation import slic
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import python_3d_seeds
    HAS_SEEDS3D = True
except ImportError:
    HAS_SEEDS3D = False

DATA_ROOT = "/work/dlclarge2/ndirt-SegFM3D/data/totalseg"
OUT_ROOT  = Path(__file__).resolve().parents[1] / "experiments" / "synth_labels"


# ---------------------------------------------------------------------------
# Supervoxel algorithms (same implementations as benchmark script)
# ---------------------------------------------------------------------------

def run_grid(vol: np.ndarray, n_segments: int) -> np.ndarray:
    D, H, W = vol.shape
    stride = max(1, int(round((D * H * W / n_segments) ** (1 / 3))))
    nW = max(1, W // stride)
    nH = max(1, H // stride)
    z_idx = (np.arange(D, dtype=np.int32) // stride).reshape(D, 1, 1)
    y_idx = (np.arange(H, dtype=np.int32) // stride).reshape(1, H, 1)
    x_idx = (np.arange(W, dtype=np.int32) // stride).reshape(1, 1, W)
    return (z_idx * nH * nW + y_idx * nW + x_idx + 1).astype(np.int32)


def run_watershed(vol: np.ndarray, n_segments: int) -> np.ndarray:
    D, H, W = vol.shape
    grad = (ndi.sobel(vol, axis=0) ** 2
            + ndi.sobel(vol, axis=1) ** 2
            + ndi.sobel(vol, axis=2) ** 2) ** 0.5
    grad_u8 = (grad / (grad.max() + 1e-8) * 255).astype(np.uint8)
    stride = max(2, int(round((D * H * W / n_segments) ** (1 / 3))))
    half = stride // 2
    zs = np.arange(half, D, stride)
    ys = np.arange(half, H, stride)
    xs = np.arange(half, W, stride)
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    seeds = np.zeros(vol.shape, dtype=np.int32)
    seeds[gz.ravel(), gy.ravel(), gx.ravel()] = np.arange(1, gz.size + 1, dtype=np.int32)
    return ndi.watershed_ift(grad_u8, seeds).astype(np.int32)


def run_slic(vol: np.ndarray, n_segments: int) -> np.ndarray:
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image not installed")
    return slic(
        vol, n_segments=n_segments, compactness=0.05,
        max_num_iter=5, channel_axis=None,
        start_label=1, enforce_connectivity=True, convert2lab=False,
    ).astype(np.int32)


def run_seeds3d(vol: np.ndarray, n_segments: int) -> np.ndarray:
    if not HAS_SEEDS3D:
        raise RuntimeError("python_3d_seeds not installed")
    D, H, W = vol.shape
    data = np.ascontiguousarray(vol, dtype=np.float32)
    sv = python_3d_seeds.createSupervoxelSEEDS(
        width=W, height=H, depth=D, channels=1,
        num_superpixels=n_segments, num_levels=4,
        prior=2, histogram_bins=15, double_step=False,
    )
    sv.iterate(data=data, num_iterations=12)
    return (sv.getLabels() + 1).astype(np.int32)  # shift to 1-based


METHODS = {
    "grid":      ("synth_grid.npy",       run_grid),
    "watershed": ("synth_watershed.npy",  run_watershed),
    "slic":      ("synth_slic.npy",       run_slic),
    "seeds3d":   ("synth_seeds.npy",      run_seeds3d),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_ct(subj_dir: Path) -> np.ndarray:
    npy = subj_dir / "ct.npy"
    if npy.exists():
        return np.load(npy, mmap_mode="r").astype(np.float32)
    from src.totalseg_dataset import HU_MIN, HU_MAX
    import nibabel as nib
    vol = nib.load(str(subj_dir / "ct.nii.gz")).get_fdata(dtype=np.float32)
    vol = np.clip(vol, HU_MIN, HU_MAX)
    return (vol - HU_MIN) / (HU_MAX - HU_MIN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_ROOT)
    parser.add_argument("--out", default=str(OUT_ROOT))
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument(
        "--methods", nargs="+",
        default=["grid", "watershed", "slic"] + (["seeds3d"] if HAS_SEEDS3D else []),
        choices=list(METHODS.keys()),
    )
    parser.add_argument("--n-segments", type=int, default=None,
                        help="Fixed n_segments; default=random U[50,500] per subject")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    subjects = subjects[:args.n_subjects]

    rng = np.random.default_rng(0)
    log_path = out_dir / "log.jsonl"
    log_fh = open(log_path, "a", buffering=1)

    if not HAS_SEEDS3D and "seeds3d" in args.methods:
        print("WARNING: python_3d_seeds not installed — skipping seeds3d")
        args.methods = [m for m in args.methods if m != "seeds3d"]

    print(f"Subjects  : {subjects}")
    print(f"Methods   : {args.methods}")
    print(f"Output    : {out_dir}")
    print()

    for subj in subjects:
        subj_dir    = data_dir / subj
        subj_out    = out_dir / subj
        subj_out.mkdir(exist_ok=True)

        n_seg = args.n_segments if args.n_segments else int(rng.integers(50, 501))

        print(f"[{subj}]  n_segments={n_seg}  shape=", end="", flush=True)
        vol = load_ct(subj_dir)
        print(vol.shape, flush=True)

        for key in args.methods:
            fname, fn = METHODS[key]
            out_path = subj_out / fname

            if out_path.exists() and not args.overwrite:
                size = np.load(out_path, mmap_mode="r").max()
                print(f"  {key:<12} skip (exists, n_actual={size})")
                continue

            t0 = time.perf_counter()
            try:
                labels = fn(vol, n_seg)
                elapsed = time.perf_counter() - t0
                n_actual = int(labels.max())
                np.save(out_path, labels)
                status = "ok"
                err = None
            except Exception as e:
                elapsed = time.perf_counter() - t0
                n_actual = 0
                status = "error"
                err = str(e)

            record = dict(
                subject=subj, method=key, n_req=n_seg, n_actual=n_actual,
                elapsed_s=round(elapsed, 3), status=status,
                shape=list(vol.shape), error=err,
            )
            log_fh.write(json.dumps(record) + "\n")

            marker = "✓" if status == "ok" else "✗"
            print(f"  {key:<12} {marker}  n_actual={n_actual:>5}  {elapsed:.1f}s"
                  + (f"  ERROR: {err}" if err else ""))

        print()

    log_fh.close()
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()

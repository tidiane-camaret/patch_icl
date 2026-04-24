"""
Generate synthetic supervoxel labels for all subjects in paths.totalseg.

Reads paths.totalseg from configs/config.yaml, loads each subject's ct.npy,
runs the chosen supervoxel algorithm, and writes label_synth_<method>.npy
alongside ct.npy and label.npy in the same subject directory.

n_segments is drawn from U[50, 500] per subject (MultiverSeg protocol) unless
--n-segments is given.

Usage
-----
  uv run scripts/synth_labels/generate.py --method grid
  uv run scripts/synth_labels/generate.py --method watershed --workers 4
  uv run scripts/synth_labels/generate.py --method slic --n-segments 200
  uv run scripts/synth_labels/generate.py --method seeds3d --overwrite
"""

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_totalseg_path() -> Path:
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["paths"]["totalseg"])


# ---------------------------------------------------------------------------
# Supervoxel algorithms
# ---------------------------------------------------------------------------

def _grid(vol: np.ndarray, n_segments: int) -> np.ndarray:
    D, H, W = vol.shape
    stride = max(1, int(round((D * H * W / n_segments) ** (1 / 3))))
    nW = max(1, W // stride)
    nH = max(1, H // stride)
    z = (np.arange(D, dtype=np.int32) // stride).reshape(D, 1, 1)
    y = (np.arange(H, dtype=np.int32) // stride).reshape(1, H, 1)
    x = (np.arange(W, dtype=np.int32) // stride).reshape(1, 1, W)
    return (z * nH * nW + y * nW + x + 1).astype(np.int32)


def _watershed(vol: np.ndarray, n_segments: int) -> np.ndarray:
    import scipy.ndimage as ndi
    D, H, W = vol.shape
    grad = (ndi.sobel(vol, axis=0) ** 2
            + ndi.sobel(vol, axis=1) ** 2
            + ndi.sobel(vol, axis=2) ** 2) ** 0.5
    grad_u8 = (grad / (grad.max() + 1e-8) * 255).astype(np.uint8)
    stride = max(2, int(round((D * H * W / n_segments) ** (1 / 3))))
    half = stride // 2
    gz, gy, gx = np.meshgrid(
        np.arange(half, D, stride),
        np.arange(half, H, stride),
        np.arange(half, W, stride),
        indexing="ij",
    )
    seeds = np.zeros(vol.shape, dtype=np.int32)
    seeds[gz.ravel(), gy.ravel(), gx.ravel()] = np.arange(1, gz.size + 1, dtype=np.int32)
    return ndi.watershed_ift(grad_u8, seeds).astype(np.int32)


def _slic(vol: np.ndarray, n_segments: int) -> np.ndarray:
    from skimage.segmentation import slic
    return slic(
        vol, n_segments=n_segments, compactness=0.05, max_num_iter=5,
        channel_axis=None, start_label=1,
        enforce_connectivity=True, convert2lab=False,
    ).astype(np.int32)


def _seeds3d(vol: np.ndarray, n_segments: int) -> np.ndarray:
    try:
        import python_3d_seeds
    except ImportError:
        raise ImportError(
            "python_3d_seeds is not installed.\n"
            "Run:  bash scripts/synth_labels/install_seeds3d.sh"
        )
    D, H, W = vol.shape
    data = np.ascontiguousarray(vol, dtype=np.float32)
    sv = python_3d_seeds.createSupervoxelSEEDS(
        width=W, height=H, depth=D, channels=1,
        num_superpixels=n_segments, num_levels=4,
        prior=2, histogram_bins=15, double_step=False,
    )
    sv.iterate(data=data, num_iterations=12)
    return (sv.getLabels() + 1).astype(np.int32)


ALGORITHMS = {
    "grid":      _grid,
    "watershed": _watershed,
    "slic":      _slic,
    "seeds3d":   _seeds3d,
}


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------

def _process(args: tuple) -> dict:
    subj_dir, method, n_segments, overwrite = args
    subj_dir = Path(subj_dir)
    out_path = subj_dir / f"label_synth_{method}.npy"

    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        return dict(subject=subj_dir.name, status="skip", n_actual=int(arr.max()),
                    elapsed_s=0.0, n_req=n_segments)

    ct_npy = subj_dir / "ct.npy"
    ct_nii = subj_dir / "ct.nii.gz"
    if not ct_npy.exists() and not ct_nii.exists():
        return dict(subject=subj_dir.name, status="error",
                    error="neither ct.npy nor ct.nii.gz found", elapsed_s=0.0, n_req=n_segments)

    try:
        if ct_npy.exists():
            vol = np.load(ct_npy, mmap_mode="r").astype(np.float32)
        else:
            import nibabel as nib
            raw = nib.load(str(ct_nii)).get_fdata(dtype=np.float32)
            vol = (np.clip(raw, -150, 250) + 150) / 400.0
        fn  = ALGORITHMS[method]
        t0  = time.perf_counter()
        labels = fn(vol, n_segments)
        elapsed = time.perf_counter() - t0
        np.save(out_path, labels)
        return dict(subject=subj_dir.name, status="ok", n_req=n_segments,
                    n_actual=int(labels.max()), elapsed_s=round(elapsed, 3),
                    shape=list(vol.shape))
    except Exception:
        return dict(subject=subj_dir.name, status="error",
                    error=traceback.format_exc(), elapsed_s=0.0, n_req=n_segments)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate label_synth_<method>.npy for every subject in paths.totalseg"
    )
    parser.add_argument(
        "--method", required=True, choices=list(ALGORITHMS),
        help="Supervoxel algorithm to use",
    )
    parser.add_argument(
        "--data", default=None,
        help="Override paths.totalseg from config.yaml",
    )
    parser.add_argument(
        "--n-segments", type=int, default=None,
        help="Fixed supervoxel count; default=random U[50,500] per subject",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help=f"Parallel worker processes (default: cpu_count={mp.cpu_count()})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even if label_synth_<method>.npy already exists",
    )
    args = parser.parse_args()

    data_dir = Path(args.data) if args.data else load_totalseg_path()
    subjects = sorted(p for p in data_dir.iterdir() if p.is_dir())

    if not subjects:
        sys.exit(f"No subject directories found in {data_dir}")

    rng = np.random.default_rng(0)
    n_segs = (
        [args.n_segments] * len(subjects)
        if args.n_segments
        else rng.integers(50, 501, size=len(subjects)).tolist()
    )

    tasks = [
        (str(subj), args.method, int(n), args.overwrite)
        for subj, n in zip(subjects, n_segs)
    ]

    print(f"Method   : {args.method}")
    print(f"Data     : {data_dir}")
    print(f"Subjects : {len(subjects)}")
    print(f"Workers  : {args.workers}")
    print(f"Output   : label_synth_{args.method}.npy  (saved next to ct.npy)")
    print()

    total = len(tasks)
    ok = skipped = errors = 0
    t_start = time.time()

    def _report(r: dict):
        nonlocal ok, skipped, errors
        subj = r["subject"]
        status = r["status"]
        if status == "ok":
            ok += 1
            print(f"  ✓  {subj:<12}  n_req={r['n_req']:>4}  n_actual={r['n_actual']:>5}"
                  f"  shape={r.get('shape', '?')}  {r['elapsed_s']:.1f}s")
        elif status == "skip":
            skipped += 1
            print(f"  -  {subj:<12}  skip (exists, n_actual={r['n_actual']})")
        else:
            errors += 1
            # Print full traceback so the root cause is visible
            err = r.get("error", "unknown error")
            print(f"  ✗  {subj:<12}  ERROR:\n{err}")

    if args.workers <= 1:
        for task in tasks:
            _report(_process(task))
    else:
        # fork is the Linux default: no re-import overhead, workers inherit the
        # already-loaded interpreter state.  Avoids the venv-mismatch issue that
        # breaks forkserver/spawn when VIRTUAL_ENV points to a different venv.
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_process, tasks, chunksize=1):
                _report(r)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s  —  ok={ok}  skipped={skipped}  errors={errors}")


if __name__ == "__main__":
    main()

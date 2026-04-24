"""
Supervoxel generation benchmark for 3-D CT volumes.

Compares three methods across the count range required by MultiverSeg (λ ~ U[50, 500]):

  GridVoronoi   : regular grid cells — pure numpy, ~170 ms/vol, no quality guarantee
  Watershed3D   : gradient + regular-grid seeds + watershed_ift — ~12 s/vol, boundary-aware
  SLIC3D        : iterative k-means style (scikit-image) — ~40 s/vol, compact supervoxels
  3D-SEEDS      : C++ method from the paper — ~10-50 ms/vol (separate install needed)

Usage
-----
    uv run scripts/benchmark_supervoxels.py                          # 128×256×256 synthetic
    uv run scripts/benchmark_supervoxels.py --size 64 128 128        # quick test
    uv run scripts/benchmark_supervoxels.py --real-volume /data/totalseg/s0000/ct.npy
    uv run scripts/benchmark_supervoxels.py --methods grid watershed  # subset of methods

Install 3D-SEEDS (optional):
    uv add git+https://github.com/zch0414/3d-seeds
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

try:
    from skimage.segmentation import slic
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    # Installed via: see script docstring for build instructions
    import python_3d_seeds  # type: ignore
    HAS_SEEDS3D = True
except ImportError:
    HAS_SEEDS3D = False


# ---------------------------------------------------------------------------
# Synthetic CT generator
# ---------------------------------------------------------------------------

def synthetic_ct(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Float32 volume in [0,1]: smooth noise + organ-like ellipsoids."""
    D, H, W = shape
    vol = rng.standard_normal(shape).astype(np.float32) * 0.05 + 0.3
    zz, yy, xx = np.ogrid[:D, :H, :W]
    for _ in range(6):
        cz = rng.integers(D // 4, 3 * D // 4)
        cy = rng.integers(H // 4, 3 * H // 4)
        cx = rng.integers(W // 4, 3 * W // 4)
        rz = rng.integers(D // 12, D // 6)
        ry = rng.integers(H // 10, H // 5)
        rx = rng.integers(W // 10, W // 5)
        inside = (((zz - cz) / rz) ** 2
                  + ((yy - cy) / ry) ** 2
                  + ((xx - cx) / rx) ** 2) < 1.0
        vol[inside] = rng.uniform(0.2, 0.9) + rng.standard_normal(inside.sum()) * 0.02
    return np.clip(vol, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def run_grid(vol: np.ndarray, n_segments: int) -> tuple[np.ndarray, float]:
    """
    Regular-grid Voronoi: divide the volume into a ~cubic grid of cells.
    Each voxel gets the label of its cell — O(N), no iteration needed.
    Fast but not boundary-aware; suitable for synthetic diversity.
    """
    t0 = time.perf_counter()
    D, H, W = vol.shape
    stride = max(1, int(round((D * H * W / n_segments) ** (1 / 3))))
    nW = max(1, W // stride)
    nH = max(1, H // stride)
    # 1-D broadcast avoids large intermediate arrays
    z_idx = (np.arange(D, dtype=np.int32) // stride).reshape(D, 1, 1)
    y_idx = (np.arange(H, dtype=np.int32) // stride).reshape(1, H, 1)
    x_idx = (np.arange(W, dtype=np.int32) // stride).reshape(1, 1, W)
    labels = (z_idx * nH * nW + y_idx * nW + x_idx + 1)
    elapsed = time.perf_counter() - t0
    return labels, elapsed


def run_watershed(vol: np.ndarray, n_segments: int) -> tuple[np.ndarray, float]:
    """
    Compact-watershed supervoxels: gradient magnitude → regular-grid seeds → watershed_ift.
    Boundary-aware; watershed_ift is the bottleneck (~10 s/vol at 128×256×256).
    """
    t0 = time.perf_counter()
    D, H, W = vol.shape

    # Gradient magnitude (3 Sobel passes)
    grad = (ndi.sobel(vol, axis=0) ** 2
            + ndi.sobel(vol, axis=1) ** 2
            + ndi.sobel(vol, axis=2) ** 2) ** 0.5
    grad_u8 = (grad / (grad.max() + 1e-8) * 255).astype(np.uint8)

    # Regular-grid seeds (vectorised)
    stride = max(2, int(round((D * H * W / n_segments) ** (1 / 3))))
    half = stride // 2
    zs = np.arange(half, D, stride)
    ys = np.arange(half, H, stride)
    xs = np.arange(half, W, stride)
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    seeds = np.zeros(vol.shape, dtype=np.int32)
    seeds[gz.ravel(), gy.ravel(), gx.ravel()] = np.arange(1, gz.size + 1, dtype=np.int32)

    labels = ndi.watershed_ift(grad_u8, seeds)
    elapsed = time.perf_counter() - t0
    return labels, elapsed


def run_slic(vol: np.ndarray, n_segments: int) -> tuple[np.ndarray, float]:
    """SLIC3D — scikit-image. Compact supervoxels; slowest option."""
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image not installed: uv add scikit-image")
    t0 = time.perf_counter()
    labels = slic(
        vol,
        n_segments=n_segments,
        compactness=0.05,       # low → intensity-driven (better for CT)
        max_num_iter=5,         # default 10; 5 cuts time ~2× with little quality loss
        channel_axis=None,      # grayscale 3-D volume
        start_label=1,
        enforce_connectivity=True,
        convert2lab=False,
    )
    elapsed = time.perf_counter() - t0
    return labels, elapsed


def run_seeds3d(vol: np.ndarray, n_segments: int) -> tuple[np.ndarray, float]:
    """
    3D-SEEDS via python_3d_seeds (https://github.com/zch0414/3d-seeds).
    The library expects (D, H, W) float32 in [0, 1] and contiguous memory.
    """
    t0 = time.perf_counter()
    D, H, W = vol.shape
    data = np.ascontiguousarray(vol, dtype=np.float32)
    seeds = python_3d_seeds.createSupervoxelSEEDS(
        width=W, height=H, depth=D, channels=1,
        num_superpixels=n_segments,
        num_levels=4,
        prior=2,
        histogram_bins=15,
        double_step=False,
    )
    seeds.iterate(data=data, num_iterations=12)
    labels = seeds.getLabels()          # (D, H, W) int32, labels start at 0
    elapsed = time.perf_counter() - t0
    return labels + 1, elapsed          # shift to 1-based to match other methods


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_METHODS: dict[str, tuple] = {
    "grid":      ("GridVoronoi", run_grid),
    "watershed": ("Watershed3D", run_watershed),
    "slic":      ("SLIC3D",      run_slic),
}
if HAS_SEEDS3D:
    ALL_METHODS["seeds3d"] = ("3D-SEEDS", run_seeds3d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-volume", type=Path, default=None,
                        help=".npy or .nii.gz CT volume")
    parser.add_argument("--size", type=int, nargs=3, default=[128, 256, 256],
                        metavar=("D", "H", "W"))
    parser.add_argument("--n-segments", type=int, nargs="+", default=[50, 100, 200, 500])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS.keys()),
                        choices=list(ALL_METHODS.keys()),
                        help="Methods to benchmark (default: all available)")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # Load or generate volume
    if args.real_volume is not None:
        p = args.real_volume
        print(f"Loading volume from {p} …")
        if p.suffix == ".npy":
            vol = np.load(p).astype(np.float32)
        else:
            import nibabel as nib  # type: ignore
            raw = nib.load(str(p)).get_fdata(dtype=np.float32)
            vol = (np.clip(raw, -150, 250) + 150) / 400.0
        if vol.ndim == 4:
            vol = vol[..., 0]
    else:
        shape = tuple(args.size)  # type: ignore[arg-type]
        print(f"Generating synthetic CT volume {shape} …")
        vol = synthetic_ct(shape, rng)

    print(f"Volume: {vol.shape}  dtype={vol.dtype}  "
          f"range=[{vol.min():.3f}, {vol.max():.3f}]  "
          f"mem={vol.nbytes/1e6:.1f} MB\n")

    methods = [(k, ALL_METHODS[k]) for k in args.methods if k in ALL_METHODS]
    if not HAS_SEEDS3D and "seeds3d" not in args.methods:
        print("Note: python_3d_seeds not installed. "
              "See script docstring for build instructions.\n")

    # Benchmark
    print(f"{'Method':<14}  {'n_req':>6}  {'n_actual':>8}  {'best_ms':>9}  {'mean_ms':>9}")
    print("-" * 55)

    results: list[dict] = []
    for n_seg in args.n_segments:
        for key, (display, fn) in methods:
            times = []
            n_actual = None
            for _ in range(args.repeats):
                labels, elapsed = fn(vol, n_seg)
                times.append(elapsed)
                if n_actual is None:
                    n_actual = int(labels.max())
            best = min(times)
            mean = sum(times) / len(times)
            print(f"  {display:<12}  {n_seg:>6}  {n_actual:>8}  "
                  f"{best*1e3:>8.1f}  {mean*1e3:>8.1f}")
            results.append(dict(method=display, n_req=n_seg, n_actual=n_actual,
                                best_ms=best * 1e3, mean_ms=mean * 1e3))
        print()

    # Per-method summary
    print("=== Method averages across all n_segments ===")
    for _, (display, _) in methods:
        rows = [r for r in results if r["method"] == display]
        avg = sum(r["mean_ms"] for r in rows) / len(rows)
        print(f"  {display:<14}  avg = {avg:>8.1f} ms/vol")

    # Offline preprocessing estimate
    print("\n=== Offline preprocessing estimate ===")
    for _, (display, _) in methods:
        rows = [r for r in results if r["method"] == display]
        avg_ms = sum(r["mean_ms"] for r in rows) / len(rows)
        print(f"  {display}:")
        for n_subj in [100, 1000, 4000]:
            mins = n_subj * avg_ms / 1000 / 60
            print(f"    {n_subj:>5} subjects → {mins:>7.1f} min")


if __name__ == "__main__":
    main()

"""
Plot CT slices overlaid with synthetic supervoxel labels for visual comparison.

Layout: one panel per subject.
  Rows    = axial slices (e.g. 25 %, 50 %, 75 % of depth)
  Columns = CT (grayscale) | GridVoronoi | Watershed3D | SLIC3D | 3D-SEEDS

Usage
-----
  uv run scripts/plot_synth_labels.py
  uv run scripts/plot_synth_labels.py --subjects s0000 s0001 s0002
  uv run scripts/plot_synth_labels.py --slices 0.25 0.5 0.75 --out results/synth_labels.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

DATA_ROOT = "/work/dlclarge2/ndirt-SegFM3D/data/totalseg"
SYNTH_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "synth_labels"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "synth_labels.png"

METHODS = [
    ("grid",      "synth_grid.npy",      "GridVoronoi"),
    ("watershed", "synth_watershed.npy", "Watershed3D"),
    ("slic",      "synth_slic.npy",      "SLIC3D"),
    ("seeds3d",   "synth_seeds.npy",     "3D-SEEDS"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ct(data_root: Path, subj: str) -> np.ndarray:
    npy = data_root / subj / "ct.npy"
    if npy.exists():
        return np.load(npy, mmap_mode="r").astype(np.float32)
    import nibabel as nib
    vol = nib.load(str(data_root / subj / "ct.nii.gz")).get_fdata(dtype=np.float32)
    vol = np.clip(vol, -150, 250)
    return (vol + 150) / 400.0


def label_to_rgb(labels: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Map integer label map to an RGBA image with a perceptually distinct colormap."""
    n = int(labels.max()) + 1
    # Use HSV colormap for many distinct colours
    cmap = plt.colormaps["hsv"].resampled(max(n, 2))
    rgba = cmap(labels.astype(np.float64) / max(n - 1, 1))   # (H, W, 4)
    rgba[labels == 0] = [0, 0, 0, 0]                          # background transparent
    rgba[..., 3] = np.where(labels > 0, alpha, 0.0)
    return rgba


def overlay(ct_slice: np.ndarray, label_slice: np.ndarray) -> np.ndarray:
    """Blend grayscale CT with coloured supervoxel RGBA overlay."""
    rgb = np.stack([ct_slice] * 3, axis=-1)                   # (H, W, 3)
    rgba = label_to_rgb(label_slice)
    a = rgba[..., 3:4]
    rgb = rgb * (1 - a) + rgba[..., :3] * a
    return np.clip(rgb, 0, 1)


def draw_boundaries(ct_slice: np.ndarray, label_slice: np.ndarray,
                    boundary_color=(1.0, 0.8, 0.0)) -> np.ndarray:
    """
    Draw supervoxel boundary lines on top of the CT slice.
    Boundaries = voxels where any of the 4-connected neighbours has a different label.
    """
    rgb = np.stack([ct_slice] * 3, axis=-1).copy()
    # Detect boundaries via neighbour differences
    edge = np.zeros(label_slice.shape, dtype=bool)
    edge[:-1, :] |= (label_slice[:-1, :] != label_slice[1:, :])
    edge[:, :-1] |= (label_slice[:, :-1] != label_slice[:, 1:])
    rgb[edge] = boundary_color
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=DATA_ROOT)
    parser.add_argument("--synth",    default=str(SYNTH_ROOT))
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject IDs; default=all found in synth dir")
    parser.add_argument("--n-subjects", type=int, default=4,
                        help="Max subjects to show when --subjects not given")
    parser.add_argument("--slices", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                        help="Fractional depth positions to plot (default: 0.25 0.5 0.75)")
    parser.add_argument("--mode", choices=["overlay", "boundary"], default="boundary",
                        help="Visualisation style: colour overlay or boundary lines")
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    data_root  = Path(args.data)
    synth_root = Path(args.synth)
    out_path   = Path(args.out)

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(p.name for p in synth_root.iterdir() if p.is_dir())
        subjects = subjects[:args.n_subjects]

    if not subjects:
        sys.exit(f"No subjects found in {synth_root}")

    # Which methods have saved labels for all subjects?
    available_methods = []
    for key, fname, display in METHODS:
        if all((synth_root / s / fname).exists() for s in subjects):
            available_methods.append((key, fname, display))

    if not available_methods:
        sys.exit(f"No synth label files found. Run generate_synth_labels.py first.")

    # Load timing from log — keep only the most recent entry per (subject, method)
    timing: dict[tuple[str, str], float] = {}
    log_path = synth_root / "log.jsonl"
    if log_path.exists():
        import json
        with open(log_path) as f:
            for line in f:
                r = json.loads(line)
                if r["status"] == "ok":
                    timing[(r["subject"], r["method"])] = r["elapsed_s"]

    print(f"Subjects  : {subjects}")
    print(f"Methods   : {[d for _, _, d in available_methods]}")
    print(f"Slice pos : {args.slices}")
    print(f"Mode      : {args.mode}")

    n_slices  = len(args.slices)
    n_methods = len(available_methods)
    n_cols    = 1 + n_methods          # CT + each method
    n_rows    = len(subjects) * n_slices

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    # Column headers on every subject's first row so they stay visible
    col_titles = ["CT (grayscale)"] + [d for _, _, d in available_methods]
    for s_idx in range(len(subjects)):
        first_row = s_idx * n_slices
        for col, title in enumerate(col_titles):
            axes[first_row, col].set_title(title, fontsize=10, fontweight="bold", pad=6)

    for s_idx, subj in enumerate(subjects):
        ct_vol = load_ct(data_root, subj)          # (D, H, W) float32

        # Load each method's labels
        label_vols: dict[str, np.ndarray] = {}
        for key, fname, _ in available_methods:
            p = synth_root / subj / fname
            label_vols[key] = np.load(p, mmap_mode="r")

        D = ct_vol.shape[0]

        for sl_idx, frac in enumerate(args.slices):
            row = s_idx * n_slices + sl_idx
            depth_idx = int(frac * (D - 1))

            ct_slice = ct_vol[depth_idx]           # (H, W)

            # Row label on first column
            axes[row, 0].set_ylabel(
                f"{subj}\nz={depth_idx}/{D-1}", fontsize=7,
                rotation=0, labelpad=65, va="center",
            )

            # CT column
            axes[row, 0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].axis("off")

            # Method columns
            for col_idx, (key, _, _) in enumerate(available_methods, start=1):
                lbl = label_vols[key][depth_idx]   # (H, W)

                if args.mode == "overlay":
                    img = overlay(ct_slice, lbl)
                else:
                    img = draw_boundaries(ct_slice, lbl)

                axes[row, col_idx].imshow(img)
                t = timing.get((subj, key))
                t_str = f" | {t:.1f}s" if t is not None else ""
                axes[row, col_idx].set_title(
                    f"n={int(lbl.max())}{t_str}", fontsize=7, pad=2
                )
                axes[row, col_idx].axis("off")

    fig.suptitle(
        f"Synthetic supervoxel labels — {args.mode} view\n"
        f"({len(subjects)} subjects × {n_slices} slices)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout(pad=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

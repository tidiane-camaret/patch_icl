"""
Plot image slices and GT overlays for a few DeepXcontrast cases.

For each selected case, shows:
  - CT axial/coronal/sagittal at best-label slice
  - T1 MRI axial/coronal/sagittal at same slice
  - GT label overlays (c1, c2, c3, nuc) on T1
  - CT label overlays (c1, c2, c3, nuc) on CT — CT labels are on a different
    grid (registered via ANTS), so the CT image is resampled to label space.

Usage
-----
  python scripts/datasets/deepxcontrast/plot_slices.py
  python scripts/datasets/deepxcontrast/plot_slices.py --n_cases 6 --out results/deepxcontrast_slices.png
  python scripts/datasets/deepxcontrast/plot_slices.py --cases 15104554 12345678
"""

import argparse
import random
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

DATASET_DIR = Path("/nfs/data/nii/data1/DeepXcontrast")

GT_LABELS = [
    ("c1.nii",               "#e74c3c", "GT gray matter"),
    ("c2.nii",               "#3498db", "GT white matter"),
    ("c3.nii",               "#2ecc71", "GT CSF"),
    ("nuc.nii",              "#f39c12", "GT nuclei"),
]

CT_LABELS = [
    ("out_CTtissue_c1.nii.gz", "#e74c3c", "CT gray matter"),
    ("out_CTtissue_c2.nii.gz", "#3498db", "CT white matter"),
    ("out_CTtissue_c3.nii.gz", "#2ecc71", "CT CSF"),
    ("out_CTnuclei.nii.gz",    "#f39c12", "CT nuclei"),
]

# grid: 4 columns — col 0 is row label, cols 1-3 are axial/coronal/sagittal
N_IMG_COLS = 3
N_COLS = 4
COL_HEADERS = ["", "Axial", "Coronal", "Sagittal"]


def load_nii(path: "Path"):
    nii = nib.as_closest_canonical(nib.load(str(path)))
    data = np.squeeze(nii.get_fdata(dtype=np.float32))
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(nii.affine)[:3])
    return data, spacing


def norm_ct(img: np.ndarray, wmin: float = -200, wmax: float = 600) -> np.ndarray:
    return np.clip((img - wmin) / (wmax - wmin), 0.0, 1.0)


def norm_ct_brain(img: np.ndarray) -> np.ndarray:
    """Narrow brain window (0–80 HU) for soft tissue contrast."""
    return np.clip((img - 0) / 80.0, 0.0, 1.0)


def norm_mri(img: np.ndarray) -> np.ndarray:
    fg = img[img > 0]
    if fg.size == 0:
        return np.zeros_like(img)
    lo, hi = float(np.percentile(fg, 0.5)), float(np.percentile(fg, 99.5))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def resample_to(img: np.ndarray, target_shape: "tuple[int, int, int]", order: int = 1) -> np.ndarray:
    """Zoom img to target_shape (for display alignment only)."""
    if img.shape == target_shape:
        return img
    factors = [t / s for t, s in zip(target_shape, img.shape)]
    return zoom(img, factors, order=order)


def best_slice_idx(masks: "list[np.ndarray]", thr: float = 0.5) -> "tuple[int, int, int]":
    """Return (iz, iy, ix) of the slice with max combined foreground per axis."""
    combined = np.zeros(masks[0].shape, dtype=np.float32)
    for m in masks:
        combined += (m > thr).astype(np.float32)
    iz = int(combined.sum(axis=(1, 2)).argmax())
    iy = int(combined.sum(axis=(0, 2)).argmax())
    ix = int(combined.sum(axis=(0, 1)).argmax())
    return iz, iy, ix


def label_ax(ax, text: str):
    """Draw a clean row label in a spine-free axes."""
    ax.axis("off")
    ax.text(0.95, 0.5, text, transform=ax.transAxes,
            fontsize=7, va="center", ha="right", wrap=True)


def show_slice(ax, sl: np.ndarray):
    ax.imshow(sl.T, cmap="gray", origin="lower", aspect="equal", interpolation="bilinear")
    ax.axis("off")


def overlay(ax, bg_norm: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.45):
    ax.imshow(bg_norm.T, cmap="gray", origin="lower", aspect="equal", interpolation="bilinear")
    if mask.max() > 0:
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        c = mcolors.to_rgba(color)
        rgba[mask > 0] = (*c[:3], alpha)
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower", aspect="equal", interpolation="nearest")
    ax.axis("off")


def plot_case(case_id: str, axes_grid, row_offset: int) -> int:
    """Fill rows in axes_grid starting at row_offset for one case. Returns rows used."""
    case_dir = DATASET_DIR / case_id / "0"

    ct_path = case_dir / "CT.nii"
    t1_path = case_dir / "T1.nii"
    if not ct_path.exists() or not t1_path.exists():
        print(f"  [{case_id}] missing CT.nii or T1.nii — skipping")
        return 0

    ct, ct_sp = load_nii(ct_path)
    t1, _     = load_nii(t1_path)

    # GT labels — same space as T1
    gt_dir = case_dir / "GT"
    gt_masks, gt_meta = [], []
    for fname, color, label in GT_LABELS:
        p = gt_dir / fname
        if p.exists():
            m, _ = load_nii(p)
            gt_masks.append(m)
            gt_meta.append((color, label))

    # CT labels — registered to CT space but potentially different grid
    ct_label_dir = case_dir / "CT"
    ct_masks, ct_label_meta = [], []
    for fname, color, label in CT_LABELS:
        p = ct_label_dir / fname
        if p.exists():
            m, _ = load_nii(p)
            ct_masks.append(m)
            ct_label_meta.append((color, label))

    iz_t1, iy_t1, ix_t1 = (
        best_slice_idx(gt_masks) if gt_masks
        else (t1.shape[0] // 2, t1.shape[1] // 2, t1.shape[2] // 2)
    )
    iz_ct, iy_ct, ix_ct = (
        best_slice_idx(ct_masks, thr=0.9) if ct_masks
        else (ct.shape[0] // 2, ct.shape[1] // 2, ct.shape[2] // 2)
    )

    row = row_offset

    def axes_row(r):
        return axes_grid[r]

    # --- raw CT (brain window 0–80 HU for soft tissue contrast) ---
    label_ax(axes_row(row)[0], f"{case_id}\nCT brain window\n{ct_sp[0]:.2f}×{ct_sp[1]:.2f}×{ct_sp[2]:.2f}mm")
    for col, sl in enumerate([
        norm_ct_brain(ct[iz_ct, :, :]),
        norm_ct_brain(ct[:, iy_ct, :]),
        norm_ct_brain(ct[:, :, ix_ct]),
    ]):
        show_slice(axes_row(row)[col + 1], sl)
    row += 1

    # --- raw T1 ---
    label_ax(axes_row(row)[0], "T1 raw")
    for col, sl in enumerate([
        norm_mri(t1[iz_t1, :, :]),
        norm_mri(t1[:, iy_t1, :]),
        norm_mri(t1[:, :, ix_t1]),
    ]):
        show_slice(axes_row(row)[col + 1], sl)
    row += 1

    # GT labels are bimodal (mostly 0s and 1s, 256 discrete levels) — threshold at 0.5.
    # CT labels are continuous ANTS soft probability maps with flat distribution — need 0.9.
    GT_THR = 0.5
    CT_THR = 0.9

    # --- GT overlays on T1 ---
    for (color, label), mask in zip(gt_meta, gt_masks):
        label_ax(axes_row(row)[0], label)
        for col, (bg, m) in enumerate(zip(
            [norm_mri(t1[iz_t1, :, :]), norm_mri(t1[:, iy_t1, :]), norm_mri(t1[:, :, ix_t1])],
            [(mask[iz_t1, :, :] > GT_THR).astype(np.float32),
             (mask[:, iy_t1, :] > GT_THR).astype(np.float32),
             (mask[:, :, ix_t1] > GT_THR).astype(np.float32)],
        )):
            overlay(axes_row(row)[col + 1], bg, m, color)
        row += 1

    # --- CT label overlays on CT (resampled to label space) ---
    # CT image may be on a different grid than CT labels — resample for alignment.
    # Use brain window (0–80 HU) to show soft tissue contrast.
    ct_in_label_space = resample_to(ct, ct_masks[0].shape, order=1) if ct_masks else ct
    for (color, label), mask in zip(ct_label_meta, ct_masks):
        label_ax(axes_row(row)[0], label)
        for col, (bg, m) in enumerate(zip(
            [norm_ct_brain(ct_in_label_space[iz_ct, :, :]),
             norm_ct_brain(ct_in_label_space[:, iy_ct, :]),
             norm_ct_brain(ct_in_label_space[:, :, ix_ct])],
            [(mask[iz_ct, :, :] > CT_THR).astype(np.float32),
             (mask[:, iy_ct, :] > CT_THR).astype(np.float32),
             (mask[:, :, ix_ct] > CT_THR).astype(np.float32)],
        )):
            overlay(axes_row(row)[col + 1], bg, m, color)
        row += 1

    return row - row_offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cases", type=int, default=3, help="Number of random cases to plot")
    parser.add_argument("--cases", nargs="+", default=None, help="Specific case IDs to plot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/datasets/deepxcontrast.png",
                        help="Save path (PNG). Default: show interactively.")
    args = parser.parse_args()

    if args.cases:
        case_ids = args.cases
    else:
        all_cases = sorted(d.name for d in DATASET_DIR.iterdir()
                           if d.is_dir() and (d / "0" / "CT.nii").exists())
        random.seed(args.seed)
        case_ids = random.sample(all_cases, min(args.n_cases, len(all_cases)))

    print(f"Plotting {len(case_ids)} cases: {case_ids}")

    # rows per case: 2 raw + 4 GT + 4 CT labels = 10
    rows_per_case = 2 + len(GT_LABELS) + len(CT_LABELS)
    n_rows = rows_per_case * len(case_ids)

    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(11, 2.2 * n_rows),
        gridspec_kw={"hspace": 0.04, "wspace": 0.02,
                     "width_ratios": [0.55, 1, 1, 1]},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers on row 0
    for col, title in enumerate(COL_HEADERS):
        axes[0][col].set_title(title, fontsize=8, fontweight="bold", pad=3)

    row = 0
    for case_id in case_ids:
        print(f"  [{case_id}] ...")
        used = plot_case(case_id, axes, row)
        row += max(used, rows_per_case)  # advance even if case was skipped

    # Hide unused rows
    for r in range(row, n_rows):
        for c in range(N_COLS):
            axes[r][c].axis("off")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

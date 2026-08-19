"""
Visualize a MAISI (NV-Generate-CTMR) generated CT image / mask pair.

The paired-generation pipeline writes two NIfTIs to its `output/` dir:
    sample_<ts>_image.nii.gz   -- synthetic CT, HU in [-1000, 1000]
    sample_<ts>_label.nii.gz   -- paired mask (filtered to `anatomy_list`)

For each pair it picks the most informative slice per anatomical axis (the slice
with the largest mask area) and renders CT + mask overlay for the three orthogonal
views plus a strip of axial slices spanning the mask extent.

Usage
-----
  # every image/label pair in a dir -> one PNG per pair (saved alongside each image):
  python experiments/3d/synth_task_generation/visualize_maisi_output.py \
      --dir /home/dpxuser/repos/NV-Generate-CTMR/output

  # write the PNGs to a separate dir:
  python experiments/3d/synth_task_generation/visualize_maisi_output.py \
      --dir /path/to/output --out_dir results/maisi_viz

  # a single explicit pair:
  python experiments/3d/synth_task_generation/visualize_maisi_output.py \
      --image /path/to/sample_..._image.nii.gz \
      --label /path/to/sample_..._label.nii.gz --out results/maisi_pair.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

MAISI_OUTPUT = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/synth_task_gen/nii")

# CT soft-tissue window (HU) for display.
WIN_CENTER, WIN_WIDTH = 40.0, 400.0


def find_pairs(out_dir: Path) -> list[tuple[Path, Path]]:
    """Return all (image, label) NIfTI pairs in `out_dir`, sorted by name.

    An image is any `*_image.nii.gz`; its label is the sibling with `_image`
    swapped for `_label`. Images without a matching label are skipped (warned).
    """
    pairs = []
    for image in sorted(out_dir.glob("*_image.nii.gz")):
        label = image.with_name(image.name.replace("_image.nii.gz", "_label.nii.gz"))
        if label.exists():
            pairs.append((image, label))
        else:
            print(f"WARN: no label for {image.name}, skipping")
    if not pairs:
        raise FileNotFoundError(f"No *_image.nii.gz / *_label.nii.gz pairs found in {out_dir}")
    return pairs


def window(ct: np.ndarray) -> np.ndarray:
    """Apply a soft-tissue CT window and normalize to [0, 1] for display."""
    lo, hi = WIN_CENTER - WIN_WIDTH / 2, WIN_CENTER + WIN_WIDTH / 2
    return np.clip((ct - lo) / (hi - lo), 0.0, 1.0)


def best_slice(mask: np.ndarray, axis: int) -> int:
    """Index of the slice with the largest mask area along `axis`
    (falls back to the volume center when the mask is empty)."""
    area = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    return int(area.argmax()) if area.any() else mask.shape[axis] // 2


_CMAP = plt.get_cmap("tab20")


def overlay(ax, ct2d: np.ndarray, mask2d: np.ndarray, title: str) -> None:
    """Draw a windowed CT slice with a semi-transparent mask overlay.

    Each label id gets its own color (multi-class friendly); a binary mask
    just shows as a single color.
    """
    ct2d = np.rot90(ct2d)
    mask2d = np.rot90(mask2d)
    ax.imshow(ct2d, cmap="gray", vmin=0, vmax=1)
    if mask2d.any():
        rgba = np.zeros((*mask2d.shape, 4))
        for lab in np.unique(mask2d):
            if lab == 0:
                continue
            r, g, b, _ = _CMAP(int(lab) % 20)
            sel = mask2d == lab
            rgba[sel] = (r, g, b, 0.45)
        ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def render_pair(image_path: Path, label_path: Path, out_path: Path, n_axial: int) -> None:
    """Render one CT/mask pair to `out_path` (top row = 3 orthogonal best-slices,
    bottom row = axial strip across the mask z-extent)."""
    ct = nib.load(str(image_path)).get_fdata().astype(np.float32)
    mask = nib.load(str(label_path)).get_fdata().astype(np.int16)
    ctw = window(ct)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    n_vox = int((mask > 0).sum())
    print(f"image {image_path.name}  shape {ct.shape}  HU [{ct.min():.0f},{ct.max():.0f}]")
    print(f"label {label_path.name}  fg labels {labels.tolist()}  fg voxels {n_vox}")

    n_ax = max(1, n_axial)
    fig = plt.figure(figsize=(3 * max(3, n_ax), 6.5))
    gs = fig.add_gridspec(2, max(3, n_ax))

    view_names = ["sagittal (x)", "coronal (y)", "axial (z)"]
    for axis, name in enumerate(view_names):
        idx = best_slice(mask, axis)
        ct2d = np.take(ctw, idx, axis=axis)
        m2d = np.take(mask, idx, axis=axis)
        ax = fig.add_subplot(gs[0, axis])
        overlay(ax, ct2d, m2d, f"{name}  slice {idx}")

    # Axial strip across the mask's z-extent (or whole volume if empty).
    z_area = (mask > 0).sum(axis=(0, 1))
    z_hit = np.where(z_area > 0)[0]
    if z_hit.size:
        z_slices = np.linspace(z_hit[0], z_hit[-1], n_ax).round().astype(int)
    else:
        z_slices = np.linspace(0, mask.shape[2] - 1, n_ax).round().astype(int)
    for col, z in enumerate(z_slices):
        ax = fig.add_subplot(gs[1, col])
        overlay(ax, ctw[:, :, z], mask[:, :, z], f"z={z}")

    fig.suptitle(f"MAISI CT/mask pair — {image_path.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def png_name(image_path: Path) -> str:
    """PNG filename for an image NIfTI: drop the `_image.nii.gz` suffix."""
    return image_path.name.replace("_image.nii.gz", "") + ".png"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=MAISI_OUTPUT, help="dir of *_image/*_label NIfTI pairs (one PNG per pair)")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/synth_task_gen/maisi"),
        help="where to write PNGs (default: the shared analysis maisi dir)",
    )
    ap.add_argument("--image", type=Path, default=None, help="single CT NIfTI (overrides --dir)")
    ap.add_argument("--label", type=Path, default=None, help="mask NIfTI for --image (default: paired sibling)")
    ap.add_argument("--out", type=Path, default=None, help="output PNG for the single-pair --image mode")
    ap.add_argument("--n_axial", type=int, default=6, help="number of axial slices in the bottom strip")
    args = ap.parse_args()

    if args.image is not None:
        # single explicit pair
        label = args.label or args.image.with_name(args.image.name.replace("_image.nii.gz", "_label.nii.gz"))
        out = args.out or args.image.with_name(png_name(args.image))
        render_pair(args.image, label, out, args.n_axial)
        return

    pairs = find_pairs(args.dir)
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"found {len(pairs)} pair(s) in {args.dir}")
    for image, label in pairs:
        out = (args.out_dir / png_name(image)) if args.out_dir else image.with_name(png_name(image))
        render_pair(image, label, out, args.n_axial)


if __name__ == "__main__":
    main()

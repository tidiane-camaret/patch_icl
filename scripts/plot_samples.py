"""
Plot a grid of (CT slice, segmentation mask) pairs from TotalSegmentator.
Loads full 3-D volumes and picks the middle annotated axial slice to display.

Usage:
    uv run python3 scripts/plot_samples.py                          # defaults
    uv run python3 scripts/plot_samples.py --n 8 --classes liver spleen pancreas
    uv run python3 scripts/plot_samples.py --out my_plot.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from totalseg_dataset import TotalSegDataset, ALL_CLASSES

DATA_ROOT = "/work/dlclarge2/ndirt-SegFM3D/data/totalseg"
LABEL_CMAP = plt.colormaps["tab20"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DATA_ROOT)
    p.add_argument("--n", type=int, default=6, help="Number of subjects to show")
    p.add_argument(
        "--classes", nargs="+",
        default=["liver", "spleen", "kidney_left", "kidney_right", "pancreas", "aorta"],
    )
    p.add_argument("--image_size", type=int, nargs=3, default=[64, 256, 256],
                   metavar=("D", "H", "W"))
    p.add_argument("--max_subjects", type=int, default=20)
    p.add_argument("--out", default="results/sample_pairs.png")
    return p.parse_args()


def pick_representative_slice(label: np.ndarray) -> int:
    """Return the axial index of the slice with the most foreground voxels."""
    counts = (label > 0).sum(axis=(1, 2))
    return int(counts.argmax())


def make_overlay(image: np.ndarray, label: np.ndarray, n_classes: int) -> np.ndarray:
    rgb = np.stack([image] * 3, axis=-1)
    overlay = np.zeros_like(rgb)
    alpha = 0.45
    for cls_idx in range(1, n_classes):
        colour = np.array(LABEL_CMAP(cls_idx / max(n_classes, 1))[:3])
        overlay[label == cls_idx] = colour
    has_fg = label > 0
    rgb[has_fg] = (1 - alpha) * rgb[has_fg] + alpha * overlay[has_fg]
    return np.clip(rgb, 0, 1)


def main():
    args = parse_args()
    classes = args.classes
    image_size = tuple(args.image_size)  # (D, H, W)

    print(f"Classes    : {classes}")
    print(f"Image size : {image_size}  (D, H, W)")

    dataset = TotalSegDataset(
        root=args.root,
        classes=classes,
        image_size=image_size,
        max_subjects=args.max_subjects,
    )

    n = min(args.n, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["CT slice", "GT label", "Overlay"]):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, idx in enumerate(indices):
        image_t, label_t = dataset[int(idx)]        # (1,D,H,W), (D,H,W)
        image_vol = image_t.squeeze(0).numpy()      # (D, H, W)
        label_vol = label_t.numpy()                 # (D, H, W)

        sl = pick_representative_slice(label_vol)
        image = image_vol[sl]                       # (H, W)
        label = label_vol[sl]                       # (H, W)

        subj = dataset.subjects[int(idx)]
        axes[row, 0].set_ylabel(f"{subj} / axial {sl}", fontsize=8,
                                rotation=0, labelpad=110, va="center")

        label_rgb = LABEL_CMAP(label / max(len(classes), 1))[:, :, :3]
        label_rgb[label == 0] = 0

        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(label_rgb)
        axes[row, 2].imshow(make_overlay(image, label, dataset.num_classes))

        for ax in axes[row]:
            ax.axis("off")

    patches = [
        mpatches.Patch(color=LABEL_CMAP(i / max(len(classes), 1)), label=classes[i - 1])
        for i in range(1, len(classes) + 1)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=min(len(classes), 6),
               fontsize=9, title="Classes", title_fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("TotalSegmentator — 3-D volumes, representative axial slice",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

"""
Thesis figure: shape-family gallery on the synthetic-bank canvas
(sections/methodology_synth.tex, Table "synth-shapes" + "Shapes." paragraph).

One row per procedural shape family, several example target crops per row, so the
table of family names/parameters has a visual counterpart. Image and mask are
plotted as separate panels (not overlaid).

Usage
-----
  python experiments/3d/plot_synth_family_gallery.py
  python experiments/3d/plot_synth_family_gallery.py --n_examples 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling `plot_synth_canvas_compare`

from plot_synth_canvas_compare import DEFAULT_BANK, _bank_row, _display_slice  # noqa: E402

FAMILIES = ["blob", "disk", "splatter", "scatter_field", "cylinder", "vessel", "torus"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default=DEFAULT_BANK)
    p.add_argument("--n_examples", type=int, default=2, help="example target crops per family")
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--crop_spacing_mm", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/3d/synth_family_gallery.png")
    args = p.parse_args()

    args.context_size = 0  # only the target crop is rendered per example

    n_cols = 2 * args.n_examples   # (image, mask) per example
    col_w = 1.8
    fig, axes = plt.subplots(len(FAMILIES), n_cols,
                             figsize=(col_w * n_cols, col_w * len(FAMILIES)),
                             squeeze=False, gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    for ex in range(args.n_examples):
        axes[0, 2 * ex].set_title("image", fontsize=8)
        axes[0, 2 * ex + 1].set_title("mask", fontsize=8)

    for row, family in enumerate(FAMILIES):
        for ex in range(args.n_examples):
            crops = _bank_row(args, family, args.seed + row * args.n_examples + ex)
            nc = crops[0]   # target only (context_size=0)
            img = nc.image.float().unsqueeze(0)
            mask = nc.label_frac.float()
            img_sl, mask_sl = _display_slice(img, mask)
            axes[row, 2 * ex].imshow(img_sl, cmap="gray")
            axes[row, 2 * ex + 1].imshow(mask_sl, cmap="hot", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(family, fontsize=10, rotation=90, labelpad=6, va="center")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Shape families, synthetic-bank canvas", fontsize=11, y=0.995)
    fig.tight_layout(h_pad=0.1, w_pad=0.1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

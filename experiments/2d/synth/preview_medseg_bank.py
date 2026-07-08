"""Montage of MedSegObjectBank renditions to eyeball the object set.

Builds the bank for a few datasets and shows, per (dataset,label) class, several
renditions as (intensity channel | mask channel) pairs — the exact tiles render.py
will paste (intensity into the image, mask into the label).

Run: .venv/bin/python experiments/2d/synth/preview_medseg_bank.py
     .venv/bin/python experiments/2d/synth/preview_medseg_bank.py --datasets busi drive kvasir cellnuclei monusac
"""
import argparse
import sys; sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.omniSynth.config import OmniMedSegConfig
from src.datasets.omniSynth.bank_medseg import MedSegObjectBank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["busi", "drive", "kvasir", "cellnuclei", "monusac", "pandental"])
    ap.add_argument("--split", default="train", help="train | val (val reads val images)")
    ap.add_argument("--cell_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=128, help="canvas size (canvas sizing)")
    ap.add_argument("--cell_margin", type=float, default=-0.15)
    ap.add_argument("--size_mode", default="canvas", choices=["canvas", "cell"])
    ap.add_argument("--n", type=int, default=6, help="renditions per class")
    ap.add_argument("--out", default="results/medseg_bank_preview.png")
    args = ap.parse_args()

    cfg = OmniMedSegConfig(train_datasets=tuple(args.datasets),
                           val_datasets=tuple(args.datasets), max_renditions_per_class=80,
                           size_mode=args.size_mode)
    bank = MedSegObjectBank(cfg, cell_size=args.cell_size, cell_margin=args.cell_margin,
                            split=args.split, image_size=args.image_size)
    cids = bank.task_ids()
    print(f"medseg bank [{args.split}]: {len(cids)} classes -> {[bank.alphabet(c) for c in cids]}")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(cids), args.n * 2, figsize=(args.n * 2 * 1.1, len(cids) * 1.2))
    axes = np.atleast_2d(axes)
    for r, cid in enumerate(cids):
        rends = bank.get(cid)
        pick = rng.choice(len(rends), size=min(args.n, len(rends)), replace=False)
        for j in range(args.n):
            c = j * 2
            if j < len(pick):
                tile = rends[int(pick[j])]
                axes[r, c].imshow(tile[0], cmap="gray", vmin=0, vmax=1)      # intensity
                axes[r, c + 1].imshow(tile[1], cmap="gray", vmin=0, vmax=1)  # mask
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c + 1].set_xticks([]); axes[r, c + 1].set_yticks([])
        axes[r, 0].set_ylabel(bank.alphabet(cid), fontsize=7)
    for c in range(args.n):
        axes[0, c * 2].set_title("int", fontsize=6)
        axes[0, c * 2 + 1].set_title("mask", fontsize=6)
    fig.suptitle(f"MedSegObjectBank renditions (cell={args.cell_size}, margin={args.cell_margin})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

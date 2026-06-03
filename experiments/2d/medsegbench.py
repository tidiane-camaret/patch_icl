"""
Visualise MedSegBench samples: target + context with mask overlay.

Usage:
    python experiments/2d/medsegbench.py
    python experiments/2d/medsegbench.py --dataset abdomenus --size 256 --n 4
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.medsegbench import MedSegBenchDataset


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Return RGB image with mask blended in red."""
    rgb = np.stack([image, image, image], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] + alpha * mask, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] - alpha * mask, 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] - alpha * mask, 0, 1)
    return rgb


def plot_sample(ax, image: np.ndarray, mask: np.ndarray, title: str):
    ax.imshow(overlay(image, mask), vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="single dataset name, or None for all")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--n", type=int, default=4, help="number of target samples to show")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/datasets/medsegbench.png", help="save path (e.g. out.png); shows interactively if None")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    datasets = [args.dataset] if args.dataset else None
    ds = MedSegBenchDataset(
        split=args.split,
        context_size=args.context_size,
        image_size=args.size,
        datasets=datasets,
    )

    indices = random.sample(range(len(ds)), min(args.n, len(ds)))
    n_cols = 1 + args.context_size  # target + k context
    fig, axes = plt.subplots(args.n, n_cols, figsize=(2.5 * n_cols, 2.5 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        item = ds[idx]
        image = item["image"][0].numpy()   # [H, W]
        label = item["label"][0].numpy()   # [H, W]
        ctx_in = item["context_in"]        # [K, 1, H, W]
        ctx_out = item["context_out"]      # [K, 1, H, W]

        plot_sample(axes[row, 0], image, label, f"target [{idx}]")

        for col in range(1, n_cols):
            ax = axes[row, col]
            k = col - 1
            if k < ctx_in.shape[0]:
                plot_sample(ax, ctx_in[k, 0].numpy(), ctx_out[k, 0].numpy(), f"ctx {k+1}")
            else:
                ax.axis("off")

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

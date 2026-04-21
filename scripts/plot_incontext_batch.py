"""
Visualise a few in-context batch items.

For each sample shows the axial mid-slice of:
  target image | target mask | ctx-1 image | ctx-1 mask | ... | ctx-K image | ctx-K mask

Usage
-----
  python scripts/plot_incontext_batch.py [--data DIR] [--n_samples 4] [--out results/incontext_batch.png]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from torch.utils.data import DataLoader


def best_slice(img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the axial slice where the mask has the most foreground pixels."""
    if img.dim() == 4:
        img = img.squeeze(0)    # (D,H,W)
    if mask.dim() == 3:
        pass                    # already (D,H,W)
    counts = mask.sum(dim=(1, 2))  # (D,)
    z = int(counts.argmax())
    return img[z], mask[z]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="/work/dlclarge2/ndirt-SegFM3D/data/totalseg")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--out",       default="results/incontext_batch.png")
    args = parser.parse_args()

    K = args.context_size
    N = args.n_samples

    ds = TotalSegInContextDataset(
        root=args.data,
        classes=["kidney_left"],
        image_size=(64, 64, 64),
        split="val",
        context_size=K,
        max_subjects=50,          # quick scan for the plot
    )

    loader = DataLoader(
        ds, batch_size=N, shuffle=True, num_workers=0,
        collate_fn=incontext_collate_fn,
    )
    batch = next(iter(loader))

    # columns: target_img, target_lbl, (ctx_img, ctx_lbl) × K
    n_cols = 2 + 2 * K
    fig, axes = plt.subplots(N, n_cols, figsize=(2.5 * n_cols, 2.5 * N))
    if N == 1:
        axes = axes[None]          # ensure 2-D array

    col_titles = ["target\nimage", "target\nmask"] + [
        f"ctx-{k+1}\nimage" if even else f"ctx-{k//2+1}\nmask"
        for k in range(2 * K)
        for even in [k % 2 == 0]
    ]
    # Simpler: build titles in order
    col_titles = (
        ["target\nimage", "target\nmask"] +
        [t for k in range(K) for t in (f"ctx {k+1}\nimage", f"ctx {k+1}\nmask")]
    )

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8)

    for row in range(N):
        lbl_name = batch["label_names"][row]
        subj     = batch["subjects"][row]

        # target — pick slice with most mask pixels
        tgt_img, tgt_lbl = best_slice(batch["image"][row], batch["label"][row])
        axes[row, 0].imshow(tgt_img.numpy(), cmap="gray")
        axes[row, 1].imshow(tgt_lbl.numpy(), cmap="gray", vmin=0, vmax=1)

        # context pairs
        for k in range(K):
            ctx_img, ctx_mask = best_slice(
                batch["context_in"][row, k], batch["context_out"][row, k]
            )
            axes[row, 2 + 2*k].imshow(ctx_img.numpy(),  cmap="gray")
            axes[row, 3 + 2*k].imshow(ctx_mask.numpy(), cmap="gray", vmin=0, vmax=1)

        axes[row, 0].set_ylabel(f"{subj}\n{lbl_name}", fontsize=7)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

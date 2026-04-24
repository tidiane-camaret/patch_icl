"""
Visualise a few items from the synth path of TotalSegInContextDataset.

Each row shows one supervoxel duplicated K+1 times with independent heavy
augmentation — target + K context copies, each as (image, mask) side by side.
Useful for checking that augmentations are reasonable and masks stay aligned.

Usage
-----
  python scripts/synth_labels/plot_synth_batch.py
  python scripts/synth_labels/plot_synth_batch.py --method slic --unions --n-samples 6
  python scripts/synth_labels/plot_synth_batch.py --data /path/to/totalseg --out results/synth.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn


def _best_slice(img: torch.Tensor, mask: torch.Tensor):
    """Return the axial slice (z) where the mask has the most foreground pixels."""
    img  = img.squeeze(0)                   # (D, H, W)
    counts = mask.sum(dim=(1, 2))           # (D,)
    z = int(counts.argmax()) if counts.max() > 0 else img.shape[0] // 2
    return img[z].numpy(), mask[z].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         default=None,
                        help="Override paths.totalseg from config.yaml")
    parser.add_argument("--method",       default="seeds3d",
                        choices=["grid", "watershed", "slic", "seeds3d"],
                        help="Synth label method to visualise")
    parser.add_argument("--unions",       action="store_true",
                        help="Load label_synth_<method>_union.npy instead")
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--n-samples",    type=int, default=4)
    parser.add_argument("--image-size",   type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--max-subjects", type=int, default=50)
    parser.add_argument("--out",          default="results/synth_batch.png")
    args = parser.parse_args()

    # Load config (paths + augmentations)
    cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    aug_cfg = OmegaConf.load(ROOT / "configs" / "augmentations.yaml").augmentations

    data_dir = args.data or cfg.paths.totalseg

    K = args.context_size
    N = args.n_samples

    ds = TotalSegInContextDataset(
        root=data_dir,
        classes=["liver"],          # real classes don't matter; synth ignores them
        image_size=tuple(args.image_size),
        split=None,
        context_size=K,
        max_subjects=args.max_subjects,
        aug_cfg=aug_cfg,
        synth_method=args.method,
        synth_unions=args.unions,
        p_synth=1.0,               # always take the synth path for this plot
    )

    if not ds._synth_subjects:
        sys.exit(f"No subjects with label_synth_{args.method}"
                 f"{'_union' if args.unions else ''}.npy found in {data_dir}")

    loader = DataLoader(ds, batch_size=N, shuffle=True, num_workers=0,
                        collate_fn=incontext_collate_fn)
    batch = next(iter(loader))

    # Layout: one column pair (image | mask) for target + each context copy
    n_pairs = 1 + K
    n_cols  = 2 * n_pairs
    fig, axes = plt.subplots(N, n_cols, figsize=(2.2 * n_cols, 2.2 * N),
                             squeeze=False)

    titles = [t for i in range(n_pairs)
              for t in (("target\nimage" if i == 0 else f"ctx {i}\nimage"),
                        ("target\nmask"  if i == 0 else f"ctx {i}\nmask"))]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=8)

    for row in range(N):
        label_name = batch["label_names"][row]
        subj       = batch["subjects"][row]

        img_s, msk_s = _best_slice(batch["image"][row], batch["label"][row])
        axes[row, 0].imshow(img_s, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(msk_s, cmap="hot",  vmin=0, vmax=1, alpha=0.9)

        for k in range(K):
            img_s, msk_s = _best_slice(batch["context_in"][row, k],
                                       batch["context_out"][row, k])
            axes[row, 2 + 2*k].imshow(img_s, cmap="gray", vmin=0, vmax=1)
            axes[row, 3 + 2*k].imshow(msk_s, cmap="hot",  vmin=0, vmax=1, alpha=0.9)

        axes[row, 0].set_ylabel(f"{subj}\n{label_name}", fontsize=7)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    method_tag = args.method + ("_union" if args.unions else "")
    fig.suptitle(f"Synth in-context items  —  method={method_tag}  "
                 f"size={args.image_size}  K={K}", fontsize=9)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

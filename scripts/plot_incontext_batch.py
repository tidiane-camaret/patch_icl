"""
Visualise in-context samples from TotalSegInContextDataset.

Builds the *training* dataset exactly as scripts/train.py does — same Hydra
`data.*` config (train split, augmentations, synth path, class balancing,
crop, image size, context size) — so the figure shows precisely what the
model is trained on.  Change anything via the same Hydra overrides train.py
accepts (they are forwarded verbatim).

Each row is one sample; columns are: target | ctx-1 | ctx-2 | … | ctx-K.
Each cell shows the axial slice with the most mask coverage, with a
semi-transparent colour overlay of the segmentation mask.

Usage
-----
  python scripts/plot_incontext_batch.py
  python scripts/plot_incontext_batch.py --n_samples 6
  python scripts/plot_incontext_batch.py --out results/my_batch.png
  # Hydra overrides are forwarded, just like train.py:
  python scripts/plot_incontext_batch.py cluster=meta
  python scripts/plot_incontext_batch.py data.p_synth=1.0    # synth-only view
  python scripts/plot_incontext_batch.py augmentations.enabled=false
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.totalseg_classes import resolve_classes
from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MASK_COLOUR = np.array([1.0, 0.2, 0.2])   # red overlay


def _best_slice(img: torch.Tensor, mask: torch.Tensor):
    """Return (img_slice, mask_slice) at the axial depth with most mask pixels."""
    img  = img.squeeze(0)       # (D, H, W)
    z    = int(mask.sum(dim=(1, 2)).argmax()) if mask.any() else img.shape[0] // 2
    return img[z].numpy(), mask[z].numpy()


def _overlay(img_slice: np.ndarray, mask_slice: np.ndarray, alpha: float = 0.45):
    """Blend a 2-D float image with a binary mask overlay."""
    # Normalise image to [0, 1] for display
    lo, hi = img_slice.min(), img_slice.max()
    img_n = (img_slice - lo) / (hi - lo + 1e-8)
    rgb = np.stack([img_n] * 3, axis=-1)
    fg = mask_slice > 0
    rgb[fg] = (1 - alpha) * rgb[fg] + alpha * MASK_COLOUR
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--out",       default="results/incontext_batch.png")
    parser.add_argument("-h", "--help", action="store_true")
    args, hydra_overrides = parser.parse_known_args()

    if args.help:
        parser.print_help(); return

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=hydra_overrides)

    # Mirror train.py's train_ds exactly so the figure reflects the training data.
    train_classes = resolve_classes(cfg.data.train_classes, cfg.paths.totalseg)
    synth_method  = cfg.data.synth_method or None

    ds = TotalSegInContextDataset(
        root=cfg.paths.totalseg,
        classes=train_classes,
        image_size=tuple(cfg.data.image_size),
        split="train",
        context_size=cfg.data.context_size,
        max_subjects=cfg.data.max_train_subjects,
        aug_cfg=cfg.augmentations,
        synth_method=synth_method,
        synth_unions=cfg.data.synth_unions,
        p_synth=cfg.data.p_synth,
        class_balanced=cfg.data.class_balanced,
        use_crop=cfg.data.use_crop,
    )

    K  = cfg.data.context_size
    N  = min(args.n_samples, len(ds))
    loader = DataLoader(ds, batch_size=N, shuffle=True, num_workers=0,
                        collate_fn=incontext_collate_fn)
    batch = next(iter(loader))

    # Layout: one column per volume (target + K contexts)
    n_cols = 1 + K
    col_w  = 2.4
    fig, axes = plt.subplots(N, n_cols, figsize=(col_w * n_cols, col_w * N),
                             squeeze=False)

    col_titles = ["target"] + [f"ctx {k+1}" for k in range(K)]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, pad=4)

    for row in range(N):
        lbl_name = batch["label_names"][row]
        subj     = batch["subjects"][row]

        # target
        img_sl, mask_sl = _best_slice(batch["image"][row], batch["label"][row])
        axes[row, 0].imshow(_overlay(img_sl, mask_sl))
        axes[row, 0].set_ylabel(f"{subj}\n{lbl_name}", fontsize=7,
                                rotation=0, labelpad=90, va="center")

        # context pairs
        for k in range(K):
            img_sl, mask_sl = _best_slice(
                batch["context_in"][row, k], batch["context_out"][row, k]
            )
            axes[row, 1 + k].imshow(_overlay(img_sl, mask_sl))

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    aug_on    = cfg.augmentations is not None and cfg.augmentations.enabled
    aug_tag   = " + aug"                        if aug_on             else ""
    synth_tag = f" + synth(p={cfg.data.p_synth})" if synth_method else ""
    fig.suptitle(
        f"Train samples  |  split=train  |  K={K}{aug_tag}{synth_tag}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

"""
Visualise a few controlSynth (data.source=synthetic) in-context samples, orig vs augmented.

Mirrors scripts/plot_biomedparse_aug.py but builds the synthetic dataset through the
real config wiring (Hydra-composed pfn_seg cfg + common.build_dataset), so the synth
difficulty knobs in configs/experiment/2d/synth/*.yaml are exactly what training uses.
Each sample shows its K context (image, mask) pairs + the query (image, GT), with an
orig row and a row after the 2D augment() with the chosen preset.

Usage
-----
    python scripts/plot_synth_aug.py
    python scripts/plot_synth_aug.py --n 5 --preset 2d_strong --synth hard_diverse
    python scripts/plot_synth_aug.py --overrides synth.live.foreground_contrast=0.3
"""

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "2d"))

from common import build_dataset
from pfn_train import augment


def _overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB overlay: grayscale image [0,1] with the mask in red."""
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    if mask.max() > 0:
        red = np.zeros_like(rgb)
        red[..., 0] = 220
        fg = (mask > 0)[..., None]
        rgb = (rgb * (1 - alpha * fg) + red * alpha * fg).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",            type=int, default=5, help="number of samples")
    ap.add_argument("--preset",       default="2d_strong")
    ap.add_argument("--synth",        default="default", help="configs/experiment/2d/synth/<name>.yaml")
    ap.add_argument("--context_size", type=int, default=3)
    ap.add_argument("--image_size",   type=int, default=128)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--overrides", nargs="*", default=[], help="extra Hydra dot-overrides")
    ap.add_argument("--out", default=str(ROOT / "results" / "controlsynth" / "aug_samples.png"))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "2d"), version_base=None):
        cfg = compose(config_name="pfn_seg", overrides=[
            "data.source=synthetic", f"synth={args.synth}",
            f"data.image_size={args.image_size}", f"data.context_size={args.context_size}",
            *args.overrides])

    aug_cfg = OmegaConf.load(ROOT / "configs" / "augmentations" / f"{args.preset}.yaml")
    ds = build_dataset(cfg, "train")          # train split = random episode per __getitem__

    items = [ds[random.randrange(len(ds))] for _ in range(args.n)]

    K, H = args.context_size, args.image_size
    imgs = torch.stack([torch.cat([it["context_in"], it["image"].unsqueeze(0)], dim=0) for it in items])
    msks = torch.stack([torch.cat([it["context_out"], torch.zeros(1, 1, H, H)], dim=0) for it in items])
    gts  = torch.stack([it["label"] for it in items])     # real query GT (un-augmented)

    aug_imgs, aug_msks = augment(imgs.clone(), msks.clone(), K, aug_cfg)

    B = len(items)
    ncols = 2 * (K + 1)
    fig, axes = plt.subplots(2 * B, ncols, figsize=(2.3 * ncols, 2.3 * 2 * B))
    col_titles = [t for k in range(K) for t in (f"ctx{k}", f"ctx{k}+m")] + ["query", "query+GT"]

    def row_cells(imgs_b, msks_b, qgt):
        cells = []
        for k in range(K):
            ci, cm = imgs_b[k, 0].numpy(), msks_b[k, 0].numpy()
            cells += [ci, _overlay(ci, cm)]
        qi = imgs_b[K, 0].numpy()
        cells += [qi, _overlay(qi, qgt)]
        return cells

    for b in range(B):
        morph = items[b]["meta"]["morphology"]
        rows = [
            ("orig", row_cells(imgs[b],     msks[b],     gts[b, 0].numpy())),
            (f"aug:{args.preset}", row_cells(aug_imgs[b], aug_msks[b], gts[b, 0].numpy())),
        ]
        for r, (tag, cells) in enumerate(rows):
            axr = axes[2 * b + r]
            for col, (ax, data) in enumerate(zip(axr, cells)):
                ax.imshow(data, cmap="gray" if data.ndim == 2 else None,
                          vmin=0, vmax=1 if data.ndim == 2 else None)
                ax.set_xticks([]); ax.set_yticks([])
                if 2 * b + r == 0:
                    ax.set_title(col_titles[col], fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"{tag}\n{morph}", fontsize=6, rotation=0, labelpad=36, va="center")

    fig.suptitle(f"controlSynth (synth={args.synth}) — {B} samples, orig vs aug={args.preset}, K={K}, {H}px. "
                 f"Geometric is context-only; query stays full-frame.", fontsize=9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

"""
Visualise in-context items from any 3D dataset, straight from the Hydra config.

Builds the dataset through experiments/3d/common.build_dataset — the same
source->dataset wiring the 3D train/eval scripts use — so the figure shows
exactly what a model sees for the selected `data.source` and split.

Each row is one item; columns are: target | ctx-1 | … | ctx-K.  Each cell shows
the axial slice with the most mask coverage, with a semi-transparent colour
overlay of the segmentation mask (red for binary; distinct colours per label id
for multi-label / random-coloured items).

Usage
-----
  python experiments/3d/plot_dataset_items.py                        # totalseg train split
  python experiments/3d/plot_dataset_items.py dataset=omnisynth3d --split val
  python experiments/3d/plot_dataset_items.py --n_samples 6 --out results/x.png
  # Hydra overrides forwarded after the argparse flags:
  python experiments/3d/plot_dataset_items.py dataset=omnisynth3d synth3d.tiles_root=results/3d/omni_tiles
  python experiments/3d/plot_dataset_items.py cluster=meta           # totalseg on meta cluster
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling `common` (dir name '3d' isn't importable)

from src.totalseg_dataloader_incontext import incontext_collate_fn
from common import build_dataset  # noqa: E402  experiments/3d/common.py


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

_BINARY_COLOUR = [1.0, 0.2, 0.2]   # red


def _label_colours(num_labels: int, palette: np.ndarray | None) -> dict[int, list[float]]:
    """Map each label id (1…num_labels) to an RGB colour.

    Uses the per-sample `label_palette` when present (random-coloured items),
    a single red for binary masks, else a distinct tab10 colour per id.
    """
    if palette is not None:
        return {i: palette[i].tolist() for i in range(1, palette.shape[0])}
    if num_labels <= 1:
        return {1: _BINARY_COLOUR}
    cmap = plt.colormaps["tab10"]
    return {i: list(cmap((i - 1) % 10)[:3]) for i in range(1, num_labels + 1)}


def _best_slice(img: torch.Tensor, mask: torch.Tensor):
    """Return (img_slice, mask_slice) at the axial depth with most foreground."""
    img = img.squeeze(0)                       # (D, H, W)
    fg  = mask > 0
    z   = int(fg.sum(dim=(1, 2)).argmax()) if fg.any() else img.shape[0] // 2
    return img[z].numpy(), mask[z].numpy()


def _raw_rgb(img_slice: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2-D slice to a [0,1] greyscale RGB image (no overlay)."""
    lo, hi = img_slice.min(), img_slice.max()
    img_n  = (img_slice - lo) / (hi - lo + 1e-8)
    return np.stack([img_n] * 3, axis=-1)


def _overlay(img_slice: np.ndarray, mask_slice: np.ndarray,
             colours: dict[int, list[float]], alpha: float = 0.5) -> np.ndarray:
    """Blend a 2-D float image with a per-label colour overlay."""
    rgb = _raw_rgb(img_slice)
    for lid, col in colours.items():
        fg = mask_slice == lid
        if fg.any():
            rgb[fg] = (1 - alpha) * rgb[fg] + alpha * np.array(col)
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split",     default="train", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--out",       default="results/3d/dataset_items.png")
    parser.add_argument("--separate_overlay", action="store_true",
                        help="render each volume as a raw-scan column plus a separate "
                             "mask-overlay column (to judge how much the object blends).")
    parser.add_argument("-h", "--help", action="store_true")
    args, hydra_overrides = parser.parse_known_args()

    if args.help:
        parser.print_help(); return

    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=hydra_overrides)

    ds = build_dataset(cfg, args.split)

    K = cfg.data.context_size
    N = min(args.n_samples, len(ds))
    num_labels = cfg.data.get("num_labels_per_sample", 1)

    loader = DataLoader(ds, batch_size=N, shuffle=True, num_workers=0,
                        collate_fn=incontext_collate_fn)
    batch = next(iter(loader))
    has_palette = "label_palette" in batch

    # Layout: one column per volume (target + K contexts), or — with
    # --separate_overlay — two columns per volume (raw scan, then mask overlay).
    sep       = args.separate_overlay
    per_vol   = 2 if sep else 1
    vol_names = ["target"] + [f"ctx {k+1}" for k in range(K)]
    n_cols    = per_vol * (1 + K)
    col_w     = 2.4
    fig, axes = plt.subplots(N, n_cols, figsize=(col_w * n_cols, col_w * N),
                             squeeze=False)

    for v, name in enumerate(vol_names):
        if sep:
            axes[0, 2 * v].set_title(name, fontsize=9, pad=4)
            axes[0, 2 * v + 1].set_title(f"{name} · seg", fontsize=9, pad=4)
        else:
            axes[0, v].set_title(name, fontsize=9, pad=4)

    for row in range(N):
        palette = batch["label_palette"][row].numpy() if has_palette else None
        colours = _label_colours(num_labels, palette)

        vols = [(batch["image"][row], batch["label"][row])]
        vols += [(batch["context_in"][row, k], batch["context_out"][row, k])
                 for k in range(K)]

        for v, (im, mk) in enumerate(vols):
            img_sl, mask_sl = _best_slice(im, mk)
            if sep:
                axes[row, 2 * v].imshow(_raw_rgb(img_sl))               # scan only
                axes[row, 2 * v + 1].imshow(_overlay(img_sl, mask_sl, colours))
            else:
                axes[row, v].imshow(_overlay(img_sl, mask_sl, colours))

        axes[row, 0].set_ylabel(f"{batch['subjects'][row]}\n{batch['label_names'][row]}",
                                fontsize=7, rotation=0, labelpad=90, va="center")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    source = cfg.data.get("source", "totalseg")
    if source == "omnisynth3d":
        # omniSynth 3D composes scenes itself (no task-aug / supervoxel-synth path); its
        # scene knobs live in cfg.synth3d, so surface those instead of the totalseg tags.
        s3 = cfg.synth3d
        extra = (f"  |  n_obj={s3.n_objects}  k={s3.k_min}-{s3.k_max}"
                 f"  mode={s3.target_mode}  bg={s3.background}")
    elif source == "anchor_synth3d":
        # anchor_synth3d draws synthetic objects at anchor-relative positions on real CT;
        # scene knobs live in cfg.anchor_synth. On the train split build_dataset also
        # forwards the multiverseg task+intensity aug (like the totalseg path), so the
        # plotted items are augmented there.
        a = cfg.anchor_synth
        aug_on  = args.split == "train" and cfg.augmentations.enabled
        aug_tag = " + aug" if aug_on else ""
        extra = (f"  |  obj={a.object_source}/{a.shape}  n_obj={a.n_objects}"
                 f"  anchors={a.n_anchors}  size_frac={a.object_size_frac_min}-"
                 f"{a.object_size_frac_max}  Δ={a.contrast_delta}{aug_tag}")
    else:
        aug_on    = args.split == "train" and cfg.augmentations.enabled
        synth_on  = args.split == "train" and bool(cfg.data.synth_method)
        aug_tag   = " + aug"                          if aug_on   else ""
        synth_tag = f" + synth(p={cfg.data.p_synth})" if synth_on else ""
        extra     = f"{aug_tag}{synth_tag}"
    fig.suptitle(
        f"{source}  |  split={args.split}  |  K={K}{extra}",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

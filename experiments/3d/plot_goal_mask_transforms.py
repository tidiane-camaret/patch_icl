"""Visualise the src/mask_transforms goal-mask ops on real dataset target masks.

Rows = items; columns = original | dilate | erode | boundary | sobel. Every column
shows the SAME axial slice (the original mask's best slice) of the target CT with the
transformed mask overlaid, so the shape change reads directly. Radii are mm -> voxels
via each item's crop pitch, exactly as src/gpu_augment._goal_mask_transform does.

  python experiments/3d/plot_goal_mask_transforms.py dataset=d1
  python experiments/3d/plot_goal_mask_transforms.py --radius_mm 5 --n_samples 8 --ball
  python experiments/3d/plot_goal_mask_transforms.py dataset=d1 --split val
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling `common`

from src.totalseg_dataloader_incontext import incontext_collate_fn          # noqa: E402
from src.mask_transforms import apply_goal_op, mm_to_vox                     # noqa: E402
from common import build_dataset, eval_cfg                                   # noqa: E402
from plot_dataset_items import _overlay, _label_colours, _raw_rgb           # noqa: E402

OPS = ["dilate", "erode", "boundary", "sobel"]


def _best_z(mask: torch.Tensor) -> int:
    fg = (mask.round() == 1) if mask.is_floating_point() else (mask > 0)
    return int(fg.sum(dim=(1, 2)).argmax()) if fg.any() else mask.shape[0] // 2


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--radius_mm", type=float, default=3.0,
                    help="dilate/erode radius & boundary half-width (mm); ignored by sobel")
    ap.add_argument("--ball", action="store_true",
                    help="Euclidean ball structuring element (default: cube / L-inf)")
    ap.add_argument("--min_keep", type=float, default=0.0,
                    help="erode/boundary: fraction of each mask's fg that erosion must leave "
                         "(0 = none; 0.3 keeps small organs from vanishing)")
    ap.add_argument("--out", default="results/3d/goal_mask_transforms.png")
    ap.add_argument("-h", "--help", action="store_true")
    args, hydra_overrides = ap.parse_known_args()
    if args.help:
        ap.print_help()
        return

    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=hydra_overrides)
    if args.split != "train":
        cfg = eval_cfg(cfg)

    ds = build_dataset(cfg, args.split)
    N = min(args.n_samples, len(ds))
    batch = next(iter(DataLoader(ds, batch_size=N, shuffle=True, num_workers=0,
                                 collate_fn=incontext_collate_fn)))

    colours = _label_colours(1, None)
    ncol = 1 + len(OPS)
    cw = 2.4
    fig, axes = plt.subplots(N, ncol, figsize=(cw * ncol + 1.4, cw * N), squeeze=False,
                             gridspec_kw={"hspace": 0.02, "wspace": 0.02})
    for c, t in enumerate(["original"] + [f"{op} {args.radius_mm:g}mm" if op != "sobel"
                                          else "sobel" for op in OPS]):
        axes[0, c].set_title(t, fontsize=9, pad=4)

    for row in range(N):
        img = batch["image"][row].squeeze(0)                    # (D,H,W)
        lbl = batch["label"][row].float()                       # (D,H,W)
        sp = (float(batch["spacing"][row][0]) if "spacing" in batch
              else float(cfg.data.get("crop_spacing_mm", 1.5)))
        r_vox = max(1, mm_to_vox(args.radius_mm, sp))
        z = _best_z(lbl)
        img_sl = img[z].numpy()

        variants = [lbl]
        for op in OPS:
            m = apply_goal_op(lbl[None], op, radius_vox=r_vox, ball=args.ball,
                              min_keep=args.min_keep)[0]
            if op == "sobel":                                   # soft edge -> show the ridge
                m = (m > 0.1).float()
            variants.append(m)

        for c, m in enumerate(variants):
            axes[row, c].imshow(_overlay(img_sl, m[z].numpy(), colours))
        fg0, fgs = int(lbl.round().sum()), [int(v.round().sum()) for v in variants[1:3]]
        axes[row, 0].set_ylabel(
            f"{batch['subjects'][row]}\n{batch['label_names'][row]}\n{sp:.2g} mm/vox"
            f"\nr={r_vox} vox\nfg {fg0} -> dil {fgs[0]} / ero {fgs[1]}",
            fontsize=6.5, rotation=0, labelpad=78, va="center")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.16, right=0.99, top=0.965, bottom=0.005)
    fig.suptitle(
        f"goal-mask ops  |  {cfg.data.get('source', 'totalseg')} {args.split}  |  "
        f"radius={args.radius_mm:g} mm  |  {'ball' if args.ball else 'cube'} SE  |  "
        f"min_keep={args.min_keep:g}",
        fontsize=11, y=0.99)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

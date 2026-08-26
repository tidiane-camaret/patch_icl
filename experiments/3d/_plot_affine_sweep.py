"""
Sweep the task-level affine geometric aug (rotation / scale / shift) on ONE real
exp30 crop, so we can pick gentle magnitudes before wiring them into a config.

Affine is a TASK aug: one shared random draw is applied to the target AND every
context in a task (keeps in-context pose correspondence). Here we replace the
random draw with deterministic values and sweep each component in isolation.

  python experiments/3d/plot_affine_sweep.py experiment=30_colipri_encoder \
      data.crop_spacing_mm=4 data.mask_downsample=occupancy data.mask_occupancy_thr=0.3
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.augmentations import _make_affine_theta, _apply_grid  # noqa: E402
from common import build_dataset  # noqa: E402

_RED = np.array([1.0, 0.2, 0.2])


def best_slice(img, mask):
    """(img_slice, mask_slice) at the axial depth with most foreground."""
    img = img.squeeze(0)
    fg = mask > 0
    z = int(fg.sum(dim=(1, 2)).argmax()) if fg.any() else img.shape[0] // 2
    return img[z].numpy(), mask[z].numpy()


def overlay(img_sl, mask_sl, alpha=0.5):
    lo, hi = img_sl.min(), img_sl.max()
    rgb = np.stack([(img_sl - lo) / (hi - lo + 1e-8)] * 3, -1)
    fg = mask_sl > 0
    rgb[fg] = (1 - alpha) * rgb[fg] + alpha * _RED
    return np.clip(rgb, 0, 1)


def apply_affine(img, mask, *, rz=0.0, scale=1.0, tx=0.0, ty=0.0):
    """Deterministic affine on (1,D,H,W) img + (D,H,W) mask. rz in radians."""
    theta = _make_affine_theta(0.0, 0.0, rz, scale, tx, ty, 0.0)
    grid = F.affine_grid(theta, (1, 1, *img.shape[1:]), align_corners=False)
    im, mk = _apply_grid(img.unsqueeze(0), mask.unsqueeze(0), grid)
    return im.squeeze(0), mk.squeeze(0)


def main():
    overrides = sys.argv[1:]
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train",
                      overrides=overrides + ["augmentations.enabled=false"])

    ds = build_dataset(cfg, "val")
    # prefer a large, clearly-oriented organ so rotation/shift read cleanly.
    # ds.samples is [(subject, class), ...] so we pick the index without loading crops.
    prefer = ["liver", "spleen", "kidney_left", "kidney_right", "stomach", "heart"]
    cand_idx = [i for i, (_, c) in enumerate(ds.samples) if c in prefer][:12]
    best = max(((ds[i] for i in cand_idx)),
               key=lambda it: int((it["label"] > 0).sum()))
    item = best
    idx = "(largest of prefer set)"
    img0, msk0 = item["image"], item["label"]
    name = item["label_name"]
    print(f"using idx={idx}  {name}  fg={(msk0 > 0).sum().item()} vox")

    # --- three sweeps: rotation (deg), scale, shift (normalised half-extent) ---
    rot_deg = [0, 5, 10, 20, 30]
    scales  = [1.00, 0.90, 1.10, 0.70, 1.40]
    shifts  = [0.00, 0.05, 0.10, 0.15, 0.20]
    T = img0.shape[1]

    rows = [
        ("rotation (in-plane)", [(f"{d}°" + (" ★cur" if d == 30 else ""),
                                  dict(rz=math.radians(d))) for d in rot_deg]),
        ("scale",              [(f"{s:.2f}" + (" ★cur" if s in (0.70, 1.40) else ""),
                                  dict(scale=s)) for s in scales]),
        ("shift (H+W)",        [(f"{t:.2f} (~{round(t * T / 2)}vox)",
                                  dict(tx=t, ty=t)) for t in shifts]),
    ]

    ncol = len(rot_deg)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.5 * ncol, 2.7 * len(rows)),
                             squeeze=False)
    for r, (label, cells) in enumerate(rows):
        for c, (title, kw) in enumerate(cells):
            im, mk = apply_affine(img0, msk0, **kw)
            axes[r, c].imshow(overlay(*best_slice(im, mk)))
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(f"task affine sweep (deterministic, single component)  |  {name}",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = ROOT / "results/3d/affine_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

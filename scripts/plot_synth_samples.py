"""
Visualize controlSynth samples for a given synth config (default: hard_diverse).

Builds the dataset through the SAME Hydra/build_dataset path the training scripts use
(experiments/2d/common.build_dataset with data.source=synthetic), so the figure
faithfully reflects `synth=<preset>`. Each row is one in-context task: the target
(image + GT mask) followed by its K context (image, mask) pairs. The mask is drawn
as a red contour over the grayscale image; the target panel is titled with the task's
morphology and a couple of difficulty knobs.

Uses the deterministic `val` split (fixed seeds -> reproducible figure) and a reduced
diversity.num_tasks for a fast bank build -- num_tasks only sets how many distinct base
shapes exist, not the per-subject appearance (that's the build/live difficulty knobs,
which are left exactly as the preset defines them).

Usage:
    python scripts/plot_synth_samples.py
    python scripts/plot_synth_samples.py --preset hard_diverse --rows 10 --num-tasks 400
    python scripts/plot_synth_samples.py --split train          # non-deterministic draws
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "2d"))

# Ensure patch_icl's src is importable before importing common.
from src.datasets.medsegbench import MedSegBenchDataset  # noqa: F401
from common import build_dataset  # noqa: E402


def _overlay(ax, img, mask, title):
    """Grayscale image + translucent red fill on the foreground + a red contour."""
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    if mask is not None and mask.any():
        red = np.zeros((*mask.shape, 4), dtype=np.float32)
        red[..., 0] = 1.0                       # R
        red[..., 3] = (mask > 0.5) * 0.45       # alpha only on the fg
        ax.imshow(red)
        ax.contour(mask, levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="hard_diverse", help="configs/experiment/2d/synth/<preset>.yaml")
    p.add_argument("--rows", type=int, default=8, help="number of task samples to plot")
    p.add_argument("--num-tasks", type=int, default=400, help="bank size (speed vs shape diversity)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--out", default=None, help="output png path")
    args = p.parse_args()

    cfg_dir = str(_ROOT / "configs" / "experiment" / "2d")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="pfn_seg", overrides=[
            "data.source=synthetic",
            f"synth={args.preset}",
            f"synth.diversity.num_tasks={args.num_tasks}",
        ])

    ds = build_dataset(cfg, args.split)
    K = cfg.data.context_size
    n = min(args.rows, len(ds))
    # Even spread across the sample list so we see different tasks/morphologies.
    idxs = np.linspace(0, len(ds) - 1, n).round().astype(int).tolist()

    ncols = 1 + K
    fig, axes = plt.subplots(n, ncols, figsize=(2.1 * ncols, 2.1 * n), squeeze=False)
    for r, idx in enumerate(idxs):
        item = ds[int(idx)]
        img = item["image"][0].numpy()
        lab = item["label"][0].numpy()
        ctx_in = item["context_in"].numpy()    # (K, 1, H, W)
        ctx_out = item["context_out"].numpy()
        meta = item.get("meta", {})
        diff = meta.get("difficulty", {})
        morph = meta.get("morphology", "?")
        rsize = diff.get("region_size"); amb = diff.get("task_ambiguity")
        sub = f"  rsz={rsize:.2f} amb={amb:.2f}" if isinstance(rsize, float) and isinstance(amb, float) else ""
        _overlay(axes[r][0], img, lab, f"target [{morph}]{sub}")
        for k in range(K):
            _overlay(axes[r][1 + k], ctx_in[k, 0], ctx_out[k, 0], f"context {k + 1}")

    fig.suptitle(f"controlSynth  synth={args.preset}  split={args.split}  "
                 f"(num_tasks={args.num_tasks}, K={K}, size={cfg.data.image_size})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out = Path(args.out) if args.out else _ROOT / "results" / "controlsynth" / f"{args.preset}_samples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}  ({n} samples, {K} contexts each)")


if __name__ == "__main__":
    main()

"""
Visualise synth_gmm shape-mode cohorts (procedural blob/splatter/disk/cylinder,
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md).

`data.gmm.p_shape` has no Hydra config wiring yet (deliberately deferred — see
docs/logs.md 2026-09-14), so plot_dataset_items.py can't reach this mode through
build_dataset. This script constructs SynthGmmProvider(cascade=True, p_shape=1.0)
directly against the real gmm_bank and renders each cohort's target + K context
NativeCrops, reusing plot_dataset_items.py's slice/overlay helpers.

Usage
-----
  python experiments/3d/plot_shape_items.py
  python experiments/3d/plot_shape_items.py --n_samples 8 --context_size 3
  python experiments/3d/plot_shape_items.py --family blob
  python experiments/3d/plot_shape_items.py --shape_between_ratio 0 --size_between_ratio 0 \
      --position_between_ratio 0   # near-identical cohort (consistency knobs at 0)
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling `plot_dataset_items`

from plot_dataset_items import _BINARY_COLOUR, _best_slice, _overlay  # noqa: E402
from src.providers.synth_gmm import SynthGmmProvider  # noqa: E402
from src.shapes3d.spec import ShapeCohortSpec  # noqa: E402
from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset  # noqa: E402

DEFAULT_BANK = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                "ANALYSIS_20251122/data/gmm_bank")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default=DEFAULT_BANK)
    p.add_argument("--n_samples", type=int, default=8, help="number of cohorts (rows)")
    p.add_argument("--context_size", type=int, default=2)
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--crop_spacing_mm", type=float, default=1.5)
    p.add_argument("--gpu_realize_max_native", type=int, default=128,
                   help="cap on the native crop side before painting (perf; see docs/logs.md)")
    p.add_argument("--family", default=None, choices=["blob", "splatter", "disk", "cylinder"],
                   help="force a single family for every row (default: mix of all 4)")
    p.add_argument("--shape_between_ratio", type=float, default=0.3)
    p.add_argument("--size_between_ratio", type=float, default=0.3)
    p.add_argument("--position_between_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/3d/shape_items.png")
    args = p.parse_args()

    T = args.image_size
    ds = SynthGmmMaisiDataset(
        bank_dir=args.bank, image_size=(T, T, T), context_size=args.context_size,
        crop_spacing_mm=args.crop_spacing_mm, classes=None, length=args.n_samples,
        var_max=5.0, background_mode="zero", class_balanced=True,
        gpu_realize=False, gpu_realize_max_native=args.gpu_realize_max_native,
    )
    family_weights = ({args.family: 1.0} if args.family
                       else {"blob": 1.0, "splatter": 1.0, "disk": 1.0, "cylinder": 1.0})
    shape_spec = ShapeCohortSpec(
        family_weights=family_weights,
        shape_between_ratio=args.shape_between_ratio,
        size_between_ratio=args.size_between_ratio,
        position_between_ratio=args.position_between_ratio,
    )
    provider = SynthGmmProvider(ds, cascade=True, p_shape=1.0, shape_spec=shape_spec)

    K = args.context_size
    N = args.n_samples
    col_w = 2.4
    fig, axes = plt.subplots(N, 1 + K, figsize=(col_w * (1 + K), col_w * N),
                             squeeze=False, gridspec_kw={"hspace": 0.02, "wspace": 0.02})
    vol_names = ["target"] + [f"ctx {k + 1}" for k in range(K)]
    for v, name in enumerate(vol_names):
        axes[0, v].set_title(name, fontsize=9, pad=4)

    for row in range(N):
        rng = random.Random(args.seed + row)
        task = provider.assemble_task(rng, args.crop_spacing_mm)
        for v, nc in enumerate(task["native_crop"]):
            img = nc.image.float().unsqueeze(0)     # (D,H,W) -> (1,D,H,W)
            mask = nc.label_frac.float()             # (D,H,W) in [0,1]
            img_sl, mask_sl = _best_slice(img, mask)
            axes[row, v].imshow(_overlay(img_sl, mask_sl, {1: _BINARY_COLOUR}))
        axes[row, 0].set_ylabel(f"{task['label_name']}\n{task['subject'].split('|')[0]}",
                                fontsize=7, rotation=0, labelpad=90, va="center")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"synth_gmm shape-mode cohorts  |  K={K}  |  family={args.family or 'mix'}  |  "
        f"between_ratio: shape={args.shape_between_ratio} size={args.size_between_ratio} "
        f"pos={args.position_between_ratio}",
        fontsize=10, y=1.01)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

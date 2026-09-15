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

  # Cascade mode: show the TARGET re-cropped across cascade levels (columns = levels)
  # instead of target+context (columns = cohort members). Level 0 comes from
  # assemble_task; each later level re-derives the SAME subject via load_native_crop
  # with a "perfect predicted center" (the host's own true centroid stands in for what
  # a well-trained cascade predictor would output) at a narrower crop_spacing_mm --
  # visualizing the docs/logs.md 2026-09-15 world-space fix: the shape should look like
  # the SAME physical object zoomed in, not a different one at each level.
  python experiments/3d/plot_shape_items.py --cascade_spacings 6,3,1.2
  python experiments/3d/plot_shape_items.py --cascade_spacings 6,3,1.2 --family cylinder
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
from src.incontext_dataset_v2 import LoadRequest  # noqa: E402
from src.providers.synth_gmm import SynthGmmProvider  # noqa: E402
from src.shapes3d.spec import ShapeCohortSpec  # noqa: E402
from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset  # noqa: E402

DEFAULT_BANK = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                "ANALYSIS_20251122/data/gmm_bank")


def _host_centroid(provider, task):
    """The host's own true centroid (e["cents"][host_cls_id]) for a shape-mode task's
    TARGET member -- a "perfect predictor" stand-in for the center a well-trained
    cascade model would predict from level 0's output. Level 0 itself never uses this
    (its crop is centered via center_mode="com" internally); only later levels'
    load_native_crop calls need an explicit center."""
    filename, _, _, host_str = task["subject"].split("|")
    host_cls_id = int(host_str[len("host"):])
    entry = provider._entry_by_file[filename]
    return tuple(int(round(v)) for v in entry["cents"][host_cls_id][:3])


def _cascade_crops(provider, task, spacings, seed):
    """[NativeCrop, ...] for the TARGET member at each crop_spacing_mm in `spacings`.
    Index 0 is `task`'s own level-0 build; each later index re-derives the SAME subject
    via load_native_crop with the host's true centroid as an explicit (perfect) center."""
    center = _host_centroid(provider, task)
    crops = [task["native_crop"][0]]
    for level, spacing in enumerate(spacings[1:], start=1):
        req = LoadRequest(rng=random.Random(seed * 1000 + level), crop_spacing_mm=spacing,
                          center=center, center_mode="com")
        crops.append(provider.load_native_crop(task["subject"], task["label_name"], req))
    return crops


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
    p.add_argument("--cascade_spacings", default=None,
                   help="comma-separated crop_spacing_mm schedule, e.g. '6,3,1.2' -- "
                        "switches to cascade mode (see module docstring)")
    args = p.parse_args()
    cascade_spacings = ([float(s) for s in args.cascade_spacings.split(",")]
                        if args.cascade_spacings else None)

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

    N = args.n_samples
    col_w = 2.4
    if cascade_spacings:
        n_cols = len(cascade_spacings)
        col_names = [f"{s:g}mm" for s in cascade_spacings]
    else:
        n_cols = 1 + args.context_size
        col_names = ["target"] + [f"ctx {k + 1}" for k in range(args.context_size)]
    fig, axes = plt.subplots(N, n_cols, figsize=(col_w * n_cols, col_w * N),
                             squeeze=False, gridspec_kw={"hspace": 0.02, "wspace": 0.02})
    for v, name in enumerate(col_names):
        axes[0, v].set_title(name, fontsize=9, pad=4)

    for row in range(N):
        rng = random.Random(args.seed + row)
        task = provider.assemble_task(rng, args.crop_spacing_mm if not cascade_spacings
                                      else cascade_spacings[0])
        crops = (_cascade_crops(provider, task, cascade_spacings, args.seed + row)
                if cascade_spacings else task["native_crop"])
        for v, nc in enumerate(crops):
            img = nc.image.float().unsqueeze(0)     # (D,H,W) -> (1,D,H,W)
            mask = nc.label_frac.float()             # (D,H,W) in [0,1]
            img_sl, mask_sl = _best_slice(img, mask)
            axes[row, v].imshow(_overlay(img_sl, mask_sl, {1: _BINARY_COLOUR}))
        axes[row, 0].set_ylabel(f"{task['label_name']}\n{task['subject'].split('|')[0]}",
                                fontsize=7, rotation=0, labelpad=90, va="center")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    if cascade_spacings:
        subtitle = (f"cascade {'->'.join(f'{s:g}' for s in cascade_spacings)}mm  |  "
                   f"family={args.family or 'mix'}  |  target only "
                   f"(level>=1 centered on the host's true centroid -- a perfect-"
                   f"predictor stand-in)")
    else:
        subtitle = (f"K={args.context_size}  |  family={args.family or 'mix'}  |  "
                   f"between_ratio: shape={args.shape_between_ratio} "
                   f"size={args.size_between_ratio} pos={args.position_between_ratio}")
    fig.suptitle(f"synth_gmm shape-mode cohorts  |  {subtitle}", fontsize=10, y=1.01)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

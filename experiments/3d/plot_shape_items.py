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

  # --preset 135b: match the 135b headline checkpoint's own synth_gmm shape-mode config
  # (configs/experiment/3d/experiment/135_..._intensitycalib.yaml) -- var_max=80,
  # paint_mask_aligned, mu/sd_group_ids=merged, sd_between_ratio=ct_mri, texture
  # n_octaves=4, p_heterogeneity=0.3, shape.host_anchored=true -- plus per-row
  # crop_spacing_mm sampled log-uniformly in train_spacing_range=[3,6]mm (same
  # distribution SpacingBatchSampler draws one-per-BATCH from at train time; here it's
  # one draw per ROW so every row's own spacing is visible).
  python experiments/3d/plot_shape_items.py --preset 135b --n_samples 20 --seed 0
"""

import argparse
import math
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
from src.providers.synth_gmm import HeterogeneitySpec, SynthGmmProvider, TextureSpec  # noqa: E402
from src.shapes3d.spec import ShapeCohortSpec  # noqa: E402
from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset  # noqa: E402

# configs/experiment/3d/experiment/135_cascade_register_varspacing_synth03_texture_pool2stage_
# ctmri_intensitycalib.yaml's own data.gmm block -- kept in sync by hand, not auto-derived.
PRESET_135B = dict(
    var_max=80.0, background_mode="zero", paint_mask_aligned=True,
    mu_group_ids="merged", sd_between_ratio="ct_mri", sd_group_ids="merged",
    texture_n_octaves=4, p_heterogeneity=0.3, host_anchored=True,
    train_spacing_range="3,6",
)

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
    p.add_argument("--family", default=None,
                   choices=["blob", "splatter", "disk", "cylinder", "scatter_field",
                            "vessel", "torus", "mix7"],
                   help="force a single family for every row; 'mix7' = equal mix of all "
                        "7 families (default: mix of the original 4, matching 135b)")
    p.add_argument("--shape_between_ratio", type=float, default=0.3)
    p.add_argument("--size_between_ratio", type=float, default=0.3)
    p.add_argument("--position_between_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/3d/shape_items.png")
    p.add_argument("--cascade_spacings", default=None,
                   help="comma-separated crop_spacing_mm schedule, e.g. '6,3,1.2' -- "
                        "switches to cascade mode (see module docstring); mutually "
                        "exclusive with --train_spacing_range, same as real training")
    p.add_argument("--train_spacing_range", default=None,
                   help="'lo,hi'mm -- sample crop_spacing_mm log-uniformly PER ROW in "
                        "[lo,hi] instead of using a fixed --crop_spacing_mm (matches "
                        "SpacingBatchSampler's per-batch draw, common.py:861)")
    p.add_argument("--var_max", type=float, default=5.0)
    p.add_argument("--background_mode", default="zero")
    p.add_argument("--paint_mask_aligned", action="store_true")
    p.add_argument("--mu_group_ids", default=None)
    p.add_argument("--sd_between_ratio", default=None)
    p.add_argument("--sd_group_ids", default=None)
    p.add_argument("--texture_n_octaves", type=int, default=1)
    p.add_argument("--p_heterogeneity", type=float, default=0.0)
    p.add_argument("--host_anchored", action="store_true")
    p.add_argument("--preset", default=None, choices=["135b"],
                   help="shortcut: set var_max/background_mode/paint_mask_aligned/"
                        "mu_group_ids/sd_between_ratio/sd_group_ids/texture_n_octaves/"
                        "p_heterogeneity/host_anchored/train_spacing_range to match the "
                        "135b headline checkpoint's own config (see PRESET_135B above); "
                        "explicit flags still override individual preset values")
    args = p.parse_args()
    if args.preset:
        preset = {"135b": PRESET_135B}[args.preset]
        for k, v in preset.items():
            if p.get_default(k) == getattr(args, k):   # not explicitly overridden on the CLI
                setattr(args, k, v)
    cascade_spacings = ([float(s) for s in args.cascade_spacings.split(",")]
                        if args.cascade_spacings else None)
    spacing_range = ([float(s) for s in args.train_spacing_range.split(",")]
                     if args.train_spacing_range else None)
    if cascade_spacings and spacing_range:
        raise ValueError("--cascade_spacings and --train_spacing_range are mutually exclusive")

    # "merged" is a preset NAME (matches configs/.../data.gmm.mu_group_ids: merged) that
    # common.py resolves to real MAISI-id groups before ever reaching the dataset --
    # SynthGmmMaisiDataset itself only accepts raw id tuples, not preset names (unlike
    # sd_between_ratio, whose string presets ARE resolved inside the dataset already).
    mu_group_ids, mu_group_rho = args.mu_group_ids, None
    sd_group_ids, sd_group_rho = args.sd_group_ids, None
    if args.mu_group_ids == "merged" or args.sd_group_ids == "merged":
        from src.gpu_gmm_intensity import (MERGED_GROUP_MAISI_IDS, MERGED_GROUP_RHO,
                                           VAR_GROUP_MAISI_IDS, VAR_GROUP_RHO)
        if args.mu_group_ids == "merged":
            mu_group_ids, mu_group_rho = MERGED_GROUP_MAISI_IDS, MERGED_GROUP_RHO
        if args.sd_group_ids == "merged":
            sd_group_ids, sd_group_rho = VAR_GROUP_MAISI_IDS, VAR_GROUP_RHO

    T = args.image_size
    ds = SynthGmmMaisiDataset(
        bank_dir=args.bank, image_size=(T, T, T), context_size=args.context_size,
        crop_spacing_mm=args.crop_spacing_mm, classes=None, length=args.n_samples,
        var_max=args.var_max, background_mode=args.background_mode, class_balanced=True,
        gpu_realize=False, gpu_realize_max_native=args.gpu_realize_max_native,
        paint_mask_aligned=args.paint_mask_aligned,
        mu_group_ids=mu_group_ids, mu_group_rho=mu_group_rho,
        sd_between_ratio=args.sd_between_ratio,
        sd_group_ids=sd_group_ids, sd_group_rho=sd_group_rho,
    )
    if args.family == "mix7":
        family_weights = {f: 1.0 for f in
                          ("blob", "splatter", "disk", "cylinder",
                           "scatter_field", "vessel", "torus")}
    elif args.family:
        family_weights = {args.family: 1.0}
    else:
        family_weights = {"blob": 1.0, "splatter": 1.0, "disk": 1.0, "cylinder": 1.0}
    shape_spec = ShapeCohortSpec(
        family_weights=family_weights,
        shape_between_ratio=args.shape_between_ratio,
        size_between_ratio=args.size_between_ratio,
        position_between_ratio=args.position_between_ratio,
        host_anchored=args.host_anchored,
    )
    texture_spec = TextureSpec(n_octaves=args.texture_n_octaves)
    provider = SynthGmmProvider(
        ds, cascade=True, p_shape=1.0, shape_spec=shape_spec, texture_spec=texture_spec,
        p_heterogeneity=args.p_heterogeneity,
        heterogeneity_spec=HeterogeneitySpec() if args.p_heterogeneity > 0 else None,
    )

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
        if spacing_range:
            lo, hi = spacing_range
            row_spacing = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            row_spacing = args.crop_spacing_mm if not cascade_spacings else cascade_spacings[0]
        task = provider.assemble_task(rng, row_spacing)
        crops = (_cascade_crops(provider, task, cascade_spacings, args.seed + row)
                if cascade_spacings else task["native_crop"])
        for v, nc in enumerate(crops):
            img = nc.image.float().unsqueeze(0)     # (D,H,W) -> (1,D,H,W)
            mask = nc.label_frac.float()             # (D,H,W) in [0,1]
            img_sl, mask_sl = _best_slice(img, mask)
            axes[row, v].imshow(_overlay(img_sl, mask_sl, {1: _BINARY_COLOUR}))
        spacing_tag = f"  {row_spacing:.2f}mm" if spacing_range else ""
        axes[row, 0].set_ylabel(
            f"{task['label_name']}\n{task['subject'].split('|')[0]}{spacing_tag}",
            fontsize=7, rotation=0, labelpad=90, va="center")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    if cascade_spacings:
        subtitle = (f"cascade {'->'.join(f'{s:g}' for s in cascade_spacings)}mm  |  "
                   f"family={args.family or 'mix'}  |  target only "
                   f"(level>=1 centered on the host's true centroid -- a perfect-"
                   f"predictor stand-in)")
    else:
        spacing_desc = (f"spacing~logU{tuple(spacing_range)}mm" if spacing_range
                        else f"spacing={args.crop_spacing_mm}mm")
        subtitle = (f"K={args.context_size}  |  family={args.family or 'mix'}  |  "
                   f"{spacing_desc}  |  between_ratio: shape={args.shape_between_ratio} "
                   f"size={args.size_between_ratio} pos={args.position_between_ratio}"
                   f"{'  |  preset=' + args.preset if args.preset else ''}")
    fig.suptitle(f"synth_gmm shape-mode cohorts  |  {subtitle}", fontsize=10, y=1.01)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

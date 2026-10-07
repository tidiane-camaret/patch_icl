"""
Thesis figure: synthetic-bank canvas vs. real-host canvas, same shape family
(sections/methodology_synth.tex, "Canvas." paragraph).

Both rows paint the SAME procedural shape family (scatter_field, the family actually
routed through real_host_families in the real-host training configs, e.g.
configs/experiment/3d/experiment/163_scatter_field_real_mri_host.yaml) as a target +
1 context task:

  - synthetic-bank row: SynthGmmProvider over the MAISI mask bank -- anatomy AND
    appearance are synthetic (GMM-painted from TotalSegmentator label statistics).
  - real-host row: RealHostShapeProvider -- the shape is stamped onto a REAL MRI
    TotalSeg brain scan at a contrast computed from that subject's own local tissue
    stats; only the shape itself is synthetic.

Image and mask are plotted as separate panels (not overlaid).

Usage
-----
  python experiments/3d/plot_synth_canvas_compare.py
  python experiments/3d/plot_synth_canvas_compare.py --family blob --n_rows 2
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

import numpy as np
import torch
import torch.nn.functional as F

from plot_dataset_items import _best_slice  # noqa: E402
from data.totalseg_classes import MRI_ALL_CLASSES  # noqa: E402
from src.providers.real_host_shape import RealHostShapeProvider  # noqa: E402
from src.providers.synth_gmm import HeterogeneitySpec, SynthGmmProvider, TextureSpec  # noqa: E402
from src.providers.totalseg import TotalSegProvider  # noqa: E402
from src.shapes3d.spec import ShapeCohortSpec  # noqa: E402
from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset  # noqa: E402

DEFAULT_BANK = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                "ANALYSIS_20251122/data/gmm_bank")
DEFAULT_MRI = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/totalsegmri")

# matches configs/experiment/3d/experiment/163_scatter_field_real_mri_host.yaml's data.gmm
PAINT_KW = dict(var_max=80.0, background_mode="zero", paint_mask_aligned=True)

DISPLAY_SIZE = 160   # native crops are anisotropic (unresampled) -- resize every 2D slice
                      # to this fixed pixel grid so bank/host panels line up in one figure


def _resize_slice(arr: np.ndarray, mode: str) -> np.ndarray:
    t = torch.from_numpy(arr).float()[None, None]
    kw = {} if mode == "nearest" else {"align_corners": False}
    t = F.interpolate(t, size=(DISPLAY_SIZE, DISPLAY_SIZE), mode=mode, **kw)
    return t[0, 0].numpy()


def _display_slice(img, mask):
    img_sl, mask_sl = _best_slice(img, mask)
    return _resize_slice(img_sl, "bilinear"), _resize_slice(mask_sl, "nearest")


def _bank_row(args, family, rng_seed):
    from src.gpu_gmm_intensity import MERGED_GROUP_MAISI_IDS, MERGED_GROUP_RHO, VAR_GROUP_MAISI_IDS, VAR_GROUP_RHO
    T = args.image_size
    ds = SynthGmmMaisiDataset(
        bank_dir=args.bank, image_size=(T, T, T), context_size=args.context_size,
        crop_spacing_mm=args.crop_spacing_mm, classes=None, length=1, class_balanced=True,
        gpu_realize=False, gpu_realize_max_native=128,
        mu_group_ids=MERGED_GROUP_MAISI_IDS, mu_group_rho=MERGED_GROUP_RHO,
        sd_between_ratio="ct_mri",
        sd_group_ids=VAR_GROUP_MAISI_IDS, sd_group_rho=VAR_GROUP_RHO,
        **PAINT_KW,
    )
    shape_spec = ShapeCohortSpec(family_weights={family: 1.0}, host_anchored=True)
    texture_spec = TextureSpec(n_octaves=4)
    provider = SynthGmmProvider(
        ds, cascade=True, p_shape=1.0, shape_spec=shape_spec, texture_spec=texture_spec,
        p_heterogeneity=0.3, heterogeneity_spec=HeterogeneitySpec(),
    )
    rng = random.Random(rng_seed)
    task = provider.assemble_task(rng, args.crop_spacing_mm)
    return task["native_crop"]


def _host_row(args, family, rng_seed):
    mri_provider = TotalSegProvider(
        root=args.mri_root, classes=MRI_ALL_CLASSES, image_size=(args.image_size,) * 3,
        split=None, modality="mri")
    shape_spec = ShapeCohortSpec(family_weights={family: 1.0}, host_anchored=False)
    texture_spec = TextureSpec(n_octaves=4)
    provider = RealHostShapeProvider(
        mri_provider, shape_spec, host_classes="brain", context_size=args.context_size,
        texture_spec=texture_spec, crop_spacing_mm=1.0)
    rng = random.Random(rng_seed)
    task = provider.assemble_task(rng, args.crop_spacing_mm)
    return task["native_crop"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default=DEFAULT_BANK)
    p.add_argument("--mri_root", default=DEFAULT_MRI)
    p.add_argument("--family", default="scatter_field",
                   choices=["blob", "splatter", "disk", "cylinder", "scatter_field",
                            "vessel", "torus"])
    p.add_argument("--n_rows", type=int, default=3, help="example cohorts per canvas")
    p.add_argument("--context_size", type=int, default=1)
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--crop_spacing_mm", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/3d/synth_canvas_compare.png")
    args = p.parse_args()

    # 2 sub-columns (image, mask) per volume (target + each context)
    n_vols = 1 + args.context_size
    n_cols = 2 * n_vols
    vol_names = ["target"] + [f"ctx {k + 1}" for k in range(args.context_size)]
    col_names = [t for name in vol_names for t in (f"{name}\nimage", f"{name}\nmask")]
    row_labels = (["synthetic-bank canvas"] + [""] * (args.n_rows - 1)
                  + ["real-host canvas"] + [""] * (args.n_rows - 1))
    n_total_rows = 2 * args.n_rows

    col_w = 2.0
    fig, axes = plt.subplots(n_total_rows, n_cols, figsize=(col_w * n_cols, col_w * n_total_rows),
                             squeeze=False, gridspec_kw={"hspace": 0.05, "wspace": 0.04})
    for v, name in enumerate(col_names):
        axes[0, v].set_title(name, fontsize=8, pad=4)

    def _plot_volumes(row, crops):
        for k, nc in enumerate(crops):
            img = nc.image.float().unsqueeze(0)
            mask = nc.label_frac.float()
            img_sl, mask_sl = _display_slice(img, mask)
            axes[row, 2 * k].imshow(img_sl, cmap="gray")
            axes[row, 2 * k + 1].imshow(mask_sl, cmap="hot", vmin=0, vmax=1)

    for i in range(args.n_rows):
        _plot_volumes(i, _bank_row(args, args.family, args.seed + i))

    for i in range(args.n_rows):
        _plot_volumes(args.n_rows + i, _host_row(args, args.family, 1000 + args.seed + i))

    for row in range(n_total_rows):
        if row_labels[row]:
            axes[row, 0].set_ylabel(row_labels[row], fontsize=10, rotation=90, labelpad=6,
                                    va="center")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Shape family: {args.family}  |  K={args.context_size}", fontsize=10, y=1.0)
    fig.tight_layout(h_pad=0.15, w_pad=0.1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

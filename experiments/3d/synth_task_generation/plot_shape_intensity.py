"""Does a synthetic shape's painted intensity (mu[shape_id]/sd[shape_id]) correlate with the
real host organ it's stamped inside? Code says no: shape pseudo-class ids (201-204) are drawn
via the SAME independent mu[1:]/sd[1:] array as every other class id, and are NOT members of
CT_GROUP_MAISI_IDS/MERGED_GROUP_MAISI_IDS/VAR_GROUP_MAISI_IDS (those only cover real anatomy
ids) -- so even with mu_group_ids/sd_group_ids calibration ON, the shape's intensity is an
arbitrary independent draw regardless of the host tissue (src/providers/synth_gmm.py:306-307,
mu_e[shape_id] = mu[shape_id] + ...). This script draws several shape-mode cohorts through the
REAL cascade pipeline (SynthGmmProvider(cascade=True) -> _build_nc -> gpu_realize_crop, same
path 130-135's p_synth>0 training actually uses) and prints host-class vs shape-class mu/sd
side by side, plus a slice plot with the shape mask contoured.

  python experiments/3d/synth_task_generation/plot_shape_intensity.py --n 4
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/home/dpxuser/dev/patch_icl")
sys.path.insert(0, str(ROOT))

from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset  # noqa: E402
from src.providers.synth_gmm import SynthGmmProvider  # noqa: E402
from src.shapes3d.spec import ShapeCohortSpec  # noqa: E402
from src.gpu_realize_crop import _realize_member  # noqa: E402
from src.totalseg_dataset import CtNormSpec  # noqa: E402
from data.maisi_classes import MAISI_IDX_TO_CLASS  # noqa: E402

BANK = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/gmm_bank"
OUT = ROOT / "results" / "synth_task_gen"
SYNTH_NORM = CtNormSpec(clip_lo=-3.0, clip_hi=3.0, mean=0.0, std=1.0)


def _norm01(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--crop-mm", type=float, default=4.24)
    ap.add_argument("--anchored", action="store_true", help="ShapeCohortSpec.host_anchored")
    args = ap.parse_args()

    ds = SynthGmmMaisiDataset(bank_dir=BANK, image_size=(128, 128, 128), context_size=1,
                              crop_spacing_mm=args.crop_mm, length=1000, var_max=80.0,
                              background_mode="zero", class_balanced=True, gpu_realize=False,
                              paint_mask_aligned=True)  # needed for NativeCrop.target_mu/sd
    prov = SynthGmmProvider(ds, cascade=True, p_shape=1.0,
                            shape_spec=ShapeCohortSpec(host_anchored=args.anchored))

    fig, axes = plt.subplots(args.n, 2, figsize=(8, 3.6 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for row in range(args.n):
        rng = random.Random(2000 + row)
        item = prov.assemble_task(rng, args.crop_mm)
        nc_target = item["native_crop"][0]
        host_cls_id = int(item["subject"].split("|host")[1].split("|")[0])
        shape_id = nc_target.class_idx
        mu_shape, sd_shape = nc_target.target_mu, nc_target.target_sd
        # Re-derive the FULL mu/sd draw the same way _draw_gmm does (same seed), just to read
        # off the host class's own value for comparison -- target_mu/target_sd on the NativeCrop
        # is already the shape's own painted value (paint_mask_aligned reuses it verbatim).
        gmm_seed = int(item["subject"].split("|")[1])
        mu, sd = prov._draw_gmm(gmm_seed)
        mu_host, sd_host = float(mu[host_cls_id]), float(sd[host_cls_id])

        img_t, mask_t = _realize_member(nc_target, T=128, mask_downsample="occupancy",
                                        occ_thr=0.1, ct_spec=SYNTH_NORM,
                                        device=torch.device("cpu"))
        img = img_t[0].numpy()
        mask = mask_t.numpy()
        per_z = mask.sum(axis=(0, 1))
        z = int(np.argmax(per_z)) if per_z.max() > 0 else img.shape[-1] // 2

        host_name = MAISI_IDX_TO_CLASS.get(host_cls_id, str(host_cls_id))
        shape_name = MAISI_IDX_TO_CLASS.get(shape_id, str(shape_id))
        print(f"row {row}: host={host_name!r} (mu={mu_host:.1f}, sd={sd_host:.1f})  "
              f"shape={shape_name!r} (mu={mu_shape:.1f}, sd={sd_shape:.1f})  "
              f"|mu diff|={abs(mu_shape - mu_host):.1f}")

        axes[row, 0].imshow(_norm01(img[:, :, z]), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].contour(mask[:, :, z], levels=[0.5], colors="red", linewidths=1.2)
        axes[row, 0].set_title(f"host={host_name}\nshape={shape_name}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].bar(["host mu", "shape mu"], [mu_host, mu_shape], color=["#4C72B0", "#C44E52"])
        axes[row, 1].bar(["host sd", "shape sd"], [sd_host, sd_shape], color=["#4C72B0", "#C44E52"],
                        alpha=0.5, bottom=0)
        axes[row, 1].set_title("host vs shape mu/sd (arbitrary domain-randomized units)", fontsize=8)

    mode = "host_anchored=True" if args.anchored else "host_anchored=False (independent, default)"
    fig.suptitle(f"Shape intensity vs. host-organ intensity ({mode})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = OUT / f"shape_vs_host_intensity_{'anchored' if args.anchored else 'independent'}.png"
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

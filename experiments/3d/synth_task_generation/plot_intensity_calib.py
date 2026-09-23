"""Visual sanity check for the 2026-09-23 intensity calibrations (mu_group_ids, sd_between_
ratio, sd_group_ids): paint the SAME cohort draws twice -- once fully uncalibrated (today's old
default), once with all three calibrations on (135's recipe) -- and plot a central slice of
each, target + K context, so cross-class shading correlation (e.g. bilateral organ pairs, bone)
is visible by eye, not just in the measured rho numbers.

  python experiments/3d/synth_task_generation/plot_intensity_calib.py --n 4
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/dpxuser/dev/patch_icl")
sys.path.insert(0, str(ROOT))

from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset  # noqa: E402
from src.gpu_gmm_intensity import (MERGED_GROUP_MAISI_IDS, MERGED_GROUP_RHO,  # noqa: E402
                                   CT_MRI_BETWEEN_WITHIN_GROUPS, CT_MRI_BETWEEN_WITHIN_DEFAULT,
                                   VAR_GROUP_MAISI_IDS, VAR_GROUP_RHO)

BANK = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/gmm_bank"
OUT = ROOT / "results" / "synth_task_gen"


def build(calibrated: bool, seed: int):
    kw = dict(bank_dir=BANK, image_size=(128, 128, 128), context_size=1, crop_spacing_mm=4.24,
              length=1000, var_max=80.0, background_mode="zero", class_balanced=True,
              gpu_realize=False, eval_seed=seed)
    if calibrated:
        kw.update(mu_group_ids=list(MERGED_GROUP_MAISI_IDS), mu_group_rho=list(MERGED_GROUP_RHO),
                  sd_between_ratio=list(_build_ratio_table()), sd_group_ids=list(VAR_GROUP_MAISI_IDS),
                  sd_group_rho=list(VAR_GROUP_RHO))
    return SynthGmmMaisiDataset(**kw)


def _build_ratio_table():
    from src.gpu_gmm_intensity import build_between_ratio_table
    return build_between_ratio_table(200, groups=CT_MRI_BETWEEN_WITHIN_GROUPS,
                                     default=CT_MRI_BETWEEN_WITHIN_DEFAULT)


def _norm01(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of cohort draws to plot")
    ap.add_argument("--crop-mm", type=float, default=4.24, help="crop_spacing_mm (mm/voxel)")
    ap.add_argument("--out", default=None, help="output PNG path (default derived from --crop-mm)")
    args = ap.parse_args()

    ds_off = build(calibrated=False, seed=0)
    ds_on = build(calibrated=True, seed=0)

    fig, axes = plt.subplots(args.n, 4, figsize=(14, 3.4 * args.n))
    if args.n == 1:
        axes = axes[None, :]

    for row in range(args.n):
        # rng (cohort/crop selection) is a stdlib random.Random (matches the v2 engine
        # convention, e.g. Random(hash((eval_seed, idx)))); nrng (GMM draw + paint noise) is a
        # numpy Generator (matches sample_grouped_uniform's own usage). SAME seed for both
        # builds -> SAME cohort/crop, only the paint calibration differs.
        rng = random.Random(1000 + row)
        nrng = np.random.default_rng(1000 + row)
        item_off = ds_off.assemble(rng, nrng, crop_mm=args.crop_mm)
        rng2 = random.Random(1000 + row)
        nrng2 = np.random.default_rng(1000 + row)
        item_on = ds_on.assemble(rng2, nrng2, crop_mm=args.crop_mm)

        # Pick the slice with the MOST target-mask voxels, not a fixed mid-slice -- the crop is
        # centered on a jittered point inside the class's mask (_resolve_center), so the object
        # is not necessarily anywhere near the volumetric center along z. A fixed z=64 slice
        # frequently misses small/off-center objects (e.g. humerus/clavicula) entirely, showing
        # an unrelated torso slice under a title that names the (correctly-wired, just
        # off-slice) target class.
        mask_np = item_off["label"].numpy()
        per_z = mask_np.sum(axis=(0, 1))
        z = int(np.argmax(per_z)) if per_z.max() > 0 else mask_np.shape[-1] // 2
        tgt_off = item_off["image"][0, :, :, z].numpy()
        tgt_on = item_on["image"][0, :, :, z].numpy()
        ctx_off = item_off["context_in"][0, 0, :, :, z].numpy()
        ctx_on = item_on["context_in"][0, 0, :, :, z].numpy()
        mask = item_off["label"][:, :, z].numpy()

        for col, (img, title) in enumerate([
            (tgt_off, f"target UNCALIBRATED\n{item_off['label_name']}"),
            (tgt_on, f"target CALIBRATED\n{item_on['label_name']}"),
            (ctx_off, "context[0] UNCALIBRATED"),
            (ctx_on, "context[0] CALIBRATED"),
        ]):
            ax = axes[row, col]
            ax.imshow(_norm01(img), cmap="gray", vmin=0, vmax=1)
            if col in (0, 1):
                ax.contour(mask, levels=[0.5], colors="red", linewidths=1.0)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle(f"synth_gmm intensity calibration @ crop_spacing_mm={args.crop_mm:g}: "
                 "uncalibrated vs mu_group_ids=merged + sd_between_ratio=ct_mri + "
                 "sd_group_ids=merged\n(same cohort/crop each row, red contour = target-class "
                 "mask, slice = argmax mask overlap)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(args.out) if args.out else OUT / f"intensity_calib_check_{args.crop_mm:g}mm.png"
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""General CLI for nifti in-context cascade inference (patchset3d spacing_cascade).

Runnable by anyone who can activate the shared `patchset` conda env — no repo checkout
needed (the code is vendored into the env; see docs). Predicts a target organ mask from
one or more context (image, binary-mask) nifti pairs for the same organ, via the
4mm->1.5mm coarse->fine cascade. GT-free for the target; pass --gt to also report Dice.

    patchset-infer \
        --target tgt_ct.nii.gz \
        --context ctx_ct.nii.gz ctx_mask.nii.gz \
        --checkpoint /nfs/.../best.pt \
        --out pred.nii.gz [--gt gt_mask.nii.gz]

Repeat --context for K>1 contexts. Extra Hydra overrides pass through via --override.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo / vendored-bundle root (…/experiments/3d -> root)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
# eval.yaml's hydra.searchpath is file://${oc.env:PWD}/configs, so the cluster/dataset/
# augmentations groups only resolve when PWD points at the bundle root. Pin it here so the
# CLI works from any working directory (and from the vendored env copy).
os.environ["PWD"] = str(ROOT)

from hydra import compose, initialize_config_dir  # noqa: E402
from infer_nifti import predict_nifti  # noqa: E402


def _build_cfg(args):
    """Compose the same cfg surface as experiments/3d/eval.py, with inference overrides."""
    cfg_dir = str(ROOT / "configs" / "experiment" / "3d")
    overrides = [
        f"eval.model={args.model}",
        f"eval.checkpoint={args.checkpoint}",
        f"eval.feat_norm={args.feat_norm}",
        f"eval.spacing_sweep=[{args.crop_spacings}]",
        f"data.mask_downsample={args.mask_downsample}",
        f"data.mask_occupancy_thr={args.mask_occupancy_thr}",
        "wandb.project=null",
        *(args.override or []),
    ]
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        return compose(config_name="eval", overrides=overrides)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="patchset-infer", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, type=Path,
                    help="target CT .nii.gz to segment")
    ap.add_argument("--context", required=True, action="append", nargs=2,
                    metavar=("IMAGE", "MASK"),
                    help="context image + binary mask (same organ); repeatable for K>1")
    ap.add_argument("--checkpoint", required=True,
                    help="trained patchset3d best.pt")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the predicted mask here (.nii.gz, on the target grid)")
    ap.add_argument("--gt", type=Path, default=None,
                    help="optional target GT mask .nii.gz -> report Dice + coarse-only Dice")
    ap.add_argument("--crop-spacings", dest="crop_spacings", default="4,1.5",
                    help="comma-separated coarse->fine CROP resolutions in mm/voxel — the "
                         "cascade's field-of-view schedule (T*mm per pass), NOT the input's "
                         "native voxel spacing (that is read from the nifti affine). "
                         "Match the checkpoint's training crops; default 4,1.5")
    ap.add_argument("--model", default="patchset3d", help="eval.model (default patchset3d)")
    ap.add_argument("--feat-norm", dest="feat_norm", default="self",
                    help="patchset3d feature-norm mode (default self)")
    ap.add_argument("--mask-downsample", dest="mask_downsample", default="occupancy",
                    help="context-mask downsample: occupancy|nearest (default occupancy)")
    ap.add_argument("--mask-occupancy-thr", dest="mask_occupancy_thr", type=float,
                    default=0.1, help="occupancy threshold (default 0.1)")
    ap.add_argument("--override", action="append", metavar="KEY=VALUE",
                    help="extra Hydra override, passed through to the cfg (repeatable)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    contexts = [(Path(img), Path(msk)) for img, msk in args.context]

    print(f"target     : {args.target}")
    for i, (img, msk) in enumerate(contexts):
        print(f"context[{i}] : {img}  mask={msk}")
    print(f"model      : {args.model}   crop_spacings=[{args.crop_spacings}]   "
          f"checkpoint={args.checkpoint}\n")

    cfg = _build_cfg(args)
    res = predict_nifti(cfg, args.target, contexts,
                        gt_path=args.gt, out_path=args.out)

    print(f"\n  pred nonzero voxels : {int(res['pred'].sum())}  shape={res['pred'].shape}")
    if res["dice"] is not None:
        print(f"  dice                : {res['dice']:.4f}")
        print(f"  coarse_only_dice    : {res['coarse_only_dice']:.4f}")
        print(f"  gain (fine-coarse)  : {res['dice'] - res['coarse_only_dice']:+.4f}")
    if res["pred_path"] is not None:
        print(f"  written             : {res['pred_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

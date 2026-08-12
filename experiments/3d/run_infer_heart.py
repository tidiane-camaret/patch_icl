"""One-off smoke run: predict_nifti on a real TotalSegmentator pair (heart).

Composes the same cfg surface as experiments/3d/eval.py, then runs the nifti
in-context cascade with s0000 as target and s0001 as the single context.

    python experiments/3d/run_infer_heart.py eval.model=patchset3d \
        eval.checkpoint=/.../best.pt eval.feat_norm=self \
        data.mask_downsample=occupancy data.mask_occupancy_thr=0.1 \
        eval.spacing_sweep=[4,1.5]
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling infer_nifti

from infer_nifti import predict_nifti


@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ts = Path(cfg.paths.totalseg)
    organ = cfg.get("organ", "urinary_bladder")  # heart is out-of-FOV for s0000/s0001 (pelvic)
    target = ts / "s0000" / "ct.nii.gz"
    contexts = [(ts / "s0001" / "ct.nii.gz", ts / "s0001" / "segmentations" / f"{organ}.nii.gz")]
    gt = ts / "s0000" / "segmentations" / f"{organ}.nii.gz"
    out = Path(f"/tmp/s0000_{organ}_pred.nii.gz")

    print(f"target : {target}")
    print(f"context: {contexts[0][0]}  mask={contexts[0][1].name}")
    print(f"gt     : {gt}")
    print(f"model  : {cfg.eval.model}  spacings={list(cfg.eval.spacing_sweep)}  "
          f"image_size={list(cfg.data.image_size)}\n")

    res = predict_nifti(cfg, target, contexts, gt_path=gt, out_path=out)
    print(f"\n  pred nonzero voxels : {int(res['pred'].sum())}  shape={res['pred'].shape}")
    print(f"  dice                : {res['dice']:.4f}")
    print(f"  coarse_only_dice    : {res['coarse_only_dice']:.4f}")
    print(f"  gain (fine vs coarse): {res['dice'] - res['coarse_only_dice']:+.4f}")
    print(f"  written             : {res['pred_path']}")


if __name__ == "__main__":
    main()

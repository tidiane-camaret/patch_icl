"""
Run TotalSegmentator's `body` task on every TotalSegmentator subject and save the body
envelope mask as `pred_body.npy` (uint8 0/1) in each subject dir, on the SAME canonical
RAS grid as label.npy / ct.npy (built via nib.as_closest_canonical in convert_to_npy.py),
so pred_body.npy aligns voxel-for-voxel with the existing labels.

Motivation: TotalSegmentator's 117-class `total` map (label.npy) has no "body" class, so the
GMM-synth pipeline (src/providers/synth_gmm.py) has no real-HU calibration for MAISI id 200
("body", present in 100% of bank crops). A per-subject body mask lets analyze_totalseg_
intensity.py measure the real soft-tissue envelope mean/std.

Resumable: skips subjects that already have pred_body.npy unless --overwrite.

  .venv_blackwell/bin/python experiments/3d/synth_task_generation/gen_body_masks.py \
      --data /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg
"""
import argparse
import tempfile
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from totalsegmentator.python_api import totalsegmentator

    subs = sorted(p for p in args.data.iterdir()
                  if p.is_dir() and (p / "ct.nii.gz").exists())
    if args.limit:
        subs = subs[:args.limit]
    print(f"gen_body_masks: {len(subs)} subjects under {args.data}", flush=True)

    done = skip = fail = 0
    t0 = time.time()
    for i, sd in enumerate(subs):
        out_npy = sd / "pred_body.npy"
        if out_npy.exists() and not args.overwrite:
            skip += 1
            continue
        try:
            ref = nib.as_closest_canonical(nib.load(str(sd / "ct.nii.gz")))
            with tempfile.TemporaryDirectory() as td:
                totalsegmentator(input=str(sd / "ct.nii.gz"), output=td, task="body",
                                 device=args.device, quiet=True)
                body = nib.as_closest_canonical(nib.load(f"{td}/body.nii.gz"))
                arr = (np.asarray(body.dataobj) > 0).astype(np.uint8)
            if arr.shape != ref.shape:
                raise RuntimeError(f"body shape {arr.shape} != ct-canonical {ref.shape}")
            lp = sd / "label.npy"
            if lp.exists():
                ls = np.load(lp, mmap_mode="r").shape
                if ls != arr.shape:
                    raise RuntimeError(f"body shape {arr.shape} != label.npy {ls}")
            np.save(out_npy, arr)
            done += 1
        except Exception as e:  # keep going; one bad subject shouldn't kill the batch
            fail += 1
            print(f"[FAIL] {sd.name}: {e}", flush=True)
            traceback.print_exc()
        if (i + 1) % 10 == 0 or i == len(subs) - 1:
            el = time.time() - t0
            rate = el / max(done, 1)
            eta_h = rate * (len(subs) - i - 1) / 3600
            print(f"{i+1}/{len(subs)}  done={done} skip={skip} fail={fail}  "
                  f"{rate:.1f}s/case  eta {eta_h:.1f}h", flush=True)
    print(f"gen_body_masks: FINISHED done={done} skip={skip} fail={fail} "
          f"elapsed {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()

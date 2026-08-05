"""Generate the native-resolution assets the use_crop eval path needs for the extra
`more_labels` subjects: a per-subject `ct.npy` and a root `spacings.json`.

The crop path (`TotalSegMoreLabelsDataset._load_crop`, via the base
`_organ_crop_arrays`) mmaps `ct.npy` and slices only the crop — decoding the full
`ct.nii.gz` per item would blow up memory under many workers — and reads the native
voxel spacing from `spacings.json` to size the crop's physical extent. The extra masks
are already on the CT's native grid (`more_labels/{task}.npy`), so only the CT and the
spacings are missing.

`ct.npy` reproduces `scripts/convert_to_npy.py` exactly: `nib.as_closest_canonical` →
`_normalise_ct` → float16 native. `spacings.json` mirrors its format
`{subject: {"spacing": [dx,dy,dz], "shape": [D,H,W]}}`, so the inherited
`_load_spacings` picks it up unchanged.

Usage
-----
  python experiments/totalseg_more_labels/generate_crop_assets.py [--workers N] [--overwrite]
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.convert_to_npy import _normalise_ct  # identical CT normalisation

DATA = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/totalseg_test_more_labels")


def convert_subject(args: tuple) -> tuple[str, str, list | None, list | None]:
    """Write native ct.npy for one subject; return (subj, status, spacing, shape)."""
    subj_dir, overwrite = args
    subj_dir = Path(subj_dir)
    subj = subj_dir.name
    ct_out = subj_dir / "ct.npy"
    try:
        img = nib.as_closest_canonical(nib.load(str(subj_dir / "ct.nii.gz")))
        spacing = [float(x) for x in nib.affines.voxel_sizes(img.affine)[:3]]
        shape = list(img.shape)
        if overwrite or not ct_out.exists():
            vol = _normalise_ct(img.get_fdata(dtype=np.float32))
            np.save(ct_out, vol.astype(np.float16))
        return subj, "ok", spacing, shape
    except Exception:
        return subj, traceback.format_exc(), None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA), help="totalseg_test_more_labels root")
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count()))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = Path(args.data)
    subjects = sorted(p for p in data.iterdir()
                      if p.is_dir() and (p / "more_labels").is_dir())
    print(f"{len(subjects)} subjects | workers={args.workers}")

    # Merge into any existing spacings.json (idempotent across reruns).
    spacings_path = data / "spacings.json"
    spacings: dict = {}
    if spacings_path.exists():
        with open(spacings_path) as f:
            spacings = json.load(f)

    tasks = [(str(s), args.overwrite) for s in subjects]
    done = ok = errors = 0
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for subj, status, spacing, shape in pool.imap_unordered(convert_subject, tasks, chunksize=1):
            done += 1
            if status == "ok":
                ok += 1
                spacings[subj] = {"spacing": spacing, "shape": shape}
            else:
                errors += 1
                print(f"\n[ERROR] {subj}:\n{status}")
            print(f"\r  {done}/{len(subjects)}  ok={ok}  err={errors}", end="", flush=True)

    with open(spacings_path, "w") as f:
        json.dump(spacings, f)
    print(f"\nWrote {spacings_path.name} ({len(spacings)} subjects)")
    print(f"Done in {(time.time()-t0)/60:.1f} min — ok={ok} errors={errors}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()

"""Laterality check (RAS: +x = patient RIGHT) + z-spacing histogram."""
import json, glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import nibabel as nib, numpy as np
ROOT = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/flare22/FLARE22Train")
def one(p):
    im = nib.load(p); lab = np.asarray(im.dataobj)
    out = {}
    for i in (2, 13, 7, 8):
        m = lab == i
        if m.any():
            xs = np.where(m.any(axis=(1, 2)))[0]
            out[i] = float(xs.mean())  # voxel x index; +x = R in RAS
    return out
if __name__ == "__main__":
    ps = sorted((ROOT/"labels").glob("*.nii.gz"))
    with ProcessPoolExecutor(16) as ex: rs = list(ex.map(one, ps))
    print("mean voxel-x centroid (RAS: larger x = patient RIGHT)")
    for i, nm in [(2,"id2"),(13,"id13"),(7,"id7"),(8,"id8")]:
        print(f"  {nm}: {np.mean([r[i] for r in rs]):.0f}")
    print("id2 > id13 in", sum(r[2] > r[13] for r in rs), "/50 cases  (id2 should be RIGHT kidney)")
    print("id7 > id8  in", sum(r[7] > r[8] for r in rs), "/50 cases  (id7 should be RIGHT adrenal)")

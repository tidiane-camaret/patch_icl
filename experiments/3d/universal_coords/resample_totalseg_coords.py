"""Resample the finest (0.8/0.8/1.6mm) totalseg coords maps onto each case's
native CT grid (1.5mm iso). Result: coords co-registered voxel-for-voxel with
ct.nii.gz and label.npy, so no affine bridge is needed downstream.

Uses full affines (the CT/coords affines carry a small rotation, so a diagonal
approximation is wrong). Trilinear per channel.
"""
import os, glob, time
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "coords_predictor", "output_totalseg")
DST = os.path.join(HERE, "coords_predictor", "output_totalseg_1p5")


def resample_one(sid):
    co_img = nib.load(os.path.join(SRC, f"{sid}_coords.nii.gz"))
    ct_img = nib.load(os.path.join(TS, sid, "ct.nii.gz"))
    co = np.asanyarray(co_img.dataobj).astype(np.float32)      # (X,Y,Z,3) @ 0.8/1.6
    shp = ct_img.shape[:3]
    gi, gj, gk = np.meshgrid(*[np.arange(s) for s in shp], indexing="ij")
    ijk1 = np.stack([gi, gj, gk, np.ones_like(gi)], -1).reshape(-1, 4).T
    world = ct_img.affine @ ijk1
    co_idx = (np.linalg.inv(co_img.affine) @ world)[:3]         # (3, N) fractional coords-voxel index
    out = np.empty((*shp, 3), np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(co[..., c], co_idx, order=1, mode="nearest").reshape(shp)
    nib.save(nib.Nifti1Image(out, ct_img.affine, ct_img.header), os.path.join(DST, f"{sid}_coords.nii.gz"))
    return out.shape


def main():
    os.makedirs(DST, exist_ok=True)
    ids = sorted(os.path.basename(f).replace("_coords.nii.gz", "")
                 for f in glob.glob(os.path.join(SRC, "*_coords.nii.gz")))
    for i, sid in enumerate(ids):
        dst = os.path.join(DST, f"{sid}_coords.nii.gz")
        if os.path.exists(dst):
            print(f"[{i+1}/{len(ids)}] {sid} exists, skip"); continue
        t0 = time.perf_counter()
        sh = resample_one(sid)
        print(f"[{i+1}/{len(ids)}] {sid} -> {sh} @1.5mm  {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

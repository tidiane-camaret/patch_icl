"""Generate coords maps for ALL TotalSeg scans, stored at native 1.5mm-iso grid
(co-registered with ct.npy / label.npy) as float16 -> <scan>/coords.npy, shape
(X,Y,Z,3).

Per scan: run the next8 model on ct.nii.gz at sampling_factor=2 (2/2/4mm; the
correspondence is resolution-invariant so fine generation is wasted), then
trilinear-resample onto the native CT grid via full affines and cast to float16.
Resumable (skips existing), OOM-safe (sf 2 -> 1 fallback), logs timing.

Run (loki/nero, patchwork_minimal env):
  export LD_LIBRARY_PATH=/software/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib:\
/software/anaconda3/envs/tf215/lib/python3.10/site-packages/nvidia/cudnn/lib
  /software/anaconda3/envs/patchwork_minimal/bin/python -u batch_all_totalseg_coords.py --split all
"""
import sys, os, csv, time, argparse, tempfile
sys.path.append("/software")
import numpy as np, nibabel as nib, tensorflow as tf
from scipy.ndimage import map_coordinates
import patchwork2.model as patchwork

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
MODEL = "/nfs/data/nii/data1/Analysis/raua___BodyComp/ANALYSIS_coord/next8/model_patchwork.json"
OUTNAME = "coords.npy"


def scan_ids(split):
    rows = list(csv.reader(open(os.path.join(TS, "meta.csv")), delimiter=";"))
    si = rows[0].index("split")
    return [r[0] for r in rows[1:] if len(r) > si and (split == "all" or r[si] == split)]


def gen_coords(model, inp, sf):
    """Run the model at the given sampling factor; return (coords X,Y,Z,3, affine).
    apply_on_nifti(ofname=None) drops the affine, so route through a temp nifti."""
    fd, tmp = tempfile.mkstemp(suffix=".nii.gz", dir="/tmp"); os.close(fd)
    try:
        model.apply_on_nifti(
            [inp], tmp, generate_type="random", out_typ="float32",
            repetitions=30, num_chunks=10, branch_factor=2,
            input_transform=lambda x: tf.where(x < -1000.0, -1000.0, x),
            postproc=lambda x: model.finalBlock.decodeCoords(x[..., 1:]),
            level="mixnohead", augment={}, scale_to_original=False,
            sampling_factor=float(sf), crop_fdim=None)
        img = nib.load(tmp)
        return np.asarray(img.dataobj, np.float32), img.affine
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def resample_to_ct(co, A_co, ct_img):
    """Trilinear-resample coords (X,Y,Z,3) onto the native CT grid -> float16."""
    shp = ct_img.shape[:3]
    gi, gj, gk = np.meshgrid(*[np.arange(s) for s in shp], indexing="ij")
    ijk1 = np.stack([gi, gj, gk, np.ones_like(gi)], -1).reshape(-1, 4).T
    co_idx = (np.linalg.inv(A_co) @ (ct_img.affine @ ijk1))[:3]
    out = np.empty((*shp, 3), np.float16)
    for c in range(3):
        out[..., c] = map_coordinates(co[..., c], co_idx, order=1, mode="nearest").reshape(shp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--sampling", type=float, default=2.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids = scan_ids(args.split)
    print(f"GPUs {len(tf.config.list_physical_devices('GPU'))} | split={args.split} "
          f"| {len(ids)} scans | sf={args.sampling} -> resample to native 1.5mm float16", flush=True)
    model = patchwork.PatchWorkModel.load(MODEL)

    ok = skip = fail = 0
    for i, sid in enumerate(ids):
        out = os.path.join(TS, sid, OUTNAME)
        inp = os.path.join(TS, sid, "ct.nii.gz")
        if os.path.exists(out) and not args.overwrite:
            skip += 1; continue
        if not os.path.exists(inp):
            print(f"[{i+1}/{len(ids)}] {sid} MISSING ct.nii.gz", flush=True); fail += 1; continue
        t0 = time.perf_counter()
        for sf in [args.sampling, 1.0]:
            if sf > args.sampling:
                continue
            try:
                co, A_co = gen_coords(model, inp, sf)
                out_arr = resample_to_ct(co, A_co, nib.load(inp))
                np.save(out, out_arr)
                print(f"[{i+1}/{len(ids)}] {sid} sf={sf:.0f} {out_arr.shape} "
                      f"{out_arr.nbytes/1e6:.0f}MB  {time.perf_counter()-t0:.1f}s", flush=True)
                ok += 1; break
            except tf.errors.ResourceExhaustedError:
                print(f"[{i+1}/{len(ids)}] {sid} sf={sf:.0f} OOM, fallback", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(ids)}] {sid} FAILED: {repr(e)[:160]}", flush=True); fail += 1; break
    print(f"\ndone: {ok} generated, {skip} skipped, {fail} failed / {len(ids)}", flush=True)


if __name__ == "__main__":
    main()

"""Batch-apply the next8 coords model over the first N cases of the ChemoTox
paths JSON at a given resolution (scalar sampling_factor). Saves one coords
nifti per case, skips existing, logs timing."""
import sys, os, json, time, argparse
sys.path.append("/software")
import numpy as np, nibabel as nib, tensorflow as tf
import patchwork2.model as patchwork

HERE = os.path.dirname(__file__)
JSON = os.path.join(HERE, "..", "coords_paths_chemotox.json")
MODEL = "/nfs/data/nii/data1/Analysis/raua___BodyComp/ANALYSIS_coord/next8/model_patchwork.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", type=float, default=4.0, help="scalar sampling_factor")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--outdir", default=os.path.join(HERE, "output_batch"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    paths = json.load(open(JSON))
    keys = list(paths)[:args.n]
    print(f"GPUs {len(tf.config.list_physical_devices('GPU'))} | sampling={args.sampling} | {len(keys)} cases")
    model = patchwork.PatchWorkModel.load(MODEL)

    for i, k in enumerate(keys):
        out = os.path.join(args.outdir, f"{k.replace('#','_')}_coords.nii.gz")
        if os.path.exists(out):
            print(f"[{i+1}/{len(keys)}] {k} exists, skip"); continue
        inp = paths[k]["img"]
        if not os.path.exists(inp):
            print(f"[{i+1}/{len(keys)}] {k} MISSING input {inp}"); continue
        t0 = time.perf_counter()
        try:
            model.apply_on_nifti([inp], out, generate_type="random", out_typ="float32",
                repetitions=30, num_chunks=10, branch_factor=2,
                input_transform=lambda x: tf.where(x < -1000.0, -1000.0, x),
                postproc=lambda x: model.finalBlock.decodeCoords(x[..., 1:]),
                level="mixnohead", augment={}, scale_to_original=False,
                sampling_factor=args.sampling, crop_fdim=None)
            sh = nib.load(out).shape
            print(f"[{i+1}/{len(keys)}] {k} -> {sh}  {time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(keys)}] {k} FAILED: {repr(e)[:160]}", flush=True)


if __name__ == "__main__":
    main()

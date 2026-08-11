"""Standalone runner for the next8 coords model (cleaned up from next8.py, which
relied on the NORA DPX_selectFiles helper and had undefined f1/f5 vars).

Runs the model on one CT and writes the coords map. `--sampling` controls output
resolution: default 1 reproduces the 4/4/8mm map; e.g. `--sampling 2 2 4 mm`
requests a 2/2/4mm output (finer), stitching the pyramid's fine levels onto a
denser grid instead of the config's destvox_mm=[4,4,8].
"""
import sys, os, argparse, json, time
sys.path.append("/software")
import numpy as np
import nibabel as nib
import tensorflow as tf
import patchwork2.model as patchwork

INPUT = "/nfs/data/nii/data1/jungm___ChemoTox/10116066/20220316122148/11_Thx_Abd_DE_venoes/Thx_Abd_DE_KM_3_0_Bf40_3_F_0_8_s002.nii"
MODEL = "/nfs/data/nii/data1/Analysis/raua___BodyComp/ANALYSIS_coord/next8/model_patchwork.json"
OUTDIR = os.path.join(os.path.dirname(__file__), "output")


def parse_sampling(vals):
    if vals is None:
        return 1
    if vals[-1] == "mm":
        return [float(v) for v in vals[:-1]] + ["mm"]
    return float(vals[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", nargs="+", default=None, help="e.g. '2 2 4 mm' or '2'")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--num_chunks", type=int, default=10)
    ap.add_argument("--repetitions", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0, help="TF intra-op threads (0=default)")
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(min(args.threads, 8))
    os.makedirs(OUTDIR, exist_ok=True)
    sf = parse_sampling(args.sampling)
    out = os.path.join(OUTDIR, f"pred_next8_coords{args.suffix}.nii.gz")
    print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))} | sampling_factor={sf} "
          f"| chunks={args.num_chunks} reps={args.repetitions} threads={args.threads or 'default'}")

    model = patchwork.PatchWorkModel.load(MODEL)
    t0 = time.perf_counter()
    nii, res = model.apply_on_nifti(
        [args.input],
        out,
        generate_type="random",
        out_typ="float32",
        repetitions=args.repetitions,
        num_chunks=args.num_chunks,
        branch_factor=2,
        input_transform=lambda x: tf.where(x < -1000.0, -1000.0, x),
        postproc=lambda x: model.finalBlock.decodeCoords(x[..., 1:]),
        level="mixnohead",
        augment={},
        scale_to_original=False,
        sampling_factor=sf,
        crop_fdim=None,
    )
    dt = time.perf_counter() - t0
    saved = nib.load(out)
    print(f"\nDONE. output shape {saved.shape}  zooms "
          f"{tuple(round(float(z),3) for z in saved.header.get_zooms())}  | apply {dt:.1f}s")


if __name__ == "__main__":
    main()

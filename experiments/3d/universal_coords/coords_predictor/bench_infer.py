"""Benchmark next8 coords inference: sweep num_chunks (and TF threads via env),
time each, and measure coords deviation vs the 10-chunk reference. Model loaded
once. Run with --threads 0 (default) and --threads 32 to compare."""
import sys, os, time, argparse
sys.path.append("/software")
import numpy as np, nibabel as nib, tensorflow as tf
import patchwork2.model as patchwork

INPUT = "/nfs/data/nii/data1/jungm___ChemoTox/10116066/20220316122148/11_Thx_Abd_DE_venoes/Thx_Abd_DE_KM_3_0_Bf40_3_F_0_8_s002.nii"
MODEL = "/nfs/data/nii/data1/Analysis/raua___BodyComp/ANALYSIS_coord/next8/model_patchwork.json"

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--sampling", type=float, default=1.0)
args = ap.parse_args()
if args.threads:
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(8)

print(f"threads={args.threads or 'default'} sampling={args.sampling} "
      f"intra={tf.config.threading.get_intra_op_parallelism_threads()}")
model = patchwork.PatchWorkModel.load(MODEL)

def run(nc):
    t0 = time.perf_counter()
    ret = model.apply_on_nifti([INPUT], None, generate_type="random", out_typ="float32",
        repetitions=30, num_chunks=nc, branch_factor=2,
        input_transform=lambda x: tf.where(x < -1000.0, -1000.0, x),
        postproc=lambda x: model.finalBlock.decodeCoords(x[..., 1:]),
        level="mixnohead", augment={}, scale_to_original=False,
        sampling_factor=args.sampling, crop_fdim=None, return_nibabel=True)
    dt = time.perf_counter() - t0
    nii = ret[0] if isinstance(ret, tuple) else ret
    arr = np.asarray(nii.dataobj, np.float32) if hasattr(nii, "dataobj") else np.asarray(nii, np.float32)
    return dt, arr

ref_t, ref = run(10)
print(f"\n{'chunks':>6} {'time(s)':>8} {'speedup':>8} {'mean|Δ| vs 10ch':>16} {'p95':>7}")
print(f"{10:>6} {ref_t:8.1f} {1.0:8.2f} {0.0:16.2f} {0.0:7.2f}")
for nc in [5, 4, 3, 2, 1]:
    t, a = run(nc)
    d = np.linalg.norm(a - ref, axis=-1)
    print(f"{nc:>6} {t:8.1f} {ref_t/t:8.2f} {d.mean():16.2f} {np.percentile(d,95):7.2f}")

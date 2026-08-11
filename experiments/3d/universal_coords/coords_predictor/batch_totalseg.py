"""Batch-apply the next8 coords model to TotalSegmentator TEST cases at the
finest feasible resolution. Reads the test split from meta.csv, feeds each
native ct.nii.gz, and writes one coords nifti per case to output_totalseg/.

Finest level = sampling_factor 5 -> 0.8/0.8/1.6mm (the pyramid's finest
supervised level). Big-FOV volumes can OOM the anti-alias conv, so each case
falls back 5 -> 4 -> 3 on ResourceExhausted. Skips existing, logs timing.
"""
import sys, os, csv, time, argparse
sys.path.append("/software")
import numpy as np, nibabel as nib, tensorflow as tf
import patchwork2.model as patchwork

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
MODEL = "/nfs/data/nii/data1/Analysis/raua___BodyComp/ANALYSIS_coord/next8/model_patchwork.json"
HERE = os.path.dirname(__file__)


def test_ids(n):
    rows = list(csv.reader(open(os.path.join(TS, "meta.csv")), delimiter=";"))
    si = rows[0].index("split")
    return [r[0] for r in rows[1:] if len(r) > si and r[si] == "test"][:n]


def run_one(model, inp, out, sf):
    model.apply_on_nifti(
        [inp], out, generate_type="random", out_typ="float32",
        repetitions=30, num_chunks=10, branch_factor=2,
        input_transform=lambda x: tf.where(x < -1000.0, -1000.0, x),
        postproc=lambda x: model.finalBlock.decodeCoords(x[..., 1:]),
        level="mixnohead", augment={}, scale_to_original=False,
        sampling_factor=float(sf), crop_fdim=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", type=float, default=5.0, help="finest sampling_factor to try")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--outdir", default=os.path.join(HERE, "output_totalseg"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ids = test_ids(args.n)
    print(f"GPUs {len(tf.config.list_physical_devices('GPU'))} | finest sf={args.sampling} | {len(ids)} test cases")
    model = patchwork.PatchWorkModel.load(MODEL)

    for i, sid in enumerate(ids):
        out = os.path.join(args.outdir, f"{sid}_coords.nii.gz")
        if os.path.exists(out):
            print(f"[{i+1}/{len(ids)}] {sid} exists, skip", flush=True); continue
        inp = os.path.join(TS, sid, "ct.nii.gz")
        if not os.path.exists(inp):
            print(f"[{i+1}/{len(ids)}] {sid} MISSING {inp}", flush=True); continue
        for sf in [args.sampling, 4.0, 3.0]:
            if sf > args.sampling:
                continue
            t0 = time.perf_counter()
            try:
                run_one(model, inp, out, sf)
                sh = nib.load(out).shape; zm = nib.load(out).header.get_zooms()[:3]
                print(f"[{i+1}/{len(ids)}] {sid} sf={sf:.0f} -> {sh} "
                      f"{tuple(round(float(z),2) for z in zm)}mm  {time.perf_counter()-t0:.1f}s", flush=True)
                break
            except tf.errors.ResourceExhaustedError:
                print(f"[{i+1}/{len(ids)}] {sid} sf={sf:.0f} OOM, falling back", flush=True)
            except Exception as e:
                print(f"[{i+1}/{len(ids)}] {sid} FAILED: {repr(e)[:160]}", flush=True); break


if __name__ == "__main__":
    main()

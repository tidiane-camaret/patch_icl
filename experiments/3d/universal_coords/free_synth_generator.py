"""Batch generator of position-corresponding synthetic in-context tasks on the
TotalSeg test cases, using the coords field. Each task:

  1. pick a REFERENCE subject + a free ellipsoid (random body position, random
     radii 15-50mm, random orientation) that sits >=30% on labelled anatomy;
  2. its coords cloud Q defines a canonical body region;
  3. for K other CONTEXT subjects, transfer Q via bin-hashing -> a mask at the
     SAME body position on each subject.

Emits K+1 (subject, binary-mask) pairs per task. A context is valid only if its
transferred mask is non-trivial (the region falls inside that subject's FOV), so
partial-body scans are skipped rather than yielding empty masks.

Masks are stored sparsely (flat nonzero indices) in one npz per task, plus a
manifest.json with subjects + cross-subject HI. coords/ct/label share the 1.5mm
grid (output_totalseg_1p5), so masks are already in each subject's voxel frame.
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib

from totalseg_ellipsoid_transfer import TS, CO, DS, load, m_bin
from plot_totalseg_free_transfer import rand_ellipsoid, label_hist, hist_int

OUT = os.path.join(os.path.dirname(__file__), "output_synth_tasks")
MIN_MASK = 50          # min transferred voxels for a context to count as valid
RETRY = 40             # ellipsoid resamples before giving up on a reference


def body_idx(sid):
    ct = np.asanyarray(nib.load(os.path.join(TS, sid, "ct.nii.gz")).dataobj)[::DS, ::DS, ::DS]
    return np.argwhere(ct > -300)


def build_task(subjects, cache, bodies, K, rng, min_hi=0.0):
    """One task: reference free blob + K valid context transfers. A context is
    accepted only if its mask is non-trivial AND lands on matching anatomy
    (HI >= min_hi) -- the reject-resample validity guard. None if it fails."""
    ref = subjects[rng.integers(len(subjects))]
    co_r, lab_r = cache[ref]
    for _ in range(RETRY):
        reg, Q = rand_ellipsoid(co_r, bodies[ref], lab_r.shape, rng)
        if (lab_r[reg] > 0).mean() >= 0.3:
            break
    else:
        return None
    hc = label_hist(lab_r.reshape(-1), reg.reshape(-1))
    task = {ref: reg}
    his = []
    for sid in rng.permutation([s for s in subjects if s != ref]):
        co_t, lab_t = cache[sid]
        sel = m_bin(Q, co_t.reshape(-1, 3)).reshape(lab_t.shape)
        if sel.sum() < MIN_MASK:
            continue
        hi = hist_int(hc, label_hist(lab_t.reshape(-1), sel.reshape(-1)))
        if hi < min_hi:                     # guard: reject off-anatomy transfers
            continue
        task[sid] = sel; his.append(hi)
        if len(task) == K + 1:
            break
    if len(task) < K + 1:
        return None
    return ref, task, float(np.mean(his))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=30)
    ap.add_argument("--K", type=int, default=3, help="context subjects per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_hi", type=float, default=0.0, help="reject context transfers below this HI")
    ap.add_argument("--save", action="store_true", help="write npz masks + manifest")
    args = ap.parse_args()

    ids = sorted(os.path.basename(f).replace("_coords.nii.gz", "")
                 for f in glob.glob(os.path.join(CO, "*_coords.nii.gz")))
    cache = {s: load(s) for s in ids}
    bodies = {s: body_idx(s) for s in ids}
    rng = np.random.default_rng(args.seed)
    if args.save:
        os.makedirs(OUT, exist_ok=True)

    manifest, his, tries = [], [], 0
    while len(manifest) < args.n_tasks and tries < args.n_tasks * 6:
        tries += 1
        res = build_task(ids, cache, bodies, args.K, rng, args.min_hi)
        if res is None:
            continue
        ref, task, hi = res
        his.append(hi)
        entry = {"task": len(manifest), "ref": ref, "hi": round(hi, 3),
                 "subjects": {s: int(m.sum()) for s, m in task.items()}}
        if args.save:
            np.savez_compressed(
                os.path.join(OUT, f"task_{len(manifest):03d}.npz"),
                ref=ref, subjects=list(task),
                shapes=np.array([m.shape for m in task.values()]),
                **{f"idx_{s}": np.flatnonzero(m.reshape(-1)).astype(np.int32) for s, m in task.items()})
        manifest.append(entry)

    print(f"generated {len(manifest)}/{args.n_tasks} tasks (K+1={args.K+1} subjects each) "
          f"from {len(ids)} cases in {tries} tries")
    print(f"cross-subject HI: mean {np.mean(his):.3f} +/- {np.std(his):.3f}  "
          f"[min {np.min(his):.3f}, max {np.max(his):.3f}]")
    voxels = [v for e in manifest for v in e["subjects"].values()]
    print(f"mask voxels/subject (~3mm): median {int(np.median(voxels))} "
          f"[p10 {int(np.percentile(voxels,10))}, p90 {int(np.percentile(voxels,90))}]")
    if args.save:
        json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
        print(f"saved {len(manifest)} npz tasks + manifest.json -> {OUT}")


if __name__ == "__main__":
    main()

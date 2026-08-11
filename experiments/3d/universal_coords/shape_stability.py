"""Which synthetic-blob SHAPE transfers most stably through our coords maps?

For a set of reference subjects we draw a random valid body position, then build
FOUR blob families at that SAME position, size-matched to ~TARGET_VOX voxels:

  ellipsoid  - single random-oriented ellipsoid (baseline)
  metaball   - union of 2-4 jittered ellipsoids (irregular, non-convex)
  noise      - smooth-noise threshold in a bbox (organic, possibly multi-lobed)
  coords_ball- region defined directly by coords proximity to the seed (upper
               bound: shape lives in the shared frame, so transfer is ~exact)

Each blob's coords values form a cloud Q; we transfer to K target subjects via
bin-hashing and score how well the landed region hits the SAME anatomy:

  HI  = bg-excluded label-histogram intersection (ctx blob vs tgt region), 0..1
  surv= fraction of transfers giving a non-trivial (>=MIN_MASK) mask

Same positions + matched sizes across families isolate the shape effect.
Run (loki): .venv_thor_fresh/bin/python experiments/3d/universal_coords/shape_stability.py
"""
import os, sys, argparse
import numpy as np
from scipy.ndimage import gaussian_filter

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
DS = 2                 # ~3mm eval grid
GMM = 1.5 * DS
TARGET_VOX = 3000      # matched blob size (~3mm voxels) ~ 80cm^3
MIN_MASK = 50
MIN_ON_ANAT = 0.3      # blob must sit >=30% on labelled anatomy to be a valid ref


def subject_ids():
    return sorted(d for d in os.listdir(TS)
                  if d.startswith("s") and os.path.exists(os.path.join(TS, d, "coords.npy")))


def load(sid):
    co = np.load(os.path.join(TS, sid, "coords.npy"))[::DS, ::DS, ::DS].astype(np.float32)
    lab = np.load(os.path.join(TS, sid, "label.npy"))[::DS, ::DS, ::DS]
    return co, lab


def label_hist(lab_flat, sel_flat):
    if sel_flat.sum() == 0:
        return None
    u, cnt = np.unique(lab_flat[sel_flat], return_counts=True)
    h = np.zeros(256); h[u] = cnt; h[0] = 0        # bg-excluded
    s = h.sum()
    return None if s == 0 else h / s


def hist_int(a, b):
    return 0.0 if a is None or b is None else float(np.minimum(a, b).sum())


def m_bin(Q, tgt, b=8.0):
    lo = np.minimum(Q.min(0), tgt.min(0))
    qb = np.floor((Q - lo) / b).astype(np.int64); tb = np.floor((tgt - lo) / b).astype(np.int64)
    dims = tb.max(0).clip(qb.max(0)) + 1
    return np.isin(np.ravel_multi_index(tb.T, dims), np.unique(np.ravel_multi_index(qb.T, dims)))


def rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.diag(R))


def _rescale(region_fn, r0, gshape, target):
    """Build region at scale r0, then rescale once to hit target voxel count."""
    reg = region_fn(r0)
    v = int(reg.sum())
    if v == 0:
        return reg
    r1 = r0 * (target / v) ** (1 / 3)
    return region_fn(r1)


def ellipsoid(co, c, gshape, rng, target):
    R = rand_rot(rng); aniso = rng.uniform(0.6, 1.6, 3)
    g = np.stack(np.meshgrid(*[np.arange(s) for s in gshape], indexing="ij"), -1).astype(np.float32)
    d = (g - c) @ R.T
    return _rescale(lambda r: ((d / (r * aniso)) ** 2).sum(-1) <= 1.0, 8.0, gshape, target)


def metaball(co, c, gshape, rng, target):
    k = rng.integers(2, 5)
    g = np.stack(np.meshgrid(*[np.arange(s) for s in gshape], indexing="ij"), -1).astype(np.float32)
    centers = c + rng.uniform(-6, 6, (k, 3)); rots = [rand_rot(rng) for _ in range(k)]
    anis = rng.uniform(0.6, 1.4, (k, 3))

    def build(r):
        m = np.zeros(gshape, bool)
        for ci, Ri, ai in zip(centers, rots, anis):
            d = (g - ci) @ Ri.T
            m |= ((d / (r * ai)) ** 2).sum(-1) <= 1.0
        return m
    return _rescale(build, 6.0, gshape, target)


def noise(co, c, gshape, rng, target):
    half = 22
    sl = tuple(slice(max(0, int(c[i]) - half), min(gshape[i], int(c[i]) + half)) for i in range(3))
    box = tuple(s.stop - s.start for s in sl)
    field = gaussian_filter(rng.normal(size=box).astype(np.float32), sigma=3.0)
    # radial falloff so the blob stays compact around the seed
    gb = np.stack(np.meshgrid(*[np.arange(b) for b in box], indexing="ij"), -1).astype(np.float32)
    cb = np.array([c[i] - sl[i].start for i in range(3)])
    field -= 0.03 * np.linalg.norm(gb - cb, axis=-1)
    reg = np.zeros(gshape, bool)
    sub = np.zeros(box, bool)
    thr = np.quantile(field, max(0.0, 1 - target / field.size))
    sub[field >= thr] = True
    reg[sl] = sub
    return reg


def coords_ball(co, c, gshape, rng, target):
    seed = co[int(c[0]), int(c[1]), int(c[2])]
    d = np.linalg.norm(co - seed, axis=-1)
    return _rescale(lambda r: d <= r, 30.0, gshape, target)


SHAPES = {"ellipsoid": ellipsoid, "metaball": metaball, "noise": noise, "coords_ball": coords_ball}


def valid_center(lab, rng):
    anat = np.argwhere(lab > 0)
    return anat[rng.integers(len(anat))].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ref", type=int, default=12, help="reference positions")
    ap.add_argument("--K", type=int, default=4, help="target subjects per transfer")
    ap.add_argument("--pool", type=int, default=24, help="subject pool size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids = subject_ids()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids); ids = ids[:args.pool]
    cache = {s: load(s) for s in ids}
    print(f"pool {len(ids)} subjects | {args.n_ref} refs x K={args.K} tgt | "
          f"target {TARGET_VOX} vox (~{TARGET_VOX*GMM**3/1000:.0f}cm^3)\n", flush=True)

    res = {k: {"hi": [], "surv": [], "vox": []} for k in SHAPES}
    made = 0
    tries = 0
    while made < args.n_ref and tries < args.n_ref * 8:
        tries += 1
        ref = ids[rng.integers(len(ids))]
        co_r, lab_r = cache[ref]
        c = valid_center(lab_r, rng)
        # one ref RNG so every shape family sees the SAME position/orientation seed
        seed = rng.integers(1 << 30)
        blobs = {}
        ok = True
        for name_, fn in SHAPES.items():
            reg = fn(co_r, c, lab_r.shape, np.random.default_rng(seed), TARGET_VOX)
            if reg.sum() < MIN_MASK or (lab_r[reg] > 0).mean() < MIN_ON_ANAT:
                ok = False; break
            blobs[name_] = reg
        if not ok:
            continue
        tgts = [t for t in ids if t != ref]
        tgts = list(np.array(tgts)[rng.permutation(len(tgts))[:args.K]])
        for name_, reg in blobs.items():
            Q = co_r[reg]
            hc = label_hist(lab_r.reshape(-1), reg.reshape(-1))
            res[name_]["vox"].append(int(reg.sum()))
            for t in tgts:
                co_t, lab_t = cache[t]
                sel = m_bin(Q, co_t.reshape(-1, 3)).reshape(lab_t.shape)
                surv = sel.sum() >= MIN_MASK
                res[name_]["surv"].append(float(surv))
                if surv:
                    res[name_]["hi"].append(
                        hist_int(hc, label_hist(lab_t.reshape(-1), sel.reshape(-1))))
        made += 1
        print(f"[{made}/{args.n_ref}] ref {ref} @ {c.astype(int)}  "
              f"sizes {[res[k]['vox'][-1] for k in SHAPES]}", flush=True)

    print(f"\n=== shape stability ({made} refs, size-matched ~{TARGET_VOX} vox) ===")
    print(f"{'shape':12} {'HI mean':>8} {'HI std':>7} {'surv':>6} {'vox':>7} {'n':>4}")
    for k in SHAPES:
        hi = res[k]["hi"]; sv = res[k]["surv"]; vx = res[k]["vox"]
        print(f"{k:12} {np.mean(hi):8.3f} {np.std(hi):7.3f} {np.mean(sv):6.2f} "
              f"{int(np.mean(vx)):7d} {len(hi):4d}")
    print("\nHI = bg-excl label-hist intersection ctx-blob vs tgt landed region "
          "(higher = same anatomy); surv = frac non-trivial masks.")


if __name__ == "__main__":
    main()

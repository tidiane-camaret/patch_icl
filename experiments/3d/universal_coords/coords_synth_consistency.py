"""Phase-0: cross-subject consistency of coords-FUNCTION synthetic labels, per
family and scale, to fix the reliable scale bands for the generator.

Every label is a soft field f(coords)->[0,1] with params sampled ONCE (from a
random reference subject's coords distribution) and evaluated on every pool
subject's coords.npy -- so correspondence is by construction. We score how much
the SAME anatomy is selected across subjects via the mean pairwise
soft-label-weighted, bg-excluded label-histogram intersection (HI, 0..1).

Families: gaussian (soft blob, sweep sigma), slab (band, sweep width),
cylinder (sweep radius), halfspace (single). Also reports mean mask mass (vox).

Run (loki): .venv_thor_fresh/bin/python experiments/3d/universal_coords/coords_synth_consistency.py
"""
import os, sys, argparse
import numpy as np

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
DS = 2
EDGE = 4.0          # hard-edge width (mm) for tier-1 primitives
MIN_MASS = 40.0     # min (soft) voxel mass for a subject to count


def subject_ids():
    return sorted(d for d in os.listdir(TS)
                  if d.startswith("s") and os.path.exists(os.path.join(TS, d, "coords.npy")))


def load(sid):
    co = np.load(os.path.join(TS, sid, "coords.npy"))[::DS, ::DS, ::DS].astype(np.float32)
    lab = np.load(os.path.join(TS, sid, "label.npy"))[::DS, ::DS, ::DS]
    return co.reshape(-1, 3), lab.reshape(-1)


# ---- field primitives: f(coords (n,3), params) -> soft [0,1] ----
def f_gaussian(co, p):
    d = co - p["mu"]
    m = np.einsum("ni,ij,nj->n", d, p["Sinv"], d)
    return np.exp(-0.5 * m)


def f_halfspace(co, p):
    return 1.0 / (1.0 + np.exp(-(co @ p["n"] - p["b"]) / EDGE))


def f_slab(co, p):
    return 1.0 / (1.0 + np.exp((np.abs(co @ p["n"] - p["b"]) - p["hw"]) / EDGE))


def f_cylinder(co, p):
    d = co - p["mu"]
    d = d - np.outer(d @ p["axis"], p["axis"])      # drop component along axis
    dist = np.linalg.norm(d, axis=1)
    return 1.0 / (1.0 + np.exp((dist - p["r"]) / EDGE))


# ---- localized (bounded, anchored) families: preferred, FOV-robust ----
def f_ellipsoid(co, p):
    """Hard anisotropic ellipsoid centred at mu (binarize at 0.5 => inside)."""
    d = (co - p["mu"]) @ p["R"].T
    dn = np.sqrt(((d / p["radii"]) ** 2).sum(1))
    return 1.0 / (1.0 + np.exp((dn - 1.0) * 20.0))


def f_cyl_capped(co, p):
    """Cylinder of radius r AND half-length L along axis (localized, not infinite)."""
    d = co - p["mu"]
    a = d @ p["axis"]
    rd = np.linalg.norm(d - np.outer(a, p["axis"]), axis=1)
    return (1.0 / (1.0 + np.exp((rd - p["r"]) / EDGE))) * \
           (1.0 / (1.0 + np.exp((np.abs(a) - p["L"]) / EDGE)))


def rand_unit(rng):
    v = rng.normal(size=3); return v / np.linalg.norm(v)


def rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.diag(R))


def sample_params(family, scale, ref_co, rng):
    mu = ref_co[rng.integers(len(ref_co))]          # a real canonical body location
    if family == "gaussian":
        aniso = rng.uniform(0.6, 1.6, 3)
        Sinv = np.diag(1.0 / (scale * aniso) ** 2)
        return {"mu": mu, "Sinv": Sinv}
    if family == "ellipsoid":
        return {"mu": mu, "R": rand_rot(rng), "radii": scale * rng.uniform(0.6, 1.6, 3)}
    if family == "cyl_capped":
        return {"mu": mu, "axis": rand_unit(rng), "r": scale,
                "L": scale * rng.uniform(1.0, 2.5)}
    if family == "halfspace":
        n = rand_unit(rng); return {"n": n, "b": float(n @ mu)}
    if family == "slab":
        n = rand_unit(rng); return {"n": n, "b": float(n @ mu), "hw": scale}
    if family == "cylinder":
        return {"mu": mu, "axis": rand_unit(rng), "r": scale}
    raise ValueError(family)


FIELDS = {"gaussian": f_gaussian, "ellipsoid": f_ellipsoid, "cyl_capped": f_cyl_capped,
          "halfspace": f_halfspace, "slab": f_slab, "cylinder": f_cylinder}

# Field families the generator should use: localized/bounded, FOV-robust.
LOCALIZED = ("gaussian", "ellipsoid", "cyl_capped")


def coords_aabb(co_flat, lab_flat):
    """Coords bounding box over labelled-anatomy voxels: which canonical body
    region a scan covers. Cheap FOV pre-filter for subject grouping."""
    m = lab_flat > 0
    c = co_flat[m]
    return c.min(0), c.max(0)


def soft_hist(lab, w):
    """bg-excluded, soft-weighted anatomy histogram (normalised)."""
    h = np.bincount(lab, weights=w, minlength=256)[:256].astype(np.float64)
    h[0] = 0.0
    s = h.sum()
    return None if s <= 0 else h / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--m", type=int, default=8, help="random instances per config")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ids = subject_ids(); rng.shuffle(ids); ids = ids[:args.pool]
    cache = {s: load(s) for s in ids}
    print(f"pool {len(ids)} subjects | {args.m} instances/config | EDGE {EDGE}mm\n", flush=True)

    configs = ([("gaussian", s) for s in (20, 40, 80, 160)] +
               [("slab", s) for s in (20, 40, 80, 160)] +          # half-width
               [("cylinder", s) for s in (20, 40, 80, 160)] +      # radius
               [("halfspace", None)])

    print(f"{'family':10} {'scale':>6} {'HI mean':>8} {'HI std':>7} {'mass(vox)':>10} {'nsubj':>6}")
    for family, scale in configs:
        fld = FIELDS[family]
        his, masses, nsub = [], [], []
        for _ in range(args.m):
            ref = cache[ids[rng.integers(len(ids))]][0]
            p = sample_params(family, scale, ref, rng)
            hists = []
            for s in ids:
                co, lab = cache[s]
                w = fld(co, p)
                if w.sum() < MIN_MASS:
                    continue
                h = soft_hist(lab, w)
                if h is not None:
                    hists.append(h); masses.append(float(w.sum()))
            if len(hists) < 2:
                continue
            nsub.append(len(hists))
            pair = [float(np.minimum(hists[i], hists[j]).sum())
                    for i in range(len(hists)) for j in range(i + 1, len(hists))]
            his.append(np.mean(pair))
        sc = f"{scale}" if scale is not None else "-"
        print(f"{family:10} {sc:>6} {np.mean(his):8.3f} {np.std(his):7.3f} "
              f"{int(np.mean(masses)):10d} {np.mean(nsub):6.1f}", flush=True)

    print("\nHI = mean pairwise soft-weighted bg-excl label-hist intersection across "
          "pool subjects (higher = same anatomy selected everywhere). mass = soft voxel sum.")


if __name__ == "__main__":
    main()

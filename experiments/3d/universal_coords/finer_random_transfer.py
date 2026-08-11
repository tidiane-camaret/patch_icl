"""Controlled test: does the genuinely finer coords map (factor-3, 1.33/1.33/2.67mm,
from the model) lift the random-ellipsoid transfer correspondence vs the original
4/4/8mm coords? Same cross-patient pairs, same seed, same bg-excluded metric.

For each scan we load coords (original OR finer), then sample totalseg + CT onto
that coords grid via affines. Then run the free-ellipsoid transfer eval.
"""
import json, os
import numpy as np
import nibabel as nib

HERE = os.path.dirname(__file__)
JSON = os.path.join(HERE, "coords_paths_chemotox.json")
FINE = os.path.join(HERE, "coords_predictor", "output_batch")
RMIN, RMAX = 15.0, 50.0


def fine_path(key):
    return os.path.join(FINE, f"{key.replace('#','_')}_coords.nii.gz")


def load_grid(coords_path, ts_path, img_path):
    """coords grid (X,Y,Z,3) + totalseg label grid + CT-on-grid, all on the coords grid."""
    co_img = nib.load(coords_path)
    co = np.asanyarray(co_img.dataobj).astype(np.float32)
    shp = co.shape[:3]
    gx, gy, gz = np.meshgrid(*[np.arange(s) for s in shp], indexing="ij")
    ijk1 = np.stack([gx, gy, gz, np.ones_like(gx)], -1).reshape(-1, 4).T
    world = co_img.affine @ ijk1
    def sample(path, fill):
        im = nib.load(path); dat = np.asanyarray(im.dataobj)
        idx = np.round(np.linalg.inv(im.affine) @ world)[:3].astype(np.int64).T
        sh = np.array(im.shape[:3]); inb = ((idx >= 0) & (idx < sh)).all(1)
        out = np.full(idx.shape[0], fill, np.float32); ii = idx[inb]
        out[inb] = dat[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out.reshape(shp)
    lab = sample(ts_path, 0).astype(np.int32)
    ct = sample(img_path, -1000.0)
    return co, lab, ct


def m_gauss(Q, tgt):
    mu = Q.mean(0); P = np.linalg.inv(np.cov(Q.T) + 1e-3 * np.eye(3))
    dq = np.einsum("ni,ij,nj->n", Q - mu, P, Q - mu)
    d2 = np.einsum("ni,ij,nj->n", tgt - mu, P, tgt - mu)
    return d2 <= np.percentile(dq, 95)


def m_bin(Q, tgt, b=8.0):
    lo = np.minimum(Q.min(0), tgt.min(0))
    qb = np.floor((Q - lo) / b).astype(np.int64); tb = np.floor((tgt - lo) / b).astype(np.int64)
    dims = tb.max(0).clip(qb.max(0)) + 1
    return np.isin(np.ravel_multi_index(tb.T, dims), np.unique(np.ravel_multi_index(qb.T, dims)))


def rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.diag(R))


def label_hist(lab_flat, sel, exclude_bg=True):
    if sel.sum() == 0:
        return None
    u, c = np.unique(lab_flat[sel], return_counts=True)
    h = np.zeros(118); h[u] = c
    if exclude_bg:
        h[0] = 0
    s = h.sum()
    return None if s == 0 else h / s


def hist_int(a, b):
    return 0.0 if a is None or b is None else float(np.minimum(a, b).sum())


def build(key, source):
    """source='orig' uses JSON coords; 'fine' uses output_batch finer coords."""
    p = json.load(open(JSON))[key]
    cpath = p["coords"] if source == "orig" else fine_path(key)
    co, lab, ct = load_grid(cpath, p["totalseg"], p["img"])
    gmm = np.array([np.linalg.norm(nib.load(cpath).affine[:3, i]) for i in range(3)])  # grid spacing mm
    g = np.stack(np.meshgrid(*[np.arange(s) for s in co.shape[:3]], indexing="ij"), -1).astype(np.float32)
    centers = np.argwhere(ct > -300)
    return dict(co=co, lab=lab.reshape(-1), body=ct > -300, gmm=gmm, g=g, centers=centers,
                tgt=co.reshape(-1, 3))


def rand_ellipsoid(S, rng):
    c = S["centers"][rng.integers(len(S["centers"]))].astype(float)
    r = rng.uniform(RMIN, RMAX, 3) / S["gmm"]
    d = (S["g"] - c) @ rand_rot(rng).T
    region = ((d / r) ** 2).sum(-1) <= 1.0
    return region.reshape(-1), S["co"][region]


def evaluate(source, pairs, keys, n_per_pair=20, seed=0):
    rng = np.random.default_rng(seed)
    agg = {"bin": [], "gauss": [], "random": []}
    cache = {}
    for ci, ti in pairs:
        C = cache.setdefault(ci, build(keys[ci], source))
        T = cache.setdefault(ti, build(keys[ti], source))
        for _ in range(n_per_pair):
            reg, Q = rand_ellipsoid(C, rng)
            if (C["lab"][reg] > 0).mean() < 0.3:
                continue
            hc = label_hist(C["lab"], reg)
            rreg, _ = rand_ellipsoid(T, rng)
            agg["bin"].append(hist_int(hc, label_hist(T["lab"], m_bin(Q, T["tgt"]))))
            agg["gauss"].append(hist_int(hc, label_hist(T["lab"], m_gauss(Q, T["tgt"]))))
            agg["random"].append(hist_int(hc, label_hist(T["lab"], rreg)))
    return agg


def main():
    paths = json.load(open(JSON))
    keys = list(paths)[:20]
    pid = [k.split("#")[0] for k in keys]
    # distinct-patient pairs among the 20 scans that have finer coords
    pairs, used = [], set()
    for i in range(20):
        for j in range(i + 1, 20):
            if pid[i] != pid[j] and os.path.exists(fine_path(keys[i])) and os.path.exists(fine_path(keys[j])):
                if i not in used and j not in used:
                    pairs.append((i, j)); used |= {i, j}
        if len(pairs) >= 8:
            break
    print(f"pairs (distinct patients): {[(pid[i],pid[j]) for i,j in pairs]}")
    for source, mm in [("orig", "4/4/8mm"), ("fine", "1.33/1.33/2.67mm")]:
        a = evaluate(source, pairs, keys)
        n = len(a["bin"])
        print(f"\n=== {source} ({mm}) | {n} scored samples ===")
        for m in ["bin", "gauss", "random"]:
            v = np.array(a[m]); print(f"  {m:7} {v.mean():.3f} ± {v.std():.3f}")


if __name__ == "__main__":
    main()

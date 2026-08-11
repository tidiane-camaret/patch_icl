"""Free (label-agnostic) ellipsoid transfer: draw an ellipsoid at a RANDOM body
position with RANDOM radii (within mm bounds) and random orientation on the ctx
scan, then transfer it to the tgt scan through coords with bin/gauss.

No label ties the ellipsoid, so evaluate anatomical correspondence indirectly:
the ctx ellipsoid covers a mix of totalseg structures; a good transfer lands on
the SAME mix in tgt. Metric = totalseg label-histogram intersection (0..1) of
ctx-region vs tgt-region, compared to a random-placement baseline (chance level).
"""
import json, os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_invariance import voxels, pick_keys, JSON, FIGDIR
from transfer_methods import render_sel

RMIN, RMAX = 15.0, 50.0        # ellipsoid radius bounds (mm)
GRID_MM = np.array([3.996, 3.996, 8.013])   # coords-grid spacing


def load_grid(p):
    """coords grid (X,Y,Z,3), totalseg label grid, CT-on-grid (for body mask)."""
    co_img = nib.load(p["coords"]); ts_img = nib.load(p["totalseg"])
    shp = co_img.shape[:3]
    cv, lab = voxels(p)
    co = cv.reshape(*shp, 3); lab = lab.reshape(shp)
    # sample CT at grid world points for a body mask
    gx, gy, gz = np.meshgrid(*[np.arange(s) for s in shp], indexing="ij")
    ijk1 = np.stack([gx, gy, gz, np.ones_like(gx)], -1).reshape(-1, 4).T
    idx = np.round(np.linalg.inv(nib.load(p["img"]).affine) @ (co_img.affine @ ijk1))[:3].astype(np.int64).T
    ct_full = nib.load(p["img"])
    cshp = np.array(ct_full.shape[:3]); inb = ((idx >= 0) & (idx < cshp)).all(1)
    ct = np.full(idx.shape[0], -1000.0, np.float32)
    ii = idx[inb]; ct[inb] = np.asanyarray(ct_full.dataobj)[ii[:, 0], ii[:, 1], ii[:, 2]]
    return co, lab, ct.reshape(shp)


def rand_rotation(rng):
    A = rng.normal(size=(3, 3)); Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def rand_ellipsoid(co, body, rng):
    """Random-center, random-radius, random-orientation ellipsoid on the grid."""
    centers = np.argwhere(body)
    c = centers[rng.integers(len(centers))].astype(float)
    r_grid = rng.uniform(RMIN, RMAX, 3) / GRID_MM
    Rm = rand_rotation(rng)
    g = np.stack(np.meshgrid(*[np.arange(s) for s in co.shape[:3]], indexing="ij"), -1).astype(float)
    d = (g - c) @ Rm.T
    region = ((d / r_grid) ** 2).sum(-1) <= 1.0
    return region, co[region]


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


def label_hist(lab_flat, sel, exclude_bg=False):
    """Normalized totalseg-label histogram over selected voxels."""
    if sel.sum() == 0:
        return None
    u, c = np.unique(lab_flat[sel], return_counts=True)
    h = np.zeros(118); h[u] = c
    if exclude_bg:
        h[0] = 0
    s = h.sum()
    return None if s == 0 else h / s


def hist_intersection(ha, hb):
    if ha is None or hb is None:
        return 0.0
    return float(np.minimum(ha, hb).sum())


def evaluate(pairs, n_per_pair=40, seed=0, min_nonbg=0.3):
    """bg-EXCLUDED label-hist intersection; only score blobs whose ctx region is
    >=min_nonbg labeled anatomy (else 'anatomy' is just fat, metric meaningless)."""
    paths = json.load(open(JSON)); keys = pick_keys(paths, 200, unique=True)
    rng = np.random.default_rng(seed)
    agg = {"bin": [], "gauss": [], "random": []}
    empty = {"bin": 0, "gauss": 0}; tot = 0; skipped = 0
    for ci, ti in pairs:
        co_c, lab_c, ct_c = load_grid(paths[keys[ci]])
        co_t, lab_t, ct_t = load_grid(paths[keys[ti]])
        body_c = ct_c > -300; body_t = ct_t > -300
        tgt = co_t.reshape(-1, 3); lab_tf = lab_t.reshape(-1); lab_cf = lab_c.reshape(-1)
        for _ in range(n_per_pair):
            region, Q = rand_ellipsoid(co_c, body_c, rng)
            rf = region.reshape(-1)
            if (lab_cf[rf] > 0).mean() < min_nonbg:      # mostly fat -> skip
                skipped += 1; continue
            hc = label_hist(lab_cf, rf, exclude_bg=True)
            sb = m_bin(Q, tgt); sg = m_gauss(Q, tgt)
            rregion, _ = rand_ellipsoid(co_t, body_t, rng)          # chance baseline
            agg["bin"].append(hist_intersection(hc, label_hist(lab_tf, sb, exclude_bg=True)))
            agg["gauss"].append(hist_intersection(hc, label_hist(lab_tf, sg, exclude_bg=True)))
            agg["random"].append(hist_intersection(hc, label_hist(lab_tf, rregion.reshape(-1), exclude_bg=True)))
            empty["bin"] += sb.sum() == 0; empty["gauss"] += sg.sum() == 0; tot += 1
    print(f"random ellipsoids: {tot} scored ({skipped} skipped <{min_nonbg} labeled), "
          f"radius {RMIN}-{RMAX}mm, free pos+orient | bg-EXCLUDED metric")
    print(f"{'method':8} {'labeled-hist intersection (mean±sd)':34} {'empty tgt %':>11}")
    for m in ["bin", "gauss", "random"]:
        a = np.array(agg[m])
        e = 100 * empty.get(m, 0) / tot if m != "random" else 0.0
        print(f"{m:8} {a.mean():.3f} ± {a.std():.3f}{'':20} {e:11.1f}")
    return agg


def figure(pair=(0, 7), seed=3, n=3):
    paths = json.load(open(JSON)); keys = pick_keys(paths, 200, unique=True)
    ci, ti = pair
    co_c, lab_c, ct_c = load_grid(paths[keys[ci]])
    co_t, lab_t, ct_t = load_grid(paths[keys[ti]])
    body_c = ct_c > -300; tgt = co_t.reshape(-1, 3); lab_tf = lab_t.reshape(-1); lab_cf = lab_c.reshape(-1)
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(n, 3, figsize=(15, 5 * n))
    for row in range(n):
        region, Q = rand_ellipsoid(co_c, body_c, rng)
        hc = label_hist(lab_cf, region.reshape(-1))
        sels = {"CTX random ellipsoid": (keys[ci], co_c.shape[:3], region),
                "TGT bin": (keys[ti], co_t.shape[:3], m_bin(Q, tgt).reshape(co_t.shape[:3])),
                "TGT gauss": (keys[ti], co_t.shape[:3], m_gauss(Q, tgt).reshape(co_t.shape[:3]))}
        for col, (ttl, (key, shp, sel)) in enumerate(sels.items()):
            ax = axs[row, col] if n > 1 else axs[col]
            ct, mask, kct = render_sel(paths[key], shp, sel)
            ax.imshow(np.clip(ct, -160, 240), cmap="gray")
            ax.imshow(np.ma.masked_equal(mask, 0), cmap="autumn", alpha=0.55, interpolation="nearest")
            extra = ""
            if col > 0:
                extra = f"  hist∩={hist_intersection(hc, label_hist(lab_tf, sel.reshape(-1))):.2f}"
            ax.set_title(f"{ttl}{extra}", fontsize=10); ax.axis("off")
    fig.suptitle("Free random ellipsoid (ctx) transferred to tgt via coords", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "random_ellipsoid_transfer.png")
    fig.savefig(out, dpi=95, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    pairs = [(2 * k, 2 * k + 1) for k in range(16)]   # 16 pairs, 32 distinct patients
    evaluate(pairs, n_per_pair=40)
    figure()

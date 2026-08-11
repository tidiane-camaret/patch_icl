"""Transfer an ellipsoid drawn on a CONTEXT scan to a TARGET scan through the
coords field, and benchmark cheap ways to do the transfer.

Pipeline: ellipsoid is defined in ctx IMAGE space (grid-index ellipsoid around an
organ) -> its voxels' coords values form an irregular cloud Q -> select tgt voxels
whose coords match Q. Four matchers:
  base  : analytic axis-aligned ellipsoid in coords space (prev approach, no cloud)
  gauss : Mahalanobis fit (mean + full 3x3 cov of Q), rotation-aware
  bin   : quantize coords to b-mm bins, keep tgt voxels in ctx-occupied bins
  knn   : KD-tree on Q, keep tgt voxels within tau of any ctx point
Quality = Dice/purity of transferred tgt mask vs the tgt organ (totalseg==L).
"""
import json, os, time
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_invariance import voxels, pick_keys, name, JSON, FIGDIR

K = 2.0            # ctx ellipsoid radius = K * organ index-std per axis
MIN_VOX = 60


def load_grid(p):
    """coords grid (X,Y,Z,3) + aligned totalseg label grid (X,Y,Z)."""
    shp = nib.load(p["coords"]).shape[:3]
    cv, lab = voxels(p)
    return cv.reshape(*shp, 3), lab.reshape(shp), shp


def ctx_cloud(co, lab, L, use_organ=False):
    """Ctx region -> coords cloud Q (M,3). Ellipsoid around organ L, or (use_organ)
    the real organ footprint itself (non-ellipsoidal source shape)."""
    idx = np.argwhere(lab == L)
    if idx.shape[0] < MIN_VOX:
        return None, None
    if use_organ:
        region = lab == L
        return co[region], region
    c = idx.mean(0); r = K * idx.std(0) + 1e-3
    gx, gy, gz = np.meshgrid(*[np.arange(s) for s in co.shape[:3]], indexing="ij")
    g = np.stack([gx, gy, gz], -1)
    region = (((g - c) / r) ** 2).sum(-1) <= 1.0        # ellipsoid on ctx grid
    return co[region], region


# ---- matchers: (ctx_coords_cloud Q, tgt_coords_flat) -> tgt bool (N,) ----

def m_gauss(Q, tgt):
    mu = Q.mean(0)
    cov = np.cov(Q.T) + 1e-3 * np.eye(3)
    P = np.linalg.inv(cov)
    d = tgt - mu
    d2 = np.einsum("ni,ij,nj->n", d, P, d)
    dq = np.einsum("ni,ij,nj->n", Q - mu, P, Q - mu)    # match ctx cloud extent
    return d2 <= np.percentile(dq, 95)


def m_bin(Q, tgt, b=8.0):
    lo = np.minimum(Q.min(0), tgt.min(0))
    qb = np.floor((Q - lo) / b).astype(np.int64)
    tb = np.floor((tgt - lo) / b).astype(np.int64)
    dims = tb.max(0).clip(qb.max(0)) + 1
    qh = np.ravel_multi_index(qb.T, dims)
    th = np.ravel_multi_index(tb.T, dims)
    return np.isin(th, np.unique(qh))


def m_knn(Q, tgt, tau=8.0):
    d, _ = cKDTree(Q).query(tgt, k=1)
    return d <= tau


def m_base(co_ctx, lab_ctx, L, tgt):
    """Analytic axis-aligned ellipsoid from ctx organ coords (no cloud transfer)."""
    m = lab_ctx.reshape(-1) == L
    cv = co_ctx.reshape(-1, 3)[m]
    c = cv.mean(0); r = K * cv.std(0) + 1e-3
    return (((tgt - c) / r) ** 2).sum(1) <= 1.0


def dice_purity(sel, gt):
    inter = (sel & gt).sum()
    dice = 2 * inter / (sel.sum() + gt.sum() + 1e-9)
    pur = inter / (sel.sum() + 1e-9)
    return dice, pur


def evaluate(labels, pairs, use_organ=False):
    paths = json.load(open(JSON))
    keys = pick_keys(paths, 200, unique=True)
    agg = {m: {"dice": [], "pur": [], "t": []} for m in ["base", "gauss", "bin", "knn"]}
    for (ci, ti) in pairs:
        gc = load_grid(paths[keys[ci]])
        gt = load_grid(paths[keys[ti]])
        co_c, lab_c, _ = gc
        co_t, lab_t, _ = gt
        tgt = co_t.reshape(-1, 3)
        for L in labels:
            Q, _ = ctx_cloud(co_c, lab_c, L, use_organ)
            gtm = lab_t.reshape(-1) == L
            if Q is None or gtm.sum() < MIN_VOX:
                continue
            runs = {"base": lambda: m_base(co_c, lab_c, L, tgt),
                    "gauss": lambda: m_gauss(Q, tgt),
                    "bin": lambda: m_bin(Q, tgt),
                    "knn": lambda: m_knn(Q, tgt)}
            for mn, fn in runs.items():
                t0 = time.perf_counter(); sel = fn(); dt = time.perf_counter() - t0
                d, p = dice_purity(sel, gtm)
                agg[mn]["dice"].append(d); agg[mn]["pur"].append(p); agg[mn]["t"].append(dt)
    print(f"{'method':6} {'dice':>6} {'purity':>7} {'ms/transfer':>12}")
    for mn, v in agg.items():
        print(f"{mn:6} {np.mean(v['dice']):6.3f} {np.mean(v['pur']):7.3f} "
              f"{1e3*np.mean(v['t']):12.1f}")


def figure(label, ci, ti):
    paths = json.load(open(JSON))
    keys = pick_keys(paths, 200, unique=True)
    co_c, lab_c, _ = load_grid(paths[keys[ci]])
    co_t, lab_t, shp = load_grid(paths[keys[ti]])
    Q, region = ctx_cloud(co_c, lab_c, label)
    tgt = co_t.reshape(-1, 3)
    sels = {"ctx ellipsoid": region.reshape(-1),
            "gauss": m_gauss(Q, tgt), "bin": m_bin(Q, tgt), "knn": m_knn(Q, tgt)}
    imgs = {"ctx ellipsoid": (keys[ci], co_c), "gauss": (keys[ti], co_t),
            "bin": (keys[ti], co_t), "knn": (keys[ti], co_t)}
    gtm = (lab_t == label).reshape(-1)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, (ttl, sel) in zip(axs, sels.items()):
        key, co = imgs[ttl]
        ct, mask, kct = render_sel(paths[key], co.shape[:3], sel.reshape(shp if key == keys[ti] else co.shape[:3]))
        ax.imshow(np.clip(ct, -160, 240), cmap="gray")
        ax.imshow(np.ma.masked_equal(mask, 0), cmap="autumn", alpha=0.55, interpolation="nearest")
        extra = ""
        if ttl != "ctx ellipsoid":
            d, p = dice_purity(sel, gtm); extra = f"  D={d:.2f} P={p:.2f}"
        who = "CTX" if ttl == "ctx ellipsoid" else "TGT"
        ax.set_title(f"{who}: {ttl}{extra}", fontsize=10); ax.axis("off")
    fig.suptitle(f"Ellipsoid transfer via coords — {name(label)}", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "transfer_methods.png")
    fig.savefig(out, dpi=95, bbox_inches="tight"); print("saved", out)


def render_sel(p, shp, sel_grid):
    """Overlay a coords-grid boolean mask onto its CT slice (max-coverage slice)."""
    co_img = nib.load(p["coords"]); ts_img = nib.load(p["img"])
    A_co, A_ct = co_img.affine, ts_img.affine
    ct_shape = ts_img.shape[:3]
    gk = np.argwhere(sel_grid)
    wz = A_co[2, 2] * gk[:, 2] + A_co[2, 3]
    kct = int(np.clip(np.median((wz - A_ct[2, 3]) / A_ct[2, 2]), 0, ct_shape[2] - 1))
    ct = np.asanyarray(ts_img.dataobj[:, :, kct]).astype(np.float32)
    i = np.arange(ct_shape[0]); j = np.arange(ct_shape[1])
    ci = np.clip(np.round((A_ct[0, 0] * i + A_ct[0, 3] - A_co[0, 3]) / A_co[0, 0]).astype(int), 0, shp[0] - 1)
    cj = np.clip(np.round((A_ct[1, 1] * j + A_ct[1, 3] - A_co[1, 3]) / A_co[1, 1]).astype(int), 0, shp[1] - 1)
    ck = int(np.clip(round((A_ct[2, 2] * kct + A_ct[2, 3] - A_co[2, 3]) / A_co[2, 2]), 0, shp[2] - 1))
    mask = sel_grid[np.ix_(ci, cj, [ck])][:, :, 0]
    return np.rot90(ct), np.rot90(mask), kct


if __name__ == "__main__":
    labels = [5, 1, 2, 3, 52, 21, 6, 7]          # liver spleen kidneys aorta bladder stomach pancreas
    pairs = [(0, 7), (3, 12), (5, 20), (9, 15), (1, 30), (8, 25)]
    print("=== source = ELLIPSOID around organ ===")
    evaluate(labels, pairs, use_organ=False)
    print("\n=== source = REAL ORGAN footprint ===")
    evaluate(labels, pairs, use_organ=True)
    figure(5, 0, 7)

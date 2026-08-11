"""Option-1 test: trilinear-upsample the existing 4/8mm coords onto finer grids
(no model re-run), then measure ellipsoid-transfer Dice vs ellipsoid size (K) for
matchers base/gauss/bin (knn excluded). Question: how much of the ~0.27 Dice
ceiling was grid quantization vs true coords limits?

All three matchers transfer the SAME ctx ellipsoid cloud Q (ellipsoid drawn on ctx
in fine-grid image space, radius K*index-std); they differ only in how they select
tgt voxels whose coords match Q. GT = totalseg sampled on the same fine grid.
"""
import json, os
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_invariance import pick_keys, name, JSON, FIGDIR

MIN_VOX = 40


def build_fine(p, f):
    """Trilinear-upsample coords by factor f; return co_fine (X,Y,Z,3) + totalseg
    label sampled on that fine grid (X,Y,Z)."""
    co_img = nib.load(p["coords"]); ts_img = nib.load(p["totalseg"])
    co = np.asanyarray(co_img.dataobj).astype(np.float32)
    S = co.shape[:3]
    axes = [np.linspace(0, s - 1, int(round(s * f))) for s in S]      # fractional orig-indices
    U = np.stack(np.meshgrid(*axes, indexing="ij"))                  # (3, X,Y,Z)
    co_f = np.stack([map_coordinates(co[..., c], U, order=1) for c in range(3)], -1)
    # sample totalseg at fine-grid world positions
    flat = U.reshape(3, -1)
    world = co_img.affine @ np.vstack([flat, np.ones(flat.shape[1])])
    idx = np.round(np.linalg.inv(ts_img.affine) @ world)[:3].astype(np.int64).T
    shp = np.array(ts_img.shape[:3])
    inb = ((idx >= 0) & (idx < shp)).all(1)
    ts = np.asanyarray(ts_img.dataobj)
    lab = np.zeros(idx.shape[0], np.int32)
    ii = idx[inb]; lab[inb] = ts[ii[:, 0], ii[:, 1], ii[:, 2]]
    return co_f, lab.reshape(co_f.shape[:3])


def ctx_cloud(co, lab, L, K):
    """Ellipsoid in fine-grid index space around organ L -> coords cloud Q."""
    idx = np.argwhere(lab == L)
    if idx.shape[0] < MIN_VOX:
        return None
    c = idx.mean(0); r = K * idx.std(0) + 1e-3
    g = np.stack(np.meshgrid(*[np.arange(s) for s in co.shape[:3]], indexing="ij"), -1)
    region = (((g - c) / r) ** 2).sum(-1) <= 1.0
    return co[region]


# --- matchers: same cloud Q, differ only in selection of tgt (N,3) ---
def m_base(Q, tgt):                       # axis-aligned ellipsoid (diagonal cov)
    mu = Q.mean(0); sd = Q.std(0) + 1e-3
    dq = (((Q - mu) / sd) ** 2).sum(1)
    return (((tgt - mu) / sd) ** 2).sum(1) <= np.percentile(dq, 95)

def m_gauss(Q, tgt):                       # full 3x3 cov, Mahalanobis
    mu = Q.mean(0); P = np.linalg.inv(np.cov(Q.T) + 1e-3 * np.eye(3))
    dq = np.einsum("ni,ij,nj->n", Q - mu, P, Q - mu)
    d2 = np.einsum("ni,ij,nj->n", tgt - mu, P, tgt - mu)
    return d2 <= np.percentile(dq, 95)

def m_bin(Q, tgt, b=8.0):                  # coords-bin hashing
    lo = np.minimum(Q.min(0), tgt.min(0))
    qb = np.floor((Q - lo) / b).astype(np.int64)
    tb = np.floor((tgt - lo) / b).astype(np.int64)
    dims = tb.max(0).clip(qb.max(0)) + 1
    return np.isin(np.ravel_multi_index(tb.T, dims), np.unique(np.ravel_multi_index(qb.T, dims)))

MATCH = {"base": m_base, "gauss": m_gauss, "bin": m_bin}


def dice(sel, gt):
    return 2 * (sel & gt).sum() / (sel.sum() + gt.sum() + 1e-9)


def main():
    paths = json.load(open(JSON))
    keys = pick_keys(paths, 200, unique=True)
    pairs = [(0, 7), (3, 12), (5, 20)]
    labels = [5, 1, 2, 3, 52, 21, 6, 7]       # liver spleen kidneys aorta bladder stomach pancreas
    factors = [1, 2]                           # 4/8mm, 2/4mm
    Ks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # res[f][method][K] = list of dice
    res = {f: {m: {k: [] for k in Ks} for m in MATCH} for f in factors}

    for f in factors:
        for ci, ti in pairs:
            co_c, lab_c = build_fine(paths[keys[ci]], f)
            co_t, lab_t = build_fine(paths[keys[ti]], f)
            tgt = co_t.reshape(-1, 3)
            for L in labels:
                gt = (lab_t == L).reshape(-1)
                if gt.sum() < MIN_VOX:
                    continue
                for K in Ks:
                    Q = ctx_cloud(co_c, lab_c, L, K)
                    if Q is None:
                        continue
                    for m, fn in MATCH.items():
                        res[f][m][K].append(dice(fn(Q, tgt), gt))
        print(f"factor {f} done")

    # table
    print(f"\n{'factor':>6} {'method':6} " + " ".join(f"K={k:<4}" for k in Ks))
    for f in factors:
        for m in MATCH:
            row = " ".join(f"{np.mean(res[f][m][k]):5.3f}" for k in Ks)
            print(f"{f:>6} {m:6} {row}")

    # plot: Dice vs K, one subplot per factor, one line per method
    fig, axs = plt.subplots(1, len(factors), figsize=(6 * len(factors), 5), sharey=True)
    vox = {1: "4/4/8mm", 2: "2/2/4mm", 3: "1.3/1.3/2.7mm"}
    for ax, f in zip(axs, factors):
        for m in MATCH:
            ax.plot(Ks, [np.mean(res[f][m][k]) for k in Ks], "-o", label=m)
        ax.set_title(f"upsample x{f}  ({vox[f]})"); ax.set_xlabel("ellipsoid K (radius = K*std)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Dice vs tgt organ"); axs[0].legend()
    fig.suptitle("Ellipsoid transfer Dice vs size, on trilinear-upsampled coords")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "finer_transfer_dice.png")
    fig.savefig(out, dpi=95); print("saved", out)


if __name__ == "__main__":
    main()

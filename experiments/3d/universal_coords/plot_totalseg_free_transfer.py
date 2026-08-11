"""Visualise FREELY-positioned ellipsoid transfer on TotalSeg (the earlier
random_ellipsoid_transfer recipe, not organ-centred). Each blob: random body
position + random radii 15-50mm + random orientation on CTX -> coords cloud Q ->
transfer to TGT via bin/gauss. Correspondence = bg-excluded label-histogram
intersection between the ctx blob and the tgt transferred region.

Rows = random blobs. Cols: CTX blob | TGT bin (HI) | TGT gauss (HI).
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from totalseg_ellipsoid_transfer import TS, CO, DS, load, m_bin, m_gauss

FIGS = os.path.join(os.path.dirname(__file__), "figs")
CTW = (-160, 240)
GMM = 1.5 * DS            # eval grid spacing (mm) after stride-DS downsample
RMIN, RMAX = 15.0, 50.0


def ct_slice(sid, k):
    ct = np.asanyarray(nib.load(os.path.join(TS, sid, "ct.nii.gz")).dataobj)[::DS, ::DS, ::DS]
    return ct[:, :, k]


def best_k(mask3d):
    return int(mask3d.sum((0, 1)).argmax())


def rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.diag(R))


def rand_ellipsoid(co, body_idx, gshape, rng):
    c = body_idx[rng.integers(len(body_idx))].astype(float)
    r = rng.uniform(RMIN, RMAX, 3) / GMM
    g = np.stack(np.meshgrid(*[np.arange(s) for s in gshape], indexing="ij"), -1).astype(np.float32)
    d = (g - c) @ rand_rot(rng).T
    region = ((d / r) ** 2).sum(-1) <= 1.0
    return region, co[region]


def label_hist(lab_flat, sel):
    if sel.sum() == 0:
        return None
    u, cnt = np.unique(lab_flat[sel], return_counts=True)
    h = np.zeros(256); h[u] = cnt; h[0] = 0        # bg-excluded
    s = h.sum()
    return None if s == 0 else h / s


def hist_int(a, b):
    return 0.0 if a is None or b is None else float(np.minimum(a, b).sum())


def panel(ax, sid, k, overlay, title, color="autumn", gt=None):
    ax.imshow(np.rot90(np.clip(ct_slice(sid, k), *CTW)), cmap="gray")
    ax.imshow(np.rot90(np.ma.masked_equal(overlay[:, :, k], 0)), cmap=color, alpha=0.55,
              interpolation="nearest")
    if gt is not None and gt[:, :, k].any():
        ax.contour(np.rot90(gt[:, :, k]).astype(float), levels=[0.5], colors="lime", linewidths=1.2)
    ax.set_title(title, fontsize=10); ax.axis("off")


def main(ci="s0040", ti="s0667", n=5, seed=1):
    os.makedirs(FIGS, exist_ok=True)
    rng = np.random.default_rng(seed)
    co_c, lab_c = load(ci); co_t, lab_t = load(ti)
    shp = lab_t.shape; tgt = co_t.reshape(-1, 3)
    lab_cf = lab_c.reshape(-1); lab_tf = lab_t.reshape(-1)
    body = np.argwhere(np.asanyarray(nib.load(os.path.join(TS, ci, "ct.nii.gz")).dataobj)[::DS, ::DS, ::DS] > -300)

    rows = []
    while len(rows) < n:
        reg, Q = rand_ellipsoid(co_c, body, lab_c.shape, rng)
        if (lab_c[reg] > 0).mean() < 0.3:            # blob must sit >=30% on labelled anatomy
            continue
        hc = label_hist(lab_cf, reg.reshape(-1))
        selb = m_bin(Q, tgt).reshape(shp); selg = m_gauss(Q, tgt).reshape(shp)
        hib = hist_int(hc, label_hist(lab_tf, selb.reshape(-1)))
        hig = hist_int(hc, label_hist(lab_tf, selg.reshape(-1)))
        # positional GT on tgt = union of the organs the ctx blob covers (>=10% each)
        ctx_labels = np.nonzero(hc >= 0.10)[0]
        gt_t = np.isin(lab_t, ctx_labels)
        rows.append((reg, selb, selg, hib, hig, gt_t))

    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axs = axs[None, :]
    for row, (reg, selb, selg, hib, hig, gt_t) in zip(axs, rows):
        kc = best_k(reg); kt = best_k(selb) if selb.sum() else best_k(selg)
        panel(row[0], ci, kc, reg, f"CTX {ci}: free blob")
        panel(row[1], ti, kt, selb, f"TGT {ti}: bin  HI={hib:.2f}", gt=gt_t)
        panel(row[2], ti, kt, selg, f"TGT {ti}: gauss  HI={hig:.2f}", gt=gt_t)
    fig.suptitle(f"Free-positioned ellipsoid transfer  {ci} -> {ti}  "
                 f"(green = ctx-covered organs on tgt; HI = bg-excl label-hist intersection)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGS, f"totalseg_free_transfer_{ci}_{ti}.png")
    fig.savefig(out, dpi=95, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()

"""Visualise ctx-ellipsoid -> tgt-transfer on TotalSeg. For a few organs, draw an
ellipsoid around the organ on the CTX case, transfer its coords cloud to the TGT
case (bin + gauss), and overlay everything on axial CT slices. coords/ct/label
share the 1.5mm-iso grid, so no affine bridge.

Rows = organs. Cols: CTX ellipsoid | TGT bin (D) | TGT gauss (D) | TGT GT organ.
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from totalseg_ellipsoid_transfer import (TS, CO, DS, K, load, ctx_cloud,
                                         m_bin, m_gauss, dice_pur)

FIGS = os.path.join(os.path.dirname(__file__), "figs")
CTW = (-160, 240)   # CT window


def ct_slice(sid, k):
    ct = np.asanyarray(nib.load(os.path.join(TS, sid, "ct.nii.gz")).dataobj)[::DS, ::DS, ::DS]
    return ct[:, :, k]


def best_k(mask3d):
    return int(mask3d.reshape(mask3d.shape[0] * mask3d.shape[1], -1).sum(0).argmax())


def panel(ax, sid, k, overlay, title, color="autumn"):
    ax.imshow(np.rot90(np.clip(ct_slice(sid, k), *CTW)), cmap="gray")
    ax.imshow(np.rot90(np.ma.masked_equal(overlay[:, :, k], 0)), cmap=color, alpha=0.55,
              interpolation="nearest")
    ax.set_title(title, fontsize=10); ax.axis("off")


def main(ci="s0040", ti="s0667", organs=(5, 6, 1, 22)):
    os.makedirs(FIGS, exist_ok=True)
    co_c, lab_c = load(ci); co_t, lab_t = load(ti)
    shp = lab_t.shape; tgt = co_t.reshape(-1, 3)
    rows = []
    for L in organs:
        Q = ctx_cloud(co_c, lab_c, L)
        if Q is None or (lab_t == L).sum() < 60:
            continue
        # ctx ellipsoid region (recompute for display)
        idx = np.argwhere(lab_c == L); c = idx.mean(0); r = K * idx.std(0) + 1e-3
        g = np.stack(np.meshgrid(*[np.arange(s) for s in lab_c.shape], indexing="ij"), -1)
        ctx_reg = ((((g - c) / r) ** 2).sum(-1) <= 1.0)
        gtm = (lab_t == L)
        selb = m_bin(Q, tgt).reshape(shp); selg = m_gauss(Q, tgt).reshape(shp)
        db, _ = dice_pur(selb.reshape(-1), gtm.reshape(-1))
        dg, _ = dice_pur(selg.reshape(-1), gtm.reshape(-1))
        rows.append((L, ctx_reg, selb, selg, gtm, db, dg))

    fig, axs = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
    if len(rows) == 1:
        axs = axs[None, :]
    for row, (L, ctx_reg, selb, selg, gtm, db, dg) in zip(axs, rows):
        kc = best_k(ctx_reg); kt = best_k(gtm)
        panel(row[0], ci, kc, ctx_reg, f"CTX {ci} L{L}: ellipsoid")
        panel(row[1], ti, kt, selb, f"TGT {ti}: bin  D={db:.2f}")
        panel(row[2], ti, kt, selg, f"TGT {ti}: gauss  D={dg:.2f}")
        panel(row[3], ti, kt, gtm, f"TGT {ti}: GT organ", color="winter")
    fig.suptitle(f"Coords ellipsoid transfer  {ci} -> {ti}", fontsize=14)
    fig.tight_layout()
    out = os.path.join(FIGS, f"totalseg_transfer_{ci}_{ti}.png")
    fig.savefig(out, dpi=95, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()

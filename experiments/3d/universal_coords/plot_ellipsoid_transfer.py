"""Visual proof of coords invariance: define an ellipsoid in coords-VALUE space
from a context subject's organ, draw it on that subject, then apply the IDENTICAL
coords-space ellipsoid to a target subject. If coords is a shared body frame the
ellipsoid lands on the same anatomy in both.

Each row = one example: [context CT + ellipsoid] | [target CT + ellipsoid].
"""
import json, os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coord_invariance import voxels, pick_keys, name, JSON, FIGDIR

K = 2.0            # ellipsoid radius = K * within-subject std per axis
MIN_VOX = 60


def ellipsoid_from(key, paths, label):
    """coords-space centroid + per-axis radius from one subject's organ."""
    cv, lab = voxels(paths[key])
    m = lab == label
    if m.sum() < MIN_VOX:
        return None
    c = cv[m].mean(0)
    r = K * cv[m].std(0) + 1e-3
    return c, r


def render(key, paths, c, r):
    """Return (ct_slice, mask_slice) at the CT slice with most ellipsoid voxels."""
    co_img = nib.load(paths[key]["coords"])
    ts_img = nib.load(paths[key]["totalseg"])          # same grid/affine as CT img
    co = np.asanyarray(co_img.dataobj).astype(np.float32)
    sel = (((co - c) / r) ** 2).sum(-1) <= 1.0         # (X,Y,Z) ellipsoid on coords grid
    if sel.sum() == 0:
        return None
    A_co, A_ct = co_img.affine, ts_img.affine
    ct_shape = ts_img.shape[:3]

    # pick CT axial slice = median CT-k of selected coords voxels
    gk = np.argwhere(sel)                              # (M,3) coords-grid idx
    world_z = A_co[2, 2] * gk[:, 2] + A_co[2, 3]
    kct = int(np.median((world_z - A_ct[2, 3]) / A_ct[2, 2]))
    kct = int(np.clip(kct, 0, ct_shape[2] - 1))

    ct = np.asanyarray(nib.load(paths[key]["img"]).dataobj[:, :, kct]).astype(np.float32)

    # resample coords-grid ellipsoid onto this CT slice (diagonal affines -> 1D maps)
    i = np.arange(ct_shape[0]); j = np.arange(ct_shape[1])
    ci = np.round((A_ct[0, 0] * i + A_ct[0, 3] - A_co[0, 3]) / A_co[0, 0]).astype(int)
    cj = np.round((A_ct[1, 1] * j + A_ct[1, 3] - A_co[1, 3]) / A_co[1, 1]).astype(int)
    ck = int(round((A_ct[2, 2] * kct + A_ct[2, 3] - A_co[2, 3]) / A_co[2, 2]))
    ci = np.clip(ci, 0, co.shape[0] - 1); cj = np.clip(cj, 0, co.shape[1] - 1)
    ck = int(np.clip(ck, 0, co.shape[2] - 1))
    mask = sel[np.ix_(ci, cj, [ck])][:, :, 0]
    return np.rot90(ct), np.rot90(mask), kct


def main():
    paths = json.load(open(JSON))
    keys = pick_keys(paths, 200, unique=True)
    # (label, context_key, target_key) examples across distinct organs/patients
    examples = [
        (5,  keys[0],  keys[7]),    # liver
        (1,  keys[3],  keys[12]),   # spleen
        (2,  keys[5],  keys[20]),   # kidney_right
        (52, keys[9],  keys[15]),   # aorta
    ]
    n = len(examples)
    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))
    for row, (lab, ck, tk) in enumerate(examples):
        ell = ellipsoid_from(ck, paths, lab)
        if ell is None:
            print(f"row {row}: label {lab} absent in context {ck}, skipping")
            continue
        c, r = ell
        for col, key in enumerate([ck, tk]):
            out = render(key, paths, c, r)
            ax = axs[row, col]
            if out is None:
                ax.set_title(f"{name(lab)} not found in {key.split('#')[0]}"); ax.axis("off"); continue
            ct, mask, kct = out
            ax.imshow(np.clip(ct, -160, 240), cmap="gray")
            ax.imshow(np.ma.masked_equal(mask, 0), cmap="autumn", alpha=0.55, interpolation="nearest")
            role = "CONTEXT (defines ellipsoid)" if col == 0 else "TARGET (same coords ellipsoid)"
            ax.set_title(f"{role}\n{name(lab)}  |  {key.split('#')[0]}  z={kct}", fontsize=10)
            ax.axis("off")
    fig.suptitle(f"Same coords-space ellipsoid (K={K}·std) transferred context -> target", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "ellipsoid_transfer.png")
    fig.savefig(out, dpi=95, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

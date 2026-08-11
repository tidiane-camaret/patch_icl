"""Assess whether the `coords` field gives body-position values that are
invariant across subjects — i.e. whether a fixed region in coords-space picks
the same anatomical element (totalseg label) in every subject.

Stage 1 (--extract): for N subjects, sample the coords 3-vector on its own grid
and the aligned totalseg label at each grid point, then store per-(subject,label)
the coords centroid, within-subject std and voxel count. Cached to an npz.

Stage 2 (--analyze): consistency ratio (between-subject / within-subject spread)
and leave-one-subject-out nearest-centroid retrieval of label identity from a
coords centroid. Saves summary CSV + figures.
"""
import argparse, json, os
import numpy as np
import nibabel as nib

HERE = os.path.dirname(__file__)
JSON = os.path.join(HERE, "coords_paths_chemotox.json")
CACHE = os.path.join(HERE, "centroids.npz")
FIGDIR = os.path.join(HERE, "figs")

# Partial TotalSegmentator v2 "total" names (common organs); fallback -> label_N.
TS = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
      5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
      9: "adrenal_gland_left", 10: "lung_upper_lobe_left",
      11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
      13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
      15: "esophagus", 16: "trachea", 17: "thyroid_gland",
      21: "urinary_bladder", 26: "vertebrae_L4", 51: "heart", 52: "aorta",
      53: "pulmonary_vein", 55: "inferior_vena_cava"}
name = lambda l: TS.get(int(l), f"label_{int(l)}")


def voxels(p):
    """Return (coords_vec (N,3), label (N,)) sampled on the coords grid."""
    co_img = nib.load(p["coords"])
    ts_img = nib.load(p["totalseg"])
    co = np.asanyarray(co_img.dataobj).astype(np.float32)          # (X,Y,Z,3)
    if co.ndim != 4 or co.shape[-1] != 3:
        return None, None
    gx, gy, gz = np.meshgrid(*[np.arange(s) for s in co.shape[:3]], indexing="ij")
    ijk1 = np.stack([gx, gy, gz, np.ones_like(gx)], -1).reshape(-1, 4).T
    world = co_img.affine @ ijk1
    idx = np.round(np.linalg.inv(ts_img.affine) @ world)[:3].astype(np.int64).T
    shp = np.array(ts_img.shape[:3])
    inb = ((idx >= 0) & (idx < shp)).all(1)
    ts = np.asanyarray(ts_img.dataobj)
    lab = np.zeros(idx.shape[0], np.int32)
    ii = idx[inb]
    lab[inb] = ts[ii[:, 0], ii[:, 1], ii[:, 2]]
    return co.reshape(-1, 3), lab


def sample_subject(p, min_vox=40):
    """Return {label: (centroid3, std3, count)} for one subject."""
    cv, lab = voxels(p)
    if cv is None:
        return {}
    out = {}
    for l in np.unique(lab):
        if l == 0:
            continue
        m = lab == l
        if m.sum() < min_vox:
            continue
        c = cv[m]
        out[int(l)] = (c.mean(0), c.std(0), int(m.sum()))
    return out


def pick_keys(paths, n, unique):
    keys = list(paths)
    if unique:                       # keep first scan per patient id
        seen, out = set(), []
        for k in keys:
            pid = k.split("#")[0]
            if pid not in seen:
                seen.add(pid); out.append(k)
        keys = out
    return keys[:n]


def extract(n_subjects, unique):
    paths = json.load(open(JSON))
    keys = pick_keys(paths, n_subjects, unique)
    recs = []  # (subj_idx, label, cx,cy,cz, sx,sy,sz, count)
    for si, k in enumerate(keys):
        try:
            d = sample_subject(paths[k])
        except Exception as e:
            print(f"  skip {k}: {e}")
            continue
        for l, (c, s, n) in d.items():
            recs.append([si, l, *c, *s, n])
        print(f"[{si+1}/{len(keys)}] {k}: {len(d)} labels")
    recs = np.array(recs, np.float32)
    np.savez(CACHE, recs=recs, keys=np.array(keys))
    print(f"saved {CACHE}: {recs.shape[0]} (subject,label) rows, "
          f"{len(np.unique(recs[:,1]))} labels")


def analyze(min_subjects=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(CACHE, allow_pickle=True)
    recs = d["recs"]
    subj = recs[:, 0].astype(int)
    lab = recs[:, 1].astype(int)
    cen = recs[:, 2:5]
    wstd = recs[:, 5:8]
    n_subj = subj.max() + 1

    labels = [l for l in np.unique(lab) if (lab == l).sum() >= min_subjects]
    rows = []
    for l in labels:
        m = lab == l
        cs = cen[m]                              # (S,3) per-subject centroids
        between = cs.std(0)                      # cross-subject spread of centroid
        within = wstd[m].mean(0)                 # typical within-subject extent
        rows.append((l, m.sum(),
                     np.linalg.norm(between),    # between RMS (mm)
                     np.linalg.norm(within),     # within RMS (mm)
                     np.linalg.norm(between) / (np.linalg.norm(within) + 1e-6)))
    rows.sort(key=lambda r: r[4])

    print(f"\n{'label':22s} {'n_subj':>6} {'between':>8} {'within':>8} {'ratio':>6}")
    for l, ns, b, w, r in rows:
        flag = "  <- tight" if r < 1 else ""
        print(f"{name(l):22s} {ns:6d} {b:8.1f} {w:8.1f} {r:6.2f}{flag}")

    # Leave-one-subject-out nearest-centroid retrieval of label from coords.
    top1 = top5 = tot = 0
    for s in range(n_subj):
        tr, te = subj != s, subj == s
        if te.sum() == 0:
            continue
        book_l, book_c = [], []
        for l in np.unique(lab[tr]):
            book_l.append(l)
            book_c.append(cen[tr & (lab == l)].mean(0))
        book_l, book_c = np.array(book_l), np.stack(book_c)
        for c, gt in zip(cen[te], lab[te]):
            dist = np.linalg.norm(book_c - c, axis=1)
            order = book_l[np.argsort(dist)]
            top1 += order[0] == gt
            top5 += gt in order[:5]
            tot += 1
    print(f"\nLOO nearest-centroid label retrieval over {tot} (subj,label) queries:")
    print(f"  top-1 = {top1/tot:.3f}   top-5 = {top5/tot:.3f}")

    # Figure: 2D projections of per-subject centroids, colored by label.
    os.makedirs(FIGDIR, exist_ok=True)
    big = [l for l, ns, *_ in rows][:14]
    cmap = plt.get_cmap("tab20")
    proj = [(0, 2, "X vs Z (coronal)"), (1, 2, "Y vs Z (sagittal)"), (0, 1, "X vs Y (axial)")]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (a, b, ttl) in zip(axs, proj):
        for i, l in enumerate(big):
            m = lab == l
            ax.scatter(cen[m, a], cen[m, b], s=18, color=cmap(i % 20), label=name(l))
        ax.set_title(ttl); ax.set_xlabel(f"coord[{a}]"); ax.set_ylabel(f"coord[{b}]")
    axs[0].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("Per-subject coords centroids per label (tight clusters => invariant)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "centroid_clusters.png"), dpi=90)
    print(f"\nsaved {FIGDIR}/centroid_clusters.png")


def synth(n_subjects, unique, radius=25.0, min_subjects=6, ellipsoid=False):
    """Downstream test: a fixed coords-ball (canonical centroid built from OTHER
    subjects) is applied to a held-out subject; measure how well the induced
    synthetic label matches the true totalseg label (purity + Dice)."""
    paths = json.load(open(JSON))
    keys = pick_keys(paths, n_subjects, unique)
    # per-subject voxel arrays + per-label centroids
    V, cents = [], {}
    for k in keys:
        cv, lab = voxels(paths[k])
        if cv is None:
            continue
        si = len(V)
        V.append((cv, lab))
        for l in np.unique(lab):
            if l == 0 or (lab == l).sum() < 40:
                continue
            cents.setdefault(int(l), {})[si] = cv[lab == l].mean(0)
    labels = [l for l, d in cents.items() if len(d) >= min_subjects]

    # per-label anisotropic extent (pooled within-subject std per axis)
    ext = {}
    for l in labels:
        stds = [V[si][0][V[si][1] == l].std(0) for si in cents[l]]
        ext[l] = np.mean(stds, 0) + 1e-3

    res = {}  # label -> list of (purity, dice)
    for l in labels:
        for si, (cv, lab) in enumerate(V):
            if si not in cents[l]:
                continue                                  # label absent here
            others = [c for s, c in cents[l].items() if s != si]
            canon = np.mean(others, 0)                    # LOO canonical centroid
            if ellipsoid:            # anisotropic: Mahalanobis-style within k*std/axis
                sel = (((cv - canon) / (radius * ext[l])) ** 2).sum(1) <= 1.0
            else:                    # isotropic ball of fixed mm radius
                sel = np.linalg.norm(cv - canon, axis=1) <= radius   # synthetic mask
            gt = lab == l
            if sel.sum() == 0:
                continue
            purity = (lab[sel] == l).mean()               # frac of ball that is right organ
            dice = 2 * (sel & gt).sum() / (sel.sum() + gt.sum())
            res.setdefault(l, []).append((purity, dice))

    rows = [(l, len(v), np.mean([x[0] for x in v]), np.mean([x[1] for x in v]))
            for l, v in res.items()]
    rows.sort(key=lambda r: -r[3])
    print(f"\nSynthetic-label round-trip (coords-ball r={radius}mm, LOO canonical centroid)")
    print(f"{'label':22s} {'n':>4} {'purity':>7} {'dice':>6}")
    for l, n, pu, di in rows:
        print(f"{name(l):22s} {n:4d} {pu:7.2f} {di:6.2f}")
    P = np.mean([r[2] for r in rows]); D = np.mean([r[3] for r in rows])
    print(f"\nMACRO over {len(rows)} labels:  purity = {P:.3f}   dice = {D:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--unique", action="store_true", help="one scan per patient")
    ap.add_argument("--radius", type=float, default=25.0)
    ap.add_argument("--ellipsoid", action="store_true", help="anisotropic k*std/axis")
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    if a.extract:
        extract(a.n, a.unique)
    if a.analyze:
        analyze()
    if a.synth:
        synth(a.n, a.unique, a.radius, ellipsoid=a.ellipsoid)

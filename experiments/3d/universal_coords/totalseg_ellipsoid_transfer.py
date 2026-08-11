"""Ellipsoid-transfer Dice on TotalSeg test cases, mirroring the ChemoTox
transfer_methods.py benchmark. For an organ L: draw an ellipsoid around L on the
ctx grid -> its coords values form a cloud Q -> select tgt voxels whose coords
match Q -> Dice vs the tgt organ L. Coords are the 1.5mm-iso maps resampled onto
the CT/label grid, so co/lab share indices (no affine bridge).

Matchers: base (analytic axis-aligned coords ellipsoid), gauss (Mahalanobis),
bin (8mm coords-bin hashing). knn omitted (slow, matcher-agnostic ceiling shown
on ChemoTox). Pairs chosen to maximise shared-organ overlap since the test set
spans head/thorax/abdomen/extremity (partial-body scans share few organs).
"""
import os, glob, time
import numpy as np
import nibabel as nib

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
HERE = os.path.dirname(__file__)
CO = os.path.join(HERE, "coords_predictor", "output_totalseg_1p5")
K = 2.0
MIN_VOX = 60
N_PAIRS = 8
DS = 2              # stride-2 downsample -> ~3mm eval (fast; comparable to ChemoTox's coarse 4/8mm grid)
TOPK_ORGANS = 12    # cap organs per pair to the largest shared ones


def load(sid):
    co = np.asanyarray(nib.load(os.path.join(CO, f"{sid}_coords.nii.gz")).dataobj).astype(np.float32)
    lab = np.load(os.path.join(TS, sid, "label.npy"))
    return co[::DS, ::DS, ::DS], lab[::DS, ::DS, ::DS]


def ctx_cloud(co, lab, L):
    idx = np.argwhere(lab == L)
    if idx.shape[0] < MIN_VOX:
        return None
    c = idx.mean(0); r = K * idx.std(0) + 1e-3
    g = np.stack(np.meshgrid(*[np.arange(s) for s in lab.shape], indexing="ij"), -1)
    region = (((g - c) / r) ** 2).sum(-1) <= 1.0
    return co[region]


def m_gauss(Q, tgt):
    mu = Q.mean(0); P = np.linalg.inv(np.cov(Q.T) + 1e-3 * np.eye(3))
    d2 = np.einsum("ni,ij,nj->n", tgt - mu, P, tgt - mu)
    dq = np.einsum("ni,ij,nj->n", Q - mu, P, Q - mu)
    return d2 <= np.percentile(dq, 95)


def m_bin(Q, tgt, b=8.0):
    lo = np.minimum(Q.min(0), tgt.min(0))
    qb = np.floor((Q - lo) / b).astype(np.int64); tb = np.floor((tgt - lo) / b).astype(np.int64)
    dims = tb.max(0).clip(qb.max(0)) + 1
    return np.isin(np.ravel_multi_index(tb.T, dims), np.unique(np.ravel_multi_index(qb.T, dims)))


def m_base(Q, tgt):
    c = Q.mean(0); r = K * Q.std(0) + 1e-3
    return (((tgt - c) / r) ** 2).sum(1) <= 1.0


def dice_pur(sel, gt):
    inter = (sel & gt).sum()
    return 2 * inter / (sel.sum() + gt.sum() + 1e-9), inter / (sel.sum() + 1e-9)


def pick_pairs(labsets):
    """Top-N distinct-case pairs by shared >=MIN_VOX organ overlap."""
    ids = list(labsets)
    scored = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            shared = labsets[ids[i]] & labsets[ids[j]]
            if shared:
                scored.append((len(shared), ids[i], ids[j]))
    scored.sort(reverse=True)
    pairs, used = [], set()
    for _, a, b in scored:
        if a in used or b in used:
            continue
        pairs.append((a, b)); used |= {a, b}
        if len(pairs) >= N_PAIRS:
            break
    return pairs


def main():
    ids = sorted(os.path.basename(f).replace("_coords.nii.gz", "")
                 for f in glob.glob(os.path.join(CO, "*_coords.nii.gz")))
    cache = {sid: load(sid) for sid in ids}
    counts = {sid: dict(zip(*np.unique(lab, return_counts=True))) for sid, (_, lab) in cache.items()}
    labsets = {sid: {int(L) for L, c in counts[sid].items() if L > 0 and c >= MIN_VOX} for sid in ids}
    pairs = pick_pairs(labsets)
    print(f"pairs (shared organs): {[(a,b,len(labsets[a]&labsets[b])) for a,b in pairs]}\n")

    agg = {m: {"dice": [], "pur": []} for m in ["base", "gauss", "bin"]}
    per_organ = {}
    for pi, (ci, ti) in enumerate(pairs):
        t0 = time.perf_counter()
        co_c, lab_c = cache[ci]; co_t, lab_t = cache[ti]
        tgt = co_t.reshape(-1, 3)
        # largest shared organs (by target voxel count / DS^3) first, capped
        shared = sorted(labsets[ci] & labsets[ti],
                        key=lambda L: -counts[ti].get(L, 0))[:TOPK_ORGANS]
        for L in shared:
            Q = ctx_cloud(co_c, lab_c, L)
            gtm = (lab_t == L).reshape(-1)
            if Q is None or gtm.sum() < MIN_VOX:
                continue
            for mn, sel in [("base", m_base(Q, tgt)), ("gauss", m_gauss(Q, tgt)), ("bin", m_bin(Q, tgt))]:
                d, p = dice_pur(sel, gtm)
                agg[mn]["dice"].append(d); agg[mn]["pur"].append(p)
                if mn == "bin":
                    per_organ.setdefault(L, []).append(d)
        print(f"  pair {pi+1}/{len(pairs)} {ci}->{ti}: {len(shared)} organs  "
              f"{time.perf_counter()-t0:.1f}s", flush=True)

    n = len(agg["bin"]["dice"])
    print(f"=== ellipsoid-transfer Dice | {len(pairs)} pairs, {n} (pair,organ) instances ===")
    print(f"{'method':6} {'dice':>7} {'purity':>7}")
    for mn, v in agg.items():
        print(f"{mn:6} {np.mean(v['dice']):7.3f} {np.mean(v['pur']):7.3f}")
    print("\nper-organ (bin) Dice, most-scored first:")
    for L, ds in sorted(per_organ.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  L{int(L):<3} n={len(ds):2d}  {np.mean(ds):.3f}")


if __name__ == "__main__":
    main()

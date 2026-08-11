"""How much STABLE info is in the coords maps? Even if fine class correspondence
is weak, the maps may robustly encode the coarse body axes. For N subjects, over
labelled-anatomy voxels, compare each coords channel to the real RAS world axes
(from the CT affine: X=L/R, Y=A/P, Z=S/I):

  1. per-channel best-matching RAS axis + |corr|, and whether the axis assignment
     + sign are CONSISTENT across subjects (a stable canonical frame => same
     channel->axis map + same sign every subject);
  2. linear-fit R^2 of each coords channel from (x,y,z) world coords (how smooth/
     global the map is -- high R^2 => coords ~ a warped body-axis coordinate);
  3. SI ordering: do organ centroids along the SI-aligned coords channel reproduce
     the known head->foot anatomical order (Spearman per subject)?

Run (loki): .venv_thor_fresh/bin/python experiments/3d/universal_coords/coords_axes.py --n 30
"""
import os, sys, argparse
import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from data.totalseg_classes import ALL_CLASSES

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
DS = 2
AX = ["LR", "AP", "SI"]                 # RAS world axes (x,y,z)
NAME2IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}

# Known head->foot anatomical order (superior -> inferior) for a landmark set.
SI_ORDER = ["brain", "vertebrae_C3", "vertebrae_C7", "thyroid_gland", "trachea",
            "vertebrae_T3", "heart", "vertebrae_T6", "liver", "vertebrae_T12",
            "kidney_left", "vertebrae_L3", "vertebrae_L5", "hip_right",
            "urinary_bladder", "femur_left"]


def subject_ids():
    return sorted(d for d in os.listdir(TS)
                  if d.startswith("s") and os.path.exists(os.path.join(TS, d, "coords.npy")))


def load(sid, max_vox=60000, rng=None):
    co = np.load(os.path.join(TS, sid, "coords.npy"))[::DS, ::DS, ::DS].astype(np.float32)
    lab = np.load(os.path.join(TS, sid, "label.npy"))[::DS, ::DS, ::DS]
    A = nib.load(os.path.join(TS, sid, "ct.nii.gz")).affine
    ijk = np.argwhere(lab > 0)                       # labelled-anatomy voxels
    if len(ijk) == 0:
        return None
    if len(ijk) > max_vox:
        ijk = ijk[rng.permutation(len(ijk))[:max_vox]]
    world = (A @ np.c_[ijk * DS, np.ones(len(ijk))].T)[:3].T   # RAS mm
    C = co[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    labv = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return C, world, lab, co, A


def si_channel(subs):
    """Coords channel index most consistently aligned with SI (world z)."""
    cc = np.array([[abs(np.corrcoef(s[0][:, a], s[1][:, 2])[0, 1]) for a in range(3)] for s in subs])
    return int(cc.mean(0).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ids = subject_ids(); rng.shuffle(ids); ids = ids[:args.n]

    subs, R2 = [], []
    corr_stack = []     # (subj, coords_chan, world_axis) signed corr
    for sid in ids:
        d = load(sid, rng=rng)
        if d is None:
            continue
        C, W, lab, co, A = d
        subs.append((C, W, lab, co))
        Wc = (W - W.mean(0)) / (W.std(0) + 1e-6)
        M = np.c_[Wc, np.ones(len(Wc))]
        cm = np.zeros((3, 3)); r2 = np.zeros(3)
        for a in range(3):
            for b in range(3):
                cm[a, b] = np.corrcoef(C[:, a], W[:, b])[0, 1]
            beta, *_ = np.linalg.lstsq(M, C[:, a], rcond=None)
            pred = M @ beta
            r2[a] = 1 - ((C[:, a] - pred) ** 2).sum() / (((C[:, a] - C[:, a].mean()) ** 2).sum() + 1e-9)
        corr_stack.append(cm); R2.append(r2)
    corr_stack = np.array(corr_stack); R2 = np.array(R2)
    print(f"{len(subs)} subjects, labelled-anatomy voxels\n")

    # 1. axis alignment: per coords channel, which RAS axis + consistency
    print("Per coords-channel alignment to RAS world axes (|corr|, signed):")
    print(f"{'chan':5} {'best axis':10} {'|corr| mean':>11} {'assign consist':>15} {'sign consist':>13} {'R2 mean':>9}")
    for a in range(3):
        best = np.abs(corr_stack[:, a, :]).argmax(1)          # per subj best world axis
        maj = np.bincount(best, minlength=3).argmax()
        assign_consist = (best == maj).mean()
        signs = np.sign(corr_stack[:, a, maj])
        sign_consist = max((signs > 0).mean(), (signs < 0).mean())
        mag = np.abs(corr_stack[:, a, maj]).mean()
        print(f"c{a:<4} {AX[maj]:10} {mag:11.3f} {assign_consist:15.2f} {sign_consist:13.2f} {R2[:,a].mean():9.3f}")

    # 2. overall: is coords a globally smooth function of body axes?
    print(f"\nlinear R^2 (coords chan from world x,y,z): mean {R2.mean():.3f} "
          f"per-chan {np.round(R2.mean(0),3)}  -> higher = coords ~ warped body-axis frame")

    # 3. SI organ ordering
    sic = si_channel(subs)
    sign = np.sign(np.mean([np.corrcoef(s[0][:, sic], s[1][:, 2])[0, 1] for s in subs]))
    order_idx = [NAME2IDX[n] for n in SI_ORDER]
    rhos = []
    for (C, W, lab, co) in subs:
        vals, ranks = [], []
        for r, L in enumerate(order_idx):
            m = lab == L
            if m.sum() < 20:
                continue
            vals.append(sign * co[..., sic][m].mean()); ranks.append(r)
        if len(vals) >= 5:
            rhos.append(spearmanr(ranks, vals).correlation)
    print(f"\nSI body-axis test: coords channel c{sic} ~ SI. Spearman(known head->foot order, "
          f"coords-SI centroid) over {len(rhos)} subjects: mean {np.mean(rhos):.3f} "
          f"[min {np.min(rhos):.3f}]  (1.0 = perfect head->foot ordering)")


if __name__ == "__main__":
    main()

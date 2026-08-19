"""Throwaway spike: estimate inter-subject deformation distribution from GT labels.

Uses TotalSegmentator GT masks (label.npy, native 1.5mm iso grid) as a proxy for
"how much does the same organ vary in pose/scale/shape across subjects?" — to check
whether the geometric-aug magnitudes (task.affine / task.elastic) are calibrated to
real anatomical variability.

Per organ, per subject we measure a similarity transform (translation / scale / rotation)
of the organ, then look at the cross-subject spread of each component. The residual
after removing similarity (via Procrustes on the multi-organ centroid constellation) is a
coarse label-only proxy for the *elastic* (nonrigid) part.

Output: a table of measured variability vs current aug knobs. No aug code is changed.
"""
import os, glob, random, json
import numpy as np

import sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
from data.totalseg_classes import ALL_CLASSES

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
SPACING = 1.5                      # mm, isotropic (verified uniform)
N_SUBJECTS = 150
MIN_VOX = 200                      # ignore tiny/partial organs
random.seed(0); np.random.seed(0)

# Current aug knobs (task) for comparison
AUG = dict(scale_min=0.70, scale_max=1.40, max_angle_deg=30.0, max_translate=0.0,
           elastic_alpha=0.12, elastic_grid_scale=8)
# alpha is displacement std in normalized grid units; on a 128^3 crop @1.5mm the
# half-extent is 96mm, so aug elastic std ~ alpha*96 mm.
CROP_HALF_MM = 96.0

IDX2NAME = {i + 1: c for i, c in enumerate(ALL_CLASSES)}

# organs to report (well-populated, mix of blobby + elongated)
FOCUS = ["aorta", "liver", "spleen", "kidney_left", "kidney_right", "urinary_bladder",
         "gallbladder", "femur_left", "vertebrae_L3", "heart", "stomach", "pancreas",
         "autochthon_left", "esophagus", "trachea"]
NAME2IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}


def organ_moments(mask_idx_coords):
    """coords: (N,3) voxel coords of one organ. Return centroid(mm), volume(mm^3),
    inertia eigenvalues(mm^2, desc), eigenvectors(cols)."""
    c = mask_idx_coords.mean(0)
    x = (mask_idx_coords - c) * SPACING
    cov = (x.T @ x) / len(x)
    w, V = np.linalg.eigh(cov)              # ascending
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    vol = len(mask_idx_coords) * SPACING ** 3
    return c * SPACING, vol, w, V


def main():
    subs = sorted(glob.glob(TS + "/s[0-9]*"))
    subs = random.sample(subs, min(N_SUBJECTS, len(subs)))

    # per-class lists
    sizes = {c: [] for c in FOCUS}          # linear size = vol^(1/3) mm
    axes = {c: [] for c in FOCUS}           # major axis (unit vec) for rotation spread
    aniso = {c: [] for c in FOCUS}          # sqrt(l1/l3) elongation
    centroids = {}                          # subject -> {class: centroid_mm}

    for s in subs:
        lab = np.load(os.path.join(s, "label.npy"))
        centroids[s] = {}
        present = np.unique(lab)
        for cname in FOCUS:
            idx = NAME2IDX[cname]
            if idx not in present:
                continue
            coords = np.argwhere(lab == idx)
            if len(coords) < MIN_VOX:
                continue
            cmm, vol, w, V = organ_moments(coords)
            sizes[cname].append(vol ** (1 / 3))
            axes[cname].append(V[:, 0])                 # major eigenvector
            aniso[cname].append(float(np.sqrt(w[0] / max(w[2], 1e-6))))
            centroids[s][cname] = cmm

    # ---- Scale variability (linear size vol^(1/3)) ----
    print("\n=== SCALE: linear organ size vol^(1/3), ratio to per-organ median ===")
    print(f"{'organ':<16}{'n':>4}{'med_mm':>8}{'CV%':>7}{'p5/med':>8}{'p95/med':>9}")
    all_logr = []
    for c in FOCUS:
        v = np.array(sizes[c])
        if len(v) < 20:
            continue
        med = np.median(v)
        r = v / med
        all_logr.extend(np.log(r))
        cv = 100 * v.std() / v.mean()
        print(f"{c:<16}{len(v):>4}{med:>8.1f}{cv:>7.1f}{np.percentile(r,5):>8.2f}{np.percentile(r,95):>9.2f}")
    lr = np.array(all_logr)
    print(f"\nPOOLED scale ratio: p5={np.exp(np.percentile(lr,5)):.2f} "
          f"p50=1.00 p95={np.exp(np.percentile(lr,95)):.2f} "
          f"(±1sd = {np.exp(-lr.std()):.2f}..{np.exp(lr.std()):.2f})")
    print(f"AUG scale range: {AUG['scale_min']:.2f}..{AUG['scale_max']:.2f}")

    # ---- Rotation variability (major-axis spread, degrees) ----
    print("\n=== ROTATION: major-axis angular spread vs population mean axis (deg) ===")
    print(f"{'organ':<16}{'n':>4}{'aniso':>7}{'std_deg':>9}{'p95_deg':>9}")
    pooled_ang = []
    for c in FOCUS:
        A = np.array(axes[c])
        if len(A) < 20:
            continue
        # mean axis via principal eigenvector of sum of outer products (sign-invariant)
        M = (A[:, :, None] * A[:, None, :]).sum(0)
        mw, mV = np.linalg.eigh(M)
        mean_ax = mV[:, -1]
        cosang = np.abs(A @ mean_ax).clip(0, 1)
        ang = np.degrees(np.arccos(cosang))
        # only meaningful for anisotropic organs
        med_aniso = np.median(aniso[c])
        if med_aniso > 1.5:
            pooled_ang.extend(ang)
        print(f"{c:<16}{len(A):>4}{med_aniso:>7.2f}{ang.std():>9.1f}{np.percentile(ang,95):>9.1f}")
    pa = np.array(pooled_ang)
    print(f"\nPOOLED (anisotropic organs) rotation: std={pa.std():.1f} deg  "
          f"p95={np.percentile(pa,95):.1f} deg  max={pa.max():.1f} deg")
    print(f"AUG max_angle_deg: {AUG['max_angle_deg']:.0f} (uniform -a..+a)")

    # ---- Elastic proxy: Procrustes residual of multi-organ centroid constellation ----
    print("\n=== ELASTIC (coarse, inter-organ constellation Procrustes residual) ===")
    common = [c for c in FOCUS if sum(c in centroids[s] for s in subs) > 0.8 * len(subs)]
    S = [s for s in subs if all(c in centroids[s] for c in common)]
    X = np.array([[centroids[s][c] for c in common] for s in S])   # (n, L, 3)
    print(f"landmarks (organs present in >80%): {common}")
    print(f"subjects with full set: {len(S)}")
    # generalized Procrustes (translation, scale, rotation) to the mean
    def procrustes_align(A, B):  # align A to B (similarity), return aligned A
        a = A - A.mean(0); b = B - B.mean(0)
        na = np.linalg.norm(a); a = a / na; b = b / np.linalg.norm(b)
        U, _, Vt = np.linalg.svd(a.T @ b)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        return (a @ R), R
    ref = X[0] - X[0].mean(0); ref /= np.linalg.norm(ref)
    for _ in range(5):
        aligned = np.array([procrustes_align(x, ref)[0] for x in X])
        ref = aligned.mean(0); ref /= np.linalg.norm(ref)
    resid = aligned - aligned.mean(0)                # per-subject residual (normalized units)
    # scale residual back to mm using median body scale (norm of centered constellation)
    body_scale_mm = np.median([np.linalg.norm(x - x.mean(0)) for x in X])
    per_lmk_rms_mm = np.sqrt((resid ** 2).sum(-1)).std(0) * body_scale_mm
    overall_rms_mm = np.sqrt((resid ** 2).sum(-1).mean()) * body_scale_mm
    print(f"\n{'organ':<16}{'resid_rms_mm':>13}")
    for c, r in zip(common, per_lmk_rms_mm):
        print(f"{c:<16}{r:>13.1f}")
    print(f"\nOVERALL nonrigid residual RMS: {overall_rms_mm:.1f} mm")
    print(f"AUG elastic std ~ alpha*{CROP_HALF_MM:.0f}mm = {AUG['elastic_alpha']*CROP_HALF_MM:.1f} mm "
          f"(alpha={AUG['elastic_alpha']}, grid_scale={AUG['elastic_grid_scale']} ~ "
          f"{AUG['elastic_grid_scale']*SPACING:.0f}mm control spacing)")
    print("NOTE: constellation residual is a WHOLE-BODY inter-organ (low-freq) proxy; "
          "task aug acts on a single-organ crop, so treat this as an upper bound on the "
          "large-scale warp, not the within-organ elastic detail.")


if __name__ == "__main__":
    main()

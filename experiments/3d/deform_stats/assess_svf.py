"""Throwaway spike: is the SVF `deform` aug calibrated to REAL inter-case deformation?

Measures, in the SAME units as aug `max_disp` (normalized [-1,1] grid units):
  (A) real inter-case nonrigid deformation between GT organ masks — by registering
      same-organ case pairs (affine to remove pose/scale/rotation = the task.affine
      analog, then a diffeomorphic SVF on the aug's coarse control grid), and
  (B) the deform op's sampled SVF at several max_disp values.
Compares displacement RMS + Jacobian-determinant spread. No aug code changed.
"""
import os, glob, random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

import sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
from data.totalseg_classes import ALL_CLASSES
from src.augmentations import _svf_displacement

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
NAME2IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}
ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", "stomach",
          "gallbladder", "urinary_bladder"]
R = 48                     # working grid
CONTROL_POINTS = 6         # velocity grid nodes per axis (resolution-invariant count)
NUM_STEPS = 6
N_PAIRS = 30
BAND = 6.0                 # SDF clip (voxels)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)


def organ_sdf(subject, organ):
    """Size-normalized signed distance field of one organ on an R^3 grid, or None."""
    lab = np.load(os.path.join(subject, "label.npy"), mmap_mode="r")
    idx = NAME2IDX[organ]
    coords = np.argwhere(np.asarray(lab) == idx)
    if len(coords) < 300:
        return None
    c = coords.mean(0)
    ext = coords.max(0) - coords.min(0)
    side = 1.5 * float(ext.max()) + 4
    lo = np.round(c - side / 2).astype(int)
    D, H, W = lab.shape
    box = np.zeros((int(side),) * 3, np.float32)
    # copy overlapping region
    src0 = np.maximum(lo, 0); src1 = np.minimum(lo + int(side), [D, H, W])
    dst0 = src0 - lo; dst1 = dst0 + (src1 - src0)
    if np.any(src1 <= src0):
        return None
    box[dst0[0]:dst1[0], dst0[1]:dst1[1], dst0[2]:dst1[2]] = (
        np.asarray(lab)[src0[0]:src1[0], src0[1]:src1[1], src0[2]:src1[2]] == idx)
    t = torch.from_numpy(box)[None, None]
    m = F.interpolate(t, size=(R, R, R), mode="trilinear", align_corners=False)[0, 0]
    b = (m > 0.5).numpy()
    if b.sum() < 30 or (~b).sum() < 30:
        return None
    sdf = distance_transform_edt(~b) - distance_transform_edt(b)
    sdf = np.clip(sdf, -BAND, BAND) / BAND
    return torch.from_numpy(sdf.astype(np.float32))[None, None]   # (1,1,R,R,R)


def _base_grid():
    return F.affine_grid(torch.eye(3, 4, device=DEV)[None], (1, 1, R, R, R), align_corners=False)


def svf_from_velocity(vc, num_steps):
    """Integrate a (learnable) coarse velocity vc (1,3,r,r,r) → disp (1,R,R,R,3)."""
    v = F.interpolate(vc, size=(R, R, R), mode="trilinear", align_corners=False)
    v = v.permute(0, 2, 3, 4, 1)
    base = _base_grid()
    phi = v / (2 ** num_steps)
    for _ in range(num_steps):
        warped = F.grid_sample(phi.permute(0, 4, 1, 2, 3), base + phi,
                               mode="bilinear", padding_mode="border",
                               align_corners=False).permute(0, 2, 3, 4, 1)
        phi = phi + warped
    return phi


def register(sdf_a, sdf_b):
    """Affine (pose/scale/rot) then diffeomorphic SVF. Return fitted SVF disp + Jac det."""
    sdf_a, sdf_b = sdf_a.to(DEV), sdf_b.to(DEV)
    base = _base_grid()
    eye = torch.eye(3, 4, device=DEV)
    # --- affine refine (12-param theta, init identity) ---
    theta = eye.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=3e-2)
    for _ in range(120):
        opt.zero_grad()
        g = F.affine_grid(theta[None], (1, 1, R, R, R), align_corners=False)
        warp = F.grid_sample(sdf_a, g, padding_mode="border", align_corners=False)
        loss = F.mse_loss(warp, sdf_b) + 1e-3 * ((theta - eye) ** 2).sum()
        loss.backward(); opt.step()
    theta = theta.detach()
    g_aff = F.affine_grid(theta[None], (1, 1, R, R, R), align_corners=False)
    sdf_a_aff = F.grid_sample(sdf_a, g_aff, padding_mode="border", align_corners=False).detach()

    # --- SVF fit on the aug's coarse control grid ---
    r = CONTROL_POINTS
    vc = torch.zeros(1, 3, r, r, r, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([vc], lr=3e-2)
    for _ in range(200):
        opt.zero_grad()
        phi = svf_from_velocity(vc, NUM_STEPS)
        warp = F.grid_sample(sdf_a_aff, base + phi, padding_mode="border", align_corners=False)
        loss = F.mse_loss(warp, sdf_b) + 1e-2 * (vc ** 2).mean()
        loss.backward(); opt.step()
    phi = svf_from_velocity(vc.detach(), NUM_STEPS).detach()
    return phi, sdf_b


def disp_rms(phi, region=None):
    mag = phi[0].pow(2).sum(-1).sqrt()          # (R,R,R) per-voxel |disp| in norm units
    if region is not None:
        mag = mag[region]
    return float(mag.mean()), float(torch.quantile(mag, 0.95))


def jac_det(phi):
    # normalized grid coords span [-1,1] over R voxels → spacing 2/R so identity→det 1
    f = (_base_grid() + phi)[0][..., [2, 1, 0]]
    sp = 2.0 / R
    J = torch.stack([torch.gradient(f, dim=d, spacing=sp)[0] for d in range(3)], dim=-2)
    return torch.linalg.det(J)[1:-1, 1:-1, 1:-1]


def main():
    subs = sorted(glob.glob(TS + "/s[0-9]*"))
    # collect available (subject, organ) sdfs lazily per organ
    real_rms, real_p95, real_jac_lo, real_jac_hi, real_jac_fold = [], [], [], [], []
    pairs_done = 0
    per_organ_target = max(2, N_PAIRS // len(ORGANS))
    for organ in ORGANS:
        cand = random.sample(subs, min(120, len(subs)))
        sdfs = []
        for s in cand:
            v = organ_sdf(s, organ)
            if v is not None:
                sdfs.append(v)
            if len(sdfs) >= 2 * per_organ_target + 2:
                break
        random.shuffle(sdfs)
        for i in range(0, min(len(sdfs) - 1, 2 * per_organ_target), 2):
            phi, sdfb = register(sdfs[i], sdfs[i + 1])
            band = (sdfb[0, 0].abs() < 0.5)                # near organ surface/interior
            m, p95 = disp_rms(phi, band)
            jd = jac_det(phi)
            real_rms.append(m); real_p95.append(p95)
            real_jac_lo.append(float(jd.min())); real_jac_hi.append(float(jd.max()))
            real_jac_fold = real_jac_fold + [float((jd <= 0).float().mean())]
            pairs_done += 1
    real_rms = np.array(real_rms); real_p95 = np.array(real_p95)
    print(f"\n=== REAL inter-case nonrigid deformation ({pairs_done} organ pairs) ===")
    print(f"disp RMS (norm units): mean={real_rms.mean():.3f}  median={np.median(real_rms):.3f}"
          f"  p10-p90={np.percentile(real_rms,10):.3f}-{np.percentile(real_rms,90):.3f}")
    print(f"disp p95 (norm units): median={np.median(real_p95):.3f}")
    print(f"Jacobian det: median lo={np.median(real_jac_lo):.2f} hi={np.median(real_jac_hi):.2f}"
          f"  worst lo={np.min(real_jac_lo):.2f} hi={np.max(real_jac_hi):.2f}"
          f"  folded-voxel frac (mean)={np.mean(real_jac_fold):.4f}")

    # ---- aug deform op at several max_disp ----
    print("\n=== SVF `deform` aug (sampled, same grid/control/num_steps) ===")
    base = _base_grid()
    for md in (0.06, 0.12, 0.24):
        rms, p95s, jlo, jhi, fold = [], [], [], [], []
        for k in range(40):
            g = torch.Generator(device=DEV).manual_seed(k)
            phi = _svf_displacement((R, R, R), CONTROL_POINTS, md, NUM_STEPS, generator=g, device=DEV)
            m, p = disp_rms(phi)
            rms.append(m); p95s.append(p)
            jd = jac_det(phi); jlo.append(float(jd.min())); jhi.append(float(jd.max()))
            fold.append(float((jd <= 0).float().mean()))
        print(f"max_disp={md:.2f}: disp RMS mean={np.mean(rms):.3f}  p95 median={np.median(p95s):.3f}"
              f"  Jac det {np.min(jlo):.2f}..{np.max(jhi):.2f}  folded frac={np.mean(fold):.4f}")
    print(f"\nCurrent aug default: max_disp=0.15, control_points={CONTROL_POINTS}, num_steps={NUM_STEPS}")


if __name__ == "__main__":
    main()

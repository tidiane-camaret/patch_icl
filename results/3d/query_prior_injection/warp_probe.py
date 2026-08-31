"""Spike: place the 6 mm (level-0) prediction onto the 3 mm (level-1) augmented grid.

Cascade fact (experiments/3d/cascade.py:227): run_cascade passes the SAME geo_gen seed at
every level, and _geometric builds the flip/affine/elastic/deform grid from that RNG on a
T^3 lattice — identical at every level. So the aug warp is level-invariant; only the crop
(crop_geom) differs between level 0 and level 1. crop_geom = [starts, crop_sizes, out_sizes,
pad_lo] per (d,h,w); starts/crop_sizes are NATIVE-CT voxels, out_sizes the resampled crop
side. That native-CT frame is the shared bridge between the two levels.

Method ladder (simple -> faithful):
  M0  crop-geom compose only          (identity grid, no flips)
  M1  + flip axis-reversal both ends
  M2  + closed-form affine conjugation (fit affine from captured grid, invert analytically)
  M3  + LM-damped Newton inversion of the nonlinear grid (deform/elastic); reduces to M2
      exactly when only affine fired. Set DEFORM_P>0 below to exercise it.

Fidelity is measured by warping level-0's GT mask and comparing to level-1's GT
(Dice + centroid offset mm). Timing is wall-clock around the grid build + grid_sample.

    .venv_blackwell/bin/python results/3d/query_prior_injection/warp_probe.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/dpxuser/dev/patch_icl")
os.chdir(ROOT)
os.environ["PWD"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from hydra import compose, initialize_config_dir           # noqa: E402
from omegaconf import OmegaConf                             # noqa: E402

CKPT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
        "ANALYSIS_20251122/results/patch_icl/3d_train/2026-08-31_66_train_spacing_range_3_6/best.pt")
S_COARSE, S_FINE = 6.0, 3.0
SEED = 0
DEFORM_P = 0.0        # >0 forces diffeomorphic deform (calibrated max_disp=0.15) so M3 has a job
ELASTIC_P = 0.0
M3_ITERS = 5          # LM-Newton iters for M3 (knee ~5 under forced calibrated deform)
TASKS_PER_CLASS = 3
MAX_CASES = 80

# Low val/dice classes at 6 mm (from wandb/latest-run/files/wandb-summary.json) — thin
# vessels, sub-cell organs, cervical vertebrae, ribs — plus spleen/liver as high-Dice
# controls. Warp fidelity is GT-based so it does not depend on the model seeing the class.
TARGET_CLASSES = [
    "common_carotid_artery_right", "common_carotid_artery_left",
    "brachiocephalic_vein_left", "subclavian_artery_right", "atrial_appendage_left",
    "portal_vein_and_splenic_vein", "iliac_artery_right",
    "adrenal_gland_right", "adrenal_gland_left", "gallbladder", "duodenum",
    "prostate", "pancreas", "esophagus", "iliopsoas_right",
    "vertebrae_C1", "vertebrae_C5", "rib_left_10", "rib_right_5",
    "spleen", "liver",
]
OUT = Path(__file__).resolve().parent / "warp_probe"
OUT.mkdir(parents=True, exist_ok=True)

with initialize_config_dir(config_dir=str(ROOT / "configs/experiment/3d"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=[
        "experiment=57_organs_encoder_from_scratch",
        "data.context_size=1",
        f"data.crop_spacing_mm={S_COARSE:g}",
        "data.train_spacing_range=[3,6]",
        f"augmentations.task.deform.p={DEFORM_P:g}",
        f"augmentations.task.elastic.p={ELASTIC_P:g}",
    ])

from common import DEVICE, make_eval_loader, _source_root      # noqa: E402
from train import build_model, _resolve_classes_for            # noqa: E402
from cascade import (_gen, INT_OFFSET, _recrop_level, _forward_level,  # noqa: E402
                     _centroid_from_logit, invert_geo_center)
from src.gpu_augment import GpuAugmentor                        # noqa: E402
from evaluate import _overlay, _grid_centroid                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402


# ---------------------------------------------------------------------------- model + data
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg.model = "patchset3d"
cfg.arch = OmegaConf.create(ckpt["arch"])
cfg.eval.workers = 0
model, _ = build_model(cfg)
model = model.to(DEVICE)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()},
                      strict=False)
model.eval()

augmentor = GpuAugmentor(cfg.augmentations, seed=SEED, ct_norm=cfg.data.get("ct_norm"))
ac = cfg.augmentations.task
print(f"aug.task: flip p=({ac.flip.p_d},{ac.flip.p_h},{ac.flip.p_w})  "
      f"affine p={ac.affine.p} ang={ac.affine.get('max_angle_deg')} "
      f"scale=[{ac.affine.scale_min},{ac.affine.scale_max}]  "
      f"elastic p={getattr(ac.get('elastic', None), 'p', 0)}  "
      f"deform p={getattr(ac.get('deform', None), 'p', 0)}")

_, root, _ = _source_root(cfg)
from data.totalseg_classes import resolve_classes            # noqa: E402
classes = resolve_classes(TARGET_CLASSES, root)
cfg.eval.batch_size = 6
cfg.eval.tasks_per_class = TASKS_PER_CLASS
loader = make_eval_loader(cfg, classes, split="val", spacing=S_COARSE)
provider = loader.dataset.provider
is_prob = False
print(f"{len(classes)} target classes, tasks_per_class={TASKS_PER_CLASS}")


# ---------------------------------------------------------------------------- warp methods
def _cg(cg, b):
    """crop_geom row -> (starts, crop_sizes, out_sizes, pad_lo) float tensors, (3,) each."""
    r = cg[b].float()
    return r[0], r[1], r[2], r[3]


def _lattice(T, device):
    a = torch.arange(T, device=device, dtype=torch.float32)
    d, h, w = torch.meshgrid(a, a, a, indexing="ij")
    return torch.stack([d, h, w], dim=-1)                     # (T,T,T,3) in (d,h,w)


def _crop_compose_g0(g1_dhw, cg0, cg1, b):
    """(...,3) level-1 aug voxel (d,h,w) -> level-0 aug voxel (d,h,w), crop-geom only."""
    s1, cs1, o1, p1 = _cg(cg1, b)
    s0, cs0, o0, p0 = _cg(cg0, b)
    p = s1 + (g1_dhw - p1) / o1 * cs1                         # native-CT voxel
    return p0 + (p - s0) / cs0 * o0


def _to_grid_sample(g0_dhw, T):
    """(...,3) voxel (d,h,w) -> normalized xyz grid for F.grid_sample (align_corners=False)."""
    n = (2.0 * g0_dhw + 1.0) / T - 1.0                        # (...,3) normalized (d,h,w)
    return n.flip(-1)                                          # -> (x=w, y=h, z=d)


def _flipax(g_dhw, flips_row, T):
    out = g_dhw.clone()
    for a in range(3):
        if bool(flips_row[a]):
            out[..., a] = (T - 1) - out[..., a]
    return out


def _fit_affine(grid_row, T, device, stride=8):
    """3x4 affine A (xyz) with grid ≈ [x,y,z,1] @ A.T, fit on a strided sub-lattice.

    grid_row: (T,T,T,3) captured sampling grid (normalized xyz). Exact when only affine
    fired (residual ~1e-5); grows with elastic/deform. stride=8 -> 16^3 points, plenty
    for an affine and ~200x cheaper than the full 2M-row lstsq."""
    idx = torch.arange(0, T, stride, device=device)
    a = (2.0 * idx.float() + 1.0) / T - 1.0
    d, h, w = torch.meshgrid(a, a, a, indexing="ij")
    base = torch.stack([w, h, d, torch.ones_like(d)], dim=-1).reshape(-1, 4)
    tgt = grid_row[idx][:, idx][:, :, idx].reshape(-1, 3).to(device)
    A = torch.linalg.lstsq(base, tgt).solution.T                              # (3,4)
    resid = (base @ A.T - tgt).abs().max().item()
    return A, resid


def _norm(g_dhw, T):
    return (2.0 * g_dhw + 1.0) / T - 1.0


def _denorm(n_dhw, T):
    return ((n_dhw + 1.0) * T - 1.0) / 2.0


def _flip_norm(n_dhw, flips_row):
    out = n_dhw.clone()
    for a in range(3):
        if bool(flips_row[a]):
            out[..., a] = -out[..., a]                        # flip == negate in normalized coords
    return out


def _grid_field(grid_row):
    """(T,T,T,3) captured grid -> (1,3,T,T,T) for grid_sample lookup of the forward map Φ."""
    return grid_row.permute(3, 0, 1, 2)[None].contiguous()


def _sample_field(gf, coord_xyz):
    """gf (1,3,T,T,T), coord_xyz (T,T,T,3) normalized xyz -> (T,T,T,3) = Φ sampled at coord."""
    return F.grid_sample(gf, coord_xyz[None], mode="bilinear",
                         padding_mode="border", align_corners=False)[0].permute(1, 2, 3, 0)


def _jac_field(grid_row, T):
    """(T,T,T,3) grid (xyz, spatial axes d,h,w) -> (1,9,T,T,T) Jacobian ∂grid_i/∂n_j,
    i,j in xyz. Finite differences (torch.gradient) scaled by T/2 (one voxel = 2/T in
    normalized coords); the coord axis is reversed d,h,w -> x,y,z to match `i`."""
    gd = torch.gradient(grid_row, dim=(0, 1, 2))          # 3 × (T,T,T,3), ∂/∂(d,h,w)
    J = torch.stack(gd, dim=-1) * (T / 2.0)               # (T,T,T,3,3): [...,i,j] i=xyz j=dhw
    J = J.flip(-1)                                        # j -> xyz
    return J.reshape(T, T, T, 9).permute(3, 0, 1, 2)[None].contiguous()


def _dewarp_native(vol_b1, grid_row, flips_row, T, use_newton):
    """Undo the augmentation on vol_b1 (1,1,T,T,T on the aug grid) -> the level's OWN
    native-crop grid: native_crop[c] = aug[Φ^-1(flip(c))]. Affine inverse, or M3 Newton
    when use_newton. Used to check 'warp pred then take COM' vs invert_geo_center(COM)."""
    dev = vol_b1.device
    c = _flip_norm(_norm(_lattice(T, dev), T), flips_row)     # (T,T,T,3) normalized (d,h,w)
    A, _ = _fit_affine(grid_row, T, dev)
    R, t = A[:, :3], A[:, 3]
    c_xyz = c.flip(-1)
    if use_newton:
        inv_xyz, _ = _invert_grid(_grid_field(grid_row), _jac_field(grid_row, T), c_xyz, R, t)
    else:
        inv_xyz = (c_xyz - t) @ torch.linalg.inv(R).T
    return F.grid_sample(vol_b1, inv_xyz[None], mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def _invert_grid(gf, jf, y_xyz, R, t, n_iter=M3_ITERS):
    """Newton solve Φ(x)=y for x (normalized xyz). gf=(1,3,T,T,T) forward map, jf=(1,9,T,T,T)
    its Jacobian. Seed x0 = R^-1(y-t) (affine inverse); x <- x + Jx^-1 (y - Φ(x)). The SVF
    warp is diffeomorphic so Jx is invertible everywhere. Returns (x, max|Φ(x)-y| in-domain)."""
    eye = torch.eye(3, device=y_xyz.device)
    x = (y_xyz - t) @ torch.linalg.inv(R).T
    for _ in range(n_iter):
        r = y_xyz - _sample_field(gf, x)                  # (T,T,T,3)
        Jx = _sample_field(jf, x).reshape(*x.shape[:-1], 3, 3)
        # Levenberg-Marquardt damping: the composed grid is clamp(-1,1)'d, so Jx is exactly
        # singular wherever the deform pushed it past the border -> λI keeps the solve well
        # posed (and degrades to a damped gradient step there, which is the right behaviour
        # since those output voxels have no true preimage).
        dx = torch.linalg.solve(Jx + 1e-2 * eye, r.unsqueeze(-1)).squeeze(-1)
        x = (x + dx).clamp(-1.2, 1.2)
    resid = (_sample_field(gf, x) - y_xyz)[(x.abs() <= 1).all(-1)].abs()
    return x, (resid.max().item() if resid.numel() else float("nan"))


def warp(method, vol6, cg0, cg1, geo0, geo1, T):
    """vol6: (B,1,T,T,T) on the level-0 aug grid -> (B,1,T,T,T) resampled onto the level-1
    aug grid. For each level-1 output voxel g1 we find the matching level-0 voxel g0 via the
    shared native-CT frame (crop_geom) and, for M2, the shared grid affine (R,t):

      g1 -> [R g1 + t] -> flip1 -> cropgeom1_fwd -> nativeCT -> cropgeom0_inv -> flip0
         -> [R^-1(. - t)] -> g0
    """
    B = vol6.shape[0]
    dev = vol6.device
    base = _lattice(T, dev)                                   # (T,T,T,3) voxel (d,h,w)
    aff = method in ("M2", "M3")
    outs, resids = [], []
    for b in range(B):
        gf = _grid_field(geo0.grid[b]) if method == "M3" else None
        jf = _jac_field(geo0.grid[b], T) if method == "M3" else None
        R = t = None
        if aff:
            A, _ = _fit_affine(geo0.grid[b], T, dev)          # grid0 == grid1 (checked)
            R, t = A[:, :3], A[:, 3]

        n = _norm(base, T)                                    # level-1 output, normalized (d,h,w)
        if aff:                                               # aug1 output -> source coord: Φ
            n_xyz = n.flip(-1)
            n_xyz = _sample_field(gf, n_xyz) if method == "M3" else n_xyz @ R.T + t
            n = n_xyz.flip(-1)
        if method != "M0":
            n = _flip_norm(n, geo1.flips[b])                  # -> native_crop_1
        g1 = _denorm(n, T)
        g0 = _crop_compose_g0(g1, cg0, cg1, b)                # native-CT compose
        n0 = _norm(g0, T)
        if method != "M0":
            n0 = _flip_norm(n0, geo0.flips[b])               # native_crop_0 -> source coord
        if aff:                                               # source coord -> aug0 voxel: Φ^-1
            n0_xyz = n0.flip(-1)
            if method == "M3":
                n0_xyz, r = _invert_grid(gf, jf, n0_xyz, R, t)
                resids.append(r)
            else:
                n0_xyz = (n0_xyz - t) @ torch.linalg.inv(R).T
            n0 = n0_xyz.flip(-1)
        grid_b = n0.flip(-1)[None]                            # -> xyz for grid_sample
        outs.append(F.grid_sample(vol6[b:b + 1], grid_b, mode="bilinear",
                                  padding_mode="zeros", align_corners=False))
    warp.last_resid = float(np.nanmean(resids)) if resids else None
    return torch.cat(outs, 0)


# ---------------------------------------------------------------------------- metrics
def _dice(a, b):
    a, b = a > 0.5, b > 0.5
    inter = (a & b).sum().item()
    s = a.sum().item() + b.sum().item()
    return 2 * inter / s if s else float("nan")


def _com_mm(m, spacing):
    m = (m > 0.5).float()
    if m.sum() < 1:
        return None
    idx = torch.nonzero(m)[:, -3:].float().mean(0)            # (d,h,w)
    return idx * spacing


METHODS = ["M0", "M1", "M2", "M3"]
agg = {m: {"dice": [], "com": [], "ms": []} for m in METHODS}
rows = []                          # per-case: {cls, subj, gt6_vox, <m>_dice, <m>_com}
fig_classes = set()
com_center_delta = []              # |invert_geo_center(COM) - COM(dewarped)| in native voxels
n_cases = 0

for step, batch in enumerate(loader):
    if n_cases >= MAX_CASES:
        break
    geo_seed = SEED * 1_000_003 + step
    cur0 = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
    cur0, geo0 = augmentor.apply(cur0, geo_gen=_gen(geo_seed, DEVICE),
                                 int_gen=_gen(geo_seed + INT_OFFSET, DEVICE), capture=True)
    T = cur0["image"].shape[-1]
    B = cur0["image"].shape[0]

    with torch.no_grad():
        logit6 = _forward_level(model, cur0, S_COARSE)        # (B,1,T,T,T)
    prob6 = torch.sigmoid(logit6)
    gt6 = cur0["label"].unsqueeze(1).float()                  # (B,1,T,T,T) aug-6 GT

    # level-0 predicted COM -> native centers (the existing cascade bridge)
    cens = _centroid_from_logit(logit6, T, is_prob)
    centers = []
    for b in range(B):
        centers.append(invert_geo_center(cens[b], geo0.grid[b], geo0.flips[b],
                                         cur0["crop_geom"][b], T))

    # DIAGNOSTIC: current center = invert_geo_center(COM_aug) vs proposed = COM(dewarped pred).
    # Equal for affine aug (COM commutes with affine); should diverge under deform.
    for b in range(B):
        if centers[b] is None:
            continue
        nat = _dewarp_native(prob6[b:b + 1], geo0.grid[b], geo0.flips[b], T,
                             use_newton=DEFORM_P > 0)[0, 0].detach().cpu().numpy()
        gc = _grid_centroid(nat)
        if gc is None:
            continue
        s, cs, o, pl = (cur0["crop_geom"][b][r].tolist() for r in range(4))
        cB = tuple(max(0, int(round(s[a] + (gc[a] - pl[a]) / max(1, o[a]) * cs[a])))
                   for a in range(3))
        com_center_delta.append(float(np.linalg.norm(np.array(centers[b]) - np.array(cB))))

    cur1 = _recrop_level(provider, batch, centers, S_FINE, step=step, seed=SEED,
                         level=1, jitter=0, recrop_workers=0)
    cur1 = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in cur1.items()}
    cur1, geo1 = augmentor.apply(cur1, geo_gen=_gen(geo_seed, DEVICE),
                                 int_gen=_gen(geo_seed + INT_OFFSET * 2, DEVICE), capture=True)
    gt3 = cur1["label"].unsqueeze(1).float()

    grid_diff = (geo0.grid - geo1.grid).abs().max().item()
    flip_eq = bool((geo0.flips == geo1.flips).all())
    if step == 0:
        print(f"\nshared-grid check: max|grid0-grid1|={grid_diff:.2e}  flips_equal={flip_eq}")
        _, r0 = _fit_affine(geo0.grid[0], T, DEVICE)
        print(f"affine-fit residual (grid0[0]): {r0:.2e}  "
              f"(near 0 => pure affine, no elastic/deform)")

    wp_gt = {}
    for m in METHODS:
        for _ in range(1 if step else 2):                    # 1 warmup on step 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wg = warp(m, gt6, cur0["crop_geom"], cur1["crop_geom"], geo0, geo1, T)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
        wp_gt[m] = wg
        agg[m]["ms"].append(dt / B)
        if m == "M3" and warp.last_resid is not None:
            agg[m].setdefault("fpres", []).append(warp.last_resid)

    wp_prob = {m: warp(m, prob6, cur0["crop_geom"], cur1["crop_geom"], geo0, geo1, T)
               for m in METHODS}
    ct3 = cur1["image"][:, 0].cpu().numpy()

    for b in range(B):
        if n_cases >= MAX_CASES:
            break
        cls, subj = batch["label_names"][b], batch["subjects"][b]
        row = {"cls": cls, "subj": subj, "gt6_vox": int(gt6[b, 0].sum())}
        c1 = _com_mm(gt3[b, 0], S_FINE)
        for m in METHODS:
            d = _dice(wp_gt[m][b, 0], gt3[b, 0])
            c0 = _com_mm(wp_gt[m][b, 0], S_FINE)
            com = float((c0 - c1).norm()) if (c0 is not None and c1 is not None) else float("nan")
            row[f"{m}_dice"], row[f"{m}_com"] = d, com
            agg[m]["dice"].append(d); agg[m]["com"].append(com)
        rows.append(row)
        n_cases += 1

        if cls not in fig_classes and int(gt3[b, 0].sum()) > 0:
            fig_classes.add(cls)
            z = int(np.asarray(gt3[b, 0].cpu()).sum(axis=(1, 2)).argmax())
            fig, ax = plt.subplots(1, 1 + len(METHODS), figsize=(4 * (1 + len(METHODS)), 4),
                                   gridspec_kw={"wspace": 0.04})
            _overlay(ax[0], ct3[b][z], [(gt3[b, 0].cpu().numpy()[z], "lime", 0.45)])
            ax[0].set_title(f"{cls} {subj}\nfine GT  gt6={row['gt6_vox']}vox", fontsize=8)
            for j, m in enumerate(METHODS):
                _overlay(ax[j + 1], ct3[b][z],
                         [(gt3[b, 0].cpu().numpy()[z], "lime", 0.30),
                          (wp_prob[m][b, 0].cpu().numpy()[z], "red", 0.45)])
                ax[j + 1].set_title(f"{m}: warped 6mm pred\n"
                                    f"gtDice={row[f'{m}_dice']:.3f} "
                                    f"comΔ={row[f'{m}_com']:.1f}mm", fontsize=8)
            fig.suptitle(f"{cls}  grid0-1 max diff {grid_diff:.1e}", fontsize=9)
            plt.savefig(OUT / f"{cls}_{subj}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

# ---------------------------------------------------------------------------- report
print(f"\ndeform p={DEFORM_P:g}  elastic p={ELASTIC_P:g}  M3 iters={M3_ITERS}  "
      f"n_cases={n_cases}")
if com_center_delta:
    cd = np.array(com_center_delta)
    print(f"center delta  invert_geo_center(COM) vs COM(dewarped pred): "
          f"mean {cd.mean():.3f}  max {cd.max():.3f}  native voxels  (n={len(cd)})")
print(f"\n{'method':<6} {'gt-warp Dice':>14} {'COM Δ mm':>13} {'ms/vol':>9} {'fp-resid':>10}")
for m in METHODS:
    d = np.array(agg[m]["dice"]); c = np.array(agg[m]["com"]); t = np.array(agg[m]["ms"])
    fp = agg[m].get("fpres")
    fps = f"{np.mean(fp):.2e}" if fp else "-"
    print(f"{m:<6} {np.nanmean(d):>8.3f}±{np.nanstd(d):<5.3f} "
          f"{np.nanmean(c):>7.2f}±{np.nanstd(c):<5.2f} {t.mean():>8.2f} {fps:>10}")

print(f"\nper-class (mean over cases)  —  M2 / M3 warp fidelity")
print(f"{'class':<30} {'n':>2} {'gt6vox':>7} "
      f"{'M2 Dice':>8} {'M2 comΔ':>8} {'M3 Dice':>8} {'M3 comΔ':>8}")
by_cls = {}
for r in rows:
    by_cls.setdefault(r["cls"], []).append(r)
for cls in sorted(by_cls, key=lambda c: np.nanmean([r["M2_com"] for r in by_cls[c]]),
                  reverse=True):
    rs = by_cls[cls]
    g = np.mean([r["gt6_vox"] for r in rs])
    print(f"{cls:<30} {len(rs):>2} {g:>7.0f} "
          f"{np.nanmean([r['M2_dice'] for r in rs]):>8.3f} "
          f"{np.nanmean([r['M2_com'] for r in rs]):>8.2f} "
          f"{np.nanmean([r['M3_dice'] for r in rs]):>8.3f} "
          f"{np.nanmean([r['M3_com'] for r in rs]):>8.2f}")
print(f"\nfigures -> {OUT}")

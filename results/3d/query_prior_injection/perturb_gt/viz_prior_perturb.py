"""Visualize src.mask_transforms.perturb_prior_mask's individual knobs on real organ
crops, at exp70's mid-range crop_spacing_mm=3.0, T=128 (same provider path as training).

Two organs (small=gallbladder, medium=kidney_right) so size-dependent vulnerability is
visible (small organs are hit much harder by the same mm-radius op).

Outputs:
  fig1_qualitative_slices.png  — axial-slice overlays: GT vs one perturbed variant per
                                  panel (yellow=agreement, green=missed GT, red=added).
  fig2_dice_vs_magnitude.png   — Dice(perturbed, GT) vs magnitude, one line per knob,
                                  with exp70's CURRENT vs PROPOSED range ceilings marked.
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# results/3d/query_prior_injection/perturb_gt/ -> repo root is 4 levels up.
OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT))
from src.incontext_dataset_v2 import LoadRequest
from src.mask_transforms import (add_gaussian_noise, dilate, erode,
                                  perturb_prior_mask, translate, mm_to_vox)
from src.providers.totalseg import TotalSegProvider

ROOT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
        "ANALYSIS_20251122/data/totalseg")
SPACING_MM = 3.0                        # mid of exp70's [1.5, 6] range
T = 128
SHIFT_DIR = torch.tensor([1.0, 1.0, 1.0]) / (3 ** 0.5)  # fixed diagonal direction

CURRENT_CFG = dict(p=1.0, ball=False, dilate_mm=[0.0, 4.0], erode_mm=[0.0, 4.0],
                   erode_min_keep=0.3, shift_mm=[0.0, 6.0], noise_std=[0.0, 0.3])
# Floors picked off the measured Dice-vs-magnitude knees (fig2/fig3), not guessed: dilate
# /erode need >=2mm on a small organ to clear the cliff between 1-2mm; shift is gradual
# (no cliff) so its floor is raised further to bite at all; noise_std is a near step
# function with its knee around 0.10-0.12, so a 0.1 floor is still pre-knee/no-op.
PROPOSED_CFG = dict(p=1.0, ball=False, dilate_mm=[2.0, 8.0], erode_mm=[2.0, 8.0],
                    erode_min_keep=0.15, shift_mm=[4.0, 12.0], noise_std=[0.15, 0.4])


def dice(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = (a > 0.5).float(), (b > 0.5).float()
    inter = (a * b).sum()
    denom = a.sum() + b.sum()
    return float((2 * inter / denom).item()) if denom > 0 else 1.0


def load_organ(cls: str, subj_idx: int = 0):
    p = TotalSegProvider(ROOT, classes=[cls], image_size=(T, T, T), split=None,
                        crop_spacing_mm=SPACING_MM)
    subs = p.subjects_for(cls)
    subj = subs[subj_idx]
    r = p.load(subj, cls, LoadRequest(rng=random.Random(0), crop_spacing_mm=SPACING_MM))
    lbl = r.label.float().unsqueeze(0).unsqueeze(0)   # (1,1,T,T,T)
    return subj, lbl


def centroid_slice(mask_1_1_t: torch.Tensor) -> int:
    """Axial (dim -3) slice index with the most foreground."""
    m = mask_1_1_t[0, 0]
    counts = m.sum(dim=(1, 2))
    return int(counts.argmax().item())


def overlay_rgb(gt_slice: np.ndarray, pred_slice: np.ndarray) -> np.ndarray:
    """(H,W) binary/float in [0,1] x2 -> (H,W,3) RGB: yellow=agree, green=missed(FN),
    red=added(FP)."""
    g, p_ = np.clip(gt_slice, 0, 1), np.clip(pred_slice, 0, 1)
    rgb = np.zeros((*g.shape, 3), dtype=np.float32)
    rgb[..., 0] = p_                 # red channel = perturbed/pred
    rgb[..., 1] = g                  # green channel = GT
    return rgb


def qualitative_figure(organs):
    cols = [
        ("GT", lambda m, sp, gen: m),
        ("dilate 3mm", lambda m, sp, gen: dilate(m, mm_to_vox(3.0, sp))),
        ("erode 3mm\n(min_keep=0.3)", lambda m, sp, gen: erode(m, mm_to_vox(3.0, sp), min_keep=0.3)),
        ("erode 3mm\n(min_keep=0)", lambda m, sp, gen: erode(m, mm_to_vox(3.0, sp), min_keep=0.0)),
        ("shift 6mm", lambda m, sp, gen: translate(m, (SHIFT_DIR * 6.0 / sp))),
        ("noise 0.3", lambda m, sp, gen: add_gaussian_noise(m, 0.3, gen)),
        ("current cfg\n(1 draw)", lambda m, sp, gen: perturb_prior_mask(m, CURRENT_CFG, sp, gen)),
        ("proposed cfg\n(1 draw)", lambda m, sp, gen: perturb_prior_mask(m, PROPOSED_CFG, sp, gen)),
    ]
    fig, axes = plt.subplots(len(organs), len(cols), figsize=(2.1 * len(cols), 2.3 * len(organs)))
    for i, (name, subj, mask) in enumerate(organs):
        z = centroid_slice(mask)
        gt_slice = mask[0, 0, z].numpy()
        for j, (title, fn) in enumerate(cols):
            gen = torch.Generator().manual_seed(1000 * i + j)
            out = fn(mask.clone(), SPACING_MM, gen)
            out_slice = out[0, 0, z].numpy()
            d = dice(torch.from_numpy(out_slice > 0.5).float()[None, None],
                    torch.from_numpy(gt_slice > 0.5).float()[None, None]) if j > 0 else 1.0
            ax = axes[i, j]
            ax.imshow(overlay_rgb(gt_slice, out_slice), origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{name}\n({subj})", fontsize=8)
            ax.text(0.02, 0.02, f"Dice={d:.2f}", color="white", fontsize=7,
                    transform=ax.transAxes, va="bottom")
    fig.suptitle(f"perturb_prior_mask knobs — axial slice through organ centroid "
                f"(crop_spacing_mm={SPACING_MM}, T={T})\n"
                f"yellow=agree with GT, green=missed(FN), red=added(FP)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def dice_vs_magnitude_figure(organs):
    dilate_mags = np.arange(0, 10.01, 1.0)
    erode_mags = np.arange(0, 10.01, 1.0)
    shift_mags = np.arange(0, 15.01, 1.5)
    noise_mags = np.arange(0, 0.61, 0.05)

    fig, axes = plt.subplots(1, len(organs), figsize=(6 * len(organs), 4.5))
    if len(organs) == 1:
        axes = [axes]
    for ax, (name, subj, mask) in zip(axes, organs):
        gt = mask
        curves = {
            "dilate_mm": (dilate_mags,
                         lambda v, gen: dilate(mask.clone(), mm_to_vox(v, SPACING_MM))),
            "erode_mm (min_keep=0.3)": (erode_mags,
                                        lambda v, gen: erode(mask.clone(), mm_to_vox(v, SPACING_MM), min_keep=0.3)),
            "shift_mm": (shift_mags,
                        lambda v, gen: translate(mask.clone(), (SHIFT_DIR * v / SPACING_MM))),
            "noise_std": (noise_mags,
                         lambda v, gen: add_gaussian_noise(mask.clone(), v, gen)),
        }
        for label, (mags, fn) in curves.items():
            ds = []
            for v in mags:
                gen = torch.Generator().manual_seed(42)
                out = fn(float(v), gen)
                ds.append(dice(out, gt))
            ax.plot(mags, ds, marker="o", markersize=3, label=label)
        # mark current vs proposed range ceilings for the matching knob families
        ax.axvline(CURRENT_CFG["dilate_mm"][1], color="C0", ls=":", alpha=0.6)
        ax.axvline(PROPOSED_CFG["dilate_mm"][1], color="C0", ls="--", alpha=0.6)
        ax.axvline(CURRENT_CFG["erode_mm"][1], color="C1", ls=":", alpha=0.6)
        ax.axvline(PROPOSED_CFG["erode_mm"][1], color="C1", ls="--", alpha=0.6)
        ax.axvline(CURRENT_CFG["shift_mm"][1], color="C2", ls=":", alpha=0.6)
        ax.axvline(PROPOSED_CFG["shift_mm"][1], color="C2", ls="--", alpha=0.6)
        ax.axvline(CURRENT_CFG["noise_std"][1], color="C3", ls=":", alpha=0.6)
        ax.axvline(PROPOSED_CFG["noise_std"][1], color="C3", ls="--", alpha=0.6)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("magnitude (mm, or std for noise)")
        ax.set_ylabel("Dice(perturbed, clean GT)")
        ax.set_title(f"{name} ({subj})")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Dice-vs-magnitude per knob. Dotted vline = CURRENT range ceiling "
                "(dataset=d1 default, exp70 as-is); dashed vline = PROPOSED ceiling.\n"
                "A ceiling landing near Dice~1.0 means that knob's current range still "
                "lets a near-clean copy through.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


if __name__ == "__main__":
    organs_spec = [("gallbladder (small)", "gallbladder"), ("kidney_right (medium)", "kidney_right")]
    organs = []
    for label, cls in organs_spec:
        subj, mask = load_organ(cls)
        n_fg = int(mask.sum().item())
        print(f"{cls}: subject={subj}, fg_voxels={n_fg}")
        organs.append((label, subj, mask))

    fig1 = qualitative_figure(organs)
    fig1.savefig(str(OUT_DIR / "fig1_qualitative_slices.png"), dpi=150)
    print("wrote fig1_qualitative_slices.png")

    fig2 = dice_vs_magnitude_figure(organs)
    fig2.savefig(str(OUT_DIR / "fig2_dice_vs_magnitude.png"), dpi=150)
    print("wrote fig2_dice_vs_magnitude.png")


def noise_fine_figure(organs):
    """Fine-grained noise_std sweep — the coarse fig2 range (0-0.6) compresses the
    interesting knee into a sliver; zoom into 0-0.25 where the current/proposed floors
    actually land."""
    mags = np.arange(0, 0.251, 0.01)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, subj, mask in organs:
        ds = []
        for v in mags:
            gen = torch.Generator().manual_seed(42)
            out = add_gaussian_noise(mask.clone(), float(v), gen)
            ds.append(dice(out, mask))
        ax.plot(mags, ds, marker="o", markersize=3, label=f"{name} ({subj})")
    ax.axvline(CURRENT_CFG["noise_std"][0], color="gray", ls=":", alpha=0.7,
              label=f'{CURRENT_CFG["noise_std"][0]:.2f} (current floor, no-op)')
    ax.axvline(PROPOSED_CFG["noise_std"][0], color="gray", ls="--", alpha=0.7,
              label=f'{PROPOSED_CFG["noise_std"][0]:.2f} (proposed floor, past the knee)')
    ax.set_xlabel("noise_std"); ax.set_ylabel("Dice(perturbed, clean GT)")
    ax.set_title("noise_std fine sweep (whole-volume additive Gaussian, T=128 raw voxels)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


organs_spec = [("gallbladder (small)", "gallbladder"), ("kidney_right (medium)", "kidney_right")]
organs = []
for label, cls in organs_spec:
    subj, mask = load_organ(cls)
    organs.append((label, subj, mask))
fig3 = noise_fine_figure(organs)
fig3.savefig(str(OUT_DIR / "fig3_noise_fine.png"), dpi=150)
print("wrote fig3_noise_fine.png")

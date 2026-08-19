"""Visualize the SVF `deform` aug on real GT labels at the calibrated max_disp=0.15.

For a few (subject, organ) examples: axial slice through the organ centroid showing
original CT+mask, the SVF-deformed CT+mask (with the original contour overlaid), and
the warped regular grid that visualizes the diffeomorphic deformation field.
Saves deform_examples.png.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
from data.totalseg_classes import ALL_CLASSES
from src.augmentations import _svf_displacement

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
NAME2IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}
# control_points is a COUNT (resolution-invariant); 6 = the calibrated correlation scale.
MAX_DISP, CONTROL_POINTS, NUM_STEPS = 0.15, 6, 6
EXAMPLES = [("s0001", "liver", 11), ("s0004", "spleen", 7), ("s0006", "kidney_left", 33),
            ("s0009", "aorta", 3)]   # (subject, organ, seed)


def grid_pattern(R, step=8):
    g = np.zeros((R, R, R), np.float32)
    g[::step] = 1; g[:, ::step] = 1; g[:, :, ::step] = 1
    return torch.from_numpy(g)[None, None]


def main():
    R = 128
    base = F.affine_grid(torch.eye(3, 4)[None], (1, 1, R, R, R), align_corners=False)
    rows = len(EXAMPLES)
    fig, ax = plt.subplots(rows, 3, figsize=(10.5, 3.4 * rows))
    if rows == 1:
        ax = ax[None]

    for r, (subj, organ, seed) in enumerate(EXAMPLES):
        ct = np.load(os.path.join(TS, subj, "ct_128x128x128.npy")).astype(np.float32)
        lab = np.load(os.path.join(TS, subj, "label_128x128x128.npy"))
        idx = NAME2IDX[organ]
        mask = (lab == idx).astype(np.float32)
        if mask.sum() < 20:
            print("skip", subj, organ, "empty"); continue
        z = int(np.round(np.argwhere(mask).mean(0)[0]))          # organ-centroid axial slice

        ct_t = torch.from_numpy(ct)[None, None]
        m_t = torch.from_numpy(mask)[None, None]
        gp = grid_pattern(R)

        g = torch.Generator().manual_seed(seed)
        phi = _svf_displacement((R, R, R), CONTROL_POINTS, MAX_DISP, NUM_STEPS, generator=g)
        grid = (base + phi).clamp(-1, 1)
        ct_d = F.grid_sample(ct_t, grid, mode="bilinear", padding_mode="border", align_corners=False)
        m_d = F.grid_sample(m_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        gp_d = F.grid_sample(gp, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        ct0, mk0 = ct[z], mask[z]
        ctd, mkd = ct_d[0, 0, z].numpy(), (m_d[0, 0, z].numpy() > 0.5)
        gpd = gp_d[0, 0, z].numpy()
        vmin, vmax = np.percentile(ct0, 2), np.percentile(ct0, 98)

        ax[r, 0].imshow(ct0, cmap="gray", vmin=vmin, vmax=vmax)
        ax[r, 0].contour(mk0, levels=[0.5], colors="red", linewidths=1.5)
        ax[r, 0].set_ylabel(f"{subj}  {organ}", fontsize=10)
        ax[r, 0].set_title("original (GT contour)" if r == 0 else "")

        ax[r, 1].imshow(ctd, cmap="gray", vmin=vmin, vmax=vmax)
        ax[r, 1].contour(mkd, levels=[0.5], colors="red", linewidths=1.5)
        ax[r, 1].contour(mk0, levels=[0.5], colors="cyan", linewidths=1.0, linestyles="dashed")
        ax[r, 1].set_title("SVF-deformed (cyan=orig)" if r == 0 else "")

        ax[r, 2].imshow(gpd, cmap="magma")
        ax[r, 2].set_title(f"warped grid (max_disp={MAX_DISP})" if r == 0 else "")
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    fig.suptitle(f"Diffeomorphic SVF deform on real GT labels "
                 f"(max_disp={MAX_DISP}, ~6 control pts)", y=1.0)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deform_examples.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

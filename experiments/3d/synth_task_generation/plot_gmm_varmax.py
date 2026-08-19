"""
Show that intra-label (within-region) texture is the var_max knob (§3/§8): the default
var_max=5 gives sigma<=2.24 on 0-255 → near-piecewise-constant (flat). Sweeping var_max
up makes per-voxel jitter visible within each slot. Same cohort seed across columns so
only sigma grows (means fixed) — isolates the within-region effect.

  .venv_thor/bin/python experiments/3d/synth_task_generation/plot_gmm_varmax.py
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.gpu_gmm_intensity import synthesize_intensities, pack_label_ids
from src.totalseg_dataloader_incontext import organ_crop_arrays, place_label

BANK = Path("/tmp/gmm_bank")
T, CROP_MM, BODY = 128, 1.5, 200
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAR_MAX = [5, 50, 200, 1000, 4000]           # sigma = sqrt(var): 2.2, 7, 14, 32, 63


def gen(s):
    g = torch.Generator(device=DEV); g.manual_seed(s); return g


def crop_label(e, rng):
    lbl_mm = np.squeeze(np.load(BANK / e["file"], mmap_mode="r"))
    organs = [k for k in e["cents"] if k != str(BODY)]
    center = tuple(e["cents"][rng.choice(organs)][:3])
    _, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
        lbl_mm, lbl_mm, center, e["spacing"], image_size=(T, T, T),
        crop_mm=CROP_MM, jitter=0, rng=random)
    small = F.interpolate(torch.from_numpy(crop_lbl.astype(np.float32))[None, None],
                          size=tuple(out_sizes), mode="nearest")[0, 0].long()
    return place_label(small, out_sizes, pad_lo, T).to(DEV)


def main():
    out = Path("results/synth_task_gen/gmm_varmax.png")
    rng = random.Random(2)
    index = json.load(open(BANK / "index.json"))
    picks = rng.sample(index, 3)
    ncol = 1 + len(VAR_MAX)
    fig, ax = plt.subplots(3, ncol, figsize=(2.6 * ncol, 2.6 * 3))
    lut = np.random.default_rng(0).random((256, 3)); lut[0] = 0
    z = T // 2
    for r, e in enumerate(picks):
        lab = crop_label(e, rng)
        packed, L = pack_label_ids(lab[None], container_id=BODY)
        ax[r, 0].imshow(lut[np.clip(lab.cpu().numpy()[:, :, z], 0, 255)])
        ax[r, 0].set_title("mask", fontsize=8); ax[r, 0].axis("off")
        for c, vm in enumerate(VAR_MAX):
            # SAME cohort seed → same means; only sigma grows with var_max
            img = synthesize_intensities(packed, L, gen(100 + r), gen(7), var_max=vm)
            ax[r, c + 1].imshow(img.cpu().numpy()[0, 0, :, :, z], cmap="gray", vmin=0, vmax=255)
            ax[r, c + 1].set_title(f"var_max={vm} (σ≤{vm ** 0.5:.0f})", fontsize=8)
            ax[r, c + 1].axis("off")
        print(f"[{r}] {e['file']} L={L}", flush=True)
    fig.suptitle("Intra-label texture = var_max knob (means fixed per row; σ=√var grows →)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

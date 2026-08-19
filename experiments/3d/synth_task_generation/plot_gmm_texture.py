"""
Show WHERE intra-label texture comes from in the chosen design (spec default): keep the
GMM flat (var_max=5) and add texture with the downstream stages —
  flat GMM  →  smooth multiplicative bias field  →  correlated Gaussian noise.
Unlike raising var_max (white per-voxel noise), bias gives spatially-correlated gradients
across a region and blur-correlated noise gives structured grain. This mirrors the GPU
stages in src/gpu_augment.py (_batched_bias_field + noise+blur), applied here on the raw
0-255 GMM scale (normalization is downstream, §5).

  .venv_thor/bin/python experiments/3d/synth_task_generation/plot_gmm_texture.py
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


def gen(s):
    g = torch.Generator(device=DEV); g.manual_seed(s); return g


def bias_field(vols, mag, coarse, g):
    """Smooth multiplicative log-normal field (batched _batched_bias_field, no CT clamp)."""
    N = vols.shape[0]
    f = torch.randn(N, 1, coarse, coarse, coarse, generator=g, device=vols.device) * mag
    f = F.interpolate(f, size=vols.shape[-3:], mode="trilinear", align_corners=False)
    return vols * f.exp()


def corr_noise(vols, std, blur_sigma, g):
    """Gaussian noise then a light blur so it becomes spatially correlated (structured grain)."""
    n = std * torch.randn(vols.shape, generator=g, device=vols.device)
    out = vols + n
    # separable-ish blur via a small average pool round-trip (cheap correlate for the demo)
    k = 3
    pad = k // 2
    w = torch.ones(1, 1, k, k, k, device=vols.device) / k ** 3
    blurred = F.conv3d(F.pad(out, [pad] * 6, mode="replicate"), w)
    return out * (1 - blur_sigma) + blurred * blur_sigma


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
    out = Path("results/synth_task_gen/gmm_texture.png")
    rng = random.Random(5)
    index = json.load(open(BANK / "index.json"))
    picks = rng.sample(index, 3)
    cols = ["mask", "GMM flat (var=5)", "+ bias field", "+ bias + noise",
            "+ bias + noise (scanner B)"]
    fig, ax = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 2.6 * 3))
    lut = np.random.default_rng(0).random((256, 3)); lut[0] = 0
    z = T // 2
    for r, e in enumerate(picks):
        lab = crop_label(e, rng)
        packed, L = pack_label_ids(lab[None], container_id=BODY)
        base = synthesize_intensities(packed, L, gen(100 + r), gen(7))       # flat
        biased = bias_field(base, mag=0.35, coarse=5, g=gen(200 + r))
        textured = corr_noise(biased, std=7.0, blur_sigma=0.6, g=gen(300 + r))
        base_b = synthesize_intensities(packed, L, gen(900 + r), gen(7))     # new scanner
        textured_b = corr_noise(bias_field(base_b, 0.35, 5, gen(210 + r)),
                                7.0, 0.6, gen(310 + r))
        views = [None, base, biased, textured, textured_b]
        for c, v in enumerate(views):
            if c == 0:
                ax[r, c].imshow(lut[np.clip(lab.cpu().numpy()[:, :, z], 0, 255)])
            else:
                ax[r, c].imshow(v.cpu().numpy()[0, 0, :, :, z], cmap="gray", vmin=0, vmax=255)
            ax[r, c].set_title(cols[c], fontsize=8); ax[r, c].axis("off")
        print(f"[{r}] {e['file']} L={L}", flush=True)
    fig.suptitle("Chosen design: flat GMM + downstream bias/noise = structured intra-label "
                 "texture (var_max stays 5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()

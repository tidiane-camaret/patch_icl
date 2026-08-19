"""
Plot samples from the GPU GMM intensity stage (src/gpu_gmm_intensity.py). Sample masks
from the /tmp bank, organ-crop, pack ids (body 200 → container L+1), then synthesize
under a few cohort seeds (= different "scanners") to show the domain-randomization
diversity. Also shows the cohort-sharing invariant: two subjects under ONE scanner.

  .venv_thor/bin/python experiments/3d/synth_task_generation/plot_gmm_gpu.py
"""
import argparse
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


def crop_label(e, rng):
    """Organ-centred 128³ multiclass label crop (int64, on device) from a bank mask."""
    lbl_mm = np.squeeze(np.load(BANK / e["file"], mmap_mode="r"))
    organs = [k for k in e["cents"] if k != str(BODY)]
    cls = rng.choice(organs)
    center = tuple(e["cents"][cls][:3])
    _, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
        lbl_mm, lbl_mm, center, e["spacing"], image_size=(T, T, T),
        crop_mm=CROP_MM, jitter=0, rng=random)
    small = F.interpolate(torch.from_numpy(crop_lbl.astype(np.float32))[None, None],
                          size=tuple(out_sizes), mode="nearest")[0, 0].long()
    full = place_label(small, out_sizes, pad_lo, T)          # (T,T,T) multiclass
    return full.to(DEV), int(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_masks", type=int, default=5)
    ap.add_argument("--scanners", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("results/synth_task_gen/gmm_gpu_samples.png"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    index = json.load(open(BANK / "index.json"))
    picks = rng.sample(index, a.n_masks)
    lut = np.random.default_rng(0).random((256, 3)); lut[0] = 0

    ncol = 2 + a.scanners           # mask | 2 subjects(same scanner) | (scanners-1) more scanners
    fig, ax = plt.subplots(a.n_masks, ncol, figsize=(2.7 * ncol, 2.7 * a.n_masks))
    for r, e in enumerate(picks):
        lab, cls = crop_label(e, rng)
        packed, L = pack_label_ids(lab[None], container_id=BODY)   # [1,D,H,W]
        z = T // 2

        ax[r, 0].imshow(lut[np.clip(lab.cpu().numpy()[:, :, z], 0, 255)])
        ax[r, 0].set_title(f"mask (L={L}, tgt id {cls})", fontsize=8); ax[r, 0].axis("off")

        # cols 1..2: TWO subjects under ONE scanner (cohort-shared mu/sigma, different noise)
        pair = packed.expand(2, -1, -1, -1).contiguous()
        img_pair = synthesize_intensities(pair, L, gen(100 + r), gen(7)).cpu().numpy()
        for j in range(2):
            ax[r, 1 + j].imshow(img_pair[j, 0, :, :, z], cmap="gray", vmin=0, vmax=255)
            ax[r, 1 + j].set_title(f"scanner A, subj {j}", fontsize=8); ax[r, 1 + j].axis("off")

        # remaining cols: other scanners (different cohort draw → different contrast)
        for s in range(a.scanners - 1):
            img = synthesize_intensities(packed, L, gen(500 + 13 * r + s), gen(7)).cpu().numpy()
            c = 3 + s
            ax[r, c].imshow(img[0, 0, :, :, z], cmap="gray", vmin=0, vmax=255)
            ax[r, c].set_title(f"scanner {chr(66 + s)}", fontsize=8); ax[r, c].axis("off")
        print(f"[{r}] {e['file']} L={L} tgt={cls}", flush=True)

    fig.suptitle("GPU GMM intensity stage — MAISI mask + per-slot Gaussian (0-255). "
                 "cols 1-2 = same scanner (shared GMM), rest = new scanners", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()

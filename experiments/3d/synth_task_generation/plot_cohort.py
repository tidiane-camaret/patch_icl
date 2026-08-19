"""
Validate cohort sampling: draw a few cohorts of K+1 similar masks (CohortSampler), organ-
crop each around the target class, and paint them with ONE cohort-shared GMM indexed by the
shared MAISI anatomical id (drawn once per cohort; per-subject noise differs). So an organ
keeps a consistent shade across the cohort ("one scanner"), while different cohorts =
different scanners. Prints per-cohort tightness stats.

  .venv_thor/bin/python experiments/3d/synth_task_generation/plot_cohort.py \
    --bank /tmp/gmm_bank_native --k 4 --cohorts 4
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.gmm_cohort_sampler import CohortSampler
from src.gpu_gmm_intensity import synthesize_intensities
from src.totalseg_dataloader_incontext import organ_crop_arrays, place_label

T, CROP_MM, MAXID = 128, 1.5, 200
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def g(s):
    x = torch.Generator(device=DEV); x.manual_seed(s); return x


def crop_multiclass(bank, e, cls):
    """Organ-centred 128³ multiclass label (int64, device) around class `cls` in mask `e`."""
    arr = np.squeeze(np.load(bank / "masks" / e["file"], mmap_mode="r"))
    center = tuple(e["cents"][cls][:3])
    _, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
        arr, arr, center, e["spacing"], image_size=(T, T, T),
        crop_mm=CROP_MM, jitter=0, rng=random)
    small = F.interpolate(torch.from_numpy(crop_lbl.astype(np.float32))[None, None],
                          size=tuple(out_sizes), mode="nearest")[0, 0].long()
    return place_label(small, out_sizes, pad_lo, T).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/gmm_bank"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cohorts", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/synth_task_gen/cohort_samples.png"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    cs = CohortSampler(a.bank, k=a.k)
    print(f"bank: {len(cs.entries)} masks, {len(cs.classes)} classes with >= k+1={a.k + 1}",
          flush=True)

    ncol = a.k + 1
    fig, ax = plt.subplots(a.cohorts, ncol, figsize=(2.7 * ncol, 2.7 * a.cohorts))
    for r in range(a.cohorts):
        cls, cohort = cs.sample_cohort(rng)
        st = cs.cohort_stats(cls, cohort)
        print(f"[cohort {r}] class={cls} span={st['span_set']} "
              f"fov_std={st['fov_mm_std']:.1f}mm size_L1={st['size_L1_meanpair']:.3f}", flush=True)
        for j, e in enumerate(cohort):
            lab = crop_multiclass(a.bank, e, cls)                 # shared-id label
            # cohort-shared GMM (same cohort seed for the whole row), per-subject noise
            img = synthesize_intensities(lab[None], MAXID, g(1000 + r), g(10 * r + j))
            im = img.cpu().numpy()[0, 0, :, :, T // 2]
            ax[r, j].imshow(im, cmap="gray", vmin=0, vmax=255)
            # target-class contour
            m = (lab == cls).cpu().numpy()[:, :, T // 2]
            if m.any():
                ax[r, j].contour(m, levels=[0.5], colors="lime", linewidths=0.8)
            ax[r, j].set_title(f"cohort{r} subj{j}" + (" (anchor)" if j == 0 else ""), fontsize=8)
            ax[r, j].axis("off")
    fig.suptitle("Cohorts: K+1 similar masks under ONE shared-id GMM (row=scanner; green=target)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()

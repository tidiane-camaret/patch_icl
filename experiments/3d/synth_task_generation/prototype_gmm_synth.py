"""
Prototype: sample masks from the MAISI candidate-mask bank and paint them with a
fully-random per-label Gaussian mixture (SynthSeg / Billot et al. recipe) instead of
running the diffusion+VAE render. Realism-optional, ~free, label-perfect alignment.

Goal here = eyeball crop quality before wiring a dataset. Reads a few masks straight
from the bank zip (the real dataset will use a converted .npy bank).

  .venv_thor/bin/python experiments/3d/synth_task_generation/prototype_gmm_synth.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --n_masks 6 --seeds 3
"""
import argparse
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ZIP_ROOT = "all_masks_flexible_size_and_spacing_4000"
BODY_LABEL = 200  # generic soft-tissue envelope
CROP = 128


def load_mask_from_zip(zf, member):
    """Read a gz-compressed nii.gz member straight from the zip into a numpy array."""
    import gzip
    raw = gzip.decompress(zf.read(member))
    img = nib.Nifti1Image.from_bytes(raw)
    return np.asarray(img.dataobj), np.abs(np.diag(img.affine))[:3]


def organ_centre_crop(lab, rng, crop=CROP):
    """Crop a `crop`^3 window centred on a random real organ (not air/body)."""
    present = np.unique(lab)
    organs = [l for l in present if l not in (0, BODY_LABEL) and (lab == l).sum() > 500]
    if not organs:
        organs = [l for l in present if l != 0]
    l = int(rng.choice(organs))
    cen = np.array(ndi.center_of_mass(lab == l)).astype(int)
    starts = []
    for c, s in zip(cen, lab.shape):
        st = int(np.clip(c - crop // 2, 0, max(0, s - crop)))
        starts.append(st)
    sl = tuple(slice(st, st + crop) for st in starts)
    out = lab[sl]
    # pad if the mask was smaller than crop on some axis
    if out.shape != (crop, crop, crop):
        pad = [(0, crop - x) for x in out.shape]
        out = np.pad(out, pad)
    return out, l


def paint_gmm(lab, rng, mu_range=(0.0, 1.0), sd_range=(0.0, 0.15),
              blur_range=(0.5, 1.6), bias_strength=0.25, noise=0.03):
    """Fully-random SynthSeg: each label region -> N(mu_l, sd_l), then blur+bias+noise."""
    img = np.zeros(lab.shape, np.float32)
    for l in np.unique(lab):
        mu = rng.uniform(*mu_range)
        sd = rng.uniform(*sd_range)
        m = lab == l
        img[m] = rng.normal(mu, sd, size=int(m.sum()))
    # spatial smoothing (partial-volume-like)
    img = ndi.gaussian_filter(img, rng.uniform(*blur_range))
    # low-frequency multiplicative bias field
    bf = rng.normal(1.0, bias_strength, size=(4, 4, 4)).astype(np.float32)
    bf = ndi.zoom(bf, np.array(lab.shape) / 4.0, order=1)
    img = img * bf
    # global noise
    img = img + rng.normal(0.0, noise, size=img.shape).astype(np.float32)
    # per-volume min-max normalise (contrast is arbitrary anyway)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--n_masks", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("results/synth_task_gen/gmm_prototype.png"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    ds = a.repo / "datasets"
    cand = json.load(open(ds / "candidate_masks_flexible_size_and_spacing_4000.json"))
    zf = zipfile.ZipFile(ds / "all_masks_flexible_size_and_spacing_4000.zip")
    members = set(zf.namelist())

    # pick masks spread across source datasets
    by_src = {}
    for e in cand:
        src = e["pseudo_label_filename"].split("/")[1]
        by_src.setdefault(src, []).append(e)
    srcs = rng.permutation(sorted(by_src))[: a.n_masks]
    picks = [by_src[s][rng.integers(len(by_src[s]))] for s in srcs]

    ncol = 1 + a.seeds
    fig, axes = plt.subplots(a.n_masks, ncol, figsize=(3 * ncol, 3 * a.n_masks))
    # distinct random colour per label id for the mask view
    lut = rng.random((256, 3)); lut[0] = 0

    for r, e in enumerate(picks):
        member = f"{ZIP_ROOT}/" + e["pseudo_label_filename"].lstrip("./")
        if member not in members:
            print("missing", member); continue
        lab_full, sp = load_mask_from_zip(zf, member)
        lab, organ = organ_centre_crop(lab_full.astype(np.int16), rng)
        z = lab.shape[2] // 2
        src = e["pseudo_label_filename"].split("/")[1]

        axes[r, 0].imshow(lut[np.clip(lab[:, :, z], 0, 255)])
        axes[r, 0].set_title(f"{src}\nmask (organ {organ})", fontsize=8)
        axes[r, 0].axis("off")
        for k in range(a.seeds):
            img = paint_gmm(lab, np.random.default_rng(1000 * r + k))
            axes[r, k + 1].imshow(img[:, :, z], cmap="gray")
            axes[r, k + 1].set_title(f"GMM seed {k}", fontsize=8)
            axes[r, k + 1].axis("off")
        print(f"[{r}] {src} organ={organ} dim={lab_full.shape} spacing={sp.round(2)}")

    fig.suptitle("MAISI mask bank + fully-random per-label GMM (SynthSeg) — 128^3 organ crops",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()

"""
Inspect 96³-crop generation quality from ONE sampled mask.

Takes a single whole-body mask from the bank (a generated MAISI .npz label, already
1.5mm / MAISI-132 vocab), crops several 96³ windows centred on different organs, and
runs the full mask→CT pipeline (30-step rflow loop + VAE decode) on each. For one crop
it also varies the noise seed to show appearance diversity. Renders a montage:
rows = organ-centred crops, cols = noise seeds, with the multi-class mask contour overlaid
so image/anatomy correspondence is visible. FOV = 96×1.5 = 144mm (below MAISI's 256mm
recommendation on purpose — we WANT to see how crop-scale conditioning degrades).

  MONAI_DATA_DIRECTORY=./temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/inspect_crop_quality.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --crop 96 --seeds 3 --organs 4
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import MetaTensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # patch_icl root
from gen_maisi_fast import build_args  # noqa: E402
from data.maisi_classes import MAISI_IDX_TO_CLASS  # noqa: E402


def crop_window(vol, center, c):
    """Clamped c³ crop of a (H,W,D) array around center (voxel idx)."""
    sl = []
    for ci, s in zip(center, vol.shape):
        lo = int(np.clip(ci - c // 2, 0, max(0, s - c)))
        sl.append(slice(lo, lo + c))
    out = vol[sl[0], sl[1], sl[2]]
    if out.shape != (c, c, c):  # edge pad
        out = np.pad(out, [(0, c - out.shape[i]) for i in range(3)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--bank", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/data/synth_task_gen/maisi"))
    ap.add_argument("--mask_idx", type=int, default=0, help="which npz in the bank")
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--organs", type=int, default=4, help="# organ-centred crops (rows)")
    ap.add_argument("--seeds", type=int, default=3, help="# noise seeds per crop (cols)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/results/synth_task_gen/crop_quality.png"))
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    from scripts.infer_image_from_mask import ldm_conditional_sample_one_image
    from scripts.utils_infer import load_image_models

    dev = torch.device("cuda")
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 1
    ae, dm, cn, scale_factor, sched = load_image_models(args, dev)
    print("[load] image models ready", flush=True)

    # ---- one sampled mask from the bank ----
    files = sorted(glob.glob(str(a.bank / "*.npz")))
    src = files[a.mask_idx]
    d = np.load(src)
    lbl = d["label"].astype(np.int64)  # (H,W,D) MAISI ids
    print(f"[mask] {Path(src).name}  shape={lbl.shape}  "
          f"has_body200={(lbl==200).any()}  n_classes={len(np.unique(lbl))}", flush=True)

    # ---- pick organ centres: largest specific organs (exclude bg/body) ----
    ids, counts = np.unique(lbl, return_counts=True)
    cand = [(c, i) for i, c in zip(ids, counts) if i not in (0, 200)]
    cand.sort(reverse=True)
    chosen = []
    for _, i in cand:
        if i in MAISI_IDX_TO_CLASS:
            chosen.append(i)
        if len(chosen) >= a.organs:
            break

    c = a.crop
    latent_shape = [args.latent_channels, c // 4, c // 4, c // 4]
    spacing_tensor = torch.FloatTensor([1.5, 1.5, 1.5]).unsqueeze(0).half().to(dev) * 1e2
    modality_tensor = torch.ones((1,), dtype=torch.long, device=dev)

    # cols: [raw seed0 | overlay seed0 | overlay seed1 | ... ] so we can judge both
    # image realism (raw) and image/mask alignment (overlays).
    n_rows, n_cols = len(chosen), 1 + a.seeds
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.3 * n_rows), squeeze=False)

    for r, organ_id in enumerate(chosen):
        cen = np.argwhere(lbl == organ_id).mean(0).astype(int)
        mcrop = crop_window(lbl, cen, c)  # (c,c,c) multi-class
        name = MAISI_IDX_TO_CLASS.get(organ_id, str(organ_id))
        mask_t = MetaTensor(torch.from_numpy(mcrop)[None, None].to(dev))
        zc = c // 2  # display slice: axial through crop centre
        mm = mcrop[:, :, zc]
        for k in range(a.seeds):
            torch.manual_seed(1000 + k)
            with torch.no_grad():
                img, _ = ldm_conditional_sample_one_image(
                    autoencoder=ae, diffusion_unet=dm, controlnet=cn,
                    noise_scheduler=sched, scale_factor=scale_factor, device=dev,
                    combine_label_or=mask_t, spacing_tensor=spacing_tensor,
                    latent_shape=latent_shape, output_size=[c, c, c], noise_factor=1.0,
                    modality_tensor=modality_tensor, num_inference_steps=a.steps,
                    autoencoder_sliding_window_infer_size=[c, c, c],
                    autoencoder_sliding_window_infer_overlap=0.5, cfg_guidance_scale=0.0,
                )
            im = img.squeeze().float().cpu().numpy()[:, :, zc]      # (c,c) HU
            if k == 0:  # raw (no overlay) reference column
                ax0 = axes[r][0]
                ax0.imshow(np.clip(im, -200, 250), cmap="gray", vmin=-200, vmax=250)
                ax0.set_xticks([]); ax0.set_yticks([])
                ax0.set_ylabel(f"{name}\n(id {organ_id})", fontsize=10)
                ax0.set_title("raw (seed 1000)", fontsize=9)
            ax = axes[r][1 + k]
            ax.imshow(np.clip(im, -200, 250), cmap="gray", vmin=-200, vmax=250)
            ax.imshow(np.ma.masked_where(mm == 0, mm), cmap="tab20", alpha=0.30, vmin=0, vmax=20)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"+mask, seed {1000+k}", fontsize=9)
        print(f"  row {r}: {name} done", flush=True)

    fig.suptitle(f"96³ crop quality — 1 bank mask ({Path(src).name}), "
                 f"{a.steps}-step, FOV {c*1.5:.0f}mm", fontsize=11)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=110)
    print(f"\n[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()

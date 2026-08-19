"""
Closer to the original pipeline: generate ONE full-body latent (diffusion sees the
whole anatomy → realistic), then crop the LATENT at 128³-image windows (=32³ latent)
at random in-body locations and decode each. Shows img + mask overlay. No SDEdit.

Why this should beat per-crop conditioning: the 30-step loop runs on the full 96³-ish
latent with global context; we only crop at DECODE time. Decode is convolutional/local,
so an interior latent-window decodes ≈ the same region of the full decode (edges may
show mild context loss — that's what we're inspecting).

  MONAI_DATA_DIRECTORY=./temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/inspect_fullbody_crops.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --n_crops 6 --crop 128
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gen_maisi_fast import build_args  # noqa: E402
from data.maisi_classes import MAISI_IDX_TO_CLASS  # noqa: E402


def gen_fullbody_latent(ae, dm, cn, sched, scale_factor, dev, mask, spacing, steps):
    """Run the mask→CT diffusion loop on the WHOLE body; return the denoised latent
    (1,4,H/4,W/4,D/4) BEFORE decode. Mirrors utils_infer.run_controlnet_conditioned_image_dm."""
    from scripts.utils import binarize_labels
    out = list(mask.shape[2:])
    latent_shape = [4, out[0] // 4, out[1] // 4, out[2] // 4]
    cond = binarize_labels(mask.long()).half()
    spacing_tensor = torch.FloatTensor(spacing).unsqueeze(0).half().to(dev) * 1e2
    modality_tensor = torch.ones((1,), dtype=torch.long, device=dev)
    include_modality = dm.num_class_embeds is not None
    with torch.no_grad(), torch.amp.autocast("cuda"):
        latents = torch.randn([1] + latent_shape, device=dev).half()
        sched.set_timesteps(num_inference_steps=steps,
                            input_img_size_numel=torch.prod(torch.tensor(latents.shape[-3:])))
        all_t = sched.timesteps
        all_next = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))
        for t, nt in zip(all_t, all_next):
            tt = torch.Tensor((t,)).to(dev)
            cn_in = {"x": latents, "timesteps": tt, "controlnet_cond": cond}
            if include_modality:
                cn_in["class_labels"] = modality_tensor
            down, mid = cn(**cn_in)
            un_in = {"x": latents, "timesteps": tt, "spacing_tensor": spacing_tensor,
                     "down_block_additional_residuals": down, "mid_block_additional_residual": mid}
            if include_modality:
                un_in["class_labels"] = modality_tensor
            v = dm(**un_in)
            latents, _ = sched.step(v, t, latents, nt)
    del cond
    torch.cuda.empty_cache()
    return latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--bank", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/data/synth_task_gen/maisi"))
    ap.add_argument("--mask_idx", type=int, default=0)
    ap.add_argument("--crop", type=int, default=128, help="image-space crop (latent=crop//4)")
    ap.add_argument("--n_crops", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--min_body", type=float, default=0.5, help="min in-body fraction of a kept crop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/results/synth_task_gen/fullbody_crops.png"))
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    from scripts.utils_infer import ReconModel, load_image_models

    dev = torch.device("cuda")
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 4  # full-body latent → decode may tile large windows
    ae, dm, cn, scale_factor, sched = load_image_models(args, dev)
    recon = ReconModel(autoencoder=ae, scale_factor=scale_factor).to(dev).eval()
    print("[load] models ready", flush=True)

    files = sorted(glob.glob(str(a.bank / "*.npz")))
    src = files[a.mask_idx]
    lbl = np.load(src)["label"].astype(np.int64)  # (H,W,D)
    mask_full = torch.from_numpy(lbl)[None, None].to(dev)
    print(f"[mask] {Path(src).name} shape={lbl.shape} body200={(lbl==200).any()}", flush=True)

    import time
    t = time.time()
    z = gen_fullbody_latent(ae, dm, cn, sched, scale_factor, dev, mask_full,
                            [1.5, 1.5, 1.5], a.steps)
    print(f"[gen] full-body latent {tuple(z.shape)} in {time.time()-t:.1f}s", flush=True)

    c = a.crop
    lc = c // 4
    Lsp = z.shape[2:]  # latent spatial (H/4,W/4,D/4)
    rng = np.random.default_rng(a.seed)

    crops = []
    tries = 0
    while len(crops) < a.n_crops and tries < 400:
        tries += 1
        lo_l = [int(rng.integers(0, Lsp[i] - lc + 1)) for i in range(3)]
        lo_i = [x * 4 for x in lo_l]
        mcrop = lbl[lo_i[0]:lo_i[0]+c, lo_i[1]:lo_i[1]+c, lo_i[2]:lo_i[2]+c]
        if mcrop.shape != (c, c, c):
            continue
        if (mcrop > 0).mean() < a.min_body:  # want mostly in-body
            continue
        zc = z[:, :, lo_l[0]:lo_l[0]+lc, lo_l[1]:lo_l[1]+lc, lo_l[2]:lo_l[2]+lc]
        with torch.no_grad(), torch.amp.autocast("cuda"):
            dec = torch.clip(recon(zc), 0, 1)
        hu = (dec.squeeze().float().cpu().numpy()) * 2000.0 - 1000.0  # [0,1]→[-1000,1000] HU
        hu[mcrop == 0] = -1000.0  # background regularisation (crop_img_body_mask)
        # dominant specific organ for the title
        ids, cnts = np.unique(mcrop[(mcrop > 0) & (mcrop != 200)], return_counts=True)
        dom = MAISI_IDX_TO_CLASS.get(int(ids[cnts.argmax()]), "?") if len(ids) else "body only"
        crops.append((hu, mcrop, lo_i, dom))
    print(f"[crop] kept {len(crops)}/{a.n_crops} (tries={tries})", flush=True)

    # ---- render: rows=crops, cols=[axial raw, axial+mask, coronal+mask] ----
    nr = len(crops)
    fig, axes = plt.subplots(nr, 3, figsize=(3.3 * 3, 3.4 * nr), squeeze=False)
    for r, (hu, mcrop, lo_i, dom) in enumerate(crops):
        za = c // 2
        for col, (img2d, m2d, ttl) in enumerate([
            (hu[:, :, za], mcrop[:, :, za], "axial raw"),
            (hu[:, :, za], mcrop[:, :, za], "axial +mask"),
            (hu[:, za, :], mcrop[:, za, :], "coronal +mask"),
        ]):
            ax = axes[r][col]
            ax.imshow(np.clip(img2d, -200, 250), cmap="gray", vmin=-200, vmax=250)
            if col > 0:
                ax.imshow(np.ma.masked_where(m2d == 0, m2d), cmap="tab20", alpha=0.30, vmin=0, vmax=20)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ttl, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"@{lo_i}\n{dom}", fontsize=8)

    fig.suptitle(f"128³ crops from ONE full-body latent ({Path(src).name}), {a.steps}-step, "
                 f"decode latent-windows", fontsize=11)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(a.out, dpi=110)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()

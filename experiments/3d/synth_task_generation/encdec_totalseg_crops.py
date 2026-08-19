"""
Does REAL TotalSeg data survive the MAISI VAE round-trip? Prereq for using this latent
space for real in-context data. Builds the real dataloader (use_crop, 1.5mm, p_synth=0),
takes 128³ crops, encodes+decodes with the MAISI image VAE (deterministic mu, no diffusion),
and reports PSNR/MAE + an orig|recon|diff montage.

Intensity bridge: dataloader CT is z-scored (hu=z*505.8-167.3, clip[-1007,1573]); MAISI VAE
wants [0,1] via (hu+1000)/2000. We map z-scored→HU→[0,1]→VAE→[0,1]→HU and compare.

  MONAI_DATA_DIRECTORY=./temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/encdec_totalseg_crops.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --n 8
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling gen_maisi_fast
sys.path.insert(0, str(ROOT / "experiments" / "3d"))              # common
from gen_maisi_fast import build_args  # noqa: E402
from src.totalseg_dataset import CT_MEAN, CT_STD  # noqa: E402

A_MIN, A_MAX = -1000.0, 1000.0  # MAISI CT intensity range


def z_to_maisi01(z):
    hu = z * CT_STD + CT_MEAN
    return np.clip((hu - A_MIN) / (A_MAX - A_MIN), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--n", type=int, default=8, help="# crops to round-trip")
    ap.add_argument("--out", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/results/synth_task_gen/encdec_totalseg.png"))
    ap.add_argument("-h", "--help", action="store_true")
    a, hydra_overrides = ap.parse_known_args()
    if a.help:
        ap.print_help(); return

    # ---- build the REAL dataloader (use_crop, 1.5mm, p_synth=0, 128³, no aug) ----
    from hydra import compose, initialize_config_dir
    from torch.utils.data import DataLoader
    from common import build_dataset  # experiments/3d/common.py
    from src.totalseg_dataloader_incontext import incontext_collate_fn

    overrides = [
        "data.source=totalseg", "data.use_crop=true", "data.crop_spacing_mm=1.5",
        "data.p_synth=0", "data.image_size=[128,128,128]", "data.context_size=2",
        "augmentations.enabled=false",
    ] + hydra_overrides
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    ds = build_dataset(cfg, "train")
    loader = DataLoader(ds, batch_size=max(2, a.n // 3 + 1), shuffle=True,
                        num_workers=0, collate_fn=incontext_collate_fn)
    batch = next(iter(loader))
    # gather target + context crops into one stack of z-scored (1,128,128,128) volumes
    crops = [batch["image"]]  # (B,1,H,W,D)
    if "context_in" in batch:  # (B,K,1,H,W,D)
        ci = batch["context_in"]
        crops += [ci[:, k] for k in range(ci.shape[1])]
    zvol = torch.cat(crops, 0)[: a.n].float().numpy()  # (n,1,128,128,128)
    print(f"[data] {zvol.shape[0]} crops, z-scored range [{zvol.min():.2f},{zvol.max():.2f}]", flush=True)

    # ---- load MAISI image VAE ----
    # patch_icl and the MAISI repo BOTH ship a top-level `scripts` package; the dataset
    # build above cached patch_icl's. Purge it so `scripts.utils_infer` resolves to the repo.
    for m in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
        del sys.modules[m]
    sys.path.insert(0, str(a.repo))
    from scripts.utils_infer import load_image_models
    dev = torch.device("cuda")
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 1
    ae, dm, cn, scale_factor, sched = load_image_models(args, dev); del dm, cn
    torch.cuda.empty_cache()
    print("[load] MAISI VAE ready", flush=True)

    # ---- round-trip: [0,1] -> encode.mu -> decode -> [0,1] ----
    x01 = z_to_maisi01(zvol)                                  # (n,1,128,128,128)
    x = torch.from_numpy(x01).to(dev)
    recon = np.empty_like(x01)
    lat_shapes = None
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(x.shape[0]):
            z_mu, _ = ae.encode(x[i:i+1])
            if lat_shapes is None:
                lat_shapes = tuple(z_mu.shape)
            xh = ae.decode_stage_2_outputs(z_mu)
            recon[i] = torch.clip(xh, 0, 1).float().cpu().numpy()
    print(f"[vae] latent shape per 128³ crop: {lat_shapes}", flush=True)

    # ---- metrics (in [0,1] and HU) ----
    err01 = recon - x01
    mae01 = np.abs(err01).mean()
    mae_hu = mae01 * (A_MAX - A_MIN)
    psnr = 10 * np.log10(1.0 / max((err01 ** 2).mean(), 1e-12))
    # soft-tissue-only MAE (0.35..0.65 of [0,1] ~ -300..300 HU) to gauge organ fidelity
    st = (x01 > 0.35) & (x01 < 0.65)
    mae_hu_soft = (np.abs(err01[st]).mean() * 2000) if st.any() else float("nan")
    print(f"[metrics] PSNR={psnr:.2f} dB | MAE={mae01:.4f} ([0,1]) = {mae_hu:.1f} HU | "
          f"soft-tissue MAE={mae_hu_soft:.1f} HU", flush=True)

    # ---- montage: rows=crops, cols=[orig, recon, |diff|] (mid-axial, soft-tissue window) ----
    def hu_win(v01):  # [0,1] -> HU -> clip window -200..250
        return np.clip(v01 * 2000 - 1000, -200, 250)
    nr = zvol.shape[0]
    fig, ax = plt.subplots(nr, 3, figsize=(3 * 3.0, nr * 3.05), squeeze=False)
    for r in range(nr):
        s = x01.shape[2] // 2
        o, rc = x01[r, 0, :, :, s], recon[r, 0, :, :, s]
        for c, (im, ttl, cmap, vmin, vmax) in enumerate([
            (hu_win(o), "orig (real CT)", "gray", -200, 250),
            (hu_win(rc), "VAE recon", "gray", -200, 250),
            (np.abs(rc - o) * 2000, "|diff| HU", "magma", 0, 200),
        ]):
            A = ax[r][c]
            A.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
            A.set_xticks([]); A.set_yticks([])
            if r == 0:
                A.set_title(ttl, fontsize=10)
    fig.suptitle(f"MAISI VAE round-trip on REAL TotalSeg 128³@1.5mm crops | "
                 f"PSNR {psnr:.1f}dB, MAE {mae_hu:.0f}HU (soft {mae_hu_soft:.0f})", fontsize=11)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=110)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()

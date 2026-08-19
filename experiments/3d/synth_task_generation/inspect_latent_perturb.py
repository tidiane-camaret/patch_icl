"""
Do latent perturbations keep organs IN PLACE? Take real TotalSeg 128³ crops (+GT label)
at 1..4mm, encode to (4,32,32,32), add Gaussian noise at increasing σ (×latent std),
decode, and overlay the fixed GT-label CONTOUR. If the organ boundary in the decoded
image keeps following the contour, the latent encodes anatomy spatially-locally
(perturb → appearance jitter, geometry preserved) — the property we need to use latent
perturbation as an appearance-diversity augmentation.

  MONAI_DATA_DIRECTORY=/home/dpxuser/repos/NV-Generate-CTMR/temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/inspect_latent_perturb.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR
(run from /home/dpxuser/dev/patch_icl)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))
from gen_maisi_fast import build_args  # noqa: E402
from src.totalseg_dataset import CT_MEAN, CT_STD  # noqa: E402

A_MIN, A_MAX = -1000.0, 1000.0
SPACINGS = [1.0, 2.0, 3.0, 4.0]
SIGMAS = [0.5, 1.0, 2.0]  # noise std as multiple of the crop's latent std


def z_to_maisi01(z):
    hu = z * CT_STD + CT_MEAN
    return np.clip((hu - A_MIN) / (A_MAX - A_MIN), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--out", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/results/synth_task_gen/latent_perturb.png"))
    ap.add_argument("-h", "--help", action="store_true")
    a, hydra_overrides = ap.parse_known_args()
    if a.help:
        ap.print_help(); return

    # ---- one labeled crop per spacing ----
    from hydra import compose, initialize_config_dir
    from torch.utils.data import DataLoader
    from common import build_dataset
    from src.totalseg_dataloader_incontext import incontext_collate_fn

    overrides = [
        "data.source=totalseg", "data.use_crop=true", "data.crop_spacing_mm=1.0",
        "data.p_synth=0", "data.image_size=[128,128,128]", "data.context_size=1",
        "augmentations.enabled=false",
    ] + hydra_overrides
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    ds = build_dataset(cfg, "train")

    crops = {}
    for s in SPACINGS:
        ds.crop_spacing_mm = float(s)
        g = torch.Generator().manual_seed(7 + int(s))
        loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                            collate_fn=incontext_collate_fn, generator=g)
        b = next(iter(loader))
        img = b["image"][0, 0].float().numpy()          # (128,128,128) z-scored
        lab = b["label"][0].float().numpy()             # (1,128,128,128) or (128,..)?
        lab = lab[0] if lab.ndim == 4 else lab
        name = b.get("label_name", ["?"])[0] if isinstance(b.get("label_name"), list) else "?"
        crops[s] = (img, (lab > 0.5).astype(np.uint8), name)
    print("[data] crops:", {s: crops[s][2] for s in SPACINGS}, flush=True)

    # ---- load MAISI VAE ----
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

    def encode(x01):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            z_mu, _ = ae.encode(torch.from_numpy(x01)[None, None].to(dev))
        return z_mu

    def decode(z):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            return torch.clip(ae.decode_stage_2_outputs(z), 0, 1).float().cpu().numpy()[0, 0]

    # ---- build figure: rows=spacings, cols=[orig, recon, +noise σ...] ----
    ncol = 2 + len(SIGMAS)
    fig, ax = plt.subplots(len(SPACINGS), ncol, figsize=(ncol * 2.9, len(SPACINGS) * 3.0),
                           squeeze=False)
    col_titles = ["orig +GT", "recon"] + [f"+noise {s:g}σ" for s in SIGMAS]
    for r, sp in enumerate(SPACINGS):
        img_z, lab, name = crops[sp]
        x01 = z_to_maisi01(img_z)
        # display slice = max GT coverage (organ visible); fallback centre
        fg = lab.sum(axis=(0, 1))
        sl = int(fg.argmax()) if fg.max() > 0 else lab.shape[2] // 2
        z = encode(x01)
        zstd = z.float().std().item()
        panels = [x01, decode(z)] + [decode(z + sig * zstd * torch.randn_like(z)) for sig in SIGMAS]
        for c, (pan, ttl) in enumerate(zip(panels, col_titles)):
            A = ax[r][c]
            A.imshow(np.clip(pan[:, :, sl] * 2000 - 1000, -200, 250), cmap="gray", vmin=-200, vmax=250)
            if lab[:, :, sl].any():
                A.contour(lab[:, :, sl], levels=[0.5], colors=["cyan"], linewidths=0.9)
            A.set_xticks([]); A.set_yticks([])
            if r == 0:
                A.set_title(ttl, fontsize=10)
            if c == 0:
                A.set_ylabel(f"{sp:.0f}mm\n{name}", fontsize=8)
        print(f"  row {r} ({sp}mm, {name}, latent std={zstd:.3f}) done", flush=True)

    fig.suptitle("Latent perturbation vs GT label (cyan) — do organs stay in place?", fontsize=12)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=110)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()

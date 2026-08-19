"""
VAE resolution capability: round-trip REAL TotalSeg 128³ crops at crop_spacing_mm
∈ {1,2,3,4,5}. Latent is (4,32,32,32) for ALL of them (same voxel grid), so this
isolates how reconstruction fidelity depends on CONTENT FREQUENCY — fine detail at
1mm (128mm FOV) vs whole-body context at 5mm (640mm FOV). MAISI CT was trained on
in-plane spacing 0.5-3mm, so 4-5mm is out-of-distribution.

  MONAI_DATA_DIRECTORY=/home/dpxuser/repos/NV-Generate-CTMR/temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/bench_vae_resolution.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --n 6
(run from /home/dpxuser/dev/patch_icl so hydra's ${PWD}/configs resolves)
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
SPACINGS = [1.0, 2.0, 3.0, 4.0, 5.0]


def z_to_maisi01(z):
    hu = z * CT_STD + CT_MEAN
    return np.clip((hu - A_MIN) / (A_MAX - A_MIN), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--n", type=int, default=6, help="# crops per spacing")
    ap.add_argument("--out", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/results/synth_task_gen/vae_resolution.png"))
    ap.add_argument("-h", "--help", action="store_true")
    a, hydra_overrides = ap.parse_known_args()
    if a.help:
        ap.print_help(); return

    # ---- real dataloader (built once; crop_spacing_mm mutated per sweep point) ----
    from hydra import compose, initialize_config_dir
    from torch.utils.data import DataLoader
    from common import build_dataset
    from src.totalseg_dataloader_incontext import incontext_collate_fn

    overrides = [
        "data.source=totalseg", "data.use_crop=true", "data.crop_spacing_mm=1.0",
        "data.p_synth=0", "data.image_size=[128,128,128]", "data.context_size=2",
        "augmentations.enabled=false",
    ] + hydra_overrides
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    ds = build_dataset(cfg, "train")

    def sample_crops(spacing, n, seed):
        ds.crop_spacing_mm = float(spacing)
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(ds, batch_size=max(2, n // 3 + 1), shuffle=True,
                            num_workers=0, collate_fn=incontext_collate_fn, generator=g)
        b = next(iter(loader))
        crops = [b["image"]]
        if "context_in" in b:
            crops += [b["context_in"][:, k] for k in range(b["context_in"].shape[1])]
        return torch.cat(crops, 0)[:n].float().numpy()  # (n,1,128,128,128) z-scored

    zvols = {s: sample_crops(s, a.n, seed=100 + int(s)) for s in SPACINGS}
    print("[data] sampled crops per spacing:", {s: v.shape[0] for s, v in zvols.items()}, flush=True)

    # ---- load MAISI VAE (purge patch_icl's shadowing `scripts` pkg first) ----
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

    def roundtrip(x01):
        x = torch.from_numpy(x01).to(dev)
        out = np.empty_like(x01)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for i in range(x.shape[0]):
                z_mu, _ = ae.encode(x[i:i+1])
                out[i] = torch.clip(ae.decode_stage_2_outputs(z_mu), 0, 1).float().cpu().numpy()
        return out

    # ---- sweep ----
    stats, examples = {}, {}
    for s in SPACINGS:
        x01 = z_to_maisi01(zvols[s])
        rec = roundtrip(x01)
        err = rec - x01
        mae_hu = np.abs(err).mean() * (A_MAX - A_MIN)
        psnr = 10 * np.log10(1.0 / max((err ** 2).mean(), 1e-12))
        st = (x01 > 0.35) & (x01 < 0.65)
        mae_soft = (np.abs(err[st]).mean() * 2000) if st.any() else float("nan")
        stats[s] = (psnr, mae_hu, mae_soft)
        examples[s] = (x01[0, 0], rec[0, 0])
        print(f"[s={s}mm | FOV {128*s:.0f}mm] PSNR={psnr:5.2f}dB  MAE={mae_hu:5.1f}HU  "
              f"soft={mae_soft:5.1f}HU", flush=True)

    # ---- figure: rows=spacings [orig|recon||diff], + PSNR/MAE curve ----
    def hu_win(v01):
        return np.clip(v01 * 2000 - 1000, -200, 250)
    nr = len(SPACINGS)
    fig, ax = plt.subplots(nr, 4, figsize=(4 * 3.0, nr * 3.0), squeeze=False,
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    for r, s in enumerate(SPACINGS):
        o, rc = examples[s]
        sl = o.shape[2] // 2
        psnr, mae_hu, mae_soft = stats[s]
        for c, (im, ttl, cmap, vmn, vmx) in enumerate([
            (hu_win(o[:, :, sl]), "orig", "gray", -200, 250),
            (hu_win(rc[:, :, sl]), "recon", "gray", -200, 250),
            (np.abs(rc - o)[:, :, sl] * 2000, "|diff|HU", "magma", 0, 200),
        ]):
            A = ax[r][c]
            A.imshow(im, cmap=cmap, vmin=vmn, vmax=vmx); A.set_xticks([]); A.set_yticks([])
            if r == 0:
                A.set_title(ttl, fontsize=10)
            if c == 0:
                A.set_ylabel(f"{s:.0f}mm/vox\nFOV {128*s:.0f}mm\nPSNR {psnr:.1f}\nMAE {mae_hu:.0f}HU",
                             fontsize=8)
    # last column: metric curves (span rows visually via a single axis in row 0..)
    gs = ax[0][3].get_gridspec()
    for r in range(nr):
        ax[r][3].remove()
    axc = fig.add_subplot(gs[:, 3])
    ss = np.array(SPACINGS)
    axc.plot(ss, [stats[s][0] for s in SPACINGS], "o-", label="PSNR (dB)", color="tab:blue")
    axc.set_xlabel("crop_spacing_mm"); axc.set_ylabel("PSNR (dB)", color="tab:blue")
    axc.tick_params(axis="y", labelcolor="tab:blue")
    axc.axvspan(0.5, 3.0, color="green", alpha=0.08, label="MAISI train range (xy)")
    ax2 = axc.twinx()
    ax2.plot(ss, [stats[s][1] for s in SPACINGS], "s--", label="MAE (HU)", color="tab:red")
    ax2.plot(ss, [stats[s][2] for s in SPACINGS], "^:", label="soft MAE (HU)", color="tab:orange")
    ax2.set_ylabel("MAE (HU)", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
    axc.set_title("fidelity vs spacing", fontsize=10)
    l1, la1 = axc.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    axc.legend(l1 + l2, la1 + la2, fontsize=7, loc="upper center")

    fig.suptitle("MAISI VAE round-trip on real TotalSeg 128³ crops vs resolution", fontsize=12)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(a.out, dpi=110)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()

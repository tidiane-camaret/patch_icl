"""
Is the MAISI VAE latent a USEFUL representation for in-context segmentation, or just a
good renderer input? Compares frozen feature spaces on the ACTUAL task via prototype
matching + fg-retrieval, on the same TotalSeg in-context tasks (use_crop, 1.5mm):

  - vae32   : MAISI VAE latent, encode(img).mu            (4, 32,32,32)   16x compress
  - vae16   : same, avg-pooled to 16³ (Primus grid)       (4, 16,16,16)
  - primus  : frozen CoLiPri Primus ViT features           (864,16,16,16) over-complete
  - rawHU   : intensity nearest-centroid baseline          (1, 32,32,32)

Per task: build fg/bg prototypes from the K CONTEXT crops (their features+masks), classify
the TARGET voxels by nearest prototype (cosine), upsample to 128³, Dice vs GT. Also
fg-retrieval@1 (each target fg voxel's nearest context voxel is fg?). Reconstruction
fidelity (PSNR) says nothing about these numbers — that's the point.

  MONAI_DATA_DIRECTORY=/home/dpxuser/repos/NV-Generate-CTMR/temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/compare_latent_vs_primus.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --tasks 40 --context 4
(run from /home/dpxuser/dev/patch_icl)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))
from gen_maisi_fast import build_args  # noqa: E402
from src.totalseg_dataset import CT_MEAN, CT_STD  # noqa: E402

A_MIN, A_MAX = -1000.0, 1000.0
COLIPRI = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/checkpoints/colipri/primus_colipri.json")


def z_to_maisi01_t(z):  # torch, z-scored -> MAISI [0,1]
    return torch.clamp((z * CT_STD + CT_MEAN - A_MIN) / (A_MAX - A_MIN), 0, 1)


def mask_to_grid(m, g):
    """binary volume (any leading dims, trailing 128³) -> occupancy>0.5 at (g,g,g)."""
    m = m.reshape(1, 1, 128, 128, 128).float()
    return (F.avg_pool3d(m, 128 // g) > 0.5).float()[0, 0]


def proto_dice(sup_f, sup_m, qf, gt_full, metric="cos"):
    """sup_f (K,C,g,g,g), sup_m (K,g,g,g), qf (C,g,g,g), gt_full (128³). Nearest-prototype."""
    C, g = sup_f.shape[1], sup_f.shape[2]
    S = sup_f.permute(0, 2, 3, 4, 1).reshape(-1, C)
    m = sup_m.reshape(-1)
    if m.sum() < 1 or (m < 0.5).sum() < 1:
        return None
    if metric == "cos":
        S = F.normalize(S, dim=1)
    fg = S[m > 0.5].mean(0, keepdim=True)
    bg = S[m < 0.5].mean(0, keepdim=True)
    Q = qf.permute(1, 2, 3, 0).reshape(-1, C)
    if metric == "cos":
        Qn = F.normalize(Q, dim=1)
        s_fg, s_bg = Qn @ F.normalize(fg, dim=1).T, Qn @ F.normalize(bg, dim=1).T
        pred = (s_fg > s_bg).float()
    else:  # L2 nearest centroid
        pred = (((Q - fg) ** 2).sum(1) < ((Q - bg) ** 2).sum(1)).float()
    pred = pred.reshape(1, 1, g, g, g)
    pred_full = F.interpolate(pred, size=(128, 128, 128), mode="nearest")[0, 0]
    inter = (pred_full * gt_full).sum()
    return float((2 * inter / (pred_full.sum() + gt_full.sum() + 1e-6)).item())


def fg_retrieval_at1(sup_f, sup_m, qf, qm, cap=1500, metric="cos"):
    """each target fg voxel's nearest context voxel — fraction that are fg. Subsampled."""
    C = sup_f.shape[1]
    S = sup_f.permute(0, 2, 3, 4, 1).reshape(-1, C)
    sm = sup_m.reshape(-1)
    Q = qf.permute(1, 2, 3, 0).reshape(-1, C)
    qfg = qm.reshape(-1) > 0.5
    if qfg.sum() < 5 or sm.sum() < 1:
        return None
    qi = torch.where(qfg)[0]
    if len(qi) > cap:
        qi = qi[torch.randperm(len(qi), device=qi.device)[:cap]]
    si = torch.randperm(S.shape[0], device=S.device)[: min(S.shape[0], 8 * cap)]
    Qs, Ss, sms = Q[qi], S[si], sm[si]
    if metric == "cos":
        Qs, Ss = F.normalize(Qs, dim=1), F.normalize(Ss, dim=1)
        nn = (Qs @ Ss.T).argmax(1)
    else:
        nn = torch.cdist(Qs, Ss).argmin(1)
    return float((sms[nn] > 0.5).float().mean().item())


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("-h", "--help", action="store_true")
    a, hydra_overrides = ap.parse_known_args()
    if a.help:
        ap.print_help(); return

    dev = torch.device("cuda")

    # ---- dataset ----
    from hydra import compose, initialize_config_dir
    from torch.utils.data import DataLoader
    from common import build_dataset
    from src.totalseg_dataloader_incontext import incontext_collate_fn
    from src.models.primus_encoder import PrimusEncoder

    overrides = [
        "data.source=totalseg", "data.use_crop=true", f"data.crop_spacing_mm={a.spacing}",
        "data.p_synth=0", "data.image_size=[128,128,128]", f"data.context_size={a.context}",
        "augmentations.enabled=false",
    ] + hydra_overrides
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    ds = build_dataset(cfg, "train")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=incontext_collate_fn,
                        generator=torch.Generator().manual_seed(0))

    # ---- Primus encoder (frozen) ----
    primus = PrimusEncoder(str(COLIPRI), resolution=16, frozen=True, device="cuda",
                           native_grid=True, precision="bf16").eval()
    print("[load] Primus ready", flush=True)

    # ---- MAISI VAE (purge shadowing scripts pkg) ----
    for mkey in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
        del sys.modules[mkey]
    sys.path.insert(0, str(a.repo))
    from scripts.utils_infer import load_image_models
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 1
    ae, dm, cn, sf, sched = load_image_models(args, dev); del dm, cn
    torch.cuda.empty_cache()
    print("[load] MAISI VAE ready", flush=True)

    def vae_feat(img_z):  # (1,1,128,128,128) z-scored -> (4,32,32,32)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            zmu, _ = ae.encode(z_to_maisi01_t(img_z).to(dev))
        return zmu[0].float()

    def primus_feat(img_z):  # -> (864,16,16,16)
        with torch.no_grad():
            return primus(img_z.to(dev))[0].float()

    spaces = ["vae32", "vae16", "primus", "rawHU"]
    dice = {s: [] for s in spaces}
    retr = {s: [] for s in spaces}
    fg_frac = []

    it = iter(loader)
    done = 0
    while done < a.tasks:
        try:
            b = next(it)
        except StopIteration:
            break
        tgt = b["image"]                     # (1,1,128,128,128)
        gt = (b["label"][0] > 0.5).float()   # (128,128,128) or (1,128,..)
        gt = gt[0] if gt.ndim == 4 else gt
        gt_full = gt.to(dev)
        ci, co = b["context_in"], b["context_out"]  # (1,K,1,128,128,128)
        K = ci.shape[1]

        # features per space
        tgt_v = vae_feat(tgt); ctx_v = [vae_feat(ci[:, k]) for k in range(K)]
        tgt_p = primus_feat(tgt); ctx_p = [primus_feat(ci[:, k]) for k in range(K)]
        tgt_r = F.avg_pool3d(z_to_maisi01_t(tgt).to(dev), 4)[0]              # (1,32,32,32)
        ctx_r = [F.avg_pool3d(z_to_maisi01_t(ci[:, k]).to(dev), 4)[0] for k in range(K)]
        cm32 = torch.stack([mask_to_grid(co[0, k], 32) for k in range(K)]).to(dev)  # (K,32,32,32)
        cm16 = torch.stack([mask_to_grid(co[0, k], 16) for k in range(K)]).to(dev)
        qm32 = mask_to_grid(gt, 32).to(dev)
        qm16 = mask_to_grid(gt, 16).to(dev)

        feats = {
            "vae32": (torch.stack(ctx_v), cm32, tgt_v, qm32, "cos"),
            "vae16": (F.avg_pool3d(torch.stack(ctx_v), 2), cm16,
                      F.avg_pool3d(tgt_v[None], 2)[0], qm16, "cos"),
            "primus": (torch.stack(ctx_p), cm16, tgt_p, qm16, "cos"),
            "rawHU": (torch.stack(ctx_r), cm32, tgt_r, qm32, "l2"),
        }
        for s, (sf_, sm_, qf_, qm_, metric) in feats.items():
            d = proto_dice(sf_, sm_, qf_, gt_full, metric)
            r = fg_retrieval_at1(sf_, sm_, qf_, qm_, metric=metric)
            if d is not None:
                dice[s].append(d)
            if r is not None:
                retr[s].append(r)
        fg_frac.append(float(qm32.mean().item()))
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{a.tasks} tasks "
                  + " | ".join(f"{s} D={np.mean(dice[s]):.3f}" for s in spaces), flush=True)

    print(f"\n=== in-context prototype matching over {done} tasks (1.5mm, K={a.context}) ===", flush=True)
    print(f"{'space':>8} {'proto-Dice':>12} {'fg-retr@1':>11} {'n':>5}", flush=True)
    for s in spaces:
        print(f"{s:>8} {np.mean(dice[s]):>12.3f} {np.mean(retr[s]):>11.3f} {len(dice[s]):>5}", flush=True)

    # thin vs thick split (by target fg fraction, median)
    med = float(np.median(fg_frac))
    fg_arr = np.array(fg_frac)
    print(f"\n--- Dice by target size (median fg_frac={med:.3f}) ---", flush=True)
    print(f"{'space':>8} {'thin(<med)':>11} {'thick(>=med)':>13}", flush=True)
    for s in spaces:
        d = np.array(dice[s]); n = min(len(d), len(fg_arr)); d, fa = d[:n], fg_arr[:n]
        thin = d[fa < med].mean() if (fa < med).any() else float("nan")
        thick = d[fa >= med].mean() if (fa >= med).any() else float("nan")
        print(f"{s:>8} {thin:>11.3f} {thick:>13.3f}", flush=True)


if __name__ == "__main__":
    main()

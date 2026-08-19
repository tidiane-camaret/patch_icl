"""
Latent-space ops on a MAISI full-body latent: is the VAE decode COHERENT, and is it
EQUIVARIANT (op-in-latent ≈ op-in-image)? Convs are translation-equivariant (crop works);
rotation/pooling are not guaranteed. We test on one in-body 32³ latent crop (→128³ image).

Fig 1 (coherence): decode(op(z)) for identity / rot90 / rot45 / flip / avg-pool / zoom.
Fig 2 (equivariance): for rot90 and rot45, decode(op(z)) vs op(decode(z)) vs |diff|,
  with HU MAE — small MAE ⇒ the latent op is a valid stand-in for the image-space op.

  MONAI_DATA_DIRECTORY=./temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/inspect_latent_ops.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gen_maisi_fast import build_args  # noqa: E402
from inspect_fullbody_crops import gen_fullbody_latent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--bank", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/data/synth_task_gen/maisi"))
    ap.add_argument("--mask_idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--outdir", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/results/synth_task_gen"))
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    from scripts.utils_infer import ReconModel, load_image_models

    dev = torch.device("cuda")
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 1
    ae, dm, cn, scale_factor, sched = load_image_models(args, dev)
    recon = ReconModel(autoencoder=ae, scale_factor=scale_factor).to(dev).eval()
    print("[load] models ready", flush=True)

    lbl = np.load(sorted(glob.glob(str(a.bank / "*.npz")))[a.mask_idx])["label"].astype(np.int64)
    mask_full = torch.from_numpy(lbl)[None, None].to(dev)
    z = gen_fullbody_latent(ae, dm, cn, sched, scale_factor, dev, mask_full, [1.5, 1.5, 1.5], a.steps)
    print(f"[gen] full-body latent {tuple(z.shape)}", flush=True)

    # pick one in-body 32³ latent crop (→128³ image)
    lc = 32
    Lsp = z.shape[2:]
    rng = np.random.default_rng(1)
    zc = None
    for _ in range(400):
        lo = [int(rng.integers(0, Lsp[i] - lc + 1)) for i in range(3)]
        ii = [x * 4 for x in lo]
        mc = lbl[ii[0]:ii[0]+128, ii[1]:ii[1]+128, ii[2]:ii[2]+128]
        if mc.shape == (128, 128, 128) and (mc > 0).mean() > 0.6:
            zc = z[:, :, lo[0]:lo[0]+lc, lo[1]:lo[1]+lc, lo[2]:lo[2]+lc].clone()
            break
    print(f"[crop] latent crop {tuple(zc.shape)} @ img {ii}", flush=True)

    def decode(zt):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            dec = torch.clip(recon(zt), 0, 1)
        return dec.squeeze().float().cpu().numpy() * 2000.0 - 1000.0  # HU

    def rot_latent(zt, ang):  # in-plane (axial H,W) rotation via scipy, per-channel
        arr = zt.squeeze(0).float().cpu().numpy()  # (4,H,W,D)
        r = ndi.rotate(arr, ang, axes=(1, 2), reshape=False, order=1, mode="nearest")
        return torch.from_numpy(r)[None].half().to(dev)

    def zoom_latent(zt, s):   # spatial zoom via trilinear interpolate
        return F.interpolate(zt.float(), scale_factor=s, mode="trilinear",
                             align_corners=False).half()

    # ---------- Fig 1: coherence ----------
    ops = [
        ("identity", lambda: decode(zc)),
        ("rot90 (k=1)", lambda: decode(torch.rot90(zc, 1, dims=(2, 3)))),
        ("rot180 (k=2)", lambda: decode(torch.rot90(zc, 2, dims=(2, 3)))),
        ("rot45 (interp)", lambda: decode(rot_latent(zc, 45))),
        ("hflip", lambda: decode(torch.flip(zc, dims=(3,)))),
        ("avg_pool2 (→64³)", lambda: decode(F.avg_pool3d(zc.float(), 2).half())),
        ("zoom×1.5 (→192³)", lambda: decode(zoom_latent(zc, 1.5))),
        ("zoom×0.6 (→76³)", lambda: decode(zoom_latent(zc, 0.6))),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(4 * 3.0, 2 * 3.2), squeeze=False)
    for i, (name, fn) in enumerate(ops):
        img = fn()
        s = img.shape[2] // 2
        A = ax[i // 4][i % 4]
        A.imshow(np.clip(img[:, :, s], -200, 250), cmap="gray", vmin=-200, vmax=250)
        A.set_title(f"{name}  {img.shape}", fontsize=9)
        A.set_xticks([]); A.set_yticks([])
    fig.suptitle("VAE decode of latent-space ops (coherence)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = a.outdir / "latent_ops_coherence.png"
    fig.savefig(p1, dpi=110); print(f"[saved] {p1}", flush=True)

    # ---------- Fig 2: equivariance (decode∘op vs op∘decode) ----------
    base = decode(zc)  # HU 128³

    def img_rot(img, ang):
        return ndi.rotate(img, ang, axes=(0, 1), reshape=False, order=1, mode="nearest")

    tests = [
        ("rot90", decode(torch.rot90(zc, 1, dims=(2, 3))), np.rot90(base, 1, axes=(0, 1))),
        ("rot45", decode(rot_latent(zc, 45)), img_rot(base, 45)),
    ]
    fig2, ax2 = plt.subplots(len(tests), 3, figsize=(3 * 3.2, len(tests) * 3.3), squeeze=False)
    for r, (nm, dec_op, op_dec) in enumerate(tests):
        s = dec_op.shape[2] // 2
        # only compare where the image-rot didn't introduce empty border
        valid = np.abs(op_dec) < 3000
        mae = float(np.mean(np.abs(dec_op[valid] - op_dec[valid])))
        for c, (im, ttl) in enumerate([
            (dec_op[:, :, s], f"decode(op(z))  [{nm}]"),
            (op_dec[:, :, s], "op(decode(z))"),
            (np.abs(dec_op - op_dec)[:, :, s], f"|diff|  MAE={mae:.0f} HU"),
        ]):
            A = ax2[r][c]
            if c < 2:
                A.imshow(np.clip(im, -200, 250), cmap="gray", vmin=-200, vmax=250)
            else:
                A.imshow(im, cmap="magma", vmin=0, vmax=400)
            A.set_title(ttl, fontsize=9); A.set_xticks([]); A.set_yticks([])
        print(f"[equivariance] {nm}: HU MAE(decode∘op vs op∘decode) = {mae:.1f}", flush=True)
    fig2.suptitle("Equivariance: latent-op then decode  vs  decode then image-op", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = a.outdir / "latent_ops_equivariance.png"
    fig2.savefig(p2, dpi=110); print(f"[saved] {p2}", flush=True)


if __name__ == "__main__":
    main()

"""
Feasibility microbench for on-the-fly crop generation from a LATENT BANK.

Motivation (see docs/logs.md / memory): the MAISI pipeline decomposes into an
expensive OFFLINE part (30-step rflow denoise loop that invents appearance from a
mask) and a cheap ONLINE part (VAE decode of a latent). For training-time crop
synthesis we want to precompute denoised latents once, then per item do only:
    crop latent window -> [optional K-step SDEdit re-noise/denoise] -> VAE decode.

This bench measures the ONLINE cost so we know if it fits a training loop:
  1. decode-only latency vs crop size (32/48/64/96 latent -> 4x image)
  2. decode throughput vs batch size (can we amortise across a mini-batch of crops?)
  3. K-step SDEdit cost (renoise to a strength then denoise K steps) for K in {2,3,5}
     + the full 30-step loop at the same crop latent size, as the baseline it replaces.

Reported per-item estimate = SDEdit(K) loop + one decode. Compare against the target
per-batch-item budget of your dataloader (well under ~100 ms to keep GPUs fed, or more
if run in a prefetch worker like gen_maisi_fast.py's).

Run inside the NV-Generate-CTMR repo env:
  MONAI_DATA_DIRECTORY=./temp_work_dir \
  /home/dpxuser/dev/patch_icl/.venv_thor/bin/python \
    experiments/3d/synth_task_generation/bench_decode_sdedit.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR \
    --infer_file config_infer_wholebody.json
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

# reuse the loader machinery from the fast generator (build_args mirrors inference.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_maisi_fast import build_args  # noqa: E402


def _sync_time(fn, n=10, warmup=2):
    """Mean/std ms over n runs of fn(), with CUDA sync + warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time()
        fn()
        torch.cuda.synchronize(); ts.append((time.time() - t) * 1e3)
    return float(np.mean(ts)), float(np.std(ts))


def load_real_mask_crop(out_dir: Path, crop: int, device) -> torch.Tensor | None:
    """A (1,1,crop,crop,crop) integer MAISI label crop from an existing npz, centred on
    the densest region — so binarize_labels sees a realistic multi-class conditioning."""
    files = sorted(glob.glob(str(out_dir / "*.npz")))
    if not files:
        return None
    d = np.load(files[0])
    lbl = d["label"].astype(np.int64)  # (H,W,D) MAISI ids
    # centre on the voxel with most non-background mass along each axis
    fg = lbl > 0
    if fg.sum() == 0:
        c = [s // 2 for s in lbl.shape]
    else:
        idx = np.array(np.nonzero(fg))
        c = idx.mean(axis=1).astype(int).tolist()
    sl = []
    for ci, s in zip(c, lbl.shape):
        lo = int(np.clip(ci - crop // 2, 0, max(0, s - crop)))
        sl.append(slice(lo, lo + crop))
    out = lbl[sl[0], sl[1], sl[2]]
    pad = [(0, crop - out.shape[i]) for i in range(3)]
    out = np.pad(out, pad)
    return torch.from_numpy(out)[None, None].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--out", type=Path,
                    default=Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                                 "ANALYSIS_20251122/data/synth_task_gen/maisi"),
                    help="dir with generated npz (for a realistic mask crop)")
    ap.add_argument("--crop", type=int, default=128, help="image-space crop edge (latent = crop//4)")
    ap.add_argument("--tp_splits", type=int, default=1,
                    help="AE channel-split (num_splits). wholebody cfg uses 4 for memory; "
                         "crops are small so 1 is faster. Set >1 to see the serialisation cost.")
    args_cli = ap.parse_args()

    sys.path.insert(0, str(args_cli.repo))
    from scripts.utils import binarize_labels
    from scripts.utils_infer import ReconModel, load_image_models

    device = torch.device("cuda")
    args = build_args(args_cli.repo, args_cli.env_file, args_cli.infer_file)
    args.autoencoder_def["num_splits"] = args_cli.tp_splits  # override wholebody's 4
    print(f"[cfg] AE num_splits={args_cli.tp_splits}  crop={args_cli.crop}  spacing={args.spacing}", flush=True)

    t0 = time.time()
    ae, dm, cn, scale_factor, sched = load_image_models(args, device)
    ae.eval(); dm.eval(); cn.eval()
    recon = ReconModel(autoencoder=ae, scale_factor=scale_factor).to(device).eval()
    print(f"[load] image models ready in {time.time()-t0:.1f}s")

    # --- probe the cuDNN slow-path warning: does autotune / channels_last help decode? ---
    print("\n=== 0. decode 128^3: cuDNN path probe ===", flush=True)
    z128 = torch.randn(1, args.latent_channels, 32, 32, 32, device=device).half()
    for tag, bench in (("benchmark=False", False), ("benchmark=True", True)):
        torch.backends.cudnn.benchmark = bench
        with torch.no_grad(), torch.amp.autocast("cuda"):
            m, s = _sync_time(lambda: recon(z128), n=8, warmup=3)
        print(f"  {tag:>18}: {m:7.1f} ± {s:.1f} ms", flush=True)
    try:
        ae_cl = recon.autoencoder.to(memory_format=torch.channels_last_3d)
        zcl = z128.to(memory_format=torch.channels_last_3d)
        torch.backends.cudnn.benchmark = True
        with torch.no_grad(), torch.amp.autocast("cuda"):
            m, s = _sync_time(lambda: ae_cl.decode_stage_2_outputs(zcl / scale_factor), n=8, warmup=3)
        print(f"  {'channels_last_3d':>18}: {m:7.1f} ± {s:.1f} ms", flush=True)
        recon.autoencoder.to(memory_format=torch.contiguous_format)
    except Exception as e:
        print(f"  channels_last_3d: failed ({type(e).__name__}: {e})", flush=True)
    torch.backends.cudnn.benchmark = True  # keep autotune on for the rest

    include_modality = dm.num_class_embeds is not None
    modality_tensor = torch.ones((1,), dtype=torch.long, device=device)  # CT=1
    spacing_tensor = torch.FloatTensor(args.spacing).unsqueeze(0).half().to(device) * 1e2

    autocast = torch.amp.autocast("cuda")

    # ------------------------------------------------------------------ 1. decode-only
    # NB: single-shot whole-body decode (latent 96^3 -> 384^3) OOMs by design — that is
    # exactly why the real pipeline tiles it with SlidingWindowInferer. Our crop use case
    # is the small end (latent <=64^3 -> <=256^3), which decodes single-shot.
    print("\n=== 1. decode-only latency (random latent -> image) ===", flush=True)
    print(f"{'latent':>14} {'image':>16} {'ms':>10} {'±':>7}", flush=True)
    for ls in (16, 32, 48, 64):
        z = torch.randn(1, args.latent_channels, ls, ls, ls, device=device).half()
        try:
            with torch.no_grad(), autocast:
                m, s = _sync_time(lambda: recon(z))
            print(f"{f'4x{ls}^3':>14} {f'{ls*4}^3':>16} {m:>10.1f} {s:>7.1f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{f'4x{ls}^3':>14} {f'{ls*4}^3':>16}        OOM", flush=True)
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 2. decode batch
    lc = args_cli.crop // 4
    print(f"\n=== 2. decode throughput vs batch (latent 4x{lc}^3 -> {args_cli.crop}^3) ===", flush=True)
    print(f"{'batch':>8} {'ms/batch':>12} {'ms/item':>10}", flush=True)
    for B in (1, 2, 4, 8):
        z = torch.randn(B, args.latent_channels, lc, lc, lc, device=device).half()
        try:
            with torch.no_grad(), autocast:
                m, _ = _sync_time(lambda: recon(z), n=6)
            print(f"{B:>8} {m:>12.1f} {m/B:>10.1f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>8}  OOM", flush=True); torch.cuda.empty_cache(); break
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 3. SDEdit steps
    print(f"\n=== 3. SDEdit K-step cost at latent 4x{lc}^3 (image {args_cli.crop}^3) ===")
    mask = load_real_mask_crop(args_cli.out, args_cli.crop, device)
    if mask is None:
        print("  no npz found for a real mask crop -> using a synthetic block mask")
        mask = torch.zeros(1, 1, args_cli.crop, args_cli.crop, args_cli.crop,
                           dtype=torch.long, device=device)
        q = args_cli.crop // 4
        mask[..., q:3*q, q:3*q, q:3*q] = 1
        mask[..., q:3*q, q:3*q, q:3*q][..., ::2] = 3
    n_classes = int(torch.unique(mask).numel())
    with torch.no_grad(), autocast:
        cond = binarize_labels(mask.long()).half()  # (1,8,crop,crop,crop)

    latent_numel = lc ** 3
    sched.set_timesteps(num_inference_steps=30, input_img_size_numel=latent_numel, device=device)
    full_ts = sched.timesteps                       # 30 warped timesteps, descending (on device)
    full_next = torch.cat((full_ts[1:], torch.tensor([0], dtype=full_ts.dtype, device=device)))

    def denoise_steps(ts_list, next_list, z):
        """One SDEdit/diffusion pass over the given (t, next_t) schedule. Mirrors
        utils_infer.run_controlnet_conditioned_image_dm's inner loop (cfg off)."""
        latents = z
        for t, nt in zip(ts_list, next_list):
            tt = torch.as_tensor([float(t)], device=device)
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
        return latents

    z0 = torch.randn(1, args.latent_channels, lc, lc, lc, device=device).half()  # stand-in banked latent

    def sdedit(K):
        ts = full_ts[-K:]                      # lowest K timesteps = denoise tail
        nx = full_next[-K:]
        t_start = ts[0]
        noise = torch.randn_like(z0)
        znoisy = sched.add_noise(z0, noise, t_start.reshape(1).float().to(device))
        return denoise_steps(list(ts), list(nx), znoisy)

    print(f"  mask crop: {n_classes} unique MAISI classes  |  cond {tuple(cond.shape)}", flush=True)
    print(f"{'variant':>16} {'loop ms':>10} {'decode ms':>11} {'total ms':>10}", flush=True)
    with torch.no_grad(), autocast:
        dm_, _ = _sync_time(lambda: recon(z0), n=8)  # decode of one crop latent
        for K in (2, 3, 5):
            lm, _ = _sync_time(lambda: sdedit(K), n=6)
            print(f"{f'SDEdit K={K}':>16} {lm:>10.1f} {dm_:>11.1f} {lm+dm_:>10.1f}", flush=True)
        fm, _ = _sync_time(lambda: denoise_steps(list(full_ts), list(full_next), z0), n=3)
        print(f"{'full 30-step':>16} {fm:>10.1f} {dm_:>11.1f} {fm+dm_:>10.1f}", flush=True)

    print(f"\n[peak GPU mem] {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print("[interpret] on-the-fly item cost ≈ SDEdit(K) total. Compare to your per-item\n"
          "            dataloader budget; batch-decode row 2 shows amortisation if you\n"
          "            collate crops and decode them together in a prefetch worker.")


if __name__ == "__main__":
    main()

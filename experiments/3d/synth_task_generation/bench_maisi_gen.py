"""
Microbenchmark for the MAISI (NV-Generate-CTMR) rflow-ct GPU generation stages.

Isolates the two GPU-bound stages of the paired-generation pipeline and sweeps
the levers we care about — torch.compile and diffusion batch size — so we can see
the achievable per-volume speedup *before* touching the full pipeline (mask prep,
QC, save are excluded on purpose; they are CPU/IO and separately parallelizable).

Stages timed (256^3 output, 64^3 latent, matching config_infer_batch.json):
  1. diffusion  -- N rflow steps of ControlNet + diffusion-UNet on a (B,4,64,64,64) latent
  2. vae_decode -- SlidingWindowInferer AE decode of the (B,4,64,64,64) latent -> (B,1,256,256,256)

For each (compile, batch) cell it reports per-*volume* ms (total / B), so a lower
number at higher B means batching is winning. Peak GPU memory is reported per cell.

Run (from the NV-Generate-CTMR repo root, with its venv):
  MONAI_DATA_DIRECTORY=./temp_work_dir \
  /home/dpxuser/dev/patch_icl/.venv_thor/bin/python \
    /home/dpxuser/dev/patch_icl/experiments/3d/synth_task_generation/bench_maisi_gen.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --steps 30 --batches 1,2,4 --reps 3
"""

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch


def build_args(repo: Path) -> Namespace:
    """Reconstruct the args Namespace inference.py builds, from the three configs."""
    args = Namespace()
    env = json.load(open(repo / "configs/environment_rflow-ct.json"))
    net = json.load(open(repo / "configs/config_network_rflow.json"))
    inf = json.load(open(repo / "configs/config_infer_batch.json"))
    for d in (env, net, inf):
        for k, v in d.items():
            setattr(args, k, v)
    # model paths are repo-relative; make them absolute so cwd doesn't matter
    for k in [
        "trained_autoencoder_path",
        "trained_diffusion_path",
        "trained_controlnet_path",
    ]:
        setattr(args, k, str(repo / getattr(args, k)))
    # tp_num_splits override (same as inference.py:113)
    if "autoencoder_tp_num_splits" in inf:
        args.autoencoder_def["num_splits"] = inf["autoencoder_tp_num_splits"]
    return args


def load_nets(args, device):
    """Load AE / diffusion-UNet / ControlNet + scale_factor + scheduler (mirrors inference.py)."""
    import monai  # noqa: F401  (registers monai targets for ConfigParser)
    from scripts.utils import define_instance

    autoencoder = define_instance(args, "autoencoder_def").to(device).eval()
    ckpt = torch.load(args.trained_autoencoder_path, weights_only=False)
    autoencoder.load_state_dict(ckpt["unet_state_dict"] if "unet_state_dict" in ckpt else ckpt)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device).eval()
    ckpt_dm = torch.load(args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(ckpt_dm["unet_state_dict"], strict=False)
    scale_factor = ckpt_dm["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet_def").to(device).eval()
    ckpt_cn = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(ckpt_cn["controlnet_state_dict"], strict=False)

    noise_scheduler = define_instance(args, "noise_scheduler")
    return autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler


def make_cond(args, batch, device):
    """Build a realistic (B,8,256,256,256) ControlNet conditioning + aux tensors.

    Mask *content* is irrelevant to timing; we use a few blobs of distinct labels.
    """
    from scripts.utils import binarize_labels

    H, W, D = args.output_size
    mask = torch.zeros((1, 1, H, W, D), dtype=torch.long, device=device)
    # a handful of labelled boxes so binarize_labels lights up several bit-planes
    for i, lab in enumerate([1, 3, 6, 11, 15, 200]):
        o = 16 * (i + 1)
        mask[..., o : o + 40, o : o + 40, o : o + 40] = lab
    cond = binarize_labels(mask).half().repeat(batch, 1, 1, 1, 1)  # (B,8,H,W,D)
    spacing_tensor = (torch.tensor([args.spacing], dtype=torch.float16, device=device) * 1e2).repeat(batch, 1)
    modality_tensor = torch.tensor([args.modality] * batch, device=device)
    return cond, spacing_tensor, modality_tensor


def diffusion_loop(diffusion_unet, controlnet, noise_scheduler, latents, cond, spacing_tensor, modality_tensor, steps):
    """Batched ControlNet + diffusion-UNet denoising loop (rflow, cfg=0, modality cond, no body region)."""
    from monai.networks.schedulers import RFlowScheduler

    device = latents.device
    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(num_inference_steps=steps, input_img_size_numel=torch.prod(torch.tensor(latents.shape[-3:])))
    else:
        noise_scheduler.set_timesteps(num_inference_steps=steps)
    ts = noise_scheduler.timesteps
    next_ts = torch.cat((ts[1:], torch.tensor([0], dtype=ts.dtype)))
    B = latents.shape[0]
    for t, next_t in zip(ts, next_ts):
        tt = torch.full((B,), float(t), device=device)  # per-sample timestep (matches class_labels batch)
        down, mid = controlnet(x=latents, timesteps=tt, controlnet_cond=cond, class_labels=modality_tensor)
        model_out = diffusion_unet(
            x=latents,
            timesteps=tt,
            spacing_tensor=spacing_tensor,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
            class_labels=modality_tensor,
        )
        if isinstance(noise_scheduler, RFlowScheduler):
            latents, _ = noise_scheduler.step(model_out, t, latents, next_t)
        else:
            latents, _ = noise_scheduler.step(model_out, t, latents)
    return latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batches", type=str, default="1,2,4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compile", type=str, default="off,on", help="comma list from {off,on}")
    args_cli = ap.parse_args()

    sys.path.insert(0, str(args_cli.repo))  # so `import scripts...` resolves
    from monai.inferers.inferer import SlidingWindowInferer

    from scripts.utils import dynamic_infer
    from scripts.utils_infer import ReconModel

    device = torch.device("cuda")
    args = build_args(args_cli.repo)
    latent_shape = [args.latent_channels] + [s // 4 for s in args.output_size]
    print(f"output_size={args.output_size}  latent_shape={latent_shape}  steps={args_cli.steps}  "
          f"vae_window={args.autoencoder_sliding_window_infer_size}  overlap={args.autoencoder_sliding_window_infer_overlap}  "
          f"tp_splits={args.autoencoder_def['num_splits']}")

    ae, dm, cn, scale_factor, sched = load_nets(args, device)
    batches = [int(b) for b in args_cli.batches.split(",")]
    compiles = [c.strip() for c in args_cli.compile.split(",")]

    def build_inferer():
        return SlidingWindowInferer(
            roi_size=list(args.autoencoder_sliding_window_infer_size),
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=args.autoencoder_sliding_window_infer_overlap,
            sw_device=device,
            device=torch.device("cpu"),
        )

    print(f"\n{'compile':>8} {'B':>3} {'diff ms/vol':>12} {'decode ms/vol':>14} {'total ms/vol':>13} {'vol/s':>7} {'peak GB':>8}")
    for comp in compiles:
        dm_use, cn_use = dm, cn
        recon = ReconModel(autoencoder=ae, scale_factor=scale_factor).to(device)
        if comp == "on":
            dm_use = torch.compile(dm, mode="default")
            cn_use = torch.compile(cn, mode="default")
            recon.autoencoder = torch.compile(ae, mode="default")
        inferer = build_inferer()  # reuse: recomputes gaussian map only once
        for B in batches:
            cond, spacing_tensor, modality_tensor = make_cond(args, B, device)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                # warmup (also triggers compile)
                lat = torch.randn([B] + latent_shape, device=device).half()
                lat = diffusion_loop(dm_use, cn_use, sched, lat, cond, spacing_tensor, modality_tensor, args_cli.steps)
                _ = dynamic_infer(inferer, recon, lat)
                torch.cuda.synchronize()

                diff_t, dec_t = [], []
                for _ in range(args_cli.reps):
                    lat = torch.randn([B] + latent_shape, device=device).half()
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    lat = diffusion_loop(dm_use, cn_use, sched, lat, cond, spacing_tensor, modality_tensor, args_cli.steps)
                    torch.cuda.synchronize(); t1 = time.perf_counter()
                    _ = dynamic_infer(inferer, recon, lat)
                    torch.cuda.synchronize(); t2 = time.perf_counter()
                    diff_t.append(t1 - t0); dec_t.append(t2 - t1)
            diff_ms = 1000 * min(diff_t) / B
            dec_ms = 1000 * min(dec_t) / B
            tot_ms = diff_ms + dec_ms
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"{comp:>8} {B:>3} {diff_ms:>12.1f} {dec_ms:>14.1f} {tot_ms:>13.1f} {1000/tot_ms:>7.2f} {peak:>8.2f}")


if __name__ == "__main__":
    main()

"""
How fast can we push the MAISI VAE decode? (the 77%/~825ms bottleneck)

Goal: cheap decode of small crops for generating DIVERSE priors (realism not required),
so we can afford 96³ crops, compile, lower precision, etc. This sweeps the decode-only
cost across the levers that could beat the cuDNN-fallback slow path:

  * crop size          96³ (latent 24³)  vs  128³ (latent 32³)
  * torch.compile      none / default / max-autotune   (may route convs off the
                       'cuDNN cannot be used for large non-batch-splittable conv' fallback)
  * precision          autocast-fp16 / autocast-bf16 / pure-fp16 (model.half, no autocast)
  * batch              1 vs 4 under the best variant (amortisation once kernels are good)

Reports ms/item; baseline (autocast-fp16, no compile) was ~825ms @128³ / measure @96³.

  MONAI_DATA_DIRECTORY=./temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/bench_vae_decode_opt.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_maisi_fast import build_args  # noqa: E402


def _time(fn, n, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time()
        fn()
        torch.cuda.synchronize(); ts.append((time.time() - t) * 1e3)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--maxautotune", action="store_true", default=True,
                    help="also try compile mode=max-autotune (slow warmup, target crop only)")
    ap.add_argument("--no-maxautotune", dest="maxautotune", action="store_false")
    args_cli = ap.parse_args()

    sys.path.insert(0, str(args_cli.repo))
    from scripts.utils_infer import ReconModel, load_image_models

    device = torch.device("cuda")
    args = build_args(args_cli.repo, args_cli.env_file, args_cli.infer_file)
    args.autoencoder_def["num_splits"] = 1  # crops are small; no channel-split needed
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ae, dm, cn, scale_factor, sched = load_image_models(args, device)
    del dm, cn  # decode-only bench
    torch.cuda.empty_cache()
    recon = ReconModel(autoencoder=ae, scale_factor=scale_factor).to(device).eval()
    print(f"[cfg] num_splits=1  tf32=on  cudnn.benchmark=on", flush=True)

    results = []

    def run(tag, crop, make_call, precision, n=8, warmup=3):
        torch.cuda.reset_peak_memory_stats()
        try:
            m, s = _time(make_call, n=n, warmup=warmup)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  {tag:>34} | {crop}^3 | {precision:>14} | {m:8.1f} ± {s:5.1f} ms | {peak:4.1f} GB", flush=True)
            results.append((tag, crop, precision, m))
        except Exception as e:
            print(f"  {tag:>34} | {crop}^3 | {precision:>14} | FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
        torch.cuda.empty_cache()

    print(f"\n{'variant':>36} | crop |      precision |          ms | peak", flush=True)

    for crop in (96, 128):
        lc = crop // 4
        z = torch.randn(1, args.latent_channels, lc, lc, lc, device=device)

        # 1. baseline autocast-fp16
        zf = z.half()
        run("baseline (autocast fp16)", crop,
            lambda: _decode_autocast(recon, zf, "float16"), "autocast-fp16")

        # 2. autocast-bf16
        run("autocast bf16", crop,
            lambda: _decode_autocast(recon, z.bfloat16(), "bfloat16"), "autocast-bf16")

        # 3. pure fp16 (model.half, no autocast)
        recon_h = ReconModel(autoencoder=ae.half(), scale_factor=scale_factor).to(device).eval()
        zh = z.half()
        run("pure fp16 (no autocast)", crop,
            lambda: _decode_plain(recon_h, zh), "pure-fp16")
        ae.float()  # restore

        # 4. torch.compile default (on pure-fp16 model)
        recon_h2 = ReconModel(autoencoder=ae.half(), scale_factor=scale_factor).to(device).eval()
        try:
            crecon = torch.compile(recon_h2, mode="default")
            run("compile default (fp16)", crop, lambda: _decode_plain(crecon, zh), "pure-fp16", n=6, warmup=5)
        except Exception as e:
            print(f"  compile default: FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
        ae.float()

        # 5. compile max-autotune (target 96 only, slow warmup)
        if args_cli.maxautotune and crop == 96:
            recon_h3 = ReconModel(autoencoder=ae.half(), scale_factor=scale_factor).to(device).eval()
            try:
                crecon2 = torch.compile(recon_h3, mode="max-autotune")
                run("compile max-autotune (fp16)", crop, lambda: _decode_plain(crecon2, zh), "pure-fp16", n=6, warmup=6)
            except Exception as e:
                print(f"  compile max-autotune: FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
            ae.float()

    # 6. batching at 96 under pure-fp16 (does it amortise now?)
    print(f"\n=== batch amortisation @96^3 (pure fp16, no compile) ===", flush=True)
    recon_hb = ReconModel(autoencoder=ae.half(), scale_factor=scale_factor).to(device).eval()
    lc = 96 // 4
    for B in (1, 2, 4, 8):
        zb = torch.randn(B, args.latent_channels, lc, lc, lc, device=device).half()
        try:
            m, _ = _time(lambda: _decode_plain(recon_hb, zb), n=5, warmup=2)
            print(f"  B={B}: {m:8.1f} ms/batch = {m/B:7.1f} ms/item", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  B={B}: OOM", flush=True); torch.cuda.empty_cache(); break
    ae.float()

    print(f"\n=== summary: best ms/item per crop ===", flush=True)
    for crop in (96, 128):
        cand = [(m, tag, prec) for tag, c, prec, m in results if c == crop]
        if cand:
            m, tag, prec = min(cand)
            print(f"  {crop}^3: {m:.1f} ms  ({tag}, {prec})", flush=True)


def _decode_autocast(recon, z, dtype):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=getattr(torch, dtype)):
        return recon(z)


def _decode_plain(recon, z):
    with torch.no_grad():
        return recon(z)


if __name__ == "__main__":
    main()

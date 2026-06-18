"""
Benchmark the frozen feature encoders (UniverSeg vs DINOv3) under matched
conditions: same input batch / image size / output grid, same precision, same
compile setting. Reports FLOPs, peak VRAM, and latency per (encoder, config).

Both encoders are how the pipeline runs them: frozen, eval, no_grad, fed a
grayscale (B, 1, H, W) batch and pooled to an out_size token grid. The encode
call in ImagePFN / the multilevel chain is on (B*T, 1, H, W) with T = K+1, so the
default batch 64 ≈ batch_size 16 × (3 context + 1 query).

Usage:
    .venv311/bin/python experiments/2d/bench_encoders.py
    .venv311/bin/python experiments/2d/bench_encoders.py --batch 64 --image-size 256 --out-size 32
    .venv311/bin/python experiments/2d/bench_encoders.py --encoders universeg dinov3 dinov3-large
"""

import argparse
import os
import sys
import time
from pathlib import Path

# DINOv3 weights are in the local HF cache only — keep transformers offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Drop this script's own dir from sys.path: experiments/2d/universeg.py would
# otherwise shadow the real `universeg` package the UniverSeg encoder imports.
_self_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _self_dir)]
from src.models.pretrained_encoders import build_image_encoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(name, level):
    """name like 'universeg' | 'dinov3' | 'dinov3-large'."""
    enc, fdim = build_image_encoder(
        {"image_encoder": name, "feature_level": level}, DEVICE)
    enc.eval()
    return enc, fdim


@torch.no_grad()
def measure_flops(enc, x, out_size):
    """Forward FLOPs for one batch (eager, fp32). Returns total GFLOPs."""
    fc = FlopCounterMode(display=False)
    with fc:
        enc(x, out_size)
    return fc.get_total_flops() / 1e9


@torch.no_grad()
def measure(enc, x, out_size, *, autocast, compile_, iters, warmup):
    """Peak VRAM (MiB) + latency (ms/iter) for a config. Returns (vram, ms)."""
    fn = torch.compile(enc, dynamic=True) if compile_ else enc
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if autocast else torch.autocast(device_type="cuda", enabled=False))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with ctx:
        for _ in range(warmup):
            fn(x, out_size)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x, out_size)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 2**20
    return vram, dt / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["universeg", "dinov3"])
    ap.add_argument("--level", default="all", help="feature level (stage) or 'all'")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--out-size", type=int, default=16, help="pooled token-grid side")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if DEVICE.type != "cuda":
        sys.exit("CUDA required for VRAM/latency benchmarking.")
    torch.set_float32_matmul_precision("high")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"input: B={args.batch} {args.image_size}x{args.image_size} (1ch) "
          f"→ grid {args.out_size}x{args.out_size}  level={args.level}\n")

    x = torch.randn(args.batch, 1, args.image_size, args.image_size, device=DEVICE)

    # config = (label, autocast, compile)
    configs = [
        ("eager fp32",     False, False),
        ("eager bf16",     True,  False),
        ("compile bf16",   True,  True),
    ]
    hdr = f"{'encoder':<16}{'feat_dim':>9}{'GFLOPs':>10}{'config':>15}{'VRAM MiB':>11}{'ms/iter':>10}"
    print(hdr)
    print("-" * len(hdr))

    for name in args.encoders:
        enc, fdim = build(name, args.level)
        gflops = measure_flops(enc, x, args.out_size)
        for label, ac, comp in configs:
            try:
                vram, ms = measure(enc, x, args.out_size, autocast=ac, compile_=comp,
                                   iters=args.iters, warmup=args.warmup)
                print(f"{name:<16}{fdim:>9}{gflops:>10.1f}{label:>15}{vram:>11.0f}{ms:>10.2f}")
            except Exception as e:
                print(f"{name:<16}{fdim:>9}{gflops:>10.1f}{label:>15}{'ERR':>11}{'':>10}  {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
        del enc
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()

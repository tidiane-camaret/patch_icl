"""Profile tap-ct-b-3d: compute time, peak VRAM and FLOPs for the full-volume
forward and for sliding-window inference.

Precision and torch.compile are CLI params. FLOPs are a property of the math
(independent of dtype/compile), so they are counted once on the eager model via
torch's FlopCounterMode (which sees the flash/SDPA kernel correctly); time and
VRAM are measured under the requested precision/compile.

Requires the SDPA attention patch (load_model(use_sdpa=True)) so the full-volume
forward does not OOM without xformers -- see tap_ct_bench.py.

Examples:
  .venv_thor/bin/python experiments/encoders/tap_ct_profile.py
  .venv_thor/bin/python experiments/encoders/tap_ct_profile.py --precision bf16 --compile
  .venv_thor/bin/python experiments/encoders/tap_ct_profile.py --depth 180 --overlap 0.75 --skip-flops
"""
import argparse
import os
import time
from contextlib import nullcontext

import torch
from torch.utils.flop_counter import FlopCounterMode

# thor/loki toolchain: force absolute g++/gcc so inductor's C++ wrapper builds.
if os.path.realpath("/bin") != os.path.realpath("/usr/bin"):
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")

from tap_ct_bench import load_model, make_input, n_tokens, INPLANE  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def amp_ctx(precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=DTYPES[precision])


def full_predictor(model):
    def fn(x):
        return model(x).last_hidden_state
    return fn


def sliding_predictor(model, roi, overlap, sw_batch):
    from monai.inferers import SlidingWindowInferer
    inferer = SlidingWindowInferer(
        roi_size=list(roi), sw_batch_size=sw_batch, overlap=overlap, mode="gaussian"
    )

    def reshape_pred(patch):
        return model(patch, reshape=True).last_hidden_state

    def fn(x):
        return inferer(x, reshape_pred)
    return fn


def measure(fn, x, precision, iters):
    """Return (mean_time_s, peak_GB, out_shape)."""
    with torch.no_grad(), amp_ctx(precision):
        for _ in range(2):  # warmup (also triggers compile)
            out = fn(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1e9
    shape = tuple(out.shape)
    del out
    return dt, peak, shape


def count_flops(fn, x):
    """Total FLOPs for one call, counted on eager fp32 (precision-independent)."""
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        fn(x)
    return fc.get_total_flops()


def report(name, dt, peak, shape, flops):
    line = f"  time {dt*1000:8.1f} ms   peak {peak:6.2f} GB"
    if flops is not None:
        tflop = flops / 1e12
        line += f"   FLOPs {tflop:8.2f} T   {tflop/dt:7.2f} TFLOP/s"
    print(f"{name}\n{line}\n  out {shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=180,
                    help="padded depth D of the (1,1,D,224,224) volume (179->180)")
    ap.add_argument("--precision", choices=list(DTYPES), default="fp32")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--roi", type=int, nargs=3, default=[12, INPLANE, INPLANE],
                    help="sliding-window roi (D H W)")
    ap.add_argument("--overlap", type=float, default=0.75)
    ap.add_argument("--sw-batch", type=int, default=1)
    ap.add_argument("--skip-flops", action="store_true")
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-sliding", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  {props.total_memory/1e9:.1f} GB")
    print(f"depth={args.depth}  tokens={n_tokens(args.depth)}  "
          f"precision={args.precision}  compile={args.compile}"
          f"{'/'+args.compile_mode if args.compile else ''}  iters={args.iters}")
    print(f"sliding: roi={tuple(args.roi)} overlap={args.overlap} sw_batch={args.sw_batch}\n")

    x = make_input(args.depth, device, torch.float32)
    model = load_model(device, use_sdpa=True)
    run_model = torch.compile(model, mode=args.compile_mode) if args.compile else model

    # FLOPs on the eager (uncompiled) model -- FlopCounterMode can't see inside
    # compiled/fused kernels, and FLOPs don't depend on precision or backend.
    if args.skip_flops:
        full_flops = slide_flops = None
    else:
        full_flops = count_flops(full_predictor(model), x)
        slide_flops = count_flops(
            sliding_predictor(model, args.roi, args.overlap, args.sw_batch), x)

    if not args.skip_full:
        dt, peak, shape = measure(full_predictor(run_model), x, args.precision, args.iters)
        report("[full volume]", dt, peak, shape, None if args.skip_flops else full_flops)

    if not args.skip_sliding:
        fn = sliding_predictor(run_model, args.roi, args.overlap, args.sw_batch)
        dt, peak, shape = measure(fn, x, args.precision, args.iters)
        report("[sliding window]", dt, peak, shape, None if args.skip_flops else slide_flops)


if __name__ == "__main__":
    main()

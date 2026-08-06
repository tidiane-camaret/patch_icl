"""Focused torch.compile test for tap-ct-b-3d at the target depth (D_pad=180).

Compares eager vs compiled on the SDPA + bf16 path. Run after tap_ct_bench.py.
"""
import os
import time

import torch

# thor/loki toolchain: force absolute g++/gcc so inductor's C++ wrapper builds.
if os.path.realpath("/bin") != os.path.realpath("/usr/bin"):
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")

from tap_ct_bench import load_model, make_input, INPLANE, n_tokens  # noqa: E402


def timed(fn, x, n=5):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(2):  # warmup / compile
            fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    device = torch.device("cuda")
    D = 180
    print(f"GPU: {torch.cuda.get_device_name(0)}  D_pad={D}  tokens={n_tokens(D)}")
    x = make_input(D, device, torch.float32)

    m = load_model(device, use_sdpa=True)

    eager = timed(m, x)
    torch.cuda.reset_peak_memory_stats()
    eager = timed(m, x)
    print(f"eager  sdpa+bf16 : {eager*1000:7.1f} ms  "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    try:
        mc = torch.compile(m, mode="max-autotune")
        comp = timed(mc, x)
        torch.cuda.reset_peak_memory_stats()
        comp = timed(mc, x)
        print(f"compiled          : {comp*1000:7.1f} ms  "
              f"peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB  "
              f"speedup {eager/comp:.2f}x")
    except Exception as e:
        print(f"torch.compile FAILED: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()

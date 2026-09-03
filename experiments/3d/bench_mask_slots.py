"""
Compute-cost spike: what does growing the c-axis (mask-slot columns) cost?

Today PatchSet3D's dual-axis transformer runs on (B, r, c, e) tokens with c=2
(1 img col + 1 mask col, see src/models/patchset3d.py::_tokens). The question here is
whether c could grow to 1 img col + M mask cols (gt, pred, bbox, point, scribble, ...,
all through one shared embed — embed cost is O(c) linear and dwarfed by attention, so
it isn't wired here) without a meaningful compute hit.

Benches the real, unmodified TransformerEncoderStack from src/models/pfn_seg_2d.py (it
already has no c=2 assumption anywhere) on synthetic tokens shaped like PatchSet3D's
default (non-full, non-register-routed) connectivity: every row attends to the
thinking+support prefix only (sep = n_think + K*N), same as _attn's default branch.

THROWAWAY: not wired into PatchSet3D; a probe script only. See conversation asking
"is it cheap to add mask-content slots on the c-axis" (2026-09-03).

    .venv_thor_fresh/bin/python experiments/3d/bench_mask_slots.py
    .venv_thor_fresh/bin/python experiments/3d/bench_mask_slots.py --ms 1 2 3 5 --ks 1 3 8
"""
import argparse
import gc
import os
import shutil as _shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Broken-usr-merge nodes: bare gcc/g++ resolve to /bin/* and fail Triton's compile.
if not os.path.islink("/bin"):
    for _var, _tool in (("CC", "gcc"), ("CXX", "g++")):
        _abs = f"/usr/bin/{_tool}"
        _found = _shutil.which(_tool)
        if _var not in os.environ and _found and _found.startswith("/bin/") and os.path.exists(_abs):
            os.environ[_var] = _abs

import torch

from src.models.pfn_seg_2d import TransformerEncoderStack

DEV = torch.device("cuda")
RES = 16                       # feature grid side -> N = RES**3 tokens/image
N = RES ** 3
E, A, L, H, N_THINK = 256, 4, 6, 512, 8     # shipping transformer shape (bench_attn_pattern.py)
RESIDUAL_DECAY = 0.95


def attn_gflops(M, K):
    """Analytical fwd FLOPs (L layers) for PatchSet3D's default connectivity, split into
    the sample-axis term (cross-image, scales linearly in c=1+M) and the feature-axis term
    (within-cell, small-seq manual attention, scales as c²). 4·Bp·E·Sq·Sk per attention op
    (matches bench_attn_pattern.py's attn_gflops shorthand)."""
    c = 1 + M
    r = N_THINK + (K + 1) * N          # thinking + K support + 1 query
    sep = N_THINK + K * N              # every row attends to this prefix only (default branch)
    sample = 4 * L * c * E * r * sep           # Bp = b·c (b=1), Sq=r, Sk=sep
    feature = 4 * L * r * E * c * c            # Bp = b·r (b=1), Sq=Sk=c
    return sample / 1e9, feature / 1e9


def make_tokens(K, M, requires_grad=True):
    c = 1 + M
    r = N_THINK + (K + 1) * N
    x = torch.randn(1, r, c, E, device=DEV, requires_grad=requires_grad)
    sep = N_THINK + K * N
    return x, sep


def measure(K, M, reps):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = TransformerEncoderStack(L, A, E, H, RESIDUAL_DECAY).to(DEV)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def one():
        opt.zero_grad(set_to_none=True)
        x, sep = make_tokens(K, M)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, sep)
        out.float().pow(2).mean().backward()
        opt.step()

    try:
        one()   # warmup (allocator + cudnn)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        one()
        torch.cuda.synchronize()
        ram = torch.cuda.max_memory_allocated() / 1e9

        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            one()
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / reps
        res = dict(ram_gb=ram, ms=ms)
    except torch.cuda.OutOfMemoryError:
        res = dict(ram_gb=float("nan"), ms=float("nan"), oom=True)
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 5],
                     help="mask-slot counts (c = 1 + M); M=1 is today's shipping c=2")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 8])
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"config: R={RES} N={N} e={E} a={A} l={L} h={H} n_think={N_THINK}  "
          f"B=1 fwd+bwd bf16, default (non-full,non-register) connectivity\n")

    results = {}
    for K in args.ks:
        for M in args.ms:
            sample_gf, feature_gf = attn_gflops(M, K)
            res = measure(K, M, args.reps)
            results[(K, M)] = (sample_gf, feature_gf, res)
            tag = "OOM" if res.get("oom") else f"{res['ram_gb']:6.2f} GB   {res['ms']:8.1f} ms"
            print(f"  K={K:<3} M={M} (c={1+M})   sample={sample_gf:9.1f}GF  "
                  f"feature={feature_gf:7.3f}GF   {tag}")

    print(f"\n{'='*92}\nSUMMARY (B=1, R={RES}, {E=} {A=} {L=}, fwd+bwd)\n{'='*92}")
    print(f"{'K':>4}{'M':>4}{'c':>4}{'sample GF':>12}{'feature GF':>12}"
          f"{'RAM GB':>10}{'ms':>10}{'ms/M=1':>10}")
    for K in args.ks:
        base_ms = results[(K, args.ms[0])][2].get("ms")
        for M in args.ms:
            sample_gf, feature_gf, res = results[(K, M)]
            if res.get("oom"):
                print(f"{K:>4}{M:>4}{1+M:>4}{sample_gf:>12.1f}{feature_gf:>12.3f}"
                      f"{'OOM':>10}{'OOM':>10}{'':>10}")
            else:
                rel = f"{res['ms']/base_ms:.2f}x" if base_ms else ""
                print(f"{K:>4}{M:>4}{1+M:>4}{sample_gf:>12.1f}{feature_gf:>12.3f}"
                      f"{res['ram_gb']:>10.2f}{res['ms']:>10.1f}{rel:>10}")


if __name__ == "__main__":
    main()

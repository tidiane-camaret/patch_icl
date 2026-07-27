"""Encoder-agnostic image-encoding cost probe for the feature-similarity study.

Measures the FROZEN / forward-only cost of one image encode (FLOPs, peak inference
VRAM, it/sec) via an adapter's `cost_target(input_res) -> (module, example_inputs)`
hook. Same call for any adapter, so cost is directly comparable across encoders.
This is an *inference* cost (no backward), unlike encoder_bench which times fwd+bwd.
"""
import sys
import time

import torch

# Reuse the FLOP counter from the encoder benchmark (fvcore, SDPA-unwrap-aware).
from encoder_bench.profiling import count_gflops


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_encode_cost(adapter, input_res, device, n_warmup=3, n_timed=10) -> dict:
    """One forward encode -> {encode_gflops, encode_vram_mb, encode_it_s}.

    encode_gflops : fvcore FLOPs of the encode forward (None if untraceable, e.g. SDPA).
    encode_vram_mb: peak CUDA memory during the forward (None on CPU).
    encode_it_s   : volumes/sec = 1000 / median forward-ms.
    """
    row = {"encode_gflops": None, "encode_vram_mb": None, "encode_it_s": None}
    if not hasattr(adapter, "cost_target"):
        return row
    module, inputs = adapter.cost_target(input_res)
    module = module.eval()

    row["encode_gflops"] = count_gflops(module, inputs)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def one():
        module(*inputs)

    try:
        for _ in range(n_warmup):
            one(); _sync(device)
        times = []
        for _ in range(n_timed):
            _sync(device); t0 = time.perf_counter()
            one(); _sync(device)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        med_ms = times[(len(times) - 1) // 2]
        row["encode_it_s"] = 1000.0 / med_ms if med_ms > 0 else None
        row["encode_vram_mb"] = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                                 if device.type == "cuda" else None)
    except Exception as e:                              # honest row on failure
        print(f"  WARN encode-cost probe failed: {e}", file=sys.stderr)
    return row

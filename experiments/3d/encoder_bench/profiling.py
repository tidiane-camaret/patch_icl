"""Device-aware measurement of one (encoder, input_size) point."""
import sys
import time
import traceback

import torch

from encoder_bench.registry import EncoderSpec, make_inputs


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_gflops(module, inputs) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None
    # fvcore's tracer can't trace a torch.compile OptimizedModule -> unwrap to the eager
    # module (shared weights, identical FLOPs) so compiled encoders still get a FLOP count.
    module = getattr(module, "_orig_mod", module)
    try:
        flops = FlopCountAnalysis(module, inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception:
        return None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_fwd_bwd(module, inputs, device, n_warmup, n_timed) -> float:
    """Median forward+backward wall-time (ms). Uses a scalar .sum() surrogate loss."""
    def one():
        for p in module.parameters():
            p.grad = None
        out = module(*inputs)
        outs = out if isinstance(out, (list, tuple)) else [out]
        loss = sum(o.float().sum() for o in outs)
        loss.backward()
    for _ in range(n_warmup):
        one(); _sync(device)
    times = []
    for _ in range(n_timed):
        _sync(device); t0 = time.perf_counter()
        one(); _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[(len(times) - 1) // 2]


def _peak_vram_mb(device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def _throughput(module, spec, input_size, device) -> float | None:
    """Largest batch that fits (exponential search) -> volumes/sec, fwd-only no_grad."""
    if device.type != "cuda":
        with torch.no_grad():
            x = torch.zeros(1, spec.in_ch, input_size, input_size, input_size, device=device)
            t0 = time.perf_counter(); module(*make_inputs(spec, x)); dt = time.perf_counter() - t0
        return 1.0 / dt if dt > 0 else None
    best_b, bs = None, 1
    while True:
        try:
            with torch.no_grad():
                x = torch.zeros(bs, spec.in_ch, input_size, input_size,
                                input_size, device=device)
                _sync(device); t0 = time.perf_counter()
                module(*make_inputs(spec, x)); _sync(device)
                dt = time.perf_counter() - t0
            best_b = (bs, dt); bs *= 2
            del x
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); break
        if bs > 256:
            break
    if best_b is None:
        return None
    b, dt = best_b
    return b / dt if dt > 0 else None


def profile_point(spec: EncoderSpec, input_size: int, device, module=None,
                  n_warmup=3, n_timed=10) -> dict:
    row = {"encoder": spec.name, "family": spec.family, "input_size": input_size,
           "params": None, "gflops": None, "fwd_bwd_ms": None,
           "train_vram_mb": None, "throughput_vol_s": None, "status": "ok"}
    if input_size % spec.size_multiple != 0:
        row["status"] = "skip:divisible"
        return row
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    try:
        if module is None:
            module = spec.factory()
        module = module.to(device).train()
        row["params"] = count_params(module)
        x = torch.zeros(1, spec.in_ch, input_size, input_size, input_size, device=device)
        inputs = make_inputs(spec, x)
        row["gflops"] = count_gflops(module, inputs)
        row["fwd_bwd_ms"] = _time_fwd_bwd(module, inputs, device, n_warmup, n_timed)
        row["train_vram_mb"] = _peak_vram_mb(device)
        row["throughput_vol_s"] = _throughput(module.eval(), spec, input_size, device)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  WARN profile error {spec.name}@{input_size}: {e}", file=sys.stderr)
        traceback.print_exc()
        row["status"] = f"error:{type(e).__name__}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row

"""
Benchmark the impact of PyTorch optimization techniques on the STU-Net 3-D
image encoder.

Techniques tested
-----------------
  Inference path (torch.no_grad):
    baseline         — plain encode_image_only
    compile_reduce   — torch.compile(mode="reduce-overhead")  [includes CUDA graphs]
    compile_autotune — torch.compile(mode="max-autotune")
    cuda_graph       — manual CUDAGraph capture + replay
    vmap             — torch.func.vmap over batch dim

  Training path (fwd + bwd):
    baseline           — plain forward + backward
    compile            — torch.compile(mode="max-autotune") + backward
    checkpoint         — per-stage gradient checkpointing
    compile_checkpoint — both combined

Usage
-----
    python experiments/encoders/benchmark_optimizations.py
    python experiments/encoders/benchmark_optimizations.py \\
        --variant base --image_sizes 64 128 --batch_sizes 1 \\
        --methods baseline compile_reduce cuda_graph checkpoint \\
        --modes inference training \\
        --n_runs 10 --n_warmup 3
    python experiments/encoders/benchmark_optimizations.py \\
        --methods baseline compile_autotune --modes inference  # long compile!

Notes
-----
  - Suppress inductor/Triton warning spam:
        python ... 2>&1 | grep -v "^[EW][0-9]"
  - compile methods fall back to eager on Python 3.12 due to a Triton
    PY_SSIZE_T_CLEAN bug; suppress_errors=True is set globally so they
    still run, just without full kernel fusion.
  - compile_checkpoint triggers a "WON'T CONVERT" warning: torch.compile
    cannot fully trace through torch.utils.checkpoint, so compile+checkpoint
    runs mostly eager (negligible benefit vs plain checkpoint).
  - vmap runs in fp32 (torch.autocast is not composable with torch.func.vmap);
    results are labelled "vmap (fp32)" for clarity.
  - cuda_graph VRAM readings are lower than baseline because graph static
    buffers are allocated outside the peak-measurement window.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.encoders.stunet import STUNetEncoder, _VARIANTS  # noqa

# Suppress inductor/Triton compilation failures (fall back to eager for broken ops).
# Required on Python 3.12 where some Triton 3-D conv templates fail with
# PY_SSIZE_T_CLEAN. Fallback is silent and correct; only some kernels lose fusion.
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Methods that apply to each mode
_INFERENCE_METHODS = {"baseline", "compile_reduce", "compile_autotune",
                      "cuda_graph", "vmap"}
_TRAINING_METHODS  = {"baseline", "compile", "checkpoint", "compile_checkpoint"}
_ALL_METHODS       = _INFERENCE_METHODS | _TRAINING_METHODS


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_image_only(
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """(B,1,D,H,W) → [skip0, …, bottleneck]. No mask path.

    No @torch.no_grad() here — callers control gradient context so this
    function works for both inference (measure_inference wraps in no_grad)
    and training (measure_training needs gradients to flow).
    """
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


def checkpointed_encode(
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """Per-stage gradient checkpointing on the image encoder.

    Recomputes activations on backward instead of storing them.
    ~50% less training VRAM at ~20-40% extra compute cost.
    use_reentrant=False is required for torch.compile compatibility.
    """
    from torch.utils.checkpoint import checkpoint as ckpt
    n = encoder._num_stages
    x = imgs
    skips: list[torch.Tensor] = []
    for stage in encoder.image_encoder.conv_blocks_context[:n - 1]:
        x = ckpt(stage, x, use_reentrant=False)
        skips.append(x)
    x = ckpt(encoder.image_encoder.conv_blocks_context[n - 1], x,
             use_reentrant=False)
    return skips + [x]


# ---------------------------------------------------------------------------
# Optimization method factories
# ---------------------------------------------------------------------------

def compile_encoder(
    encoder: STUNetEncoder,
    mode: str,
    encode_fn: Callable,
    imgs: torch.Tensor,
    amp: bool,
    device: torch.device,
) -> tuple[STUNetEncoder, Callable, float]:
    """torch.compile the encoder, trigger first-call kernel compilation.

    Returns (compiled_encoder, encode_fn, compile_time_seconds).
    The encode_fn is unchanged — pass the compiled encoder to it.
    """
    print(f"  torch.compile(mode={mode!r}) … compiling (may take minutes) …",
          end="  ", flush=True)
    compiled = torch.compile(encoder, mode=mode)
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
        _ = encode_fn(compiled, imgs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compile_time = time.perf_counter() - t0
    cached = compile_time < 0.5   # inductor cache hit: skip re-compilation
    print(f"done in {compile_time:.1f} s{'  (inductor cache hit)' if cached else ''}")
    return compiled, encode_fn, compile_time


def build_cuda_graph_encode(
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
    amp: bool,
    device: torch.device,
    n_warmup: int = 3,
) -> Callable:
    """Capture a CUDAGraph for encode_image_only; return a replay callable.

    The callable signature matches encode_image_only: (encoder, imgs) → list[Tensor].
    The encoder/imgs args are ignored at replay time (graph uses static buffers).
    imgs must have the same shape on every call.
    """
    if device.type != "cuda":
        raise RuntimeError("CUDA graphs require a CUDA device.")

    static_input = imgs.clone()

    # Warmup on a side stream before graph capture
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
                _ = encode_image_only(encoder, static_input)
    torch.cuda.current_stream(device).wait_stream(stream)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
            static_output = encode_image_only(encoder, static_input)

    def _replay(_encoder: STUNetEncoder, _imgs: torch.Tensor) -> list[torch.Tensor]:
        static_input.copy_(_imgs)
        g.replay()
        return static_output   # static GPU buffers — valid until next replay

    return _replay


def make_vmap_encode(
    encoder: STUNetEncoder,
    amp: bool = True,
    device: torch.device = torch.device("cuda"),
) -> Callable:
    """Return a vmap-batched encode function.

    Maps a single-image forward over the batch dim via torch.func.vmap.
    torch.autocast does not propagate through functional transforms, so it is
    applied explicitly inside the vmapped function.

    The returned callable has the same signature as encode_image_only:
        (encoder, imgs: (B,1,D,H,W)) → list[Tensor]
    """
    def _single(img: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # img: (1, D, H, W) — vmap strips the batch dim.
        # torch.autocast is not composable with torch.func.vmap in PyTorch 2.x;
        # cast to fp32 explicitly to match model weight dtype.
        feats = encode_image_only(encoder, img.float().unsqueeze(0))
        return tuple(f.squeeze(0) for f in feats)

    _vmapped = torch.func.vmap(_single, in_dims=0)

    def vmap_encode(_encoder: STUNetEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
        # Disable any outer autocast: vmap functional transforms are incompatible
        # with torch.autocast. Input cast to fp32 to match model weight dtype.
        with torch.amp.autocast(device_type=device.type, enabled=False):
            return list(_vmapped(imgs.float()))

    return vmap_encode


# ---------------------------------------------------------------------------
# Core measurement functions
# ---------------------------------------------------------------------------

def measure_inference(
    encode_fn: Callable,
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
    n_warmup: int = 3,
    n_runs: int = 10,
    amp: bool = True,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Measure inference latency and peak VRAM.

    encode_fn: (encoder, imgs) -> list[Tensor], called under no_grad + autocast.
    """
    result: dict = {
        "latency_ms_mean": None, "latency_ms_std": None,
        "latency_ms_per_img": None, "peak_vram_mb": None, "status": "ok",
    }
    batch_size = imgs.shape[0]

    try:
        def _fwd():
            with torch.no_grad(), \
                 torch.autocast(device_type=device.type, enabled=amp):
                return encode_fn(encoder, imgs)

        for _ in range(n_warmup):
            _fwd()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        _fwd()   # one clean pass for VRAM

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

        times: list[float] = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _fwd()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1e3)

        mean_t = sum(times) / len(times)
        std_t  = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        result["latency_ms_mean"]    = mean_t
        result["latency_ms_std"]     = std_t
        result["latency_ms_per_img"] = mean_t / batch_size

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except Exception as exc:
        result["status"] = f"ERROR: {exc}"
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return result


def measure_training(
    encode_fn: Callable,
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
    n_warmup: int = 3,
    n_runs: int = 10,
    amp: bool = True,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Measure training (fwd + bwd) latency and peak VRAM.

    encode_fn: (encoder, imgs) -> list[Tensor].
    Uses lr=0 — parameter values never change; this is a throughput benchmark.
    """
    result: dict = {
        "latency_ms_mean": None, "latency_ms_std": None,
        "latency_ms_per_img": None, "peak_vram_mb": None, "status": "ok",
    }
    batch_size = imgs.shape[0]

    trainable = [p for p in encoder.parameters() if p.requires_grad]
    optimizer  = torch.optim.SGD(trainable, lr=0.0)
    scaler     = (torch.amp.GradScaler("cuda")
                  if (amp and device.type == "cuda") else None)

    try:
        def _fwd_bwd():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                feats = encode_fn(encoder, imgs)
                loss  = sum(f.mean() for f in feats)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

        for _ in range(n_warmup):
            _fwd_bwd()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        _fwd_bwd()   # one clean pass for VRAM

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

        times: list[float] = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _fwd_bwd()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1e3)

        mean_t = sum(times) / len(times)
        std_t  = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        result["latency_ms_mean"]    = mean_t
        result["latency_ms_std"]     = std_t
        result["latency_ms_per_img"] = mean_t / batch_size

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except Exception as exc:
        result["status"] = f"ERROR: {exc}"
    finally:
        del optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f} ms"


def _mb(x: float | None) -> str:
    if x is None:
        return "—"
    if x >= 1000:
        return f"{x/1000:.2f} GB"
    if x >= 1:
        return f"{x:.1f} MB"
    return f"{x*1000:.0f} KB"


def _speedup(val: float | None, baseline: float | None) -> str:
    if val is None or baseline is None or val == 0:
        return "—"
    return f"{baseline / val:.2f}×"


def _vram_delta(val: float | None, baseline: float | None) -> str:
    if val is None or baseline is None:
        return "—"
    delta = val - baseline
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{_mb(abs(delta))}" if delta >= 0 else f"-{_mb(abs(delta))}"


def print_comparison_table(
    rows: list[dict],
    mode: str,
    variant: str,
    img_size: int,
    batch_size: int,
    amp: bool,
) -> None:
    """Print a per-config method comparison table."""
    prec  = "fp16" if amp else "fp32"
    title = (f"STUNet-{variant}  {img_size}³  B={batch_size}  "
             f"{prec}  [{mode.upper()}]")
    print(f"\n{'─'*78}")
    print(f"  {title}")
    print(f"{'─'*78}")

    baseline_lat  = next(
        (r["latency_ms_mean"] for r in rows
         if r["method"] == "baseline" and r["status"] == "ok"), None
    )
    baseline_vram = next(
        (r["peak_vram_mb"] for r in rows
         if r["method"] == "baseline" and r["status"] == "ok"), None
    )

    print(f"  {'method':<22}  {'latency':>16}  {'/ img':>8}  "
          f"{'speedup':>8}  {'peak_vram':>10}  {'ΔVRAM':>10}")
    print(f"  {'─'*22}  {'─'*16}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")

    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['method']:<22}  ✗  {r['status']}")
            continue
        lat_str = (f"{r['latency_ms_mean']:.1f}±{r['latency_ms_std']:.1f} ms"
                   if r.get("latency_ms_std") is not None
                   else _ms(r["latency_ms_mean"]))
        ct_str  = (f"  [compile {r['compile_time_s']:.0f}s]"
                   if r.get("compile_time_s") else "")
        print(
            f"  {r['method']:<22}  {lat_str:>16}  "
            f"{_ms(r['latency_ms_per_img']):>8}  "
            f"{_speedup(r['latency_ms_mean'], baseline_lat):>8}  "
            f"{_mb(r['peak_vram_mb']):>10}  "
            f"{_vram_delta(r['peak_vram_mb'], baseline_vram):>10}"
            f"{ct_str}"
        )


def print_sweep_summary(all_rows: list[dict]) -> None:
    """Compact multi-config summary across all methods and configs."""
    if not all_rows:
        return

    hdr = (f"  {'mode':>9}  {'method':<22}  {'img':>5}  {'B':>2}  "
           f"{'t/img':>8}  {'speedup':>8}  {'peak_vram':>10}  {'ΔVRAM':>10}")
    width = len(hdr) - 2
    print(f"\n{'═'*width}")
    print("SWEEP SUMMARY")
    print(f"{'═'*width}")
    print(hdr)
    print(f"  {'─'*9}  {'─'*22}  {'─'*5}  {'─'*2}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")

    from itertools import groupby
    key_fn = lambda r: (r["mode"], r["img_size"], r["batch_size"])
    for _, grp_iter in groupby(sorted(all_rows, key=key_fn), key=key_fn):
        grp = list(grp_iter)
        bl  = next((r for r in grp if r["method"] == "baseline"
                    and r["status"] == "ok"), None)
        bl_lat  = bl["latency_ms_per_img"] if bl else None
        bl_vram = bl["peak_vram_mb"]       if bl else None
        for r in grp:
            if r["status"] != "ok":
                print(f"  {r['mode']:>9}  {r['method']:<22}  "
                      f"{r['img_size']:>5}  {r['batch_size']:>2}  "
                      f"✗ {r['status'][:40]}")
                continue
            print(
                f"  {r['mode']:>9}  {r['method']:<22}  "
                f"{r['img_size']:>5}  {r['batch_size']:>2}  "
                f"{_ms(r['latency_ms_per_img']):>8}  "
                f"{_speedup(r['latency_ms_per_img'], bl_lat):>8}  "
                f"{_mb(r['peak_vram_mb']):>10}  "
                f"{_vram_delta(r['peak_vram_mb'], bl_vram):>10}"
            )
    print(f"{'═'*width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--variant",      default="base",
                        choices=list(_VARIANTS))
    parser.add_argument("--image_sizes",  nargs="+", type=int, default=[64, 128])
    parser.add_argument("--batch_sizes",  nargs="+", type=int, default=[1])
    parser.add_argument("--methods",      nargs="+", default=sorted(_ALL_METHODS),
                        choices=sorted(_ALL_METHODS),
                        metavar="METHOD")
    parser.add_argument("--modes",        nargs="+", default=["inference", "training"],
                        choices=["inference", "training"])
    parser.add_argument("--n_runs",       type=int, default=10)
    parser.add_argument("--n_warmup",     type=int, default=3)
    parser.add_argument("--no_amp",       action="store_true",
                        help="Run in fp32 instead of fp16 autocast")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    amp        = not args.no_amp

    print(f"Device  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU     : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    print(f"AMP     : {'fp16 autocast' if amp else 'fp32'}")
    print(f"Runs    : {args.n_warmup} warmup + {args.n_runs} measured")
    print(f"Methods : {args.methods}")
    print(f"Modes   : {args.modes}\n")

    inf_methods   = [m for m in args.methods if m in _INFERENCE_METHODS]
    train_methods = [m for m in args.methods if m in _TRAINING_METHODS]

    all_rows: list[dict] = []

    # ── INFERENCE ──────────────────────────────────────────────────────────
    if "inference" in args.modes and inf_methods:
        print(f"Building STUNet-{args.variant} (frozen, eval) …",
              end="  ", flush=True)
        enc_inf = STUNetEncoder(
            in_channels=1, variant=args.variant, freeze_encoder=True,
        ).to(device).eval()
        print(f"{sum(p.numel() for p in enc_inf.parameters())/1e6:.1f} M params\n")

        for img_size in args.image_sizes:
            for batch_size in args.batch_sizes:
                imgs = torch.randn(
                    batch_size, 1, img_size, img_size, img_size,
                    device=device,
                    dtype=torch.float16 if amp else torch.float32,
                )
                config_rows: list[dict] = []

                def _add(method: str, result: dict, compile_time_s: float | None = None):
                    row = {
                        "mode": "inference", "method": method,
                        "img_size": f"{img_size}³", "batch_size": batch_size,
                        **result,
                    }
                    if compile_time_s is not None:
                        row["compile_time_s"] = compile_time_s
                    config_rows.append(row)
                    all_rows.append(row)

                if "baseline" in inf_methods:
                    _add("baseline", measure_inference(
                        encode_image_only, enc_inf, imgs,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    ))

                for method, mode_str in [
                    ("compile_reduce",   "reduce-overhead"),
                    ("compile_autotune", "max-autotune"),
                ]:
                    if method not in inf_methods:
                        continue
                    try:
                        compiled_enc, enc_fn, ct = compile_encoder(
                            enc_inf, mode_str, encode_image_only,
                            imgs, amp, device,
                        )
                        _add(method, measure_inference(
                            enc_fn, compiled_enc, imgs,
                            n_warmup=args.n_warmup, n_runs=args.n_runs,
                            amp=amp, device=device,
                        ), compile_time_s=ct)
                    except Exception as exc:
                        _add(method, {"status": f"ERROR: {exc}",
                                      "latency_ms_mean": None, "latency_ms_std": None,
                                      "latency_ms_per_img": None, "peak_vram_mb": None})
                    finally:
                        if "compiled_enc" in dir():
                            del compiled_enc
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()

                if "cuda_graph" in inf_methods:
                    try:
                        graph_encode = build_cuda_graph_encode(
                            enc_inf, imgs, amp, device, n_warmup=args.n_warmup,
                        )
                        _add("cuda_graph", measure_inference(
                            graph_encode, enc_inf, imgs,
                            n_warmup=0, n_runs=args.n_runs,  # graph is already warm
                            amp=amp, device=device,
                        ))
                    except Exception as exc:
                        _add("cuda_graph", {"status": f"GRAPH_ERROR: {exc}",
                                            "latency_ms_mean": None, "latency_ms_std": None,
                                            "latency_ms_per_img": None, "peak_vram_mb": None})

                if "vmap" in inf_methods:
                    try:
                        vmap_encode = make_vmap_encode(enc_inf, amp=amp, device=device)
                        _add("vmap (fp32)", measure_inference(
                            vmap_encode, enc_inf, imgs,
                            n_warmup=args.n_warmup, n_runs=args.n_runs,
                            amp=amp, device=device,
                        ))
                    except Exception as exc:
                        _add("vmap (fp32)", {"status": f"VMAP_ERROR: {exc}",
                                      "latency_ms_mean": None, "latency_ms_std": None,
                                      "latency_ms_per_img": None, "peak_vram_mb": None})

                print_comparison_table(
                    config_rows, "inference",
                    args.variant, img_size, batch_size, amp,
                )

        del enc_inf
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ── TRAINING ────────────────────────────────────────────────────────────
    if "training" in args.modes and train_methods:
        print(f"\nBuilding STUNet-{args.variant} (trainable) …",
              end="  ", flush=True)
        enc_train = STUNetEncoder(
            in_channels=1, variant=args.variant, freeze_encoder=False,
        ).to(device).train()
        print(f"{sum(p.numel() for p in enc_train.parameters())/1e6:.1f} M params\n")

        for img_size in args.image_sizes:
            for batch_size in args.batch_sizes:
                imgs_t = torch.randn(
                    batch_size, 1, img_size, img_size, img_size,
                    device=device,
                    dtype=torch.float16 if amp else torch.float32,
                )
                config_rows_t: list[dict] = []

                def _add_t(method: str, result: dict, compile_time_s: float | None = None):
                    row = {
                        "mode": "training", "method": method,
                        "img_size": f"{img_size}³", "batch_size": batch_size,
                        **result,
                    }
                    if compile_time_s is not None:
                        row["compile_time_s"] = compile_time_s
                    config_rows_t.append(row)
                    all_rows.append(row)

                if "baseline" in train_methods:
                    _add_t("baseline", measure_training(
                        encode_image_only, enc_train, imgs_t,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    ))

                if "compile" in train_methods:
                    try:
                        # reduce-overhead is more stable than max-autotune for 3D conv
                        # training on Python 3.12 (avoids Triton PY_SSIZE_T_CLEAN bug)
                        compiled_enc, enc_fn, ct = compile_encoder(
                            enc_train, "reduce-overhead", encode_image_only,
                            imgs_t, amp, device,
                        )
                        _add_t("compile", measure_training(
                            enc_fn, compiled_enc, imgs_t,
                            n_warmup=args.n_warmup, n_runs=args.n_runs,
                            amp=amp, device=device,
                        ), compile_time_s=ct)
                    except Exception as exc:
                        _add_t("compile", {"status": f"ERROR: {exc}",
                                           "latency_ms_mean": None, "latency_ms_std": None,
                                           "latency_ms_per_img": None, "peak_vram_mb": None})
                    finally:
                        if "compiled_enc" in dir():
                            del compiled_enc
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()

                if "checkpoint" in train_methods:
                    _add_t("checkpoint", measure_training(
                        checkpointed_encode, enc_train, imgs_t,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    ))

                if "compile_checkpoint" in train_methods:
                    try:
                        # Use reduce-overhead: max-autotune Triton kernels for 3-D conv
                        # fail on Python 3.12 (PY_SSIZE_T_CLEAN). suppress_errors lets
                        # those ops fall back to eager while the rest stay compiled.
                        _ckpt_mode = "reduce-overhead"
                        print(f"  torch.compile(mode={_ckpt_mode!r}) + checkpoint …",
                              end="  ", flush=True)
                        compiled_ckpt = torch.compile(
                            checkpointed_encode, mode=_ckpt_mode
                        )
                        t0 = time.perf_counter()
                        with torch.autocast(device_type=device.type, enabled=amp):
                            _ = compiled_ckpt(enc_train, imgs_t)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        ct = time.perf_counter() - t0
                        print(f"done in {ct:.1f} s")
                        _add_t("compile_checkpoint", measure_training(
                            compiled_ckpt, enc_train, imgs_t,
                            n_warmup=args.n_warmup, n_runs=args.n_runs,
                            amp=amp, device=device,
                        ), compile_time_s=ct)
                    except Exception as exc:
                        _add_t("compile_checkpoint",
                               {"status": f"ERROR: {exc}",
                                "latency_ms_mean": None, "latency_ms_std": None,
                                "latency_ms_per_img": None, "peak_vram_mb": None})

                print_comparison_table(
                    config_rows_t, "training",
                    args.variant, img_size, batch_size, amp,
                )

    # ── SWEEP SUMMARY ───────────────────────────────────────────────────────
    print_sweep_summary(all_rows)


if __name__ == "__main__":
    main()

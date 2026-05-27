# Encoder Optimization Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `experiments/encoders/benchmark_optimizations.py` that measures the latency and VRAM impact of torch.compile, CUDA graphs, torch.vmap, and gradient checkpointing on the STU-Net 3D image encoder across both inference and training paths.

**Architecture:** Single standalone script with one shared `measure_inference()` and one `measure_training()` core function — all optimization techniques are expressed as interchangeable `encode_fn` callables `(encoder, imgs) → list[Tensor]`. The main loop sweeps method × config × mode and prints a unified comparison table.

**Tech Stack:** PyTorch 2.6, CUDA 12.4, `torch.compile`, `torch.func.vmap`, `torch.utils.checkpoint`, `torch.cuda.CUDAGraph`, `torch.amp.GradScaler`. Uses `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-05-27-encoder-optimization-benchmark-design.md`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `experiments/encoders/benchmark_optimizations.py` | Full benchmark script |
| **Read** (no modify) | `experiments/encoders/benchmark_encoder.py` | Reuse `encode_image_only` pattern |
| **Read** (no modify) | `src/models/encoders/stunet.py` | `STUNetEncoder`, `_VARIANTS`, `_ImageEncoder.conv_blocks_context` |

---

## Task 1: File Skeleton, Imports, and CLI

**Files:**
- Create: `experiments/encoders/benchmark_optimizations.py`

- [ ] **Step 1: Create the file with all imports and argparse**

```python
"""
Benchmark the impact of PyTorch optimization techniques on the STU-Net 3-D
image encoder.

Techniques tested
-----------------
  Inference path (torch.no_grad):
    baseline        — plain encode_image_only
    compile_reduce  — torch.compile(mode="reduce-overhead")  [includes CUDA graphs]
    compile_autotune— torch.compile(mode="max-autotune")
    cuda_graph      — manual CUDAGraph capture + replay
    vmap            — torch.func.vmap over batch dim

  Training path (fwd + bwd):
    baseline        — plain forward + backward
    compile         — torch.compile(mode="max-autotune") + backward
    checkpoint      — per-stage gradient checkpointing
    compile_checkpoint — both combined

Usage
-----
    python experiments/encoders/benchmark_optimizations.py
    python experiments/encoders/benchmark_optimizations.py \\
        --variant base --image_sizes 64 128 --batch_sizes 1 \\
        --methods baseline compile_reduce cuda_graph \\
        --modes inference training \\
        --n_runs 10 --n_warmup 3
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

# Methods that apply to each mode
_INFERENCE_METHODS = {"baseline", "compile_reduce", "compile_autotune",
                      "cuda_graph", "vmap"}
_TRAINING_METHODS  = {"baseline", "compile", "checkpoint", "compile_checkpoint"}
_ALL_METHODS       = _INFERENCE_METHODS | _TRAINING_METHODS


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
                        choices=sorted(_ALL_METHODS))
    parser.add_argument("--modes",        nargs="+", default=["inference", "training"],
                        choices=["inference", "training"])
    parser.add_argument("--n_runs",       type=int, default=10)
    parser.add_argument("--n_warmup",     type=int, default=3)
    parser.add_argument("--no_amp",       action="store_true")
    parser.add_argument("--device",       default=None)
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    amp        = not args.no_amp

    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU    : {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    print(f"AMP    : {'fp16' if amp else 'fp32'}")
    print(f"Runs   : {args.n_warmup} warmup + {args.n_runs} measured\n")
    print(f"Methods: {args.methods}")
    print(f"Modes  : {args.modes}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py --help
```

Expected: prints the docstring and all flags, exits 0.

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 32 --methods baseline --modes inference
```

Expected: prints device/GPU/AMP/runs info, exits 0.

---

## Task 2: `encode_image_only` Helper and `measure_inference` Core

**Files:**
- Modify: `experiments/encoders/benchmark_optimizations.py` (add before `main`)

- [ ] **Step 1: Add `encode_image_only` and `measure_inference`**

Add the following functions before `main()`:

```python
# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_only(
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """(B,1,D,H,W) → [skip0, …, bottleneck]. Mirrors benchmark_encoder.py."""
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


# ---------------------------------------------------------------------------
# Core inference measurement  (shared by all inference methods)
# ---------------------------------------------------------------------------

def measure_inference(
    encode_fn:  Callable,
    encoder:    STUNetEncoder,
    imgs:       torch.Tensor,
    n_warmup:   int  = 3,
    n_runs:     int  = 10,
    amp:        bool = True,
    device:     torch.device = torch.device("cuda"),
) -> dict:
    """Run encode_fn under no_grad, return latency + VRAM stats.

    encode_fn signature: (encoder, imgs) -> list[Tensor]
    """
    result: dict = {
        "latency_ms_mean":  None,
        "latency_ms_std":   None,
        "latency_ms_per_img": None,
        "peak_vram_mb":     None,
        "status":           "ok",
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

        _fwd()   # one pass for VRAM measurement

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
```

- [ ] **Step 2: Wire baseline inference into main and smoke-test**

Inside `main()`, after the info prints, add:

```python
    # --- build encoder (inference: freeze=True) ---
    if "inference" in args.modes:
        print(f"Building STUNet-{args.variant} (frozen) …", end="  ", flush=True)
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
                if "baseline" in args.methods:
                    r = measure_inference(
                        encode_image_only, enc_inf, imgs,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    )
                    print(f"[inference/baseline  {img_size}³ B={batch_size}] "
                          f"status={r['status']}  "
                          f"t={r['latency_ms_mean']:.1f}ms  "
                          f"vram={r['peak_vram_mb']:.0f}MB")
```

Run:
```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 64 --batch_sizes 1 \
    --methods baseline --modes inference \
    --n_runs 3 --n_warmup 1
```

Expected: prints one `[inference/baseline 64³ B=1] status=ok  t=…ms  vram=…MB` line.

---

## Task 3: Compile Inference Methods (`compile_reduce`, `compile_autotune`)

**Files:**
- Modify: `experiments/encoders/benchmark_optimizations.py`

- [ ] **Step 1: Add `compile_encoder` helper**

Add before `main()`:

```python
def compile_encoder(
    encoder: STUNetEncoder,
    mode: str,
    encode_fn: Callable,
    imgs: torch.Tensor,
    amp: bool,
    device: torch.device,
) -> tuple[STUNetEncoder, Callable, float]:
    """torch.compile the encoder, trigger first-call compilation, return
    (compiled_encoder, encode_fn_that_uses_it, compile_time_seconds).

    The encode_fn is unchanged — it is called with the compiled encoder.
    First-call compilation happens here so timing runs don't include it.
    """
    print(f"  torch.compile(mode={mode!r}) … compiling (may take minutes) …",
          end="  ", flush=True)
    compiled = torch.compile(encoder, mode=mode)

    t0 = time.perf_counter()
    with torch.no_grad(), \
         torch.autocast(device_type=device.type, enabled=amp):
        _ = encode_fn(compiled, imgs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compile_time = time.perf_counter() - t0
    print(f"done in {compile_time:.1f} s")
    return compiled, encode_fn, compile_time
```

- [ ] **Step 2: Add compile methods to the inference block in `main()`**

Inside the `for img_size / for batch_size` loop, after the baseline block:

```python
                compile_times: dict[str, float] = {}

                for method, mode_str in [
                    ("compile_reduce",   "reduce-overhead"),
                    ("compile_autotune", "max-autotune"),
                ]:
                    if method not in args.methods:
                        continue
                    compiled_enc, enc_fn, ct = compile_encoder(
                        enc_inf, mode_str, encode_image_only,
                        imgs, amp, device,
                    )
                    compile_times[method] = ct
                    r = measure_inference(
                        enc_fn, compiled_enc, imgs,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    )
                    print(f"[inference/{method:<16} {img_size}³ B={batch_size}] "
                          f"status={r['status']}  "
                          f"t={r['latency_ms_mean']:.1f}ms  "
                          f"vram={r['peak_vram_mb']:.0f}MB  "
                          f"compile={ct:.1f}s")
                    del compiled_enc
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
```

- [ ] **Step 3: Smoke-test compile_reduce**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 64 --batch_sizes 1 \
    --methods baseline compile_reduce --modes inference \
    --n_runs 3 --n_warmup 1
```

Expected: prints baseline line, then compile line (compile_reduce typically takes 30-90 s). Both `status=ok`.

---

## Task 4: CUDA Graph Inference Method

**Files:**
- Modify: `experiments/encoders/benchmark_optimizations.py`

- [ ] **Step 1: Add `build_cuda_graph_encode` factory**

Add before `main()`:

```python
def build_cuda_graph_encode(
    encoder: STUNetEncoder,
    imgs:    torch.Tensor,
    amp:     bool,
    device:  torch.device,
    n_warmup: int = 3,
) -> Callable:
    """Capture a CUDAGraph for encode_image_only and return a replay callable.

    The callable has signature (encoder, imgs) -> list[Tensor] to match
    measure_inference — the encoder/imgs arguments are ignored at replay time
    (the graph uses static buffers). imgs must have the same shape each call.

    Returns None on non-CUDA devices or if capture fails.
    """
    if device.type != "cuda":
        raise RuntimeError("CUDA graphs require a CUDA device.")

    # Static input buffer (graph will replay against this)
    static_input = imgs.clone()

    # Warmup outside the graph (required before capture)
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            with torch.no_grad(), \
                 torch.autocast(device_type="cuda", enabled=amp):
                _ = encode_image_only(encoder, static_input)
    torch.cuda.current_stream(device).wait_stream(stream)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        with torch.no_grad(), \
             torch.autocast(device_type="cuda", enabled=amp):
            static_output = encode_image_only(encoder, static_input)

    def _replay(_encoder, _imgs):
        """Copy real input into static buffer, replay graph."""
        static_input.copy_(_imgs)
        g.replay()
        return static_output   # list of tensors in static GPU memory

    return _replay
```

- [ ] **Step 2: Add cuda_graph to the inference block in `main()`**

After the compile methods block, inside the same `for img_size / for batch_size` loop:

```python
                if "cuda_graph" in args.methods:
                    try:
                        graph_encode = build_cuda_graph_encode(
                            enc_inf, imgs, amp, device,
                            n_warmup=args.n_warmup,
                        )
                        r = measure_inference(
                            graph_encode, enc_inf, imgs,
                            n_warmup=0,          # graph is already warm
                            n_runs=args.n_runs,
                            amp=amp, device=device,
                        )
                        print(f"[inference/cuda_graph      {img_size}³ B={batch_size}] "
                              f"status={r['status']}  "
                              f"t={r['latency_ms_mean']:.1f}ms  "
                              f"vram={r['peak_vram_mb']:.0f}MB")
                    except Exception as exc:
                        print(f"[inference/cuda_graph      {img_size}³ B={batch_size}] "
                              f"GRAPH_ERROR: {exc}")
```

- [ ] **Step 3: Smoke-test cuda_graph**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 64 --batch_sizes 1 \
    --methods baseline cuda_graph --modes inference \
    --n_runs 3 --n_warmup 1
```

Expected: both lines `status=ok`. CUDA graph latency should be ≤ baseline (typically 1.3–1.7× faster at B=1).

---

## Task 5: `torch.func.vmap` Inference Method

**Files:**
- Modify: `experiments/encoders/benchmark_optimizations.py`

- [ ] **Step 1: Add `make_vmap_encode` factory**

Add before `main()`:

```python
def make_vmap_encode(encoder: STUNetEncoder) -> Callable:
    """Return a vmap-batched encode function.

    Maps encode_image_only over the batch dim: each call sees a single image
    (1, D, H, W), unsqueezes to (1, 1, D, H, W), runs the encoder, and
    returns a tuple of feature tensors (without the batch dim).
    torch.func.vmap stacks them back into a (B, …) batch.

    The returned callable has the same signature as encode_image_only:
        (encoder, imgs) -> list[Tensor]
    """
    def _single(img: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # img: (1, D, H, W)  →  unsqueeze → (1, 1, D, H, W)
        feats = encode_image_only(encoder, img.unsqueeze(0))
        # squeeze batch dim back out for vmap stacking
        return tuple(f.squeeze(0) for f in feats)

    _vmapped = torch.func.vmap(_single, in_dims=0)

    def vmap_encode(_encoder: STUNetEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
        # imgs: (B, 1, D, H, W)
        result = _vmapped(imgs)       # tuple of (B, C_i, D_i, H_i, W_i)
        return list(result)

    return vmap_encode
```

- [ ] **Step 2: Add vmap to the inference block in `main()`**

After the cuda_graph block:

```python
                if "vmap" in args.methods:
                    try:
                        vmap_encode = make_vmap_encode(enc_inf)
                        r = measure_inference(
                            vmap_encode, enc_inf, imgs,
                            n_warmup=args.n_warmup, n_runs=args.n_runs,
                            amp=amp, device=device,
                        )
                        print(f"[inference/vmap            {img_size}³ B={batch_size}] "
                              f"status={r['status']}  "
                              f"t={r['latency_ms_mean']:.1f}ms  "
                              f"vram={r['peak_vram_mb']:.0f}MB")
                    except Exception as exc:
                        print(f"[inference/vmap            {img_size}³ B={batch_size}] "
                              f"VMAP_ERROR: {exc}")
```

- [ ] **Step 3: Smoke-test vmap**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 64 --batch_sizes 1 4 \
    --methods baseline vmap --modes inference \
    --n_runs 3 --n_warmup 1
```

Expected: both methods run for both batch sizes, `status=ok`. At B=1 vmap ≈ baseline; at B=4 vmap may be slightly slower (overhead) but within ~20%.

---

## Task 6: Training Measurement Core + Baseline + Compile

**Files:**
- Modify: `experiments/encoders/benchmark_optimizations.py`

- [ ] **Step 1: Add `measure_training` core function**

Add before `main()`:

```python
# ---------------------------------------------------------------------------
# Core training measurement  (shared by all training methods)
# ---------------------------------------------------------------------------

def measure_training(
    encode_fn:  Callable,
    encoder:    STUNetEncoder,
    imgs:       torch.Tensor,
    n_warmup:   int  = 3,
    n_runs:     int  = 10,
    amp:        bool = True,
    device:     torch.device = torch.device("cuda"),
) -> dict:
    """Run encode_fn + scalar backward, return latency + VRAM stats.

    encode_fn signature: (encoder, imgs) -> list[Tensor]
    The optimizer uses lr=0 so parameter values never change — this is a
    pure memory/throughput benchmark, not actual training.
    """
    result: dict = {
        "latency_ms_mean":    None,
        "latency_ms_std":     None,
        "latency_ms_per_img": None,
        "peak_vram_mb":       None,
        "status":             "ok",
    }
    batch_size = imgs.shape[0]

    trainable = [p for p in encoder.parameters() if p.requires_grad]
    optimizer  = torch.optim.SGD(trainable, lr=0.0)
    scaler     = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") \
                 else None

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

        _fwd_bwd()   # one pass for VRAM measurement

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
```

- [ ] **Step 2: Add training block to `main()`**

After the inference block (still inside `main()`), add:

```python
    # --- build encoder (training: freeze=False) ---
    if "training" in args.modes:
        print(f"\nBuilding STUNet-{args.variant} (trainable) …", end="  ", flush=True)
        enc_train = STUNetEncoder(
            in_channels=1, variant=args.variant, freeze_encoder=False,
        ).to(device).train()
        print(f"{sum(p.numel() for p in enc_train.parameters())/1e6:.1f} M params\n")

        for img_size in args.image_sizes:
            for batch_size in args.batch_sizes:
                imgs_train = torch.randn(
                    batch_size, 1, img_size, img_size, img_size,
                    device=device,
                    dtype=torch.float16 if amp else torch.float32,
                )

                if "baseline" in args.methods:
                    r = measure_training(
                        encode_image_only, enc_train, imgs_train,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    )
                    print(f"[training/baseline         {img_size}³ B={batch_size}] "
                          f"status={r['status']}  "
                          f"t={r['latency_ms_mean']:.1f}ms  "
                          f"vram={r['peak_vram_mb']:.0f}MB")

                if "compile" in args.methods:
                    compiled_enc, enc_fn, ct = compile_encoder(
                        enc_train, "max-autotune", encode_image_only,
                        imgs_train, amp, device,
                    )
                    r = measure_training(
                        enc_fn, compiled_enc, imgs_train,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    )
                    print(f"[training/compile          {img_size}³ B={batch_size}] "
                          f"status={r['status']}  "
                          f"t={r['latency_ms_mean']:.1f}ms  "
                          f"vram={r['peak_vram_mb']:.0f}MB  "
                          f"compile={ct:.1f}s")
                    del compiled_enc
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
```

- [ ] **Step 3: Smoke-test training baseline**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 64 --batch_sizes 1 \
    --methods baseline --modes training \
    --n_runs 3 --n_warmup 1
```

Expected: one training/baseline line, `status=ok`.

---

## Task 7: Gradient Checkpointing + `compile_checkpoint` Training Methods

**Files:**
- Modify: `experiments/encoders/benchmark_optimizations.py`

- [ ] **Step 1: Add `checkpointed_encode` function**

Add before `main()`:

```python
def checkpointed_encode(
    encoder: STUNetEncoder,
    imgs:    torch.Tensor,
) -> list[torch.Tensor]:
    """Per-stage gradient checkpointing on the image encoder.

    Wraps each conv_blocks_context[i] call with torch.utils.checkpoint so
    that intermediate activations are recomputed on backward rather than
    stored. Reduces training VRAM at the cost of ~20-40% extra compute.
    use_reentrant=False is required for compatibility with torch.compile.
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
```

- [ ] **Step 2: Add checkpoint and compile_checkpoint to the training block in `main()`**

After the `compile` block inside the `for img_size / for batch_size` loop:

```python
                if "checkpoint" in args.methods:
                    r = measure_training(
                        checkpointed_encode, enc_train, imgs_train,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    )
                    print(f"[training/checkpoint       {img_size}³ B={batch_size}] "
                          f"status={r['status']}  "
                          f"t={r['latency_ms_mean']:.1f}ms  "
                          f"vram={r['peak_vram_mb']:.0f}MB")

                if "compile_checkpoint" in args.methods:
                    print(f"  torch.compile(mode='max-autotune') + checkpoint … ",
                          end="", flush=True)
                    compiled_ckpt = torch.compile(
                        checkpointed_encode, mode="max-autotune"
                    )
                    # Trigger compilation with one warmup call
                    t0 = time.perf_counter()
                    with torch.autocast(device_type=device.type, enabled=amp):
                        _ = compiled_ckpt(enc_train, imgs_train)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    ct = time.perf_counter() - t0
                    print(f"done in {ct:.1f} s")

                    r = measure_training(
                        compiled_ckpt, enc_train, imgs_train,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    )
                    print(f"[training/compile_checkpoint {img_size}³ B={batch_size}] "
                          f"status={r['status']}  "
                          f"t={r['latency_ms_mean']:.1f}ms  "
                          f"vram={r['peak_vram_mb']:.0f}MB  "
                          f"compile={ct:.1f}s")
```

- [ ] **Step 3: Smoke-test checkpoint**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 64 --batch_sizes 1 \
    --methods baseline checkpoint --modes training \
    --n_runs 3 --n_warmup 1
```

Expected: both lines `status=ok`. Checkpoint VRAM should be noticeably less than baseline (target: ~50% reduction). Latency should be ~20-40% higher.

---

## Task 8: Output Formatting — Comparison Tables

**Files:**
- Modify: `experiments/encoders/benchmark_optimizations.py`

This task replaces the inline `print(f"[method …]")` calls in Tasks 2–7 with a structured result collector and proper formatted tables.

- [ ] **Step 1: Add formatting helpers before `main()`**

```python
# ---------------------------------------------------------------------------
# Formatting helpers
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
    return f"{sign}{_mb(delta)}"


def print_comparison_table(
    rows:       list[dict],
    mode:       str,
    variant:    str,
    img_size:   int,
    batch_size: int,
    amp:        bool,
) -> None:
    """Print a per-config method comparison table.

    Each row dict must have keys:
        method, status, latency_ms_mean, latency_ms_std,
        latency_ms_per_img, peak_vram_mb, compile_time_s (optional)
    """
    prec  = "fp16" if amp else "fp32"
    title = (f"STUNet-{variant}  {img_size}³  B={batch_size}  "
             f"{prec}  [{mode.upper()}]")
    print(f"\n{'─'*76}")
    print(f"  {title}")
    print(f"{'─'*76}")

    baseline_lat  = next(
        (r["latency_ms_mean"] for r in rows if r["method"] == "baseline"
         and r["status"] == "ok"), None
    )
    baseline_vram = next(
        (r["peak_vram_mb"] for r in rows if r["method"] == "baseline"
         and r["status"] == "ok"), None
    )

    hdr = (f"  {'method':<20}  {'latency':>12}  {'/ img':>8}  "
           f"{'speedup':>8}  {'peak_vram':>10}  {'ΔVRAM':>10}")
    print(hdr)
    print(f"  {'─'*20}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")

    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['method']:<20}  ✗  {r['status']}")
            continue
        lat_str = (f"{r['latency_ms_mean']:.1f} ±{r['latency_ms_std']:.1f} ms"
                   if r['latency_ms_std'] is not None
                   else _ms(r['latency_ms_mean']))
        ct_str = (f"  [compile: {r['compile_time_s']:.0f}s]"
                  if r.get("compile_time_s") else "")
        print(
            f"  {r['method']:<20}  {lat_str:>12}  "
            f"{_ms(r['latency_ms_per_img']):>8}  "
            f"{_speedup(r['latency_ms_mean'], baseline_lat):>8}  "
            f"{_mb(r['peak_vram_mb']):>10}  "
            f"{_vram_delta(r['peak_vram_mb'], baseline_vram):>10}"
            f"{ct_str}"
        )


def print_sweep_summary(all_rows: list[dict]) -> None:
    """Print a compact multi-row sweep across all configs."""
    ok = [r for r in all_rows if r["status"] == "ok"]
    if not ok:
        return

    hdr = (f"  {'mode':>9}  {'method':<20}  {'img':>5}  {'B':>2}  "
           f"{'t/img':>8}  {'speedup':>8}  {'peak_vram':>10}  {'ΔVRAM':>10}")
    width = len(hdr) - 2
    print(f"\n{'═'*width}")
    print("SWEEP SUMMARY")
    print(f"{'═'*width}")
    print(hdr)
    print(f"  {'─'*9}  {'─'*20}  {'─'*5}  {'─'*2}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")

    # group by (mode, img_size, batch_size) to compute per-group speedup
    from itertools import groupby
    key = lambda r: (r["mode"], r["img_size"], r["batch_size"])
    for _, group in groupby(sorted(all_rows, key=key), key=key):
        grp = list(group)
        bl  = next((r for r in grp if r["method"] == "baseline"
                    and r["status"] == "ok"), None)
        bl_lat  = bl["latency_ms_per_img"] if bl else None
        bl_vram = bl["peak_vram_mb"]       if bl else None
        for r in grp:
            if r["status"] != "ok":
                print(f"  {r['mode']:>9}  {r['method']:<20}  "
                      f"{r['img_size']:>5}  {r['batch_size']:>2}  ✗ {r['status'][:30]}")
                continue
            print(
                f"  {r['mode']:>9}  {r['method']:<20}  "
                f"{r['img_size']:>5}  {r['batch_size']:>2}  "
                f"{_ms(r['latency_ms_per_img']):>8}  "
                f"{_speedup(r['latency_ms_per_img'], bl_lat):>8}  "
                f"{_mb(r['peak_vram_mb']):>10}  "
                f"{_vram_delta(r['peak_vram_mb'], bl_vram):>10}"
            )
    print(f"{'═'*width}")
```

- [ ] **Step 2: Refactor the inline prints in `main()` to use the collector**

Replace all the individual `print(f"[mode/method …]")` statements in `main()` with a result-collecting pattern. Replace the full `main()` body (keeping the arg parsing + device setup unchanged) with:

```python
    all_rows: list[dict] = []

    # ── INFERENCE ──────────────────────────────────────────────────────────
    if "inference" in args.modes:
        print(f"Building STUNet-{args.variant} (frozen) …", end="  ", flush=True)
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

                def _add(method, result, compile_time_s=None):
                    row = {
                        "mode": "inference", "method": method,
                        "img_size": f"{img_size}³", "batch_size": batch_size,
                        **result,
                    }
                    if compile_time_s is not None:
                        row["compile_time_s"] = compile_time_s
                    config_rows.append(row)
                    all_rows.append(row)

                if "baseline" in args.methods:
                    _add("baseline", measure_inference(
                        encode_image_only, enc_inf, imgs,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    ))

                for method, mode_str in [
                    ("compile_reduce",   "reduce-overhead"),
                    ("compile_autotune", "max-autotune"),
                ]:
                    if method not in args.methods:
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
                        del compiled_enc
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as exc:
                        _add(method, {"status": f"ERROR: {exc}",
                                      "latency_ms_mean": None,
                                      "latency_ms_std": None,
                                      "latency_ms_per_img": None,
                                      "peak_vram_mb": None})

                if "cuda_graph" in args.methods:
                    try:
                        graph_encode = build_cuda_graph_encode(
                            enc_inf, imgs, amp, device,
                            n_warmup=args.n_warmup,
                        )
                        _add("cuda_graph", measure_inference(
                            graph_encode, enc_inf, imgs,
                            n_warmup=0, n_runs=args.n_runs,
                            amp=amp, device=device,
                        ))
                    except Exception as exc:
                        _add("cuda_graph", {"status": f"GRAPH_ERROR: {exc}",
                                            "latency_ms_mean": None,
                                            "latency_ms_std": None,
                                            "latency_ms_per_img": None,
                                            "peak_vram_mb": None})

                if "vmap" in args.methods:
                    try:
                        vmap_encode = make_vmap_encode(enc_inf)
                        _add("vmap", measure_inference(
                            vmap_encode, enc_inf, imgs,
                            n_warmup=args.n_warmup, n_runs=args.n_runs,
                            amp=amp, device=device,
                        ))
                    except Exception as exc:
                        _add("vmap", {"status": f"VMAP_ERROR: {exc}",
                                      "latency_ms_mean": None,
                                      "latency_ms_std": None,
                                      "latency_ms_per_img": None,
                                      "peak_vram_mb": None})

                print_comparison_table(
                    config_rows, "inference", args.variant,
                    img_size, batch_size, amp,
                )

        del enc_inf
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # ── TRAINING ────────────────────────────────────────────────────────────
    if "training" in args.modes:
        print(f"\nBuilding STUNet-{args.variant} (trainable) …", end="  ", flush=True)
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

                def _add_t(method, result, compile_time_s=None):
                    row = {
                        "mode": "training", "method": method,
                        "img_size": f"{img_size}³", "batch_size": batch_size,
                        **result,
                    }
                    if compile_time_s is not None:
                        row["compile_time_s"] = compile_time_s
                    config_rows_t.append(row)
                    all_rows.append(row)

                if "baseline" in args.methods:
                    _add_t("baseline", measure_training(
                        encode_image_only, enc_train, imgs_t,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    ))

                if "compile" in args.methods:
                    try:
                        compiled_enc, enc_fn, ct = compile_encoder(
                            enc_train, "max-autotune", encode_image_only,
                            imgs_t, amp, device,
                        )
                        _add_t("compile", measure_training(
                            enc_fn, compiled_enc, imgs_t,
                            n_warmup=args.n_warmup, n_runs=args.n_runs,
                            amp=amp, device=device,
                        ), compile_time_s=ct)
                        del compiled_enc
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as exc:
                        _add_t("compile", {"status": f"ERROR: {exc}",
                                           "latency_ms_mean": None,
                                           "latency_ms_std": None,
                                           "latency_ms_per_img": None,
                                           "peak_vram_mb": None})

                if "checkpoint" in args.methods:
                    _add_t("checkpoint", measure_training(
                        checkpointed_encode, enc_train, imgs_t,
                        n_warmup=args.n_warmup, n_runs=args.n_runs,
                        amp=amp, device=device,
                    ))

                if "compile_checkpoint" in args.methods:
                    try:
                        print("  torch.compile + checkpoint … compiling …",
                              end="  ", flush=True)
                        compiled_ckpt = torch.compile(
                            checkpointed_encode, mode="max-autotune"
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
                                "latency_ms_mean": None,
                                "latency_ms_std": None,
                                "latency_ms_per_img": None,
                                "peak_vram_mb": None})

                print_comparison_table(
                    config_rows_t, "training", args.variant,
                    img_size, batch_size, amp,
                )

    # ── SWEEP SUMMARY ───────────────────────────────────────────────────────
    print_sweep_summary(all_rows)
```

- [ ] **Step 3: Verify table formatting**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base --image_sizes 64 --batch_sizes 1 \
    --methods baseline checkpoint --modes inference training \
    --n_runs 3 --n_warmup 1
```

Expected output includes:
```
────────────────────────────────────────────────────────────────────────
  STUNet-base  64³  B=1  fp16  [INFERENCE]
────────────────────────────────────────────────────────────────────────
  method                  latency       / img   speedup   peak_vram       ΔVRAM
  baseline            …ms  1.00×  …GB     —
  …
SWEEP SUMMARY
═══════════════
```

---

## Task 9: End-to-End Integration Run

**Files:**
- No code changes — integration test only.

- [ ] **Step 1: Run a quick full sweep (skip compile_autotune for speed)**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base \
    --image_sizes 64 128 \
    --batch_sizes 1 \
    --methods baseline compile_reduce cuda_graph vmap checkpoint compile_checkpoint \
    --modes inference training \
    --n_runs 5 --n_warmup 2
```

Expected: all six methods print for each config, `status=ok` for all (except vmap/cuda_graph may show a note). Final sweep summary table is printed.

- [ ] **Step 2: Verify expected patterns in results**

Check the output manually against these expected patterns:
- `compile_reduce` speedup ≥ 1.3× over baseline (inference)
- `cuda_graph` speedup ≥ 1.2× over baseline at B=1 (inference)
- `checkpoint` VRAM ≤ 70% of baseline VRAM (training)
- `checkpoint` latency ≥ 1.1× of baseline (training, i.e. slower due to recompute)
- `vmap` speedup ≈ 1.0× at B=1 (no benefit expected at B=1)

If any of these are not observed, it indicates a measurement or implementation bug worth investigating (not necessarily a failure — the benchmark exists to reveal real numbers).

- [ ] **Step 3: Run with compile_autotune to complete the inference picture**

```bash
.venv/bin/python experiments/encoders/benchmark_optimizations.py \
    --variant base \
    --image_sizes 64 \
    --batch_sizes 1 \
    --methods baseline compile_autotune \
    --modes inference \
    --n_runs 10 --n_warmup 3
```

Expected: `compile_autotune` takes 2–10 min to compile (A6000 + PyTorch 2.6), then runs timed. Speedup should be ≥ `compile_reduce`.

- [ ] **Step 4: Log results to `docs/logs.md`**

Append a summary of the actual numbers observed from Step 1 to `docs/logs.md`. The entry format already used in that file should be followed. Include: date, variants tested, key findings (e.g. "compile_reduce: 2.1× faster, cuda_graph: 1.4× at B=1, checkpoint: 48% VRAM reduction").

---

## Self-Review Notes

- **Spec coverage:** All 5 inference methods covered (Tasks 2–5); all 4 training methods covered (Tasks 6–7); output format matches spec (Task 8); CLI matches spec (Task 1). Non-goals respected: `STUNetEncoder` source is unchanged, mask encoder not benchmarked.  
- **No placeholders:** All steps contain complete code or exact commands.  
- **Type consistency:** `encode_fn` signature is `(STUNetEncoder, Tensor) -> list[Tensor]` throughout. `measure_inference` / `measure_training` share identical return dict keys. `_add` / `_add_t` closures inject `mode`, `img_size`, `batch_size` consistently. `compile_encoder` returns `(compiled_enc, encode_fn, float)` — used correctly in Tasks 3, 6.  
- **No git commits** — omitted throughout per user preference.

# Encoder Optimization Benchmark Design

**Date:** 2026-05-27  
**Project:** patch_icl — in-context 3D medical image segmentation  
**Status:** Approved

---

## Overview

Add `experiments/encoders/benchmark_optimizations.py` — a standalone script that measures the impact of four PyTorch optimization techniques on the STU-Net 3D encoder:

1. `torch.compile` (modes: `reduce-overhead`, `max-autotune`)
2. CUDA graphs (manual capture + replay)
3. `torch.func.vmap` (vectorized batch processing)
4. Gradient checkpointing (`torch.utils.checkpoint`)

The existing `benchmark_encoder.py` (stage storage / VRAM breakdown) is **not modified**.

---

## Scope

- **GPU:** RTX A6000, PyTorch 2.6 / CUDA 12.4
- **Encoder:** `STUNetEncoder` from `src/models/encoders/stunet.py`
- **Both paths:** inference (no grad) and training (fwd + bwd)
- **Default config:** `variant=base`, `image_sizes=[64, 128]`, `batch_sizes=[1]`

---

## File

```
experiments/encoders/benchmark_optimizations.py   ← new
```

---

## Methods

### Inference path (`@torch.no_grad()`)

| ID | Method | Implementation |
|---|---|---|
| `baseline` | Plain forward | `encode_image_only(encoder, imgs)` |
| `compile_reduce` | `torch.compile` reduce-overhead | Includes CUDA graphs internally |
| `compile_autotune` | `torch.compile` max-autotune | Heaviest kernel fusion; long first-call |
| `cuda_graph` | Manual CUDA graph | `CUDAGraph` capture + `g.replay()`, static shapes |
| `vmap` | `torch.func.vmap` over batch | Single-image fwd vmapped across B |

### Training path (fwd + bwd, gradients enabled)

| ID | Method | Implementation |
|---|---|---|
| `baseline` | Plain fwd + bwd | `loss.backward()` |
| `compile` | `torch.compile` max-autotune | Applied before training loop |
| `checkpoint` | Per-stage gradient checkpointing | `checkpoint(stage, x, use_reentrant=False)` per conv block |
| `compile+checkpoint` | Both combined | compile wraps the checkpointed encode function |

---

## Key Implementation Details

### Gradient checkpointing
Wraps each stage of `_ImageEncoder.conv_blocks_context` at call time without modifying the class:

```python
def checkpointed_encode(encoder, imgs):
    from torch.utils.checkpoint import checkpoint
    n = encoder._num_stages
    x = imgs
    skips = []
    for stage in encoder.image_encoder.conv_blocks_context[:n - 1]:
        x = checkpoint(stage, x, use_reentrant=False)
        skips.append(x)
    x = checkpoint(encoder.image_encoder.conv_blocks_context[n - 1], x,
                   use_reentrant=False)
    return skips + [x]
```

### CUDA graphs
One graph per `(image_size, batch_size)` config. Warmup → capture → timed replays:

```python
g = torch.cuda.CUDAGraph()
static_in = imgs.clone()
with torch.cuda.graph(g):
    static_out = encode_image_only(encoder, static_in)
# each timed run: static_in.copy_(real_input); g.replay()
```

### torch.vmap
Since `encode_image_only` returns a list of tensors with varying spatial shapes, the vmapped function returns a `tuple`; vmap stacks each element over the batch dim automatically:

```python
def single_fwd(img):            # img: (1, D, H, W) — no batch dim
    feats = encode_image_only(encoder, img.unsqueeze(0))
    return tuple(f.squeeze(0) for f in feats)

vmapped = torch.func.vmap(single_fwd)
feats = vmapped(imgs)           # imgs: (B, 1, D, H, W)
```

### torch.compile
Applied to the full `STUNetEncoder` module. `max-autotune` compilation time (2–5 min) is measured and reported separately from inference latency.

### Training benchmark
Encoder rebuilt with `freeze_encoder=False`. Each training iteration:
```python
feats = encode_fn(encoder, imgs, masks)
loss  = sum(f.mean() for f in feats)
loss.backward()
optimizer.zero_grad()
```

---

## CLI

```bash
python experiments/encoders/benchmark_optimizations.py \
  --variant base \
  --image_sizes 64 128 \
  --batch_sizes 1 \
  --methods baseline compile_reduce compile_autotune cuda_graph vmap checkpoint \
  --modes inference training \
  --n_runs 10 --n_warmup 3 \
  --no_amp \          # optional: fp32
  --device cuda:0
```

Default methods: all. Default modes: both.

---

## Output Format

Per-config comparison block:
```
────────────────────────────────────────────────────────────────────────
STUNet-base  64³  B=1  fp16  [INFERENCE]
────────────────────────────────────────────────────────────────────────
  method              latency (ms)   /img    speedup   peak_vram   ΔVRAM
  baseline              42.3 ±0.8   42.3ms   1.00×    1.2 GB       —
  compile_reduce        18.1 ±0.3   18.1ms   2.34×    1.1 GB    -0.1 GB
  compile_autotune      16.9 ±0.2   16.9ms   2.50×    1.1 GB    -0.1 GB
  cuda_graph            31.2 ±0.1   31.2ms   1.36×    1.2 GB    +0.0 GB
  vmap                  44.1 ±0.9   44.1ms   0.96×    1.3 GB    +0.1 GB
```

Compile overhead reported separately:
```
[compile_reduce]   compilation time: 38.2 s  (one-time cost)
[compile_autotune] compilation time: 187.4 s
```

Final sweep summary table across all configs (variant × img_size × batch_size × mode).

---

## Metrics

| Metric | Description |
|---|---|
| `latency_ms_mean` | Wall-clock mean over `n_runs` (ms) |
| `latency_ms_std` | Std deviation (ms) |
| `latency_ms_per_img` | Mean / batch_size |
| `speedup` | baseline_latency / method_latency |
| `peak_vram_mb` | `torch.cuda.max_memory_allocated` after reset |
| `vram_delta_mb` | peak_vram − baseline_peak_vram |
| `compile_time_s` | Seconds for first compilation (compile methods only) |

---

## Error Handling

- OOM: caught per method, reported as `OOM` in table; benchmark continues with next method
- Compile error: caught, reported as `COMPILE_ERROR: <msg>`; skip that method
- CUDA graph with dynamic shapes: detected at capture time, reported as `GRAPH_ERROR: shape mismatch`
- `max-autotune` long compile: print ETA warning before starting

---

## Non-Goals

- Does not modify `STUNetEncoder` or `_ImageEncoder` classes
- Does not benchmark the mask encoder or full `STUNetEncoder.forward` (both image + mask paths)
- Does not produce persistent result files (stdout only; pipe to tee if needed)
- Does not test combinations beyond `compile+checkpoint`

---

## Research Sources

- [Accelerating PyTorch with CUDA Graphs — pytorch.org](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [torch.compile + CUDA Graphs for LLM Inference — spheron.network](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)
- [Maximizing AI/ML Model Performance with PyTorch Compilation — towardsdatascience.com](https://towardsdatascience.com/maximizing-ai-ml-model-performance-with-pytorch-compilation/)
- [Gradient Checkpointing Tutorial — markaicode.com](https://markaicode.com/gradient-checkpointing-tutorial-train-larger-models-less-vram/)
- [torch.func.vmap — PyTorch 2.x docs](https://docs.pytorch.org/docs/stable/generated/torch.func.vmap.html)

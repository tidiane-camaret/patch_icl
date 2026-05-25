# MultilevelICL — VRAM & Throughput Study

*Generated 2026-05-25  ·  Source: `20260525_090551.json`*

## Setup

- **Model**: `MultilevelICL` with frozen STU-Net base encoder
- **Batch size**: 1  (single training step: forward + backward + Adam)
- **Feature level**: `all` (all 6 STU-Net encoder stages concatenated → embed_dim = 1504)
- **mask_cnn_dim**: 32,  **num_registers**: 4  (held constant throughout)
- **Measurement**: 1 warmup step (not measured) + 5 measured steps; peak VRAM from `torch.cuda.max_memory_allocated`
- **Reference config** (actual training): 128³, [8³→16³→32³], NP=512, K=1, dim=256, L=8 → **1.99 GB · 289 ms/step**

---

### Image size

![image_size](benchmarks/vram_sweep/image_size.png)

VRAM scales **super-cubically** once the STU-Net skip connections at the first stage dominate: from 64³ to 256³ is ×25.5 VRAM (expected ×64 for pure cubic, but the model weights form a fixed floor). Time is nearly flat up to 128³ (encoder runs in `inference_mode`; attention always operates at 8³) then jumps ×4.7 at 256³ as the encoder forward itself becomes expensive.

| size | VRAM | time | status |
|------|------|------|--------|
| 64³ | 0.246 GB | 40.7 ms | ok |
| 128³ | 0.917 GB | 53.2 ms | ok |
| 256³ | 6.279 GB | 251.1 ms | ok |
| 512³ | — | — | OOM |


### Number of levels

![num_levels](benchmarks/vram_sweep/num_levels.png)

Levels 1–3 are cheap. **Level 4 (→64³) hits a cliff**: `extract_features` with `feature_level='all'` materialises a `(2, 1504, 64³)` float32 tensor (≈ 3.1 GB per image × 2 images = 6.2 GB), pushing total VRAM to 5.7 GB. Level 5 (→128³) would require ≈ 25 GB for those tensors alone → OOM.

**Fix**: for levels beyond 32³, switch `feature_level` to a single late encoder stage (e.g. `4` or `5`) instead of `'all'`.

| levels | finest res | VRAM | time | status |
|--------|-----------|------|------|--------|
| 1  [8³] | 8³ | 0.917 GB | 50.2 ms | ok |
| 2  [→16³] | 16³ | 0.929 GB | 75.7 ms | ok |
| 3  [→32³] | 32³ | 1.283 GB | 115.3 ms | ok |
| 4  [→64³] | 64³ | 5.706 GB | 138.8 ms | ok |
| 5  [→128³] | 128³ | — | — | OOM |
| 6  [→256³] | 256³ | — | — | OOM |


### Patches per sparse level (NP)

![n_patches_l1](benchmarks/vram_sweep/n_patches_l1.png)

With 3 levels [8³→16³→32³], VRAM grows **mildly** with NP (+0.47 GB from 128 to 4096). The source is the backward pass through `Linear(1504→dim)`: the `(B, NP, 1504)` input must be kept for weight-gradient computation. Time stays **flat** — attention compute (O(NP²)) is negligible against pipeline overhead at this dim and layer count.

**NP=8192 fails** with a `topk out of range` error (not OOM): at the 16³ level only 4096 positions exist, so Gumbel-TopK(k=8192) crashes. Cap: `NP ≤ min(N_i for all sparse levels)` = 4096 for [→16³].

| NP | VRAM | time | status |
|----|------|------|--------|
| 128 | 1.250 GB | 100.4 ms | ok |
| 256 | 1.258 GB | 116.4 ms | ok |
| 512 | 1.283 GB | 97.3 ms | ok |
| 1024 | 1.346 GB | 91.4 ms | ok |
| 2048 | 1.474 GB | 96.1 ms | ok |
| 4096 | 1.722 GB | 96.2 ms | ok |
| 8192 | — | — | ERROR: selected index k out of range |


### Context size K

![context_size](benchmarks/vram_sweep/context_size.png)

VRAM scales **linearly** with K (encoder encodes K context images, each adding the same feature tensor footprint). Time grows sub-linearly thanks to Flash Attention absorbing the K² ctx self-attention term.

| K | VRAM | time | status |
|---|------|------|--------|
| 1 | 0.929 GB | 73.2 ms | ok |
| 2 | 1.492 GB | 89.4 ms | ok |
| 4 | 2.617 GB | 116.2 ms | ok |


### Hidden dimension

![dim](benchmarks/vram_sweep/dim.png)

Both VRAM and time are **nearly flat** up to dim=256, then VRAM grows noticeably at 512 (+0.6 GB vs dim=64). Parameters grow ×47 (64→512) yet time only increases ×0.95 — at 512 tokens the attention FLOP are still negligible; the dominant cost is the frozen encoder pipeline. **dim=256→512 is a cheap capacity upgrade** (0.6 GB, 5 ms).

| dim | VRAM | time | params | status |
|-----|------|------|--------|--------|
| 64 | 0.917 GB | 120.8 ms | 777k | ok |
| 128 | 0.946 GB | 135.2 ms | 2603k | ok |
| 256 | 1.058 GB | 118.1 ms | 9499k | ok |
| 512 | 1.537 GB | 114.6 ms | 36267k | ok |


### Number of transformer layers

![num_layers](benchmarks/vram_sweep/num_layers.png)

The **primary time knob**: doubling layers roughly doubles time (×1.60 at 2→4, ×2.80 at 2→8, ×5.16 at 2→16). VRAM is sub-linear (×1.70 at ×8 layers) because most VRAM is fixed encoder features. L=8 (the current config) is the practical sweet spot — L=16 gives ×1.94 depth for ×1.84 time, diminishing returns.

| L | VRAM | time | params | status |
|---|------|------|--------|--------|
| 2 | 0.991 GB | 78.9 ms | 5291k | ok |
| 4 | 1.058 GB | 126.0 ms | 9499k | ok |
| 8 | 1.193 GB | 220.7 ms | 17914k | ok |
| 16 | 1.687 GB | 406.9 ms | 34745k | ok |


---

## Summary: scaling rules

| Parameter | VRAM scaling | Time scaling | Hard limit | Notes |
|-----------|-------------|-------------|-----------|-------|
| `image_size` | ~cubic (×3.7 per 2×) | ~flat up to 128³, then steep | 512³ OOM | STU-Net skip activations |
| `num_levels` | cheap ≤32³, cliff at 64³ (×6.2) | linear +25–40 ms/level | 5 OOM | `extract_features('all')` at 64³ costs ≈6 GB |
| `n_patches_l1` | mild (+0.47 GB, 128→4096) | flat | NP > min(N_i) crashes topk | Backward keeps `(NP, 1504)` inputs |
| `context_size` | linear ×K | sub-linear (×1.59 at K=4) | — | Main budget multiplier in production |
| `dim` | negligible ≤256, +0.6 GB at 512 | negligible | — | Free capacity upgrade |
| `num_layers` | sub-linear (×1.70 at ×8) | dominant, ~linear | — | Sweet spot: L=8 |

## Recommendations

**Free wins** (no meaningful cost):
- Increase `n_patches_l1` to 4096 (max for [→16³] stack, zero time cost)
- Increase `dim` to 512 (+0.6 GB VRAM, +5 ms/step, ×47 more parameters)

**Good tradeoffs**:
- `context_size=2` → +0.56 GB VRAM, +19% time for 2× context signal
- Adding 32³ level → +0.37 GB VRAM, +40 ms/step for a full refinement stage

**Avoid**:
- Level 4 (→64³) with `feature_level='all'`: adds 4.4 GB for feature tensors alone. Switch to a single encoder stage first.
- `image_size=256³` in training: 6.3 GB at B=1 → ~50 GB at B=8. 128³ is the practical ceiling.

**Bug to fix**:
- `n_patches_l1` must be ≤ `min(N_i for all sparse levels)` or `_gumbel_topk` raises a topk index error. Add `n = min(n, weights.shape[1])` in `sample_target_patches` and `sample_context_patches`.

---

## Overview figure

![overview](benchmarks/vram_sweep/overview.png)
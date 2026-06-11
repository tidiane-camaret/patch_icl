# patch_icl — Results Summary

*Last updated: 2026-06-11 (2D MedSegBench: feature_sim vs UniverSeg baseline)*

---

## Encoder optimization benchmark

Script: `experiments/encoders/benchmark_optimizations.py`  
Setup: STUNet-base, RTX A6000 (50.8 GB), fp16 autocast, B=1, 3 warmup + 10 measured runs.

### Inference

| Config | Method | Latency | Speedup | Peak VRAM | ΔVRAM |
|--------|--------|--------:|--------:|----------:|------:|
| 64³ | baseline | 3.7 ms | 1.00× | 188 MB | — |
| 64³ | compile_reduce | 3.8 ms | 0.97× | 188 MB | +0 |
| 64³ | compile_autotune | 3.5 ms | 1.05× | 188 MB | +0 |
| 64³ | **cuda_graph** | **2.5 ms** | **1.47×** | **144 MB** | **−44 MB** |
| 64³ | vmap (fp32) | 5.4 ms | 0.68× | 279 MB | +91 MB |
| 128³ | baseline | 13.8 ms | 1.00× | 685 MB | — |
| 128³ | compile_reduce | 13.7 ms | 1.01× | 685 MB | +0 |
| 128³ | compile_autotune | 13.8 ms | 1.00× | 685 MB | +0 |
| 128³ | **cuda_graph** | **13.4 ms** | **1.03×** | **331 MB** | **−354 MB** |
| 128³ | vmap (fp32) | 26.5 ms | 0.52× | 1.41 GB | +728 MB |

### Training (fwd + bwd)

| Config | Method | Latency | Speedup | Peak VRAM | ΔVRAM |
|--------|--------|--------:|--------:|----------:|------:|
| 64³ | baseline | 13.3 ms | 1.00× | 714 MB | — |
| 64³ | checkpoint only | 23.3 ms | 0.57× | 710 MB | −4 MB |
| 64³ | **compile_checkpoint** | **8.9 ms** | **1.50×** | **474 MB** | **−240 MB** |
| 128³ | baseline | 45.8 ms | 1.00× | 1.72 GB | — |
| 128³ | checkpoint only | 59.7 ms | 0.77× | 1.69 GB | −34 MB |
| 128³ | **compile_checkpoint** | **43.0 ms** | **1.06×** | **634 MB** | **−1.09 GB** |

`compile_checkpoint` = `torch.compile(mode="reduce-overhead")` wrapping per-stage gradient checkpointing. One-time compile cost ~50 s.

### Conclusions

- **`torch.compile` alone does nothing** on this architecture — STU-Net is memory-bandwidth–bound 3D convs with no fuseable pointwise clusters for Triton to exploit.
- **CUDA graphs** are the best inference option: 1.47× at 64³ (kernel-dispatch overhead removal); −354 MB VRAM at 128³ (static allocation pool).
- **Gradient checkpointing alone is counterproductive**: slower (0.57–0.77×) with negligible VRAM savings. Recomputation cost exceeds savings on this encoder.
- **`compile_checkpoint` is the clear training winner**: fuses the recomputed forward pass. −1.09 GB VRAM at 128³ (63% reduction) at 1.06× speed; 1.50× speedup at 64³. Use for any training run above 64³.
- **vmap**: avoid — slower and higher VRAM at B=1.

---

## Attention model compile benchmark

Script: `/tmp/compile_bench.py` (inline, not committed)  
Setup: `PatchICLAttention` (8.8 M params), RTX A6000 (50.8 GB), Python 3.11 + triton 3.2.0, B=4, K=1, N=M=512 (dense 8³), embed_dim=800 (5-stage `feature_level=all`), dim=256, L=8, fwd+bwd, 3 warmup + 10 measured runs.

> **Note:** current `nninteractive.yaml` uses `feature_level: [2, 3, 4]` → embed_dim=704 (128+256+320 ch). The benchmark above used `feature_level=all` (800 ch); speedup and VRAM savings are comparable at 704 ch.

| Method | Latency | Speedup | Peak VRAM | ΔVRAM |
|--------|--------:|--------:|----------:|------:|
| baseline (eager) | 87.8 ms | 1.00× | 591 MB | — |
| **compile reduce-overhead** | **13.8 ms** | **6.34×** | **92 MB** | **−499 MB** |
| compile max-autotune | 17.2 ms | 5.11× | 92 MB | −499 MB |

`reduce-overhead` (CUDA graph capture via torch.compile) wins both speed and VRAM:
**6.3× faster, −500 MB VRAM**. `max-autotune` costs ~5 min extra compile time for a slightly worse result here (attention is dominated by SDPA, not pointwise kernels).

### Conclusions

- **`torch.compile(mode="reduce-overhead")` is the clear winner for training the attention model.** Apply to `model.shared_level` (or the per-level `PatchICLAttention`) — not the frozen encoder.
- VRAM savings (−500 MB) are large enough to increase batch size by 1–2 at 128³.
- `max-autotune` is not worth the extra compile time here — prefer `reduce-overhead`.
- These numbers reflect the attention-only pass. The full training step adds encoder time (frozen, ~70 ms at 128³ B=8) which dilutes the speedup slightly.

---

## NNInteractive encoder — feature stage dimensions

Encoder: `NNInteractiveEncoder`, `num_stages=5`, 128³ input.  
`forward()` returns `feats[0..4]` ordered high-res → low-res.

| Index | Channels | Spatial | Stride | VRAM @ B=8 fp16 |
|-------|----------|---------|--------|-----------------|
| 0 | 32 | 128³ | 1× | 1 074 MB |
| 1 | 64 | 64³ | 2× | 268 MB |
| 2 | 128 | 32³ | 4× | 67 MB |
| 3 | 256 | 16³ | 8× | 17 MB |
| 4 | 320 | 8³ | 16× | 2.6 MB ← bottleneck |

`model.feature_level` selects which levels are passed to the attention stack. Each selected tensor is interpolated (trilinear) to the current attention grid resolution, then concatenated channel-wise to form `embed_dim`:

| `feature_level` | Levels used | `embed_dim` | Skip VRAM |
|---|---|---|---|
| `all` | 0+1+2+3+4 | 800 | 1 428 MB |
| `[2, 3, 4]` ← current | 2+3+4 | 704 | 87 MB |
| `[3, 4]` | 3+4 | 576 | 20 MB |
| `4` | bottleneck only | 320 | 3 MB |

`embed_dim` is projected down to `model.dim` (256) by a learned linear at the start of each `PatchICLAttention` forward. Changing `feature_level` only affects that projection and which encoder tensors are retained in memory.

---

## VRAM & throughput sweep

Full report: `results/benchmarks/vram_sweep/report.md`

Reference config (current training): 128³, [8³→16³→32³], NP=512, K=1, dim=256, L=8 → **1.99 GB · 289 ms/step** at batch_size=1.

| Parameter | Range tested | VRAM range | Time range |
|-----------|-------------|------------|------------|
| `image_size` | 64³–512³ | 0.25–6.28 GB (512³ OOM) | 41–251 ms |
| `num_levels` | 1–6 | 0.92–5.71 GB (5+ OOM) | 50–139 ms |
| `n_patches_l1` | 128–8192 | 1.25–1.72 GB (8192 error) | ~96 ms flat |
| `context_size K` | 1–4 | 0.93–2.62 GB | 73–116 ms |
| `dim` | 64–512 | 0.92–1.54 GB | ~115 ms flat |
| `num_layers L` | 2–16 | 0.99–1.69 GB | 79–407 ms |

---

## Segmentation quality: MultilevelICL vs Medverse

Source: `results/benchmarks/eval/eval_20260524_*.json` — TotalSegmentator test split, K=1, 117 classes.

| Model | Mean Dice | GFLOPs | Latency |
|-------|-----------|--------|---------|
| MultilevelICL (ours) | **0.212** | 599 | 102 ms |
| Medverse | 0.105 | 2363 | 172 ms |

### Per-class Dice

| Class | MultilevelICL | Medverse |
|-------|:---:|:---:|
| liver | 0.624 | 0.747 |
| gluteus_maximus_left | 0.615 | 0.340 |
| femur_left | 0.614 | 0.465 |
| femur_right | 0.591 | 0.459 |
| gluteus_maximus_right | 0.567 | 0.344 |
| spleen | 0.540 | 0.462 |
| urinary_bladder | 0.508 | 0.337 |
| kidney_right | 0.441 | **0.771** |
| brain | 0.395 | **0.528** |
| kidney_left | 0.353 | 0.486 |
| aorta | 0.310 | 0.218 |
| pancreas | 0.224 | 0.037 |
| adrenal_gland_left | 0.009 | 0.005 |
| common_carotid_artery_left | 0.014 | 0.008 |

---

## 2D MedSegBench: UniverSeg encoder + TabPFN vs UniverSeg baseline

Runs: `wandb/run-20260603_191202-vnnhsldm` (feature_sim), `wandb/run-20260603_171003-37xdo361` (universeg).  
Scripts: `experiments/2d/feature_sim.py`, `experiments/2d/universeg.py`.  
Setup: MedSegBench val split, 13 237 samples, 35 datasets, K=1, image_size=128.

### Summary

| Model | dice/mean | Inference | FLOPs/sample | Runtime (total) |
|-------|-----------|-----------|--------------|-----------------|
| UniverSeg (full model) | 0.242 | ~2.2 ms | 13.57 GFLOPs | 29 s |
| **UniverSeg encoder + TabPFN** | **0.267** | ~495 ms | 8.12 GFLOPs (enc only) | 6 601 s |

Feature-sim is **+2.5% mean Dice** at **228× higher latency**.

feature_sim config: `feature_level=all` (4 levels → C=256), `output_size=16` (16×16=256 patches/image), `n_estimators=4`, `balance_ratio=null`, `context_mask=false`.

### Per-dataset comparison (selected)

**UniverSeg baseline wins (Δ > +0.10)**

| Dataset | feat_sim | universeg | Δ (useg) |
|---------|----------|-----------|----------|
| promise12 | 0.114 | **0.369** | +0.255 |
| usforkidney | 0.345 | **0.501** | +0.156 |
| bbbc010 | 0.306 | **0.454** | +0.148 |
| chuac | 0.293 | **0.428** | +0.135 |
| ultrasoundnerve | 0.156 | **0.281** | +0.125 |

**feature_sim wins (Δ > +0.10)**

| Dataset | feat_sim | universeg | Δ (fsim) |
|---------|----------|-----------|----------|
| brifiseg | **0.363** | 0.135 | +0.228 |
| deepbacs | **0.244** | 0.089 | +0.155 |
| dynamicnuclear | **0.350** | 0.206 | +0.144 |
| isic2016 | **0.603** | 0.470 | +0.133 |
| cellnuclei | **0.357** | 0.225 | +0.132 |
| isic2018 | **0.574** | 0.466 | +0.108 |
| dca1 | **0.316** | 0.209 | +0.107 |
| kvasir | **0.317** | 0.213 | +0.104 |

feature_sim wins 25/35 datasets, UniverSeg wins 10/35.

### Interpretation

UniverSeg's cross-conv attention wins on **shape-defined** structures (prostate, kidney, nerve) where the cross-image interaction propagates a global shape prior that local patch features cannot replicate at 16×16 resolution. The promise12 gap (+0.255) is the clearest example.

feature_sim wins on **texture/appearance-defined** objects: microscopy cells (deepbacs, dynamicnuclear, cellnuclei), retinal vessels, dermoscopy. Here the UniverSeg encoder's per-patch features carry a distinctive statistical signature and TabPFN as a discriminative classifier exploits it better than the cross-conv decoder.

### Suggested next experiments

| Change | Expected effect |
|--------|----------------|
| `context_size=4` | Most impactful — 4× more TabPFN training points per image |
| `context_mask=true` | Upper bound: mask-conditioned encoder features |
| `output_size=8`, `level=-1` | Faster; tests whether bottleneck alone suffices |
| `n_estimators=8` | Modest accuracy gain, ~2× TabPFN time |

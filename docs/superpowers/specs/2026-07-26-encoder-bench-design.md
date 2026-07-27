# Encoder compute/latency scaling benchmark — design

Date: 2026-07-26
Status: approved (design)

## Goal & scope

A standalone harness that measures **how each 3D encoder's compute cost scales with
input size**, so we can choose backbones for the in-context segmentation model.

- Encoders are measured **in isolation**: a single volume `(B, C, D, H, W)`, batch dim
  only. K (context count) is a separate batch multiplier, reasoned about outside this study.
- Each encoder is measured at its **best-optimized config only** (one curve per encoder).
- Primary question: find the **crossover input size** where transformer/Mamba encoders
  overtake CNNs on compute cost, and rank encoders by training-step cost, complexity,
  and throughput.
- **Accuracy / Dice is explicitly out of scope.** Pure compute/latency/VRAM.

## Target hardware

- Node: **thor** — NVIDIA RTX A6000, Ampere **sm_86**, 48 GB, driver → CUDA 12.3.
- Env: **`.venv_thor/bin/python`** (torch 2.5.1+cu121). Run everything with this interpreter.
- Kernel availability: SDPA → FlashAttention-2 (no FA3, that needs Hopper); `torch.compile`
  mature; Mamba `mamba_ssm` prebuilt wheels exist for cu121/Ampere.
- **thor `torch.compile` gotcha:** inductor's C++ build fails unless
  `export CXX=/usr/bin/g++ CC=/usr/bin/gcc` is set (bare `g++` resolves to `/bin/g++`
  whose install prefix `/` omits libstdc++ headers). The harness sets these env vars
  automatically at startup when `/bin` is not a symlink.

## Layout & components

New directory `experiments/3d/encoder_bench/` (sibling of `feature_sim/`):

- **`registry.py`** — `name -> factory(in_ch, **kw) -> nn.Module`. Returns a bare encoder
  taking `(B, C, D, H, W)` and returning a feature map or list of feature maps. Each entry
  also declares:
  - `in_ch`: input channels the backbone expects.
  - `size_multiple`: input-size divisibility constraint (e.g. 16 for patch-16 ViT,
    32 for Swin). Sizes not divisible are skipped and logged as `NaN`.
  - `opt_profile`: best-optimization settings (compile mode, autocast dtype,
    channels_last flag).
  Wraps the 6 existing `src/models/encoders/` encoders + `ConvEncoder3D` from
  `src/models/patchset3d.py`; adds 2 compute-only stand-ins (Primus, SegMamba).
- **`optimize.py`** — applies an `opt_profile` to a module: `channels_last_3d` memory
  format, `torch.autocast(device, bf16)` wrapping, SDPA/flash context, and
  `torch.compile(mode=...)`. Sets `CC`/`CXX` to `/usr/bin/*` on non-usr-merged nodes.
- **`profile.py`** — measurement core:
  - warmup (≥3 iters, absorbs compile trace + recompile),
  - CUDA-event timing, `torch.cuda.synchronize()` around each, median of ≥10 timed iters,
  - `reset_peak_memory_stats` / `max_memory_allocated` for peak VRAM (with grads),
  - FLOPs + params via `fvcore.nn.FlopCountAnalysis` (fallback `thop`),
  - largest-batch throughput: exponential batch search until OOM, report vol/s at the
    largest batch that fits.
  - OOM / unsupported-size caught → `NaN` row + logged, never crash the sweep.
- **`run.py`** — Hydra-style entry mirroring `feature_sim/run.py`. Sweeps
  `encoders × input_sizes`, writes CSV, renders scaling-curve PNGs, optional wandb.

## Roster (9 encoders)

| name | family | source | notes |
|---|---|---|---|
| `conv_encoder3d` | CNN | `src/models/patchset3d.py` | current in-context encoder |
| `resenc` | CNN | zoo | nnUNet ResEncM, scratch |
| `vocomni_nnunet` | CNN | zoo | PlainConvUNet |
| `stunet` | CNN | zoo | STU-Net |
| `nninteractive` | CNN | zoo | ResidualEncoder |
| `vocomni_swin` | Transformer | zoo | SwinUNETR |
| `threedino` | Transformer | zoo | ViT-L-3D patch16 |
| `primus` | Transformer | **stand-in** | high-res-token ViT; faithful block/dims/patch, scratch |
| `segmamba` | Mamba/SSM | **stand-in** | CNN-stem + tri-scan SSM blocks; routes through real `mamba_ssm.selective_scan` if importable, else a reference PyTorch scan |

**Weight loading is optional and off by default.** For a compute study, random/`meta`
init yields identical FLOPs/latency/VRAM; loading real pretrained checkpoints is a flag so
the benchmark runs anywhere, including nodes without the checkpoints. Compute-only
stand-ins never load weights.

## Measurement protocol

- **Input sizes**: `32³, 64³, 96³, 128³` (all divisible by 16 and 32). Encoders requiring
  coarser divisibility skip non-fitting sizes → `NaN`, logged.
- **Per (encoder, input_size)** at best-optimized config:
  `params`, `gflops`, `fwd_bwd_ms` (median), `train_vram_mb` (peak w/ grads),
  `throughput_vol_s` (largest fitting batch).
- Fixed-seed dummy input. ≥3 warmup iters (covers compile trace). Median of ≥10 timed.
  `synchronize` around each timed iter. `empty_cache` + `reset_peak_memory_stats` between
  configs.
- **Compile confound control**: compile once per (encoder, input_size) — dynamic shapes
  recompile, so that recompile occurs during warmup and is excluded from timing.
- OOM caught → `NaN` + logged.

## Output

- `results/encoder_bench/encoder_bench.csv` — tidy, one row per (encoder, input_size).
  Source of truth.
- `results/encoder_bench/*.png` — scaling curves (one line per encoder, log-y, colored by
  family): `fwd_bwd_ms` vs size, `train_vram_mb` vs size, `gflops` vs size — the crossover
  plots.
- wandb table + the same curves as media, gated on `wandb.project` (like `feature_sim`).

## Non-goals

- No Dice / segmentation-accuracy measurement.
- No full in-context forward (transformer set-attention excluded); encoder backbone only.
- No plain-vs-optimized ablation; only the best-optimized config per encoder.
- No multi-node run; thor only for the headline numbers.

# Inference-time / FLOPs comparison: medverse vs patchset3d vs patchset3d_v2

Script: `bench_inference_compare.py` (this directory). Raw numbers: `results.json`.

## Conditions

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, torch 2.12.1+cu130
- B=1, K=1 (single context pair), 128³ volumes — matches `69_medverse_varspacing_6_1_5.yaml`'s
  and `bench_arch.py`'s own convention
- `model.predict()` called directly (the same method `experiments/3d/evaluate.py`'s real eval
  loop calls), 20 timed reps after 5 warmup reps, `torch.cuda.synchronize()` around timing

## What "fair" means here

Each model is benchmarked at the precision it was actually trained/evaluated at, not an
artificially-equalized one:

- **medverse**: plain fp32, no autocast anywhere. `MedverseModel` has no internal autocast, and
  `69_medverse_varspacing_6_1_5.yaml` / the released checkpoint were never trained or validated
  under bf16 — forcing bf16 on it would benchmark a regime it was never calibrated for.
- **patchset3d / patchset3d_v2**: bf16 autocast (`arch.encoder_precision=bf16`), matching every
  real training run of both classes in this repo, and `evaluate.py::_eval_autocast`'s own
  bf16-CUDA-autocast convention.

Forcing all three into one precision was considered and mostly dropped: `PatchSet3D`/
`PatchSetV2`'s encoder classes (`plainconv_ts.py`) apply their own **internal** bf16 autocast
unconditionally at construction time (`encoder_precision=bf16`), independent of any outer
autocast context — so externally disabling autocast to force fp32 doesn't actually make them
fp32; it just creates a dtype mismatch between the (still-bf16) encoder output and the (now
fp32) downstream layers. A genuine fp32 comparison for these two would need separate
`encoder_precision=fp32` model instances, out of scope here.

`MedverseModel` has no such internal autocast, so the reverse direction — externally wrapping
it in bf16 even though it was never trained/validated there — works cleanly with no dtype
conflicts. Added as a supplementary data point (not the headline "fair" number) below, purely
to show medverse's own bf16 speed potential; its *accuracy* under bf16 was not checked.

## Architectures benchmarked (each model's actual real-run configuration)

| Model | Config source |
|---|---|
| medverse | Released weights, `sw_roi_size=(128,128,128)` — matches `69`'s single-forward (no cascade) setup |
| patchset3d | `97_iris_decoder_ct_only` / `99`'s own arch verbatim (wandb run `x1gz71wj`'s saved config): `plainconv_ts`, `e=768, h=3072, l=4, a=12`, `decoder=iris` |
| patchset3d_v2 | `101_patchset_v2_mask8_wide`'s own arch verbatim: `plainconv_ts` widened to 768ch, `mask_patch_size=8`, `feat_norm=context`, `img_embed_mlp=true` |

## Results

| model | params | GFLOPs (predict) | inference (native precision) | peak memory (native) | bf16 (supplementary)† |
|---|---:|---:|---:|---:|---:|
| medverse | 71.1M | 2362.6 | 68.3 ms (fp32) | 3.30 GB | 44.1 ms, 2.32 GB |
| patchset3d (97/99, decoder=iris) | 71.9M | not measurable* | 36.4 ms (bf16) | 3.36 GB | — (already native) |
| patchset3d_v2 (101, mask8_wide) | 67.9M | 1190.5 [encoder 804.5, transformer 36.9, other 349.1] | 36.6 ms (bf16) | 2.63 GB | — (already native) |

† medverse's bf16 column is an external `torch.autocast` wrap, not its trained precision —
speed/memory data point only, accuracy under bf16 not checked. See "What 'fair' means here".

\* `FlopCounterMode` raises `AssertionError: Expected gradient function to be set` for this
specific architecture — root-caused (not a benchmark bug, see below). Given `patchset3d`'s
architecture and parameter count (71.9M) are close to `patchset3d_v2`'s (67.9M, 1190.5 GFLOPs),
its true FLOPs are almost certainly in the same order of magnitude, but that's an inference, not
a measurement — no number is reported.

### Why patchset3d's FLOPs count fails

Traced the exact crash site: `PatchSet3D._iris_task_encode` feeds a raw `nn.Parameter`
(`self.iris_ctx_query`) directly into `nn.MultiheadAttention` as the query tensor
(`q = self.iris_ctx_query.unsqueeze(0).expand(B, -1, -1)`). Under `predict()`'s
`@torch.no_grad()`, this expand produces a tensor with no `grad_fn` and `requires_grad=False`.
`FlopCounterMode` internally uses `torch.utils.module_tracker.ModuleTracker` to attribute FLOPs
to submodules, and `ModuleTracker`'s forward-pre-hook calls
`torch.autograd.graph.register_multi_grad_hook` on every module's inputs, which asserts a valid
grad function exists — it doesn't, hence the crash.

`patchset3d_v2` has an analogous raw parameter (`compress_slots`) used the same way, but does
**not** hit this: its `_compress_all` runs it through a custom `RowCrossAttention` module (a
plain `F.scaled_dot_product_attention` call), not `nn.MultiheadAttention` — `nn.MultiheadAttention`
apparently interacts with `ModuleTracker`'s hook in a way the plain SDPA path doesn't. This is a
PyTorch/architecture interaction specific to `patchset3d`'s already-deployed iris-decoder code,
not something introduced by this benchmark — fixing it (if wanted) would mean changing
`patchset3d.py`'s attention call convention, out of scope here.

## Reading the numbers

- **Inference time**: `patchset3d` and `patchset3d_v2` are effectively tied (36.4 vs 36.6 ms) and
  both ~1.9x faster than medverse (68.3 ms) at their respective native precisions — expected,
  since medverse runs fp32 while both patchset variants run bf16. Under the supplementary bf16
  wrap, medverse drops to 44.1 ms (peak memory 3.30→2.32 GB too) — a real ~1.55x speedup from
  precision alone (68.3/44.1), narrowing the remaining gap to the patchset variants' own bf16
  numbers to ~1.2x (44.1/36.5) — most of medverse's native-precision speed disadvantage here
  traces to the fp32 vs bf16 choice, not the architecture itself.
- **Peak memory**: `patchset3d_v2` is the lightest of the three (2.63 GB vs 3.30-3.36 GB) despite
  having a wider bottleneck encoder stage (768ch) than `patchset3d`'s multi-scale concat
  (256+512=768ch too, same total) — consistent with `patchset3d_v2`'s architecture avoiding the
  O(R³)-scale attention `patchset3d`'s iris decoder reintroduces in its `_iris_task_encode`/
  `_decode_iris` cross-attention (see `docs/logs.md` 2026-09-18).
- **Params**: all three are close (67.9-71.9M) — not a capacity-driven comparison, the
  architectures were sized independently to land in a similar range.
- **GFLOPs vs wall-clock**: medverse has ~2x `patchset3d_v2`'s FLOPs (2362.6 vs 1190.5 GFLOPs)
  but is only ~1.9x slower in wall-clock, not ~2x — consistent with medverse's fp32 compute
  being less FLOP-efficient per unit time on this GPU than bf16 (expected: Blackwell's bf16
  tensor-core throughput is roughly 2x its fp32 throughput), so the wall-clock gap undershoots
  the raw FLOPs ratio.

## Caveats

- Single-sample timing (B=1); no batching efficiency is captured here (see `docs/logs.md`
  entries on `_pool_all`'s and `arch.seq_compress`'s own batch-size-dependent memory behavior for
  that separate question).
- `patchset3d`'s FLOPs gap means the GFLOPs column isn't a complete three-way comparison; treat
  the wall-clock and memory columns as the primary result for that model.
- Neither `patchset3d` config benchmarked here has been evaluated to convergence under
  identical data/epochs as the other (see `docs/logs.md` 2026-09-19 "PatchSetV2 first real run"
  for the accuracy-side comparison, a separate question from this performance benchmark).

# Inference-time / FLOPs comparison: medverse vs patchset3d (iris / conv) vs patchset3d_v2

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
| patchset3d (iris) | `97_iris_decoder_ct_only` / `99`'s own arch verbatim (wandb run `x1gz71wj`'s saved config): `plainconv_ts`, `e=768, h=3072, l=4, a=12`, `decoder=iris` |
| patchset3d (conv) | `92_multisource_synth`'s own arch verbatim (Hydra-resolved: `92 → 89_multisource_cascade → 88_cascade`, which selects `model=m2_patchset_decoder`): same `plainconv_ts`/`e/h/l/a`, `decoder=conv` — does **not** skip the R³-token main transformer (see Follow-up below) |
| patchset3d_v2 | `101_patchset_v2_mask8_wide`'s own arch verbatim: `plainconv_ts` widened to 768ch, `mask_patch_size=8`, `feat_norm=context`, `img_embed_mlp=true` |
| patchset3d_v2 (103) | `103_patchset_v2_cascade`'s own arch verbatim (Hydra-resolved), benchmarked single-level (K=1, no cascade re-crop — `run_cascade` just calls this same forward N times per task): `plainconv_ts_features_per_stage=[32,64,256,768]`, `nnunet_ts_stages=[3]`, `encoder_input_norm=zscore`, `encoder_spacing_aware=true`, `mask_embed=conv` (`MaskConvEmbedV2`), `decode_layers=3` |

## Results

| model | params | GFLOPs (predict) | inference (native precision) | peak memory (native) | bf16 (supplementary)† |
|---|---:|---:|---:|---:|---:|
| medverse | 71.1M | 2362.6 | 69.6 ms (fp32) | 3.30 GB | 45.5 ms, 2.32 GB |
| patchset3d (97/99, decoder=iris) | 71.9M | not measurable* | 36.4 ms (bf16) | 3.36 GB | — (already native) |
| patchset3d (92_multisource_synth, decoder=conv) | 48.0M | 3891.1 [encoder 877.0, transformer 2736.3, other 277.8] | 43.5 ms (bf16) | 2.50 GB | — (already native) |
| patchset3d_v2 (101, mask8_wide) | 77.4M | 1230.3 [encoder 804.5, transformer 36.9, other 388.9] | 35.9 ms (bf16) | 2.67 GB | — (already native) |
| patchset3d_v2 (103, cascade arch, decode_layers=3) | 109.6M | 1993.4 [encoder 1050.9, transformer 36.9, other 905.5] | 42.6 ms (bf16) | 3.78 GB | — (already native) |

† medverse's bf16 column is an external `torch.autocast` wrap, not its trained precision —
speed/memory data point only, accuracy under bf16 not checked. See "What 'fair' means here".

\* `FlopCounterMode` raises `AssertionError: Expected gradient function to be set` for this
specific architecture — root-caused below (not a benchmark bug). Unlike the `decoder=conv`
variant, `decoder=iris` also skips the real cost driver entirely (see Follow-up), so its true
FLOPs would land well below the conv variant's 3891.1 GFLOPs, not above it — no number reported
since it isn't measurable, not because it's assumed large.

(2026-09-20 update: added the `patchset3d_v2 (103)` row. `patchset3d_v2 (101)`'s own numbers
shifted too (67.9M→77.4M params, 1190.5→1230.3 GFLOPs) — a real effect, not benchmark noise:
`_decode`'s single `iris_t2f`/`iris_f2t` pair was replaced by `DecodeCrossBlock` (the
`arch.decode_layers` change, docs/logs.md 2026-09-20), which adds a per-side MLP even at the
new default `decode_layers=1`. Earlier in the day the whole table was also remeasured on an
idle GPU after adding the `decoder=conv` row, correcting numbers that had briefly been
measured while a training job shared the GPU; relative ordering is unchanged throughout.)

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

### Follow-up: does "v1 uses R³-token attention, v2 compresses to m" show up as a cost gap?

`patchset3d`'s main cross-context transformer (`_attn`, gated by `self.N = resolution³ = 4096`
raw per-cell tokens, `seq_compress=False` in this arch → **no** compression, `full_attn=True` →
dense self-attention over all `(K+1)*N` tokens) genuinely is R³-scale if it ever runs. Whether
it runs depends entirely on `decoder_kind`:

- **`decoder=iris`** (the `97`/`99`/`x1gz71wj` config, benchmarked above as "patchset3d (iris)"):
  `forward()` at `patchset3d.py:1108` explicitly skips `_attn` whenever `decoder_kind == "iris"`
  and `cascade_registers=False` (both true here) — the docstring even says so ("The main
  cross-context transformer ... is unused for arch.decoder=iris's final logit ... Skip `_attn`
  entirely"). Confirmed empirically with a forward-hook stage-timer
  (`profile_stage_split.py`, hooks on `net.encoder`/`net.transformer`): `self.transformer`
  fires **0 times** per `predict()` call. `decoder=iris`'s *actual* decode path
  (`_iris_task_encode` → `_decode_iris`) is itself an m-scale design (`iris_m=10` query rows
  cross-attending into the raw `R³` grid on the KV side only — `O(m×R³)`, linear in the big
  side, never `R³×R³`) — the same family as v2's `compress_m`, just m=10 instead of m=128.
- **`decoder=conv`** (`92_multisource_synth`'s config, benchmarked above as "patchset3d (conv)"):
  does **not** hit the `patchset3d.py:1108` skip — `_attn`'s dense R³-token self-attention runs
  for real. This is the config the "v1 has R³ token length instead of m" question was actually
  about.

Measured with the same forward-hook stage-timer, GPU idle (`profile_stage_split.py`):

| variant | total | encoder | transformer | other |
|---|---:|---:|---:|---:|
| patchset3d (iris) | 36.8 ms | 15.9 ms (43%) | 0.00 ms (0%) | 20.8 ms (57%) |
| patchset3d (conv) | 43.4 ms | 15.4 ms (35%) | **13.4 ms (31%)** | 14.6 ms (34%) |
| patchset3d_v2 | 36.7 ms | 15.8 ms (43%) | 0.7 ms (2%) | 20.2 ms (55%) |

So the R³-token transformer, once actually exercised, is real and visible: it costs ~19×
`patchset3d_v2`'s compressed-transformer time (13.4 vs 0.7 ms) and dominates the FLOPs table
(2736.3 of 3891.1 GFLOPs, 70%). But because attention matmuls run efficiently on this GPU's
bf16 tensor cores, that 13.4 ms only adds ~18% to *total* wall-clock (36.8→43.4 ms) — the
encoder and fine-decode convolution stages ("other") still account for the majority of time in
every variant, so a ~19× cost multiplier on one sub-stage translates to a much smaller multiplier
on the number that actually matters for throughput. `patchset3d (conv)`'s peak memory (2.50 GB)
is even slightly *below* the iris variant's (3.36 GB) despite the extra attention compute —
`decoder=iris` carries additional iris-specific parameters/activations (pixel-shuffle upsampling
blocks, class-embedding path) that the conv decoder doesn't.

## Reading the numbers

- **Inference time**: `patchset3d (iris)` and `patchset3d_v2 (101)` are effectively tied (36.4
  vs 35.9 ms); `patchset3d (conv)` (43.5 ms) and `patchset3d_v2 (103)` (42.6 ms) sit ~18-19%
  higher, for two DIFFERENT reasons — `patchset3d (conv)` because it's the only v1 variant that
  actually runs the R³-token main transformer (see Follow-up), `patchset3d_v2 (103)` because of
  its own accumulation of changes (wider encoder, `decode_layers=3`, `mask_embed=conv`) rather
  than any R³-scale attention (its Stage B transformer is unchanged at 36.9 GFLOPs, identical to
  101's). All patchset variants are ~1.6-1.9x faster than medverse (69.6 ms) at their respective
  native precisions — expected, since medverse runs fp32 while every patchset variant runs
  bf16. Under the supplementary bf16 wrap, medverse drops to 45.5 ms (peak memory 3.30→2.32 GB
  too) — a real ~1.5x speedup from precision alone, narrowing the remaining gap to the patchset
  variants' own bf16 numbers to ~1.05-1.27x — most of medverse's native-precision speed
  disadvantage here traces to the fp32 vs bf16 choice, not the architecture itself.
- **Peak memory**: `patchset3d (conv)` is the lightest (2.50 GB), then `patchset3d_v2 (101)`
  (2.67 GB), then `patchset3d (iris)` (3.36 GB) — the iris decoder's own extra
  parameters/activations (pixel-shuffle upsampling blocks, class-embedding path) cost more
  memory than the conv variant's real R³-token attention does. `patchset3d_v2 (103)` is the
  heaviest of all five (3.78 GB) — its own accumulation of changes (wider encoder concat,
  `decode_layers=3`'s extra activations, `MaskConvEmbedV2`'s own conv layers) adds up, despite
  never touching R³-scale attention either.
- **Params**: `patchset3d (iris)` (71.9M) and `patchset3d_v2 (101)` (77.4M) are close;
  `patchset3d (conv)` is notably smaller (48.0M) — the iris decoder carries real extra
  parameters beyond what `_attn`+conv-decoder needs. `patchset3d_v2 (103)` is the largest
  (109.6M) — every one of its changes vs 101 (wider `plainconv_ts_features_per_stage`,
  `decode_layers=3`'s 3x more `DecodeCrossBlock` params, `MaskConvEmbedV2`'s conv front end
  vs a bare `Linear`) adds params, none removes any.
- **GFLOPs vs wall-clock**: `patchset3d (conv)`'s transformer stage is 2736.3 of 3891.1 total
  GFLOPs (70%) but only 13.4 of 43.4 ms wall-clock (~31%, per the Follow-up's stage-timer) —
  dense bf16 attention matmuls run much closer to this GPU's peak FLOP/s than the conv-heavy
  encoder/decode stages do, so a FLOPs-heavy stage doesn't cost proportionally as much
  wall-clock. `patchset3d_v2 (103)` vs `(101)` shows the same pattern from a different angle:
  +62% GFLOPs (1993.4 vs 1230.3) but only +19% wall-clock (42.6 vs 35.9 ms) — its transformer
  stage is IDENTICAL between the two (36.9 GFLOPs; `decode_layers` doesn't touch Stage B), so
  all of that extra GFLOPs is encoder conv work (804.5→1050.9) and `_decode`'s own cross
  -attention/MLP/conv cost (388.9→905.5, the `decode_layers=3` + `MaskConvEmbedV2` effect) —
  and conv-heavy FLOPs translate to wall-clock less efficiently than attention FLOPs do here,
  same story as the R³-transformer case just via a different mechanism. Medverse has ~1.9x
  `patchset3d_v2 (101)`'s FLOPs (2362.6 vs 1230.3) but only ~1.9x slower in wall-clock —
  roughly proportional here, since medverse's fp32 compute is less FLOP-efficient per unit time
  than bf16 (Blackwell's bf16 tensor-core throughput is roughly 2x its fp32 throughput).

## Caveats

- Single-sample timing (B=1); no batching efficiency is captured here (see `docs/logs.md`
  entries on `_pool_all`'s and `arch.seq_compress`'s own batch-size-dependent memory behavior for
  that separate question).
- `patchset3d (iris)`'s FLOPs count still isn't measurable (`FlopCounterMode` crash, root-caused
  above) — its true FLOPs are almost certainly *lower* than the `conv` variant's 3891.1 GFLOPs,
  since it's the variant that skips the R³-token transformer, but no number is reported.
- None of the `patchset3d`/`patchset3d_v2` configs benchmarked here have been evaluated to
  convergence under identical data/epochs as each other (see `docs/logs.md` 2026-09-19
  "PatchSetV2 first real run" for the accuracy-side comparison, a separate question from this
  performance benchmark). `patchset3d (conv)`'s 92_multisource_synth arch and
  `patchset3d_v2 (103)`'s cascade arch have never been directly Dice-compared against anything
  else here — this benchmark is purely about inference cost, not accuracy.
- `patchset3d_v2 (103)` is benchmarked in single-level (K=1) mode only, matching every other
  row's convention — it says nothing about the cost of a full multi-level cascade run
  (`cascade.py::run_cascade` calls this same forward once per cascade level per task, so total
  cascade cost is roughly `n_levels ×` this row, plus re-crop/GPU-realize overhead not measured
  here).

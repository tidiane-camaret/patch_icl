# PatchSetV2 — clean Iris-style in-context 3D segmentation

## Motivation

Experiments 91-99 integrated Iris-decoder elements (task encoding, mask decoding, Eq 2-6 of
`docs/methods/iris.md`) into `PatchSet3D` (`src/models/patchset3d.py`) incrementally, one knob
at a time: `mask_slots`, `decoder`/`fine_decode`/`fine_stage`, `pool_token`, `seq_compress`,
`iris_pixelshuffle_r`, `cascade_registers`. The result works but has real accretion problems:

- The iris decoder path is asymmetric — `_iris_task_encode` builds a task representation from
  **support only**; the query only meets it via a separate, expensive bidirectional
  cross-attention (`iris_t2f`/`iris_f2t`) directly against its raw `R³`-cell grid.
- `query_prior` (the cascade coarse-level prediction) is silently dropped whenever
  `decoder_kind=="iris" and not cascade_registers` — `forward()` skips `_attn` entirely in that
  case, and `_attn` is the only place `query_prior` is consumed (`patchset3d.py:1171`).
- `arch.seq_compress`'s Stage C (`_expand`) exists to decompress the query back to per-cell
  resolution for the old `fine_filter`/`conv` decoders — unnecessary once the decoder is
  Iris-style, since Iris's own decode (Eq 5-6) never compresses the query's feature map in the
  first place (see "Trace: Iris's own decoder needs no decompression" below).
- The Iris pixel-shuffle trick (Eq 3-4) exists to fuse mask into features *without* a learned
  attention step; once a learned, mask-aware compressor is available (`RowCrossAttention`,
  already used by `seq_compress`'s Stage A), it's redundant.
- Net effect: a config surface where most knobs don't apply to most configs, and a user has to
  understand `mask_slots`/`register_routed`/`full_attn`/`seq_compress`/`pool_token`/
  `fine_decode`/`decoder` interactions to reason about any one of them.

`PatchSetV2` is a from-scratch, minimal class that reuses the existing low-level building
blocks but is not another branch grafted onto `PatchSet3D`. `PatchSet3D` is left untouched
(existing configs/checkpoints unaffected).

## Architecture

### Inputs

Per forward call: target image `x_q` + prior/prediction mask `prev_q` (soft probability,
matching today's `query_prior` — support-mean fallback when absent, same as `_attn` does now),
and K context pairs `(x_s, y_s)` with real binary GT masks. `T = K+1` volumes total.

### Per-volume tokenize + pool + compress (weight-shared across all T volumes)

For each volume `v` in `[target, context_1, ..., context_K]`, independently:

1. **Encode.** Shared encoder (any existing `arch.encoder` choice) → coarse `R³` grid
   `feat_v` (`Cf` channels) + unpooled multiscale pyramid `F_v_all` (the `fine_stage` taps,
   unchanged mechanism).
2. **Per-cell tokens.** `img_embed(feat_v) + mask_embed(occupancy_v) + FourierPE` →
   `(N, 2, e)`, `N=R³`, columns `[img, mask]`. Mask column holds real GT for context volumes,
   `prev_q`/support-mean for the target volume — same convention `_tokens` uses today, no
   `mask_slots` tagging needed (the target/context distinction is structural, not a flag).
3. **`pool_v` (Iris Eq 2, generalized to every volume).** Upsample `v`'s finest available
   `fine_stage` feature map to `v`'s **native** `(D,H,W)` resolution — never `R`, never a
   coarser stage's own resolution — mask by `v`'s own mask/prior at that native resolution,
   masked-average over foreground voxels → one row `(e,)` after projection, broadcast into
   both img and mask columns → `(1, 2, e)`.
4. **Stage A compress** (`RowCrossAttention`, weight-shared across all `T` volumes, unchanged
   mechanism from `PatchSet3D._compress`): `compress_m` learnable slots cross-attend into `v`'s
   own raw `(N, 2, e)` KV, scoped to `v` only → `(compress_m, 2, e)`.
5. **Concatenate:** `v`'s final sequence = `[pool_v ; compressed_m_v]` → `(compress_m+1, 2, e)`.

No pixel-shuffle/unshuffle step (see Motivation).

### Stage B — shared cross-volume transformer (unchanged `TransformerEncoderStack`)

- All `T` volumes' `(compress_m+1, 2, e)` sequences concatenate →
  `(T·(compress_m+1), 2, e)`.
- `ThinkingRows` prepended, as today.
- `cascade_registers` (if enabled): previous cascade level's thinking-row output, projected
  and tagged (`cascade_proj`/`cascade_type`), prepended as extra rows — unchanged mechanism
  from `PatchSet3D._attn`.
- Full self-attention across every row — no `register_routed`, no block-masking, no
  `full_attn` toggle (it's the only mode: every volume's tokens can attend to every other
  volume's tokens and to thinking/cascade rows).
- No RoPE (compressed rows have no single cell position — same reasoning `seq_compress`
  already applies today). Fourier PE is baked in per-cell before compression (step 2 above).

### Decode (Iris Eq 5-6, unchanged mechanism)

- Take the target volume's own post-Stage-B `(compress_m+1, 2, e)` slice as `T`. This is
  already context-aware — Stage B let it exchange information with every context volume —
  richer than plain Iris's support-only `T`.
- Bidirectional cross-attention (`iris_t2f`/`iris_f2t`-equivalent, unchanged) between `T` and
  the target's **raw, never-compressed** per-cell grid `F_q` (the same `img_embed` output from
  step 2 above, before any compression) — both sides update, cardinality unchanged on either
  side.
- Conv up-path (`iris_blocks`/`_ConvNormAct`-equivalent, unchanged) with skip connections from
  the target's multiscale pyramid `F_q_all`, recovering full native resolution.
- Class embedding from refined `T'` dotted against the resulting per-voxel mask-feature map →
  logits. (Single logit map per forward, matching this repo's own K=1-class-per-task use —
  unlike Iris's own multi-class-per-pass design, which this repo has never reproduced.)

### Trace: Iris's own decoder needs no decompression

From `docs/methods/iris.md` §5: Eq 5's `CrossAttn(F_q, T)` only refines values on both sides —
`F'_q` stays `d·h·w` cells, `T'` stays `K(m+1)` tokens; no step changes either side's row
count. Eq 6's mask prediction recovers full resolution via the **decoder up-path's conv
upsampling + skip fusion**, not via any token-level expansion. `F_q` is never reduced to a
handful of summary tokens in Iris to begin with. This is exactly what `_decode_iris` already
implements, and exactly why v2's decode needs no Stage-C equivalent: the target's raw grid is
never discarded, so there's nothing to re-expand.

## Reused verbatim (imported, not reimplemented)

- `RowCrossAttention`, `TransformerEncoderStack`, `ThinkingRows`, `FourierPositionalEncoding`,
  RoPE helpers — from `pfn_seg_2d.py` / `patchset_pfn.py` / `rope.py`.
- `MaskConvEmbed`, `_ConvNormAct` — from `patchset3d.py`.
- `_down_to`, `_mask_tiles_3d` — from `patchset3d.py` (module-level free functions).
- Encoder classes: `ConvEncoder3D`, `PlainConvTSEncoder`, `ResEncTSEncoder`, `NnUNetTSEncoder`,
  `PrimusEncoder`, `TapCTEncoder` — unchanged.

## New shared helper: `build_encoder`

`PatchSet3D.__init__`'s encoder-selection `if/elif` (`patchset3d.py:315-387`, ~70 lines) is
extracted into a standalone `build_encoder(name, resolution, **kwargs) -> nn.Module`. Both
`PatchSet3D` and `PatchSetV2` call it instead of duplicating the dispatch logic. Pure
refactor — the dispatch logic itself is unchanged, only relocated (proposed location:
`src/models/encoders/__init__.py`, alongside the individual encoder modules it already
dispatches to). `PatchSet3D`'s own behavior/checkpoints are unaffected.

## Dropped vs `PatchSet3D` (not ported to v2)

| Dropped | Why |
|---|---|
| `register_routed`/`register_flex`/`build_register_block_mask` | Explicit requirement this round. Also structurally moot: per-cell tokens are pure KV for Stage A only, never self-attend to each other beyond their own volume's compression. |
| `mask_slots` (content-type tag, distinct from `mask_embed` — see Kept) | In `PatchSet3D`, an additive Fourier tag marking a mask column "gt" vs "pred" since support/target rows are otherwise embedded identically. Redundant once `context_id_embed` is kept: it already assigns the target row its own identity distinct from every context row, which tells the network which volume — and therefore whether that volume's mask column is GT or prediction — as a byproduct of a strictly more general signal. |
| `decode_source` | Always the target's own Stage-B slice / img-embed pair, mirroring Iris exactly. |
| `fine_decode` / `decoder` mode switch, `fine_filter`, `conv` decoders | v2 has exactly one decoder (Iris-style). |
| `pool_token` flag | Pooling is unconditionally part of every volume's sequence, not an optional prefix-row mechanism. |
| `seq_compress` toggle | Always on. Stage C / `_expand` / `self.expander` dropped entirely — nothing to re-expand (see Trace above). |
| `iris_pixelshuffle_r`, `_pixel_shuffle_3d`/`_pixel_unshuffle_3d`, `iris_ctx_conv` | Superseded by Stage A's mask-aware learned compression. |
| `full_attn` / `query_self_attn` | Stage B is always full self-attention across every volume's tokens. |
| `token_mask_ratio_support` / `token_mask_ratio_query` (SimMIM) | No current loss consumes it; out of scope until a reconstruction objective is wired in. |

## Kept / carried over unchanged

- `cascade_registers` mechanism (`cascade_proj`, `cascade_type`, prepended rows) — explicit
  requirement this round.
- Fourier positional encoding and mask embedding (`mask_embed`: linear or conv) — explicit
  requirement this round.
- `compress_m` / `compress_layers` (renamed from `seq_compress`'s knobs — always active in v2).
- `context_id_embed` — kept, for two reasons: (1) forward-looking support for K>1 context
  items needing individually distinguishable identity, not just joint pooling; (2) it now also
  carries the job `mask_slots` used to do (see Dropped table) — the target row's `qry_id` vs.
  each context row's own `ctx_id[k]` already tells the network which volume, and therefore
  whether that volume's mask column holds GT or prediction, without a separate tag.
- `fine_stage` (which encoder stages are exposed unpooled — feeds both `pool_v` and the decode
  skip pyramid).
- Encoder choice and its own sub-knobs (`encoder`, `nnunet_ts_stages`,
  `plainconv_ts_features_per_stage`, `encoder_frozen`, `encoder_input_norm`, etc.) — unchanged,
  routed through `build_encoder`.

## Known gap this design fixes

`query_prior` is currently silently ignored whenever `decoder_kind=="iris" and not
cascade_registers` (`patchset3d.py:1171`). In v2, `prev_q` is a first-class input to the
target volume's own tokenize/pool step — always consumed, cascade-prior-aware by construction,
with no separate code path to fall out of.

## Cost expectation

Stage B's self-attention operates over `T·(compress_m+1)` rows, replacing both the old
per-cell dual-axis transformer (`T·N` rows) and the iris-decoder path's separate
`iris_t2f`/`iris_f2t` reciprocal attention (which put the query's full `R³` grid on both sides
of an attention op). Expected to be substantially cheaper on the `attn`/`decode` timing
buckets `profile_timing` already tracks (see `docs/logs.md` 2026-09-18). No formal FLOPs or
wall-clock comparison has been run yet — recommend a `profile_timing` pass against the
`99_iris_decoder_plainconv_doubling` baseline once implemented.

## Config / integration point

`experiments/3d/train.py::build_model(cfg)` dispatches on `cfg.model` (currently `"medverse"`
| `"patchset3d"`, raising `ValueError` otherwise, `train.py:368-443`). Add
`if name == "patchset3d_v2":` importing `PatchSetV2` and building its own `arch` dict from
`cfg.arch` — a much smaller dict than `patchset3d`'s (no `mask_slots`, `register_routed`,
`full_attn`, `seq_compress`, `pool_token`, `fine_decode`, `decoder`, `iris_pixelshuffle_r`,
etc.). New experiment/model config files under `configs/experiment/3d/` and
`configs/experiment/3d/model/` follow the existing Hydra composition pattern.

## Open questions for the implementation plan

1. **`mask_patch_size`** (occupancy tiling granularity, `p>1`) — does v2 need it, or is `p=1`
   (single-voxel occupancy per cell) sufficient now that `pool_v` separately handles
   high-resolution mask detail via native-resolution masked pooling?
2. **`forward()` return signature** — today's `mask_support`/`mask_query` outputs exist for a
   SimMIM reconstruction loss that's dropped in v2 (see Dropped table). v2's `forward()` should
   return just `final_logit` + `registers` (for cascade use) unless a future loss needs more.
3. Whether `cascade.py`/`evaluate.py` need any changes beyond the `build_model` dispatch to
   support `patchset_v2` (native-grid eval, checkpoint loading, etc.) — needs a pass once the
   class exists; not expected to need changes given the same `predict()`/`train_forward()`
   contract `PatchSet3D` already satisfies, but unverified.
4. **Feature normalization before `img_embed`** — `PatchSet3D` always calls `_feat_norm`
   (context/self/none per-channel z-score) before its own `img_embed`; `PatchSetV2` has no
   equivalent anywhere in the tokenize path. This was an oversight during implementation, not
   a deliberate simplification — flagged by the final whole-branch review
   (2026-09-19). Needs a design decision (which mode, applied where — per-volume like
   `_pool_all`'s existing z-score, or context-relative like `PatchSet3D`'s `self`/`context`
   modes) before a real training run, since raw frozen-encoder features (e.g. `plainconv_ts`'s
   704-channel multi-scale concat) currently go straight into `img_embed` and then into
   `nn.MultiheadAttention` (which has no internal input normalization) unnormalized.

## Non-goals for this spec

- No change to `PatchSet3D` behavior — the `build_encoder` extraction is a pure refactor,
  location-only.
- No training run or benchmarking — first implementation-plan milestone, not part of this
  design.

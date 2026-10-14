# Multi-patch cascade re-crop — design

**Date:** 2026-10-13
**Status:** draft (pending spec review)
**Branch:** `multipatch-cascade`

## Goal

The cascade (`experiments/3d/cascade.py::run_cascade`) currently re-crops a
single `T^3` window per role, per level `i>=1`: one target window (centered on
level `i-1`'s predicted centroid or a single random foreground voxel,
`data.cascade_center_mode`), and one window per context subject (always
GT-centered). This is a single point of attention per level — large, diffuse,
or multi-focal ROIs (donut-shaped organs, multi-focal lesions like ISLES22
strokes or GNC kidney cysts, elongated structures) cannot be captured, because
a single window discards everything outside it.

This design generalizes the re-crop step from "1 window per role" to "N
windows per role," sampled from a continuous probability field (the previous
level's prediction for the target, GT occupancy for support) rather than a
single point, and processed in **one joint forward pass** so all N target
windows and all support windows attend to each other (PatchSet3D's row axis
is already a set-of-patches transformer — see Background). This directly
targets the known limitation documented in docs/logs.md 2026-09-12 ("COM
re-crop centering is suboptimal for large/non-convex ROIs") and memory
`project_colipri_self_ctx_ceiling` / `project_gnc_kidney_lesions`.

**First deliverable** (this spec covers only this): a **zero-shot,
inference-only** validation on the existing best checkpoint
(`results/patch_icl/3d_train/.../167_real_host_all_shapes_wide_spacing` chain
→ `168_cascade_real_host_all_shapes`, wandb run `efuqm74l`'s lineage, i.e.
whatever checkpoint `168`'s `train.checkpoint` resolves to at `best.pt`). No
training-recipe change, no new synth data. If this validates a real
Dice/NSD improvement on diffuse-ROI sources, a training-recipe follow-up
(new spec) teaches the model to expect N-window context directly instead of
relying on zero-shot generalization of attention it was trained with K+1.

## Non-goals

- No training changes (loss, data loader, synth recipe) — this spec is
  inference/eval-only.
- No change to level 0 (dataset-driven bootstrap, unaffected — there is no
  prediction yet to sample a probability field from).
- No change to `arch.decoder=iris`, `arch.seq_compress`, or
  `arch.cascade_registers` paths in `patchset3d.py` — all three are off in
  the checkpoint under test; the multi-query generalization asserts against
  `M>1` combined with any of them rather than extending code nothing
  exercises.
- No change to `mask_slots>=2`, `register_routed`, or `context_id_embed`
  semantics beyond what's needed for N>1 support windows per subject (see
  below) — the checkpoint under test has `mask_slots=1`,
  `register_routed=false`, `context_id_embed=true`.

## Background: why this is mostly inference-side surgery

`PatchSet3D`'s row axis (`src/models/patchset3d.py::_attn`) is already a flat
set of every volume's `R^3` cell-tokens — `thinking + K*N + Q*N` rows, with
full dense attention (checkpoint has `arch.full_attn=true`) across every row,
for every one of the `c` (img/mask) columns. Support is already a pure count
`K` (no model code assumes K == "number of context *subjects*"), so widening
support from "1 crop per subject" to "N crops per subject" needs **zero
model changes** — only `cascade.py`'s bookkeeping of which support row
belongs to which subject changes.

The query side is different: `_grid_tokens`, `_attn`, and `_pool_tokens`
hardcode exactly one query volume (`Q=1`, `feat[:, K:K+1]`,
`qry_ijk = self.ijk_base.unsqueeze(0).expand(B, N, 3)`, etc.). Generalizing
`Q=1` to `Q=M` is the actual model change (§1 below).

RoPE positions (`arch.transformer_rope=true`) are per-volume-local
(`ijk_base`, always `0..R-1` regardless of the volume's real-world
location) — two different windows' cell `(0,0,0)` already collide in
position today whenever K>1 (every support subject is a different anatomy
with its own locally-indexed window), and the model already relies on
content-based attention, not a shared global coordinate frame, to
disambiguate. Adding more such windows (extra support crops per subject,
multiple target crops) is consistent with what the model already handles,
not a new regime — this is why a zero-shot test is a reasonable first probe
instead of committing straight to a training change.

## 1. Model change — `src/models/patchset3d.py`

Generalize the query axis from exactly 1 volume to `M` volumes (`M>=1`,
`M=1` byte-identical to today). `M` here is the same quantity `n` in §2's
`data.cascade_n_patches` — `M` is `patchset3d.py`'s forward-signature name
for "how many query windows arrived in this call," `n` is `cascade.py`'s
config name for "how many windows to sample per role at this level"; the
model has no opinion on where its `M` windows came from.

- `forward(image, context_in, context_out, ...)`: accept `image` as either
  `(B,1,D,H,W)` (legacy 4-D-after-batch shape, `M=1`) or `(B,M,1,D,H,W)`
  (new). Detect by `image.ndim`. `T = K + M` (was `K + 1`).
- `_grid_tokens(feat_map, B, T, K)`: change
  `feat[:, K:].reshape(B, self.N, Cf)` → `feat[:, K:].reshape(B, (T - K) * self.N, Cf)`.
- `_attn`: `qry_ijk` tiles `ijk_base` `M` times (mirrors `sup_ijk`'s
  `.repeat(K, 1)`); the support-mean prior fallback
  (`sup_occ.mean(dim=1, keepdim=True).expand(B, N, ...)`) expands to
  `M * N` rows instead of `N`.
- `query_prior`: generalize from a single `(B,1,D,H,W)` soft-prob volume to
  `(B,M,1,D,H,W)` (or `None`) — `_prior_occupancy` runs per query window and
  concatenates, giving `(B, M*N, p^3)` instead of `(B, N, p^3)`.
- `_pool_tokens`: `sup_feat, qry_feat = feat[:, :K], feat[:, K:K+1]` →
  `feat[:, K:K+1+?]`... becomes `feat[:, K:]` (all `M` query rows); the
  `query_prior` mask branch and the `masked_avg` reduction both need the `M`
  axis threaded through (currently assumes exactly one query row when
  building `qry_mask`).
- `_attn`'s `pool_feat`/`cascade_regs` concatenation paths that assume
  exactly one query row in the `K+1` pool/mem tensors (`ctx_tag`
  concatenation, `pred_tag.unsqueeze(0)`, etc.) generalize the trailing `1`
  to `M`.
- Decode needs **no new code**: after `_attn`, reshape the `M` query
  volumes' token output `(B, M*N, e)` → `(B*M, N, e)` and call the existing
  `_decode`/`_decode_conv` unchanged (decode has no cross-volume structure).
  The `fine` maps threaded into `_decode_conv` get the same `(B*M, ...)`
  reshape on their query slice.
- Add an assertion: `M > 1` requires
  `decoder_kind != "iris" and not seq_compress and not cascade_registers`
  (the combinations this spec explicitly doesn't generalize).

Verification for this section: a unit test forwarding `M=1` through both the
legacy 4-D `image` shape and the new 5-D `(B,1,1,D,H,W)` shape and asserting
byte-identical logits; a second test with `M=3` synthetic windows asserting
output shape `(B,3,1,G,G,G)` and that decode is permutation-equivariant in
the `M` axis (shuffling the 3 windows' order shuffles the 3 output logits
identically, i.e. no cross-window leakage in decode itself — only in
`_attn`'s shared transformer, which is the intended effect).

## 2. Cascade re-crop mechanics — `experiments/3d/cascade.py`

Level 0 unchanged. For `i>=1`, add a `data.cascade_n_patches` config knob
(default `1`, preserving today's exact behavior) controlling `n`, the
number of windows sampled per role at that level.

- New `_prob_weighted_points_multi(fields, T, is_prob, n, *, seed, step,
  level)`: sibling of `_random_fg_point`/`_centroid_from_logit`, but takes
  `fields`, a **list of M** `(B,1,T,T,T)` logit/prob tensors (one per
  level-`i` window) instead of one. Per `b`, flattens and concatenates all
  `M` windows' probabilities into one length-`M*T^3` vector, renormalizes
  to a simplex (empty → `None`, same fallback convention as
  `_random_fg_point`), and draws `n` indices via
  `torch.multinomial(flat, n, replacement=True)`. Each drawn flat index
  decodes to `(window_idx, local_voxel)` via `divmod(idx, T**3)` then
  `np.unravel_index` — so the return is, per `b`, a list of `n`
  `(window_idx, (d,h,w))` pairs. Overlapping windows naturally accumulate
  *more* sampling mass where both predict foreground — no separate
  "stitch a native array" step needed, and nothing here differs between
  train and eval (pure seeded tensor ops, like the function it generalizes).
- `invert_geo_center` is called once per sampled point, using **that
  point's own originating window's** `geo` (captured aug grid) and
  `crop_geom` — `run_cascade`'s level loop already tracks one `geo`/`geom`
  per window per `b`; for `M` windows it tracks `M` of each instead of one,
  and `_prob_weighted_points_multi`'s `window_idx` picks which one each
  sampled point inverts through. This is the exact mechanism already used
  for the `M=1` case (there, `window_idx` is always `0`), so `n=1, M=1`
  reproduces today's single-point inversion byte-for-byte.
- Support-side sampling needs **no new sampling code at all**: GT is
  already native and binary, so "`n` points from a continuous density"
  *is* `n` independently-seeded calls to the existing
  `random_fg`-mode provider path (`src/providers/totalseg.py`'s
  `_resolve_center`, already wired for `NativeGridProvider`/FLARE22 too) —
  same `center=None, center_mode="random_fg"` `LoadRequest`, `n` different
  RNG seed suffixes (`f"{seed}_{step}_{level}_{b}_{k}_{j}"`, `j in
  range(n)`) instead of one. `context_id_embed`'s per-subject tag (`ctx_id`)
  is assigned **per original context subject `k`**, not per sampled window,
  so all `n` windows of context subject `k` share one `ctx_id` row — matches
  the embedding's existing semantics (distinguish *subjects*, not crops)
  and needs no model change.
- `_recrop_level`: currently builds one `(b, -1, subj, center, seed, mod)`
  task for the target and one `(b, k, ctx_subj, None, seed, mod)` task per
  context index `k`. Generalize to `n` target tasks per `b` (centers from
  `_prob_weighted_points_multi` on the `M` level-`i-1` target windows'
  predictions) and `n` tasks per `(b, k)` context slot (centers from `n`
  independently-seeded `random_fg` provider calls) — i.e. the flat task
  list grows from `B*(K+1)` to `B*(K+1)*n`.
- `run_cascade`: `_forward_level` now passes `image` as `(B, n, 1, T, T, T)`
  (the `n` sampled target windows) instead of `(B,1,T,T,T)`, and
  `context_in`/`context_out` grow from `K` to `K*n` rows (same subject,
  repeated `ctx_id`, different window). `n=1` reproduces today's tensor
  shapes exactly (`image.squeeze(1)` degenerates to the legacy 4-D shape),
  so this is a strict generalization, not a parallel code path.
- `CascadeResult` gains `n_patches: list[int]` (per level, `1` for level 0)
  so `evaluate_cascade`/figure code can tell which levels ran multi-patch.

Verification for this section: `data.cascade_n_patches=1` run against the
existing `test_cascade.py` suite — byte-identical logits/Dice to current
`main` (regression guard that the generalization is truly a no-op at
`n=1`). A new synthetic two-disjoint-blob test (mirrors the existing
`random_fg` two-blob smoke test) asserting that at `n=2` both blobs receive
a sampled window (not just one), and that the stitched probability field
after one level has nonzero mass over both blobs.

## 3. Validation — the actual deliverable

A comparison script (new, reuses `cascade.evaluate_cascade` twice — once
with `data.cascade_n_patches=1`, once with `>1` — against the same
checkpoint, same eval seed) run on:

- **FLARE22** (`src/providers/flare22.py`, `dataset=flare22` in `eval.py`,
  13 abdominal organ classes: liver, kidney L/R, spleen, pancreas, aorta,
  IVC, adrenal L/R, gallbladder, esophagus, stomach, duodenum, native
  anisotropic ~0.8×0.8×2.5mm). Unlike ISLES22/GNC_kidney (multi-focal
  lesions), FLARE22's organs are mostly solid single-focus structures — this
  is an **unseen-acquisition-centre generalization check**, not a
  diffuse-ROI stress test: does sampling N windows from a continuous
  probability field help or hurt ordinary organ localization under domain
  shift (different scanner/protocol than TotalSegmentator), where a single
  well-centered window is already close to sufficient. A regression here
  (multi-patch underperforms single-patch on solid organs) would be as
  informative as a win on a diffuse ROI — it would mean the generalization
  needs a training-time fix (support window diversity, decode
  disambiguation) before being safe to turn on universally, consistent
  with the §1 note that attention across many same-content windows is new
  territory the checkpoint wasn't trained against. `duodenum`/`esophagus`
  (thin, elongated — closer to the diffuse case) and `liver`/`spleen`
  (large solid, closest to today's working case) are worth separating out
  in the per-class breakdown rather than only looking at the macro mean.

Report per-class Dice/NSD deltas (`n_patches=1` vs `n_patches∈{2,4}`,
sweeping `n` is cheap since it's inference-only) and a qualitative pass over
a few worst-case samples: does the failure mode shift from "missed the
second focus entirely" (visible in the stitched prediction) to something
else. No success threshold is pre-committed in this spec — the point of a
zero-shot probe is to learn whether the direction is worth a training
investment, not to hit a target number.

## Risks / open questions (flagged, not blocking)

- Compute: `n` windows per role means `n*(K+1)` volumes through the encoder
  and an `O((n*(K+1)*N)^2)`-ish dense attention (checkpoint has
  `arch.full_attn=true`, no register-routing) — likely fine for
  inference-only probing at small `n` (2-4), but should be measured, not
  assumed, before any training-recipe follow-up.
- The GT-occupancy sampling for support is binary (uniform-over-foreground
  generalization of `random_fg`), not a genuinely "continuous" density
  distinguishing e.g. core-vs-boundary — fine for a first probe; a
  training-recipe follow-up might want a smoother support density (e.g.
  distance-transform-weighted) if zero-shot validates the direction.
- Letting overlapping windows' probability mass accumulate in the
  concatenated-multinomial sampler (§2) means a region two windows agree
  on is sampled roughly twice as often as one only one window covers —
  a design choice, not a proven-better one vs. e.g. deduplicating by native
  coordinate first. Flagged here for the validation pass to actually look
  at, not re-litigated before trying it.

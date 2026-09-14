# Differentiable cascade re-crop sampling — feasibility notes

**Status:** investigation only, not designed/approved. Captures the 2026-09-12 brainstorm on
whether `run_cascade` (`experiments/3d/cascade.py`) can be made to backprop a finer level's
loss into a coarser level's prediction, and how. No code changes yet — this is the reference
to work from if/when we pick it up.

**Related:** `docs/logs.md` 2026-09-12 entry (`data.cascade_center_mode`, the non-differentiable
random-fg sampler shipped from this same discussion); `src/models/scatter_sampling.py` +
`docs/superpowers/specs/2026-07-13-scatter-refine-sampling-design.md` (the existing Gumbel-topk
sampler this was compared against).

## Motivation

Today each cascade level trains against its own loss only — `_cascade_loss` (`cascade.py`) is a
plain weighted sum `Σ w_i · loss_fn(logit_i, target_i)`, and nothing connects `logit_i` to
`logit_{i+1}`'s computation graph. A finer level's loss has no way to tell a coarser level "you
cropped/centered me badly." We asked: could `level_{i+1}`'s loss flow back into `level_i`'s
prediction, so the cascade trains as one connected system instead of N independent single-level
models glued by a non-differentiable re-crop? We'd previously built a Gumbel-topk differentiable
*sampler* for a different purpose (`src/models/scatter_sampling.py`, the `PatchSetCNN` scatter
refine) — the question was whether that mechanism (or something like it) transfers here.

## What's already differentiable (not the blocker)

Traced every step between a level's logit and the next level's forward:

- `_centroid_from_logit`'s prob-weighted COM (`cascade.py`) is a plain differentiable weighted
  mean of `sigmoid(logit)`. It's non-differentiable today only because of an explicit
  `.detach().cpu().numpy()` at the end — not because the math requires it.
- `invert_geo_center`'s augmentation-undo (`cascade.py`) is `grid_sample` (differentiable w.r.t.
  its query coordinates) composed with flip/crop-geom affine arithmetic (`+`,`-`,`*`,`/`) — all
  continuous. Currently wrapped in `.detach()`/fp32-no-grad blocks by choice, not necessity.
- This exactness only holds for **affine-only** augmentation (flips + rotation) — the docstring
  already says the M2 warp's residual "grows with deform/elastic." `88_cascade.yaml` /
  `89_multisource_cascade.yaml` already run with `deform: {p: 0}` / `elastic: {p: 0}`, so this
  holds for the configs that matter today.

## What actually blocks it

1. **The predicted center gets rounded to an integer voxel** (`invert_geo_center`'s
   `int(round(...))`) before it's used.
2. **The re-crop itself is CPU/numpy disk I/O**, entirely outside autograd: `_recrop_level` →
   `provider.load` / `load_native_crop` → `np.load(mmap)` + literal array slicing
   (`crop_ct[d0:d0+cs, ...]`), run in a thread pool. Even with a continuous, gradient-carrying
   center, slicing a numpy array at that location has no gradient — nothing computes
   `d(content)/d(coord)` unless the read is itself a differentiable resample.
3. **(Separate, already-deliberate)** `_build_query_prior`'s `pred` mode explicitly detaches
   (`src = prev_logit.detach().float()`, "each level keeps its own loss") — a second,
   intentionally-cut cross-level path. Not what this investigation targets, but the other place
   information crosses levels, and worth reconsidering in the same pass if this is ever built.

`scatter_sampling.py`'s Gumbel-topk doesn't solve this either, even reused verbatim: it's a
**hard `.topk`** over a score, and the productionized `PatchSetCNN` model explicitly `.detach()`s
the coarse map before scoring. Its job there is *reproducible stochastic cell selection*
(sampling WHICH cells to compute a sparse loss on), not a gradient path back to what produced the
score. Reusing it as-is would give stochastic center selection (which is what
`data.cascade_center_mode="random_fg"` already does, shipped separately) but no
`level_{i+1} → level_i` gradient signal.

## Two ways to actually get the gradient

### A. Reparameterized differentiable recrop — recommended, unifies with `cascade_center_mode`

Keep level-0's native (or lightly decimated) CT+label buffer resident on GPU per target item
across the whole cascade — crop physical extent only shrinks level-to-level (spacing decreases
while grid size `T` stays fixed), so a level-0 window very likely already covers every finer
level's need. Re-crop levels ≥1 for the **target only** (context never needs gradient — it
doesn't depend on any prediction) via `F.grid_sample` against that buffer instead of re-reading
from disk, with a sampling grid built from a **continuous** center.

Generalize the random-fg idea into a **Gumbel-softmax soft-argmax**:

```
w = softmax((log(p + ε) + gumbel) / τ)     over candidate voxels
center = Σ w · coord
```

— differentiable, stochastic, and it interpolates between the two things we want: `τ → ∞`
approaches today's deterministic COM; `τ → 0` approaches a genuine random draw from the mask
(the `cascade_center_mode="random_fg"` behavior, but soft). One mechanism gets both properties,
and ordinary `loss.backward()` naturally pulls `loss_{i+1}` back into `logit_i` — no extra loss
term needed.

**Cost / scope:**
- A real architecture change: a new GPU-resident buffer per target item + a **batched** (not the
  current per-`b` python loop) differentiable recrop path. The existing `_warp_prior_m2`/
  `_fit_grid_affine` per-`b` loop with `torch.linalg.lstsq` is not compile/graph-friendly and is
  only tolerable today because it lives on a small aux path outside the main graph (the
  `query_prior` warp) — a differentiable version living *inside* the trained graph should be
  vectorized over `B` instead (for affine-only aug we know the transform params analytically
  from the aug config, so this doesn't need to fit an arbitrary captured grid at all).
- Memory grows by one retained buffer per item; should stay bounded if decimated to "fine enough
  for the finest level" — same idea as `NativeCrop.decim` today, just retained across levels
  instead of discarded per-level.
- Only the **target** path needs rework; contexts keep the existing disk-backed, non-
  differentiable `_recrop_level` path unchanged (no gradient needed there regardless of mode).

### B. Score-function (REINFORCE) fallback

Leave the entire existing recrop pipeline untouched (disk-backed, non-differentiable, discrete
center) — REINFORCE doesn't need to differentiate through the crop content at all. Add
`-log p(center | logit_i) · reward_{i+1}` to level `i`'s loss, where `reward_{i+1}` is a
*detached* scalar (e.g. `-loss_{i+1}` or a Dice delta) and `p(center | logit_i)` is the
(differentiable) probability of the sampled center under a categorical/Gumbel distribution built
from `logit_i`.

**Cost / scope:** zero changes to `_recrop_level`/providers/disk I/O — much smaller footprint.
Brings the usual RL-training headaches though: reward baseline/variance reduction, reward scaling
across levels, and it's a materially noisier signal than true backprop.

## Recommendation

**A** if we want a real signal — more work, but it's the only one that's actually "level
`i+1`'s loss refines level `i`," and it happens to unify cleanly with the shipped random-fg
sampling via one temperature knob (a single soft-argmax mechanism covers both the existing
"crop over the mask" feature and this gradient-flow goal). **B** is cheap to bolt on but is a
materially different (weaker, RL-flavored) mechanism — worth reaching for only if A's buffer/
compile cost turns out prohibitive on inspection.

## Open questions / next steps (before designing)

- Does the level-0 window reliably cover every finer level in practice, or does the predicted
  center drift outside it often enough (e.g. under a bad early-training coarse prediction) that
  a plain "retain level-0's crop" buffer isn't enough, and a slightly larger margin buffer is
  needed?
- Precision/memory budget for the retained buffer at the finest configured spacing (`88`/`89`'s
  ladder is `[6, 3, 1.5]`) — worth a quick bench before committing to a design.
- Should `_build_query_prior`'s `pred`-mode `.detach()` be revisited in the same pass, or kept
  cut (current framing: "each level keeps its own loss")? Doing both at once conflates two
  different signals into one gradient path and would be harder to attribute if results move.
- Whether the Gumbel-softmax soft-argmax should replace `cascade_center_mode="random_fg"`
  outright (one mechanism, `τ` config knob) or ship alongside it as a separate opt-in mode —
  affects whether this is additive or a refactor of the existing feature.

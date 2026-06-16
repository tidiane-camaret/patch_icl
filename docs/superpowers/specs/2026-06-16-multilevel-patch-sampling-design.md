# Multilevel patch sampling reformulation — threshold core + neighbor fill

Date: 2026-06-16
Experiment: `experiments/2d/multilevel` (stage-2 `PatchSetPFN` refinement)

## Problem

The current sampler (`sampling.py:sample_patch_indices`) splits the `M` query/support
patches into a **fixed** `n_uncertain=192` cells closest to 0.5 plus `n_certain=64`
cells farthest from 0.5. Two issues observed:

1. **Fixed-budget overflow.** Genuine boundary cells average ~127 (12.4% of 1024) — fewer
   than `n_uncertain=192`. Under `ds_gt`, pure cells saturate at `d=0.5`, so once the
   real boundary runs out the "uncertain" bucket pads with arbitrary tie-broken pure
   cells. The "uncertain" set is therefore boundary ∪ arbitrary padding.
2. **No spatial coverage control.** The selected cells can be scattered, leaving holes in
   the refined output surface (only the `M` sampled cells get composited into the final
   map; the rest stay coarse).

## Goal

Replace the certain/uncertain split with: a **variable-size uncertain core** (cells whose
sampling-map value is near 0.5, by threshold), and **fill the remaining budget with
patches sampled near the core** so the refined surface has fewer holes — while keeping the
total `M` constant for batched tensors.

## Decisions (from brainstorming)

- **Budget**: fixed total `M=256` per image. Core has two tiers — a threshold-based
  **boundary core** (variable count) plus a fixed **fg-core quota** of `n_fg_core` random
  foreground cells. Neighbors fill the remainder.
- **fg core**: motivated by the coverage finding that the boundary core alone misses most
  GT foreground (large-object interiors). A fixed quota of `n_fg_core` random foreground
  cells (value≥0.5, excluding boundary core) is force-promoted into the core, independent
  of the neighbor field. It is the most robust fg-coverage contributor across the GT→pred
  swap (random predicted-fg cells are mostly true fg). Default `n_fg_core=64`; see Tuning.
- **Fill method**: blurred proximity field + Gumbel-top-k (stochastic, biased near the
  boundary core, with a uniform floor so far cells retain a chance).
- **Scope**: applies to **both** query (target) and support (context) selection. The old
  `sample_patch_indices` is retired.

## Algorithm

A single combined score + one top-k handles per-image variable core count without ragged
tensors. Three priority tiers (boundary core > fg core > neighbors):

```python
d      = (values - 0.5).abs()                # (B,N) distance from boundary
core_b = d < tau                             # tier 1: boundary core (variable count)

# tier 2: fixed quota of random foreground cells (excluding boundary core)
fg_core = zeros_like(core_b)
if n_fg_core > 0:
    fg_pool = (values >= 0.5) & ~core_b
    key     = where(fg_pool, rand_like(values), -1.0)
    take    = key.topk(n_fg_core, dim=1).indices
    fg_core = zeros_like(core_b).scatter_(1, take, True) & fg_pool   # guard: <n_fg_core fg

# tier 3: neighbor proximity field around the boundary core
g     = gaussian_blur(core_b.float(), sigma) # (B,N) high near boundary core
w     = g + floor                            # floor → far cells keep a chance
noise = sample_gumbel(w.shape) if stochastic else 0
neigh_score = (w + 1e-12).log() + temperature * noise

BIG_B, BIG_F = 2e4, 1e4                       # boundary > fg core > neighbors
score = where(core_b, BIG_B - d, where(fg_core, BIG_F, neigh_score))
idx   = score.topk(n_total, dim=1).indices    # (B, n_total) — fixed count
is_fg_core = fg_core.gather(1, idx)           # confident-fg anchors
is_core    = (core_b | fg_core).gather(1, idx)
is_boundary = is_core & ~is_fg_core           # genuinely uncertain → metric scope
```

Case behaviour (all in one vectorized top-k — one conv + one topk, no Python loops):

- `boundary + n_fg_core ≤ n_total` → all core selected; remainder are the highest-scoring
  neighbors (proximity-biased random).
- `boundary count > n_total` → only the `n_total` most-boundary cells survive (ranked by
  `BIG_B − d`); fg core and neighbors squeezed out. Budget-capped gracefully.
- `boundary count == 0` → `g≈0`, `w≈floor` → neighbor fill degenerates to uniform random
  (after the fg quota). Sensible fallback, no crash.
- fewer than `n_fg_core` fg cells in the image → the `& fg_pool` guard drops the shortfall;
  those slots go to neighbors instead.

Knobs: `tau` = boundary core width; `n_fg_core` = forced fg-interior coverage; `sigma` =
how tightly neighbors hug the boundary; `floor` = global coverage; `temperature` =
neighbor randomness.

### `gaussian_blur` helper

Separable fixed Gaussian kernel applied via `F.conv2d` with reflect padding. Operates on
`(B,N)` reshaped to `(B,1,grid_res,grid_res)` and flattened back. Kernel size
`2*ceil(2*sigma)+1`, precomputed from `sigma`. Reflect padding avoids edge bias.

## Config (`configs/experiment/2d/multilevel.yaml`)

Replace `n_uncertain` / `n_certain` with:

```yaml
sample:
  grid_res: 32
  n_total: 256          # M: total patches per image (constant)
  tau: 0.30             # boundary core: cells with |value-0.5| < tau (tuned, see Tuning)
  n_fg_core: 64         # fixed quota of random foreground cells forced into the core
  blur_sigma: 1.0       # neighbor proximity width (grid cells)
  floor: 0.005          # uniform floor → far cells keep a chance
  temperature: 1.0      # gumbel temperature for neighbor fill
  eval_deterministic: true   # drop gumbel at eval for reproducible metrics
  train: prev_pred      # prev_pred | ds_gt  (unchanged — sampling-map source)
  eval: prev_pred
```

## Tuning (sweep findings)

Defaults were chosen from a `tau × sigma × floor` sweep over the full MedSegBench val
set (13,237 images), comparing the two sampling-map sources at their correct resolutions
(`experiments/2d/multilevel/plot_sampling.py --sweep`). Coverage metric = fg→miss (% of
true GT foreground patches that are never sampled; lower is better), fg always from true GT.

Head-to-head at `sigma=1.0, floor=0.005`, `M=256, grid=32, n_fg_core=64`:

| tau  | ds_gt fg→miss | prev_pred fg→miss | prev_pred core %fg | prev_pred fg→neigh |
|------|---------------|-------------------|--------------------|--------------------|
| 0.15 | 39.0%         | 41.2%             | 66.4%              | 14.8%              |
| 0.30 | 36.7%         | **36.2%**         | 57.4%              | 4.2%               |
| 0.45 | 36.3%         | 41.2%             | 46.2%              | 0.9%               |

Key points:

- **`tau=0.30` is the deployable optimum.** Under `prev_pred` (real frozen stage-1) it
  reaches fg→miss 36.2%, matching the GT-oracle ceiling (36.7%). It cannot be read off the
  oracle: `tau=0.45` is near-best for `ds_gt` (36.3%) but *regresses* to 41.2% for
  `prev_pred`.
- **Neighbor fill collapses as tau grows under prev_pred** (fg→neigh 14.8% → 4.2% → 0.9%):
  a wide predicted boundary band is fat/misplaced, leaving few neighbor slots and gluing
  them to the wrong edge. At `tau=0.30`, prev_pred compensates via a fatter core
  (fg→core 59.6% vs oracle 49.9%) — the blurry res-16 band sweeps in more true interior.
- **`floor=0.005` and `sigma≥1.0`** are best in both regimes; `sigma=0.5` and high `floor`
  only hurt; little leverage above `sigma=1.0`.
- The other lever on the ~36% fg→miss floor is the budget `M` relative to `grid_res²`
  (res-64 at `M=512` covers a smaller fraction → fg→miss ~60%); not pursued here.

## Integration

- **`sampling.py`**: add `sample_patches(values, n_total, tau, n_fg_core, blur_sigma,
  floor, grid_res, temperature=1.0, stochastic=True)` returning
  `(idx, is_core, is_fg_core)`; add `gaussian_blur(x_flat, grid_res, sigma)`. Retire
  `sample_patch_indices`. Keep `idx_to_ij` and `gather_grid` unchanged. (A reference
  implementation, validated against the stats, lives in `plot_sampling.py`.)
- **`pipeline.py`**:
  - Query path (currently line 79): call `sample_patches` on `sampling_map`
    (`gt32` if `sampling_source=="ds_gt"` else `coarse_flat`), `stochastic` driven by the
    train/eval determinism flag. Set `qry_is_uncertain = is_core & ~is_fg_core`
    (the **boundary core only** — the genuinely uncertain cells), so the
    `refine/uncertain` metric is not inflated by the confident fg-core anchors.
  - Support path (currently line 91): call `sample_patches` on `ctx_frac`
    (batched over `B*K`). Support is always GT-based (context masks known); the core
    flags are unused for support.
  - Thread `n_total`, `tau`, `n_fg_core`, `blur_sigma`, `floor`, `temperature`, and the
    determinism flag from `cfg.sample`. `build_patch_batch` already takes
    `sampling_source`; add a `stochastic` argument (True in train,
    `not cfg.sample.eval_deterministic` in eval).
- **`train.py`**: extend run-name tag with `_tau{tau}_fg{n_fg_core}`; log change to
  `docs/logs.md`.

## Metrics impact

- `refine/uncertain/*` scope = the **boundary core** (`is_core & ~is_fg_core`), so its
  denominator now varies per image (true boundary size) instead of a fixed 192. The
  per-`b` boolean indexing in `run_eval` (`u = unc[b]; if u.any(): ...`) already handles
  variable-size cores — no code change needed there.
- The fg-core anchors are confident foreground, deliberately excluded from the `uncertain`
  scope; they still count in the `sampled` (all 256) and `full`/native `dice/mean` scopes.
- Not directly comparable to old runs' `uncertain` numbers (different denominator). The
  `sampled` and native `dice/mean` scopes remain comparable.

## Out of scope

- No change to `PatchSetPFN`, the stage-1 model, or the fusion/compositing logic.
- No change to the `prev_pred | ds_gt` sampling-map source mechanism (added previously).
- `M`-varies-per-image (ragged tensors) was considered and rejected — keeps batching simple.

## Testing

Unit-test `sample_patches` on synthetic `(B,N)` maps for the invariants:
- output shape `(B, n_total)`, indices unique per row;
- `core_count==0`, `core_count<n_total`, `core_count>n_total` all return exactly
  `n_total` indices;
- with a single core cell and small `sigma`, the filled neighbors are spatially adjacent
  to it more often than chance (proximity bias);
- `is_boundary` (= `is_core & ~is_fg_core`) matches `d < tau` at the returned indices;
- with `n_fg_core>0`, exactly `min(n_fg_core, #fg outside boundary)` cells are flagged
  `is_fg_core`, all with `value>=0.5`; with `n_fg_core=0` none are.

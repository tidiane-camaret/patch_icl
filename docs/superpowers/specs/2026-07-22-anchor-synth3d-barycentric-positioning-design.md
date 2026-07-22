# anchor_synth3d: barycentric multi-anchor object positioning

**Date:** 2026-07-22
**Status:** approved (design)
**Supersedes:** the single-anchor offset positioning in
`docs/superpowers/specs/2026-07-22-anchor-synth3d-design.md`.

## Motivation

`anchor_synth3d` draws a synthetic object at a consistent position relative to a
landmark organ, on real CT backgrounds; the model must infer that anchor-relative
rule from the K context examples. The current mechanism places the object at
`centroid + offset · extent`, where `extent` is the anchor bbox's **per-axis side
length**. That side length is **orientation-dependent** — reorienting the scan
changes each axis extent — so the intended "anchor-relative" position is not a
stable, frame-invariant quantity.

Object **size** was already decoupled from the anchor (independent draw). This
change decouples **position** from any single anchor's orientation-dependent
geometry by expressing it in **barycentric coordinates over 4 landmark organs**,
which is invariant to rigid reorientation and to global scale.

## Mechanism

### Task construction (subject-first)

Per item, with a seeded RNG (`item_rng`):

1. Pick a **target subject** (train: random from `eligible_subjects`; val:
   enumerated — see Determinism).
2. `present = classes_present_in(target) ∩ anchor_pool` (pool = resolved
   `anchor_classes`, `[] = all`). The target is eligible only if
   `len(present) ≥ n_anchors`.
3. Up to `max_select_tries`, sample `n_anchors` distinct classes from `present`
   and compute their **co-occurrence set** `cooccur = (⋂ subject_sets[c]) − {target}`
   (subjects containing all selected anchor classes). Prefer a set with
   `len(cooccur) ≥ context_size`; accept the last sampled set otherwise.
4. **Contexts:** sample `context_size` subjects from `cooccur`; if `cooccur` has
   fewer than `context_size`, pad by resampling from it; if `cooccur` is empty,
   fall back to target self-context (rare; matches current behaviour, leakage
   flagged only in that degenerate case).
5. The **same `n_anchors` anchor classes** are used for all K+1 scenes — required
   so the barycentric frame is consistent across target and contexts.

The anchors are landmarks only; they are never labeled. The label is the drawn
object(s).

### Position (barycentric, weights shared across scenes)

Drawn **once per object** in `_draw_specs` (shared across the K+1 scenes):

- Base convex weights `u ~ Dirichlet(weight_concentration · 1_{n})`.
- Affine expansion around the barycenter to allow mild extrapolation outside the
  hull:
  `w = 1/n + (1 + extrapolation) · (u − 1/n)` ⟹ `Σ wᵢ = 1`, some `wᵢ` mildly
  negative when `extrapolation > 0` (`extrapolation = 0` ⟹ strictly inside hull).

Per scene, in `_render_subject`:

- Load the scene subject's resized label volume once; compute each anchor class's
  **bbox centroid** `cᵢ` (via `anchor_stats`, reused for the centroid only).
- `center = Σ wᵢ · cᵢ` (element-wise over the `(n, 3)` centroid array), then
  clamped so the `size³` object tile stays fully in-bounds.

`extent` is no longer used for positioning. `offset_range` and
`offset_to_center` are removed; a new `barycentric_center(centroids, weights,
size, vol_shape)` replaces them in `draw.py`.

Per-object **size** (independent draw), **shape** geometry, and **contrast** are
unchanged and still shared across scenes; **rotation** (`rotate_jitter`) and a
small size `scale_jitter` still vary per scene.

### Edge cases

- **Anchor empty at 128³:** the scan cache is built from native-res labels, so a
  small anchor may vanish after resize. If any of the `n_anchors` centroids is
  absent in a scene, that scene draws **no object** (empty label). Mitigated by
  keeping `anchor_classes` to reasonably large organs. Rare; counted, not fatal.
- **Near-coplanar centroids:** relying on the in-bounds clamp to bound extreme
  extrapolated positions; no explicit degeneracy check (YAGNI). A `min_tetra_vol`
  reject can be added later if placements look unstable.

## Configuration (`configs/experiment/3d/dataset/anchor_synth3d.yaml`)

New / changed keys under `anchor_synth`:

```yaml
n_anchors: 4              # landmark organs defining the barycentric frame
extrapolation: 0.3        # affine expansion around barycenter (0 = strictly inside hull)
weight_concentration: 1.0 # Dirichlet alpha for base weights (1 = uniform on simplex)
max_select_tries: 20      # retries to find a co-occurring anchor set
anchor_classes: []        # allowed anchor POOL ([] = all) — object not tied to one class
# removed: offset_range
```

Unchanged: `object_source`, `shape`, `n_objects`, `object_size_min`,
`object_size_max_frac`, `scale_jitter`, `rotate_jitter`, `contrast_delta`,
`edge_blur`, `boundary_complexity`, `eval_subjects_per_task`,
`eval_seed_namespace`, `epoch_length`.

## Eval / determinism

- `label_name = object shape` (`blob` / `elongated` / `tubular`), so validation
  Dice is grouped **by shape**.
- Val class list = distinct shapes the config emits: `("blob","elongated",
  "tubular")` when `shape=mix`, else `[shape]`. A helper `anchor_shapes(cfg)`
  provides this; `train.py`'s anchor_synth3d val branch and `make_eval_loader`
  use it instead of the anchor pool. `resolve_anchor_classes` is retained but only
  for the **anchor pool**, not val grouping.
- Deterministic val index:
  `_eval_index = [(target_subject, s) for target_subject in eligible_val_subjects
  for s in range(eval_subjects_per_task)]`, each item seeded by
  `SeedSequence([eval_seed_namespace, subject_idx, s])`.

## Data structures (init)

Built once in `AnchorSynth3DICLDataset.__init__` from the parent's
`label_to_subjects` / scan cache:

- `subject_sets: dict[class -> set[subject]]` (for cheap co-occurrence
  intersection).
- `subject_to_classes: dict[subject -> set[pool class]]`.
- `eligible_subjects: list[subject]` = subjects with `≥ n_anchors` poolable
  present classes (train sampling pool and val enumeration base).

## Meta (debug provenance)

Item `meta` gains `anchors` (the `n_anchors` class names) and `weights` (the
shared barycentric weights) alongside the existing `shapes` / `contrasts`. Used by
plot/analyze scripts only — not by metric reporting.

## Files to change

- `src/datasets/anchor_synth/draw.py` — add `barycentric_center`; remove
  `offset_to_center`.
- `src/datasets/anchor_synth/dataset3d.py` — init co-occurrence structures +
  `eligible_subjects`; subject-first selection in `__getitem__`; weights in
  `_draw_specs`; barycentric center + multi-anchor centroid load in
  `_render_subject`; `label_name = shape`; extended meta. Remove `offset_range`;
  add `n_anchors`, `extrapolation`, `weight_concentration`, `max_select_tries`.
- `experiments/3d/common.py` — forward new knobs; `anchor_shapes(cfg)` helper;
  val-class resolution → shapes.
- `experiments/3d/train.py` — anchor_synth3d val branch uses `anchor_shapes`.
- `configs/experiment/3d/dataset/anchor_synth3d.yaml` — new knobs; drop
  `offset_range`.
- `experiments/3d/plot_dataset_items.py`, `analyze_object_blend.py` — captions.
- `src/datasets/anchor_synth/test_dataset3d.py`, `test_wiring.py` — update
  constructor/cfg keys; add a co-occurrence / barycentric-placement test.

## Testing

- Unit: `barycentric_center` — weights summing to 1 reproduce the barycenter at
  `u = 1/n`; a one-hot weight lands on that centroid; extrapolation with a
  negative weight lands outside the hull; result always in-bounds.
- Dataset (synthetic fake root with ≥ n_anchors co-occurring classes): item
  contract (shapes/dtypes), object drawn (`label.sum() > 0`), same anchor set
  across target+contexts, `label_name` ∈ shapes, determinism across instances.
- Occupancy/orientation sanity left to the existing probe workflow.
```

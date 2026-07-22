# anchor_synth3d — anchor-relative synthetic objects on real CT

**Date:** 2026-07-22
**Status:** design approved, pending implementation plan

## Motivation

`omnisynth3d` (`src/datasets/omniSynth/dataset3d.py`) composes in-context tasks by
pasting bbox-cropped TotalSegmentator organs at **random** positions onto a
black/noise canvas. The only structure shared across the K+1 target/context
examples is the target organ *class*; there is no anatomical background and no
positional coherence.

`anchor_synth3d` inverts what is real vs. synthetic. It keeps the **real CT
volume** as a coherent anatomical background, and defines the segmentation task
as a **synthetic object drawn at a consistent position relative to a shared
anatomical landmark** (the "anchor" organ). Coherence across the K+1 examples
comes from the object always sitting at the same anatomical position relative to
the anchor; task diversity comes from the per-task object shape, offset, and
contrast being drawn randomly.

This gives the model tasks whose *answer* is not derivable from intensity alone —
the object blends into the background, so it can only be localised via the
anchor-relative position that the K context examples demonstrate.

## Core idea

Per item:

1. Pick an **anchor class** (class-balanced), then pull **K+1 subjects that all
   contain it** — exactly the existing class-balanced path of
   `TotalSegInContextDataset`.
2. Draw a **task spec once**, shared across all K+1 scenes: an offset vector, a
   base blob shape, a scale, and a contrast delta.
3. In each subject's real CT volume, locate the anchor organ, place the object at
   `anchor_centroid + offset · anchor_extent`, size it as `scale · anchor_extent`,
   blend it in, and emit its mask as the label.

The **anchor organ is never labeled** — it is a pure landmark. The only labeled
structure is the drawn object. The image is the real CT (the anchor organ is
visible in it, but not in the label); the model must learn from context that the
target is the blob near the anchor, not the anchor itself.

## Architecture (approach A: subclass the existing dataset)

New code lives in a small package `src/datasets/anchor_synth/`:
`dataset3d.py` (the dataset), `shapes.py` (object generators), `draw.py`
(placement + compositing helpers). This mirrors `controlSynth/`'s `shapes/`
subpackage convention while keeping the dataset the single entry point.

### `src/datasets/anchor_synth/dataset3d.py` — `AnchorSynth3DICLDataset(TotalSegInContextDataset)`

Subclasses `TotalSegInContextDataset` to inherit, unchanged:

- the subject→classes **scan cache** and `label_to_subjects` index,
- **class-balanced** anchor/subject sampling,
- the pre-resized **fast-path** volume loading (`_load`),
- spacing lookup, `incontext_collate_fn` compatibility.

Overrides `__getitem__`:

1. Choose an anchor class (balanced over `active_classes`) and K+1 subjects that
   share it (the target subject + up to K context subjects from
   `label_to_subjects[anchor_cls]`), mirroring the base class's
   `class_balanced=True` branch.
2. Draw the per-item **task spec** from an item-level RNG (see Determinism).
   The spec holds, for each of `n_objects` objects: `offset_norm` (3-vector),
   blob shape params, `scale_frac`, `contrast_delta`.
3. For each of the K+1 subjects:
   - load `(image, anchor_label)` via the inherited fast path,
   - compute anchor stats (`centroid`, per-axis `extent`) from `anchor_label`,
   - for each object: build the blob tile, map `offset_norm`/`scale_frac` to a
     voxel `center`/`size` using the anchor stats, apply small per-scene
     scale/rotation jitter, composite into `image`, and write the object mask
     into the label volume with ID `1..n_objects`,
   - discard the anchor mask (never emitted).
4. Emit the standard contract: `image (1,D,H,W)`, `label (D,H,W) int64`,
   `context_in (K,1,D,H,W)`, `context_out (K,D,H,W) int64`, `subject`,
   `label_name` (the anchor class), `spacing`, and a `meta` dict recording the
   task spec (offset(s), scale, contrast, anchor class) for the eval sample table.

New constructor params (all with defaults so the base signature is unaffected):
`object_source ("blob"|"organ")`, `shape ("blob"|"elongated"|"mix")`, `n_objects`,
`anchor_classes`, `offset_range`, `scale_frac`, `scale_jitter`, `rotate_jitter`,
`contrast_delta`, `edge_blur`, `boundary_complexity`, `deterministic`,
`eval_seed_namespace`, `eval_subjects_per_task`.

### `src/datasets/anchor_synth/shapes.py` — object generators (pure, testable)

Follows `controlSynth/shapes/blob.py`'s approach, extended to 3D: a base
ellipsoid whose radius is modulated by **low-frequency radial harmonics** for
organic, irregular (non-spherical) shapes — fully **analytic**, no scipy in the
default path.

- `make_object(size, params, rng) -> alpha_tile`
  Evaluates a rotated ellipsoid field `((x/ax)^2 + (y/ay)^2 + (z/az)^2) <= r(θ,φ)`
  on a `size³` grid, where `r(θ,φ)` is a product of a few low-order harmonics
  (`1 + Σ a_k cos(k·angle + φ_k)`, the 3D analog of `_radial_harmonics`). A `shape`
  param selects the base geometry (`blob` = near-isotropic, `elongated` =
  eccentric axes, `mix` = sampled per task). Semi-axes/eccentricity/orientation are
  jittered from `params`. Returns a soft `alpha` tile in `[0, 1]`; `edge_blur`
  controls edge softness (analytic falloff at the level set, not a scipy blur).

- `roughen(alpha, c, rng) -> alpha` *(optional, opt-in)*
  Mirrors `controlSynth/shapes/boundary.py`: perturbs the tile's signed-distance
  field with smoothed noise and re-thresholds for heavier irregularity. Uses scipy
  (`distance_transform_edt`, `gaussian_filter`), so it is the explicitly-opt-in
  heavier path (`boundary_complexity > 0`); default `0` keeps the analytic hot path.

### `src/datasets/anchor_synth/draw.py` — placement/compositing helpers (pure, testable)

No dataset/IO dependencies (constructed with plain numpy arrays), so testable with
trivial inputs:

- `anchor_stats(mask) -> (centroid, extent, bbox)`
  Computed via **axis-projection reductions** (`mask.any(axis=(1,2))`, etc.), not
  full `np.nonzero` or scipy. `extent` is the per-axis bbox side length; `centroid`
  is the bbox centre (cheap and robust; no weighted centre-of-mass needed).

- `place_object(image, alpha_tile, center, contrast_delta) -> object_mask`
  Composites on the **local sub-volume slice** only (like `render3d._slices_3d`):
  `img_local = img_local · (1 - α) + (bg_local + Δ) · α`, where `bg_local` is the
  mean CT intensity under the tile's footprint and `Δ = contrast_delta`. Soft α
  edges make the object **blend** into the background (no intensity cliff at the
  boundary). Binary object mask = `α > 0.5`.

- offset→center mapping: `center = centroid + offset_norm · extent`, clamped so the
  object tile stays fully in-volume.

## Efficiency (primary constraint)

- **No extra volume reads** beyond the K+1 the base class already loads; rides the
  pre-resized `ct_{size}.npy` / `label_{size}.npy` fast path (no interpolation).
- Anchor stats via **axis projections** — sub-millisecond at 128³, versus a full
  `nonzero` scan.
- **Analytic** object generation (radial-harmonics-perturbed ellipsoid) with
  per-scene jitter baked into params — the default shape path calls no scipy. The
  scipy-based `roughen`/`boundary_complexity` step is opt-in and off by default.
- Object composited on a **small local slice**, never a full-volume operation.
- Base geometric/intensity augmentation is **off by default**: the per-task object
  drawing *is* the randomization, keeping scipy warps out of the hot path.
- `object_source="organ"` (real organ tiles from the existing `TotalSegObjectBank`
  + `render.affine_jitter`) is the explicitly-opt-in, heavier path; `"blob"` is the
  fast default.

## Multi-object

`n_objects > 1` uses the **same single anchor** with N independent offset/shape
specs, producing labels `1..N`. This keeps the candidate subject pool equal to the
subjects sharing the one anchor class (no pool shrinkage from requiring several
shared classes). Multi-*anchor* (distinct anchor per object) is a possible future
extension, out of scope for v1.

## Determinism

Train draws fresh entropy per item. A `deterministic` mode (val/test) derives the
item-level task-spec RNG from `SeedSequence(eval_seed_namespace, anchor_key,
sample_index)`, where `anchor_key` is the anchor class's integer index
(`_ALL_CLASSES_IDX[anchor_cls]`), and enumerates `(anchor_class, sample)` pairs
into a fixed index —
mirroring `OmniSynthICLDataset`'s determinism. The task spec is drawn once per item
so the query and all K contexts share the same task definition; per-scene jitter
uses per-subject sub-RNGs.

## Config & wiring

- **`configs/experiment/3d/dataset/anchor_synth3d.yaml`** (mirrors
  `omnisynth3d.yaml`): `data.source: anchor_synth3d`, `data.image_size`,
  `data.context_size`, plus an `anchor_synth` block:
  `object_source`, `shape`, `n_objects`, `anchor_classes` (`[]`=all; also accepts
  `benchmark`/`not_benchmark`), `offset_range`, `scale_frac`, `scale_jitter`,
  `rotate_jitter`, `contrast_delta`, `edge_blur`, `boundary_complexity`,
  `eval_subjects_per_task`, `eval_seed_namespace`, `epoch_length`.
- **`experiments/3d/common.py`**: add an `anchor_synth3d` branch to `build_dataset`
  (construct `AnchorSynth3DICLDataset` with `root = cfg.paths.totalseg`, anchor
  classes resolved via `resolve_classes`, `deterministic = split != "train"`) and to
  `make_eval_loader` (deterministic multi-anchor eval, alongside the `omnisynth3d`
  branch).
- **`experiments/3d/eval.py`**: add a class-listing branch for
  `source == "anchor_synth3d"` — the class list is the resolved anchor pool.

## Testing

- **Unit tests** (`src/datasets/anchor_synth/`): `make_object` size scales with the
  requested size and harmonics produce non-spherical (irregular) shapes; `shape`
  variants differ; optional `roughen` changes the boundary; offset→center mapping +
  in-volume clamp; soft-edge blend (interior contrast ≈ Δ over local background,
  mask = α>0.5); determinism given a seeded RNG.
- **One integration test**: build `AnchorSynth3DICLDataset` with a small
  `max_subjects`, assert the contract tensor shapes/dtypes, that `label ⊆` the drawn
  object region, that anchor voxels are **not** in the label, and that the K+1 share
  one task spec.

## Out of scope (v1)

- Multi-anchor tasks (distinct anchor per object).
- Learned/real texture synthesis for objects beyond local-background + contrast delta.
- Additional morphologies beyond blob/elongated + SDF roughening (e.g. tubular,
  annular, scattered from `controlSynth/shapes/`) — easy to add later via
  `shapes.py`, out of scope for v1.
- Heavy base augmentation in the hot path (available via `object_source="organ"`
  and the existing aug config, but off by default).

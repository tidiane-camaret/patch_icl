# Cohort-consistent synthetic shapes for synth_gmm

**Date:** 2026-09-14
**Status:** approved design, not yet implemented

## 1. Motivation

On OOD eval tasks (tumors, sub-regions — see `docs/logs.md` eval-expansion entries for
ISLES22 / GNC kidney lesions / HU_LWK1), the model tends to fall back to predicting
familiar whole-organ shapes rather than the actual novel structure. The synth_gmm
training curriculum (`src/providers/synth_gmm.py`, wired into `92_multisource_synth`)
never shows the model a task whose target is a sub-region shape living inside a known
organ — every synth cohort paints a real whole-organ (or SLIC-supervoxel) silhouette
from the MAISI/TotalSeg bank. This design adds a new cohort mode that stamps a
procedurally generated geometric shape (not required to be anatomically plausible) into
a real host organ crop, so the model has to learn to find "a distinct blob-like region
inside this organ" as a general geometric-reasoning skill rather than only recognizing
whole known organs.

Two systems inform the design (inspected before this spec was written):

- **NV-Generate-CTMR's tumor augmentation** (`scripts/augmentation.py` in
  `/home/dpxuser/repos/NV-Generate-CTMR`): transplants a *real* donor tumor mask into a
  (possibly different) host organ instance, via a reject-until-valid loop that jitters
  the tumor mask (translate/rotate/scale) and keeps retrying until enough of it survives
  inside the host organ's mask. Establishes the "transplant a foreground shape into a
  real host, position independent per case" pattern, but requires real donor lesion
  masks (which we don't have at scale) and a retry loop we don't need once containment
  is loose.
- **controlSynth** (`src/datasets/controlSynth/`, 2D, already implemented and
  calibrated — see `docs/datasets/controlSynth_difficulty_findings.md`): a procedural
  shape-family generator (`blob`, `elongated`, `annular`, `vessel`, `scattered`) with a
  build/live parameter split for perf. Its calibration study found that **breaking
  context↔query consistency (`context_consistency`, `support_query_shift`,
  `context_copy_fraction`) is the dominant driver of in-context task difficulty** for a
  context-matching model, ahead of geometry knobs like `region_size`; shape/intensity
  distractor similarity (`task_ambiguity`) was empirically inert. This is the strongest
  evidence that investing in per-cohort *consistency* controls (this design's actual
  subject) is worth more than shape realism.
- **omniSynth** (`src/datasets/omniSynth/`): `scene.target_mode` (`identical | aug |
  class`) is a single combined consistency knob shared by shape+position+everything. The
  3D port only supports `identical | class` (no per-item warp yet). This design
  generalizes that single knob into four independent per-axis knobs.
- **synth_gmm's existing intensity-consistency pattern** (`src/providers/synth_gmm.py`):
  already implements exactly the mechanic this design needs, but only for intensity — a
  cohort-shared `mu`/`sd` draw (`_draw_gmm`, keyed by `gmm_seed`) plus a `between_ratio`
  knob that scales each member's independent deviation from the shared mean
  (`mu_e = mu + between_ratio * sd * member_nrng.standard_normal(...)`, keyed by
  `(gmm_seed, member_idx)`). This is cascade-consistent by construction: `load_native_crop`
  re-derives the identical `mu`/`sd`/`mu_e` at every cascade re-crop level from the
  subject string alone. This design generalizes that pattern from intensity to shape,
  size, and position.

## 2. Scope decisions (from brainstorming)

These were settled before this design and constrain everything below:

1. **Hosting**: shapes are placed inside real gmm_bank organ crops from day one (not a
   standalone synthetic-canvas prototype first).
2. **Consistency knobs**: continuous `[0,1]` `*_between_ratio` per axis (0=identical
   across cohort, 1=fully independent), mirroring synth_gmm's existing `between_ratio`
   — except shape *family* itself, which is a discrete per-cohort choice (not blended).
3. **Host containment**: loose. The shape's *position* is drawn from inside the host
   organ (reusing today's `_resolve_center`/`fg_samples` machinery, unchanged), but the
   rasterized shape is not clipped or retried against the organ boundary — it may cross
   into neighboring tissue. No reject-until-valid loop.
4. **Shape primitives (v1, four families)**: `blob` (harmonic-perturbed sphere),
   `splatter` (multi-component scatter — several small blobs under one label, 3D analog
   of controlSynth's `scattered.py`, distinct from a single rough-boundary blob),
   `disk` (flattened ellipsoid), `cylinder` (capsule: distance-to-line-segment
   threshold).
5. **Label semantics**: a shape overwrites the host organ's label at its footprint and
   becomes its own new pseudo-class (like the existing `body`=200 pseudo-class in
   `data/maisi_classes.py`), not an additive overlay. That pseudo-class is the
   in-context segmentation target for the cohort.
6. **Host sampling**: each of the K+1 cohort members gets its own independently-sampled
   host donor entry (mirrors `GmmCohortSampler.sample_cohort` today) — not one shared
   host volume for the whole cohort.
7. **Host class pool**: unrestricted — hosts are drawn via
   `self.ds.cs.sample_cohort(rng)` with `target_class=None`, exactly as today, no new
   "eligible host classes" config. (`sample_cohort`'s signature already supports this;
   see `src/gmm_cohort_sampler.py:155`.)
8. **Integration strategy**: extend `SynthGmmProvider` in place (`_build_nc` gains one
   new step), gated by `p_shape=0.0` default so `92_multisource_synth` and any other
   existing run is unaffected until explicitly turned on. Rejected alternatives: a
   standalone `SynthShapeProvider` (would duplicate nearly all of `_build_nc`'s
   crop/resample/paint/cascade-seed machinery once hosting was required) and subclassing
   `SynthGmmProvider` (couples to private methods not designed as an extension seam).

## 3. Components

### 3.1 `src/shapes3d/` — new package, pure geometry, no torch/provider dependencies

Mirrors the separation controlSynth already uses (`shapes/` vs `dataset.py`): the
primitives and the per-axis instantiation logic are independently testable and know
nothing about NativeCrop, GMM painting, or the bank.

- **`primitives.py`**: one function per family,
  `make_<family>(grid_size: int, params: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]`
  returning a `uint8[T,T,T]` mask and a `realized_meta` dict (actual size fraction,
  component count, etc. — mirrors controlSynth's "record realized, not requested"
  practice, `docs/datasets/controlSynth.md` §10.2).
  - `make_blob`: 3D generalization of `controlSynth/shapes/blob.py::make_blob` — a
    spherical radius field perturbed by a handful of low-frequency spherical-harmonic-like
    angular terms (replacing the 2D radial-harmonic perturbation), thresholded.
  - `make_splatter`: 3D generalization of `controlSynth/shapes/scattered.py` — a point
    process (count, clustering) placing several small blob stamps under one label.
  - `make_disk`: an ellipsoid with one axis scaled down by an `aspect_ratio` parameter
    (flattened blob).
  - `make_cylinder`: a capsule — points within `radius` of a line segment of length
    `length` at a random orientation, with optional rounded or flat end caps.
- **`spec.py`**: `ShapeCohortSpec` dataclass — the config surface (§4).
- **`instantiate.py`**:
  - `draw_cohort_hyperparams(rng, spec) -> CohortHyperparams`: called once per cohort
    from the cohort-level RNG (keyed by `gmm_seed`, same RNG that drives `_draw_gmm`
    today). Picks `family` (discrete, from `spec.family_weights`) and a baseline value
    for every continuous param (shape params, `size_frac`, and a *relative* position
    descriptor — see below).
  - `draw_member_shape(member_rng, cohort_hp, spec) -> np.ndarray`: called once per
    member from the per-member RNG (keyed by `(gmm_seed, member_idx)`, same RNG family
    as the existing `member_nrng` in `_build_nc`). For each axis, blends the cohort
    baseline and a fresh independent draw by that axis's `*_between_ratio`:
    - **shape params / size_frac**: linear blend, `value = cohort_value + between_ratio *
      (fresh_draw - cohort_value)` — same shape as the existing `mu_e` formula.
    - **position**: "identical across cohort" cannot mean the same absolute voxel
      (each member has its own host volume). It means the same *relative* offset within
      each member's own host organ's bounding region — e.g. cohort hyperparameters
      include a normalized `(u, v, w) ∈ [0,1]^3` offset relative to the host organ's
      bounding box, which `position_between_ratio=0` reuses verbatim per member and
      `=1` redraws independently per member. Falls back to today's `_resolve_center`
      voxel resolution (`center_mode="com"` or `"random_fg"`) as the base coordinate
      system so this doesn't touch that machinery.
    - **intensity**: unchanged — reuses `synth_gmm.py`'s existing `mu`/`sd`/
      `between_ratio` exactly as today; the shape's pseudo-class gets its own `mu`/`sd`
      row like any other class id.

### 3.2 `SynthGmmProvider` changes (`src/providers/synth_gmm.py`)

- New pseudo-class ids for the four shape families, registered in
  `data/maisi_classes.py` the same way `200: 'body'` was (e.g. `201: 'shape_blob'`,
  `202: 'shape_splatter'`, `203: 'shape_disk'`, `204: 'shape_cylinder'`), so
  `class_idx`/`label_name` keep flowing through the existing NativeCrop, eval, and
  logging paths unchanged.
- New config knobs (all additive, defaulted off): `data.gmm.p_shape: float = 0.0` and a
  `data.gmm.shape: ShapeCohortSpec` block.
- `assemble_task`: after the existing `gmm_seed = rng.getrandbits(64)` and
  `mu, sd = self._draw_gmm(gmm_seed)`, draw a coin flip from the same `rng` gated by
  `p_shape`. If it fires: draw `cohort_hp = draw_cohort_hyperparams(...)` (also keyed by
  `gmm_seed`, so it's reproducible the same way `mu`/`sd` are), and call
  `self.ds.cs.sample_cohort(rng)` unchanged (host class pool is unrestricted, per §2.7)
  to get the K+1 host donor entries. `cls_id` passed to `_build_nc` becomes the shape's
  pseudo-class id instead of the host's real class id; `label_name` is set from the
  shape pseudo-class, not the host.
- `_build_nc`: gains one new step. The existing flow is: load host `arr` → resolve
  `center` via `_resolve_center` (unchanged, still resolves against the host's own real
  anatomy — e.g. "a voxel inside this lung") → crop via `organ_crop_arrays` → cap native
  crop → `_resample_paint_mask`. The new step sits between crop and cap: when the cohort
  is shape-mode, call `draw_member_shape(member_nrng, cohort_hp, spec)` to get a mask at
  the crop's grid resolution, and overwrite `crop_lbl` at the shape's footprint with the
  shape's pseudo-class id, before the existing native-crop cap and
  `_resample_paint_mask` call. Everything downstream (cap, resample, GMM paint,
  NativeCrop construction) is untouched.
- `load_native_crop`: no structural change. The subject string already carries
  `(filename, gmm_seed, member_idx)`; re-deriving `cohort_hp` and the member's shape
  draw from those same two RNGs reproduces the identical shape at any cascade re-crop
  level, the same way `mu`/`sd`/`mu_e` already do.

## 4. Config surface

```yaml
data:
  gmm:
    p_shape: 0.0                    # probability a sampled cohort becomes shape-mode
    shape:
      family_weights: {blob: 1, splatter: 1, disk: 1, cylinder: 1}  # per-cohort mixture
      size_frac: [0.02, 0.15]        # target host-crop volume fraction, sampled range
      shape_between_ratio: 0.3
      size_between_ratio: 0.3
      position_between_ratio: 0.5
      intensity_between_ratio: null  # falls back to existing gmm.between_ratio if unset
      # per-family param ranges (roughness, aspect_ratio, n_components, radius/length, ...)
      # deferred to implementation — not a design-level decision
```

## 5. Error handling

Rasterization is pure NumPy with no failure modes requiring retries (loose containment,
per §2.3, means there is nothing to reject). The only degenerate case is a shape
footprint partly or fully clipped by the crop's grid bounds after position jitter —
clipped silently (same border-clip behavior controlSynth already has), never raised.

## 6. Testing

- Per-primitive unit tests (footprint volume fraction near requested `size_frac`,
  splatter's component count, disk/cylinder aspect sanity) — same style as
  `src/datasets/omniSynth/test_bank_common3d.py`.
- A determinism test mirroring the existing cascade-consistency contract: same
  `gmm_seed` → same `cohort_hp` across two independent calls to
  `draw_cohort_hyperparams`; `assemble_task`'s level-0 shape mask is reconstructible
  byte-for-byte via `load_native_crop` from the subject string alone.
- No integration test against real training is required for this design; validate the
  generator and consistency knobs standalone before wiring `p_shape > 0` into any
  training config.

## 7. Explicitly deferred (not in this design)

- Per-family shape-parameter ranges (roughness, aspect ratios, cylinder length/radius
  priors) — implementation detail, not architectural.
- Any curriculum/scheduling of `p_shape` or the `*_between_ratio` knobs over training.
- Calibrating whether this curriculum actually improves OOD (tumor/sub-region) eval —
  out of scope for this design; follows the same calibrate-before-adopt discipline
  controlSynth used (`docs/datasets/controlSynth_difficulty_findings.md`).
- Reject-until-valid / strict containment (only revisit if loose containment measurably
  underperforms).

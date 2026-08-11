# Coords-function synthetic labels for in-context 3D segmentation

Date: 2026-08-11
Status: design approved (brainstorm), pending spec review

## Goal

Add a synthetic-label generator that injects **cross-subject positional
correspondence** into in-context training, using the `coords.npy` canonical body
frame we generated for all 1228 TotalSeg scans. This complements the existing
supervoxel synth path, which teaches appearance/augmentation-invariance at an
*identical* position (K+1 augmented copies of one subject) but carries no
cross-subject position signal.

## What the coords maps can and cannot do (measured)

- `coords_axes.py` (30 subj): each coords channel is a near-perfect, sign-
  consistent affine image of a real RAS body axis — c0↔LR |r|0.986, c1↔AP 0.976,
  c2↔SI 0.931; linear R²=0.982. coords ≈ **~98% an affine transform of scanner
  RAS per subject**; only ~2% is nonlinear cross-subject normalization.
- `coords_quality.py` (50 subj, 61 balanced classes): LOO organ retrieval top-1
  0.386, cross-subject centroid spread 60–120 mm. Fine organ correspondence is
  weak.
- **Consequence:** a synthetic label is trustworthy only if its geometry lives at
  the scale/shape the affine frame represents faithfully — **large and axis-
  aligned**. A small hard blob "inside a vertebra" will land in a different
  vertebra across subjects.

## Core design: label = smooth function of coords

Every synthetic label is a soft field `f(coords) → [0,1]` with randomly sampled
parameters, evaluated **independently on each subject's `coords.npy`**. Because
coords is a shared frame, the same `f` applied to different subjects yields
**corresponding** labels by construction — no ctx→tgt matcher, no bin-hash, no
HI validity guard, no transfer "survival" failures. Correspondence quality =
coords quality, controlled by the field's scale.

"Hard" vs "soft" is one parameter (edge width), so both tiers share code:

All fields must be **localized/bounded** (anchored at a body point μ). Visual QA
(`plot_coords_synth.py`) showed unbounded primitives (half-space, full slab, long
cylinder) FAIL on heterogeneous FOVs: a fixed coords half-space anchored in the
chest lands on lungs in a chest scan but on the **skull** in a head-only scan
(HI-averaging hid this bimodality). Bounded fields don't have this problem — a
subject either contains the anchored region (mass guard passes) or doesn't. So
half-space/slab/unbounded-cylinder are RETIRED.

- **Tier 1 — hard, bounded (binary label):**
  - anisotropic ellipsoid: `‖R(coords−μ)/radii‖ ≤ 1`, random rotation R
  - capped cylinder: radius r AND half-length L along a random axis
  - Orientation is **arbitrary** (any oblique direction): since coords ≈ affine
    of RAS, any linear function of coords is affine-consistent. Binarize at
    0.5 → integer label.
- **Tier 2 — soft, focused (float label):**
  - coords-Gaussian: `exp(−½ (coords−μ)ᵀ Σ⁻¹ (coords−μ))`, μ a real canonical
    body location, Σ anisotropic. Softness encodes positional uncertainty; small
    Σ blurs toward the reliable scale rather than asserting a false boundary.

### Scale bands (from Phase-0 `coords_synth_consistency.py`, 20 subj)

Cross-subject anatomy consistency (mean pairwise soft-weighted, bg-excluded
label-histogram intersection) sits ~0.25–0.31 everywhere (≈5–6× the ~0.05
random-placement chance). The **variance** is the selector:

| family | scale | HI mean | HI std |
|--------|-------|---------|--------|
| gaussian | σ=20 | 0.294 | 0.121 |
| gaussian | σ=40 | 0.295 | 0.039 |
| gaussian | σ=160 | 0.314 | 0.020 |
| slab | hw=20 | 0.312 | 0.106 |
| slab | hw=40 | 0.254 | 0.042 |
| cylinder | r=20 | 0.209 | 0.059 |
| halfspace | – | 0.266 | 0.043 |

**Rule: floor the characteristic scale at ≈40 mm.** Below that, std doubles/
triples (the "sometimes right, sometimes garbage" failure) and masks frequently
miss FOV. Sampling ranges: gaussian σ∈[40,160], slab hw∈[40,160], cylinder
r∈[40,160], half-spaces always allowed.

## Task assembly (multi-subject, FOV-aware)

1. Sample a field `f` (localized family + params), anchor μ = coords value at a
   random labelled voxel of a random reference subject.
2. **FOV pre-filter:** keep only subjects whose precomputed coords bounding box
   (min/max coords over labelled voxels = the canonical region the scan covers)
   contains μ. This is the cheap grouping filter that prevents pairing a
   chest-anchored field with a head-only scan.
3. For each candidate: evaluate `f`, apply a **mass guard** (in-crop label mass ≥
   `min_mass`) and collect K+1. Center that subject's crop on its own instance of
   the region (argmax / centroid of `f`), matching how the supervoxel path
   centers crops under `use_crop`.
4. **Consistency backstop:** require the K+1 picked subjects to agree on anatomy
   (mean pairwise soft-weighted label-histogram intersection ≥ `min_hi`, ~0.15);
   otherwise redraw the field. Catches residual near-FOV-boundary clipping.
5. Emit `(image, label)` per subject; item[0] = target, items[1:] = K contexts.

Validated in `plot_coords_synth.py`: with this assembly every montage row hits
the same anatomy across target + contexts (HI 0.53–0.83), and the earlier
skull-for-lung failures disappear.

This mirrors `_get_synth_item` but swaps "one subject, K+1 aug copies" for "K+1
subjects, one shared field". Per-subject augmentation still applies on top.

## Integration into `TotalSegInContextDataset`

- New synth mode selected by a probability knob. Current `__getitem__` routes to
  `_get_synth_item` with prob `p_synth`; add `p_coords` so an item is:
  real (1−p_synth), supervoxel synth (p_synth·(1−p_coords)), or coords synth
  (p_synth·p_coords). Keeps one synth budget, mixes the two synth flavours.
- New method `_get_coords_item()` implementing the assembly above, reusing the
  existing `use_crop` crop machinery (`_organ_crop_arrays` / native-crop slice)
  with the crop center = coords-region center instead of organ centroid.
- Precompute + cache a per-subject coords AABB (`coords_aabb`) once at init
  (like the existing scan/bbox caches) for the FOV pre-filter.
- Config: `data.p_coords`, `data.coords_fname="coords.npy"`, field sampling
  ranges (localized families, scale bands, hard/soft mix), `min_mass`, `min_hi`.

### Label dtype: phased

- **Phase A (hard only):** Tier-1 fields binarized → `uint8`/`int64`. Drops into
  the existing integer pipeline (`label`, `context_out`, collate, `bce_dice`,
  metrics, random-coloring) with **zero changes**. Delivers the coarse positional
  curriculum first and validates the multi-subject assembly.
- **Phase B (soft):** Tier-2 float labels in [0,1]. Requires float-label support:
  `bce_dice` already accepts soft targets and Medverse outputs a probability, but
  verify no site does `.long()` / binarize / argmax on the label; metrics
  (Dice) need a soft or thresholded variant; `random_coloring`/`_sample_palette`
  assume integer labels → soft mode runs with coloring off (or thresholded for
  coloring only). Gated behind a `coords_soft` flag; off in Phase A.

## Performance

Loading `coords.npy` (≈30–250 MB) for K+1 subjects per item is the cost risk
(the supervoxel path uses small pre-resized files). Mitigations:
- center-finding on a strided (e.g. ÷4) coords read, native crop only for the
  final T³ window;
- optional pre-resized `coords_{T}.npy` for the non-crop fast path, generated
  alongside like `ct_{size}.npy`.
Measure worker throughput before/after; only add pre-resize if needed.

## Validation artifacts

- Phase-0 consistency table (done) → scale bands.
- K+1-panel montage per field family (target + K contexts, CT + label overlay)
  for visual QA that the same anatomy is selected across subjects.
- A short training smoke run with `p_coords>0` (hard mode) confirming loss/
  metrics flow and that coords tasks look sane in W&B.

## Out of scope

- Raising the coords fidelity ceiling (needs a better/finer-trained coords model
  or post-transfer intensity refinement) — the missing ~2% nonlinear warp.
- Replacing the supervoxel path; coords synth is additive.

## Open items to confirm in review

1. Phasing hard-first (A) then soft (B) — agree?
2. Crop-centering each subject on its region (removes absolute-position signal,
   teaches appearance + relative position) vs. random/full-volume crops (keeps
   absolute position). Default = crop-centered to match the pipeline.
3. Field family mix / weights and the hard:soft ratio for Phase B.

# ChemoTox in-context eval — design

**Date:** 2026-08-12
**Goal:** Evaluate patchset3d (and the other 3D in-context models) on the ChemoTox
cohort, both label schemes, reusing the existing TotalSeg eval harness
(`experiments/3d/eval.py` → `common.make_eval_loader` → `evaluate.evaluate_classes`).

## Cohort summary

`experiments/3d/universal_coords/coords_paths_chemotox.json` — **366 subjects**
keyed `patientID#studyDate`. Each entry: `img`, `bclabels`, `totalseg`, `coords`
(all present on NFS) + metadata (`age`, `sex`, `weight`, `height`).

Geometry (all subjects sampled identical): `img` / `totalseg` / `bclabels` share
one native grid **`(1024, 1024, 212)` @ 0.35×0.35×3.0 mm** (thorax/abdomen DE-CT).
`coords` is a separate `(90,90,80,3)` field — **not used** by this eval.

Two label sources:

- **`totalseg`** (`ML/total_seg_total.nii.gz`, uint8): standard TotalSegmentator v2
  `total` numbering, IDs **1–117**, all mapping cleanly to the project's
  `_ALL_CLASSES_IDX`. Coverage is FOV-driven across the cohort: 64 classes
  near-universal (≥95% of subjects), 19 partial, 28 sparse (abdominal organs
  present only where the scan FOV reaches them).
- **`bclabels`** (`uncropped_BCLabels.nii.gz`, int16) is **4-D `(...,2)`**:
  - channel 0 = body-composition tissue map, **4 classes**:
    `1=muscle, 2=sat, 3=vat, 4=imat`.
  - channel 1 = instance/region IDs (thousands of values) — **discarded**.

## Constraints / decisions (confirmed)

- **bc classes:** muscle / sat / vat / imat (channel 0 only).
- **bc crop:** reuse the organ-centroid crop machinery unchanged (per-class
  centroid ≈ body-center for diffuse tissue; context and target are both crops, so
  in-context matching still holds).
- **Config surface:** two `cfg.data.source` values — `chemotox` (totalseg labels)
  and `chemotox_bc` (bc labels).
- **Fast I/O:** decompressing 366 large gzipped NIfTIs per `__getitem__` (gzip has
  no random access → every crop reads the whole volume) would make eval I/O-bound.
  Convert once to uncompressed, mmap-friendly `.npy` (crops touch only their bytes).
  `.npy`, not `.npz` — zip compression defeats mmap cropping.
- **Cache resolution:** pre-resample to **1.5 mm isotropic** (~35 GB total). Caps
  eval `crop_spacing_mm ≥ 1.5`; matches the typical 1.5–4 mm eval regime.
- **Cache location:** new `paths.chemotox`, default
  `/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/chemotox`.
- **totalseg `val_classes`:** default to the standard eval list (`benchmark`);
  FOV-sparse classes are included (they simply contribute fewer samples).

## Component 1 — generalize `scripts/convert_to_npy.py`

Keep a single conversion tool. Add a **`--source {totalseg,chemotox}`** dispatch
(default `totalseg` = current behavior, unchanged) plus two general options. The
source-specific bits sit behind a thin, dict-dispatched seam (picklable into
workers); the shared machinery (normalise CT → resample → save `.npy` +
`spacings.json`) stays one code path.

### Per-source seam

- `enumerate_subjects(source, data, out) -> list[spec]`, where each `spec` carries
  `subj_id`, resolved input paths, and the per-subject output dir.
  - `totalseg`: directories under `--data` (as today).
  - `chemotox`: entries of `coords_paths_chemotox.json`; `subj_id =
    "{patientID}_{date}"` (from the `patientID#date` key), output dir `out/{subj_id}`.
- `load_raw(source, spec) -> (raw_ct: f32, spacing: [3], labels: dict[name→array])`.
  - `totalseg`: read `ct.nii.gz`; build merged label from `segmentations/{cls}.nii.gz`
    in `ALL_CLASSES` order → `{"label": merged}` (unchanged behavior).
  - `chemotox`: read `img` NIfTI; read `total_seg_total.nii.gz` and **remap** its
    TS-v2 integer IDs → project `_CLASS_TO_IDX` **by name** using the official
    TotalSegmentator `total` id→name map → `{"label": …}`; read `bclabels`, take
    channel 0 → `{"bc": …}`. Assert anchor IDs (liver/heart/aorta) resolve
    consistently before writing; if `ALL_CLASSES` already equals TS order the remap
    is identity. All three share one grid → no cross-volume resampling; spacing from
    the `img` affine (`nib.affines.voxel_sizes`), no canonicalization.

### Two new general options

- `--out DIR` — output root; **default = `--data`** (in-place; preserves current
  totalseg runs). ChemoTox reads a read-only cohort tree and writes to
  `paths.chemotox`.
- `--target-spacing MM` — when set, the "native" outputs are resampled to `MM`
  isotropic (CT trilinear, every label nearest) via a new
  `_resample_to_spacing(vol, native_sp, target_sp, order)`
  (`out_shape = round(shape · native_sp / target_sp)`, `ndi.zoom`) instead of
  stored full-native. ChemoTox uses `1.5`. `spacings.json` records the **cache**
  spacing (`[1.5,1.5,1.5]`) + cache shape, which is what the `use_crop` path reads.

### Shared `convert_subject`

`load_raw` → normalise CT (`_normalise_ct`) → optional `--target-spacing` resample
of CT + every label → write `ct.npy` (f16) and one `.npy` per label name
(`label.npy`, `bc.npy`, uint8) → optional `--size` sized variants → return
spacing/shape for `spacings.json`. Multi-label output is just iterating the
`labels` dict. For sources that declare no split (chemotox), write `meta.csv`
(`image_id;split`, all `test`) so `eval.split=test` works uniformly.

**Run:**
```bash
python scripts/convert_to_npy.py --source chemotox \
    --out /nfs/.../ANALYSIS_20251122/data/chemotox \
    --target-spacing 1.5 --workers 32
```

## Component 2 — `source=chemotox` (117 totalseg classes): no new dataset class

The converted tree (`ct.npy`, `label.npy`, `spacings.json`, `meta.csv`) is
byte-compatible with the base `use_crop` path, so it routes to
**`TotalSegInContextDataset`** unchanged: mmap `ct.npy`/`label.npy`, its own
scan+bbox caches, `spacings.json`-driven fixed-extent crops. Wiring: add
`"chemotox"` to `common._TOTALSEG_SOURCES` (root `paths.chemotox`, `is_mri=False`).
The direct-totalseg branch of `make_eval_loader` then serves it, including spacing
sweeps.

## Component 3 — `source=chemotox_bc` (4 tissue classes): thin subclass

`src/chemotox_dataset.py::ChemoToxBCDataset(TotalSegInContextDataset)`, mirroring
`TotalSegMoreLabelsDataset`. Eval-only, `use_crop`-only. Classes
`{muscle:1, sat:2, vat:3, imat:4}` (local id = `bc.npy` value). Overrides:

- `_get_subjects` — list output dirs (those containing `bc.npy`); all `test`
  (assert `split in (None, "test")`).
- `_load_or_build_cache` — every subject has all 4 bc classes (diffuse tissue),
  so return `{subj: frozenset(BC_NAMES)}` directly; no label scan.
- `_load_or_build_bbox_cache` — per-class centroid from `bc.npy` (parallel, pickled,
  keyed by subject-list hash), like the more_labels centroid cache.
- `_load_crop` — organ-centred native crop of fixed physical extent
  (`T·crop_spacing_mm`) from `ct.npy` + `bc.npy == local_id`, resampled to T³; reuse
  the base `_organ_crop_arrays` / `_place_image` / `_place_label` / `_resample_binary`.

## Component 4 — wiring (`experiments/3d/common.py`)

- `_source_root`: add `chemotox` (→ `paths.chemotox`) to `_TOTALSEG_SOURCES`;
  special-case `chemotox_bc` → `paths.chemotox` (same tree, reads `bc.npy`).
- `build_dataset`: add a `chemotox_bc` branch (before the generic totalseg build,
  like the `totalseg_more_labels` branch) constructing `ChemoToxBCDataset`
  (`root=paths.chemotox`, `classes=BC_NAMES`, `image_size`, `split`,
  `context_size`, `max_subjects`, `eval_seed`, `use_crop=True`, `crop_spacing_mm`,
  `crop_jitter`). `chemotox` needs no branch (falls through to the generic
  `TotalSegInContextDataset` build).
- `make_eval_loader`: add `chemotox_bc` to the subclass special-case set alongside
  `totalseg_more_labels` (honors the `(idx, spacing)` per-item crop override for
  spacing sweeps). `chemotox` is served by the direct-totalseg branch unchanged.

## Component 5 — configs

- `configs/experiment/3d/dataset/chemotox.yaml` — mirror `totalseg.yaml`:
  `source: chemotox`, `image_size: [128,128,128]`, `context_size: 1`,
  `use_crop: true`, `crop_spacing_mm: 1.5`, `val_classes: benchmark`, synth/aug off
  for eval. (FOV-coverage caveat noted in a comment.)
- `configs/experiment/3d/dataset/chemotox_bc.yaml` — `source: chemotox_bc`,
  `val_classes: [muscle, sat, vat, imat]`, otherwise as `chemotox.yaml`.
- `configs/cluster/nfs.yaml` — add
  `paths.chemotox: /nfs/.../ANALYSIS_20251122/data/chemotox`.

## End-to-end usage

```bash
# once
python scripts/convert_to_npy.py --source chemotox \
    --out /nfs/.../data/chemotox --target-spacing 1.5 --workers 32
# eval
python experiments/3d/eval.py dataset=chemotox    eval.model=... eval.checkpoint=...
python experiments/3d/eval.py dataset=chemotox_bc eval.model=... eval.checkpoint=...
```

## Out of scope / notes

- `coords` field is unused here.
- bc crops centred on diffuse-tissue centroids won't cover the whole body; that is
  intended (in-context crop-vs-crop matching). If whole-body bc segmentation is
  wanted later, that is a separate whole-volume-resample regime.
- 1.5 mm cache caps `crop_spacing_mm ≥ 1.5`; a finer cache (1.0 mm / native) is a
  re-run of the same tool with a different `--target-spacing`.
- TS numbering parity: verify the official TS `total` id→name map vs project
  `ALL_CLASSES` during implementation; the remap-by-name makes correctness
  independent of whether the orderings coincide.

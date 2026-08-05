# Design: Eval on TotalSegmentator `more_labels` extra classes

Date: 2026-08-05
Status: approved (pre-implementation)

## Goal

Make any of the extra TotalSegmentator `more_labels` classes (converted in
`experiments/totalseg_more_labels/convert_more_labels.py`) nameable in
`cfg.data.val_classes` and evaluated by `experiments/3d/eval.py` exactly like a
normal in-context TotalSeg class — no change to training or to the existing
per-class Dice / timing / GFLOPs reporting.

The extra labels live in a **separate data tree**:
`.../ANALYSIS_20251122/data/totalseg_test_more_labels/`, containing per subject:

- `ct.nii.gz`                          — raw CT (native)
- `more_labels/{task}.npy`             — uint8, native, canonical orientation
- `more_labels/{task}_{D}x{H}x{W}.npy` — uint8, iso-resized (nearest), aligned to
                                         the main tree's `ct_{size}.npy` grid
- `segmentations/{task}.nii.gz`        — source multilabel masks

and at the root:

- `more_labels_classes.json`          — global index: `classes: [{global_id, task,
                                        local_id, name}]`, 362 entries / 329 unique
                                        names / 37 tasks
- `more_labels_subject_classes.json`  — `{subject: [global_id present (>0 vox)]}`,
                                        25 subjects

Only `ct.nii.gz` (no pre-sized `ct_{size}.npy`) exists in this tree; CT is loaded
directly and resized on the fly. Pre-sized CT files may be added later and will be
used automatically when present.

## Key facts driving the design

- **362 classes, 329 unique names, 33 name collisions** across tasks (e.g.
  `tissue_types` vs `tissue_4_types` both have `skeletal_muscle`; `vertebrae_pp`
  vs `vertebrae_pp_refined`). A class is therefore identified by the
  **task-qualified key** `"{task}/{name}"`, which is unique.
- **285 of 362** classes appear in ≥2 of the 25 test subjects. Only these are
  context-viable (in-context eval needs target + ≥1 same-class context from the
  same 25-subject pool). The other 77 are dropped from the default class list.
- CT + the pre-resized grid the masks were aligned against are produced by
  `scripts/convert_to_npy.py` as `nib.as_closest_canonical` → `_normalise_ct` →
  `_iso_resize(vol, size, order=1, aa=True, spacing=sp)`. Reproducing this exactly
  guarantees the loaded CT is pixel-aligned with `more_labels/{task}_{size}.npy`
  (which was produced with the same `_iso_resize`, `order=0`).

## Architecture

### 1. `TotalSegMoreLabelsDataset` (new: `src/totalseg_more_labels_dataset.py`)

Subclass of `TotalSegInContextDataset`. Overrides only class identity and loading;
inherits context sampling (`_lazy_shuffle`), `eval_seed` determinism, `__getitem__`
(single-label branch), and the collate contract unchanged.

**Class identity.** Each class = `"{task}/{name}"`. On init, read
`more_labels_classes.json` into:

- `self._resolve: {"{task}/{name}": (task, local_id)}`
- `self._gid_to_key: {global_id: "{task}/{name}"}`

Read `more_labels_subject_classes.json` (global_ids per subject) and map through
`_gid_to_key` to `{subject: frozenset("{task}/{name}")}`.

**Cache override.** Override `_load_or_build_cache()` to return that JSON-derived
`{subject: frozenset[class]}` directly — no `label.npy` scan, no `.scan_cache`
pickle. The inherited `__init__` then builds `label_to_subjects` and `samples`
from it as usual. No bbox / synth / spacings caches are built (all off for this
eval). `_get_subjects` still applies `max_subjects`; `split` is accepted for
signature compatibility but not used to filter (all 25 subjects are test).

**Eval-only guardrails.** Assert `use_crop is False`, `synth_method is None`,
`num_labels_per_sample == 1`, `aug_cfg is None`. (Training / crop / synth / multi-
label paths are explicitly out of scope; asserting prevents silent misuse.)

**`_load(subj, cls)`** — the only load path:

- *CT:* prefer `ct_{size}.npy` if present; else load `ct.nii.gz` →
  `nib.as_closest_canonical` → `_normalise_ct` → `_iso_resize(vol, size, order=1,
  aa=True, spacing=sp)` where `sp = nib.affines.voxel_sizes(img.affine)[:3]`. Cache
  the resulting `(1, D, H, W)` float32 tensor per subject in an in-memory dict
  (`self._ct_cache`); only 25 subjects (~26 MB/worker), so contexts never re-decode
  the NIfTI. Return `.clone()` of the cached tensor per call (downstream aug is off,
  but keep the contract that `_load` returns a fresh tensor).
- *Mask:* `task, local_id = self._resolve[cls]`; load `more_labels/{task}_{size}.npy`
  (mmap); return `torch.from_numpy(arr == local_id).long()` shape `(D, H, W)`. If the
  sized file is absent, fall back to native `more_labels/{task}.npy` +
  `_iso_resize(arr, size, order=0, aa=False, spacing=sp)`.

Both return `(image_t (1,D,H,W) float32, label_t (D,H,W) int64)`, matching the base
`_load` contract exactly.

### 2. Config & eval wiring

- **`experiments/3d/common.py`**
  - Extend the source dispatch to include `"totalseg_more_labels"`.
  - `_source_root`: resolve its root from `cfg.paths.totalseg_more_labels`
    (`is_mri=False`).
  - `build_dataset` + `make_eval_loader`: when `source=="totalseg_more_labels"`,
    construct `TotalSegMoreLabelsDataset` (deterministic `eval_seed=cfg.eval.seed`,
    aug/synth off, `class_balanced=False`, `shuffle=False`), mirroring the existing
    `omnisynth3d` / `anchor_synth3d` special-cases.

- **`data/totalseg_classes.py`**
  - New `resolve_more_labels_classes(root, value)`: reads
    `{root}/more_labels_classes.json` + `more_labels_subject_classes.json`.
    `value=="all"` → the 285 classes present in ≥2 subjects, sorted; a list →
    passed through (each entry a `"{task}/{name}"` key, validated against the index).

- **`experiments/3d/eval.py`**
  - Add a `source=="totalseg_more_labels"` branch in `main()` that sets `root` from
    `cfg.paths.totalseg_more_labels` and `classes` via
    `resolve_more_labels_classes(root, cfg.data.val_classes)`, alongside the existing
    `anchor_synth3d` / `omnisynth3d` branches.

- **Config:** `configs/experiment/3d/dataset/totalseg_more_labels.yaml`
  - `data.source=totalseg_more_labels`, `data.val_classes=all`, `data.use_crop=false`.
  - `paths.totalseg_more_labels` default = the ANALYSIS_20251122
    `totalseg_test_more_labels` path (overridable per cluster).
  - Run: `python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse`

## Data flow (one eval item)

```
eval.py main()
  source==totalseg_more_labels
    root      = cfg.paths.totalseg_more_labels
    classes   = resolve_more_labels_classes(root, cfg.data.val_classes)   # 285 keys
  evaluate_classes(model, cfg, classes)
    make_eval_loader -> TotalSegMoreLabelsDataset(root, classes, eval_seed=...)
      __getitem__(idx)                                # inherited
        subj, cls = samples[idx]                      # cls = "task/name"
        image_t, label_t = _load(subj, cls)           # overridden
          CT   : ct.nii.gz -> normalise -> iso_resize (cached per subj)
          mask : more_labels/{task}_{size}.npy == local_id
        context: _lazy_shuffle over label_to_subjects[cls] (same-class subjects)
      -> {image, label, context_in, context_out, label_name=cls, ...}
    model.predict(...) -> per-class Dice / time / GFLOPs   # unchanged
```

## Verification

1. **Alignment unit check.** For a subject shared with the main tree (e.g. `s0002`),
   assert the CT produced by `TotalSegMoreLabelsDataset._load` equals the main
   tree's `ct_{size}.npy` (byte / allclose), proving mask–CT pixel alignment. Assert
   `label` foreground equals `(more_labels/{task}_{size}.npy == local_id)`.
2. **Smoke eval.** `python experiments/3d/eval.py dataset=totalseg_more_labels
   eval.model=medverse eval.n_subjects=4` over a few classes → non-crashing per-class
   Dice rows, sane occupancy.
3. Log the change in `docs/logs.md`.

## Out of scope

- Training on the extra labels; the crop path; synth; multi-label-per-sample; MRI.
- Curating / de-duplicating the 33 collision or blanket/reference-background tasks
  — all 285 viable classes are exposed; curation is a `val_classes` list decision
  left to the user.

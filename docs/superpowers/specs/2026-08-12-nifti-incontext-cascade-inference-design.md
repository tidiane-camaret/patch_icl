# Nifti in-context cascade inference — design

**Date:** 2026-08-12
**Status:** approved (pending spec review)

## Goal

Provide one importable function that takes a target CT nifti + a list of context
(image, binary-mask) nifti pairs for the same organ, and returns a predicted
binary mask on the target's native grid — optionally writing it as a nifti and
computing Dice against a target GT nifti. It runs the **spacing-cascade** eval
mode (coarse→fine, default `[4, 1.5]` mm) so it reproduces the accuracy path of:

```
experiments/3d/eval.py dataset=totalseg data.val_classes=all \
  eval.model=patchset3d eval.split=test eval.checkpoint=.../best.pt \
  eval.feat_norm=self data.mask_downsample=occupancy data.mask_occupancy_thr=0.1 \
  eval.spacing_sweep=[4,1.5] eval.spacing_cascade=true eval.crop_jitter=0
```

but on **arbitrary nifti files** and **GT-free for the target**.

### Why the cascade is GT-free for the target

The dataset's high-accuracy crop path organ-centres the target on its GT
centroid, which a real target does not have. The cascade removes that
dependency: the coarse pass FOV per axis is `T·s0` mm. For the test checkpoint
`T = 128` (see below) at `s0 = 4mm` → **512 mm** (51.2 cm), which spans the whole
trunk. Because `_organ_crop_arrays` clamps the crop to `min(native_extent,
T·s0mm)`, any axis whose scan extent is ≤ that is captured in full (centering is
moot), and larger axes get a centred `T·s0mm` window — either way the coarse FOV
reliably contains the organ for localization, so it can crop on the **volume
centre** with no target GT. The fine pass (`s1 = 1.5mm` → 192 mm FOV at T=128)
then re-centres the target crop on the **coarse prediction's centroid**. Only the
*contexts* need masks (provided), used to organ-crop them on the mask centroid.

### Fidelity params come from the composed cfg (T = 128 via `dataset=totalseg`)

`data.image_size` (T) and the other data-fidelity knobs are owned by the eval
config, not restored from the checkpoint. They arrive automatically through the
`dataset=totalseg` config group, which sets `image_size: [128,128,128]`,
`use_crop: true`, `crop_spacing_mm: 1.5`, `context_size: 1`, and
`mask_downsample`. Because `predict_nifti` takes the **same cfg surface** as
`eval.py` (composed with `dataset=totalseg`), T=128 flows in for free — the
caller does not pass it explicitly. This matches the test checkpoint's patchset3d
`conv` ConvEncoder3D, whose `enc_dims` length (4) encodes three stride-2 stages
(128→64→32→16 = `log2(128/16)+1`) and therefore needs 128³ input. As a safety
net the method still calls `eval._warn_uninherited_data(cfg)` to flag any
train/eval drift on the fidelity keys vs the checkpoint's stored `data`.

## Non-goals

- No batching / multi-class loop, no wandb, no figure generation (that is
  `evaluate_spacing_sweep`'s job). This is a single-target predict.
- No support for the resized (`use_crop=false`) path — the method is
  cascade-only.
- No new model types; whatever `eval._build_model(cfg)` produces (patchset3d,
  medverse, native_resenc, …) is used as-is via its `.predict`.

## Public interface

New module `experiments/3d/infer_nifti.py`:

```python
def predict_nifti(
    cfg,                       # OmegaConf cfg — same surface as experiments/3d/eval.py
    target_path,               # str | Path — target CT .nii.gz
    context_pairs,             # list[(img_path, mask_path)] — mask = binary organ
    gt_path=None,              # optional target GT (binary) .nii.gz for metrics
    out_path=None,             # optional; write predicted mask .nii.gz here
) -> dict:
    ...
```

Returns:

```python
{
    "pred":              np.ndarray,   # bool, native target grid (D,H,W)
    "affine":            np.ndarray,   # target affine (4,4), for reference
    "dice":              float | None, # native binary Dice vs GT (stitched cascade)
    "coarse_only_dice":  float | None, # native Dice from the coarse pass alone
    "pred_path":         Path | None,  # set when out_path given
}
```

Config keys consumed (all already on the eval cfg surface):
`eval.model`, `eval.checkpoint`, `eval.feat_norm`, `eval.medverse_ckpt`,
`eval.sw_roi_size`; `data.image_size`, `data.mask_downsample`,
`data.mask_occupancy_thr`; and the cascade spacings from
`eval.spacing_sweep` (default `[4, 1.5]`). `eval.crop_jitter` is forced to `0`
(deterministic centred crops) for inference regardless of cfg.

## Reuse strategy (approach A)

Zero duplication of the crop geometry and native-stitch math. Reused as-is:

- `experiments/3d/eval.py::_build_model(cfg)` — model construction + checkpoint
  restore (patchset3d arch-from-checkpoint, feat_norm override, medverse, …).
- `src.totalseg_dataset.normalize_ct` — CT → model input space (global pointwise,
  so crop == whole-volume normalisation; we normalise once up front).
- `experiments/3d/evaluate.py::_predicted_native_center(prob, geom)` — coarse
  prediction centroid → native voxel centre (or `"volume_center"` when empty).
- `experiments/3d/evaluate.py::_write_native(native, pred, geom)` — composite a
  crop-grid prediction into the native volume at its crop location (finer
  overwrites coarser), used for the stitched output + metrics.

### Refactor: extract `organ_crop_arrays` as a pure function

Today `TotalSegInContextDataset._organ_crop_arrays` mixes (a) disk IO
(`_load_native_ct_mmap`, raw_ct normalisation) with (b) pure array-level crop
geometry (target/crop sizes, jittered starts, slicing, `out_sizes`/`pad_lo`,
`crop_geom`). Extract (b) into a module-level function in
`src/totalseg_dataloader_incontext.py`:

```python
def organ_crop_arrays(ct_mm, label_mm, center, sp, *, image_size, crop_mm,
                      jitter, rng):
    """Pure array-level organ crop -> (crop_ct, crop_lbl, out_sizes, pad_lo, crop_geom)."""
```

`_organ_crop_arrays` becomes a thin wrapper: it loads/normalises the CT (disk
concerns) then calls `organ_crop_arrays`, keeping `self._last_crop_geom` set as
before. **Behaviour-preserving** — same numbers, same jitter draw order from
`rng`. Also reuse the resample-to-`T³` placement:

- `_place_image`, `_place_label`, `_resample_binary` are small instance methods
  that read only `self.image_size` / `self.mask_downsample` /
  `self.mask_occupancy_thr`. Extract their bodies into module-level helpers
  (`place_image`, `place_label`, `resample_binary(..., mode, occ_thr)`) that the
  methods then call, so `infer_nifti` shares them without instantiating a
  dataset. Behaviour-preserving.

The single-target 2-pass cascade loop (~40 lines) lives in `infer_nifti.py`; it
is far simpler than the batched, multi-class, figure-emitting
`evaluate_spacing_sweep`, so reimplementing that thin orchestration is cleaner
than bending the dataset-coupled loop.

## Data flow

For each pass at spacing `s` (coarse `s0=4`, fine `s1=1.5`):

1. **Load** (once, cached across passes): target CT array + affine via nibabel;
   each context CT array + binary mask array. Native mm/voxel spacing per volume
   = `abs(affine)` zooms (`nibabel.affines.voxel_sizes`), replacing the dataset's
   `spacings.json`. Normalise every CT with `normalize_ct` up front.
2. **Context crops**: for each context, centre = binary-mask centroid (voxel
   COM); `organ_crop_arrays(ct, mask, center, sp_ctx, ...)` → `place_image` /
   `resample_binary` → `(1,T,T,T)` image + `(T,T,T)` mask. Stack to
   `context_in (K,1,T,T,T)`, `context_out (K,T,T,T)`.
3. **Target crop**:
   - coarse (`s0`): centre = target **volume centre**.
   - fine (`s1`): centre = `_predicted_native_center(coarse_prob, coarse_geom)`
     (or volume centre if `"volume_center"`).
   Keep the target `crop_geom` for stitching.
4. **Predict**: `model.predict(target_img[None], context_in[None],
   context_out[None])` → `(1,T,T,T)` → squeeze to `(T,T,T)` hard mask.
   Wrap in `torch.no_grad()` + the eval device; a spacing kwarg is passed only
   when `getattr(model, "spacing_aware", False)` (mirrors `evaluate.py`).
5. **Stitch**: allocate native-shaped `bool` volume; `_write_native` the coarse
   pred, then `_write_native` the fine pred (fine overwrites). That is the output
   mask on the native target grid.

K handling: K = the number of provided context pairs. `predict` accepts any K,
so we do **not** pad/truncate to `data.context_size` — the number of contexts the
caller passes is exactly what the model sees.

## Output & metrics

- `out_path`: `nib.save(nib.Nifti1Image(pred.astype(uint8), target_affine),
  out_path)`.
- `gt_path` given: load GT (binary, `>0`), resample to native target shape if
  its grid differs (nearest); `dice` = binary Dice of stitched pred vs GT;
  `coarse_only_dice` = binary Dice of the coarse-only native composite (coarse
  `_write_native` alone) vs GT. Uses the same Dice as `evaluate.dice_binary`
  (imported).

## Error handling

- Missing file / unreadable nifti → raise `FileNotFoundError` / propagate
  nibabel error (fail loud).
- Empty context mask (no foreground) → fall back to that volume's centre for its
  crop, with a `warnings.warn` (mirrors the dataset's empty-centroid fallback).
- `context_pairs` empty → `ValueError` (in-context needs ≥1 context).
- Empty coarse prediction → fine pass crops on the volume centre (via
  `_predicted_native_center` returning `"volume_center"`), no crash.

## Testing

One lightweight test (`experiments/3d/tests/`) using tiny synthetic niftis
(e.g. a cube organ in a small volume, K=1 self-context): assert output shape ==
native target shape, dtype bool, and that a self-context run yields
`dice > 0.5`. No GPU/model-heavy assertions — the goal is wiring/geometry, not
accuracy. Use `medverse` (no checkpoint) or a tiny stub model to keep it cheap.

## Files touched

- `experiments/3d/infer_nifti.py` — new module (`predict_nifti` + helpers).
- `src/totalseg_dataloader_incontext.py` — extract pure `organ_crop_arrays`,
  `place_image`, `place_label`, `resample_binary`; make the existing methods thin
  callers (behaviour-preserving).
- `experiments/3d/tests/test_infer_nifti.py` — new minimal test.
- `docs/logs.md` — log the change.

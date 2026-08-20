# In-Context DataLoader v2 — Design

**Date:** 2026-08-20
**Status:** Design (approved in brainstorming; pending spec review)

## Motivation

`src/totalseg_dataloader_incontext.py` (`TotalSegInContextDataset`, ~1600 lines)
has accreted across many research iterations. It tangles three concerns that
should be independent:

1. **In-context task assembly** — sample a target, sample K contexts of the same
   class, augment, package the item dict.
2. **TotalSegmentator-specific I/O** — four load paths (pre-resized fast /
   organ-crop / `.nii.gz` slow / `raw_ct`), bbox/scan/adjacency caches, npy
   layout, crop geometry.
3. **Research-probe machinery** — supervoxel synth (`p_synth`), `self_context.*`
   probes, multi-label + SegGPT coloring, coarse→fine cascade eval.

Because assembly and I/O are fused, every other data source
(`synth_gmm_maisi`, `synth_gen_maisi`, `omnisynth3d`, `anchor_synth3d`,
`chemotox_bc`, `totalseg_more_labels`) re-implements both, and correctness leans
on mutable per-item instance state (`_cur_rng`, `_cur_crop_spacing`,
`_last_crop_geom`, `_pred_centers`) that is safe only because a worker processes
one item at a time.

### Goals

- **Separate task-assembly from I/O** via a small source-specific provider
  interface reused by a generic engine.
- **Simplify the config surface** — collapse the ~30 interacting `__init__`
  params into engine-side (sampling/aug) vs provider-side (I/O) knobs; drop
  multiplicatively-combined flag paths that are not required.
- **Throughput** — one disk read per load, no redundant `np.unique`/resample, no
  side-channel state.

### Non-goals (explicitly out of scope for v2)

- Supervoxel synth (`p_synth`) and adjacency-merge caches.
- `self_context.*` probes (per-image/intensity re-aug, ellipse/supervoxel
  synth-mask substitution, `coords.npy`).
- `num_labels_per_sample > 1`, SegGPT `label_palette`.
- Coarse→fine **cascade** logic (only *anticipation hooks* are in scope — see
  "Cascade anticipation").
- Fast (pre-resized) / pre-normalized `ct.npy` / `.nii.gz` slow load paths.

The old `TotalSegInContextDataset` stays untouched and keeps serving all of the
above; v2 coexists behind a config flag.

## Architecture

Approach A: **thin provider + generic engine**, with an explicit
`LoadRequest`/`LoadResult` dataclass pair carrying all per-item state so the
mutable side-channels disappear.

```
InContextDataset (generic engine, the only torch Dataset)
    └── holds a VolumeProvider
            └── TotalSegProvider (the one source-specific adapter for v2)
```

New files:

- `src/incontext_dataset_v2.py` — `InContextDataset`, `LoadRequest`,
  `LoadResult`, `VolumeProvider` protocol.
- `src/providers/totalseg.py` — `TotalSegProvider`.

`src/totalseg_dataloader_incontext.py` is **not modified**.

### Provider interface

```python
@dataclass
class LoadRequest:
    rng: random.Random         # per-item RNG (eval determinism or global `random`)
    crop_spacing_mm: float     # physical crop pitch for THIS item (variable-spacing aware)
    center: tuple[int, int, int] | None = None
        # native-voxel crop center. None -> provider uses its own default
        # (bbox centroid). A caller-supplied center is the seam a future cascade
        # uses to place the fine crop on a predicted location. v2 always passes
        # None; the provider fills the centroid internally.

@dataclass
class LoadResult:
    image: Tensor      # (1, T, T, T) float32, normalized
    label: Tensor      # (T, T, T) int64, binary {0,1}
    spacing: Tensor    # (3,) mm/voxel of the output tensor
    crop_geom: Tensor  # (4, 3) long: [starts, crop_sizes, out_sizes, pad_lo]
    # Future cascade extension (NOT in v2): an optional `buffer` field carrying
    # the pre-crop high-res (image, label) + geometry so the GPU loop can re-crop
    # at a second spacing without a disk round-trip. Adding it is a new field,
    # not a contract change.

class VolumeProvider(Protocol):
    classes: list[str]
    def subjects_for(self, cls: str) -> list[str]: ...
    def load(self, subject: str, cls: str, req: LoadRequest) -> LoadResult: ...
```

### `TotalSegProvider` (single raw_ct crop path)

Owns: scan cache (subject→classes, all 117 classes), bbox cache
(subject→class→native centroid), `spacings.json`, and MRI `ct_stats.json`. No
adjacency cache, no synth cache.

`load(subject, cls, req)` does exactly:

1. mmap native `ct_raw.npy` (int16 raw HU for CT; raw + per-volume `ct_stats`
   for MRI) and `label.npy`. **Missing `ct_raw.npy` is a hard error** — no
   fallback to `ct.npy` / `.nii.gz`.
2. `center = req.center or bbox_cache[subject][cls]` (volume center if absent).
3. `crop_and_place(ct_raw, label, center, req.crop_spacing_mm, T)` (see below).
4. Normalize the cropped image slice on the fly (`normalize_ct` global /
   `normalize_mri` per-volume) — normalize the **crop**, not the whole volume.
5. Return `LoadResult(image, label=(crop == class_idx), spacing=crop_spacing_mm
   isotropic, crop_geom)`.

### The pure crop function (single source of crop geometry)

Extract one device-agnostic function that is the *only* place crop geometry is
computed:

```python
def crop_and_place(image_np, label_np, center, spacing_mm, T, *,
                   crop_spacing_mm, jitter, rng, mask_downsample, occ_thr):
    # physical extent T*crop_spacing_mm -> crop_sizes -> slice around center
    # (+jitter) -> resample real slice to out_sizes (trilinear image /
    # occupancy|nearest label) -> centre-pad to T^3.
    # returns (image (1,T,T,T), label (T,T,T), crop_geom (4,3))
```

This reuses/absorbs the existing pure helpers `organ_crop_arrays`,
`place_image`, `place_label`, `resample_binary`. Keeping this math in exactly
one place is what makes a future GPU (torch) twin of the same geometry safe.

### Generic engine (`InContextDataset`)

Config knobs (engine-side only): `provider`, `context_size`, `class_balanced`,
`aug_cfg`, `defer_aug`, `crop_spacing_mm` (default), `eval_seed`. Provider-side
I/O knobs (`mask_downsample`, `mask_occupancy_thr`, `crop_jitter`, `raw_ct`
paths, `modality`) live on `TotalSegProvider`.

`__init__` builds `samples = [(subj, cls) for cls in provider.classes for subj in
provider.subjects_for(cls)]` and `active_classes`.

`__getitem__(idx)`:

1. Decode `idx`: plain `int`, or `(idx, spacing)` from `SpacingBatchSampler` →
   local `crop_spacing` (else default). No instance mutation.
2. `rng = Random(hash((eval_seed, idx)))` when `eval_seed` set, else global
   `random`.
3. Pick target `(subj, cls)` — `class_balanced` (class-uniform then subject) or
   `samples[idx]`.
4. `req = LoadRequest(rng, crop_spacing)`; `tgt = provider.load(subj, cls, req)`.
5. Context sampling: `_lazy_shuffle(rng, subjects_for(cls) − subj)`,
   `provider.load` each until `context_size`; on load failure skip. Clone-pad if
   short; if zero candidates, leaky self-context clone + `warnings.warn`.
6. Augment: when `aug_cfg.enabled and not defer_aug`, `apply_task_aug` (shared
   geometry across target+K) then per-volume `apply_intensity_aug`; else emit raw
   clones. `aug_mode` tag preserved (0 = real cross-subject).
7. Package the item dict.

### Item schema & collate (contract preserved)

Emits exactly the keys the current cross-subject real path emits:

```
image, label, context_in, context_out, subject, context_subjects,
label_name, spacing, aug_mode, crop_geom
```

`incontext_collate_fn` and `SpacingBatchSampler` are **reused unchanged**. The
dropped optional keys (`synth_radii_mm`, `synth_coord`, `label_palette`) are
simply never produced.

## Cascade anticipation (hooks only, no cascade code in v2)

A future training-time cascade (coarse pass → predicted center → fine crop at
finer spacing) must keep the dataloader off the critical path between passes.
The efficient interaction is: **the dataloader ships one high-res buffer per
item; the GPU loop crops both scales from it** (mirroring the existing
`SynthRealizer`/`GpuAugmentor` `gpu_realize` pattern). Feeding predicted centers
back into `__getitem__` for a second disk read (today's `_pred_centers` eval
approach) is rejected for training — it serializes GPU→sync→disk→GPU and kills
prefetch.

v2 bakes in three cheap hooks so cascade is a later *extension*, not a rewrite:

1. **`center` is a `LoadRequest` field.** Crop location is a general geometry
   input; the engine/provider never learn where it came from. v2 passes `None`
   (provider uses the centroid); cascade passes a predicted center.
2. **`crop_and_place` is one pure function.** A future GPU realizer reuses the
   same physical-extent→sizes→pad geometry math; no forked CPU/GPU crop logic.
3. **`LoadResult` has room for a `buffer` variant.** A future provider mode
   returns the pre-crop high-res `(image, label)` + geometry instead of the
   cropped T³ — a new field, not a changed contract.

## Wiring

`experiments/3d/common.py` gains a v2 branch gated by a config flag
(`data.loader_v2: true`) that builds `InContextDataset(TotalSegProvider(...))`
for the totalseg family. Every existing run (flag absent/false) is unaffected —
the old `build_dataset` path is the default. `train_loader` / `make_eval_loader`
reuse `SpacingBatchSampler` and `incontext_collate_fn` for the v2 path exactly as
today.

## Testing

- Provider unit test: `load()` returns correct shapes/dtypes, normalized image
  range, binary label, isotropic reported spacing, valid `crop_geom`; hard-fails
  on missing `ct_raw.npy`.
- `crop_and_place` unit test: physical extent → out_sizes/pad invariants;
  non-empty label under occupancy downsample for a thin structure.
- Engine test: item schema keys + shapes match the current cross-subject path;
  `eval_seed` reproducibility across two `__getitem__` calls; K-padding and
  zero-candidate self-context fallback.
- Parity spot-check: for a fixed (subject, class) and centroid center, v2's
  raw_ct crop matches the old `_load_crop` (raw_ct=True) output within
  resampling tolerance.

## Risks

- **Precondition shift**: v2 requires `ct_raw.npy` + `label.npy` + bbox cache +
  `spacings.json` (+ MRI `ct_stats.json`) for every subject. Datasets built only
  with pre-resized/normalized files won't work under v2 until `ct_raw.npy`
  exists. Surfaced as a hard error at load.
- **Buffer size (future cascade)**: the high-res buffer spans
  `coarse_extent_mm / fine_spacing_mm` voxels/axis; cap it (cf.
  `gpu_realize_max_native`). Not a v2 concern.

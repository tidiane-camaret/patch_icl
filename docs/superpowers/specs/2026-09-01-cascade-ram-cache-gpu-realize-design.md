# Cascade RAM cache + GPU crop/realize — design

Date: 2026-09-01
Status: approved (brainstorming), pending spec review
Driver: `experiments/3d/experiment/59_organs_cascade_from_scratch.yaml`

## Problem

The N-level coarse→fine cascade (`experiments/3d/cascade.py`, `run_cascade`) re-loads and
re-crops `.npy` image + label volumes at every training step. The config header itself
documents the cost:

- `_recrop_level` (levels ≥ 1) runs **synchronously inside the training step with the GPU
  idle**: `B * (K+1)` = 32 provider loads, each an NFS-mmap `np.load` + `organ_crop_arrays`
  (numpy slice) + `place_image` (`F.interpolate` trilinear) + `resample_binary`
  (`avg_pool3d`) + `normalize_ct`. Measured **~+0.3 s/step**.
- The cascade val pass (`evaluate_cascade`) runs the same re-crop loop: **~100 s per val
  pass**.

Level 0 is produced by the normal v2 DataLoader workers and overlaps the GPU, so it is not
the acute cost — but under this design it is converted too (see Scope).

## Established facts (verified 2026-09-01)

- TotalSegmentator root has **1230 subject dirs, 1228 with data**. `spacings.json`: **every
  subject is 1.5 mm isotropic** (min/median/max all 1.5). So `ct_raw.npy` *is* the 1.5 mm
  volume; a `ct_raw_1.5mm.npy` cache would be a byte-identical copy — not built.
- On-disk totals: `ct_raw.npy` (fp16) **32 GB**, `label.npy` (uint8) **3.2 GB**,
  `ct_raw_3mm.npy` (fp16) 3.9 GB, `ct_raw_6mm.npy` ~0.3 GB.
- Data lives on **NFS** (`configs/cluster/nfs.yaml`) — mmap random reads pay network
  round-trips.
- `59_organs_cascade_from_scratch` chain: `57_organs_encoder_from_scratch` →
  `48_abdomen_ceiling` → `dataset=d1`. `image_size = [128,128,128]` (T=128),
  `cascade_spacings = [3, 1.5]`, `context_size = 3`, `mask_downsample: soft` (from 57),
  `mask_occupancy_thr: 0.5`, `ct_norm: null` → `fingerprint_1228`
  (`CtNormSpec(clip_lo=-1007, clip_hi=1573, mean=-167.3, std=505.8)`).
- An analogous GPU-realize path already exists for the synth source: `data.gpu_realize`
  flag, `gpu_realize_max_native`, list-preserving collate (`common.py:455`,
  `common.py:582`), `src/gpu_synth_realize.py::SynthRealizer` + `_resample_member`,
  invoked from `train.py:576` (`if synth_realizer is not None and "native_lbls" in batch`).
  The v2 engine already tolerates items with no `"image"` key
  (`src/incontext_dataset_v2.py:112` — "gpu_realize items ship a native-crop payload …
  painted + augmented on-GPU downstream").
- GPU aug is already mandatory for cascade: `_assert_cascade_supported` errors unless
  `augmentations.gpu=true` when `augmentations.enabled` (`common.py:216`).

## Decisions (from brainstorming)

1. **Scope**: RAM cache **+ full GPU pipeline** — both cascade levels realized on GPU.
2. **Cache contents**: all 1228 subjects, `ct_raw.npy` + `label.npy` only. **Drop the 3 mm
   cache** — the coarse (3 mm) level is derived from the cached 1.5 mm volume.
3. **Cache mechanism**: fork-COW `dict` of read-only numpy arrays, preloaded in
   `TotalSegProvider.__init__` before the DataLoader forks workers.
4. **Non-cascade v2 path**: the GPU-realize path is *built* but wired to `run_cascade` only.
   The plain single-level v2 train/eval loaders keep the current CPU worker crop.
5. **Correctness bar**: semantic equivalence (interpolation-tolerance image error, exact
   occupancy/soft semantics, byte-identical `crop_geom`) — not bit-parity. `exp59` is
   from-scratch; no in-flight run to reproduce.
6. **Naming**: reuse the pattern, not the code — `data.gpu_realize_crop` flag, new
   `src/gpu_realize_crop.py` sibling of `src/gpu_synth_realize.py`.
7. **Coarser-level H2D**: integer-decimate the RAM crop toward `out_sizes` with an
   `avg_pool3d` prefilter before H2D, then finish the non-integer resample on GPU. Keeps
   H2D near ~T³/member regardless of the level's pitch.
9. **Any `cascade_spacings`**: length ≥ 2, arbitrary values — `[3, 1.5]`, `[6, 3, 1.5]`,
   non-2× ratios. Nothing assumes 2 levels or power-of-two pitch ratios; the decimation
   factor is derived per-axis from the crop, not from the spacing.
8. **Singleton**: one process-lifetime cache shared by the train provider and the separate
   eval-loader provider (both constructed before any fork).

## Architecture

```
TotalSegProvider.__init__
  └─ volume_cache.get_cache(root, subjects)         # singleton, fork-COW, read-only ndarrays
                                                    #   {subject: {"ct_raw": fp16(D,H,W), "label": uint8(D,H,W)}}

level 0  (cascade train loader, gpu_realize_crop=true)
  worker: InContextDataset.__getitem__
    └─ provider.load_native_crop(...)  ×(K+1)       # RAM slice + int-decimate + avg_pool3d prefilter
    → item = {"native_crop": [NativeCrop ...], "subject", "context_subjects", "label_name", ...}   # no "image"
  collate: list-preserving (native_crop stays a list)
  train_epoch cascade branch:
    └─ crop_realizer(batch, DEVICE)                 # → batch{image,label,context_in,context_out,spacing,crop_geom}
    └─ augmentor.apply(...)                         # unchanged
    └─ run_cascade(...)

level i ≥ 1  (run_cascade._recrop_level)
  └─ provider.load_native_crop(...)  ×B(K+1)  over the existing _RECROP_POOL threadpool
  └─ realize_native_crops(natives, device=DEVICE)  # same collated batch shape as today
  └─ (rest of run_cascade unchanged: augmentor, _forward_level, invert_geo_center, prior warp)
```

### Component A — `src/providers/volume_cache.py`

```python
_CACHE: dict[str, dict[str, np.ndarray]] = {}   # module singleton keyed by str(root)

def get_cache(root, subjects, *, max_subjects=None, workers=16) -> dict[str, dict]:
    """Load ct_raw.npy (fp16) + label.npy (uint8) for `subjects` into a process-lifetime
    dict. Arrays are set read-only (arr.flags.writeable = False) so fork() COW keeps the
    buffers page-shared across DataLoader workers and the main-process provider. Idempotent:
    a second call with the same root returns the existing dict and tops up any missing
    subjects. Threaded np.load (GIL released during I/O)."""
```

- Called from `TotalSegProvider.__init__` only when `ram_cache` is set.
- `max_subjects` → `data.ram_cache_max_subjects` (debug).
- Memory: ~35 GB resident in the parent; ~0 extra per worker (COW).
- Startup: one-time threaded NFS read (minutes). Logged with an elapsed line.

### Component B1 — `TotalSegProvider.load_native_crop`

```python
@dataclass
class NativeCrop:
    image: torch.Tensor      # (d,h,w) fp16, native-pitch, integer-decimated toward T
    label: torch.Tensor      # (d,h,w) uint8, SAME decimation as image
    class_idx: int
    out_sizes: list[int]     # from organ_crop_arrays
    pad_lo: list[int]
    crop_geom: torch.Tensor  # (4,3) i64 — identical to crop_and_place's
    crop_spacing_mm: float
    decim: tuple[int,int,int]

def load_native_crop(self, subject, cls, req: LoadRequest) -> NativeCrop:
    ...
```

- Geometry via the existing `organ_crop_arrays(label_cache, label_cache, center, native_sp,
  image_size=(T,T,T), crop_mm=req.crop_spacing_mm, jitter=..., rng=req.rng)` — consumes
  `req.rng` **exactly once** (eval determinism preserved), yields the same `crop_geom` the
  current path returns.
- `center` default: `self._bbox[subject][cls]` (unchanged); `req.center` seam for level ≥ 1.
- Slice `ct_raw` + `label` caches with the crop box. `.contiguous()` copies the small crop
  out of the COW buffer (~a few MB) — the cache pages stay clean.
- `decim[a] = max(1, floor(crop_sizes[a] / out_sizes[a]))` — per-axis, derived from the
  crop geometry, not the pitch. Guarantees the decimated crop stays ≥ `out_sizes`
  (GPU never has to upsample), handles `[6,3,1.5]` (decim 4/2/1) and non-2× ratios, and
  self-limits to 1 when the crop is already small (clamped by the volume dims). Apply
  strided `avg_pool3d` (image) and area-pool→uint8 (label — multi-class, class selection
  happens on GPU) by `decim`; the remainder is dropped (prefilter only, GPU `F.interpolate`
  to exact `out_sizes` fixes final geometry). `crop_geom` is computed from the **native**
  crop and passed through untouched.
- No `normalize_ct`, no `F.interpolate` to `out_sizes`, no placement — all GPU.

### Component B2 — `src/gpu_realize_crop.py`

```python
def realize_native_crops(
    natives: list[list[NativeCrop]] | list[NativeCrop],   # per-b list of (K+1), or flat
    *, T, mask_downsample, occ_thr, ct_spec, device,
) -> dict:
    """→ {image (B,1,T,T,T) f32, label (B,T,T,T) {soft f32 | occupancy i64},
         context_in (B,K,1,T,T,T), context_out (B,K,T,T,T),
         spacing (B,3) f32, crop_geom (B,4,3) i64}."""
```

Per member, on `device`, autocast disabled:

- **image**: `src = crop.float()[None,None]`; `F.interpolate(src, out_sizes,
  mode="area")` if any axis downsamples else `mode="trilinear", align_corners=False`;
  GPU `normalize_ct` = `((x.clamp(clip_lo, clip_hi) - mean) / std)`; scatter into
  `torch.full((1,T,T,T), float(img.min()))` at `pad_lo` — the *resampled* member's own
  normalized min, byte-for-byte the rule `place_image` (`totalseg_dataloader_incontext.py`)
  and `_resample_member` (`gpu_synth_realize.py:91`) already use for the air pad.
- **mask**: `binm = (crop == class_idx).float()[None,None]`; `frac = _area_pool_3d(binm,
  out_sizes)` (GPU); then branch ported verbatim from `resample_binary`:
  - `soft`: `frac.clamp(0,1)`; if `binm.any()` and `peak < occ_thr` → lift the peak
    cell(s) to `occ_thr`. Output f32.
  - `occupancy`: `out = frac >= occ_thr`; if empty and `binm.any()` →
    `out.view(-1)[frac.argmax()] = True`. Output i64.
  - scatter into `torch.zeros((T,T,T))` at `pad_lo`.
- Ragged `out_sizes` across members → per-member Python loop (B·(K+1) ≈ 32 tiny GPU
  calls — the shape `SynthRealizer._resample_member` already runs at).
- `spacing` = `full((3,), crop_spacing_mm)`; `crop_geom` passed straight through.

`ct_spec` GPU helper: a 3-line `normalize_ct_gpu(t, spec)` reusing the `CtNormSpec` fields
and its normalized-floor property (`totalseg_dataset.py:46`); keep the numpy `normalize_ct`
untouched for the mmap path.

### Component C — cascade wiring

**`cascade.py::_recrop_level`**: add a `realize` switch (provider has `load_native_crop`
and `gpu_realize_crop` on):

```python
if use_realize:
    natives = pool.map(lambda t: provider.load_native_crop(t.subj, clss[t.b],
        LoadRequest(rng=random.Random(t.rk), crop_spacing_mm=sp,
                    center=t.center, jitter=jitter)), tasks)
    return realize_native_crops(_regroup(natives, B, K), T=T,
        mask_downsample=..., occ_thr=..., ct_spec=provider.ct_spec, device=device)
```

Returns the same dict shape `incontext_collate_fn(items)` returns today → the rest of
`run_cascade` (`_forward_level`, `target_like`, `invert_geo_center`, `_build_query_prior`
M2 warp, `_centroid_from_logit`) is unchanged. `crop_geom` identical, so
`_stitched_native_dice_multi` / `evaluate_cascade` unchanged.

Keep `_RECROP_POOL` — per-task CPU work is now `organ_crop_arrays` + one `avg_pool3d`,
near-zero, but the pool is harmless and preserves task-order reassembly.

**Level 0**: `common.train_loader` — when `cfg.data.get("gpu_realize_crop")` **and**
`cascade_spacings` set:
- `InContextDataset` is told `gpu_realize_crop=True` → in the non-cohort branch,
  `__getitem__` builds `context_in/out` as a `native_crop` list via
  `provider.load_native_crop` and returns `{"native_crop": [...], "subject",
  "context_subjects", "label_name", "spacing_mm": crop_spacing, "aug_mode": 0}` with **no
  `"image"`** (engine already skips its CPU aug for imageless items).
- collate: reuse/extend the list-preserving collate at `common.py:582` (currently gated to
  `source == "synth_gmm_maisi"`) to also fire for `gpu_realize_crop`.
- `train.py` `main()`: build `crop_realizer = make_crop_realizer(cfg)` (mirrors
  `synth_realizer` at `train.py:1121`), pass into `train_epoch`.
- `train_epoch` cascade branch (before `run_cascade`): `if "native_crop" in batch: batch =
  crop_realizer(batch, DEVICE)` — then `augmentor.apply` and `run_cascade` as today.

**`common._assert_cascade_supported`**: if `cascade_spacings` set —
- default `gpu_realize_crop` → `True` (opt out with `=false`);
- default `ram_cache` → `True`; error if `gpu_realize_crop and not ram_cache` with a
  message pointing here (RAM cache is the point; NFS mmap under realize is worse).

### Component D — config

`configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml`:
- add `data.ram_cache: true`, `data.gpu_realize_crop: true`.
- replace the "IMAGE CACHES (measured)" header block (about building `ct_raw_3mm.npy` /
  `ct_raw_1.5mm.npy`) with a note that the run holds `ct_raw.npy` + `label.npy` for all
  subjects in RAM (~35 GB) and realizes both levels on GPU; recommend a ≥ 96 GB node.

## Testing

- `experiments/3d/tests/test_gpu_realize_crop.py` (needs the totalseg root; skip if
  absent): for ~6 `(subject, class, center, spacing∈{3,1.5})` cases, compare
  `realize_native_crops([load_native_crop(...)])` against `crop_and_place(...)`:
  - normalized image: `max |Δ|` < 2e-2, `mean |Δ|` < 2e-3;
  - `soft` mask: `max |Δ|` < 1e-4; `occupancy` mask: Dice == 1.0;
  - `crop_geom`: `torch.equal`.
  Run on CPU (`device="cpu"`) so it needs no GPU.
- `test_cascade.py`: add cases that run one `run_cascade` step with
  `gpu_realize_crop=true` on `device="cpu"` (monkeypatched tiny provider) for **both**
  `cascade_spacings=[3,1.5]` and `[6,3,1.5]` — asserts per-level shapes, `centers` /
  `empty_frac` populated for every `i≥1`, `crop_geom` dtype/shape, and 3-level prior-warp
  chaining (`prev_*` across levels 0→1→2).
- `test_volume_cache.py`: `get_cache` returns `writeable=False` arrays, is idempotent per
  root, and `TotalSegProvider.subjects_for` is unchanged with `ram_cache` on vs off.

## Risks / mitigations

| Risk | Mitigation |
|---|---|
| COW page copy if a cached array is mutated | `arr.flags.writeable = False`; crops `.contiguous()` out small copies; parity test would catch a silent write |
| ~35 GB resident on a 64 GB node | `_assert_cascade_supported` logs the recommendation; `ram_cache_max_subjects` for debug; default node bump to ≥ 96 GB |
| Startup NFS read latency (~minutes) | threaded `np.load`; one-time per process; `persistent_workers=true` already set so workers fork once |
| Coarse-level decimation ≠ offline `ct_raw_3mm.npy` filter | `avg_pool3d` prefilter before the GPU resample; within the semantic-equivalence bar (decision 5/7) |
| `mask_downsample: soft` peak-floor + occupancy argmax-fallback semantics | ported verbatim from `resample_binary`; asserted by the parity test in both modes |
| eval-loader builds its own provider | singleton `get_cache(root, …)` — both providers share the arrays, constructed before any fork |
| non-cascade v2 path regression | untouched — `gpu_realize_crop` / list-preserving collate / `crop_realizer` only fire when `cascade_spacings` is set |

## Non-goals

- Converting the plain single-level v2 train/eval loaders to GPU realize (deferred; the
  hook is built).
- Anisotropic sources (FLARE22 / NasalSeg providers) — separate classes, out of scope.
- Deleting any on-disk `.npy` — the mmap fallback path stays fully functional.
- Bit-exact reproduction of prior runs.

## Expected outcome

- `_recrop_level`: 32 NFS-mmap `np.load` + CPU `F.interpolate` → RAM slice (µs) +
  ~130 MB H2D + ~32 tiny GPU resamples on the idle GPU. The documented **~0.3 s/step** and
  **~100 s/val pass** should largely disappear.
- Level 0: worker CPU data-time (~6 ms/item) drops to a RAM slice + one `avg_pool3d`.
- Cost: one-time ~35 GB RAM + startup read.

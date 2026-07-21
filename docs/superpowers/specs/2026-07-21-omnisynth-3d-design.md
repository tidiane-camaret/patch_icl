# omniSynth 3D — TotalSegmentator organs on a 3D canvas

**Date:** 2026-07-21
**Status:** Design approved, pending spec review

## Goal

Extend omniSynth to compose **3D** in-context scenes by painting bbox-cropped
TotalSegmentator organs at random 3D positions onto a `D×H×W` canvas. Reuse
omniSynth's existing bank + dataset-orchestration layers; add a **parallel
`render3d.py`** for volumetric composition (the 2D `render.py` is left untouched).

Output matches the existing **3D pipeline** contract emitted by
`TotalSegInContextDataset` (verified against `src/totalseg_dataloader_incontext.py`)
so `ResEncInContext3D` / `ViTInContext3D` and `incontext_collate_fn` consume it
with no changes:

| key | shape | dtype |
|---|---|---|
| `image` | `(1, D, H, W)` | float32 |
| `label` | `(D, H, W)` (no channel dim) | int64 |
| `context_in` | `(K, 1, D, H, W)` | float32 |
| `context_out` | `(K, D, H, W)` (no channel dim) | int64 |
| `subject` | — | str |
| `label_name` | — | str |
| `spacing` | `(3,)` | float32 |

Note this differs from the 2D omniSynth contract (which carries a channel dim on
the label and a `meta` dict) — the 3D dataset packages the 3D-pipeline contract.

**Primary non-functional constraint: dataloading time.** Every design fork below
is resolved in favour of the fastest hot path.

## Scope decisions

| Decision | Choice | Rationale |
|---|---|---|
| Render layer | **Parallel `render3d.py`**, 2D `render.py` unchanged | Low risk to the working 2D path; the two share the bank + dataset layers, not the compositing helpers |
| Scene model | **Free 3D placement** (random positions + anti-overlap), native canvas-relative organ sizes | Preserves organ size diversity; no grid quantisation. Grid placement in 3D is **out of scope** |
| Object source | **Precompute-once, read-small tile cache** (see Bank) | Fastest possible dataloading; no full-volume reads or cropping in the hot path |
| `dataset.py` shape | **Thin subclass** `OmniSynth3DICLDataset(OmniSynthICLDataset)` | Keeps the 2D path byte-identical; reuses RNG/target-mode/copy helpers verbatim |
| `target_mode` (v1) | **`identical` + `class` only**; `aug` deferred | `identical`/`class` need **zero** per-item scipy warps — the hot path stays pure numpy compositing |

## Contour-accurate pasting (not bbox rectangles)

Objects are already pasted by their **true contour**, not as filled bbox
rectangles — this is existing 2D behaviour that carries over unchanged to 3D:

- A rendition is `[2, T, T, T]`: **ch0 = intensity** (zeroed outside the mask),
  **ch1 = binary mask**. The bbox is only the *crop extent*; the real shape lives
  in ch1.
- Compositing is `region = region*(1 - mask) + intensity` — texture lands only
  where the mask is 1.
- The label uses `np.maximum(label, mask)` — the real contour.
- **Anti-overlap operates on the true footprint** (`mask_tile > 0`), so bbox
  rectangles may overlap as long as the actual voxels do not. This matters *more*
  in 3D: large irregular organs (ribs, vessels) have sparse contours inside loose
  bboxes, so bbox-based occupancy would reject far too many placements.
  `render3d.py`'s `_occupy_3d` / `_overlap_frac_3d` stay contour-based.

No extra cost vs bbox: the mask channel is loaded regardless (it defines the
organ), and the compositing/occupancy math is identical elementwise ops.

## Components

### 1. `bank_totalseg.py` — `TotalSegObjectBank`

Mirrors the bank interface (`task_ids(split)` / `get(class_id)` / `alphabet(class_id)`)
so it drops into the same sampler + render machinery.

- `class_id` = a TotalSeg organ label; name from `data.totalseg_classes.ALL_CLASSES`.
  `alphabet(cid)` returns the class name.
- A **rendition** = one subject's organ, bbox-cropped from the pre-resized
  `label_{D}x{H}x{W}.npy`, with the intensity patch from `ct_{D}x{H}x{W}.npy`,
  built into a `[2, T, T, T]` fp16 tile at its **final on-canvas size** (canvas
  `size_mode`, aspect preserved).
- Splits from `meta.csv` (train / val / test). Reuses the existing
  `{subject: frozenset[classes]}` scan-cache concept to build the
  `label → subjects` index for the split.

**Dataloading-optimised storage (the core of this design):**

- A **one-time build script** (`scripts/synth3d/build_totalseg_tiles.py`, à la
  `convert_to_npy.py`) crops every organ once and writes a **per-class tile cache**
  to disk: `totalseg_tiles/T{T}/{split}/class_{id}.npz` holding that class's
  renditions as fp16 tiles (ragged sizes stored as an object array or padded stack
  + shapes).
- At train time, `get(cid)` loads that one small class file **once** and indexes
  renditions in RAM; an **LRU across classes** bounds memory. The hot path does
  **no** full-volume reads, **no** cropping, **no** resizing — only numpy
  compositing of `T³` tiles.
- Size estimate: ~117 classes × ≤200 renditions × `2·T³` fp16 ≈ **1–3 GB** on disk
  for `T ∈ [24, 32]`.
- `get(cid)` returns an indexable view (`__len__` = #renditions, `__getitem__(i)` →
  `[2,T,T,T]`) so `make_target_sampler` / `make_distractor_sampler` work unchanged.

### 2. `render3d.py` — 3D twin of `render.py` (free-placement path)

`render_scene_3d(rng, scene, target_sampler, distractor_sampler,
background_sampler=None) → (image[D,H,W], mask[D,H,W], k, info)`.

3D versions of the helpers, same semantics as 2D:
`_tile_slices_3d`, `_composite_3d`, `_paste_3d` (union), `_occupy_3d`,
`_overlap_frac_3d`, `_place_random_3d`, `_clamp_center_3d`, `_paste_centroid_3d`.
`k` target objects of the target class + distractors from other classes, placed at
random 3D positions with contour-based anti-overlap rejection (`tries` candidates,
keep least-overlapping, accept early under `placement_max_overlap`). Background v1:
black or noise-field (real-volume "image" background deferred).

### 3. `dataset.py` — `OmniSynth3DICLDataset(OmniSynthICLDataset)`

Thin subclass:
- Overrides bank construction → `TotalSegObjectBank`.
- Overrides the per-item `scene(rng)` call → `render_scene_3d`.
- **Reuses verbatim**: `_subject_rngs`, `_item_rng`, `_resolve_target_mode`,
  copy-slot injection, deterministic eval seeding.
- `_to_img_tensor` generalised to add a channel dim for a 2D **or** 3D array.
- `meta` reports 3D centroids; the 2D grid `(row, col)` provenance (`divmod`) does
  not apply under free placement and is dropped.

The 2D `OmniSynthICLDataset` path stays byte-identical.

### 4. `config.py` — `OmniTotalSegConfig`

- `data_root` ← `cfg.paths.totalseg`; `tiles_root` for the precomputed cache.
- `classes` subset (`() = all present`), `tile_size` `T`, `size_mode` /
  `size_scale`, `max_renditions_per_class`, `min_mask_vox`, `max_subjects`.
- Canvas = `image_size³`; object count / `k_min` / `k_max` / anti-overlap `tries` /
  `placement_max_overlap` reuse `OmniSceneConfig`.

### 5. Integration — `experiments/3d/common.py`

`build_dataset(cfg, split)` gains a `source=omnisynth3d` branch →
`OmniSynth3DICLDataset`, mirroring how `experiments/2d/common.py` wires
`source=omnisynth`. A `configs/experiment/3d/…` config selects it.

## Data flow

```
build_dataset(cfg, split)            # source=omnisynth3d
  └─ OmniSynth3DICLDataset
       per __getitem__(idx):
         seed rngs (deterministic for eval)
         resolve target_mode (identical | class)
         TotalSegObjectBank → target / distractor samplers   # tiles from disk cache (LRU)
         render_scene_3d(query rng)          → image, mask, k, info
         render_scene_3d(context rng) × K
         optional copy-slot injection (train)
       return {                              # TotalSegInContextDataset contract
         image:       [1, D, H, W] float32,
         label:       [D, H, W]    int64,
         context_in:  [K, 1, D, H, W] float32,
         context_out: [K, D, H, W]    int64,
         subject:     f"omni_{class_id}_{sample_index}",
         label_name:  class_name,
         spacing:     [3] float32 (ones — synthetic canvas),
       }
       # collated by the existing incontext_collate_fn
```

## Testing

Parallel to the existing `test_render.py` / `test_bank_*`:

- `test_render3d.py` — trivial samplers; output shapes `[D,H,W]`; `k` target cells
  in the label; contour-based anti-overlap; union labels; determinism given a seed.
- `test_bank_totalseg.py` — interface parity (`task_ids`/`get`/`alphabet`); tile
  shape `[2,T,T,T]`; split scoping; LRU behaviour. Uses a tiny synthetic tile cache
  fixture (no dependency on the real TotalSeg store).
- `test_dataset3d.py` — integration: correct 3D tensor shapes; eval determinism
  (byte-identical items for a fixed `(namespace, task_id, sample_index)`).

## YAGNI / deferred

- **`aug` target_mode** (3D affine jitter): start with none; add scale+translate
  later; **3D rotation deferred** (needs per-plane axis handling and adds scipy
  warps to the hot path).
- **Grid placement in 3D** — out of scope (free placement chosen).
- **Real-volume "image" backgrounds** and **biomedparse-3D** — deferred; 3D
  backgrounds are black / noise-field in v1.

## Open build-order note

The precomputed tile cache is a hard prerequisite for training, so the build
script (`build_totalseg_tiles.py`) and `TotalSegObjectBank` land first and are
validated on a small subject subset before `render3d.py` / the dataset wrapper.

# Feature-Sim: Pluggable Primus Encoder (frozen CoLiPri) — Design

**Date:** 2026-07-27
**Status:** design (approved for spec)
**Env:** `.venv_thor` (thor, RTX A6000 sm_86, torch 2.5.1+cu121)

## Goal

Let the feature-similarity study (`experiments/3d/feature_sim/`) evaluate an **arbitrary
frozen encoder** — first target: **CoLiPri** (`microsoft/colipri`), whose vision tower is a
stock nnUNet **Primus-M** ViT. We want to see how a frozen, pretrained Primus encoder behaves
on the existing intrinsic feature-matching metrics (`retrieval_at1`, `transfer_dice`,
`prototype_cosine`) *without training anything*. Whether to later train the full in-context
model with a Primus encoder is a separate, downstream decision this study informs.

Alongside feature-matching quality, log the **image-encoding compute cost** of whichever
encoder is running — **FLOPs, peak VRAM, and it/sec (volumes/s)** — so quality-vs-cost is
captured in one place. Encoding here is frozen/forward-only, so this is an *inference*-cost
measurement (distinct from `encoder_bench`, which times training fwd+bwd).

Non-goals: no training, no in-context head for CoLiPri, no new comparison/baseline machinery
(comparison against existing PatchSet3D runs is done offline from the CSVs), no domain
tagging. YAGNI.

## Background / why this is clean

- CoLiPri's `ImageEncoder(backbone: Primus, projector: Conv3d, pooler: AttentionPool1D)` wraps
  `from dynamic_network_architectures.architectures.primus import Primus` — the **standard
  nnUNet Primus**, already installed in `.venv_thor`. So its weights are a plain Primus
  `state_dict`; nothing custom.
- Dense feature extraction is a plain ViT backbone forward → tokens `(B,N,C)` → grid
  `(B,C,g,g,g)`. The CoLiPri-specific parts (processor preprocessing, attention pooling, CLIP
  projection) all live on the **pooled** path, which feature-matching does not use.
- Primus-M uses patch `8³` and (per the CoLiPri report) **no absolute positional embeddings**
  (dynamic input size). So the adapter accepts our native grid (`128³ → 16³` tokens);
  resampling toward CoLiPri's ~2mm training spacing is a *feature-quality knob*, not a hard
  requirement.
- The intrinsic metrics in `run.py` (`prototype_cosine`, `fg_match_margin`, `retrieval_at1`,
  `label_transfer`) are computed from **features + labels only** → encoder-agnostic already.
  Only `real_dice` (from `model.predict`) and the `transformer_*` probes are PatchSet3D-specific.

## Architecture

```
cfg.eval.model ─▶ build_adapter(cfg) ─┬─ "patchset3d" → PatchSet3DEncoderAdapter
                                      │        (+ real_dice via model.predict, + transformer tiers)
                                      └─ "primus"     → PrimusEncoderAdapter
                                               (frozen; intrinsic metrics only; real_dice=None)
```

The encoder becomes a swappable component behind the existing `EncoderAdapter` ABC. `real_dice`
and transformer tiers are treated as **PatchSet3D-only extras** — absent for a generic encoder.
The PatchSet3D path stays behaviorally identical (regression-safe).

## Components

### 1. `PrimusEncoderAdapter(EncoderAdapter)`  (new)
Location: `experiments/3d/feature_sim/adapters.py` (sibling of `PatchSet3DEncoderAdapter`).

- **Construction:** `PrimusEncoderAdapter(weights_path, primus_kwargs, resample_spacing=None)`.
  Builds `Primus(**primus_kwargs)`, loads `weights_path` (a plain Primus `state_dict`;
  `None` = random-init floor), moves to device, `.eval()`, `requires_grad_(False)`.
- **Feature tap:** returns the ViT **token features** the same way CoLiPri's `ImageEncoder`
  taps the backbone (encoder tokens *before* the segmentation decoder). The exact submodule /
  forward-hook point is fixed during implementation by reading CoLiPri's `image.py`; recorded
  in the extraction step (Component 2) alongside `primus_kwargs`.
- **ABC methods** (mirror `PatchSet3DEncoderAdapter` semantics):
  - `R` → native token grid res at the study's input res (`input_res // 8`).
  - `tiers()` → `["backbone"]` (raw 768-dim token grid). Optional future tier `"projector"`
    (CoLiPri's Conv3d dense head) — omitted for now.
  - `native_res("backbone", input_res)` → `input_res // 8`.
  - `features(volumes, tier, res)` → `_preprocess(volumes)` → backbone tokens → reshape to
    `(B,C,g,g,g)` → `_down_to(res)`.
  - `sample_features(volumes, tier, coords)` → same grid → `F.grid_sample` (copy the
    PatchSet3D adapter's coord flip / bilinear logic).
- **`_preprocess(volumes)`:** the loader stores z-scored HU (`x = (clip(HU,MIN,MAX) - CT_MEAN)/CT_STD`).
  Reconstruct approx HU (`HU = x·CT_STD + CT_MEAN`), apply CoLiPri's intensity normalization,
  and (if `resample_spacing` set) trilinearly resample toward ~2mm. Exact CoLiPri normalization
  values come from its processor (read during implementation); default `resample_spacing=None`
  (feed native grid) with the option recorded for a quality sweep.
- **No** `predict` / `transformer_*` methods.

### 2. `scripts/extract_colipri_backbone.py`  (one-off)
Isolates **all** CoLiPri-package coupling to a single offline step.
- `pip install colipri` into `.venv_thor` (one-off).
- Load CoLiPri, pull the `backbone.*` sub-`state_dict`, save `results/checkpoints/primus_colipri.pt`.
- Emit the exact `primus_kwargs` used by CoLiPri (embed_dim, patch_embed_size=8³, eva_depth,
  eva_numheads, `use_abs_pos_embed=False`, `use_rot_pos_emb`, input_channels=1, num_classes)
  and the processor's intensity-normalization constants → written into a small JSON/py sidecar
  next to the weights so the adapter/config need no CoLiPri import at run time.

### 3. `run.py`  (small refactor, regression-safe)
- Add `build_adapter(cfg)` factory dispatching on `cfg.eval.model`:
  - `"patchset3d"` → existing `_load_patchset` + `PatchSet3DEncoderAdapter` (unchanged).
  - `"primus"` → `PrimusEncoderAdapter(weights=cfg.eval.weights, primus_kwargs=..., ...)`.
- `real_dice`: computed only when the adapter exposes a segmenter (PatchSet3D). Otherwise
  `None`. The coupling analyses (`_spearman`/`_partial_spearman` vs `real_dice`) already drop
  non-numeric rows → they degrade to `None` cleanly; no crash.
- Transformer tiers (`transformer_q`, `transformer_layers`) run only when present in
  `cfg.feature_sim.tiers`; the Primus config omits them.
- Everything else (sweep, metrics, CSV, aggregation) is untouched.

### 4. Encoder cost probe  (new, generic — works for any adapter)
Location: `experiments/3d/feature_sim/cost.py` (small helper), reuses
`encoder_bench.profiling.count_gflops`.

- **What:** one forward encode of a single volume `(1,1,D,H,W)` at the study input res, on the
  adapter's primary encoder module, **frozen / `no_grad` / autocast bf16** (matches how
  `features()` runs).
- **Metrics:** `encode_gflops` (fvcore on the encode forward; `None` if untraceable, e.g. SDPA),
  `encode_vram_mb` (`torch.cuda.max_memory_allocated` peak, inference), `encode_it_s`
  (`1000 / median_forward_ms` over a few warmup + timed iters → volumes/s).
- **Generic hook:** each adapter exposes `cost_target(input_res) -> (module, example_inputs)`
  returning the traceable encoder module + the inputs it sees (Primus backbone on preprocessed
  volume; PatchSet3D `enc` on `[image, zero-mask]`). The probe is encoder-agnostic — same call
  for `patchset3d` and `primus`, so cost is directly comparable.
- **When/where:** measured once at run start (before the task loop); written to
  `encode_cost.csv` in `out_dir` and logged to wandb as summary scalars. Not per-task (encoding
  cost is a per-encoder/per-res constant, not a task property).

### 5. Config
- `configs/experiment/3d/feature_sim.yaml`: add `eval.model` switch surface and
  `eval.weights` (path; required for `model=primus`, ignored for `patchset3d`).
- A Primus variant (e.g. `configs/experiment/3d/experiment/feature_sim_primus.yaml` or CLI
  overrides): `eval.model=primus`, `eval.weights=results/checkpoints/primus_colipri.pt`,
  `feature_sim.tiers=[backbone]`, `feature_sim.resolutions=[8,16]` (native ≤ 16³ at 128³ input),
  `primus_kwargs` sourced from the extraction sidecar.

## Data flow (Primus path)

```
eval loader → volume (1,1,128,128,128) z-scored HU
  → _preprocess: reconstruct HU → CoLiPri norm → [resample ~2mm] 
  → Primus backbone (patch 8³) → tokens (1,N,768) → grid (1,768,16,16,16)
  → _down_to(res) / grid_sample(coords)
  → prototype_cosine / retrieval_at1 / label_transfer(target ↔ K contexts)
  → per-(task,tier,res) CSV row (real_dice = None)

once at start:  cost_target → fvcore FLOPs + peak inference VRAM + median-forward it/s
                → encode_cost.csv + wandb scalars
```

## Error handling / edge cases

- **Missing weights file** → clear error at adapter construction.
- **`model=primus` without `eval.weights`** → explicit config error (mirrors the existing
  "checkpoint required" guard for patchset3d).
- **Random-init floor:** `eval.weights=null` builds an untrained Primus (architecture-only
  baseline) — supported, not required.
- **Input not divisible by 8** → resample or pad in `_preprocess`; assert the token grid is
  integer.
- **Regression safety:** with `eval.model=patchset3d` (default) the driver must produce
  identical rows to today. Verified by a before/after diff on a small `n_subjects` run.

## Testing

- Unit: `PrimusEncoderAdapter.features`/`sample_features` return expected shapes/res on a
  random-init Primus (no weights, no CoLiPri needed) — cheap CPU/GPU smoke test.
- Integration: `feature_sim/run.py eval.model=primus eval.weights=…` on a handful of subjects
  writes a well-formed CSV with `real_dice` empty and non-null `retrieval_at1`/`transfer_dice`.
- Regression: `eval.model=patchset3d` on a small run yields rows identical to pre-refactor.
- Cost probe: `encode_cost.csv` is written with sane, positive `encode_it_s`/`encode_vram_mb`
  for a random-init Primus; `encode_gflops` populated (or explicitly `None` if untraceable).

## Open implementation details (resolved during the plan, not blockers)

1. Exact backbone feature-tap point in CoLiPri's `image.py` (which token tensor = "features").
2. CoLiPri processor's intensity-normalization constants + whether ~2mm resampling materially
   changes features (a quick `resample_spacing` on/off sweep).
3. Exact `primus_kwargs` from CoLiPri's config (embed_dim / eva_depth / eva_numheads for "M").

# omniSynth — in-context Omniglot dataset (design)

Date: 2026-06-29

## Purpose

An in-context 2D segmentation dataset built from the Omniglot handwritten-character
dataset. Each item is a 64×64 scene of characters laid out on a 4×4 grid; the task is
to segment every cell holding a given **target character class**, using K context
(image, mask) pairs that share that target class. It plugs into the existing 2D
pipeline as a new `data.source=omnisynth`, returning the same contract as
`SynthICLDataset` / `MedSegBenchDataset` (4-key dict + `meta` + `.samples`).

## Task & scene definition

- A **task** = one target character *class* (one `(alphabet, characterNN)`).
- A **4×4 grid** on a 64×64 canvas → sixteen 16×16 cells, **all 16 filled**.
- `k ~ U[k_min, k_max]` cells (drawn fresh per image) hold the **target class**; the
  remaining `16 − k` hold **distractors** (any other character class, any alphabet in
  the split, excluding the target class).
- Background uniform `0`; characters white `1` (Omniglot ink is inverted on load).
- **Mask** = the `k` target cells (binary).
- The **query** and all **K contexts** share the same target class; each is rendered
  independently (its own `k`, cell positions, distractor classes).

### `target_mode` — what "same target" means across the k cells + query/contexts

- `identical` — one rendition bitmap reused in every target placement.
- `aug` — one base rendition + independent light geometric aug (rotate/scale/translate)
  per placement.
- `class` — a different rendition (drawing/person) of the same class per placement.

## Data flow & rendering

### Bank (`bank.py`)

Built once at init, process-shared via a module-level cache so forked DataLoader
workers inherit it (mirrors controlSynth's `_BANK_CACHE`).

- Opens the split's zip: `images_background.zip` (train) or `images_evaluation.zip`
  (val/test), read directly with `zipfile` — no unzip step.
- Builds an index `class_id → [rendition arrays]` enumerating every
  `(alphabet, characterNN)` in that split, plus `class_id → alphabet` (kept for future
  within-alphabet distractors; unused in V1).
- Each rendition is loaded once, **inverted** (ink→1), resized to cell size with
  anti-alias, re-binarized at 0.5, stored as small `uint8`. ~32k tiny images → a few MB.
- API: `task_ids(split)` → class pool; `get(class_id)` → rendition list.

### Item construction (`dataset.py` `__getitem__`)

1. Pick target `class_id` (train: random from pool; eval: from `_eval_index`).
2. Get K+1 independent RNGs. Eval: spawned from
   `SeedSequence([namespace, task_id, sample_index])`. Train: fresh entropy.
3. For the query and each of K contexts, `_render_scene(target_class, rng)`:
   - draw `k ~ U[k_min, k_max]`; choose `k` target cells + `16 − k` distractor cells via
     a random permutation of the 16 cells.
   - choose distractor classes (one random rendition each, from pool minus target).
   - per target cell, pick the rendition per `target_mode`
     (`identical`: fixed bitmap; `aug`: base bitmap + per-cell affine jitter;
     `class`: random rendition of target).
   - paste each cell's bitmap into its 16×16 block (centered, `cell_margin` padding);
     build the binary mask from target cells.
4. Return query `image`/`label` and stacked context `context_in`/`context_out`.

### Tensors

`_to_img_tensor` → `[1,64,64]` float32; contexts stacked `[K,1,64,64]`. Same helper and
shapes as controlSynth.

## Params & config

```python
@dataclass
class OmniDiversityConfig:
    master_seed: int = 42
    omniglot_root: str = "/home/dpxuser/repos/omniglot/python"  # dir holding the zips
    train_zip: str = "images_background.zip"
    eval_zip:  str = "images_evaluation.zip"
    val_test_split: float = 0.5   # fraction of eval-alphabet classes -> val (rest -> test)

@dataclass
class OmniSceneConfig:
    grid: int = 4                 # 4x4 cells
    k_min: int = 1                # target cells ~ U[k_min, k_max]
    k_max: int = 6
    cell_margin: float = 0.1      # fractional padding inside each cell
    target_mode: str = "class"    # identical | aug | class
    aug_rotate: float = 15.0      # deg; aug-mode per-cell jitter
    aug_scale: float = 0.1        # +/- log-scale
    aug_translate: float = 0.1    # fraction of cell

@dataclass
class OmniSamplingConfig:
    epoch_length: int = 10000
    eval_subjects_per_task: int = 4
    eval_seed_namespace: int = 0
```

### Dataset signature (matches `SynthICLDataset` style)

```python
OmniSynthICLDataset(split="train", context_size=3, image_size=64,
                    diversity=..., scene=..., sampling=..., deterministic=None)
```

`image_size` must be divisible by `grid`; cell size = `image_size // grid`. `deterministic`
defaults to `split != "train"`.

### Splits

- train pool = all character classes in the **background** alphabets (`train_zip`).
- val/test pools = classes in the **evaluation** alphabets (`eval_zip`), partitioned by
  `val_test_split` (deterministic split keyed on `master_seed`).
- distractors are drawn from the **same split's** pool, so val/test stay cleanly held out.

### Determinism

train draws fresh entropy per subject (infinite diversity). val/test derive every
subject seed from `(eval_seed_namespace, task_id, sample_index)` → byte-identical eval
set across runs. The bank's class index is deterministic in `master_seed`.

### `.samples` contract

- train → `[("omniglot/train", i, 1) for i in range(epoch_length)]`.
- eval → one entry per `(class_id, subject)` with a self-documenting name
  `omniglot/<alphabet>`, so `run_eval`'s per-dataset grouping stratifies Dice for free
  (same mechanism controlSynth uses).

## Hydra config

`configs/experiment/2d/synth/omniglot.yaml` (`# @package synth`), structured like
`synth/default.yaml` with `diversity` / `scene` / `sampling` blocks. `paths.omniglot`
points at the dir holding the two zips (default `/home/dpxuser/repos/omniglot/python`).

## Integration

New branch in `build_dataset` (`experiments/2d/common.py`), alongside `synthetic`:

```python
if source == "omnisynth":
    from src.datasets.omniSynth import (
        OmniDiversityConfig, OmniSceneConfig, OmniSamplingConfig, OmniSynthICLDataset,
    )
    s = cfg.synth
    return OmniSynthICLDataset(
        split=split, context_size=cfg.data.context_size, image_size=cfg.data.image_size,
        diversity=OmniDiversityConfig(omniglot_root=cfg.paths.omniglot, **dict(s.diversity)),
        scene=OmniSceneConfig(**dict(s.scene)),
        sampling=OmniSamplingConfig(**dict(s.sampling)),
    )
```

Add `omnisynth` to the unknown-source error list. No training/eval script changes —
they route through `build_dataset` and consume the same 4-key dict + `.samples`.

### File layout

Small package `src/datasets/omniSynth/`:
- `config.py` — the three dataclasses.
- `bank.py` — zip index + image loading/resize/cache.
- `dataset.py` — `OmniSynthICLDataset`.
- `__init__.py` — exports configs + dataset.

## Testing

Light (repo guideline: tests only when necessary). Cover what's most likely to break:

- shapes: `image`/`label` `[1,64,64]`, contexts `[K,1,64,64]`; `image_size % grid != 0`
  guard raises.
- mask correctness: target cells are 1, distractor cells 0; `k ∈ [k_min, k_max]`.
- determinism: same `(split=val)` index → byte-identical tensors across two instances;
  train differs.
- split disjointness: train (background) and eval (evaluation) class pools don't overlap.
- optional preview script: render a grid of items to PNG for eyeballing (like the
  Omniglot demo).

## Logging

Append a changelog entry to `docs/logs.md`.

## Out of scope (V1)

- within-alphabet distractors (index keeps the `alphabet` field for later).
- per-task difficulty sweep / binned val grid (controlSynth-style).
- offline precompute / LMDB store.

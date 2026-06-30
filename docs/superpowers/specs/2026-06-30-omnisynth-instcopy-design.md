# omniSynth instCopy (copy-tasks where target = context)

**Date:** 2026-06-30
**Status:** design approved, pending implementation

## Motivation

In the in-context segmentation contract, the model can cheat by solving the query
from learned semantic priors (segment the salient/known character) without reading
the support. That simplicity-bias trap is the segmentation analog of the IWL
shortcut in the induction-head ICL literature. The mitigation borrowed from that
work is **instCopy**: seed the support with an exact/lightly-augmented copy of the
query so the image→mask look-up starts as a near-identity mapping, which lets the
matching mechanism form early and then generalize to non-identical support. A
single copy in the context is sufficient.

This spec adds copy-tasks to `OmniSynthICLDataset` as a **training-only**
bootstrap. Eval (val/test) is left byte-identical to today so few-shot ICL numbers
stay honest (no instance leakage between support and query).

## Scope

- One new pure helper in `render.py` (scene-level paired affine jitter).
- A gated copy block in `OmniSynthICLDataset.__getitem__`.
- Two new `OmniSceneConfig` fields (`p_copy`, `n_copy`).
- Two new `meta` fields (`is_copy`, `copy_slot`).
- Tests in `test_dataset.py`.

Out of scope: changes to the model, training loop, or any other dataset. No new
augmentation parameters — copy jitter reuses the existing `aug_*` fields.

## Mechanism

In `__getitem__`, the query `(t_img, t_seg)` and the K context scenes are rendered
exactly as today. Then, **only when `not self.deterministic`** and a fresh isolated
rng draws `< p_copy`:

1. Choose `n = min(n_copy, context_size)` distinct context slots uniformly.
2. For each chosen slot `j`, overwrite `ctx[j]` with a lightly-augmented copy of the
   query scene: `affine_jitter_scene(t_img, t_seg, scene, rng)` (independent jitter
   per slot). The slot's reported `k` is the query's `t_k`.

The remaining context slots keep their normal renditions of the same target class,
so the task definition (segment *this* class) is preserved; the copy only makes one
or more support pairs a near-identity example of the query.

Scenes here are **binary** (image and mask both in {0,1}), so the copy jitter is
purely geometric — there is no photometric component. The same transform is applied
to the image and the mask so they stay aligned.

### Determinism

- The copy decision, slot choice, and jitter all draw from a **fresh
  `np.random.default_rng()`** created inside the copy block, so they never perturb
  the existing `_subject_rngs` / `_item_rng` seeding.
- The block is gated on `not self.deterministic`. Val/test therefore produce **no
  copies regardless of `p_copy`** and remain byte-identical to the current dataset.
- `p_copy = 0.0` disables the feature entirely (train output identical to today).

## Config (`OmniSceneConfig`)

```python
p_copy: float = 0.9   # train-only per-item probability of injecting copy slot(s)
n_copy: int = 1       # number of context slots to copy when an item is a copy-task
                      # (clamped to context_size); each copied slot jittered independently
```

Copy jitter magnitude reuses the existing `aug_rotate / aug_scale / aug_translate`
fields, applied at full scene resolution rather than per cell.

## `render.py` — `affine_jitter_scene`

```
affine_jitter_scene(img, mask, scene, rng) -> (img_out float32 [H,W], mask_out float32 [H,W])
```

- Draws one rotate / scale / translate (same parameterization as the per-cell
  `affine_jitter`: `aug_rotate` deg, `2**U[-aug_scale, aug_scale]`,
  `aug_translate * H` pixels).
- Applies the **same** transform to both `img` and `mask` (rotate `reshape=False`,
  `_zoom_to_size` back to `[H,W]`, shift), `order=1`, `mode="constant"`, `cval=0`.
- Returns `img` as float32 and `mask` re-thresholded to {0,1} float32.

Pure and bank-free, mirroring the existing render helpers so it is unit-testable
with trivial inputs.

## `meta` additions

- `is_copy: bool` — whether this item injected any copy slot.
- `copy_slot: int` — the first copied slot index, or `-1` when none. (With `n_copy>1`
  this is the lowest copied index; a full list is unnecessary for current logging.)

## Tests (`test_dataset.py`)

1. **Copy present & aligned:** with `p_copy=1.0, n_copy=1`, exactly one context slot
   has high mask-IoU with the query mask and the others are below a low threshold;
   `meta["is_copy"]` is True and `meta["copy_slot"]` points at that slot.
2. **Multi-slot:** with `p_copy=1.0, n_copy=2` (and `context_size>=2`), two distinct
   slots are near-copies.
3. **Eval has no copies:** deterministic split with `p_copy=1.0` yields
   `is_copy=False` for every item and is byte-identical to `p_copy=0.0`.
4. **Disabled is unchanged:** train with `p_copy=0.0` matches current behavior
   (`is_copy=False`, no copy slot).

## Risks / notes

- A whole-scene affine rotates the entire grid of characters; image/mask stay
  aligned so this is a valid "same scene, lightly moved" rendition and avoids a
  degenerate exact-duplicate detector.
- `n_copy` is clamped to `context_size`; with `context_size==1` a copy replaces the
  only context, which collapses to pure identity — acceptable but worth noting when
  configuring small K.

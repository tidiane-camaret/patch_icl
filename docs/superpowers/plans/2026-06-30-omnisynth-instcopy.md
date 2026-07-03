# omniSynth instCopy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add training-only "copy-tasks" to `OmniSynthICLDataset` — with probability `p_copy`, overwrite `n_copy` context slots with a lightly-augmented copy of the query scene — to bootstrap the image→mask induction look-up.

**Architecture:** A new pure scene-level paired affine helper in `render.py`; a gated copy block in `OmniSynthICLDataset.__getitem__` that runs only on the non-deterministic (train) path and draws from an isolated rng so existing seeding is untouched; two new `OmniSceneConfig` fields and two new `meta` fields.

**Tech Stack:** Python, numpy, scipy.ndimage, torch, plain assert-function tests (no pytest).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-30-omnisynth-instcopy-design.md`.
- Python interpreter: `.venv311/bin/python`. **No pytest** — tests are plain functions run via a `__main__` block.
- Run all tests from the repo root `/home/dpxuser/dev/patch_icl` (test files do `sys.path.insert(0, ".")`).
- **Do not run `git commit` or `git add`** — version control is the user's. Each task ends by appending a line to `docs/logs.md` instead of committing.
- Scenes are **binary** (image and mask in {0,1}); copy jitter is geometric only.
- Copy jitter reuses existing `aug_rotate / aug_scale / aug_translate` — **no new aug params**.
- Eval (val/test) must stay byte-identical to today regardless of `p_copy`: copy logic is gated on `not self.deterministic`.

---

### Task 1: `affine_jitter_scene` scene-level paired jitter

**Files:**
- Modify: `src/datasets/omniSynth/render.py` (add function after `affine_jitter`, ~line 64)
- Test: `src/datasets/omniSynth/test_render.py`

**Interfaces:**
- Consumes: existing module-level imports in `render.py` (`np`, `nd_rotate`, `nd_shift`, `_zoom_to_size`) and `OmniSceneConfig` fields `aug_rotate`, `aug_scale`, `aug_translate`.
- Produces: `affine_jitter_scene(img, mask, scene, rng) -> (img_out: np.ndarray float32 [H,W], mask_out: np.ndarray float32 [H,W])`. One shared rotate/scale/translate applied to both inputs; `img_out` float32, `mask_out` re-thresholded to {0,1} float32.

- [ ] **Step 1: Write the failing test**

Add to `src/datasets/omniSynth/test_render.py` (and import the new symbol on line 4: `from src.datasets.omniSynth.render import render_scene, affine_jitter, affine_jitter_scene`):

```python
def test_affine_jitter_scene_pairs_and_aligns():
    # A small square in both image and mask: after a shared jitter, image and mask
    # stay aligned (mask is exactly the >0 part of the image here), shapes preserved,
    # mask binary float32.
    scene = OmniSceneConfig()
    H = GRID * CELL
    img = np.zeros((H, H), dtype=np.float32)
    img[20:40, 20:40] = 1.0
    mask = img.copy()
    img_out, mask_out = affine_jitter_scene(img, mask, scene, np.random.default_rng(7))
    assert img_out.shape == (H, H) and mask_out.shape == (H, H)
    assert img_out.dtype == np.float32 and mask_out.dtype == np.float32
    assert set(np.unique(mask_out)).issubset({0.0, 1.0})
    # identical input transforms => mask tracks image foreground exactly
    assert np.array_equal((mask_out > 0.5), (img_out > 0.5))

def test_affine_jitter_scene_zero_params_is_identity():
    # with all jitter ranges 0, the transform is a no-op (round-trip preserves content)
    scene = OmniSceneConfig(aug_rotate=0.0, aug_scale=0.0, aug_translate=0.0)
    H = GRID * CELL
    img = np.zeros((H, H), dtype=np.float32)
    img[20:40, 20:40] = 1.0
    img_out, mask_out = affine_jitter_scene(img, img.copy(), scene, np.random.default_rng(0))
    assert np.array_equal((img_out > 0.5), (img > 0.5))
```

Add both calls to the `__main__` block and before the final print.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dpxuser/dev/patch_icl && .venv311/bin/python src/datasets/omniSynth/test_render.py`
Expected: FAIL with `ImportError: cannot import name 'affine_jitter_scene'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/datasets/omniSynth/render.py` after `affine_jitter` (before `make_target_sampler`):

```python
def affine_jitter_scene(img, mask, scene, rng):
    """Apply one shared rotate/scale/translate to a full-resolution (image, mask)
    pair so they stay aligned. Returns (img float32 [H,W], mask float32 {0,1} [H,W]).
    Used for instCopy: a lightly-augmented copy of the whole query scene."""
    size = img.shape[0]
    angle = rng.uniform(-scene.aug_rotate, scene.aug_rotate)
    scale = 2.0 ** rng.uniform(-scene.aug_scale, scene.aug_scale)
    dy = rng.uniform(-scene.aug_translate, scene.aug_translate) * size
    dx = rng.uniform(-scene.aug_translate, scene.aug_translate) * size

    def _xform(a):
        a = nd_rotate(a.astype(np.float32), angle, reshape=False, order=1,
                      mode="constant", cval=0.0)
        a = _zoom_to_size(a, scale, size)
        a = nd_shift(a, (dy, dx), order=1, mode="constant", cval=0.0)
        return a

    img_out = _xform(img).astype(np.float32)
    mask_out = (_xform(mask) > 0.5).astype(np.float32)
    return img_out, mask_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/dpxuser/dev/patch_icl && .venv311/bin/python src/datasets/omniSynth/test_render.py`
Expected: PASS — prints `ALL RENDER TESTS PASSED`.

- [ ] **Step 5: Log the change**

Append to `docs/logs.md`:

```
- omniSynth: added render.affine_jitter_scene (scene-level paired affine jitter) for instCopy copy-tasks.
```

---

### Task 2: config fields + dataset copy block + meta

**Files:**
- Modify: `src/datasets/omniSynth/config.py` (`OmniSceneConfig`, ~lines 17-26)
- Modify: `src/datasets/omniSynth/dataset.py` (import ~line 19; `__getitem__` ~lines 104-138)
- Test: `src/datasets/omniSynth/test_dataset.py`

**Interfaces:**
- Consumes: `affine_jitter_scene(img, mask, scene, rng)` from Task 1; existing `render_scene`, samplers, and `_to_img_tensor`.
- Produces: two `OmniSceneConfig` fields `p_copy: float = 0.9`, `n_copy: int = 1`; two `meta` keys `is_copy: bool` and `copy_slot: int` (lowest copied slot index, or `-1`).

- [ ] **Step 1: Write the failing tests**

Add the import on line 4 of `src/datasets/omniSynth/test_dataset.py` if not present (it already imports `OmniSceneConfig, OmniSamplingConfig`). Add these tests (define `_iou` helper near the top, after `_ds`):

```python
def _iou(a, b):
    a = a > 0.5; b = b > 0.5
    inter = (a & b).sum().item(); union = (a | b).sum().item()
    return inter / union if union else 1.0


def test_copy_injects_one_aligned_slot():
    # p_copy=1, n_copy=1 (train): exactly one context slot is a near-copy of the
    # query (high mask-IoU), the rest are not; meta flags it.
    ds = _ds("train", p_copy=1.0, n_copy=1)
    item = ds[0]
    assert item["meta"]["is_copy"] is True
    slot = item["meta"]["copy_slot"]
    assert 0 <= slot < K
    ious = [_iou(item["context_out"][j, 0], item["label"][0]) for j in range(K)]
    assert ious[slot] > 0.5, f"copy slot IoU too low: {ious}"
    others = [v for j, v in enumerate(ious) if j != slot]
    # a real same-class context can occasionally overlap; require the copy to be the max
    assert ious[slot] == max(ious), f"copy slot not the strongest match: {ious}"


def test_copy_multi_slot():
    # p_copy=1, n_copy=2: two distinct slots are near-copies of the query.
    ds = _ds("train", p_copy=1.0, n_copy=2)
    item = ds[0]
    assert item["meta"]["is_copy"] is True
    ious = [_iou(item["context_out"][j, 0], item["label"][0]) for j in range(K)]
    n_high = sum(v > 0.5 for v in ious)
    assert n_high >= 2, f"expected >=2 copy slots, ious={ious}"


def test_eval_never_copies():
    # deterministic split must ignore p_copy entirely.
    ds = _ds("val", p_copy=1.0, n_copy=2)
    for i in range(min(4, len(ds))):
        assert ds[i]["meta"]["is_copy"] is False


def test_copy_disabled_by_default_pcopy_zero():
    ds = _ds("train", p_copy=0.0)
    item = ds[0]
    assert item["meta"]["is_copy"] is False
    assert item["meta"]["copy_slot"] == -1
```

Add all four calls to the `__main__` block before the final print.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/dpxuser/dev/patch_icl && .venv311/bin/python src/datasets/omniSynth/test_dataset.py`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'p_copy'` (config field missing).

- [ ] **Step 3a: Add config fields**

In `src/datasets/omniSynth/config.py`, add to `OmniSceneConfig` (after `aug_translate`):

```python
    p_copy: float = 0.9           # train-only per-item prob of injecting copy slot(s)
    n_copy: int = 1               # number of context slots to copy when an item is a
                                  # copy-task (clamped to context_size); each jittered independently
```

- [ ] **Step 3b: Wire the copy block into `__getitem__`**

In `src/datasets/omniSynth/dataset.py`, extend the render import (line 19) to include the new helper:

```python
from .render import make_distractor_sampler, make_target_sampler, render_scene, affine_jitter_scene
```

Replace the tail of `__getitem__` (from `t_img, t_seg, t_k = scene(rngs[0])` through the `return {...}` block) with:

```python
        t_img, t_seg, t_k = scene(rngs[0])
        ctx = [scene(rngs[1 + i]) for i in range(self.context_size)]

        is_copy = False
        copy_slot = -1
        if not self.deterministic and self.context_size > 0 and self.scene.p_copy > 0.0:
            crng = np.random.default_rng()       # isolated: never perturbs subject/item seeds
            if crng.random() < self.scene.p_copy:
                n = max(1, min(int(self.scene.n_copy), self.context_size))
                slots = crng.permutation(self.context_size)[:n].tolist()
                for j in slots:
                    cj_img, cj_seg = affine_jitter_scene(t_img, t_seg, self.scene, crng)
                    ctx[j] = (cj_img, cj_seg, t_k)
                is_copy = True
                copy_slot = min(slots)

        return {
            "image":       _to_img_tensor(t_img),
            "label":       _to_img_tensor(t_seg),
            "context_in":  torch.stack([_to_img_tensor(c[0]) for c in ctx]),
            "context_out": torch.stack([_to_img_tensor(c[1]) for c in ctx]),
            "meta": {
                "class_id": int(class_id),
                "alphabet": self.bank.alphabet(class_id),
                "subject_index": int(sample_index),
                "target_mode": mode,
                "k_target": int(t_k),
                "is_copy": bool(is_copy),
                "copy_slot": int(copy_slot),
            },
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/dpxuser/dev/patch_icl && .venv311/bin/python src/datasets/omniSynth/test_dataset.py`
Expected: PASS — prints `ALL DATASET TESTS PASSED`.

- [ ] **Step 5: Run the render tests too (regression)**

Run: `cd /home/dpxuser/dev/patch_icl && .venv311/bin/python src/datasets/omniSynth/test_render.py`
Expected: PASS — prints `ALL RENDER TESTS PASSED`.

- [ ] **Step 6: Log the change**

Append to `docs/logs.md`:

```
- omniSynth: instCopy copy-tasks — OmniSceneConfig.p_copy (default 0.9) / n_copy (default 1); train-only, isolated rng; meta gains is_copy/copy_slot. Eval byte-identical.
```

---

## Self-Review

**Spec coverage:**
- Mechanism (gated copy of query into n slots, train-only, isolated rng) → Task 2 Step 3b. ✓
- `affine_jitter_scene` helper → Task 1. ✓
- Config `p_copy`, `n_copy` → Task 2 Step 3a. ✓
- `meta` `is_copy`, `copy_slot` → Task 2 Step 3b. ✓
- Determinism: gated on `not self.deterministic`; eval-no-copy test → Task 2 `test_eval_never_copies`. ✓
- `p_copy=0` disables → `test_copy_disabled_by_default_pcopy_zero`. ✓
- Binary/geometric-only jitter → `affine_jitter_scene` thresholds mask; no photometric path. ✓

**Placeholder scan:** none — all steps carry concrete code/commands.

**Type consistency:** `affine_jitter_scene(img, mask, scene, rng) -> (np.ndarray, np.ndarray)` defined in Task 1, consumed identically in Task 2. `copy_slot` is `int` everywhere; `is_copy` is `bool` everywhere. ✓

# omniSynth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-context 2D segmentation dataset from Omniglot — a 64×64 scene of handwritten characters on a 4×4 grid where the task is to segment the cells holding a sampled target character class, given K context (image, mask) pairs that share that class.

**Architecture:** A small package `src/datasets/omniSynth/` mirroring `controlSynth`: dataclass configs, a process-shared `OmniglotBank` that reads PNGs directly from the Omniglot zips and caches cell-sized binary bitmaps, and `OmniSynthICLDataset` that composes grid scenes per `__getitem__`. Wires into the existing 2D pipeline as a new `data.source=omnisynth` branch in `experiments/2d/common.py:build_dataset`, returning the same 4-key dict (`image`, `label`, `context_in`, `context_out`) + `meta` + `.samples` as `SynthICLDataset`.

**Tech Stack:** Python 3.11 (`.venv311/bin/python`), numpy, torch, Pillow (PIL 12.2), scipy.ndimage, Hydra/OmegaConf. Spec: `docs/superpowers/specs/2026-06-29-omnisynth-design.md`.

## Global Constraints

- Interpreter: `.venv311/bin/python` (cuda venv). **No pytest, no uv, no conda.**
- Tests: plain `test_*.py` files next to the code, each function called from an `if __name__ == "__main__":` block ending with a `print("ALL ... TESTS PASSED")`; run with `.venv311/bin/python <path>`. Begin each test file with `import sys; sys.path.insert(0, ".")`.
- **Never stage or commit** — leave all version control to the user. The "Commit" steps below are written for completeness per the plan format, but DO NOT run them; instead, at each commit point, report the files changed and let the user commit.
- Output contract per item (must match `SynthICLDataset`): `{"image": [1,H,W] float32, "label": [1,H,W] float32, "context_in": [K,1,H,W] float32, "context_out": [K,1,H,W] float32, "meta": {...}}`. Expose `.samples` as a list of `(name, sample_idx, label_value)`.
- `image_size` must be divisible by `grid`; cell size = `image_size // grid`. Default `image_size=64`, `grid=4` → cell 16.
- Omniglot data: zips at `/home/dpxuser/repos/omniglot/python/` (`images_background.zip` = 30 train alphabets, `images_evaluation.zip` = 20 eval alphabets). PNGs are 105×105, **black ink (0) on white (255)** — invert so characters are 1, background 0.
- Append a changelog entry to `docs/logs.md`.

---

### Task 1: Config dataclasses + Omniglot bank

**Files:**
- Create: `src/datasets/omniSynth/__init__.py` (minimal for now; full exports in Task 4)
- Create: `src/datasets/omniSynth/config.py`
- Create: `src/datasets/omniSynth/bank.py`
- Test: `src/datasets/omniSynth/test_bank.py`

**Interfaces:**
- Produces:
  - `OmniDiversityConfig(master_seed:int=42, omniglot_root:str=".../omniglot/python", train_zip:str="images_background.zip", eval_zip:str="images_evaluation.zip", val_test_split:float=0.5)`
  - `OmniSceneConfig(grid:int=4, k_min:int=1, k_max:int=6, cell_margin:float=0.1, target_mode:str="class", aug_rotate:float=15.0, aug_scale:float=0.1, aug_translate:float=0.1)`
  - `OmniSamplingConfig(epoch_length:int=10000, eval_subjects_per_task:int=4, eval_seed_namespace:int=0)`
  - `OmniglotBank(diversity, cell_size:int)` with `.cell_size:int`, `.task_ids(split:str)->list[int]`, `.get(class_id:int)->list[np.ndarray]` (each `[cell,cell]` uint8 in {0,1}), `.alphabet(class_id:int)->str`
  - `get_or_build_bank(diversity, cell_size)->OmniglotBank` (process-level cache)

- [ ] **Step 1: Create the package `__init__.py` (minimal)**

`src/datasets/omniSynth/__init__.py`:
```python
"""omniSynth: in-context 2D segmentation from Omniglot characters on a grid.

See docs/superpowers/specs/2026-06-29-omnisynth-design.md. A task = one target
character class; each item is a 4x4 grid of characters where k cells hold the
target (mask = those cells) and the rest are distractors. Plugs into the 2D
pipeline via data.source=omnisynth.
"""
```

- [ ] **Step 2: Write `config.py`**

`src/datasets/omniSynth/config.py`:
```python
"""Config dataclasses for omniSynth (split into diversity / scene / sampling,
mirroring controlSynth's separation of concerns)."""

from dataclasses import dataclass


@dataclass
class OmniDiversityConfig:
    master_seed: int = 42
    omniglot_root: str = "/home/dpxuser/repos/omniglot/python"  # dir holding the zips
    train_zip: str = "images_background.zip"
    eval_zip: str = "images_evaluation.zip"
    val_test_split: float = 0.5   # fraction of eval-alphabet classes -> val (rest -> test)


@dataclass
class OmniSceneConfig:
    grid: int = 4                 # grid x grid cells fill the canvas
    k_min: int = 1                # target cells ~ U[k_min, k_max] (clamped to [1, grid*grid])
    k_max: int = 6
    cell_margin: float = 0.1      # fractional padding inside each cell
    target_mode: str = "class"    # identical | aug | class
    aug_rotate: float = 15.0      # deg; aug-mode per-placement jitter
    aug_scale: float = 0.1        # +/- log2 scale
    aug_translate: float = 0.1    # fraction of cell


@dataclass
class OmniSamplingConfig:
    epoch_length: int = 10000
    eval_subjects_per_task: int = 4
    eval_seed_namespace: int = 0
```

- [ ] **Step 3: Write the failing test**

`src/datasets/omniSynth/test_bank.py`:
```python
import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.config import OmniDiversityConfig
from src.datasets.omniSynth.bank import get_or_build_bank

DIV = OmniDiversityConfig()
CELL = 16


def test_pools_nonempty_and_disjoint():
    bank = get_or_build_bank(DIV, CELL)
    train, val, test = bank.task_ids("train"), bank.task_ids("val"), bank.task_ids("test")
    assert len(train) > 100          # ~964 background classes
    assert len(val) > 0 and len(test) > 0
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


def test_renditions_are_cell_sized_binary_foreground():
    bank = get_or_build_bank(DIV, CELL)
    cid = bank.task_ids("val")[0]
    rends = bank.get(cid)
    assert len(rends) >= 1
    r = rends[0]
    assert r.shape == (CELL, CELL) and r.dtype == np.uint8
    assert set(np.unique(r)).issubset({0, 1})
    assert r.sum() > 0               # inverted: foreground (ink) is 1, not all-zero
    assert r.sum() < r.size          # not all-foreground (background present)


def test_alphabet_lookup():
    bank = get_or_build_bank(DIV, CELL)
    cid = bank.task_ids("train")[0]
    assert isinstance(bank.alphabet(cid), str) and len(bank.alphabet(cid)) > 0


def test_val_test_split_deterministic():
    b1 = get_or_build_bank(OmniDiversityConfig(), CELL)
    # fresh config object, same values -> identical val pool (seeded split)
    b2 = get_or_build_bank(OmniDiversityConfig(), CELL)
    assert b1.task_ids("val") == b2.task_ids("val")


if __name__ == "__main__":
    test_pools_nonempty_and_disjoint()
    test_renditions_are_cell_sized_binary_foreground()
    test_alphabet_lookup()
    test_val_test_split_deterministic()
    print("ALL BANK TESTS PASSED")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv311/bin/python src/datasets/omniSynth/test_bank.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.datasets.omniSynth.bank'`

- [ ] **Step 5: Write `bank.py`**

`src/datasets/omniSynth/bank.py`:
```python
"""OmniglotBank: reads character PNGs directly from the Omniglot zips and caches
cell-sized binary bitmaps. Built once and process-shared so forked DataLoader
workers inherit it (the in-memory analog of controlSynth's GeometryBank cache).

Splits follow the Omniglot convention: background alphabets -> train; evaluation
alphabets -> val/test (partitioned by val_test_split, seeded on master_seed).
class_id is a global int across both zips. Renditions are inverted (ink->1),
resized to an inner box (cell_margin padding) and centered into a cell_size tile.
"""

import io
import os
import re
import zipfile

import numpy as np
from PIL import Image

from .config import OmniDiversityConfig

_BANK_CACHE: dict = {}

# zip entries look like: images_background/Greek/character05/0123_07.png
_ENTRY = re.compile(r"^[^/]+/([^/]+)/(character\d+)/([^/]+\.png)$")


def get_or_build_bank(diversity: OmniDiversityConfig, cell_size: int) -> "OmniglotBank":
    key = (repr(diversity), int(cell_size))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = OmniglotBank(diversity, cell_size)
    return _BANK_CACHE[key]


class OmniglotBank:
    def __init__(self, diversity: OmniDiversityConfig, cell_size: int):
        self.cell_size = int(cell_size)
        self._renditions: dict[int, list[np.ndarray]] = {}
        self._alphabet: dict[int, str] = {}
        self._pools: dict[str, list[int]] = {"train": [], "val": [], "test": []}

        root = diversity.omniglot_root
        train_zip = os.path.join(root, diversity.train_zip)
        eval_zip = os.path.join(root, diversity.eval_zip)
        next_id = 0

        # Train pool: every (alphabet, character) in the background zip.
        next_id = self._ingest(train_zip, target_pools=["train"], start_id=next_id,
                               diversity=diversity)
        # Eval pool: ingest all eval classes, then split into val/test.
        eval_ids_start = next_id
        next_id = self._ingest(eval_zip, target_pools=None, start_id=next_id,
                               diversity=diversity)
        eval_ids = list(range(eval_ids_start, next_id))
        rng = np.random.default_rng(diversity.master_seed)
        perm = rng.permutation(len(eval_ids))
        n_val = int(round(len(eval_ids) * diversity.val_test_split))
        val_set = {eval_ids[i] for i in perm[:n_val]}
        for cid in eval_ids:
            self._pools["val" if cid in val_set else "test"].append(cid)
        self._pools["val"].sort()
        self._pools["test"].sort()

    def _ingest(self, zip_path, target_pools, start_id, diversity):
        """Read a zip, group PNGs by (alphabet, character), assign class_ids.

        target_pools: list of pool names to append class_ids to (e.g. ["train"]),
        or None to leave pool assignment to the caller (eval val/test split)."""
        next_id = start_id
        groups: dict[tuple, list[bytes]] = {}
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                m = _ENTRY.match(name)
                if not m:
                    continue
                alphabet, character, _png = m.groups()
                groups.setdefault((alphabet, character), []).append(zf.read(name))
        for (alphabet, character) in sorted(groups):
            cid = next_id
            next_id += 1
            self._alphabet[cid] = alphabet
            self._renditions[cid] = [self._to_bitmap(b) for b in groups[(alphabet, character)]]
            if target_pools:
                for p in target_pools:
                    self._pools[p].append(cid)
        return next_id

    def _to_bitmap(self, png_bytes: bytes) -> np.ndarray:
        """PNG bytes -> [cell,cell] uint8 in {0,1}, inverted, margin-padded, centered."""
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        arr = np.asarray(img)
        fg = (arr < 128)                       # ink (black) -> foreground
        cell = self.cell_size
        inner = max(1, int(round(cell * (1.0 - 2.0 * 0.1))))  # margin baked at load (0.1)
        resized = np.asarray(
            Image.fromarray((fg * 255).astype(np.uint8)).resize((inner, inner), Image.BILINEAR)
        )
        bm_inner = (resized >= 128).astype(np.uint8)
        out = np.zeros((cell, cell), dtype=np.uint8)
        off = (cell - inner) // 2
        out[off:off + inner, off:off + inner] = bm_inner
        return out

    def task_ids(self, split: str) -> list[int]:
        return list(self._pools[split])

    def get(self, class_id: int) -> list[np.ndarray]:
        return self._renditions[class_id]

    def alphabet(self, class_id: int) -> str:
        return self._alphabet[class_id]
```

Note: `cell_margin` is fixed at 0.1 in `_to_bitmap` for V1 (it must be baked at load time since bitmaps are cached). If a future task needs a configurable margin, thread `diversity`/`scene` margin into the cache key and `_to_bitmap`. The `OmniSceneConfig.cell_margin` field documents the intended knob.

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv311/bin/python src/datasets/omniSynth/test_bank.py`
Expected: `ALL BANK TESTS PASSED` (first run builds both banks; a few seconds).

- [ ] **Step 7: Commit** *(report files; do NOT run — user commits)*

```bash
git add src/datasets/omniSynth/__init__.py src/datasets/omniSynth/config.py src/datasets/omniSynth/bank.py src/datasets/omniSynth/test_bank.py
git commit -m "feat(omniSynth): config dataclasses + Omniglot zip bank"
```

---

### Task 2: Scene rendering (pure) + samplers

**Files:**
- Create: `src/datasets/omniSynth/render.py`
- Test: `src/datasets/omniSynth/test_render.py`

**Interfaces:**
- Consumes: `OmniSceneConfig` (Task 1); `OmniglotBank` (Task 1, used only via the sampler factories).
- Produces:
  - `render_scene(rng, scene, grid, cell_size, target_sampler, distractor_sampler) -> (image float32 [H,W], mask float32 [H,W], k int)` where `H = W = grid*cell_size`. `target_sampler(rng)` and `distractor_sampler(rng)` each return a `[cell_size,cell_size]` uint8 bitmap.
  - `affine_jitter(base: np.ndarray, scene: OmniSceneConfig, rng) -> np.ndarray` (`[cell,cell]` uint8)
  - `make_target_sampler(bank, class_id, scene, base_rng) -> callable(rng)->bitmap`
  - `make_distractor_sampler(bank, pool, target_class) -> callable(rng)->bitmap`

- [ ] **Step 1: Write the failing test**

`src/datasets/omniSynth/test_render.py`:
```python
import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.config import OmniSceneConfig
from src.datasets.omniSynth.render import render_scene, affine_jitter

CELL, GRID = 16, 4


def _const_sampler(value):
    # returns a full-cell bitmap of the given constant (1=target ink, etc.)
    return lambda rng: np.full((CELL, CELL), value, dtype=np.uint8)


def test_shapes_and_k_range():
    scene = OmniSceneConfig(k_min=2, k_max=5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        img, mask, k = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(0))
        assert img.shape == (GRID * CELL, GRID * CELL) and img.dtype == np.float32
        assert mask.shape == img.shape
        assert 2 <= k <= 5


def test_mask_marks_exactly_k_cells():
    # target sampler fills its cell with 1s; mask must equal the painted target cells.
    scene = OmniSceneConfig(k_min=3, k_max=3)
    rng = np.random.default_rng(1)
    img, mask, k = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(0))
    assert k == 3
    # exactly 3 of the 16 cells are fully masked, the rest empty
    cells_masked = 0
    for r in range(GRID):
        for c in range(GRID):
            block = mask[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
            s = block.sum()
            assert s == 0 or s == CELL * CELL          # whole-cell, never partial
            cells_masked += int(s > 0)
    assert cells_masked == 3


def test_mask_is_target_cells_not_distractors():
    # distractor sampler also paints 1s into the image, but must NOT be in the mask.
    scene = OmniSceneConfig(k_min=4, k_max=4)
    rng = np.random.default_rng(2)
    img, mask, k = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(1))
    assert img.sum() == GRID * GRID * CELL * CELL        # every cell painted
    assert mask.sum() == 4 * CELL * CELL                 # only 4 target cells masked


def test_k_clamped_to_valid_range():
    scene = OmniSceneConfig(k_min=99, k_max=99)          # absurd -> clamp to n_cells
    rng = np.random.default_rng(3)
    _, _, k = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(0))
    assert k == GRID * GRID


def test_affine_jitter_preserves_shape_and_binary():
    scene = OmniSceneConfig()
    base = np.zeros((CELL, CELL), dtype=np.uint8)
    base[4:12, 4:12] = 1
    out = affine_jitter(base, scene, np.random.default_rng(4))
    assert out.shape == (CELL, CELL) and out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 1})


if __name__ == "__main__":
    test_shapes_and_k_range()
    test_mask_marks_exactly_k_cells()
    test_mask_is_target_cells_not_distractors()
    test_k_clamped_to_valid_range()
    test_affine_jitter_preserves_shape_and_binary()
    print("ALL RENDER TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python src/datasets/omniSynth/test_render.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.datasets.omniSynth.render'`

- [ ] **Step 3: Write `render.py`**

`src/datasets/omniSynth/render.py`:
```python
"""Pure scene composition + character samplers for omniSynth.

render_scene is bank-free (takes sampler callables) so it is unit-testable with
trivial samplers. The sampler factories encode target_mode:
  identical -> one fixed rendition reused everywhere (chosen via base_rng so it is
               shared across the query + all contexts of an item)
  aug       -> one fixed base rendition + independent affine jitter per placement
  class     -> a fresh random rendition of the target class per placement
"""

import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, zoom as nd_zoom


def render_scene(rng, scene, grid, cell_size, target_sampler, distractor_sampler):
    """Compose a grid scene. Returns (image float32 [H,W], mask float32 [H,W], k)."""
    n_cells = grid * grid
    k = int(rng.integers(scene.k_min, scene.k_max + 1))
    k = max(1, min(k, n_cells))                       # clamp to [1, n_cells]
    cells = rng.permutation(n_cells)
    target_cells = set(cells[:k].tolist())

    H = W = grid * cell_size
    image = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    for cell in range(n_cells):
        r, c = divmod(cell, grid)
        y0, x0 = r * cell_size, c * cell_size
        if cell in target_cells:
            bm = target_sampler(rng)
            image[y0:y0 + cell_size, x0:x0 + cell_size] = bm
            mask[y0:y0 + cell_size, x0:x0 + cell_size] = bm
        else:
            bm = distractor_sampler(rng)
            image[y0:y0 + cell_size, x0:x0 + cell_size] = bm
    return image, mask, k


def _zoom_to_size(img, scale, size):
    """Zoom by `scale` then center-crop/pad back to (size, size)."""
    z = nd_zoom(img, scale, order=1)
    out = np.zeros((size, size), dtype=img.dtype)
    # center-crop source / center-place into dest
    sy = max(0, (z.shape[0] - size) // 2)
    sx = max(0, (z.shape[1] - size) // 2)
    cropped = z[sy:sy + size, sx:sx + size]
    dy = (size - cropped.shape[0]) // 2
    dx = (size - cropped.shape[1]) // 2
    out[dy:dy + cropped.shape[0], dx:dx + cropped.shape[1]] = cropped
    return out


def affine_jitter(base, scene, rng):
    """Per-placement rotate/scale/translate jitter of a base bitmap -> uint8 {0,1}."""
    cell = base.shape[0]
    img = base.astype(np.float32)
    angle = rng.uniform(-scene.aug_rotate, scene.aug_rotate)
    img = nd_rotate(img, angle, reshape=False, order=1, mode="constant", cval=0.0)
    scale = 2.0 ** rng.uniform(-scene.aug_scale, scene.aug_scale)
    img = _zoom_to_size(img, scale, cell)
    dy = rng.uniform(-scene.aug_translate, scene.aug_translate) * cell
    dx = rng.uniform(-scene.aug_translate, scene.aug_translate) * cell
    img = nd_shift(img, (dy, dx), order=1, mode="constant", cval=0.0)
    return (img > 0.5).astype(np.uint8)


def make_target_sampler(bank, class_id, scene, base_rng):
    rends = bank.get(class_id)
    mode = scene.target_mode
    if mode == "class":
        return lambda rng: rends[rng.integers(len(rends))].copy()
    base = rends[base_rng.integers(len(rends))]       # fixed per item (shared across subjects)
    if mode == "identical":
        return lambda rng: base.copy()
    if mode == "aug":
        return lambda rng: affine_jitter(base, scene, rng)
    raise ValueError(f"unknown target_mode {mode!r} (identical | aug | class)")


def make_distractor_sampler(bank, pool, target_class):
    others = [c for c in pool if c != target_class]
    if not others:
        raise ValueError("distractor pool empty after excluding target class")

    def sample(rng):
        cid = others[rng.integers(len(others))]
        rends = bank.get(cid)
        return rends[rng.integers(len(rends))].copy()

    return sample
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python src/datasets/omniSynth/test_render.py`
Expected: `ALL RENDER TESTS PASSED`

- [ ] **Step 5: Commit** *(report files; do NOT run)*

```bash
git add src/datasets/omniSynth/render.py src/datasets/omniSynth/test_render.py
git commit -m "feat(omniSynth): pure grid scene rendering + character samplers"
```

---

### Task 3: OmniSynthICLDataset

**Files:**
- Create: `src/datasets/omniSynth/dataset.py`
- Modify: `src/datasets/omniSynth/__init__.py` (add exports)
- Test: `src/datasets/omniSynth/test_dataset.py`

**Interfaces:**
- Consumes: `OmniDiversityConfig`, `OmniSceneConfig`, `OmniSamplingConfig` (Task 1); `get_or_build_bank` (Task 1); `render_scene`, `make_target_sampler`, `make_distractor_sampler` (Task 2).
- Produces:
  - `OmniSynthICLDataset(split="train", context_size=3, image_size=64, diversity=None, scene=None, sampling=None, deterministic=None)` — a `torch.utils.data.Dataset` with `.samples` and the 4-key+meta dict contract.

- [ ] **Step 1: Write the failing test**

`src/datasets/omniSynth/test_dataset.py`:
```python
import sys; sys.path.insert(0, ".")
import torch
from src.datasets.omniSynth.config import OmniSceneConfig, OmniSamplingConfig
from src.datasets.omniSynth.dataset import OmniSynthICLDataset

K, IMG = 3, 64


def _ds(split, **scene_kw):
    return OmniSynthICLDataset(
        split=split, context_size=K, image_size=IMG,
        scene=OmniSceneConfig(**scene_kw),
        sampling=OmniSamplingConfig(eval_subjects_per_task=2),
    )


def test_item_shapes_and_keys():
    ds = _ds("train")
    item = ds[0]
    assert set(item) == {"image", "label", "context_in", "context_out", "meta"}
    assert item["image"].shape == (1, IMG, IMG)
    assert item["label"].shape == (1, IMG, IMG)
    assert item["context_in"].shape == (K, 1, IMG, IMG)
    assert item["context_out"].shape == (K, 1, IMG, IMG)


def test_label_matches_target_cells_only():
    ds = _ds("val", k_min=2, k_max=2)
    item = ds[0]
    lbl = item["label"]
    assert lbl.max() <= 1.0 and lbl.min() >= 0.0
    # 2 target cells of 16x16 each in a 64x64 image (cell=16): mask area == 2*256
    assert float(lbl.sum()) == 2 * 16 * 16


def test_image_size_divisible_guard():
    try:
        OmniSynthICLDataset(split="train", context_size=K, image_size=63,
                            scene=OmniSceneConfig(grid=4))
        assert False, "expected ValueError for non-divisible image_size"
    except ValueError:
        pass


def test_val_deterministic_train_not():
    v1, v2 = _ds("val"), _ds("val")
    a, b = v1[0], v2[0]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["label"], b["label"])
    assert torch.equal(a["context_in"], b["context_in"])
    # train: two reads almost surely differ
    t = _ds("train")
    assert not torch.equal(t[0]["image"], t[1]["image"])


def test_samples_contract():
    tr = _ds("train")
    assert tr.samples[0][0] == "omniglot/train"
    va = _ds("val")
    assert len(va.samples) == len(va)
    name, idx, lab = va.samples[0]
    assert name.startswith("omniglot/") and lab == 1


def test_identical_mode_repeats_target_cells():
    # identical: every target cell is the same bitmap -> all target cells byte-identical
    ds = _ds("val", target_mode="identical", k_min=3, k_max=3)
    item = ds[0]
    img = item["image"][0]
    cells = []
    for cell in range(16):
        r, c = divmod(cell, 4)
        block = img[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        if item["label"][0, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].sum() > 0:
            cells.append(block)
    assert len(cells) == 3
    for blk in cells[1:]:
        assert torch.equal(blk, cells[0])


if __name__ == "__main__":
    test_item_shapes_and_keys()
    test_label_matches_target_cells_only()
    test_image_size_divisible_guard()
    test_val_deterministic_train_not()
    test_samples_contract()
    test_identical_mode_repeats_target_cells()
    print("ALL DATASET TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python src/datasets/omniSynth/test_dataset.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.datasets.omniSynth.dataset'`

- [ ] **Step 3: Write `dataset.py`**

`src/datasets/omniSynth/dataset.py`:
```python
"""OmniSynthICLDataset: composes grid scenes of Omniglot characters into the
in-context contract (image, label, context_in, context_out + meta), matching
SynthICLDataset so the existing TaggedDataset/collate wrappers work unchanged.

Determinism (mirrors controlSynth): train draws fresh entropy per subject; val/
test derive every subject seed from (eval_seed_namespace, task_id, sample_index)
-> byte-identical eval set. A separate item-level rng (distinct spawn key) fixes
the shared target base bitmap for identical/aug modes across query + contexts.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .bank import get_or_build_bank
from .config import OmniDiversityConfig, OmniSamplingConfig, OmniSceneConfig
from .render import make_distractor_sampler, make_target_sampler, render_scene


def _to_img_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


class OmniSynthICLDataset(Dataset):
    def __init__(self, split="train", context_size=3, image_size=64,
                 diversity=None, scene=None, sampling=None, deterministic=None):
        self.split = split
        self.context_size = context_size
        self.image_size = image_size
        self.diversity = diversity or OmniDiversityConfig()
        self.scene = scene or OmniSceneConfig()
        self.sampling = sampling or OmniSamplingConfig()
        self.deterministic = (split != "train") if deterministic is None else deterministic

        grid = self.scene.grid
        if image_size % grid != 0:
            raise ValueError(f"image_size {image_size} not divisible by grid {grid}")
        self.cell_size = image_size // grid

        self.bank = get_or_build_bank(self.diversity, self.cell_size)
        self.pool = self.bank.task_ids(split)
        if not self.pool:
            raise ValueError(f"empty class pool for split {split!r}")

        if self.deterministic:
            self._eval_index = []                       # idx -> (class_id, subject_index)
            self.samples = []
            for class_id in self.pool:
                alph = self.bank.alphabet(class_id)
                for s in range(self.sampling.eval_subjects_per_task):
                    self.samples.append((f"omniglot/{alph}", len(self._eval_index), 1))
                    self._eval_index.append((class_id, s))
        else:
            self._eval_index = None
            self.samples = [("omniglot/train", i, 1)
                            for i in range(self.sampling.epoch_length)]

    def __len__(self):
        return len(self.samples)

    def _subject_rngs(self, task_id, sample_index):
        n = self.context_size + 1
        if self.deterministic:
            ss = np.random.SeedSequence([int(self.sampling.eval_seed_namespace),
                                         int(task_id), int(sample_index)])
            return [np.random.default_rng(c) for c in ss.spawn(n)]
        return [np.random.default_rng() for _ in range(n)]

    def _item_rng(self, task_id, sample_index):
        """Item-level rng for the shared target base (identical/aug). Distinct
        namespace offset so it never collides with the subject seeds above."""
        if self.deterministic:
            ss = np.random.SeedSequence([int(self.sampling.eval_seed_namespace) + 1,
                                         int(task_id), int(sample_index)])
            return np.random.default_rng(ss)
        return np.random.default_rng()

    def __getitem__(self, idx):
        if self.deterministic:
            class_id, sample_index = self._eval_index[idx]
        else:
            class_id = int(self.pool[np.random.default_rng().integers(len(self.pool))])
            sample_index = idx

        rngs = self._subject_rngs(class_id, sample_index)
        base_rng = self._item_rng(class_id, sample_index)

        target_sampler = make_target_sampler(self.bank, class_id, self.scene, base_rng)
        distractor_sampler = make_distractor_sampler(self.bank, self.pool, class_id)

        def scene(rng):
            return render_scene(rng, self.scene, self.scene.grid, self.cell_size,
                                target_sampler, distractor_sampler)

        t_img, t_seg, t_k = scene(rngs[0])
        ctx = [scene(rngs[1 + i]) for i in range(self.context_size)]

        return {
            "image":       _to_img_tensor(t_img),
            "label":       _to_img_tensor(t_seg),
            "context_in":  torch.stack([_to_img_tensor(c[0]) for c in ctx]),
            "context_out": torch.stack([_to_img_tensor(c[1]) for c in ctx]),
            "meta": {
                "class_id": int(class_id),
                "alphabet": self.bank.alphabet(class_id),
                "subject_index": int(sample_index),
                "target_mode": self.scene.target_mode,
                "k_target": int(t_k),
            },
        }
```

- [ ] **Step 4: Add exports to `__init__.py`**

Append to `src/datasets/omniSynth/__init__.py`:
```python
from .config import OmniDiversityConfig, OmniSceneConfig, OmniSamplingConfig
from .dataset import OmniSynthICLDataset

__all__ = [
    "OmniDiversityConfig",
    "OmniSceneConfig",
    "OmniSamplingConfig",
    "OmniSynthICLDataset",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv311/bin/python src/datasets/omniSynth/test_dataset.py`
Expected: `ALL DATASET TESTS PASSED`

- [ ] **Step 6: Commit** *(report files; do NOT run)*

```bash
git add src/datasets/omniSynth/dataset.py src/datasets/omniSynth/__init__.py src/datasets/omniSynth/test_dataset.py
git commit -m "feat(omniSynth): OmniSynthICLDataset with deterministic eval + .samples"
```

---

### Task 4: Pipeline integration (build_dataset + Hydra config + logs)

**Files:**
- Modify: `experiments/2d/common.py` (add `omnisynth` branch in `build_dataset`, ~line 97-118)
- Create: `configs/experiment/2d/synth/omniglot.yaml`
- Modify: `configs/config.yaml` (add `paths.omniglot` default) and `configs/cluster/nfs.yaml` + `configs/cluster/meta.yaml` if they define `paths` (add `omniglot` there too)
- Modify: `docs/logs.md` (changelog entry)
- Test: `src/datasets/omniSynth/test_integration.py`

**Interfaces:**
- Consumes: `OmniDiversityConfig`, `OmniSceneConfig`, `OmniSamplingConfig`, `OmniSynthICLDataset` (Task 3).
- Produces: `build_dataset(cfg, split)` returns an `OmniSynthICLDataset` when `cfg.data.source == "omnisynth"`.

- [ ] **Step 1: Write the failing integration test**

`src/datasets/omniSynth/test_integration.py`:
```python
import sys; sys.path.insert(0, ".")
import torch
from omegaconf import OmegaConf
from experiments.common_import import build_dataset  # see Step 2 note

CFG = OmegaConf.create({
    "data": {"source": "omnisynth", "context_size": 3, "image_size": 64},
    "paths": {"omniglot": "/home/dpxuser/repos/omniglot/python"},
    "synth": {
        "diversity": {"master_seed": 42, "train_zip": "images_background.zip",
                      "eval_zip": "images_evaluation.zip", "val_test_split": 0.5},
        "scene": {"grid": 4, "k_min": 1, "k_max": 6, "cell_margin": 0.1,
                  "target_mode": "class", "aug_rotate": 15.0, "aug_scale": 0.1,
                  "aug_translate": 0.1},
        "sampling": {"epoch_length": 100, "eval_subjects_per_task": 2,
                     "eval_seed_namespace": 0},
    },
})


def test_build_dataset_omnisynth():
    ds = build_dataset(CFG, "train")
    item = ds[0]
    assert item["image"].shape == (1, 64, 64)
    assert item["context_in"].shape == (3, 1, 64, 64)


if __name__ == "__main__":
    test_build_dataset_omnisynth()
    print("ALL INTEGRATION TESTS PASSED")
```

Note for Step 2: import `build_dataset` from its real location. Replace the placeholder import line with `from experiments.__dummy__ import ...` resolution at implementation time — the actual module is `experiments/2d/common.py`. Since `experiments/2d` is not a package path that imports cleanly by dotted name (`2d` is not a valid identifier), the test must add it to `sys.path` and import by file: replace the import with:
```python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "omnisynth_common", str(pathlib.Path("experiments/2d/common.py")))
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
build_dataset = common.build_dataset
```
Use this importlib form in the test instead of the `from experiments.common_import import build_dataset` placeholder line.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python src/datasets/omniSynth/test_integration.py`
Expected: FAIL — `ValueError: unknown data.source 'omnisynth' ...`

- [ ] **Step 3: Add the `omnisynth` branch to `build_dataset`**

In `experiments/2d/common.py`, immediately after the `if source == "synthetic":` block (before the final `raise ValueError`), insert:
```python
    if source == "omnisynth":
        from src.datasets.omniSynth import (
            OmniDiversityConfig, OmniSamplingConfig, OmniSceneConfig, OmniSynthICLDataset,
        )
        s = cfg.synth
        return OmniSynthICLDataset(
            split=split,
            context_size=cfg.data.context_size,
            image_size=cfg.data.image_size,
            diversity=OmniDiversityConfig(omniglot_root=cfg.paths.omniglot,
                                          **dict(s.diversity)),
            scene=OmniSceneConfig(**dict(s.scene)),
            sampling=OmniSamplingConfig(**dict(s.sampling)),
        )
```
And update the final error string to include `omnisynth`:
```python
    raise ValueError(
        f"unknown data.source {source!r} "
        "(medsegbench | biomedparse | totalseg2d | synthetic | omnisynth)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python src/datasets/omniSynth/test_integration.py`
Expected: `ALL INTEGRATION TESTS PASSED`

- [ ] **Step 5: Create the Hydra config**

`configs/experiment/2d/synth/omniglot.yaml`:
```yaml
# omniSynth generator (only used when data.source=omnisynth). In-context Omniglot:
# a 4x4 grid of characters; k cells hold the target class (mask = those cells),
# the rest are distractors. See docs/superpowers/specs/2026-06-29-omnisynth-design.md.
# Reuses the `synth` package slot, so set data.source=omnisynth to select it.
# @package synth
diversity:
  master_seed: 42
  train_zip: images_background.zip   # 30 background alphabets -> train
  eval_zip: images_evaluation.zip    # 20 evaluation alphabets -> val/test
  val_test_split: 0.5                # fraction of eval classes -> val (rest -> test)
scene:
  grid: 4                # grid x grid cells fill the canvas (image_size must divide by grid)
  k_min: 1               # target cells ~ U[k_min, k_max]
  k_max: 6
  cell_margin: 0.1       # fractional padding inside each cell (baked at load in V1)
  target_mode: class     # identical | aug | class  (what "same target" means)
  aug_rotate: 15.0       # deg; aug-mode per-placement jitter
  aug_scale: 0.1         # +/- log2 scale
  aug_translate: 0.1     # fraction of cell
sampling:
  epoch_length: 10000
  eval_subjects_per_task: 4
  eval_seed_namespace: 0
```

- [ ] **Step 6: Add `paths.omniglot` default**

In `configs/config.yaml` under the `paths:` block, add:
```yaml
  omniglot: /home/dpxuser/repos/omniglot/python   # dir holding images_background.zip / images_evaluation.zip
```
If `configs/cluster/nfs.yaml` and/or `configs/cluster/meta.yaml` define a `paths:` block that overrides per-cluster, add an `omniglot:` entry there too (point at the cluster's Omniglot copy, or repeat the dev path if none).

- [ ] **Step 7: Append changelog entry to `docs/logs.md`**

Add a dated entry near the top describing: new `omniSynth` dataset (`src/datasets/omniSynth/`), in-context Omniglot grid segmentation, wired as `data.source=omnisynth` with config `configs/experiment/2d/synth/omniglot.yaml`; `target_mode` (identical|aug|class) and `k_min/k_max` knobs; train=background alphabets, val/test=evaluation alphabets.

- [ ] **Step 8: Commit** *(report files; do NOT run)*

```bash
git add experiments/2d/common.py configs/experiment/2d/synth/omniglot.yaml configs/config.yaml docs/logs.md src/datasets/omniSynth/test_integration.py
git commit -m "feat(omniSynth): wire data.source=omnisynth + hydra config + logs"
```

---

### Task 5 (optional): Preview script for eyeballing

**Files:**
- Create: `experiments/2d/synth/preview_omnisynth.py`

**Interfaces:**
- Consumes: `OmniSynthICLDataset` (Task 3).
- Produces: a PNG grid of query + context images/masks for visual sanity-checking. No test (visual artifact).

- [ ] **Step 1: Write the preview script**

`experiments/2d/synth/preview_omnisynth.py`:
```python
"""Render a few omniSynth items (query image+mask + contexts) to a PNG for eyeballing.

Run: .venv311/bin/python experiments/2d/synth/preview_omnisynth.py --mode class --n 4
"""
import argparse
import sys; sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets.omniSynth import OmniSceneConfig, OmniSynthICLDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="class", choices=["identical", "aug", "class"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="results/omnisynth_preview.png")
    args = ap.parse_args()

    ds = OmniSynthICLDataset(split=args.split, context_size=3, image_size=64,
                             scene=OmniSceneConfig(target_mode=args.mode))
    cols = 2 + 3 * 2     # query img, query mask, then 3 contexts (img+mask)
    fig, axes = plt.subplots(args.n, cols, figsize=(cols * 1.4, args.n * 1.4))
    axes = axes.reshape(args.n, cols)
    for i in range(args.n):
        item = ds[i]
        panels = [("q-img", item["image"][0]), ("q-mask", item["label"][0])]
        for c in range(3):
            panels.append((f"c{c}-img", item["context_in"][c, 0]))
            panels.append((f"c{c}-msk", item["context_out"][c, 0]))
        for j, (title, im) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
    fig.suptitle(f"omniSynth {args.split} / target_mode={args.mode}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and eyeball the output**

Run: `.venv311/bin/python experiments/2d/synth/preview_omnisynth.py --mode class --n 4`
Expected: prints `wrote results/omnisynth_preview.png`; open the PNG — query mask should cover exactly the cells whose character matches the target across query + contexts; distractors unmasked.

- [ ] **Step 3: Commit** *(report files; do NOT run)*

```bash
git add experiments/2d/synth/preview_omnisynth.py
git commit -m "feat(omniSynth): preview script for visual sanity-checking"
```

---

## Self-Review

**Spec coverage:**
- Scene/task (4×4, all cells filled, k targets, mask=target cells) → Task 2 `render_scene` + Task 3 mask. ✓
- Distractors = any other class in split → Task 2 `make_distractor_sampler`. ✓
- `target_mode` identical|aug|class → Task 2 sampler factories + `affine_jitter`; Task 3 shared base via `_item_rng`. ✓
- `k ~ U[k_min,k_max]` per image → Task 2 `render_scene`. ✓
- Bank reads zips directly, inverted, cell-sized, cached, process-shared → Task 1. ✓
- Split: train=background, val/test=evaluation (seeded split), distractors from same split → Task 1 pools + Task 3 `self.pool`. ✓
- Determinism (subject seeds from namespace/task/sample) → Task 3 `_subject_rngs` / `_item_rng`. ✓
- Output contract + `.samples` (`omniglot/<alphabet>`) → Task 3. ✓
- Config dataclasses + Hydra yaml + `paths.omniglot` → Task 1 + Task 4. ✓
- `build_dataset` branch → Task 4. ✓
- Tests (shapes, mask, k-range, determinism, disjoint pools) → Tasks 1–4 test files. ✓
- `docs/logs.md` entry → Task 4 Step 7. ✓
- Optional preview → Task 5. ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to" left; the one indirection (integration test import) is spelled out with concrete importlib code in Task 4 Step 1.

**Type consistency:** `render_scene` returns `(image, mask, k)` — consumed as `c[0]`/`c[1]`/`t_k` in Task 3. ✓ Sampler signatures `callable(rng)->bitmap` match factories and the const-sampler test stubs. ✓ `bank.get`/`task_ids`/`alphabet` names consistent across Tasks 1–3. ✓ Config field names match between `config.py`, the yaml, and `build_dataset` `**dict(...)` expansion. ✓

# anchor_synth3d Barycentric Positioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place the synthetic object at affine-invariant **barycentric coordinates over 4 landmark organs**, and scale its size by the **anchor-frame length**, so target and context objects share a consistent anatomical position and apparent size.

**Architecture:** `anchor_synth3d` becomes subject-first: pick a target subject, choose 4 co-occurring anchor organs, draw shared barycentric weights + a size fraction, and render the object at `Σ wᵢ·centroidᵢ` with side `size_frac · L` (`L` = mean pairwise centroid distance). Anchors are landmarks only; the label is the drawn object. Validation groups by object shape.

**Tech Stack:** Python, NumPy, PyTorch, Hydra, pytest. Env: `.venv_thor` on host `thor` (`.venv_thor/bin/python`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-22-anchor-synth3d-barycentric-positioning-design.md`.
- `n_anchors = 4`; barycentric weights **affine** (sum to 1, may go mildly negative), `extrapolation = 0.3`.
- Frame length `L = mean pairwise distance` of the 4 anchor centroids (orientation-invariant).
- Object side `= max(object_size_min_vox, round(size_frac · L · jit))`; `size_frac ~ U[frac_min, frac_max]` shared across the K+1 scenes.
- All K+1 subjects share the **same 4 anchor classes**; contexts drawn from their co-occurrence set (target excluded). Self-context only as a last-resort fallback.
- `label_name = object shape`; val classes = shapes emitted (`mix → ["blob","elongated","tubular"]`, else `[shape]`).
- Run tests with `.venv_thor/bin/python -m pytest`. Full-dataset probes use `TotalSegmentator` at `cfg.paths.totalseg`.
- Remove the single-anchor path entirely: `offset_range`, `offset_to_center`, `object_size_min`, `object_size_max_frac`.

---

## File Structure

- `src/datasets/anchor_synth/draw.py` — add pure-geometry helpers `affine_weights`, `frame_length`, `barycentric_center`; remove `offset_to_center`. (`anchor_stats`, `place_object`, `_slices_3d` unchanged.)
- `src/datasets/anchor_synth/dataset3d.py` — subject-first selection, co-occurrence structures, barycentric render, frame-length sizing, shape `label_name`, extended meta.
- `experiments/3d/common.py` — `anchor_shapes(cfg)` helper; forward new knobs in `build_dataset`.
- `experiments/3d/train.py` — anchor_synth3d val branch → `anchor_shapes`.
- `configs/experiment/3d/dataset/anchor_synth3d.yaml` — new knobs.
- `experiments/3d/plot_dataset_items.py`, `experiments/3d/analyze_object_blend.py` — captions.
- `src/datasets/anchor_synth/test_dataset3d.py`, `test_wiring.py` — updated fake roots + constructor/cfg keys.

---

### Task 0: Baseline commit of current working changes

The working tree already contains committed-spec plus **uncommitted** anchor_synth3d iteration (aug wiring, independent absolute sizing, per-shape `meta`/spec `shape`). Commit it so later per-task commits are clean.

**Files:** all currently-modified tracked files.

- [ ] **Step 1: Inspect the working tree**

Run: `git status --short && git --no-pager diff --stat`
Expected: modifications to `experiments/3d/common.py`, `src/datasets/anchor_synth/dataset3d.py`, `experiments/3d/plot_dataset_items.py`, `experiments/3d/analyze_object_blend.py`, `src/datasets/anchor_synth/shapes.py`, `configs/experiment/3d/dataset/anchor_synth3d.yaml`, `src/datasets/anchor_synth/test_*.py`, `docs/logs.md`.

- [ ] **Step 2: Run the existing anchor_synth tests (green baseline)**

Run: `.venv_thor/bin/python -m pytest src/datasets/anchor_synth/ -q`
Expected: PASS (18 passed).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(anchor-synth3d): apply augs, decouple object size, record per-shape meta"
```

---

### Task 1: Geometry + weight helpers in `draw.py`

**Files:**
- Modify: `src/datasets/anchor_synth/draw.py`
- Test: `src/datasets/anchor_synth/test_draw.py` (create)

**Interfaces:**
- Produces:
  - `affine_weights(rng: np.random.Generator, n: int, extrapolation: float = 0.0, concentration: float = 1.0) -> np.ndarray` — shape `(n,)`, sums to 1.
  - `frame_length(centroids: np.ndarray) -> float` — mean pairwise distance of `(n,3)` centroids.
  - `barycentric_center(centroids: np.ndarray, weights: np.ndarray, tile_size: int, vol_shape) -> np.ndarray` — `(3,)` voxel centre, clamped in-bounds.
- `offset_to_center` is removed (its only caller, `dataset3d.py`, is refactored in Task 2).

- [ ] **Step 1: Write the failing tests**

Create `src/datasets/anchor_synth/test_draw.py`:
```python
import numpy as np
from src.datasets.anchor_synth.draw import (
    affine_weights, frame_length, barycentric_center,
)

TETRA = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])


def test_affine_weights_sum_to_one_and_convex_when_no_extrapolation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = affine_weights(rng, 4, extrapolation=0.0, concentration=1.0)
        assert w.shape == (4,)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= -1e-12).all()          # convex: inside the hull


def test_affine_weights_extrapolation_allows_negative():
    rng = np.random.default_rng(1)
    saw_negative = any(
        affine_weights(rng, 4, extrapolation=1.0).min() < 0 for _ in range(200)
    )
    assert saw_negative                     # mild extrapolation can leave the hull


def test_frame_length_is_rotation_invariant():
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))     # random rotation
    L0 = frame_length(TETRA)
    L1 = frame_length((q @ TETRA.T).T + np.array([5., -3., 2.]))  # rotate + translate
    assert abs(L0 - L1) < 1e-6
    assert L0 > 0


def test_barycentric_center_barycenter_and_onehot():
    vol = (100, 100, 100)
    bc = barycentric_center(TETRA, np.full(4, 0.25), tile_size=4, vol_shape=vol)
    assert np.allclose(bc, TETRA.mean(0))
    oh = barycentric_center(TETRA, np.array([0., 1., 0., 0.]), tile_size=4, vol_shape=vol)
    assert np.allclose(oh, TETRA[1])


def test_barycentric_center_clamped_in_bounds():
    vol = (32, 32, 32)
    far = np.array([[0., 0., 0.]] * 4)
    bc = barycentric_center(far, np.array([2., -1., 0., 0.]), tile_size=8, vol_shape=vol)
    assert (bc >= 4).all() and (bc <= 28).all()          # tile_size/2 .. vol - tile_size/2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv_thor/bin/python -m pytest src/datasets/anchor_synth/test_draw.py -q`
Expected: FAIL (ImportError: cannot import name `affine_weights`).

- [ ] **Step 3: Implement the helpers**

In `src/datasets/anchor_synth/draw.py`, **delete** `offset_to_center` (lines defining it) and add:
```python
def affine_weights(rng, n, extrapolation=0.0, concentration=1.0):
    """`n` barycentric weights summing to 1. Base convex `u ~ Dirichlet`, expanded
    around the barycenter 1/n by (1+extrapolation) so weights may go mildly negative
    (extrapolation=0 -> strictly inside the hull)."""
    u = rng.dirichlet([float(concentration)] * int(n))
    b = 1.0 / int(n)
    return b + (1.0 + float(extrapolation)) * (u - b)


def frame_length(centroids):
    """Mean pairwise Euclidean distance of centroids (n,3) — an orientation- and
    translation-invariant characteristic length of the landmark frame."""
    c = np.asarray(centroids, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    diffs = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(n, k=1)
    return float(d[iu].mean())


def barycentric_center(centroids, weights, tile_size, vol_shape):
    """Voxel centre = Σ wᵢ·centroidᵢ, clamped so a `tile_size` cube stays fully
    inside `vol_shape`."""
    c = np.asarray(centroids, dtype=np.float64)          # (n, 3)
    w = np.asarray(weights, dtype=np.float64)            # (n,)
    center = (w[:, None] * c).sum(0)                     # (3,)
    half = tile_size / 2.0
    return np.clip(center, half, np.asarray(vol_shape, dtype=np.float64) - half)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv_thor/bin/python -m pytest src/datasets/anchor_synth/test_draw.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/datasets/anchor_synth/draw.py src/datasets/anchor_synth/test_draw.py
git commit -m "feat(anchor-synth3d): barycentric/frame-length geometry helpers"
```

---

### Task 2: Refactor `AnchorSynth3DICLDataset` to barycentric multi-anchor

**Files:**
- Modify: `src/datasets/anchor_synth/dataset3d.py`
- Test: `src/datasets/anchor_synth/test_dataset3d.py`

**Interfaces:**
- Consumes: `affine_weights`, `frame_length`, `barycentric_center` (Task 1); `anchor_stats`, `place_object` (`draw.py`); `render_object`, `small_rotation`, `roughen`, `sample_object_spec` (`shapes.py`); `_ALL_CLASSES_IDX`.
- Produces: `AnchorSynth3DICLDataset(..., n_anchors=4, extrapolation=0.3, weight_concentration=1.0, max_select_tries=20, object_size_frac_min=0.3, object_size_frac_max=0.8, object_size_min_vox=6, ...)`. Item dict adds `meta["anchors"]` (list of `n_anchors` class names) and `meta["weights"]`; `label_name` is the object shape.

- [ ] **Step 1: Update the test fake root + constructor to the new contract**

Rewrite `src/datasets/anchor_synth/test_dataset3d.py`:
```python
import sys; sys.path.insert(0, ".")
import numpy as np
import torch

from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX

SIZE = 16
ANCHORS = ["aorta", "liver", "spleen", "kidney_left"]   # 4 co-occurring landmarks
# non-coplanar blocks (tetrahedron corners) so the frame is well-conditioned
BLOCKS = [(2, 2, 2), (2, 10, 10), (10, 2, 10), (10, 10, 2)]


def _make_root(tmp_path, n=5):
    rows = ["image_id;split"]
    for i in range(n):
        subj = f"s{i:04d}"
        d = tmp_path / subj
        d.mkdir()
        label = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        for cls, (z, y, x) in zip(ANCHORS, BLOCKS):
            label[z:z + 3, y:y + 3, x:x + 3] = _ALL_CLASSES_IDX[cls]
        ct = (0.3 + 0.01 * i) * np.ones((SIZE, SIZE, SIZE), dtype=np.float16)
        np.save(d / "label.npy", label)
        np.save(d / f"label_{SIZE}x{SIZE}x{SIZE}.npy", label)
        np.save(d / f"ct_{SIZE}x{SIZE}x{SIZE}.npy", ct)
        rows.append(f"{subj};val")
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def _ds(root, **kw):
    return AnchorSynth3DICLDataset(
        root=root, classes=list(ANCHORS), image_size=(SIZE, SIZE, SIZE),
        split="val", context_size=2, eval_subjects_per_task=2,
        n_anchors=4, object_size_frac_min=0.6, object_size_frac_max=1.2,
        object_size_min_vox=3, contrast_delta=0.3, **kw)


def test_contract_shapes_and_object_drawn(tmp_path):
    ds = _ds(_make_root(tmp_path))
    assert len(ds) == 2 * 5                       # eligible subjects (5) * subjects/task (2)
    item = ds[0]
    assert item["image"].shape == (1, SIZE, SIZE, SIZE)
    assert item["label"].shape == (SIZE, SIZE, SIZE)
    assert item["label"].dtype == torch.int64
    assert item["context_in"].shape == (2, 1, SIZE, SIZE, SIZE)
    assert item["context_out"].shape == (2, SIZE, SIZE, SIZE)
    assert item["label"].sum() > 0                # a blob was drawn
    assert item["label_name"] in ("blob", "elongated", "tubular")
    assert len(item["meta"]["anchors"]) == 4
    assert len(item["meta"]["weights"][0]) == 4


def test_anchor_not_emitted_as_label(tmp_path):
    ds = _ds(_make_root(tmp_path))
    item = ds[0]
    full = np.load(tmp_path / f"{item['subject']}/label_{SIZE}x{SIZE}x{SIZE}.npy")
    anchor_union = np.isin(full, [_ALL_CLASSES_IDX[c] for c in ANCHORS])
    assert not np.array_equal(item["label"].numpy() > 0, anchor_union)
    assert int(item["label"].max()) <= 1          # n_objects=1: bg(0) or object(1)


def test_deterministic_across_instances(tmp_path):
    root = _make_root(tmp_path)
    a = _ds(root)[0]
    b = _ds(root)[0]
    assert torch.equal(a["label"], b["label"])
    assert torch.equal(a["image"], b["image"])


def test_organ_source_not_implemented(tmp_path):
    import pytest
    with pytest.raises(NotImplementedError):
        _ds(_make_root(tmp_path), object_source="organ")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv_thor/bin/python -m pytest src/datasets/anchor_synth/test_dataset3d.py -q`
Expected: FAIL (TypeError: unexpected keyword argument `n_anchors`, or KeyError `meta['anchors']`).

- [ ] **Step 3: Rewrite the dataset imports + `__init__`**

In `src/datasets/anchor_synth/dataset3d.py`, change the draw import line to:
```python
from .draw import anchor_stats, barycentric_center, frame_length, affine_weights
from .draw import place_object
```
Replace the whole `__init__` with:
```python
    def __init__(self, root, classes, image_size=(128, 128, 128), split="train",
                 context_size=1, object_source="blob", shape="blob", n_objects=1,
                 n_anchors=4, extrapolation=0.3, weight_concentration=1.0,
                 max_select_tries=20, object_size_frac_min=0.3, object_size_frac_max=0.8,
                 object_size_min_vox=6, scale_jitter=0.15, rotate_jitter=12.0,
                 contrast_delta=0.15, edge_blur=0.08, boundary_complexity=0.0,
                 harmonic_amp=0.30, eccentricity=3.0, n_harmonics=4, deterministic=None,
                 eval_seed_namespace=0, eval_subjects_per_task=4, epoch_length=10000,
                 max_subjects=None, aug_cfg=None):
        if object_source != "blob":
            raise NotImplementedError(
                f"object_source={object_source!r} not implemented in v1 (blob only)")
        super().__init__(root=root, classes=classes, image_size=image_size,
                         split=split, context_size=context_size,
                         max_subjects=max_subjects, class_balanced=True)
        self.aug_cfg = aug_cfg          # set AFTER super().__init__ (which nulls it)
        self.object_source = object_source
        self.shape = shape
        self.n_objects = int(n_objects)
        self.n_anchors = int(n_anchors)
        self.extrapolation = float(extrapolation)
        self.weight_concentration = float(weight_concentration)
        self.max_select_tries = int(max_select_tries)
        self.object_size_frac_min = float(object_size_frac_min)
        self.object_size_frac_max = float(object_size_frac_max)
        self.object_size_min_vox = int(object_size_min_vox)
        self.scale_jitter = float(scale_jitter)
        self.rotate_jitter = float(rotate_jitter)
        self.contrast_delta = float(contrast_delta)
        self.edge_blur = float(edge_blur)
        self.boundary_complexity = float(boundary_complexity)
        self.harmonic_amp = float(harmonic_amp)
        self.eccentricity = float(eccentricity)
        self.n_harmonics = int(n_harmonics)
        self.eval_seed_namespace = int(eval_seed_namespace)
        self.eval_subjects_per_task = int(eval_subjects_per_task)
        self.epoch_length = int(epoch_length)
        self.anchor_deterministic = (split != "train") if deterministic is None else deterministic

        # Co-occurrence structures over the anchor pool (self.classes), from the
        # parent's per-split label_to_subjects. Anchors define POSITION + SIZE only.
        self.subject_sets = {c: set(s) for c, s in self.label_to_subjects.items() if s}
        self.subject_to_classes: dict[str, set] = {}
        for c, subs in self.subject_sets.items():
            for s in subs:
                self.subject_to_classes.setdefault(s, set()).add(c)
        self.eligible_subjects = sorted(
            s for s, cs in self.subject_to_classes.items() if len(cs) >= self.n_anchors)

        if not self.eligible_subjects:
            raise ValueError(
                f"AnchorSynth3DICLDataset: no subject in split {split!r} has "
                f">= n_anchors={self.n_anchors} pool classes (classes={list(self.classes)!r}).")

        if self.anchor_deterministic:
            self._eval_index = [(subj, s) for subj in self.eligible_subjects
                                for s in range(self.eval_subjects_per_task)]
            self._n = len(self._eval_index)
        else:
            self._eval_index = None
            self._n = self.epoch_length
```

- [ ] **Step 4: Replace `_draw_specs`, add `_select_anchors` + `_load_scene`, rewrite `_render_subject`**

Replace `_draw_specs` and `_render_subject` and add the two helpers:
```python
    def _draw_specs(self, rng):
        """Per-item task spec (shared across the K+1 scenes): shape geometry,
        barycentric weights, size fraction, contrast — all anchor-independent."""
        specs = []
        for _ in range(self.n_objects):
            specs.append({
                "geom": sample_object_spec(
                    rng, shape=self.shape, eccentricity=self.eccentricity,
                    n_harmonics=self.n_harmonics, harmonic_amp=self.harmonic_amp,
                    edge_blur=self.edge_blur),
                "weights": affine_weights(rng, self.n_anchors, self.extrapolation,
                                          self.weight_concentration),
                "size_frac": float(rng.uniform(self.object_size_frac_min,
                                               self.object_size_frac_max)),
                "contrast": float(rng.uniform(-1.0, 1.0) * self.contrast_delta),
            })
        return specs

    def _select_anchors(self, subj, rng):
        """Pick n_anchors classes present in `subj`, preferring a set whose mutual
        co-occurrence yields >= context_size other subjects. Returns (anchors, cooccur)."""
        present = sorted(self.subject_to_classes[subj])
        best = None
        for _ in range(self.max_select_tries):
            pick = [present[i] for i in
                    rng.choice(len(present), self.n_anchors, replace=False)]
            cooccur = set.intersection(*(self.subject_sets[c] for c in pick)) - {subj}
            if best is None or len(cooccur) > len(best[1]):
                best = (pick, cooccur)
            if len(cooccur) >= self.context_size:
                break
        return best

    def _load_scene(self, subj):
        """Fast-path load of (ct float32 (D,H,W), full label volume (D,H,W))."""
        subj_dir = self.root / subj
        image = np.load(subj_dir / f"ct_{self._size_str}.npy", mmap_mode="r")
        full = np.load(subj_dir / f"label_{self._size_str}.npy", mmap_mode="r")
        return np.array(image, dtype=np.float32), np.asarray(full)

    def _render_subject(self, subj, anchors, specs, scene_rng):
        img, full = self._load_scene(subj)                 # img is a writable copy
        label = np.zeros(img.shape, dtype=np.int64)
        centroids = []
        for c in anchors:                                  # anchor -> POSITION + SIZE only
            st = anchor_stats(full == _ALL_CLASSES_IDX[c])
            if st is None:                                 # anchor vanished at this res
                return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label)
            centroids.append(st[0])
        centroids = np.stack(centroids)                    # (n_anchors, 3)
        L = frame_length(centroids)                        # orientation-invariant scale
        for lid, spec in enumerate(specs, 1):
            jit = 1.0 + scene_rng.uniform(-self.scale_jitter, self.scale_jitter)
            size = max(self.object_size_min_vox, int(round(spec["size_frac"] * L * jit)))
            alpha = render_object(size, spec["geom"],
                                  R_extra=small_rotation(scene_rng, self.rotate_jitter))
            if self.boundary_complexity > 0.0:
                alpha = roughen(alpha, self.boundary_complexity, scene_rng)
            center = barycentric_center(centroids, spec["weights"], size, img.shape)
            place_object(img, alpha, center, spec["contrast"], label, lid)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label)
```

- [ ] **Step 5: Rewrite `__getitem__` (subject-first + shape label_name + meta)**

Replace the whole `__getitem__` with:
```python
    def __getitem__(self, idx):
        if self.anchor_deterministic:
            target, sample_index = self._eval_index[idx]
            item_rng = np.random.default_rng(np.random.SeedSequence(
                [self.eval_seed_namespace,
                 self.eligible_subjects.index(target), sample_index]))
        else:
            item_rng = np.random.default_rng()
            target = self.eligible_subjects[item_rng.integers(len(self.eligible_subjects))]

        anchors, cooccur = self._select_anchors(target, item_rng)
        pool = sorted(cooccur)
        contexts = ([pool[i] for i in item_rng.permutation(len(pool))][:self.context_size]
                    if pool else [])
        while len(contexts) < self.context_size:
            if pool:
                contexts.append(pool[int(item_rng.integers(len(pool)))])
            else:
                contexts.append(target)                    # last-resort self-context (rare)
        chosen = [target] + contexts

        specs = self._draw_specs(item_rng)
        scene_seeds = item_rng.integers(0, 2 ** 32, size=len(chosen))
        scenes = [self._render_subject(subj, anchors, specs,
                                       np.random.default_rng(int(s)))
                  for subj, s in zip(chosen, scene_seeds)]

        image_t, label_t = scenes[0]
        context_in = [c[0] for c in scenes[1:]]
        context_out = [c[1] for c in scenes[1:]]

        if self.aug_cfg is not None and self.aug_cfg.enabled and len(context_in) > 0:
            all_images = torch.cat([image_t.unsqueeze(0), torch.stack(context_in)], dim=0)
            all_masks  = torch.cat([label_t.unsqueeze(0), torch.stack(context_out)], dim=0)
            all_images, all_masks = apply_task_aug(all_images, all_masks, self.aug_cfg.task)
            for i in range(all_images.shape[0]):
                all_images[i] = apply_intensity_aug(all_images[i], self.aug_cfg.intensity)
            image_t     = all_images[0]
            label_t     = all_masks[0]
            context_in  = list(all_images[1:])
            context_out = list(all_masks[1:])

        return {
            "image":       image_t,
            "label":       label_t,
            "context_in":  torch.stack(context_in),
            "context_out": torch.stack(context_out),
            "subject":     target,
            "label_name":  specs[0]["geom"].get("shape", self.shape),   # group by shape
            "spacing":     self._get_spacing(target),
            "meta": {"anchors": list(anchors),
                     "n_objects": self.n_objects,
                     "shapes": [s["geom"].get("shape") for s in specs],
                     "weights": [np.asarray(s["weights"]).tolist() for s in specs],
                     "contrasts": [s["contrast"] for s in specs]},
        }
```

- [ ] **Step 6: Run the dataset tests**

Run: `.venv_thor/bin/python -m pytest src/datasets/anchor_synth/test_dataset3d.py -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

```bash
git add src/datasets/anchor_synth/dataset3d.py src/datasets/anchor_synth/test_dataset3d.py
git commit -m "feat(anchor-synth3d): subject-first barycentric multi-anchor placement + frame-length sizing"
```

---

### Task 3: Config + `common.py` plumbing

**Files:**
- Modify: `configs/experiment/3d/dataset/anchor_synth3d.yaml`
- Modify: `experiments/3d/common.py:43-113`
- Test: `src/datasets/anchor_synth/test_wiring.py`

**Interfaces:**
- Consumes: refactored `AnchorSynth3DICLDataset` (Task 2).
- Produces: `anchor_shapes(cfg) -> list[str]` in `common.py`; `build_dataset` forwards the new knobs.

- [ ] **Step 1: Update the YAML knobs**

In `configs/experiment/3d/dataset/anchor_synth3d.yaml`, replace the `offset_range` / `object_size_*` lines under `anchor_synth:` with:
```yaml
  n_anchors: 4               # landmark organs defining the barycentric frame
  extrapolation: 0.3         # affine expansion around barycenter (0 = strictly inside hull)
  weight_concentration: 1.0  # Dirichlet alpha for base weights (1 = uniform simplex)
  max_select_tries: 20       # retries to find a co-occurring anchor set
  object_size_frac_min: 0.3  # object side ~ U[min,max] * L (L = mean pairwise anchor dist)
  object_size_frac_max: 0.8
  object_size_min_vox: 6     # absolute voxel floor (guards empty/degenerate renders)
```
Also update the `anchor_classes` comment to note it is the allowed anchor **pool**. Leave `object_source`, `shape`, `n_objects`, `scale_jitter`, `rotate_jitter`, `contrast_delta`, `edge_blur`, `boundary_complexity`, `eval_*`, `epoch_length` unchanged.

- [ ] **Step 2: Update the wiring test to the new cfg keys**

In `src/datasets/anchor_synth/test_wiring.py`: make the fake root contain the 4 anchor classes and update the cfg block. Replace `_make_root` label line and the `anchor_synth` cfg dict:
```python
    # in _make_root: place 4 non-coplanar anchor blocks
    from src.totalseg_dataset import _ALL_CLASSES_IDX
    for cls, (z, y, x) in zip(["aorta", "liver", "spleen", "kidney_left"],
                              [(2, 2, 2), (2, 10, 10), (10, 2, 10), (10, 10, 2)]):
        label[z:z + 3, y:y + 3, x:x + 3] = _ALL_CLASSES_IDX[cls]
```
```python
        "anchor_synth": {"object_source": "blob", "shape": "blob", "n_objects": 1,
                         "anchor_classes": ["aorta", "liver", "spleen", "kidney_left"],
                         "n_anchors": 4, "extrapolation": 0.3,
                         "weight_concentration": 1.0, "max_select_tries": 20,
                         "object_size_frac_min": 0.6, "object_size_frac_max": 1.2,
                         "object_size_min_vox": 3, "scale_jitter": 0.15,
                         "rotate_jitter": 12.0, "contrast_delta": 0.3,
                         "edge_blur": 0.08, "boundary_complexity": 0.0,
                         "eval_subjects_per_task": 2, "eval_seed_namespace": 0,
                         "epoch_length": 5},
```
Remove the old `ANCHOR`-single-class assumption: set `SIZE = 16` (unchanged) and keep `context_size: 1`. Keep the assertions `len(ds) == 5` and `item["label"].sum() > 0`.

- [ ] **Step 3: Run to verify failure**

Run: `.venv_thor/bin/python -m pytest src/datasets/anchor_synth/test_wiring.py -q`
Expected: FAIL (build_dataset still passes removed `offset_range`/`object_size_min` kwargs → TypeError).

- [ ] **Step 4: Add `anchor_shapes` and update `build_dataset`**

In `experiments/3d/common.py`, after `resolve_anchor_classes`, add:
```python
_ANCHOR_SHAPES = ("blob", "elongated", "tubular")


def anchor_shapes(cfg):
    """Validation 'classes' for anchor_synth3d = the object shapes it emits, which is
    the val grouping key (label_name). `shape=mix` -> all three, else the single shape."""
    shape = cfg.anchor_synth.get("shape", "blob")
    return list(_ANCHOR_SHAPES) if shape == "mix" else [shape]
```
Replace the `AnchorSynth3DICLDataset(...)` construction (the `offset_range` / `object_size_min` / `object_size_max_frac` kwargs) with:
```python
            n_anchors=int(a.get("n_anchors", 4)),
            extrapolation=float(a.get("extrapolation", 0.3)),
            weight_concentration=float(a.get("weight_concentration", 1.0)),
            max_select_tries=int(a.get("max_select_tries", 20)),
            object_size_frac_min=float(a.get("object_size_frac_min", 0.3)),
            object_size_frac_max=float(a.get("object_size_frac_max", 0.8)),
            object_size_min_vox=int(a.get("object_size_min_vox", 6)),
            scale_jitter=float(a.get("scale_jitter", 0.15)),
```
(Leave the rest of the kwargs — `rotate_jitter`, `contrast_delta`, `edge_blur`, `boundary_complexity`, `eval_*`, `epoch_length`, `deterministic`, `aug_cfg`, `max_subjects` — unchanged.)

- [ ] **Step 5: Run the wiring test + full anchor_synth suite**

Run: `.venv_thor/bin/python -m pytest src/datasets/anchor_synth/ -q`
Expected: PASS (all tests).

- [ ] **Step 6: Commit**

```bash
git add configs/experiment/3d/dataset/anchor_synth3d.yaml experiments/3d/common.py src/datasets/anchor_synth/test_wiring.py
git commit -m "feat(anchor-synth3d): config knobs + anchor_shapes val grouping"
```

---

### Task 4: `train.py` val branch + plot/analyze captions

**Files:**
- Modify: `experiments/3d/train.py` (anchor_synth3d val branch, ~line 276)
- Modify: `experiments/3d/plot_dataset_items.py` (caption, ~line 174)
- Modify: `experiments/3d/analyze_object_blend.py` (caption, ~line 114)

**Interfaces:**
- Consumes: `anchor_shapes` (Task 3).

- [ ] **Step 1: Point the val branch at shapes**

In `experiments/3d/train.py`, replace the anchor_synth3d val branch:
```python
    if cfg.data.get("source") == "anchor_synth3d":
        # anchor_synth3d groups val by object shape (each item's label_name = its shape).
        from common import anchor_shapes
        val_classes = anchor_shapes(cfg)
```

- [ ] **Step 2: Update the plot caption**

In `experiments/3d/plot_dataset_items.py`, replace the `anchor_synth3d` caption `extra` block (which references `a.object_size_min` / `a.object_size_max_frac`) with:
```python
        a = cfg.anchor_synth
        aug_on  = args.split == "train" and cfg.augmentations.enabled
        aug_tag = " + aug" if aug_on else ""
        extra = (f"  |  obj={a.object_source}/{a.shape}  n_obj={a.n_objects}"
                 f"  anchors={a.n_anchors}  size_frac={a.object_size_frac_min}-"
                 f"{a.object_size_frac_max}  Δ={a.contrast_delta}{aug_tag}")
```

- [ ] **Step 3: Update the analyze caption**

In `experiments/3d/analyze_object_blend.py`, replace the config-print line referencing `object_size` with:
```python
    print(f"config: contrast_delta={a.contrast_delta}  "
          f"anchors={a.n_anchors}  size_frac={a.object_size_frac_min}-{a.object_size_frac_max}  "
          f"extrapolation={a.extrapolation}  edge_blur={a.edge_blur}")
```

- [ ] **Step 4: Smoke-check the plot path (renders + caption, no crash)**

Run:
```bash
.venv_thor/bin/python experiments/3d/plot_dataset_items.py dataset=anchor_synth3d \
  --split val --n_samples 2 --out results/3d/_smoke.png 2>&1 | tail -3
```
Expected: `Saved → results/3d/_smoke.png` (no `KeyError`/`AttributeError`). Then `rm -f results/3d/_smoke.png`.

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/train.py experiments/3d/plot_dataset_items.py experiments/3d/analyze_object_blend.py
git commit -m "feat(anchor-synth3d): shape-grouped val + updated captions"
```

---

### Task 5: Empirical verification + tune size fractions

Not a TDD task — a full-dataset sanity check on real `TotalSegmentator` data confirming (a) zeros stay negligible, (b) the object's **apparent size is consistent** across target and contexts, and (c) placement is anatomically sensible. Tune `object_size_frac_*` if occupancy is off.

**Files:**
- Create (scratch, not committed): `/tmp/verify_barycentric.py`

- [ ] **Step 1: Write the verification probe**

Create `/tmp/verify_barycentric.py`:
```python
import sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from hydra import compose, initialize_config_dir
from tqdm import tqdm

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "3d"))
os.environ.setdefault("PWD", str(ROOT))
from common import make_eval_loader, anchor_shapes

with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                           version_base="1.3"):
    cfg = compose(config_name="train",
                  overrides=["model=patchset3d", "experiment=1_medverse_benchmark",
                             "arch.l=2", "dataset=anchor_synth3d"])
va = make_eval_loader(cfg, anchor_shapes(cfg), split="val")
by_shape = defaultdict(list)
# per-task apparent-size consistency: coefficient of variation of occupancy^(1/3)
consist = []
for b in tqdm(va, desc="val"):
    lbl, ctx = b["label"], b["context_out"]
    for i in range(lbl.shape[0]):
        tv = int((lbl[i] > 0).sum())
        cv = (ctx[i] > 0).flatten(1).sum(1).numpy()      # (K,)
        by_shape[b["meta"][i]["shapes"][0]].append(tv)
        sides = np.cbrt([tv, *cv.tolist()])
        if sides.mean() > 0:
            consist.append(sides.std() / sides.mean())   # lower = more consistent
print("\nzero-rate:", np.mean([v == 0 for vs in by_shape.values() for v in vs]))
for s, vs in by_shape.items():
    vs = np.array(vs); nz = vs[vs > 0]
    print(f"  {s:10s} N={vs.size:4d} zero={100*(vs==0).mean():4.1f}%  "
          f"occ_med={np.median(nz) if nz.size else 0:.0f}")
print(f"apparent-size CV across target+ctx (median): {np.median(consist):.3f}  "
      f"(lower = more consistent; absolute-size baseline was large)")
```

- [ ] **Step 2: Run it**

Run: `.venv_thor/bin/python /tmp/verify_barycentric.py 2>&1 | grep -avE "scan cache|class counts|hu_jitter" | tail -12`
Expected: `zero-rate` ≲ 0.02; per-shape `occ_med` in a healthy range (hundreds–thousands of voxels); apparent-size CV noticeably lower than the absolute-voxel design.

- [ ] **Step 3: Tune if needed**

If `occ_med` is too small/large, adjust `object_size_frac_min/max` in `configs/experiment/3d/dataset/anchor_synth3d.yaml` and re-run Step 2. Commit only the YAML if changed:
```bash
git add configs/experiment/3d/dataset/anchor_synth3d.yaml
git commit -m "chore(anchor-synth3d): tune object_size_frac range"
```

- [ ] **Step 4: Log the change**

Append a `docs/logs.md` entry summarising the barycentric positioning + frame-length sizing and the verification numbers. Commit:
```bash
git add docs/logs.md
git commit -m "docs: log anchor_synth3d barycentric positioning"
```

- [ ] **Step 5: Clean up**

Run: `rm -f /tmp/verify_barycentric.py`

---

## Self-Review

**Spec coverage:** subject-first selection (T2), barycentric position (T1+T2), frame-length size (T1+T2), affine weights + extrapolation (T1), co-occurrence structures/eligible subjects (T2), removal of single-anchor path (T1 `offset_to_center`, T2/T3 knobs), config (T3), shape-grouped val (T3 `anchor_shapes`, T4 train branch), meta anchors/weights (T2), captions (T4), empty-anchor guard (T2 `_render_subject`), determinism (T2 `_eval_index`+seed), testing (T1/T2/T3 + T5 probe). No degeneracy check — intentionally deferred per spec.

**Placeholders:** none — every code step shows full code; commands have expected output.

**Type consistency:** `affine_weights`/`frame_length`/`barycentric_center` signatures match between T1 definition and T2 use; `anchor_shapes` returns `list[str]` used identically in T3/T4; `meta["anchors"]`/`meta["weights"]`/`meta["shapes"]` produced in T2 consumed in T4/T5; `_load_scene`/`_select_anchors` used only within T2.

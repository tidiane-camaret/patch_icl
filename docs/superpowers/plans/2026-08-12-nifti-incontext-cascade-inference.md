# Nifti In-Context Cascade Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one importable function `predict_nifti(cfg, target_path, context_pairs, gt_path=None, out_path=None)` that runs the 4mm→1.5mm in-context cascade on arbitrary nifti files (GT-free target) and returns a native-grid binary mask, optionally writing a nifti and computing Dice.

**Architecture:** Extract the array-level crop/resample geometry out of `TotalSegInContextDataset` into behavior-preserving module-level helpers, then a new `experiments/3d/infer_nifti.py` reuses those helpers plus the eval model builder (`eval._build_model`) and native-stitch helpers (`evaluate._write_native`, `evaluate._predicted_native_center`) to implement a slim single-target two-pass cascade.

**Tech Stack:** Python, PyTorch, nibabel 5.4.2, OmegaConf/Hydra, numpy, pytest.

## Global Constraints

- Short docstrings on every function (project style). Log the change in `docs/logs.md`.
- Reuse existing helpers — do NOT duplicate crop geometry or native-stitch math.
- The refactor in `src/totalseg_dataloader_incontext.py` MUST be behavior-preserving (identical numbers, identical `self._cur_rng` draw order).
- Image is cubic: `T = image_size[0]` (matches existing code).
- `model.predict(target, ctx_in, ctx_out)` shapes: target `(1,1,T,T,T)`, ctx_in `(1,K,1,T,T,T)`, ctx_out `(1,K,T,T,T)` → returns `(1,T,T,T)` hard mask.
- Tests use the existing pattern: `sys.path.insert` for ROOT (`parents[3]`) and the sibling dir (`parents[1]`), `OmegaConf.create` for cfgs. Run with `pytest`.
- Spec: `docs/superpowers/specs/2026-08-12-nifti-incontext-cascade-inference-design.md`.

---

### Task 1: Extract behavior-preserving crop/resample helpers

Pull the pure array-level geometry out of `TotalSegInContextDataset` so `infer_nifti` can reuse it without instantiating a dataset. Four new module-level functions; the existing methods become thin callers.

**Files:**
- Modify: `src/totalseg_dataloader_incontext.py` (add module-level `organ_crop_arrays`, `place_image`, `place_label`, `resample_binary`; rewrite `_organ_crop_arrays`, `_place_image`, `_place_label`, `_resample_binary` as callers)
- Test: `experiments/3d/tests/test_crop_helpers.py` (new)

**Interfaces:**
- Produces:
  - `organ_crop_arrays(ct_mm, label_mm, center, sp, *, image_size, crop_mm, jitter, rng) -> (crop_ct: np.ndarray, crop_lbl: np.ndarray, out_sizes: list[int], pad_lo: list[int], crop_geom: torch.LongTensor)` where `crop_geom` is `(4,3)` = `[starts, crop_sizes, out_sizes, pad_lo]`.
  - `place_image(crop_ct: np.ndarray, out_sizes, pad_lo, T: int) -> torch.Tensor` shape `(1,T,T,T)`.
  - `place_label(label_small: torch.Tensor, out_sizes, pad_lo, T: int) -> torch.Tensor` shape `(T,T,T)` long.
  - `resample_binary(bin_np: np.ndarray, size, *, mode: str, occ_thr: float) -> torch.Tensor` long 0/1.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_crop_helpers.py`:

```python
"""Unit tests for the extracted pure crop/resample helpers (Task 1)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import (  # noqa: E402
    organ_crop_arrays, place_image, place_label, resample_binary,
)


def test_organ_crop_centered_no_jitter():
    # 20^3 volume, isotropic 1mm, T=8, crop_mm=1 -> target extent 8 voxels, centered crop.
    ct = np.arange(20 ** 3, dtype=np.float32).reshape(20, 20, 20)
    lbl = np.zeros((20, 20, 20), dtype=np.uint8)
    lbl[9:11, 9:11, 9:11] = 1
    rng = random.Random(0)
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, lbl, center=(10, 10, 10), sp=[1.0, 1.0, 1.0],
        image_size=(8, 8, 8), crop_mm=1.0, jitter=0, rng=rng)
    assert crop_ct.shape == (8, 8, 8)          # extent 8, fits in 20
    assert out_sizes == [8, 8, 8]              # full T, no padding
    assert pad_lo == [0, 0, 0]
    # starts = center - cs//2 = 10 - 4 = 6
    assert geom[0].tolist() == [6, 6, 6]
    assert geom.shape == (4, 3)


def test_organ_crop_thin_axis_padded():
    # A thin axis (size 4 < extent 8) is captured whole and maps to <T with centre pad.
    ct = np.zeros((4, 20, 20), dtype=np.float32)
    lbl = np.zeros((4, 20, 20), dtype=np.uint8)
    rng = random.Random(0)
    _, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, lbl, center=(2, 10, 10), sp=[1.0, 1.0, 1.0],
        image_size=(8, 8, 8), crop_mm=1.0, jitter=0, rng=rng)
    assert geom[1].tolist()[0] == 4            # crop_sizes[0] clamped to native 4
    assert out_sizes[0] == 4 and pad_lo[0] == 2  # 4 maps to 4/8, centred -> pad 2


def test_place_image_pads_with_air():
    crop = np.full((4, 8, 8), -3.0, dtype=np.float32)
    img = place_image(crop, out_sizes=[4, 8, 8], pad_lo=[2, 0, 0], T=8)
    assert img.shape == (1, 8, 8, 8)
    assert float(img[0, 0, 0, 0]) == -3.0      # padded region filled with crop.min()


def test_resample_binary_occupancy_keeps_thin():
    m = np.zeros((8, 8, 8), dtype=bool)
    m[0, 0, 0] = True                          # single voxel
    out = resample_binary(m, (2, 2, 2), mode="occupancy", occ_thr=0.1)
    assert out.sum() >= 1                       # densest voxel kept (non-empty guarantee)
    out_near = resample_binary(m, (2, 2, 2), mode="nearest", occ_thr=0.5)
    assert out_near.shape == (2, 2, 2)


def test_place_label_centers():
    small = torch.ones(4, 8, 8, dtype=torch.long)
    lab = place_label(small, out_sizes=[4, 8, 8], pad_lo=[2, 0, 0], T=8)
    assert lab.shape == (8, 8, 8)
    assert lab[0].sum() == 0 and lab[2].sum() == 64   # padded slices 0, body slices set
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/3d/tests/test_crop_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'organ_crop_arrays'`.

- [ ] **Step 3: Add the four module-level helpers**

In `src/totalseg_dataloader_incontext.py`, add these module-level functions (place them just above the `class TotalSegInContextDataset` definition, after the existing module helpers like `_lazy_shuffle`). Copy the geometry EXACTLY from the current methods (`_organ_crop_arrays` lines ~1068-1113, `_place_image` ~1115-1128, `_place_label` ~1130-1139, `_resample_binary` ~1141-1156):

```python
def organ_crop_arrays(ct_mm, label_mm, center, sp, *, image_size, crop_mm, jitter, rng):
    """Pure array-level organ crop. Returns (crop_ct, crop_lbl, out_sizes, pad_lo, crop_geom).

    Slices a centre crop of fixed physical extent (T*crop_mm/axis) from ct_mm/label_mm
    (same shape), records where it lands in the final T³ grid. Behaviour extracted verbatim
    from TotalSegInContextDataset._organ_crop_arrays; `rng` supplies the crop jitter."""
    T = image_size[0]
    cd, ch, cw = center
    D, H, W = label_mm.shape
    phys_ref = T * crop_mm
    target_sizes = [max(1, round(phys_ref / spi)) for spi in sp]
    crop_sizes = [min(dim, t) for t, dim in zip(target_sizes, (D, H, W))]
    starts = []
    for c, s, cs in zip((cd, ch, cw), (D, H, W), crop_sizes):
        smax = max(0, s - cs)
        ideal = c - cs // 2
        lo = min(max(0, ideal - jitter), smax)
        hi = min(max(0, ideal + jitter), smax)
        starts.append(rng.randint(lo, hi))
    d0, h0, w0 = starts
    crop_ct = ct_mm[d0:d0 + crop_sizes[0], h0:h0 + crop_sizes[1], w0:w0 + crop_sizes[2]]
    crop_lbl = label_mm[d0:d0 + crop_sizes[0], h0:h0 + crop_sizes[1], w0:w0 + crop_sizes[2]]
    out_sizes = [max(1, min(T, round(cs / t * T))) for cs, t in zip(crop_sizes, target_sizes)]
    pad_lo = [(T - o) // 2 for o in out_sizes]
    crop_geom = torch.tensor([starts, list(crop_sizes), out_sizes, pad_lo], dtype=torch.long)
    return crop_ct, crop_lbl, out_sizes, pad_lo, crop_geom


def place_image(crop_ct, out_sizes, pad_lo, T):
    """Resample the native CT slice to out_sizes (trilinear) and centre it in an
    air-filled T³ tensor. Returns (1, T, T, T)."""
    img_small = F.interpolate(
        torch.from_numpy(crop_ct.astype(np.float32))[None, None],
        size=tuple(out_sizes), mode="trilinear", align_corners=False)[0]
    if all(o == T for o in out_sizes):
        return img_small
    image_t = torch.full((1, T, T, T), float(crop_ct.min()), dtype=torch.float32)
    sl = (slice(None),) + tuple(slice(p, p + o) for p, o in zip(pad_lo, out_sizes))
    image_t[sl] = img_small
    return image_t


def place_label(label_small, out_sizes, pad_lo, T):
    """Centre an already-resampled label (spatial dims out_sizes, long) in a
    background-0 T³ tensor. Returns (T, T, T)."""
    if all(o == T for o in out_sizes):
        return label_small
    label_t = torch.zeros(T, T, T, dtype=torch.long)
    sl = tuple(slice(p, p + o) for p, o in zip(pad_lo, out_sizes))
    label_t[sl] = label_small
    return label_t


def resample_binary(bin_np, size, *, mode, occ_thr):
    """Resize a binary mask to `size` -> long (0/1). "occupancy" area-pools + thresholds
    (thin structures survive; non-empty input never returns empty); "nearest" point-samples."""
    t = torch.from_numpy(np.ascontiguousarray(bin_np, dtype=np.float32))[None, None]
    if mode == "occupancy":
        frac = F.interpolate(t, size=size, mode="area")[0, 0]
        out = frac >= occ_thr
        if not bool(out.any()) and bin_np.any():
            out.view(-1)[int(frac.argmax())] = True
        return out.long()
    return (F.interpolate(t, size=size, mode="nearest")[0, 0] > 0.5).long()
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `pytest experiments/3d/tests/test_crop_helpers.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Rewrite the four methods as thin callers (behavior-preserving)**

Replace the bodies of the existing methods so they delegate to the module functions. `_organ_crop_arrays` keeps the disk load + assert + raw_ct normalize + `self._last_crop_geom`:

```python
    def _organ_crop_arrays(self, subj_dir, label_mm, center, sp):
        """See module-level organ_crop_arrays. Adds the disk CT load, shape assert,
        raw_ct crop normalisation, and records self._last_crop_geom."""
        ct_mm = self._load_native_ct_mmap(subj_dir)
        assert ct_mm.shape == label_mm.shape, (
            f"{subj_dir.name}: ct.npy {ct_mm.shape} != label.npy {label_mm.shape} — "
            f"conversion must resample labels onto the CT grid (see convert_to_npy chemotox)")
        crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            ct_mm, label_mm, center, sp, image_size=self.image_size,
            crop_mm=self._crop_mm, jitter=self.crop_jitter, rng=self._cur_rng)
        if self.raw_ct:
            crop_ct = self._normalize_native(subj_dir.name, np.ascontiguousarray(crop_ct))
        self._last_crop_geom = geom
        return crop_ct, crop_lbl, out_sizes, pad_lo

    def _place_image(self, crop_ct, out_sizes, pad_lo):
        return place_image(crop_ct, out_sizes, pad_lo, self.image_size[0])

    def _place_label(self, label_small, out_sizes, pad_lo):
        return place_label(label_small, out_sizes, pad_lo, self.image_size[0])

    def _resample_binary(self, bin_np, size):
        return resample_binary(bin_np, size, mode=self.mask_downsample,
                               occ_thr=self.mask_occupancy_thr)
```

Note: the original `_organ_crop_arrays` applied `raw_ct` normalisation before computing `out_sizes`; here it is applied to `crop_ct` after the geometry, which is equivalent (normalisation is pointwise and does not touch shapes). Keep the `_cur_rng` usage — the jitter draw is now inside `organ_crop_arrays`, same call order.

- [ ] **Step 6: Run the extraction test + existing dataloader-adjacent tests**

Run: `pytest experiments/3d/tests/test_crop_helpers.py experiments/3d/tests/test_sweep_guard.py experiments/3d/tests/test_locator.py -v`
Expected: PASS (crop-helper tests still pass; the two existing tests are unaffected).

- [ ] **Step 7: Commit**

```bash
/usr/bin/git add src/totalseg_dataloader_incontext.py experiments/3d/tests/test_crop_helpers.py
/usr/bin/git commit -m "refactor(dataloader): extract pure organ_crop_arrays/place_*/resample_binary helpers"
```

---

### Task 2: Nifti IO + crop-prep helpers

The building blocks `predict_nifti` composes: load a nifti to `(array, affine)`, derive voxel spacing, compute a mask centroid, and turn a native (ct[, mask]) + centre into model-ready `T³` tensors.

**Files:**
- Create: `experiments/3d/infer_nifti.py`
- Test: `experiments/3d/tests/test_infer_nifti.py` (new)

**Interfaces:**
- Consumes (Task 1): `organ_crop_arrays`, `place_image`, `place_label`, `resample_binary`; `src.totalseg_dataset.normalize_ct`.
- Produces:
  - `load_nifti(path) -> (arr: np.ndarray, affine: np.ndarray)` (arr float32 for CT-like data).
  - `voxel_spacing(affine) -> list[float]` (3 mm/voxel).
  - `mask_centroid(mask: np.ndarray) -> tuple[int,int,int]` (COM; volume centre + warn if empty).
  - `prep_context(ct, mask, sp, center, *, T, crop_mm, mask_downsample, occ_thr) -> (img_t (1,T,T,T), mask_t (T,T,T) long)`.
  - `prep_target(ct, sp, center, *, T, crop_mm) -> (img_t (1,T,T,T), geom (4,3) long tensor)`.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_infer_nifti.py`:

```python
"""Tests for the nifti in-context cascade inference module."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sibling infer_nifti / evaluate

from infer_nifti import (  # noqa: E402
    load_nifti, voxel_spacing, mask_centroid, prep_context, prep_target,
)


def _write_nifti(tmp_path, name, arr, spacing=(1.0, 1.0, 1.0)):
    aff = np.diag([*spacing, 1.0])
    p = tmp_path / name
    nib.save(nib.Nifti1Image(arr, aff), str(p))
    return p


def test_load_and_spacing(tmp_path):
    arr = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    p = _write_nifti(tmp_path, "ct.nii.gz", arr, spacing=(2.0, 1.5, 1.0))
    got, aff = load_nifti(p)
    assert got.shape == (4, 5, 6)
    assert voxel_spacing(aff) == [2.0, 1.5, 1.0]


def test_mask_centroid_and_empty():
    m = np.zeros((10, 10, 10), dtype=bool)
    m[4:6, 4:6, 4:6] = True
    assert mask_centroid(m) == (4, 4, 4)      # COM of the cube (floored)
    # empty -> volume centre (with a warning)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mask_centroid(np.zeros((8, 8, 8), bool)) == (4, 4, 4)


def test_prep_target_shapes():
    ct = np.zeros((40, 40, 40), dtype=np.float32)
    img_t, geom = prep_target(ct, [1.0, 1.0, 1.0], (20, 20, 20),
                              T=8, crop_mm=1.0)
    assert img_t.shape == (1, 8, 8, 8)
    assert geom.shape == (4, 3)


def test_prep_context_shapes():
    ct = np.zeros((40, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[18:22, 18:22, 18:22] = True
    img_t, mask_t = prep_context(ct, mask, [1.0, 1.0, 1.0], (20, 20, 20),
                                 T=8, crop_mm=1.0, mask_downsample="occupancy", occ_thr=0.1)
    assert img_t.shape == (1, 8, 8, 8)
    assert mask_t.shape == (8, 8, 8)
    assert mask_t.sum() > 0                    # organ survives into the crop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/3d/tests/test_infer_nifti.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'infer_nifti'`.

- [ ] **Step 3: Create `experiments/3d/infer_nifti.py` with IO + prep helpers**

```python
"""Nifti in-context cascade inference — predict a target organ mask from context
(image, binary-mask) nifti pairs via the 4mm->1.5mm cascade, GT-free for the target.

See docs/superpowers/specs/2026-08-12-nifti-incontext-cascade-inference-design.md.
"""
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.affines import voxel_sizes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling eval/evaluate/common

from src.totalseg_dataset import normalize_ct
from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
)


def load_nifti(path):
    """Load a nifti -> (array, affine). Array is the stored data with scaling applied."""
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine


def voxel_spacing(affine):
    """Per-axis mm/voxel (3,) from the affine, aligned with the array axes."""
    return [float(v) for v in voxel_sizes(affine)]


def mask_centroid(mask):
    """Integer centre-of-mass (d,h,w) of a binary mask; volume centre + warn if empty."""
    fg = np.asarray(mask) > 0
    if not fg.any():
        warnings.warn("infer_nifti: empty context/target mask — using volume centre.",
                      stacklevel=2)
        return tuple(s // 2 for s in fg.shape)
    idx = np.nonzero(fg)
    return tuple(int(a.mean()) for a in idx)


def prep_target(ct, sp, center, *, T, crop_mm):
    """Native CT (normalised) + centre -> (img_t (1,T,T,T), crop_geom (4,3)).

    No target label, so ct doubles as the label array for organ_crop_arrays (its
    crop_lbl output is discarded). rng is unused at jitter=0 (centred crop)."""
    import random
    crop_ct, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, ct, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    return place_image(crop_ct, out_sizes, pad_lo, T), geom


def prep_context(ct, mask, sp, center, *, T, crop_mm, mask_downsample, occ_thr):
    """Native (CT, binary mask) + centre -> (img_t (1,T,T,T), mask_t (T,T,T) long)."""
    import random
    assert ct.shape == mask.shape, f"context ct {ct.shape} != mask {mask.shape}"
    crop_ct, crop_mask, out_sizes, pad_lo, _ = organ_crop_arrays(
        ct, mask, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    img_t = place_image(crop_ct, out_sizes, pad_lo, T)
    mask_small = resample_binary(np.asarray(crop_mask) > 0, tuple(out_sizes),
                                 mode=mask_downsample, occ_thr=occ_thr)
    return img_t, place_label(mask_small, out_sizes, pad_lo, T)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest experiments/3d/tests/test_infer_nifti.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
/usr/bin/git add experiments/3d/infer_nifti.py experiments/3d/tests/test_infer_nifti.py
/usr/bin/git commit -m "feat(infer_nifti): nifti IO + crop-prep helpers"
```

---

### Task 3: `predict_nifti` cascade orchestration + metrics

Compose the helpers into the single-target two-pass cascade: coarse (volume-centre) → recenter on the coarse prediction → fine, stitch into the native volume, optionally write a nifti and compute Dice + coarse-only Dice.

**Files:**
- Modify: `experiments/3d/infer_nifti.py` (add `predict_nifti` + a small `_dice` import + a stitch helper reuse)
- Modify: `experiments/3d/tests/test_infer_nifti.py` (add the integration test with a stub model)
- Modify: `docs/logs.md` (log the new capability)

**Interfaces:**
- Consumes (Task 2): `load_nifti`, `voxel_spacing`, `mask_centroid`, `prep_target`, `prep_context`.
- Consumes (existing): `eval._build_model(cfg)`, `eval._warn_uninherited_data(cfg)`, `evaluate._write_native(native, pred, geom)`, `evaluate._predicted_native_center(prob, geom)`, `evaluate.dice_binary(pred, target)`, `common.DEVICE`.
- Produces:
  - `predict_nifti(cfg, target_path, context_pairs, gt_path=None, out_path=None) -> dict` with keys `pred` (bool ndarray, native target grid), `affine`, `dice` (float|None), `coarse_only_dice` (float|None), `pred_path` (Path|None).

- [ ] **Step 1: Write the failing integration test**

Append to `experiments/3d/tests/test_infer_nifti.py`:

```python
from omegaconf import OmegaConf  # noqa: E402


class _StubModel:
    """Minimal model with .predict: emits a centred cube in the T³ grid (independent of
    input), so the cascade wiring/stitch/metrics can be exercised without a checkpoint."""
    spacing_aware = False

    def predict(self, target_img, context_imgs, context_masks, **kw):
        B, _, T, _, _ = target_img.shape
        out = torch.zeros(B, T, T, T)
        q = T // 4
        out[:, q:T - q, q:T - q, q:T - q] = 1.0
        return out


def _cfg():
    return OmegaConf.create({
        "data": {"image_size": [16, 16, 16], "crop_spacing_mm": 1.5,
                 "use_crop": True, "mask_downsample": "occupancy",
                 "mask_occupancy_thr": 0.1, "source": "totalseg"},
        "eval": {"model": "stub", "checkpoint": None, "spacing_sweep": [4, 1.5]},
    })


def test_predict_nifti_end_to_end(tmp_path, monkeypatch):
    import infer_nifti
    # Bypass the real model builder + drift warning (no checkpoint in the test).
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _StubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)

    shape = (32, 32, 32)
    ct = np.zeros(shape, dtype=np.int16)
    organ = np.zeros(shape, dtype=np.uint8)
    organ[12:20, 12:20, 12:20] = 1
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    tgt = tmp_path / "tgt.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(tgt))
    cimg = tmp_path / "cimg.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(cimg))
    cmsk = tmp_path / "cmsk.nii.gz"; nib.save(nib.Nifti1Image(organ, aff), str(cmsk))
    gt = tmp_path / "gt.nii.gz"; nib.save(nib.Nifti1Image(organ, aff), str(gt))
    out = tmp_path / "pred.nii.gz"

    res = infer_nifti.predict_nifti(
        _cfg(), tgt, [(cimg, cmsk)], gt_path=gt, out_path=out)

    assert res["pred"].shape == shape
    assert res["pred"].dtype == bool
    assert res["pred"].any()                    # stub emits a non-empty cube
    assert 0.0 <= res["dice"] <= 1.0
    assert 0.0 <= res["coarse_only_dice"] <= 1.0
    assert out.exists()
    loaded, _ = load_nifti(out)
    assert loaded.shape == shape


def test_predict_nifti_requires_context(tmp_path, monkeypatch):
    import infer_nifti
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _StubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)
    ct = np.zeros((8, 8, 8), dtype=np.int16)
    aff = np.eye(4)
    tgt = tmp_path / "t.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(tgt))
    import pytest
    with pytest.raises(ValueError, match="context"):
        infer_nifti.predict_nifti(_cfg(), tgt, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/3d/tests/test_infer_nifti.py::test_predict_nifti_end_to_end -v`
Expected: FAIL with `AttributeError: module 'infer_nifti' has no attribute '_build_model'` (the monkeypatch target does not exist yet).

- [ ] **Step 3: Add imports + `predict_nifti` to `experiments/3d/infer_nifti.py`**

Add these imports near the top (after the existing `from src...` imports):

```python
from eval import _build_model, _warn_uninherited_data
from evaluate import _write_native, _predicted_native_center, dice_binary
from common import DEVICE
```

Then add the orchestration function at the end of the module:

```python
def _resample_gt(gt, shape):
    """Nearest-resample a binary GT to `shape` if it differs (bool)."""
    if gt.shape == shape:
        return gt.astype(bool)
    t = torch.from_numpy(gt.astype(np.float32))[None, None]
    return (torch.nn.functional.interpolate(t, size=shape, mode="nearest")[0, 0] > 0.5).numpy()


def predict_nifti(cfg, target_path, context_pairs, gt_path=None, out_path=None):
    """Run the coarse->fine in-context cascade on nifti files (GT-free target).

    cfg            : OmegaConf cfg (same surface as experiments/3d/eval.py). Uses
                     data.image_size / mask_downsample / mask_occupancy_thr and
                     eval.model / eval.checkpoint / eval.spacing_sweep.
    target_path    : target CT .nii.gz.
    context_pairs  : list[(image_path, binary_mask_path)] for the same organ (K = len).
    gt_path        : optional target GT (binary) .nii.gz -> Dice + coarse-only Dice.
    out_path       : optional -> write the predicted mask as .nii.gz on the target grid.

    Returns {"pred", "affine", "dice", "coarse_only_dice", "pred_path"}.
    """
    if not context_pairs:
        raise ValueError("predict_nifti needs at least one context pair (in-context model)")

    _warn_uninherited_data(cfg)
    model = _build_model(cfg)
    T = int(cfg.data.image_size[0])
    crop_ds = cfg.data.get("mask_downsample", "occupancy")
    crop_thr = float(cfg.data.get("mask_occupancy_thr", 0.1))
    spacings = [float(s) for s in cfg.eval.spacing_sweep]

    # --- load target + contexts once (arrays reused across passes) --------------
    tgt_ct, affine = load_nifti(target_path)
    tgt_ct = normalize_ct(tgt_ct)
    tgt_sp = voxel_spacing(affine)
    shape = tgt_ct.shape

    contexts = []  # (ct_norm, mask_bool, spacing, centroid)
    for img_p, msk_p in context_pairs:
        c_ct, c_aff = load_nifti(img_p)
        c_msk, _ = load_nifti(msk_p)
        c_msk = np.asarray(c_msk) > 0
        contexts.append((normalize_ct(c_ct), c_msk, voxel_spacing(c_aff),
                         mask_centroid(c_msk)))

    native = np.zeros(shape, dtype=bool)   # stitched (coarse then fine overwrite)
    coarse_native = None
    center = tuple(s // 2 for s in shape)  # coarse: volume centre
    prev_pred = prev_geom = None

    for i, s in enumerate(spacings):
        if i > 0:
            c = _predicted_native_center(
                torch.from_numpy(prev_pred.astype(np.float32)),
                torch.from_numpy(prev_geom.astype(np.int64)))
            center = tuple(s2 // 2 for s2 in shape) if c == "volume_center" else c

        tgt_img, geom = prep_target(tgt_ct, tgt_sp, center, T=T, crop_mm=s)
        ctx_in, ctx_out = [], []
        for c_ct, c_msk, c_sp, c_center in contexts:
            im, mk = prep_context(c_ct, c_msk, c_sp, c_center, T=T, crop_mm=s,
                                  mask_downsample=crop_ds, occ_thr=crop_thr)
            ctx_in.append(im)
            ctx_out.append(mk)
        target_b = tgt_img.unsqueeze(0).to(DEVICE)                      # (1,1,T,T,T)
        ctx_in_b = torch.stack(ctx_in).unsqueeze(0).to(DEVICE)         # (1,K,1,T,T,T)
        ctx_out_b = torch.stack(ctx_out).unsqueeze(0).to(DEVICE)       # (1,K,T,T,T)

        kw = {"spacing": s} if getattr(model, "spacing_aware", False) else {}
        with torch.no_grad():
            pred = model.predict(target_b, ctx_in_b, ctx_out_b, **kw)   # (1,T,T,T)
        pred = pred.squeeze(0).cpu().numpy()

        geom_np = geom.numpy()
        _write_native(native, pred, geom_np)
        if i == 0:
            coarse_native = native.copy()
        prev_pred, prev_geom = pred, geom_np

    # --- output + metrics -------------------------------------------------------
    pred_path = None
    if out_path is not None:
        nib.save(nib.Nifti1Image(native.astype(np.uint8), affine), str(out_path))
        pred_path = Path(out_path)

    dice = coarse_only = None
    if gt_path is not None:
        gt, _ = load_nifti(gt_path)
        gt = _resample_gt(np.asarray(gt) > 0, shape)
        gt_t = torch.from_numpy(gt)
        dice = float(dice_binary(torch.from_numpy(native), gt_t))
        coarse_only = float(dice_binary(torch.from_numpy(coarse_native), gt_t))

    return {"pred": native, "affine": affine, "dice": dice,
            "coarse_only_dice": coarse_only, "pred_path": pred_path}
```

- [ ] **Step 4: Run the integration tests to verify they pass**

Run: `pytest experiments/3d/tests/test_infer_nifti.py -v`
Expected: PASS (all Task 2 + Task 3 tests). If `_build_model`/`_warn_uninherited_data` import from `eval` triggers Hydra/module import side effects, confirm the sibling `sys.path.insert` (parents[1]) is present — the test adds it too.

- [ ] **Step 5: Log the change in `docs/logs.md`**

Append a dated entry:

```markdown
## 2026-08-12 — nifti in-context cascade inference

Added `experiments/3d/infer_nifti.py::predict_nifti(cfg, target_path, context_pairs,
gt_path=None, out_path=None)`: runs the 4mm->1.5mm in-context cascade on arbitrary
nifti files (GT-free target; coarse crops on the volume centre, fine recenters on the
coarse prediction centroid). Reuses eval._build_model + evaluate._write_native /
_predicted_native_center + the newly extracted crop helpers (organ_crop_arrays /
place_image / place_label / resample_binary, refactored out of TotalSegInContextDataset,
behaviour-preserving). Returns the native-grid mask + optional Dice and coarse-only Dice.
```

- [ ] **Step 6: Commit**

```bash
/usr/bin/git add experiments/3d/infer_nifti.py experiments/3d/tests/test_infer_nifti.py docs/logs.md
/usr/bin/git commit -m "feat(infer_nifti): predict_nifti coarse->fine cascade + metrics"
```

---

## Self-Review

**Spec coverage:**
- Public interface `predict_nifti(cfg, target_path, context_pairs, gt_path, out_path)` + return dict → Task 3. ✓
- GT-free coarse crop on volume centre; fine recenter on coarse prediction → Task 3 loop. ✓
- Contexts organ-cropped on binary-mask centroid → Task 2 `prep_context` + `mask_centroid`. ✓
- Spacing from nifti affine → Task 2 `voxel_spacing`. ✓
- Reuse model builder + native-stitch/recenter helpers → Task 3 imports. ✓
- Refactor: extract `organ_crop_arrays`/`place_image`/`place_label`/`resample_binary` behavior-preserving → Task 1. ✓
- Recenter uses hard `predict` mask centroid (one forward/pass) → Task 3 (`_predicted_native_center(prev_pred...)`). ✓
- Metrics: native Dice + coarse-only Dice → Task 3. ✓
- Fidelity params from composed cfg (T=128 via dataset=totalseg); `_warn_uninherited_data` safety net → Task 3. ✓
- Error handling: empty context_pairs → ValueError (Task 3 test); empty mask centroid → volume centre + warn (Task 2 `mask_centroid`); empty coarse pred → `_predicted_native_center` returns "volume_center" handled (Task 3). ✓
- Test with tiny synthetic niftis + cheap stub model → Tasks 2 & 3. ✓

**Placeholder scan:** No TBD/TODO; all code blocks concrete and directly runnable. ✓

**Type consistency:** `organ_crop_arrays` returns a 5-tuple `(crop_ct, crop_lbl, out_sizes, pad_lo, crop_geom)` — used consistently in Task 1 methods and Task 2 `prep_*`. `predict` returns `(1,T,T,T)` → squeezed to `(T,T,T)` → `_write_native` expects `(T,T,T)` + `(4,3)` geom (numpy). `_predicted_native_center(prob, geom)` gets a tensor prob + int64 geom tensor, returns tuple or `"volume_center"`. `dice_binary` takes two tensors. All aligned. ✓

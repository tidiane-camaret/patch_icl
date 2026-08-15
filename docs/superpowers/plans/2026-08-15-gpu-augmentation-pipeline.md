# GPU Augmentation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the 3D in-context augmentation pipeline off CPU DataLoader workers into a single batched GPU stage that runs in the training loop between `batch.to(device)` and `model(...)`.

**Architecture:** The dataset stops calling `apply_*_aug` and instead emits raw (un-augmented) volumes plus a per-item `aug_mode` code; a new `GpuAugmentor` (own batched torch ops) applies all augmentation on-device. Task construction (sampling, crop, I/O, label synthesis, clone replication) stays on CPU workers. Behind a config flag; CPU path is the fallback.

**Tech Stack:** PyTorch (`F.grid_sample`, `F.affine_grid`, `F.conv3d`, `F.interpolate`), Hydra/OmegaConf configs, pytest.

**Spec:** `docs/superpowers/specs/2026-08-15-gpu-augmentation-pipeline-design.md`

## Global Constraints

- **Own torch ops only** — no MONAI/Kornia/new dependencies.
- **Non-differentiable** — all aug runs under `torch.no_grad()`; in-place ops allowed.
- **Device-agnostic** — every op infers device/dtype from its input tensor so tests run on CPU; production runs on CUDA.
- **In-context semantics** — for `real`/`self_context` modes ONE geometric transform is shared across the K+1 volumes of a task; intensity is independent per volume.
- **Value range** — CT images are z-scored in `[CT_NORM_MIN, CT_NORM_MAX]` (import from `src.totalseg_dataset`); intensity ops clamp to this range.
- **Exact reproduction of the CPU pipeline is a NON-goal** — tests assert shape/dtype/range/semantics and distributional plausibility, not bit-exactness.
- **aug_mode codes:** `0 = real`, `1 = synth`, `2 = self_context`.
- **Batch tensor shapes:** `image (B,1,D,H,W)`, `label (B,D,H,W)` int64, `context_in (B,K,1,D,H,W)`, `context_out (B,K,D,H,W)` int64, `aug_mode (B,)` int64. `T = K+1`.
- Reuse `_make_affine_theta` and `CT_NORM_MIN/MAX` from existing modules; do not re-derive.
- Log the change in `docs/logs.md` (project rule).

---

## File Structure

- **Create** `src/gpu_augment.py` — `GpuAugmentor` + batched primitives (`_stack_task`, `_unstack_task`, `_geometric`, `_batched_intensity`, `_batched_gin_ipa`).
- **Create** `tests/test_gpu_augment.py` — unit tests for all primitives + the augmentor (run on CPU).
- **Modify** `src/totalseg_dataloader_incontext.py` — `defer_aug_to_gpu` param, gate the `apply_*_aug` calls, emit `aug_mode`; `incontext_collate_fn` stacks `aug_mode`.
- **Modify** `experiments/3d/common.py` — forward `defer_aug_to_gpu` to the train dataset from `cfg.augmentations.gpu`.
- **Modify** `experiments/3d/train.py` — build `GpuAugmentor`, move batch to device once, call the augmentor before `model(...)`.
- **Modify** `configs/augmentations/nnunet.yaml` — add `augmentations.gpu: false`.

---

### Task 1: Module scaffold + stack/unstack helpers

**Files:**
- Create: `src/gpu_augment.py`
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Produces:
  - `_stack_task(batch: dict) -> tuple[Tensor, Tensor, int, int]` returning `(vols, masks, B, T)` where `vols` is `(B*T, 1, D, H, W)`, `masks` is `(B*T, D, H, W)` int64, and volume index `b*T + 0` is task `b`'s target, `b*T + 1..K` its contexts.
  - `_unstack_task(vols, masks, B, T, batch) -> None` writes the augmented volumes back into `batch["image"/"label"/"context_in"/"context_out"]` in place.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_augment.py
import sys; sys.path.insert(0, ".")
import torch
from src.gpu_augment import _stack_task, _unstack_task


def _fake_batch(B=2, K=3, D=6, H=6, W=6):
    return {
        "image":       torch.randn(B, 1, D, H, W),
        "label":       torch.randint(0, 2, (B, D, H, W)),
        "context_in":  torch.randn(B, K, 1, D, H, W),
        "context_out": torch.randint(0, 2, (B, K, D, H, W)),
        "aug_mode":    torch.zeros(B, dtype=torch.long),
    }


def test_stack_unstack_roundtrip():
    b = _fake_batch()
    ref = {k: v.clone() for k, v in b.items()}
    vols, masks, B, T = _stack_task(b)
    assert vols.shape == (B * T, 1, 6, 6, 6)
    assert masks.shape == (B * T, 6, 6, 6)
    assert masks.dtype == torch.long
    # target of task 0 is vols[0]; first context of task 0 is vols[1]
    assert torch.equal(vols[0, 0], ref["image"][0, 0])
    assert torch.equal(vols[1, 0], ref["context_in"][0, 0, 0])
    _unstack_task(vols, masks, B, T, b)
    for k in ("image", "label", "context_in", "context_out"):
        assert torch.equal(b[k], ref[k])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_augment.py::test_stack_unstack_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (no `src.gpu_augment`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/gpu_augment.py
"""Batched GPU augmentation for 3D in-context segmentation.

Runs in the training loop after batch.to(device), replacing the per-item CPU
augmentation in totalseg_dataloader_incontext. All ops are device/dtype-agnostic
and run under torch.no_grad(). See docs/superpowers/specs/2026-08-15-*.
"""
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from src.augmentations import _make_affine_theta
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX

REAL, SYNTH, SELF_CONTEXT = 0, 1, 2


def _stack_task(batch: dict) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """batch tensors -> (vols (B*T,1,D,H,W), masks (B*T,D,H,W), B, T). vol b*T+0 = target b."""
    img, ctx = batch["image"], batch["context_in"]          # (B,1,D,H,W),(B,K,1,D,H,W)
    lbl, cout = batch["label"], batch["context_out"]         # (B,D,H,W),(B,K,D,H,W)
    B, K = ctx.shape[0], ctx.shape[1]
    T = K + 1
    D, H, W = img.shape[-3:]
    vols = torch.cat([img.unsqueeze(1), ctx], dim=1).reshape(B * T, 1, D, H, W)
    masks = torch.cat([lbl.unsqueeze(1), cout], dim=1).reshape(B * T, D, H, W).long()
    return vols, masks, B, T


def _unstack_task(vols: torch.Tensor, masks: torch.Tensor, B: int, T: int, batch: dict) -> None:
    D, H, W = vols.shape[-3:]
    v = vols.reshape(B, T, 1, D, H, W)
    m = masks.reshape(B, T, D, H, W)
    batch["image"] = v[:, 0]
    batch["context_in"] = v[:, 1:]
    batch["label"] = m[:, 0]
    batch["context_out"] = m[:, 1:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_augment.py::test_stack_unstack_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpu_augment.py tests/test_gpu_augment.py
git commit -m "feat(gpu_augment): task stack/unstack helpers"
```

---

### Task 2: Batched geometric aug (shared or independent)

**Files:**
- Modify: `src/gpu_augment.py`
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Consumes: `_make_affine_theta` (existing).
- Produces: `_geometric(vols, masks, group_size, cfg, gen) -> (vols, masks)`. One shared flip/affine/elastic transform per consecutive group of `group_size` volumes. `group_size=T` → shared across a task; `group_size=1` → independent per volume. `cfg` has `.flip.{p_d,p_h,p_w}`, `.affine.{p,max_angle_deg,scale_min,scale_max,max_translate}`, `.elastic.{p,alpha,grid_scale}` (grid_scale optional, default 8). `gen` is a `torch.Generator` on the input device.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_gpu_augment.py
from types import SimpleNamespace
from src.gpu_augment import _geometric

def _geo_cfg(affine_p=1.0, flip_p=0.0, elastic_p=0.0):
    return SimpleNamespace(
        flip=SimpleNamespace(p_d=flip_p, p_h=flip_p, p_w=flip_p),
        affine=SimpleNamespace(p=affine_p, max_angle_deg=30.0, scale_min=0.9,
                               scale_max=1.1, max_translate=0.1),
        elastic=SimpleNamespace(p=elastic_p, alpha=0.1, grid_scale=4),
    )

def test_geometric_shared_within_group():
    # 1 task, T=3 identical volumes -> shared transform keeps them identical
    D = 8
    vol = torch.randn(1, 1, D, D, D)
    vols = vol.repeat(3, 1, 1, 1, 1)                 # 3 identical volumes, one group
    masks = torch.randint(0, 2, (1, D, D, D)).repeat(3, 1, 1, 1)
    gen = torch.Generator().manual_seed(0)
    out, om = _geometric(vols.clone(), masks.clone(), group_size=3, cfg=_geo_cfg(), gen=gen)
    assert out.shape == vols.shape
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])   # shared
    assert not torch.allclose(out[0], vols[0])       # actually transformed

def test_geometric_independent_diverges():
    D = 8
    vols = torch.randn(1, 1, D, D, D).repeat(4, 1, 1, 1, 1)
    masks = torch.zeros(4, D, D, D, dtype=torch.long)
    gen = torch.Generator().manual_seed(1)
    out, _ = _geometric(vols.clone(), masks.clone(), group_size=1, cfg=_geo_cfg(), gen=gen)
    assert not torch.allclose(out[0], out[1])        # independent per volume

def test_geometric_mask_follows_image():
    # a mask blob and an image blob at the same voxels move together
    D = 10
    vols = torch.zeros(2, 1, D, D, D); vols[:, 0, 2:5, 2:5, 2:5] = 1.0
    masks = torch.zeros(2, D, D, D, dtype=torch.long); masks[:, 2:5, 2:5, 2:5] = 1
    gen = torch.Generator().manual_seed(2)
    out, om = _geometric(vols, masks, group_size=2, cfg=_geo_cfg(), gen=gen)
    assert om.dtype == torch.long
    # where the mask is 1, the image is high (they co-moved)
    m = om[0] == 1
    assert m.sum() > 0 and out[0, 0][m].mean() > 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_augment.py -k geometric -v`
Expected: FAIL (`_geometric` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/gpu_augment.py
def _rand(gen, device, *shape):
    return torch.rand(*shape, generator=gen, device=device)

def _uniform(gen, device, lo, hi):
    return (lo + (hi - lo) * torch.rand((), generator=gen, device=device)).item()

def _geometric(vols, masks, group_size, cfg, gen):
    """Shared (group_size=T) or independent (group_size=1) flip/affine/elastic."""
    N = vols.shape[0]
    device = vols.device
    G = N // group_size                              # number of groups
    D, H, W = vols.shape[-3:]

    # --- flips: one decision per group, per axis ---
    fp = cfg.flip
    for g in range(G):
        sl = slice(g * group_size, (g + 1) * group_size)
        for vol_dim, mask_dim, p in [(2, 1, fp.p_d), (3, 2, fp.p_h), (4, 3, fp.p_w)]:
            if _rand(gen, device, 1).item() < p:
                vols[sl] = vols[sl].flip(vol_dim)
                masks[sl] = masks[sl].flip(mask_dim)

    # --- affine: one theta per group (built with the existing helper) ---
    ac = cfg.affine
    thetas = []
    for g in range(G):
        if _rand(gen, device, 1).item() < ac.p:
            mr = ac.max_angle_deg * math.pi / 180.0
            rx, ry, rz = (_uniform(gen, device, -mr, mr) for _ in range(3))
            scale = _uniform(gen, device, ac.scale_min, ac.scale_max)
            tx, ty, tz = (_uniform(gen, device, -ac.max_translate, ac.max_translate)
                          for _ in range(3))
            thetas.append(_make_affine_theta(rx, ry, rz, scale, tx, ty, tz)[0])
        else:
            thetas.append(torch.eye(3, 4))
    theta = torch.stack(thetas).to(device)                       # (G,3,4)
    theta = theta.repeat_interleave(group_size, dim=0)           # (N,3,4)
    grid = F.affine_grid(theta, vols.shape, align_corners=False)  # (N,D,H,W,3)

    # --- elastic: one coarse displacement field per group, added to grid ---
    ec = getattr(cfg, "elastic", None)
    if ec is not None and ec.p > 0:
        gs = max(int(getattr(ec, "grid_scale", 8)), 2)
        sd, sh, sw = max(D // gs, 2), max(H // gs, 2), max(W // gs, 2)
        for g in range(G):
            if _rand(gen, device, 1).item() < ec.p:
                disp = torch.randn(1, 3, sd, sh, sw, generator=gen, device=device) * ec.alpha
                disp = F.interpolate(disp, size=(D, H, W), mode="trilinear",
                                     align_corners=False).permute(0, 2, 3, 4, 1)  # (1,D,H,W,3)
                sl = slice(g * group_size, (g + 1) * group_size)
                grid[sl] = (grid[sl] + disp).clamp(-1.0, 1.0)

    vols = F.grid_sample(vols, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)
    m = F.grid_sample(masks.unsqueeze(1).float(), grid, mode="nearest",
                      padding_mode="zeros", align_corners=False)
    return vols, m.squeeze(1).long()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_augment.py -k geometric -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpu_augment.py tests/test_gpu_augment.py
git commit -m "feat(gpu_augment): batched shared/independent geometric aug"
```

---

### Task 3: Batched per-volume intensity

**Files:**
- Modify: `src/gpu_augment.py`
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Produces: `_batched_intensity(vols, cfg, gen) -> vols`. `vols` is `(N,1,D,H,W)`. Applies, per volume independently and each gated by its own probability: brightness/contrast, gamma, gaussian noise, gaussian blur, sharpness (optional), simulate-low-resolution (optional). Reads keys via `getattr` like the CPU code so both `aug_cfg.intensity` and `aug_cfg.synth` cfgs work. GIN/IPA is added in Task 4. Output clamped to `[CT_NORM_MIN, CT_NORM_MAX]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_gpu_augment.py
from src.gpu_augment import _batched_intensity
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX

def _int_cfg():
    return SimpleNamespace(
        brightness_contrast=SimpleNamespace(p=1.0, brightness=0.1, contrast_range=[0.8, 1.2]),
        gamma=SimpleNamespace(p=1.0, range=[0.8, 1.3]),
        gaussian_noise=SimpleNamespace(p=1.0, max_std=0.1),
        gaussian_blur=SimpleNamespace(p=1.0, sigma_range=[0.5, 1.0]),
    )

def test_intensity_shape_range_and_changes():
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = torch.rand(5, 1, 8, 8, 8) * span + CT_NORM_MIN
    gen = torch.Generator().manual_seed(3)
    out = _batched_intensity(vols.clone(), _int_cfg(), gen)
    assert out.shape == vols.shape
    assert out.min() >= CT_NORM_MIN - 1e-4 and out.max() <= CT_NORM_MAX + 1e-4
    assert not torch.allclose(out, vols)

def test_intensity_p_zero_is_noop():
    cfg = _int_cfg()
    for k in ("brightness_contrast", "gamma", "gaussian_noise", "gaussian_blur"):
        getattr(cfg, k).p = 0.0
    vols = torch.rand(3, 1, 8, 8, 8)
    gen = torch.Generator().manual_seed(4)
    out = _batched_intensity(vols.clone(), cfg, gen)
    assert torch.allclose(out, vols)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_augment.py -k intensity -v`
Expected: FAIL (`_batched_intensity` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/gpu_augment.py
def _per_vol_mask(gen, N, device, p):
    """(N,1,1,1,1) bool: which volumes this op fires on."""
    return (torch.rand(N, generator=gen, device=device) < p).view(N, 1, 1, 1, 1)

def _grouped_gaussian_blur(vols, sigma):
    """Separable 3D gaussian blur, same sigma for all volumes in `vols` (N,1,D,H,W)."""
    radius = max(1, int(math.ceil(2.0 * sigma)))
    size = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, dtype=vols.dtype, device=vols.device)
    k = torch.exp(-0.5 * (coords / sigma) ** 2); k = k / k.sum()
    x = vols
    for dim, view in ((2, (1, 1, size, 1, 1)), (3, (1, 1, 1, size, 1)), (4, (1, 1, 1, 1, size))):
        pad = [0, 0, 0]; pad[dim - 2] = radius
        x = F.conv3d(x, k.view(view), padding=(pad[0], pad[1], pad[2]))
    return x

def _batched_intensity(vols, cfg, gen):
    N = vols.shape[0]
    device = vols.device
    span = CT_NORM_MAX - CT_NORM_MIN

    bc = getattr(cfg, "brightness_contrast", None)
    if bc is not None and bc.p > 0:
        mask = _per_vol_mask(gen, N, device, bc.p)
        bright = (-bc.brightness + 2 * bc.brightness *
                  torch.rand(N, 1, 1, 1, 1, generator=gen, device=device))
        c0, c1 = bc.contrast_range
        contrast = c0 + (c1 - c0) * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
        aug = (vols * contrast + bright).clamp(CT_NORM_MIN, CT_NORM_MAX)
        vols = torch.where(mask, aug, vols)

    gc = getattr(cfg, "gamma", None)
    if gc is not None and gc.p > 0:
        mask = _per_vol_mask(gen, N, device, gc.p)
        g0, g1 = gc.range
        gamma = g0 + (g1 - g0) * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
        norm = ((vols - CT_NORM_MIN) / span).clamp(0, 1).pow(gamma)
        aug = norm * span + CT_NORM_MIN
        vols = torch.where(mask, aug, vols)

    sc = getattr(cfg, "sharpness", None)
    if sc is not None and getattr(sc, "p", 0) > 0:
        mask = _per_vol_mask(gen, N, device, sc.p)
        blur = _grouped_gaussian_blur(vols, sigma=1.0)
        aug = (vols + sc.factor * (vols - blur)).clamp(CT_NORM_MIN, CT_NORM_MAX)
        vols = torch.where(mask, aug, vols)

    bl = getattr(cfg, "gaussian_blur", None)
    if bl is not None and bl.p > 0:
        mask = _per_vol_mask(gen, N, device, bl.p)
        s0, s1 = bl.sigma_range
        sigma = _uniform(gen, device, s0, s1)
        aug = _grouped_gaussian_blur(vols, sigma)
        vols = torch.where(mask, aug, vols)

    nc = getattr(cfg, "gaussian_noise", None)
    if nc is not None and nc.p > 0:
        mask = _per_vol_mask(gen, N, device, nc.p)
        if hasattr(nc, "max_std"):
            std = _uniform(gen, device, 0.0, nc.max_std); mean = 0.0
        else:                                                   # synth schema
            mean = _uniform(gen, device, *nc.mean_range)
            std = _uniform(gen, device, *nc.std_range)
        noise = mean + std * torch.randn(vols.shape, generator=gen, device=device)
        aug = (vols + noise).clamp(CT_NORM_MIN, CT_NORM_MAX)
        vols = torch.where(mask, aug, vols)

    lr = getattr(cfg, "simulate_low_resolution", None)
    if lr is not None and lr.p > 0:
        mask = _per_vol_mask(gen, N, device, lr.p)
        D, H, W = vols.shape[-3:]
        scale = _uniform(gen, device, lr.scale_min, lr.scale_max)
        small = (max(1, int(D * scale)), max(1, int(H * scale)), max(1, int(W * scale)))
        down = F.interpolate(vols, size=small, mode="trilinear", align_corners=False)
        aug = F.interpolate(down, size=(D, H, W), mode="trilinear", align_corners=False)
        vols = torch.where(mask, aug, vols)

    return vols
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_augment.py -k intensity -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpu_augment.py tests/test_gpu_augment.py
git commit -m "feat(gpu_augment): batched per-volume intensity aug"
```

---

### Task 4: Batched GIN / IPA (grouped conv)

**Files:**
- Modify: `src/gpu_augment.py`
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Produces: `_batched_gin_ipa(vols, cfg, gen) -> vols`. `cfg` is the `gin` section (`p,mode,n_layer,interm_channel,scale_pool,out_norm,ipa_copies,ipa_control_points`). Each volume gets an independent GIN warp via grouped conv (`groups=N`, fresh random kernels); `mode="ipa"` blends `ipa_copies` warps with a per-volume smooth field. Output clamped. Called from `_batched_intensity` when `cfg.gin.p>0` (wire-in below).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_gpu_augment.py
from src.gpu_augment import _batched_gin_ipa

def _gin_cfg(mode="ipa"):
    return SimpleNamespace(p=1.0, mode=mode, n_layer=4, interm_channel=2,
                           scale_pool=[1, 3], out_norm="frob",
                           ipa_copies=2, ipa_control_points=3)

def test_gin_ipa_shape_range_changes():
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = torch.rand(4, 1, 8, 8, 8) * span + CT_NORM_MIN
    gen = torch.Generator().manual_seed(5)
    for mode in ("gin", "ipa"):
        out = _batched_gin_ipa(vols.clone(), _gin_cfg(mode), gen)
        assert out.shape == vols.shape
        assert out.min() >= CT_NORM_MIN - 1e-4 and out.max() <= CT_NORM_MAX + 1e-4
        assert not torch.allclose(out, vols)

def test_intensity_invokes_gin_when_configured():
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = torch.rand(3, 1, 8, 8, 8) * span + CT_NORM_MIN
    cfg = _int_cfg()
    for k in ("brightness_contrast", "gamma", "gaussian_noise", "gaussian_blur"):
        getattr(cfg, k).p = 0.0
    cfg.gin = _gin_cfg("gin")
    gen = torch.Generator().manual_seed(6)
    out = _batched_intensity(vols.clone(), cfg, gen)
    assert not torch.allclose(out, vols)             # gin fired even with others off
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_augment.py -k gin -v`
Expected: FAIL (`_batched_gin_ipa` not defined / not wired).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/gpu_augment.py
def _gin_once(vols, cfg, gen):
    """One GIN warp per volume via grouped conv (groups=N, fresh random kernels)."""
    N = vols.shape[0]
    device, dtype = vols.device, vols.dtype
    scale_pool = list(getattr(cfg, "scale_pool", (1, 3)))
    n_layer = int(getattr(cfg, "n_layer", 4))
    interm = int(getattr(cfg, "interm_channel", 2))
    x = vols                                          # (N,1,D,H,W)
    ch_in = 1
    for li in range(n_layer):
        out_ch = 1 if li == n_layer - 1 else interm
        k = scale_pool[torch.randint(len(scale_pool), (1,), generator=gen).item()]
        ker = torch.randn(N * out_ch, ch_in, k, k, k, generator=gen, device=device, dtype=dtype)
        shift = torch.randn(N * out_ch, 1, 1, 1, generator=gen, device=device, dtype=dtype)
        xg = x.reshape(1, N * ch_in, *x.shape[-3:])
        xg = F.conv3d(xg, ker, padding=k // 2, groups=N) + shift.reshape(1, N * out_ch, 1, 1, 1)
        x = xg.reshape(N, out_ch, *x.shape[-3:])
        if li < n_layer - 1:
            x = F.leaky_relu(x)
        ch_in = out_ch
    alpha = torch.rand(N, 1, 1, 1, 1, generator=gen, device=device, dtype=dtype)
    mixed = alpha * x + (1.0 - alpha) * vols
    if getattr(cfg, "out_norm", "frob") == "frob":
        in_f = torch.norm(vols.reshape(N, -1), dim=1).view(N, 1, 1, 1, 1)
        self_f = torch.norm(mixed.reshape(N, -1), dim=1).view(N, 1, 1, 1, 1)
        mixed = mixed * (in_f / (self_f + 1e-5))
    return mixed

def _batched_gin_ipa(vols, cfg, gen):
    if getattr(cfg, "mode", "gin") != "ipa":
        out = _gin_once(vols, cfg, gen)
    else:
        copies = [_gin_once(vols, cfg, gen) for _ in range(max(2, int(getattr(cfg, "ipa_copies", 2))))]
        N, _, D, H, W = vols.shape
        cp = max(2, int(getattr(cfg, "ipa_control_points", 4)))
        field = torch.randn(N, 1, cp, cp, cp, generator=gen, device=vols.device, dtype=vols.dtype)
        field = F.interpolate(field, size=(D, H, W), mode="trilinear", align_corners=False)
        fmin = field.amin(dim=(2, 3, 4), keepdim=True)
        fmax = field.amax(dim=(2, 3, 4), keepdim=True)
        field = (field - fmin) / (fmax - fmin + 1e-20)
        out = copies[0]
        for c in copies[1:]:
            out = out * (1.0 - field) + c * field
    return out.clamp(CT_NORM_MIN, CT_NORM_MAX)
```

Then wire GIN into `_batched_intensity` — add at the END of that function, before `return vols`:

```python
    gin = getattr(cfg, "gin", None)
    if gin is not None and getattr(gin, "p", 0) > 0:
        mask = _per_vol_mask(gen, N, device, gin.p)
        aug = _batched_gin_ipa(vols, gin, gen)
        vols = torch.where(mask, aug, vols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_augment.py -k gin -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpu_augment.py tests/test_gpu_augment.py
git commit -m "feat(gpu_augment): batched GIN/IPA via grouped conv"
```

---

### Task 5: GpuAugmentor with mode dispatch

**Files:**
- Modify: `src/gpu_augment.py`
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Consumes: `_stack_task`, `_unstack_task`, `_geometric`, `_batched_intensity`.
- Produces: `class GpuAugmentor:` with `__init__(self, aug_cfg, self_context_per_image: bool = False, seed: int = 0)` and `__call__(self, batch: dict, training: bool) -> dict`. When `training=False` or `aug_cfg` is None / `not aug_cfg.enabled`, returns `batch` unchanged. Otherwise dispatches per `aug_mode` on masked task-subsets:
  - `real` & `self_context`: shared geometric (`aug_cfg.task`, group_size=T) → per-volume intensity (`aug_cfg.intensity`) on all; for `self_context` if `self_context_per_image`, an extra independent per-volume geometric (`aug_cfg.per_image`, group_size=1) on the K context volumes.
  - `synth`: independent per-volume geometric (`aug_cfg.synth`, group_size=1) → intensity (`aug_cfg.synth`) on all.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_gpu_augment.py
from src.gpu_augment import GpuAugmentor

def _full_cfg():
    return SimpleNamespace(
        enabled=True,
        task=_geo_cfg(affine_p=1.0),
        per_image=_geo_cfg(affine_p=1.0),
        synth=SimpleNamespace(**vars(_geo_cfg(affine_p=1.0)), **{
            "brightness_contrast": SimpleNamespace(p=1.0, brightness=0.1, contrast_range=[0.8, 1.2]),
            "gamma": SimpleNamespace(p=1.0, range=[0.8, 1.2]),
            "gaussian_noise": SimpleNamespace(p=1.0, mean_range=[0.0, 0.05], std_range=[0.0, 0.05]),
            "gaussian_blur": SimpleNamespace(p=0.0, sigma_range=[0.5, 1.0]),
        }),
        intensity=_int_cfg(),
    )

def test_eval_is_identity():
    b = _fake_batch()
    ref = {k: v.clone() for k, v in b.items()}
    aug = GpuAugmentor(_full_cfg())
    out = aug(b, training=False)
    for k in ("image", "context_in"):
        assert torch.allclose(out[k], ref[k])

def test_real_mode_shares_geometry_across_task():
    # target and its contexts identical -> shared geo keeps their geometry aligned
    B, K, D = 1, 3, 8
    base = torch.randn(1, 1, D, D, D)
    b = {
        "image": base.clone(),
        "label": torch.zeros(B, D, D, D, dtype=torch.long),
        "context_in": base.view(1, 1, 1, D, D, D).repeat(1, K, 1, 1, 1, 1),
        "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
        "aug_mode": torch.tensor([0]),
    }
    cfg = _full_cfg()
    for kk in ("brightness_contrast", "gamma", "gaussian_noise", "gaussian_blur"):
        getattr(cfg.intensity, kk).p = 0.0            # isolate geometry
    aug = GpuAugmentor(cfg)
    out = aug(b, training=True)
    # target and each context underwent the SAME geometric transform
    for k in range(K):
        assert torch.allclose(out["image"][0, 0], out["context_in"][0, k, 0], atol=1e-5)

def test_mixed_modes_route_and_preserve_shape():
    B, K, D = 3, 2, 8
    b = {
        "image": torch.rand(B, 1, D, D, D),
        "label": torch.zeros(B, D, D, D, dtype=torch.long),
        "context_in": torch.rand(B, K, 1, D, D, D),
        "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
        "aug_mode": torch.tensor([0, 1, 2]),          # real, synth, self_context
    }
    aug = GpuAugmentor(_full_cfg(), self_context_per_image=True)
    out = aug(b, training=True)
    assert out["image"].shape == (B, 1, D, D, D)
    assert out["context_in"].shape == (B, K, 1, D, D, D)
    assert out["context_out"].dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_augment.py -k "eval_is_identity or real_mode or mixed_modes" -v`
Expected: FAIL (`GpuAugmentor` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/gpu_augment.py
class GpuAugmentor:
    def __init__(self, aug_cfg, self_context_per_image: bool = False, seed: int = 0):
        self.cfg = aug_cfg
        self.self_context_per_image = bool(self_context_per_image)
        self._seed = seed
        self._step = 0

    @torch.no_grad()
    def __call__(self, batch: dict, training: bool) -> dict:
        cfg = self.cfg
        if (not training) or cfg is None or not getattr(cfg, "enabled", False):
            return batch
        vols, masks, B, T = _stack_task(batch)
        device = vols.device
        gen = torch.Generator(device=device)
        gen.manual_seed(self._seed + self._step)
        self._step += 1

        modes = batch.get("aug_mode", torch.zeros(B, dtype=torch.long, device=device))
        modes = modes.to(device)
        for mode in (REAL, SYNTH, SELF_CONTEXT):
            task_sel = (modes == mode).nonzero(as_tuple=True)[0]
            if task_sel.numel() == 0:
                continue
            # volume indices for the selected tasks
            vidx = (task_sel.view(-1, 1) * T + torch.arange(T, device=device)).reshape(-1)
            v, m = vols[vidx], masks[vidx]

            if mode == SYNTH:
                v, m = _geometric(v, m, group_size=1, cfg=cfg.synth, gen=gen)
                v = _batched_intensity(v, cfg.synth, gen)
            else:  # REAL or SELF_CONTEXT
                v, m = _geometric(v, m, group_size=T, cfg=cfg.task, gen=gen)
                v = _batched_intensity(v, cfg.intensity, gen)
                if mode == SELF_CONTEXT and self.self_context_per_image:
                    # extra independent geo jitter on the K context volumes only
                    G = task_sel.numel()
                    ctx_local = torch.tensor([g * T + t for g in range(G) for t in range(1, T)],
                                             device=device)
                    cv, cm = _geometric(v[ctx_local], m[ctx_local],
                                        group_size=1, cfg=cfg.per_image, gen=gen)
                    v[ctx_local], m[ctx_local] = cv, cm

            vols[vidx], masks[vidx] = v, m

        _unstack_task(vols, masks, B, T, batch)
        return batch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_augment.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpu_augment.py tests/test_gpu_augment.py
git commit -m "feat(gpu_augment): GpuAugmentor mode dispatch"
```

---

### Task 6: Dataset defer-to-GPU + collate aug_mode

**Files:**
- Modify: `src/totalseg_dataloader_incontext.py` (aug call sites ~1082-1089, ~1202-1265; return dicts ~1094, ~1267; `incontext_collate_fn` ~1497)
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Consumes: `GpuAugmentor` mode codes (`0 real, 1 synth, 2 self_context`).
- Produces:
  - `TotalSegInContextDataset.__init__` gains `defer_aug_to_gpu: bool = False` (stored as `self.defer_aug`).
  - Each item dict gains `item["aug_mode"]` (a 0-dim `torch.long` tensor).
  - When `self.defer_aug` is True, `__getitem__` and `_get_synth_item` skip the `apply_*_aug` calls but keep task construction (context selection, self-context cloning, label synthesis).
  - `incontext_collate_fn` stacks `aug_mode` into `(B,)` when present.

- [ ] **Step 1: Write the failing test (collate)**

```python
# add to tests/test_gpu_augment.py
import torch
from src.totalseg_dataloader_incontext import incontext_collate_fn

def _item(mode, K=2, D=6):
    return {
        "image": torch.randn(1, D, D, D),
        "label": torch.zeros(D, D, D, dtype=torch.long),
        "context_in": torch.randn(K, 1, D, D, D),
        "context_out": torch.zeros(K, D, D, D, dtype=torch.long),
        "subject": "s0", "label_name": "x",
        "spacing": torch.ones(3),
        "context_subjects": ["s1", "s2"],
        "aug_mode": torch.tensor(mode, dtype=torch.long),
    }

def test_collate_stacks_aug_mode():
    out = incontext_collate_fn([_item(0), _item(2)])
    assert "aug_mode" in out
    assert out["aug_mode"].tolist() == [0, 2]
    assert out["aug_mode"].dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_augment.py::test_collate_stacks_aug_mode -v`
Expected: FAIL (`aug_mode` not in collate output → KeyError).

- [ ] **Step 3: Implement**

In `incontext_collate_fn` (after the `crop_geom` block, before the `synth_radii_mm` block) add:

```python
    if "aug_mode" in batch[0]:
        out["aug_mode"] = torch.stack([b["aug_mode"] for b in batch])  # (B,) int64
```

In `TotalSegInContextDataset.__init__` signature add `defer_aug_to_gpu: bool = False,` and in the body add `self.defer_aug = bool(defer_aug_to_gpu)`.

In `_get_synth_item` — gate the synth aug (the block at ~1082-1089). Replace:

```python
        if self.aug_cfg is not None and self.aug_cfg.enabled:
            items = [
                apply_synth_aug(image_t.clone(), mask_t.clone(), self.aug_cfg.synth)
                for _ in range(self.context_size + 1)
            ]
        else:
            items = [(image_t.clone(), mask_t.clone()) for _ in range(self.context_size + 1)]
```

with:

```python
        if self.aug_cfg is not None and self.aug_cfg.enabled and not self.defer_aug:
            items = [
                apply_synth_aug(image_t.clone(), mask_t.clone(), self.aug_cfg.synth)
                for _ in range(self.context_size + 1)
            ]
        else:                                   # defer: emit K+1 RAW clones
            items = [(image_t.clone(), mask_t.clone()) for _ in range(self.context_size + 1)]
```

and in that item dict (~1094) add `"aug_mode": torch.tensor(1, dtype=torch.long),  # synth`.

In `__getitem__` — gate the shared aug (block ~1202-1212). Wrap the existing
`if self.aug_cfg is not None and self.aug_cfg.enabled and len(context_in) > 0:`
condition to also require `and not self.defer_aug`.

Gate the self-context re-aug (block ~1255): change
`if do_augs and self.aug_cfg is not None and self.aug_cfg.enabled:` to also require
`and not self.defer_aug` (the CLONING at ~1251-1253 stays — only the per-clone aug defers).

In the main-path item dict (~1267) add:

```python
        item["aug_mode"] = torch.tensor(
            2 if (self.self_context_p > 0 and self._sc_fired) else 0, dtype=torch.long)
```

To know whether self-context fired, set a local flag: at the `if self.self_context_p > 0 and self._cur_rng.random() < self.self_context_p:` line (~1222) capture the boolean into `self._sc_fired` first:

```python
        self._sc_fired = (self.self_context_p > 0
                          and self._cur_rng.random() < self.self_context_p)
        synth_coord = synth_radii = None
        if self._sc_fired:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gpu_augment.py::test_collate_stacks_aug_mode -v`
Expected: PASS

Also run the existing dataloader-touching tests to confirm no regression:
Run: `python -m pytest tests/test_synth_ellipsoid.py -q`
Expected: PASS (unchanged — `defer_aug_to_gpu` defaults False).

- [ ] **Step 5: Commit**

```bash
git add src/totalseg_dataloader_incontext.py tests/test_gpu_augment.py
git commit -m "feat(dataloader): defer_aug_to_gpu emits raw volumes + aug_mode"
```

---

### Task 7: Config flag + common.py wiring

**Files:**
- Modify: `configs/augmentations/nnunet.yaml`
- Modify: `experiments/3d/common.py` (`build_dataset`, ~186-237)
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Consumes: dataset `defer_aug_to_gpu` param (Task 6).
- Produces: `cfg.augmentations.gpu` (bool, default false); `build_dataset(cfg, "train")` passes `defer_aug_to_gpu=bool(cfg.augmentations.get("gpu", False))` to the train `TotalSegInContextDataset`; eval/val builds pass `False` (they already pass `aug_cfg=None`).

- [ ] **Step 1: Add the config key**

In `configs/augmentations/nnunet.yaml`, under the top-level `augmentations:` (next to `enabled: true`):

```yaml
  # Run augmentation on GPU in the training loop (src/gpu_augment.GpuAugmentor)
  # instead of per-item on CPU workers. When true the dataset emits raw volumes
  # + aug_mode and skips apply_*_aug. Default false (CPU path).
  gpu: false
```

- [ ] **Step 2: Write the failing test**

```python
# add to tests/test_gpu_augment.py
from omegaconf import OmegaConf

def test_nnunet_config_has_gpu_flag():
    cfg = OmegaConf.load("configs/augmentations/nnunet.yaml")
    assert cfg.augmentations.gpu is False
```

Run: `python -m pytest tests/test_gpu_augment.py::test_nnunet_config_has_gpu_flag -v`
Expected: PASS (config edited in Step 1). If FAIL, fix the yaml.

- [ ] **Step 3: Wire common.py**

In `experiments/3d/common.py`, in `build_dataset`, at the train-dataset construction
(the `return TotalSegInContextDataset(...)` around line 214 and the multi-source
branch around line 201), add the argument:

```python
        defer_aug_to_gpu=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
```

Pass it to every `TotalSegInContextDataset(...)` / subclass construction that receives
`aug_cfg` in the train path. (Eval loader `build_eval_dataset` passes `aug_cfg=None`;
leave it — the augmentor is train-only and won't run there.)

- [ ] **Step 4: Verify import + smoke**

Run: `python -c "import experiments.3d.common" 2>/dev/null || python -c "import importlib,sys; importlib.import_module('experiments.3d.common')"`

If the package path is awkward, instead syntax-check:
Run: `python -m py_compile experiments/3d/common.py`
Expected: no output (compiles).

- [ ] **Step 5: Commit**

```bash
git add configs/augmentations/nnunet.yaml experiments/3d/common.py tests/test_gpu_augment.py
git commit -m "feat(config): augmentations.gpu flag wired to defer_aug_to_gpu"
```

---

### Task 8: Training-loop integration + end-to-end smoke

**Files:**
- Modify: `experiments/3d/train.py` (`train_epoch` ~275-323; model construction ~497-501)
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Consumes: `GpuAugmentor` (Task 5), `cfg.augmentations.gpu`.
- Produces: a `GpuAugmentor` built once when `cfg.augmentations.gpu` is true and passed into `train_epoch`; the batch is moved to device once and augmented before the forward. When the flag is false, behaviour is unchanged.

- [ ] **Step 1: Write the failing test (augmentor end-to-end on a fake batch)**

```python
# add to tests/test_gpu_augment.py
def test_augmentor_end_to_end_batch_smoke():
    # emulate the train-loop call: raw batch -> to(device) -> augmentor -> shapes intact
    B, K, D = 2, 3, 8
    b = {
        "image": torch.rand(B, 1, D, D, D),
        "label": torch.zeros(B, D, D, D, dtype=torch.long),
        "context_in": torch.rand(B, K, 1, D, D, D),
        "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
        "aug_mode": torch.tensor([2, 2]),
        "spacing": torch.ones(B, 3),
    }
    aug = GpuAugmentor(_full_cfg(), self_context_per_image=True)
    out = aug(b, training=True)
    assert out["image"].shape == (B, 1, D, D, D)
    assert out["context_in"].shape == (B, K, 1, D, D, D)
    assert torch.isfinite(out["image"]).all()
    assert torch.isfinite(out["context_in"]).all()
```

Run: `python -m pytest tests/test_gpu_augment.py::test_augmentor_end_to_end_batch_smoke -v`
Expected: PASS (uses only Task-5 code; this locks the train-loop contract).

- [ ] **Step 2: Integrate into `train.py`**

At model/loop setup (where `train_epoch` is called, near line 497-501), build the augmentor once:

```python
    from src.gpu_augment import GpuAugmentor
    gpu_aug = (GpuAugmentor(cfg.augmentations,
                            self_context_per_image=bool(
                                cfg.data.get("self_context", {}).get("augs", {}).get("per_image", False)),
                            seed=int(cfg.get("seed", 0)))
               if cfg.augmentations.get("gpu", False) else None)
```

Pass `gpu_aug` into `train_epoch(...)` (add a `gpu_aug=None` kwarg to its signature).

In `train_epoch`, replace the per-tensor `.to(DEVICE)` at the forward site (lines
304, 314-316) with a single move + augment at the top of the loop body:

```python
    for batch in pbar:
        if prof:
            tsum["data"] += (time.perf_counter() - t_prev) * 1000
            prof_items += batch["image"].shape[0]
        if gpu_aug is not None:
            for k in ("image", "label", "context_in", "context_out", "aug_mode", "spacing"):
                if k in batch:
                    batch[k] = batch[k].to(DEVICE, non_blocking=True)
            batch = gpu_aug(batch, training=True)
            lbl = batch["label"].float()
        else:
            lbl = batch["label"].to(DEVICE, non_blocking=True).float()
```

and at the model call, use the already-moved tensors when `gpu_aug` is set:

```python
                img_in = batch["image"] if gpu_aug is not None else batch["image"].to(DEVICE, non_blocking=True)
                cin = batch["context_in"] if gpu_aug is not None else batch["context_in"].to(DEVICE, non_blocking=True)
                cout = batch["context_out"] if gpu_aug is not None else batch["context_out"].to(DEVICE, non_blocking=True)
                out = model(img_in, context_in=cin, context_out=cout, mode="train", spacing=spacing)
```

(Keep the non-patchset branch as-is; it already reads `batch[...]` directly.)

- [ ] **Step 3: Syntax-check train.py**

Run: `python -m py_compile experiments/3d/train.py`
Expected: no output.

- [ ] **Step 4: Run the full new-module test suite**

Run: `python -m pytest tests/test_gpu_augment.py -v`
Expected: PASS (all)

- [ ] **Step 5: Log + commit**

Add to `docs/logs.md` (top, under `# Change log`):

```markdown
## 2026-08-15 — GPU augmentation pipeline (batched, replaces CPU per-item augs)
- NEW `src/gpu_augment.py::GpuAugmentor` — batched on-device aug run in the train loop after batch.to(device), before model(). Own torch ops (no deps), non-differentiable. `_geometric` (shared per task group_size=T, or independent group_size=1), `_batched_intensity` (brightness/contrast/gamma/noise/blur/sharpness/low-res), `_batched_gin_ipa` (grouped conv, groups=N). Mode dispatch on `aug_mode` (0 real, 1 synth, 2 self_context).
- Dataset `defer_aug_to_gpu` (from `augmentations.gpu`): __getitem__/_get_synth_item skip apply_*_aug, emit RAW volumes + aug_mode; collate stacks aug_mode. train.py moves batch to device once + augments. Behind `augmentations.gpu` (default false → CPU path unchanged). Exact CPU repro is a non-goal; tests assert shape/range/K+1-sharing/eval-identity. See docs/superpowers/specs/2026-08-15-gpu-augmentation-pipeline-design.md.
```

```bash
git add experiments/3d/train.py docs/logs.md tests/test_gpu_augment.py
git commit -m "feat(train): run GpuAugmentor in the training loop behind augmentations.gpu"
```

---

## Self-Review

**Spec coverage:**
- Dataset separation (raw emit + aug_mode) → Task 6. ✓
- `GpuAugmentor` interface + placement → Tasks 5, 8. ✓
- Batched geometric (shared K+1) → Task 2. ✓
- Batched intensity + GIN/IPA grouped conv → Tasks 3, 4. ✓
- Mode dispatch (real/synth/self_context) → Task 5. ✓
- Config `augmentations.gpu` + common wiring → Task 7. ✓
- Eval identity → Task 5 test. ✓
- Testing (distributional/shape/mask/K+1/eval) → Tasks 2,5,6,8. ✓
- Rollout behind flag → Task 7/8. ✓
- Throughput microbench: NOTE — the existing `experiments/3d/bench_cpu_aug.py` measures the CPU path; a GPU counterpart is optional follow-up (not a code requirement here). Left out of tasks intentionally; add if desired.

**Placeholder scan:** No TBD/TODO; all code steps carry real code. ✓

**Type consistency:** `_stack_task` returns `(vols, masks, B, T)` used consistently in Task 5; `_geometric(vols, masks, group_size, cfg, gen)` signature matches all call sites; `_batched_intensity(vols, cfg, gen)` and `_batched_gin_ipa(vols, cfg, gen)` consistent; `aug_mode` is a 0-dim long tensor per item, stacked to `(B,)` — consistent across Tasks 5/6/8. ✓

**Known simplification (documented, per non-goal):** `self_context` reuses the shared-geo + per-volume-intensity path (contexts diverge from target via independent intensity + optional per_image jitter) rather than bit-reproducing the CPU clone-then-reaugment order. Intentional.

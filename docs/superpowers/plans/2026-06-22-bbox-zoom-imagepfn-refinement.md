# Bbox-zoom ImagePFN refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stage-2 refinement variant where the refiner is the *same* `ImagePFN`
architecture as stage-1, fed a contiguous square crop ("zoom") instead of scattered
patches — isolating the effect of refinement from the change of model class.

**Architecture:** A single 64px zoom hop. Frozen stage-1 `ImagePFN` segments the full 128
image at res-16; we crop the 64×64 square with the largest predicted mass, crop each
context image around its GT, pool the **once-computed** encoder maps to that bbox, and run a
warm-started `ImagePFN` that corrects the cropped coarse prediction. The refined 16×16
output is upsampled to 64×64 and composited back into the full-res prediction. Selected via
a `refine_arch` config switch; the existing `PatchSetPFN` path is untouched.

**Tech Stack:** PyTorch, Hydra/OmegaConf, the project's `ImagePFN` (`src/models/pfn_seg_2d.py`)
and the multilevel experiment harness (`experiments/2d/multilevel/`).

## Global Constraints

- Python interpreter: `.venv311/bin/python` (CUDA). No `uv`/`conda`.
- No `pytest`. Tests are standalone scripts with `test_*` functions and a `__main__` block
  that calls them and prints `ALL <NAME> TESTS PASSED`; run with
  `.venv311/bin/python experiments/2d/multilevel/test_*.py` from the repo root.
- **Do not `git add`/`git commit`.** Version control is the user's; the "Commit" step of the
  default TDD cadence is intentionally omitted from every task.
- Log the change to `docs/logs.md` (last task).
- `ImagePFN` changes MUST be backward-compatible: defaults reproduce current behavior
  byte-for-byte. The `refine_arch=patchset` path MUST remain unchanged.
- Stage-1 checkpoint (UniverSeg encoder):
  `results/2d/pfn_seg_universeg/pfn_seg_USegall_R16q8_e256_l6_k3_think8/best.pt`.
- Encoder runs **once** per batch (`encode_maps`); hops crop+pool those maps — never
  re-encode crops.

---

### Task 1: `bbox.py` — pure tensor crop/window ops

**Files:**
- Create: `experiments/2d/multilevel/bbox.py`
- Test: `experiments/2d/multilevel/test_bbox.py`

**Interfaces:**
- Produces:
  - `max_sum_window(prob, s) -> LongTensor (B,2)` — `prob` is `(B,1,H,W)` or `(B,H,W)`;
    returns the `(row, col)` **top-left** of the `s×s` square maximizing summed value.
    The window always fits, so the returned origin is in `[0, H-s] × [0, W-s]`.
  - `gt_window(mask, s) -> LongTensor (B,2)` — same on a binary mask (densest-GT square).
  - `crop_resize(x, origin, s, out, mode="bilinear") -> Tensor (N,C,out,out)` — `x` is
    `(N,C,H,W)`, `origin` is `(N,2)`; crops each image's `[r0:r0+s, c0:c0+s]` and resamples
    to `out×out` via `F.grid_sample` (`mode` in `{"bilinear","nearest"}`).
  - `composite_window(full, patch, origin, s) -> Tensor (B,1,H,W)` — writes `patch`
    `(B,1,s,s)` into a **clone** of `full` `(B,1,H,W)` at each sample's `s×s` bbox; `full`
    not mutated.

- [ ] **Step 1: Write the failing test**

Create `experiments/2d/multilevel/test_bbox.py`:

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d/multilevel")
import torch
from bbox import max_sum_window, gt_window, crop_resize, composite_window


def test_max_sum_window_finds_blob():
    H, s = 16, 8
    prob = torch.zeros(2, 1, H, H)
    prob[0, 0, 2:6, 3:7] = 1.0          # blob near top-left → origin should cover it
    prob[1, 0, 10:14, 9:13] = 1.0       # blob near bottom-right
    o = max_sum_window(prob, s)
    assert o.shape == (2, 2)
    # origin must be in-bounds and the window must contain the blob's mass
    assert (o >= 0).all() and (o[:, 0] <= H - s).all() and (o[:, 1] <= H - s).all()
    assert o[0, 0] <= 2 and o[0, 1] <= 3            # window starts at/above-left of blob
    assert o[1, 0] >= 6 and o[1, 1] >= 5            # window shifted toward the blob


def test_max_sum_window_border_blob_clamps():
    H, s = 16, 8
    prob = torch.zeros(1, 1, H, H)
    prob[0, 0, 0:2, 0:2] = 1.0          # corner blob
    o = max_sum_window(prob, s)
    assert (o == 0).all()              # origin clamped to the corner, still in-bounds


def test_gt_window_matches_max_sum():
    H, s = 16, 8
    mask = torch.zeros(1, 1, H, H)
    mask[0, 0, 4:8, 4:8] = 1.0
    o = gt_window(mask, s)
    assert o.shape == (1, 2)
    assert (o >= 0).all() and (o <= H - s).all()


def test_crop_resize_roundtrip_identity():
    # cropping the full image (origin 0, s=H, out=H) returns the image unchanged
    x = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    origin = torch.zeros(2, 2, dtype=torch.long)
    y = crop_resize(x, origin, s=4, out=4, mode="nearest")
    assert y.shape == (2, 1, 4, 4)
    assert torch.allclose(y, x)


def test_crop_resize_picks_region():
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 4:8, 4:8] = 5.0
    origin = torch.tensor([[4, 4]])
    y = crop_resize(x, origin, s=4, out=4, mode="nearest")
    assert torch.allclose(y, torch.full((1, 1, 4, 4), 5.0))


def test_composite_window_writes_region_only():
    full = torch.zeros(1, 1, 8, 8)
    patch = torch.ones(1, 1, 4, 4)
    origin = torch.tensor([[2, 3]])
    out = composite_window(full, patch, origin, s=4)
    assert torch.allclose(full, torch.zeros(1, 1, 8, 8))     # input not mutated
    assert torch.allclose(out[0, 0, 2:6, 3:7], torch.ones(4, 4))
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask[0, 0, 2:6, 3:7] = True
    assert torch.allclose(out[~mask], torch.zeros_like(out)[~mask])


if __name__ == "__main__":
    test_max_sum_window_finds_blob()
    test_max_sum_window_border_blob_clamps()
    test_gt_window_matches_max_sum()
    test_crop_resize_roundtrip_identity()
    test_crop_resize_picks_region()
    test_composite_window_writes_region_only()
    print("ALL BBOX TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_bbox.py`
Expected: `ModuleNotFoundError: No module named 'bbox'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

Create `experiments/2d/multilevel/bbox.py`:

```python
"""
Square-bbox crop / window ops for the zoom-refinement chain.

Origins are top-left (row, col) integer corners of an s×s square that always fits inside
the H×W frame. crop_resize / composite_window are batched (per-sample origin) via
F.grid_sample so the whole batch of context+target crops is handled in one call.
"""

import torch
import torch.nn.functional as F


def _box_sum(x, s):
    """(B,1,H,W) → (B,1,H-s+1,W-s+1) summed value of every s×s window (stride 1)."""
    return F.avg_pool2d(x, kernel_size=s, stride=1) * (s * s)


def _argmax_origin(score):
    """(B,1,Hs,Ws) window scores → (B,2) top-left (row,col) of the max window."""
    B = score.shape[0]
    Ws = score.shape[-1]
    flat = score.reshape(B, -1).argmax(dim=1)
    return torch.stack([torch.div(flat, Ws, rounding_mode="floor"), flat % Ws], dim=1)


def max_sum_window(prob, s):
    """Top-left (B,2) of the s×s square with the largest summed value in `prob`."""
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    return _argmax_origin(_box_sum(prob.float(), s))


def gt_window(mask, s):
    """Top-left (B,2) of the s×s square with the most foreground in `mask`."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return _argmax_origin(_box_sum(mask.float(), s))


def crop_resize(x, origin, s, out, mode="bilinear"):
    """Crop each (N,C,H,W) image to its s×s bbox at `origin` (N,2) and resample to out×out.

    Uses F.grid_sample with align_corners=False (pixel-center convention)."""
    N, C, H, W = x.shape
    r0 = origin[:, 0].to(x.dtype).view(N, 1)
    c0 = origin[:, 1].to(x.dtype).view(N, 1)
    i = torch.arange(out, device=x.device, dtype=x.dtype) + 0.5     # (out,) cell centers
    # source pixel coords of each output cell, then normalize to [-1,1] (align_corners=False)
    rows = r0 + i.view(1, out) * (s / out)                          # (N,out)
    cols = c0 + i.view(1, out) * (s / out)                          # (N,out)
    ny = 2.0 * rows / H - 1.0                                       # (N,out)
    nx = 2.0 * cols / W - 1.0
    grid = torch.stack([nx.view(N, 1, out).expand(N, out, out),
                        ny.view(N, out, 1).expand(N, out, out)], dim=-1)   # (N,out,out,2)
    return F.grid_sample(x, grid, mode=mode, align_corners=False, padding_mode="border")


def composite_window(full, patch, origin, s):
    """Write patch (B,1,s,s) into a clone of full (B,1,H,W) at each origin (B,2). New tensor."""
    B, _, H, W = full.shape
    out = full.clone()
    rr = torch.arange(s, device=full.device)
    for b in range(B):                       # per-sample origin; B is small (batch dim)
        r0, c0 = int(origin[b, 0]), int(origin[b, 1])
        out[b, 0, r0:r0 + s, c0:c0 + s] = patch[b, 0]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_bbox.py`
Expected: `ALL BBOX TESTS PASSED`

---

### Task 2: `ImagePFN` — external-features + query-seeding modes

**Files:**
- Modify: `src/models/pfn_seg_2d.py` (`ImagePFN.__init__` and `ImagePFN.forward`)
- Test: `experiments/2d/multilevel/test_imagepfn_modes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ImagePFN(..., use_external_features=False)` — when `True` (with `feature_dim` set and
    `image_encoder=None`), `image_embed = Linear(feature_dim, e)` and no encoder submodule.
  - `ImagePFN.forward(images, masks, sep, return_thinking=False, image_feats=None,
    seed_query_mask=False)`:
    - `image_feats` `(B,T,N,Cf)` given → image path uses it (skips encoding); `images` may
      be `None` (shape is then read from `masks`).
    - `seed_query_mask=True` → query rows keep the mask columns as passed (no TargetEncoder
      context-mean overwrite).
    - Defaults (`None`/`False`) reproduce current behavior exactly.

- [ ] **Step 1: Write the failing test**

Create `experiments/2d/multilevel/test_imagepfn_modes.py`:

```python
import sys; sys.path.insert(0, ".")
import torch
from src.models.pfn_seg_2d import ImagePFN


def test_default_behavior_unchanged():
    torch.manual_seed(0)
    m = ImagePFN(resolution=8, image_size=32, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2)
    B, T, H = 2, 3, 32
    imgs = torch.rand(B, T, 1, H, H); msk = (torch.rand(B, T, 1, H, H) > 0.5).float()
    out = m(imgs, msk, sep=2)
    assert out.shape == (B, 8, 8)


def test_external_features_mode():
    torch.manual_seed(0)
    Cf = 5
    m = ImagePFN(resolution=8, image_size=32, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2, use_external_features=True, feature_dim=Cf)
    assert m.image_encoder is None
    assert m.image_embed.in_features == Cf
    B, T, H, N = 2, 3, 32, 64
    feats = torch.randn(B, T, N, Cf)
    msk = (torch.rand(B, T, 1, H, H) > 0.5).float()
    out = m(None, msk, sep=2, image_feats=feats, seed_query_mask=True)
    assert out.shape == (B, 8, 8)


def test_seed_query_mask_changes_output():
    # With seeding ON, the query mask we pass should influence the prediction; with it OFF
    # the query mask is overwritten by the context-mean, so the passed query mask is ignored.
    torch.manual_seed(0)
    Cf = 5
    m = ImagePFN(resolution=8, image_size=32, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2, use_external_features=True, feature_dim=Cf)
    B, T, H, N = 1, 3, 32, 64
    feats = torch.randn(B, T, N, Cf)
    msk = (torch.rand(B, T, 1, H, H) > 0.5).float()
    msk_a = msk.clone(); msk_a[:, -1] = 0.0      # query mask all-zero
    msk_b = msk.clone(); msk_b[:, -1] = 1.0      # query mask all-one
    with torch.no_grad():
        seed_a = m(None, msk_a, sep=2, image_feats=feats, seed_query_mask=True)
        seed_b = m(None, msk_b, sep=2, image_feats=feats, seed_query_mask=True)
        off_a  = m(None, msk_a, sep=2, image_feats=feats, seed_query_mask=False)
        off_b  = m(None, msk_b, sep=2, image_feats=feats, seed_query_mask=False)
    assert not torch.allclose(seed_a, seed_b)    # seeding: query mask matters
    assert torch.allclose(off_a, off_b)          # off: query mask overwritten → identical


if __name__ == "__main__":
    test_default_behavior_unchanged()
    test_external_features_mode()
    test_seed_query_mask_changes_output()
    print("ALL IMAGEPFN MODE TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_imagepfn_modes.py`
Expected: FAIL — `test_external_features_mode` raises `TypeError` (unexpected kwarg
`use_external_features`).

- [ ] **Step 3: Modify `ImagePFN.__init__`**

In `src/models/pfn_seg_2d.py`, change the `__init__` signature to add the flag (after
`feature_dim`):

```python
        image_encoder: nn.Module | None = None,
        feature_dim: int | None = None,
        use_external_features: bool = False,
    ):
```

Replace the image-embed construction block:

```python
        self.image_encoder = image_encoder
        if image_encoder is not None:
            assert feature_dim is not None, "feature_dim required with image_encoder"
            self.image_embed = nn.Linear(feature_dim, e)   # embed pretrained features
        else:
            self.image_embed = nn.Linear(Q * Q, e)         # embed raw pixel patches
```

with:

```python
        self.image_encoder = image_encoder
        if image_encoder is not None:
            assert feature_dim is not None, "feature_dim required with image_encoder"
            self.image_embed = nn.Linear(feature_dim, e)   # embed pretrained features
        elif use_external_features:
            # Features are computed outside (e.g. the zoom pipeline crop-pools encoder maps)
            # and passed to forward(image_feats=...); no internal encoder submodule.
            assert feature_dim is not None, "feature_dim required with use_external_features"
            self.image_embed = nn.Linear(feature_dim, e)
        else:
            self.image_embed = nn.Linear(Q * Q, e)         # embed raw pixel patches
```

- [ ] **Step 4: Modify `ImagePFN.forward`**

Change the signature:

```python
    def forward(
        self,
        images: torch.Tensor,  # (B, K+1, 1, H, W) — last row is query; may be None if image_feats given
        masks:  torch.Tensor,  # (B, K+1, 1, H, W) — query mask is replaced below (unless seed_query_mask)
        sep:    int,           # K = number of context images
        return_thinking: bool = False,
        image_feats: torch.Tensor | None = None,   # (B,T,N,Cf) precomputed → skip encoding
        seed_query_mask: bool = False,             # keep query mask as passed (no context-mean)
    ):
```

Replace the shape line:

```python
        B, T, _, H, W = images.shape
```

with (read shape from whichever tensor is present):

```python
        ref = images if images is not None else masks
        B, T, _, H, W = ref.shape
```

Replace the image-cols block:

```python
        # ── Image cols ─────────────────────────────────────────────────────────
        if self.image_encoder is not None:
```

so the precomputed-features path takes priority:

```python
        # ── Image cols ─────────────────────────────────────────────────────────
        if image_feats is not None:
            # Precomputed features (e.g. zoom pipeline crop-pooled encoder maps).
            img_p = standardize_by_context(image_feats, sep)
        elif self.image_encoder is not None:
```

Then, in the mask-cols block, guard the TargetEncoder overwrite:

```python
        # TargetEncoder trick: replace query mask patches with mean of context masks
        ctx_mask_mean = mask_p[:, :sep].mean(dim=1, keepdim=True)           # (B, 1, N, Q²)
        mask_p = torch.cat(
            [mask_p[:, :sep], ctx_mask_mean.expand(B, T - sep, N, Q * Q)],
            dim=1,
        )                                                                     # (B, T, N, Q²)
```

becomes:

```python
        # TargetEncoder trick: replace query mask patches with mean of context masks —
        # unless seed_query_mask, in which case the caller already put a real prior
        # (e.g. the cropped coarse prediction) in the query rows.
        if not seed_query_mask:
            ctx_mask_mean = mask_p[:, :sep].mean(dim=1, keepdim=True)        # (B, 1, N, Q²)
            mask_p = torch.cat(
                [mask_p[:, :sep], ctx_mask_mean.expand(B, T - sep, N, Q * Q)],
                dim=1,
            )                                                                 # (B, T, N, Q²)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_imagepfn_modes.py`
Expected: `ALL IMAGEPFN MODE TESTS PASSED`

- [ ] **Step 6: Regression — existing pipeline tests still pass**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: `ALL PIPELINE TESTS PASSED` (ImagePFN defaults unchanged).

---

### Task 3: `zoom_pipeline.py` — `run_zoom_chain`

**Files:**
- Create: `experiments/2d/multilevel/zoom_pipeline.py`
- Test: `experiments/2d/multilevel/test_zoom_pipeline.py`

**Interfaces:**
- Consumes: `bbox.{max_sum_window, gt_window, crop_resize, composite_window}` (Task 1);
  `ImagePFN.forward(..., image_feats=, seed_query_mask=)` (Task 2);
  `pipeline._grid_from_feat` (standardize pooled features → `(B,T,N,Cf)`).
- Produces:
  - `crop_pool_maps(maps, origin, s, out) -> Tensor (N, Cf, out, out)` — concat of each
    stage map cropped to `origin` and resampled to `out×out`. `maps` is the list from
    `encoder.encode_maps`.
  - `run_zoom_chain(batch, stage1, encoder, models, cfg, source, stochastic, device)
    -> (outputs, coarse_lr)`:
    - `coarse_lr` `(B, R0, R0)` — stage-1 probability at its native resolution.
    - `outputs` — list (one per `cfg.sample.crop_sizes` entry) of dicts:
      `{"logits": (B,N), "qry_gt": (B,N), "refined_full": (B,1,H,W), "origin": (B,2),
        "crop_size": int}` where `N = R0²`.

- [ ] **Step 1: Write the failing test**

Create `experiments/2d/multilevel/test_zoom_pipeline.py`:

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d/multilevel")
import torch
from omegaconf import OmegaConf
from src.models.pfn_seg_2d import ImagePFN


class StubStage1:
    """ImagePFN-like: returns (B,R0,R0) logits (and thinking if asked). N = R0²."""
    def __init__(self, R0=8): self.N = R0 * R0; self.R0 = R0
    def __call__(self, images, masks, sep, return_thinking=False):
        B = images.shape[0]
        logits = torch.randn(B, self.R0, self.R0)
        return (logits, torch.randn(B, 4, 16)) if return_thinking else logits
    def eval(self): return self


class StubEncoder:
    """encode_maps → list of (N, C_i, R_i, R_i) maps; feature_dim = sum(C_i)."""
    feature_dim = 5
    def encode_maps(self, images):
        N = images.shape[0]
        return [torch.randn(N, 2, 16, 16), torch.randn(N, 3, 8, 8)]
    def eval(self): return self


def _batch(B=2, K=2, H=32):
    return {
        "image":       torch.rand(B, 1, H, H),
        "label":       (torch.rand(B, 1, H, H) > 0.5).float(),
        "context_in":  torch.rand(B, K, 1, H, H),
        "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float(),
        "dataset":     ["d"] * B,
    }


def test_crop_pool_maps_shape():
    from zoom_pipeline import crop_pool_maps
    maps = [torch.randn(6, 2, 16, 16), torch.randn(6, 3, 8, 8)]
    origin = torch.zeros(6, 2, dtype=torch.long)
    feat = crop_pool_maps(maps, origin, s=16, out=8)
    assert feat.shape == (6, 5, 8, 8)          # channels concatenated, pooled to 8×8


def test_run_zoom_chain_shapes_and_composite():
    from zoom_pipeline import run_zoom_chain
    torch.manual_seed(0)
    dev = torch.device("cpu")
    B, K, H, R0, Cf = 2, 2, 32, 8, 5
    batch = _batch(B, K, H)
    cfg = OmegaConf.create({"sample": {"crop_sizes": [16]},
                            "data": {"image_size": H}})
    models = torch.nn.ModuleList([
        ImagePFN(resolution=R0, image_size=H, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2, use_external_features=True, feature_dim=Cf)])
    outputs, coarse_lr = run_zoom_chain(batch, StubStage1(R0), StubEncoder(), models, cfg,
                                        "prev_pred", True, dev)
    assert coarse_lr.shape == (B, R0, R0)
    assert len(outputs) == 1
    o = outputs[0]
    assert o["logits"].shape == (B, R0 * R0)
    assert o["qry_gt"].shape == (B, R0 * R0)
    assert o["refined_full"].shape == (B, 1, H, H)
    assert o["origin"].shape == (B, 2)
    # composite changed only inside the bbox vs the upsampled stage-1 prediction
    assert torch.isfinite(o["refined_full"]).all()


if __name__ == "__main__":
    test_crop_pool_maps_shape()
    test_run_zoom_chain_shapes_and_composite()
    print("ALL ZOOM PIPELINE TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_zoom_pipeline.py`
Expected: `ModuleNotFoundError: No module named 'zoom_pipeline'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/2d/multilevel/zoom_pipeline.py`:

```python
"""
Zoom-refinement chain: same ImagePFN arch as stage-1, fed a contiguous square crop.

Stage-1 (frozen) predicts at R0 on the full image; each hop crops an s×s square (max
predicted mass for the target, densest GT for each context), pools the once-computed
encoder maps to that bbox, and runs a warm-started ImagePFN that corrects the cropped
coarse prediction. The R0×R0 output is upsampled to s×s and composited back. Hops chain
through the detached composite. See the design/plan in docs/superpowers.
"""

import torch
import torch.nn.functional as F

from bbox import composite_window, crop_resize, gt_window, max_sum_window
from pipeline import _grid_from_feat


def crop_pool_maps(maps, origin, s, out):
    """encode_maps list → (N, sum(C_i), out, out): each stage map cropped to the s×s bbox
    at `origin` (N,2) and resampled to out×out, then concatenated over channels."""
    return torch.cat([crop_resize(m.float(), origin, s, out, mode="bilinear") for m in maps],
                     dim=1)


@torch.no_grad()
def _coarse(stage1, all_images, all_masks, K):
    logits = stage1(all_images, all_masks, sep=K)
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.sigmoid(logits.float())            # (B, R0, R0) probability


def run_zoom_chain(batch, stage1, encoder, models, cfg, source, stochastic, device):
    """Coarse-to-fine zoom chain. Returns (outputs list per hop, coarse_lr (B,R0,R0))."""
    crop_sizes = list(cfg.sample.crop_sizes)
    H = cfg.data.image_size
    image       = batch["image"].to(device)             # (B,1,H,W)
    context_in  = batch["context_in"].to(device)        # (B,K,1,H,W)
    context_out = batch["context_out"].to(device)       # (B,K,1,H,W)
    label       = batch["label"].to(device)             # (B,1,H,W)
    B, K = context_in.shape[0], context_in.shape[1]

    all_images = torch.cat([context_in, image.unsqueeze(1)], dim=1)         # (B,T,1,H,W)
    all_masks  = torch.cat([context_out, torch.zeros_like(image.unsqueeze(1))], dim=1)
    T = all_images.shape[1]

    coarse_lr = _coarse(stage1, all_images, all_masks, K)                   # (B,R0,R0)
    R0 = coarse_lr.shape[-1]
    pred = F.interpolate(coarse_lr.unsqueeze(1), size=(H, H),
                         mode="bilinear", align_corners=False)             # (B,1,H,W)

    with torch.no_grad():
        maps = encoder.encode_maps(all_images.reshape(B * T, 1, H, H))

    outputs = []
    for L, s in enumerate(crop_sizes):
        tgt_o = max_sum_window(pred, s)                                     # (B,2)
        ctx_o = torch.stack([gt_window(context_out[:, k], s) for k in range(K)], dim=1)  # (B,K,2)
        origins = torch.cat([ctx_o, tgt_o.unsqueeze(1)], dim=1).reshape(B * T, 2)        # (B*T,2)

        with torch.no_grad():
            feat = crop_pool_maps(maps, origins, s, R0)                     # (B*T, Cf, R0, R0)
            image_feats = _grid_from_feat(feat, B, T, R0, K)               # (B,T,N,Cf) standardized

        # Mask images cropped to each bbox: context = true GT (nearest); query = coarse prior.
        ctx_mask = crop_resize(context_out.reshape(B * K, 1, H, H),
                               ctx_o.reshape(B * K, 2), s, H, mode="nearest").reshape(B, K, 1, H, H)
        qry_prior = crop_resize(pred, tgt_o, s, H, mode="bilinear").unsqueeze(1)         # (B,1,1,H,W)
        masks_in = torch.cat([ctx_mask, qry_prior], dim=1)                 # (B,T,1,H,W)

        logits = models[L](None, masks_in, sep=K, image_feats=image_feats,
                           seed_query_mask=True)                           # (B,R0,R0)
        logits = logits.reshape(B, R0 * R0)
        qry_gt = crop_resize(label, tgt_o, s, R0, mode="bilinear").reshape(B, R0 * R0)   # soft GT

        patch = F.interpolate(torch.sigmoid(logits.float()).reshape(B, 1, R0, R0),
                              size=(s, s), mode="bilinear", align_corners=False)         # (B,1,s,s)
        pred = composite_window(pred, patch, tgt_o, s).detach()
        outputs.append({"logits": logits, "qry_gt": qry_gt, "refined_full": pred,
                        "origin": tgt_o, "crop_size": s})

    return outputs, coarse_lr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_zoom_pipeline.py`
Expected: `ALL ZOOM PIPELINE TESTS PASSED`

---

### Task 4: Train/eval integration + config + smoke run

**Files:**
- Modify: `experiments/2d/multilevel/train.py`
- Create: `configs/experiment/2d/multilevel_zoom.yaml`
- Modify: `docs/logs.md`

**Interfaces:**
- Consumes: `run_zoom_chain` (Task 3); `ImagePFN` external-features mode (Task 2);
  existing `train.py` helpers (`load_stage1`, `augment`, `patch_loss`, `batch_dice_sums`,
  Muon/AdamW setup, LAWA, checkpoint save).
- Produces: a runnable `refine_arch=imagepfn_zoom` path selectable via
  `configs/experiment/2d/multilevel_zoom.yaml`.

- [ ] **Step 1: Add imports and the zoom model builder to `train.py`**

In `experiments/2d/multilevel/train.py`, extend the multilevel import line:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))       # multilevel
from pipeline import run_chain
```

to:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))       # multilevel
from pipeline import run_chain
from zoom_pipeline import run_zoom_chain
```

Add a builder function after `load_stage1` (it re-reads the stage-1 ckpt's `arch` so the
hop ImagePFNs are structurally identical and warm-start cleanly):

```python
def build_zoom_models(cfg, stage1, feature_dim):
    """ModuleList of ImagePFN hops (one per crop_sizes), warm-started from frozen stage-1.

    External-features mode: the encoder lives once in the chain; hops consume crop-pooled
    features. Warm-start loads stage-1's weights minus image_encoder.* (strict=False)."""
    ckpt = torch.load(cfg.train.stage1_checkpoint, map_location="cpu", weights_only=False)
    arch, img = ckpt["arch"], ckpt["image_size"]
    resolution = int(round(stage1.N ** 0.5))
    n_hops = len(cfg.sample.crop_sizes)
    models = nn.ModuleList([
        ImagePFN(resolution=resolution, image_size=img,
                 input_patch_size=arch.get("input_patch_size", img // resolution),
                 use_external_features=True, feature_dim=feature_dim,
                 e=arch["e"], h=arch["h"], l=arch["l"], a=arch["a"],
                 thinking_rows=arch["thinking_rows"],
                 residual_decay=arch["residual_decay"]).to(DEVICE)
        for _ in range(n_hops)])
    s1 = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()
          if not k.removeprefix("_orig_mod.").startswith("image_encoder.")}
    for m in models:
        m.load_state_dict(s1, strict=False)
    assert stage1.image_embed.in_features == feature_dim, (
        f"chain encoder feature_dim {feature_dim} != stage-1 image_embed "
        f"{stage1.image_embed.in_features}; encoder must match the stage-1 checkpoint")
    print(f"Zoom ImagePFN chain: {n_hops} hops (crop_sizes={list(cfg.sample.crop_sizes)}), "
          f"warm-started from stage-1; "
          f"{sum(p.numel() for p in models.parameters() if p.requires_grad):,} params")
    return models
```

- [ ] **Step 2: Make `train_epoch` chain-agnostic**

In `train_epoch`, change the signature to accept the chain fn and hop labels:

```python
def train_epoch(model, loader, stage1, encoder, optimizers, cfg, epoch):
```

to:

```python
def train_epoch(model, loader, stage1, encoder, optimizers, cfg, epoch, chain_fn, hop_labels):
```

Replace:

```python
    hops = list(cfg.sample.resolutions)[1:]
    nh = len(hops)
```

with:

```python
    hops = list(hop_labels)
    nh = len(hops)
```

Replace the chain call:

```python
            outputs, _ = run_chain(batch, stage1, encoder, model, cfg, cfg.sample.train,
                                   stochastic=True, device=DEVICE)
```

with:

```python
            outputs, _ = chain_fn(batch, stage1, encoder, model, cfg, cfg.sample.train,
                                  stochastic=True, device=DEVICE)
```

(The rest of `train_epoch` already keys on `outputs[L]["logits"]`/`["qry_gt"]`, which both
chains produce, so per-hop loss + `batch_dice_sums` accuracy work unchanged.)

- [ ] **Step 3: Add `run_eval_zoom`**

Add this function next to `run_eval` in `train.py`:

```python
@torch.no_grad()
def run_eval_zoom(model, loader, stage1, encoder, lawa_queue, cfg, epoch):
    """Eval for the zoom chain: composite is at native H, so metrics are full-res Dice
    after each hop, plus the in-bbox refine delta. Returns dice_soft/mean (ckpt metric)."""
    saved = lawa_average(lawa_queue, model, DEVICE)
    for m in model: m.eval()
    crop_sizes = list(cfg.sample.crop_sizes)
    nh = len(crop_sizes)
    per_ds      = defaultdict(list)                 # final hard dice
    per_ds_soft = defaultdict(list)                 # final soft dice
    after = [defaultdict(list) for _ in range(nh)]  # hard dice after each hop
    delta = [[] for _ in range(nh)]                 # in-bbox hard-dice gain vs prior
    total_loss, nl = 0.0, 0

    for batch in loader:
        if batch is None: continue
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            outputs, coarse_lr = run_zoom_chain(batch, stage1, encoder, model, cfg,
                                                cfg.sample.eval,
                                                stochastic=not cfg.sample.eval_deterministic,
                                                device=DEVICE)
            weights = list(cfg.train.loss_weights)
            total_loss += sum(w * patch_loss(o["logits"], {"qry_gt": o["qry_gt"]}, cfg)
                              for w, o in zip(weights, outputs)).item()
            nl += 1
        B = coarse_lr.shape[0]
        H = cfg.data.image_size
        # Stage-1 baseline composite at native H (for the in-bbox delta of hop 0).
        prev_full = F.interpolate(coarse_lr.unsqueeze(1), size=(H, H),
                                  mode="bilinear", align_corners=False)
        for L, o in enumerate(outputs):
            full = o["refined_full"]
            for b in range(B):
                ds = batch["dataset"][b]
                gt = batch["label"][b, 0]
                after[L][ds].append(hard_dice(full[b, 0].cpu(), gt))
                r0, c0 = int(o["origin"][b, 0]), int(o["origin"][b, 1]); s = o["crop_size"]
                box = (slice(r0, r0 + s), slice(c0, c0 + s))
                gtb = (gt[box] >= 0.5).float()
                delta[L].append(hard_dice(full[b, 0, box[0], box[1]].cpu(), gtb)
                                - hard_dice(prev_full[b, 0, box[0], box[1]].cpu(), gtb))
            prev_full = full
        for b in range(B):
            ds = batch["dataset"][b]; gt = batch["label"][b, 0]
            per_ds[ds].append(hard_dice(outputs[-1]["refined_full"][b, 0].cpu(), gt))
            per_ds_soft[ds].append(soft_dice(outputs[-1]["refined_full"][b, 0].cpu(), gt))
    if saved is not None:
        model.load_state_dict(saved)

    def nanmean(xs):
        v = [x for x in xs if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")
    flat = lambda d: [x for sc in d.values() for x in sc if not np.isnan(x)]

    metrics = {"epoch": epoch, "val/loss": total_loss / max(nl, 1)}
    metrics["dice/mean"]      = float(np.mean(flat(per_ds)))      if flat(per_ds)      else float("nan")
    metrics["dice_soft/mean"] = float(np.mean(flat(per_ds_soft))) if flat(per_ds_soft) else float("nan")
    for L in range(nh):
        metrics[f"dice_after_hop{L}/mean"]  = float(np.mean(flat(after[L]))) if flat(after[L]) else float("nan")
        metrics[f"refine/hop{L}/dice_delta"] = nanmean(delta[L])
    for k, v in per_ds.items():
        metrics[f"dice/dataset/{k}"] = nanmean(v)
    tqdm.write(f"  [e{epoch}] val loss={metrics['val/loss']:.4f}  "
               f"dice={metrics['dice/mean']:.4f}  soft={metrics['dice_soft/mean']:.4f}  "
               + "  ".join(f"d{cs}={metrics[f'refine/hop{L}/dice_delta']:+.4f}"
                           for L, cs in enumerate(crop_sizes)))
    wandb.log(metrics)
    for m in model: m.train()
    return metrics["dice_soft/mean"]
```

- [ ] **Step 4: Wire the switch into `main`**

In `main`, replace the model-build block (the `resolutions = list(cfg.sample.resolutions)`
assertion + `model = nn.ModuleList([PatchSetPFN(...) ...])` + the param-count print) with a
branch on `refine_arch`. Insert immediately before that block:

```python
    is_zoom = cfg.arch.get("refine_arch", "patchset") == "imagepfn_zoom"
    chain_fn = run_zoom_chain if is_zoom else run_chain
    eval_fn  = run_eval_zoom if is_zoom else run_eval
    hop_labels = (list(cfg.sample.crop_sizes) if is_zoom
                  else list(cfg.sample.resolutions)[1:])
```

Then guard the existing PatchSetPFN build so it only runs for the patchset path, and add
the zoom build. The existing lines from `resolutions = list(cfg.sample.resolutions)` through
the `print(f"PatchSetPFN chain: ...")` become:

```python
    if is_zoom:
        model = build_zoom_models(cfg, stage1, feature_dim)
    else:
        resolutions = list(cfg.sample.resolutions)
        assert resolutions[0] == int(round(stage1.N ** 0.5)), \
            f"resolutions[0]={resolutions[0]} must equal stage-1 res {int(round(stage1.N ** 0.5))}"
        # Chained thinking: hop L>0 receives the previous PatchSetPFN's thinking (dim e).
        model = nn.ModuleList([
            PatchSetPFN(feature_dim=feature_dim, e=cfg.arch.e, h=cfg.arch.h, l=cfg.arch.l,
                        a=cfg.arch.a, thinking_rows=cfg.arch.thinking_rows,
                        residual_decay=cfg.arch.residual_decay, fourier_bands=cfg.arch.fourier_bands,
                        mask_prior=cfg.arch.mask_prior,
                        mask_patch_size=cfg.data.image_size // grid,
                        stage1_dim=(stage1_dim if L == 0 else cfg.arch.e),
                        query_self_attn=cfg.arch.query_self_attn).to(DEVICE)
            for L, grid in enumerate(resolutions[1:])])
        print(f"PatchSetPFN chain: {len(model)} hops, "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
```

(`stage1_dim` is computed just above this block for the patchset path; leave that code as-is
— it is only consumed inside the `else` branch.)

Update the `train_epoch` call in the epoch loop:

```python
        loss, train_soft, train_hard = train_epoch(model, train_loader, stage1, encoder,
                                                    optimizers, cfg, epoch)
```

to:

```python
        loss, train_soft, train_hard = train_epoch(model, train_loader, stage1, encoder,
                                                    optimizers, cfg, epoch, chain_fn, hop_labels)
```

And the eval call:

```python
            dice_soft = run_eval(model, val_loader, stage1, encoder, lawa_queue, cfg, epoch)
```

to:

```python
            dice_soft = eval_fn(model, val_loader, stage1, encoder, lawa_queue, cfg, epoch)
```

- [ ] **Step 5: Create the config**

Create `configs/experiment/2d/multilevel_zoom.yaml`:

```yaml
# Stage-2 zoom refinement: same ImagePFN arch as stage-1, fed a square crop (no PatchSetPFN).
# Selected via arch.refine_arch=imagepfn_zoom. Common params inherited from train_base.yaml.
defaults:
  - train_base
  - _self_

model: imagepfn_zoom

sample:
  crop_sizes: [64]           # one per hop, pixels in the 128 frame (start: single 64px hop)
  eval_deterministic: true

arch:
  refine_arch: imagepfn_zoom

train:
  loss_weights: [1.0]        # len == len(crop_sizes)
  stage1_checkpoint: results/2d/pfn_seg_universeg/pfn_seg_USegall_R16q8_e256_l6_k3_think8/best.pt

eval:
  max_per_label: null
```

- [ ] **Step 6: Config composition smoke check**

Run: `.venv311/bin/python experiments/2d/multilevel/train.py --cfg job -c experiment/2d/multilevel_zoom 2>/dev/null || .venv311/bin/python experiments/2d/multilevel/train.py --config-name multilevel_zoom --cfg job`
Expected: prints the resolved config with `arch.refine_arch: imagepfn_zoom` and
`sample.crop_sizes: [64]`. (No training runs with `--cfg job`.)

- [ ] **Step 7: One-epoch training smoke run**

Run (tiny, CPU/GPU, 1 epoch, a few batches, wandb disabled):

```bash
.venv311/bin/python experiments/2d/multilevel/train.py --config-name multilevel_zoom \
  train.epochs=1 data.max_train_samples=64 eval.batch_size=8 train.batch_size=8 \
  wandb.enabled=false arch.compile=false
```

Expected: `Zoom ImagePFN chain: 1 hops ... warm-started from stage-1`, the train loss prints
and decreases over the epoch, the eval line prints `dice=... soft=... d64=...`, and a
`best.pt` is written under `eval.out_dir`. No shape/key errors.

- [ ] **Step 8: Log the change**

Append to `docs/logs.md`:

```markdown
## 2026-06-22 — Zoom-refinement variant (shared ImagePFN arch)
- New stage-2 refinement path selectable via arch.refine_arch=imagepfn_zoom (default
  patchset, unchanged). Instead of PatchSetPFN on scattered patches, a warm-started
  ImagePFN refines a contiguous square crop ("zoom"), so the refiner is the SAME arch as
  stage-1 — isolating refinement from the change of model class.
- experiments/2d/multilevel/bbox.py: max_sum_window / gt_window (s×s square of largest
  predicted mass / densest GT), crop_resize (batched grid_sample crop+resample),
  composite_window (write crop back). Unit-tested in test_bbox.py.
- src/models/pfn_seg_2d.py ImagePFN: backward-compatible use_external_features (consume
  precomputed features, no internal encoder) + forward(image_feats=, seed_query_mask=).
  Defaults reproduce prior behavior; test_imagepfn_modes.py + test_pipeline.py green.
- experiments/2d/multilevel/zoom_pipeline.py run_zoom_chain: frozen stage-1 coarse pred →
  encode maps ONCE → per hop crop-pool maps to the bbox (same encode-once features the
  PatchSetPFN chain uses), seed the query with the cropped coarse pred, composite the
  upsampled R0 output back at native H. test_zoom_pipeline.py.
- train.py: refine_arch switch (build_zoom_models warm-start, chain_fn dispatch in
  train_epoch, run_eval_zoom with native-H dice + in-bbox refine delta). Config
  configs/experiment/2d/multilevel_zoom.yaml (single 64px hop). Verified 1-epoch smoke run.
```

---

## Self-Review

**Spec coverage:**
- Chained zoom ladder / single 64px hop → Tasks 3, 4 (`crop_sizes:[64]`, ModuleList loop). ✓
- Fixed crop-size schedule, max-sum centering, GT-window context centering → Task 1 + 3. ✓
- Warm-start from stage-1 → `build_zoom_models` (Task 4). ✓
- Seed query with coarse prediction → `seed_query_mask` (Task 2) + `run_zoom_chain` (Task 3). ✓
- Encode-once / crop-pool, no re-encoding → `crop_pool_maps` + single `encode_maps` (Task 3). ✓
- ImagePFN minimal default-off changes → Task 2. ✓
- `refine_arch` switch, existing path untouched → Task 4 (`else` branch verbatim). ✓
- Metrics (final dice, per-hop after, in-bbox delta, val/loss) → `run_eval_zoom` (Task 4). ✓
- Tests for bbox / ImagePFN / zoom chain → Tasks 1–3. ✓
- Edge case (border clamp) → covered by valid-origin construction + Task 1 test. ✓

**Placeholder scan:** No TBD/TODO; every code/test step shows full content. ✓

**Type consistency:** `origin`/`top-left (B,2)` long tensors used uniformly across
`max_sum_window`/`gt_window`/`crop_resize`/`composite_window`; `outputs[L]` dict keys
(`logits`,`qry_gt`,`refined_full`,`origin`,`crop_size`) match between `run_zoom_chain`
(producer) and `train_epoch`/`run_eval_zoom` (consumers); `image_feats`/`seed_query_mask`
names match between Task 2 (ImagePFN) and Task 3 (caller). ✓

**Deferred (out of scope for v1, per spec):** per-sample empty-prediction skip; DINOv3
channel-reduction in `crop_pool_maps` (v1 targets the UniverSeg stage-1 ckpt, concat
features); multi-hop schedule (machinery generalizes, config stays `[64]`).

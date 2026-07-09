# PatchSetCNN bbox-zoom refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional single-level bbox-zoom refinement mode to `PatchSetCNN` that produces a native-resolution fused output, trainable/evaluable through the existing `experiments/2d/train.py` and `experiments/2d/eval_incontext.py` with no changes to the training/eval loops.

**Architecture:** The same `PatchSetCNN` runs twice — a coarse R×R pass over the full image, then a refine R×R pass over a square crop centered on the densest predicted region (contexts cropped on densest GT). The refine logit is added as a residual into the upsampled coarse logit at the bbox (logit fusion), giving one native-resolution `final_logit` supervised by the trainer's existing single BCE + soft-Dice loss.

**Tech Stack:** PyTorch 2.5 (`.venv/bin/python`, py3.12, cuda), Hydra configs, pytest 9.

## Global Constraints

- Python interpreter: `.venv/bin/python` (never `python`/`uv`/conda). Tests run from repo root as `.venv/bin/python -m pytest ...` so `import src...` resolves.
- **Version control is the user's job.** Do NOT `git add`/`git commit`/`git stage`. Each task ends at a review checkpoint (a self-contained, tested unit); leave committing to the user.
- Log the change in `docs/logs.md` (prepend a dated entry) as part of the final task.
- Defaults must be backward-compatible: with `refine=False`, `PatchSetCNN` is byte-for-byte unchanged (returns `(B,1,R,R)`).
- Keep code readable with short docstrings (repo convention).

---

### Task 1: `src/models/bbox_refine.py` — pure bbox tensor ops

**Files:**
- Create: `src/models/bbox_refine.py`
- Test: `tests/test_bbox_refine.py`

**Interfaces:**
- Consumes: nothing (torch only).
- Produces:
  - `max_sum_window(prob, s) -> LongTensor (B,2)` — top-left `(row,col)` origin of the s×s window with max summed value; accepts `(B,1,H,W)` or `(B,H,W)`; empty maps center the crop; origins in-bounds.
  - `gt_window(mask, s) -> LongTensor (B,2)` — same on a binary/soft mask.
  - `crop_resize(x, origin, s, out, mode="bilinear") -> Tensor (N,C,out,out)` — batched per-sample crop of `(N,C,H,W)` to the s×s bbox at `origin (N,2)`, resampled to `out×out`.
  - `fuse_window(full, patch, origin, s) -> Tensor (B,1,H,W)` — additive: returns a clone of `full` with `patch (B,1,s,s)` **added** into the s×s window at each `origin (B,2)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bbox_refine.py`:

```python
import sys; sys.path.insert(0, ".")
import torch
from src.models.bbox_refine import max_sum_window, gt_window, crop_resize, fuse_window


def test_max_sum_window_finds_blob():
    H, s = 16, 8
    prob = torch.zeros(2, 1, H, H)
    prob[0, 0, 2:6, 3:7] = 1.0            # blob near top-left
    prob[1, 0, 10:14, 9:13] = 1.0         # blob near bottom-right
    o = max_sum_window(prob, s)
    assert o.shape == (2, 2)
    assert (o >= 0).all() and (o[:, 0] <= H - s).all() and (o[:, 1] <= H - s).all()
    assert o[0, 0] <= 2 and o[0, 1] <= 3          # window covers top-left blob
    assert o[1, 0] >= 6 and o[1, 1] >= 5          # window shifted toward bottom-right blob


def test_max_sum_window_empty_centers():
    H, s = 16, 8
    prob = torch.zeros(1, 1, H, H)                # no mass → center, not corner (0,0)
    o = max_sum_window(prob, s)
    assert o[0, 0] == (H - s) // 2 and o[0, 1] == (H - s) // 2


def test_gt_window_matches_blob():
    H, s = 16, 8
    mask = torch.zeros(1, 1, H, H)
    mask[0, 0, 4:8, 4:8] = 1.0
    o = gt_window(mask, s)
    assert o[0, 0] <= 4 and o[0, 1] <= 4 and o[0, 0] >= 0 and o[0, 1] >= 0


def test_crop_resize_recovers_region():
    # crop the exact 8x8 region back to 8x8 (out=s) should reproduce it (bilinear, aligned cells)
    H, s = 16, 8
    x = torch.zeros(1, 1, H, H)
    x[0, 0, 4:12, 4:12] = 1.0
    o = torch.tensor([[4, 4]])
    y = crop_resize(x, o, s, out=s, mode="nearest")
    assert y.shape == (1, 1, s, s)
    assert y.min() > 0.5                          # every cell inside the all-ones region


def test_fuse_window_adds_into_window_only():
    H, s = 16, 8
    full = torch.zeros(2, 1, H, H)
    patch = torch.ones(2, 1, s, s)
    o = torch.tensor([[0, 0], [8, 8]])
    out = fuse_window(full, patch, o, s)
    assert out.shape == (2, 1, H, H)
    assert (full == 0).all()                      # input not mutated
    assert out[0, 0, 0:s, 0:s].eq(1).all()        # window filled
    assert out[0, 0, s:, s:].eq(0).all()          # outside untouched
    assert out[1, 0, 8:16, 8:16].eq(1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bbox_refine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.bbox_refine'`.

- [ ] **Step 3: Write the module**

Create `src/models/bbox_refine.py` (adapted from `experiments/2d/multilevel/bbox.py`; `fuse_window` is the additive variant of that file's `composite_window`):

```python
"""Square-bbox crop / fuse ops for PatchSetCNN's bbox-zoom refinement.

Origins are top-left (row, col) integer corners of an s×s square that always fits
inside the H×W frame. crop_resize is batched (per-sample origin) via F.grid_sample.
fuse_window ADDS a patch into the window (logit-space residual fusion), unlike a
hard-replace composite. Adapted from experiments/2d/multilevel/bbox.py.
"""

import torch
import torch.nn.functional as F


def _box_sum(x, s):
    """(B,1,H,W) → (B,1,H-s+1,W-s+1): summed value of every s×s window (stride 1)."""
    return F.avg_pool2d(x, kernel_size=s, stride=1) * (s * s)


def _window_origin(score, s, H, W, eps=0.5):
    """(B,1,Hs,Ws) window scores → (B,2) argmax top-left per sample. Windows holding
    essentially no mass (max ≤ eps, e.g. an empty prediction) center the crop instead of
    collapsing to the (0,0) corner. eps=0.5 ≈ 'less than half a cell of mass'."""
    B, Ws = score.shape[0], score.shape[-1]
    flat = score.reshape(B, -1)
    idx = flat.argmax(dim=1)
    origin = torch.stack([torch.div(idx, Ws, rounding_mode="floor"), idx % Ws], dim=1)
    center = origin.new_tensor([(H - s) // 2, (W - s) // 2])
    return torch.where((flat.amax(dim=1) <= eps).unsqueeze(1), center, origin)


def max_sum_window(prob, s):
    """Top-left (B,2) of the s×s square with the largest summed value in `prob`
    ((B,1,H,W) or (B,H,W)). Empty maps center the crop."""
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    prob = prob.float()
    H, W = prob.shape[-2:]
    return _window_origin(_box_sum(prob, s), s, H, W)


def gt_window(mask, s):
    """Top-left (B,2) of the s×s square with the most foreground in `mask`
    ((B,1,H,W) or (B,H,W)). Empty masks center the crop."""
    return max_sum_window(mask, s)


def crop_resize(x, origin, s, out, mode="bilinear"):
    """Crop each (N,C,H,W) image to its s×s bbox at `origin` (N,2) and resample to
    out×out via F.grid_sample (align_corners=False, border padding)."""
    N, C, H, W = x.shape
    r0 = origin[:, 0].to(x.dtype).view(N, 1)
    c0 = origin[:, 1].to(x.dtype).view(N, 1)
    i = torch.arange(out, device=x.device, dtype=x.dtype) + 0.5      # cell centers
    rows = r0 + i.view(1, out) * (s / out)
    cols = c0 + i.view(1, out) * (s / out)
    ny = 2.0 * rows / H - 1.0
    nx = 2.0 * cols / W - 1.0
    grid = torch.stack([nx.view(N, 1, out).expand(N, out, out),
                        ny.view(N, out, 1).expand(N, out, out)], dim=-1)
    return F.grid_sample(x, grid, mode=mode, align_corners=False, padding_mode="border")


def fuse_window(full, patch, origin, s):
    """Return a clone of full (B,1,H,W) with patch (B,1,s,s) ADDED into the s×s window at
    each origin (B,2). Additive (logit-space) fusion; input not mutated. Per-sample loop
    (B is the small batch dim)."""
    out = full.clone()
    for b in range(full.shape[0]):
        r0, c0 = int(origin[b, 0]), int(origin[b, 1])
        out[b, 0, r0:r0 + s, c0:c0 + s] += patch[b, 0]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bbox_refine.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Review checkpoint**

`src/models/bbox_refine.py` + its tests are a self-contained unit. Stop for user review/commit.

---

### Task 2: PatchSetCNN `refine` mode

**Files:**
- Modify: `src/models/patchset_cnn.py` (imports; `__init__` adds `refine`/`refine_crop`; refactor `forward` body into `_segment`; add `_refine_forward`; new `forward` dispatch)
- Test: `tests/test_patchset_cnn_refine.py`

**Interfaces:**
- Consumes: `max_sum_window`, `gt_window`, `crop_resize`, `fuse_window` from Task 1.
- Produces:
  - `PatchSetCNN(..., refine: bool = False, refine_crop: int = 64)`.
  - `PatchSetCNN._segment(image, context_in, context_out) -> Tensor (B,1,R,R)` — the coarse logit (old forward body).
  - `PatchSetCNN.forward(image, context_in, context_out, mode="train") -> {"final_logit": Tensor}` — `(B,1,R,R)` when `refine=False`, `(B,1,H,W)` when `refine=True`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_patchset_cnn_refine.py`:

```python
import sys; sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from src.models.patchset_cnn import PatchSetCNN


def _model(refine, H=32, R=8):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=R, enc_dims=[16], e=32, h=64, l=1, a=2,
                       thinking_rows=1, refine=refine, refine_crop=16)


def _batch(B=2, K=2, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_refine_false_unchanged_shape():
    m = _model(refine=False)
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    assert out.shape == (2, 1, 8, 8)                         # native R×R
    # forward == _segment when refine is off
    seg = m._segment(img, cin, cout)
    assert torch.equal(out, seg)


def test_refine_true_native_shape_and_finite():
    m = _model(refine=True)
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    assert out.shape == (2, 1, 32, 32)                       # native H×W fused
    assert torch.isfinite(out).all()


def test_refine_grad_reaches_encoder_and_decoder():
    m = _model(refine=True)
    img, cin, cout = _batch()
    lbl = (torch.rand(2, 1, 32, 32) > 0.5).float()
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    loss = F.binary_cross_entropy_with_logits(out, lbl)
    loss.backward()
    assert m.decoder[0].weight.grad is not None
    assert m.encoder.stem[0].weight.grad is not None         # coarse+refine both use encoder
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_patchset_cnn_refine.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'refine'`.

- [ ] **Step 3: Add imports**

In `src/models/patchset_cnn.py`, after the existing `from src.models.pfn_seg_2d import ...` line, add:

```python
from src.models.bbox_refine import crop_resize, fuse_window, gt_window, max_sum_window
```

- [ ] **Step 4: Add constructor args**

In `PatchSetCNN.__init__`, add two params to the signature (after `max_context: int = 16,`):

```python
        max_context: int = 16,
        refine: bool = False,
        refine_crop: int = 64,
```

and store them (near `self.max_context = max_context`):

```python
        self.refine = refine
        self.refine_crop = refine_crop
```

- [ ] **Step 5: Refactor `forward` body into `_segment`**

Rename the existing `def forward(self, image, context_in, context_out, mode="train"):` to `def _segment(self, image, context_in, context_out):`, update its docstring first line, and change its final two lines from returning a dict to returning the bare logit. The method now ends:

```python
        q = x[:, sep_t:, 0, :]                                            # (B,Q,e) query img-col
        logit = self.decoder(q).squeeze(-1).reshape(B, 1, R, R)
        return logit                                                      # (B,1,R,R)
```

Update its docstring to:

```python
        """Coarse single-pass segmentation → (B,1,R,R) logits.

        image (B,1,H,W); context_in/out (B,K,1,H,W). Support = all K·N context patches
        (known mask occupancy); query = the N target patches (mask = support-mean prior)."""
```

(The `mode` argument is dropped here; it is reinstated on the public `forward` below.)

- [ ] **Step 6: Add `_refine_forward` and the new `forward`**

Immediately after `_segment`, add:

```python
    def _refine_forward(self, image, context_in, context_out):
        """Coarse pass → bbox-zoom refine pass → logit-space fusion → (B,1,H,W).

        Crop the target on its densest predicted region (max_sum_window) and each context on
        its densest GT (gt_window), resize crops back to H, re-segment with the SAME weights,
        and ADD the refine logit as a residual into the upsampled coarse logit at the bbox."""
        B, K = context_in.shape[0], context_in.shape[1]
        H, W = image.shape[-2:]
        s = self.refine_crop

        coarse = self._segment(image, context_in, context_out)            # (B,1,R,R)
        coarse_up = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)
        prob_up = F.interpolate(torch.sigmoid(coarse).detach(), size=(H, W),
                                mode="bilinear", align_corners=False)      # bbox selection only

        tgt_o = max_sum_window(prob_up, s)                                # (B,2)
        ctx_o = torch.stack([gt_window(context_out[:, k], s) for k in range(K)], dim=1)  # (B,K,2)

        tgt_img = crop_resize(image, tgt_o, s, H, mode="bilinear")        # (B,1,H,W)
        ctx_img = crop_resize(context_in.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              s, H, mode="bilinear").reshape(B, K, 1, H, W)
        ctx_msk = crop_resize(context_out.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              s, H, mode="nearest").reshape(B, K, 1, H, W)

        refine = self._segment(tgt_img, ctx_img, ctx_msk)                 # (B,1,R,R), same weights
        refine_s = F.interpolate(refine, size=(s, s), mode="bilinear", align_corners=False)
        return fuse_window(coarse_up, refine_s, tgt_o, s)                 # (B,1,H,W)

    def forward(self, image, context_in, context_out, mode="train"):
        """image (B,1,H,W); context_in/out (B,K,1,H,W) → {"final_logit": ...}.

        refine=False → coarse (B,1,R,R). refine=True → native (B,1,H,W) fused output.
        `mode` is accepted for interface parity with the UniverSeg baseline; unused."""
        if not self.refine:
            return {"final_logit": self._segment(image, context_in, context_out)}
        return {"final_logit": self._refine_forward(image, context_in, context_out)}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_patchset_cnn_refine.py -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Regression — bbox_refine tests still pass**

Run: `.venv/bin/python -m pytest tests/test_bbox_refine.py tests/test_patchset_cnn_refine.py -v`
Expected: PASS (8 passed).

- [ ] **Step 9: Review checkpoint**

`PatchSetCNN` refine mode + tests are a self-contained unit. Stop for user review/commit.

---

### Task 3: Config wiring, checkpoint metadata, smoke run, log

**Files:**
- Modify: `experiments/2d/train.py` (`build_model` `patchset_cnn` branch: thread `refine`/`refine_crop` into the `arch` dict)
- Modify: `configs/experiment/2d/model/patchset_cnn.yaml` (add `refine`/`refine_crop` under `arch:`)
- Create: `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`
- Modify: `docs/logs.md` (prepend a dated entry)

**Interfaces:**
- Consumes: `PatchSetCNN(..., refine, refine_crop)` from Task 2.
- Produces: checkpoints whose `arch` block carries `refine`/`refine_crop` so `eval_incontext.py` rebuilds the refine model with zero drift; a runnable `--config-name 2_omnisynth_medseg_refine`.

- [ ] **Step 1: Thread the flags through `build_model`**

In `experiments/2d/train.py`, in `build_model`'s `patchset_cnn` branch, extend the `arch` dict (add the two keys alongside the existing ones, before the closing brace):

```python
            "query_self_attn": a.get("query_self_attn", False),
            "context_id_embed": a.get("context_id_embed", False),
            "max_context": a.get("max_context", 16),
            "refine": a.get("refine", False),
            "refine_crop": a.get("refine_crop", 64),
        }
```

(No other `train.py` change: the loop already pools GT to `final_logit`'s size and computes one BCE + soft-Dice loss, which now lands at native resolution when `refine=True`.)

- [ ] **Step 2: Add config defaults to the model group**

In `configs/experiment/2d/model/patchset_cnn.yaml`, under the `arch:` block (after `max_context: 16`), add:

```yaml
  refine: false              # enable coarse→fine bbox-zoom refinement (native-res fused output)
  refine_crop: 64            # square bbox side length (pixels in the image_size frame)
```

- [ ] **Step 3: Create the runnable experiment leaf**

Create `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`:

```yaml
# Experiment 2 — PatchSetCNN with single-level bbox-zoom refinement, on the same
# omniSynth/MedSeg distribution as experiment 1. Output is native-resolution (fused),
# so val logs `dice`/`dice_soft` at native res; ds_metric_res keeps the coarse-grid Dice.
#   python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine
#   ... vary refine window on the CLI, e.g. arch.refine_crop=48
defaults:
  - 1_omnisynth_medseg
  - _self_

arch:
  refine: true
  # refine_crop: 64          # override to change the zoom window

eval:
  ds_metric_res: [16, 32]    # keep pooled coarse Dice alongside the new native Dice
```

- [ ] **Step 4: Verify config composition (dry run)**

Run:
```bash
.venv/bin/python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine --cfg job 2>/dev/null | grep -E "^ *(refine|refine_crop|model|resolution):|ds_metric_res"
```
Expected: `model: patchset_cnn`, `refine: true`, `refine_crop: 64`, `resolution: 32`, `ds_metric_res: [16, 32]` all present.

- [ ] **Step 5: Smoke run — 1 epoch on a tiny subset**

Run (wandb disabled; a tiny subset to confirm the loop, loss, and checkpoint save under the refine path). In the 2D path the train subset is `data.max_train_samples` (a random subset per epoch) and the val cap is `eval.max_per_label`:
```bash
.venv/bin/python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine \
  train.epochs=1 train.batch_size=4 eval.batch_size=4 \
  data.max_train_samples=64 eval.max_per_label=4 wandb.enabled=false
```
Expected: builds `PatchSetCNN (...)`, trains 1 epoch without shape errors, prints a `val dice=...`, writes `best.pt`.

- [ ] **Step 6: Verify eval reloads the refine checkpoint**

Using the `best.pt` path printed in Step 5:
```bash
.venv/bin/python experiments/2d/eval_incontext.py \
  eval.checkpoint=<path/to/best.pt> wandb.enabled=false eval.max_per_label=4
```
Expected: `Loaded patchset_cnn (...)`, prints `dice/mean=...` with no rebuild error (confirms `arch.refine`/`refine_crop` round-tripped through the checkpoint).

- [ ] **Step 7: Log the change**

Prepend a dated entry to `docs/logs.md` describing: new `PatchSetCNN` `refine`/`refine_crop` mode (two-pass shared-weight bbox zoom + additive logit fusion → native `final_logit`); new `src/models/bbox_refine.py`; `build_model` threads the flags; config group default (off) + `2_omnisynth_medseg_refine.yaml`; note that `refine=True` makes the model log like a native predictor (checkpoint metric `dice` not `cossim`), with `ds_metric_res=[16,32]` preserving coarse-grid Dice; no `train.py`/`eval_incontext.py` loop changes.

- [ ] **Step 8: Review checkpoint**

Full feature wired and smoke-tested end to end. Stop for user review/commit.

---

## Notes for the implementer

- **Why fusion is additive, not replace:** the coarse logit keeps contributing (and receiving gradient) inside the object region, so the coarse pass learns to localize the object — which it must, since it places the next bbox. A hard replace would starve the coarse pass of signal exactly where it needs to be accurate.
- **Bbox origins are detached** (`argmax` on `prob_up.detach()`), so they are hard routing constants; all learning flows through the two `_segment` calls and the additive fuse.
- **Extension points (not now):** a multi-hop ladder (loop `_refine_forward` over a list of crop sizes, chaining the detached composite) and top-k windows (greedy non-overlapping selection + k refine forwards) are deliberately out of scope; the single-window/single-level structure was chosen to keep it readable.

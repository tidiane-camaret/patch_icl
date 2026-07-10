# PatchSetCNN multi-resolution per-level refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise `PatchSetCNN`'s refinement from a single additively-fused native output to a **multi-resolution, per-level** scheme: `arch.resolutions=[32,64]` (effective full-image resolutions, constant token count), per-level losses, and per-level + fused metrics — trainable/evaluable through the existing `experiments/2d/train.py` and `experiments/2d/eval_incontext.py`.

**Architecture:** The same `PatchSetCNN` runs a coarse pass over the full image and one refine pass over a **derived** square crop (`c = image_size·resolutions[0]/resolutions[k]`), both at the same `T=resolutions[0]` token grid (same compute). The forward returns per-level heads (no fusion); the trainer sums a per-level BCE+soft-Dice loss and logs `dice@64` (refine, on the crop) and `dice_fused@64` (the replace-stitch of coarse+refine, metric only); checkpoints select on `dice_fused@{resolutions[-1]}`.

**Tech Stack:** PyTorch 2.5 (`.venv/bin/python`, py3.12, cuda), Hydra, pytest 9.

## Global Constraints

- Python interpreter: `.venv/bin/python` only (never `python`/`uv`/conda). Tests run from repo root: `.venv/bin/python -m pytest ...` (cwd on path → `import src...` and `experiments/2d` imports resolve; test files insert both `"."` and `"experiments/2d"`).
- Branch `patchset-refine`, **commit per task** (user authorized this for the feature). Commit trailer, verbatim last line: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- This revises code already on the branch (the additive-fusion version). `refine`/`refine_crop` are **replaced** by `resolutions`; `_refine_forward` no longer fuses.
- `resolutions` semantics (verbatim): effective full-image resolutions per level; token grid `T=resolutions[0]` constant across levels; derived crop `c_k = image_size·resolutions[0]/resolutions[k]`; each `resolutions[k]` (k≥1) must be a multiple of `resolutions[0]` and divide `image_size·resolutions[0]`. Single-element/absent list = the plain model, byte-for-byte unchanged.
- Metric names (verbatim): `dice@{Rf}` (refine level), `dice_fused@{Rf}` (+ `dice_soft@{Rf}`, `dice_fused_soft@{Rf}`), where `Rf = resolutions[-1]`. Checkpoint selection: `dice_fused@{Rf}/mean` when present.
- Log the change in `docs/logs.md` (final task).

---

### Task 1: `place_window` (replace-stitch) in `bbox_refine.py`

**Files:**
- Modify: `src/models/bbox_refine.py`
- Test: `tests/test_bbox_refine.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `place_window(full, patch, origin, s) -> Tensor (B,1,H,W)` — clone of `full` with `patch (B,1,s,s)` **written (replace)** into the s×s window at each `origin (B,2)`; input not mutated. (Distinct from additive `fuse_window`.)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bbox_refine.py` (update the import line to include `place_window`):

```python
def test_place_window_replaces_not_adds():
    H, s = 16, 8
    full = torch.zeros(2, 1, H, H)
    full[0, 0, 0:s, 0:s] = 1.0                    # non-zero window → REPLACE (not add)
    full[1, 0, 8:16, 8:16] = 1.0
    patch = torch.full((2, 1, s, s), 2.0)
    o = torch.tensor([[0, 0], [8, 8]])
    out = place_window(full, patch, o, s)
    assert out.shape == (2, 1, H, H)
    assert full[0, 0, 0, 0] == 1.0                # input not mutated
    assert out[0, 0, 0:s, 0:s].eq(2.0).all()      # window overwritten to patch (2), not 1+2=3
    assert out[0, 0, s:, s:].eq(0).all()          # outside window untouched
    assert out[1, 0, 8:16, 8:16].eq(2.0).all()
```

Change the file's first import line from
`from src.models.bbox_refine import max_sum_window, gt_window, crop_resize, fuse_window`
to
`from src.models.bbox_refine import max_sum_window, gt_window, crop_resize, fuse_window, place_window`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bbox_refine.py::test_place_window_replaces_not_adds -v`
Expected: FAIL — `ImportError: cannot import name 'place_window'`.

- [ ] **Step 3: Implement `place_window`**

Append to `src/models/bbox_refine.py`:

```python
def place_window(full, patch, origin, s):
    """Return a clone of full (B,1,H,W) with patch (B,1,s,s) WRITTEN (replace) into the s×s
    window at each origin (B,2). Overwrite semantics — used to stitch the refine crop into the
    coarse prediction for the fused metric (cf. additive fuse_window, reserved for a fused loss).
    Input not mutated. Per-sample loop (B is the small batch dim)."""
    out = full.clone()
    for b in range(full.shape[0]):
        r0, c0 = int(origin[b, 0]), int(origin[b, 1])
        out[b, 0, r0:r0 + s, c0:c0 + s] = patch[b, 0]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bbox_refine.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/models/bbox_refine.py tests/test_bbox_refine.py
git commit -m "Add place_window (replace-stitch) to bbox_refine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `PatchSetCNN` `resolutions` + per-level forward

**Files:**
- Modify: `src/models/patchset_cnn.py`
- Test: `tests/test_patchset_cnn_refine.py`

**Interfaces:**
- Consumes: `max_sum_window`, `gt_window`, `crop_resize` from `bbox_refine`.
- Produces:
  - `PatchSetCNN(..., resolutions: list[int] | None = None)` (replaces `refine`/`refine_crop`); attributes `self.resolutions` (list), `self.resolution = resolutions[0]`, `self.refine_crops` (derived px crops, one per level ≥1).
  - `forward(...) -> dict`: single level → `{"final_logit": (B,1,T,T)}`; multi level → `{"final_logit": coarse (B,1,T,T), "refine_logit": (B,1,T,T), "refine_origin": (B,2), "refine_crop": int, "resolutions": list}`.

- [ ] **Step 1: Rewrite the tests**

Replace the entire contents of `tests/test_patchset_cnn_refine.py` with:

```python
import sys; sys.path.insert(0, ".")
import pytest
import torch
from src.models.patchset_cnn import PatchSetCNN


def _model(resolutions, H=32):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolutions[0], enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, resolutions=resolutions)


def _batch(B=2, K=2, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_single_level_unchanged():
    m = _model([8])
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert set(out) == {"final_logit"}
    assert out["final_logit"].shape == (2, 1, 8, 8)
    assert torch.equal(out["final_logit"], m._segment(img, cin, cout))


def test_multi_level_heads_and_derived_crop():
    m = _model([8, 16])                       # image_size 32 → crop = 32*8/16 = 16
    assert m.refine_crops == [16]
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)     # coarse, T=8
    assert out["refine_logit"].shape == (2, 1, 8, 8)    # refine, same T
    assert out["refine_origin"].shape == (2, 2)
    assert out["refine_crop"] == 16
    assert out["resolutions"] == [8, 16]


def test_derived_crop_full_zoom():
    m = _model([8, 32])                       # crop = 32*8/32 = 8
    assert m.refine_crops == [8]


def test_invalid_resolutions_rejected():
    with pytest.raises(AssertionError):
        _model([8, 12])                       # 12 % 8 != 0


def test_grad_reaches_shared_weights_from_both_heads():
    m = _model([8, 16])
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    loss = out["final_logit"].mean() + out["refine_logit"].mean()   # both go through _segment
    loss.backward()
    assert m.decoder[0].weight.grad is not None
    assert m.encoder.stem[0].weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_patchset_cnn_refine.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'resolutions'`.

- [ ] **Step 3: Update the `bbox_refine` import**

In `src/models/patchset_cnn.py`, change:
```python
from src.models.bbox_refine import crop_resize, fuse_window, gt_window, max_sum_window
```
to (drop `fuse_window` — no longer used here; it stays in `bbox_refine` for the future fused loss):
```python
from src.models.bbox_refine import crop_resize, gt_window, max_sum_window
```

- [ ] **Step 4: Replace the constructor head (resolutions logic)**

Replace this block (constructor signature tail + first stored attrs):
```python
        max_context: int = 16,
        refine: bool = False,
        refine_crop: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.resolution = resolution
        self.N = resolution ** 2
        self.query_self_attn = query_self_attn
        self.context_id_embed = context_id_embed
        self.max_context = max_context
        self.refine = refine
        self.refine_crop = refine_crop
        self.encoder = ConvEncoder(1, tuple(enc_dims), resolution)
```
with:
```python
        max_context: int = 16,
        resolutions: list[int] | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        # `resolutions` = effective full-image resolutions per level (level 0 = coarse over the
        # full image). The token grid T is constant across levels and equals resolutions[0]; each
        # further level k crops the image to c_k = image_size*resolutions[0]/resolutions[k] px so
        # its T tokens resolve a finer effective resolution. None → single level = plain model.
        self.resolutions = [resolution] if resolutions is None else [int(r) for r in resolutions]
        assert len(self.resolutions) <= 2, \
            "multi-hop refinement (>2 levels) not implemented yet; use resolutions=[T] or [T, R1]"
        resolution = self.resolutions[0]                 # token grid T (drives the encoder)
        self.resolution = resolution
        self.N = resolution ** 2
        self.query_self_attn = query_self_attn
        self.context_id_embed = context_id_embed
        self.max_context = max_context
        # Derived per-level crop sizes (px in the image_size frame); level 0 is the full image.
        self.refine_crops = []
        for rk in self.resolutions[1:]:
            assert rk % resolution == 0 and (image_size * resolution) % rk == 0, \
                f"resolutions[k]={rk} must be a multiple of resolutions[0]={resolution} and " \
                f"divide image_size*resolutions[0]={image_size * resolution}"
            c = image_size * resolution // rk
            assert 0 < c <= image_size, f"derived crop {c} out of range for resolutions[k]={rk}"
            self.refine_crops.append(c)
        self.encoder = ConvEncoder(1, tuple(enc_dims), resolution)
```

- [ ] **Step 5: Replace `_refine_forward` and `forward`**

Replace the whole `_refine_forward` method and the whole `forward` method (the additive-fusion versions) with:
```python
    def _refine_forward(self, image, context_in, context_out):
        """Coarse pass over the full image + one bbox-zoom refine pass (SAME weights) → per-level
        heads. Crop the target on its densest predicted region and each context on its densest GT,
        resize crops to the encoder input, re-segment at the same T-token grid. No fusion — levels
        are supervised/metricked separately (the fused stitch is a metric only, built elsewhere)."""
        B, K = context_in.shape[0], context_in.shape[1]
        H, W = image.shape[-2:]
        c = self.refine_crops[0]                                          # derived crop (px)

        coarse = self._segment(image, context_in, context_out)           # (B,1,T,T)
        prob_up = F.interpolate(torch.sigmoid(coarse).detach(), size=(H, W),
                                mode="bilinear", align_corners=False)     # bbox selection only
        tgt_o = max_sum_window(prob_up, c)                               # (B,2) px origin
        ctx_o = torch.stack([gt_window(context_out[:, k], c) for k in range(K)], dim=1)  # (B,K,2)

        tgt_img = crop_resize(image, tgt_o, c, H, mode="bilinear")       # (B,1,H,W)
        ctx_img = crop_resize(context_in.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              c, H, mode="bilinear").reshape(B, K, 1, H, W)
        ctx_msk = crop_resize(context_out.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              c, H, mode="nearest").reshape(B, K, 1, H, W)

        refine = self._segment(tgt_img, ctx_img, ctx_msk)                # (B,1,T,T), same weights
        return {"final_logit": coarse, "refine_logit": refine,
                "refine_origin": tgt_o, "refine_crop": c, "resolutions": self.resolutions}

    def forward(self, image, context_in, context_out, mode="train"):
        """image (B,1,H,W); context_in/out (B,K,1,H,W).

        Single level (len(resolutions)==1): {"final_logit": (B,1,T,T)} — the plain model.
        Multi level: per-level heads (final_logit=coarse, refine_logit, refine_origin,
        refine_crop, resolutions). `mode` is accepted for interface parity; unused."""
        if len(self.resolutions) == 1:
            return {"final_logit": self._segment(image, context_in, context_out)}
        return self._refine_forward(image, context_in, context_out)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_patchset_cnn_refine.py tests/test_bbox_refine.py -v`
Expected: PASS (5 + 6 = 11 passed).

- [ ] **Step 7: Commit**

```bash
git add src/models/patchset_cnn.py tests/test_patchset_cnn_refine.py
git commit -m "PatchSetCNN: replace additive-fusion refine with multi-resolution per-level heads

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `refine_geometry` helper + per-level/fused metrics in `validate()`

**Files:**
- Modify: `experiments/2d/evaluate.py`
- Test: `tests/test_refine_geometry.py`

**Interfaces:**
- Consumes: `crop_resize`, `place_window` from `bbox_refine`; `hard_dice`, `soft_dice`, `log_summary` from `common` (already imported in evaluate.py).
- Produces:
  - `refine_geometry(out, lbl) -> dict | None` — `None` when `out` has no `"refine_logit"`; else `{"refine_prob": (B,1,T,T), "refine_target": (B,1,T,T), "fused_R": (B,1,Rf,Rf), "gt_R": (B,1,Rf,Rf), "Rf": int}`. Must be called under `no_grad` (callers guarantee it).
  - `validate(...)` additionally emits `dice@{Rf}`, `dice_soft@{Rf}`, `dice_fused@{Rf}`, `dice_fused_soft@{Rf}` (each with `/mean`) for the refine model.

- [ ] **Step 1: Write the failing test**

Create `tests/test_refine_geometry.py`:

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import torch
from evaluate import refine_geometry


def _out(B=2, T=4):
    torch.manual_seed(0)
    return {"final_logit": torch.randn(B, 1, T, T),
            "refine_logit": torch.randn(B, 1, T, T),
            "refine_origin": torch.tensor([[0, 0], [8, 8]]),
            "refine_crop": 8, "resolutions": [4, 8]}


def test_none_for_single_level():
    assert refine_geometry({"final_logit": torch.randn(2, 1, 4, 4)},
                           torch.rand(2, 1, 16, 16)) is None


def test_shapes_and_ranges():
    out = _out()
    lbl = (torch.rand(2, 1, 16, 16) > 0.5).float()
    rg = refine_geometry(out, lbl)
    assert rg["Rf"] == 8
    assert rg["refine_prob"].shape == (2, 1, 4, 4)
    assert rg["refine_target"].shape == (2, 1, 4, 4)
    assert rg["fused_R"].shape == (2, 1, 8, 8)
    assert rg["gt_R"].shape == (2, 1, 8, 8)
    assert (rg["fused_R"] >= 0).all() and (rg["fused_R"] <= 1).all()   # probabilities


def test_fused_takes_refine_inside_window():
    # coarse all -inf (prob 0), refine all +inf (prob 1) → fused prob is 1 inside the crop
    # window and 0 outside. With origin (0,0) crop 8 on a 16 image, pooled-to-8 fused should be
    # 1 in the top-left 4x4 (the crop) and 0 elsewhere.
    B, T = 1, 4
    out = {"final_logit": torch.full((B, 1, T, T), -30.0),
           "refine_logit": torch.full((B, 1, T, T), 30.0),
           "refine_origin": torch.tensor([[0, 0]]), "refine_crop": 8, "resolutions": [4, 8]}
    rg = refine_geometry(out, torch.zeros(B, 1, 16, 16))
    f = rg["fused_R"][0, 0]
    assert f[:4, :4].mean() > 0.99            # crop region takes refine (prob 1)
    assert f[4:, 4:].mean() < 0.01            # outside stays coarse (prob 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_refine_geometry.py -v`
Expected: FAIL — `ImportError: cannot import name 'refine_geometry'`.

- [ ] **Step 3: Add the `bbox_refine` import to evaluate.py**

In `experiments/2d/evaluate.py`, after the `from common import (...)` block, add:
```python
from src.models.bbox_refine import crop_resize, place_window
```

- [ ] **Step 4: Implement `refine_geometry`**

In `experiments/2d/evaluate.py`, add this function just above `def validate(`:
```python
def refine_geometry(out: dict, lbl: torch.Tensor) -> dict | None:
    """Per-level + fused tensors for a multi-resolution refine output; None if single-level.

    out: model output; multi-level has coarse `final_logit` (B,1,T,T), `refine_logit` (B,1,T,T),
    `refine_origin` (B,2 px), `refine_crop` (int px), `resolutions` (list). lbl: (B,1,H,W) GT.
    Returns detached maps for metrics (call under no_grad):
      refine_prob   (B,1,T,T)  sigmoid(refine_logit)
      refine_target (B,1,T,T)  crop_resize(lbl, origin, c, T) — soft cropped GT
      fused_R/gt_R  (B,1,Rf,Rf) fused prob (coarse with refine placed in the crop) and GT,
                    both avg-pooled to Rf = resolutions[-1]
    """
    if "refine_logit" not in out:
        return None
    coarse = out["final_logit"].float()
    refine = out["refine_logit"].float()
    origin = out["refine_origin"]
    c = int(out["refine_crop"])
    Rf = int(out["resolutions"][-1])
    T = refine.shape[-1]
    H = lbl.shape[-1]
    refine_prob = torch.sigmoid(refine)
    refine_target = crop_resize(lbl, origin, c, T, mode="bilinear")
    coarse_up = F.interpolate(torch.sigmoid(coarse), size=(H, H),
                              mode="bilinear", align_corners=False)
    refine_up = F.interpolate(refine_prob, size=(c, c), mode="bilinear", align_corners=False)
    fused = place_window(coarse_up, refine_up, origin, c)              # (B,1,H,H) native stitch
    return {"refine_prob": refine_prob, "refine_target": refine_target,
            "fused_R": F.adaptive_avg_pool2d(fused, (Rf, Rf)),
            "gt_R": F.adaptive_avg_pool2d(lbl, (Rf, Rf)), "Rf": Rf}
```

- [ ] **Step 5: Run the helper test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_refine_geometry.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Wire the metrics into `validate()`**

(6a) After the `cos_ds, cos_lab = ...` / `topk_ds, topk_lab = ...` accumulator declarations near the top of `validate()`, add:
```python
    ref_h_ds, ref_h_lab = defaultdict(list), defaultdict(list)   # refine hard dice@Rf
    ref_s_ds, ref_s_lab = defaultdict(list), defaultdict(list)   # refine soft
    fus_h_ds, fus_h_lab = defaultdict(list), defaultdict(list)   # fused hard dice_fused@Rf
    fus_s_ds, fus_s_lab = defaultdict(list), defaultdict(list)   # fused soft
    fused_res = None                                             # resolutions[-1] when refine
```

(6b) Right after `logit = out["final_logit"].float()`, add:
```python
        rg = refine_geometry(out, lbl)
        if rg is not None:
            fused_res = rg["Rf"]
```

(6c) Inside the `for b in range(B):` loop, after the existing `table.add_data(...)` line, add:
```python
            if rg is not None:
                ref_h_ds[ds].append(hard_dice(rg["refine_prob"][b, 0], rg["refine_target"][b, 0]))
                ref_h_lab[key].append(hard_dice(rg["refine_prob"][b, 0], rg["refine_target"][b, 0]))
                ref_s_ds[ds].append(soft_dice(rg["refine_prob"][b, 0], rg["refine_target"][b, 0]))
                ref_s_lab[key].append(soft_dice(rg["refine_prob"][b, 0], rg["refine_target"][b, 0]))
                fus_h_ds[ds].append(hard_dice(rg["fused_R"][b, 0], rg["gt_R"][b, 0]))
                fus_h_lab[key].append(hard_dice(rg["fused_R"][b, 0], rg["gt_R"][b, 0]))
                fus_s_ds[ds].append(soft_dice(rg["fused_R"][b, 0], rg["gt_R"][b, 0]))
                fus_s_lab[key].append(soft_dice(rg["fused_R"][b, 0], rg["gt_R"][b, 0]))
```

(6d) In the summary-building section, after the `for R in res_list:` pooled-Dice block (the one ending with the `dice_ds_soft@{R}` update), add:
```python
    if fused_res is not None:                             # refine model: per-level + fused Dice
        summary.update(log_summary(ref_h_ds, ref_h_lab, prefix=f"dice@{fused_res}",
                                   metric_label=f"refine@{fused_res}", per_group=per_group))
        summary.update(log_summary(ref_s_ds, ref_s_lab, prefix=f"dice_soft@{fused_res}",
                                   metric_label=f"refine soft@{fused_res}", per_group=per_group))
        summary.update(log_summary(fus_h_ds, fus_h_lab, prefix=f"dice_fused@{fused_res}",
                                   metric_label=f"fused@{fused_res}", per_group=per_group))
        summary.update(log_summary(fus_s_ds, fus_s_lab, prefix=f"dice_fused_soft@{fused_res}",
                                   metric_label=f"fused soft@{fused_res}", per_group=per_group))
```

- [ ] **Step 7: Verify the helper tests still pass and evaluate.py imports cleanly**

Run: `.venv/bin/python -m pytest tests/test_refine_geometry.py -v && .venv/bin/python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'experiments/2d'); import evaluate; print('evaluate imports OK')"`
Expected: PASS (3 passed) and `evaluate imports OK`.

- [ ] **Step 8: Commit**

```bash
git add experiments/2d/evaluate.py tests/test_refine_geometry.py
git commit -m "evaluate: refine_geometry helper + per-level/fused val metrics

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Trainer — per-level loss, train metrics, `resolutions` wiring, fused checkpoint selection

**Files:**
- Modify: `experiments/2d/train.py`
- Test: `tests/test_select_metric.py`

**Interfaces:**
- Consumes: `refine_geometry` (evaluate.py), `crop_resize` (bbox_refine), `PatchSetCNN(resolutions=...)`.
- Produces: `_select_metric(summary) -> (str, float)` — `(metric_key_without_/mean, value)`; prefers `dice_fused@*` (non-soft), else `cossim`, else `dice`. Train logs `train/dice@{Rf}` and `train/dice_fused@{Rf}` via `dsr_out`. Checkpoints select on the fused metric when present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_select_metric.py`:

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
from train import _select_metric


def test_prefers_fused_hard_not_soft():
    s = {"dice_fused@64/mean": 0.7, "dice_fused_soft@64/mean": 0.6,
         "cossim/mean": 0.5, "dice/mean": 0.4}
    assert _select_metric(s) == ("dice_fused@64", 0.7)


def test_falls_back_to_cossim():
    assert _select_metric({"cossim/mean": 0.5, "dice/mean": 0.4}) == ("cossim", 0.5)


def test_falls_back_to_dice():
    assert _select_metric({"dice/mean": 0.4}) == ("dice", 0.4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_select_metric.py -v`
Expected: FAIL — `ImportError: cannot import name '_select_metric'`.

- [ ] **Step 3: Add imports**

In `experiments/2d/train.py`, change:
```python
from evaluate import validate, _target_like, _upsample_to, _as_res_list
```
to:
```python
from evaluate import validate, _target_like, _upsample_to, _as_res_list, refine_geometry
from src.models.bbox_refine import crop_resize
```

- [ ] **Step 4: Add `_select_metric` (module level)**

In `experiments/2d/train.py`, add this function just above `def main(` (after `train_epoch`):
```python
def _select_metric(summary: dict) -> tuple[str, float]:
    """Checkpoint-selection metric from a val summary: prefer the fused refine metric
    (dice_fused@R, hard — not its _soft sibling), else cossim, else dice. Returns
    (metric_key_without_'/mean', mean_value)."""
    fused = next((k for k in summary
                  if k.startswith("dice_fused@") and "soft" not in k and k.endswith("/mean")),
                 None)
    if fused is not None:
        return fused[: -len("/mean")], summary[fused]
    m = "cossim" if "cossim/mean" in summary else "dice"
    return m, summary.get(f"{m}/mean", float("nan"))
```

- [ ] **Step 5: Thread `resolutions` through `build_model`**

In `build_model`'s `patchset_cnn` branch, replace the two refine lines in the `arch` dict:
```python
            "refine": a.get("refine", False),
            "refine_crop": a.get("refine_crop", 64),
```
with:
```python
            "resolutions": list(a.resolutions) if a.get("resolutions", None) is not None else None,
```

- [ ] **Step 6: Add the per-level refine loss**

In `train_epoch`, right after `loss = bce + cfg.train.dice_weight * dice`, add:
```python
        if out.get("refine_logit") is not None:            # multi-level: add the refine loss
            rlogit = out["refine_logit"].float()
            rtarget = crop_resize(lbl, out["refine_origin"], int(out["refine_crop"]),
                                  rlogit.shape[-1], mode="bilinear")   # soft cropped GT at T
            rbce = F.binary_cross_entropy_with_logits(rlogit, rtarget)
            rdice = soft_dice_loss(torch.sigmoid(rlogit), rtarget)
            loss = loss + float(cfg.train.get("refine_loss_weight", 1.0)) * (
                rbce + cfg.train.dice_weight * rdice)
```

- [ ] **Step 7: Add the train-side per-level + fused metric sums**

(7a) In `train_epoch`, next to the other running-sum initializers (near `lr_hard_sum = lr_hard_cnt = 0.0`), add:
```python
    refine_hard_sum = refine_hard_cnt = 0.0   # refine hard Dice at Rf (on the crop)
    fused_hard_sum = fused_hard_cnt = 0.0      # fused hard Dice at Rf (stitched full image)
    fused_res = None
```

(7b) Inside the `with torch.no_grad():` monitoring block, after the `for R in res_list:` pooled-Dice loop, add:
```python
            rg = refine_geometry(out, lbl)
            if rg is not None:
                rh, rhc = _hard_sum(rg["refine_prob"], (rg["refine_target"] >= 0.5).float())
                fh, fhc = _hard_sum(rg["fused_R"], (rg["gt_R"] >= 0.5).float())
                refine_hard_sum += rh; refine_hard_cnt += rhc
                fused_hard_sum += fh; fused_hard_cnt += fhc
                fused_res = rg["Rf"]
```

(7c) In `train_epoch`, after the `if low_res is not None:` block that fills `dsr_out` (just before the `return (...)`), add:
```python
    if fused_res is not None:                   # refine model: per-level + fused train Dice
        dsr_out[f"dice@{fused_res}"] = float(refine_hard_sum) / max(float(refine_hard_cnt), 1)
        dsr_out[f"dice_fused@{fused_res}"] = float(fused_hard_sum) / max(float(fused_hard_cnt), 1)
```
(These flow to wandb as `train/dice@{Rf}` / `train/dice_fused@{Rf}` via the existing
`log.update({f"train/{k}": v for k, v in train_dsr.items()})` in `main`.)

- [ ] **Step 8: Use `_select_metric` for checkpointing**

In `main`, replace:
```python
            metric = "cossim" if "cossim/mean" in summary else "dice"
            mean_dice = summary.get(f"{metric}/mean", float("nan"))
```
with:
```python
            metric, mean_dice = _select_metric(summary)
```

- [ ] **Step 9: Run the unit test + full test suite**

Run: `.venv/bin/python -m pytest tests/test_select_metric.py tests/test_bbox_refine.py tests/test_patchset_cnn_refine.py tests/test_refine_geometry.py -v`
Expected: PASS (3 + 6 + 5 + 3 = 17 passed).

- [ ] **Step 10: Commit**

```bash
git add experiments/2d/train.py tests/test_select_metric.py
git commit -m "train: per-level refine loss + train/val fused metrics + fused checkpoint selection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Config, smoke run, log

**Files:**
- Modify: `configs/experiment/2d/model/patchset_cnn.yaml`
- Modify: `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`
- Modify: `docs/logs.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a runnable `--config-name 2_omnisynth_medseg_refine` (multi-resolution refine) whose checkpoint `arch` carries `resolutions`.

- [ ] **Step 1: Remove the old refine flags from the model group**

In `configs/experiment/2d/model/patchset_cnn.yaml`, delete the two lines under `arch:`:
```yaml
  refine: false              # enable coarse→fine bbox-zoom refinement (native-res fused output)
  refine_crop: 64            # square bbox side length (pixels in the image_size frame)
```
(Leave `resolution` and the rest of the `arch:` block unchanged.)

- [ ] **Step 2: Rewrite the experiment leaf**

Replace the entire contents of `configs/experiment/2d/2_omnisynth_medseg_refine.yaml` with:
```yaml
# Experiment 2 — PatchSetCNN with multi-resolution per-level bbox-zoom refinement, on the same
# omniSynth/MedSeg distribution as experiment 1. resolutions = effective full-image resolutions;
# token count is constant (T=32), the refine crop is derived (128·32/64 = 64px). Per-level losses
# (coarse@32 + refine@64 on the crop); checkpoint selects on dice_fused@64.
#   python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine
#   ... deeper zoom: arch.resolutions=[32,128]   (32px crop, effective 128)
defaults:
  - 1_omnisynth_medseg
  - _self_

arch:
  resolutions: [32, 64]      # effective full-image resolutions; T=32 tokens/level, crop=64px derived

train:
  refine_loss_weight: 1.0    # weight of the refine-level loss relative to the coarse loss

eval:
  ds_metric_res: [16, 32]    # coarse-grid pooled Dice, comparable to the plain model
```

- [ ] **Step 3: Verify config composition (dry run)**

Run:
```bash
.venv/bin/python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine --cfg job 2>/dev/null | grep -E "^ *(resolutions|resolution|model|refine_loss_weight):|- 32|- 64|ds_metric_res"
```
Expected: `model: patchset_cnn`, a `resolutions:` list with `- 32` / `- 64`, `refine_loss_weight: 1.0`, and `ds_metric_res` present.

- [ ] **Step 4: Smoke train — 1 epoch, tiny subset, assert fused metric is logged**

Run:
```bash
.venv/bin/python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine \
  train.epochs=1 train.batch_size=4 eval.batch_size=4 \
  data.max_train_samples=64 eval.max_per_label=4 wandb.enabled=false 2>&1 | tee /tmp/refine_smoke.log
grep -E "dice_fused@64|val dice_fused" /tmp/refine_smoke.log || echo "MISSING dice_fused@64"
```
Expected: builds `PatchSetCNN`, trains 1 epoch without shape errors, the console `val ...=` line reports the `dice_fused@64` selection metric, and `best.pt` is written. (The `grep` confirms the fused metric surfaced; it must NOT print `MISSING`.)

- [ ] **Step 5: Verify eval reload of the refine checkpoint**

Using the `best.pt` path printed in Step 4:
```bash
.venv/bin/python experiments/2d/eval_incontext.py \
  eval.checkpoint=<path/to/best.pt> wandb.enabled=false eval.max_per_label=4
```
Expected: `Loaded patchset_cnn (...)` with no rebuild error (confirms `arch.resolutions` round-tripped), prints `dice/mean=...`.

- [ ] **Step 6: Log the change**

Prepend a dated entry to `docs/logs.md` covering: `PatchSetCNN` refine reworked to multi-resolution per-level (`arch.resolutions` = effective full-image resolutions, constant token count T, derived crop `image_size·R0/Rk`); replaced `refine`/`refine_crop`; per-level losses (`train.refine_loss_weight`); new `place_window` (replace) + `refine_geometry` helper; metrics `dice@64` / `dice_fused@64` (+ soft) in train and val; checkpoint selection on `dice_fused@{resolutions[-1]}`; `fuse_window` retained for a future fused loss; note old additive-fusion checkpoints (with `arch.refine`) no longer reload — retrain.

- [ ] **Step 7: Commit**

```bash
git add configs/experiment/2d/model/patchset_cnn.yaml configs/experiment/2d/2_omnisynth_medseg_refine.yaml docs/logs.md
git commit -m "config+docs: multi-resolution refine leaf (arch.resolutions) + logs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Effective resolution ≠ token count.** Every pass emits `T=resolutions[0]` tokens; the refine pass just looks at a smaller crop, so those `T` tokens resolve a finer *effective* full-image resolution. This is why compute is equal across levels and why `_segment` needs no resolution argument.
- **No fusion in the model.** The forward returns per-level logits. The only "fusion" is the metric-only replace-stitch (`place_window`) inside `refine_geometry`. `fuse_window` (additive, logit-space) stays in `bbox_refine` unused, reserved for a future fused *loss*.
- **Bbox origins are detached** (`argmax` on `sigmoid(coarse).detach()`); learning flows only through the two `_segment` calls.
- **Old checkpoints:** additive-fusion `best.pt` files from the previous plan carry `arch.refine`/`refine_crop` and will not rebuild under the new constructor — they are throwaway smoke artifacts; retrain.
```

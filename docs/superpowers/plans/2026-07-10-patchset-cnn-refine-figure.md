# PatchSetCNN refine qualitative figure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 2×3 qualitative figure for multi-resolution refine `PatchSetCNN` checkpoints that visualizes the coarse→fine bbox-zoom flow (sampled bboxes, per-level predictions, fused stitch), emitted during eval when `eval.save_figures=true`.

**Architecture:** Two thin data additions surface geometry the figure needs (context bbox origins from the model; the native fused map from `refine_geometry`), then a backend-agnostic `save_refine_figure` renders the panel and is wired into `validate()`'s existing gated figure block as an *additional* figure alongside the standard one.

**Tech Stack:** PyTorch, matplotlib (Agg backend, already configured in `evaluate.py`), pytest.

## Global Constraints

- Python interpreter: `.venv/bin/python` only (never bare `python`, `uv`, or `conda`).
- Version control: feature work on the current branch; **commit per task** (user authorized for this line of work). Commit trailer, verbatim: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Log the change in `docs/logs.md` (project convention).
- Design spec (source of truth): `docs/superpowers/specs/2026-07-10-patchset-cnn-refine-figure-design.md`.
- Tests only where they add real value (project guideline); reuse existing test file patterns.

---

### Task 1: Surface the figure's geometry (model context origins + native fused map)

**Files:**
- Modify: `src/models/patchset_cnn.py` (the `_refine_forward` return dict, ~`:262`)
- Modify: `experiments/2d/evaluate.py` (the `refine_geometry` return dict, ~`:195`)
- Test: `tests/test_patchset_cnn_refine.py`, `tests/test_refine_geometry.py`

**Interfaces:**
- Consumes: existing `PatchSetCNN._refine_forward` (already computes `ctx_o`, `fused`-precursor) and `refine_geometry(out, lbl)`.
- Produces:
  - `PatchSetCNN.forward(...)` multi-level output gains key `"refine_ctx_origin"` → tensor `(B, K, 2)` integer px top-left origins (the context crops).
  - `refine_geometry(out, lbl)` return dict gains key `"fused"` → tensor `(B, 1, H, H)` native stitched probability map (coarse with refine placed in the crop).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patchset_cnn_refine.py`:
```python
def test_multi_level_returns_ctx_origin():
    m = _model([8, 16])                       # image_size 32 → crop = 16
    img, cin, cout = _batch(B=2, K=2, H=32)
    out = m(img, context_in=cin, context_out=cout)
    assert out["refine_ctx_origin"].shape == (2, 2, 2)          # (B, K, 2)
    # origins are in-bounds top-left px for a crop of size 16 on a 32 image
    assert (out["refine_ctx_origin"] >= 0).all()
    assert (out["refine_ctx_origin"] <= 32 - 16).all()


def test_single_level_has_no_ctx_origin():
    m = _model([8])
    img, cin, cout = _batch(H=32)
    assert "refine_ctx_origin" not in m(img, context_in=cin, context_out=cout)
```

Append to `tests/test_refine_geometry.py`:
```python
def test_returns_native_fused_map():
    out = _out()
    lbl = (torch.rand(2, 1, 16, 16) > 0.5).float()
    rg = refine_geometry(out, lbl)
    assert rg["fused"].shape == (2, 1, 16, 16)                  # native H×H
    assert (rg["fused"] >= 0).all() and (rg["fused"] <= 1).all()   # probabilities
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_patchset_cnn_refine.py::test_multi_level_returns_ctx_origin tests/test_refine_geometry.py::test_returns_native_fused_map -v`
Expected: FAIL — `KeyError: 'refine_ctx_origin'` and `KeyError: 'fused'`.

- [ ] **Step 3: Add `refine_ctx_origin` to the model output**

In `src/models/patchset_cnn.py`, in `_refine_forward`, change the return statement:
```python
        return {"final_logit": coarse, "refine_logit": refine,
                "refine_origin": tgt_o, "refine_crop": c, "resolutions": self.resolutions}
```
to:
```python
        return {"final_logit": coarse, "refine_logit": refine,
                "refine_origin": tgt_o, "refine_ctx_origin": ctx_o,
                "refine_crop": c, "resolutions": self.resolutions}
```
(`ctx_o` is already computed just above as `torch.stack([gt_window(...) ...], dim=1)`.)

- [ ] **Step 4: Add the native `fused` map to `refine_geometry`**

In `experiments/2d/evaluate.py`, in `refine_geometry`, change the return statement:
```python
    return {"refine_prob": refine_prob, "refine_target": refine_target,
            "fused_R": F.adaptive_avg_pool2d(fused, (Rf, Rf)),
            "gt_R": F.adaptive_avg_pool2d(lbl, (Rf, Rf)), "Rf": Rf}
```
to:
```python
    return {"refine_prob": refine_prob, "refine_target": refine_target,
            "fused": fused,
            "fused_R": F.adaptive_avg_pool2d(fused, (Rf, Rf)),
            "gt_R": F.adaptive_avg_pool2d(lbl, (Rf, Rf)), "Rf": Rf}
```
(`fused` is already built above as `place_window(coarse_up, refine_up, origin, c)`.)

- [ ] **Step 5: Run the refine test suite to verify pass**

Run: `.venv/bin/python -m pytest tests/test_patchset_cnn_refine.py tests/test_refine_geometry.py -v`
Expected: PASS — all existing tests plus the 3 new ones (`test_multi_level_returns_ctx_origin`, `test_single_level_has_no_ctx_origin`, `test_returns_native_fused_map`).

- [ ] **Step 6: Commit**

```bash
git add src/models/patchset_cnn.py experiments/2d/evaluate.py tests/test_patchset_cnn_refine.py tests/test_refine_geometry.py
git commit -m "refine: expose context bbox origins + native fused map for figures

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `save_refine_figure` render function + `validate()` wiring

**Files:**
- Modify: `experiments/2d/evaluate.py` (add `_refine_overlay_ax`, `save_refine_figure`; wire into the gated figure block ~`:320`; add the `Rectangle` import)
- Test: `tests/test_refine_figure.py` (create)
- Modify: `docs/logs.md`

**Interfaces:**
- Consumes (from Task 1): `out["refine_ctx_origin"]` `(B,K,2)`, `rg["fused"]` `(B,1,H,H)`; plus existing `out["refine_origin"]` `(B,2)`, `out["refine_crop"]` int, `rg["refine_prob"]` `(B,1,T,T)`, and the `validate()` locals `img/lbl/cin/cout/prob_nat/rg`.
- Produces:
  - `save_refine_figure(tgt_image, tgt_gt, ctx_image, ctx_gt, coarse_pred, fused_pred, refine_pred, tgt_box, ctx_box, out_path, title="")` — writes a 2×3 PNG. `*_image`/`*_gt`/`coarse_pred`/`fused_pred` are `(H,W)` numpy; `refine_pred` is `(T,T)` numpy; `tgt_box`/`ctx_box` are `(r0, c0, size)` int tuples; `out_path` is a `pathlib.Path`.
  - A `{ds}_l{lv}_refine.png` per `(ds, lv)` for refine checkpoints (+ `figures_refine/{ds}/label_{lv}` in wandb when `figures_to_wandb`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_refine_figure.py`:
```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import numpy as np
from pathlib import Path
from evaluate import save_refine_figure


def _img(n=16):
    rng = np.random.default_rng(0)
    return rng.random((n, n)).astype("float32")


def test_writes_png(tmp_path):
    p = tmp_path / "sub" / "case_refine.png"       # parent dir does not exist yet
    save_refine_figure(
        tgt_image=_img(), tgt_gt=(_img() > 0.5).astype("float32"),
        ctx_image=_img(), ctx_gt=(_img() > 0.5).astype("float32"),
        coarse_pred=_img(), fused_pred=_img(), refine_pred=_img(8),   # T=8 over a 16 image
        tgt_box=(2, 3, 8), ctx_box=(4, 4, 8), out_path=p, title="t")
    assert p.exists() and p.stat().st_size > 0


def test_border_clamped_box_ok(tmp_path):
    p = tmp_path / "border_refine.png"
    # box flush against the bottom-right corner (origin + size == H): must not raise
    save_refine_figure(
        tgt_image=_img(), tgt_gt=(_img() > 0.5).astype("float32"),
        ctx_image=_img(), ctx_gt=(_img() > 0.5).astype("float32"),
        coarse_pred=_img(), fused_pred=_img(), refine_pred=_img(8),
        tgt_box=(8, 8, 8), ctx_box=(0, 0, 8), out_path=p, title="t")
    assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_refine_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_refine_figure'`.

- [ ] **Step 3: Add the `Rectangle` import**

In `experiments/2d/evaluate.py`, just after `import matplotlib.pyplot as plt` (near `:18`), add:
```python
from matplotlib.patches import Rectangle
```

- [ ] **Step 4: Add `_refine_overlay_ax` and `save_refine_figure`**

In `experiments/2d/evaluate.py`, add just after `save_figure` (i.e. after its `plt.close(fig)`, ~`:91`):
```python
def _refine_overlay_ax(ax, image, title, *, gt=None, pred=None, pred_extent=None, boxes=()):
    """Gray base + optional pred heat (Reds) + optional GT contour (lime) + bbox rectangles.
    pred_extent stretches a coarse pred map over the crop; None = pixel-aligned to `image`."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if pred is not None:
        ax.imshow(pred, cmap="Reds", alpha=0.45, vmin=0, vmax=1, extent=pred_extent)
    if gt is not None and float(gt.max()) > 0:      # contour needs a level present
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.0)
    for (r0, c0, s, color) in boxes:
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), s, s, fill=False, edgecolor=color, lw=1.5))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_refine_figure(
    tgt_image, tgt_gt, ctx_image, ctx_gt,          # full-frame (H,W)
    coarse_pred, fused_pred,                       # target preds (H,W): res0, fused
    refine_pred,                                   # target refine pred (T,T)
    tgt_box, ctx_box,                              # (r0, c0, size) int px
    out_path, title="",
):
    """2×3 refine panel. Row 0 = target, row 1 = first context; col 2 row 1 is empty.
    Col 0: full frame + GT contour + (res0 pred / bbox). Col 1: bbox crop + GT contour +
    (res1 pred on target). Col 2: full frame + GT contour + fused pred (target only)."""
    tr0, tc0, tc = tgt_box
    cr0, cc0, cc = ctx_box
    tgt_crop     = tgt_image[tr0:tr0 + tc, tc0:tc0 + tc]
    tgt_crop_gt  = tgt_gt[tr0:tr0 + tc, tc0:tc0 + tc]
    ctx_crop     = ctx_image[cr0:cr0 + cc, cc0:cc0 + cc]
    ctx_crop_gt  = ctx_gt[cr0:cr0 + cc, cc0:cc0 + cc]
    # refine_pred is T×T over the tc×tc crop: stretch it across the crop's display extent
    crop_extent  = (-0.5, tc - 0.5, tc - 0.5, -0.5)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.5), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.05)

    _refine_overlay_ax(axes[0, 0], tgt_image, "Target + GT + res0 pred",
                       gt=tgt_gt, pred=coarse_pred, boxes=[(tr0, tc0, tc, "yellow")])
    _refine_overlay_ax(axes[1, 0], ctx_image, "Ctx0 + GT",
                       gt=ctx_gt, boxes=[(cr0, cc0, cc, "cyan")])
    _refine_overlay_ax(axes[0, 1], tgt_crop, "Target crop + GT + res1 pred",
                       gt=tgt_crop_gt, pred=refine_pred, pred_extent=crop_extent)
    _refine_overlay_ax(axes[1, 1], ctx_crop, "Ctx0 crop + GT", gt=ctx_crop_gt)
    _refine_overlay_ax(axes[0, 2], tgt_image, "Target + GT + fused pred",
                       gt=tgt_gt, pred=fused_pred)
    axes[1, 2].axis("off")

    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_refine_figure.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Wire the refine figure into `validate()`**

In `experiments/2d/evaluate.py`, inside the gated figure block, immediately after the existing
`save_figure(...)` call and its `if figures.get("to_wandb"): wandb.log(...)` (i.e. right after
line ~`:336`, still inside the `if figures and fig_key not in saved and len(saved) < max_fig:`
block), add:
```python
                if rg is not None:               # refine model: extra coarse→fine panel
                    c_px = int(out["refine_crop"])
                    fig_path_refine = Path(figures["out_dir"]) / f"{ds}_l{lv}_refine.png"
                    tr0, tc0 = (int(v) for v in out["refine_origin"][b])
                    cr0, cc0 = (int(v) for v in out["refine_ctx_origin"][b, 0])
                    save_refine_figure(
                        tgt_image=img[b, 0].cpu().numpy(), tgt_gt=lbl[b, 0].cpu().numpy(),
                        ctx_image=cin[b, 0, 0].cpu().numpy(), ctx_gt=cout[b, 0, 0].cpu().numpy(),
                        coarse_pred=prob_nat[b, 0].cpu().numpy(),
                        fused_pred=rg["fused"][b, 0].cpu().numpy(),
                        refine_pred=rg["refine_prob"][b, 0].cpu().numpy(),
                        tgt_box=(tr0, tc0, c_px), ctx_box=(cr0, cc0, c_px),
                        out_path=fig_path_refine,
                        title=f"{ds} label={lv} sample={si} refine")
                    if figures.get("to_wandb"):
                        wandb.log({f"figures_refine/{ds}/label_{lv}":
                                   wandb.Image(str(fig_path_refine))})
```
(`rg`, `out`, `img`, `lbl`, `cin`, `cout`, `prob_nat`, `b`, `ds`, `lv`, `si` are all already in
scope at this point in the loop.)

- [ ] **Step 7: End-to-end smoke — refine figure is emitted**

Run a 1-epoch refine train with figures on, then confirm a `*_refine.png` was written. First
train (writes `best.pt`):
```bash
.venv/bin/python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine \
  train.epochs=1 train.batch_size=4 eval.batch_size=4 \
  data.max_train_samples=64 eval.max_per_label=4 wandb.enabled=false 2>&1 | tail -5
```
Then eval that checkpoint with figures enabled (use the `best.pt` path printed above):
```bash
.venv/bin/python experiments/2d/eval_incontext.py \
  eval.checkpoint=<path/to/best.pt> wandb.project=null eval.max_per_label=4 \
  eval.save_figures=true eval.max_figures=4 2>&1 | tail -3
ls <eval out_dir>/*_refine.png     # expect at least one refine panel
```
Expected: eval runs without error, prints `dice/mean=...`, and at least one `*_refine.png`
exists alongside the standard `*.png` panels. If the smoke reveals a genuine bug in the figure
code, fix it (re-run the unit test), do not paper over it.

- [ ] **Step 8: Log the change**

Prepend a dated (2026-07-10) entry to `docs/logs.md`: refine `PatchSetCNN` eval now emits an
extra `{ds}_l{lv}_refine.png` qualitative panel (2×3: sampled target/context bboxes, res0/res1/
fused target predictions with GT contours), gated by `eval.save_figures`; `forward` now returns
`refine_ctx_origin` and `refine_geometry` returns the native `fused` map to feed it.

- [ ] **Step 9: Commit**

```bash
git add experiments/2d/evaluate.py tests/test_refine_figure.py docs/logs.md
git commit -m "eval: coarse→fine refine qualitative figure (save_refine_figure)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Exact-slice crops.** Bbox origins are integer, in-bounds, top-left px (`max_sum_window` /
  `gt_window`), so `x[r0:r0+s, c0:c0+s]` is always a full s×s slice — no padding needed.
- **`extent` for the T×T refine pred.** The refine prediction is a T×T grid spanning the whole
  crop; drawing it with `extent=(-0.5, s-0.5, s-0.5, -0.5)` stretches it over the crop's display
  regardless of the T:s ratio, so any valid `resolutions` config renders.
- **Additional, not replacement.** The standard `save_figure` panel is untouched and still
  emitted; the refine panel is written alongside it under the same one-per-`(ds,lv)` gate.
- **Single-level safety.** `rg is None` for plain checkpoints → the refine block is skipped and
  `refine_ctx_origin` is never read.

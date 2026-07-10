# Refine qualitative figure for PatchSetCNN eval

Date: 2026-07-10
Experiment: qualitative visualization for the multi-resolution refine `PatchSetCNN`
(`arch.resolutions=[32,64]`), rendered during `experiments/2d/eval_incontext.py` /
`validate()` when `eval.save_figures=true` and the checkpoint is a refine model.

## Problem

The existing `save_figure` panel (`evaluate.py`) shows target/context images with GT and
prediction overlays, but it has no notion of the refine model's two-level, bbox-zoom
structure. For a refine checkpoint we want a figure that makes the coarse→fine flow legible:
where the target/context bboxes were sampled, what each level predicted inside them, and how
the stitched (fused) prediction looks on the full frame.

## What the figure shows

A 2×3 panel — **row 0 = target**, **row 1 = first context** (K≥1; only the first context is
drawn). Column 2 row 1 is empty.

| | Col 0 — full frame | Col 1 — cropped to bbox | Col 2 — full frame |
|---|---|---|---|
| **Row 0 (target)** | tgt img + GT contour + **res0 (coarse)** pred heat + tgt bbox | tgt **crop** + GT contour + **res1 (refine)** pred heat | tgt img + GT contour + **fused** pred heat |
| **Row 1 (1st ctx)** | ctx img + GT contour + ctx bbox | ctx **crop** + GT contour | *(empty axis)* |

- **res0 pred** = `sigmoid(final_logit)` upsampled to native H (the trainer's `prob_nat`).
- **res1 pred** = `sigmoid(refine_logit)` (T×T), drawn over the crop via matplotlib `extent`
  (stretches T→crop regardless of the T:crop ratio, so any valid config renders).
- **fused pred** = the native stitched map (coarse with the refine crop placed in), at H.

## Decisions (from brainstorming)

- **Context bboxes: returned from the model.** `_refine_forward` already computes `ctx_o`
  (`gt_window` per context); add it to the forward output so the figure is faithful to what
  the model actually cropped, with no duplicated selection logic.
- **Additional figure, not a replacement.** The standard `save_figure` panel is still emitted
  for refine checkpoints; the refine panel is written alongside it as `{ds}_l{lv}_refine.png`.
- **Overlay encoding:** GT as a green (`lime`) contour outline; prediction as a semi-transparent
  `Reds` heatmap (alpha 0.45, matching the existing overlay); bbox as a `Rectangle` outline
  (target = yellow, context = cyan). All three stay legible on one axis.
- **Exact-slice crops.** `max_sum_window` / `gt_window` return integer, in-bounds, top-left px
  origins, so the displayed crop is a plain numpy slice `x[r0:r0+c, c0:c0+c]` — no resampling,
  no drift from what the model cropped.
- **No new config.** Reuses `eval.save_figures`, `eval.max_figures`, `eval.figures_to_wandb`.

## Architecture

### 1. `src/models/patchset_cnn.py` — surface the context origins

In `_refine_forward`, add one key to the returned dict:

```python
return {"final_logit": coarse, "refine_logit": refine,
        "refine_origin": tgt_o, "refine_ctx_origin": ctx_o,   # (B,K,2) px, NEW
        "refine_crop": c, "resolutions": self.resolutions}
```

`ctx_o` is already computed (`:253`). This is a pure output addition: no constructor/param
change, existing checkpoints load unchanged, and downstream code reads by explicit key so the
extra entry is inert for the loss and metric paths.

### 2. `experiments/2d/evaluate.py` — surface the native fused map

`refine_geometry` already builds the native stitch `fused` (B,1,H,H) before pooling (`:194`).
Add it to the return dict:

```python
return {"refine_prob": refine_prob, "refine_target": refine_target,
        "fused": fused,                                       # (B,1,H,H) native, NEW
        "fused_R": F.adaptive_avg_pool2d(fused, (Rf, Rf)),
        "gt_R": F.adaptive_avg_pool2d(lbl, (Rf, Rf)), "Rf": Rf}
```

Metric code ignores the extra key; nothing else changes.

### 3. `experiments/2d/evaluate.py` — `save_refine_figure` + overlay helper

A backend-agnostic render function, styled like `save_figure`:

```python
def _refine_overlay_ax(ax, image, title, *, gt=None, pred=None, pred_extent=None, boxes=()):
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if pred is not None:   # pred_extent stretches a T×T map over the crop; None = pixel-aligned
        ax.imshow(pred, cmap="Reds", alpha=0.45, vmin=0, vmax=1, extent=pred_extent)
    if gt is not None:
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.0)
    for (r0, c0, s, color) in boxes:
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), s, s, fill=False, edgecolor=color, lw=1.5))
    ax.set_title(title, fontsize=8); ax.axis("off")

def save_refine_figure(tgt_image, tgt_gt, ctx_image, ctx_gt,      # full-frame (H,W)
                       coarse_pred, fused_pred,                    # target preds (H,W)
                       refine_pred,                                # target refine pred (T,T)
                       tgt_box, ctx_box,                           # (r0,c0,size) each
                       out_path, title=""):
    ...
```

`save_refine_figure` owns all geometry: it slices the crops from the full-frame arrays using
`tgt_box`/`ctx_box`, draws the res1 pred over the target crop via `imshow(..., extent=...)`, and
lays out the 2×3 grid (col 2 row 1 axis off). `Rectangle` is imported from
`matplotlib.patches`. Fewer args, one place for the crop/extent logic.

### 4. `experiments/2d/evaluate.py` — integration in `validate()`

In the existing gated figure block (`:320`, one figure per `(ds, lv)`), after the standard
`save_figure` call, add — only when `rg is not None`:

```python
if rg is not None:
    c = int(out["refine_crop"])
    fig_path_refine = Path(figures["out_dir"]) / f"{ds}_l{lv}_refine.png"
    tr0, tc0 = (int(v) for v in out["refine_origin"][b])
    cr0, cc0 = (int(v) for v in out["refine_ctx_origin"][b, 0])
    save_refine_figure(
        tgt_image=img[b, 0].cpu().numpy(), tgt_gt=lbl[b, 0].cpu().numpy(),
        ctx_image=cin[b, 0, 0].cpu().numpy(), ctx_gt=cout[b, 0, 0].cpu().numpy(),
        coarse_pred=prob_nat[b, 0].cpu().numpy(),
        fused_pred=rg["fused"][b, 0].cpu().numpy(),
        refine_pred=rg["refine_prob"][b, 0].cpu().numpy(),
        tgt_box=(tr0, tc0, c), ctx_box=(cr0, cc0, c),
        out_path=fig_path_refine,
        title=f"{ds} label={lv} sample={si} refine")
    if figures.get("to_wandb"):
        wandb.log({f"figures_refine/{ds}/label_{lv}": wandb.Image(str(fig_path_refine))})
```

The refine panel is gated by the same `figures` dict + one-per-`(ds,lv)` `saved` set, so it
respects `max_figures` and only appears for refine checkpoints (where `rg is not None`).

## Data flow

`model.forward` → `out` carries `refine_origin`, `refine_ctx_origin`, `refine_crop`,
`refine_logit`. `refine_geometry(out, lbl)` → `rg` carries `refine_prob` (res1) and `fused`
(native). `validate()` already has `prob_nat` (res0) and the raw `img/lbl/cin/cout`. All figure
inputs are thus in hand at the `(ds, lv)` gate; nothing new is recomputed.

## Edge cases

- **Border-clamped bbox:** origins are clamped in-bounds at selection, so `x[r0:r0+c, c0:c0+c]`
  is always a full c×c slice — no partial-slice / padding handling needed.
- **Single-level (plain) checkpoint:** `rg is None` → the refine panel is skipped entirely; only
  the standard `save_figure` runs. No `refine_ctx_origin` is present and none is read.
- **Empty target prediction:** `max_sum_window` centers the crop; the figure still renders (the
  pred heat is simply near-zero).
- **T ≠ crop size:** handled by `imshow(extent=...)` stretching the T×T pred over the crop.

## Testing

- **`save_refine_figure` smoke test:** call it with small synthetic arrays (incl. a
  border-clamped box) and assert the PNG is written, non-empty, and no exception is raised.
  (Visual correctness is not unit-assertable; matplotlib runs under the `Agg` backend already
  set in `evaluate.py`.)
- **End-to-end:** the existing refine smoke run with `eval.save_figures=true` should now also
  produce `{ds}_l{lv}_refine.png` alongside the standard panel. Kept minimal per the project's
  "tests only when necessary" guideline.

## Out of scope (YAGNI)

- Drawing more than the first context (spec'd to first only).
- A combined single-figure-per-batch or per-sample-index mode (kept the existing one-per-
  `(ds, lv)` gate).
- New config knobs (reuses the existing figure config).
- Changing or merging the standard `save_figure` panel.

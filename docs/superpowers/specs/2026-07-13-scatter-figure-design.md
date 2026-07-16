# Scatter refinement qualitative figure — design

**Date:** 2026-07-13
**Status:** approved (design); implementation pending
**Related:** `2026-07-13-scatter-refine-sampling-design.md` (the scatter mode this visualizes),
`experiments/2d/evaluate.py` (`save_figure`, `save_refine_figure`, `validate`),
`experiments/2d/multilevel/plot_sampling.py` (tier-coloring reference).

## Motivation

The scatter refine mode (`refine_mode="scatter"`) produces correct metrics, but eval figures
under-represent it: the main `save_figure` panel shows only the coarse prediction, and the
bbox-specific `save_refine_figure` panel is skipped for scatter (guarded on `refine_origin`,
which scatter lacks). So a scatter eval run gives no visual of **where** cells were sampled or
what the fused stitch looks like — exactly the behavior the approach is about. This adds a
scatter-specific qualitative panel so runs launched from `eval_incontext.py` (with
`eval.save_figures=true`) are inspectable.

## Design decisions (locked)

1. **Tier-colored cell overlay** — cells colored by sampler tier (boundary-core / fg-core /
   neighbor), mirroring `plot_sampling.py`, so the sampler's decisions are visible. Requires
   threading `is_core`/`is_fg_core` (already computed, currently discarded) through the model
   output.
2. **2×3 panel layout** with a context row (like `save_refine_figure`).
3. **New function** `save_scatter_figure` (not an extension of `save_refine_figure`, which is
   bbox-specific with crop rectangles).

## Architecture

### A. Model output — `src/models/patchset_cnn.py::_refine_scatter`

The two `sample_patches` calls already return `(idx, is_core, is_fg_core)` but currently discard
the tier flags (`qidx, _, _` and `sidx, _, _`). Capture them and add 5 keys to the returned dict
(all ignored by the trainer loss and eval metrics, which key on `refine_idx` and read only the
metric-relevant keys — additive, non-breaking):

- `refine_is_core (B, M)` bool — query-cell boundary∪fg core membership.
- `refine_is_fg (B, M)` bool — query-cell fg-core membership.
- `refine_sup_idx (B, K, M)` long — support cell flat indices (reshaped from the `(B*K, M)` sample).
- `refine_sup_is_core (B, K, M)` bool, `refine_sup_is_fg (B, K, M)` bool — support-cell tiers.

The query idx is already returned as `refine_idx (B, M)`. `refine_grid_res` (= Rf) is already present.

### B. `save_scatter_figure(...)` in `experiments/2d/evaluate.py`

A 2×3 matplotlib panel. Signature (all target/context images native `(H,W)` numpy; preds native
`(H,W)`; cells are Rf-grid arrays):
```
save_scatter_figure(
    tgt_image, tgt_gt,                 # (H,W)
    coarse_pred, fused_pred,           # (H,W) native soft maps
    qry_ij, qry_is_core, qry_is_fg,    # (M,2), (M,), (M,)  query cells + tiers (Rf grid)
    ctx_image, ctx_gt,                 # (H,W) first context
    sup_ij, sup_is_core, sup_is_fg,    # (M,2), (M,), (M,)  ctx0 support cells + tiers
    grid_res,                          # Rf, for cell->pixel scaling
    out_path, title="")
```
Layout:
- **Row 0 (target):** `[GT contour(lime) + tier-colored query cells]` | `[coarse native pred heatmap]`
  | `[fused native pred heatmap]`.
- **Row 1 (ctx0):** `[GT contour(lime) + tier-colored support cells]` | blank | blank.

The two heatmap panels reuse the existing `_refine_overlay_ax` helper (gray base + Reds pred +
lime GT contour). The cell-overlay panels draw the gray image + lime GT contour + three scatter
layers.

**Cell → pixel mapping.** For a cell flat index → `i = idx // Rf`, `j = idx % Rf`; center in image
pixels `y = (i + 0.5) * H/Rf`, `x = (j + 0.5) * W/Rf`. Marker = square of size `~H/Rf` px so cells
visually tile the grid. Tier partition (non-overlapping, matches `plot_sampling.py`):
`fg_core = is_fg` → **orange**; `boundary_core = is_core & ~is_fg` → **red**;
`neighbor = ~is_core` → **cyan**.

### C. `validate()` figure block

Add a branch parallel to the existing bbox one (after the `if rg is not None and
out.get("refine_origin") is not None:` bbox panel):
```python
elif rg is not None and out.get("refine_idx") is not None:   # scatter panel
    # gather ctx0 support cells + tiers from out; map indices to ij via idx_to_ij
    save_scatter_figure(..., out_path=fig_path_scatter, title=...)
    if figures.get("to_wandb"):
        wandb.log({f"figures_scatter/{ds}/label_{lv}": wandb.Image(str(fig_path_scatter))})
```
`idx_to_ij` is imported from `src.models.scatter_sampling`. The main `save_figure` panel and the
bbox path are unchanged.

## Testing

- **`_refine_scatter` keys** (extend `tests/test_patchset_scatter.py`): the 5 new keys exist with
  shapes `refine_is_core (B,M)`, `refine_is_fg (B,M)`, `refine_sup_idx/is_core/is_fg (B,K,M)`;
  flags are bool; `refine_sup_idx` in `[0, Rf²)`. Existing scatter + bbox suites stay green
  (additive change).
- **`save_scatter_figure` smoke test** (new `tests/test_scatter_figure.py`): `matplotlib.use("Agg")`,
  build tiny synthetic arrays + random cell indices/tiers, call the function to a tmp path, assert
  the PNG is created and non-empty. Checks it runs and writes, not pixel content.

## Scope guardrails (YAGNI)

Only the first context (`ctx0`) gets a support-cell panel; K>1 extra contexts are not drawn. No
per-cell probability coloring, no interactive/animated output. The main `save_figure` panel and the
bbox `save_refine_figure` path are untouched.

## Files touched

- `src/models/patchset_cnn.py` — 5 additional return keys in `_refine_scatter`.
- `experiments/2d/evaluate.py` — `save_scatter_figure` + the `validate()` scatter branch.
- `tests/test_patchset_scatter.py` — new-key assertions.
- `tests/test_scatter_figure.py` (new) — figure smoke test.

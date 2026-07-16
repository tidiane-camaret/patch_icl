# Scatter refinement sampling for PatchSetCNN — design

**Date:** 2026-07-13
**Status:** approved (design); implementation pending
**Related:** `2026-07-09-patchset-cnn-bbox-refine-design.md` (the bbox refine this replaces the
window of), `project_patchset_refine_2lvl` memory, `experiments/2d/multilevel/` (reference impl).

## Motivation

The deployed `PatchSetCNN` two-level refine crops a single square **bbox** around the densest
predicted region, refines it, and stitches it back (`src/models/bbox_refine.py`). Analysis of the
medsegbench and omnisynth_medseg runs showed the refine *encoder* works (+0.05–0.10 dice on its
crop) but the **single-bbox stitch erases the gain** (macro stitch cost −0.065 to −0.082), because
one contiguous window cannot cover thin / multi-region / multi-instance targets (retinal vessels,
scattered nuclei, surgical tools). Net final ≈ coarse.

A GT-oracle coverage diagnostic (`plot_sampling.py` + a bbox-vs-scatter comparison) confirmed that
**unconstrained "scatter" sampling** — selecting a budget of individual grid cells anywhere the
coarse map is uncertain/foreground, then scattering refinements back — covers ~10 pts more
foreground on exactly the datasets where bbox loses (deepbacs +44, nuclei +36, idrib +21,
m2caiseg +7), while covering 100% of the boundary on compact objects (so it should not regress the
bbox wins). The scatter mechanism already exists in `experiments/2d/multilevel/sampling.py` +
`pipeline.py` (with the `PatchSetPFN` model) but is **not wired into the deployed `PatchSetCNN`**.

This spec is **iteration 1**: the simplest faithful scatter refine, to validate that scatter beats
bbox on the deployed model before adding sophistication.

## Design decisions (locked)

1. **Faithful sampled-cell loss** — the refine head outputs logits on the M sampled cells `(B, M)`
   plus their indices; loss is BCE + soft-Dice on the GT gathered at those cells (mirrors
   `multilevel/pipeline.py`). *Not* a dense-grid masquerade.
2. **Sample both query and support** — query = M cells from the target sampling map; support = M
   cells per context from its mask-fraction map. Both at the fine grid `Rf`, so features/coords
   match exactly. (`K·M` support tokens, bounded.)
3. **Sample source = coarse prediction (prev_pred)** — sample cells from `sigmoid(coarse)`
   upsampled to `Rf`, the same signal available at test time. No teacher forcing, no train/test gap.
4. **New `refine_mode="scatter"`** coexisting with `reencode`/`encode_once`; bbox path stays for
   A/B comparison.

## Architecture

### A. New module `src/models/scatter_sampling.py`

Productionized copy of `experiments/2d/multilevel/sampling.py`, porting the **capped** variant from
`plot_sampling.py` (keeps `src/` free of `experiments/` imports — same pattern as `bbox_refine.py`
← `bbox.py`). Pure tensor ops, no model deps.

Exports:
- `sample_patches(values, n_total, tau, blur_sigma, floor, grid_res, temperature=1.0,
  stochastic=True, n_fg_core=0, boundary_tier=True, n_boundary_core=0)` → `(idx, is_core,
  is_fg_core)`, each `(B, n_total)`. Three priority tiers combined into one score + a single top-k:
  (1) boundary core `|value−0.5| < tau` (optionally capped to the `n_boundary_core` cells closest
  to 0.5); (2) fixed `n_fg_core` random-foreground quota; (3) Gaussian-blurred proximity field over
  `core ∪ fg_core` + uniform `floor` + Gumbel-top-k neighbor fill.
- `idx_to_ij(idx, grid_res)` → `(B, M, 2)` row/col.
- `gather_grid(x, idx)` — gather along the cell axis for `(B,N,C)` or `(B,N)`.
- `composite_predictions(coarse_flat, idx, vals)` → `(B, N)` new tensor with `vals` scattered in.
- `gaussian_blur(x_flat, grid_res, sigma)` — separable blur helper.

### B. `patchset_cnn.py`

**B.1 Refactor `_attn` → `_attn_core` (backward-compatible).** Extract the body of `_attn` into
`_attn_core(sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij, res, K, ctx_count, mem=None,
return_think=False, flat_out=False)`. It does the support-stat standardization, token build
(`self.pos(ij, res)`), optional context-id / memory rows, thinking rows, transformer, and decode.
When `flat_out` is False it reshapes the query logits to `(B,1,res,res)` (current behavior); when
True it returns `(B, Q)` unshaped. `context_id_embed` uses `repeat_interleave(ctx_count)` (N for the
grid path, M for scatter).

`_attn` becomes a thin wrapper that fills the grid defaults — `qry_occ = sup_occ.mean` prior,
`sup_ij/qry_ij` = full lattice via `ij_base`, `res=self.resolution`, `ctx_count=self.N`,
`flat_out=False` — so the coarse/single-level path is **bitwise-identical** (regression-guarded by a
test).

**B.2 New `_refine_scatter(image, context_in, context_out)`:**
1. `imgs = cat([context_in, image], 1)`; `maps = encoder.encode_maps(...)` (once).
2. Coarse: `sup_c, qry_c = _grid_tokens(pool_maps(maps, T), B, T, K)`; `coarse[, coarse_think] =
   _attn(sup_c, qry_c, _occupancy(context_out), K[, return_think=refine_memory])`.
3. Fine features: `feat = pool_maps(maps, Rf)` reshaped to `(B, T, Rf², Cf)`.
4. Query sampling: `q_map = interpolate(sigmoid(coarse).detach(), (Rf,Rf)).reshape(B, Rf²)`;
   `qidx = sample_patches(q_map, M, …)`; `qry_feat = gather(feat[:,-1], qidx)`;
   `qry_ij = idx_to_ij(qidx, Rf)`; `qry_prior = gather(q_map, qidx).unsqueeze(-1)` (coarse prob).
5. Support sampling: `ctx_frac = adaptive_avg_pool2d(context_out, (Rf,Rf)).reshape(B*K, Rf²)`;
   `sidx = sample_patches(ctx_frac, M, …, n_fg_core=n_fg_core_ctx)`;
   `sup_feat = gather(feat[:,:K]…, sidx).reshape(B, K·M, Cf)`;
   `sup_occ = gather(ctx_frac, sidx)…(B, K·M, 1)`; `sup_ij = idx_to_ij(sidx, Rf)…(B, K·M, 2)`.
6. `mem = coarse_think.detach()` if `refine_memory` else None.
7. `refine_logit = _attn_core(sup_feat, qry_feat, sup_occ, qry_prior, sup_ij, qry_ij, res=Rf, K,
   ctx_count=M, mem=mem, flat_out=True)` → `(B, M)`.

Sampling is stochastic in train, deterministic in eval (`stochastic = self.training`).

## Output contract

`_refine_scatter` returns:
```python
{"final_logit": coarse,          # (B,1,T,T) unchanged coarse head
 "refine_logit": refine_logit,   # (B, M) per-sampled-cell logits
 "refine_idx":   qidx,           # (B, M) flat Rf-grid cell indices
 "refine_grid_res": Rf,          # int
 "resolutions": self.resolutions}
```
Absence of `refine_origin`/`refine_crop` is how downstream code distinguishes scatter from bbox.

## Trainer / eval wiring

**`experiments/2d/train.py`** — add a scatter branch to the refine-loss block (the existing
`crop_resize` branch stays for bbox modes):
```python
if out.get("refine_idx") is not None:          # scatter
    Rf = out["refine_grid_res"]
    gt_Rf = F.adaptive_avg_pool2d(lbl, (Rf, Rf)).reshape(B, Rf*Rf)
    rtarget = gather_grid(gt_Rf, out["refine_idx"])       # (B, M)
    rlogit  = out["refine_logit"].float()                  # (B, M)
    loss += refine_loss_weight * (bce(rlogit, rtarget) + dice_weight * soft_dice(sigmoid(rlogit), rtarget))
```

**`experiments/2d/evaluate.py::refine_geometry`** — add a scatter branch returning the same keys as
the bbox branch so all downstream metrics/plots keep working:
- `refine_prob = sigmoid(refine_logit)` `(B, M)`; `refine_target = gather_grid(gt_Rf, idx)`.
- `fused_R = composite_predictions(coarse_prob_Rf, idx, refine_prob).reshape(B,1,Rf,Rf)`;
  `gt_R = gt_Rf.reshape(B,1,Rf,Rf)`.
- `fused` (native) = `interpolate(fused_R, native)`.
- `coarse_nat`/`coarse_R` (the existing refine-off counterfactual) = the coarse grid up/at Rf → the
  exact per-sample refine delta `dice − dice_coarse` comes for free.
- Sample-table columns are unchanged (`dice@Rf`, `dice_soft@Rf`, `dice_fused@Rf`, `dice_coarse@Rf`,
  `dice_coarse`); "refine@Rf" is now scored on the sampled cells rather than a crop.

## Config & checkpoint

- `cfg.arch.refine_mode: scatter`.
- `cfg.sample` block: `n_total, tau, blur_sigma, floor, n_fg_core, n_fg_core_ctx, temperature,
  n_boundary_core`. Defaults seeded from `plot_sampling`: `n_total=256, tau=0.30, blur_sigma=1.0,
  floor=0.005, n_fg_core=64, n_fg_core_ctx=64, temperature=1.0, n_boundary_core=0` (uncapped —
  the coverage diagnostic showed the cap slightly reduces LOSE-set coverage; it stays a knob for later).
- `build_model` stores the sample params + `refine_mode` in the checkpoint `arch` dict so `eval.py`
  rebuilds `PatchSetCNN(image_size=…, **arch)` with zero drift.

## Testing

- `tests/test_scatter_sampling.py`: output shapes; idx in `[0, N)`; budget respected
  (`idx.shape[-1]==n_total`); seed determinism; `n_boundary_core` cap honored; fg/boundary coverage
  sane on a synthetic mask (a compact blob → boundary fully in core; a scattered mask → fg spread
  across core+neighbor).
- `tests/test_patchset_scatter.py`: `_refine_scatter` forward → `refine_logit (B,M)`, idx valid,
  finite; the `_attn`→`_attn_core` refactor leaves the coarse/single-level output **bitwise-identical**
  (regression guard, `torch.equal` on fixed seed); a few train steps with `refine_mode=scatter` on a
  synthetic batch → finite loss, `backward()` runs.

## Scope guardrails (YAGNI for iteration 1)

Single refine level only (`resolutions=[T, Rf]`, honoring the existing ≤2 assertion). **Out of
scope:** multi-hop scatter, learned/adaptive budget, alternative fusion (soft-blend vs hard
scatter), sampling-source scheduling (GT→pred warmup), and any encoder change. `context_id_embed`
is supported via `repeat_interleave(M)` but not a focus.

## Files touched

- `src/models/scatter_sampling.py` (new)
- `src/models/patchset_cnn.py` (`_attn_core` refactor, `_refine_scatter`, `refine_mode` dispatch,
  sample-param constructor args)
- `experiments/2d/train.py` (scatter refine-loss branch)
- `experiments/2d/evaluate.py` (`refine_geometry` scatter branch)
- a config under `configs/experiment/2d/` (scatter variant of the patchset_cnn refine config)
- `tests/test_scatter_sampling.py`, `tests/test_patchset_scatter.py` (new)

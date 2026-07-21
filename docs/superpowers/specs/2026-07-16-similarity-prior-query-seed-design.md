# Max-cosine similarity prior for the PatchSetCNN query token — design

**Date:** 2026-07-16
**Status:** approved (design); implementation pending
**Related:** `configs/experiment/2d/5_full_res_decode.yaml` (baseline this improves on),
`experiments/2d/feature_sim.py` (heavyweight training-free cousin: UniverSeg features + TabPFN),
`src/models/scatter_sampling.py` (the foreground-balanced-support follow-up lever, out of scope
here), `project_patchset_refine_2lvl` memory. Method is PFENet's prior-mask generation
(Tian et al. 2020) adapted to this model's own encoder features.

## Motivation

On the single-level `PatchSetCNN` (`5_full_res_decode`: res=32 → 1024 tokens,
`mask_patch_decode_size=4` → 128² output, K context images, `full_attn`), small objects are a
needle-in-a-haystack. Diagnosis of where it bites:

1. **The query's prior is ~zero for small objects.** In `_attn` (`src/models/patchset_cnn.py:295`)
   the target's mask token is `qry_occ = sup_occ.mean(...)` — the support-mean occupancy. For a
   ≤32px object at 128², support occupancy ≈ 0.2%, so *every* query token starts from a flat
   "this is background" prior with no localized hint of where to look.
2. **Attention dilution.** Each query token content-matches over K·N = 1024 support tokens, of
   which only ~4 are foreground; softmax over 1024 keys struggles to isolate them.
3. **Background-dominated loss** and **4px token quantization** (unchanged by this spec).

This spec attacks **(1)** only, in isolation, so its effect on small-object dice is cleanly
attributable against the `5_full_res_decode` baseline. It replaces the flat support-mean query
prior with a **localized max-cosine similarity prior**: for each query cell, "how much do I look
like the context's foreground cells", computed from feature similarity with **no learned
parameters**. Foreground-balanced support (fixing the keys, lever 2) and any auxiliary loss are
deliberately deferred to separate experiments.

## Design decisions (locked)

1. **Max-cosine to foreground, not softmax-mean.** The per-query-cell prior is the *max* cosine
   similarity to the support's foreground cells. `max` is imbalance-robust — even with ~4 fg
   support cells, each query cell is scored by its best match. A softmax-weighted average would
   re-introduce the exact dilution we are escaping.
2. **Full concatenated encoder features.** Similarity uses the full `sum(enc_dims)`-channel
   feature already in hand (not a deepest-stage subset). Simplicity; revisit only if the prior is
   noisy.
3. **Replace, not blend.** When active, `qry_occ` is built entirely from the prior. The flat prior
   it replaces was ≈0, so this strictly adds signal; a learnable blend gate is a noted future
   variant, not v1.
4. **Detached.** The prior is a `.detach()`-ed input signal (à la PFENet's frozen features), so the
   model cannot game the similarity instead of learning to segment. The encoder still learns good
   features from the main objective (those same features feed the image tokens), so the prior
   sharpens with training.
5. **Degenerate fallback.** Images with zero foreground support cells keep the existing
   `sup_occ.mean()` prior (no fg exemplars → no meaningful max-cosine).
6. **Off by default.** New `sim_prior: bool = False`. When false the model is byte-identical to
   current behaviour, and existing checkpoints reload unchanged.
7. **Single-level grid path only.** Wired into `_attn` (used by `_segment`). The scatter/`_attn_core`
   flat path is out of scope (it already samples query cells from the coarse prediction).

## Architecture

### New method: `PatchSetCNN._similarity_prior`

Computed inside `_attn`, where `sup_feat (B,S,Cf)`, `qry_feat (B,N,Cf)`, `sup_occ (B,S,p²)` are all
already available (`src/models/patchset_cnn.py:291-295`).

```
def _similarity_prior(qry_feat, sup_feat, sup_occ) -> (prior (B,N) in [0,1], valid (B,) bool):
    occ = sup_occ.mean(-1)                       # (B,S) per-support-cell scalar occupancy
    fg  = occ >= 0.5                             # (B,S) foreground support cells
    q = F.normalize(qry_feat, dim=-1)           # L2
    s = F.normalize(sup_feat, dim=-1)
    sim = q @ s.transpose(1, 2)                 # (B,N,S) cosine; ~one attention-score matmul
    sim = sim.masked_fill(~fg[:, None, :], -inf)
    prior_raw = sim.max(dim=-1).values          # (B,N) max-cosine to any fg exemplar
    valid = fg.any(dim=-1)                      # (B,) images with >=1 fg support cell
    # per-image min-max over query cells -> [0,1]; guard constant maps -> 0
    prior = minmax_normalize(prior_raw)         # (B,N)
    return prior.detach(), valid
```

Cost: one `(B,N,S)` matmul (N=1024, S=K·1024) — same order as a single attention score, negligible
vs the l-layer transformer.

### Seeding in `_attn`

Current (`src/models/patchset_cnn.py:294-295`):
```
qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])   # flat prior
```
New, when `self.sim_prior`:
```
prior, valid = self._similarity_prior(qry_feat, sup_feat, sup_occ)     # (B,N), (B,)
prior_tile = prior[..., None].expand(B, N, sup_occ.shape[-1])          # uniform-fill p² tile
mean_tile  = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])
qry_occ = torch.where(valid[:, None, None], prior_tile, mean_tile)     # fallback for no-fg images
```
The prior scalar uniform-fills the p²=`mask_patch_size²` tile, staying in the same [0,1] occupancy
scale the shared `mask_embed` sees for support tokens.

### Plumbing

- `PatchSetCNN.__init__`: add `sim_prior: bool = False`, store `self.sim_prior`.
- `experiments/2d/train.py:156` arch dict: add `"sim_prior": a.get("sim_prior", False)` so it is
  recorded in the checkpoint and reloaded by eval with zero drift.
- New `configs/experiment/2d/6_sim_prior.yaml`: `defaults: [5_full_res_decode, _self_]`, sets
  `arch.sim_prior: true`. `5_full_res_decode.yaml` is left untouched as the A/B baseline.

## Testing

- **Unit (`_similarity_prior`):**
  - output shape `(B,N)`, range ⊂ [0,1], `requires_grad is False`;
  - a query cell whose feature equals a fg support cell's scores at (near) the per-image max;
  - all-background support → `valid` is False for that image (fallback path taken).
- **Backward-compat:** `sim_prior=False` yields logits identical to the current `_segment` on a
  fixed input (guards the default path / existing checkpoints).
- **Smoke:** `sim_prior=True` forward runs and returns the unchanged output shape (B,1,128,128).

## Evaluation

Train `6_sim_prior`; reuse the size-binned dice notebook
(`results/experiments/13_full_res_decode.py`) to compare against `5_full_res_decode` on the ≤32px
bucket and micro mean. **Success = ≤32px dice up with no regression above ~130px.**

## Risks

- **Cold start** — random early features → noisy prior. Mitigated by `detach` + degenerate
  fallback; the prior rides the model's own encoder and sharpens as training proceeds.
- **Low-contrast tiny objects** — features may not be discriminative → false-positive prior. It is
  only an input signal on the mask column (not a hard gate), so the transformer can override it.

## Out of scope (future levers)

- Foreground-balanced support sampling (fixing the attention *keys*, `scatter_sampling.py`).
- Auxiliary supervision of the prior heatmap against pooled GT.
- Learnable blend gate between prior and support-mean; deepest-stage-only similarity features.

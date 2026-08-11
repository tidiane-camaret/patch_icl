# PatchSet3D random token masking — design

**Date:** 2026-08-11
**Model:** `src/models/patchset3d.py` (`PatchSet3D`, the CoLiPri set-of-patches matcher)
**Status:** approved design, ready for implementation plan

## Goal

Train the dual-axis attention head to rely on **incomplete** input by randomly masking
feature-patch tokens during training. Near-term this is a regularization/augmentation:
the matcher must segment despite missing context patches (support) and missing target
patches (query). The design is deliberately **reconstruction-compatible** so an optional
auxiliary token-reconstruction loss can be added later (Phase 2) without reworking the
masking mechanism.

## Background: token layout

`PatchSet3D.forward` encodes each volume (K context + 1 target) to an `R³` feature grid,
then `_grid_tokens` splits the cells into:

- **support** tokens: `K·N` (N = R³), and
- **query** tokens: `N`.

Each token is a **2-column** structure built in `_tokens`: an image-feature column
(`img_embed(feat)`) and a mask/occupancy column (`mask_embed(occ)`), stacked to `(B, M, 2, e)`.
Support columns are both real (encoder feature + known label occupancy); the query's mask
column is only the support-mean prior. `_attn` prepends `thinking_rows`, concatenates to a
row sequence `[thinking | K·N support | N query]`, and `TransformerEncoderStack` runs
dual-axis attention (feature-axis across a token's 2 columns; sample-axis across rows, where
query/support rows attend to `[:sep]` = thinking+support).

## Decisions (from brainstorming)

1. **Purpose:** masking-as-augmentation now; auxiliary reconstruction loss deferred to Phase 2.
2. **Target sets:** both **support** and **query** tokens.
3. **What is masked:** the **whole token** — both the image and mask/occupancy columns.
4. **Rate:** **separate** support/query ratios, fixed per step, both default `0.0` (off).
5. **When:** training only.

## Mechanism — SimMIM-style in-place masking (not MAE token-dropping)

Masked tokens are **replaced in place** by a learned `[MASK]` embedding, **not** removed
from the sequence. Rationale — three things assume a fixed token count and would break under
token-dropping:

- the fixed `R³` grid layout and `_tile_logits` inverse,
- the `torch.compile`'d transformer's static sequence length (dropping → variable length →
  recompilation / dynamic-shape churn), and
- the RoPE-by-row-index positional scheme (`_rope` maps row index → `(i,j,k)`).

In-place replacement keeps `K·N` support + `N` query rows intact, and — crucially for the
future reconstruction loss — leaves every masked cell present in the sequence, positioned,
and still producing an output.

### Learned mask token

Add one parameter:

```python
self.mask_token = nn.Parameter(torch.zeros(2, e))   # row 0 = image col, row 1 = mask col
nn.init.normal_(self.mask_token, std=0.02)
```

Only allocated/used when either ratio > 0 (allocating it unconditionally is harmless and
keeps state_dict stable; allocate unconditionally for simplicity).

### Applying the mask (`_tokens`)

`_tokens(feat, occ, ijk, mask=None)` gains an optional boolean `mask` of shape `(B, M)`:

```python
img = self.img_embed(feat)
msk = self.mask_embed(occ)
if mask is not None:
    m = mask.unsqueeze(-1)                       # (B, M, 1)
    img = torch.where(m, self.mask_token[0], img)
    msk = torch.where(m, self.mask_token[1], msk)
if self.pos is not None:                         # additive Fourier PE (non-RoPE mode)
    pos = self.pos(ijk, self.resolution)
    img = img + pos
    msk = msk + pos
return torch.stack([img, msk], dim=2)
```

Replacement happens **before** the additive PE, so a masked cell is exactly
`[MASK] + position`. In RoPE mode (`self.pos is None`) position is carried by the preserved
row index inside attention — automatically correct, since no rows are dropped.

Context-id embedding and thinking rows are still added afterward in `_attn` (unchanged order),
so a masked support token still knows which context volume it belongs to — only its *content*
is hidden.

### Sampling the mask (`_attn`)

Gate on `self.training` (eval/`predict`/`validate_mean` run under `net.eval()` → never mask):

```python
def _sample_mask(self, B, M, ratio, device):
    if not self.training or ratio <= 0.0:
        return None
    return torch.rand(B, M, device=device) < ratio     # independent per cell (Bernoulli)
```

- `mask_support = _sample_mask(B, K*N, self.token_mask_ratio_support, dev)`
- `mask_query   = _sample_mask(B, N,   self.token_mask_ratio_query,   dev)`

Independent per-cell Bernoulli is chosen for simplicity. Exact per-sample count
(`randperm`) is a noted alternative if degenerate all-/none-masked draws prove a problem;
not expected at the ratios of interest.

`_feat_norm` is computed **before** masking (over the real support features), so masked
tokens do not perturb the context z-score statistics — kept intentionally: the stats remain
a property of the true context.

### Return value (`forward`)

```python
return {"final_logit": logit, "mask_support": mask_support, "mask_query": mask_query}
```

`mask_*` are `None` when masking is inactive. The training loop (`experiments/3d/train.py`
`train_epoch`) reads `out["final_logit"]` exactly as today — **no training-loop change** in
Phase 1. The masks are the clean hook for Phase 2.

`_attn` returns `(logit, mask_support, mask_query)` so `forward` can surface them; the
internal signature change is confined to `patchset3d.py`.

## Loss behavior (Phase 1)

Unchanged. Masked **query** cells still receive a decoder logit and still incur the existing
`bce_dice` segmentation loss at their grid cell — that *is* the "segment despite a missing
target patch" learning signal. Masked **support** cells remain attendable keys carrying only
position + context-id + `[MASK]` (no feature, no label).

## Config

`configs/experiment/3d/model/patchset3d.yaml` gains two knobs (both default `0.0` → behavior
byte-identical to today):

```yaml
arch:
  token_mask_ratio_support: 0.0   # fraction of K·N context tokens masked per step (train only)
  token_mask_ratio_query:   0.0   # fraction of N target tokens masked per step (train only)
```

Wired through `experiments/3d/train.py::build_model`'s `arch` dict via
`a.get("token_mask_ratio_support", 0.0)` / `a.get("token_mask_ratio_query", 0.0)` and the
`PatchSet3D.__init__` signature. Because both flow through the arch dict, they are stored in
the checkpoint's `arch` block and reproduced by `eval.py` (which rebuilds from the checkpoint
arch) — harmless there since eval never trains.

## Compile / performance

Mask generation and `_tokens` live outside the compiled submodule (only `net.transformer` is
compiled). Sequence length is unchanged (replace, not drop), so no recompilation or
dynamic-shape churn. Overhead is one `torch.rand` + two `torch.where` per forward — negligible.

## Out of scope (Phase 2 sketch — do NOT build now)

Optional auxiliary reconstruction loss. Given the returned `mask_support`/`mask_query` and the
in-place `[MASK]` placeholders, a small head would read the transformer output at masked
positions and predict the original encoder feature (and/or the known occupancy) there,
added to the seg loss with a weight. Nothing in Phase 1 needs to change to enable this; it is
documented only to confirm the mechanism supports it.

## Testing

- **No-op guarantee:** with both ratios `0.0`, one training forward produces logits
  bit-identical (or within fp noise) to the pre-change model, and `mask_*` are `None`.
- **Masking active:** with a ratio > 0 under `net.train()`, the expected fraction of cells is
  replaced (check `mask_*` shapes/mean), gradients flow to `mask_token`, and `forward` still
  returns the correct `(B,1,Rd,Rd,Rd)` logit shape.
- **Eval untouched:** under `net.eval()` the masks are `None` regardless of ratio.
- Keep tests minimal (repo guideline: tests only when necessary) — a single focused unit test
  on the masking path is sufficient.

## Logging

Record the change in `docs/logs.md` per repo convention.

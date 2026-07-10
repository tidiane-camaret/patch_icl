# PatchSetCNN cross-level thinking memory — design

**Date:** 2026-07-10
**Component:** `src/models/patchset_cnn.py` (+ trainer/config wiring)
**Status:** approved, pending implementation plan

## Goal

When `PatchSetCNN` runs in refine (multi-resolution) mode, let the **refine
(second-level) pass attend to the thinking tokens produced by the coarse
(first-level) pass**. This mirrors the cross-level memory mechanism already used
in `experiments/2d/multilevel/` (`PatchSetPFN.stage1_think`), adapted to
PatchSetCNN's single-model / shared-weight structure.

## Reference mechanism (multilevel PatchSetPFN)

Three parts, in `src/models/patchset_pfn.py` + `experiments/2d/multilevel/pipeline.py`:

1. **Produce** — after the transformer, `forward(..., return_thinking=True)`
   returns `think = x[:, :n_think].mean(dim=2)` → `(B, n_think, e)`, the
   post-attention thinking rows averaged over the img|mask columns.
2. **Consume** — the next level receives `stage1_think=(B,T1,e1)`, projects
   `e1→e`, adds a learned type token, broadcasts to both columns, and
   **prepends the rows into the support block** (`sep += T1`). Because they land
   inside `[:sep_t]`, every row (support and query) attends to them.
3. **Thread** — `run_chain` passes coarse `think` to hop 1 as `prev_think` and
   each hop emits `this_think`; values are **detached** between hops because
   stage-1 and each level are separate, frozen/independent models.

## Key difference driving this design

In PatchSetCNN the coarse and refine passes **share the same weights** — `_attn`
is called twice (`_refine_reencode`, `_refine_encode_once`) with the same `e` and
the same transformer. Multilevel used *separate* models with a *frozen* stage-1,
which forced both the `e1→e` projection and the detachment. Here those are
**choices**, and we make the minimal ones.

## Design decisions

- **Gradient flow: detach.** The coarse thinking rows are `.detach()`ed before
  the refine pass consumes them. The coarse pass is shaped only by the coarse
  loss; the refine pass treats coarse thinking as read-only memory. (Matches the
  reference; more stable than joint end-to-end coupling.)
- **Adapter: type token only.** The coarse thinking rows are already in the
  model's `e` latent space (shared weights), so **no projection** is needed. A
  single learned `mem_type` parameter (`e`,) marks the rows as coarse-memory and
  is added before injection. No positional `(i,j)` features (memory rows have no
  patch location).
- **Opt-in flag.** New `arch.refine_memory: bool`, default `false`. Every
  existing config and checkpoint is unchanged and gains zero parameters.
- **Both refine modes** are covered automatically because the injection lives in
  the shared `_attn`.

## Changes

### `src/models/patchset_cnn.py`

**Constructor:** add `refine_memory: bool = False`. Store `self.refine_memory`.
Create the parameter only when enabled:
```python
if refine_memory:
    self.mem_type = nn.Parameter(torch.zeros(e))
    nn.init.normal_(self.mem_type, std=0.02)
```

**`_attn`** gains two optional args, `mem=None` and `return_think=False`:
```python
def _attn(self, sup_feat, qry_feat, sup_occ, K, mem=None, return_think=False):
    ...                                        # unchanged: standardize, build sup_tok / qry_tok
    sep = K * N
    rows = [sup_tok, qry_tok]
    if mem is not None:                        # mem: (B, T1, e), detached, already in e-space
        T1 = mem.shape[1]
        m = (mem + self.mem_type).unsqueeze(2).expand(B, T1, 2, e)   # broadcast to both cols
        rows = [m] + rows                      # layout: [memory | support | query]
        sep += T1
    x = torch.cat(rows, dim=1)
    x, sep_t = self.thinking(x, sep)           # -> [thinking | memory | support | query]
    ...                                        # attn_mask logic unchanged: [:sep_t] covers memory
    x = self.transformer(x, sep_t, attn_mask=attn_mask)
    q = x[:, sep_t:, 0, :]                      # query rows unchanged (memory is left of sep_t)
    logit = self.decoder(q).squeeze(-1).reshape(B, 1, R, R)
    if return_think:
        return logit, x[:, :self.thinking.n].mean(dim=2)   # (B, n_think, e)
    return logit
```

**`_segment`** (used by `_refine_reencode`) forwards the same two optional args
so the coarse call can request thinking and the refine call can inject memory.

**`_refine_reencode` / `_refine_encode_once`:** when `self.refine_memory`, run
the coarse `_attn`/`_segment` with `return_think=True`, then pass
`mem=coarse_think.detach()` into the refine `_attn`/`_segment`. When the flag is
off, behavior is identical to today (`mem=None`, no thinking captured).

Docstring note: `refine_memory` is inert for single-level (`len(resolutions)==1`)
configs — there is no coarse pass to summarize. No assert; silently ignored.

### `experiments/2d/train.py`

`build_model`'s `arch` dict: add
`"refine_memory": a.get("refine_memory", False)` so eval rebuilds the model
identically and old checkpoints still load (`.get` default).

### `configs/experiment/2d/train_base.yaml`

Add to the `arch:` block:
```yaml
  refine_memory: false   # refine pass attends to detached coarse thinking rows (multi-res only)
```
The `2_omnisynth_medseg_refine.yaml` leaf sets `refine_memory: true` to enable it.

### `docs/logs.md`

Append a change-log entry describing the mechanism and the flag.

## Compile note

The refine transformer call now carries `n_think` more rows than the coarse call
(two distinct row counts). `torch.compile(..., dynamic=True)` — the mode used in
`train.py` — already treats `sep` and the row axis as symbolic, so this adds at
most one extra guard and does not reintroduce the `adaptive_avg_pool2d` lowering
break (that lives in the eager encoder, outside `model.transformer`).

## Testing (`tests/test_patchset_cnn_refine.py`, extend)

1. **Backward-compat.** `refine_memory=False` gives bitwise-identical output to
   the pre-change model; no `mem_type` in `state_dict`.
2. **Shapes.** `refine_memory=True` forward runs for both `refine_mode`s;
   `final_logit` / `refine_logit` shapes unchanged.
3. **Wiring is real.**
   - `mem_type` receives a non-None gradient after a refine-loss backward
     (proves the refine pass actually routes through the memory rows).
   - Perturbing the captured `coarse_think` before injection changes
     `refine_logit` (refine attends to memory).
   - **Detach:** the tensor passed as `mem` into the refine `_attn` has no grad
     history (`grad_fn is None` / `requires_grad=False`), so refine-loss gradient
     cannot flow back through the memory path into the coarse pass. (Note: coarse
     and refine share weights, so the coarse *parameters* still receive gradient
     from the refine loss via the refine pass's ordinary forward — detach only
     cuts the memory-row path, which is what this asserts.)

## Out of scope

- Joint (non-detached) gradient flow.
- Any `e1→e` projection / larger memory adapter.
- More than 2 levels (the model already asserts `len(resolutions) <= 2`).
- Changes to the encoder / crop / bbox-selection path.

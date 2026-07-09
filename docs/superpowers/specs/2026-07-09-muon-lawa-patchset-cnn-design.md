# Muon + LAWA for patchset_cnn in the unified 2D trainer

**Date:** 2026-07-09
**Status:** Approved design, ready for implementation plan
**File touched:** `experiments/2d/train.py` only

## Problem

The unified 2D trainer `experiments/2d/train.py` trains both `universeg` and
`patchset_cnn` through one AdamW loop. The `muon_*` and `lawa_k` keys already
exist in `configs/experiment/2d/train_base.yaml` but this trainer ignores them —
they are only used by `experiments/2d/pfn_seg.py` (ImagePFN/PatchSetPFN). We want
`patchset_cnn` to train with the same Muon + LAWA recipe, since it has the same
`self.transformer` submodule the recipe targets.

## Decision

**Always-on for patchset_cnn.** `patchset_cnn` always trains with Muon (for its
transformer 2D matrices) + LAWA. `universeg` keeps its exact current path (single
AdamW, no LAWA) — its baseline is preserved byte-for-byte for the model comparison.

Gating flag: `is_patchset = (model_name == "patchset_cnn")`.

## Reference implementation

Reuse the proven pattern from `experiments/2d/pfn_seg.py` and the shared utilities
in `experiments/2d/pfn_train.py` (`Muon`, `lawa_average`). No new optimizer/averaging
code is written — only wiring.

## Design

### 1. Optimizer split (cf. `pfn_seg.py:334-349`)

```python
muon_params = [p for n, p in model.named_parameters()
               if p.requires_grad and p.ndim == 2 and "transformer" in n]
adam_params = [p for n, p in model.named_parameters()
               if p.requires_grad and not (p.ndim == 2 and "transformer" in n)]

optimizers = [torch.optim.AdamW(adam_params, lr=cfg.train.lr,
                                weight_decay=cfg.train.get("adam_wd", 0.01))]
if is_patchset:
    optimizers.append(Muon(muon_params,
                           lr=cfg.train.muon_lr_scale * cfg.train.lr,
                           momentum=cfg.train.muon_momentum,
                           weight_decay=cfg.train.muon_wd))
```

- `self.transformer` qkv/MLP 2D weights → Muon.
- Encoder convs (4D), `img_embed`/`mask_embed`, `pos`, `decoder`, thinking rows,
  ctx-id embeddings → AdamW.
- universeg has no `transformer` submodule, so `muon_params` is empty; Muon is not
  constructed for it (an empty-param Muon would raise).

### 2. Scheduler

Cosine + linear warmup stays attached to the **AdamW optimizer only**, stepped
per-batch exactly as today (step-based `warmup_steps` / `total_steps`). Muon runs at
constant LR (matches pfn_seg — Muon is unscheduled).

`train_epoch` changes: the single `optimizer` parameter becomes an `optimizers`
list. `optimizer.zero_grad(...)` and `optimizer.step()` become loops over the list.
`scheduler` is still passed separately and stepped per-batch; `scheduler.get_last_lr()`
still drives LR logging.

### 3. LAWA (cf. `pfn_seg.py:361/401/416`; only when `is_patchset`)

- `lawa_queue = collections.deque(maxlen=cfg.train.lawa_k)`.
- End of each epoch: `lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})`.
- Around the existing `validate(...)` call:
  1. `saved = lawa_average(lawa_queue, model, DEVICE)` **before** validate — model now
     holds averaged weights for eval, so the best-checkpoint `torch.save` stores the
     **averaged** weights.
  2. run `validate(...)` and the existing best-checkpoint save block unchanged.
  3. **after** the eval block: `if saved: model.load_state_dict(saved)` — restore the
     raw training weights so optimization continues from them.
- Epoch 1 (queue length 1): `lawa_average` returns `None` (its `len(queue) <= 1`
  guard), so nothing is averaged and nothing is restored. Correct by construction.
- When `eval_every` skips an epoch, no averaging happens that epoch and there is
  nothing to restore.

### 4. Config

No config changes required — `muon_lr_scale=0.1`, `muon_momentum=0.96`,
`muon_wd=0.1`, `adam_wd=0.01`, `lawa_k=10` already resolve into patchset_cnn via
`train_base.yaml`. Add a one-line comment in
`configs/experiment/2d/patchset_cnn_train.yaml` noting these keys are now active for
this model.

## Edge cases

- **universeg unchanged:** `optimizers` is a 1-element list `[AdamW]`, no LAWA queue,
  no averaging/restore — identical numerics to today.
- **Restore correctness:** training always resumes from raw (un-averaged) weights;
  only eval and the saved checkpoint see averaged weights.
- **LAWA memory:** `lawa_k` CPU state-dict clones; negligible for this model size.

## Comparison impact

This changes the patchset_cnn training recipe. The two existing checkpoints are
therefore not a clean optimizer before/after. To later attribute a Dice change to
architecture vs optimizer, keep a previous AdamW patchset_cnn run as a reference.

## Out of scope

- No Muon/LAWA for universeg.
- No config-flag opt-in (decided against — always-on for patchset_cnn).
- No change to the loss, data pipeline, or scheduler shape.

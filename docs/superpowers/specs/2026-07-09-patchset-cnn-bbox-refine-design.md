# Multi-resolution bbox-zoom refinement for PatchSetCNN

Date: 2026-07-09 (revised — supersedes the single-composite / additive-fusion approach below)
Experiment: `src/models/patchset_cnn.py` (multi-level `resolutions` mode), trainable/evaluable
through the existing `experiments/2d/train.py` and `experiments/2d/eval_incontext.py`.

## Problem

`PatchSetCNN` predicts a coarse T×T token grid over the full image (T=16/32). Small or
detailed objects lose sharpness at that resolution. We want a **coarse→fine refinement**:
after a first pass over the full image, zoom into the densest region and re-segment it with
the **same model** at the **same token count**, so the second pass resolves finer detail
(higher *effective* full-image resolution) at equal compute.

## Core geometry (the key idea)

`arch.resolutions` is a list of **effective full-image resolutions**, one per level. The
**token grid `T` is constant across levels** and equals `resolutions[0]` — every pass emits a
`T×T` grid and costs the same. A pass over a `c`-px crop (resized to the `image_size` encoder
input) yields `T` tokens spanning `c` px, i.e. effective full-image resolution
`T · image_size / c`. Setting that equal to the level's target resolution gives the **derived
crop size**:

```
c_k = image_size · resolutions[0] / resolutions[k]
```

For `resolutions=[32, 64]`, `image_size=128`:
- Level 0 (coarse): `c = 128·32/32 = 128` → the full image, effective resolution 32.
- Level 1 (refine): `c = 128·32/64 = 64` → a 64px crop, 32×32 tokens, effective resolution 64.

`refine_crop` is therefore **not a configured knob** — it is derived from `resolutions`.
Constraint: each `resolutions[k]` must be a multiple of `resolutions[0]` such that `c_k` is a
whole number in `(0, image_size]`.

Because every level uses the same `T`, `_segment` needs **no resolution parameterization** —
the refine pass is just `_segment` run on the crop (resized to `image_size`). Multi-level is
active when `len(resolutions) > 1`; a single-element list is the plain model, unchanged.

## Decisions (from brainstorming)

- **Per-level losses now; unified/fused loss later.** Each level is supervised on its own
  grid against the appropriately pooled/cropped GT. Stitching the levels into a single fused
  prediction is used for **metrics only** for now; a fused *loss* is deferred.
- **Effective-resolution semantics** for `resolutions` (full-image), constant token count, derived crop (above).
- **Bbox source (target): prediction max-prob window** in both train and eval (deployment-faithful).
- **Bbox source (context): densest-GT window** per context image.
- **Single top-1 window / single refine level to start** (`resolutions=[32,64]`), structured so
  a third entry (e.g. `128` → 32px crop) chains another level with no redesign.
- **Fused metric named `_fused`** (`dice_fused@R`, `dice_fused_soft@R`): the stitched
  full-image prediction (coarse with the refine crop placed in), scored — no loss on it yet.
- **Checkpoint selection** is on `dice_fused@{resolutions[-1]}` (the fused full-image metric at
  the finest effective resolution) when the refine model is active — it tracks the actual
  final-prediction quality. Plain single-level models keep the existing selection metric.

## Architecture

### `PatchSetCNN` changes (`src/models/patchset_cnn.py`)

1. **Constructor**: replace `refine`/`refine_crop` with `resolutions: list[int] | None = None`.
   `None` → `[resolution]` (single level = plain model, unchanged). `self.token_res =
   resolutions[0]` is the token grid `T` (drives the encoder, as `resolution` does today).
   `self.resolutions = list(resolutions)`. Validate each `resolutions[k]` (k≥1) is a multiple
   of `resolutions[0]` and yields an integer crop `c_k = image_size·resolutions[0]/resolutions[k]`
   in `(0, image_size]`; store the derived crop sizes.
2. **`_segment(image, context_in, context_out) -> (B,1,T,T)`**: the current coarse forward body
   (unchanged — fixed at the `T` token grid). Both passes call it.
3. **`forward(image, context_in, context_out, mode="train")`**:
   - Single level (`len(resolutions)==1`): `return {"final_logit": self._segment(...)}` —
     byte-for-byte the plain model.
   - Multi level: run the per-level flow and return per-level heads + geometry (below). No fusion.

Multi-level forward (2 levels; H,W = image_size; T = resolutions[0]; c = derived crop):

```
coarse = self._segment(image, context_in, context_out)             # (B,1,T,T) full image
prob   = torch.sigmoid(coarse).detach()
prob_up = interpolate(prob, (H,W), bilinear)                       # for bbox selection
tgt_o = max_sum_window(prob_up, c)                                 # (B,2) px origin, target
ctx_o = stack[ gt_window(context_out[:,k], c) for k in K ]         # (B,K,2) px origins, context
tgt_img = crop_resize(image,           tgt_o,        c, H, bilinear)
ctx_img = crop_resize(context_in|>BK,  ctx_o|>BK,    c, H, bilinear) |> (B,K,1,H,W)
ctx_msk = crop_resize(context_out|>BK, ctx_o|>BK,    c, H, nearest) |> (B,K,1,H,W)
refine = self._segment(tgt_img, ctx_img, ctx_msk)                  # (B,1,T,T) the crop, same weights
return {
  "final_logit":  coarse,          # (B,1,T,T)  effective resolutions[0]; drives the existing suite + ckpt
  "refine_logit": refine,          # (B,1,T,T)  effective resolutions[1]; the crop
  "refine_origin": tgt_o,          # (B,2) px top-left in the H×W frame
  "refine_crop":  c,               # int px
  "resolutions":  self.resolutions,
}
```

Bbox origins come from `argmax` (detached). Both passes share weights via `_segment`.

### `bbox_refine.py` addition

- `place_window(full, patch, origin, s) -> (B,1,H,W)` — **replace** variant of `fuse_window`:
  clones `full` and writes `patch (B,1,s,s)` into the s×s window at each `origin` (overwrite,
  not add). Used to build the fused/stitched prediction for the metric. `fuse_window`
  (additive, logit-space) is retained for the deferred fused *loss*.

### Shared metric helper

Per-level + fused metrics are computed the same way in training and validation, so a single
helper (in `evaluate.py`, reused by `train.py`) takes `(out_dict, label)` and returns:
- `dice@{resolutions[0]}` / soft — coarse `final_logit` vs GT pooled to T over the full image.
- `dice@{resolutions[1]}` / soft — `refine_logit` vs `crop_resize(label, refine_origin, c, T)`.
- `dice_fused@{resolutions[1]}` / soft — the **fused** (stitched) prediction: build at native
  `image_size` as `place_window(upsample(sigmoid(coarse),H), upsample(sigmoid(refine),c),
  refine_origin, c)`, then pool to `resolutions[1]` and score vs GT pooled to `resolutions[1]`.

(The coarse level continues to flow through the trainer's/`validate()`'s existing single-logit
metric suite via `final_logit`, so `dice`, `dice_ds@…`, `cossim`, etc. stay as the effective-32
level with no relabeling; the two extra families above are added for the refine model.)

### Training integration (`experiments/2d/train.py`)

- Loss: keep the existing `final_logit` (coarse) BCE+soft-Dice as level 0. When
  `out["refine_logit"]` is present, add `refine_loss_weight · (BCE + dice_weight·softdice)` of
  the refine logit vs `crop_resize(label, refine_origin, refine_crop, T)`. `loss = coarse +
  refine_loss_weight·refine`. New knob `train.refine_loss_weight: 1.0`.
- Metrics: after the existing coarse metrics, call the shared helper to log `train/dice@64`,
  `train/dice_fused@64` (+ soft) when the refine head is present.
- `build_model` (patchset_cnn branch): pass `resolutions` (list) into the `arch` dict instead of
  `refine`/`refine_crop`, so eval rebuilds identically.
- Checkpoint selection: when the val summary contains `dice_fused@{resolutions[-1]}`, select on
  it (best = max); otherwise keep the existing selection metric. The console/`best_*` logging
  follows the selected metric.

### Eval integration (`experiments/2d/eval_incontext.py` / `evaluate.py`)

`validate()` computes the same shared-helper metrics (`dice@64`, `dice_fused@64`) for the
refine model, in addition to the existing coarse suite on `final_logit`. `eval_incontext.py`
needs no change beyond what `validate()` emits. Checkpoint `arch` carries `resolutions`, so the
model rebuilds with zero drift.

### Config

- `configs/experiment/2d/model/patchset_cnn.yaml`: keep `resolution` scalar (token grid for the
  plain model); the refine leaf sets the list.
- `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`:
  ```yaml
  defaults:
    - 1_omnisynth_medseg
    - _self_
  arch:
    resolutions: [32, 64]      # effective full-image resolutions; T=32 tokens/level, refine crop=64px derived
  train:
    refine_loss_weight: 1.0
  eval:
    ds_metric_res: [16, 32]    # coarse-grid pooled Dice, comparable to the plain model
  ```

## Metrics summary

| metric | level | grid | vs |
|---|---|---|---|
| `dice`, `dice_ds@…`, `cossim`, … | coarse (eff. 32) | T×T full image | existing suite on `final_logit` |
| `dice@64` (+ soft) | refine (eff. 64) | T×T on the crop | `crop_resize(GT, origin, c, T)` |
| `dice_fused@64` (+ soft) | fused/stitched (eff. 64) | 64×64 full image | GT pooled to 64 |

Checkpoint metric: `dice_fused@{resolutions[-1]}` (= `dice_fused@64` here) for the refine model;
the existing metric for plain single-level models.

## Edge cases

- **Empty coarse prediction**: `max_sum_window` centers the crop; the refine level still runs
  and is scored, its contribution near-zero where nothing is present.
- **Empty context GT**: `gt_window` centers that context's crop.
- **Non-integer / out-of-range crop**: rejected at construction by the `resolutions` validation.
- **Object larger than the crop**: only the in-crop part is refined/fused; outside the crop the
  fused prediction is the coarse value (by construction).

## Testing

- `bbox_refine.py`: existing tests + `place_window` (replace into the right window, input not
  mutated, distinct from additive `fuse_window`).
- `PatchSetCNN`: single-level (`resolutions=[32]` / default) output unchanged `(B,1,T,T)` equal
  to `_segment`; multi-level (`resolutions=[32,64]`) returns the per-level dict with correct
  shapes, derived crop, detached origins; gradient reaches shared weights from BOTH heads;
  `resolutions` validation rejects a bad list (e.g. `[32, 48]` → non-integer crop).
- Shared metric helper: on a hand-built case, `dice@64` uses the cropped GT and `dice_fused@64`
  equals coarse Dice when the refine logit matches the coarse crop (fused ≡ coarse when refine
  adds nothing), and improves when refine is correct inside the crop.
- Smoke run: `train.py --config-name 2_omnisynth_medseg_refine train.epochs=1
  data.max_train_samples=64 eval.max_per_label=4 wandb.enabled=false` — loss decreases,
  `dice@64` / `dice_fused@64` logged, checkpoint reloads under `eval_incontext.py`.

## Out of scope (YAGNI)

- Fused **loss** on the stitched prediction (metric only for now).
- >2 levels / top-k windows (structured to extend; not built).
- Teacher-forced or scheduled bbox source (prediction-driven chosen).
- Query-prior seeding of the refine pass.

---

## Superseded approach (2026-07-09 original): single native composite via additive logit fusion

The original design produced one native-resolution `final_logit` by additively fusing the
refine logit (residual) into the upsampled coarse logit at the bbox, trained under the
trainer's single BCE+soft-Dice loss. It was implemented (branch `patchset-refine`, commits
`4a2d836`..`704c5cd`) and then revised because: (a) it gave no independent per-level training
signal or per-level metrics, and (b) attempts to control the refine pass's resolution exposed a
geometric conflict (a single fused canvas forces one global resolution; a sub-image crop cannot
be both "effective 64 over the full image" and a genuine higher-token grid without either
landing at native 128 or discarding the refine detail). The per-level design above replaces it;
`fuse_window` is retained for the future fused-loss experiment.

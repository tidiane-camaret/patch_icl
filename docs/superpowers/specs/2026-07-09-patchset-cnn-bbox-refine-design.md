# Single-level bbox-zoom refinement for PatchSetCNN

Date: 2026-07-09
Experiment: `src/models/patchset_cnn.py` (new `refine` mode), trainable/evaluable through the
existing `experiments/2d/train.py` and `experiments/2d/eval_incontext.py`.

## Problem

`PatchSetCNN` predicts a coarse R×R logit over the full image (R=16/32). Small or
detailed objects lose sharpness at that resolution. We want a **coarse→fine refinement**:
after a first R×R prediction, crop a square window around the densest predicted region,
re-segment that crop with the **same model**, and fuse the finer result back — so detail
is recovered where it matters without a decoder/upsampling head.

Prior art in `experiments/2d/multilevel/` (zoom_pipeline.py, bbox.py, its own train.py)
built this around the older `ImagePFN`/`PatchSetPFN` with a **frozen** stage-1 plus
**separate per-hop** trainable copies and a dedicated training script. This design is
deliberately simpler: **one shared `PatchSetCNN`** applied twice, folded inside the model's
`forward`, so the existing unified trainer/eval run it with no changes.

## Decisions (from brainstorming)

- **Shared weights, two passes.** The same `PatchSetCNN` (same encoder/transformer/decoder)
  runs the coarse pass on the full image and the refine pass on the crop. No frozen stage,
  no per-hop copies.
- **Bbox source (target): prediction max-prob window**, in **both** train and eval
  (deployment-faithful). Early-training coarse predictions are near-random, so early refine
  crops are noisy; accepted, because the additive fusion (below) keeps the coarse pass
  directly supervised so it learns to localize quickly.
- **Bbox source (context): densest-GT window** per context image (`gt_window`), mirroring
  the target's "densest region" selection but using the known context mask.
- **Single top-1 window** per image (one refine forward). Structured so top-k / a multi-hop
  ladder is an easy later extension; not built now (YAGNI).
- **Logit fusion (single loss).** The refine logit is added as a **residual in logit space**
  into the coarse canvas at the bbox: `fused = upsample(coarse); fused[bbox] += upsample(refine)`.
  Because the coarse logit still contributes inside the object region, it keeps receiving a
  direct training signal there (it must stay accurate to place the next bbox), while the
  refine pass learns a correction. One `sigmoid(fused)` → the trainer's existing single
  BCE + soft-Dice loss. No auxiliary losses, no trainer change.
- **Native-resolution output.** The fused canvas is at `image_size` (H×W). This is the point
  of the experiment — recover detail the coarse grid can't represent — and it makes the
  output directly comparable to the UniverSeg native-Dice baseline.
- **No query-prior seeding** for the refine pass. `PatchSetCNN` already derives its query
  occupancy prior from the (cropped) context masks; the additive coarse logit supplies the
  rest. The refine pass is an ordinary `forward` call on the cropped images — identical
  interface — which keeps the code readable.

## Architecture

### `PatchSetCNN` changes (`src/models/patchset_cnn.py`) — minimal, default-off

1. **Constructor**: add `refine: bool = False`, `refine_crop: int = 64` (square-bbox side
   length in the H×W image frame). Defaults reproduce today's behavior exactly.
2. **Refactor** the current `forward` body into a private
   `self._segment(image, context_in, context_out) -> (B,1,R,R)` returning the coarse logit.
   Both passes call it — guaranteeing identical weights and behavior.
3. **`forward`**:
   - `refine=False` → `return {"final_logit": self._segment(...)}` — byte-for-byte unchanged
     `(B,1,R,R)`.
   - `refine=True` → run the two-pass flow and return `{"final_logit": fused}` `(B,1,H,W)`.

Two-pass flow (H,W = image_size; s = refine_crop; R = resolution):

```
coarse   = self._segment(image, context_in, context_out)          # (B,1,R,R) logits
coarse_up = interpolate(coarse, (H,W), bilinear)                   # logit-space canvas
prob_up   = interpolate(sigmoid(coarse).detach(), (H,W), bilinear) # for bbox selection only

tgt_o = max_sum_window(prob_up, s)                                 # (B,2) target crop origin
ctx_o = stack[ gt_window(context_out[:,k], s) for k in K ]         # (B,K,2) context origins

# crop each image/mask to its s×s window and resize back to H (same input distribution the
# encoder was trained on); context masks use nearest, images/prob use bilinear.
tgt_img = crop_resize(image,           tgt_o,        s, H, bilinear)   # (B,1,H,W)
ctx_img = crop_resize(context_in|>BK,  ctx_o|>BK,    s, H, bilinear)   # -> (B,K,1,H,W)
ctx_msk = crop_resize(context_out|>BK, ctx_o|>BK,    s, H, nearest)    # -> (B,K,1,H,W)

refine  = self._segment(tgt_img, ctx_img, ctx_msk)                # (B,1,R,R) logits, same weights
refine_s = interpolate(refine, (s,s), bilinear)                   # (B,1,s,s)
fused   = fuse_window(coarse_up, refine_s, tgt_o, s)              # (B,1,H,W): coarse_up[bbox] += refine_s
return {"final_logit": fused}
```

`tgt_o`/`ctx_o` come from `argmax` and are **detached** (hard routing constants). Everything
else is differentiable: gradient reaches the shared weights from both the coarse background
(loss outside the bbox and, additively, inside it) and the refine residual (inside the bbox).

### New module: `src/models/bbox_refine.py` (pure tensor ops, unit-tested)

Lifted/adapted from `experiments/2d/multilevel/bbox.py` so `src/models/` needs no
`experiments/` import. Functions:

- `max_sum_window(prob, s) -> (B,2)` — top-left origin of the s×s window with the largest
  summed value (box-sum via `avg_pool2d` stride-1 + argmax); empty maps (max ≤ ε) center the
  crop instead of collapsing to the corner. Origins clamped in-bounds.
- `gt_window(mask, s) -> (B,2)` — same on a binary/soft mask (densest-GT window).
- `crop_resize(x, origin, s, out, mode) -> (N,C,out,out)` — batched per-sample crop to the
  s×s bbox at `origin`, resampled to `out×out` via `grid_sample` (align_corners=False,
  border padding). Resolution-agnostic, vectorized over the batch.
- `fuse_window(full, patch, origin, s) -> (B,1,H,W)` — the **additive** variant of the
  reference's `composite_window`: writes `full[b, :, r0:r0+s, c0:c0+s] += patch[b]` into a
  clone (input not mutated). Per-sample loop over the batch dim (B small).

The existing `experiments/2d/multilevel/bbox.py` is left untouched.

### `build_model` (`experiments/2d/train.py`) — thread the two flags

In the `patchset_cnn` branch, add to the `arch` dict:
`"refine": a.get("refine", False), "refine_crop": a.get("refine_crop", 64)`.
Nothing else in `train.py` changes — the loop already pools GT to `final_logit`'s size and
computes one BCE + soft-Dice loss, which now lands at native resolution.

### Config

- `configs/experiment/2d/model/patchset_cnn.yaml`: add under `arch:` (default off)
  ```yaml
  refine: false        # enable coarse→fine bbox-zoom refinement (native-res fused output)
  refine_crop: 64      # square bbox side length (pixels in the image_size frame)
  ```
- New runnable leaf `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`:
  ```yaml
  # Experiment 2 — PatchSetCNN with single-level bbox-zoom refinement, on the same
  # omniSynth/MedSeg distribution as experiment 1.
  #   python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine
  defaults:
    - 1_omnisynth_medseg
    - _self_
  arch:
    refine: true
    # refine_crop: 64
  eval:
    ds_metric_res: [16, 32]   # keep pooled coarse Dice alongside the new native Dice
  ```
  `1_omnisynth_medseg` defaults `model: patchset_cnn`, so this inherits it; enable ad-hoc on
  any run with `arch.refine=true`.

## Metrics / eval

`final_logit` is native H×W, so `validate()` treats the model like a native predictor:

- `dice/mean`, `dice_soft/mean` — hard/soft Dice of the fused native output vs GT. With
  native output the trainer's checkpoint-selection metric becomes `dice` (no `cossim`), which
  is the right target for a refinement model.
- `dice_ds@16 / dice_ds@32` (and their `_soft`) — from `eval.ds_metric_res: [16, 32]`: the
  fused output avg-pooled to R×R, so coarse-grid Dice stays comparable to the non-refine
  PatchSetCNN runs.
- No changes to `evaluate.py` / `eval_incontext.py`; the checkpoint's `arch` block now carries
  `refine`/`refine_crop`, so eval rebuilds the refine model automatically.

## Edge cases

- **Empty coarse prediction** (max window ≤ ε): `max_sum_window` centers the crop; refine runs
  on the centered window and its residual is fused normally (near-zero where nothing is there).
- **Empty context GT**: `gt_window` centers that context's crop.
- **Object larger than `refine_crop`**: only the in-bbox part is sharpened; outside the bbox
  the fused output is the coarse prediction (by construction). Default `refine_crop=64` covers
  typical omniSynth objects; tune via config.
- **Multiple objects**: single top-1 window refines the densest one; others stay coarse
  (accepted for now; top-k is the documented extension).

## Testing

- `tests` for `bbox_refine.py`: `max_sum_window`/`gt_window` on hand-built blobs (centered,
  off-center, border→clamped, empty→centered); `crop_resize` crop-then-place round-trip;
  `fuse_window` adds into the right window and leaves the rest unchanged; in-bounds invariants.
- `PatchSetCNN` forward tests: `refine=False` output shape unchanged `(B,1,R,R)` and equal to
  a direct `_segment` call; `refine=True` returns `(B,1,H,W)`, is finite, and a fused-loss
  backward populates `decoder`/`encoder` grads (both passes contribute).
- Smoke run: `python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine
  train.epochs=1 data.max_train_subjects=...` (tiny) to confirm the loss decreases and a
  checkpoint that reloads under `eval_incontext.py`.

## Out of scope (YAGNI)

- Multi-hop zoom ladder and top-k windows (single level / single window now; structure allows
  adding a loop and NMS-style selection later).
- Teacher-forced or scheduled bbox source (prediction-driven chosen).
- Query-prior seeding of the refine pass (additive coarse logit supplies the prior).
- Re-encoding vs encode-once feature caching (each pass is a plain `forward`; the coarse and
  refine encodes are separate and cheap at this resolution — no shared-feature plumbing).

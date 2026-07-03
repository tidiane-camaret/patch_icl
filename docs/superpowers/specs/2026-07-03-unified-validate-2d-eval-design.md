# Unified `validate()` + focused 2D eval script (universeg / patchset_cnn)

**Date:** 2026-07-03
**Status:** Approved design, ready for implementation plan

## Problem

`experiments/2d/eval.py` is a ~1100-line, 5-backend dispatcher (universeg_featuresim /
pfn_seg_2d / patchset_pfn / imagepfn_zoom / universeg). It is hard to extend and does
**not** handle `patchset_cnn` at all (that model would fall through to the UniverSeg
branch and be mis-scored, since its logit is low-res R×R, not native H×W).

Separately, `train.py` has its own `validate()` loop. Its metric set overlaps but does
not match `eval.py`'s, so training-time val Dice and eval-time Dice can silently diverge.

## Goal

A focused evaluation path for the two trained in-context models `universeg` and
`patchset_cnn`, built on a **single shared `validate()` used by both `train.py` and the
new eval script**, so metrics are coherent by construction. Expensive extras (figures,
CSVs) are opt-in; everything else is always logged.

Out of scope: TabPFN / pfn_seg_2d / multilevel / zoom backends (they keep using the
existing `eval.py` unchanged), Strategy-A `encode_size`, `stage1_checkpoint`.

## Architecture

New module **`experiments/2d/evaluate.py`** owns the shared eval logic. Both `train.py`
and the new wrapper import from it. `common.py` stays a lean data/metrics library; the
big `eval.py` is left untouched.

```
common.py            data + metrics + log_summary (unchanged)
evaluate.py  (NEW)   validate(), save_figure(), _sample_detail() formatter
train.py             imports validate; drops its own copy
eval_incontext.py (NEW)  thin Hydra wrapper: load ckpt → build model → validate()
```

## Component: `evaluate.py`

### `validate()`

```python
def validate(model, loader, *, topk_k=16, epoch=0,
             figures=None,        # None | {"out_dir": Path, "max_figures": int, "to_wandb": bool}
             patch_csv=None,      # None | str path   (patchset_cnn low-res grid only)
             synth_csv=None,      # None | str path   (data.source == "synthetic" only)
             compute_flops=False  # measure once on the first batch
) -> tuple[dict, "wandb.Table", float | None]:
    """Returns (summary, sample_table, flops).

    summary is wandb-ready (keys: dice/mean, dice/dataset/*, dice/macro,
    dice_ds/*, dice_ds_soft/*, cossim/*, top{k}/*, plus flops_giga when measured).
    """
```

**Model-agnostic loop** (identical to today's forward path):
`model(img, context_in=cin, context_out=cout, mode="val") -> {"final_logit"}`, under
bf16 autocast. Native vs low-res is detected by comparing the logit spatial size to the
label size (reuse the existing `_target_like` / `_upsample_to` helpers).

**Always computed** (NaN-skipped when native, so UniverSeg reports only `dice`):
- `dice` — native hard Dice (preds upsampled to H×W vs full-res GT). Comparable across models.
- `dice_ds` — low-res hard Dice (pred grid vs GT avg-pooled to grid, binarized at ≥0.5).
- `dice_ds_soft` — low-res soft/shape Dice (pred grid vs soft pooled GT).
- `cossim` — scale-invariant low-res similarity (skipped at native res).
- `top{topk_k}` — top-k patch overlap (skipped at native res).
- Per-sample table (see below).

**Aggregation is internal:** `validate()` runs the `log_summary` calls and returns the
merged `summary` dict. This is the coherence win — one place aggregates. Callers just
`wandb.log(summary)`. `train.py` reads `summary["cossim/mean"]` or `summary["dice/mean"]`
for best-checkpoint selection; that policy stays in `train.py`.

**FLOPs (always-on, measured once):** when `compute_flops=True`, wrap the first batch's
forward in `FlopCounterMode`; return GFLOPs and add `flops_giga` to `summary`. Model
interface is uniform, so no per-model dummy shapes are needed.

**Gated extras** fire only when their arg is set:
- `figures` → save qualitative panels (Section: Figures).
- `patch_csv` → per-low-res-patch error rows (pred, soft GT, signed error, gt_size,
  ctx_dice). Only meaningful for `patchset_cnn` (low-res grid); no-op/skip for native.
- `synth_csv` → per-element controlSynth difficulty rows; only when `source == "synthetic"`.

### Per-sample table (adaptive, fixed columns)

Columns: `[epoch, dataset, sample_idx, label, dice, dice_ds, dice_ds_soft, detail]`.

`detail` is a single string built by `_sample_detail(meta)`:
- omniSynth meta → `"alphabet/class mode=<target_mode> cells=<target_cells> tf=<transforms>"`
  (preserves the current train.py debugging info).
- controlSynth meta → morphology / task_id.
- no meta (medsegbench) → `""`.

Fixed columns keep `wandb.Table` happy across sources; source-specific richness lives in
`detail`.

### `save_figure()`

Moved verbatim from `eval.py`. One figure per `(dataset, label_value)`, capped at
`figures["max_figures"]`. Row 0: target+GT, target+pred, and (GT↓ | pred↓) heatmaps when
the model has a coarse grid (patchset_cnn); the low-res pair is omitted for native
UniverSeg. Row 1: the K context overlays. Optional wandb upload via `figures["to_wandb"]`.

## Component: `train.py` migration

- Remove `train.py`'s `validate()` and its private helpers (`_fmt_transforms`,
  `SAMPLE_COLS`, `_topk_sum`/`_soft_sum` bits that only served it); move them into
  `evaluate.py`.
- `main()` calls `validate(model, val_loader, topk_k=..., epoch=epoch)` with all gated
  extras off and `compute_flops=(epoch == 0)` — so FLOPs is logged once at the start of
  training (not re-measured every epoch) as well as on every eval run.
- Keep the existing checkpoint policy unchanged:
  `metric = "cossim" if "cossim/mean" in summary else "dice"`; save when
  `summary[f"{metric}/mean"]` improves.
- **Behavior must be identical**: same metric keys logged, same best-checkpoint logic.
  This is a pure extraction. Verification: run one debug training epoch before and after
  and confirm the same wandb metric keys appear with matching values.

## Component: `eval_incontext.py` (new wrapper)

Hydra entrypoint (~60–80 lines). **Reuses the existing `eval_base.yaml`** as its
`config_name` — it already carries every field the wrapper needs (`data.split`,
`eval.checkpoint`, `eval.save_figures`/`max_figures`/`figures_to_wandb`,
`eval.patch_csv`, `eval.synth_csv`, `wandb.*`). The tabpfn/stage1/encode_size fields are
irrelevant to these two models and simply go unread — no new config file.

1. `ckpt = torch.load(cfg.eval.checkpoint, ...)`; **dispatch on `ckpt["model_name"]`**.
2. Rebuild the one model:
   - `universeg` → `UniverSegBaseline(pretrained=True, input_size=ckpt["image_size"])`;
     load `ckpt["model"]` (strip `_orig_mod.`).
   - `patchset_cnn` → `PatchSetCNN(image_size=ckpt["image_size"], **ckpt["arch"])`;
     load state. **Fail loudly** if `ckpt.get("arch")` is missing (pre-`arch` checkpoint).
3. `loader = build_loader(cfg)` (any source, `cfg.data.split`).
4. `summary, table, flops = validate(model, loader, epoch=0,
   figures={...} if cfg.eval.save_figures else None,
   patch_csv=cfg.eval.get("patch_csv"), synth_csv=cfg.eval.get("synth_csv"),
   compute_flops=True)`.
5. `wandb.init(...)`, `wandb.log(summary)`, `wandb.log({"samples": table})`, write CSVs,
   `run.finish()`.

## Data flow

```
checkpoint (best.pt: model_name, image_size, arch, ...) ──► build model
Hydra cfg.data ──► build_loader ──► batches
batches + model ──► validate() ──► summary + table (+ figures/CSVs/flops)
summary ──► wandb.log
```

## Error handling

- Missing `ckpt["arch"]` for `patchset_cnn` → raise with a clear message ("checkpoint
  predates full-arch storage; retrain or re-save").
- Unknown `ckpt["model_name"]` → raise (`universeg | patchset_cnn` only).
- Empty/NaN Dice rows → already handled by `hard_dice`/`soft_dice`/`log_summary`
  (NaN-filtered).
- `patch_csv` requested for a native (non-grid) model → skip with a warning.

## Testing / verification

- **Extraction parity (primary risk):** one debug training epoch
  (`data.max_train_samples` small, `train.epochs=1`) before vs after the migration logs
  the same metric keys with matching values.
- **patchset_cnn round-trip:** load a `patchset_cnn` best.pt, rebuild via
  `PatchSetCNN(image_size=..., **ckpt["arch"])`, run `validate()` end-to-end, confirm
  `dice_ds`/`dice_ds_soft`/`cossim`/`top-k` populated and `dice` (native) present.
- **universeg smoke:** run the wrapper on a universeg checkpoint; confirm only native
  `dice` populated, low-res metrics NaN-skipped, no figures crash.
- No new unit tests beyond these (per repo guideline: tests only when necessary).

## Files touched

- NEW `experiments/2d/evaluate.py`
- NEW `experiments/2d/eval_incontext.py` (uses existing `eval_base.yaml`)
- EDIT `experiments/2d/train.py` (drop local `validate()`, import shared)
- `eval.py` and `eval_base.yaml` unchanged.
- Log to `docs/logs.md`.

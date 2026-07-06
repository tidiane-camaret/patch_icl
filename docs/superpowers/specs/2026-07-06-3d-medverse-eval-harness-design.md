# 3D in-context experiments harness — eval sub-project (Medverse)

Date: 2026-07-06

## Goal

Build `experiments/3d/{eval,train}.py` that both train and evaluate 3D in-context
models (Medverse first), sharing `common.py` + `evaluate.py` — mirroring how
`experiments/2d/train.py` and `experiments/2d/eval_incontext.py` share
`experiments/2d/common.py` + `experiments/2d/evaluate.py`.

Sequenced in two sub-projects (decision: eval first, train after):
- **This spec — Sub-project A (eval harness):** shared modules + `eval.py` with
  Medverse eval working end-to-end. `train.py`'s interface is designed but its
  training loop and the trainable-Medverse forward are deferred.
- **Sub-project B (train harness, future spec):** `train.py` loop + a trainable
  single-ROI forward on `MedverseModel` (the current adapter is inference-only:
  `.predict()` is multi-scale sliding-window under `@torch.no_grad`).

## Module layout (mirrors 2D)

| 2D | 3D |
|---|---|
| `common.py` (build_dataset, make_loader, DEVICE) | `experiments/3d/common.py` (extend) |
| `evaluate.py` (`validate`) | `experiments/3d/evaluate.py` (new) |
| `train.py` | `experiments/3d/train.py` (future) |
| `eval_incontext.py` | `experiments/3d/eval.py` (new) |

### `experiments/3d/common.py` (extend — already has `build_dataset`)
- Add `DEVICE`.
- Add `make_loader(cfg, cls, split="test") -> DataLoader`: a single-class eval
  loader. Builds `TotalSegInContextDataset(classes=[cls], split, image_size,
  context_size, use_crop, max_subjects=cfg.eval.n_subjects, aug_cfg=None,
  synth_method=None, p_synth=0, class_balanced=False)`, root/is_mri resolved from
  `cfg.data.source` (reuse the same dispatch as `build_dataset`). Loader params
  (`batch_size`, `workers`) from `cfg.eval`.

### `experiments/3d/evaluate.py` (new — the shared eval unit)
Ported from `scripts/eval.py` (harness independence; scripts/eval.py stays legacy):
- `dice_binary(pred, target) -> float`
- `save_eval_figure(...)`
- `measure_flops(model, image_size, K, device) -> gflops`
- `validate(model, loader, cls, *, fig_dir=None, compute_flops=False) ->
  (summary_row, cases)` — runs `model.predict()` over the loader, per-case Dice +
  time, optional first-case figure. Both `eval.py` (now) and `train.py`'s val step
  (later) call this via the `model.predict()` interface every benchmark model
  exposes.

### `experiments/3d/eval.py` (new — orchestration, mirrors `eval_incontext.py`)
- `@hydra.main(config_path="../../configs", config_name="config")`, run with
  `experiment=3d/eval`.
- Model: `load_model(cfg.eval.model, ...)`, default `medverse`. Small special-case
  so a custom `cfg.eval.medverse_ckpt` / `sw_roi_size` reaches `MedverseModel`
  (load_model drops `ckpt_path` for medverse). Non-medverse models take
  `ckpt_path=cfg.eval.checkpoint`.
- Classes: `resolve_classes(cfg.data.val_classes, root, is_mri)`.
- Loop classes → `common.make_loader` → `evaluate.validate` → aggregate
  mean-over-classes. Write CSV + JSON + figures to
  `${paths.results}/3d_eval/<date>_<run>/`; log per-class + overall to wandb.
  FLOPs measured once.

### `configs/experiment/3d/eval.yaml` (new, `# @package _global_`)
```yaml
data: { source: totalseg, image_size: [128,128,128], context_size: 1,
        use_crop: true, val_classes: benchmark }
eval: { model: medverse, split: test, n_subjects: 50, batch_size: 8, workers: 20,
        save_figures: true, max_figures: 200, out_dir: ${paths.results}/3d_eval,
        medverse_ckpt: null, sw_roi_size: null, checkpoint: null, seed: 0 }
wandb: { project: patch_icl_3d_eval, name: null }
```

## Sub-project B — train.py (Medverse fine-tune)

Decision: full fine-tune, config-driven size, train resolution 128³.

- **`src/benchmark_models/medverse.py`**: add `train_forward(target, ctx_img,
  ctx_mask, l=None) -> logits` (grad-enabled single-ROI `self.model.forward`, same
  normalization as `predict`) and `load_finetuned(state_dict)`. `predict` unchanged.
  `LightningModel.forward` returns raw logits (output block ends in 1×1 conv).
- **`experiments/3d/common.py`**: add `train_loader(cfg)` — DataLoader over
  `build_dataset(cfg,"train")` with train batch/workers + optional
  `RandomSampler(max_ds_len_train)`.
- **`experiments/3d/evaluate.py`**: add `evaluate_classes(model, cfg, classes,
  fig_dir=None) -> (rows, cases)`; refactor `eval.py` to use it; `train.py` val step
  reuses it for mean-Dice checkpoint selection.
- **`experiments/3d/train.py`**: Hydra `experiment=3d/train`. Full fine-tune
  `medverse.model.parameters()`, AdamW + cosine-warmup, AMP bf16. Loss =
  BCE-with-logits + dice_weight·soft_dice(sigmoid(logits)). Val every `eval_every`
  via `evaluate_classes`; save best `{"model": medverse.model.state_dict(),
  "model_name":"medverse", image_size, context_size, ...}`.
- **`configs/experiment/3d/train.yaml`** (`# @package _global_`): data (image_size
  [128,128,128], synth/class params), train block, minimal eval block (for the
  val make_loader), model: medverse, wandb.
- **eval loop closure**: `eval.py` medverse builder loads `cfg.eval.checkpoint`
  via `load_finetuned` when set.

## Testing
Smoke: `python experiments/3d/eval.py experiment=3d/eval eval.model=medverse
eval.n_subjects=3 data.val_classes='[liver]'` → produces a Dice number, a figure,
and a CSV. No unit tests (per CLAUDE.md). `scripts/eval.py` untouched.

## Notes
- ~60 lines of loop logic are duplicated from `scripts/eval.py` into
  `evaluate.py` (intentional). A later refactor could point `scripts/eval.py` at
  the harness; out of scope.
- load_model can't pass a custom medverse checkpoint through its `ckpt_path`
  param (bound + dropped for the medverse branch) — handled by the eval.py
  special-case.

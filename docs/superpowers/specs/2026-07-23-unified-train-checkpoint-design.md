# Unified `train.checkpoint` for 3D medverse training

Date: 2026-07-23

## Goal

Collapse the three medverse weight-source knobs — `train.random_init`,
`train.base_ckpt`, and the path-resume role of `train.checkpoint` — into a single
`train.checkpoint` field.

## Accepted values

| `train.checkpoint` | Behavior (medverse) |
|---|---|
| `"orig_weights"` | Fine-tune from the released `Medverse.ckpt` (`MEDVERSE_CKPT`). Default for medverse. |
| `"random"` | Train from scratch (random init from the checkpoint's hparams). |
| `<path>` | Warm-start from our finetuned `best.pt` via `MedverseModel.load_finetuned`. |

`SENTINELS = {"orig_weights", "random"}`. Any non-sentinel, non-null value is a path.

## Changes

1. `experiments/3d/train.py` `build_model` (medverse branch): drop `base_ckpt` /
   `random_init` handling. Special-case only `checkpoint == "random"` →
   `random_init=True`. `"orig_weights"` and path both construct with released weights
   (the path case is overridden in step 2).
2. `experiments/3d/train.py` `main()` checkpoint block: only `torch.load` + apply
   weights when `checkpoint` is a real path (not a sentinel/null). Patchset3d resume is
   unchanged: a path → `load_state_dict`; a sentinel/null → fresh init.
3. `configs/experiment/3d/model/medverse.yaml`: remove `base_ckpt` and `random_init`;
   add `train.checkpoint: orig_weights` (medverse default, overrides `train.yaml` null).
4. `configs/experiment/3d/train.yaml`: keep `checkpoint: null` (generic default:
   patchset null=fresh / path=resume); document the sentinels in the comment.
5. `src/benchmark_models/medverse.py`: unchanged (keeps `random_init` / `ckpt_path`
   constructor params; only the config layer stops using `base_ckpt`).

## Trade-offs

- Drops `base_ckpt` (pointing `"orig_weights"` at a different raw Medverse `.ckpt`).
  Currently `null` everywhere and unused; YAGNI. `MEDVERSE_CKPT` stays a module constant.
- `eval.yaml`'s `checkpoint` is a separate concern (eval entrypoint) and is unchanged.

No new tests (pure config plumbing; no new logic).

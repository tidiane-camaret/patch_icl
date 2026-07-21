# Clarify `configs/experiment/3d/` — train/eval entrypoints composing dataset + model groups

**Date:** 2026-07-21
**Status:** Design approved, pending spec review

## Goal

Reorganize the flat, mixed-concern `configs/experiment/3d/` directory into a clean
Hydra layout: `train.yaml` / `eval.yaml` entrypoints that **compose** `dataset` and
`model` config groups and can be **overridden by an optional `experiment` preset**.
This makes the two file *kinds* explicit — reusable mixins (`dataset/`, `model/`) vs
runnable entrypoints (`train.yaml`, `eval.yaml`) — and removes the current duplication
(the `synth3d` block lives in both `omnisynth3d.yaml` and `omnisynth3d_eval.yaml`).

No dataset/model/eval **logic** changes; this is a config + hydra-entry refactor.
Retro-compatibility is explicitly **not** a goal — old `experiment=3d/<name>`
invocations are replaced by the new forms below.

## Verified mechanism

The 3D scripts need `paths.totalseg` / `paths.results`, which come from the global
`configs/config.yaml` + `configs/cluster/` chain. Moving the hydra entry to a 3D-local
root would normally lose that. **Hydra `searchpath` solves it** (verified end-to-end):
a `train.yaml` under `configs/experiment/3d/` with

```yaml
hydra:
  searchpath:
    - file://${oc.env:PWD}/configs
```

resolves the global `cluster` group (→ `paths.totalseg`, `paths.results`) and the
`augmentations` group, while `dataset`/`model` resolve as local groups — so
`dataset=omnisynth3d` swaps in short form with no path duplication. `${oc.env:PWD}` is
used (not `${hydra:runtime.cwd}`, which fails under the `compose` API that plot uses);
it works under both `@hydra.main` and `compose`, given the scripts are run from the repo
root (the established convention).

## Target layout

```
configs/experiment/3d/
  train.yaml         # config_name for train.py
  eval.yaml          # config_name for eval.py
  dataset/
    totalseg.yaml    # @package _global_ — totalseg data + training knobs + aug override (was dataset.yaml)
    omnisynth3d.yaml # @package _global_ — data.source=omnisynth3d + synth3d block (the one synth3d source)
  model/
    medverse.yaml    # @package _global_ — model + optimizer/scheduler/loss/lr/batch recipe
  experiment/        # optional override presets, layered LAST (ships empty)
```

The old flat files are removed (clean break): `dataset.yaml`, `medverse.yaml`,
`eval.yaml`, `omnisynth3d.yaml`, `omnisynth3d_eval.yaml`.

**Why no `_base.yaml`:** `hydra.searchpath` is only reliably honored in the *primary*
config, not in a composed default. So the searchpath (and the `cluster`/`augmentations`
group selections that depend on it) live directly in each primary entrypoint
(`train.yaml`, `eval.yaml`) — exactly the shape verified in the probe. The tiny
duplication (searchpath + two group lines) is the price of robustness.

## File contents & block partition

**`dataset/totalseg.yaml`** (`@package _global_`) = current `dataset.yaml` verbatim:
the full totalseg `data:` block (source, image_size `[128,128,128]`, context_size 1,
use_crop, synth_method/p_synth/n_synth_merge, class_balanced, random_coloring,
num_labels_per_sample, max_ds_len_train 2000, train/val_classes) + its
`defaults: [override /augmentations: multiverseg_v2]`.

**`dataset/omnisynth3d.yaml`** (`@package _global_`) = current `omnisynth3d.yaml`
verbatim: `data.source=omnisynth3d`, image_size `[64,64,64]`, context_size 3, and the
`synth3d:` block. **Single source of truth** for the omniSynth-3D scene params.

**`model/medverse.yaml`** (`@package _global_`) — model + recipe (per approved
partition, model-recipe = optimizer/loss/scheduler/lr/batch):
```yaml
model: medverse
train:
  batch_size: 4
  optimizer: adam
  lr: 3.0e-5
  weight_decay: 0.0
  scheduler: plateau
  lr_factor: 0.5
  lr_patience: 5
  lr_min_factor: 0.01
  warmup_epochs: 1          # scheduler-coupled -> recipe
  loss: smooth_l1
  smooth_l1_beta: 1.0
  loss_scale: 50.0
  dice_weight: 1.0
  base_ckpt: null
  random_init: true
data:
  max_ds_len_train: 200     # medverse fine-tune sample cap (overrides dataset/totalseg's 2000)
```

**`train.yaml`** — run-generic entrypoint (carries the searchpath + base groups):
```yaml
# @package _global_
defaults:
  - cluster: nfs
  - augmentations: multiverseg
  - dataset: totalseg
  - model: medverse
  - _self_
  - optional experiment:
hydra:
  searchpath:
    - file://${oc.env:PWD}/configs
train:
  epochs: 100
  workers: 8
  seed: 42
  eval_every: 1
  grad_clip: 1.0
  checkpoint: null
  out_dir: ${paths.results}/3d_train
eval:                       # per-class validation during training
  split: val
  n_subjects: 20
  batch_size: 4
  workers: 8
wandb:
  project: patch_icl_3d_exps
  name: null
```
`_self_` sets no `data.*` keys, so `model/medverse`'s `data.max_ds_len_train: 200`
override is not clobbered. `optional experiment:` is listed **after** `_self_` so a
preset overrides the entrypoint.

**`eval.yaml`** — eval entrypoint (composes dataset only; the model is loaded by
`eval.model` name in eval.py, not the `model/` group):
```yaml
# @package _global_
defaults:
  - cluster: nfs
  - augmentations: multiverseg
  - dataset: totalseg
  - _self_
  - optional experiment:
hydra:
  searchpath:
    - file://${oc.env:PWD}/configs
eval:
  model: medverse
  split: test
  n_subjects: 50
  batch_size: 8
  workers: 20
  seed: 0
  save_figures: true
  max_figures: 200
  out_dir: ${paths.results}/3d_eval
  medverse_ckpt: null
  sw_roi_size: null
  checkpoint: null
wandb:
  project: patch_icl_3d_eval
  name: null
```
Eval inherits `image_size`/`context_size`/`use_crop`/`val_classes` from
`dataset/totalseg`; the training-only synth/sampling keys it also inherits are ignored
by `make_eval_loader` (which forces aug/synth off). `dataset=omnisynth3d` switches the
eval set to omniSynth-3D (already wired in `common.make_eval_loader` / `eval.py`).

## Scripts repointed (3 hydra entrypoints)

- **`experiments/3d/train.py`**: `@hydra.main(config_path="../../configs/experiment/3d", config_name="train", version_base="1.3")`.
- **`experiments/3d/eval.py`**: same `config_path`, `config_name="eval"`.
- **`experiments/3d/plot_dataset_items.py`**: `initialize_config_dir(config_dir=ROOT/"configs"/"experiment"/"3d")` + `compose(config_name="train", overrides=hydra_overrides)`, so `dataset=…` selects what is visualized.

No change to the script bodies otherwise — they already read `cfg.data`, `cfg.train`,
`cfg.eval`, `cfg.augmentations`, `cfg.wandb`, `cfg.paths`, `cfg.synth3d`, all still
present after composition.

## Migration map (usage)

| Old | New |
|---|---|
| `train.py experiment=3d/medverse` | `train.py` (totalseg default) |
| — | `train.py dataset=omnisynth3d` |
| `eval.py experiment=3d/eval` | `eval.py` (totalseg default) |
| `eval.py experiment=3d/omnisynth3d_eval eval.split=val` | `eval.py dataset=omnisynth3d eval.split=val` |
| `plot_dataset_items.py experiment=3d/omnisynth3d` | `plot_dataset_items.py dataset=omnisynth3d` |
| `plot_dataset_items.py experiment=3d/dataset` | `plot_dataset_items.py` (totalseg default) |
| any run + ablation | `… +experiment=<preset>` |

## References to update

- Docstring usage blocks in `experiments/3d/{train,eval,plot_dataset_items}.py`.
- Header usage comments in each new config file.
- `docs/logs.md` (the `configs/experiment/3d/omnisynth3d.yaml` path pointer).
- `scripts/synth3d/build_totalseg_tiles.py` docstring (the `omnisynth3d.yaml`
  reference → `dataset/omnisynth3d.yaml`).
- Add a changelog entry.

Archival spec/plan docs (`2026-07-06-3d-medverse-eval-harness-design.md`,
`2026-07-21-omnisynth-3d.md`) are **left as-is** — they record what was built then.

## Out of scope / YAGNI

- `scripts/train.py` (the older global-config trainer) is untouched. The
  `dataset/*.yaml` files remain `@package _global_`, so they can still be selected as
  `experiment=3d/dataset/omnisynth3d` through the global entry if ever needed.
- `experiment/` ships empty (mechanism wired via `optional experiment:`); presets are
  added when a concrete ablation needs one.
- No backward-compat shim configs.
- No true top-level `dataset`/`model` groups shared with 2D (2D keeps its own root).

## Verification

- Each entrypoint composes and resolves the expected fields, checked with `hydra`
  `compose` (mirroring the probe already run):
  - `train` (default): `data.source==totalseg`, `data.image_size==[128,128,128]`,
    `model==medverse`, `train.optimizer==adam`, `data.max_ds_len_train==200`,
    `paths.totalseg` resolved.
  - `train dataset=omnisynth3d`: `data.source==omnisynth3d`, `image_size==[64,64,64]`,
    `synth3d.tiles_root` resolved.
  - `eval` (default): `eval.model==medverse`, `data.source==totalseg`, `paths.results`
    resolved.
  - `eval dataset=omnisynth3d eval.split=val`: `data.source==omnisynth3d`,
    `synth3d` present.
- A short smoke: `plot_dataset_items.py dataset=omnisynth3d synth3d.tiles_root=results/3d/omni_tiles --split val`
  still renders (reusing the existing demo tile cache).
- **Searchpath resolver:** confirm `file://${oc.env:PWD}/configs` resolves the
  `cluster`/`augmentations` groups under BOTH entry modes — `@hydra.main` (train/eval)
  and the `initialize_config_dir` + `compose` API (plot). If it fails to interpolate
  under the compose API, fall back to `file://${oc.env:PWD}/configs` (both entry points
  are run from the repo root).

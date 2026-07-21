# 3D Config Layout Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `configs/experiment/3d/` into `train.yaml`/`eval.yaml` entrypoints that compose `dataset/` + `model/` config groups (overridable by an optional `experiment` preset), and repoint the three 3D hydra entrypoints accordingly.

**Architecture:** The 3D scripts move their hydra entry to a 3D-local config root; a `hydra.searchpath` back to `configs/` keeps the global `cluster`→`paths` chain. `dataset`/`model` become swappable groups. Pure config + hydra-entry refactor — no dataset/model/eval logic changes.

**Tech Stack:** Hydra / OmegaConf, Python.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-21-3d-config-layout-design.md`.
- **Searchpath resolver:** `file://${oc.env:PWD}/configs` (verified under both `@hydra.main` and the `compose` API; `${hydra:runtime.cwd}` raises under `compose`). **All commands run from the repo root** `/home/dpxuser/dev/patch_icl`.
- **Python env:** the `python` on PATH is the correct `.venv_nero`; a torch CUDA driver-version warning at import is harmless (not a finding).
- **`@package _global_`** on `dataset/*.yaml`, `model/*.yaml`, `train.yaml`, `eval.yaml` (top line).
- **Block partition:** model-recipe (optimizer/loss/scheduler/lr/batch + base_ckpt/random_init + `data.max_ds_len_train: 200`) in `model/medverse.yaml`; run-generic (epochs/workers/seed/eval_every/grad_clip/checkpoint/out_dir + `eval:` + `wandb:`) in `train.yaml`; data in `dataset/*`.
- **Defaults order in entrypoints:** `cluster`, `augmentations`, `dataset`, [`model`,] `_self_`, `optional experiment:` — augmentations BEFORE dataset (so `dataset/totalseg`'s `override /augmentations` resolves); `_self_` sets no `data.*` (so `model`'s `max_ds_len_train: 200` is not clobbered); `optional experiment:` LAST (overrides the entrypoint).
- **No logic changes**; retro-compat is a non-goal (old `experiment=3d/*` invocations are dropped).
- Archival spec/plan docs (`2026-07-06-3d-medverse-eval-harness-design.md`, `2026-07-21-omnisynth-3d.md`) are left untouched.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `configs/experiment/3d/dataset/totalseg.yaml` | create | totalseg data block (from `dataset.yaml`) |
| `configs/experiment/3d/dataset/omnisynth3d.yaml` | create | omniSynth-3D data + `synth3d` block (from `omnisynth3d.yaml`) |
| `configs/experiment/3d/model/medverse.yaml` | create | medverse model + train recipe |
| `configs/experiment/3d/train.yaml` | create | training entrypoint (`config_name=train`) |
| `configs/experiment/3d/eval.yaml` | overwrite | eval entrypoint (`config_name=eval`) — replaces the old flat `eval.yaml` |
| `configs/experiment/3d/experiment/.gitkeep` | create | make the optional override group dir exist |
| `configs/experiment/3d/{dataset,medverse,omnisynth3d,omnisynth3d_eval}.yaml` | delete | old flat files |
| `experiments/3d/train.py` | modify | repoint `@hydra.main` + docstring |
| `experiments/3d/eval.py` | modify | repoint `@hydra.main` + docstring |
| `experiments/3d/plot_dataset_items.py` | modify | repoint `initialize_config_dir`/`compose` + docstring |
| `docs/logs.md`, `scripts/synth3d/build_totalseg_tiles.py` | modify | doc references |

---

## Task 1: dataset/ + model/ groups + `train.yaml` entrypoint

Create the training-side composition. Verification composes `train.yaml` for both datasets.

**Files:**
- Create: `configs/experiment/3d/dataset/totalseg.yaml`, `configs/experiment/3d/dataset/omnisynth3d.yaml`, `configs/experiment/3d/model/medverse.yaml`, `configs/experiment/3d/train.yaml`, `configs/experiment/3d/experiment/.gitkeep`

**Interfaces:**
- Consumes: global `cluster`/`augmentations` groups (via searchpath), existing `configs/augmentations/multiverseg_v2.yaml`.
- Produces: `config_name="train"` composing `data`/`model`/`train`/`eval`/`wandb`/`paths`/`synth3d(when dataset=omnisynth3d)`; group `dataset` (options `totalseg`, `omnisynth3d`), group `model` (option `medverse`).

- [ ] **Step 1: Write the failing verification**

Create `configs/experiment/3d/experiment/.gitkeep` (empty file) first so the optional group dir exists:

```bash
mkdir -p configs/experiment/3d/experiment && : > configs/experiment/3d/experiment/.gitkeep
```

Then the verification command (run now — it must FAIL because `train.yaml` doesn't exist yet):

```bash
python - <<'PY' 2>&1 | grep -v "UserWarning\|_C._cuda\|torch/cuda" | tail -8
from hydra import compose, initialize_config_dir
from pathlib import Path
cfgdir = str(Path("configs/experiment/3d").resolve())
with initialize_config_dir(config_dir=cfgdir, version_base="1.3"):
    d = compose(config_name="train", overrides=[])
    assert d.data.source == "totalseg", d.data.source
    assert list(d.data.image_size) == [128,128,128]
    assert d.model == "medverse", d.model
    assert d.train.optimizer == "adam" and float(d.train.lr) == 3.0e-5
    assert d.train.epochs == 100 and d.train.batch_size == 4
    assert d.data.max_ds_len_train == 200, d.data.max_ds_len_train   # model override wins
    assert d.paths.get("totalseg") and d.paths.get("results")
    o = compose(config_name="train", overrides=["dataset=omnisynth3d"])
    assert o.data.source == "omnisynth3d" and list(o.data.image_size) == [64,64,64]
    assert o.synth3d.tiles_root and o.synth3d.n_objects == 4
print("TRAIN COMPOSE OK")
PY
```

- [ ] **Step 2: Run it to verify it fails**

Run the command above.
Expected: FAIL — a Hydra error like `Cannot find primary config 'train'` (or a compose exception), no `TRAIN COMPOSE OK`.

- [ ] **Step 3: Create the group + entrypoint files**

Create `configs/experiment/3d/dataset/totalseg.yaml`:

```yaml
# @package _global_
# totalseg 3D dataset block — composed by train.yaml / eval.yaml as `dataset=totalseg`
# (default). Also selectable as experiment=3d/dataset/totalseg through the global
# config entry (scripts/train.py). Only params that reach TotalSegInContextDataset.
defaults:
  - override /augmentations: multiverseg_v2   # matches multilevel's aug_preset
data:
  source: totalseg              # totalseg | totalsegmri  (3D build_dataset dispatch)
  image_size: [128, 128, 128]
  context_size: 1
  max_ds_len_train: 2000        # samples per train epoch (loader cap, not dataset content)
  synth_method: seeds3d
  n_synth_merge_min: 1          # min adjacent SVs merged per synth label (inclusive)
  n_synth_merge_max: 5          # max adjacent SVs merged per synth label (inclusive)
  p_synth: 1
  class_balanced: true
  use_crop: true
  random_coloring: false
  num_labels_per_sample: 1
  train_classes: not_benchmark
  val_classes: benchmark
```

Create `configs/experiment/3d/dataset/omnisynth3d.yaml`:

```yaml
# @package _global_
# omniSynth 3D dataset — TotalSegmentator organs on a synthetic canvas. Composed as
# `dataset=omnisynth3d`; the single source of truth for the synth3d scene params.
# Prerequisite: build the tile cache once, e.g.
#   python scripts/synth3d/build_totalseg_tiles.py --split train --size 64 64 64
data:
  source: omnisynth3d
  image_size: [64, 64, 64]
  context_size: 3

synth3d:
  tiles_root: ${paths.totalseg}/omni_tiles
  size: [64, 64, 64]
  classes: []            # [] = all classes present in the cache
  n_objects: 4
  k_min: 1
  k_max: 2
  placement_tries: 4
  placement_max_overlap: 0.1
  target_mode: class     # identical | class
  background: black       # black | noise
  eval_subjects_per_task: 4
  epoch_length: 10000
```

Create `configs/experiment/3d/model/medverse.yaml`:

```yaml
# @package _global_
# Medverse model + fine-tune recipe (optimizer/scheduler/loss/lr/batch), composed as
# `model=medverse`. Loss/optimizer/scheduler follow the Medverse/Neuroverse3D (Hu 2025)
# recipe. 128³ inputs only (Medverse runs level=1, no AR).
model: medverse
train:
  batch_size: 4
  optimizer: adam               # adam | adamw
  lr: 3.0e-5
  weight_decay: 0.0
  scheduler: plateau            # plateau | cosine | constant
  lr_factor: 0.5
  lr_patience: 5
  lr_min_factor: 0.01           # min_lr = lr * lr_min_factor
  warmup_epochs: 1              # cosine only (scheduler-coupled)
  loss: smooth_l1               # smooth_l1 | bce_dice
  smooth_l1_beta: 1.0
  loss_scale: 50.0
  dice_weight: 1.0              # bce_dice only
  base_ckpt: null               # override the pretrained Medverse.ckpt
  random_init: true             # true -> train from scratch (ignore pretrained weights)
data:
  max_ds_len_train: 200         # fine-tune sample cap (overrides dataset/totalseg's 2000)
```

Create `configs/experiment/3d/train.yaml`:

```yaml
# @package _global_
# 3D in-context training entrypoint (experiments/3d/train.py, config_name=train).
#   python experiments/3d/train.py                       # medverse on totalseg
#   python experiments/3d/train.py dataset=omnisynth3d   # train on omniSynth-3D
#   python experiments/3d/train.py train.loss=bce_dice   # ad-hoc overrides
#   python experiments/3d/train.py +experiment=<preset>  # ablation layer
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
  checkpoint: null              # resume/warm-start fine-tuned weights
  out_dir: ${paths.results}/3d_train
eval:                           # per-class validation during training
  split: val
  n_subjects: 20
  batch_size: 4
  workers: 8
wandb:
  project: patch_icl_3d_exps
  name: null
```

- [ ] **Step 4: Run the verification to confirm it passes**

Run the Step-1 command again.
Expected: PASS — prints `TRAIN COMPOSE OK`, no assertion error.

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/3d/dataset configs/experiment/3d/model configs/experiment/3d/train.yaml configs/experiment/3d/experiment/.gitkeep
git commit -m "feat(3d-config): dataset/model groups + train.yaml entrypoint"
```

---

## Task 2: `eval.yaml` entrypoint

Create the eval entrypoint composing `dataset/totalseg`. Overwrites the old flat `eval.yaml` (same path).

**Files:**
- Modify (overwrite): `configs/experiment/3d/eval.yaml`

**Interfaces:**
- Consumes: `dataset` group from Task 1 (`totalseg` default, `omnisynth3d` swap).
- Produces: `config_name="eval"` composing `data`/`eval`/`wandb`/`paths` (+ `synth3d` when `dataset=omnisynth3d`).

- [ ] **Step 1: Write the failing verification**

Run now — must FAIL because `eval.yaml` is still the OLD flat file (has no `eval` block layered on `dataset` group / has `data.source` set inline, and lacks the group composition). Concretely it will fail the `dataset=omnisynth3d` swap assertion:

```bash
python - <<'PY' 2>&1 | grep -v "UserWarning\|_C._cuda\|torch/cuda" | tail -8
from hydra import compose, initialize_config_dir
from pathlib import Path
cfgdir = str(Path("configs/experiment/3d").resolve())
with initialize_config_dir(config_dir=cfgdir, version_base="1.3"):
    d = compose(config_name="eval", overrides=[])
    assert d.eval.model == "medverse", d.eval.model
    assert d.data.source == "totalseg", d.data.source
    assert d.paths.get("results")
    o = compose(config_name="eval", overrides=["dataset=omnisynth3d","eval.split=val"])
    assert o.data.source == "omnisynth3d", o.data.source
    assert o.synth3d.tiles_root and o.eval.split == "val"
print("EVAL COMPOSE OK")
PY
```

- [ ] **Step 2: Run it to verify it fails**

Run the command above.
Expected: FAIL — an assertion/compose error (the old `eval.yaml` doesn't compose the `dataset` group), no `EVAL COMPOSE OK`.

- [ ] **Step 3: Overwrite `configs/experiment/3d/eval.yaml`**

```yaml
# @package _global_
# 3D in-context eval entrypoint (experiments/3d/eval.py, config_name=eval). Composes a
# dataset group; the model is chosen by eval.model (loaded by name in eval.py).
#   python experiments/3d/eval.py                                   # medverse on totalseg (test)
#   python experiments/3d/eval.py eval.model=native_resenc \
#       eval.checkpoint=results/checkpoints/resenc_in_context_best.pt
#   python experiments/3d/eval.py dataset=omnisynth3d eval.split=val   # eval on omniSynth-3D
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
  model: medverse             # medverse | native_resenc | native_vit | multilevel | nninteractive
  split: test
  n_subjects: 50
  batch_size: 8
  workers: 20
  seed: 0
  save_figures: true
  max_figures: 200
  out_dir: ${paths.results}/3d_eval
  medverse_ckpt: null         # override Medverse.ckpt path; null = adapter default
  sw_roi_size: null           # sliding-window ROI; null = adapter default (128³)
  checkpoint: null            # trained best.pt (required for native_*/multilevel/nninteractive)
wandb:
  project: patch_icl_3d_eval  # null to disable
  name: null
```

- [ ] **Step 4: Run the verification to confirm it passes**

Run the Step-1 command again.
Expected: PASS — prints `EVAL COMPOSE OK`.

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/3d/eval.yaml
git commit -m "feat(3d-config): eval.yaml entrypoint composing dataset group"
```

---

## Task 3: Repoint scripts, delete old flat files, smoke

Repoint the three hydra entrypoints, update their usage docstrings, delete the now-unused flat configs, and smoke-test the real scripts.

**Files:**
- Modify: `experiments/3d/train.py` (`@hydra.main` line 194 + docstring lines 12-14)
- Modify: `experiments/3d/eval.py` (`@hydra.main` line 60 + docstring lines 8-12)
- Modify: `experiments/3d/plot_dataset_items.py` (`initialize_config_dir`/`compose` lines 100-101 + docstring lines 15-20)
- Delete: `configs/experiment/3d/dataset.yaml`, `medverse.yaml`, `omnisynth3d.yaml`, `omnisynth3d_eval.yaml`

**Interfaces:**
- Consumes: `train.yaml` (Task 1), `eval.yaml` (Task 2).
- Produces: runnable `python experiments/3d/{train,eval,plot_dataset_items}.py` on the new config root.

- [ ] **Step 1: Repoint `experiments/3d/train.py`**

Change the decorator (line ~194) from:

```python
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
```

to:

```python
@hydra.main(config_path="../../configs/experiment/3d", config_name="train", version_base="1.3")
```

And update the docstring usage block (lines ~13-14) from:

```python
    python experiments/3d/train.py experiment=3d/medverse
    python experiments/3d/train.py experiment=3d/medverse train.loss=bce_dice train.optimizer=adamw
```

to:

```python
    python experiments/3d/train.py                       # medverse on totalseg (default)
    python experiments/3d/train.py dataset=omnisynth3d   # train on omniSynth-3D
    python experiments/3d/train.py train.loss=bce_dice train.optimizer=adamw
```

- [ ] **Step 2: Repoint `experiments/3d/eval.py`**

Change the decorator (line ~60) from:

```python
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
```

to:

```python
@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
```

And update the docstring usage block (lines ~8-12) from:

```python
    python experiments/3d/eval.py experiment=3d/eval
    python experiments/3d/eval.py experiment=3d/eval eval.model=native_resenc \
        eval.checkpoint=results/checkpoints/resenc_in_context_best.pt
    python experiments/3d/eval.py experiment=3d/eval data.source=totalsegmri \
        eval.n_subjects=20 data.val_classes='[liver,spleen]'
```

to:

```python
    python experiments/3d/eval.py
    python experiments/3d/eval.py eval.model=native_resenc \
        eval.checkpoint=results/checkpoints/resenc_in_context_best.pt
    python experiments/3d/eval.py dataset=omnisynth3d eval.split=val   # eval on omniSynth-3D
```

- [ ] **Step 3: Repoint `experiments/3d/plot_dataset_items.py`**

Change the compose block (lines ~100-101) from:

```python
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=hydra_overrides)
```

to:

```python
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=hydra_overrides)
```

And update the docstring usage block (lines ~15-20) — replace the `experiment=3d/...` / `data.source=...` examples with:

```python
  python experiments/3d/plot_dataset_items.py                        # totalseg train split
  python experiments/3d/plot_dataset_items.py dataset=omnisynth3d --split val
  python experiments/3d/plot_dataset_items.py --n_samples 6 --out results/x.png
  # Hydra overrides forwarded after the argparse flags:
  python experiments/3d/plot_dataset_items.py dataset=omnisynth3d synth3d.tiles_root=results/3d/omni_tiles
```

- [ ] **Step 4: Delete the old flat config files**

```bash
git rm configs/experiment/3d/dataset.yaml configs/experiment/3d/medverse.yaml \
       configs/experiment/3d/omnisynth3d.yaml configs/experiment/3d/omnisynth3d_eval.yaml
```

- [ ] **Step 5: Smoke — real scripts compose + plot renders**

`--cfg job` makes Hydra print the composed config and exit WITHOUT running the body (validates `@hydra.main` config_path/name + searchpath under the real entry, from repo root):

```bash
python experiments/3d/train.py --cfg job 2>&1 | grep -E "^  (source|model):|source:|^model:" | head
python experiments/3d/eval.py  --cfg job 2>&1 | grep -E "model:" | head
```
Expected: `train --cfg job` prints a config containing `model: medverse` and `source: totalseg`; `eval --cfg job` prints `model: medverse` (under `eval:`), both exit 0.

Then the plot smoke against the existing demo cache:

```bash
python experiments/3d/plot_dataset_items.py dataset=omnisynth3d \
  synth3d.tiles_root=results/3d/omni_tiles --split val --n_samples 3 \
  --out results/3d/omnisynth3d_layout_check.png 2>&1 | tail -2
ls -la results/3d/omnisynth3d_layout_check.png
```
Expected: prints `Saved → results/3d/omnisynth3d_layout_check.png`; the PNG exists.

- [ ] **Step 6: Commit**

```bash
git add experiments/3d/train.py experiments/3d/eval.py experiments/3d/plot_dataset_items.py
git commit -m "refactor(3d): repoint hydra entrypoints to 3d config root; drop flat configs"
```

---

## Task 4: Update doc references

External references to the moved/renamed configs.

**Files:**
- Modify: `docs/logs.md`, `scripts/synth3d/build_totalseg_tiles.py`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `docs/logs.md`**

In the 2026-07-21 omniSynth-3D entry, change the parenthetical:

```
  `data.source=omnisynth3d` (config `configs/experiment/3d/omnisynth3d.yaml`).
```

to:

```
  `data.source=omnisynth3d` (config group `configs/experiment/3d/dataset/omnisynth3d.yaml`,
  selected via `dataset=omnisynth3d`).
```

- [ ] **Step 2: Update the build-script docstring reference**

In `scripts/synth3d/build_totalseg_tiles.py`, the docstring line referencing the config path:

```
paths.totalseg/omni_tiles (matching configs/experiment/3d/omnisynth3d.yaml's
```

becomes:

```
paths.totalseg/omni_tiles (matching configs/experiment/3d/dataset/omnisynth3d.yaml's
```

- [ ] **Step 3: Add a changelog entry**

Prepend under `# Change log` in `docs/logs.md`:

```markdown
## 2026-07-21 — 3D config layout: train/eval entrypoints + dataset/model groups
- refactor(3d-config): `configs/experiment/3d/` reorganized into `train.yaml`/`eval.yaml`
  entrypoints that compose `dataset/{totalseg,omnisynth3d}` + `model/medverse` groups,
  overridable by an `optional experiment:` preset. `hydra.searchpath`
  (`file://${oc.env:PWD}/configs`, run from repo root) keeps the global cluster→paths
  chain. The 3 hydra entrypoints (train.py/eval.py/plot_dataset_items.py) now point at
  the 3D config root. Old flat files removed; synth3d block de-duplicated (one source in
  dataset/omnisynth3d.yaml). New usage: `train.py dataset=omnisynth3d`,
  `eval.py dataset=omnisynth3d eval.split=val`. Spec: 2026-07-21-3d-config-layout-design.md.
```

- [ ] **Step 4: Verify no stale references remain**

```bash
grep -rn "experiment=3d/medverse\|experiment=3d/eval\b\|experiment=3d/omnisynth3d\|3d/omnisynth3d_eval\|configs/experiment/3d/omnisynth3d.yaml\|configs/experiment/3d/dataset.yaml" \
  --include=*.py --include=*.md . 2>/dev/null | grep -v "docs/superpowers/specs/2026-07-06\|docs/superpowers/specs/2026-07-21-3d-config-layout\|docs/superpowers/plans/2026-07-21-omnisynth-3d.md\|docs/superpowers/plans/2026-07-21-3d-config-layout.md\|\.ipynb_checkpoints"
```
Expected: no output (all live references updated; archival spec/plan docs excluded).

- [ ] **Step 5: Commit**

```bash
git add docs/logs.md scripts/synth3d/build_totalseg_tiles.py
git commit -m "docs(3d-config): update config references + changelog"
```

---

## Self-Review

**1. Spec coverage:**
- Layout (train/eval + dataset/model groups + experiment/) → Tasks 1–2 + `.gitkeep`. ✓
- searchpath `${oc.env:PWD}` in each primary entrypoint → Task 1/2 files. ✓
- Block partition (model-recipe vs run-generic) → Task 1 (`model/medverse.yaml` vs `train.yaml`). ✓
- synth3d dedup (one source) → `dataset/omnisynth3d.yaml`; `eval.yaml` no longer carries it. ✓
- `optional experiment:` last → Task 1/2 defaults lists. ✓
- Repoint 3 scripts → Task 3. ✓
- Delete old flat files (clean break) → Task 3 Step 4 (+ eval.yaml overwritten in Task 2). ✓
- Doc references (logs.md, build script, changelog) + docstrings → Task 3 (docstrings) + Task 4. ✓
- Verification (compose per entrypoint + plot smoke + searchpath under both modes) → Task 1/2 compose checks (compose API) + Task 3 `--cfg job` (@hydra.main) + plot smoke. ✓
- Archival docs untouched → not modified; Task 4 grep excludes them. ✓

**2. Placeholder scan:** No TBD/TODO; every file's full content is given; every command has expected output. ✓

**3. Type/name consistency:**
- Group names `dataset` (options `totalseg`/`omnisynth3d`) and `model` (`medverse`) consistent across train.yaml/eval.yaml defaults and CLI swaps. ✓
- `data.max_ds_len_train: 200` set only in `model/medverse.yaml`; `train.yaml` sets no `data.*` (partition preserved so the override wins). ✓
- `config_name` values: `train` (train.py + plot compose), `eval` (eval.py) — consistent with the created filenames. ✓
- searchpath string identical in both entrypoints: `file://${oc.env:PWD}/configs`. ✓

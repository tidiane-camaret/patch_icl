# omniSynth batch augmentation (unified trainer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply `configs/augmentations/` presets to omniSynth (and any source) training run through the unified `experiments/2d/train.py`, reusing `pfn_train.augment()`, opt-in so existing runs are unchanged.

**Architecture:** Add a repo-root constant + a preset load in `main()` (gated by a new `augment: false` flag) and a pure `_augment_batch` helper called per-batch in `train_epoch`. The helper assembles the `(B,T,1,H,W)` batch (query at index `K`), calls the existing `augment()`, and splits back — the query GT is never passed in, so it stays valid. Config edits then turn it on for the omniSynth refine experiment.

**Tech Stack:** PyTorch, Hydra/OmegaConf, pytest. Python via `.venv_nero/bin/python`.

## Global Constraints

- Python interpreter is `.venv_nero/bin/python`; run tests with `.venv_nero/bin/python -m pytest` (pytest is installed there). Do NOT use `.venv` — it is corrupted.
- `augment: false` is the default; existing `train.py` runs must be byte-identical until a config opts in. The gate is `cfg.get("augment", False)`.
- Reuse `experiments/2d/pfn_train.py`'s `augment(images, masks, K, cfg)` unchanged — do NOT modify it or the preset schema.
- The query GT (`lbl`) must never be passed to `augment()`; `augment()` transforms contexts geometrically and applies at most intensity/task ops to the query image.
- Follow existing style in `train.py` and the config files.
- Branch `patchset-refine` uses commit-per-task (user-authorized). Stage only changed files with path-scoped `git add`; end every commit message with the trailer line exactly: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `_augment_batch` helper + gated wiring in `train.py`

**Files:**
- Modify: `experiments/2d/train.py` (add `_ROOT` at line ~52; add `_augment_batch` helper; load preset in `main()` ~line 287; gated call in `train_epoch` after line 171)
- Test: `tests/test_train_augment.py` (create)

**Interfaces:**
- Consumes: `augment(images, masks, K, cfg)` from `pfn_train` — `images`/`masks` are `(B,T,1,H,W)`, query at index `K`; returns transformed `(images, masks)`.
- Produces: `_augment_batch(img, cin, cout, aug_cfg) -> (img, cin, cout)` where `img` is `(B,1,H,W)`, `cin`/`cout` are `(B,K,1,H,W)`. `K` is derived as `cin.shape[1]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train_augment.py`:

```python
import sys; sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2d")
import torch
from omegaconf import OmegaConf
from train import _augment_batch
from src.models.patchset_cnn import PatchSetCNN


def _geom_only_cfg():
    # rotate always on; every intensity/task op off -> query stays byte-identical.
    return OmegaConf.create({
        "enabled": True,
        "geometric": {"hflip_p": 0.0, "vflip_p": 0.0,
                      "rotate": {"p": 1.0, "max_angle_deg": 45.0}},
        "intensity": {},
    })


def test_augment_batch_leaves_query_untouched_and_changes_context():
    torch.manual_seed(0)
    B, K, H = 2, 2, 16
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    out_img, out_cin, out_cout = _augment_batch(img.clone(), cin.clone(), cout.clone(),
                                                _geom_only_cfg())
    # query (index K) is never geometrically transformed and no intensity op is active
    assert torch.equal(out_img, img)
    # contexts were rotated
    assert not torch.equal(out_cin, cin)


def test_augment_batch_shapes():
    B, K, H = 2, 3, 16
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    out_img, out_cin, out_cout = _augment_batch(img, cin, cout, _geom_only_cfg())
    assert out_img.shape == (B, 1, H, H)
    assert out_cin.shape == (B, K, 1, H, H)
    assert out_cout.shape == (B, K, 1, H, H)


def test_augmented_batch_trains_one_step():
    torch.manual_seed(0)
    B, K, H = 2, 2, 32
    model = PatchSetCNN(image_size=H, resolution=8, enc_dims=[16], e=32, h=64, l=1, a=2,
                        thinking_rows=1, resolutions=[8, 16])
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    aug = OmegaConf.create({
        "enabled": True,
        "geometric": {"hflip_p": 0.5, "vflip_p": 0.5, "rotate": {"p": 0.5, "max_angle_deg": 20.0}},
        "intensity": {"brightness": {"p": 0.5, "max_delta": 0.15}},
    })
    img, cin, cout = _augment_batch(img, cin, cout, aug)
    out = model(img, context_in=cin, context_out=cout, mode="train")
    loss = out["final_logit"].mean() + out["refine_logit"].mean()
    loss.backward()
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv_nero/bin/python -m pytest tests/test_train_augment.py -v`
Expected: FAIL — `ImportError: cannot import name '_augment_batch' from 'train'`.

- [ ] **Step 3: Add `augment` to the pfn_train import and add `_ROOT`**

In `experiments/2d/train.py`, change the import (line ~56) from:

```python
from pfn_train import Muon, lawa_average, soft_dice_loss
```
to:
```python
from pfn_train import Muon, augment, lawa_average, soft_dice_loss
```

Immediately after `sys.path.insert(0, str(Path(__file__).resolve().parent))` (line ~52), add:

```python
_ROOT = str(Path(__file__).resolve().parents[2])   # experiments/2d/train.py -> repo root
```

- [ ] **Step 4: Add the `_augment_batch` helper**

Add this function just above `def train_epoch` (line ~151) in `experiments/2d/train.py`:

```python
def _augment_batch(img, cin, cout, aug_cfg):
    """Augment context pairs + query intensity via pfn_train.augment.

    img (B,1,H,W); cin/cout (B,K,1,H,W). Returns (img, cin, cout). The query GT (lbl)
    is never passed in, so it stays valid: augment() geometrically transforms contexts
    only; the query (index K) receives at most intensity/task ops."""
    K = cin.shape[1]
    imgs = torch.cat([cin, img.unsqueeze(1)], dim=1)              # (B,T,1,H,W), query at index K
    msks = torch.cat([cout, torch.zeros_like(img.unsqueeze(1))], dim=1)
    imgs, msks = augment(imgs, msks, K, aug_cfg)
    return imgs[:, K], imgs[:, :K], msks[:, :K]                   # img, cin, cout
```

- [ ] **Step 5: Load the preset in `main()` (opt-in)**

In `main()`, after the seeding/precision block (after `torch.set_float32_matmul_precision("high")`, line ~287) and before `train_loader = make_loader(...)` (line ~289), add:

```python
    # Augmentation is opt-in: only when a config sets `augment: true` do we load the
    # configs/augmentations/<aug_preset>.yaml preset into cfg.aug (mirrors pfn_seg.py).
    # Default off keeps every existing train.py run byte-identical.
    if cfg.get("augment", False):
        _aug = OmegaConf.load(Path(_ROOT) / "configs" / "augmentations" / f"{cfg.aug_preset}.yaml")
        cfg.aug = OmegaConf.merge(_aug, cfg.aug) if cfg.get("aug", None) else _aug
        print(f"Augmentation ON (preset={cfg.aug_preset}, enabled={cfg.aug.get('enabled', True)})")
```

- [ ] **Step 6: Call the helper in `train_epoch` (train only)**

In `train_epoch`, immediately after the four `.to(DEVICE...)` lines (after `cout = batch["context_out"]...`, line ~171), add:

```python
        if cfg.get("augment", False) and cfg.aug.get("enabled", True):
            img, cin, cout = _augment_batch(img, cin, cout, cfg.aug)
```

- [ ] **Step 7: Run the tests**

Run: `.venv_nero/bin/python -m pytest tests/test_train_augment.py -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git add experiments/2d/train.py tests/test_train_augment.py
git commit -m "train: opt-in batch augmentation via pfn_train.augment (_augment_batch)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: config opt-in + logs

**Files:**
- Modify: `configs/experiment/2d/train_base.yaml`
- Modify: `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`
- Modify: `docs/logs.md`
- Test: `tests/test_train_augment.py` (add config-level tests)

**Interfaces:**
- Consumes: the `cfg.get("augment", False)` gate from Task 1.
- Produces: `augment: false` default in `train_base.yaml`; `augment: true` in `2_omnisynth_medseg_refine.yaml`.

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_train_augment.py`:

```python
from pathlib import Path


def test_train_base_augment_defaults_false():
    c = OmegaConf.load("configs/experiment/2d/train_base.yaml")
    assert c.get("augment", False) is False


def test_omnisynth_refine_opts_in():
    c = OmegaConf.load("configs/experiment/2d/2_omnisynth_medseg_refine.yaml")
    assert c.get("augment", False) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv_nero/bin/python -m pytest tests/test_train_augment.py -k "augment_defaults or opts_in" -v`
Expected: FAIL — `test_omnisynth_refine_opts_in` fails (`augment` key absent → `False`).

- [ ] **Step 3: Add the opt-in flag to `train_base.yaml`**

In `configs/experiment/2d/train_base.yaml`, find the `aug_preset: 2d` line near the bottom and its comment block:

```yaml
# Augmentation params live in the single shared file configs/augmentations/<preset>.yaml
# (2D schema: enabled/geometric/intensity). Loaded into cfg.aug in main().
# Override a field at the CLI with the +-prefix, e.g. +aug.enabled=false.
aug_preset: 2d
```

Replace that block with:

```yaml
# Augmentation is opt-in in the unified train.py: set `augment: true` to load the
# configs/augmentations/<aug_preset>.yaml preset (2D schema: enabled/geometric/intensity)
# into cfg.aug in main() and apply it per-batch in training. Default false keeps runs
# un-augmented. Override a preset field at the CLI with the +-prefix, e.g. +aug.geometric.rotate.p=0.
augment: false
aug_preset: 2d
```

- [ ] **Step 4: Opt in on the omniSynth refine config**

`augment` is a TOP-LEVEL key (sibling to `arch:`/`train:`), matching where `aug_preset` lives in `train_base.yaml`. In `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`, the `train:` block currently reads:

```yaml
train:
  refine_loss_weight: 1.0    # weight of the refine-level loss relative to the coarse loss
```

Add a top-level `augment: true` after it (leave the `train:` block unchanged), so that region becomes:

```yaml
train:
  refine_loss_weight: 1.0    # weight of the refine-level loss relative to the coarse loss

augment: true                # apply configs/augmentations/<aug_preset> (2d) to this omniSynth run
```

- [ ] **Step 5: Run the config tests**

Run: `.venv_nero/bin/python -m pytest tests/test_train_augment.py -k "augment_defaults or opts_in" -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Add a change-log entry**

Prepend to `docs/logs.md` (under `# Change log`) a dated entry: the unified `train.py` now supports opt-in batch augmentation via the existing `pfn_train.augment()` (`_augment_batch` helper, gated by top-level `augment:` flag, preset from `aug_preset`); default off preserves all existing runs; `2_omnisynth_medseg_refine.yaml` opts in with the `2d` preset. Note the query GT is never augmented (augment transforms contexts geometrically; query gets intensity/task only).

- [ ] **Step 7: Full test run**

Run: `.venv_nero/bin/python -m pytest tests/test_train_augment.py -v`
Expected: PASS (5 passed).

- [ ] **Step 8: Commit**

```bash
git add configs/experiment/2d/train_base.yaml configs/experiment/2d/2_omnisynth_medseg_refine.yaml docs/logs.md tests/test_train_augment.py
git commit -m "config: opt-in augment flag; enable on omnisynth refine + logs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Import `augment`, add `_ROOT`, preset load in `main()` → Task 1 Steps 3, 5. ✓
- `_augment_batch` helper (exact signature/return) → Task 1 Step 4. ✓
- Gated per-batch call in `train_epoch`, train only → Task 1 Step 6. ✓
- `augment: false` default in `train_base.yaml` + fixed comment → Task 2 Step 3. ✓
- `augment: true` on `2_omnisynth_medseg_refine.yaml` → Task 2 Step 4. ✓
- `docs/logs.md` entry → Task 2 Step 6. ✓
- Tests: split alignment (query untouched / context changed), shapes, smoke → Task 1 Step 1; opt-in default (config) → Task 2 Step 1. ✓
- Query-GT correctness: `lbl` never passed to `_augment_batch` (not an argument) — structural. ✓

**Placeholder scan:** No TBD/TODO/vague steps remain; every code/config step shows concrete content. Task 2 Step 4 explicitly flags `augment` as a top-level key to prevent mis-nesting under `train:`.

**Type consistency:** `_augment_batch(img, cin, cout, aug_cfg) -> (img, cin, cout)` is used identically in Task 1 helper, Task 1 tests, and (via the gate) Task 1 Step 6. `augment(images, masks, K, cfg)` matches `pfn_train.py`. The `augment` top-level flag name is consistent across `main()`, `train_epoch`, both configs, and all tests.

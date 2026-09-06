# Multi-source (CT + MRI) in-context dataloader — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and eval one in-context segmentation model jointly over `totalseg` (CT) and `totalsegmri` (MRI), drawing a per-task modality regime — 1/3 all-CT, 1/3 all-MRI, 1/3 forced cross-modality.

**Architecture:** A new `MultiSourceProvider` composes two modality-locked `TotalSegProvider` instances and implements the v2 engine's cohort `assemble_task` hook (the `src/providers/synth_gmm.py` pattern), so `InContextDataset` keeps only aug + the per-item RNG. Per task it picks a class uniformly, draws a regime, resolves a modality per K+1 slot (with fallback to the other modality when a class has no subjects in the wanted one), loads each slot through the matching sub-provider, and stacks. A new Hydra dataset group `multisource_ct_mri` and experiment `81_multisource_ct_mri` wire it onto the `experiment=80_varspacing_hard_tgt_prior` lineage; `common.py`/`train.py` gain a `multisource` branch for source-root resolution, class-list union, dataset build, and the eval-loader route.

**Tech Stack:** Python, PyTorch 2.5 (CUDA), Hydra/OmegaConf configs, pytest (pure-logic unit tests), manual `_check_*.py` scripts for NFS/GPU integration checks.

**Spec:** `docs/superpowers/specs/2026-09-06-multi-source-ct-mri-dataloader-design.md` (read it alongside this plan).

## Global Constraints

- **Python env is node-specific** (see memory `feedback_python_env`). On the current dev node `python` already resolves to a working `torch 2.5.1+cu121` with CUDA available (RTX A6000) — use `python` directly. If an `import torch` fails on another node, activate that node's venv first (`.venv_nero` on Ampere, `.venv_blackwell` on odin) — do not touch `.venv` (corrupted).
- **Commits are owned by this controller session** on the `patch_icl` NFS mount (memory `project_nfs_git_commit_hang`). If `git commit` hangs, check for a stale `.git/index.lock`; do not spawn subagents to retry commits.
- **Repo code style:** short docstrings; follow existing patterns; write tests only when necessary. Pure-logic tests are co-located `test_*.py` pytest files (cf. `src/datasets/omniSynth/test_build_totalseg_tiles.py`); data/GPU integration checks are `experiments/3d/_check_*.py` / `_plot_*.py` scripts run by hand.
- **Log every change to `docs/logs.md`** (dated bullet, newest at the section it belongs in — match the file's existing format).
- **v2 cohort-provider contract:** `assemble_task(rng, crop_spacing_mm)` returns the standard item dict (`image`, `label`, `context_in` `(K,1,T,T,T)`, `context_out` `(K,T,T,T)`, `spacing`, `crop_geom`, `subject`, `context_subjects`, `label_name`, `aug_mode`, plus optional `modality` / `meta`). The engine applies shared task-aug + per-volume intensity aug when the item carries an `"image"` key; it does not call `subjects_for` / `load` in cohort mode.
- **Regime label values** stored in `meta["regime"]` are the modality key verbatim (`"ct"` / `"mri"`) for the pure regimes, or the literal `"cross"`.
- **Fixed paths (NFS `cluster/nfs.yaml`):** CT root `paths.totalseg = /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg`; MRI root `paths.totalsegmri = .../data/totalsegmri`. MRI `meta.csv` has only `train` / `test` splits (no `val`).

---

### Task 1: `MultiSourceProvider` (pure logic)

**Files:**
- Create: `src/providers/multisource.py`
- Test: `src/providers/test_multisource.py`

**Interfaces:**
- Consumes: `src.incontext_dataset_v2.LoadRequest` (dataclass: `rng: random.Random`, `crop_spacing_mm: float`, `center=None`, `jitter=None`); `LoadResult` (dataclass: `image`, `label`, `spacing`, `crop_geom`, `modality="ct"`).
- Produces:
  - `class MultiSourceProvider(sub_providers: dict[str, SubProvider], *, context_size: int, regime_p: tuple[float,float,float] = (1/3,1/3,1/3), epoch_length: int = 1000)`
  - attrs: `.classes: list[str]`, `.epoch_length: int`, `._avail: dict[str, list[str]]`, `._mods: list[str]`
  - `.subjects_for(cls) -> list` (always `[]`), `.load(*a, **k)` (raises `RuntimeError`)
  - `.assemble_task(rng: random.Random, crop_spacing_mm: float) -> dict` with keys `image, label, context_in, context_out, spacing, crop_geom, subject, context_subjects, label_name, modality, aug_mode, meta`; `meta = {"regime": str, "tgt_mod": str, "ctx_mod": str}`.
  - A "sub-provider" is any object with `.classes: list[str]`, `.subjects_for(cls) -> list[str]`, `.load(subject, cls, LoadRequest) -> LoadResult`-like (`.image`, `.label`, `.spacing`, `.crop_geom`).

- [ ] **Step 1: Write the failing test**

Create `src/providers/test_multisource.py`:

```python
"""Unit tests for MultiSourceProvider (pure logic, fake sub-providers)."""
import random

import pytest
import torch

from src.incontext_dataset_v2 import LoadResult
from src.providers.multisource import MultiSourceProvider


class _FakeSub:
    """Minimal sub-provider: fixed class->subjects map, zero-tensor loads."""

    def __init__(self, modality, class_to_subjects):
        self.modality = modality
        self.classes = list(class_to_subjects)
        self._c2s = {c: list(s) for c, s in class_to_subjects.items()}
        self.loaded = []  # (subject, cls) log for assertions

    def subjects_for(self, cls):
        return self._c2s.get(cls, [])

    def load(self, subject, cls, req):
        self.loaded.append((subject, cls))
        return LoadResult(
            image=torch.zeros(1, 4, 4, 4),
            label=torch.zeros(4, 4, 4, dtype=torch.long),
            spacing=torch.full((3,), float(req.crop_spacing_mm)),
            crop_geom=torch.zeros(4, 3, dtype=torch.long),
            modality=self.modality,
        )


def _mk(regime_p=(1 / 3, 1 / 3, 1 / 3), context_size=1):
    ct = _FakeSub("ct", {"a": ["ca0", "ca1", "ca2"],
                          "b": ["cb0", "cb1", "cb2", "cb3"],
                          "c": ["cc0", "cc1"]})
    mri = _FakeSub("mri", {"b": ["mb0", "mb1", "mb2"],
                            "c": ["mc0", "mc1", "mc2"],
                            "d": ["md0", "md1"]})
    prov = MultiSourceProvider({"ct": ct, "mri": mri},
                               context_size=context_size, regime_p=regime_p,
                               epoch_length=99)
    return prov, ct, mri


def test_class_union_and_availability():
    prov, _, _ = _mk()
    assert prov.classes == ["a", "b", "c", "d"]
    assert prov._avail["a"] == ["ct"]
    assert prov._avail["b"] == ["ct", "mri"]
    assert prov._avail["c"] == ["ct", "mri"]
    assert prov._avail["d"] == ["mri"]
    assert prov.epoch_length == 99


def test_item_dict_shape_k2():
    prov, _, _ = _mk(context_size=2)
    rng = random.Random(0)
    it = prov.assemble_task(rng, 3.0)
    assert set(it) >= {"image", "label", "context_in", "context_out", "spacing",
                       "crop_geom", "subject", "context_subjects", "label_name",
                       "modality", "aug_mode", "meta"}
    assert it["context_in"].shape == (2, 1, 4, 4, 4)
    assert it["context_out"].shape == (2, 4, 4, 4)
    assert len(it["context_subjects"]) == 2
    assert it["label_name"] in prov.classes
    assert it["meta"]["regime"] in ("ct", "mri", "cross")
    assert int(it["aug_mode"]) == 0


def test_regime_frequencies():
    prov, _, _ = _mk(regime_p=(0.5, 0.3, 0.2))
    rng = random.Random(0)
    tally = {"ct": 0, "mri": 0, "cross": 0}
    n = 6000
    for _ in range(n):
        tally[prov.assemble_task(rng, 3.0)["meta"]["regime"]] += 1
    assert abs(tally["ct"] / n - 0.5) < 0.03
    assert abs(tally["mri"] / n - 0.3) < 0.03
    assert abs(tally["cross"] / n - 0.2) < 0.03


def test_cross_is_cross_modality_when_both_available():
    prov, _, _ = _mk()
    rng = random.Random(1)
    seen = 0
    for _ in range(2000):
        it = prov.assemble_task(rng, 3.0)
        if it["meta"]["regime"] == "cross" and it["label_name"] in ("b", "c"):
            seen += 1
            assert it["meta"]["tgt_mod"] != it["meta"]["ctx_mod"]
            assert it["modality"] in ("ct", "mri")
    assert seen > 50  # sanity: the branch was actually exercised


def test_ct_only_class_never_produces_mri():
    prov, _, mri = _mk()
    rng = random.Random(2)
    for _ in range(500):
        it = prov.assemble_task(rng, 3.0)
        if it["label_name"] == "a":
            assert it["modality"] == "ct"
            assert it["meta"]["ctx_mod"] == "ct"
            assert all(s.startswith("ca") for s in it["context_subjects"])
            assert it["subject"].startswith("ca")


def test_pure_regime_falls_back_for_missing_modality():
    # regime forced to 'mri' (index 1) always; class 'a' has no MRI -> falls back to ct.
    prov, _, _ = _mk(regime_p=(0.0, 1.0, 0.0))
    rng = random.Random(3)
    for _ in range(200):
        it = prov.assemble_task(rng, 3.0)
        if it["label_name"] == "a":
            assert it["meta"]["regime"] == "mri"       # regime label is unchanged
            assert it["modality"] == "ct"              # resolved slot modality fell back
            assert it["meta"]["ctx_mod"] == "ct"


def test_determinism_same_seed_same_item():
    p1, _, _ = _mk()
    p2, _, _ = _mk()
    r1, r2 = random.Random(123), random.Random(123)
    for _ in range(50):
        a = p1.assemble_task(r1, 2.5)
        b = p2.assemble_task(r2, 2.5)
        assert a["subject"] == b["subject"]
        assert a["context_subjects"] == b["context_subjects"]
        assert a["meta"] == b["meta"]
        assert torch.equal(a["context_in"], b["context_in"])


def test_short_pool_warns_and_repeats():
    ct = _FakeSub("ct", {"x": ["only0"]})
    mri = _FakeSub("mri", {"x": ["mx0", "mx1"]})
    prov = MultiSourceProvider({"ct": ct, "mri": mri}, context_size=1,
                               regime_p=(1.0, 0.0, 0.0), epoch_length=10)
    rng = random.Random(0)
    with pytest.warns(UserWarning, match="repeating"):
        it = prov.assemble_task(rng, 3.0)
    assert it["subject"] == "only0"
    assert it["context_subjects"] == ["only0"]


def test_rejects_wrong_subprovider_count():
    ct = _FakeSub("ct", {"a": ["ca0"]})
    with pytest.raises(ValueError, match="exactly 2"):
        MultiSourceProvider({"ct": ct}, context_size=1, epoch_length=1)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest src/providers/test_multisource.py -q`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'src.providers.multisource'`.

- [ ] **Step 3: Write the implementation**

Create `src/providers/multisource.py`:

```python
"""Multi-source in-context provider (v2 cohort hook).

Composes two modality-locked TotalSegProviders into one provider that draws, per
task, a modality REGIME (all-source-0 / all-source-1 / forced cross) and loads the
K+1 cases accordingly. A class with no subjects in the wanted modality falls back
to the other modality. Implements the engine's `assemble_task` hook (like
src/providers/synth_gmm.py), so InContextDataset owns only aug + the per-item RNG.
"""
import warnings

import torch

from src.incontext_dataset_v2 import LoadRequest


class MultiSourceProvider:
    """Cohort-hook provider over exactly two modality-locked sub-providers."""

    def __init__(self, sub_providers, *, context_size, regime_p=(1 / 3, 1 / 3, 1 / 3),
                 epoch_length=1000):
        if len(sub_providers) != 2:
            raise ValueError(f"MultiSourceProvider expects exactly 2 sub-providers, "
                             f"got {list(sub_providers)}")
        self.subs = dict(sub_providers)          # insertion order defines m0, m1
        self._mods = list(self.subs)
        self.context_size = int(context_size)
        self.regime_p = tuple(float(x) for x in regime_p)
        if len(self.regime_p) != 3:
            raise ValueError(f"regime_p needs 3 entries (m0, m1, cross), got {regime_p}")
        self.epoch_length = int(epoch_length)

        all_classes = set()
        for p in self.subs.values():
            all_classes.update(p.classes)
        self._avail = {}
        for c in sorted(all_classes):
            mods = [m for m, p in self.subs.items() if p.subjects_for(c)]
            if mods:
                self._avail[c] = mods
        self.classes = list(self._avail)
        if not self.classes:
            raise ValueError("MultiSourceProvider: no class has subjects in any sub-provider")

    # --- VolumeProvider protocol stubs (engine uses assemble_task in cohort mode) ---
    def subjects_for(self, cls):
        return []

    def load(self, *a, **k):
        raise RuntimeError("MultiSourceProvider is a cohort provider; use assemble_task")

    # --- helpers ---
    def _draw_subjects(self, rng, mod, cls, n):
        """`n` distinct subjects for (mod, cls); repeat with a warning if the pool is short."""
        pool = list(self.subs[mod].subjects_for(cls))
        rng.shuffle(pool)
        if len(pool) >= n:
            return pool[:n]
        warnings.warn(
            f"MultiSourceProvider: only {len(pool)} {mod} subject(s) for {cls!r}, "
            f"need {n}; repeating (metrics leakage-inflated).", stacklevel=2)
        out = list(pool)
        while len(out) < n:
            out.append(pool[len(out) % len(pool)])
        return out

    # --- cohort hook ---
    def assemble_task(self, rng, crop_spacing_mm):
        cls = rng.choice(self.classes)
        avail = self._avail[cls]
        m0, m1 = self._mods
        regime = rng.choices([m0, m1, "cross"], weights=self.regime_p, k=1)[0]

        if regime == "cross":
            tgt_mod = rng.choice(avail)
            other = m1 if tgt_mod == m0 else m0
            ctx_mod = other if other in avail else tgt_mod
        else:
            tgt_mod = regime if regime in avail else avail[0]
            ctx_mod = tgt_mod

        k = self.context_size
        if tgt_mod == ctx_mod:
            subs = self._draw_subjects(rng, tgt_mod, cls, k + 1)
            tgt_subj, ctx_subjs = subs[0], subs[1:]
        else:
            tgt_subj = self._draw_subjects(rng, tgt_mod, cls, 1)[0]
            ctx_subjs = self._draw_subjects(rng, ctx_mod, cls, k)

        def _load(mod, subj):
            return self.subs[mod].load(
                subj, cls, LoadRequest(rng=rng, crop_spacing_mm=float(crop_spacing_mm)))

        tgt = _load(tgt_mod, tgt_subj)
        ctx = [_load(ctx_mod, s) for s in ctx_subjs]

        return {
            "image": tgt.image,
            "label": tgt.label,
            "context_in": torch.stack([r.image for r in ctx]),
            "context_out": torch.stack([r.label for r in ctx]),
            "spacing": tgt.spacing,
            "crop_geom": tgt.crop_geom,
            "subject": tgt_subj,
            "context_subjects": list(ctx_subjs),
            "label_name": cls,
            "modality": tgt_mod,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "meta": {"regime": regime, "tgt_mod": tgt_mod, "ctx_mod": ctx_mod},
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest src/providers/test_multisource.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Log + commit**

Add a bullet to `docs/logs.md` under a `2026-09-06` heading:
`- multi-source dataloader: add MultiSourceProvider (v2 cohort hook) — per-task modality regime (all-ct / all-mri / forced cross) over two modality-locked TotalSegProviders, with per-class fallback to the other modality. src/providers/multisource.py (+ unit tests).`

```bash
git add src/providers/multisource.py src/providers/test_multisource.py docs/logs.md
git commit -m "feat: MultiSourceProvider — per-task CT/MRI modality regime (v2 cohort hook)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019Cu129vDtFyCnYmHpwG8T2"
```

---

### Task 2: Hydra configs + `common.py` / `train.py` wiring

**Files:**
- Create: `configs/experiment/3d/dataset/multisource_ct_mri.yaml`
- Create: `configs/experiment/3d/experiment/81_multisource_ct_mri.yaml`
- Modify: `experiments/3d/common.py` (`_source_root`; new `resolve_multisource_classes`; `build_dataset`; `make_eval_loader`)
- Modify: `experiments/3d/train.py` (`_resolve_classes_for`)
- Test: `experiments/3d/_check_multisource.py` (real NFS data, no GPU)

**Interfaces:**
- Consumes: `MultiSourceProvider` from Task 1; `src.providers.totalseg.TotalSegProvider(root, classes, image_size, split=None, max_subjects=None, crop_spacing_mm=1.5, crop_jitter=None, mask_downsample="occupancy", mask_occupancy_thr=0.1, modality="ct", ct_norm=None, ram_cache=False, ram_cache_max_subjects=None)`; `src.incontext_dataset_v2.InContextDataset(provider, context_size=3, class_balanced=False, aug_cfg=None, defer_aug=False, crop_spacing_mm=1.5, eval_seed=None, max_tasks_per_class=None, gpu_realize_crop=False)`; `data.totalseg_classes.resolve_classes(value, totalseg_root=None, is_mri=False) -> list[str]`.
- Produces:
  - `common.resolve_multisource_classes(cfg, which: str) -> list[str]` where `which in ("train", "val")` — sorted union of `resolve_classes` over `cfg.data.source_mix.sources`.
  - `_source_root(cfg)` returns `("multisource", <cfg.paths.totalseg>, False)` when `cfg.data.source == "multisource"`.
  - `build_dataset(cfg, "train"|"val"|"test")` returns an `InContextDataset` in cohort mode for `source == "multisource"`.
  - `make_eval_loader(cfg, classes, split, spacing=None)` returns a forkserver `DataLoader` for `source == "multisource"` (routed through `build_dataset`).

- [ ] **Step 1: Write the dataset config**

Create `configs/experiment/3d/dataset/multisource_ct_mri.yaml`:

```yaml
# @package _global_
# multisource_ct_mri — joint CT (totalseg) + MRI (totalsegmri) in-context training,
# v2 cohort path. Selected as `dataset=multisource_ct_mri`. Geometry matches the
# d2 varspacing regime (128^3, K=1, log-uniform [1.5, 6] mm crop pitch, eval fixed
# at 3 mm). Per task the loader draws a modality REGIME:
#   ct    (~1/3): target + all K contexts CT
#   mri   (~1/3): target + all K contexts MRI
#   cross (~1/3): target one modality, all K contexts the other (forced)
# A class with no subjects in the wanted modality falls back to the other one
# (see src/providers/multisource.py). Eval is a stochastic-but-seeded mix of
# `eval_epoch_length` items; regime/modality breakdown is read from the wandb
# sample-table `detail` column (evaluate.py::_sample_detail).
#
#   python experiments/3d/train.py experiment=81_multisource_ct_mri
#   python experiments/3d/plot_dataset_items.py dataset=multisource_ct_mri --split train
paths:
  # Repeated for cluster-independence (also in cluster/nfs.yaml).
  totalseg:    /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg
  totalsegmri: /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalsegmri
data:
  source: multisource
  loader_v2: true
  image_size: [128, 128, 128]
  context_size: 1
  train_spacing_range: [1.5, 6.0]     # per-batch log-uniform crop pitch (SpacingBatchSampler)
  crop_spacing_mm: 3                  # fixed eval pitch (train_spacing_range is train-only)
  mask_downsample: soft              # eval maps back to "occupancy" in make_eval_loader
  mask_occupancy_thr: 0.5
  class_balanced: true               # (cohort assemble_task samples class uniformly anyway)
  max_ds_len_train: 1000             # train samples per epoch
  ct_norm: null                      # CT frame; the MRI sub-provider uses per-subject z-score
  train_classes: union               # sentinel — resolved by common.resolve_multisource_classes
  val_classes: union
  source_mix:
    sources: [totalseg, totalsegmri]
    modalities: [ct, mri]                        # parallel to `sources`
    per_source_train_classes: [balanced, train]  # spec fed to resolve_classes per source
    per_source_val_classes:   [all, test]        # CT val on all 117; MRI on its `test` classes
    regime_p: [0.334, 0.333, 0.333]              # (all-ct, all-mri, cross)
    split_map: {totalsegmri: {val: test}}        # MRI has no `val` split -> eval on `test`
    eval_epoch_length: 1400                      # ~20 tasks/class over ~70 union classes
```

- [ ] **Step 2: Write the experiment config**

Create `configs/experiment/3d/experiment/81_multisource_ct_mri.yaml`:

```yaml
# @package _global_
# 81_multisource_ct_mri — exp80 (hard target prior, mask_embed=conv, instance input
# norm) trained JOINTLY on CT (totalseg) + MRI (totalsegmri) with a per-task modality
# regime (1/3 all-ct, 1/3 all-mri, 1/3 forced cross). See
# docs/superpowers/specs/2026-09-06-multi-source-ct-mri-dataloader-design.md and
# configs/experiment/3d/dataset/multisource_ct_mri.yaml.
#
#   python experiments/3d/train.py experiment=81_multisource_ct_mri
#
# exp80 already ships arch.encoder_input_norm=instance ("ready for the planned CT+MRI
# joint run") — this experiment is what consumes it.
defaults:
  - 80_varspacing_hard_tgt_prior
  - override /dataset: multisource_ct_mri
  - _self_

wandb:
  name: 81_multisource_ct_mri
```

- [ ] **Step 3: Write the failing integration check**

Create `experiments/3d/_check_multisource.py`:

```python
"""Integration check for dataset=multisource_ct_mri (real NFS data, no GPU).

    python experiments/3d/_check_multisource.py

Builds the train + eval datasets via Hydra compose, pulls items, and asserts:
  - regime frequencies are ~1/3 each (over classes available in both modalities);
  - every `cross` task on a both-modality class is genuinely cross-modality;
  - a CT-only union class never yields an MRI slot;
  - the eval dataset is deterministic (idx -> identical item across two builds);
  - every union class with subjects appears at least once in the eval pass.
Exit code 0 = all pass.
"""
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir

from common import build_dataset


def _cfg(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                              version_base=None):
        return compose(config_name="train", overrides=overrides)


def main():
    base = ["experiment=81_multisource_ct_mri", "cluster=nfs"]
    cfg = _cfg(base)

    # --- train dataset: regime mix + cross-modality + CT-only guard ---
    ds = build_dataset(cfg, "train")
    prov = ds.provider
    print(f"union classes: {len(prov.classes)}  both-modality: "
          f"{sum(len(v) == 2 for v in prov._avail.values())}")
    assert ds.cohort_mode, "expected cohort_mode for multisource"

    rng = __import__("random").Random(0)
    regimes = Counter()
    cross_checked = 0
    for _ in range(600):
        it = prov.assemble_task(rng, 3.0)
        m = it["meta"]
        regimes[m["regime"]] += 1
        if m["regime"] == "cross" and len(prov._avail[it["label_name"]]) == 2:
            cross_checked += 1
            assert m["tgt_mod"] != m["ctx_mod"], m
        if prov._avail[it["label_name"]] == ["ct"]:
            assert it["modality"] == "ct" and m["ctx_mod"] == "ct", m
    print("train regime mix:", dict(regimes))
    tot = sum(regimes.values())
    for r in ("ct", "mri", "cross"):
        assert abs(regimes[r] / tot - 1 / 3) < 0.06, (r, regimes)
    assert cross_checked > 20, "cross branch never hit a both-modality class"

    # --- eval dataset: determinism + class coverage ---
    ev1 = build_dataset(cfg, "val")
    ev2 = build_dataset(cfg, "val")
    assert ev1.cohort_mode and len(ev1) == cfg.data.source_mix.eval_epoch_length
    for idx in (0, 1, 7, 123, len(ev1) - 1):
        a, b = ev1[idx], ev2[idx]
        assert a["subject"] == b["subject"]
        assert a["context_subjects"] == b["context_subjects"]
        assert a["meta"] == b["meta"]
        assert torch.equal(a["context_in"], b["context_in"]), idx
    seen = Counter(ev1[i]["label_name"] for i in range(len(ev1)))
    missing = [c for c in ev1.provider.classes if seen[c] == 0]
    print(f"eval covers {len(seen)}/{len(ev1.provider.classes)} classes; "
          f"missing: {missing[:10]}")
    assert not missing, f"{len(missing)} eval classes never sampled: {missing[:10]}"

    print("OK")


if __name__ == "__main__":
    main()
```

Run: `python experiments/3d/_check_multisource.py`
Expected: FAIL — `_source_root` raises `unknown data.source 'multisource'` (via `build_dataset` → class resolution / `_source_root`).

- [ ] **Step 4: Wire `_source_root` + `resolve_multisource_classes` in `common.py`**

In `experiments/3d/common.py`, in `_source_root(cfg)` (currently starts ~line 92), add a branch **before** the `if source not in _TOTALSEG_SOURCES:` check:

```python
    if source == "multisource":
        root = cfg.paths.get("totalseg")
        if root is None:
            raise ValueError("cfg.paths.totalseg is not set (needed for data.source=multisource)")
        return source, root, False
```

Then add a module-level helper (put it next to `_source_root`):

```python
def resolve_multisource_classes(cfg, which: str) -> list[str]:
    """Sorted union of per-source class lists for data.source=multisource.

    `which` is "train" or "val"; the per-source spec comes from
    data.source_mix.per_source_{which}_classes (parallel to .sources / .modalities).
    """
    sm = cfg.data.source_mix
    key = f"per_source_{which}_classes"
    specs = sm[key]
    out: set[str] = set()
    for src, mod, spec in zip(sm.sources, sm.modalities, specs):
        root = cfg.paths.get(src)
        if root is None:
            raise ValueError(f"cfg.paths.{src} is not set (data.source_mix.sources)")
        out.update(resolve_classes(spec, root, is_mri=(mod == "mri")))
    return sorted(out)
```

- [ ] **Step 5: Add the `multisource` branch to `build_dataset`**

In `experiments/3d/common.py::build_dataset`, add this branch **immediately before** the `if d.get("loader_v2", False) and d.get("source", ...) in _TOTALSEG_SOURCES:` branch (~line 295):

```python
    if d.get("source") == "multisource":
        from src.incontext_dataset_v2 import InContextDataset
        from src.providers.totalseg import TotalSegProvider
        from src.providers.multisource import MultiSourceProvider
        is_train = split == "train"
        sm = d.source_mix
        which = "train" if is_train else "val"
        specs = sm[f"per_source_{which}_classes"]
        split_map = sm.get("split_map", {}) or {}
        subs = {}
        for src, mod, spec in zip(sm.sources, sm.modalities, specs):
            root = cfg.paths.get(src)
            sub_split = split_map.get(src, {}).get(split, split)
            classes = resolve_classes(spec, root, is_mri=(mod == "mri"))
            subs[mod] = TotalSegProvider(
                root=root, classes=classes, image_size=tuple(d.image_size),
                split=sub_split,
                max_subjects=(d.get("max_train_subjects") if is_train
                              else d.get("max_val_subjects")),
                crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
                crop_jitter=(d.get("crop_jitter") if is_train
                             else cfg.get("eval", {}).get("crop_jitter", 0)),
                mask_downsample=(d.get("mask_downsample", "occupancy") if is_train
                                 else ("occupancy" if d.get("mask_downsample") == "soft"
                                       else d.get("mask_downsample", "occupancy"))),
                mask_occupancy_thr=d.get("mask_occupancy_thr", 0.1),
                modality=mod, ct_norm=d.get("ct_norm"), ram_cache=False)
        provider = MultiSourceProvider(
            subs, context_size=d.context_size,
            regime_p=tuple(sm.get("regime_p", (1 / 3, 1 / 3, 1 / 3))),
            epoch_length=((d.get("max_ds_len_train") or 1000) if is_train
                          else int(sm.eval_epoch_length)))
        return InContextDataset(
            provider, context_size=d.context_size,
            aug_cfg=(cfg.augmentations if is_train else None),
            defer_aug=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            eval_seed=(None if is_train else int(cfg.get("eval", {}).get("seed", 0))))
```

- [ ] **Step 6: Route `make_eval_loader` for `multisource`**

In `experiments/3d/common.py::make_eval_loader`, find the special-case tuple (~line 655):

```python
    if d.get("source") in ("omnisynth3d", "anchor_synth3d", "totalseg_more_labels",
                            "chemotox_bc", "synth_gmm_maisi", "flare22", "nasalseg"):
```

Add `"multisource"` to that tuple. That branch already: calls `build_dataset(cfg, split)`, builds the forkserver `DataLoader`, and handles `spacing` only for `totalseg_more_labels` (multisource is never called with `spacing` set from `train.py`'s val loop — leave that sub-branch alone).

- [ ] **Step 7: Wire `_resolve_classes_for` in `train.py`**

In `experiments/3d/train.py::_resolve_classes_for` (~line 854), add **before** the `_, root, is_mri = _source_root(cfg)` tail:

```python
    if src == "multisource":
        from common import resolve_multisource_classes
        return resolve_multisource_classes(cfg, "train" if classes_key == "train_classes" else "val")
```

- [ ] **Step 8: Run the integration check**

Run: `python experiments/3d/_check_multisource.py`
Expected: prints `union classes: …`, `train regime mix: …`, `eval covers …`, then `OK`. Exit 0.

If `eval covers` reports missing classes, raise `data.source_mix.eval_epoch_length` in the dataset config until coverage is complete, then re-run.

- [ ] **Step 9: Visual sanity check**

Run: `python experiments/3d/plot_dataset_items.py dataset=multisource_ct_mri --split train`
Expected: a grid of items; CT and MRI targets both look plausible, masks aligned; no exception. (Inspect the saved figure path it prints.)

- [ ] **Step 10: Log + commit**

Add to `docs/logs.md` (2026-09-06 section):
`- multi-source dataloader: dataset=multisource_ct_mri + experiment=81_multisource_ct_mri; common.py gains _source_root/build_dataset/make_eval_loader branches + resolve_multisource_classes; train.py._resolve_classes_for branch. CT eval on val, MRI eval on test (split_map). Integration check: experiments/3d/_check_multisource.py.`

```bash
git add configs/experiment/3d/dataset/multisource_ct_mri.yaml \
        configs/experiment/3d/experiment/81_multisource_ct_mri.yaml \
        experiments/3d/_check_multisource.py experiments/3d/common.py \
        experiments/3d/train.py docs/logs.md
git commit -m "feat: dataset=multisource_ct_mri + exp 81 — joint CT/MRI in-context training

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019Cu129vDtFyCnYmHpwG8T2"
```

---

### Task 3: Regime breakdown in the eval sample table

**Files:**
- Modify: `experiments/3d/evaluate.py` (`_sample_detail`, ~line 264)
- Test: `experiments/3d/test_sample_detail_regime.py`

**Interfaces:**
- Consumes: `evaluate._sample_detail(meta: dict | None) -> str`. `meta` for a multisource item is `{"regime": "ct"|"mri"|"cross", "tgt_mod": "ct"|"mri", "ctx_mod": "ct"|"mri"}` (from Task 1). `incontext_collate_fn` already forwards per-item `meta` as `batch["meta"]`, and `evaluate_classes` already sets `case["detail"] = _sample_detail(metas[i])`.
- Produces: `_sample_detail` returns `"<regime> <tgt_mod><-<ctx_mod>"` (e.g. `"cross ct<-mri"`) when `meta` has a `"regime"` key; unchanged for every other input.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/test_sample_detail_regime.py`:

```python
"""_sample_detail renders a multisource regime meta into the detail column."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import _sample_detail


def test_regime_meta_renders():
    assert _sample_detail({"regime": "cross", "tgt_mod": "ct", "ctx_mod": "mri"}) == "cross ct<-mri"
    assert _sample_detail({"regime": "ct", "tgt_mod": "ct", "ctx_mod": "ct"}) == "ct ct<-ct"


def test_non_regime_meta_unchanged():
    assert _sample_detail(None) == ""
    assert _sample_detail({}) == ""
    assert _sample_detail({"class_id": 3, "target_mode": "x", "sample_index": 1}) == \
        "mode=x class=3 sub=1"
```

Run: `python -m pytest experiments/3d/test_sample_detail_regime.py -q`
Expected: FAIL on `test_regime_meta_renders` (`_sample_detail` returns `""` for regime meta).

- [ ] **Step 2: Implement**

In `experiments/3d/evaluate.py::_sample_detail`, add a branch after the `if not meta:` guard and before the omniSynth `if "class_id" in meta:` branch:

```python
    if "regime" in meta:  # multisource (src/providers/multisource.py)
        return f"{meta['regime']} {meta.get('tgt_mod', '?')}<-{meta.get('ctx_mod', '?')}"
```

Update the docstring's source list to mention `multisource meta -> "<regime> <tgt><-<ctx>"`.

- [ ] **Step 3: Run the test to verify it passes**

Run: `python -m pytest experiments/3d/test_sample_detail_regime.py -q`
Expected: PASS (2 tests).

- [ ] **Step 4: Log + commit**

Add to `docs/logs.md` (2026-09-06 section):
`- multi-source dataloader: evaluate.py::_sample_detail renders multisource regime meta ("<regime> <tgt><-<ctx>") into the wandb sample-table `detail` column, so eval Dice can be sliced by regime/modality post-hoc.`

```bash
git add experiments/3d/evaluate.py experiments/3d/test_sample_detail_regime.py docs/logs.md
git commit -m "feat: regime/modality breakdown in the eval sample table (multisource)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019Cu129vDtFyCnYmHpwG8T2"
```

---

### Task 4: End-to-end training smoke run

**Files:**
- No source changes. Uses `experiments/3d/train.py` + `experiment=81_multisource_ct_mri`.
- Runs on the local RTX A6000.

**Interfaces:**
- Consumes: everything from Tasks 1–3, composed via `experiment=81_multisource_ct_mri`.
- Produces: confidence that the full train+eval loop (SpacingBatchSampler → cohort `assemble_task` → CPU aug → collate → train step → val loop → `build_sample_table`) runs without error and yields a finite val Dice.

- [ ] **Step 1: Tiny run, wandb disabled**

Run:

```bash
python experiments/3d/train.py experiment=81_multisource_ct_mri \
  wandb.project=null train.epochs=1 train.eval_every=1 \
  data.max_ds_len_train=16 train.batch_size=2 train.workers=2 \
  data.source_mix.eval_epoch_length=24 eval.batch_size=2 eval.workers=2 \
  2>&1 | tee /tmp/claude-1011/-home-dpxuser-dev-patch-icl/d2ef5e20-9a74-4dba-81b9-d4a27bcd12ea/scratchpad/ms_smoke.log
```

Expected: completes with a printed epoch line including a non-`nan` `val_dice` (or `val/dice`). No traceback. The run may be slow on first compile — allow up to ~15 min.

- [ ] **Step 2: Inspect the log for regime coverage**

Run: `grep -Ei "regime|cross|val.?dice|Traceback" /tmp/claude-1011/-home-dpxuser-dev-patch-icl/d2ef5e20-9a74-4dba-81b9-d4a27bcd12ea/scratchpad/ms_smoke.log | head -40`
Expected: a finite val dice line; no `Traceback`.

- [ ] **Step 3: Confirm mixed-modality batches actually formed**

Run:

```bash
python - <<'PY'
import random, sys
from pathlib import Path
ROOT = Path("/home/dpxuser/dev/patch_icl")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "3d"))
from hydra import compose, initialize_config_dir
from common import build_dataset
with initialize_config_dir(config_dir=str(ROOT / "configs/experiment/3d"), version_base=None):
    cfg = compose(config_name="train", overrides=["experiment=81_multisource_ct_mri", "cluster=nfs"])
ds = build_dataset(cfg, "train")
rng = random.Random(0)
from collections import Counter
c = Counter(ds.provider.assemble_task(rng, 3.0)["meta"]["regime"] for _ in range(300))
print("regime mix over 300:", dict(c))
assert c["cross"] > 60 and c["ct"] > 60 and c["mri"] > 60, c
print("OK")
PY
```

Expected: `regime mix over 300: {...}` with all three regimes well represented, then `OK`.

- [ ] **Step 4: Log**

Add to `docs/logs.md` (2026-09-06 section):
`- multi-source dataloader: end-to-end smoke — experiment=81_multisource_ct_mri runs 1 epoch on an A6000 (tiny caps), finite val dice, all three regimes present. Ready for a full run.`

```bash
git add docs/logs.md
git commit -m "docs: log multisource end-to-end smoke pass

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019Cu129vDtFyCnYmHpwG8T2"
```

---

## Self-Review

**Spec coverage:**
- MultiSourceProvider + cohort hook + regime draw + fallback → Task 1. ✓
- Union class set (`per_source_*_classes`, `resolve_multisource_classes`) → Task 2 steps 4, 7; config step 1. ✓
- Forced cross-modality (`context_size=1`) → Task 1 (`assemble_task` cross branch) + tests `test_cross_is_cross_modality_when_both_available`. ✓
- Stochastic-but-seeded eval, fixed `eval_epoch_length` → Task 2 (`build_dataset` eval branch sets `epoch_length`, `eval_seed`) + check step 3 determinism/coverage. ✓
- MRI eval on `test` via `split_map` → Task 2 config step 1 + `build_dataset` `sub_split`. ✓
- `encoder_input_norm=instance`, hard prior, `mask_embed=conv` inherited → Task 2 experiment config `defaults: [80_varspacing_hard_tgt_prior, ...]`. ✓
- Regime breakdown in wandb sample table → Task 3. ✓
- `_assert_cascade_supported` rejecting `multisource` + cascade: already covered by the existing `source not in _TOTALSEG_SOURCES` guard in that function and the `gpu_realize_crop`-without-`cascade_spacings` guard — no new code needed; noted here so the executor does not add a redundant branch. ✓
- Testing: pytest unit (`test_multisource.py`), integration script (`_check_multisource.py`), `plot_dataset_items` visual, e2e smoke → Tasks 1–4. ✓
- Known limitations (MRI anisotropy, CT-only fallback classes, MRI-on-test, statistical eval coverage) — documented in spec; surfaced by check-script asserts/prints; no code owed. ✓

**Placeholder scan:** No TBD/TODO; every code step has real content; test bodies are concrete; no "similar to Task N".

**Type consistency:** `MultiSourceProvider(sub_providers, *, context_size, regime_p, epoch_length)` and the `.classes/._avail/._mods/.epoch_length` attrs and `assemble_task` return keys (`meta={"regime","tgt_mod","ctx_mod"}`) are used identically in Task 1 tests, Task 2 `build_dataset` + `_check_multisource.py`, and Task 3 `_sample_detail`. `resolve_multisource_classes(cfg, which)` signature matches both call sites (Task 2 step 4 defn, step 7 train.py call). `build_dataset` returns an `InContextDataset` whose `.cohort_mode` / `.provider` / `len()` the check script relies on — those exist on the current engine.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-06-multi-source-ct-mri-dataloader.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

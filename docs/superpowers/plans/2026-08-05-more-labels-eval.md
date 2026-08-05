# TotalSegmentator `more_labels` Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any context-viable extra TotalSegmentator `more_labels` class be named in `cfg.data.val_classes` and evaluated by `experiments/3d/eval.py` exactly like a normal in-context class.

**Architecture:** A thin `TotalSegMoreLabelsDataset(TotalSegInContextDataset)` roots at the separate `totalseg_test_more_labels/` tree, overriding only class identity (task-qualified `"{task}/{name}"` keys from `more_labels_classes.json` + `_subject_classes.json`) and `_load` (CT reproduced from `ct.nii.gz`, mask = `task array == local_id`). A new `data.source="totalseg_more_labels"` routes it through `common.py` / `eval.py` alongside the existing synth sources.

**Tech Stack:** Python 3.12, PyTorch, nibabel, Hydra/OmegaConf, numpy. Data on NFS.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-08-05-more-labels-eval-design.md`.
- Env is node-specific (see memory): activate the project venv for this session with `source .venv_thor/bin/activate` (adjust to the node's venv if different). Do NOT use `uv`.
- Project rule (CLAUDE.md): "Write tests only when necessary" — the one correctness-critical check (CT↔mask alignment) is a runnable script, not a pytest suite. Log every change in `docs/logs.md`.
- Class identifier is always the task-qualified key `"{task}/{name}"` (bare names collide across the 37 tasks).
- Extra masks are pre-sized at **64³ only**; the more_labels dataset config uses `image_size: [64, 64, 64]`. CT has no pre-sized file yet — loaded from `ct.nii.gz` and reproduced to match `convert_to_npy.py`'s `ct_{size}.npy`.
- Data roots (nfs cluster):
  - more_labels: `/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg_test_more_labels`
  - main totalseg (for the alignment check): `.../ANALYSIS_20251122/data/totalseg`
- Commit at the end of each task per your normal git workflow (branch already `feat/feature-sim-primus-encoder`; keep the spec/plan and unrelated staged changes out of feature commits).

---

### Task 1: `TotalSegMoreLabelsDataset`

**Files:**
- Create: `src/totalseg_more_labels_dataset.py`
- Create (runnable check = the test): `experiments/totalseg_more_labels/check_more_labels_dataset.py`

**Interfaces:**
- Consumes: `TotalSegInContextDataset` (base, `src/totalseg_dataloader_incontext.py`); `_iso_resize`, `_normalise_ct` (`scripts/convert_to_npy.py`).
- Produces: `TotalSegMoreLabelsDataset(root, classes, image_size=(64,64,64), split=None, context_size=3, max_subjects=None, eval_seed=0)` — a `Dataset` whose items have the same keys as `TotalSegInContextDataset` (`image (1,D,H,W) f32`, `label (D,H,W) i64`, `context_in`, `context_out`, `label_name="{task}/{name}"`, `subject`, `spacing`). Attributes used by later tasks/checks: `_resolve: {key:(task,local_id)}`, `_gid_to_key: {global_id:key}`, `label_to_subjects`, `samples`.

- [ ] **Step 1: Write the failing check script**

Create `experiments/totalseg_more_labels/check_more_labels_dataset.py`:

```python
"""Sanity check for TotalSegMoreLabelsDataset (the plan's correctness gate).

Proves (a) the CT it loads from ct.nii.gz aligns pixel-for-pixel with the main
tree's ct_{size}.npy (so it aligns with the pre-resized more_labels masks), and
(b) its binary label equals (more_labels/{task}_{size}.npy == local_id).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.totalseg_more_labels_dataset import TotalSegMoreLabelsDataset

DATA = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data")
MORE = DATA / "totalseg_test_more_labels"
MAIN = DATA / "totalseg"
SIZE = (64, 64, 64)


def main():
    idx = json.load(open(MORE / "more_labels_classes.json"))
    sc = json.load(open(MORE / "more_labels_subject_classes.json"))
    gid_to = {int(c["global_id"]): c for c in idx["classes"]}

    # a class present in >=2 subjects (context-viable), on a subject that also
    # exists in the main tree (for the CT-alignment reference).
    cnt = Counter(g for v in sc.values() for g in v)
    gid = next(g for g, k in cnt.items()
               if k >= 2 and (MAIN / next(s for s, v in sc.items() if g in v)
                              / f"ct_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy").exists())
    c = gid_to[gid]
    key = f"{c['task']}/{c['name']}"
    subj = next(s for s, v in sc.items() if gid in v
                and (MAIN / s / f"ct_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy").exists())

    ds = TotalSegMoreLabelsDataset(root=MORE, classes=[key], image_size=SIZE, split="test")
    img, lbl = ds._load(subj, key)

    ref_ct = np.load(MAIN / subj / f"ct_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy").astype(np.float32)
    assert img.shape == (1, *SIZE), img.shape
    assert np.allclose(img[0].numpy(), ref_ct, atol=1e-2), \
        f"CT misaligned: max|diff|={np.abs(img[0].numpy()-ref_ct).max()}"

    task_arr = np.load(MORE / subj / "more_labels" / f"{c['task']}_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy")
    exp = (task_arr == c["local_id"]).astype(np.int64)
    assert np.array_equal(lbl.numpy(), exp), "label != (task_arr == local_id)"

    # end-to-end item: same keys as the base dataset, with a matching-class context.
    item = ds[0]
    for k in ("image", "label", "context_in", "context_out", "label_name", "subject"):
        assert k in item, k
    assert item["label_name"] == ds.samples[0][1]
    print(f"OK  subj={subj}  class={key}  fg={int(lbl.sum())}  "
          f"ctx={item['context_in'].shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source .venv_thor/bin/activate
python experiments/totalseg_more_labels/check_more_labels_dataset.py
```

Expected: `ModuleNotFoundError: No module named 'src.totalseg_more_labels_dataset'`.

- [ ] **Step 3: Implement the dataset**

Create `src/totalseg_more_labels_dataset.py`:

```python
"""In-context EVAL dataset over the extra TotalSegmentator `more_labels` classes.

Reuses TotalSegInContextDataset for context sampling, eval-seed determinism, the
single-label __getitem__ path, and the collate contract. Overrides only:

  * class identity  — classes are task-qualified keys "{task}/{name}" from
                      more_labels_classes.json (329 unique names collide across the
                      37 tasks, so the bare name is not unique); subject->classes
                      comes from more_labels_subject_classes.json, not a label.npy scan.
  * loading (_load) — CT from ct.nii.gz, reproducing convert_to_npy's normalise +
                      iso_resize so it aligns pixel-for-pixel with the pre-resized
                      more_labels/{task}_{size}.npy masks; binary mask = task array
                      == local_id.

Eval-only: use_crop / synth / augmentation / multi-label are asserted off.
"""
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from scripts.convert_to_npy import _iso_resize, _normalise_ct
from src.totalseg_dataloader_incontext import TotalSegInContextDataset


class TotalSegMoreLabelsDataset(TotalSegInContextDataset):
    def __init__(
        self,
        root: str | Path,
        classes: list[str],
        image_size: Optional[tuple[int, int, int]] = (64, 64, 64),
        split: Optional[str] = None,
        context_size: int = 3,
        max_subjects: Optional[int] = None,
        eval_seed: int = 0,
    ):
        root = Path(root)
        # Read the global index BEFORE super().__init__: the overridden
        # _load_or_build_cache (called inside super) needs _gid_to_key.
        with open(root / "more_labels_classes.json") as f:
            index = json.load(f)
        self._resolve: dict[str, tuple[str, int]] = {}
        self._gid_to_key: dict[int, str] = {}
        for c in index["classes"]:
            key = f"{c['task']}/{c['name']}"
            self._resolve[key] = (c["task"], int(c["local_id"]))
            self._gid_to_key[int(c["global_id"])] = key
        with open(root / "more_labels_subject_classes.json") as f:
            self._subject_gids: dict[str, list[int]] = json.load(f)
        self._ct_cache: dict[str, torch.Tensor] = {}

        super().__init__(
            root=root,
            classes=classes,
            image_size=image_size,
            split=split,
            context_size=context_size,
            max_subjects=max_subjects,
            aug_cfg=None,
            synth_method=None,
            p_synth=0.0,
            class_balanced=False,
            use_crop=False,
            num_labels_per_sample=1,
            eval_seed=eval_seed,
        )

    # --- overrides -----------------------------------------------------------
    def _get_subjects(self, split, meta_csv, max_subjects) -> list[str]:
        """No meta.csv in this tree; the 25 subjects are all 'test'. List dirs that
        actually carry a more_labels/ folder (ignores the two root JSON files)."""
        subs = sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "more_labels").is_dir())
        if max_subjects is not None:
            subs = subs[:max_subjects]
        return subs

    def _load_or_build_cache(self) -> dict[str, frozenset]:
        """subject -> frozenset("{task}/{name}") straight from the JSON — no label.npy
        scan, no .scan_cache pickle."""
        return {
            subj: frozenset(self._gid_to_key[g] for g in gids if g in self._gid_to_key)
            for subj, gids in self._subject_gids.items()
        }

    def _load_ct_resized(self, subj: str) -> torch.Tensor:
        """(1, D, H, W) f32 CT, resized to match the main tree's ct_{size}.npy. Cached
        per subject (25 subjects, ~26 MB/worker) so contexts don't re-decode the NIfTI."""
        t = self._ct_cache.get(subj)
        if t is not None:
            return t
        subj_dir = self.root / subj
        pre = (subj_dir / f"ct_{self._size_str}.npy") if self._size_str else None
        if pre is not None and pre.exists():
            t = torch.from_numpy(np.load(pre, mmap_mode="r").astype(np.float32)).unsqueeze(0)
        else:
            img = nib.as_closest_canonical(nib.load(str(subj_dir / "ct.nii.gz")))
            sp = tuple(float(x) for x in nib.affines.voxel_sizes(img.affine)[:3])
            vol = _normalise_ct(img.get_fdata(dtype=np.float32))
            if self.image_size is not None:
                vol = _iso_resize(vol, self.image_size, order=1, aa=True, spacing=sp)
            t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).unsqueeze(0)
        self._ct_cache[subj] = t
        return t

    def _load(self, subj: str, cls: str) -> tuple[torch.Tensor, torch.Tensor]:
        image_t = self._load_ct_resized(subj).clone()
        task, local_id = self._resolve[cls]
        mdir = self.root / subj / "more_labels"
        sized = (mdir / f"{task}_{self._size_str}.npy") if self._size_str else None
        if sized is not None and sized.exists():
            arr = np.asarray(np.load(sized, mmap_mode="r"))
        else:
            native = np.asarray(np.load(mdir / f"{task}.npy", mmap_mode="r"))
            arr = (_iso_resize(native, self.image_size, order=0, aa=False)
                   if self.image_size is not None else native)
        label_t = torch.from_numpy((arr == local_id).astype(np.int64))
        return image_t, label_t
```

- [ ] **Step 4: Run the check to verify it passes**

```bash
python experiments/totalseg_more_labels/check_more_labels_dataset.py
```

Expected: `OK  subj=... class=.../...  fg=<n>  ctx=torch.Size([3, 1, 64, 64, 64])` and no assertion error.

- [ ] **Step 5: Commit**

```bash
git add src/totalseg_more_labels_dataset.py experiments/totalseg_more_labels/check_more_labels_dataset.py
git commit -m "feat(more_labels): TotalSegMoreLabelsDataset for in-context eval"
```

---

### Task 2: `resolve_more_labels_classes` helper

**Files:**
- Modify: `data/totalseg_classes.py` (append a new function near `resolve_classes`, ~line 250)

**Interfaces:**
- Produces: `resolve_more_labels_classes(root, value) -> list[str]`. `root` = more_labels tree; `value=="all"` → the classes present in ≥2 subjects (context-viable), sorted; a list/ListConfig → validated pass-through of `"{task}/{name}"` keys.

- [ ] **Step 1: Write the failing check**

Append to the bottom of `experiments/totalseg_more_labels/check_more_labels_dataset.py` a second entrypoint (or a quick inline check). Add this function and call it from `main()` before the print:

```python
def _check_resolve():
    from data.totalseg_classes import resolve_more_labels_classes
    allc = resolve_more_labels_classes(MORE, "all")
    assert len(allc) == 285, len(allc)          # classes present in >=2 subjects
    assert all("/" in k for k in allc)          # task-qualified keys
    picked = resolve_more_labels_classes(MORE, [allc[0]])
    assert picked == [allc[0]]
    print(f"resolve OK  n_all={len(allc)}")
```

Add `_check_resolve()` as the first line of `main()`.

- [ ] **Step 2: Run to verify it fails**

```bash
python experiments/totalseg_more_labels/check_more_labels_dataset.py
```

Expected: `ImportError: cannot import name 'resolve_more_labels_classes'`.

- [ ] **Step 3: Implement the helper**

Append to `data/totalseg_classes.py` (after `resolve_classes`):

```python
def resolve_more_labels_classes(root, value) -> list[str]:
    """Resolve the extra `more_labels` eval class list (task-qualified "{task}/{name}").

    Reads {root}/more_labels_classes.json + more_labels_subject_classes.json.
      value == "all" -> every class present in >=2 subjects (context-viable: an
                        in-context item needs a target + >=1 same-class context),
                        sorted.
      list/ListConfig -> validated pass-through (each entry must be a known key).
    """
    import json
    from collections import Counter
    from pathlib import Path

    root = Path(root)
    index = json.load(open(root / "more_labels_classes.json"))
    subj_gids = json.load(open(root / "more_labels_subject_classes.json"))
    gid_to_key = {int(c["global_id"]): f"{c['task']}/{c['name']}" for c in index["classes"]}
    keys = set(gid_to_key.values())

    if not isinstance(value, str):
        want = list(value)
        bad = [k for k in want if k not in keys]
        if bad:
            raise ValueError(f"unknown more_labels classes: {bad[:5]}")
        return want

    if value != "all":
        raise ValueError(f"more_labels val_classes must be 'all' or a list, got {value!r}")

    cnt = Counter(g for gids in subj_gids.values() for g in gids)
    return sorted(gid_to_key[g] for g, k in cnt.items() if k >= 2 and g in gid_to_key)
```

- [ ] **Step 4: Run to verify it passes**

```bash
python experiments/totalseg_more_labels/check_more_labels_dataset.py
```

Expected: `resolve OK  n_all=285` then the Task-1 `OK ...` line.

- [ ] **Step 5: Commit**

```bash
git add data/totalseg_classes.py experiments/totalseg_more_labels/check_more_labels_dataset.py
git commit -m "feat(more_labels): resolve_more_labels_classes (all=285 viable)"
```

---

### Task 3: Wire the source into `common.py`

**Files:**
- Modify: `experiments/3d/common.py` — `_TOTALSEG_SOURCES`/`_source_root` (lines ~29-42), `build_dataset` (~133-159), `make_eval_loader` (~233-272).

**Interfaces:**
- Consumes: `TotalSegMoreLabelsDataset` (Task 1). `cfg.paths.totalseg_more_labels`.
- Produces: `build_dataset(cfg, split)` and `make_eval_loader(cfg, classes, split)` return a `TotalSegMoreLabelsDataset` when `cfg.data.source == "totalseg_more_labels"`.

- [ ] **Step 1: Add the import + source-root branch**

At the top of `experiments/3d/common.py`, alongside the base dataset import:

```python
from src.totalseg_more_labels_dataset import TotalSegMoreLabelsDataset
```

In `_source_root`, allow the new source and resolve its root:

```python
def _source_root(cfg) -> tuple[str, str, bool]:
    """Resolve (source, root, is_mri) from cfg.data.source — shared by all builders."""
    source = cfg.data.get("source", "totalseg")
    if source == "totalseg_more_labels":
        root = cfg.paths.get("totalseg_more_labels")
        if root is None:
            raise ValueError("cfg.paths.totalseg_more_labels is not set "
                             "(needed for data.source=totalseg_more_labels)")
        return source, root, False
    if source not in _TOTALSEG_SOURCES:
        raise ValueError(
            f"unknown data.source {source!r} (expected one of {_TOTALSEG_SOURCES})"
        )
    root = cfg.paths.get(source)
    if root is None:
        raise ValueError(f"cfg.paths.{source} is not set (needed for data.source={source!r})")
    return source, root, source == "totalsegmri"
```

- [ ] **Step 2: Route build_dataset + make_eval_loader**

In `build_dataset`, just before the `d = cfg.data` / `_source_root` block for the totalseg path (i.e. after the `anchor_synth3d` branch), add:

```python
    if cfg.data.get("source") == "totalseg_more_labels":
        d = cfg.data
        root = cfg.paths.get("totalseg_more_labels")
        classes = resolve_more_labels_classes(root, d.val_classes)
        return TotalSegMoreLabelsDataset(
            root=root, classes=classes, image_size=tuple(d.image_size),
            split=split, context_size=d.context_size,
            max_subjects=d.get("max_val_subjects"),
            eval_seed=int(cfg.get("eval", {}).get("seed", 0)),
        )
```

Add the import at the top of `common.py` next to `resolve_classes`:

```python
from data.totalseg_classes import resolve_classes, resolve_more_labels_classes
```

In `make_eval_loader`, extend the early synth special-case so the new source is also routed through `build_dataset` (it composes its own deterministic dataset):

```python
    if d.get("source") in ("omnisynth3d", "anchor_synth3d", "totalseg_more_labels"):
```

(the existing body already builds `ds = build_dataset(cfg, split)` and wraps it in a non-shuffled DataLoader — no other change needed).

- [ ] **Step 3: Smoke-check the loader builds**

```bash
python - <<'PY'
import sys; from pathlib import Path
sys.path.insert(0, "experiments/3d"); sys.path.insert(0, ".")
from omegaconf import OmegaConf
from common import make_eval_loader
from data.totalseg_classes import resolve_more_labels_classes
MORE="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg_test_more_labels"
cfg = OmegaConf.create({
  "data": {"source": "totalseg_more_labels", "image_size": [64,64,64],
           "context_size": 1, "val_classes": "all", "use_crop": False},
  "paths": {"totalseg_more_labels": MORE},
  "eval": {"seed": 0, "batch_size": 2, "workers": 0, "n_subjects": None},
})
classes = resolve_more_labels_classes(MORE, "all")[:3]
loader = make_eval_loader(cfg, classes, split="test")
batch = next(iter(loader))
print("batch image", batch["image"].shape, "labels", batch["label_names"])
PY
```

Expected: prints a batch with `image torch.Size([2, 1, 64, 64, 64])` and 2 `label_names` of the form `task/name`.

- [ ] **Step 4: Commit**

```bash
git add experiments/3d/common.py
git commit -m "feat(more_labels): route data.source=totalseg_more_labels in common.py"
```

---

### Task 4: `eval.py` branch + dataset config

**Files:**
- Modify: `experiments/3d/eval.py` `main()` source dispatch (lines ~136-154).
- Create: `configs/experiment/3d/dataset/totalseg_more_labels.yaml`.

**Interfaces:**
- Consumes: `resolve_more_labels_classes` (Task 2), the `totalseg_more_labels` source wiring (Task 3).
- Produces: `python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse` resolves the 285 classes and runs the shared eval loop.

- [ ] **Step 1: Add the eval.py source branch**

In `experiments/3d/eval.py` `main()`, in the source dispatch chain, add before the `else:` totalseg branch:

```python
    elif source == "totalseg_more_labels":
        from data.totalseg_classes import resolve_more_labels_classes
        root = cfg.paths.get("totalseg_more_labels")
        classes = resolve_more_labels_classes(root, cfg.data.val_classes)
```

(`root` and `classes` are the same locals the existing branches set; the rest of `main` is unchanged.)

- [ ] **Step 2: Create the dataset config**

Create `configs/experiment/3d/dataset/totalseg_more_labels.yaml`:

```yaml
# @package _global_
# Extra TotalSegmentator `more_labels` eval classes (converted by
# experiments/totalseg_more_labels/convert_more_labels.py). Composed as
# `dataset=totalseg_more_labels`. Eval-only: fast path, no crop/synth/aug.
#   python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse
paths:
  totalseg_more_labels: /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg_test_more_labels
data:
  source: totalseg_more_labels
  image_size: [64, 64, 64]     # only grid the extra masks are pre-sized at
  context_size: 1
  use_crop: false
  val_classes: all             # all = 285 classes present in >=2 subjects; or a list of "task/name"
  max_val_subjects: null
```

- [ ] **Step 3: Verify config composes**

```bash
python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse \
    eval.n_subjects=2 --cfg job 2>/dev/null | grep -E "source|image_size|val_classes|totalseg_more_labels" | head
```

Expected: shows `source: totalseg_more_labels`, `image_size: [64, 64, 64]`, `val_classes: all`, and the `totalseg_more_labels` path.

- [ ] **Step 4: Commit**

```bash
git add experiments/3d/eval.py configs/experiment/3d/dataset/totalseg_more_labels.yaml
git commit -m "feat(more_labels): eval.py source branch + dataset config"
```

---

### Task 5: End-to-end smoke eval + docs

**Files:**
- Modify: `docs/logs.md` (append dated entry).

**Interfaces:** none produced; final verification.

- [ ] **Step 1: Run a small real eval**

```bash
source .venv_thor/bin/activate
python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse \
    eval.n_subjects=4 data.val_classes='[body/body_trunk, brain_structures/brainstem]' \
    wandb.project=null
```

Expected: prints per-class `dice=…` rows for the two classes (non-crashing; Dice values plausible for medverse), a mean line, and `Saved -> …/3d_eval/…`. If a chosen class has <2 subjects it prints an `error: no samples` row — swap for a class from `resolve_more_labels_classes(MORE, "all")`.

- [ ] **Step 2: Append the docs/logs.md entry**

Add under today's date in `docs/logs.md`:

```markdown
2026-08-05: **Wired the extra TotalSegmentator `more_labels` classes into eval.**
New `src/totalseg_more_labels_dataset.py` (`TotalSegMoreLabelsDataset`, subclass of
`TotalSegInContextDataset`) roots at `totalseg_test_more_labels/`: class identity is
the task-qualified key `"{task}/{name}"` from `more_labels_classes.json` (329 unique
names collide across 37 tasks), subject→classes from `more_labels_subject_classes.json`
(no label.npy scan). `_load` loads CT from `ct.nii.gz` reproducing `convert_to_npy`'s
normalise + `_iso_resize` (aligns pixel-for-pixel with the pre-sized `{task}_64³.npy`
masks; verified by `experiments/totalseg_more_labels/check_more_labels_dataset.py`),
and the binary mask as `task_array == local_id`. New `data.source=totalseg_more_labels`
routes through `common.py`/`eval.py`; `resolve_more_labels_classes` exposes the 285
classes present in ≥2 subjects (`val_classes=all`). Run:
`python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse`.
Eval-only: fast path (64³), no crop/synth/aug.
```

- [ ] **Step 3: Commit**

```bash
git add docs/logs.md
git commit -m "docs: log more_labels eval wiring"
```

---

## Self-review notes

- **Spec coverage:** dataset (Task 1) · class resolution (Task 2) · common.py wiring (Task 3) · eval.py + config (Task 4) · verification + logs (Task 5). CT-alignment check and label check are in Task 1's script; smoke eval in Task 5. All spec sections mapped.
- **Type consistency:** `resolve_more_labels_classes(root, value)`, `TotalSegMoreLabelsDataset(root, classes, image_size, split, context_size, max_subjects, eval_seed)`, keys `"{task}/{name}"`, and attributes `_resolve`/`_gid_to_key` are used identically across tasks.
- **Out of scope (unchanged):** training/crop/synth/multi-label/MRI paths.
```

# Unified `validate()` + focused 2D eval script — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a single shared `validate()` used by both `train.py` and a new focused eval script (`eval_incontext.py`) for the `universeg` and `patchset_cnn` in-context models, so training-time and eval-time metrics are coherent by construction.

**Architecture:** New module `experiments/2d/evaluate.py` owns `validate()`, the qualitative `save_figure()`, and a per-source `_sample_detail()` formatter. `train.py` drops its own `validate()` and imports the shared one. A thin Hydra wrapper `eval_incontext.py` loads a checkpoint, dispatches on `model_name`, rebuilds the one model, and calls `validate()` with the expensive extras (figures, CSVs, FLOPs) enabled. The big 5-backend `eval.py` is left untouched.

**Tech Stack:** Python 3.12, PyTorch (bf16 autocast), Hydra/OmegaConf configs, Weights & Biases logging, matplotlib for figures. Design spec: `docs/superpowers/specs/2026-07-03-unified-validate-2d-eval-design.md`.

## Global Constraints

- **No pytest in this repo.** Verify with inline scripts run as `.venv/bin/python - <<'EOF' … EOF` and with short Hydra smoke runs. (`.venv/bin/python` has CUDA; training smoke runs may use `.venv311/bin/python` if that is your training env.)
- **Never stage or commit** — leave all version control to the user. "Commit" steps are replaced by verification checkpoints. Log changes to `docs/logs.md` (repo convention).
- **Data/paths:** omniSynth loader reads omniglot at `/home/dpxuser/repos/omniglot/python` (already present). Checkpoints live under `results/2d/…/best.pt` and the NFS `…/2d_train/*/best.pt`.
- **Model interface (both models):** `model(img, context_in=cin, context_out=cout, mode="val") -> {"final_logit": tensor}`. UniverSeg logit is native `(B,1,H,W)`; PatchSetCNN logit is low-res `(B,1,R,R)`.
- **Checkpoint format (saved by `train.py`):** `{"model", "model_name", "image_size", "context_size", "best_val_dice", "epoch", "data", "synth", **ckpt_meta}`. `ckpt_meta` for `patchset_cnn` is `{"arch": {...full ctor kwargs minus image_size...}}` (added this session); for `universeg` it is `{"pretrained": bool}`. Pre-existing checkpoints have NO `arch` key.

---

## File Structure

- `experiments/2d/evaluate.py` (NEW) — `validate()`, `save_figure()` (+ `_overlay_ax`/`_heatmap_ax`), `_sample_detail()`, `_fmt_transforms()`, `_target_like()`, `_upsample_to()`, `SAMPLE_COLS`.
- `experiments/2d/train.py` (MODIFY) — remove local `validate()` and its private helpers; import + call the shared one.
- `experiments/2d/eval.py` (MODIFY, minimal) — delete its 3 figure-helper defs; `from evaluate import save_figure`. No other change.
- `experiments/2d/eval_incontext.py` (NEW) — Hydra wrapper (config_name = `eval_base`).
- `docs/logs.md` (MODIFY) — log entry.

---

## Task 1: Scaffold `evaluate.py` — figure + detail helpers

**Files:**
- Create: `experiments/2d/evaluate.py`
- Modify: `experiments/2d/eval.py` — delete its copies of `save_figure`/`_overlay_ax`/`_heatmap_ax` (`:290-356`) and import them from `evaluate` instead (DRY — single home for the figure helpers).
- Reference (move from): `experiments/2d/eval.py:290-356` (figure helpers), `experiments/2d/train.py:211-226` (`_fmt_transforms`, `SAMPLE_COLS`)

**Interfaces:**
- Produces:
  - `save_figure(tgt_image, tgt_gt, pred_native, ctx_images, ctx_gts, out_path, title="", pred_lowres=None, gt_lowres=None) -> None`
  - `_sample_detail(meta: dict | None) -> str`
  - `_fmt_transforms(transforms) -> str`
  - `SAMPLE_COLS = ["epoch", "dataset", "sample_idx", "label", "dice", "dice_ds", "dice_ds_soft", "detail"]`

- [ ] **Step 1: Create the module with imports + verbatim figure helpers**

Create `experiments/2d/evaluate.py` with this top matter, then paste `_overlay_ax`, `_heatmap_ax`, and `save_figure` **verbatim** from `experiments/2d/eval.py:290-356` (unchanged):

```python
"""Shared 2D in-context evaluation: a single validate() used by train.py and
eval_incontext.py, plus qualitative figures and a per-source sample-detail
formatter. Keeps training-time and eval-time metrics coherent by construction.
"""
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (DEVICE, hard_dice, soft_dice, cosine_sim, topk_overlap,
                    downsample_mask, log_summary)

# <-- paste _overlay_ax, _heatmap_ax, save_figure verbatim from eval.py:290-356 here -->
```

- [ ] **Step 2: Add `_fmt_transforms` (verbatim from train.py) and `SAMPLE_COLS`**

Copy `_fmt_transforms` **verbatim** from `experiments/2d/train.py:209-224`, then add the new columns constant (the old train.py `SAMPLE_COLS` is replaced by this one):

```python
SAMPLE_COLS = ["epoch", "dataset", "sample_idx", "label",
               "dice", "dice_ds", "dice_ds_soft", "detail"]
```

- [ ] **Step 3: Add the adaptive `_sample_detail` formatter**

```python
def _sample_detail(meta: dict | None) -> str:
    """One compact string describing a sample, adapting to the data source.

    omniSynth meta -> "alphabet/class mode=<m> cells=<...> tf=<...>";
    controlSynth meta -> "<morphology> task=<id>"; anything else (e.g. medsegbench,
    or missing meta) -> "". Keeps the wandb sample table's columns fixed across sources.
    """
    if not meta:
        return ""
    if "alphabet" in meta:  # omniSynth
        return (f"{meta.get('alphabet')}/{meta.get('class_id')} "
                f"mode={meta.get('target_mode', '')} "
                f"cells={meta.get('target_cells', [])} "
                f"tf={_fmt_transforms(meta.get('target_transforms'))}")
    if "morphology" in meta:  # controlSynth
        return f"{meta.get('morphology')} task={int(meta.get('task_id', -1))}"
    return ""
```

- [ ] **Step 4: DRY `eval.py` — import the figure helpers instead of duplicating**

In `experiments/2d/eval.py`, delete the `_overlay_ax`, `_heatmap_ax`, and `save_figure`
definitions (currently `:290-356`, under the `# ── Visualisation` header) and add an
import near the other local imports (`from common import ...` at `eval.py:51`):
```python
from evaluate import save_figure  # _overlay_ax/_heatmap_ax used only inside save_figure
```
`eval.py` calls only `save_figure` (its `_overlay_ax`/`_heatmap_ax` were private helpers of
it), so this single import suffices. Leave everything else in `eval.py` unchanged.

- [ ] **Step 5: Verify imports + `_sample_detail` behavior + eval.py still imports**

Run:
```bash
cd /home/dpxuser/dev/patch_icl && .venv/bin/python - <<'EOF'
import sys; from pathlib import Path
sys.path.insert(0, "experiments/2d"); sys.path.insert(0, ".")
from evaluate import _sample_detail, SAMPLE_COLS, save_figure
assert SAMPLE_COLS[-1] == "detail" and len(SAMPLE_COLS) == 8
assert _sample_detail(None) == ""
assert _sample_detail({}) == ""
omni = {"alphabet": "Latin", "class_id": 3, "target_mode": "aug",
        "target_cells": [(0, 1)], "target_transforms": None}
d = _sample_detail(omni); assert d.startswith("Latin/3 mode=aug"), d
ctrl = _sample_detail({"morphology": "blob", "task_id": 7}); assert ctrl == "blob task=7", ctrl
assert _sample_detail({"dataset": "abdomenus"}) == ""
print("OK Task1:", d, "|", ctrl)
import importlib, eval as _evalmod  # eval.py must still import after the DRY edit
importlib.reload(_evalmod)
assert hasattr(_evalmod, "save_figure"), "eval.py should re-export save_figure via import"
print("OK eval.py imports save_figure from evaluate")
EOF
```
Expected: `OK Task1: Latin/3 mode=aug cells=[(0, 1)] tf=... | blob task=7` then `OK eval.py imports save_figure from evaluate`.

- [ ] **Step 6: Checkpoint** — imports resolve, `_sample_detail` covers all three source shapes, `eval.py` still imports cleanly (no duplication). Do not commit.

---

## Task 2: Implement shared `validate()` in `evaluate.py`

**Files:**
- Modify: `experiments/2d/evaluate.py`

**Interfaces:**
- Consumes: `common.{DEVICE, hard_dice, soft_dice, cosine_sim, topk_overlap, downsample_mask, log_summary}`; `save_figure`, `_sample_detail`, `SAMPLE_COLS` from Task 1; the model interface from Global Constraints.
- Produces:
  ```python
  def validate(model, loader, *, topk_k=16, epoch=0,
               figures=None,        # None | {"out_dir": Path, "max_figures": int, "to_wandb": bool}
               patch_csv=None,      # None | str  (low-res models only; native -> skipped)
               synth_csv=None,      # None | str  (controlSynth meta only)
               compute_flops=False  # measure once on the first batch
  ) -> tuple[dict, "wandb.Table", float | None]:  # (summary, sample_table, flops)
  ```
  `summary` keys: `dice/{mean,macro,dataset/*,class/*}`, `dice_ds/*`, `dice_ds_soft/*`, `cossim/*` + `top{topk_k}/*` (only when non-native rows exist), `time/inference_ms`, and `flops_giga` (only when measured).

- [ ] **Step 1: Add the res-matching helpers**

```python
def _target_like(lbl: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    """Avg-pool the (B,1,H,W) GT to the logit's spatial size (no-op when equal)."""
    if lbl.shape[-2:] == logit.shape[-2:]:
        return lbl
    return F.adaptive_avg_pool2d(lbl, logit.shape[-2:])


def _upsample_to(x: torch.Tensor, size) -> torch.Tensor:
    """Bilinear-resize (B,1,h,w) -> (B,1,*size); no-op when already at `size`."""
    return (x if x.shape[-2:] == tuple(size)
            else F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False))
```

- [ ] **Step 2: Write `validate()` — the loop**

```python
@torch.no_grad()
def validate(model, loader, *, topk_k=16, epoch=0, figures=None,
             patch_csv=None, synth_csv=None, compute_flops=False):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    hard_ds, hard_lab = defaultdict(list), defaultdict(list)   # native hard dice
    dsh_ds,  dsh_lab  = defaultdict(list), defaultdict(list)   # low-res hard dice_ds
    soft_ds, soft_lab = defaultdict(list), defaultdict(list)   # low-res soft dice_ds_soft
    cos_ds,  cos_lab  = defaultdict(list), defaultdict(list)   # populated only when not native
    topk_ds, topk_lab = defaultdict(list), defaultdict(list)
    table = wandb.Table(columns=SAMPLE_COLS)
    inf_times, flops, saved = [], None, set()
    patch_rows = [] if patch_csv else None
    synth_rows = [] if synth_csv else None
    max_fig = int(figures["max_figures"]) if figures else 0

    for batch in tqdm(loader, desc="val", leave=False):
        if batch is None:
            continue
        img  = batch["image"].to(DEVICE, non_blocking=True)
        lbl  = batch["label"].to(DEVICE, non_blocking=True).float()
        cin  = batch["context_in"].to(DEVICE, non_blocking=True)
        cout = batch["context_out"].to(DEVICE, non_blocking=True)
        B, _, H, W = img.shape
        K = cin.shape[1]

        ac = torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                            enabled=DEVICE.type == "cuda")
        t0 = time.perf_counter()
        if compute_flops and flops is None:
            with FlopCounterMode(display=False) as fc, ac:
                out = model(img, context_in=cin, context_out=cout, mode="val")
            flops = fc.get_total_flops()
        else:
            with ac:
                out = model(img, context_in=cin, context_out=cout, mode="val")
        logit = out["final_logit"].float()
        inf_times.append((time.perf_counter() - t0) / B)

        target   = _target_like(lbl, logit)             # (B,1,hp,wp) soft pooled GT
        prob     = torch.sigmoid(logit)                 # (B,1,hp,wp)
        prob_nat = _upsample_to(prob, lbl.shape[-2:])   # (B,1,H,W)
        native   = logit.shape[-2:] == lbl.shape[-2:]
        metas = batch.get("meta")

        for b in range(B):
            ds  = batch["dataset"][b]
            lv  = int(batch["label_value"][b])
            si  = int(batch["sample_idx"][b])
            key = f"{ds}/label_{lv}"

            h = hard_dice(prob_nat[b, 0], lbl[b, 0])     # native hard dice
            hard_ds[ds].append(h); hard_lab[key].append(h)
            if not native:
                dh = hard_dice(prob[b, 0], (target[b, 0] >= 0.5).float())
                s  = soft_dice(prob[b, 0], target[b, 0])
                c  = cosine_sim(prob[b, 0], target[b, 0])
                t  = topk_overlap(prob[b, 0], target[b, 0], topk_k)
                cos_ds[ds].append(c); cos_lab[key].append(c)
                topk_ds[ds].append(t); topk_lab[key].append(t)
            else:
                dh = s = float("nan")                    # native: no coarse grid
            dsh_ds[ds].append(dh); dsh_lab[key].append(dh)
            soft_ds[ds].append(s); soft_lab[key].append(s)

            detail = _sample_detail(metas[b]) if metas is not None else ""
            table.add_data(epoch, ds, si, lv, h, dh, s, detail)

            # ── gated: qualitative figure (one per dataset/label) ──
            fig_key = (ds, lv)
            if figures and fig_key not in saved and len(saved) < max_fig:
                saved.add(fig_key)
                fig_path = Path(figures["out_dir"]) / f"{ds}_l{lv}.png"
                low = None if native else prob[b, 0].cpu().numpy()
                glow = None if native else target[b, 0].cpu().numpy()
                save_figure(
                    tgt_image=img[b, 0].cpu().numpy(), tgt_gt=lbl[b, 0].cpu().numpy(),
                    pred_native=prob_nat[b, 0].cpu().numpy(),
                    ctx_images=[cin[b, k, 0].cpu().numpy() for k in range(K)],
                    ctx_gts=[cout[b, k, 0].cpu().numpy() for k in range(K)],
                    out_path=fig_path,
                    title=f"{ds} label={lv} sample={si} dice={h:.3f}",
                    pred_lowres=low, gt_lowres=glow)
                if figures.get("to_wandb"):
                    wandb.log({f"figures/{ds}/label_{lv}": wandb.Image(str(fig_path))})

            # ── gated: per-low-res-patch CSV (only meaningful when not native) ──
            if patch_rows is not None and not native:
                pp = prob[b, 0].cpu().numpy(); gp = target[b, 0].cpu().numpy()
                gt_size = float((lbl[b, 0] > 0).sum())
                ctx_d = [hard_dice(lbl[b, 0], cout[b, k, 0]) for k in range(K)]
                cd = float(np.nanmean(ctx_d)) if ctx_d else float("nan")
                for i in range(pp.shape[0]):
                    for j in range(pp.shape[1]):
                        patch_rows.append((ds, lv, si, i, j, float(pp[i, j]),
                                           float(gp[i, j]), float(pp[i, j] - gp[i, j]),
                                           gt_size, cd))

            # ── gated: per-element controlSynth params CSV ──
            if synth_rows is not None and metas is not None and "morphology" in (metas[b] or {}):
                m = metas[b]
                row = {"dataset": ds, "sample_idx": si, "label_value": lv,
                       "dice_native": h, "dice_ds": dh,
                       "morphology": m["morphology"], "task_id": int(m["task_id"]),
                       "subject_index": int(m.get("subject_index", -1)),
                       "fg_frac": float((lbl[b, 0] > 0).float().mean())}
                row.update({k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in m.get("difficulty", {}).items()})
                synth_rows.append(row)
```

- [ ] **Step 3: Write `validate()` — aggregate, write CSVs, return**

Append to the end of `validate()` (same indentation as the `for batch` loop):

```python
    extra = {"time/inference_ms": (float(np.mean(inf_times)) * 1000
                                   if inf_times else float("nan"))}
    if flops is not None:
        extra["flops_giga"] = flops / 1e9

    summary = {}
    summary.update(log_summary(hard_ds, hard_lab, prefix="dice",
                               metric_label="native", extra=extra))
    summary.update(log_summary(dsh_ds, dsh_lab, prefix="dice_ds",
                               metric_label="downsampled"))
    summary.update(log_summary(soft_ds, soft_lab, prefix="dice_ds_soft",
                               metric_label="low-res soft"))
    if cos_ds:   # populated only when some batch was non-native
        summary.update(log_summary(cos_ds, cos_lab, prefix="cossim",
                                   metric_label="cos sim"))
        summary.update(log_summary(topk_ds, topk_lab, prefix=f"top{topk_k}",
                                   metric_label=f"top{topk_k}"))

    if patch_rows is not None:
        p = Path(patch_csv); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "label_value", "sample_idx", "patch_i", "patch_j",
                        "pred", "gt", "error", "gt_size", "ctx_dice"])
            w.writerows(patch_rows)
        print(f"Wrote {len(patch_rows)} patch rows to {p}")
    if synth_rows:
        p = Path(synth_csv); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(synth_rows[0].keys()))
            w.writeheader(); w.writerows(synth_rows)
        print(f"Wrote {len(synth_rows)} synth rows to {p}")

    return summary, table, flops
```

- [ ] **Step 4: Verify end-to-end on a tiny omniSynth loader (CPU ok)**

This builds a fresh (untrained) PatchSetCNN via `build_model`, runs `validate` on a few omniSynth batches, and asserts the low-res metric families and flops are present.

```bash
cd /home/dpxuser/dev/patch_icl && .venv/bin/python - <<'EOF'
import sys; from pathlib import Path
sys.path.insert(0, "experiments/2d"); sys.path.insert(0, ".")
import wandb; wandb.init(mode="disabled")
from omegaconf import OmegaConf
from common import build_loader
from train import build_model
from evaluate import validate

cfg = OmegaConf.create({
  "model": "patchset_cnn",
  "data": {"source": "omnisynth", "image_size": 64, "context_size": 2,
           "split": "val", "dataset": None, "max_train_samples": None},
  # enc_dims length must be log2(image_size/resolution)+1 = log2(64/16)+1 = 3.
  "arch": {"resolution": 16, "enc_dims": [32,32,32], "e": 64, "h": 128, "l": 1,
           "a": 2, "thinking_rows": 1, "residual_decay": 0.95,
           "query_self_attn": False, "context_id_embed": False, "max_context": 16},
  "eval": {"batch_size": 4, "workers": 0},
  "synth": {"diversity": {"master_seed": 0, "train_zip": "images_background.zip",
              "eval_zip": "images_evaluation.zip", "val_test_split": 1},
            "scene": {"grid": 4, "k_min": 1, "k_max": 1, "cell_margin": 0.1,
              "target_mode": "aug", "aug_rotate": 20.0, "aug_scale": 0.2,
              "aug_translate": 0.2, "p_copy": 0.0, "n_copy": 0},
            "sampling": {"epoch_length": 8, "eval_subjects_per_task": 4,
              "eval_seed_namespace": 0}},
  "paths": {"omniglot": "/home/dpxuser/repos/omniglot/python"},
})
model, name, meta = build_model(cfg)
loader = build_loader(cfg)
summary, table, flops = validate(model, loader, topk_k=8, epoch=0, compute_flops=True)
print("keys:", sorted(k for k in summary if k.endswith("/mean")))
assert "dice/mean" in summary and "dice_ds/mean" in summary
assert "dice_ds_soft/mean" in summary and "cossim/mean" in summary and "top8/mean" in summary
assert "flops_giga" in summary and summary["flops_giga"] > 0
assert len(table.data) > 0
print("OK Task2  flops_giga=%.3f  rows=%d" % (summary["flops_giga"], len(table.data)))
EOF
```
Expected: prints `keys: [...]` including `cossim/mean`, `dice/mean`, `dice_ds/mean`, `dice_ds_soft/mean`, `top8/mean`, then `OK Task2 flops_giga=… rows=…`.

- [ ] **Step 5: Checkpoint** — low-res model produces all five metric families + flops + table rows. Do not commit.

---

## Task 3: Migrate `train.py` to the shared `validate()`

**Files:**
- Modify: `experiments/2d/train.py` — delete local `validate()` (currently `train.py:233`) and the helpers exclusive to it (`_fmt_transforms` at `:211`, `SAMPLE_COLS` at `:228`). Delete the local `_target_like` (`:142`) / `_upsample_to` (`:52`) and **import them from `evaluate` instead** — they are ALSO used by `train_epoch` (`:171`, `:188`), so they cannot simply be removed. Keep `_soft_sum`/`_topk_sum`/`_hard_sum` (train-accuracy monitors used by `train_epoch`). Update the call site (`train.py:~343-390`).

**Interfaces:**
- Consumes: `validate`, `_target_like`, `_upsample_to` from Task 2's `evaluate.py`.

- [ ] **Step 1: Confirm which helpers `train_epoch` still needs**

Run:
```bash
cd /home/dpxuser/dev/patch_icl
grep -n '_target_like\|_upsample_to\|_soft_sum\|_topk_sum\|_hard_sum\|_fmt_transforms\|SAMPLE_COLS\|def validate' experiments/2d/train.py
```
Expected: `_target_like`/`_upsample_to` appear in BOTH `train_epoch` (~171/188) and `validate` (~263/265) → move to `evaluate.py` and import (do NOT orphan them). `_fmt_transforms`/`SAMPLE_COLS`/`validate` appear only in the validate path → delete. `_soft_sum`/`_topk_sum`/`_hard_sum` are train monitors → keep.

- [ ] **Step 2: Update imports; remove `validate`, `_fmt_transforms`, `SAMPLE_COLS`, and the local `_target_like`/`_upsample_to`**

At the top of `train.py`, alongside the existing `from common import (...)`, add:
```python
from evaluate import validate, _target_like, _upsample_to
```
Then: delete the `def validate(...)` block, the `_fmt_transforms` def, the `SAMPLE_COLS` assignment, and the local `def _target_like` / `def _upsample_to` defs (now imported). Keep `_soft_sum`/`_topk_sum`/`_hard_sum`. `train_epoch` keeps calling `_target_like`/`_upsample_to` unchanged — they now resolve to the imported versions.

- [ ] **Step 3: Update the call site to the new return contract**

Find the block in `main()` that calls the old `validate` and builds `summary` (around `train.py:343-390`). Replace the unpack + inline `log_summary` calls with:

```python
            summary, sample_table, _ = validate(
                model, val_loader, topk_k=topk_k, epoch=epoch,
                compute_flops=(epoch == 0))
            metric = "cossim" if "cossim/mean" in summary else "dice"
            mean_dice = summary.get(f"{metric}/mean", float("nan"))
            log.update(summary)
            log["val/samples"] = sample_table
```

Leave the rest of the block (the `tqdm.write` progress line, the `if mean_dice > best_dice:` checkpoint save with `**ckpt_meta`) unchanged — it already reads `summary.get(...)` keys that `validate` still emits (`dice/mean`, `top{topk_k}/mean`, `dice_ds_soft/mean`, `dice/mean`).

- [ ] **Step 4: Verify a 1-epoch debug training run completes and logs the metric keys**

Run (uses the training env; `.venv311` if that is yours, else `.venv`):
```bash
cd /home/dpxuser/dev/patch_icl && .venv/bin/python experiments/2d/train.py \
  --config-name patchset_cnn_train \
  train.epochs=1 train.batch_size=4 train.workers=0 \
  data.image_size=64 arch.resolution=8 \
  synth.sampling.epoch_length=16 synth.sampling.eval_subjects_per_task=4 \
  wandb.enabled=false 2>&1 | tail -30
```
Expected: run completes without error; the per-epoch line prints `val cossim=… top8=… ds_soft=… dice=…` and `Done. Best val cossim=…  Checkpoint: …`. (patchset_cnn is low-res → checkpoint metric is `cossim`, matching pre-migration behavior.)

- [ ] **Step 5: Verify universeg (native) still selects the `dice` metric**

Run:
```bash
cd /home/dpxuser/dev/patch_icl && .venv/bin/python experiments/2d/train.py \
  --config-name universeg_train \
  train.epochs=1 train.batch_size=4 train.workers=0 \
  data.image_size=64 \
  synth.sampling.epoch_length=16 synth.sampling.eval_subjects_per_task=4 \
  wandb.enabled=false 2>&1 | tail -20
```
Expected: completes; progress line shows `val dice=…` and `Done. Best val dice=…` (native model → `dice`, not `cossim`). Confirms the checkpoint-selection policy is unchanged.

- [ ] **Step 6: Checkpoint** — both models train one epoch; low-res picks `cossim`, native picks `dice`. Do not commit.

---

## Task 4: New wrapper `eval_incontext.py`

**Files:**
- Create: `experiments/2d/eval_incontext.py`
- Reference (config): `configs/experiment/2d/eval_base.yaml` (reused as-is; has `data.split`, `eval.checkpoint`, `eval.save_figures`/`max_figures`/`figures_to_wandb`, `eval.patch_csv`, `eval.synth_csv`, `wandb.*`).

**Interfaces:**
- Consumes: `validate()` from Task 2; `common.build_loader`; the checkpoint format + model interface from Global Constraints.

- [ ] **Step 1: Write the wrapper**

Create `experiments/2d/eval_incontext.py`:

```python
"""Focused 2D eval for the two in-context models saved by train.py:
`universeg` (native H×W logit) and `patchset_cnn` (low-res R×R logit). Dispatches
on the checkpoint's `model_name`, rebuilds the one model, and runs the SHARED
validate() (evaluate.py) — the same loop/metrics used during training — with
figures + CSVs + FLOPs enabled.

    python experiments/2d/eval_incontext.py eval.checkpoint=results/2d/.../best.pt
    python experiments/2d/eval_incontext.py eval.checkpoint=<p> data.source=omnisynth data.split=test
"""
import datetime
import random
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, build_loader
from evaluate import validate


def _load_model(ckpt: dict):
    """Rebuild the trained model from a train.py checkpoint (dispatch on model_name)."""
    name = ckpt.get("model_name")
    img = ckpt["image_size"]
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    if name == "universeg":
        from src.models.universeg_baseline import UniverSegBaseline
        model = UniverSegBaseline(pretrained=True, input_size=img).to(DEVICE)
    elif name == "patchset_cnn":
        from src.models.patchset_cnn import PatchSetCNN
        arch = ckpt.get("arch")
        if not arch:
            raise ValueError(
                "patchset_cnn checkpoint has no 'arch' block — it predates full-arch "
                "storage. Retrain (or re-save) so the checkpoint is self-contained.")
        model = PatchSetCNN(image_size=img, **arch).to(DEVICE)
    else:
        raise ValueError(f"unknown model_name {name!r} (universeg | patchset_cnn)")
    model.load_state_dict(state)
    return model.eval(), name


@hydra.main(config_path="../../configs/experiment/2d", config_name="eval_base",
            version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.eval.seed); torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if not cfg.eval.get("checkpoint"):
        raise ValueError("set eval.checkpoint=<path/to/best.pt>")
    ckpt = torch.load(cfg.eval.checkpoint, map_location="cpu", weights_only=False)
    model, model_name = _load_model(ckpt)
    # Serve images at the size the checkpoint was trained on.
    with open_dict(cfg):
        cfg.data.image_size = ckpt["image_size"]
        cfg.data.context_size = ckpt.get("context_size", cfg.data.context_size)
    print(f"Loaded {model_name} (size={ckpt['image_size']}, ctx={cfg.data.context_size}) "
          f"from {cfg.eval.checkpoint}")

    loader = build_loader(cfg)
    # wandb.project=null (or wandb.enabled=false) disables logging, per repo convention.
    wb_on = bool(cfg.wandb.get("project")) and cfg.wandb.get("enabled", True)
    run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name,
                     mode="online" if wb_on else "disabled",
                     config={"model": model_name, "checkpoint": str(cfg.eval.checkpoint),
                             "source": cfg.data.get("source"), "split": cfg.data.split,
                             "image_size": ckpt["image_size"],
                             "context_size": cfg.data.context_size})
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or model_name
    out_dir = Path(cfg.eval.out_dir) / f"{datetime.date.today():%Y-%m-%d}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = ({"out_dir": out_dir, "max_figures": int(cfg.eval.get("max_figures", 200)),
                "to_wandb": bool(cfg.eval.get("figures_to_wandb", False))}
               if cfg.eval.get("save_figures", False) else None)
    summary, table, flops = validate(
        model, loader, topk_k=int(cfg.eval.get("topk_k", 16)), epoch=0,
        figures=figures, patch_csv=cfg.eval.get("patch_csv"),
        synth_csv=(cfg.eval.get("synth_csv") if cfg.data.get("source") == "synthetic" else None),
        compute_flops=True)
    summary["samples"] = table
    wandb.log(summary)
    print(f"dice/mean={summary.get('dice/mean'):.4f}  "
          f"dice_ds/mean={summary.get('dice_ds/mean', float('nan')):.4f}  "
          f"flops={summary.get('flops_giga', float('nan')):.2f}G  out={out_dir}")
    run.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the fail-loud path on an arch-less patchset_cnn checkpoint**

Run (uses an existing pre-arch checkpoint):
```bash
cd /home/dpxuser/dev/patch_icl && .venv/bin/python experiments/2d/eval_incontext.py \
  eval.checkpoint=/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/2d_train/2026-07-02_vibrant-frog-90/best.pt \
  wandb.project=null 2>&1 | tail -5
```
Expected: raises `ValueError: patchset_cnn checkpoint has no 'arch' block …`. (Confirms old checkpoints fail clearly rather than silently mis-building.)

- [ ] **Step 3: Verify the universeg native path end-to-end**

Run (existing universeg checkpoint; omniSynth val split):
```bash
cd /home/dpxuser/dev/patch_icl && .venv/bin/python experiments/2d/eval_incontext.py \
  eval.checkpoint=/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/2d_train/2026-07-03_fallen-leaf-95/best.pt \
  data.source=omnisynth data.split=val eval.save_figures=false \
  eval.batch_size=8 eval.workers=0 wandb.project=null 2>&1 | tail -8
```
Expected: prints `Loaded universeg …` then `dice/mean=… dice_ds/mean=nan flops=…G out=…` (native → `dice_ds` is NaN; no crash).

- [ ] **Step 4: Verify the patchset_cnn low-res path on a fresh (arch-ful) checkpoint**

First train a tiny arch-ful checkpoint, then eval it with figures on:
```bash
cd /home/dpxuser/dev/patch_icl
.venv/bin/python experiments/2d/train.py --config-name patchset_cnn_train \
  train.epochs=1 train.batch_size=4 train.workers=0 data.image_size=64 arch.resolution=8 \
  synth.sampling.epoch_length=16 synth.sampling.eval_subjects_per_task=4 \
  eval.out_dir=/tmp/pc_ckpt wandb.enabled=false 2>&1 | tail -3
CKPT=$(find /tmp/pc_ckpt -name best.pt | head -1)
.venv/bin/python experiments/2d/eval_incontext.py eval.checkpoint=$CKPT \
  data.source=omnisynth data.split=val eval.save_figures=true eval.max_figures=3 \
  eval.batch_size=8 eval.workers=0 eval.out_dir=/tmp/pc_eval wandb.project=null 2>&1 | tail -8
find /tmp/pc_eval -name '*.png' | head
```
Expected: eval prints `Loaded patchset_cnn …` then `dice/mean=… dice_ds/mean=<real number> flops=…G`, and at least one `.png` figure exists under `/tmp/pc_eval`.

- [ ] **Step 5: Checkpoint** — fail-loud works; universeg gives native-only metrics; patchset_cnn gives low-res metrics + figures. Do not commit.

---

## Task 5: Documentation

**Files:**
- Modify: `docs/logs.md`

- [ ] **Step 1: Append a log entry**

Add to the end of `docs/logs.md`:
```markdown
## 2026-07-03 — Shared validate() + focused eval_incontext.py (universeg / patchset_cnn)
- Extracted train.py's per-epoch validate() into experiments/2d/evaluate.py as the single
  shared eval loop (metrics + adaptive sample table + gated figures/CSVs/FLOPs). train.py
  now imports it; behavior unchanged (low-res models checkpoint on cossim, native on dice).
- New experiments/2d/eval_incontext.py: thin Hydra wrapper (reuses eval_base.yaml) that
  loads a train.py checkpoint, dispatches on model_name, rebuilds universeg or patchset_cnn
  (via **ckpt["arch"]), and runs the shared validate() with figures/CSVs/FLOPs on. Fails
  loudly on pre-arch patchset_cnn checkpoints. eval.py (5-backend) now imports the shared
  save_figure from evaluate.py (its only change); its dispatch logic is untouched.
```

- [ ] **Step 2: Checkpoint** — log entry added. Do not commit (user handles VCS).

---

## Notes for the implementer

- Run every command from the repo root `/home/dpxuser/dev/patch_icl`.
- If a smoke run is slow on CPU, it will still complete (tiny sizes chosen); prefer the CUDA env for the training smoke runs.
- The ONLY change to `experiments/2d/eval.py` is the figure-helper DRY edit in Task 1 Step 4 (delete 3 defs, add 1 import). Do NOT modify `configs/experiment/2d/eval_base.yaml`.
- The old checkpoints under NFS lack `arch`; only freshly trained patchset_cnn checkpoints are self-contained (that is expected and is what Task 4 Step 2 verifies).

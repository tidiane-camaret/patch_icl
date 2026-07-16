# Scatter Refinement Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a scatter-specific qualitative figure to 2D eval so `eval_incontext.py` runs on a `refine_mode="scatter"` checkpoint show where cells were sampled (tier-colored) and the coarse→fused prediction.

**Architecture:** Thread the sampler's `is_core`/`is_fg_core` tier flags (already computed, currently discarded) through `_refine_scatter`'s output dict; add a `save_scatter_figure` 2×3 panel to `evaluate.py` and a scatter branch in `validate()`'s figure block keyed on `refine_idx`. Additive only — trainer loss, eval metrics, and the bbox path are untouched.

**Tech Stack:** PyTorch, matplotlib (Agg backend), pytest. Spec: `docs/superpowers/specs/2026-07-13-scatter-figure-design.md`.

## Global Constraints

- Commit per task on the current `patchset-refine` branch. End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Run tests with `.venv311/bin/python -m pytest <path> -v` from repo root (torch 2.5.1 + pytest installed there; do NOT use bare `python`).
- Additive change: the new output-dict keys and figure branch must not alter the scatter loss/metrics or the bbox refine path. Existing suites `tests/test_patchset_scatter.py`, `tests/test_refine_geometry_scatter.py`, `tests/test_patchset_cnn_refine.py` must stay green.
- Tier partition (non-overlapping, matches `plot_sampling.py`): `fg_core = is_fg` → orange; `boundary_core = is_core & ~is_fg` → red; `neighbor = ~is_core` → cyan.
- Cell→pixel: for a cell `(i,j)` on the `Rf×Rf` grid, center is `y=(i+0.5)*H/Rf`, `x=(j+0.5)*W/Rf`.

---

### Task 1: Thread tier flags through `_refine_scatter`

**Files:**
- Modify: `src/models/patchset_cnn.py` (`_refine_scatter`: the two `sample_patches` calls + the return dict, ~lines 424-449)
- Test: `tests/test_patchset_scatter.py` (append)

**Interfaces:**
- Consumes: `sample_patches(...) -> (idx, is_core, is_fg_core)` (from `src.models.scatter_sampling`, already imported).
- Produces: `_refine_scatter` output dict gains 5 keys: `refine_is_core (B,M) bool`, `refine_is_fg (B,M) bool`, `refine_sup_idx (B,K,M) long`, `refine_sup_is_core (B,K,M) bool`, `refine_sup_is_fg (B,K,M) bool`. Consumed by Task 2's `validate()` branch.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patchset_scatter.py`:
```python
def test_scatter_returns_tier_keys():
    m = _scatter_model((8, 16), H=32)          # K=2, M=20, Rf=16
    img, cin, cout = _batch(B=2, K=2, H=32)
    out = m(img, context_in=cin, context_out=cout)
    B, K, M, Rf = 2, 2, 20, 16
    assert out["refine_is_core"].shape == (B, M)
    assert out["refine_is_fg"].shape == (B, M)
    assert out["refine_is_core"].dtype == torch.bool
    assert out["refine_is_fg"].dtype == torch.bool
    assert out["refine_sup_idx"].shape == (B, K, M)
    assert out["refine_sup_is_core"].shape == (B, K, M)
    assert out["refine_sup_is_fg"].shape == (B, K, M)
    assert out["refine_sup_idx"].dtype == torch.long
    assert int(out["refine_sup_idx"].max()) < Rf * Rf and int(out["refine_sup_idx"].min()) >= 0
    # fg-core is a subset of core (partition invariant)
    assert bool((out["refine_is_fg"] & ~out["refine_is_core"]).any()) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_patchset_scatter.py::test_scatter_returns_tier_keys -v`
Expected: FAIL — `KeyError: 'refine_is_core'` (key not in output dict yet).

- [ ] **Step 3: Write minimal implementation**

In `src/models/patchset_cnn.py::_refine_scatter`, capture the query tier flags. Replace:
```python
        qidx, _, _ = sample_patches(q_map, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
                                    temperature=s["temperature"], stochastic=stoch,
                                    n_fg_core=s["n_fg_core"], n_boundary_core=s["n_boundary_core"])
```
with:
```python
        qidx, q_is_core, q_is_fg = sample_patches(
            q_map, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
            temperature=s["temperature"], stochastic=stoch,
            n_fg_core=s["n_fg_core"], n_boundary_core=s["n_boundary_core"])
```
Capture the support tier flags. Replace:
```python
        sidx, _, _ = sample_patches(ctx_frac, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
                                    temperature=s["temperature"], stochastic=stoch,
                                    n_fg_core=s["n_fg_core_ctx"], n_boundary_core=s["n_boundary_core"])
```
with:
```python
        sidx, s_is_core, s_is_fg = sample_patches(
            ctx_frac, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
            temperature=s["temperature"], stochastic=stoch,
            n_fg_core=s["n_fg_core_ctx"], n_boundary_core=s["n_boundary_core"])
```
Extend the return dict. Replace:
```python
        return {"final_logit": coarse, "refine_logit": refine_logit, "refine_idx": qidx,
                "refine_grid_res": Rf, "resolutions": self.resolutions}
```
with:
```python
        return {"final_logit": coarse, "refine_logit": refine_logit, "refine_idx": qidx,
                "refine_grid_res": Rf, "resolutions": self.resolutions,
                # tier flags for the scatter qualitative figure (unused by loss/metrics)
                "refine_is_core": q_is_core, "refine_is_fg": q_is_fg,
                "refine_sup_idx": sidx.reshape(B, K, M),
                "refine_sup_is_core": s_is_core.reshape(B, K, M),
                "refine_sup_is_fg": s_is_fg.reshape(B, K, M)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv311/bin/python -m pytest tests/test_patchset_scatter.py tests/test_refine_geometry_scatter.py tests/test_patchset_cnn_refine.py -v`
Expected: PASS — the new tier-key test passes; all existing scatter forward/backward/eval + geometry + bbox tests stay green (additive change).

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset_cnn.py tests/test_patchset_scatter.py
git commit -m "feat(patchset): surface sampler tier flags (is_core/is_fg) from _refine_scatter for viz

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `save_scatter_figure` + `validate()` scatter panel

**Files:**
- Modify: `experiments/2d/evaluate.py` (import ~line 24; add `_scatter_cells_ax` + `save_scatter_figure` near `save_refine_figure` ~line 147; add the `elif` branch in the `validate()` figure block ~line 472)
- Test: `tests/test_scatter_figure.py` (new)

**Interfaces:**
- Consumes: `_refine_scatter` output keys from Task 1 (`refine_idx`, `refine_is_core`, `refine_is_fg`, `refine_sup_idx`, `refine_sup_is_core`, `refine_sup_is_fg`, `refine_grid_res`); `rg["fused"]` from `refine_geometry`; `idx_to_ij` from `src.models.scatter_sampling`; existing `_refine_overlay_ax` helper.
- Produces: `save_scatter_figure(tgt_image, tgt_gt, coarse_pred, fused_pred, qry_ij, qry_is_core, qry_is_fg, ctx_image, ctx_gt, sup_ij, sup_is_core, sup_is_fg, grid_res, out_path, title="")` → writes a 2×3 PNG.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scatter_figure.py`:
```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import numpy as np
from pathlib import Path


def test_save_scatter_figure_writes_png(tmp_path):
    from evaluate import save_scatter_figure
    H, Rf, M = 32, 16, 10
    rng = np.random.default_rng(0)
    tgt_image = rng.random((H, H)).astype("float32")
    tgt_gt = (rng.random((H, H)) > 0.5).astype("float32")
    coarse = rng.random((H, H)).astype("float32")
    fused = rng.random((H, H)).astype("float32")
    q_ij = np.stack([rng.integers(0, Rf, M), rng.integers(0, Rf, M)], axis=-1)
    q_core = rng.random(M) > 0.5
    q_fg = q_core & (rng.random(M) > 0.5)          # fg is subset of core
    s_ij = np.stack([rng.integers(0, Rf, M), rng.integers(0, Rf, M)], axis=-1)
    s_core = rng.random(M) > 0.5
    s_fg = s_core & (rng.random(M) > 0.5)
    out = tmp_path / "scatter.png"
    save_scatter_figure(tgt_image, tgt_gt, coarse, fused,
                        q_ij, q_core, q_fg,
                        tgt_image, tgt_gt, s_ij, s_core, s_fg,
                        Rf, out, title="smoke")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python -m pytest tests/test_scatter_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_scatter_figure' from 'evaluate'`.

- [ ] **Step 3: Write minimal implementation**

In `experiments/2d/evaluate.py`, extend the scatter_sampling import (currently `from src.models.scatter_sampling import gather_grid, composite_predictions`):
```python
from src.models.scatter_sampling import gather_grid, composite_predictions, idx_to_ij
```

Add these two functions just after `save_refine_figure` (after ~line 147):
```python
def _scatter_cells_ax(ax, image, gt, ij, is_core, is_fg, grid_res, title):
    """Gray image + lime GT contour + sampled cells colored by tier (Rf grid -> image px).
    ij: (M,2) row/col on the grid_res grid; is_core/is_fg: (M,) bool. Tiers are a partition:
    fg-core (orange) subset of core; boundary-core (red) = core & ~fg; neighbor (cyan) = ~core."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if gt is not None and float(gt.max()) > 0:
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.0)
    scale = image.shape[0] / grid_res
    y = (ij[:, 0] + 0.5) * scale
    x = (ij[:, 1] + 0.5) * scale
    fg = is_fg.astype(bool)
    bcore = is_core.astype(bool) & ~fg
    neigh = ~is_core.astype(bool)
    ax.scatter(x[neigh], y[neigh], s=12, c="cyan", marker="s", edgecolors="none",
               label=f"neighbor ({int(neigh.sum())})")
    ax.scatter(x[bcore], y[bcore], s=12, c="red", marker="s", edgecolors="none",
               label=f"boundary ({int(bcore.sum())})")
    ax.scatter(x[fg], y[fg], s=12, c="orange", marker="s", edgecolors="none",
               label=f"fg-core ({int(fg.sum())})")
    ax.legend(loc="upper right", fontsize=5, framealpha=0.6)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_scatter_figure(tgt_image, tgt_gt, coarse_pred, fused_pred,
                        qry_ij, qry_is_core, qry_is_fg,
                        ctx_image, ctx_gt, sup_ij, sup_is_core, sup_is_fg,
                        grid_res, out_path, title=""):
    """2×3 scatter-refine panel. Row 0 (target): [GT + tier-colored query cells | coarse native
    pred | fused native pred]. Row 1: [ctx0 GT + tier-colored support cells | blank | blank].
    Cells live on the grid_res grid and are scaled to image pixels."""
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.5), squeeze=False)
    fig.subplots_adjust(hspace=0.35, wspace=0.05)
    _scatter_cells_ax(axes[0, 0], tgt_image, tgt_gt, qry_ij, qry_is_core, qry_is_fg,
                      grid_res, "Target + GT + sampled cells")
    _refine_overlay_ax(axes[0, 1], tgt_image, "Target + coarse pred", gt=tgt_gt, pred=coarse_pred)
    _refine_overlay_ax(axes[0, 2], tgt_image, "Target + fused pred", gt=tgt_gt, pred=fused_pred)
    _scatter_cells_ax(axes[1, 0], ctx_image, ctx_gt, sup_ij, sup_is_core, sup_is_fg,
                      grid_res, "Ctx0 + GT + support cells")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

In `validate()`'s figure block, add a scatter branch right after the bbox `if rg is not None and out.get("refine_origin") is not None:` block closes (i.e. after the block that logs `figures_refine/...`). Insert:
```python
                elif rg is not None and out.get("refine_idx") is not None:  # scatter panel
                    Rf_g = int(out["refine_grid_res"])
                    fig_path_scatter = Path(figures["out_dir"]) / f"{ds}_l{lv}_scatter.png"
                    q_ij = idx_to_ij(out["refine_idx"][b:b + 1], Rf_g)[0].cpu().numpy()      # (M,2)
                    s_ij = idx_to_ij(out["refine_sup_idx"][b, 0:1], Rf_g)[0].cpu().numpy()   # ctx0 (M,2)
                    save_scatter_figure(
                        tgt_image=img[b, 0].cpu().numpy(), tgt_gt=lbl[b, 0].cpu().numpy(),
                        coarse_pred=prob_nat[b, 0].cpu().numpy(),
                        fused_pred=rg["fused"][b, 0].cpu().numpy(),
                        qry_ij=q_ij, qry_is_core=out["refine_is_core"][b].cpu().numpy(),
                        qry_is_fg=out["refine_is_fg"][b].cpu().numpy(),
                        ctx_image=cin[b, 0, 0].cpu().numpy(), ctx_gt=cout[b, 0, 0].cpu().numpy(),
                        sup_ij=s_ij, sup_is_core=out["refine_sup_is_core"][b, 0].cpu().numpy(),
                        sup_is_fg=out["refine_sup_is_fg"][b, 0].cpu().numpy(),
                        grid_res=Rf_g, out_path=fig_path_scatter,
                        title=f"{ds} label={lv} sample={si} scatter")
                    if figures.get("to_wandb"):
                        wandb.log({f"figures_scatter/{ds}/label_{lv}":
                                   wandb.Image(str(fig_path_scatter))})
```
Note: the existing bbox block is an `if`; converting the following-line indentation so this becomes its `elif` sibling. Confirm the `elif` aligns with the `if rg is not None and out.get("refine_origin")...` at the same indentation level, and that the whole thing sits inside `if figures and fig_key not in saved and len(saved) < max_fig:`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv311/bin/python -m pytest tests/test_scatter_figure.py -v`
Expected: PASS — the PNG is written.
Also confirm nothing else broke and `evaluate.py` parses:
```bash
.venv311/bin/python -c "import ast; ast.parse(open('experiments/2d/evaluate.py').read())"
.venv311/bin/python -m pytest tests/test_refine_geometry_scatter.py tests/test_patchset_cnn_refine.py -v
```
Expected: no output from the ast check; both suites PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/2d/evaluate.py tests/test_scatter_figure.py
git commit -m "feat(eval): save_scatter_figure — 2x3 tier-colored scatter panel + validate() branch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Tier flags threaded through `_refine_scatter` (5 keys) → Task 1. ✓
- `save_scatter_figure` 2×3 panel (target sampling | coarse | fused; ctx0 support | blank | blank) → Task 2. ✓
- Tier-colored overlay (red/orange/cyan), cell→pixel mapping → Task 2 `_scatter_cells_ax`. ✓
- `validate()` scatter branch keyed on `refine_idx`, wandb `figures_scatter/...` → Task 2. ✓
- ctx0-only support panel (YAGNI) → Task 2 (uses `[b, 0]`). ✓
- Additive: loss/metrics/bbox untouched → both tasks re-run the metric + bbox suites. ✓
- Testing (key shapes + figure smoke) → Tasks 1 & 2. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full content.

**Type consistency:** `refine_is_core/is_fg (B,M)`, `refine_sup_idx/is_core/is_fg (B,K,M)` defined in Task 1 and consumed with matching indexing in Task 2 (`out["refine_is_core"][b]` → `(M,)`; `out["refine_sup_is_core"][b, 0]` → `(M,)`; `out["refine_sup_idx"][b, 0:1]` → `(1,M)` for `idx_to_ij`). `save_scatter_figure` signature identical in Task 2's Produces block, the smoke test, and the `validate()` call site.

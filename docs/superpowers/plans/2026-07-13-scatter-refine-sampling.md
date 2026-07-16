# Scatter Refinement Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an unconstrained "scatter" refinement mode to `PatchSetCNN` that samples individual grid cells anywhere the coarse prediction is uncertain/foreground (instead of a single bbox crop), refines them, and scatters the results back.

**Architecture:** New `refine_mode="scatter"` reuses the encode-once path (encode K+1 images once, pool to a fine grid `Rf`), samples M query cells from the coarse prediction and M support cells per context from the true mask, runs the existing set-attention core on that sampled set, and returns per-cell logits + indices. The refine loss and eval geometry gain a scatter branch keyed on the presence of `refine_idx`. The bbox modes (`reencode`/`encode_once`) are untouched.

**Tech Stack:** PyTorch, Hydra configs, pytest. Spec: `docs/superpowers/specs/2026-07-13-scatter-refine-sampling-design.md`.

## Global Constraints

- **Commit per task** on the current `patchset-refine` branch. Each task's final step commits its files with a clear message; end commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Single refine level only: `resolutions=[T, Rf]` (the constructor already asserts ≤2). For scatter, `Rf = resolutions[-1]` is the **fine token grid resolution**, not a crop size.
- Sampling is stochastic in train, deterministic in eval (`stochastic = self.training`).
- The `_attn`→`_attn_core` refactor MUST leave the coarse / single-level / bbox-refine outputs unchanged: `tests/test_patchset_cnn_refine.py` must stay green.
- Run tests with the activated env `.venv311`: `.venv311/bin/python -m pytest <path> -v` from repo root (repo root is on `sys.path` via `sys.path.insert(0, ".")` at the top of each test file). torch 2.5.1+cu124, pytest installed.
- Sampler defaults (from the coverage diagnostic): `n_total=256, tau=0.30, blur_sigma=1.0, floor=0.005, n_fg_core=64, n_fg_core_ctx=64, temperature=1.0, n_boundary_core=0`.

---

### Task 1: Scatter sampling module

**Files:**
- Create: `src/models/scatter_sampling.py`
- Test: `tests/test_scatter_sampling.py`

**Interfaces:**
- Produces:
  - `sample_patches(values, n_total, tau, blur_sigma, floor, grid_res, temperature=1.0, stochastic=True, n_fg_core=0, boundary_tier=True, n_boundary_core=0) -> (idx, is_core, is_fg_core)` each `(B, n_total)` long/bool.
  - `idx_to_ij(idx, grid_res) -> (B, M, 2)` long.
  - `gather_grid(x, idx)` — `(B,N,C)->(B,M,C)` or `(B,N)->(B,M)`.
  - `composite_predictions(coarse_flat, idx, vals) -> (B, N)` new tensor.
  - `gaussian_blur(x_flat, grid_res, sigma) -> (B, N)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scatter_sampling.py`:
```python
import sys; sys.path.insert(0, ".")
import torch
from src.models.scatter_sampling import (
    sample_patches, idx_to_ij, gather_grid, composite_predictions)


def test_sample_patches_shapes_and_budget():
    torch.manual_seed(0)
    B, R, M = 3, 16, 64
    values = torch.rand(B, R * R)
    idx, is_core, is_fg = sample_patches(values, M, tau=0.30, blur_sigma=1.0,
                                         floor=0.005, grid_res=R, n_fg_core=8)
    assert idx.shape == (B, M) and is_core.shape == (B, M) and is_fg.shape == (B, M)
    assert idx.min() >= 0 and idx.max() < R * R
    # indices unique per row (top-k over distinct cells)
    for b in range(B):
        assert idx[b].unique().numel() == M


def test_sample_patches_deterministic_when_seeded():
    R, M = 16, 32
    values = torch.rand(1, R * R)
    torch.manual_seed(7); a, _, _ = sample_patches(values, M, 0.3, 1.0, 0.005, R)
    torch.manual_seed(7); b, _, _ = sample_patches(values, M, 0.3, 1.0, 0.005, R)
    assert torch.equal(a, b)


def test_boundary_core_cap_limits_core_count():
    R, M = 16, 128
    # a smooth ramp so many cells fall in the tau band
    values = torch.linspace(0, 1, R * R).reshape(1, R * R)
    _, core_uncapped, _ = sample_patches(values, M, 0.45, 1.0, 0.005, R, stochastic=False)
    _, core_capped, _ = sample_patches(values, M, 0.45, 1.0, 0.005, R, stochastic=False,
                                       n_boundary_core=10)
    assert int(core_capped.sum()) <= int(core_uncapped.sum())
    assert int(core_capped.sum()) <= 10


def test_compact_blob_boundary_in_core():
    # a solid square blob → its fractional-boundary cells should land in the core tier
    R, M = 16, 96
    g = torch.zeros(R, R); g[4:12, 4:12] = 1.0
    values = g.reshape(1, R * R)
    idx, is_core, _ = sample_patches(values, M, 0.30, 1.0, 0.005, R, stochastic=False, n_fg_core=16)
    # every selected core cell that is fractional (0<v<1) — here the blob is binary so use fg cells
    sel_fg = gather_grid(values, idx)[is_core]
    assert (sel_fg >= 0.5).float().mean() > 0.5   # core is dominated by foreground


def test_gather_and_composite_roundtrip():
    B, R, M = 2, 8, 10
    coarse = torch.rand(B, R * R)
    idx = torch.stack([torch.randperm(R * R)[:M] for _ in range(B)])
    vals = torch.ones(B, M)
    out = composite_predictions(coarse, idx, vals)
    assert out.shape == coarse.shape
    assert torch.allclose(gather_grid(out, idx), vals)          # scattered cells overwritten
    assert not out.data_ptr() == coarse.data_ptr()              # new tensor


def test_idx_to_ij():
    idx = torch.tensor([[0, 1, 8, 9]])
    ij = idx_to_ij(idx, 8)
    assert torch.equal(ij, torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scatter_sampling.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'src.models.scatter_sampling'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/models/scatter_sampling.py`:
```python
"""Unconstrained scatter patch sampling for PatchSetCNN refinement.

Productionized copy of experiments/2d/multilevel/sampling.py (the capped variant from
plot_sampling.py). Selects a budget of individual grid cells via three priority tiers —
boundary core, a fixed foreground-core quota, and a blurred-proximity neighbor fill — then
gathers features/coords for them and scatters refined predictions back. Pure tensor ops.
"""

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_blur(x_flat: torch.Tensor, grid_res: int, sigma: float) -> torch.Tensor:
    """(B, N) -> (B, N) separable Gaussian blur on the grid_res x grid_res grid."""
    B, N = x_flat.shape
    x = x_flat.reshape(B, 1, grid_res, grid_res)
    k = int(2 * np.ceil(2 * sigma) + 1)
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    pad = k // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), g.view(1, 1, k, 1))
    return x.reshape(B, N)


def sample_patches(values: torch.Tensor, n_total: int, tau: float, blur_sigma: float,
                   floor: float, grid_res: int, temperature: float = 1.0,
                   stochastic: bool = True, n_fg_core: int = 0, boundary_tier: bool = True,
                   n_boundary_core: int = 0):
    """values: (B, N) in [0,1]. Returns (idx, is_core, is_fg_core), each (B, n_total).

    Three priority tiers combined into one score + a single top-k:
      1. boundary core: |value-0.5| < tau (ranked by closeness to 0.5); optionally capped to
         the n_boundary_core cells closest to 0.5. Disabled when boundary_tier=False.
      2. fg core: a fixed n_fg_core quota of value>=0.5 cells chosen uniformly at random.
    The remaining budget is a blurred proximity field over (core u fg_core) + uniform floor +
    Gumbel-top-k neighbor fill.
    """
    d = (values - 0.5).abs()
    core_b = (d < tau) if boundary_tier else torch.zeros_like(values, dtype=torch.bool)
    if boundary_tier and n_boundary_core > 0:
        masked_d = torch.where(core_b, d, torch.full_like(d, 2.0))          # non-core -> large
        keep = masked_d.topk(min(n_boundary_core, d.shape[1]), dim=1, largest=False).indices
        core_b = torch.zeros_like(core_b).scatter_(1, keep, True) & core_b   # real core only

    fg_core = torch.zeros_like(core_b)
    if n_fg_core > 0:
        fg_pool = (values >= 0.5) & ~core_b
        key = torch.where(fg_pool, torch.rand_like(values), values.new_full((), -1.0))
        take = key.topk(n_fg_core, dim=1).indices
        fg_core = torch.zeros_like(core_b).scatter_(1, take, True) & fg_pool  # guard: <n_fg_core fg

    g = gaussian_blur((core_b | fg_core).float(), grid_res, blur_sigma)
    w = g + floor
    if stochastic:
        u = torch.rand_like(w).clamp(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        neigh_score = (w + 1e-12).log() + temperature * gumbel
    else:
        neigh_score = (w + 1e-12).log()

    BIG_B, BIG_F = 2e4, 1e4                                                   # boundary > fg > neighbor
    score = torch.where(core_b, BIG_B - d, torch.where(fg_core, BIG_F, neigh_score))
    idx = score.topk(n_total, dim=1).indices
    is_fg_core = fg_core.gather(1, idx)
    is_core = (core_b | fg_core).gather(1, idx)
    return idx, is_core, is_fg_core


def idx_to_ij(idx: torch.Tensor, grid_res: int) -> torch.Tensor:
    """Flat cell index (B, M) -> (B, M, 2) row/col on a grid_res x grid_res grid (row-major)."""
    return torch.stack([torch.div(idx, grid_res, rounding_mode="floor"),
                        idx % grid_res], dim=-1)


def gather_grid(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along the cell axis. x: (B, N, C) -> (B, M, C), or x: (B, N) -> (B, M)."""
    if x.dim() == 3:
        C = x.shape[-1]
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    return torch.gather(x, 1, idx)


def composite_predictions(coarse_flat: torch.Tensor, idx: torch.Tensor,
                          vals: torch.Tensor) -> torch.Tensor:
    """(B,N) dense map + (B,M) indices + (B,M) values -> (B,N) NEW tensor with vals scattered in."""
    refined = coarse_flat.clone()
    refined.scatter_(1, idx, vals)
    return refined
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scatter_sampling.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit** — `git add` the task's files and commit with a clear message (end with the `Co-Authored-By` trailer).

---

### Task 2: Refactor `_attn` into `_attn_core` (behavior-preserving)

**Files:**
- Modify: `src/models/patchset_cnn.py` (`_tokens` ~192-197, `_attn` ~230-293)
- Test: `tests/test_patchset_cnn_refine.py` (existing — must stay green), plus one new test in `tests/test_patchset_scatter.py`

**Interfaces:**
- Produces: `_attn_core(sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij, res, K, ctx_count, mem=None, return_think=False, flat_out=False)`. When `flat_out=False` returns `(B,1,res,res)` (optionally `(logit, think)`); when `flat_out=True` returns `(B, Q)` query-cell logits.
- Consumes: `_tokens(feat, occ, ij, res=None)` now takes an optional `res` (defaults to `self.resolution`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_patchset_scatter.py` with a refactor-parity test (this asserts the coarse head is still internally consistent and that `_attn_core` exists):
```python
import sys; sys.path.insert(0, ".")
import torch
from src.models.patchset_cnn import PatchSetCNN


def _model(resolutions, H=32, refine_mode="reencode"):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolutions[0], enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, resolutions=resolutions, refine_mode=refine_mode)


def _batch(B=2, K=2, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_attn_core_grid_path_matches_segment():
    # After the refactor, the single-level forward (which routes through _attn -> _attn_core)
    # must equal a direct _segment call bit-for-bit.
    m = _model([8])
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert torch.equal(out["final_logit"], m._segment(img, cin, cout))
    assert hasattr(m, "_attn_core")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_patchset_scatter.py::test_attn_core_grid_path_matches_segment -v`
Expected: FAIL — `AssertionError` on `hasattr(m, "_attn_core")` (method does not exist yet).

- [ ] **Step 3: Write minimal implementation**

In `src/models/patchset_cnn.py`, replace `_tokens` (add optional `res`):
```python
    def _tokens(self, feat, occ, ij, res=None):
        """feat (B,M,Cf); occ (B,M,1); ij (B,M,2) -> (B,M,2,e) = [img-token | mask-token].
        `res` is the grid resolution used to normalize the Fourier position (defaults to the
        token grid T; the scatter refine passes the fine grid Rf)."""
        res = self.resolution if res is None else res
        p = self.pos(ij, res)                          # (B,M,e) Fourier position feature
        img = self.img_embed(feat) + p
        msk = self.mask_embed(occ) + p
        return torch.stack([img, msk], dim=2)
```

Replace the whole `_attn` method with a thin wrapper + the extracted `_attn_core`:
```python
    def _attn(self, sup_feat, qry_feat, sup_occ, K, mem=None, return_think=False):
        """Grid path: full R x R query lattice, support-mean prior. Wraps _attn_core with the
        grid defaults so the coarse / single-level output is unchanged."""
        B, N = sup_feat.shape[0], self.N
        qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, 1)          # (B,Q,1) prior
        sup_ij = self.ij_base.repeat(K, 1).unsqueeze(0).expand(B, K * N, 2)  # (B,S,2)
        qry_ij = self.ij_base.unsqueeze(0).expand(B, N, 2)                   # (B,Q,2)
        return self._attn_core(sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij,
                               self.resolution, K, self.N, mem=mem,
                               return_think=return_think, flat_out=False)

    def _attn_core(self, sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij, res, K,
                   ctx_count, mem=None, return_think=False, flat_out=False):
        """Set-of-patches attention half over an arbitrary support/query set.

        sup_feat (B,S,Cf), qry_feat (B,Q,Cf), sup_occ (B,S,1), qry_occ (B,Q,1); sup_ij/qry_ij
        are (·,2) grid coords normalized by `res`. ctx_count = patches per context image (N for
        the grid path, M for scatter) — used to broadcast the per-context id embedding.
        flat_out=False -> (B,1,res,res); flat_out=True -> (B,Q). return_think adds (B,n_think,e)."""
        B = sup_feat.shape[0]

        # per-channel standardize features by SUPPORT-patch stats
        mu = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        sup_feat = ((sup_feat - mu) / sig).clamp(-10, 10)
        qry_feat = ((qry_feat - mu) / sig).clamp(-10, 10)

        sup_tok = self._tokens(sup_feat, sup_occ, sup_ij, res)              # (B,S,2,e)
        qry_tok = self._tokens(qry_feat, qry_occ, qry_ij, res)             # (B,Q,2,e)

        if self.context_id_embed:
            assert K <= self.max_context, \
                f"context_size {K} exceeds max_context {self.max_context}"
            e_dim = sup_tok.shape[-1]
            ctx_emb = self.ctx_id(torch.arange(K, device=sup_tok.device))  # (K,e)
            ctx_emb = ctx_emb.repeat_interleave(ctx_count, dim=0)          # (K*ctx_count,e) image-major
            sup_tok = sup_tok + ctx_emb.view(1, K * ctx_count, 1, e_dim)
            qry_tok = qry_tok + self.qry_id.view(1, 1, 1, e_dim)

        sep = K * ctx_count
        rows = [sup_tok, qry_tok]
        if mem is not None:
            T1 = mem.shape[1]
            m = (mem + self.mem_type).unsqueeze(2).expand(mem.shape[0], T1, 2, mem.shape[-1])
            rows = [m] + rows                                              # [memory | support | query]
            sep += T1
        x = torch.cat(rows, dim=1)

        x, sep_t = self.thinking(x, sep)      # -> [thinking | memory | support | query]
        attn_mask = None
        if self.query_self_attn:
            r = x.shape[1]
            attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
            attn_mask[:, :sep_t] = True            # all rows -> thinking + support
            attn_mask[sep_t:, sep_t:] = True        # query -> query
        x = self.transformer(x, sep_t, attn_mask=attn_mask)

        q = x[:, sep_t:, 0, :]                                             # (B,Q,e) query img-col
        logit = self.decoder(q).squeeze(-1)                               # (B,Q)
        if not flat_out:
            logit = logit.reshape(B, 1, res, res)
        if return_think:
            return logit, x[:, :self.thinking.n].mean(dim=2)             # (B,n_think,e)
        return logit
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset_scatter.py::test_attn_core_grid_path_matches_segment tests/test_patchset_cnn_refine.py -v`
Expected: PASS — the new parity test AND every existing refine test (coarse, reencode, encode_once) stay green (this is the regression guard).

- [ ] **Step 5: Commit** — `git add` the task's files and commit with a clear message (end with the `Co-Authored-By` trailer).

---

### Task 3: `_refine_scatter` + `refine_mode="scatter"` dispatch + sample params

**Files:**
- Modify: `src/models/patchset_cnn.py` (top imports ~32-38; `__init__` assert ~133 and add sample params; `_refine_forward` ~307-310; add `_refine_scatter`)
- Test: `tests/test_patchset_scatter.py`

**Interfaces:**
- Consumes: `sample_patches, idx_to_ij, gather_grid` from `src.models.scatter_sampling`; `_attn_core` (Task 2).
- Produces: `PatchSetCNN(..., refine_mode="scatter", sample=<dict|None>)`. `forward(...)` for a scatter model returns `{"final_logit": (B,1,T,T), "refine_logit": (B,M), "refine_idx": (B,M), "refine_grid_res": int, "resolutions": list}`. `self.sample` is the resolved sampler-param dict.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patchset_scatter.py`:
```python
def _scatter_model(resolutions=(8, 16), H=32):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolutions[0], enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, resolutions=list(resolutions),
                       refine_mode="scatter", sample={"n_total": 20, "n_fg_core": 4,
                                                      "n_fg_core_ctx": 4})


def test_scatter_forward_shapes():
    m = _scatter_model((8, 16), H=32)          # fine grid Rf=16
    img, cin, cout = _batch(B=2, K=2, H=32)
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)     # coarse at T=8
    assert out["refine_logit"].shape == (2, 20)         # M sampled cells
    assert out["refine_idx"].shape == (2, 20)
    assert out["refine_grid_res"] == 16
    assert int(out["refine_idx"].max()) < 16 * 16 and int(out["refine_idx"].min()) >= 0
    assert "refine_origin" not in out                    # scatter != bbox
    assert torch.isfinite(out["refine_logit"]).all()


def test_scatter_backward_runs():
    m = _scatter_model()
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    out["refine_logit"].sum().backward()                 # gradients flow
    assert any(p.grad is not None for p in m.parameters())


def test_scatter_deterministic_in_eval():
    m = _scatter_model().eval()
    img, cin, cout = _batch()
    with torch.no_grad():
        a = m(img, context_in=cin, context_out=cout)["refine_idx"]
        b = m(img, context_in=cin, context_out=cout)["refine_idx"]
    assert torch.equal(a, b)                              # stochastic=False in eval
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_patchset_scatter.py::test_scatter_forward_shapes -v`
Expected: FAIL — `AssertionError: bad refine_mode 'scatter'` (constructor rejects it).

- [ ] **Step 3: Write minimal implementation**

In `src/models/patchset_cnn.py`, extend the top import block:
```python
from src.models.bbox_refine import crop_pool_maps, crop_resize, gt_window, max_sum_window
from src.models.scatter_sampling import sample_patches, idx_to_ij, gather_grid
```

Add a module-level default just above the class `PatchSetCNN`:
```python
DEFAULT_SAMPLE = dict(n_total=256, tau=0.30, blur_sigma=1.0, floor=0.005,
                      n_fg_core=64, n_fg_core_ctx=64, temperature=1.0, n_boundary_core=0)
```

In `__init__`, add a `sample` parameter to the signature (after `refine_memory: bool = False`):
```python
        refine_memory: bool = False,
        sample: dict | None = None,
```
Change the refine_mode assert to admit scatter, and resolve the sample dict (place right after `self.refine_mode = refine_mode`):
```python
        assert refine_mode in ("reencode", "encode_once", "scatter"), \
            f"bad refine_mode {refine_mode!r}"
        self.refine_mode = refine_mode
        self.sample = {**DEFAULT_SAMPLE, **(sample or {})}
```

Update `_refine_forward` to dispatch scatter:
```python
    def _refine_forward(self, image, context_in, context_out):
        if self.refine_mode == "scatter":
            return self._refine_scatter(image, context_in, context_out)
        if self.refine_mode == "encode_once":
            return self._refine_encode_once(image, context_in, context_out)
        return self._refine_reencode(image, context_in, context_out)
```

Add the new method (next to `_refine_encode_once`):
```python
    def _refine_scatter(self, image, context_in, context_out):
        """Coarse pass at T + unconstrained scatter refine at the fine grid Rf.

        Encode once; pool to Rf. Sample M query cells from the coarse prediction (prev_pred) and
        M support cells/context from the true mask fraction; run the attention core on that set.
        Returns per-sampled-cell logits + their flat Rf-grid indices (scattered back downstream)."""
        B, K = context_in.shape[0], context_in.shape[1]
        H, W = image.shape[-2:]
        T, Rf = self.resolution, self.resolutions[-1]
        Nf = Rf * Rf
        s = self.sample
        M = int(s["n_total"])
        stoch = self.training

        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)          # (B,Timgs,1,H,W)
        Timgs = imgs.shape[1]
        maps = self.encoder.encode_maps(imgs.reshape(B * Timgs, 1, H, W))  # native multi-scale, ONCE

        # ── coarse at the token grid T ──
        sup_c, qry_c = self._grid_tokens(self.encoder.pool_maps(maps, T), B, Timgs, K)
        if self.refine_memory:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K,
                                              return_think=True)             # (B,1,T,T), (B,n,e)
        else:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K), None

        # ── fine features at Rf ──
        fine = self.encoder.pool_maps(maps, Rf)                            # (B*Timgs,Cf,Rf,Rf)
        Cf = fine.shape[1]
        feat = fine.flatten(2).transpose(1, 2).reshape(B, Timgs, Nf, Cf)   # (B,Timgs,Nf,Cf)

        # ── query: sample from the coarse prediction upsampled to Rf (prev_pred) ──
        coarse_prob = torch.sigmoid(coarse).detach()                       # (B,1,T,T)
        q_map = F.interpolate(coarse_prob, size=(Rf, Rf), mode="bilinear",
                              align_corners=False).reshape(B, Nf)           # (B,Nf)
        qidx, _, _ = sample_patches(q_map, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
                                    temperature=s["temperature"], stochastic=stoch,
                                    n_fg_core=s["n_fg_core"], n_boundary_core=s["n_boundary_core"])
        qry_feat = gather_grid(feat[:, -1], qidx)                          # (B,M,Cf)  target is last
        qry_ij = idx_to_ij(qidx, Rf)                                       # (B,M,2)
        qry_occ = gather_grid(q_map, qidx).unsqueeze(-1)                  # (B,M,1) coarse-prob prior

        # ── support: sample from each context's true mask fraction at Rf ──
        ctx_frac = F.adaptive_avg_pool2d(context_out.reshape(B * K, 1, H, W),
                                         (Rf, Rf)).reshape(B * K, Nf)       # (B*K,Nf)
        sidx, _, _ = sample_patches(ctx_frac, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
                                    temperature=s["temperature"], stochastic=stoch,
                                    n_fg_core=s["n_fg_core_ctx"], n_boundary_core=s["n_boundary_core"])
        ctx_feat = feat[:, :K].reshape(B * K, Nf, Cf)
        sup_feat = gather_grid(ctx_feat, sidx).reshape(B, K * M, Cf)
        sup_occ = gather_grid(ctx_frac, sidx).reshape(B, K * M, 1)
        sup_ij = idx_to_ij(sidx, Rf).reshape(B, K * M, 2)

        mem = coarse_think.detach() if coarse_think is not None else None
        refine_logit = self._attn_core(sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij,
                                       Rf, K, M, mem=mem, flat_out=True)     # (B,M)
        return {"final_logit": coarse, "refine_logit": refine_logit, "refine_idx": qidx,
                "refine_grid_res": Rf, "resolutions": self.resolutions}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset_scatter.py tests/test_patchset_cnn_refine.py -v`
Expected: PASS — scatter forward/backward/eval-determinism tests pass; existing refine tests still green.

- [ ] **Step 5: Commit** — `git add` the task's files and commit with a clear message (end with the `Co-Authored-By` trailer).

---

### Task 4: Scatter refine-loss branch in the trainer

**Files:**
- Modify: `experiments/2d/train.py` (import ~54-57; refine-loss block ~199-206)
- Test: `tests/test_train_scatter_loss.py`

**Interfaces:**
- Consumes: `out["refine_idx"], out["refine_grid_res"], out["refine_logit"]` (Task 3); `gather_grid` (Task 1).
- Produces: a finite scalar refine loss added to the total when `out["refine_idx"]` is present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_scatter_loss.py` (unit-tests the loss/target math in isolation, no Hydra — it locks the exact expression that Step 3 adds to `train.py`):
```python
import sys; sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from src.models.scatter_sampling import gather_grid


def _refine_target(lbl, refine_idx, Rf):
    B = lbl.shape[0]
    gt_Rf = F.adaptive_avg_pool2d(lbl, (Rf, Rf)).reshape(B, Rf * Rf)
    return gather_grid(gt_Rf, refine_idx)


def test_refine_target_shape_and_range():
    B, M, Rf = 2, 20, 16
    lbl = (torch.rand(B, 1, 32, 32) > 0.5).float()
    idx = torch.stack([torch.randperm(Rf * Rf)[:M] for _ in range(B)])
    t = _refine_target(lbl, idx, Rf)
    assert t.shape == (B, M)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_refine_loss_finite():
    B, M, Rf = 2, 20, 16
    lbl = (torch.rand(B, 1, 32, 32) > 0.5).float()
    idx = torch.stack([torch.randperm(Rf * Rf)[:M] for _ in range(B)])
    rlogit = torch.randn(B, M, requires_grad=True)
    rtarget = _refine_target(lbl, idx, Rf)
    bce = F.binary_cross_entropy_with_logits(rlogit, rtarget)
    assert torch.isfinite(bce)
    bce.backward()
    assert rlogit.grad is not None
```

- [ ] **Step 2: Run test to verify it passes (it locks the math; the integration edit is Step 3)**

Run: `python -m pytest tests/test_train_scatter_loss.py -v`
Expected: PASS — this test depends only on `src.models.scatter_sampling` + torch, so it passes immediately and pins the exact target/loss expression that Step 3 wires into `train.py`. (The real integration point is `train.py`; verified in Step 4.)

- [ ] **Step 3: Write minimal implementation**

In `experiments/2d/train.py`, add `gather_grid` to the imports near `from src.models.bbox_refine import crop_resize`:
```python
from src.models.bbox_refine import crop_resize
from src.models.scatter_sampling import gather_grid
```
Replace the refine-loss block:
```python
        if out.get("refine_logit") is not None:            # multi-level: add the refine loss
            rlogit = out["refine_logit"].float()
            if out.get("refine_idx") is not None:          # scatter: GT gathered at sampled cells
                Rf = int(out["refine_grid_res"])
                gt_Rf = F.adaptive_avg_pool2d(lbl, (Rf, Rf)).reshape(lbl.shape[0], Rf * Rf)
                rtarget = gather_grid(gt_Rf, out["refine_idx"])                # (B,M)
            else:                                          # bbox: soft cropped GT at T
                rtarget = crop_resize(lbl, out["refine_origin"], int(out["refine_crop"]),
                                      rlogit.shape[-1], mode="bilinear")
            rbce = F.binary_cross_entropy_with_logits(rlogit, rtarget)
            rdice = soft_dice_loss(torch.sigmoid(rlogit), rtarget)
            loss = loss + float(cfg.train.get("refine_loss_weight", 1.0)) * (
                rbce + cfg.train.dice_weight * rdice)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_scatter_loss.py -v`
Expected: PASS (2 tests).
Also verify train.py still imports cleanly: `python -c "import ast; ast.parse(open('experiments/2d/train.py').read())"` → no output.

- [ ] **Step 5: Commit** — `git add` the task's files and commit with a clear message (end with the `Co-Authored-By` trailer).

---

### Task 5: Scatter branch in `refine_geometry` (eval + train monitoring)

**Files:**
- Modify: `experiments/2d/evaluate.py` (`refine_geometry` ~224-253)
- Test: `tests/test_refine_geometry_scatter.py`

**Interfaces:**
- Consumes: the scatter `out` dict (Task 3); `gather_grid, composite_predictions` (Task 1).
- Produces: `refine_geometry(out, lbl)` returns the SAME keys for scatter as for bbox: `refine_prob (B,1,M)`, `refine_target (B,1,M)`, `fused (B,1,H,H)`, `fused_R (B,1,Rf,Rf)`, `gt_R (B,1,Rf,Rf)`, `Rf`, `coarse_nat (B,1,H,H)`, `coarse_R (B,1,Rf,Rf)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_refine_geometry_scatter.py` (`evaluate.py` lives in `experiments/2d/`, so put that dir on the path to import it directly):
```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import torch
from src.models.patchset_cnn import PatchSetCNN


def _scatter_out(B=2, K=2, H=32, res=(8, 16), M=20):
    torch.manual_seed(0)
    m = PatchSetCNN(image_size=H, resolution=res[0], enc_dims=[16], e=32, h=64, l=1, a=2,
                    thinking_rows=1, resolutions=list(res), refine_mode="scatter",
                    sample={"n_total": M, "n_fg_core": 4, "n_fg_core_ctx": 4}).eval()
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    lbl = (torch.rand(B, 1, H, H) > 0.5).float()
    with torch.no_grad():
        out = m(img, context_in=cin, context_out=cout)
    return out, lbl


def test_refine_geometry_scatter_keys_and_shapes():
    import importlib
    ev = importlib.import_module("evaluate")   # experiments/2d on sys.path at runtime
    out, lbl = _scatter_out()
    rg = ev.refine_geometry(out, lbl)
    B, H, Rf, M = 2, 32, 16, 20
    assert rg["Rf"] == Rf
    assert rg["refine_prob"].shape == (B, 1, M)
    assert rg["refine_target"].shape == (B, 1, M)
    assert rg["fused"].shape == (B, 1, H, H)
    assert rg["fused_R"].shape == (B, 1, Rf, Rf)
    assert rg["gt_R"].shape == (B, 1, Rf, Rf)
    assert rg["coarse_nat"].shape == (B, 1, H, H)
    assert rg["coarse_R"].shape == (B, 1, Rf, Rf)
    # fused is a valid probability map everywhere
    assert torch.isfinite(rg["fused"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_refine_geometry_scatter.py -v`
Expected: FAIL — `refine_geometry` currently reads `out["refine_logit"].shape[-1]` as a grid and `out["refine_origin"]`, raising `KeyError: 'refine_origin'` for a scatter `out`.

- [ ] **Step 3: Write minimal implementation**

In `experiments/2d/evaluate.py`, add the imports near the top (with the other `src.models` imports):
```python
from src.models.scatter_sampling import gather_grid, composite_predictions
```
At the start of `refine_geometry`, branch to scatter when `refine_idx` is present (place right after the `if "refine_logit" not in out: return None` guard):
```python
    if "refine_logit" not in out:
        return None
    if out.get("refine_idx") is not None:
        return _refine_geometry_scatter(out, lbl)
```
Add the scatter helper just above `refine_geometry`:
```python
def _refine_geometry_scatter(out: dict, lbl: torch.Tensor) -> dict:
    """Scatter-refine geometry: per-sampled-cell prob/target + fused stitch (coarse with refined
    cells scattered in). Returns the SAME keys as the bbox refine_geometry so downstream metrics
    are model-agnostic. refine_prob/target are (B,1,M) so callers' [b,0] indexing yields (M,)."""
    coarse = out["final_logit"].float()                       # (B,1,T,T)
    refine_logit = out["refine_logit"].float()                # (B,M)
    idx = out["refine_idx"]                                    # (B,M)
    Rf = int(out["refine_grid_res"])
    B, H = lbl.shape[0], lbl.shape[-1]
    Nf = Rf * Rf
    refine_prob = torch.sigmoid(refine_logit)                 # (B,M)
    gt_Rf = F.adaptive_avg_pool2d(lbl, (Rf, Rf)).reshape(B, Nf)
    refine_target = gather_grid(gt_Rf, idx)                    # (B,M)
    coarse_prob = torch.sigmoid(coarse)
    coarse_up = F.interpolate(coarse_prob, size=(H, H), mode="bilinear", align_corners=False)
    coarse_Rf = F.adaptive_avg_pool2d(coarse_prob, (Rf, Rf)).reshape(B, Nf)     # (B,Nf)
    fused_flat = composite_predictions(coarse_Rf, idx, refine_prob)             # (B,Nf)
    fused_R = fused_flat.reshape(B, 1, Rf, Rf)
    fused = F.interpolate(fused_R, size=(H, H), mode="bilinear", align_corners=False)
    return {"refine_prob": refine_prob.unsqueeze(1),          # (B,1,M)
            "refine_target": refine_target.unsqueeze(1),      # (B,1,M)
            "fused": fused,                                    # (B,1,H,H)
            "fused_R": fused_R, "gt_R": gt_Rf.reshape(B, 1, Rf, Rf), "Rf": Rf,
            "coarse_nat": coarse_up,                           # (B,1,H,H)
            "coarse_R": coarse_Rf.reshape(B, 1, Rf, Rf)}       # (B,1,Rf,Rf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_refine_geometry_scatter.py -v`
Expected: PASS. Also confirm the bbox path is untouched: `python -m pytest tests/test_patchset_cnn_refine.py -v` → still green.

- [ ] **Step 5: Commit** — `git add` the task's files and commit with a clear message (end with the `Co-Authored-By` trailer).

---

### Task 6: Config for a scatter training run

**Files:**
- Create: `configs/experiment/2d/3_omnisynth_medseg_scatter.yaml`
- Modify: `experiments/2d/train.py` `build_model` (the patchset_cnn `arch` dict ~135-148) to pass `sample` + accept `refine_mode="scatter"`.

**Interfaces:**
- Consumes: `cfg.arch.refine_mode`, `cfg.arch.sample` (Task 3 constructor).
- Produces: a runnable `python experiments/2d/train.py --config-name 3_omnisynth_medseg_scatter`; the checkpoint `arch` dict carries `refine_mode` + `sample` so `eval.py` rebuilds with zero drift.

- [ ] **Step 1: Write the config**

Create `configs/experiment/2d/3_omnisynth_medseg_scatter.yaml`:
```yaml
# Experiment 3 — PatchSetCNN with UNCONSTRAINED SCATTER refinement, same omniSynth/MedSeg
# distribution as experiments 1 & 2. resolutions=[T, Rf]: coarse token grid T=32, fine scatter
# grid Rf=64. The refine level samples cfg.arch.sample.n_total cells from the coarse prediction
# (query) and the true mask (support), refines them, and scatters them back. Per-level losses
# (coarse@32 + refine on the sampled cells); checkpoint selects on native `dice` (fused).
#   python experiments/2d/train.py --config-name 3_omnisynth_medseg_scatter
defaults:
  - 1_omnisynth_medseg
  - _self_

arch:
  resolutions: [32, 64]      # T=32 coarse token grid; Rf=64 fine scatter grid
  refine_mode: scatter
  sample:
    n_total: 256             # M sampled cells per image (query and per-context support)
    tau: 0.30                # boundary-core band |p-0.5| < tau
    blur_sigma: 1.0          # neighbor proximity-field blur
    floor: 0.005             # uniform floor keeping far cells reachable
    n_fg_core: 64            # forced random foreground quota (query)
    n_fg_core_ctx: 64        # forced random foreground quota (support)
    temperature: 1.0         # Gumbel-top-k temperature
    n_boundary_core: 0       # 0 = uncapped boundary band (see coverage diagnostic)

train:
  refine_loss_weight: 1.0

augment: true
```

- [ ] **Step 2: Update `build_model`**

In `experiments/2d/train.py`, add `sample` to the patchset_cnn `arch` dict so it reaches the constructor and is persisted in the checkpoint:
```python
            "refine_mode": a.get("refine_mode", "reencode"),
            "refine_memory": a.get("refine_memory", False),
            "sample": OmegaConf.to_container(a.get("sample"), resolve=True) if a.get("sample", None) is not None else None,
        }
        model = PatchSetCNN(image_size=cfg.data.image_size, **arch)
```
(`OmegaConf` is already imported in train.py.)

- [ ] **Step 3: Smoke-test config resolution + model build (no training)**

Run a 1-step dry check that Hydra composes the config and the model builds:
```bash
python -c "
import sys; sys.path.insert(0,'experiments/2d')
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import os
with initialize_config_dir(config_dir=os.path.abspath('configs/experiment/2d'), version_base=None):
    cfg = compose(config_name='3_omnisynth_medseg_scatter')
print('refine_mode:', cfg.arch.refine_mode, '| sample.n_total:', cfg.arch.sample.n_total)
assert cfg.arch.refine_mode == 'scatter' and cfg.arch.sample.n_total == 256
print('OK')
"
```
Expected: prints `refine_mode: scatter | sample.n_total: 256` then `OK`.
(If the base config `1_omnisynth_medseg` requires cluster/path overrides to compose, instead assert via `python -c "import yaml; d=yaml.safe_load(open('configs/experiment/2d/3_omnisynth_medseg_scatter.yaml')); assert d['arch']['refine_mode']=='scatter'; print('OK')"`.)

- [ ] **Step 4: Full test sweep**

Run: `python -m pytest tests/test_scatter_sampling.py tests/test_patchset_scatter.py tests/test_train_scatter_loss.py tests/test_refine_geometry_scatter.py tests/test_patchset_cnn_refine.py -v`
Expected: all PASS (scatter suite + the untouched bbox refine suite).

- [ ] **Step 5: Commit** — record a one-line entry in `docs/logs.md` describing the scatter refine mode, then `git add` the task's files + logs and commit (end with the `Co-Authored-By` trailer).

---

## Self-Review

**Spec coverage:**
- Module `scatter_sampling.py` (capped `sample_patches` + helpers) → Task 1. ✓
- `_attn`→`_attn_core` refactor, backward-compatible → Task 2. ✓
- `_refine_scatter` + `refine_mode="scatter"` dispatch + sample params → Task 3. ✓
- Output contract (`refine_logit`, `refine_idx`, `refine_grid_res`) → Task 3. ✓
- Trainer scatter loss branch (gather GT at cells) → Task 4. ✓
- `refine_geometry` scatter branch (fused via `composite_predictions`, counterfactual keys) → Task 5. ✓
- Config + checkpoint `arch.sample` persistence → Task 6. ✓
- Testing (sampler, forward/backward/eval-determinism, refactor parity, loss, geometry) → Tasks 1–6. ✓
- Scope guardrails (single level; scatter interprets `Rf` as fine grid) → Global Constraints + Task 3. ✓

**Placeholder scan:** No TBD/TODO. Every code step shows the actual content. Task 4/Task 5 tests are self-contained (import only `src` + torch, or add `experiments/2d` to the path for `evaluate`).

**Type consistency:** `refine_idx` is `(B,M)` everywhere; `refine_logit` `(B,M)`; `refine_grid_res` int; `gather_grid` used identically in Tasks 1/4/5; `_attn_core(... res, K, ctx_count, mem, return_think, flat_out)` signature identical in Tasks 2 and 3; scatter `refine_geometry` returns the bbox key set (verified against `evaluate.py` usage `rg["refine_prob"][b,0]`, `rg["fused"]`, `rg["fused_R"]`, `rg["gt_R"]`, `rg["coarse_nat"]`, `rg["coarse_R"]`, `rg["Rf"]`).

# PatchSet3D Encoder Feature-Similarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build transformer-free machinery to measure how well a trained (or pluggable) encoder separates target↔context foreground vs background in feature space, swept over representation tier and pooling resolution.

**Architecture:** Three pure-function modules (`metrics`, `labels`) + a model-facing `adapters` module wrapping a loaded PatchSet3D encoder, driven by a Hydra `run.py` that loops the shared 3D eval loader and emits a tidy per-`(task,tier,res)` CSV. Metrics operate identically on dense grid cells or sampled points, so dense-small-`R'` and native-res point modes share code.

**Tech Stack:** PyTorch (pure-torch AUROC/AP — no sklearn), Hydra, the repo's `experiments/3d/common.py` eval loader and `eval.py` checkpoint-loading path.

## Global Constraints

- New code lives under `experiments/3d/feature_sim/` (metrics.py, labels.py, adapters.py, run.py). Tests under `tests/`.
- Import style matches the repo: tests start with `import sys; sys.path.insert(0, ".")`; to import the new package add `sys.path.insert(0, "experiments/3d")` then `from feature_sim.<mod> import ...` (the dir `experiments/3d` starts with a digit, so it is added to `sys.path`, never imported as `experiments.3d`).
- Tensor shapes from the loader (`src/totalseg_dataloader_incontext.py`): `image (B,1,D,H,W)`, `label (B,D,H,W)` int64 GT, `context_in (B,K,1,D,H,W)`, `context_out (B,K,D,H,W)` int64.
- `R = 16` is the model token grid (`resolution`). Occupancy rule = `_down_to` avg-pool then threshold `>= 0.5` (mirrors `patchset3d.py:_occupancy`).
- No new pip dependencies. AUROC/AP implemented in pure torch.
- Metrics are pure functions of feature arrays + `{0,1}` labels — no model, no I/O.
- Run pytest from repo root with the node's project env, e.g. `python -m pytest tests/<file> -v`.

---

### Task 1: Core matching metrics (`metrics.py`)

**Files:**
- Create: `experiments/3d/feature_sim/__init__.py` (empty)
- Create: `experiments/3d/feature_sim/metrics.py`
- Test: `tests/test_feature_sim_metrics.py`

**Interfaces:**
- Produces:
  - `l2norm(x: Tensor, dim=-1) -> Tensor`
  - `auroc(scores: Tensor[N], labels: Tensor[N]) -> float` (rank-based Mann–Whitney U)
  - `average_precision(scores: Tensor[N], labels: Tensor[N]) -> float`
  - `prototype_cosine(target_feats: Tensor[N,C], target_labels: Tensor[N], ctx_feats: Tensor[M,C], ctx_labels: Tensor[M], mode: str = "dense") -> dict` — `{"auroc","soft_dice"}` when `mode=="dense"`, `{"auroc","ap"}` when `mode=="point"`.
  - `fg_match_margin(target_feats, target_labels, ctx_feats, ctx_labels) -> float`
  - `retrieval_at1(target_feats, target_labels, ctx_feats, ctx_labels) -> float`
  - All accept `labels` as `{0,1}` int/float 1-D tensors.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_sim_metrics.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from feature_sim.metrics import (
    l2norm, auroc, average_precision,
    prototype_cosine, fg_match_margin, retrieval_at1)


def _separable(n=64, c=8, sep=6.0, seed=0):
    """FG clustered near +e0, BG near -e0; both target and context share the geometry."""
    g = torch.Generator().manual_seed(seed)
    def make(nfg, nbg):
        f = 0.3 * torch.randn(nfg + nbg, c, generator=g)
        f[:nfg, 0] += sep; f[nfg:, 0] -= sep
        lab = torch.cat([torch.ones(nfg), torch.zeros(nbg)])
        return f, lab
    tf, tl = make(n, n)
    cf, cl = make(n, n)
    return tf, tl, cf, cl


def test_auroc_perfect_and_chance():
    s = torch.tensor([0.9, 0.8, 0.2, 0.1]); y = torch.tensor([1., 1., 0., 0.])
    assert abs(auroc(s, y) - 1.0) < 1e-6
    # reversed labels -> 0.0
    assert abs(auroc(s, 1 - y) - 0.0) < 1e-6


def test_average_precision_perfect():
    s = torch.tensor([0.9, 0.8, 0.2, 0.1]); y = torch.tensor([1., 1., 0., 0.])
    assert abs(average_precision(s, y) - 1.0) < 1e-6


def test_l2norm_unit():
    x = torch.randn(5, 8)
    assert torch.allclose(l2norm(x).norm(dim=-1), torch.ones(5), atol=1e-5)


def test_prototype_cosine_separable_dense():
    tf, tl, cf, cl = _separable()
    out = prototype_cosine(tf, tl, cf, cl, mode="dense")
    assert out["auroc"] > 0.95 and out["soft_dice"] > 0.9


def test_prototype_cosine_separable_point():
    tf, tl, cf, cl = _separable()
    out = prototype_cosine(tf, tl, cf, cl, mode="point")
    assert out["auroc"] > 0.95 and out["ap"] > 0.9
    assert "soft_dice" not in out


def test_margin_and_retrieval_separable():
    tf, tl, cf, cl = _separable()
    assert fg_match_margin(tf, tl, cf, cl) > 0.3
    assert retrieval_at1(tf, tl, cf, cl) > 0.95


def test_random_features_are_chance():
    g = torch.Generator().manual_seed(1)
    tf = torch.randn(200, 8, generator=g); cf = torch.randn(200, 8, generator=g)
    tl = (torch.arange(200) % 2).float(); cl = (torch.arange(200) % 2).float()
    out = prototype_cosine(tf, tl, cf, cl, mode="dense")
    assert 0.35 < out["auroc"] < 0.65
    assert abs(fg_match_margin(tf, tl, cf, cl)) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_feature_sim_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'feature_sim'`.

- [ ] **Step 3: Implement `metrics.py`**

```python
# experiments/3d/feature_sim/metrics.py
"""Transformer-free target<->context matching metrics on encoder feature rows.

Rows are either dense grid cells or sampled points — the functions don't care.
Pure torch (no sklearn); AUROC is the rank-based Mann-Whitney U statistic."""
import torch


def l2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def auroc(scores, labels):
    """P(score[pos] > score[neg]) via mean rank of positives (ties -> average rank)."""
    labels = labels.float()
    n_pos = labels.sum().item(); n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
    # average tied ranks so exact ties score 0.5
    uniq, inv = torch.unique(scores, return_inverse=True)
    mean_rank = torch.zeros_like(uniq, dtype=torch.float).scatter_reduce(
        0, inv, ranks, reduce="mean", include_self=False)
    ranks = mean_rank[inv]
    sum_pos = ranks[labels == 1].sum().item()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(scores, labels):
    """Area under precision-recall via the step sum sum_k P(k) * dRecall(k)."""
    labels = labels.float()
    n_pos = labels.sum().item()
    if n_pos == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    y = labels[order]
    tp = torch.cumsum(y, 0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float)
    return (precision * y).sum().item() / n_pos


def _best_soft_dice(scores, labels):
    """Max Dice over thresholds at each unique score (scores expected in a bounded range)."""
    labels = labels.float()
    thr = torch.unique(scores)
    best = 0.0
    for t in thr:
        pred = (scores >= t).float()
        inter = (pred * labels).sum().item()
        den = pred.sum().item() + labels.sum().item()
        d = (2 * inter) / den if den > 0 else 0.0
        best = max(best, d)
    return best


def _prototype_scores(target_feats, ctx_feats, ctx_labels):
    proto = l2norm(l2norm(ctx_feats)[ctx_labels == 1].mean(0), dim=0)
    return l2norm(target_feats) @ proto


def prototype_cosine(target_feats, target_labels, ctx_feats, ctx_labels, mode="dense"):
    scores = _prototype_scores(target_feats, ctx_feats, ctx_labels)
    out = {"auroc": auroc(scores, target_labels)}
    if mode == "dense":
        out["soft_dice"] = _best_soft_dice(scores, target_labels)
    elif mode == "point":
        out["ap"] = average_precision(scores, target_labels)
    else:
        raise ValueError(f"mode must be 'dense' or 'point', got {mode!r}")
    return out


def fg_match_margin(target_feats, target_labels, ctx_feats, ctx_labels):
    tf = l2norm(target_feats)[target_labels == 1]
    cf = l2norm(ctx_feats)
    sims = tf @ cf.T                                   # (n_tfg, M)
    fg = sims[:, ctx_labels == 1].mean(1)
    bg = sims[:, ctx_labels == 0].mean(1)
    return (fg - bg).mean().item()


def retrieval_at1(target_feats, target_labels, ctx_feats, ctx_labels):
    tf = l2norm(target_feats)[target_labels == 1]
    cf = l2norm(ctx_feats)
    nn = (tf @ cf.T).argmax(1)
    return (ctx_labels[nn] == 1).float().mean().item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_feature_sim_metrics.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/feature_sim/__init__.py experiments/3d/feature_sim/metrics.py tests/test_feature_sim_metrics.py
git commit -m "feat(feature-sim): transformer-free target<->context matching metrics"
```

---

### Task 2: Cell / point labeling (`labels.py`)

**Files:**
- Create: `experiments/3d/feature_sim/labels.py`
- Test: `tests/test_feature_sim_labels.py`

**Interfaces:**
- Consumes: `_down_to` from `src.models.patchset3d`.
- Produces:
  - `grid_labels(mask: Tensor, res: int) -> Tensor` — `mask` is `(D,H,W)` or `(1,D,H,W)`; returns `(res,res,res)` float `{0,1}` via avg-pool + `>=0.5` (occupancy rule).
  - `sample_points(mask: Tensor[D,H,W], n_fg: int, n_bg: int, band: int | None = None, generator: torch.Generator | None = None) -> tuple[Tensor[N,3], Tensor[N]]` — normalized coords in `[-1,1]` in `(z,y,x)=(d,h,w)` axis order and `{0,1}` labels. FG sampled from `mask>0`; BG from `mask==0` (restricted to a `band`-voxel dilated shell around the object when `band` is set). Samples with replacement if a region is smaller than requested.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_sim_labels.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from feature_sim.labels import grid_labels, sample_points


def _blob(S=16):
    m = torch.zeros(S, S, S)
    m[4:12, 4:12, 4:12] = 1.0
    return m


def test_grid_labels_occupancy_matches_threshold():
    m = _blob(16)
    g = grid_labels(m, res=8)                 # 16->8, each cell = 2^3 block
    assert g.shape == (8, 8, 8)
    # cell (2,2,2) covers voxels [4:6] fully inside the blob -> 1
    assert g[2, 2, 2] == 1.0
    # corner cell fully outside -> 0
    assert g[0, 0, 0] == 0.0


def test_sample_points_counts_and_labels():
    m = _blob(16)
    coords, labels = sample_points(m, n_fg=50, n_bg=70,
                                   generator=torch.Generator().manual_seed(0))
    assert coords.shape == (120, 3) and labels.shape == (120,)
    assert labels.sum() == 50 and (labels == 0).sum() == 70
    assert coords.min() >= -1.0 and coords.max() <= 1.0


def test_sample_points_band_restricts_bg_near_object():
    m = _blob(16)
    coords, labels = sample_points(m, n_fg=10, n_bg=40, band=2,
                                   generator=torch.Generator().manual_seed(0))
    # all BG points should fall within a 2-voxel shell of the blob: convert the
    # normalized (d,h,w) coord back to a voxel index and check dist to [4,12).
    bg = coords[labels == 0]
    idx = ((bg + 1) / 2 * (16 - 1)).round().long()    # (n_bg,3) voxel indices
    inside_core = ((idx >= 4) & (idx < 12)).all(dim=1)
    assert not inside_core.any()                       # band excludes the FG core
    near = ((idx >= 2) & (idx < 14)).all(dim=1)
    assert near.all()                                  # within a 2-voxel shell
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_feature_sim_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'feature_sim.labels'`.

- [ ] **Step 3: Implement `labels.py`**

```python
# experiments/3d/feature_sim/labels.py
"""FG/BG labeling for feature-similarity: R'^3 occupancy grids (dense) and
native-res point sampling (point mode). Coords are grid_sample-ready."""
import torch
import torch.nn.functional as F

from src.models.patchset3d import _down_to


def grid_labels(mask, res):
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(0)                    # (1,D,H,W)
    occ = _down_to(m.unsqueeze(0), res)       # (1,1,res,res,res)
    return (occ >= 0.5).float().squeeze(0).squeeze(0)


def _to_norm_coords(idx, shape):
    """Voxel indices (N,3) in (d,h,w) -> normalized [-1,1] coords, same axis order."""
    dims = torch.tensor(shape, dtype=torch.float)
    return (idx.float() / (dims - 1)) * 2 - 1


def _dilate(mask, band):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    d = F.max_pool3d(m, kernel_size=2 * band + 1, stride=1, padding=band)
    return d.squeeze(0).squeeze(0) > 0


def _pick(coords_pool, n, generator):
    k = coords_pool.shape[0]
    if k == 0:
        return coords_pool.new_zeros((0, 3))
    replace = k < n
    sel = torch.randint(k, (n,), generator=generator) if replace \
        else torch.randperm(k, generator=generator)[:n]
    return coords_pool[sel]


def sample_points(mask, n_fg, n_bg, band=None, generator=None):
    m = mask
    fg_idx = torch.nonzero(m > 0, as_tuple=False)
    bg_mask = (m == 0)
    if band is not None:
        bg_mask = bg_mask & _dilate(m > 0, band)
    bg_idx = torch.nonzero(bg_mask, as_tuple=False)
    fg = _pick(fg_idx, n_fg, generator)
    bg = _pick(bg_idx, n_bg, generator)
    idx = torch.cat([fg, bg], dim=0)
    labels = torch.cat([torch.ones(fg.shape[0]), torch.zeros(bg.shape[0])])
    return _to_norm_coords(idx, m.shape), labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_feature_sim_labels.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/feature_sim/labels.py tests/test_feature_sim_labels.py
git commit -m "feat(feature-sim): R'^3 occupancy labels + native-res point sampler"
```

---

### Task 3: Encoder adapter (`adapters.py`)

**Files:**
- Create: `experiments/3d/feature_sim/adapters.py`
- Test: `tests/test_feature_sim_adapters.py`

**Interfaces:**
- Consumes: `PatchSet3D`, `ConvEncoder3D`, `_down_to` from `src.models.patchset3d`; `l2norm` unused here.
- Produces:
  - `class EncoderAdapter` (ABC): `tiers() -> list[str]`; `native_res(tier: str, input_res: int) -> int`; `features(volumes: Tensor[B,1,D,H,W], tier: str, res: int) -> Tensor[B,C,res,res,res]`; `sample_features(volumes: Tensor[B,1,D,H,W], tier: str, coords: Tensor[B,N,3]) -> Tensor[B,N,C]`; property `R: int`.
  - `class PatchSet3DEncoderAdapter(EncoderAdapter)`:
    - `__init__(self, model: PatchSet3D)`
    - conv tiers: `"stage:0".."stage:n"`, `"concat"`, `"img_embed"`.
    - `transformer_query(image, context_in, context_out) -> Tensor[B,N,e]` — decoder-input capture, `res=R` only.
    - `n_stages` property.
  - `coords` passed to `sample_features` are `(z,y,x)` normalized (as produced by `sample_points`); the adapter flips to `(x,y,z)` internally for `grid_sample`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_sim_adapters.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from src.models.patchset3d import PatchSet3D
from feature_sim.adapters import PatchSet3DEncoderAdapter


def _model():
    return PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                      thinking_rows=2, fourier_bands=4)


def _vols(B=2, S=16):
    return torch.randn(B, 1, S, S, S)


def test_tiers_and_native_res():
    ad = PatchSet3DEncoderAdapter(_model())
    ts = ad.tiers()
    assert "concat" in ts and "img_embed" in ts and "stage:0" in ts
    assert ad.R == 4
    # stem is full res, stage:1 is halved
    assert ad.native_res("stage:0", 16) == 16
    assert ad.native_res("stage:1", 16) == 8
    assert ad.native_res("concat", 16) == 16


def test_features_dense_shapes():
    ad = PatchSet3DEncoderAdapter(_model())
    v = _vols()
    f = ad.features(v, "concat", res=8)
    assert f.shape[0] == 2 and f.shape[2:] == (8, 8, 8)
    fs = ad.features(v, "stage:1", res=6)
    assert fs.shape[2:] == (6, 6, 6)
    fe = ad.features(v, "img_embed", res=4)
    assert fe.shape[1] == 32 and fe.shape[2:] == (4, 4, 4)   # e=32 channels


def test_sample_features_shape():
    ad = PatchSet3DEncoderAdapter(_model())
    v = _vols()
    coords = torch.rand(2, 20, 3) * 2 - 1
    s = ad.sample_features(v, "concat", coords)
    assert s.shape[:2] == (2, 20) and s.shape[2] == ad._concat_ch


def test_transformer_query_shape():
    m = _model(); ad = PatchSet3DEncoderAdapter(m)
    img = torch.randn(2, 1, 16, 16, 16)
    cin = torch.randn(2, 2, 1, 16, 16, 16)
    cout = (torch.rand(2, 2, 16, 16, 16) > 0.5).float()
    q = ad.transformer_query(img, cin, cout)
    assert q.shape == (2, ad.R ** 3, 32)                     # (B, N, e)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_feature_sim_adapters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'feature_sim.adapters'`.

- [ ] **Step 3: Implement `adapters.py`**

```python
# experiments/3d/feature_sim/adapters.py
"""Encoder-agnostic feature adapters for the similarity study.

EncoderAdapter maps volumes -> per-cell feature grids at an arbitrary resolution
(dense) or trilinearly-sampled point features (native res). PatchSet3DEncoderAdapter
wraps a loaded PatchSet3D. Future SAM/DINO adapters implement the same interface."""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from src.models.patchset3d import _down_to


class EncoderAdapter(ABC):
    @property
    @abstractmethod
    def R(self) -> int: ...

    @abstractmethod
    def tiers(self) -> list[str]: ...

    @abstractmethod
    def native_res(self, tier: str, input_res: int) -> int: ...

    @abstractmethod
    def features(self, volumes, tier, res): ...

    @abstractmethod
    def sample_features(self, volumes, tier, coords): ...


class PatchSet3DEncoderAdapter(EncoderAdapter):
    def __init__(self, model):
        self.model = model.eval()
        self.enc = model.encoder
        self._concat_ch = self.enc.out_ch

    @property
    def R(self):
        return self.model.resolution

    @property
    def n_stages(self):
        return len(self.enc.stages)               # excludes the stem

    def tiers(self):
        stages = [f"stage:{i}" for i in range(self.n_stages + 1)]
        return stages + ["concat", "img_embed"]

    def native_res(self, tier, input_res):
        if tier.startswith("stage:"):
            return input_res >> int(tier.split(":")[1])
        if tier in ("concat", "img_embed"):
            return input_res                       # stem-limited, finest genuine
        raise ValueError(f"unknown tier {tier!r}")

    @torch.no_grad()
    def _stage_feats(self, volumes):
        feats = [self.enc.stem(volumes)]
        for stage in self.enc.stages:
            feats.append(stage(feats[-1]))
        return feats                               # [stem, stage1, ...] native res

    @torch.no_grad()
    def _concat_native(self, feats):
        """Concat all stages at the finest (stem) native res — matches encoder semantics
        but keeps native detail instead of pooling to R."""
        r = feats[0].shape[-1]
        return torch.cat([_down_to(f, r) if f.shape[-1] != r else f for f in feats], 1)

    @torch.no_grad()
    def features(self, volumes, tier, res):
        feats = self._stage_feats(volumes)
        if tier.startswith("stage:"):
            f = feats[int(tier.split(":")[1])]
        elif tier == "concat":
            f = self._concat_native(feats)
        elif tier == "img_embed":
            f = self._concat_native(feats)         # projected below at target res
        else:
            raise ValueError(f"unknown tier {tier!r}")
        f = _down_to(f, res)                        # (B,C,res,res,res)
        if tier == "img_embed":
            B, C = f.shape[0], f.shape[1]
            flat = f.flatten(2).transpose(1, 2)     # (B, res^3, C)
            emb = self.model.img_embed(flat)        # (B, res^3, e)
            f = emb.transpose(1, 2).reshape(B, emb.shape[-1], res, res, res)
        return f

    @torch.no_grad()
    def sample_features(self, volumes, tier, coords):
        """coords (B,N,3) normalized in (z,y,x)=(d,h,w) order -> (B,N,C)."""
        feats = self._stage_feats(volumes)
        if tier.startswith("stage:"):
            f = feats[int(tier.split(":")[1])]
        elif tier in ("concat", "img_embed"):
            f = self._concat_native(feats)
        else:
            raise ValueError(f"unknown tier {tier!r}")
        xyz = coords.flip(-1).view(coords.shape[0], coords.shape[1], 1, 1, 3)  # ->(x,y,z)
        s = F.grid_sample(f, xyz, mode="bilinear", align_corners=True)          # (B,C,N,1,1)
        s = s.squeeze(-1).squeeze(-1).transpose(1, 2)                           # (B,N,C)
        if tier == "img_embed":
            s = self.model.img_embed(s)
        return s

    @torch.no_grad()
    def transformer_query(self, image, context_in, context_out):
        """Post-transformer query rep (B,N,e) via a decoder-input hook (res=R only)."""
        captured = {}
        h = self.model.decoder.register_forward_pre_hook(
            lambda mod, args: captured.setdefault("q", args[0]))
        try:
            self.model(image, context_in=context_in, context_out=context_out, mode="train")
        finally:
            h.remove()
        return captured["q"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_feature_sim_adapters.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/feature_sim/adapters.py tests/test_feature_sim_adapters.py
git commit -m "feat(feature-sim): PatchSet3D encoder adapter (tiers, dense + point features)"
```

---

### Task 4: Sweep planner + Hydra driver (`run.py`) + config

**Files:**
- Create: `experiments/3d/feature_sim/run.py`
- Create: `configs/experiment/3d/feature_sim.yaml`
- Modify: `docs/logs.md` (append an entry)
- Test: `tests/test_feature_sim_sweep.py`

**Interfaces:**
- Consumes: `PatchSet3DEncoderAdapter`; `grid_labels`, `sample_points`; `prototype_cosine`, `fg_match_margin`, `retrieval_at1`; `common.make_eval_loader`, `common.DEVICE`; the PatchSet3D checkpoint-load block from `experiments/3d/eval.py:55-83`.
- Produces:
  - `plan_sweep(tiers: list[str], resolutions: list[int], budget: int, R: int) -> list[dict]` — each `{"tier","res","mode"}`; `mode="point"` when `res**3 > budget` else `"dense"`; `transformer_q` (if present) forced to `res=R, mode="dense"`; deduped.
  - `main(cfg)` Hydra entry writing `feature_sim.csv` (tidy long form).

- [ ] **Step 1: Write the failing test (planner only — the driver is validated by a documented smoke run)**

```python
# tests/test_feature_sim_sweep.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
from feature_sim.run import plan_sweep


def test_plan_sweep_mode_and_budget():
    rows = plan_sweep(["concat", "stage:0"], [16, 64], budget=48 ** 3, R=16)
    modes = {(r["tier"], r["res"]): r["mode"] for r in rows}
    assert modes[("concat", 16)] == "dense"       # 16^3 <= 48^3
    assert modes[("concat", 64)] == "point"       # 64^3 > 48^3
    assert len(rows) == 4


def test_plan_sweep_transformer_q_pinned_to_R():
    rows = plan_sweep(["transformer_q"], [16, 64], budget=10 ** 9, R=16)
    assert len(rows) == 1
    assert rows[0] == {"tier": "transformer_q", "res": 16, "mode": "dense"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feature_sim_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'feature_sim.run'`.

- [ ] **Step 3: Implement `run.py`**

```python
# experiments/3d/feature_sim/run.py
"""Driver: load a PatchSet3D checkpoint, sweep (tier x resolution) over the shared 3D
eval loader, and write a tidy per-(task,tier,res) CSV of matching metrics + real Dice.

    python experiments/3d/feature_sim/run.py eval.checkpoint=results/.../best.pt \
        eval.model=patchset3d
"""
import csv
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))      # common / eval / evaluate

from common import DEVICE, make_eval_loader, _source_root          # noqa: E402
from data.totalseg_classes import resolve_classes                  # noqa: E402
from feature_sim.adapters import PatchSet3DEncoderAdapter          # noqa: E402
from feature_sim.labels import grid_labels, sample_points          # noqa: E402
from feature_sim.metrics import (                                  # noqa: E402
    prototype_cosine, fg_match_margin, retrieval_at1)


def plan_sweep(tiers, resolutions, budget, R):
    rows, seen = [], set()
    for tier in tiers:
        if tier == "transformer_q":
            key = (tier, R, "dense")
            if key not in seen:
                seen.add(key); rows.append({"tier": tier, "res": R, "mode": "dense"})
            continue
        for res in resolutions:
            mode = "point" if res ** 3 > budget else "dense"
            key = (tier, res, mode)
            if key not in seen:
                seen.add(key); rows.append({"tier": tier, "res": res, "mode": mode})
    return rows


def _load_patchset(cfg):
    """Rebuild PatchSet3D from the checkpoint's stored arch (mirrors eval.py:55-83)."""
    from train import build_model
    ckpt = torch.load(cfg.eval.checkpoint, map_location=DEVICE, weights_only=False)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.model = "patchset3d"
        if ckpt.get("arch") is not None:
            cfg.arch = OmegaConf.create(ckpt["arch"])
    model, _ = build_model(cfg)
    model = model.to(DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    return model.eval()


def _rows_for_task(adapter, model, item, cfg, plan, input_res, gen):
    """One task (single batch index already unbatched to B=1 tensors). Yields dict rows."""
    fs = cfg.feature_sim
    image = item["image"].to(DEVICE)              # (1,1,D,H,W)
    cin = item["context_in"].to(DEVICE)           # (1,K,1,D,H,W)
    cout = item["context_out"].to(DEVICE)         # (1,K,D,H,W)
    gt = item["label"][0]                         # (D,H,W)
    K = cin.shape[1]
    cls = item.get("label_name", ["?"])[0]
    obj_vox = int((gt > 0).sum().item())
    with torch.no_grad():
        real = model.predict(image, cin, cout)   # cin already (1,K,1,D,H,W) -> (1,D,H,W)
    inter = (real[0] * (gt.to(DEVICE) > 0)).sum().item()
    den = real[0].sum().item() + (gt > 0).sum().item()
    real_dice = (2 * inter) / den if den > 0 else 0.0

    ctx_imgs = cin[0].squeeze(1)                  # (K,D,H,W)
    for p in plan:
        tier, res, mode = p["tier"], p["res"], p["mode"]
        if tier == "transformer_q":
            q = adapter.transformer_query(image, cin, cout)[0]          # (N,e)
            tl = grid_labels(gt, adapter.R).flatten()
            cl = torch.stack([grid_labels(cout[0, k], adapter.R).flatten()
                              for k in range(K)]).flatten()
            # context query reps aren't produced by the hook; use encoder concat@R for ctx
            cf = adapter.features(ctx_imgs.unsqueeze(1), "concat", adapter.R)
            cf = cf.flatten(2).transpose(1, 2).reshape(-1, cf.shape[1])
            yield _metric_row(cls, obj_vox, real_dice, tier, res, mode,
                              adapter.native_res("concat", input_res),
                              q.cpu(), tl, cf.cpu(), cl, K)
            continue
        if mode == "dense":
            tf = adapter.features(image, tier, res)[0]                  # (C,res,res,res)
            tf = tf.flatten(1).transpose(0, 1)                         # (res^3, C)
            tl = grid_labels(gt, res).flatten()
            cvol = adapter.features(ctx_imgs.unsqueeze(1), tier, res)  # (K,C,res^3...)
            cf = cvol.flatten(2).transpose(1, 2).reshape(-1, cvol.shape[1])
            cl = torch.stack([grid_labels(cout[0, k], res).flatten()
                              for k in range(K)]).flatten()
        else:
            tcoords, tl = sample_points(gt, fs.n_fg, fs.n_bg,
                                        band=fs.get("band"), generator=gen)
            tf = adapter.sample_features(image, tier, tcoords.to(DEVICE).unsqueeze(0))[0]
            cfs, cls = [], []
            for k in range(K):
                cc, ll = sample_points(cout[0, k].cpu(), fs.n_fg, fs.n_bg,
                                       band=fs.get("band"), generator=gen)
                cfs.append(adapter.sample_features(
                    ctx_imgs[k][None, None], tier, cc.to(DEVICE).unsqueeze(0))[0])
                cls.append(ll)
            cf = torch.cat(cfs, 0); cl = torch.cat(cls, 0); tf = tf
        yield _metric_row(cls, obj_vox, real_dice, tier, res, mode,
                          adapter.native_res("concat" if tier == "transformer_q" else tier,
                                              input_res),
                          tf.cpu(), tl, cf.cpu(), cl, K)


def _metric_row(cls, obj_vox, real_dice, tier, res, mode, tier_native,
                tf, tl, cf, cl, K):
    proto = prototype_cosine(tf, tl, cf, cl, mode=mode)
    row = {"class": cls, "obj_vox": obj_vox, "real_dice": real_dice,
           "tier": tier, "res": res, "mode": mode, "tier_native_res": tier_native,
           "K": K, "auroc": proto["auroc"],
           "soft_dice": proto.get("soft_dice", ""), "ap": proto.get("ap", ""),
           "margin": fg_match_margin(tf, tl, cf, cl),
           "retrieval_at1": retrieval_at1(tf, tl, cf, cl)}
    return row


@hydra.main(config_path="../../../configs/experiment/3d",
            config_name="feature_sim", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.eval.seed)
    _, root, is_mri = _source_root(cfg)
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    loader = make_eval_loader(cfg, val_classes, split=cfg.eval.split)
    model = _load_patchset(cfg)
    adapter = PatchSet3DEncoderAdapter(model)
    input_res = int(cfg.data.image_size[-1])
    tiers = list(cfg.feature_sim.tiers)
    plan = plan_sweep(tiers, list(cfg.feature_sim.resolutions),
                      int(cfg.feature_sim.budget), adapter.R)
    gen = torch.Generator().manual_seed(cfg.eval.seed)

    out_dir = Path(cfg.eval.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_sim.csv"
    n = 0
    with open(csv_path, "w", newline="") as fh:
        writer = None
        for batch in loader:
            B = batch["image"].shape[0]
            for b in range(B):
                item = {k: (v[b:b + 1] if torch.is_tensor(v) else [v[b]])
                        for k, v in batch.items()}
                for row in _rows_for_task(adapter, model, item, cfg, plan,
                                          input_res, gen):
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row); n += 1
            if n and n % 200 == 0:
                print(f"  wrote {n} rows...")
    print(f"Done. {n} rows -> {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the planner test to verify it passes**

Run: `python -m pytest tests/test_feature_sim_sweep.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Create the Hydra config**

```yaml
# configs/experiment/3d/feature_sim.yaml
# @package _global_
# Feature-similarity study entrypoint (experiments/3d/feature_sim/run.py). Reuses the
# eval loader/checkpoint surface (eval.*) and adds a feature_sim.* sweep block.
#   python experiments/3d/feature_sim/run.py eval.model=patchset3d \
#       eval.checkpoint=results/checkpoints/<run>/best.pt eval.split=val
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
  model: patchset3d
  split: val
  n_subjects: 25
  batch_size: 4
  workers: 8
  seed: 0
  out_dir: ${paths.results}/3d_feature_sim
  checkpoint: null            # REQUIRED: trained PatchSet3D best.pt
feature_sim:
  tiers: [stage:0, stage:1, stage:2, concat, img_embed, transformer_q]
  resolutions: [16, 32, 48, 64]     # >48^3 auto-switches to point mode
  budget: 110592                    # 48^3; res^3 above this uses point sampling
  n_fg: 512
  n_bg: 512
  band: null                        # null = BG anywhere; int = dilated shell (voxels)
wandb:
  project: null
  name: null
```

- [ ] **Step 6: Smoke-run the driver on a real checkpoint (documented check, not a unit test)**

Run (substitute an existing PatchSet3D checkpoint):
```bash
python experiments/3d/feature_sim/run.py \
    eval.checkpoint=$(ls -t results/checkpoints/*/best.pt 2>/dev/null | head -1) \
    eval.n_subjects=2 eval.batch_size=2 \
    feature_sim.resolutions='[16,64]'
```
Expected: prints `Done. <N> rows -> .../feature_sim.csv`; CSV has columns
`class,obj_vox,real_dice,tier,res,mode,tier_native_res,K,auroc,soft_dice,ap,margin,retrieval_at1`
with `mode=dense` for res 16 and `mode=point` for res 64.

- [ ] **Step 7: Append a `docs/logs.md` entry**

```markdown
## Encoder feature-similarity study (2026-07-25)

Added `experiments/3d/feature_sim/` — transformer-free target<->context matching metrics
(prototype cosine -> AUROC/soft-Dice or AP; FG-match margin; top-1 retrieval), a
PatchSet3D encoder adapter (per-stage / concat / img_embed / transformer_q tiers, dense
`R'^3` grids + native-res point sampling), and a Hydra driver (`run.py`) that sweeps
(tier x resolution) over the shared eval loader and writes a tidy `feature_sim.csv` with
the model's real Dice per task. Spec:
docs/superpowers/specs/2026-07-25-patchset3d-encoder-feature-similarity-design.md.
SAM/DINO adapters and Dice-correlation analysis are phase 2.
```

- [ ] **Step 8: Commit**

```bash
git add experiments/3d/feature_sim/run.py configs/experiment/3d/feature_sim.yaml \
        tests/test_feature_sim_sweep.py docs/logs.md
git commit -m "feat(feature-sim): (tier x resolution) sweep driver + Hydra config"
```

---

## Self-Review notes

- **Spec coverage:** metrics (prototype cosine + margin + retrieval) → Task 1; occupancy labels + point sampler → Task 2; tier sweep + point/dense feature extraction + transformer_q ceiling → Task 3; resolution sweep + dense/point auto-switch + real-Dice column + CSV → Task 4. Pretrained adapters are interface-only (`EncoderAdapter` ABC) per phase-1 scope.
- **Deferred (phase 2, not in this plan):** `concat_std` standardization tier, SAM/DINO concrete adapters, Dice-correlation plots. `concat_std` is intentionally dropped from phase 1 to avoid threading support-set statistics through the adapter — revisit if the raw-tier results warrant it.
- **Type consistency:** `plan_sweep` dict keys `{tier,res,mode}` are consumed unchanged in `_rows_for_task`; `sample_points` returns `(coords[N,3] in (z,y,x), labels[N])` and `adapter.sample_features` flips to `(x,y,z)` — matched. Metric functions take `(target_feats, target_labels, ctx_feats, ctx_labels)` in that order everywhere.
```

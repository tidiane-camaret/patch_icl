# Max-cosine Similarity Query Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat `sup_occ.mean()` query prior in the single-level `PatchSetCNN` with a detached, PFENet-style max-cosine similarity prior mask that seeds the query mask token, to improve small-object (needle-in-haystack) segmentation.

**Architecture:** A new `PatchSetCNN._similarity_prior` computes, per query cell, the max cosine similarity between its encoder feature and the context's *foreground* patch features (min-max normalized to [0,1], detached). When a new `sim_prior` flag is on, `_attn` seeds the query occupancy tile with this prior instead of the support-mean, falling back to the mean for images with no foreground support cell. Off by default → byte-identical to current behavior; adds zero parameters so existing checkpoints reload unchanged.

**Tech Stack:** PyTorch, pytest, Hydra configs. Model in `src/models/patchset_cnn.py`; trainer/config plumbing in `experiments/2d/train.py` and `configs/experiment/2d/`.

## Global Constraints

- **Zero new parameters** — `sim_prior` must not add any `nn.Module`/`nn.Parameter`, so existing checkpoints load with `strict=True`.
- **Off-by-default, backward compatible** — `sim_prior: bool = False`; with it False the forward output is unchanged from today.
- **Detached prior** — the similarity map is `.detach()`-ed; it is an input signal on the mask column, never a gradient path.
- **Single-level grid path only** — wire into `_attn` (used by `_segment`); do NOT touch `_attn_core`'s scatter/flat path.
- **Prior stays in [0,1] occupancy scale** — it is uniform-filled into the `mask_patch_size²` tile and consumed by the shared `mask_embed`.
- **Full concatenated features, replace (not blend)** — similarity uses the full `sum(enc_dims)` feature; the prior fully replaces the mean prior when active.
- Run tests with the active interpreter: `python -m pytest <path> -v` (this repo's venv is already the active `python`).

---

### Task 1: `_similarity_prior` method + `sim_prior` constructor flag

**Files:**
- Modify: `src/models/patchset_cnn.py` (add `sim_prior` param in `PatchSetCNN.__init__` ~line 149; add `_similarity_prior` method near `_attn`, ~line 291)
- Test: `tests/test_patchset_sim_prior.py` (create)

**Interfaces:**
- Consumes: nothing new (uses `torch`, `torch.nn.functional as F` already imported in the module).
- Produces:
  - `PatchSetCNN(..., sim_prior: bool = False)` — stored as `self.sim_prior`.
  - `PatchSetCNN._similarity_prior(qry_feat, sup_feat, sup_occ) -> (prior, valid)` where
    `qry_feat` is `(B,N,Cf)`, `sup_feat` is `(B,S,Cf)`, `sup_occ` is `(B,S,p²)`;
    returns `prior` `(B,N)` float in `[0,1]` (detached) and `valid` `(B,)` bool
    (True where the image has ≥1 foreground support cell).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_patchset_sim_prior.py`:

```python
import sys; sys.path.insert(0, ".")
import torch
from src.models.patchset_cnn import PatchSetCNN


def _model(sim_prior=True, H=32, resolution=8, mask_patch_size=2):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolution, enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, mask_patch_size=mask_patch_size,
                       sim_prior=sim_prior)


def test_similarity_prior_shape_range_detached():
    m = _model()
    B, N, S, Cf, p2 = 2, m.N, 3 * m.N, m.encoder.out_ch, m.mask_patch_size ** 2
    torch.manual_seed(1)
    qry = torch.rand(B, N, Cf, requires_grad=True)
    sup = torch.rand(B, S, Cf)
    occ = (torch.rand(B, S, p2) > 0.5).float()
    prior, valid = m._similarity_prior(qry, sup, occ)
    assert prior.shape == (B, N)
    assert valid.shape == (B,) and valid.dtype == torch.bool
    assert float(prior.min()) >= 0.0 and float(prior.max()) <= 1.0
    assert not prior.requires_grad                       # detached input signal


def test_similarity_prior_matches_fg_exemplar():
    # A query cell whose feature EQUALS a foreground support cell's feature has cosine 1.0
    # (the global max), so after per-image min-max it must sit at the per-image maximum.
    m = _model()
    B, N, Cf, p2 = 1, m.N, m.encoder.out_ch, m.mask_patch_size ** 2
    torch.manual_seed(2)
    sup = torch.rand(B, N, Cf)                            # S == N here (K=1 worth)
    occ = torch.zeros(B, N, p2)
    occ[0, 5] = 1.0                                       # support cell 5 is foreground
    qry = torch.rand(B, N, Cf)
    qry[0, 9] = sup[0, 5]                                 # query cell 9 == fg support cell 5
    prior, valid = m._similarity_prior(qry, sup, occ)
    assert bool(valid[0]) is True
    assert torch.argmax(prior[0]).item() == 9            # cell 9 is the peak
    assert float(prior[0, 9]) > 0.99                      # normalized to the max (~1.0)


def test_similarity_prior_degenerate_no_fg():
    # No foreground support cells -> valid False (caller falls back to the flat mean prior).
    m = _model()
    B, N, Cf, p2 = 2, m.N, m.encoder.out_ch, m.mask_patch_size ** 2
    torch.manual_seed(3)
    qry = torch.rand(B, N, Cf)
    sup = torch.rand(B, N, Cf)
    occ = torch.zeros(B, N, p2)                           # all background
    prior, valid = m._similarity_prior(qry, sup, occ)
    assert bool(valid.any()) is False
    assert torch.isfinite(prior).all()                   # degenerate rows are finite (0), not -inf
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patchset_sim_prior.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'sim_prior'` (flag not added yet).

- [ ] **Step 3: Add the `sim_prior` constructor flag**

In `src/models/patchset_cnn.py`, add the parameter to `PatchSetCNN.__init__` (after `mask_patch_decode_size: int = 1,` in the signature, ~line 148):

```python
        mask_patch_decode_size: int = 1,
        sim_prior: bool = False,
```

And store it (near the other flag assignments, e.g. after `self.full_attn = full_attn` ~line 183):

```python
        # Max-cosine similarity query prior (PFENet-style): when True, _attn seeds the query
        # mask token with a localized foreground-similarity prior instead of the flat
        # support-mean. Adds NO parameters (checkpoint-compatible); grid/single-level path only.
        self.sim_prior = bool(sim_prior)
```

- [ ] **Step 4: Add the `_similarity_prior` method**

In `src/models/patchset_cnn.py`, add this method just above `_attn` (~line 291):

```python
    def _similarity_prior(self, qry_feat, sup_feat, sup_occ):
        """Max-cosine similarity prior mask (PFENet, Tian et al. 2020) → (prior, valid).

        qry_feat (B,N,Cf), sup_feat (B,S,Cf), sup_occ (B,S,p²). For each query cell, the MAX
        cosine similarity between its feature and the FOREGROUND support-cell features
        (occupancy≥0.5), min-max normalized per image to [0,1]. `max` (not softmax-mean) is
        imbalance-robust — the whole point for a needle. Returns a DETACHED (B,N) prior and a
        (B,) bool `valid` marking images with ≥1 fg support cell (callers fall back to the flat
        support-mean prior for the rest)."""
        occ = sup_occ.mean(dim=-1)                                   # (B,S) scalar occupancy
        fg = occ >= 0.5                                              # (B,S) foreground cells
        q = F.normalize(qry_feat, dim=-1)
        s = F.normalize(sup_feat, dim=-1)
        sim = torch.bmm(q, s.transpose(1, 2))                       # (B,N,S) cosine
        neg_inf = torch.finfo(sim.dtype).min
        sim = sim.masked_fill(~fg.unsqueeze(1), neg_inf)            # keep only fg exemplars
        prior = sim.max(dim=-1).values                              # (B,N) max-cosine to any fg
        valid = fg.any(dim=-1)                                      # (B,)
        prior = prior.masked_fill(~valid.unsqueeze(1), 0.0)        # degenerate rows -> 0 (finite)
        lo = prior.amin(dim=1, keepdim=True)
        hi = prior.amax(dim=1, keepdim=True)
        prior = (prior - lo) / (hi - lo).clamp_min(1e-6)          # per-image min-max -> [0,1]
        return prior.detach(), valid
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset_sim_prior.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add src/models/patchset_cnn.py tests/test_patchset_sim_prior.py
git commit -m "feat(patchset): _similarity_prior method + sim_prior flag"
```

---

### Task 2: Seed the query token in `_attn` (gated by `sim_prior`)

**Files:**
- Modify: `src/models/patchset_cnn.py` (`_attn`, ~line 291-300)
- Modify: `experiments/2d/train.py` (arch dict, ~line 156-171)
- Test: `tests/test_patchset_sim_prior.py` (append)

**Interfaces:**
- Consumes: `self.sim_prior` and `PatchSetCNN._similarity_prior` from Task 1.
- Produces: no new public interface; `_attn` now branches on `self.sim_prior`. `experiments/2d/train.py`'s `arch` dict records `"sim_prior"` so eval reload is drift-free.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patchset_sim_prior.py`:

```python
def _batch(B=2, K=1, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_sim_prior_no_new_params():
    # Zero-parameter feature: enabling sim_prior must not change the parameter set
    # (existing checkpoints must load strict=True).
    n_off = sum(p.numel() for p in _model(sim_prior=False).parameters())
    n_on = sum(p.numel() for p in _model(sim_prior=True).parameters())
    assert n_off == n_on


def test_sim_prior_changes_output():
    # Same seed -> identical weights (sim_prior adds no params/RNG draw), so any output
    # difference is due to the prior actually being wired into _attn.
    off = _model(sim_prior=False)
    on = _model(sim_prior=True)
    img, cin, cout = _batch()
    with torch.no_grad():
        a = off(img, context_in=cin, context_out=cout)["final_logit"]
        b = on(img, context_in=cin, context_out=cout)["final_logit"]
    assert a.shape == b.shape
    assert not torch.allclose(a, b)


def test_sim_prior_forward_backward_smoke():
    m = _model(sim_prior=True)               # H=32, resolution=8, mask_patch_size=2
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)     # one logit per token (decode=1)
    assert torch.isfinite(out["final_logit"]).all()
    out["final_logit"].sum().backward()                 # gradients still flow through the model
    assert any(p.grad is not None for p in m.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patchset_sim_prior.py -v`
Expected: `test_sim_prior_changes_output` FAILS (`not torch.allclose` assertion) because `_attn` does not yet use the prior — the two models produce identical output. (`no_new_params` and the smoke test may already pass.)

- [ ] **Step 3: Wire the prior into `_attn`**

In `src/models/patchset_cnn.py`, replace the first two lines of `_attn` (currently):

```python
        B, N = sup_feat.shape[0], self.N
        qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])  # (B,Q,p²) prior
```

with:

```python
        B, N = sup_feat.shape[0], self.N
        mean_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])  # (B,Q,p²) flat prior
        if self.sim_prior:
            prior, valid = self._similarity_prior(qry_feat, sup_feat, sup_occ)        # (B,N), (B,)
            prior_tile = prior.unsqueeze(-1).expand(B, N, sup_occ.shape[-1])          # uniform-fill p² tile
            qry_occ = torch.where(valid.view(B, 1, 1), prior_tile, mean_occ)          # fallback for no-fg images
        else:
            qry_occ = mean_occ
```

(The `sim_prior=False` branch is `qry_occ = mean_occ`, identical to the original expression — backward compatible.)

- [ ] **Step 4: Record `sim_prior` in the trainer arch dict**

In `experiments/2d/train.py`, add to the `arch` dict (after `"mask_patch_decode_size": a.get("mask_patch_decode_size", 1),` ~line 170):

```python
            "mask_patch_decode_size": a.get("mask_patch_decode_size", 1),
            "sim_prior": a.get("sim_prior", False),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset_sim_prior.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Run the existing patchset suite for no regressions**

Run: `python -m pytest tests/test_patchset_scatter.py tests/test_patchset_cnn_refine.py -v`
Expected: PASS (all existing tests green — the `sim_prior=False` default leaves every existing path unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/models/patchset_cnn.py experiments/2d/train.py tests/test_patchset_sim_prior.py
git commit -m "feat(patchset): seed query token with similarity prior in _attn"
```

---

### Task 3: Experiment config + end-to-end smoke

**Files:**
- Create: `configs/experiment/2d/6_sim_prior.yaml`
- Modify: `docs/logs.md` (append a log line — repo convention per CLAUDE.md)

**Interfaces:**
- Consumes: the `arch.sim_prior` key plumbed in Task 2.
- Produces: a runnable experiment `6_sim_prior` (inherits `5_full_res_decode`, sets `arch.sim_prior: true`).

- [ ] **Step 1: Create the experiment config**

Create `configs/experiment/2d/6_sim_prior.yaml`:

```yaml
# Experiment 6 - max-cosine similarity query prior (PFENet-style) on top of full-res decode.
# Seeds the query mask token with a localized foreground-similarity prior instead of the flat
# support-mean, to help the small-object needle-in-haystack. A/B against 5_full_res_decode.
# Design: docs/superpowers/specs/2026-07-16-similarity-prior-query-seed-design.md
defaults:
  - 5_full_res_decode
  - _self_

arch:
  sim_prior: true
```

- [ ] **Step 2: Smoke-test the full pipeline (tiny run)**

Run a minimal training run to confirm the config composes and the model trains end-to-end with the prior active:

```bash
python experiments/2d/train.py --config-name 6_sim_prior \
  train.epochs=1 data.max_train_subjects=8 train.wandb_project=null
```

Expected: the run starts, prints `Building PatchSetCNN (...)`, completes 1 epoch without error, and exits 0. (If the node's data/GPU is unavailable, instead confirm config composition only — see Step 3.)

- [ ] **Step 3: Verify config composition (data-independent fallback)**

If Step 2 cannot run on this node, verify the Hydra config composes with `sim_prior` set:

```bash
python -c "
import sys; sys.path.insert(0, 'experiments/2d')
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import os
with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
    cfg = compose(config_name='config', overrides=['experiment/2d=6_sim_prior'])
    assert cfg.arch.sim_prior is True, cfg.arch.sim_prior
    print('OK: arch.sim_prior =', cfg.arch.sim_prior)
"
```

Expected: `OK: arch.sim_prior = True`. (If the repo's Hydra entrypoint differs, fall back to asserting the YAML content directly; the authoritative check is Step 2.)

- [ ] **Step 4: Log the change**

Append one line to `docs/logs.md` (match the existing terse style):

```
- feat(patchset): sim_prior — max-cosine similarity query prior (PFENet-style) seeding the query
  mask token; single-level, zero params, off by default. Config 6_sim_prior (A/B vs 5). Targets
  small-object needle-in-haystack. Spec: 2026-07-16-similarity-prior-query-seed-design.md
```

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/2d/6_sim_prior.yaml docs/logs.md
git commit -m "feat(patchset): 6_sim_prior experiment config + log"
```

---

## Self-Review

**Spec coverage:**
- Max-cosine to foreground, full features, replace, detached, degenerate fallback, off-by-default, single-level → Task 1 (`_similarity_prior`) + Task 2 (`_attn` seeding). ✓
- Zero new parameters (checkpoint reload) → `test_sim_prior_no_new_params` (Task 2). ✓
- Trainer arch-dict plumbing → Task 2 Step 4. ✓
- `6_sim_prior.yaml` config, `5` untouched → Task 3. ✓
- Testing (unit shape/range/detached, fg-exemplar, degenerate, backward-compat, smoke) → Tasks 1-2. ✓
- Evaluation via the size-binned notebook → operational (run after training); no code task needed.

**Placeholder scan:** No TBD/TODO; all code and commands are concrete.

**Type consistency:** `_similarity_prior(qry_feat, sup_feat, sup_occ) -> (prior (B,N), valid (B,))` is defined in Task 1 and consumed with the same names/shapes in Task 2's `_attn` edit. `sim_prior` flag name is consistent across constructor, `_attn`, arch dict, and config.

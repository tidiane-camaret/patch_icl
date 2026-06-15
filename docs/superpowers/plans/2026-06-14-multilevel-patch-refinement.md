# Multilevel Patch Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a stage-2 patch-set in-context transformer that refines a frozen res-16 model's uncertain patches, resampled at resolution 32, and measure error reduction on the uncertain region.

**Architecture:** Per task, a frozen res-16 `ImagePFN` produces a coarse target prediction; a frozen `UniverSegFeatureEncoder` produces res-32 features for all images. We sample 256 patches/image (192 closest-to-0.5 + 64 most-certain): target patches → queries, context patches → support. A new `PatchSetPFN` (nanoTabPFN-shaped: rows=patches, cols=`[img|mask]`, 2-D Fourier positional encoding) predicts the query labels.

**Tech Stack:** PyTorch 2.5.1 (CUDA), Hydra, W&B, frozen UniverSeg encoder, Muon+AdamW, LAWA.

**Spec:** `docs/superpowers/specs/2026-06-14-multilevel-patch-refinement-design.md`

**Conventions for this plan:**
- Interpreter: `.venv311/bin/python` (per user; GPU env). Run all commands from repo root `/home/dpxuser/dev/patch_icl`.
- No pytest available → tests are standalone runnable scripts using `assert`; "fails" means a non-zero exit / traceback.
- **No git commits** (project rule: version control is left to the user). Each task ends with a **Checkpoint** the user may commit if desired — no `git` commands are issued.

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `experiments/2d/pfn_train.py` | Create | Shared training utils factored out of `pfn_seg.py`: `Muon`, `_newtonschulz5_batched`, `augment`, `lawa_average`, `soft_dice_loss`. |
| `experiments/2d/pfn_seg.py` | Modify | Import the above from `pfn_train` instead of defining them locally. No behavior change. |
| `src/models/patchset_pfn.py` | Create | `FourierPositionalEncoding` + `PatchSetPFN` model. |
| `experiments/2d/multilevel/sampling.py` | Create | Pure tensor ops: `sample_patch_indices`, `idx_to_ij`, `gather_grid`. |
| `experiments/2d/multilevel/pipeline.py` | Create | `coarse_predict`, `encode_grid`, `build_patch_batch` (coarse→sample→assemble). |
| `experiments/2d/multilevel/train.py` | Create | Hydra training script: loaders, frozen stage-1 + encoder, optimizers, train loop, eval/metrics, W&B, checkpoint. |
| `configs/experiment/2d/multilevel.yaml` | Create | Hydra config. |
| `experiments/2d/multilevel/test_sampling.py` | Create | Tests for `sampling.py`. |
| `experiments/2d/multilevel/test_patchset.py` | Create | Tests for `patchset_pfn.py`. |
| `experiments/2d/multilevel/test_pipeline.py` | Create | Tests for `pipeline.py` (stubbed stage-1/encoder). |
| `docs/logs.md` | Modify | Log the new experiment. |

---

## Task 1: Factor shared training utils into `pfn_train.py`

**Files:**
- Create: `experiments/2d/pfn_train.py`
- Modify: `experiments/2d/pfn_seg.py` (remove local defs, import from `pfn_train`)
- Test: `experiments/2d/multilevel/test_sampling.py` is later; here verify via a one-liner.

- [ ] **Step 1: Create `experiments/2d/pfn_train.py`**

Copy the five definitions **verbatim** from `experiments/2d/pfn_seg.py` (lines ~61–116 for `_newtonschulz5_batched`+`Muon`, ~120–188 for `augment`, ~246–254 for `lawa_average`, ~259–269 for `soft_dice_loss`). The new file:

```python
"""
Shared training utilities for the 2D ImagePFN / PatchSetPFN scripts.

Factored out of pfn_seg.py so pfn_seg.py and experiments/2d/multilevel/train.py
share one copy: Muon optimizer, batched GPU augmentation, LAWA averaging, soft-Dice.
"""

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Muon optimizer ────────────────────────────────────────────────────────────

def _newtonschulz5_batched(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Batched approximate matrix orthogonalization via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for hidden-layer 2D weight matrices (Newton-Schulz orthogonalized grads)."""
    def __init__(self, params, lr: float = 3e-4, momentum: float = 0.95,
                 weight_decay: float = 0.0, steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, steps=steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, wd, ns = group['lr'], group['momentum'], group['weight_decay'], group['steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(mu).add_(g)
                g = g.add(buf, alpha=mu)  # Nesterov
                if g.ndim == 2 and g.size(0) == 3 * g.size(1):
                    g_batch = g.view(3, g.size(1), g.size(1))
                    g_orth  = _newtonschulz5_batched(g_batch, steps=ns).view_as(g)
                    scale   = g.size(1) ** 0.5
                else:
                    g_orth = _newtonschulz5_batched(g.unsqueeze(0), steps=ns).squeeze(0)
                    scale  = max(g.size(0), g.size(1)) ** 0.5
                p.data.add_(g_orth, alpha=-lr * scale)
                if wd > 0:
                    p.data.mul_(1 - lr * wd)


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(images: torch.Tensor, masks: torch.Tensor, K: int, cfg):
    """Batched GPU augmentation. Geometric on context pairs; intensity on all images.

    images, masks: (B, T, 1, H, W) float32 on device; query is at index K. cfg = cfg.aug.
    """
    B, T, _, H, W = images.shape
    dev = images.device
    BK  = B * K

    c_imgs = images[:, :K].reshape(BK, 1, H, W)
    c_msks = masks[:, :K].reshape(BK, 1, H, W)

    g = cfg.geometric
    if g.hflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.hflip_p
        c_imgs = torch.where(m, c_imgs.flip(-1), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-1), c_msks)
    if g.vflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.vflip_p
        c_imgs = torch.where(m, c_imgs.flip(-2), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-2), c_msks)
    if g.rotate.p > 0:
        active = torch.rand(BK, device=dev) < g.rotate.p
        angles = (torch.rand(BK, device=dev) * 2 - 1) * g.rotate.max_angle_deg * active.float()
        rad    = torch.deg2rad(angles)
        cos_t, sin_t = torch.cos(rad), torch.sin(rad)
        z     = torch.zeros_like(cos_t)
        theta = torch.stack([cos_t, -sin_t, z, sin_t, cos_t, z], dim=1).reshape(BK, 2, 3)
        grid  = F.affine_grid(theta, (BK, 1, H, W), align_corners=False)
        c_imgs = F.grid_sample(c_imgs, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        c_msks = F.grid_sample(c_msks, grid, mode="nearest",  align_corners=False, padding_mode="zeros")

    images = torch.cat([c_imgs.reshape(B, K, 1, H, W), images[:, K:]], dim=1)
    masks  = torch.cat([c_msks.reshape(B, K, 1, H, W), masks[:, K:]],  dim=1)

    BT   = B * T
    imgs = images.reshape(BT, 1, H, W)
    ic   = cfg.intensity
    if ic.brightness.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.brightness.p
        d = (torch.rand(BT, 1, 1, 1, device=dev) * 2 - 1) * ic.brightness.max_delta
        imgs = torch.where(m, (imgs + d).clamp(0, 1), imgs)
    if ic.contrast.p > 0:
        lo, hi = ic.contrast.range
        m  = torch.rand(BT, 1, 1, 1, device=dev) < ic.contrast.p
        s  = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        mu = imgs.mean(dim=(-2, -1), keepdim=True)
        imgs = torch.where(m, ((imgs - mu) * s + mu).clamp(0, 1), imgs)
    if ic.gamma.p > 0:
        lo, hi = ic.gamma.range
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.gamma.p
        gm = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        imgs = torch.where(m, imgs.clamp(1e-6).pow(gm), imgs)
    if ic.noise.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.noise.p
        n = torch.randn_like(imgs) * ic.noise.std
        imgs = torch.where(m, (imgs + n).clamp(0, 1), imgs)

    return imgs.reshape(B, T, 1, H, W), masks


# ── LAWA ─────────────────────────────────────────────────────────────────────

def lawa_average(queue: collections.deque, model: nn.Module, device: torch.device):
    """Average checkpoint queue into model weights; return original state for restore."""
    if len(queue) <= 1:
        return None
    avg = {k: sum(s[k].float() for s in queue) / len(queue) for k in queue[0]}
    avg = {k: v.to(dtype=queue[0][k].dtype, device=device) for k, v in avg.items()}
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(avg)
    return saved


# ── Loss ──────────────────────────────────────────────────────────────────────

def soft_dice_loss(p: torch.Tensor, t: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-sample soft Dice loss between probability map p and soft target t (both flattened per row)."""
    p = p.flatten(1).float()
    t = t.flatten(1).float()
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1 - num / den).mean()
```

- [ ] **Step 2: Edit `experiments/2d/pfn_seg.py` to import from `pfn_train`**

Remove the local definitions of `_newtonschulz5_batched`, `Muon`, `augment`, `lawa_average`, `soft_dice_loss` (the `# ── Muon optimizer`, `# ── Augmentation`, `# ── LAWA`, and `# ── Loss` sections). Replace the existing `common` import block:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, TaggedDataset, collate, downsample_mask, hard_dice, soft_dice, log_summary
```

with (add the `pfn_train` import directly after it):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, TaggedDataset, collate, downsample_mask, hard_dice, soft_dice, log_summary
from pfn_train import Muon, augment, lawa_average, soft_dice_loss
```

Leave `import collections` in `pfn_seg.py` (still used by `collections.deque`). Leave `make_model_inputs`, `build_split_loader`, `train_epoch`, `run_eval`, `main` unchanged.

- [ ] **Step 3: Verify `pfn_seg.py` still imports and utils round-trip**

Run:
```bash
.venv311/bin/python -c "
import sys; sys.path.insert(0,'experiments/2d')
import torch
from pfn_train import Muon, augment, lawa_average, soft_dice_loss
# soft_dice_loss sanity: identical maps -> ~0 loss
p = torch.ones(2,4,4); t = torch.ones(2,4,4)
print('dice_loss(ones,ones)=', float(soft_dice_loss(p,t)))
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('pfn_seg', 'experiments/2d/pfn_seg.py')
print('pfn_seg.py parses & imports OK (module object):', spec is not None)
"
```
Expected: `dice_loss(ones,ones)=` a value < 0.05, and the OK line. (We don't exec `main`; importing the module file validates the import edits compile.)

- [ ] **Step 4: Checkpoint**

Shared utils factored out; `pfn_seg.py` behavior unchanged. User may commit.

---

## Task 2: Patch sampling ops (`sampling.py`)

**Files:**
- Create: `experiments/2d/multilevel/sampling.py`
- Test: `experiments/2d/multilevel/test_sampling.py`

- [ ] **Step 1: Write the failing test**

Create `experiments/2d/multilevel/test_sampling.py`:

```python
import sys; sys.path.insert(0, "experiments/2d/multilevel")
import torch
from sampling import sample_patch_indices, idx_to_ij, gather_grid

def test_selects_closest_and_farthest_from_half():
    # N=8 values; distances to 0.5 are well separated.
    v = torch.tensor([[0.50, 0.49, 0.51, 0.7, 0.3, 0.95, 0.05, 0.99]])
    idx = sample_patch_indices(v, n_uncertain=3, n_certain=2)
    unc, cer = idx[:, :3], idx[:, 3:]
    # 3 closest to 0.5 are values {0.50,0.49,0.51} -> indices {0,1,2}
    assert set(unc[0].tolist()) == {0, 1, 2}, unc
    # 2 farthest from 0.5 are {0.05,0.99} -> indices {6,7}
    assert set(cer[0].tolist()) == {6, 7}, cer
    # disjoint
    assert len(set(idx[0].tolist())) == 5

def test_idx_to_ij_roundtrip():
    R = 32
    idx = torch.tensor([[0, 1, 33, 1023]])
    ij = idx_to_ij(idx, R)
    assert ij.shape == (1, 4, 2)
    assert ij[0, 0].tolist() == [0, 0]
    assert ij[0, 1].tolist() == [0, 1]
    assert ij[0, 2].tolist() == [1, 1]
    assert ij[0, 3].tolist() == [31, 31]

def test_gather_grid_features_and_values():
    x = torch.arange(2*8*3).float().reshape(2, 8, 3)   # (B,N,C)
    idx = torch.tensor([[1, 4], [0, 7]])
    g = gather_grid(x, idx)
    assert g.shape == (2, 2, 3)
    assert torch.equal(g[0, 0], x[0, 1]) and torch.equal(g[1, 1], x[1, 7])
    vals = torch.arange(2*8).float().reshape(2, 8)      # (B,N)
    gv = gather_grid(vals, idx)
    assert gv.shape == (2, 2)
    assert gv[0, 0] == vals[0, 1] and gv[1, 1] == vals[1, 7]

if __name__ == "__main__":
    test_selects_closest_and_farthest_from_half()
    test_idx_to_ij_roundtrip()
    test_gather_grid_features_and_values()
    print("ALL SAMPLING TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_sampling.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sampling'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/2d/multilevel/sampling.py`:

```python
"""
Pure tensor ops for multilevel patch sampling.

Given per-cell values on a flattened grid, select cells nearest to 0.5 (uncertain)
and farthest from 0.5 (certain), and gather features/coords for the selected cells.
"""

import torch


def sample_patch_indices(values: torch.Tensor, n_uncertain: int, n_certain: int) -> torch.Tensor:
    """values: (B, N) in [0,1]. Returns (B, n_uncertain + n_certain) long indices:
    the n_uncertain cells closest to 0.5 followed by the n_certain cells farthest
    from 0.5. Disjoint as long as n_uncertain + n_certain <= N."""
    d = (values - 0.5).abs()                 # (B, N): 0 == on the 0.5 boundary
    order = d.argsort(dim=1)                  # ascending: closest-to-0.5 first
    unc = order[:, :n_uncertain]
    cer = order[:, order.shape[1] - n_certain:]   # farthest from 0.5 (largest d)
    return torch.cat([unc, cer], dim=1)


def idx_to_ij(idx: torch.Tensor, grid_res: int) -> torch.Tensor:
    """Flat cell index (B, M) → (B, M, 2) row/col coords on a grid_res×grid_res grid (row-major)."""
    return torch.stack([torch.div(idx, grid_res, rounding_mode="floor"),
                        idx % grid_res], dim=-1)


def gather_grid(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along the cell axis. x: (B, N, C) → (B, M, C), or x: (B, N) → (B, M)."""
    if x.dim() == 3:
        C = x.shape[-1]
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    return torch.gather(x, 1, idx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_sampling.py`
Expected: `ALL SAMPLING TESTS PASSED`

- [ ] **Step 5: Checkpoint**

Sampling ops done. User may commit.

---

## Task 3: `PatchSetPFN` model + Fourier PE (`patchset_pfn.py`)

**Files:**
- Create: `src/models/patchset_pfn.py`
- Test: `experiments/2d/multilevel/test_patchset.py`

- [ ] **Step 1: Write the failing test**

Create `experiments/2d/multilevel/test_patchset.py`:

```python
import sys; sys.path.insert(0, ".")
import torch
from src.models.patchset_pfn import PatchSetPFN, FourierPositionalEncoding

def test_fourier_pe_shape_and_resolution_generalizes():
    pe = FourierPositionalEncoding(e=16, num_bands=6)
    ij = torch.randint(0, 32, (2, 10, 2))
    out32 = pe(ij, grid_res=32)
    assert out32.shape == (2, 10, 16)
    # same module runs at a different grid resolution (generalization) without error
    ij64 = torch.randint(0, 64, (2, 10, 2))
    out64 = pe(ij64, grid_res=64)
    assert out64.shape == (2, 10, 16)

def _mk(coarse_prior):
    return PatchSetPFN(feature_dim=32, e=32, h=64, l=2, a=4, thinking_rows=2,
                       fourier_bands=6, coarse_prior=coarse_prior)

def test_forward_shapes_and_query_grad_only():
    B, S, Q, Fd, R = 2, 12, 8, 32, 32
    m = _mk(coarse_prior=True)
    sup_feat = torch.randn(B, S, Fd); sup_label = torch.rand(B, S); sup_ij = torch.randint(0, R, (B, S, 2))
    qry_feat = torch.randn(B, Q, Fd); qry_prior = torch.rand(B, Q); qry_ij = torch.randint(0, R, (B, Q, 2))
    logits = m(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij, grid_res=R)
    assert logits.shape == (B, Q)
    logits.sum().backward()
    assert m.decoder[0].weight.grad is not None        # learns
    assert m.img_embed.weight.grad is not None

def test_coarse_prior_false_runs():
    B, S, Q, Fd, R = 2, 12, 8, 32, 32
    m = _mk(coarse_prior=False)
    out = m(torch.randn(B,S,Fd), torch.rand(B,S), torch.randint(0,R,(B,S,2)),
            torch.randn(B,Q,Fd), torch.rand(B,Q), torch.randint(0,R,(B,Q,2)), grid_res=R)
    assert out.shape == (B, Q)

if __name__ == "__main__":
    test_fourier_pe_shape_and_resolution_generalizes()
    test_forward_shapes_and_query_grad_only()
    test_coarse_prior_false_runs()
    print("ALL PATCHSET TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_patchset.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.patchset_pfn'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/models/patchset_pfn.py`:

```python
"""
PatchSetPFN: stage-2 patch-set in-context refinement model.

A nanoTabPFN-shaped transformer: rows = sampled patches, cols = [img-token | mask-token].
Reuses ImagePFN's dual-axis TransformerEncoderStack and ThinkingRows.

  - img-token  = Linear(feature_dim → e) on the patch's frozen-encoder feature
  - mask-token = Linear(1 → e) on the patch's mask value (support: true fraction;
                 query: coarse prediction if coarse_prior else support-mean prior)
  - 2-D Fourier positional encoding of the patch's (i,j) grid cell, added to both
    tokens (resolution-generalizable: normalized coords + fixed frequencies)
  - sample-axis attention: query patches attend to thinking + support rows only
  - decoder reads each query's img-col → per-query logit
"""

import math

import torch
import torch.nn as nn

from src.models.pfn_seg_2d import ThinkingRows, TransformerEncoderStack


class FourierPositionalEncoding(nn.Module):
    """2-D Fourier features of normalized (i,j) → Linear → e. Resolution-generalizable."""
    def __init__(self, e: int, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        freqs = 2.0 ** torch.arange(num_bands).float()      # (L,) geometric: 1,2,4,...
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(4 * num_bands, e)

    def forward(self, ij: torch.Tensor, grid_res: int) -> torch.Tensor:
        # ij: (..., 2) integer cell coords on a grid_res×grid_res grid
        uv  = (ij.float() + 0.5) / grid_res                 # (...,2) in (0,1)
        ang = 2 * math.pi * uv.unsqueeze(-1) * self.freqs   # (...,2,L)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)   # (...,2,2L)
        feats = feats.flatten(-2)                           # (...,4L)
        return self.proj(feats)                             # (...,e)


class PatchSetPFN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        e: int = 256,
        h: int = 512,
        l: int = 6,
        a: int = 4,
        thinking_rows: int = 8,
        residual_decay: float = 0.95,
        fourier_bands: int = 8,
        coarse_prior: bool = True,
    ):
        super().__init__()
        self.coarse_prior = coarse_prior
        self.img_embed  = nn.Linear(feature_dim, e)
        self.mask_embed = nn.Linear(1, e)
        self.pos        = FourierPositionalEncoding(e, fourier_bands)
        self.thinking   = ThinkingRows(thinking_rows, e)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
        self.decoder    = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, 1))

    def _tokens(self, feat, label, ij, grid_res):
        # feat (B,R,F), label (B,R), ij (B,R,2) → (B,R,2,e)
        p   = self.pos(ij, grid_res)                        # (B,R,e)
        img = self.img_embed(feat) + p
        msk = self.mask_embed(label.unsqueeze(-1)) + p
        return torch.stack([img, msk], dim=2)               # (B,R,2,e)

    def forward(self, sup_feat, sup_label, sup_ij,
                qry_feat, qry_prior, qry_ij, grid_res):
        B, S, _ = sup_feat.shape
        Q = qry_feat.shape[1]

        # Per-channel feature normalization using support statistics.
        mu  = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        sup_feat = ((sup_feat - mu) / sig).clamp(-10, 10)
        qry_feat = ((qry_feat - mu) / sig).clamp(-10, 10)

        # Query mask prior: coarse pred, or the support-mean fraction (TargetEncoder analog).
        if not self.coarse_prior:
            qry_prior = sup_label.mean(dim=1, keepdim=True).expand(B, Q)

        sup_tok = self._tokens(sup_feat, sup_label, sup_ij, grid_res)   # (B,S,2,e)
        qry_tok = self._tokens(qry_feat, qry_prior, qry_ij, grid_res)   # (B,Q,2,e)
        x = torch.cat([sup_tok, qry_tok], dim=1)                        # (B,S+Q,2,e)

        x, sep_t = self.thinking(x, S)          # prepend thinking rows; sep_t = n_think + S
        x = self.transformer(x, sep_t)

        q = x[:, sep_t:, 0, :]                  # query rows, img-col → (B,Q,e)
        return self.decoder(q).squeeze(-1)      # (B,Q)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_patchset.py`
Expected: `ALL PATCHSET TESTS PASSED`

- [ ] **Step 5: Checkpoint**

Model + Fourier PE done. User may commit.

---

## Task 4: Coarse→sample→assemble pipeline (`pipeline.py`)

**Files:**
- Create: `experiments/2d/multilevel/pipeline.py`
- Test: `experiments/2d/multilevel/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `experiments/2d/multilevel/test_pipeline.py`. Uses stubs for the frozen stage-1 model and encoder so it runs on CPU without UniverSeg:

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d/multilevel")
import torch
from pipeline import build_patch_batch

class StubStage1:
    """Returns res-16 logits; here a fixed gradient so |pred-0.5| ranking is well-defined."""
    def __call__(self, images, masks, sep):
        B = images.shape[0]
        # logits ramp across the 16x16 grid → varied sigmoid values
        row = torch.linspace(-4, 4, 16)
        grid = row.view(1, 16, 1).expand(B, 16, 16).clone()
        return grid
    def eval(self): return self

class StubEncoder:
    """Returns (B*T, C, R, R) features. forward(images, out_size)."""
    feature_dim = 5
    def __call__(self, images, out_size):
        N = images.shape[0]
        return torch.randn(N, self.feature_dim, out_size, out_size)
    def eval(self): return self

class Cfg:
    class sample: grid_res = 32; n_uncertain = 192; n_certain = 64
    class arch:   coarse_prior = True

def test_build_patch_batch_shapes():
    B, K, H = 2, 3, 128
    batch = {
        "image":       torch.rand(B, 1, H, H),
        "label":       (torch.rand(B, 1, H, H) > 0.5).float(),
        "context_in":  torch.rand(B, K, 1, H, H),
        "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float(),
    }
    out = build_patch_batch(batch, StubStage1(), StubEncoder(), Cfg, torch.device("cpu"))
    M = 256
    assert out["qry_feat"].shape  == (B, M, 5)
    assert out["sup_feat"].shape  == (B, K * M, 5)
    assert out["qry_ij"].shape    == (B, M, 2)
    assert out["sup_ij"].shape    == (B, K * M, 2)
    assert out["qry_gt"].shape    == (B, M)
    assert out["qry_coarse"].shape == (B, M)
    assert out["qry_prior"].shape == (B, M)
    assert out["qry_is_uncertain"].shape == (B, M)
    # first 192 queries are the uncertain ones
    assert bool(out["qry_is_uncertain"][0, :192].all())
    assert bool((~out["qry_is_uncertain"][0, 192:]).all())
    # support labels/coords in valid ranges
    assert out["sup_label"].min() >= 0 and out["sup_label"].max() <= 1
    assert out["qry_ij"].max() < 32

if __name__ == "__main__":
    test_build_patch_batch_shapes()
    print("ALL PIPELINE TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 3: Write minimal implementation**

Create `experiments/2d/multilevel/pipeline.py`:

```python
"""
Coarse → sample → assemble for the multilevel refinement task.

Given a data batch, a frozen stage-1 res-16 ImagePFN, and a frozen res-32 feature
encoder, build the support/query patch tensors consumed by PatchSetPFN.
"""

import torch
import torch.nn.functional as F

from sampling import sample_patch_indices, idx_to_ij, gather_grid


@torch.no_grad()
def coarse_predict(stage1, all_images, all_masks, K, grid_res):
    """Frozen stage-1 target prediction, upsampled to (grid_res, grid_res). Returns (B, grid_res, grid_res)."""
    logits = stage1(all_images, all_masks, sep=K)            # (B, R1, R1)
    p = torch.sigmoid(logits.float())
    p = F.interpolate(p.unsqueeze(1), size=(grid_res, grid_res),
                      mode="bilinear", align_corners=False).squeeze(1)
    return p


@torch.no_grad()
def encode_grid(encoder, images, grid_res):
    """images (B, T, 1, H, W) → features (B, T, grid_res², Cf) in row-major cell order."""
    B, T, _, H, W = images.shape
    feat = encoder(images.reshape(B * T, 1, H, W), grid_res)  # (B*T, Cf, R2, R2)
    Cf = feat.shape[1]
    return feat.flatten(2).transpose(1, 2).reshape(B, T, grid_res * grid_res, Cf)


def _grid_fractions(masks, grid_res):
    """masks (B, T, 1, H, W) → soft mask fraction per cell (B, T, grid_res²)."""
    B, T, _, H, W = masks.shape
    f = F.adaptive_avg_pool2d(masks.reshape(B * T, 1, H, W).float(), (grid_res, grid_res))
    return f.reshape(B, T, grid_res * grid_res)


@torch.no_grad()
def build_patch_batch(batch, stage1, encoder, cfg, device):
    """Returns a dict of tensors on `device` for PatchSetPFN + metrics.

    Keys: sup_feat (B,K*M,Cf), sup_label (B,K*M), sup_ij (B,K*M,2),
          qry_feat (B,M,Cf), qry_prior (B,M), qry_ij (B,M,2),
          qry_gt (B,M), qry_coarse (B,M), qry_is_uncertain (B,M bool).
    M = n_uncertain + n_certain.
    """
    R2 = cfg.sample.grid_res
    n_unc, n_cer = cfg.sample.n_uncertain, cfg.sample.n_certain
    M = n_unc + n_cer

    image       = batch["image"].to(device)         # (B,1,H,W)
    label       = batch["label"].to(device)         # (B,1,H,W)
    context_in  = batch["context_in"].to(device)    # (B,K,1,H,W)
    context_out = batch["context_out"].to(device)   # (B,K,1,H,W)
    B, K = context_in.shape[0], context_in.shape[1]

    # Stack: target is the LAST row; query mask is zeros (stage-1 fills its own prior).
    all_images = torch.cat([context_in, image.unsqueeze(1)], dim=1)             # (B,T,1,H,W)
    all_masks  = torch.cat([context_out, torch.zeros_like(image.unsqueeze(1))], dim=1)

    coarse = coarse_predict(stage1, all_images, all_masks, K, R2)               # (B,R2,R2)
    coarse_flat = coarse.reshape(B, R2 * R2)                                    # (B,N)

    feats = encode_grid(encoder, all_images, R2)                               # (B,T,N,Cf)
    fracs = _grid_fractions(all_masks, R2)                                     # (B,T,N)
    gt32  = fracs[:, -1]                                                        # (B,N) target GT fraction

    # ── Query (target) patches: rank by coarse pred ──────────────────────────
    qidx = sample_patch_indices(coarse_flat, n_unc, n_cer)                      # (B,M)
    qry_feat   = gather_grid(feats[:, -1], qidx)                                # (B,M,Cf)
    qry_coarse = gather_grid(coarse_flat, qidx)                                 # (B,M)
    qry_gt     = gather_grid(gt32, qidx)                                        # (B,M)
    qry_ij     = idx_to_ij(qidx, R2)                                            # (B,M,2)
    qry_prior  = qry_coarse if cfg.arch.coarse_prior else torch.zeros_like(qry_coarse)
    is_unc = torch.zeros(B, M, dtype=torch.bool, device=device)
    is_unc[:, :n_unc] = True

    # ── Support (context) patches: rank by true mask fraction, batched over K ─
    ctx_feat = feats[:, :K].reshape(B * K, R2 * R2, feats.shape[-1])            # (B*K,N,Cf)
    ctx_frac = fracs[:, :K].reshape(B * K, R2 * R2)                             # (B*K,N)
    sidx = sample_patch_indices(ctx_frac, n_unc, n_cer)                         # (B*K,M)
    sup_feat  = gather_grid(ctx_feat, sidx).reshape(B, K * M, feats.shape[-1])  # (B,K*M,Cf)
    sup_label = gather_grid(ctx_frac, sidx).reshape(B, K * M)                   # (B,K*M)
    sup_ij    = idx_to_ij(sidx, R2).reshape(B, K * M, 2)                        # (B,K*M,2)

    return {
        "sup_feat": sup_feat, "sup_label": sup_label, "sup_ij": sup_ij,
        "qry_feat": qry_feat, "qry_prior": qry_prior, "qry_ij": qry_ij,
        "qry_gt": qry_gt, "qry_coarse": qry_coarse, "qry_is_uncertain": is_unc,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: `ALL PIPELINE TESTS PASSED`

- [ ] **Step 5: Checkpoint**

Pipeline done. User may commit.

---

## Task 5: Hydra config (`multilevel.yaml`)

**Files:**
- Create: `configs/experiment/2d/multilevel.yaml`

- [ ] **Step 1: Create the config**

Create `configs/experiment/2d/multilevel.yaml`:

```yaml
model: patchset_pfn

data:
  image_size: 128
  context_size: 3
  dataset: null
  max_train_samples: null

sample:
  grid_res: 32        # res-32 grid (32×32 = 1024 cells per image)
  n_uncertain: 192    # cells closest to 0.5
  n_certain: 64       # most-certain cells

arch:
  feature_level: all  # UniverSeg encoder level → feature_dim 256
  coarse_prior: true  # query mask-token = stage-1 coarse pred (else neutral 0)
  fourier_bands: 8
  e: 256
  h: 512
  l: 6
  a: 4
  thinking_rows: 8
  residual_decay: 0.95
  compile: true       # graph-breaks at the frozen encoder (as in pfn_seg)

train:
  epochs: 200
  batch_size: 8
  lr: 1.0e-3
  grad_clip: 2.0
  dice_weight: 1.0
  lawa_k: 10
  muon_lr_scale: 0.1
  muon_momentum: 0.96
  muon_wd: 0.1
  adam_wd: 0.01
  warmup_epochs: 5
  eval_every: 1
  workers: 16
  seed: 42
  checkpoint: null    # warm-start for PatchSetPFN (not the stage-1 model)
  stage1_checkpoint: results/2d/pfn_seg_low_res_loss/pfn_seg_P8_e256_l6_k3_think8/best.pt

eval:
  batch_size: 16
  workers: 8
  max_per_label: 20
  out_dir: results/2d/multilevel

wandb:
  project: patch_icl_2d_exps_train
  name: null
  enabled: true

# mirrors configs/augmentations/medsegbench.yaml
aug:
  enabled: true
  geometric: { hflip_p: 0.5, vflip_p: 0.5, rotate: { p: 0.5, max_angle_deg: 20.0 } }
  intensity:
    brightness: { p: 0.5, max_delta: 0.15 }
    contrast:   { p: 0.5, range: [0.8, 1.2] }
    gamma:      { p: 0.3, range: [0.75, 1.33] }
    noise:      { p: 0.3, std: 0.04 }
```

- [ ] **Step 2: Verify it loads**

Run:
```bash
.venv311/bin/python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('configs/experiment/2d/multilevel.yaml')
print('grid_res', c.sample.grid_res, 'queries/img', c.sample.n_uncertain + c.sample.n_certain,
      'coarse_prior', c.arch.coarse_prior, 'stage1', c.train.stage1_checkpoint)
"
```
Expected: `grid_res 32 queries/img 256 coarse_prior True stage1 results/2d/pfn_seg_low_res_loss/...best.pt`

- [ ] **Step 3: Checkpoint**

Config in place. User may commit.

---

## Task 6: Training script (`multilevel/train.py`)

**Files:**
- Create: `experiments/2d/multilevel/train.py`

This script wires everything together. It loads the frozen stage-1 `ImagePFN` (deriving its arch from the checkpoint, mirroring `eval.py`), builds the frozen encoder, and trains `PatchSetPFN`.

- [ ] **Step 1: Create the script**

Create `experiments/2d/multilevel/train.py`:

```python
"""
Stage-2 multilevel patch refinement training.

Frozen res-16 ImagePFN (stage 1) + frozen UniverSeg encoder produce coarse target
predictions and res-32 features; we sample 256 patches/image and train a PatchSetPFN
to refine the uncertain target patches. Metric of interest: |error| reduction on the
sampled uncertain region vs the stage-1 coarse value.

Usage:
    python experiments/2d/multilevel/train.py
    python experiments/2d/multilevel/train.py arch.coarse_prior=false train.lr=5e-4
"""

import collections
import math
import os
import socket
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_cache_root = os.path.join(tempfile.gettempdir(), f"{os.environ.get('USER','user')}_compile_{socket.gethostname()}")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "inductor"))

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _ROOT)
# Cache patch_icl's src before common.py inserts ic_segmentation's shadowing src.
from src.datasets.medsegbench import MedSegBenchDataset   # noqa: F401
from src.models.pfn_seg_2d import ImagePFN
from src.models.patchset_pfn import PatchSetPFN
from src.models.pretrained_encoders import UniverSegFeatureEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # experiments/2d
from common import DEVICE, TaggedDataset, collate, hard_dice
from pfn_train import Muon, augment, lawa_average, soft_dice_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))       # multilevel
from pipeline import build_patch_batch

from torch.utils.data import DataLoader, RandomSampler


def build_split_loader(cfg, split, shuffle):
    datasets = [cfg.data.dataset] if cfg.data.dataset else None
    ds = MedSegBenchDataset(split=split, context_size=cfg.data.context_size,
                            image_size=cfg.data.image_size, datasets=datasets)
    if split == "val" and cfg.eval.max_per_label:
        import random
        groups = {}
        for i, (name, _, lv) in enumerate(ds.samples):
            groups.setdefault((name, lv), []).append(i)
        keep = []
        for idxs in groups.values():
            keep.extend(random.sample(idxs, min(cfg.eval.max_per_label, len(idxs))))
        ds.samples = [ds.samples[i] for i in sorted(keep)]
    bs = cfg.train.batch_size if split == "train" else cfg.eval.batch_size
    nw = cfg.train.workers   if split == "train" else cfg.eval.workers
    max_train = cfg.data.get("max_train_samples", None)
    sampler = (RandomSampler(ds, replacement=False, num_samples=max_train)
               if split == "train" and max_train is not None else None)
    return DataLoader(TaggedDataset(ds), batch_size=bs,
                      shuffle=(shuffle and sampler is None), sampler=sampler,
                      num_workers=nw, collate_fn=collate,
                      pin_memory=DEVICE.type == "cuda",
                      persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)


def load_stage1(cfg):
    """Load the frozen res-16 ImagePFN from its checkpoint (arch read from the .pt)."""
    ckpt = torch.load(cfg.train.stage1_checkpoint, map_location="cpu", weights_only=False)
    arch, img_size = ckpt["arch"], ckpt["image_size"]
    resolution = arch.get("resolution", img_size // arch["patch_size"] if "patch_size" in arch else None)
    input_patch_size = arch.get("input_patch_size", img_size // resolution)
    image_encoder, feature_dim = None, None
    if arch.get("image_encoder", "patch") == "universeg":
        image_encoder = UniverSegFeatureEncoder(
            level=arch.get("feature_level", "all"), input_size=128,
            resize_to_input=arch.get("encoder_resize_to_input", False)).to(DEVICE)
        feature_dim = image_encoder.feature_dim
    model = ImagePFN(resolution=resolution, image_size=img_size,
                     input_patch_size=input_patch_size,
                     image_encoder=image_encoder, feature_dim=feature_dim,
                     e=arch["e"], h=arch["h"], l=arch["l"], a=arch["a"],
                     thinking_rows=arch["thinking_rows"],
                     residual_decay=arch["residual_decay"]).to(DEVICE)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Stage-1 loaded: resolution={resolution}, encoder={arch.get('image_encoder','patch')}")
    return model


def patch_loss(logits, batch, cfg):
    target = batch["qry_gt"]
    bce  = F.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(torch.sigmoid(logits.float()), target)
    return bce + cfg.train.dice_weight * dice


def train_epoch(model, loader, stage1, encoder, optimizers, cfg, epoch):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        # Augment whole images first; coarse pred + features are computed on them.
        if cfg.aug.enabled:
            img = batch["image"].unsqueeze(1).to(DEVICE)            # (B,1,1,H,W)
            ctx = batch["context_in"].to(DEVICE)                    # (B,K,1,H,W)
            imgs = torch.cat([ctx, img], dim=1)
            cout = batch["context_out"].to(DEVICE)
            msks = torch.cat([cout, batch["label"].unsqueeze(1).to(DEVICE)], dim=1)
            K = ctx.shape[1]
            imgs, msks = augment(imgs, msks, K, cfg.aug)
            batch = {**batch, "context_in": imgs[:, :K].cpu(), "image": imgs[:, K, 0:1].cpu(),
                     "context_out": msks[:, :K].cpu(), "label": msks[:, K, 0:1].cpu()}
        pb = build_patch_batch(batch, stage1, encoder, cfg, DEVICE)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = model(pb["sup_feat"], pb["sup_label"], pb["sup_ij"],
                           pb["qry_feat"], pb["qry_prior"], pb["qry_ij"], cfg.sample.grid_res)
            loss = patch_loss(logits, pb, cfg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()
        total += loss.item(); n += 1
        pbar.set_postfix(loss=f"{total/n:.4f}")
    return total / max(n, 1)


@torch.no_grad()
def run_eval(model, loader, stage1, encoder, lawa_queue, cfg, epoch):
    saved = lawa_average(lawa_queue, model, DEVICE)
    model.eval()
    d_err_unc, d_dice_unc, d_dice_unc_coarse = [], [], []
    cert_err_stage2, cert_err_coarse = [], []
    pbar = tqdm(loader, desc=f"eval e{epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        pb = build_patch_batch(batch, stage1, encoder, cfg, DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = model(pb["sup_feat"], pb["sup_label"], pb["sup_ij"],
                           pb["qry_feat"], pb["qry_prior"], pb["qry_ij"], cfg.sample.grid_res)
        pred = torch.sigmoid(logits.float())
        gt, coarse, unc = pb["qry_gt"], pb["qry_coarse"], pb["qry_is_uncertain"]
        B = gt.shape[0]
        for b in range(B):
            u = unc[b]
            if u.any():
                d_err_unc.append((coarse[b][u] - gt[b][u]).abs().mean().item()
                                 - (pred[b][u] - gt[b][u]).abs().mean().item())
                d_dice_unc.append(hard_dice(pred[b][u], (gt[b][u] >= 0.5).float()))
                d_dice_unc_coarse.append(hard_dice(coarse[b][u], (gt[b][u] >= 0.5).float()))
            c = ~u
            if c.any():
                cert_err_stage2.append((pred[b][c] - gt[b][c]).abs().mean().item())
                cert_err_coarse.append((coarse[b][c] - gt[b][c]).abs().mean().item())
    if saved is not None:
        model.load_state_dict(saved)

    nanmean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")
    metrics = {
        "epoch": epoch,
        "refine/delta_err_uncertain": nanmean(d_err_unc),           # >0 = improvement
        "refine/dice_uncertain_stage2": nanmean(d_dice_unc),
        "refine/dice_uncertain_coarse": nanmean(d_dice_unc_coarse),
        "refine/certain_err_stage2": nanmean(cert_err_stage2),
        "refine/certain_err_coarse": nanmean(cert_err_coarse),
    }
    tqdm.write(f"  [e{epoch}] Δerr(unc)={metrics['refine/delta_err_uncertain']:.4f}  "
               f"dice(unc) {metrics['refine/dice_uncertain_coarse']:.3f}→{metrics['refine/dice_uncertain_stage2']:.3f}")
    wandb.log(metrics)
    return metrics["refine/delta_err_uncertain"]


@hydra.main(config_path="../../../configs/experiment/2d", config_name="multilevel", version_base=None)
def main(cfg: DictConfig):
    import random
    random.seed(cfg.train.seed); np.random.seed(cfg.train.seed); torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high"); torch.backends.cudnn.benchmark = True

    print("Building data loaders...")
    train_loader = build_split_loader(cfg, "train", shuffle=True)
    val_loader   = build_split_loader(cfg, "val",   shuffle=False)

    stage1  = load_stage1(cfg)
    encoder = UniverSegFeatureEncoder(level=cfg.arch.feature_level, input_size=128).to(DEVICE)
    feature_dim = encoder.feature_dim

    model = PatchSetPFN(feature_dim=feature_dim, e=cfg.arch.e, h=cfg.arch.h, l=cfg.arch.l,
                        a=cfg.arch.a, thinking_rows=cfg.arch.thinking_rows,
                        residual_decay=cfg.arch.residual_decay, fourier_bands=cfg.arch.fourier_bands,
                        coarse_prior=cfg.arch.coarse_prior).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchSetPFN: {trainable:,} trainable params")

    if cfg.train.get("checkpoint", None):
        raw = torch.load(cfg.train.checkpoint, map_location="cpu", weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        msd = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(compat, strict=False)
        print(f"Warm-start PatchSetPFN: loaded {len(compat)}/{len(msd)} tensors")

    if cfg.arch.compile:
        model = torch.compile(model, dynamic=True)

    muon_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2 and "transformer" in n]
    adam_params = [p for n, p in model.named_parameters() if p.requires_grad and not (p.ndim == 2 and "transformer" in n)]
    opt_muon = Muon(muon_params, lr=cfg.train.muon_lr_scale * cfg.train.lr,
                    momentum=cfg.train.muon_momentum, weight_decay=cfg.train.muon_wd)
    opt_adam = torch.optim.AdamW(adam_params, lr=cfg.train.lr, weight_decay=cfg.train.adam_wd)
    def lr_lambda(epoch):
        if epoch < cfg.train.warmup_epochs:
            return (epoch + 1) / cfg.train.warmup_epochs
        t = (epoch - cfg.train.warmup_epochs) / max(cfg.train.epochs - cfg.train.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_adam, lr_lambda)
    optimizers = [opt_muon, opt_adam]

    lawa_queue = collections.deque(maxlen=cfg.train.lawa_k)
    run_name = cfg.wandb.name or (f"multilevel_{'coarse' if cfg.arch.coarse_prior else 'scratch'}"
                                  f"_R{cfg.sample.grid_res}_k{cfg.data.context_size}_l{cfg.arch.l}")
    wandb.init(project=cfg.wandb.project, name=run_name,
               config={"arch": dict(cfg.arch), "train": dict(cfg.train),
                       "data": dict(cfg.data), "sample": dict(cfg.sample)},
               mode="online" if cfg.wandb.enabled else "disabled")

    ckpt_dir = Path(cfg.eval.out_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = -1e9
    for epoch in tqdm(range(1, cfg.train.epochs + 1), desc="epochs", dynamic_ncols=True):
        loss = train_epoch(model, train_loader, stage1, encoder, optimizers, cfg, epoch)
        scheduler.step()
        wandb.log({"epoch": epoch, "train/loss": loss, "train/lr": scheduler.get_last_lr()[0]})
        lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs:
            delta = run_eval(model, val_loader, stage1, encoder, lawa_queue, cfg, epoch)
            if delta > best:
                best = delta
                saved = lawa_average(lawa_queue, model, DEVICE)
                torch.save({"model": model.state_dict(), "arch": dict(cfg.arch),
                            "sample": dict(cfg.sample), "image_size": cfg.data.image_size,
                            "context_size": cfg.data.context_size}, ckpt_dir / "best.pt")
                if saved:
                    model.load_state_dict(saved)
                tqdm.write(f"  [best] Δerr(unc)={best:.4f} → {ckpt_dir}/best.pt")

    wandb.log({"best_delta_err_uncertain": best})
    wandb.finish()
    print(f"\nDone. Best Δerr(uncertain): {best:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (1 epoch, tiny subset, no compile, wandb disabled)**

Run (GPU; takes a few minutes incl. UniverSeg load + first encode):
```bash
.venv311/bin/python experiments/2d/multilevel/train.py \
    train.epochs=1 data.max_train_samples=16 train.batch_size=4 \
    eval.max_per_label=2 train.workers=0 eval.workers=0 \
    arch.compile=false wandb.enabled=false
```
Expected: prints `Stage-1 loaded: …`, `PatchSetPFN: N trainable params`, runs one train epoch + eval, prints `[e1] Δerr(unc)=…  dice(unc) X→Y`, writes `results/2d/multilevel/<run>/best.pt`, ends with `Done. Best Δerr(uncertain): …`. No shape errors.

- [ ] **Step 3: Verify coarse_prior=false path also runs**

Run:
```bash
.venv311/bin/python experiments/2d/multilevel/train.py \
    train.epochs=1 data.max_train_samples=16 train.batch_size=4 \
    eval.max_per_label=2 train.workers=0 eval.workers=0 \
    arch.compile=false wandb.enabled=false arch.coarse_prior=false
```
Expected: same successful completion; run name contains `scratch`.

- [ ] **Step 4: Checkpoint**

End-to-end training runs. User may commit.

---

## Task 7: Log the experiment

**Files:**
- Modify: `docs/logs.md`

- [ ] **Step 1: Prepend a log entry**

Add under the top `# Change log` header in `docs/logs.md`:

```markdown
## 2026-06-14 — multilevel patch refinement (stage-2 PatchSetPFN)

New experiment `experiments/2d/multilevel/`. A frozen res-16 ImagePFN (stage 1) +
frozen UniverSeg encoder produce a coarse target prediction and res-32 features; we
sample 256 patches/image (192 closest-to-0.5 + 64 most-certain) — target→query,
context→support — and train `src/models/patchset_pfn.py:PatchSetPFN` (nanoTabPFN-shaped:
rows=patches, cols=[img|mask], 2-D Fourier PE, query-attends-to-support) to refine the
query patches. Metric: `refine/delta_err_uncertain` = |error| reduction on the uncertain
target region vs the stage-1 coarse value (with the 64 certain patches as a regression
check). `arch.coarse_prior` toggles using the coarse pred as the query prior.

Shared training utils (`Muon`, `augment`, `lawa_average`, `soft_dice_loss`) factored
out of `pfn_seg.py` into `experiments/2d/pfn_train.py`; both scripts import them.
Spec: `docs/superpowers/specs/2026-06-14-multilevel-patch-refinement-design.md`.
```

- [ ] **Step 2: Checkpoint**

Logged. User may commit.

---

## Self-Review (completed during planning)

**Spec coverage:** Pipeline (coarse→features→sample→refine) → Tasks 4, 6. PatchSetPFN + Fourier PE → Task 3. 256/image budget (192+64), target=query/context=support → Tasks 2, 4. Frozen encoder features → Tasks 4, 6. coarse_prior param → Tasks 3, 4, 5. Metrics (Δerr on uncertain, certain regression check) → Task 6. Refactor of shared utils → Task 1. Config → Task 5. Logging → Task 7. All covered.

**Placeholder scan:** No TBD/TODO; all code blocks are complete and runnable.

**Type consistency:** `build_patch_batch` keys (`sup_feat/sup_label/sup_ij/qry_feat/qry_prior/qry_ij/qry_gt/qry_coarse/qry_is_uncertain`) match `PatchSetPFN.forward(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij, grid_res)` and the train/eval call sites. `sample_patch_indices(values, n_uncertain, n_certain)`, `idx_to_ij(idx, grid_res)`, `gather_grid(x, idx)` signatures consistent across Tasks 2/4. `UniverSegFeatureEncoder(level, input_size, resize_to_input)` and `.feature_dim`/`forward(images, out_size)` match the existing module.

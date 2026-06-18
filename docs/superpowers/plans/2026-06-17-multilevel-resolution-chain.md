# Multilevel Resolution Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the 2-level (res-16 → res-32) patch refinement into an N-level coarse-to-fine chain (16→32→64→128) with per-level weights, detached gradients between levels, and chained "thinking" memory.

**Architecture:** A reusable per-hop unit `refine_level` (sample → gather → forward → composite) driven by a thin `run_chain` loop over a configurable resolution ladder. Each level is its own `PatchSetPFN` (an `nn.ModuleList`); levels train independently on the detached composite of the level below. Spec: `docs/superpowers/specs/2026-06-17-multilevel-resolution-chain-design.md`.

**Tech Stack:** PyTorch, Hydra/OmegaConf, the existing UniverSeg encoder + frozen res-16 `ImagePFN` + `PatchSetPFN`.

**Project conventions (IMPORTANT):**
- Run everything with `.venv311/bin/python` (the CUDA venv). No `uv`/`conda`/`pytest`.
- Tests are plain-python modules: `test_*` functions + a `if __name__ == "__main__":` runner that calls them and prints `ALL ... TESTS PASSED`. Match `experiments/2d/multilevel/test_sampling.py`.
- **Do NOT run `git add`/`git commit`.** Version control is the user's job. Each task ends at a passing verification; the user commits when they choose.

---

## File Structure

- `src/models/patchset_pfn.py` — **Modify.** `PatchSetPFN.forward` gains `return_thinking`.
- `experiments/2d/multilevel/pipeline.py` — **Modify.** Add `composite_predictions`, `refine_level`, `run_chain`; keep `coarse_predict`; retire `build_patch_batch`.
- `experiments/2d/multilevel/train.py` — **Modify.** Build models as `nn.ModuleList`; `train_epoch`/`run_eval` call `run_chain`; per-level losses; per-resolution metric ladder.
- `configs/experiment/2d/multilevel.yaml` — **Modify.** `sample.resolutions` + per-level budget/loss lists.
- `experiments/2d/multilevel/test_patchset.py` — **Modify.** Add `return_thinking` test.
- `experiments/2d/multilevel/test_pipeline.py` — **Modify.** Add `composite_predictions`, `refine_level`, `run_chain` tests.

---

## Task 1: `PatchSetPFN.forward` returns thinking

**Files:**
- Modify: `src/models/patchset_pfn.py:94-140`
- Test: `experiments/2d/multilevel/test_patchset.py`

- [ ] **Step 1: Write the failing test** — append to `experiments/2d/multilevel/test_patchset.py`

```python
def test_return_thinking_shape():
    import torch
    from src.models.patchset_pfn import PatchSetPFN
    torch.manual_seed(0)
    B, S, Q, Fdim, e, nthink = 2, 12, 6, 8, 16, 4
    m = PatchSetPFN(feature_dim=Fdim, e=e, h=32, l=2, a=2, thinking_rows=nthink,
                    mask_prior="scalar", mask_patch_size=1, stage1_dim=None,
                    query_self_attn=True)
    sup_feat = torch.randn(B, S, Fdim); sup_label = torch.rand(B, S)
    sup_ij = torch.randint(0, 8, (B, S, 2))
    qry_feat = torch.randn(B, Q, Fdim); qry_prior = torch.rand(B, Q)
    qry_ij = torch.randint(0, 8, (B, Q, 2))
    logits, think = m(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij,
                      grid_res=8, return_thinking=True)
    assert logits.shape == (B, Q), logits.shape
    assert think.shape == (B, nthink, e), think.shape
    # default path unchanged: returns only logits
    out = m(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij, grid_res=8)
    assert out.shape == (B, Q), out.shape
```

Add `test_return_thinking_shape()` to the `__main__` runner block.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_patchset.py`
Expected: FAIL — `TypeError: forward() got an unexpected keyword argument 'return_thinking'`.

- [ ] **Step 3: Implement** — edit `PatchSetPFN.forward` in `src/models/patchset_pfn.py`

Change the signature (line 94-95) and the return (line 139-140):

```python
    def forward(self, sup_feat, sup_label, sup_ij,
                qry_feat, qry_prior, qry_ij, grid_res, stage1_think=None,
                return_thinking=False):
```

```python
        q = x[:, sep_t:, 0, :]                  # query rows, img-col → (B,Q,e)
        out = self.decoder(q).squeeze(-1)       # (B,Q)
        if return_thinking:
            # post-transformer thinking rows, pooled over the 2 columns → (B, n_think, e)
            think = x[:, :self.thinking.n].mean(dim=2)
            return out, think
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_patchset.py`
Expected: PASS — prints `ALL ... TESTS PASSED`.

---

## Task 2: `composite_predictions` helper

**Files:**
- Modify: `experiments/2d/multilevel/pipeline.py` (add helper near `gather_grid` usage, after imports)
- Test: `experiments/2d/multilevel/test_pipeline.py`

- [ ] **Step 1: Write the failing test** — append to `experiments/2d/multilevel/test_pipeline.py`

```python
def test_composite_predictions():
    import torch
    import sys; sys.path.insert(0, "experiments/2d/multilevel")
    from pipeline import composite_predictions
    B, N, M = 2, 16, 5
    coarse = torch.rand(B, N)
    qidx = torch.stack([torch.randperm(N)[:M] for _ in range(B)])
    vals = torch.rand(B, M)
    out = composite_predictions(coarse, qidx, vals)
    assert out.shape == (B, N)
    for b in range(B):
        sel = set(qidx[b].tolist())
        for j in range(N):
            if j in sel:
                pos = (qidx[b] == j).nonzero()[0, 0]
                assert torch.allclose(out[b, j], vals[b, pos])     # overwritten
            else:
                assert torch.allclose(out[b, j], coarse[b, j])     # untouched
    assert out is not coarse                                       # no in-place mutation
```

Add `test_composite_predictions()` to the `__main__` runner.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: FAIL — `ImportError: cannot import name 'composite_predictions'`.

- [ ] **Step 3: Implement** — add to `experiments/2d/multilevel/pipeline.py`

```python
def composite_predictions(coarse_flat, qidx, vals):
    """(B,N) dense map + (B,M) indices + (B,M) values → (B,N) with vals scattered in.

    Returns a NEW tensor (coarse_flat is not mutated); unsampled cells keep coarse value."""
    refined = coarse_flat.clone()
    refined.scatter_(1, qidx, vals)
    return refined
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: PASS.

---

## Task 3: `refine_level` — one coarse-to-fine hop

**Files:**
- Modify: `experiments/2d/multilevel/pipeline.py` (add `refine_level`; move the body of `build_patch_batch` into it)
- Test: `experiments/2d/multilevel/test_pipeline.py`

**Interface (referenced by Task 4 — keep names exact):**
```python
refine_level(model, batch, feats, coarse_grid, prev_think, grid_res, s,
             source, stochastic, device) -> dict
# feats: (B, T, N, Cf) encoder features at grid_res, T = K+1 (contexts then target)
# coarse_grid: (B, N) detached prev prediction at grid_res (sampling map + query prior)
# prev_think: (B, n_think, e1) or None
# s: level config (has .n_total .n_fg_core .n_fg_core_ctx .tau .blur_sigma .floor
#    .temperature .mask_prior)
# returns dict keys: refined_grid (B,N), logits (B,M), qry_gt (B,M), qry_coarse (B,M),
#   qry_is_uncertain (B,M bool), qidx (B,M), this_think (B,n_think,e), gt_grid (B,N)
```

- [ ] **Step 1: Write the failing test** — append to `experiments/2d/multilevel/test_pipeline.py`

```python
def test_refine_level_shapes_and_composite():
    import torch
    import sys; sys.path.insert(0, "experiments/2d/multilevel")
    from pipeline import refine_level
    from src.models.patchset_pfn import PatchSetPFN
    from types import SimpleNamespace
    torch.manual_seed(0)
    dev = torch.device("cpu")
    B, K, R, Cf, e, nth = 2, 2, 8, 8, 16, 4
    N, T = R * R, K + 1
    H = 32                                   # native image size for this toy (p = H//R = 4)
    batch = {
        "image":       torch.rand(B, 1, H, H),
        "label":       (torch.rand(B, 1, H, H) > 0.5).float(),
        "context_in":  torch.rand(B, K, 1, H, H),
        "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float(),
    }
    feats = torch.randn(B, T, N, Cf)
    coarse_grid = torch.rand(B, N)
    model = PatchSetPFN(feature_dim=Cf, e=e, h=32, l=2, a=2, thinking_rows=nth,
                        mask_prior="scalar", mask_patch_size=H // R, stage1_dim=e,
                        query_self_attn=True)
    s = SimpleNamespace(n_total=10, n_fg_core=2, n_fg_core_ctx=4, tau=0.3,
                        blur_sigma=1.0, floor=0.01, temperature=1.0, mask_prior="scalar")
    prev_think = torch.randn(B, nth, e)
    out = refine_level(model, batch, feats, coarse_grid, prev_think, R, s,
                       "prev_pred", True, dev)
    assert out["logits"].shape == (B, s.n_total)
    assert out["refined_grid"].shape == (B, N)
    assert out["this_think"].shape == (B, nth, e)
    assert out["qidx"].shape == (B, s.n_total)
    # composite: unsampled cells equal coarse_grid; sampled cells equal sigmoid(logits)
    import torch as _t
    sel = _t.zeros(B, N, dtype=_t.bool).scatter_(1, out["qidx"], True)
    assert _t.allclose(out["refined_grid"][~sel], coarse_grid[~sel])
```

Add to the `__main__` runner.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: FAIL — `ImportError: cannot import name 'refine_level'`.

- [ ] **Step 3: Implement** — add `refine_level` to `experiments/2d/multilevel/pipeline.py`. Port the body of `build_patch_batch` (`pipeline.py:99-141`), replacing `coarse_flat`→`coarse_grid`, `R2`→`grid_res`, stage-1 think→`prev_think`, and scalar config→`s`:

```python
def refine_level(model, batch, feats, coarse_grid, prev_think, grid_res, s,
                 source, stochastic, device):
    """One coarse-to-fine hop at grid_res. See module docstring / spec."""
    label       = batch["label"].to(device)        # (B,1,H,W)
    context_out = batch["context_out"].to(device)  # (B,K,1,H,W)
    B, K = context_out.shape[0], context_out.shape[1]
    N = grid_res * grid_res
    M = s.n_total

    gt_grid = F.adaptive_avg_pool2d(label.float(), (grid_res, grid_res)).reshape(B, N)
    ctx_frac_grid = _grid_fractions(context_out, grid_res).reshape(B, K, N)  # true masks

    # ── Query (target) patches ──
    sampling_map = gt_grid if source == "ds_gt" else coarse_grid
    qidx, q_is_core, q_is_fg = sample_patches(
        sampling_map, M, s.tau, s.n_fg_core, s.blur_sigma, s.floor, grid_res,
        temperature=s.temperature, stochastic=stochastic)
    qry_feat   = gather_grid(feats[:, -1], qidx)                 # (B,M,Cf)
    qry_coarse = gather_grid(coarse_grid, qidx)
    qry_gt     = gather_grid(gt_grid, qidx)
    qry_ij     = idx_to_ij(qidx, grid_res)
    is_unc     = q_is_core & ~q_is_fg

    # ── Support (context) patches ──
    ctx_feat = feats[:, :K].reshape(B * K, N, feats.shape[-1])
    ctx_frac = ctx_frac_grid.reshape(B * K, N)
    sidx, _, _ = sample_patches(
        ctx_frac, M, s.tau, s.n_fg_core_ctx, s.blur_sigma, s.floor, grid_res,
        temperature=s.temperature, stochastic=stochastic)
    sup_feat = gather_grid(ctx_feat, sidx).reshape(B, K * M, feats.shape[-1])
    sup_ij   = idx_to_ij(sidx, grid_res).reshape(B, K * M, 2)

    # ── Mask-token: scalar or p×p tile ──
    if s.mask_prior == "patch":
        p = label.shape[-1] // grid_res
        ctx_tiles = torch.stack([_mask_tiles(context_out[:, k], grid_res, p) for k in range(K)],
                                dim=1).reshape(B * K, N, p * p)
        sup_label = gather_grid(ctx_tiles, sidx).reshape(B, K * M, p * p)
        coarse_tiles = _mask_tiles(coarse_grid.reshape(B, 1, grid_res, grid_res), grid_res, p)
        qry_prior = gather_grid(coarse_tiles, qidx)
    else:
        sup_label = gather_grid(ctx_frac, sidx).reshape(B, K * M)
        qry_prior = qry_coarse

    logits, this_think = model(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij,
                               grid_res, stage1_think=prev_think, return_thinking=True)
    refined_grid = composite_predictions(coarse_grid, qidx, torch.sigmoid(logits.float()))
    return {"refined_grid": refined_grid, "logits": logits, "qry_gt": qry_gt,
            "qry_coarse": qry_coarse, "qry_is_uncertain": is_unc, "qidx": qidx,
            "this_think": this_think, "gt_grid": gt_grid}
```

Note: `_mask_tiles` accepts `(B,1,grid,grid)` and upsamples to `grid*p` internally, so passing `coarse_grid.reshape(B,1,grid,grid)` is correct.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: PASS.

---

## Task 4: `run_chain` — the driver

**Files:**
- Modify: `experiments/2d/multilevel/pipeline.py` (add `run_chain`; retire `build_patch_batch`)
- Test: `experiments/2d/multilevel/test_pipeline.py`

**Interface:**
```python
run_chain(batch, stage1, encoder, models, cfg, source, stochastic, device)
    -> (outputs: list[dict], coarse_lr: (B, R0, R0))
# models: nn.ModuleList, one PatchSetPFN per hop (len == len(resolutions)-1)
# outputs[L] is the dict returned by refine_level for hop L (grid = resolutions[L+1])
```

- [ ] **Step 1: Write the failing test** — append to `experiments/2d/multilevel/test_pipeline.py`. This test stubs `stage1`/`encoder` so it needs no checkpoint.

```python
def test_run_chain_detaches_and_shapes():
    import torch
    import sys; sys.path.insert(0, "experiments/2d/multilevel")
    from pipeline import run_chain
    from src.models.patchset_pfn import PatchSetPFN
    from omegaconf import OmegaConf
    torch.manual_seed(0)
    dev = torch.device("cpu")
    B, K, H, Cf, e, nth = 2, 2, 32, 8, 16, 4
    R0, ladder = 16, [16, 32]                          # single hop → grid 32
    batch = {"image": torch.rand(B, 1, H, H),
             "label": (torch.rand(B, 1, H, H) > 0.5).float(),
             "context_in": torch.rand(B, K, 1, H, H),
             "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float()}

    class StubStage1:        # mimics ImagePFN: stage1(imgs, masks, sep=K, return_thinking=True)
        def __call__(self_, all_images, all_masks, sep, return_thinking=False):
            b = all_images.shape[0]
            logits = torch.rand(b, R0, R0)              # res-16 logits
            think = torch.randn(b, nth, e)
            return (logits, think) if return_thinking else logits
    def stub_encoder(images, grid):                    # encode_grid calls encoder(imgs, grid)
        bT = images.shape[0]
        return torch.randn(bT, Cf, grid, grid)

    cfg = OmegaConf.create({"sample": {
        "resolutions": ladder, "n_total": [10], "n_fg_core": [2], "n_fg_core_ctx": [4],
        "tau": 0.3, "blur_sigma": 1.0, "floor": 0.01, "temperature": 1.0,
        "mask_prior": "scalar"}, "data": {"image_size": H}})
    models = torch.nn.ModuleList([
        PatchSetPFN(feature_dim=Cf, e=e, h=32, l=2, a=2, thinking_rows=nth,
                    mask_prior="scalar", mask_patch_size=H // 32, stage1_dim=e,
                    query_self_attn=True)])
    outputs, coarse_lr = run_chain(batch, StubStage1(), stub_encoder, models, cfg,
                                   "prev_pred", True, dev)
    assert len(outputs) == 1
    assert outputs[0]["refined_grid"].shape == (B, 32 * 32)
    assert coarse_lr.shape == (B, R0, R0)
```

The stub must match how `pipeline.coarse_predict` and `encode_grid` call their args — verify against `pipeline.py:15-34`. Add to the `__main__` runner.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: FAIL — `ImportError: cannot import name 'run_chain'`.

- [ ] **Step 3: Implement** — add `run_chain` to `experiments/2d/multilevel/pipeline.py`, then delete `build_patch_batch`.

```python
def _level_cfg(cfg, L):
    """Per-hop config namespace, reading list entries from cfg.sample at index L."""
    from types import SimpleNamespace
    s = cfg.sample
    pick = lambda v: v[L] if isinstance(v, (list, ListConfig)) else v
    return SimpleNamespace(
        n_total=pick(s.n_total), n_fg_core=pick(s.n_fg_core),
        n_fg_core_ctx=pick(s.get("n_fg_core_ctx", s.n_fg_core)),
        tau=pick(s.tau), blur_sigma=pick(s.blur_sigma), floor=pick(s.floor),
        temperature=pick(s.temperature), mask_prior=cfg.arch.mask_prior)


def run_chain(batch, stage1, encoder, models, cfg, source, stochastic, device):
    """Coarse-to-fine chain. Returns (outputs list per hop, coarse_lr (B,R0,R0))."""
    resolutions = list(cfg.sample.resolutions)
    image       = batch["image"].to(device)
    context_in  = batch["context_in"].to(device)
    context_out = batch["context_out"].to(device)
    B, K = context_in.shape[0], context_in.shape[1]

    all_images = torch.cat([context_in, image.unsqueeze(1)], dim=1)
    all_masks  = torch.cat([context_out, torch.zeros_like(image.unsqueeze(1))], dim=1)

    R0 = resolutions[0]
    # Frozen stage-1 + encoder: no grad (preserves the old @torch.no_grad build_patch_batch
    # behaviour — only the per-level PatchSetPFNs train).
    with torch.no_grad():
        _, coarse_lr, think = coarse_predict(stage1, all_images, all_masks, K, R0)  # p_lowres @R0
    prev_dense = coarse_lr.reshape(B, R0 * R0)
    prev_think = think
    prev_res = R0

    outputs = []
    for L, grid in enumerate(resolutions[1:]):
        coarse_grid = F.interpolate(prev_dense.reshape(B, 1, prev_res, prev_res),
                                    size=(grid, grid), mode="bilinear",
                                    align_corners=False).reshape(B, grid * grid)
        with torch.no_grad():
            feats = encode_grid(encoder, all_images, grid)
        s = _level_cfg(cfg, L)
        hop = refine_level(models[L], batch, feats, coarse_grid, prev_think, grid, s,
                           source, stochastic, device)
        outputs.append(hop)
        prev_dense = hop["refined_grid"].detach()
        prev_think = hop["this_think"].detach()
        prev_res = grid
    return outputs, coarse_lr
```

Add `from omegaconf import ListConfig` near the top imports of `pipeline.py`. Delete the now-unused `build_patch_batch` function.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv311/bin/python experiments/2d/multilevel/test_pipeline.py`
Expected: PASS — `ALL ... TESTS PASSED`.

---

## Task 5: Config — resolution ladder + per-level lists

**Files:**
- Modify: `configs/experiment/2d/multilevel.yaml:9-24` (the `sample:` block) and `train:` block.

- [ ] **Step 1: Edit `sample:` block** — replace the scalar `grid_res`/`n_total`/`n_fg_core`/`n_fg_core_ctx` with the ladder + per-hop lists. Keep `tau`/`blur_sigma`/`floor`/`temperature` as shared scalars.

```yaml
sample:
  resolutions: [16, 32, 64, 128]   # resolutions[0] MUST equal stage-1 resolution (asserted)
  grid_res: 32        # DEPRECATED single-level alias; kept only for the diagnostic scripts
  n_total:       [256, 256, 256]   # per hop  (len == len(resolutions)-1)
  n_fg_core:     [64, 64, 64]
  n_fg_core_ctx: [160, 160, 160]
  tau: 0.30
  blur_sigma: 1.0
  floor: 0.005
  temperature: 1.0
  eval_deterministic: true
  train: prev_pred
  eval:  prev_pred
```

- [ ] **Step 2: Add per-hop loss weights** — in the `train:` block add:

```yaml
  loss_weights: [1.0, 1.0, 1.0]   # per hop (len == len(resolutions)-1)
```

- [ ] **Step 3: Verify config parses**

Run:
```
.venv311/bin/python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/experiment/2d/multilevel.yaml'); print(list(c.sample.resolutions), list(c.sample.n_total), list(c.train.loss_weights))"
```
Expected: `[16, 32, 64, 128] [256, 256, 256] [1.0, 1.0, 1.0]`

---

## Task 6: `train.py` — models as ModuleList + `train_epoch` via `run_chain`

**Files:**
- Modify: `experiments/2d/multilevel/train.py` model construction (`main`, ~lines 270-296) and `train_epoch` (lines 114-148).

- [ ] **Step 1: Build models as an `nn.ModuleList`** — in `main`, replace the single `PatchSetPFN(...)` construction with one per hop and assert the ladder seed:

```python
    resolutions = list(cfg.sample.resolutions)
    assert resolutions[0] == int(round(stage1.N ** 0.5)), \
        f"resolutions[0]={resolutions[0]} must equal stage-1 res {int(round(stage1.N ** 0.5))}"
    stage1_dim = stage1.thinking.tokens.shape[-1] if cfg.arch.use_stage1_thinking else None
    # Chained thinking: hop L>0 receives the previous PatchSetPFN's thinking (dim e).
    model = nn.ModuleList([
        PatchSetPFN(feature_dim=feature_dim, e=cfg.arch.e, h=cfg.arch.h, l=cfg.arch.l,
                    a=cfg.arch.a, thinking_rows=cfg.arch.thinking_rows,
                    residual_decay=cfg.arch.residual_decay, fourier_bands=cfg.arch.fourier_bands,
                    mask_prior=cfg.arch.mask_prior,
                    mask_patch_size=cfg.data.image_size // grid,
                    stage1_dim=(stage1_dim if L == 0 else cfg.arch.e),
                    query_self_attn=cfg.arch.query_self_attn).to(DEVICE)
        for L, grid in enumerate(resolutions[1:])])
    print(f"PatchSetPFN chain: {len(model)} hops, "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
```

The Muon/Adam param split (`p.ndim == 2 and "transformer" in n`) and `torch.compile` still work on the `ModuleList` unchanged (names become `0.transformer...`). The warm-start checkpoint loader stays as-is (it filters by shape).

- [ ] **Step 2: Rewrite `train_epoch`'s forward/loss** — replace the `build_patch_batch` + single-forward block (lines 132-142) with `run_chain` + summed per-level loss:

```python
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            # run_chain internally no_grads the frozen stage-1/encoder; the per-level
            # PatchSetPFN forwards run here under autocast + grad.
            outputs, _ = run_chain(batch, stage1, encoder, model, cfg, cfg.sample.train,
                                   stochastic=True, device=DEVICE)
            weights = list(cfg.train.loss_weights)
            loss = sum(w * patch_loss(o["logits"], {"qry_gt": o["qry_gt"]}, cfg)
                       for w, o in zip(weights, outputs))
        loss.backward()
```

Update the import at the top of `train.py`: `from pipeline import run_chain` (was `build_patch_batch`). Note `patch_loss` already reads `batch["qry_gt"]`, so passing `{"qry_gt": o["qry_gt"]}` works unchanged.

- [ ] **Step 2b: Fix the `torch.compile` block for the ModuleList** — the existing `train.py:289-290` compiles a single module; a `ModuleList` is not callable. Replace it with per-submodule compile:

```python
    if cfg.arch.compile:
        model = nn.ModuleList([torch.compile(m, dynamic=True) for m in model])
```

(Validate with `arch.compile=false` first; the Muon filter `"transformer" in n` still matches the `_orig_mod.` -prefixed names, and the warm-start loader already strips that prefix.)

- [ ] **Step 3: Verify a single training step runs** (1 epoch, tiny)

Run:
```
.venv311/bin/python experiments/2d/multilevel/train.py train.epochs=1 data.dataset=busi \
  data.max_train_samples=8 eval.max_per_label=2 arch.compile=false wandb.enabled=false 2>&1 | tail -6
```
Expected: training bar runs without error and prints an eval line (eval is updated in Task 7; until then it may error in `run_eval` — if so, proceed to Task 7 and re-run). The train loss should print a finite number.

---

## Task 7: `train.py` — `run_eval` resolution-ladder metrics

**Files:**
- Modify: `experiments/2d/multilevel/train.py` `run_eval` (lines 168-260).

- [ ] **Step 1: Rewrite `run_eval` to loop the chain.** Replace the per-batch forward + the hard-coded r16/r32 metric block with a per-hop ladder. Key structure:

```python
@torch.no_grad()
def run_eval(model, loader, stage1, encoder, lawa_queue, cfg, epoch):
    saved = lawa_average(lawa_queue, model, DEVICE)
    for m in model: m.eval()
    H = cfg.data.image_size
    resolutions = list(cfg.sample.resolutions)
    hops = resolutions[1:]                       # grids of each trained level
    # per-resolution Dice accumulators (true-res, like the 2-level version)
    per_ds = {r: defaultdict(list) for r in resolutions}     # r0 = stage-1, others = hops
    # per-hop refinement diagnostics (s2 = this hop, s1 = its upsampled input)
    acc = {L: {k: [] for k in ("derr", "dd", "sdd")} for L in range(len(hops))}

    for batch in loader:
        if batch is None: continue
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            outputs, coarse_lr = run_chain(batch, stage1, encoder, model, cfg,
                                           cfg.sample.eval,
                                           stochastic=not cfg.sample.eval_deterministic,
                                           device=DEVICE)
        B = coarse_lr.shape[0]
        for b in range(B):
            ds_name   = batch["dataset"][b]
            gt_native = batch["label"][b, 0]
            R0 = resolutions[0]
            gt_r0 = (downsample_mask(gt_native, R0) >= 0.5).float()
            per_ds[R0][ds_name].append(hard_dice(coarse_lr[b].cpu(), gt_r0))
            prev_grid = coarse_lr[b].reshape(1, 1, R0, R0).float()
            for L, grid in enumerate(hops):
                o = outputs[L]
                refined = o["refined_grid"][b]                      # (grid²,)
                gt_g = (o["gt_grid"][b] >= 0.5).float()
                per_ds[grid][ds_name].append(hard_dice(refined.cpu(), gt_g.cpu()))
                # refine delta vs this hop's input (coarse_grid), on sampled cells
                up = F.interpolate(prev_grid, size=(grid, grid), mode="bilinear",
                                   align_corners=False).reshape(-1)
                qg, qi = o["qry_gt"][b], o["qidx"][b]
                pred_q = torch.sigmoid(o["logits"][b].float())
                coarse_q = up[qi]
                acc[L]["derr"].append((coarse_q - qg).abs().mean().item()
                                      - (pred_q - qg).abs().mean().item())
                acc[L]["dd"].append(hard_dice(pred_q, (qg >= 0.5).float())
                                    - hard_dice(coarse_q, (qg >= 0.5).float()))
                acc[L]["sdd"].append(soft_dice(pred_q, qg) - soft_dice(coarse_q, qg))
                prev_grid = refined.reshape(1, 1, grid, grid).float()
    if saved is not None:
        model.load_state_dict(saved)

    def nanmean(xs):
        v = [x for x in xs if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")
    flat = lambda d: [x for sc in d.values() for x in sc if not np.isnan(x)]

    metrics = {"epoch": epoch}
    for r in resolutions:
        metrics[f"dice_r{r}/mean"] = (float(np.mean(flat(per_ds[r])))
                                      if flat(per_ds[r]) else float("nan"))
    metrics["dice/mean"] = metrics[f"dice_r{resolutions[-1]}/mean"]   # final = native
    for L in range(len(hops)):
        metrics[f"refine/hop{L}/delta_err"]       = nanmean(acc[L]["derr"])
        metrics[f"refine/hop{L}/dice_delta"]      = nanmean(acc[L]["dd"])
        metrics[f"refine/hop{L}/soft_dice_delta"] = nanmean(acc[L]["sdd"])
    for r in resolutions:
        for k, v in per_ds[r].items():
            metrics[f"dice/dataset_r{r}/{k}"] = nanmean(v)

    tqdm.write(f"  [e{epoch}] " + "  ".join(
        f"r{r}={metrics[f'dice_r{r}/mean']:.4f}" for r in resolutions))
    wandb.log(metrics)
    for m in model: m.train()
    return metrics["dice/mean"]
```

Notes: `lawa_average`/`lawa_queue.append` operate on `model.state_dict()` — an `nn.ModuleList` has a normal `state_dict`, so those lines in `main` are unchanged. The final hop's grid is `H` (128), so `dice_r128/mean` is computed directly at native res (no upsample), and `dice/mean` aliases it.

- [ ] **Step 2: Verify eval runs and logs the ladder** (1 epoch, full 4-level ladder, tiny data)

Run:
```
.venv311/bin/python experiments/2d/multilevel/train.py train.epochs=1 data.dataset=busi \
  data.max_train_samples=8 eval.max_per_label=2 arch.compile=false wandb.enabled=false 2>&1 | tail -6
```
Expected: prints a line like `[e1] r16=0.xxxx  r32=0.xxxx  r64=0.xxxx  r128=0.xxxx` and `Best dice/mean=...`, no error.

---

## Task 8: End-to-end + backward-compatibility verification

**Files:** none (verification only).

- [ ] **Step 1: Full 4-level smoke run** (a few epochs, one dataset)

Run:
```
.venv311/bin/python experiments/2d/multilevel/train.py train.epochs=2 data.dataset=busi \
  data.max_train_samples=64 eval.max_per_label=4 arch.compile=false wandb.enabled=false 2>&1 | tail -8
```
Expected: 2 epochs complete; the `r16…r128` ladder prints each epoch; `dice/mean` is finite; a `best.pt` is saved. Sanity: `r16` ≈ the stage-1 baseline (~0.7–0.9 on busi), later levels are learning (may be low early but should not be NaN).

- [ ] **Step 2: Backward-compat — single hop reduces to the old 2-level behavior**

Run:
```
.venv311/bin/python experiments/2d/multilevel/train.py train.epochs=1 data.dataset=busi \
  data.max_train_samples=16 eval.max_per_label=4 arch.compile=false wandb.enabled=false \
  'sample.resolutions=[16,32]' 'sample.n_total=[256]' 'sample.n_fg_core=[64]' \
  'sample.n_fg_core_ctx=[160]' 'train.loss_weights=[1.0]' 2>&1 | tail -5
```
Expected: runs with exactly one hop; prints `r16=… r32=…`; `dice/mean == dice_r32/mean`. Confirms the chain generalizes the prior single-hop pipeline.

- [ ] **Step 3: Run the unit test suite**

Run:
```
.venv311/bin/python experiments/2d/multilevel/test_patchset.py && \
.venv311/bin/python experiments/2d/multilevel/test_pipeline.py
```
Expected: both print `ALL ... TESTS PASSED`.

- [ ] **Step 4: Update the change log** — append a dated entry to `docs/logs.md` summarizing the chain (resolutions ladder, per-level weights, detached training, chained thinking, metric ladder). No commit.

---

## Notes for the implementer

- The diagnostic scripts (`plot_sampling.py`) still use the old scalar `sample.grid_res`; the deprecated alias in the config keeps them working. Do not wire them to `resolutions`.
- `coarse_predict` already returns `(p, p_lowres, think)` (added earlier) — `run_chain` uses `p_lowres` as the res-16 seed.
- If `arch.compile=true`, compile is applied in `main` as today; with a `ModuleList` you may compile each submodule or leave compile off for first validation. Validate with `arch.compile=false` first.

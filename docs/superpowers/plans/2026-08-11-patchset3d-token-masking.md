# PatchSet3D Random Token Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add training-only random masking of feature-patch tokens (support + query) to `PatchSet3D`, so the attention head learns to segment from incomplete input.

**Architecture:** SimMIM-style *in-place* masking — selected tokens have both their image and mask/occupancy columns replaced by a single learned `[MASK]` parameter, keeping the fixed R³ token count (so the compiled transformer and RoPE-by-row-index are untouched) and leaving masked cells present for a future reconstruction loss. Two ratios (support, query) default to 0.0 → behavior identical to today. Gated on `self.training` so eval/`predict` never mask.

**Tech Stack:** PyTorch, Hydra config, pytest. All model changes confined to `src/models/patchset3d.py`; config in `configs/experiment/3d/model/patchset3d.yaml`; wiring in `experiments/3d/train.py::build_model`.

## Global Constraints

- Both new ratios default to `0.0`; with defaults, forward output and the val/checkpoint path must be unchanged from today (repo relies on eval reproducibility).
- Masking active ONLY when `self.training` is True (eval/`predict`/`validate_mean` run under `net.eval()`).
- Masks are replaced in place, never dropped — token sequence length stays `[thinking | K·N support | N query]` so `torch.compile` and RoPE indexing are unaffected.
- Repo guideline: write tests only when necessary; keep the masking unit tests minimal. Log changes in `docs/logs.md`.
- Design spec: `docs/superpowers/specs/2026-08-11-patchset3d-token-masking-design.md`.

---

### Task 1: Token masking in PatchSet3D (model + unit tests)

**Files:**
- Modify: `src/models/patchset3d.py` — `PatchSet3D.__init__` (add args + `mask_token`), `_tokens` (accept `mask`), add `_sample_mask`, `_attn` (sample + thread masks, return them), `forward` (surface masks in dict).
- Test: `tests/test_patchset3d.py` (append three tests).

**Interfaces:**
- Consumes: existing `PatchSet3D.__init__(..., feat_norm="context")` tail; `_tokens(self, feat, occ, ijk)`; `_attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None) -> logit`; `forward(...) -> {"final_logit": logit}`.
- Produces:
  - `PatchSet3D.__init__(..., token_mask_ratio_support: float = 0.0, token_mask_ratio_query: float = 0.0)`
  - `self.mask_token: nn.Parameter` shape `(2, e)`
  - `_tokens(self, feat, occ, ijk, mask=None)` — `mask` optional bool `(B, M)`
  - `_sample_mask(self, B, M, ratio, device) -> Optional[BoolTensor(B, M)]` (None when not training or ratio ≤ 0)
  - `_attn(...) -> (logit, mask_support, mask_query)` where `mask_*` are `Optional[BoolTensor]`
  - `forward(...) -> {"final_logit", "mask_support", "mask_query"}` (`mask_*` None when inactive)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patchset3d.py`:

```python
def test_token_masking_noop_when_ratios_zero():
    """Default ratios (0.0): masks are None even in train mode; logit shape unchanged."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    m.train()
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    assert out["mask_support"] is None and out["mask_query"] is None


def test_token_masking_active_in_train():
    """ratio>0 under train(): masks have right shape, ~right fraction, grad flows to mask_token."""
    torch.manual_seed(0)
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   token_mask_ratio_support=0.5, token_mask_ratio_query=0.5)
    m.train()
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)   # N=64, support M=128, query M=64
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    assert out["mask_support"].shape == (2, 128) and out["mask_support"].dtype == torch.bool
    assert out["mask_query"].shape == (2, 64)
    assert abs(out["mask_support"].float().mean().item() - 0.5) < 0.15
    out["final_logit"].mean().backward()
    assert m.mask_token.grad is not None and torch.isfinite(m.mask_token.grad).all()


def test_token_masking_off_in_eval():
    """Even with ratio>0, eval mode never masks (eval/predict reproducibility)."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   token_mask_ratio_support=0.5, token_mask_ratio_query=0.5)
    m.eval()
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["mask_support"] is None and out["mask_query"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patchset3d.py -k token_masking -v`
Expected: FAIL — `PatchSet3D.__init__` has no `token_mask_ratio_support` (TypeError), and `out` has no `"mask_support"` key.

- [ ] **Step 3: Add constructor args + `mask_token` parameter**

In `PatchSet3D.__init__`, add two params to the signature (after `feat_norm: str = "context",`):

```python
        feat_norm: str = "context",
        token_mask_ratio_support: float = 0.0,
        token_mask_ratio_query: float = 0.0,
```

Store them and allocate the learned mask token. Put this right after `self.feat_norm = feat_norm` (near line 183):

```python
        self.feat_norm = feat_norm
        # SimMIM-style in-place token masking (training only; both default 0.0 = off). A masked
        # cell has BOTH its image and mask/occupancy columns replaced by mask_token, keeping the
        # R³ token count intact (compiled transformer + RoPE-by-index unaffected) and leaving the
        # cell in the sequence for a future reconstruction loss. See
        # docs/superpowers/specs/2026-08-11-patchset3d-token-masking-design.md.
        self.token_mask_ratio_support = float(token_mask_ratio_support)
        self.token_mask_ratio_query = float(token_mask_ratio_query)
        self.mask_token = nn.Parameter(torch.zeros(2, e))   # row 0 = image col, row 1 = mask col
        nn.init.normal_(self.mask_token, std=0.02)
```

- [ ] **Step 4: Add `_sample_mask` and thread `mask` through `_tokens`**

Change `_tokens` (currently near line 225) to accept an optional `mask` and replace in place BEFORE the positional encoding:

```python
    def _tokens(self, feat, occ, ijk, mask=None):
        img = self.img_embed(feat)
        msk = self.mask_embed(occ)
        if mask is not None:                                # SimMIM in-place [MASK] replacement
            m = mask.unsqueeze(-1)                          # (B,M,1) bool
            img = torch.where(m, self.mask_token[0], img)
            msk = torch.where(m, self.mask_token[1], msk)
        if self.pos is not None:                            # additive Fourier PE (non-RoPE mode)
            pos = self.pos(ijk, self.resolution)
            img = img + pos                                 # masked token keeps its position
            msk = msk + pos
        return torch.stack([img, msk], dim=2)               # (B,M,2,e)
```

Add `_sample_mask` as a new method (place it just above `_attn`, near line 278):

```python
    def _sample_mask(self, B, M, ratio, device):
        """Random per-cell boolean mask (B,M) at the given ratio; None when not training or
        ratio<=0. Independent Bernoulli per cell (in-place SimMIM masking, not token-dropping)."""
        if not self.training or ratio <= 0.0:
            return None
        return torch.rand(B, M, device=device) < ratio
```

- [ ] **Step 5: Sample + thread masks in `_attn`, return them**

Edit `_attn` (near line 278). Add mask sampling at the top and pass into `_tokens`; change the return to a tuple.

Add after `B, N = sup_feat.shape[0], self.N`:

```python
        dev = sup_feat.device
        mask_support = self._sample_mask(B, K * N, self.token_mask_ratio_support, dev)
        mask_query = self._sample_mask(B, N, self.token_mask_ratio_query, dev)
```

Change the two `_tokens` calls (near lines 286-287) to pass the masks (note: `_feat_norm`
still runs on the unmasked features just above these lines — intentional, stats stay a
property of the real context):

```python
        sup_tok = self._tokens(sup_feat, sup_occ, sup_ijk, mask=mask_support)   # (B,S,2,e)
        qry_tok = self._tokens(qry_feat, qry_occ, qry_ijk, mask=mask_query)     # (B,Q,2,e)
```

Change the final return (currently `return self._tile_logits(self.decoder(q))`, near line 308):

```python
        logit = self._tile_logits(self.decoder(q))          # (B,1,Rd,Rd,Rd)
        return logit, mask_support, mask_query
```

- [ ] **Step 6: Surface masks in `forward`**

Edit `forward` (near line 317). Change the `_attn` call + return:

```python
        logit, mask_support, mask_query = self._attn(
            sup_feat, qry_feat, self._occupancy(context_out), K, spacing=spacing)
        return {"final_logit": logit, "mask_support": mask_support, "mask_query": mask_query}
```

(`_native_logit` / `train_forward` / `predict` read `["final_logit"]` unchanged.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset3d.py -k token_masking -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Run the full model test file for no regressions**

Run: `python -m pytest tests/test_patchset3d.py tests/test_patchset3d_rope.py -v`
Expected: PASS (existing tests unaffected — default ratios keep `mask_*` None and output shape identical).

- [ ] **Step 9: Commit**

```bash
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "$(printf 'feat(patchset3d): in-place random token masking (support+query)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: Config knobs + build_model wiring + logs

**Files:**
- Modify: `configs/experiment/3d/model/patchset3d.yaml` (add two arch keys).
- Modify: `experiments/3d/train.py::build_model` (pass the two keys into the arch dict).
- Modify: `docs/logs.md` (append a log entry).

**Interfaces:**
- Consumes: `PatchSet3D.__init__(..., token_mask_ratio_support, token_mask_ratio_query)` from Task 1; `build_model`'s existing `arch = {...}` dict pattern using `a.get("key", default)`.
- Produces: `arch.token_mask_ratio_support` / `arch.token_mask_ratio_query` reachable via Hydra overrides, stored in the checkpoint's `arch` block (rebuilt by `eval.py`, harmless there — eval never trains).

- [ ] **Step 1: Add the two arch keys to the config**

In `configs/experiment/3d/model/patchset3d.yaml`, add after the `feat_norm:` block (near line 31), keeping the surrounding comment style:

```yaml
  token_mask_ratio_support: 0.0   # fraction of K·N context tokens masked per step (SimMIM,
                                  # train only). 0.0 = off (identical to no masking).
  token_mask_ratio_query: 0.0     # fraction of N target tokens masked per step (train only).
                                  # Masked query cells are still segmented from context.
```

- [ ] **Step 2: Wire the keys into `build_model`**

In `experiments/3d/train.py`, inside `build_model`'s `patchset3d` branch, add two entries to the `arch` dict (alongside `"feat_norm": a.get("feat_norm", "context"),`, near line 261):

```python
            "feat_norm": a.get("feat_norm", "context"),
            "token_mask_ratio_support": a.get("token_mask_ratio_support", 0.0),
            "token_mask_ratio_query": a.get("token_mask_ratio_query", 0.0),
```

- [ ] **Step 3: Verify config resolves and reaches the model**

Run:
```bash
python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('configs/experiment/3d/model/patchset3d.yaml')
assert c.arch.token_mask_ratio_support == 0.0
assert c.arch.token_mask_ratio_query == 0.0
print('config ok:', c.arch.token_mask_ratio_support, c.arch.token_mask_ratio_query)
"
```
Expected: prints `config ok: 0.0 0.0` with no assertion error.

- [ ] **Step 4: Append a log entry**

Add to `docs/logs.md` (follow the file's existing dated-entry format):

```markdown
## 2026-08-11 — PatchSet3D random token masking

Added SimMIM-style in-place token masking to `PatchSet3D` (`src/models/patchset3d.py`):
`arch.token_mask_ratio_support` / `arch.token_mask_ratio_query` (both default 0.0 = off)
randomly replace whole tokens (image + mask columns) with a learned `mask_token` during
training only (`self.training`), keeping the R³ token count so the compiled transformer and
RoPE are unaffected. `forward` now also returns `mask_support` / `mask_query` (None when off)
as the hook for a future auxiliary reconstruction loss. Eval/predict never mask.
Spec: docs/superpowers/specs/2026-08-11-patchset3d-token-masking-design.md.
```

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/3d/model/patchset3d.yaml experiments/3d/train.py docs/logs.md
git commit -m "$(printf 'feat(patchset3d): config + build_model wiring for token masking\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Self-Review

**Spec coverage:**
- In-place SimMIM masking, learned `mask_token (2,e)`, position preserved → Task 1 Steps 3-4. ✓
- Both support + query, separate ratios → Task 1 Steps 3-5; Task 2 Steps 1-2. ✓
- Training-only gate → Task 1 Step 4 (`_sample_mask`), tested Step 1 (`test_token_masking_off_in_eval`). ✓
- `forward` returns `mask_support`/`mask_query` (Phase-2 hook) → Task 1 Step 6. ✓
- `feat_norm` computed before masking → Task 1 Step 5 note. ✓
- Config defaults 0.0 → no-op → Task 2 Step 1; tested Task 1 Step 1 (`test_token_masking_noop_when_ratios_zero`). ✓
- Checkpoint arch stores keys / eval rebuilds harmlessly → Task 2 Interfaces + Step 2. ✓
- Reconstruction loss explicitly Phase 2 (not built) → not in plan, correct. ✓
- Log entry → Task 2 Step 4. ✓

**Placeholder scan:** No TBD/TODO; all code steps contain concrete code. ✓

**Type consistency:** `token_mask_ratio_support`/`token_mask_ratio_query` (float), `mask_token` `(2,e)`, `_sample_mask -> Optional[BoolTensor(B,M)]`, `_attn -> (logit, mask_support, mask_query)`, `forward` dict keys `final_logit`/`mask_support`/`mask_query` — consistent across Tasks 1-2 and the tests. ✓

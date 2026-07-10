# PatchSetCNN cross-level thinking memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In refine (multi-resolution) mode, let the refine pass of `PatchSetCNN` attend to the coarse pass's thinking rows, injected as detached memory tokens.

**Architecture:** The coarse `_attn` pass optionally returns its post-transformer thinking rows `(B, n_think, e)`. The refine `_attn` pass optionally receives them (detached) as extra rows prepended into the support block plus a learned `mem_type` marker, so every row attends to them. Gated by `arch.refine_memory` (default off); the shared `_attn` covers both refine modes.

**Tech Stack:** PyTorch, Hydra config, pytest. Run Python via `.venv_nero/bin/python`.

## Global Constraints

- Python interpreter: `.venv_nero/bin/python` (torch 2.6.0+cu124). Run tests with `.venv_nero/bin/python -m pytest`.
- `arch.refine_memory` defaults to `false`; existing configs and checkpoints MUST load and behave identically (`.get(..., False)` everywhere it is read).
- Memory rows are `.detach()`ed before the refine pass consumes them.
- No projection: memory rows are already in `e`-space; only a learned `mem_type` (`e`,) marker is added.
- `refine_memory` is inert for single-level (`len(resolutions)==1`) models — no assert, silently unused.
- Do NOT auto-commit. This repo's convention is that the user handles all git commits. Where a step says "checkpoint," stage nothing — just stop for review.

---

### Task 1: Constructor flag + `mem_type` parameter

**Files:**
- Modify: `src/models/patchset_cnn.py` (`PatchSetCNN.__init__`, roughly lines 104–181)
- Test: `tests/test_patchset_cnn_refine.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PatchSetCNN(..., refine_memory: bool = False)`; attribute `self.refine_memory: bool`; parameter `self.mem_type: nn.Parameter` of shape `(e,)` present **only** when `refine_memory=True`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_patchset_cnn_refine.py`. Note `_model` builds with `e=32`, so `mem_type` shape is `(32,)`.

```python
def test_refine_memory_default_off_no_param():
    m = _model([8, 16])
    assert m.refine_memory is False
    assert not hasattr(m, "mem_type")
    assert "mem_type" not in dict(m.named_parameters())


def test_refine_memory_on_creates_mem_type():
    m = PatchSetCNN(image_size=32, resolution=8, enc_dims=[16], e=32, h=64, l=1, a=2,
                    thinking_rows=1, resolutions=[8, 16], refine_memory=True)
    assert m.refine_memory is True
    assert m.mem_type.shape == (32,)
    assert "mem_type" in dict(m.named_parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py::test_refine_memory_on_creates_mem_type -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'refine_memory'`.

- [ ] **Step 3: Add the constructor parameter and attribute**

In `PatchSetCNN.__init__`, add `refine_memory: bool = False,` to the signature (next to `refine_mode`). After `self.refine_mode = refine_mode` (line ~133), add:

```python
        self.refine_memory = refine_memory
```

Then, after `self.thinking = ThinkingRows(thinking_rows, e)` (line ~175), add:

```python
        # Cross-level memory: the refine pass attends to the coarse pass's (detached)
        # thinking rows, prepended as extra rows plus this learned type marker. Only
        # created when enabled, so default checkpoints gain zero parameters. Inert for
        # single-level models (no coarse pass to summarize).
        if refine_memory:
            self.mem_type = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.mem_type, std=0.02)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py -k refine_memory -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Checkpoint** — stop for review. Do not commit (user handles git).

---

### Task 2: `_attn` memory injection + refine-forward threading

**Files:**
- Modify: `src/models/patchset_cnn.py` (`_attn` ~219–268; `_segment` ~206–217; `_refine_reencode` ~287–308; `_refine_encode_once` ~310–338)
- Test: `tests/test_patchset_cnn_refine.py`

**Interfaces:**
- Consumes: `self.refine_memory`, `self.mem_type` (from Task 1); `self.thinking.n`.
- Produces:
  - `_attn(self, sup_feat, qry_feat, sup_occ, K, mem=None, return_think=False)` — returns `logit (B,1,R,R)` when `return_think=False`, else `(logit, think)` where `think` is `(B, n_think, e)`.
  - `_segment(self, image, context_in, context_out, mem=None, return_think=False)` — same return contract, forwarding both args to `_attn`.
  - Both refine forwards, when `self.refine_memory`, capture `coarse_think` and pass `mem=coarse_think.detach()` into the refine pass.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_patchset_cnn_refine.py`. A helper builds an enabled model:

```python
def _mem_model(refine_mode="reencode"):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=32, resolution=8, enc_dims=[16], e=32, h=64, l=1, a=2,
                       thinking_rows=1, resolutions=[8, 16], refine_mode=refine_mode,
                       refine_memory=True)


@pytest.mark.parametrize("mode", ["reencode", "encode_once"])
def test_refine_memory_shapes_unchanged(mode):
    m = _mem_model(mode)
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)
    assert out["refine_logit"].shape == (2, 1, 8, 8)


@pytest.mark.parametrize("mode", ["reencode", "encode_once"])
def test_refine_memory_coarse_unaffected(mode):
    # Coarse head is a plain full-image segment: memory must not leak into it.
    m = _mem_model(mode)
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert torch.allclose(out["final_logit"], m._segment(img, cin, cout), atol=1e-5)


@pytest.mark.parametrize("mode", ["reencode", "encode_once"])
def test_mem_type_receives_gradient(mode):
    # If refine routes through the memory rows, mem_type gets a gradient.
    m = _mem_model(mode)
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    out["refine_logit"].mean().backward()
    assert m.mem_type.grad is not None
    assert m.mem_type.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["reencode", "encode_once"])
def test_memory_is_detached(mode):
    # Capture the `mem` tensor passed into the refine _attn; it must carry no grad history.
    m = _mem_model(mode)
    img, cin, cout = _batch()
    seen = []
    orig = m._attn
    def spy(*a, **kw):
        seen.append(kw.get("mem", a[4] if len(a) > 4 else None))
        return orig(*a, **kw)
    m._attn = spy
    m(img, context_in=cin, context_out=cout)
    mems = [x for x in seen if x is not None]
    assert len(mems) == 1                      # exactly the refine pass carries memory
    assert mems[0].grad_fn is None and not mems[0].requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py -k "refine_memory_shapes or mem_type_receives or memory_is_detached or coarse_unaffected" -v`
Expected: FAIL — `_attn()`/`_segment()` reject the `mem`/`return_think` kwargs (`TypeError`).

- [ ] **Step 3: Add `mem`/`return_think` to `_attn`**

Change the `_attn` signature (line ~219) to:

```python
    def _attn(self, sup_feat, qry_feat, sup_occ, K, mem=None, return_think=False):
```

Replace the row-assembly + thinking block. Currently (lines ~252–254):

```python
        x = torch.cat([sup_tok, qry_tok], dim=1)                          # (B,S+Q,2,e)

        x, sep_t = self.thinking(x, K * N)
```

with:

```python
        # Optional cross-level memory rows (coarse thinking), prepended into the support
        # block so every row (support + query) attends to them. mem: (B,T1,e), detached.
        sep = K * N
        rows = [sup_tok, qry_tok]
        if mem is not None:
            T1 = mem.shape[1]
            m = (mem + self.mem_type).unsqueeze(2).expand(mem.shape[0], T1, 2, mem.shape[-1])
            rows = [m] + rows                                             # [memory | support | query]
            sep += T1
        x = torch.cat(rows, dim=1)                                        # (B, (T1+)S+Q, 2, e)

        x, sep_t = self.thinking(x, sep)      # -> [thinking | memory | support | query]
```

At the end of `_attn`, replace `return logit` (line ~268) with:

```python
        if return_think:
            return logit, x[:, :self.thinking.n].mean(dim=2)             # (B, n_think, e)
        return logit
```

(The existing `attn_mask` block is unchanged: it uses `sep_t` and `x.shape[1]`, which already account for the prepended memory rows.)

- [ ] **Step 4: Add passthrough to `_segment`**

Change `_segment` (line ~206) signature and its `_attn` call:

```python
    def _segment(self, image, context_in, context_out, mem=None, return_think=False):
```

and the final line (line ~217) from:

```python
        return self._attn(sup_feat, qry_feat, self._occupancy(context_out), K)
```

to:

```python
        return self._attn(sup_feat, qry_feat, self._occupancy(context_out), K,
                          mem=mem, return_think=return_think)
```

- [ ] **Step 5: Thread memory through `_refine_reencode`**

In `_refine_reencode`, replace the coarse call (line ~296):

```python
        coarse = self._segment(image, context_in, context_out)           # (B,1,T,T)
```

with:

```python
        if self.refine_memory:
            coarse, coarse_think = self._segment(image, context_in, context_out,
                                                 return_think=True)       # (B,1,T,T), (B,n,e)
        else:
            coarse, coarse_think = self._segment(image, context_in, context_out), None
```

and replace the refine call (line ~305):

```python
        refine = self._segment(tgt_img, ctx_img, ctx_msk)                # (B,1,T,T), same weights
```

with:

```python
        mem = coarse_think.detach() if coarse_think is not None else None
        refine = self._segment(tgt_img, ctx_img, ctx_msk, mem=mem)       # (B,1,T,T), same weights
```

- [ ] **Step 6: Thread memory through `_refine_encode_once`**

In `_refine_encode_once`, replace the coarse call (line ~326):

```python
        coarse = self._attn(sup_c, qry_c, self._occupancy(context_out), K)   # (B,1,T,T)
```

with:

```python
        if self.refine_memory:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K,
                                              return_think=True)             # (B,1,T,T), (B,n,e)
        else:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K), None
```

and replace the refine call (line ~335):

```python
        refine = self._attn(sup_r, qry_r, self._occupancy(ctx_msk), K)   # (B,1,T,T), same weights
```

with:

```python
        mem = coarse_think.detach() if coarse_think is not None else None
        refine = self._attn(sup_r, qry_r, self._occupancy(ctx_msk), K, mem=mem)  # same weights
```

- [ ] **Step 7: Run the new tests**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py -k "refine_memory_shapes or mem_type_receives or memory_is_detached or coarse_unaffected" -v`
Expected: PASS (8 passed — 4 tests × 2 modes).

- [ ] **Step 8: Run the whole file (backward-compat regression)**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py -v`
Expected: PASS — all pre-existing tests (flag-off path) still pass unchanged.

- [ ] **Step 9: Checkpoint** — stop for review. Do not commit.

---

### Task 3: Config flag + checkpoint rebuild + logs

**Files:**
- Modify: `experiments/2d/train.py` (`build_model`, `arch` dict ~123–133)
- Modify: `configs/experiment/2d/train_base.yaml` (`arch:` block ~15–23)
- Modify: `docs/logs.md`
- Test: `tests/test_patchset_cnn_refine.py`

**Interfaces:**
- Consumes: `PatchSetCNN(..., refine_memory=...)` (Task 1).
- Produces: checkpoint `arch` dict carries `refine_memory`; config exposes `arch.refine_memory`.

- [ ] **Step 1: Write the failing test**

This test asserts the checkpoint `arch` dict round-trips the flag. Add to `tests/test_patchset_cnn_refine.py`:

```python
def test_build_model_arch_dict_carries_refine_memory():
    import sys as _sys
    from pathlib import Path
    _sys.path.insert(0, str(Path("experiments/2d").resolve()))
    from omegaconf import OmegaConf
    from train import build_model
    cfg = OmegaConf.create({
        "model": "patchset_cnn",
        "data": {"image_size": 32},
        "arch": {"resolution": 8, "enc_dims": [16], "e": 32, "h": 64, "l": 1, "a": 2,
                 "thinking_rows": 1, "residual_decay": 0.95, "resolutions": [8, 16],
                 "refine_memory": True},
    })
    model, name, meta = build_model(cfg)
    assert meta["arch"]["refine_memory"] is True
    assert model.refine_memory is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py::test_build_model_arch_dict_carries_refine_memory -v`
Expected: FAIL — `KeyError: 'refine_memory'` (not in the built `arch` dict).

- [ ] **Step 3: Add the key to `build_model`'s arch dict**

In `experiments/2d/train.py`, inside the `patchset_cnn` branch of `build_model`, add to the `arch` dict (after the `"refine_mode"` entry, line ~132):

```python
            "refine_memory": a.get("refine_memory", False),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py::test_build_model_arch_dict_carries_refine_memory -v`
Expected: PASS.

- [ ] **Step 5: Add the config flag**

In `configs/experiment/2d/train_base.yaml`, in the `arch:` block (after `compile: true`, line ~23), add:

```yaml
  refine_memory: false   # refine pass attends to detached coarse thinking rows (multi-res only)
```

- [ ] **Step 6: Verify config resolves and Hydra accepts the key**

Run: `.venv_nero/bin/python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/experiment/2d/train_base.yaml'); print('refine_memory' in c.arch, c.arch.refine_memory)"`
Expected output: `True False`

- [ ] **Step 7: Add a change-log entry**

Append to `docs/logs.md` (top, under `# Change log`) a dated entry summarizing: refine pass now optionally attends to the coarse pass's detached thinking rows via a prepended memory token (`mem_type`), gated by `arch.refine_memory` (default off), covering both refine modes; mirrors `multilevel/`'s `stage1_think` but with shared weights and a type-token-only adapter (no projection).

- [ ] **Step 8: Full test run**

Run: `.venv_nero/bin/python -m pytest tests/test_patchset_cnn_refine.py -v`
Expected: PASS (all tests).

- [ ] **Step 9: Checkpoint** — stop for review. Do not commit.

---

## Self-Review

**Spec coverage:**
- Detach → Task 2 Step 5/6 (`.detach()`) + `test_memory_is_detached`. ✓
- Type-token-only adapter → Task 1 `mem_type`, Task 2 `mem + self.mem_type`, no Linear. ✓
- Opt-in flag default off → Task 1 signature default, Task 3 config + `.get(...,False)`. ✓
- Both refine modes → Task 2 Steps 5 & 6, tests parametrized over both. ✓
- `_attn` produce/consume contract → Task 2 Step 3/4. ✓
- Checkpoint rebuild → Task 3 Step 3 + test. ✓
- Single-level inert → constructor only creates param; refine forwards not reached for single level (no assert). ✓
- Compile note → no code change needed (dynamic=True handles the extra rows); documented in spec. ✓
- Tests: backward-compat (Task 2 Step 8 regression + `test_refine_memory_default_off_no_param`), shapes (`test_refine_memory_shapes_unchanged`), wiring (`test_mem_type_receives_gradient`), detach (`test_memory_is_detached`), coarse-unaffected (`test_refine_memory_coarse_unaffected`). ✓

**Placeholder scan:** none — all steps carry concrete code/commands.

**Type consistency:** `_attn(..., mem=None, return_think=False)` and `_segment(..., mem=None, return_think=False)` signatures match across Tasks 2–3; `mem_type` name consistent Tasks 1–2; `refine_memory` key consistent Tasks 1–3.

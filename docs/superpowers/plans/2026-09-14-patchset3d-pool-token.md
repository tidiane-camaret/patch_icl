# PatchSet3D Pool Token Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an IRIS-style foreground-masked pooling token (`T_f`) to `PatchSet3D` — one extra per-volume prefix row (K support + 1 query) computed as a mask-weighted average of `arch.fine_decode`'s finest unpooled encoder features, gated by a new `arch.pool_token` flag.

**Architecture:** `forward` extends `fine_decode`'s row selection from query-only to all `K+1` volumes when `pool_token=True`, then a new `_pool_tokens` method computes the masked average per volume (support: real GT mask; query: soft prior/support-mean, both at fine resolution — not routed through the coarse `R³`-tiled occupancy). `_attn` projects and inserts the result as extra prefix rows using the exact same mechanism `cascade_regs`/`mem` already use (prepended before `ThinkingRows`, generic `n_extra` accounting) — so `register_routed`'s block mask, `_rope`, `_grid_tokens`, and `context_id_embed`'s per-cell `repeat_interleave` are all completely untouched; the pool rows just reuse `ctx_id`/`qry_id`/`slot_pos` tagging the same way real content rows already do.

**Tech Stack:** PyTorch, Hydra config, pytest. All model changes confined to `src/models/patchset3d.py`; config in `configs/experiment/3d/model/patchset3d.yaml`; wiring in `experiments/3d/train.py::build_model`.

**Spec:** `docs/superpowers/specs/2026-09-14-patchset3d-pool-token-design.md`

## Global Constraints

- `arch.pool_token` defaults to `False` — with the default, forward output must be
  byte-identical (or within fp noise) to today's model; no new params (`pool_proj`,
  `pool_type`) allocated; `forward`'s `fine_rows` selection stays query-only.
- `arch.pool_token=True` **requires** `arch.fine_decode=True` (assert) — needs unpooled
  per-stage maps.
- `arch.pool_token=True` is **assert-incompatible** with `arch.register_routed=True` — same
  reason `cascade_registers` already is (block-mask partitioning assumes no prefix rows besides
  thinking rows).
- Pool tokens pool over `min(self.fine_stage)` (the finest requested fine-decode stage) — no
  new stage-selection config knob.
- The query's fine-resolution pooling mask is derived **directly** from `query_prior`/
  `context_out` at native resolution (via `F.interpolate` to the pool stage's side), **not**
  by upsampling the coarse `R³`-tiled `qry_occ`/`sup_occ` representation.
- `_decode`'s `fine` argument must stay query-only (`(B, Cf, S,S,S)` per stage) even when
  `pool_token=True` extends fine-map extraction to all `K+1` volumes — re-slice back down
  after pooling.
- Pool rows are extra prefix rows (mirrors `cascade_regs`), never an extra per-cell column —
  `register_routed`, `_rope`, `_grid_tokens`, `context_id_embed`'s `repeat_interleave` are not
  modified anywhere in this plan.
- Repo guideline: write tests only when necessary; keep tests focused. Log changes in
  `docs/logs.md`.

---

### Task 1: Pool-token computation + integration in `PatchSet3D`

**Files:**
- Modify: `src/models/patchset3d.py` — `PatchSet3D.__init__` (add `pool_token` arg + asserts +
  `pool_proj`/`pool_type`), new `_pool_tokens` method, `_attn` (add `pool_feat` param +
  insertion block), `forward` (extend `fine_rows` selection, call `_pool_tokens`, re-slice
  `fine`, thread `pool_feat` into `_attn`).
- Test: `tests/test_patchset3d.py` (append).

**Interfaces:**
- Consumes: existing `PatchSet3D.__init__(..., decoder_dim: int = 64)` tail (signature end);
  `self.fine_stage: tuple[int,...]` (always set); `self.encoder.fine_stage_channels(stage) ->
  int`; existing `_attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None, query_prior=None,
  cascade_regs=None) -> (q, mask_support, mask_query, regs)`; existing `forward`'s `fine_rows`
  selection (`rows = torch.arange(B, device=x.device) * T + K`) and `self._encode(x, spacing,
  fine_rows=rows)` call.
- Produces:
  - `PatchSet3D.__init__(..., pool_token: bool = False)`
  - `self.pool_token: bool`, `self._pool_stage: int` (only set when `pool_token=True`)
  - `self.pool_proj: nn.Linear`, `self.pool_type: nn.Parameter` shape `(e,)` (only allocated
    when `pool_token=True`)
  - `_pool_tokens(self, fine_finest: Tensor[B*T,Cf,S,S,S], context_out: Tensor[B,K,D,H,W],
    query_prior: Tensor[B,1,D,H,W] | None, B: int, K: int, T: int) -> Tensor[B,K+1,Cf]`
  - `_attn(..., pool_feat: Tensor[B,K+1,Cf] | None = None)` — external return type unchanged
  - `forward`'s external signature/return dict unchanged

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patchset3d.py`:

```python
# --- arch.pool_token: IRIS-style foreground-masked pooling token (finest fine_decode stage) ---

_FINE_KW = dict(image_size=(16, 16, 16), fine_decode=True, fine_stage=1)


def test_pool_token_default_off():
    """Default (pool_token=False): no new params, forward shape/behavior unchanged, fine map
    stays query-only."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   **_FINE_KW)
    assert not hasattr(m, "pool_proj") and not hasattr(m, "pool_type")
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)


def test_pool_token_forward_shape_and_backward():
    """pool_token=True: output shape unchanged; gradient reaches pool_proj/pool_type."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   pool_token=True, **_FINE_KW)
    assert m.pool_proj.in_features == 8      # dims[1] (fine_stage=1 -> enc_dims[1])
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    out["final_logit"].mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"


def test_pool_token_changes_output_vs_off():
    """Sanity: with the same seed, pool_token=True must produce different logits than
    pool_token=False."""
    torch.manual_seed(0)
    kw = dict(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2, **_FINE_KW)
    m_off = PatchSet3D(pool_token=False, **kw)
    torch.manual_seed(0)
    m_on = PatchSet3D(pool_token=True, **kw)
    m_off.eval(); m_on.eval()
    torch.manual_seed(1)
    img, cin, cout = _dummy_batch(S=16)
    out_off = m_off(img, context_in=cin, context_out=cout)["final_logit"]
    out_on = m_on(img, context_in=cin, context_out=cout)["final_logit"]
    assert out_off.shape == out_on.shape
    assert not torch.allclose(out_off, out_on)


def test_pool_token_requires_fine_decode():
    try:
        PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                  pool_token=True, fine_decode=False)
        assert False, "should have raised"
    except AssertionError as exc:
        assert "fine_decode" in str(exc)


def test_pool_token_rejects_register_routed():
    try:
        PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                  pool_token=True, register_routed=True, **_FINE_KW)
        assert False, "should have raised"
    except AssertionError as exc:
        assert "register_routed" in str(exc)


def test_pool_token_fine_map_stays_query_only_for_decode():
    """_decode must still receive query-only fine maps (B,Cf,S,S,S), not (B*T,...) -- the
    easiest place for an off-by-one/wrong-slice bug to hide."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   pool_token=True, **_FINE_KW)
    captured = {}
    orig_decode = m._decode

    def spy(q, fine=None):
        captured["fine_shapes"] = [f.shape for f in fine]
        return orig_decode(q, fine)
    m._decode = spy
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)
    m(img, context_in=cin, context_out=cout, mode="train")
    assert captured["fine_shapes"] == [(2, 8, 8, 8, 8)]   # (B, Cf, S, S, S), B=2 not B*T


def test_pool_token_with_context_id_embed_and_cascade_registers_and_mask_slots():
    """Combined smoke test: context_id_embed/cascade_registers/mask_slots tagging all apply
    correctly to pool rows alongside real content rows, and everything still trains."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   pool_token=True, context_id_embed=True, cascade_registers=True,
                   mask_slots=2, **_FINE_KW)
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)
    prev_regs = torch.randn(2, 2, 32)
    out = m(img, context_in=cin, context_out=cout, cascade_regs=prev_regs)
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    out["final_logit"].mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"


def test_pool_tokens_all_background_support_mask_no_nan():
    """An all-zero support mask for one context volume must not produce NaN/Inf (den clamp)."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   pool_token=True, **_FINE_KW)
    img, cin, _ = _dummy_batch(B=2, K=2, S=16)
    cout = torch.zeros(2, 2, 16, 16, 16, dtype=torch.long)     # all background
    out = m(img, context_in=cin, context_out=cout, mode="train")["final_logit"]
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patchset3d.py -k pool_token -v`
Expected: FAIL — `PatchSet3D.__init__` has no `pool_token` argument (TypeError).

- [ ] **Step 3: Add the `pool_token` constructor arg**

In `PatchSet3D.__init__`'s signature, add one param right after `decoder_dim: int = 64,`
(the last existing param, end of the signature):

```python
        decoder_dim: int = 64,
        pool_token: bool = False,
    ):
```

- [ ] **Step 4: Add the `pool_token` setup block in `__init__`**

Insert this block right after the `else: raise ValueError(f"arch.decoder {self.decoder_kind!r}
(fine_filter | conv)")` line that closes the `fine_decode` if/else (immediately before the
`# (i,j,k) lattice, row-major over R³` comment):

```python
        # pool_token (IRIS-style T_f): foreground-masked average of fine-resolution image
        # features, one extra prefix row per volume (K support + 1 query) -- inserted the same
        # way arch.cascade_registers' carried memory is (see _attn), not as an extra per-cell
        # token, so register_routed/_rope/_grid_tokens/context_id_embed's N-cells-per-volume
        # invariant is untouched. See docs/superpowers/specs/2026-09-14-patchset3d-pool-token-
        # design.md.
        self.pool_token = bool(pool_token)
        assert not (self.pool_token and not self.fine_decode), (
            "arch.pool_token=True requires arch.fine_decode=True (needs unpooled per-stage maps)")
        assert not (self.pool_token and self.register_routed), (
            "arch.pool_token=True adds K+1 extra prefix rows -- arch.register_routed's "
            "block-mask partitioning assumes no prefix rows besides thinking rows (same reason "
            "cascade_registers is incompatible)")
        if self.pool_token:
            self._pool_stage = min(self.fine_stage)
            self.pool_proj = nn.Linear(self.encoder.fine_stage_channels(self._pool_stage), e)
            self.pool_type = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.pool_type, std=0.02)
```

- [ ] **Step 5: Add the `_pool_tokens` method**

Add this new method right after `_expand` (before `_attn`):

```python
    def _pool_tokens(self, fine_finest, context_out, query_prior, B, K, T):
        """fine_finest: (B*T, Cf, S, S, S) -- ALL volumes' finest requested-stage map (forward's
        `fine` indexed at self._pool_stage, before it's re-sliced back to query-only). Returns
        (B, K+1, Cf): support-major (index 0..K-1) then query (index K), raw foreground-masked
        -average feature vectors -- NOT yet projected to e (projection + tagging happens in
        _attn, mirroring how cascade_regs is projected there via cascade_proj). See
        docs/superpowers/specs/2026-09-14-patchset3d-pool-token-design.md."""
        S = fine_finest.shape[-1]
        Cf = fine_finest.shape[1]
        feat = fine_finest.reshape(B, T, Cf, S, S, S)
        sup_feat, qry_feat = feat[:, :K], feat[:, K:K + 1]      # (B,K,Cf,...), (B,1,Cf,...)

        sup_mask = F.interpolate(
            context_out.reshape(B * K, 1, *context_out.shape[-3:]).float(),
            size=(S, S, S), mode="trilinear", align_corners=False).reshape(B, K, S, S, S)
        if query_prior is not None:
            qry_mask = F.interpolate(query_prior.float(), size=(S, S, S), mode="trilinear",
                                     align_corners=False)                   # (B,1,S,S,S)
        else:
            qry_mask = sup_mask.mean(dim=1, keepdim=True)                  # (B,1,S,S,S)

        def masked_avg(f, m):
            w = m.unsqueeze(2)                                # (B,n,1,S,S,S)
            num = (f * w).sum(dim=(-3, -2, -1))
            den = w.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
            return num / den                                   # (B,n,Cf)

        return torch.cat([masked_avg(sup_feat, sup_mask), masked_avg(qry_feat, qry_mask)],
                         dim=1)
```

- [ ] **Step 6: Thread `pool_feat` through `_attn`**

Change `_attn`'s signature (currently `def _attn(self, sup_feat, qry_feat, sup_occ, K,
spacing=None, query_prior=None, cascade_regs=None):`) to add the new param:

```python
    def _attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None, query_prior=None,
             cascade_regs=None, pool_feat=None):
```

Insert this block right after the existing `cascade_regs` block (after `sep += n_extra` and
before `x, sep_t = self.thinking(x, sep)`):

```python
        if pool_feat is not None:
            assert self.pool_token, "pool_feat given but arch.pool_token=False on this model"
            pool = self.pool_proj(pool_feat) + self.pool_type       # (B,K+1,e)
            if self.context_id_embed:
                ctx_tag = torch.cat([
                    self.ctx_id(torch.arange(K, device=pool.device)).unsqueeze(0).expand(B, -1, -1),
                    self.qry_id.view(1, 1, -1).expand(B, 1, -1)], dim=1)   # (B,K+1,e)
                pool = pool + ctx_tag
            if self.mask_slots >= 2:
                gt_tag = self._slot_pos_vec(self._mask_content_index["gt"], pool.device, pool.dtype)
                pred_tag = self._slot_pos_vec(self._mask_content_index["pred"], pool.device, pool.dtype)
                pool = pool + torch.cat([gt_tag.expand(K, -1), pred_tag.unsqueeze(0)],
                                        dim=0).unsqueeze(0)
            pool = pool.unsqueeze(2).expand(-1, -1, x.shape[2], -1)   # (B,K+1,c,e)
            x = torch.cat([pool, x], dim=1)
            n_extra += pool.shape[1]
            sep += pool.shape[1]
        x, sep_t = self.thinking(x, sep)
```

(`n_extra` and `sep` already exist at this point from the preceding `cascade_regs` block, which
still sets `n_extra = mem.shape[1]` unchanged — the new block's `+=` correctly accumulates on
top whether or not `cascade_regs` ran, since `n_extra` starts at `0` either way.)

- [ ] **Step 7: Extend `forward`'s fine-map extraction and thread `pool_feat` through**

Replace `forward`'s body from `fine = None` through the `_attn` call with:

```python
        fine = None
        pool_feat = None
        if self.fine_decode:
            qidx = torch.arange(B, device=x.device) * T + K
            if self.pool_token:
                rows = torch.arange(B * T, device=x.device)   # every volume (support + query)
            else:
                # The query volume is the last of T, so its flat rows are b*T + K; only those
                # keep an unpooled map (one encoder pass, the rest freed with the stage list).
                rows = qidx
            feat_map, fine = self._encode(x, spacing, fine_rows=rows)
            if self.pool_token:
                finest_idx = self.fine_stage.index(self._pool_stage)
                pool_feat = self._pool_tokens(fine[finest_idx], context_out, query_prior,
                                              B, K, T)
                fine = tuple(f[qidx] for f in fine)     # re-slice back to query-only for _decode
        else:
            feat_map = self._encode(x, spacing)                        # (B*T,Cf,R,R,R)
        sup_feat, qry_feat = self._grid_tokens(feat_map, B, T, K)
        q, mask_support, mask_query, regs = self._attn(
            sup_feat, qry_feat, self._occupancy(context_out), K, spacing=spacing,
            query_prior=query_prior, cascade_regs=cascade_regs, pool_feat=pool_feat)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset3d.py -k pool_token -v`
Expected: PASS (8 tests).

- [ ] **Step 9: Run the full model test suite for no regressions**

Run: `python -m pytest tests/test_patchset3d.py tests/test_patchset3d_rope.py tests/test_pfn_seg_2d.py -v`
Expected: PASS (every existing test, unaffected by the `pool_token=False` default).

- [ ] **Step 10: Commit**

```bash
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "$(printf 'feat(patchset3d): IRIS-style foreground-masked pool token (arch.pool_token)\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Bp6fgSQtz7XQfHXHWuRsrw')"
```

---

### Task 2: Config knob, `build_model` wiring, and logs

**Files:**
- Modify: `configs/experiment/3d/model/patchset3d.yaml` (add one arch key).
- Modify: `experiments/3d/train.py::build_model` (pass the key into the arch dict).
- Modify: `docs/logs.md` (append a log entry).

**Interfaces:**
- Consumes: `PatchSet3D.__init__(..., pool_token)` from Task 1; `build_model`'s existing
  `arch = {...}` dict pattern using `a.get("key", default)`.
- Produces: `arch.pool_token` reachable via Hydra overrides, stored in the checkpoint's `arch`
  block (rebuilt by `eval.py`, harmless there — eval never trains).

- [ ] **Step 1: Add the arch key to the config**

In `configs/experiment/3d/model/patchset3d.yaml`, add after the `compress_layers:` line
(the last `arch.*` entry before `compile:`):

```yaml
  pool_token: false            # IRIS-style foreground-masked pooling token per volume (K+1
                               # extra prefix rows, one per support + query). Pools
                               # arch.fine_decode's finest requested stage -- requires
                               # arch.fine_decode=true. Extends fine-map extraction to all K+1
                               # volumes (memory cost scales with K, vs. query-only otherwise).
                               # Incompatible with arch.register_routed (same reason
                               # cascade_registers is).
```

- [ ] **Step 2: Wire the key into `build_model`**

In `experiments/3d/train.py`, inside `build_model`'s `patchset3d` branch, add one entry to the
`arch` dict (alongside `"compress_layers": a.get("compress_layers", 1),`):

```python
            "compress_layers": a.get("compress_layers", 1),
            "pool_token": a.get("pool_token", False),
```

- [ ] **Step 3: Verify config resolves and reaches the model**

Run:
```bash
python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('configs/experiment/3d/model/patchset3d.yaml')
assert c.arch.pool_token == False
print('config ok:', c.arch.pool_token)
"
```
Expected: prints `config ok: False` with no assertion error.

- [ ] **Step 4: Add a log entry at the top**

`docs/logs.md` is reverse-chronological (newest entry first, right after the `# Change log`
header on line 1). Insert this new entry as the new first entry — right after line 1's `#
Change log` header and its blank line 2, BEFORE the current top entry (currently `## 2026-09-14
— PatchSet3D IRIS-style sequence compression...` at line 3) — do NOT append at the physical end
of the file:

```markdown
## 2026-09-14 — PatchSet3D IRIS-style pooling token (`arch.pool_token`)

Added an optional foreground-masked pooling token to `PatchSet3D` (`src/models/patchset3d.py`),
mirroring Iris's `T_f` (`docs/methods/iris.md` §4.1: `T_f = Pool(Upsample(F_s) ⊙ y_s)`). Unlike
a version pooling over the coarse `R³` token grid, this pools over `arch.fine_decode`'s finest
requested unpooled stage (`min(self.fine_stage)`), matching the property Iris's own ablation
credits for a large small-object Dice gain (masking AFTER upsampling, not before). One extra
prefix row per volume (`K` support + 1 query, `K+1` total) computed as a mask-weighted average
of fine image features — support: real GT mask; query: soft prior/support-mean, both derived
directly from native-resolution `context_out`/`query_prior` (not routed through the coarse
`R³`-tiled occupancy `qry_occ` already used elsewhere). Requires `arch.fine_decode=true`
(extends its row selection from query-only to all `K+1` volumes — a real `K`-scaling memory
cost, accepted deliberately rather than compromise on resolution); assert-incompatible with
`arch.register_routed` for the same reason `arch.cascade_registers` already is. Inserted as
extra prefix rows via the exact mechanism `cascade_regs`/`mem` already use, so
`register_routed`'s block mask, `_rope`, `_grid_tokens`, and `context_id_embed`'s
`repeat_interleave` are completely untouched; pool rows reuse `ctx_id`/`qry_id`/`slot_pos`
tagging the same way real content rows do. `arch.pool_token` defaults to `false` (byte-identical
to today). Spec: docs/superpowers/specs/2026-09-14-patchset3d-pool-token-design.md.
```

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/3d/model/patchset3d.yaml experiments/3d/train.py docs/logs.md
git commit -m "$(printf 'feat(patchset3d): config + build_model wiring for pool_token\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Bp6fgSQtz7XQfHXHWuRsrw')"
```

---

## Self-Review

**Spec coverage:**
- Fine-feature extraction extended to all K+1 volumes, gated on `pool_token` → Task 1 Step 7. ✓
- Masked pooling (support real GT, query prior/support-mean fallback derived at native
  resolution, not via coarse `qry_occ`) → Task 1 Step 5 (`_pool_tokens`). ✓
- Projection + broadcast-to-both-columns + insertion as extra prefix rows (mirrors
  `cascade_regs`) → Task 1 Step 6. ✓
- `pool_type` learned tag, `ctx_id`/`qry_id` reuse, `slot_pos` gt/pred reuse → Task 1 Step 6. ✓
- `fine_decode` hard prerequisite assert, `register_routed` incompatibility assert → Task 1
  Step 4, tested Task 1 Step 1 (`test_pool_token_requires_fine_decode`,
  `test_pool_token_rejects_register_routed`). ✓
- `_decode` still receives query-only `fine` (re-slicing) → Task 1 Step 7, tested
  (`test_pool_token_fine_map_stays_query_only_for_decode`). ✓
- `cascade_registers`/`seq_compress`/`transformer_rope` compatibility (no code changes needed
  per spec) → verified by not touching `_rope`, Stage C's `kv_expand` slice, or
  `cascade_registers`' own block anywhere in this plan; combined smoke test in Task 1 Step 1
  (`test_pool_token_with_context_id_embed_and_cascade_registers_and_mask_slots`) exercises
  `cascade_registers` + `context_id_embed` + `mask_slots` together. ✓
- Config default off, no-op guarantee → Task 2 Step 1; tested Task 1 Step 1
  (`test_pool_token_default_off`, `test_pool_token_changes_output_vs_off`). ✓
- All-background edge case (den clamp) → Task 1 Step 5, tested
  (`test_pool_tokens_all_background_support_mask_no_nan`). ✓
- Checkpoint arch stores the key / eval rebuilds harmlessly → Task 2 Step 2. ✓
- Log entry → Task 2 Step 4. ✓

**Placeholder scan:** No TBD/TODO; all code steps contain concrete, complete code. ✓

**Type consistency:** `pool_token: bool`, `_pool_tokens(self, fine_finest, context_out,
query_prior, B, K, T) -> Tensor[B,K+1,Cf]` (Task 1 Step 5) is called with exactly this
signature in `forward` (Task 1 Step 7). `_attn(..., pool_feat=None)` (Task 1 Step 6) is called
with `pool_feat=pool_feat` from `forward` (Task 1 Step 7) — names and types consistent.
`_attn`'s external return tuple `(q, mask_support, mask_query, regs)` and `forward`'s return
dict keys are unchanged in both branches. Config key name `pool_token` consistent across
`__init__`, yaml, `build_model`, and all tests. ✓

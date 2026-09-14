# PatchSet3D Sequence Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an IRIS-style optional compression pipeline to `PatchSet3D` that compresses each volume's `R³` raw cell tokens down to `compress_m` tokens before the heavy transformer, and expands back to per-cell resolution only at decode — cutting attention compute at the model's current `R`/`K` without changing output shape or requiring larger grids.

**Architecture:** Three stages inside `_attn`. Stage A (new `RowCrossAttention` module, `compress_layers` deep) cross-attends `compress_m` learned per-volume query slots into each volume's raw `N` cell tokens, weight-shared across all `K+1` volumes. Stage B is the EXISTING `TransformerEncoderStack`, unmodified, now fed the compressed `K·m+m` row sequence instead of `K·N+N`. Stage C (another `RowCrossAttention` stack) cross-attends the query's original, never-compressed per-cell tokens into Stage B's context-aware compressed output, reconstituting a per-cell `q` for `_decode`. Gated by `arch.seq_compress` (default `False` = today's code path, byte-identical).

**Tech Stack:** PyTorch, Hydra config, pytest. New module in `src/models/pfn_seg_2d.py`; integration in `src/models/patchset3d.py`; config in `configs/experiment/3d/model/patchset3d.yaml`; wiring in `experiments/3d/train.py::build_model`.

**Spec:** `docs/superpowers/specs/2026-09-14-patchset3d-sequence-compression-design.md`

## Global Constraints

- `arch.seq_compress` defaults to `False` — with the default, forward output must be byte-identical (or within fp noise) to today's model; no new params allocated.
- `arch.seq_compress=True` is **assert-incompatible** with `arch.register_routed=True` (Stage A already partitions per-volume) and with `arch.transformer_rope=True` (Stage A/C carry no RoPE — see spec's "Known open questions"; this plan makes that gap an explicit fail-fast rather than a silent position loss).
- Stage A/C use a NEW `RowCrossAttention` module (asymmetric Q/KV row counts) — `TransformerEncoderLayer`'s self-attention cannot express this, so it is a sibling class, not a modification of the existing one.
- `context_id_embed`'s per-volume tag applies at the POST-compression granularity (`compress_m` rows per volume, not `N`) when `seq_compress=True`.
- `cascade_registers`' carried memory rows are independent of `N`/`compress_m` and require no change — Stage B consumes them exactly as today.
- Repo guideline: write tests only when necessary; keep new tests focused (mirrors this repo's existing `PatchSet3D` test density — one file for the new module, a handful of forward/backward/incompatibility tests for the integration). Log changes in `docs/logs.md`.
- Design spec: `docs/superpowers/specs/2026-09-14-patchset3d-sequence-compression-design.md`.

---

### Task 1: `RowCrossAttention` module (`pfn_seg_2d.py`)

**Files:**
- Modify: `src/models/pfn_seg_2d.py` — add `RowCrossAttention` class (place it directly after `TransformerEncoderLayer`, before `TransformerEncoderStack`, near line 264).
- Test: `tests/test_pfn_seg_2d.py` (new file).

**Interfaces:**
- Consumes: module-level `batched_sdpa`, `_small_seq_attn`, `_SMALL_SEQ_ATTN`, `LowerPrecisionRMSNorm` (all already defined in `pfn_seg_2d.py`, used the same way `TransformerEncoderLayer` uses them).
- Produces: `RowCrossAttention(a: int, e: int, h: int)` with `forward(q_in: Tensor[B,r_q,c,e], kv_in: Tensor[B,r_kv,c,e]) -> Tensor[B,r_q,c,e]` — `r_q` and `r_kv` may differ; `c` and `e` must match.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pfn_seg_2d.py`:

```python
import torch
from src.models.pfn_seg_2d import RowCrossAttention


def test_row_cross_attention_shape_with_mismatched_row_counts():
    """r_q != r_kv must work -- this is the whole point (self-attention can't do this)."""
    m = RowCrossAttention(a=2, e=16, h=32)
    q_in = torch.randn(3, 5, 2, 16)     # B=3, r_q=5
    kv_in = torch.randn(3, 40, 2, 16)   # r_kv=40
    out = m(q_in, kv_in)
    assert out.shape == (3, 5, 2, 16)


def test_row_cross_attention_backward_reaches_all_params():
    m = RowCrossAttention(a=2, e=16, h=32)
    q_in = torch.randn(2, 4, 2, 16, requires_grad=True)
    kv_in = torch.randn(2, 10, 2, 16, requires_grad=True)
    out = m(q_in, kv_in)
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"
    assert q_in.grad is not None and kv_in.grad is not None


def test_row_cross_attention_output_depends_on_kv_content():
    """Sanity: changing kv_in must change the output (proves cross-attention actually reads
    kv, not just passing q_in through via the residual)."""
    torch.manual_seed(0)
    m = RowCrossAttention(a=2, e=16, h=32)
    m.eval()
    q_in = torch.randn(1, 3, 2, 16)
    kv_a = torch.randn(1, 8, 2, 16)
    kv_b = torch.randn(1, 8, 2, 16)
    out_a = m(q_in, kv_a)
    out_b = m(q_in, kv_b)
    assert not torch.allclose(out_a, out_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pfn_seg_2d.py -v`
Expected: FAIL — `ImportError: cannot import name 'RowCrossAttention'`.

- [ ] **Step 3: Add the `RowCrossAttention` class**

In `src/models/pfn_seg_2d.py`, insert directly after `TransformerEncoderLayer`'s closing (before `class TransformerEncoderStack`, near line 264):

```python
class RowCrossAttention(nn.Module):
    """Row-axis cross-attention: Q rows attend into a DIFFERENT (kv) row-set's K/V, then the
    Q-side output runs the same column-axis self-attention + MLP TransformerEncoderLayer uses.
    Q and KV row counts may differ (r_q != r_kv) -- the asymmetric primitive
    TransformerEncoderLayer's self-attention (single input, single row count) can't express.
    Used by PatchSet3D's compress (Stage A: q=learned slots, kv=raw cells) and expand
    (Stage C: q=raw query cells, kv=compressed rows) stages -- same module, opposite argument
    order. No RoPE support (see PatchSet3D's arch.seq_compress/arch.transformer_rope
    incompatibility assert). See docs/superpowers/specs/2026-09-14-patchset3d-sequence
    -compression-design.md."""

    def __init__(self, a: int, e: int, h: int):
        super().__init__()
        assert e % a == 0
        self.a = a
        self.d = e // a
        self.q_proj = nn.Linear(e, e)
        self.kv_proj = nn.Linear(e, 2 * e)
        self.qkv_col = nn.Linear(e, 3 * e)
        self.norm_q = LowerPrecisionRMSNorm(e)
        self.norm_kv = LowerPrecisionRMSNorm(e)
        self.norm_col = LowerPrecisionRMSNorm(e)
        self.norm_mlp = LowerPrecisionRMSNorm(e)
        self.mlp = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, e))

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        """q_in (B,r_q,c,e), kv_in (B,r_kv,c,e), same c and e -> (B,r_q,c,e)."""
        b, rq, c, e = q_in.shape
        rkv = kv_in.shape[1]
        a, d = self.a, self.d

        # -- Row-axis cross-attention: rq queries read rkv keys/values, per column --------
        qn = self.norm_q(q_in).permute(0, 2, 1, 3).reshape(b * c, rq, e)
        kn = self.norm_kv(kv_in).permute(0, 2, 1, 3).reshape(b * c, rkv, e)
        qh = self.q_proj(qn).reshape(b * c, rq, a, d).transpose(1, 2)          # (b*c,a,rq,d)
        kvh = self.kv_proj(kn).reshape(b * c, rkv, 2, a, d).permute(2, 0, 3, 1, 4)
        x = batched_sdpa(qh, kvh[0], kvh[1])                                    # (b*c,a,rq,d)
        x = x.transpose(1, 2).reshape(b * c, rq, e).reshape(b, c, rq, e).permute(0, 2, 1, 3)
        q_in = q_in + x                                                         # (b,rq,c,e)

        # -- Column-axis self-attention (img/mask mix) -- identical shape/logic to
        # TransformerEncoderLayer's feature-axis block, applied to the cross-attended Q ---
        y = q_in.reshape(b * rq, c, e)
        res = y
        y = self.norm_col(y)
        qkv = self.qkv_col(y).reshape(b * rq, c, 3, a, d).permute(2, 0, 3, 1, 4)
        if c <= _SMALL_SEQ_ATTN:
            y = _small_seq_attn(qkv[0], qkv[1], qkv[2])
        else:
            y = batched_sdpa(qkv[0], qkv[1], qkv[2])
        y = y.transpose(1, 2).reshape(b * rq, c, e)
        q_in = (res + y).reshape(b, rq, c, e)

        # -- MLP --
        return q_in + self.mlp(self.norm_mlp(q_in))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pfn_seg_2d.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/models/pfn_seg_2d.py tests/test_pfn_seg_2d.py
git commit -m "$(printf 'feat(pfn_seg_2d): add RowCrossAttention (asymmetric Q/KV row-axis attention)\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Bp6fgSQtz7XQfHXHWuRsrw')"
```

---

### Task 2: Wire the compress/expand pipeline into `PatchSet3D`

**Files:**
- Modify: `src/models/patchset3d.py` — import `RowCrossAttention`; `__init__` (add args, new modules, incompatibility asserts); add `_compress`/`_expand` helper methods; rewrite `_attn`.
- Test: `tests/test_patchset3d.py` (append).

**Interfaces:**
- Consumes: `RowCrossAttention(a, e, h)` from Task 1; existing `PatchSet3D.__init__(..., cascade_registers: bool = False, ...)` tail (near the `self.thinking = ThinkingRows(...)` block); existing `_attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None, query_prior=None, cascade_regs=None) -> (q, mask_support, mask_query, regs)`.
- Produces:
  - `PatchSet3D.__init__(..., seq_compress: bool = False, compress_m: int = 32, compress_layers: int = 1)`
  - `self.seq_compress: bool`, `self.compress_m: int`
  - `self.compress_slots: nn.Parameter` shape `(compress_m, e)`, `self.compressor` / `self.expander`: `nn.ModuleList[RowCrossAttention]`, each `compress_layers` long (only allocated when `seq_compress=True`)
  - `_compress(self, tok: Tensor[B, n_vol*N, 2, e], n_vol: int) -> Tensor[B, n_vol*compress_m, 2, e]`
  - `_expand(self, q_raw: Tensor[B, N, 2, e], kv: Tensor[B, K*compress_m+compress_m, 2, e]) -> Tensor[B, N, 2, e]`
  - `_attn`'s external signature/return type is unchanged: `(q, mask_support, mask_query, regs)`, `q` still `(B, N, e)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patchset3d.py`:

```python
# --- arch.seq_compress: IRIS-style per-volume token compression before the heavy transformer ---

def test_seq_compress_default_off():
    """Default (seq_compress=False): no new params, forward shape/behavior unchanged."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    assert not hasattr(m, "compressor") and not hasattr(m, "expander")
    assert not hasattr(m, "compress_slots")
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)


def test_seq_compress_forward_shape_and_backward():
    """seq_compress=True: output shape unchanged (compression is internal); gradient reaches
    every param including the new compressor/expander/compress_slots."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   seq_compress=True, compress_m=3, compress_layers=2)
    assert len(m.compressor) == 2 and len(m.expander) == 2
    assert m.compress_slots.shape == (3, 32)
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    out["final_logit"].mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"


def test_seq_compress_changes_output_vs_uncompressed():
    """Sanity: with the same seed, seq_compress=True must produce different logits than
    seq_compress=False (different architecture, not silently falling back)."""
    torch.manual_seed(0)
    kw = dict(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    m_off = PatchSet3D(seq_compress=False, **kw)
    torch.manual_seed(0)
    m_on = PatchSet3D(seq_compress=True, compress_m=3, compress_layers=1, **kw)
    m_off.eval(); m_on.eval()
    torch.manual_seed(1)
    img, cin, cout = _dummy_batch(S=16)
    out_off = m_off(img, context_in=cin, context_out=cout)["final_logit"]
    out_on = m_on(img, context_in=cin, context_out=cout)["final_logit"]
    assert out_off.shape == out_on.shape
    assert not torch.allclose(out_off, out_on)


def test_seq_compress_rejects_register_routed():
    try:
        PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                  seq_compress=True, register_routed=True)
        assert False, "should have raised"
    except AssertionError as exc:
        assert "register_routed" in str(exc)


def test_seq_compress_rejects_transformer_rope():
    try:
        PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                  seq_compress=True, transformer_rope=True)
        assert False, "should have raised"
    except AssertionError as exc:
        assert "transformer_rope" in str(exc)


def test_seq_compress_works_with_context_id_embed_and_cascade_registers():
    """Both context_id_embed's per-volume tag and cascade_registers' carried memory must
    still work correctly at the compressed (compress_m-per-volume) granularity."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   seq_compress=True, compress_m=3, compress_layers=1,
                   context_id_embed=True, cascade_registers=True)
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)
    prev_regs = torch.randn(2, 2, 32)
    out = m(img, context_in=cin, context_out=cout, cascade_regs=prev_regs)
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    assert out["registers"].shape == (2, 2, 32)
    out["final_logit"].mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patchset3d.py -k seq_compress -v`
Expected: FAIL — `PatchSet3D.__init__` has no `seq_compress` argument (TypeError).

- [ ] **Step 3: Import `RowCrossAttention`**

In `src/models/patchset3d.py`, change the import near the top (line 21-22):

```python
from src.models.pfn_seg_2d import (
    ThinkingRows, TransformerEncoderStack, build_register_block_mask, RowCrossAttention)
```

- [ ] **Step 4: Add constructor args**

In `PatchSet3D.__init__`'s signature, add three params right after `cascade_registers: bool = False,` (near line 200):

```python
        cascade_registers: bool = False,
        seq_compress: bool = False,
        compress_m: int = 32,
        compress_layers: int = 1,
```

- [ ] **Step 5: Build the compress/expand modules**

In `__init__`'s body, insert between the existing `cascade_registers` block and `self.transformer = TransformerEncoderStack(...)` (near line 435):

```python
        self.thinking = ThinkingRows(thinking_rows, e)
        if self.cascade_registers:
            # Projects the previous level's mean-pooled thinking rows (B, thinking_rows, e)
            # into this level's own token space + tags them as carried memory (distinct from
            # this level's own fresh thinking rows) via a learned type vector.
            self.cascade_proj = nn.Linear(e, e)
            self.cascade_type = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.cascade_type, std=0.02)
        # seq_compress (IRIS-style): compress each volume's N raw cell tokens down to
        # compress_m tokens before the heavy transformer (Stage A), run the transformer over
        # the compressed sequence (Stage B, unmodified below), then expand back to N per-cell
        # tokens for decode (Stage C) via a fresh cross-attention read-out. See
        # docs/superpowers/specs/2026-09-14-patchset3d-sequence-compression-design.md.
        self.seq_compress = bool(seq_compress)
        self.compress_m = int(compress_m)
        assert not (self.seq_compress and self.transformer_rope), (
            "arch.seq_compress=True does not carry RoPE through the compress/expand stages -- "
            "incompatible with arch.transformer_rope=True for now")
        assert not (self.seq_compress and self.register_routed), (
            "arch.seq_compress=True already partitions per-volume in Stage A -- "
            "arch.register_routed's block-mask has nothing left to restrict in Stage B")
        if self.seq_compress:
            assert compress_layers >= 1, "arch.compress_layers must be >= 1 when seq_compress=True"
            self.compress_slots = nn.Parameter(torch.empty(self.compress_m, e))
            nn.init.normal_(self.compress_slots, std=0.02)
            self.compressor = nn.ModuleList(
                [RowCrossAttention(a, e, h) for _ in range(compress_layers)])
            self.expander = nn.ModuleList(
                [RowCrossAttention(a, e, h) for _ in range(compress_layers)])
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
```

- [ ] **Step 6: Add `_compress`/`_expand` helper methods**

Add these two methods just above `_attn` (near line 663, right before `def _attn`):

```python
    def _compress(self, tok: torch.Tensor, n_vol: int) -> torch.Tensor:
        """(B, n_vol*N, 2, e) -> (B, n_vol*compress_m, 2, e): per-volume compression via
        self.compressor (arch.compress_layers RowCrossAttention layers, weight-shared across
        all n_vol volumes by folding them into the batch dim). Each layer iteratively refines
        the running query against the SAME raw N-cell kv (DETR-decoder-style), starting from
        the learned self.compress_slots. See docs/superpowers/specs/2026-09-14-patchset3d-
        sequence-compression-design.md Stage A."""
        B, e = tok.shape[0], tok.shape[-1]
        kv = tok.reshape(B * n_vol, self.N, 2, e)
        q = self.compress_slots.unsqueeze(0).unsqueeze(2).expand(B * n_vol, -1, 2, -1)
        for layer in self.compressor:
            q = layer(q, kv)
        return q.reshape(B, n_vol * self.compress_m, 2, e)

    def _expand(self, q_raw: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """(B,N,2,e) raw query cells, (B,K*compress_m+compress_m,2,e) post-transformer
        compressed rows -> (B,N,2,e): per-cell read-out via self.expander (Stage C). q_raw
        supplies real spatial identity (never compressed away); kv is fixed across all
        arch.compress_layers layers (DETR-decoder-style iterative refinement of q_raw against
        the same memory)."""
        q = q_raw
        for layer in self.expander:
            q = layer(q, kv)
        return q
```

- [ ] **Step 7: Rewrite `_attn`**

Replace the entire `_attn` method body (from `def _attn(self, sup_feat, ...)` through its final `return q, mask_support, mask_query, regs`, near lines 663-738) with:

```python
    def _attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None, query_prior=None,
             cascade_regs=None):
        B, N = sup_feat.shape[0], self.N
        dev = sup_feat.device
        mask_support = self._sample_mask(B, K * N, self.token_mask_ratio_support, dev)
        mask_query = self._sample_mask(B, N, self.token_mask_ratio_query, dev)
        if query_prior is not None:                          # cascade: coarse-level prediction
            qry_occ = self._prior_occupancy(query_prior).to(sup_occ.dtype)
        else:
            qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])  # support-mean prior
        sup_ijk = self.ijk_base.repeat(K, 1).unsqueeze(0).expand(B, K * N, 3)
        qry_ijk = self.ijk_base.unsqueeze(0).expand(B, N, 3)

        sup_feat, qry_feat = self._feat_norm(sup_feat, qry_feat)

        # support always carries real GT (never a prediction); query never carries real GT
        # (only ever a prior/prediction, or the support-mean fallback) — see forward's
        # docstring and cascade.py's context_out handling. mask_slots>=2 tags the mask
        # column accordingly so the network knows which; mask_slots=1 tags neither (legacy).
        ctype_sup = "gt" if self.mask_slots >= 2 else None
        ctype_qry = "pred" if self.mask_slots >= 2 else None
        sup_tok = self._tokens(sup_feat, sup_occ, sup_ijk, mask=mask_support, content_type=ctype_sup)
        qry_tok = self._tokens(qry_feat, qry_occ, qry_ijk, mask=mask_query, content_type=ctype_qry)

        # arch.seq_compress (Stage A): compress each of the K+1 volumes' N raw cell tokens to
        # compress_m tokens BEFORE the heavy transformer. qry_tok_raw is kept aside, untouched,
        # as Stage C's per-cell read-out query — see docs/superpowers/specs/2026-09-14-
        # patchset3d-sequence-compression-design.md.
        qry_tok_raw = qry_tok
        n_per_vol = N
        if self.seq_compress:
            combined = self._compress(torch.cat([sup_tok, qry_tok], dim=1), K + 1)
            n_per_vol = self.compress_m
            sup_tok, qry_tok = combined[:, :K * n_per_vol], combined[:, K * n_per_vol:]

        if self.context_id_embed:
            assert K <= self.max_context, f"context_size {K} exceeds max_context {self.max_context}"
            e_dim = sup_tok.shape[-1]
            ctx_emb = self.ctx_id(torch.arange(K, device=sup_tok.device)).repeat_interleave(n_per_vol, dim=0)
            sup_tok = sup_tok + ctx_emb.view(1, K * n_per_vol, 1, e_dim)
            qry_tok = qry_tok + self.qry_id.view(1, 1, 1, e_dim)

        sep = K * n_per_vol
        x = torch.cat([sup_tok, qry_tok], dim=1)
        n_extra = 0
        if cascade_regs is not None:
            assert self.cascade_registers, (
                "cascade_regs given but arch.cascade_registers=False on this model")
            mem = self.cascade_proj(cascade_regs) + self.cascade_type   # (B, R, e)
            mem = mem.unsqueeze(2).expand(-1, -1, x.shape[2], -1)       # (B, R, c, e)
            x = torch.cat([mem, x], dim=1)
            n_extra = mem.shape[1]
            sep += n_extra
        x, sep_t = self.thinking(x, sep)
        attn_mask = None
        block_mask = None
        if not self.full_attn and self.register_routed:
            n_t = self.thinking.n
            block_mask = (build_register_block_mask(n_t, N, K + 1, x.device)
                          if self.register_flex else None)
            if block_mask is None:
                r = x.shape[1]
                attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
                attn_mask[:n_t, :] = True
                attn_mask[:, :n_t] = True
                for m in range(K + 1):
                    s = n_t + m * N
                    attn_mask[s:s + N, s:s + N] = True
        elif not self.full_attn and self.query_self_attn:
            r = x.shape[1]
            attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
            attn_mask[:, :sep_t] = True
            attn_mask[sep_t:, sep_t:] = True
        rope = (self._rope(K, spacing, x.device, n_extra=n_extra)
               if self.transformer_rope else None)
        x = self.transformer(x, sep_t, attn_mask=attn_mask, full_attn=self.full_attn,
                             rope=rope, block_mask=block_mask)

        if self.seq_compress:
            # Stage C: qry_tok_raw's REAL per-cell tokens read out of the post-transformer
            # (context-aware) compressed rows -- support AND query compressed rows both, per
            # the design spec (the query's own compressed representation is itself context
            # -informed and should feed its own expansion).
            kv_expand = x[:, self.thinking.n + n_extra:]     # (B, K*compress_m+compress_m, 2, e)
            q_out = self._expand(qry_tok_raw, kv_expand)
            q = q_out[:, :, self._decode_col, :]              # (B,N,e)
        else:
            q = x[:, sep_t:, self._decode_col, :]      # (B,Q,e) query row, arch.decode_source col
        regs = x[:, :self.thinking.n].mean(dim=2) if self.cascade_registers else None
        return q, mask_support, mask_query, regs
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset3d.py -k seq_compress -v`
Expected: PASS (6 tests).

- [ ] **Step 9: Run the full model test suite for no regressions**

Run: `python -m pytest tests/test_patchset3d.py tests/test_patchset3d_rope.py tests/test_pfn_seg_2d.py -v`
Expected: PASS (every existing test, unaffected by the `seq_compress=False` default).

- [ ] **Step 10: Commit**

```bash
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "$(printf 'feat(patchset3d): IRIS-style seq_compress pipeline (Stage A/B/C)\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Bp6fgSQtz7XQfHXHWuRsrw')"
```

---

### Task 3: Config knobs, `build_model` wiring, and logs

**Files:**
- Modify: `configs/experiment/3d/model/patchset3d.yaml` (add three arch keys).
- Modify: `experiments/3d/train.py::build_model` (pass the three keys into the arch dict).
- Modify: `docs/logs.md` (append a log entry).

**Interfaces:**
- Consumes: `PatchSet3D.__init__(..., seq_compress, compress_m, compress_layers)` from Task 2; `build_model`'s existing `arch = {...}` dict pattern using `a.get("key", default)`.
- Produces: `arch.seq_compress` / `arch.compress_m` / `arch.compress_layers` reachable via Hydra overrides, stored in the checkpoint's `arch` block (rebuilt by `eval.py`, harmless there — eval never trains).

- [ ] **Step 1: Add the three arch keys to the config**

In `configs/experiment/3d/model/patchset3d.yaml`, add after the `register_flex:` line and before `compile: true` (near line 57):

```yaml
  seq_compress: false          # IRIS-style: compress each volume's N raw cell tokens to
                               # compress_m before the heavy transformer (Stage A), run the
                               # transformer over the compressed sequence, expand back to N
                               # per-cell tokens for decode (Stage C). Incompatible with
                               # register_routed and transformer_rope (see
                               # docs/superpowers/specs/2026-09-14-patchset3d-sequence-
                               # compression-design.md).
  compress_m: 32               # tokens per volume after Stage A compression (tune; only
                               # used when seq_compress=true)
  compress_layers: 1           # depth of the Stage A/Stage C RowCrossAttention stacks, each
```

- [ ] **Step 2: Wire the keys into `build_model`**

In `experiments/3d/train.py`, inside `build_model`'s `patchset3d` branch, add three entries to the `arch` dict (alongside `"cascade_registers": a.get("cascade_registers", False),`, near line 404):

```python
            "cascade_registers": a.get("cascade_registers", False),
            "seq_compress": a.get("seq_compress", False),
            "compress_m": a.get("compress_m", 32),
            "compress_layers": a.get("compress_layers", 1),
```

- [ ] **Step 3: Verify config resolves and reaches the model**

Run:
```bash
python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('configs/experiment/3d/model/patchset3d.yaml')
assert c.arch.seq_compress == False
assert c.arch.compress_m == 32
assert c.arch.compress_layers == 1
print('config ok:', c.arch.seq_compress, c.arch.compress_m, c.arch.compress_layers)
"
```
Expected: prints `config ok: False 32 1` with no assertion error.

- [ ] **Step 4: Append a log entry**

Add to `docs/logs.md` (follow the file's existing dated-entry format; append after the last entry):

```markdown
## 2026-09-14 — PatchSet3D IRIS-style sequence compression (`arch.seq_compress`)

Added an optional 3-stage compression pipeline to `PatchSet3D` (`src/models/patchset3d.py`),
inspired by Iris's decoupled task-encoding (`docs/methods/iris.md`). New `RowCrossAttention`
module (`src/models/pfn_seg_2d.py`) does asymmetric Q/KV row-axis cross-attention (Q and KV
row counts may differ — `TransformerEncoderLayer`'s self-attention can't express this). Stage
A compresses each of the K+1 volumes' `R³` raw cell tokens to `arch.compress_m` tokens
(weight-shared, `arch.compress_layers` deep). Stage B is the existing `TransformerEncoderStack`
unmodified, now fed the compressed `K·m+m` sequence instead of `K·N+N`. Stage C cross-attends
the query's original (never-compressed) per-cell tokens into Stage B's output to reconstitute
a per-cell `q` for `_decode`. `arch.seq_compress` defaults to `False` (byte-identical to
today); assert-incompatible with `arch.register_routed` (Stage A already partitions
per-volume) and `arch.transformer_rope` (Stage A/C carry no RoPE yet — a known gap, not
silently ignored). Spec: docs/superpowers/specs/2026-09-14-patchset3d-sequence-compression-design.md.
```

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/3d/model/patchset3d.yaml experiments/3d/train.py docs/logs.md
git commit -m "$(printf 'feat(patchset3d): config + build_model wiring for seq_compress\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Bp6fgSQtz7XQfHXHWuRsrw')"
```

---

## Self-Review

**Spec coverage:**
- `RowCrossAttention` new sibling module (asymmetric Q/KV) → Task 1. ✓
- Stage A: per-volume, weight-shared, learned `compress_m` slots, block-restricted (achieved via the `B*n_vol` batch fold, not a mask tensor) → Task 2 Step 5-6 (`_compress`). ✓
- Stage B: existing `TransformerEncoderStack` unmodified, fed the compressed sequence → Task 2 Step 7 (`_attn` calls `self.transformer` exactly as before, just on shorter `x`). ✓
- Stage C: query's original per-cell tokens cross-attend into ALL `K·m+m` post-transformer compressed rows (support AND query) → Task 2 Step 6-7 (`_expand`, `kv_expand = x[:, thinking.n+n_extra:]`). ✓
- `arch.seq_compress` / `compress_m` / `compress_layers` config knobs, default off → Task 2 Step 4, Task 3 Steps 1-2. ✓
- `register_routed` incompatibility assert → Task 2 Step 5, tested Task 2 Step 1 (`test_seq_compress_rejects_register_routed`). ✓
- `transformer_rope` incompatibility assert (plan's explicit resolution of the spec's open RoPE gap) → Task 2 Step 5, tested Task 2 Step 1 (`test_seq_compress_rejects_transformer_rope`). ✓
- `ThinkingRows` unaffected, operates on compressed sequence in Stage B → Task 2 Step 7 (`self.thinking(x, sep)` unchanged). ✓
- `cascade_registers` compatible with no change → Task 2 Step 7 (`mem`/`cascade_regs` handling untouched) + tested (`test_seq_compress_works_with_context_id_embed_and_cascade_registers`). ✓
- `context_id_embed` adapted to `compress_m` granularity → Task 2 Step 7 (`n_per_vol` swap) + tested. ✓
- `fine_decode` untouched (reads raw unpooled encoder stages, orthogonal to row sequence) → not modified anywhere in this plan, correct (no task touches `_decode`/`fine`). ✓
- Backward compatibility (`seq_compress=False` byte-identical, no new params) → tested `test_seq_compress_default_off`. ✓
- Log entry → Task 3 Step 4. ✓

**Placeholder scan:** No TBD/TODO; every code step contains concrete, complete code. ✓

**Type consistency:** `RowCrossAttention(a, e, h).forward(q_in, kv_in) -> Tensor` (Task 1) is called identically in `_compress`/`_expand` (Task 2). `seq_compress: bool`, `compress_m: int`, `compress_layers: int` — consistent names/types across `__init__` signature, config yaml, `build_model`'s `a.get(...)` calls, and all tests. `_attn`'s return tuple `(q, mask_support, mask_query, regs)` and `q`'s final shape `(B, N, e)` are unchanged whether `seq_compress` is on or off — verified by both branches of the Step-7 `_attn` rewrite ending in shape-identical tensors. ✓

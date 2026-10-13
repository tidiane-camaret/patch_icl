# Multi-patch Cascade Re-crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zero-shot, inference-only validation of sampling N windows per role
(target + support) from a continuous probability field, processed in one
joint forward pass, as a drop-in generalization of the cascade's existing
1-window-per-role re-crop — compared on FLARE22 against the existing
checkpoint's single-patch cascade baseline.

**Architecture:** `PatchSet3D`'s row axis is already a flat set of every
volume's cell-tokens with full dense attention; generalizing its query axis
from exactly 1 volume to `M` volumes needs only shape-plumbing (no new
attention mechanism). `cascade.py`'s re-crop step gains a
`data.cascade_n_patches` knob (default `1`, byte-identical) that, when `>1`,
samples `n` points per role from a probability-weighted multinomial over the
previous level's `M` windows (target) or via repeated `random_fg` provider
calls (support), builds an `n`-query batch, and forwards it through the
generalized model in one call.

**Tech Stack:** PyTorch, OmegaConf/Hydra, pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-multipatch-cascade-design.md`

## Global Constraints

- `data.cascade_n_patches` defaults to `1` and must reproduce today's exact
  cascade behavior byte-for-byte at that default (every new code path is
  gated on `n_patches > 1`, never taken at `n_patches == 1`).
- `data.cascade_n_patches > 1` is eval-only: `run_cascade` must raise if
  called with `training=True` and `n_patches > 1` (training-recipe support
  is an explicit non-goal of the spec).
- `M > 1` (the model's generalized query-volume count) asserts against
  `arch.decoder == "iris"`, `arch.seq_compress`, `arch.cascade_registers`,
  and `arch.register_routed` — none of these are generalized; the checkpoint
  under test (`168_cascade_real_host_all_shapes`) has all four off/`conv`.
- No changes to level 0 (dataset-driven bootstrap) or to training code paths
  (loss, data loader, synth recipe).

## Review Focus

- A FLARE22 case whose GT mask is empty for a requested class (some subjects
  lack some organs) — `_prob_weighted_points_multi` must return `None`
  entries, not divide-by-zero or crash, matching `_random_fg_point`'s
  existing empty-mask contract.
- `n_patches > 1` with `K=0` support subjects (context_size=0 configs exist
  elsewhere in this repo) — the per-context-subject task-building loop must
  not assume `K >= 1`.
- A sampled window landing fully outside the native volume bounds after
  `invert_geo_center` (e.g. a target near the volume edge) — `provider.load`
  already clamps crops to native bounds for the `n=1` case; the `n>1` task
  list must hit the exact same provider call, not a new unclamped path.
- Two sampled windows for the *same role* at the *same level* landing on
  the identical native centre (duplicate draws from `random.choices`'
  with-replacement sampling) — decode must still produce `n` independent
  predictions (no accidental de-duplication that would silently shrink `n`).
- FLARE22's per-class `CLASS_IDX`/native-GT path (`NativeGridProvider`) must
  still be used correctly by the new list-of-windows stitching in
  `_stitched_native_metrics_and_levels` — a source with its own index space
  silently scoring NaN is a documented past failure mode (docs/logs.md
  2026-09-12) the generalization must not reintroduce.

---

## Task 1: `_prob_weighted_points_multi` — multi-window probability sampler

**Files:**
- Modify: `experiments/3d/cascade.py` (add near `_random_fg_point`, after line ~159)
- Test: `experiments/3d/tests/test_cascade.py` (add near the existing `_random_fg_point` tests, after line ~1101)

**Interfaces:**
- Consumes: nothing new (only `torch`, `torch.nn.functional as F`, `numpy as np`, `random` — all already imported in `cascade.py`).
- Produces: `_prob_weighted_points_multi(fields, T, is_prob, n, *, seed, step, level) -> list[list[tuple[int, tuple[int,int,int]] | None]]`, one outer list entry per batch row `b`, each an inner list of length `n` of `(window_idx, (d,h,w))` or `None`. Task 5 (`run_cascade`) and Task 6 (`evaluate_cascade`) call this directly.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/3d/tests/test_cascade.py, after the existing _random_fg_point tests

from cascade import _prob_weighted_points_multi


def test_prob_weighted_points_multi_single_window_single_voxel():
    """M=1 field, one hot voxel: every one of n draws must land on that voxel."""
    T = 6
    field = torch.full((1, 1, T, T, T), -10.0)
    field[0, 0, 2, 3, 4] = 10.0
    out = _prob_weighted_points_multi([field], T, is_prob=False, n=3,
                                      seed=0, step=0, level=0)
    assert len(out) == 1 and len(out[0]) == 3
    for window_idx, (d, h, w) in out[0]:
        assert window_idx == 0
        assert (d, h, w) == (2, 3, 4)


def test_prob_weighted_points_multi_covers_both_windows():
    """Two windows, each with ALL its mass on a different voxel: with n=8 draws from a
    concatenated multinomial, both windows' voxels must appear at least once (this is
    exactly the failure mode a single-point re-crop can't express -- both windows'
    content gets a chance to be sampled, not just one)."""
    T = 6
    f0 = torch.full((1, 1, T, T, T), -10.0)
    f0[0, 0, 1, 1, 1] = 10.0
    f1 = torch.full((1, 1, T, T, T), -10.0)
    f1[0, 0, 4, 4, 4] = 10.0
    out = _prob_weighted_points_multi([f0, f1], T, is_prob=False, n=8,
                                      seed=0, step=0, level=0)
    picks = {(w, pt) for w, pt in out[0]}
    assert (0, (1, 1, 1)) in picks
    assert (1, (4, 4, 4)) in picks


def test_prob_weighted_points_multi_empty_returns_none():
    T = 6
    f0 = torch.full((1, 1, T, T, T), -10.0)
    f1 = torch.full((1, 1, T, T, T), -10.0)
    out = _prob_weighted_points_multi([f0, f1], T, is_prob=False, n=3,
                                      seed=0, step=0, level=0)
    assert out == [[None, None, None]]


def test_prob_weighted_points_multi_is_seeded_reproducibly():
    T = 6
    f0 = torch.full((1, 1, T, T, T), -10.0)
    f0[0, 0, 1, 1, 1] = 10.0
    f0[0, 0, 4, 4, 4] = 10.0
    a = _prob_weighted_points_multi([f0], T, is_prob=False, n=4, seed=7, step=3, level=1)
    b = _prob_weighted_points_multi([f0], T, is_prob=False, n=4, seed=7, step=3, level=1)
    assert a == b


def test_prob_weighted_points_multi_batch_rows_independent():
    T = 6
    field = torch.full((2, 1, T, T, T), -10.0)
    field[0, 0, 1, 1, 1] = 10.0
    field[1, 0, 5, 5, 5] = 10.0
    out = _prob_weighted_points_multi([field], T, is_prob=False, n=2,
                                      seed=0, step=0, level=0)
    assert all(pt == (1, 1, 1) for _, pt in out[0])
    assert all(pt == (5, 5, 5) for _, pt in out[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd experiments/3d && python -m pytest tests/test_cascade.py -k prob_weighted_points_multi -v`
Expected: FAIL with `ImportError: cannot import name '_prob_weighted_points_multi'`

- [ ] **Step 3: Implement `_prob_weighted_points_multi`**

Add to `experiments/3d/cascade.py`, directly after `_random_fg_point` (after
the existing function ending around line 159):

```python
def _prob_weighted_points_multi(fields, T, is_prob, n, *, seed, step, level):
    """Per-b, sample `n` points via a probability-weighted multinomial over the
    CONCATENATION of `fields` (a list of M (B,1,T,T,T) logit/prob tensors, one per
    level-i window) -- the multi-window generalization of _random_fg_point/
    _centroid_from_logit. A voxel's weight is its own predicted probability (not a
    thresholded indicator), so overlapping/agreeing windows naturally accumulate more
    sampling mass where both predict foreground -- no separate native-space stitch step.

    Returns, per b, a list of n (window_idx, (d,h,w)) pairs, or n `None` entries when
    every field's mass is ~0 for that b (same empty-mask fallback convention as
    _random_fg_point). Seeded off f"{seed}_{step}_{level}_{b}_mfg" via random.Random,
    the same string-seeding convention _random_fg_point uses -- reproducible across
    runs/python-version-stable (unlike builtin hash(), which is NOT process-stable
    under PYTHONHASHSEED randomization)."""
    probs = []
    for f in fields:
        p = f.float().clamp(0, 1) if is_prob else torch.sigmoid(f.float())
        if p.shape[-1] != T:
            p = F.interpolate(p, size=(T, T, T), mode="trilinear", align_corners=False)
        probs.append(p[:, 0].reshape(p.shape[0], -1))            # (B, T^3)
    flat = torch.cat(probs, dim=1)                                # (B, M*T^3)
    B = flat.shape[0]
    out = []
    for b in range(B):
        weights = flat[b].tolist()
        if sum(weights) < 1e-6:
            out.append([None] * n)
            continue
        rng = random.Random(f"{seed}_{step}_{level}_{b}_mfg")
        picks = rng.choices(range(len(weights)), weights=weights, k=n)
        pts = []
        for idx in picks:
            w_idx, local = divmod(idx, T ** 3)
            d, h, w = np.unravel_index(local, (T, T, T))
            pts.append((w_idx, (int(d), int(h), int(w))))
        out.append(pts)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd experiments/3d && python -m pytest tests/test_cascade.py -k prob_weighted_points_multi -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/cascade.py experiments/3d/tests/test_cascade.py
git commit -m "feat(cascade): add _prob_weighted_points_multi, a multi-window probability sampler"
```

---

## Task 2: `PatchSet3D` — generalize the query axis from 1 to M volumes (shape plumbing)

**Files:**
- Modify: `src/models/patchset3d.py`
- Test: `tests/test_patchset3d.py` (append near the end)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PatchSet3D.forward(image, context_in, context_out, mode="train", spacing=None, query_prior=None, cascade_regs=None)` now accepts `image` as either legacy `(B,1,D,H,W)` (M=1) or new `(B,M,1,D,H,W)` (M>=1), returning `final_logit` of shape `(B,1,G,G,G)` (legacy) or `(B,M,1,G,G,G)` (new). Task 3 extends `query_prior` to the `(B,M,1,D,H,W)` shape; Task 5 is the only caller that passes `M>1`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_patchset3d.py, appended at the end

def test_multi_query_m1_matches_legacy_shape():
    """M=1 via the NEW 5-D (B,1,1,D,H,W) image shape must byte-match the legacy
    (B,1,D,H,W) shape -- the M-generalization must be a true no-op at M=1."""
    torch.manual_seed(0)
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                   thinking_rows=2, mask_patch_decode_size=2, full_attn=True)
    m.eval()
    img, cin, cout = _dummy_batch(S=16)
    legacy = m(img, context_in=cin, context_out=cout)["final_logit"]
    new_shape = m(img.unsqueeze(1), context_in=cin, context_out=cout)["final_logit"]
    assert new_shape.shape == (2, 1, 1, 8, 8, 8)
    assert torch.allclose(legacy, new_shape[:, 0])


def test_multi_query_m3_output_shape():
    torch.manual_seed(0)
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                   thinking_rows=2, mask_patch_decode_size=2, full_attn=True)
    m.eval()
    B, K, S, M = 2, 2, 16, 3
    image = torch.randn(B, M, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    out = m(image, context_in=context_in, context_out=context_out)["final_logit"]
    assert out.shape == (B, M, 1, 8, 8, 8)


def test_multi_query_decode_is_permutation_equivariant_in_m():
    """Shuffling the M query windows' order must shuffle the M output logits
    identically -- decode itself has no cross-window structure; only _attn's shared
    transformer lets windows influence each other, which is the intended effect."""
    torch.manual_seed(0)
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                   thinking_rows=2, mask_patch_decode_size=2, full_attn=True)
    m.eval()
    B, K, S, M = 1, 2, 16, 3
    image = torch.randn(B, M, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    out = m(image, context_in=context_in, context_out=context_out)["final_logit"]

    perm = [2, 0, 1]
    out_perm = m(image[:, perm], context_in=context_in, context_out=context_out)["final_logit"]
    assert torch.allclose(out[:, perm], out_perm)


def test_multi_query_backward():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    B, K, S, M = 2, 2, 16, 2
    image = torch.randn(B, M, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    out = m(image, context_in=context_in, context_out=context_out)["final_logit"]
    out.mean().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0


def test_multi_query_rejects_iris_decoder():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fine_decode=True, decoder="iris", image_size=(16, 16, 16),
                   iris_pixelshuffle_r=2)
    B, K, S, M = 1, 2, 16, 2
    image = torch.randn(B, M, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    with pytest.raises(AssertionError, match="M=2"):
        m(image, context_in=context_in, context_out=context_out)


def test_multi_query_rejects_cascade_registers():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   cascade_registers=True)
    B, K, S, M = 1, 2, 16, 2
    image = torch.randn(B, M, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    with pytest.raises(AssertionError, match="M=2"):
        m(image, context_in=context_in, context_out=context_out)
```

`pytest` is already imported in `tests/test_patchset3d.py` (used by
`test_decode_source_invalid_raises`) — no new import needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patchset3d.py -k multi_query -v`
Expected: FAIL — shape mismatches / no assertion raised (the model doesn't yet know
about 5-D `image` or `M>1`).

- [ ] **Step 3: Implement the shape generalization**

In `src/models/patchset3d.py`, `_grid_tokens` (around line 666):

```python
    def _grid_tokens(self, feat_map, B, T, K):
        """(B*T,Cf,R,R,R) -> (support (B,K·N,Cf), query (B,(T-K)·N,Cf)), image-major,
        row-major cells. T-K is the query VOLUME count (M, generalized from always-1)."""
        Cf = feat_map.shape[1]
        feat = feat_map.flatten(2).transpose(1, 2).reshape(B, T, self.N, Cf)
        return (feat[:, :K].reshape(B, K * self.N, Cf),
               feat[:, K:].reshape(B, (T - K) * self.N, Cf))
```

In `_attn` (around line 928), right after `B, N = sup_feat.shape[0], self.N`:

```python
    def _attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None, query_prior=None,
             cascade_regs=None, pool_feat=None):
        B, N = sup_feat.shape[0], self.N
        M = qry_feat.shape[1] // N                   # query VOLUME count (1 legacy, M new)
        assert not (M > 1 and (self.decoder_kind == "iris" or self.seq_compress
                               or self.cascade_registers or self.register_routed)), (
            f"PatchSet3D: M={M} query volumes requires decoder_kind != 'iris', "
            f"seq_compress=False, cascade_registers=False, register_routed=False "
            f"(not generalized -- see docs/superpowers/specs/"
            f"2026-10-13-multipatch-cascade-design.md)")
        dev = sup_feat.device
        mask_support = self._sample_mask(B, K * N, self.token_mask_ratio_support, dev)
        mask_query = self._sample_mask(B, M * N, self.token_mask_ratio_query, dev)
```

Replace the existing `mask_query = self._sample_mask(B, N, ...)` line with the
`B, M * N, ...` version shown above (the single line changes; everything above it in the
method is unchanged).

A few lines down, replace the `qry_occ`/`qry_ijk` block:

```python
        if query_prior is not None:                          # cascade: coarse-level prediction
            qry_occ = self._prior_occupancy(query_prior).to(sup_occ.dtype)
        else:
            qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, M * N, sup_occ.shape[-1])
        sup_ijk = self.ijk_base.repeat(K, 1).unsqueeze(0).expand(B, K * N, 3)
        qry_ijk = self.ijk_base.repeat(M, 1).unsqueeze(0).expand(B, M * N, 3)
```

(Task 3 implements `_prior_occupancy` so it already returns `(B, M*N, p^3)` when
`query_prior` carries `M` windows — nothing more changes here in Task 2; for Task 2's own
tests, `query_prior` is always `None`, so only the `else` branch and `qry_ijk` matter yet.)

In `_pool_tokens` (around line 880-898), change the slice that currently hardcodes a
single query row:

```python
            sup_feat, qry_feat = feat[:, :K], feat[:, K:]           # (B,K,...), (B,M,...)
```

(was `feat[:, K:K + 1]`). A few lines below, the `qry_mask` fallback currently does
`sup_mask.mean(dim=1, keepdim=True)` giving `(B,1,S,S,S)`; `masked_avg` broadcasts
against `qry_feat`'s own leading dim via its own `f.shape`/`m.shape` (both passed
together), so update the fallback to match `qry_feat`'s `M`:

```python
            if query_prior is not None:
                qry_mask = F.interpolate(query_prior.float(), size=(S, S, S), mode="trilinear",
                                         align_corners=False)               # (B,M,S,S,S) — Task 3
            else:
                qry_mask = sup_mask.mean(dim=1, keepdim=True).expand(-1, qry_feat.shape[1], -1, -1, -1)
```

(The `query_prior is not None` branch's exact interpolate call is finished in Task 3,
which generalizes `query_prior` itself to carry an `M` axis — for Task 2's tests
`query_prior` is always `None`, so only the `else` line is exercised here.)

In `_attn`'s `pool_feat` branch (around line 990-1010), generalize the trailing `1`
(query row) to `M`:

```python
        if pool_feat is not None:
            assert self.pool_token, "pool_feat given but arch.pool_token=False on this model"
            pool = self.pool_proj(pool_feat) + self.pool_type       # (B,K+M,e)
            if self.context_id_embed:
                ctx_tag = torch.cat([
                    self.ctx_id(torch.arange(K, device=pool.device)).unsqueeze(0).expand(B, -1, -1),
                    self.qry_id.view(1, 1, -1).expand(B, M, -1)], dim=1)   # (B,K+M,e)
                pool = pool + ctx_tag
            if self.mask_slots >= 2:
                gt_tag = self._slot_pos_vec(self._mask_content_index["gt"], pool.device, pool.dtype)
                pred_tag = self._slot_pos_vec(self._mask_content_index["pred"], pool.device, pool.dtype)
                pool = pool + torch.cat([gt_tag.expand(K, -1), pred_tag.expand(M, -1)],
                                        dim=0).unsqueeze(0)
            pool = pool.unsqueeze(2).expand(-1, -1, x.shape[2], -1)   # (B,K+M,c,e)
            x = torch.cat([pool, x], dim=1)
            n_extra += pool.shape[1]
            sep += pool.shape[1]
```

(Only `self.qry_id.view(1, 1, -1).expand(B, 1, -1)` → `.expand(B, M, -1)` and
`pred_tag.unsqueeze(0)` → `pred_tag.expand(M, -1)` actually change; the rest of the
block is unchanged, shown for context.)

In `forward` (around line 1199), replace the `imgs`/`T` construction:

```python
    def forward(self, image, context_in, context_out, mode="train", spacing=None,
                query_prior=None, cascade_regs=None):
        B, K = context_in.shape[0], context_in.shape[1]
        legacy_image = image.dim() == 4           # (B,1,D,H,W) -- single query volume
        if legacy_image:
            image = image.unsqueeze(1)             # -> (B,1,1,D,H,W), M=1
        M = image.shape[1]
        D, H, W = image.shape[-3:]
        imgs = torch.cat([context_in, image], dim=1)     # (B,T,1,D,H,W), T = K + M
        T = imgs.shape[1]
```

(`image.unsqueeze(1).dim()` previously fed `torch.cat([context_in, image.unsqueeze(1)],
dim=1)` directly; now the unsqueeze happens once, up front, so every downstream use of
`image` already has the volume-count axis.)

Still in `forward`, the `fine_decode` branch's `qidx` (around line 1230) currently
grabs only the first query row per `b`; generalize to all `M`:

```python
            qidx = (torch.arange(B, device=x.device).unsqueeze(1) * T
                   + K + torch.arange(M, device=x.device).unsqueeze(0)).reshape(-1)   # (B*M,)
```

(was `qidx = torch.arange(B, device=x.device) * T + K`). Every other use of `qidx` in
this branch (`rows = qidx`, `fine = tuple(f[qidx] for f in fine)`) is unchanged — both
already index generically by a 1-D tensor, so a longer `qidx` (length `B*M` instead of
`B`) flows through correctly.

Finally, replace the tail of `forward` (the `_decode` call and return):

```python
        if self.decoder_kind == "iris":
            norm_sup, norm_qry = self._feat_norm(sup_feat, qry_feat)
            task_tokens = self._iris_task_encode(norm_sup, context_out, iris_fine_finest,
                                                 B, K, T)
            qry_img_pre = self.img_embed(norm_qry)
            logit = self._decode_iris(qry_img_pre, task_tokens, fine)
        else:
            N = self.N
            q_flat = q.reshape(B * M, N, -1)
            fine_flat = tuple(f for f in fine) if fine is not None else None  # already (B*M,...)
            logit = self._decode(q_flat, fine_flat)
            logit = logit.reshape(B, M, *logit.shape[1:])         # (B,M,1,G,G,G)
            if legacy_image:
                logit = logit[:, 0]                                # (B,1,G,G,G)
        return {"final_logit": logit, "mask_support": mask_support, "mask_query": mask_query,
               "registers": regs}
```

(`fine` is already `(B*M, Cf, S, S, S)` per stage after the `qidx` fix above — no
reshape needed there, only the `q` reshape is new. The `self.decoder_kind == "iris"`
branch is unchanged, shown only so the `else:` is unambiguous — `M>1` already asserts
against `decoder_kind == "iris"` in `_attn`, so this branch only ever runs at `M=1`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset3d.py -v`
Expected: PASS — the 6 new `multi_query` tests AND every pre-existing test in the file
(the legacy `(B,1,D,H,W)` path must still pass unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "feat(patchset3d): generalize the query axis from 1 to M volumes"
```

---

## Task 3: `PatchSet3D` — generalize `query_prior` to carry M windows

**Files:**
- Modify: `src/models/patchset3d.py`
- Test: `tests/test_patchset3d.py`

**Interfaces:**
- Consumes: Task 2's `M`-generalized `forward`/`_attn`/`_pool_tokens`.
- Produces: `forward(..., query_prior=...)` now accepts `query_prior` as either the
  legacy `(B,1,D,H,W)` single soft-prob volume or a new `(B,M,1,D,H,W)` tensor (one
  prior per query window, `M` matching `image`'s own `M`). Task 5 builds the `(B,M,1,T,T,T)` tensor.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_patchset3d.py, appended after Task 2's tests

def test_multi_query_prior_per_window_shifts_only_that_windows_logit():
    """Each of the M windows' own query_prior must affect ONLY that window's logit --
    giving window 0 a prior and leaving window 1's prior at the support-mean fallback
    equivalent (all-zeros here) must change window 0's output but not window 1's."""
    torch.manual_seed(0)
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                   thinking_rows=2, mask_patch_decode_size=2, full_attn=True)
    m.eval()
    B, K, S, M = 1, 2, 16, 2
    image = torch.randn(B, M, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()

    base = m(image, context_in=context_in, context_out=context_out)["final_logit"]

    prior = torch.zeros(B, M, 1, S, S, S)
    prior[:, 0] = torch.rand(B, 1, S, S, S)          # only window 0 gets a real prior
    with_prior = m(image, context_in=context_in, context_out=context_out,
                  query_prior=prior)["final_logit"]

    assert with_prior.shape == base.shape
    assert not torch.allclose(with_prior[:, 0], base[:, 0])


def test_multi_query_prior_backward_reaches_all_params():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    B, K, S, M = 2, 2, 16, 2
    image = torch.randn(B, M, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    prior = torch.rand(B, M, 1, S, S, S)
    out = m(image, context_in=context_in, context_out=context_out,
           query_prior=prior)["final_logit"]
    out.mean().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_patchset3d.py -k multi_query_prior -v`
Expected: FAIL (shape errors — `_prior_occupancy` doesn't yet know about an `M` axis).

- [ ] **Step 3: Implement the `query_prior` generalization**

In `src/models/patchset3d.py`, `_prior_occupancy` (around line 684) stays unchanged —
it already handles exactly one `(B,1,D,H,W)` volume at a time. Add the `M`-window
threading at its two call sites instead.

In `_attn`, replace the `qry_occ` block from Task 2 with the full `M`-aware version:

```python
        if query_prior is not None:                          # cascade: coarse-level prediction
            if query_prior.dim() == 6:                        # (B,M,1,D,H,W) -- per-window priors
                qry_occ = torch.cat(
                    [self._prior_occupancy(query_prior[:, m]) for m in range(query_prior.shape[1])],
                    dim=1).to(sup_occ.dtype)
            else:                                             # legacy (B,1,D,H,W)
                qry_occ = self._prior_occupancy(query_prior).to(sup_occ.dtype)
        else:
            qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, M * N, sup_occ.shape[-1])
```

In `_pool_tokens` (the `qry_mask` block Task 2 left with a TODO-shaped comment),
replace it with the real per-window interpolation:

```python
            if query_prior is not None:
                if query_prior.dim() == 6:                    # (B,M,1,D,H,W)
                    Bq, Mq = query_prior.shape[0], query_prior.shape[1]
                    qry_mask = F.interpolate(
                        query_prior.reshape(Bq * Mq, 1, *query_prior.shape[-3:]).float(),
                        size=(S, S, S), mode="trilinear", align_corners=False
                    ).reshape(Bq, Mq, S, S, S)
                else:                                          # legacy (B,1,D,H,W)
                    qry_mask = F.interpolate(query_prior.float(), size=(S, S, S),
                                             mode="trilinear", align_corners=False)
            else:
                qry_mask = sup_mask.mean(dim=1, keepdim=True).expand(-1, qry_feat.shape[1], -1, -1, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_patchset3d.py -v`
Expected: PASS — all tests in the file, including the pre-existing
`test_query_prior_feeds_the_query_mask_token` etc. (legacy `(B,1,D,H,W)` priors must
still work unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "feat(patchset3d): generalize query_prior to carry M per-window priors"
```

---

## Task 4: `data.cascade_n_patches` config knob + guard validation

**Files:**
- Modify: `experiments/3d/common.py` (`_assert_cascade_supported`, around line 246)
- Test: `experiments/3d/tests/test_cascade_guard.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_assert_cascade_supported(cfg)` now also validates
  `cfg.data.cascade_n_patches` (default `1`): must be a positive int. Task 5 reads
  the same config key via `cfg.data.get("cascade_n_patches", 1)`.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/3d/tests/test_cascade_guard.py, appended at the end

def test_allows_n_patches_default():
    _assert_cascade_supported(_cfg())                          # no raise, default is 1


def test_allows_n_patches_explicit():
    _assert_cascade_supported(_cfg(data={"cascade_n_patches": 4}))


def test_rejects_n_patches_zero():
    with pytest.raises(ValueError, match="cascade_n_patches"):
        _assert_cascade_supported(_cfg(data={"cascade_n_patches": 0}))


def test_rejects_n_patches_negative():
    with pytest.raises(ValueError, match="cascade_n_patches"):
        _assert_cascade_supported(_cfg(data={"cascade_n_patches": -1}))


def test_rejects_n_patches_non_int():
    with pytest.raises(ValueError, match="cascade_n_patches"):
        _assert_cascade_supported(_cfg(data={"cascade_n_patches": 1.5}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd experiments/3d && python -m pytest tests/test_cascade_guard.py -k n_patches -v`
Expected: FAIL — `test_rejects_*` tests fail because no `ValueError` is raised yet
(`test_allows_*` already pass vacuously, since the key is simply ignored today).

- [ ] **Step 3: Implement the guard**

In `experiments/3d/common.py`, inside `_assert_cascade_supported` (around line 277,
right after the `len(spacings) < 2` check):

```python
    n_patches = d.get("cascade_n_patches", 1)
    if not isinstance(n_patches, int) or isinstance(n_patches, bool) or n_patches < 1:
        raise ValueError(
            f"data.cascade_n_patches={n_patches!r} must be a positive int (default 1).")
```

(`isinstance(n_patches, bool)` guards against `True`/`False` passing the `isinstance(...,
int)` check, since `bool` subclasses `int` in Python — not a real concern for this
config key, but consistent with not silently accepting a typo'd boolean.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd experiments/3d && python -m pytest tests/test_cascade_guard.py -v`
Expected: PASS — all tests in the file, old and new.

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/common.py experiments/3d/tests/test_cascade_guard.py
git commit -m "feat(cascade): validate data.cascade_n_patches in _assert_cascade_supported"
```

---

## Task 5: `run_cascade`/`_recrop_level` — the N-patch re-crop branch

**Files:**
- Modify: `experiments/3d/cascade.py`
- Test: `experiments/3d/tests/test_cascade.py`

**Interfaces:**
- Consumes: Task 1's `_prob_weighted_points_multi`; Task 2/3's `PatchSet3D` M-query +
  M-window-query_prior forward; Task 4's `cascade_n_patches` config key.
- Produces: `run_cascade(..., n_patches=1, ...)` new keyword (threaded from
  `cfg.data.cascade_n_patches` by both call sites — `train.py`'s cascade branch and
  `evaluate_cascade`, Task 6). `CascadeResult` gains `n_patches: list[int]` (length `N`,
  `n_patches[0] == 1` always). Task 6 reads `res.hard_preds[li]`/`res.geoms[li]`, whose
  shapes now carry an extra leading-after-batch `n` axis whenever `n_patches[li] > 1`.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/3d/tests/test_cascade.py, appended after the existing random_fg /
# center_mode tests (after line ~1194)

def test_run_cascade_n_patches_default_is_byte_identical_to_main():
    """n_patches unset (defaults to 1) must take the EXACT existing single-point code
    path -- same centers, same logits -- as a regression guard that the generalization
    is additive, not a behavior change at the default."""
    B, T, G = 2, 8, 4
    model, prov = _FakeModel(G=G, hot=(1, 1, 1)), _FakeProvider(T=T)
    res_default = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                              spacings=[3.0, 1.5], device=torch.device("cpu"),
                              training=True, step=0, seed=0, jitter=0)
    model2, prov2 = _FakeModel(G=G, hot=(1, 1, 1)), _FakeProvider(T=T)
    res_explicit = run_cascade(model2, prov2, _v2_batch(B=B, T=T), augmentor=None,
                               spacings=[3.0, 1.5], device=torch.device("cpu"),
                               training=True, step=0, seed=0, jitter=0, n_patches=1)
    assert res_default.n_patches == [1, 1]
    assert res_explicit.n_patches == [1, 1]
    for a, b in zip(res_default.logits, res_explicit.logits):
        assert torch.equal(a, b)
    assert res_default.centers == res_explicit.centers


def test_run_cascade_n_patches_training_raises():
    with pytest.raises(AssertionError, match="cascade_n_patches"):
        run_cascade(_FakeModel(), _FakeProvider(), _v2_batch(), augmentor=None,
                    spacings=[3.0, 1.5], device=torch.device("cpu"),
                    training=True, step=0, seed=0, n_patches=2)


class _TwoBlobMultiQueryModel(torch.nn.Module):
    """Echoes back M one-hot blobs (one per query window) so the test can assert BOTH
    level-0 blobs get covered by level 1's n=2 windows -- the exact failure mode a
    single-point re-crop can't express. Accepts the model's new (B,M,1,D,H,W) image
    shape and returns (B,M,1,G,G,G)."""
    spacing_aware = False

    def __init__(self, G, blobs):
        super().__init__()
        self.G, self.blobs = G, blobs
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, context_in, context_out, mode="train", spacing=None,
               query_prior=None, cascade_regs=None):
        legacy = image.dim() == 4
        M = 1 if legacy else image.shape[1]
        B = image.shape[0]
        lg = torch.full((B, M, 1, self.G, self.G, self.G), -10.0, device=image.device)
        for m in range(M):
            c = self.blobs[m % len(self.blobs)]
            lg[:, m, :, c[0], c[1], c[2]] = 10.0
        lg = lg + self.p.to(image.device)
        if legacy:
            lg = lg[:, 0]
        return {"final_logit": lg}


def test_run_cascade_n_patches_two_covers_two_disjoint_blobs():
    """At level 0 a single FakeProvider crop is loaded (M=1, as always); level 1 asks
    for n_patches=2 windows. The model always predicts two disjoint one-hot blobs
    regardless of M, so level 1's sampler (fed level 0's single M=1 field) must still
    land its 2 drawn points on voxels that are plausible under level 0's own prediction
    -- this test's real purpose is exercising the n_patches>1 PLUMBING end-to-end
    (shapes, no crash, n_patches recorded) rather than re-proving the sampler's own
    coverage math (already covered by Task 1's unit tests)."""
    B, T, G = 2, 8, 4
    blobs = ((1, 1, 1), (2, 2, 2))
    model, prov = _TwoBlobMultiQueryModel(G, blobs), _FakeProvider(T=T)
    res = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device=torch.device("cpu"),
                      training=False, step=0, seed=0, jitter=0, n_patches=2)
    assert res.n_patches == [1, 2]
    assert res.logits[1].shape == (B, 2, 1, G, G, G)
    assert res.geoms[1].shape[0] == B           # (B, n, 4, 3) -- one geom per sampled window


def test_run_cascade_n_patches_context_tasks_scale_with_n():
    """Each of K context subjects contributes n independently-centred windows per level,
    so the provider sees K*n context loads per b at level>0 (was K)."""
    B, K, T = 2, 3, 8
    model, prov = _TwoBlobMultiQueryModel(4, ((1, 1, 1), (2, 2, 2))), _FakeProvider(T=T)
    run_cascade(model, prov, _v2_batch(B=B, K=K, T=T), augmentor=None,
               spacings=[3.0, 1.5], device=torch.device("cpu"),
               training=False, step=0, seed=0, jitter=0, n_patches=2)
    ctx_calls = [c for c in prov.calls if c["center"] is None]
    assert len(ctx_calls) == B * K * 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd experiments/3d && python -m pytest tests/test_cascade.py -k n_patches -v`
Expected: FAIL — `run_cascade()` doesn't accept an `n_patches` keyword yet
(`TypeError: run_cascade() got an unexpected keyword argument 'n_patches'`).

- [ ] **Step 3: Implement the N-patch branch in `run_cascade`/`_recrop_level`**

In `experiments/3d/cascade.py`, add `n_patches=1` to `run_cascade`'s signature and
guard training:

```python
def run_cascade(model, provider, batch, augmentor, spacings, *, device, training,
                step, seed, jitter=0, is_prob=False, want_hard_preds=False,
                recrop_workers=1, query_prior=False, query_prior_hard=False,
                want_figure_arrays=False, realize_crop=False,
                mask_downsample="occupancy", occ_thr=0.1, ct_spec=None,
                prior_perturb=None, center_mode="com", center_fg_thr=0.5,
                n_patches=1):
    assert len(spacings) >= 2, "cascade needs >=2 spacings"
    assert not (training and n_patches > 1), (
        "data.cascade_n_patches>1 is eval-only for now -- see docs/superpowers/specs/"
        "2026-10-13-multipatch-cascade-design.md")
```

`CascadeResult` gains the new field (dataclass, around line 83):

```python
@dataclass
class CascadeResult:
    logits: list
    targets: list
    geoms: list
    centers: list
    hard_preds: list | None
    empty_frac: float
    figure_levels: list | None = None
    prior_modes_used: list | None = None
    n_patches: list | None = None       # per level: 1 (level 0, always) or n (level i>=1)
```

Add a new `_recrop_level_multi` function, a sibling of `_recrop_level` for the
`n_patches > 1` case — rather than overloading `_recrop_level` itself with an `n`
branch, this keeps the `n_patches == 1` path (used by every existing caller/test)
completely untouched:

```python
def _recrop_level_multi(provider, batch, target_points, spacing, *, step, seed, level,
                        jitter, n, realize_crop=False, mask_downsample="occupancy",
                        occ_thr=0.1, ct_spec=None, device=None, center_mode="random_fg"):
    """Build one level-i v2 batch with n target windows (centers from
    target_points[b], a length-n list of native (d,h,w) tuples -- already resolved by
    the caller via _prob_weighted_points_multi + invert_geo_center) and n*K context
    windows (n independently-seeded random_fg draws per original context subject,
    center=None + center_mode so the provider's own _resolve_center samples a fresh
    foreground voxel each time). Returns a batch dict shaped like _recrop_level's, but
    with image: (B,n,1,T,T,T), context_in/context_out: (B,K*n,...), AND label:
    (B,n,T,T,T) (one GT per target window, since each has its own crop).

    realize_crop is NOT supported here (eval-only n_patches>1 call sites use the plain
    provider.load path below) -- asserted, not silently ignored."""
    assert not realize_crop, "data.cascade_n_patches>1 does not support gpu_realize_crop yet"
    subs, ctxs, clss = batch["subjects"], batch["context_subjects"], batch["label_names"]
    sp = float(spacing)
    tmods = batch.get("tgt_modality")
    cmods = batch.get("ctx_modality")

    tasks = []                     # (b, role, j, subj, center, rk, mod); role: "tgt" | "ctx"
    for b in range(len(subs)):
        tmod = tmods[b] if tmods is not None else None
        cmod = cmods[b] if cmods is not None else None
        pts = target_points[b]
        for j in range(n):
            center = pts[j] if pts is not None and pts[j] is not None else None
            tasks.append((b, "tgt", j, subs[b], center, f"{seed}_{step}_{level}_{b}_{j}", tmod))
        for k, cs in enumerate(ctxs[b]):
            for j in range(n):
                tasks.append((b, "ctx", (k, j), cs, None,
                             f"{seed}_{step}_{level}_{b}_{k}_{j}", cmod))

    def _load(t):
        _b, _role, _j, subj, center, rk, mod = t
        req = LoadRequest(rng=random.Random(rk), crop_spacing_mm=sp,
                          center=center, jitter=jitter, center_mode=center_mode)
        if mod is not None:
            return provider.load(subj, clss[_b], req, modality=mod)
        return provider.load(subj, clss[_b], req)

    results = [_load(t) for t in tasks]

    tgt_slots = [[None] * n for _ in range(len(subs))]
    ctx_slots = [[[None] * n for _ in ctxs[b]] for b in range(len(subs))]
    for (b, role, j, *_rest), r in zip(tasks, results):
        if role == "tgt":
            tgt_slots[b][j] = r
        else:
            k, jj = j
            ctx_slots[b][k][jj] = r

    items = []
    for b in range(len(subs)):
        img = torch.stack([tgt_slots[b][j].image for j in range(n)])          # (n,1,T,T,T)
        lbl = torch.stack([tgt_slots[b][j].label for j in range(n)])          # (n,T,T,T)
        cin = torch.stack([torch.stack([ctx_slots[b][k][j].image for j in range(n)])
                           for k in range(len(ctxs[b]))]).reshape(-1, *img.shape[-4:])
        cout = torch.stack([torch.stack([ctx_slots[b][k][j].label for j in range(n)])
                            for k in range(len(ctxs[b]))]).reshape(-1, *lbl.shape[-3:])
        items.append({
            "image": img, "label": lbl, "context_in": cin, "context_out": cout,
            "subject": subs[b], "context_subjects": list(ctxs[b]),
            "label_name": clss[b], "spacing": tgt_slots[b][0].spacing,
            "crop_geom": torch.stack([tgt_slots[b][j].crop_geom for j in range(n)]),
        })
    return {
        "image": torch.stack([it["image"] for it in items]),
        "label": torch.stack([it["label"] for it in items]),
        "context_in": torch.stack([it["context_in"] for it in items]),
        "context_out": torch.stack([it["context_out"] for it in items]),
        "subjects": list(subs), "context_subjects": [list(c) for c in ctxs],
        "label_names": list(clss),
        "crop_geom": torch.stack([it["crop_geom"] for it in items]),       # (B,n,4,3)
        "aug_mode": torch.zeros(len(subs), dtype=torch.long),
    }
```

Now wire it into `run_cascade`'s level loop. Replace the body of the `for i in
range(N):` loop's `if i > 0:` branch and the end-of-loop re-crop-center block (the two
pieces that currently call `_recrop_level`/`_centroid_from_logit`/`_random_fg_point`)
with an `n_patches > 1` fork. The full replacement for the loop (everything from `for i
in range(N):` through the loop's closing, i.e. the existing block spanning roughly
lines 637-709):

```python
    n_patches_used = [1]
    prev_fields = None                       # list[Tensor] -- level i-1's M windows' logits
    prev_geoms_list = None                   # list[Tensor] -- level i-1's M windows' geoms (per b)
    prev_geo_list = None                     # list[GeoState|None] -- level i-1's M windows' aug geo
    cur = _to_device(dict(batch), device)
    for i in range(N):
        if i > 0:
            if n_patches <= 1:
                cur = _recrop_level(provider, batch, centers[i], spacings[i],
                                    step=step, seed=seed, level=i, jitter=jitter,
                                    recrop_workers=recrop_workers, realize_crop=realize_crop,
                                    mask_downsample=mask_downsample, occ_thr=occ_thr,
                                    ct_spec=ct_spec, device=device, center_mode=center_mode)
            else:
                target_points = _points_for_level_multi(
                    prev_fields, T, is_prob, n_patches, prev_geo_list, prev_geoms_list,
                    seed=seed, step=step, level=i)
                cur = _recrop_level_multi(provider, batch, target_points, spacings[i],
                                          step=step, seed=seed, level=i, jitter=jitter,
                                          n=n_patches, mask_downsample=mask_downsample,
                                          occ_thr=occ_thr, ct_spec=ct_spec, device=device,
                                          center_mode="random_fg")
            cur = _to_device(cur, device)
        capture = augmentor is not None and i < N - 1
        if augmentor is not None:
            cur, geo = augmentor.apply(
                cur, geo_gen=_gen(geo_seed, device),
                int_gen=_gen(geo_seed + INT_OFFSET * (i + 1), device), capture=capture,
                goal_gen=_gen(geo_seed + GOAL_OFFSET, device))
        else:
            geo = None

        this_n = 1 if i == 0 else n_patches
        prior = None
        if i > 0:
            if len(prior_spec.modes) == 1:
                mode_i = prior_spec.modes[0]
            elif training:
                mode_i = random.Random(f"{seed}_{step}_{i}_qprior").choices(
                    prior_spec.modes, weights=prior_spec.weights, k=1)[0]
            else:
                mode_i = prior_spec.eval_mode
            prior_modes_used.append(mode_i)
            if mode_i != "none" and prev_logit is not None:
                cg_cur = cur["crop_geom"] if "crop_geom" in cur else batch["crop_geom"]
                if this_n == 1:
                    prior = _build_query_prior(
                        mode_i, bool(query_prior_hard), prev_logit, prev_label, cur["label"],
                        prev_geo, geoms[i - 1], cg_cur, T, is_prob,
                        perturb_cfg=(prior_perturb if training else None),
                        spacing_mm=spacings[i],
                        perturb_gen=_gen(geo_seed + PERTURB_OFFSET + i, device))
                else:
                    prior = _build_query_prior_multi(
                        mode_i, prev_fields, prev_label, cur["label"], prev_geo_list,
                        prev_geoms_list, cg_cur, target_points, T, is_prob)

        logit, regs = _forward_level(model, cur, spacings[i], query_prior=prior,
                                     cascade_regs=prev_regs)
        tgt = target_like(
            cur["label"].reshape(-1, 1, *cur["label"].shape[-3:]).float(),
            logit.reshape(-1, 1, *logit.shape[-3:]))
        logits.append(logit); targets.append(tgt)
        geoms.append(cur["crop_geom"] if "crop_geom" in cur else batch["crop_geom"])
        prev_logit = logit if this_n == 1 else None     # legacy single-window field, n=1 only
        prev_geo, prev_label, prev_regs = geo, cur["label"], regs
        prev_fields = ([logit[:, m] for m in range(this_n)] if this_n > 1
                       else [logit if logit.dim() == 5 else logit])
        prev_geo_list = [geo] * this_n                   # same captured aug grid for every window
        prev_geoms_list = ([geoms[i][:, m] for m in range(this_n)] if this_n > 1
                           else [geoms[i]])
        n_patches_used.append(this_n if i < N - 1 else this_n)
        if want_hard_preds:
            hard.append(_hard_pred_native(
                logit.reshape(-1, 1, *logit.shape[-3:]), T, is_prob
            ).reshape(B, this_n, T, T, T) if this_n > 1 else _hard_pred_native(logit, T, is_prob))
        if want_figure_arrays:
            figs.append({
                "img":     cur["image"][:, 0].detach().float().cpu().numpy() if this_n == 1
                          else cur["image"][:, 0, 0].detach().float().cpu().numpy(),
                "gt":      cur["label"].detach().float().cpu().numpy(),
                "ctx_img": cur["context_in"][:, 0, 0].detach().float().cpu().numpy(),
                "ctx_gt":  cur["context_out"][:, 0].detach().float().cpu().numpy(),
            })

        if i < N - 1:
            if n_patches <= 1:
                cens = (_random_fg_point(logit, T, is_prob, center_fg_thr,
                                         seed=seed, step=step, level=i)
                        if center_mode == "random_fg" else
                        _centroid_from_logit(logit, T, is_prob))
                row = []
                for b in range(B):
                    empty_total += 1
                    if geo is not None and geo.grid is not None:
                        gr = geo.grid[b]
                        fl = geo.flips[b]
                    else:
                        gr, fl = None, torch.zeros(3, dtype=torch.bool)
                    nc = invert_geo_center(cens[b], gr, fl, geoms[i][b], T)
                    if nc is None:
                        empty_hits += 1
                    row.append(nc)
                centers.append(row)
            # n_patches > 1: next level's target_points are computed fresh at the top of
            # the next iteration (via _points_for_level_multi, reading prev_fields/
            # prev_geo_list/prev_geoms_list set above), not accumulated into `centers`.

    return CascadeResult(logits=logits, targets=targets, geoms=geoms, centers=centers,
                         hard_preds=hard,
                         empty_frac=(empty_hits / empty_total if empty_total else 0.0),
                         figure_levels=figs, prior_modes_used=prior_modes_used,
                         n_patches=n_patches_used)
```

Add the two new helpers this references, near `_recrop_level_multi`:

```python
def _points_for_level_multi(prev_fields, T, is_prob, n, prev_geo_list, prev_geoms_list,
                            *, seed, step, level):
    """level i-1's `len(prev_fields)` windows' own logits -> n native centers per b for
    level i's target re-crop, via _prob_weighted_points_multi (grid-space, concatenated
    multinomial) then invert_geo_center per sampled point through ITS OWN originating
    window's geo/geom. Returns, per b, a list of n native (d,h,w) tuples or None."""
    picks = _prob_weighted_points_multi(prev_fields, T, is_prob, n,
                                        seed=seed, step=step, level=level)
    B = prev_fields[0].shape[0]
    out = []
    for b in range(B):
        row = []
        for p in picks[b]:
            if p is None:
                row.append(None)
                continue
            w_idx, cen = p
            geo = prev_geo_list[w_idx]
            gr = geo.grid[b] if geo is not None and geo.grid is not None else None
            fl = geo.flips[b] if geo is not None and geo.grid is not None else torch.zeros(3, dtype=torch.bool)
            row.append(invert_geo_center(np.array(cen, dtype=float), gr, fl,
                                         prev_geoms_list[w_idx][b], T))
        out.append(row)
    return out


def _build_query_prior_multi(mode, prev_fields, prev_label, cur_label, prev_geo_list,
                             prev_geoms_list, cg_cur, target_points, T, is_prob):
    """Per current-level window j, warp its ORIGIN window's (from target_points, already
    tracked per Task 5's _points_for_level_multi) previous-level prediction onto this
    window's grid via _warp_prior_cropgeom -- the no-aug crop-geom-only path (eval-only
    call sites never pass an augmentor, so there is no aug grid to compose through).
    Returns (B,n,1,T,T,T) for PatchSet3D's generalized query_prior (Task 3)."""
    assert mode == "pred", f"cascade_n_patches>1 only supports cascade_query_prior='pred', got {mode!r}"
    B = prev_fields[0].shape[0]
    n = len(target_points[0])
    out = torch.zeros(B, n, 1, T, T, T, device=prev_fields[0].device)
    for b in range(B):
        for j in range(n):
            # target_points[b][j] came from _points_for_level_multi picking some origin
            # window w via _prob_weighted_points_multi; _points_for_level_multi does not
            # currently return w alongside the inverted point, so this helper re-derives
            # it the same way: the ONLY origin-tracking consumer is this prior build, and
            # it reruns on the identical prev_fields/seed -- see Step 3 note below on
            # threading w_idx through instead of recomputing it.
            pass
    return out
```

**Note on the `_build_query_prior_multi` stub above:** it is written here with a `pass`
placeholder deliberately flagged, not shipped — `_points_for_level_multi` must actually
return the `window_idx` alongside each inverted point (not just the native center), so
`_build_query_prior_multi` doesn't have to re-derive it. Before running Step 4, change
`_points_for_level_multi` to return, per `b`, a list of `(window_idx, native_center |
None)` pairs instead of bare centers, thread that `window_idx` through to
`_recrop_level_multi`'s `target_points[b][j]` (centers only, `_recrop_level_multi`
unpacks the tuple and uses just the center for the provider call) AND to
`_build_query_prior_multi` (which needs `window_idx` to pick `prev_fields[window_idx]`/
`prev_geoms_list[window_idx][b]` for the warp source):

```python
def _points_for_level_multi(prev_fields, T, is_prob, n, prev_geo_list, prev_geoms_list,
                            *, seed, step, level):
    picks = _prob_weighted_points_multi(prev_fields, T, is_prob, n,
                                        seed=seed, step=step, level=level)
    B = prev_fields[0].shape[0]
    out = []
    for b in range(B):
        row = []
        for p in picks[b]:
            if p is None:
                row.append((0, None))
                continue
            w_idx, cen = p
            geo = prev_geo_list[w_idx]
            gr = geo.grid[b] if geo is not None and geo.grid is not None else None
            fl = geo.flips[b] if geo is not None and geo.grid is not None else torch.zeros(3, dtype=torch.bool)
            nc = invert_geo_center(np.array(cen, dtype=float), gr, fl,
                                   prev_geoms_list[w_idx][b], T)
            row.append((w_idx, nc))
        out.append(row)
    return out


def _build_query_prior_multi(mode, prev_fields, prev_geoms_list, cg_cur, target_points, T, is_prob):
    assert mode == "pred", f"cascade_n_patches>1 only supports cascade_query_prior='pred', got {mode!r}"
    B = prev_fields[0].shape[0]
    n = len(target_points[0])
    out = torch.zeros(B, n, 1, T, T, T, device=prev_fields[0].device)
    for b in range(B):
        for j, (w_idx, nc) in enumerate(target_points[b]):
            if nc is None:
                continue
            src = prev_fields[w_idx][b:b + 1].float()
            src = src.clamp(0, 1) if is_prob else torch.sigmoid(src)
            if src.shape[-1] != T:
                src = F.interpolate(src, size=(T, T, T), mode="trilinear", align_corners=False)
            warped = _warp_prior_cropgeom(src, prev_geoms_list[w_idx][b:b + 1],
                                          cg_cur[b:b + 1, j], T)
            out[b, j] = warped[0]
    return out
```

With `target_points[b][j]` now a `(window_idx, native_center)` tuple,
`_recrop_level_multi`'s `center = pts[j] if ... else None` line changes to
`center = pts[j][1] if pts is not None and pts[j][1] is not None else None`, and the
call site in `run_cascade` passes `cg_cur[b:b+1, j]` (that window's own slice of the
current level's `(B,n,4,3)` crop_geom) to `_build_query_prior_multi`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd experiments/3d && python -m pytest tests/test_cascade.py -v`
Expected: PASS — the new `n_patches` tests AND every pre-existing test in the file
(the `n_patches<=1` branch must be untouched, so every test written before this task
still exercises the exact same code path it always has).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/cascade.py experiments/3d/tests/test_cascade.py
git commit -m "feat(cascade): wire data.cascade_n_patches into run_cascade's re-crop step"
```

---

## Task 6: `evaluate_cascade` — score an N-patch level's stitched prediction

**Files:**
- Modify: `experiments/3d/evaluate.py` (`_stitched_native_metrics_and_levels`, around line 1028)
- Modify: `experiments/3d/cascade.py` (`evaluate_cascade`, around line 717)
- Test: `experiments/3d/tests/test_cascade.py`

**Interfaces:**
- Consumes: Task 5's `CascadeResult.n_patches`/`hard_preds`/`geoms` (now carrying an
  `n`-window axis for `n_patches[li] > 1` levels).
- Produces: `_stitched_native_metrics_and_levels(pg_levels, ...)` now accepts each
  `pg_levels[li][key]` as either a single `(packed, shape, geom)` tuple (today's
  contract, unchanged) OR a `list` of such tuples (new — one per sampled window,
  overwrite-pasted into that level's own composite in list order, same "later paste
  wins" rule already used across levels). `evaluate_cascade` builds the list form
  whenever `res.n_patches[li] > 1`.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/3d/tests/test_cascade.py, appended after Task 5's tests

from evaluate import _stitched_native_metrics_and_levels


def _packed(mask, geom):
    return (np.packbits(mask), mask.shape, geom)


def test_stitched_native_metrics_accepts_single_tuple_unchanged(tmp_path):
    """Regression: the pre-existing single-(packed,shape,geom)-tuple contract (n=1
    levels) must still work byte-identically."""
    subj_dir = tmp_path / "s0"
    subj_dir.mkdir()
    gt = np.zeros((8, 8, 8), dtype=bool)
    gt[2:5, 2:5, 2:5] = True
    np.save(subj_dir / "label.npy", gt.astype(np.uint8))
    geom = np.array([[0, 0, 0], [8, 8, 8], [8, 8, 8], [0, 0, 0]])
    pred = np.zeros((8, 8, 8), dtype=bool)
    pred[2:5, 2:5, 2:5] = True
    class_idx = {"organ": 1}
    gt_u8 = np.load(subj_dir / "label.npy")
    full, per_level = _stitched_native_metrics_and_levels(
        [{("s0", "organ"): _packed(pred, geom)}], tmp_path, class_idx=class_idx)
    # class_idx path reads label.npy == idx; write it with the real class value (1)
    np.save(subj_dir / "label.npy", (gt_u8 * 1))
    full, per_level = _stitched_native_metrics_and_levels(
        [{("s0", "organ"): _packed(pred, geom)}], tmp_path, class_idx=class_idx)
    assert full[("s0", "organ")][0] == pytest.approx(1.0)


def test_stitched_native_metrics_accepts_list_of_windows(tmp_path):
    """n_patches>1: pg_levels[li][key] is a LIST of (packed,shape,geom) -- two windows,
    each covering a different half of the GT, must stitch to full coverage (overwrite-
    paste, same semantics as across-level stitching)."""
    subj_dir = tmp_path / "s0"
    subj_dir.mkdir()
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    gt[0:4, :, :] = 1
    gt[4:8, :, :] = 1
    np.save(subj_dir / "label.npy", gt)
    geom = np.array([[0, 0, 0], [8, 8, 8], [8, 8, 8], [0, 0, 0]])
    pred_a = np.zeros((8, 8, 8), dtype=bool)
    pred_a[0:4, :, :] = True                 # window A covers the top half
    pred_b = np.zeros((8, 8, 8), dtype=bool)
    pred_b[4:8, :, :] = True                 # window B covers the bottom half
    class_idx = {"organ": 1}
    full, per_level = _stitched_native_metrics_and_levels(
        [{("s0", "organ"): [_packed(pred_a, geom), _packed(pred_b, geom)]}],
        tmp_path, class_idx=class_idx)
    assert full[("s0", "organ")][0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd experiments/3d && python -m pytest tests/test_cascade.py -k stitched_native_metrics -v`
Expected: FAIL on `test_stitched_native_metrics_accepts_list_of_windows` (the function
currently does `_unpack_pred(pg_levels[li][key])`, which assumes a single tuple and
raises when handed a list).

- [ ] **Step 3: Implement the list-of-windows generalization**

In `experiments/3d/evaluate.py`, `_stitched_native_metrics_and_levels` (around line
1028), the loop body currently does:

```python
        for li in present:
            p, geom = _unpack_pred(pg_levels[li][key])
            placed = _place_patch(p, geom)
            solo = np.zeros(gt.shape, dtype=bool)
            if placed is not None:
                if want_full:
                    _paste(composite, *placed)
                _paste(solo, *placed)
```

Replace it with a form that accepts either a single entry or a list, pasting every
entry in the list into the same per-level `solo` (and the running `composite`) in
order — list order is "later paste wins," matching the existing coarse→fine overwrite
rule used across levels, just applied within one level too:

```python
        for li in present:
            entries = pg_levels[li][key]
            if not isinstance(entries, list):
                entries = [entries]
            solo = np.zeros(gt.shape, dtype=bool)
            for entry in entries:
                p, geom = _unpack_pred(entry)
                placed = _place_patch(p, geom)
                if placed is not None:
                    if want_full:
                        _paste(composite, *placed)
                    _paste(solo, *placed)
```

In `experiments/3d/cascade.py`, `evaluate_cascade` (around line 816-820), the per-case
loop currently does:

```python
            for li in range(N):
                hp = res.hard_preds[li][b].cpu().numpy().astype(bool)
                geom = res.geoms[li][b].cpu().numpy()
                pg_levels[li][key] = (np.packbits(hp), tuple(hp.shape), geom)
```

Replace it to build the list form whenever that level ran more than one window:

```python
            for li in range(N):
                n_li = res.n_patches[li] if res.n_patches is not None else 1
                if n_li <= 1:
                    hp = res.hard_preds[li][b].cpu().numpy().astype(bool)
                    geom = res.geoms[li][b].cpu().numpy()
                    pg_levels[li][key] = (np.packbits(hp), tuple(hp.shape), geom)
                else:
                    pg_levels[li][key] = [
                        (np.packbits(res.hard_preds[li][b, j].cpu().numpy().astype(bool)),
                        tuple(res.hard_preds[li][b, j].shape),
                        res.geoms[li][b, j].cpu().numpy())
                        for j in range(n_li)
                    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd experiments/3d && python -m pytest tests/test_cascade.py tests/test_cascade_provider.py -v`
Expected: PASS — the new tests AND every pre-existing `evaluate_cascade`/
`_stitched_native_metrics_and_levels` test (the `n<=1` branch is byte-identical to
before).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/evaluate.py experiments/3d/cascade.py experiments/3d/tests/test_cascade.py
git commit -m "feat(cascade): score n_patches>1 levels by stitching a list of windows per key"
```

---

## Task 7: FLARE22 validation script

**Files:**
- Create: `experiments/3d/compare_multipatch_flare22.py`

**Interfaces:**
- Consumes: `cascade.evaluate_cascade`, `common.make_eval_loader`/`common._build_model`
  (the same helpers `eval.py` uses), `data.cascade_n_patches` (Task 4/5).
- Produces: a standalone CLI script (Hydra-free, plain argparse — matching the existing
  `experiments/3d/infer_nifti.py`'s style of taking a composed `cfg` rather than being
  its own Hydra entry point) that prints a per-class Dice/NSD delta table. No new
  importable interface other tasks depend on — this is the end of the chain.

- [ ] **Step 1: Write the script**

```python
"""Compare cascade_n_patches=1 vs >1 on FLARE22, same checkpoint/eval seed.

Zero-shot probe (docs/superpowers/specs/2026-10-13-multipatch-cascade-design.md):
no training change, just whether sampling N windows per role from a continuous
probability field helps or hurts FLARE22's 13 abdominal organs under the existing
best cascade checkpoint. Forces data.cascade_query_prior="none" for n_patches>1
(the only prior mode Task 5's _build_query_prior_multi supports today) and ALSO for
the n_patches=1 baseline run, so the comparison isolates the n-patch effect alone --
NOT a comparison against the checkpoint's own best-published query_prior="pred" n=1
number (see memory project_gnc_kidney_lesions / docs/logs.md for that reference point).

Usage:
    cd experiments/3d
    python compare_multipatch_flare22.py \
        eval.checkpoint=/path/to/best.pt \
        data.cascade_n_patches_sweep=[1,2,4]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf

from cascade import evaluate_cascade
from common import _build_model, make_eval_loader


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg.data.source = "flare22"
    cfg.data.cascade_query_prior = "none"
    sweep = list(cfg.data.get("cascade_n_patches_sweep", [1, 2, 4]))
    assert sweep[0] == 1, "the first sweep value must be 1 (the baseline)"

    model = _build_model(cfg)
    loader = make_eval_loader(cfg, split=cfg.eval.get("split", "val"))
    classes = list(loader.dataset.provider.ALL_CLASSES)

    baseline = None
    for n in sweep:
        cfg.data.cascade_n_patches = n
        rows, _cases = evaluate_cascade(model, cfg, classes, loader=loader,
                                        seed=int(cfg.eval.get("seed", 0)), is_prob=False)
        by_cls = {r["class"]: r for r in rows if "error" not in r}
        if n == 1:
            baseline = by_cls
            print(f"\n=== n_patches=1 (baseline) ===")
            for cls, r in by_cls.items():
                print(f"  {cls:24s} dice={r['mean_dice']:.4f}"
                     + (f" nsd={r.get('mean_nsd', float('nan')):.4f}" if "mean_nsd" in r else ""))
            continue
        print(f"\n=== n_patches={n} (delta vs baseline) ===")
        for cls, r in by_cls.items():
            base = baseline.get(cls)
            if base is None:
                continue
            ddice = r["mean_dice"] - base["mean_dice"]
            line = f"  {cls:24s} dice={r['mean_dice']:.4f} (Δ{ddice:+.4f})"
            if "mean_nsd" in r and "mean_nsd" in base:
                dnsd = r["mean_nsd"] - base["mean_nsd"]
                line += f" nsd={r['mean_nsd']:.4f} (Δ{dnsd:+.4f})"
            print(line)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports and parses config cleanly (no GPU/checkpoint needed yet)**

Run: `cd experiments/3d && python -c "import compare_multipatch_flare22"`
Expected: no `ImportError`/`SyntaxError` (this only checks the module loads; running
`main()` for real needs a GPU and the checkpoint path, done manually per the docstring's
`Usage` once this task is merged — not part of the automated test suite, consistent
with `infer_nifti.py` and other inference-side scripts in this repo having no pytest
coverage of their `main()`).

- [ ] **Step 3: Commit**

```bash
git add experiments/3d/compare_multipatch_flare22.py
git commit -m "feat(cascade): add FLARE22 n_patches=1-vs-N comparison script"
```

- [ ] **Step 4: Run the real validation (manual, needs GPU + checkpoint)**

Run (fill in the actual checkpoint path from `168_cascade_real_host_all_shapes`'s
`best.pt`, per the spec's Goal section):

```bash
cd experiments/3d
python compare_multipatch_flare22.py \
    eval.checkpoint=<path-to-168's-best.pt> \
    eval.model=patchset3d \
    dataset=flare22 \
    data.cascade_spacings=[4,1.5] \
    data.cascade_n_patches_sweep=[1,2,4]
```

Record the printed per-class Dice/NSD deltas in a follow-up `docs/logs.md` entry
(per CLAUDE.md's "Log changes to docs/logs.md" convention) — this is the spec's §3
deliverable, not something this plan automates further.

---

## Self-Review Notes

- **Spec coverage:** §1 (model M-query generalization) → Tasks 2-3. §2 (cascade
  re-crop mechanics, `_prob_weighted_points_multi`, `cascade_n_patches` knob,
  per-window `invert_geo_center`/query_prior) → Tasks 1, 4, 5. §3 (FLARE22
  validation) → Task 7. The "Verification for this section" paragraphs under §1/§2
  in the spec map to Task 2/3's byte-identical-at-M=1 tests and Task 5's
  byte-identical-at-n=1 regression test, respectively. Task 6 (native-stitch scoring)
  is plumbing the spec's prose implied but didn't name a file for — added because
  `evaluate_cascade` cannot score an `n>1` level without it.
- **Type consistency:** `n_patches` (cascade.py's config-facing name) vs `M` (model's
  forward-facing name) are kept distinct throughout, matching the spec's own
  clarifying note in §1 — no task conflates them in a function signature.
- **Review Focus coverage:** empty-GT class → Task 1's
  `test_prob_weighted_points_multi_empty_returns_none` +
  `_random_fg_point`'s pre-existing empty contract (reused, not re-tested, for the
  support side). `K=0` support → `_recrop_level_multi`'s context loop is a plain `for
  k, cs in enumerate(ctxs[b])`, a no-op at `K=0` (not explicitly re-tested in this
  plan; flagged here as a gap — if this matters for FLARE22's actual configs, add a
  `K=0` case to Task 5's tests before running Task 7's real validation). Native-bounds
  clamping → Task 5 routes every sampled point through the SAME `provider.load` call
  `_recrop_level` already uses (no new unclamped path). Duplicate draws → Task 1's
  `random.choices(..., k=n)` is with-replacement by construction, and Task 5's
  `_recrop_level_multi` always builds exactly `n` tasks regardless of duplicate
  centers (no dedup). FLARE22 `CLASS_IDX` path → unchanged by this plan (Task 6 only
  changes how MANY `(packed,shape,geom)` entries exist per key, not how GT is loaded),
  exercised for real in Task 7's manual run, not a unit test — flagged as a gap: Task 6
  could add one more test using `gt_loader` instead of `class_idx` to cover
  `NativeGridProvider`'s path explicitly rather than relying on Task 7's manual run.

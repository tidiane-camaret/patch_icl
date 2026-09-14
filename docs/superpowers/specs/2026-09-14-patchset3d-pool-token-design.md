# PatchSet3D IRIS-style foreground-masked pooling token — design

**Date:** 2026-09-14
**Model:** `src/models/patchset3d.py` (`PatchSet3D`)
**Status:** approved design, ready for implementation plan
**Reference:** `docs/methods/iris.md` §4.1 (Foreground stream, `T_f`)

## Goal

Add a per-volume "class prototype" token — a foreground-masked average of fine-resolution
image features — mirroring Iris's `T_f = Pool(Upsample(F_s) ⊙ y_s)`. Iris's own ablation
credits masking *after* upsampling (not before) for a large small-object Dice gain
(62.13 → 78.92); `PatchSet3D`'s existing per-cell tokens live at the coarse `R³` grid, so this
feature deliberately pools over `arch.fine_decode`'s finer unpooled encoder stage instead, to
reproduce the property Iris's ablation actually found valuable, not a diminished low-res
version of it.

## Background: what's already there, what's missing

`PatchSet3D` has no analog of `T_f` today. The closest existing quantity, `qry_occ =
sup_occ.mean(dim=1, keepdim=True)` in `_attn` (the query's support-mean fallback prior), pools
**mask occupancy**, not image features gated by the mask — a different thing entirely.
`arch.fine_decode` (existing feature) already gives the model access to an unpooled,
finer-than-`R³` encoder feature map, but only for the **query** row (`forward`'s `rows =
arange(B)*T + K`) — a deliberate memory-saving choice, since only the query needs per-cell
detail for its own decode.

## Decisions (from brainstorming)

1. **Resolution:** pool over `arch.fine_decode`'s finest requested stage (`min(self.fine_stage)`
   — reuses the existing config, no new stage knob), not the coarse `R³` grid.
2. **Scope:** compute a pool token for **both** support and query volumes (not support-only
   like Iris) — query's fine map is already computed today for its own decode, so this is
   near-free; support's fine maps are the new cost.
3. **Memory:** accept the `K`-scaling cost of keeping unpooled maps for all `K+1` volumes
   (extends `forward`'s `rows` selection) rather than compromise on resolution.
4. **Column content:** the pooled+projected vector is broadcast to **both** c-axis columns
   (img, mask) — like `ThinkingRows.tokens`/cascade-memory — since there is no natural
   distinct "mask-side" content for a pure image-feature prototype (nothing to pool *against*
   on that side; the mask **is** the pooling weight, not a channel being pooled).
5. **Insertion:** as extra prefix rows (mirrors `cascade_regs`/`mem`'s existing mechanism)
   rather than as an extra per-volume cell — keeps the `N`-cells-per-volume invariant that
   `register_routed`, `_rope`, `_grid_tokens`, and `context_id_embed`'s `repeat_interleave`
   all assume completely untouched.

## Refinement made during spec write-up (beyond the chat-approved sketch)

The chat design said the query's fine pooling mask comes from "upsampling the existing
`qry_occ`." On closer look this is unnecessarily indirect: `qry_occ` is already a **discretized,
`R³`-tiled** representation (`mask_patch_size`-shaped tiles, not a plain volume), so upsampling
it would require first inverting the tiling (`_tile_logits`-style) before interpolating —
extra machinery for no benefit. Cleaner: derive the query's fine-resolution pooling mask
**directly** from the same source `qry_occ` itself comes from, at native/near-native
resolution, before any `R³` discretization:
- `query_prior` given (cascade case): `F.interpolate(query_prior, size=(S,S,S))`.
- `query_prior` is `None` (fallback, same case `qry_occ`'s support-mean branch covers): average
  the `K` support masks (`context_out`, each resampled to `S` via `F.interpolate`) → one
  `(B,1,S,S,S)` volume.

This mirrors the two existing `qry_occ` branches in spirit (same fallback logic) without
routing through the `R³`-tiled intermediate.

## Mechanism

### 1. Fine-feature extraction (`forward`)

```python
if self.fine_decode:
    if self.pool_token:
        rows = torch.arange(B * T, device=x.device)        # every volume (support + query)
    else:
        rows = torch.arange(B, device=x.device) * T + K    # query only (today's behavior,
                                                             # unchanged when pool_token=False)
    feat_map, fine = self._encode(x, spacing, fine_rows=rows)
```

`fine[st]` is now `(B*T, Cf, S,S,S)` per requested stage when `pool_token=True` (vs. `(B,
Cf, S,S,S)` today). `_decode`'s fine-filter/conv paths expect query-only maps, so after pooling
(`_pool_tokens` below), re-slice `fine` down to the query rows before it reaches `_decode`:
`qidx = torch.arange(B, device=x.device) * T + K`; `fine = tuple(f[qidx] for f in fine)`.

All four fine-capable encoders (`ConvEncoder3D`, `NnUNetTSEncoder`, `ResEncTSEncoder`,
`PlainConvTSEncoder`) share the identical `forward(x, spacing=None, fine_rows=None,
fine_stage=None) -> (coarse, tuple_of_fine_maps)` contract and the same
`feats[st].index_select(0, fine_rows)` mechanism — this is a single change at the `forward()`
call site in `patchset3d.py`, no per-encoder changes.

### 2. Masked pooling + projection (`_pool_tokens`, new method)

```python
def _pool_tokens(self, fine_finest, context_out, query_prior, B, K, T):
    """fine_finest: (B*T, Cf, S, S, S) -- ALL volumes' finest-stage map (forward's `fine`
    indexed at self._pool_stage). Returns (B, K+1, Cf) raw masked-average feature vectors,
    support-major (index 0..K-1) then query (index K) -- NOT yet projected to e (that happens
    in _attn, mirroring cascade_regs's cascade_proj)."""
    S = fine_finest.shape[-1]
    feat = fine_finest.reshape(B, T, -1, S, S, S)
    sup_feat, qry_feat = feat[:, :K], feat[:, K:K+1]            # (B,K,Cf,S,S,S),(B,1,Cf,S,S,S)

    sup_mask = F.interpolate(context_out.float(), size=(S, S, S), mode="trilinear",
                             align_corners=False)                # (B,K,S,S,S) -- wait, needs
                                                                   # per-K interpolate; see impl
                                                                   # note below
    if query_prior is not None:
        qry_mask = F.interpolate(query_prior.float(), size=(S, S, S), mode="trilinear",
                                 align_corners=False)             # (B,1,S,S,S)
    else:
        qry_mask = sup_mask.mean(dim=1, keepdim=True)             # (B,1,S,S,S)

    def masked_avg(f, m):                                         # f:(B,n,Cf,S,S,S) m:(B,n,S,S,S)
        w = m.unsqueeze(2)                                        # (B,n,1,S,S,S)
        num = (f * w).sum(dim=(-3, -2, -1))
        den = w.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
        return num / den                                          # (B,n,Cf)

    return torch.cat([masked_avg(sup_feat, sup_mask), masked_avg(qry_feat, qry_mask)], dim=1)
```

Implementation note: `context_out` is `(B,K,D,H,W)`; `F.interpolate` needs a `(N,C,...)` layout,
so reshape to `(B*K,1,D,H,W)` before interpolating and back to `(B,K,S,S,S)` after — the sketch
above elides this reshape for readability; the implementation must include it.

### 3. Insertion (`_attn`)

New optional `pool_feat` parameter (the `(B,K+1,Cf)` output of `_pool_tokens`, threaded through
`forward`, mirroring how `cascade_regs` is threaded through):

```python
def _attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None, query_prior=None,
         cascade_regs=None, pool_feat=None):
    ...
    n_extra = 0
    if cascade_regs is not None:
        ...                                    # existing cascade mem block, unchanged
    if pool_feat is not None:
        assert self.pool_token, "pool_feat given but arch.pool_token=False on this model"
        pool = self.pool_proj(pool_feat) + self.pool_type          # (B,K+1,e)
        if self.context_id_embed:
            ctx_tag = torch.cat([
                self.ctx_id(torch.arange(K, device=pool.device)).unsqueeze(0).expand(B, -1, -1),
                self.qry_id.view(1, 1, -1).expand(B, 1, -1)], dim=1)   # (B,K+1,e)
            pool = pool + ctx_tag
        if self.mask_slots >= 2:
            gt_tag = self._slot_pos_vec(self._mask_content_index["gt"], pool.device, pool.dtype)
            pred_tag = self._slot_pos_vec(self._mask_content_index["pred"], pool.device, pool.dtype)
            pool = pool + torch.cat([gt_tag.expand(K, -1), pred_tag.unsqueeze(0)], dim=0).unsqueeze(0)
        pool = pool.unsqueeze(2).expand(-1, -1, x.shape[2], -1)     # (B,K+1,c,e) -- broadcast
                                                                     # to both columns, same
                                                                     # pattern as ThinkingRows
        x = torch.cat([pool, x], dim=1)
        n_extra += pool.shape[1]
        sep += pool.shape[1]
    x, sep_t = self.thinking(x, sep)
    ...
```

Placed in the same `n_extra`-accumulating position as the existing `cascade_regs` block (order
relative to `mem` doesn't matter functionally — both are non-spatial prefix rows getting the
same `(0,0,0)` RoPE treatment via `_rope`'s existing `n_extra` parameter, which already sums
however many such rows precede `thinking` — no change needed to `_rope` itself beyond it
continuing to receive the correct total `n_extra`). Stage C's (`seq_compress`) `kv_expand = x[:,
self.thinking.n + n_extra:]` slice already uses `n_extra` generically, so pool rows are
automatically excluded from Stage C's kv the same way cascade-memory rows already are — no
`seq_compress`-side change needed.

### 4. New params (`__init__`, after the existing `fine_decode`/`decoder` setup block)

```python
pool_token: bool = False,
...
self.pool_token = bool(pool_token)
assert not (self.pool_token and not self.fine_decode), (
    "arch.pool_token=True requires arch.fine_decode=True (needs unpooled per-stage maps)")
assert not (self.pool_token and self.register_routed), (
    "arch.pool_token=True adds K+1 extra prefix rows -- arch.register_routed's block-mask "
    "partitioning assumes no prefix rows besides thinking rows (same reason cascade_registers "
    "is incompatible)")
if self.pool_token:
    self._pool_stage = min(self.fine_stage)
    self.pool_proj = nn.Linear(self.encoder.fine_stage_channels(self._pool_stage), e)
    self.pool_type = nn.Parameter(torch.zeros(e))
    nn.init.normal_(self.pool_type, std=0.02)
```

Placed after the existing `fine_decode`/`decoder_kind` validation block (`self.fine_stage` is
always set regardless of `fine_decode`, but `fine_stage_channels`/`n_fine_stages` are only
guaranteed valid once the `fine_decode=True` branch has already range-checked them — so this
block must come after that validation, which the `assert ... not self.fine_decode` above
already guarantees by construction order).

## Config

`configs/experiment/3d/model/patchset3d.yaml` gains one knob:

```yaml
arch:
  pool_token: false            # IRIS-style foreground-masked pooling token per volume (K+1
                               # extra prefix rows, one per support + query). Pools
                               # arch.fine_decode's finest requested stage -- requires
                               # arch.fine_decode=true. Extends fine-map extraction to all K+1
                               # volumes (memory cost scales with K, vs. query-only otherwise).
                               # Incompatible with arch.register_routed (same reason
                               # cascade_registers is).
```

Threaded through `experiments/3d/train.py::build_model`'s `arch` dict via
`a.get("pool_token", False)`, same pattern as every other `arch.*` knob — stored in the
checkpoint's `arch` block, reproduced by `eval.py` automatically.

## Interactions with existing options

- **`arch.fine_decode`** — hard prerequisite, asserted.
- **`register_routed`** — assert-incompatible, same mechanism/reason as `cascade_registers`.
- **`cascade_registers`** — compatible, no change: pool rows and cascade-memory rows are both
  generic `n_extra` prefix rows, independently accumulated and summed before `ThinkingRows`.
- **`seq_compress`** — compatible, no change: Stage C's `kv_expand` slice already skips
  `self.thinking.n + n_extra` rows generically.
- **`transformer_rope`** — compatible: pool rows get `(0,0,0)` (no rotation), the same
  treatment `_rope` already gives thinking/cascade-memory rows via its `n_extra` parameter.
- **`context_id_embed`** — reused: pool rows get the same `ctx_id`/`qry_id` tags real
  support/query content rows get.
- **`mask_slots>=2`** — reused: pool rows get the same gt/pred `slot_pos` tag real
  support/query content rows get (support-derived pool rows tagged "gt", the query-derived one
  tagged "pred").
- **`decoder_kind` (`fine_filter` | `conv`)** — untouched; `_decode` still only ever sees the
  query-only `fine` slices, re-sliced back down after pooling extracts what it needs.

## Backward compatibility

`pool_token=False` (default) allocates none of the new params (`pool_proj`, `pool_type`) and
leaves `forward`'s `rows` selection and `_attn`'s row sequence byte-identical to today — old
checkpoints load and run unchanged.

## Known open questions (tune during implementation, not blocking)

- Whether `pool_proj` should be a plain `Linear` or an MLP (mirrors the existing
  `img_embed`/`img_embed_mlp` choice) — default to plain `Linear`, matching `img_embed`'s
  default.
- Whether the all-background masked-average edge case (`den` clamped to `1e-6`) needs a more
  principled fallback (e.g. falling back to an unweighted spatial mean) — the clamp prevents
  NaN/Inf; whether the resulting near-zero-weighted average is *useful* signal in that case is
  untested.

## Testing

- **No-op guarantee:** `pool_token=False` (default) — a training forward produces logits
  bit-identical (or within fp noise) to the pre-change model; `forward`'s `rows` selection and
  `fine` shape are unchanged; no `pool_proj`/`pool_type` attributes exist.
- **Shape/gradient:** `pool_token=True` — verify `_pool_tokens` output shape `(B,K+1,Cf)`,
  `_attn`'s sequence length grows by exactly `K+1`, `forward`'s output shape is unchanged, and
  gradient reaches `pool_proj`/`pool_type`.
- **Fine-map re-slicing:** with `pool_token=True` AND `fine_decode=True`, confirm `_decode`
  still receives query-only `fine` maps (shape `(B, Cf, S,S,S)` per stage, not `(B*T, ...)`) —
  this is the easiest place for an off-by-one/wrong-slice bug to hide.
- **Incompatibility asserts:** `pool_token=True` with `fine_decode=False` raises;
  `pool_token=True` with `register_routed=True` raises.
- **Edge case:** an all-zero support mask for one context volume doesn't produce NaN/Inf in the
  pooled token or the final logit.
- Keep tests minimal (repo guideline: tests only when necessary) — the above list is the
  minimum needed to trust the no-op guarantee and the new mechanism; not every combination with
  `cascade_registers`/`context_id_embed`/`mask_slots`/`seq_compress` needs its own test given
  each of those paths is already independently tested against the generic `n_extra` mechanism
  by the `cascade_registers`/`seq_compress` test suites — one combined smoke test covering all
  of them together (mirroring `test_seq_compress_works_with_context_id_embed_and_cascade_registers`)
  is enough.

## Logging

Record the change in `docs/logs.md` per repo convention.

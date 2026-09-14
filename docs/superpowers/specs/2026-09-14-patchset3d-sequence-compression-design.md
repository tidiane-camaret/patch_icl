# PatchSet3D sequence compression (IRIS-style) — design

**Date:** 2026-09-14
**Model:** `src/models/patchset3d.py` (`PatchSet3D`), `src/models/pfn_seg_2d.py` (shared dual-axis transformer)
**Status:** approved design, ready for implementation plan
**Reference:** `docs/methods/iris.md` (Iris, CVPR 2025 — arXiv:2503.19359), §4 Task encoding module, §5 Mask decoding module

## Goal

Reduce the attention compute/latency of `PatchSet3D` at its **current** `R` (grid resolution)
and `K` (context size) — not to enable larger `R`/`K`, purely to cut per-step cost. Inspired
by Iris's decoupled task-encoding: compress each volume's raw per-cell tokens down to a small
fixed-size set before the expensive multi-layer transformer, then expand back to per-cell
resolution only at the very end, right before decode.

## Background: where the cost actually lives

`PatchSet3D._attn` builds a row sequence `[thinking | K·N support | N query]` (`N = R³`) and
feeds it through `TransformerEncoderStack` (`l` layers, default 6). Each
`TransformerEncoderLayer` does two attention passes per layer:

- **Column axis** (`c = 2`, img/mask): cheap (`_small_seq_attn`, tiny sequence).
- **Row axis** (`r = thinking + K·N + N`): full/masked SDPA over every cell of every volume.
  This is the actual cost driver — `O(r²)` per layer, paid `l` times.

`ThinkingRows` add a fixed-size register set on top of this sequence but do not remove the raw
per-cell rows from it — additive, not a replacement. `register_routed`'s block-mask already
restricts row-axis attention to per-volume blocks plus a register border, but the row *count*
`r` is unchanged; it only reduces which pairs actually compute scores, not how many rows the
column-axis pass and MLP still process.

## Design: 3-stage pipeline

```
Stage A — Compress (new, arch.compress_layers layers, default 1)
  Per volume (each of the K support + 1 query, weight-shared), m learned
  "compressor" query rows cross-attend, row-axis, block-restricted to that
  volume's own N raw cell rows only. Column axis (img/mask) self-attention
  and MLP run as in the existing block. Support/query never mix here.
    r: K·N + N  →  K·m + m                    cost: O(K·m·N) block-local

Stage B — Heavy transformer (existing TransformerEncoderStack, unmodified)
  Runs over the COMPRESSED rows only (+ thinking rows, + cascade memory
  rows when arch.cascade_registers is set). This is where the saving
  compounds across layers.
    r: K·m + m (+ thinking, + cascade mem)     cost: l · O((K·m+m)²)

Stage C — Expand / read-out (new, arch.compress_layers layers, default 1)
  The query's ORIGINAL (pre-compression) N raw cell rows — kept around,
  never discarded — cross-attend, row-axis, into Stage B's OUTPUT
  compressed rows (now context-aware) to reconstitute a per-cell `q` for
  `_decode`. Every query cell may read every compressed row (cheap: m is
  small).
    r: N query rows × (K·m+m) keys/values      cost: O(N·(K·m+m))
```

Total: `O(K·m·N) + l·O((K·m+m)²) + O(N·(K·m+m))`, vs. today's `l·O((K·N+N)²)`. Since `m ≪ N`
and the multiplier `l` only applies to the now-compressed Stage B, this is the intended win.

Worked example (`R=16 → N=4096`, `K=3`, `m=32`, `l=6`): Stage B sequence shrinks from `16384`
to `4192` rows, and that reduction is realized on every one of the 6 layers — Stage A and C
pay only a one-time linear-in-`N` cost.

## New module: `RowCrossAttention`

`TransformerEncoderLayer.forward` always derives Q, K, V from the *same* input tensor (self
-attention, optionally masked/sliced). Stages A and C need **asymmetric** cross-attention —
Q from one row-set, K/V from a different, differently-sized row-set — which the existing block
cannot express. Add a sibling class in `pfn_seg_2d.py`:

```python
class RowCrossAttention(nn.Module):
    """Row-axis cross-attention: Q from `q_in` (r_q rows), K/V from `kv_in` (r_kv rows).
    Column axis (img/mask) self-attention + MLP applied to the Q-side output afterward,
    mirroring TransformerEncoderLayer's per-row structure. Used by PatchSet3D's compress
    (Stage A: q_in=learned slots, kv_in=raw cells) and expand (Stage C: q_in=raw query
    cells, kv_in=compressed rows) stages — same module, opposite argument order."""
```

Shape contract: `q_in: (B, r_q, c, e)`, `kv_in: (B, r_kv, c, e)` → `(B, r_q, c, e)`. Column
axis (`c=2`) is attended the same way `TransformerEncoderLayer` does today (shared, unmodified
`_small_seq_attn`/`batched_sdpa` helper). Pre-norm + residual + MLP, matching
`LowerPrecisionRMSNorm` conventions used elsewhere in the file.

`PatchSet3D` gets two small stacks of this module (`self.compressor`, `self.expander`, each
`arch.compress_layers` deep), built only when `arch.seq_compress=True`.

### Stage A specifics

- Learned queries: `nn.Parameter(m, e)`, shared weights across every one of the `K+1` volumes
  (mirrors `ThinkingRows.tokens` and Iris's shared `Q`). Being learned parameters, the `m`
  slots are already mutually distinguishable — no separate positional encoding needed.
- Block restriction: one volume at a time (`K+1` independent cross-attention problems,
  batched). Reuse the `register_routed` block-mask pattern conceptually, but simpler — Stage A
  never needs a *dense* fallback mask like `register_routed` does, since Q and K/V are already
  separate tensors per volume (just a batched `RowCrossAttention` call per volume group, no
  mask tensor needed at all).
- Applied to `sup_tok` and `qry_tok` (the `(B, M, 2, e)` tensors `_tokens` already builds) —
  compression sits *after* `_tokens` (mask-content tagging, positional encoding, SimMIM
  masking all unchanged) and *before* the `torch.cat([sup_tok, qry_tok])` that today feeds
  `ThinkingRows`.

### Stage C specifics

- Q = the query's raw (pre-compression) `qry_tok`, kept as a separate reference (not
  overwritten by Stage A's compression).
- K/V = **all** `K·m+m` compressed rows (support *and* query — not just the `K·m` support
  rows), since after Stage B the query's own compressed representation is itself
  context-informed and should feed back into its own per-cell expansion, exactly as Iris's
  Eq. 5 lets `F_q` attend across the full task token set.
- Output feeds `_decode` exactly where `q = x[:, sep_t:, self._decode_col, :]` does today.

## Config

`configs/experiment/3d/model/patchset3d.yaml` (or wherever `arch.*` is set) gains:

```yaml
arch:
  seq_compress: false      # off by default -> byte-identical to today, no new params allocated
  compress_m: 32            # compressed tokens per volume (tune; Iris uses m=10 at a smaller grid)
  compress_layers: 1        # depth of the compress (Stage A) and expand (Stage C) stacks, each
```

Threaded through `experiments/3d/train.py::build_model`'s `arch` dict → `PatchSet3D.__init__`,
same pattern as every other `arch.*` knob. Stored in the checkpoint's `arch` block, so `eval.py`
reproduces it automatically from a trained checkpoint.

## Interactions with existing options

- **`register_routed`** — assert-incompatible with `seq_compress` (same pattern as the existing
  `cascade_registers`/`register_routed` guard at construction time): once Stage A already
  partitions per-volume, Stage B's block-diagonal masking has nothing left to restrict.
- **`ThinkingRows`** — unchanged, operates in Stage B on the compressed sequence. Still useful
  as a global bus layered on top of per-volume compression.
- **`cascade_registers`** — compatible with no change. `cascade_regs` is `(B, thinking_rows, e)`,
  independent of `N`, so it slots into Stage B's (now shorter) sequence exactly as today.
- **`context_id_embed`** — needs a one-line adaptation: today `repeat_interleave(N)` is applied
  per raw cell row before the transformer; with compression it becomes `repeat_interleave(m)`
  applied to Stage A's *output* (after compression, since the embed marks which of the `K+1`
  volumes a compressed row summarizes — identical purpose, smaller tensor).
- **`full_attn`, `query_self_attn`, `transformer_rope`** — apply to Stage B's compressed
  sequence unchanged. RoPE positions for compressed rows get the same `(0,0,0)` no-rotation
  treatment `_rope` already gives thinking/cascade-memory rows — a compressed row isn't tied to
  a single spatial cell anymore.
- **`fine_decode` / `decoder=conv|fine_filter`** — untouched. Both read raw unpooled encoder
  stages directly (`_encode`'s `fine` return), orthogonal to the row sequence Stages A-C
  operate on.
- **`token_mask_ratio_support/query` (SimMIM masking)** — applied in `_tokens`, before Stage A,
  unchanged. A masked cell still contributes its `[MASK]`-replaced embedding into Stage A's
  compression, same as any other cell.
- **`mask_slots` (gt/pred content tags)** — unaffected; tagging happens in `_tokens`, before
  compression.

## Backward compatibility

`seq_compress=False` (default) allocates none of the new modules and leaves `_attn`'s existing
code path byte-identical — old checkpoints load and run unchanged. `seq_compress=True` is a new
architecture variant requiring checkpoints trained with it; no migration path from an existing
checkpoint (new params, no equivalent to warm-start from).

## Known open questions (tune during implementation, not blocking)

- `compress_m` and `compress_layers` defaults are best-guess starting points, not derived —
  expect a small sweep once the mechanism is wired and training-stable.
- Whether Stage A/C should share weights with each other (compress vs. expand as literal
  mirror-image operations) or stay fully independent modules — default to **independent**
  (simpler, matches Iris's asymmetric Eqs. 2-4 vs. Eq. 5 having no shared parameters either).
- Whether the query volume's Stage A compression should use the same learned slot parameters
  as support volumes, or its own separate set — default to **shared** (fewer new params,
  consistent with the mask-content-tag mechanism already distinguishing support/query content
  inside the *same* embedding space rather than via separate weights).

## Testing

- **No-op guarantee:** `seq_compress=False` (default) — a training forward produces logits
  bit-identical (or within fp noise) to the pre-change model.
- **Shape/compression active:** with `seq_compress=True`, verify Stage A output shape
  `(B, K·m+m, 2, e)`, Stage B unchanged internals, Stage C output shape matches today's
  `q = x[:, sep_t:, decode_col, :]` `(B, N, e)`, and `forward` still returns the correct
  `(B, 1, Rd, Rd, Rd)` logit shape.
- **Gradient flow:** compressor/expander learned parameters receive gradients under a backward
  pass.
- Keep tests minimal (repo guideline: tests only when necessary) — focused unit tests on the
  new `RowCrossAttention` module and the no-op guarantee are sufficient.

## Logging

Record the change in `docs/logs.md` per repo convention.

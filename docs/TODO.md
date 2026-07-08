# TODO

## Label injection — notes from TabPFN comparison

### Current mechanism (patch_icl)

Context patch token = `concat_proj(Linear(704→256)(feat), Linear(1→256)(label))` → 256-dim.
Label is merged into the feature token via a learned 512→256 projection.

**Current injection order** (after 2025-05-28 fix): label is fused *before* scale/role embeddings
and RoPE, so all token-level conditioning (physical scale, context-image index, spatial position)
applies to the already-unified (feature+label) token.

### TabPFN approach

In TabPFN, features and labels are **completely separate tokens**:
- Each feature: `Linear(1, emsize)([x_f])` → standalone token
- Label: `Linear(2, emsize)([y, is_nan])` → standalone token
- All `F+1` tokens sit in the feature dimension; integration happens via **feature attention**
  (within-sample attention across the F+1 tokens), not via a merge projection.

### Why patch_icl must merge

patch_icl has no feature-attention axis — only one token per spatial patch. Without a separate
attention axis to integrate them, features and labels must be merged before attention runs.
`concat_proj` is the only integration point.

### Potential experiment

Replace scalar `ctx_labels` with a learned 2-value encoding `[avg_pool_value, is_foreground]`
(analogous to TabPFN's `[y, is_nan]`) fed through `Linear(2, 256, bias=False)` instead of the
current `Linear(1, 256, bias=False)`. Adds an explicit "is any foreground present" signal
alongside the soft avg-pool value.

---

## GPU memory reduction during encoder forward

### CPU-offload stored encoder features
Move `tgt_feats` / `ctx_feats_flat` to CPU immediately after encoding; pull each level
back to GPU inside the per-resolution loop, release it afterwards.
Only the level currently being processed needs to live on GPU simultaneously.

```python
tgt_feats      = [f.cpu() if f is not None else None for f in encode_target(encoder, images)]
ctx_feats_flat = [f.cpu() if f is not None else None for f in encode_context(...)]

# inside the level loop, before extract_features:
tgt_feats_gpu = [f.to(device) if f is not None else None for f in tgt_feats]
ctx_feats_gpu = [f.to(device) if f is not None else None for f in ctx_feats_flat]
tgt_feat_i = extract_features(tgt_feats_gpu, level, res, num_levels)
```

### Chunk context encoding over K
Currently `encode_context` takes `(B*K, …)` in a single call.
For `context_size > 1` this multiplies peak encoding memory by K.
Process one context at a time and cat features:

```python
ctx_feats_flat = None
for k in range(K):
    fk = encode_context(encoder, ctx_imgs_flat[k::K], ctx_masks_flat[k::K])
    if ctx_feats_flat is None:
        ctx_feats_flat = fk
    else:
        ctx_feats_flat = [
            torch.cat([a, b], dim=0) if a is not None else None
            for a, b in zip(ctx_feats_flat, fk)
        ]
```

Peak encoding memory then scales with B only, not B*K.

### torch.compile the encoder
The encoder is frozen and always sees the same input shape, making it safe to use
`reduce-overhead` (CUDA graph capture), unlike the attention module.

```python
encoder = torch.compile(encoder, mode="reduce-overhead")
```

Expected: 10–30% reduction in intermediate tensor allocations via kernel fusion,
plus a speedup on the encoder forward.
Apply after `.to(device).to(torch.bfloat16).eval()` and before the dummy forward.

---

## PatchSetCNN: refinement-pass extensions (two orthogonal ideas)

Both target `src/models/patchset_cnn.py` and are motivated by a **coarse → refinement**
two-pass design: the coarse pass predicts a low-res mask; a second pass re-samples the
volume where the coarse pass is uncertain. They are independent and can land separately.

### Idea 1 — Patch-level sampling maps for context images

**Goal.** Emit a per-patch `sampling` score for *every* patch (context **and** target), so a
refinement pass knows where to draw finer crops. Currently only the `N` target patches get a
prediction (mask); context patches produce nothing.

**Attention change.** Today the sample-axis is masked so the "train set" is read-only:
context/thinking rows attend only to `[thinking + support]`, never to the query
(`query_self_attn=True` only adds the query→query block). To give context rows a
target-aware representation, **drop the mask entirely** → full `r×r` attention (every row,
including thinking + support, attends to every row).

- *Cost:* ≈0 vs `query_self_attn=True` — the score/`AV` matmuls are already dense `r×r`;
  removing the mask drops the `masked_fill` + mask tensor and re-enables the fused
  (flash) SDPA kernel, so it's marginally *cheaper* and lower-memory. Replace the
  `sdpa(q, k[:, :, :sep], v[:, :, :sep])` slice with an **unmasked full-`k,v`** call.
- *Semantics:* this intentionally **breaks the "context is read-only" invariant**. No label
  leak (query rows carry only the support-mean occupancy prior, not GT), but context
  representations now depend on the target.

**Head change.** Decode from **all `(K+1)·N` patch rows** instead of the `N` query rows.
Single shared 2-channel head `(mask, sampling)`; keep the target's `mask` + `sampling`,
keep every patch's `sampling`, **discard the predicted context masks**.

- *Cost:* the decoder is <1% of the network; going from `N` → `(K+1)·N` rows keeps it <1%.
  A single 2-ch head vs two heads differs by <0.5% — prefer the **single head** (fewer
  params, symmetric). Extra `sampling` channel is free.
- *Supervision:* target `mask` as today (GT pooled to `R`); `sampling` against an
  uncertainty/error target (e.g. `|coarse_pred − GT|`), for both target and context patches.

### Idea 2 — High-res outputs via the Medverse "pool QK / hi-res V" trick

**Goal.** Produce an `R_hi = f·R` output map while keeping attention at the coarse `R×R`
grid — cheap weights, high-res values, full-res output. (Ref: Medverse
`MultiContextSpatialCrossAttention3D`, `/home/dpxuser/repos/Medverse`.)

**Key insight (why it's clean).** Medverse pools the axis that becomes V's **channels**, not
the attention **sequence**. Port: keep the `R`-grid tokens for Q/K (the attention map
`A : (N_q × S)` is unchanged), but let each support patch's **value carry its `f×f` sub-cells
folded into channels** (`V` dim `Cv·f²`). Then `A @ V → (N_q, Cv·f²)` and a `pixel_shuffle(f)`
un-shuffles the sub-cells back to space → `(R·f, R·f)`. No A-upsampling, no sequence-length
change, and — because the detail lives in V's channels and is reassembled spatially — it does
**not** collapse to mean-pooled V (the failure mode when pooling on the sequence axis).

**Sketch.** A `HiResReadout` module after the transformer:
`A = softmax(Wq(q_tok) Wk(s_tok)ᵀ)` from coarse post-transformer tokens (dim `e`);
`v_hi` = `[hi-res encoder feats ‖ hi-res mask occupancy]` sampled at `R·f`, `f×f` blocks
folded into channels in `(Cv, f1, f2)` order (to match `pixel_shuffle`); support-major /
patch-row-major to align with `sup_feat` and `ij_base`. Head = `Conv3×3 → GELU → Conv1×1`
→ `out_ch` (e.g. `[mask, sampling]`) at `R·f`.

- *Cost:* only the `A @ V` output width and V memory scale by `f²` (the grid factor);
  `Q Kᵀ`, softmax, and the score-matrix size are **unchanged** (governed by `N`, `S`,
  `dqk`). V has **no projection** (identity fold, Medverse-style).
- *Encoder change:* split `ConvEncoder` into `encode()` (raw multi-scale maps) + a
  resample step, so one encode feeds both the `R` tokens and the `R_hi` value grid.

**Open decisions.** `f = 2` (safe first step, `sub=4`) vs `4` (`sub=16`); V content =
`[feats ‖ mask]` vs mask-occupancy only (pure hi-res label copy, `Cv=1`); fresh read-out
`Wq/Wk` vs reusing the last sample-axis attention map.

### How they compose

Idea 2's read-out naturally emits `(mask, sampling)` at `R_hi`; running it a second time with
`q_tok = s_tok` (support-as-query) reuses `v_hi` to produce per-**context** hi-res sampling
maps — i.e. Idea 1's context outputs at high resolution, for one extra `A @ V`.

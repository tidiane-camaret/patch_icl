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

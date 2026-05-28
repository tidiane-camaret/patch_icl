# TODO

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

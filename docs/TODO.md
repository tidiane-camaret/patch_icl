# TODO

## anchor_synth3d — follow-ups (from blend analysis, 2026-07-22)

Spec/plan: `docs/superpowers/specs/2026-07-22-anchor-synth3d-design.md`,
`docs/superpowers/plans/2026-07-22-anchor-synth3d.md`. Analysis tool:
`experiments/3d/analyze_object_blend.py` (per-object local separability: Cohen's d +
direction-agnostic intensity AUC obj-vs-shell).

**Context.** CT is z-normalized (intensity range ≈ [-1.7, 3.4]), so `contrast_delta`
is in **std units**. Δ sweep at scale_frac=0.4, offset_range=0.6:
`0.05 → med local_auc 0.61 (82% blended)`, `0.15 → 0.70 (52% blended, 15% too-easy)`,
`0.30 → 0.735 (38% blended, 30% too-easy)`. Default 0.15 is a good sweet spot.
Two weaknesses are placement-driven, not contrast-driven:

### Contrast relative to local std
Set `Δ = k · shell_std` instead of a fixed value, so blending is consistent across
regions and the "too-easy" tail (objects in uniform regions) shrinks. Caveat: in air
(std≈0) the object goes invisible — floor the effective Δ or combine with the
in-body constraint below.

### Constrain placement to inside the body
The too-easy tail is mostly objects pushed into **air / outside the body** by
`offset_range`, where any Δ is trivially separable. Reject candidate centers whose
local intensity ≈ air (below a background threshold), or clamp the offset to keep the
object within the body mask, before compositing.

### Minimum object-size floor
`scale_frac · mean(anchor_extent)` yields 1-voxel objects on thin/small anchors
(`obj_voxels` p10 ≈ 1). Add a minimum object size (voxels) so targets are never
degenerate.

---

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

---

## Cascade: keep query_prior gradients attached across levels (ablation, not yet built)

Companion to `arch.cascade_registers` (implemented 2026-09-13, `src/models/patchset3d.py` /
`experiments/3d/cascade.py`; see `project_cascade_register_carry` memory) — that feature
carries level i-1's thinking-row state into level i **with gradient attached** (a deliberate
choice, verified to be a genuine departure from the rest of the cascade: `query_prior` and
the re-crop center are both explicitly severed from gradient today). The natural next ablation
is the mirror-image question: what if `query_prior` itself also carried gradient?

**Current state**: `experiments/3d/cascade.py::_build_query_prior`, `pred` mode —
`src = prev_logit.detach().float()`, then geometrically warped (`_warp_prior_m2` /
`_warp_prior_cropgeom`, `grid_sample`-based, forced fp32 via `torch.autocast(enabled=False)`)
onto level i's crop and fed into level i's query mask token. Removing the `.detach()` would
let level i's loss backprop through the warp, into level i-1's decode head and transformer,
teaching level i-1 to predict masks that make a *better prior for level i*, not just a good
mask on its own.

**Estimated cost (analytical, not yet benchmarked — see 2026-09-13 conversation)**: materially
larger than `cascade_registers`, for three concrete reasons, using exp92's real dims
(`resolution=16, e=768, thinking_rows=8, mask_patch_decode_size=8` → decode grid
`T=128`, `B=2`):
1. **Tensor size crossing the new backward edge**: `prev_logit` at `T=128` is
   `(B,1,128,128,128)` ≈ 4.2M elements vs `cascade_regs`' `(B,8,768)` ≈ 12K elements —
   **~340× more data**.
2. **A genuinely new op sits on the path**: the geometric warp isn't needed by level i-1's
   own loss at all (unlike the decode-head activations `cascade_registers` piggybacks on,
   which are already retained regardless). `grid_sample`'s backward saves its input volume
   *and* sampling grid, both full-res fp32 — a `(B,128,128,128,3)` grid alone is ~50MB.
3. **Bigger fan-out at the next level**: the warped prior feeds `mask_embed` for every one
   of the `N=16³=4096` query tokens (via `qry_occ`), not a fixed 8 register rows.

Rough estimate: tens of MB extra per level-transition (not negligible, unlike
`cascade_registers`) — worth an actual `torch.cuda.max_memory_allocated()` + timing
measurement before committing to it, not just this reasoning. Only the `pred` mode is
affected (`gt_coarse`/`gt_fine` modes use dataloader GT tensors, already non-differentiable,
so detaching them is a no-op either way).

---

## seq_compress: wire RoPE through Stage A/C (currently asserted incompatible)

Spec: `docs/superpowers/specs/2026-09-14-patchset3d-sequence-compression-design.md`. Today
`PatchSet3D.__init__` hard-fails the combination:

```python
assert not (self.seq_compress and self.transformer_rope), (
    "arch.seq_compress=True does not carry RoPE through the compress/expand stages -- "
    "incompatible with arch.transformer_rope=True for now")
```

**Why this is a real correctness gap, not just a missing perf path.** When
`arch.transformer_rope=True`, `self.pos` (the additive Fourier position embedding baked into
every token in `_tokens`) is set to `None` — position is instead carried entirely by RoPE
rotations inside the main transformer's row-axis attention (`_rope`). `RowCrossAttention`
(Stage A/C, `src/models/pfn_seg_2d.py`) has no RoPE support at all — plain cross-attention,
no positional term. So under `transformer_rope=True` + `seq_compress=True`, a raw per-cell
token entering Stage A would carry **zero** position information (no additive PE, no RoPE),
and the compressor wouldn't know which cell it's pooling; Stage C's expand wouldn't know
where in the volume each output belongs either. The `assert` exists to fail loudly instead of
silently shipping position-blind compression/expansion.

**Practical scope today:** `transformer_rope` is unset (default `False`) in
`configs/experiment/3d/model/patchset3d.yaml` and in the `92_multisource_synth` lineage, so
`seq_compress` works fine there. It blocks `seq_compress` on `model/m1.yaml`,
`model/m2_patchset_decoder.yaml`, and experiments 37/40/42/43 (all set `transformer_rope: true`).

**Sketch to lift it.** Wire real RoPE cos/sin into `RowCrossAttention` for the *raw per-cell*
side only:
- Stage A: KV side (the volume's raw `N` cells) gets real `(i,j,k)` positions; the Q side
  (learned `compress_slots`) gets `(0,0,0)` — no rotation, same treatment `_rope` already
  gives `thinking`/cascade-memory rows.
- Stage C: Q side (the query's raw per-cell tokens) gets real positions; the KV side
  (post-transformer compressed rows) gets `(0,0,0)`.

Needs `RowCrossAttention.forward` to accept separate `rope_q`/`rope_kv` cos/sin pairs (Q and
KV have different row-position meanings and counts, unlike `TransformerEncoderLayer`'s
self-attention where q/k share one row indexing) and `apply_rope` calls on each side
independently before the cross-attention SDPA call. Deliberately scoped out of `seq_compress`
v1 to avoid shipping an untested RoPE-through-cross-attention scheme; revisit if a
`transformer_rope=True` recipe wants `seq_compress` too.

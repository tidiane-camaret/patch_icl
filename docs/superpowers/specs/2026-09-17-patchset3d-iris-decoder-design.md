# PatchSet3D — literal Iris task-encoding + mask-decoding modules — design

**Date:** 2026-09-17
**Model:** `src/models/patchset3d.py` (`PatchSet3D`)
**Status:** approved design, ready for implementation plan
**Reference:** `docs/methods/iris.md` §4.2 (Contextual stream, Eq. 3–4), §5 (Mask decoding
module, Eq. 5–6)

## Goal

Add `arch.decoder="iris"`, a literal reproduction of Iris's contextual-stream task encoding
(§4.2) **and** mask-decoding module (§5), as a third `decoder` option alongside today's
`fine_filter` / `conv`. Neither is built by adapting the existing `_build_conv_decoder` /
`_decode_conv` machinery (FiLM fuse, per-skip z-score, static conv head, token-only residual) —
those are patchset3d-specific fusion tricks Iris doesn't use. This is a self-contained pair of
new modules, built directly from the paper's equations.

**Still explicitly out of scope:** Iris's foreground-pooled stream `T_f` (Eq. 2, §4.1) —
`arch.pool_token` already implements that mechanism for other purposes, but this decoder uses
only the contextual stream `T_c` as `T` (decision carried over from the first brainstorming
round: "thinking_rows only, no pool_token" — now realized as "T_c only, no T_f").

**The main cross-context set-transformer (dense full-attention, RoPE, thinking rows) is
untouched** — same "attention pattern stays the same for now" constraint as before. One
consequence worth stating plainly: since `T_c` (below) is computed **independently of the query
and of the main transformer**, and `F_q` is the query's pre-transformer embedding, the
`decoder="iris"` path does not use `q` (the main transformer's query-row output) at all for its
final logits. `_attn`/`self.transformer` still run unchanged (for parity and for the SimMIM
`mask_support`/`mask_query` auxiliary returns), just wastefully for this path — accepted for now.

## Background: what Iris's task encoding + decoder actually do

**§4.2 Contextual stream (Eq. 3–4)** — support-only, computed once, reused across queries:

```
F'_s = PixelShuffle(F_s)                                (C/r³, D, H, W)   # Eq. 3
     = Conv1³( Concat[F'_s, y_s] )                      (C/r³, D, H, W)
F̂_s  = PixelUnshuffle( · )                              (C,   d, h, w)    # Eq. 4  (back to F_s's
                                                                             shape/resolution)
Q (learned params)            (m, C)
flatten F̂_s                   (d·h·w, C)          # keys / values
cross-attention  Q ← F̂_s      (m, C)
self-attention   Q            (m, C)
                              -> T_c : (m, C)
```
`m=10` **[paper]**. Layer count/heads **[assumed]**: "2 blocks, each {cross-attn, self-attn,
MLP}, 8 heads, pre-LN."

**§5 Mask decoding module (Eq. 5–6)**:

```
5.1  F'_q, T' = CrossAttn(F_q, T)     — bidirectional: tokens attend the image, image attends
                                         tokens. F_q is the RAW encoder feature, untouched by
                                         any task interaction before this line.
5.2  mask_features = UNetUp(F'_q)  →  (C_m, D, H, W)   — skip-connected conv up-path, no fusion
                                                          tricks specified or implied ("[assumed]"
                                                          standard query-based head).
     class_embed = Linear(T', C_m)
     logits = einsum('kc,cdhw->kdhw', class_embed, mask_features)   — ONE global filter per
                                                                       class/task, dotted at
                                                                       every voxel.
```

## Sizing the mechanism to this model

- **`F_s` / `F_q`** = the support's / query's pre-transformer image embedding —
  `self.img_embed(sup_feat)` / `self.img_embed(qry_feat)` — computed directly from `_grid_tokens`'
  output in `forward()`, **before** `_attn`/`self.transformer` ever run, and with no additive
  Fourier positional encoding or SimMIM masking applied (those are `_tokens()`-pipeline-specific;
  Iris's `E(x)` carries neither). This is *more* literal than the first design round's
  `qry_img_pre` (which was captured mid-`_tokens()`), and — because `T_c` no longer depends on
  the transformer either — means `_attn` needs **zero changes** for this decoder, a cleaner
  outcome than the earlier "widen the `regs`/`need_think` condition" plan.
- **`r=4`**: `e=768` (the `m2_patchset_decoder` config this is being built for) is divisible by
  `r³=64` cleanly (→12 channels post-shuffle) — same `r=4` Iris's own §7 worked example uses,
  for a different reason (theirs is PixelShuffle-constraint-driven; here it's `e` happening to
  factor nicely). Asserted at construction (`e % r**3 == 0`) rather than hardcoded, so a
  different `e` config still works as long as it divides.
- **Joint KV across K**: `F̂_s` for all `K` support volumes is flattened into **one**
  `(B, K·R³, e)` key/value sequence for `Q`'s cross-attention — matches how the rest of this
  model already treats K-shot context (joint attention), rather than Iris's own
  context-ensemble-averaging (an optional *inference*-time strategy in Iris, not part of Eq. 3–4
  itself, which is inherently single-reference in the paper).
- **`m=10`**, **2 blocks**, **pre-LN** — taken directly from the paper/its own stated defaults.
  Heads reuse the model's own `a` (already asserted to divide `e`, since the main transformer
  needs that too).

## Mechanism

### 1. 3D PixelShuffle / PixelUnshuffle (new module-level helpers)

PyTorch only ships 2D `nn.PixelShuffle`. Two small helpers, exact inverses of each other by
construction:

```python
def _pixel_shuffle_3d(x: torch.Tensor, r: int) -> torch.Tensor:
    """(N,C,D,H,W) -> (N,C/r^3,D*r,H*r,W*r), C % r^3 == 0."""
    N, C, D, H, W = x.shape
    Co = C // (r ** 3)
    x = x.reshape(N, Co, r, r, r, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)          # N,Co,D,r,H,r,W,r
    return x.reshape(N, Co, D * r, H * r, W * r)

def _pixel_unshuffle_3d(x: torch.Tensor, r: int) -> torch.Tensor:
    """(N,C,D,H,W) -> (N,C*r^3,D/r,H/r,W/r) -- inverse of _pixel_shuffle_3d."""
    N, C, D, H, W = x.shape
    Dd, Hh, Ww = D // r, H // r, W // r
    x = x.reshape(N, C, Dd, r, Hh, r, Ww, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)          # N,C,r,r,r,Dd,Hh,Ww
    return x.reshape(N, C * (r ** 3), Dd, Hh, Ww)
```

### 2. Support-only contextual-stream task encoding (`_iris_task_encode`, new method)

Called from `forward()`, independent of `_attn`:

```python
def _iris_task_encode(self, sup_feat, context_out, B, K):
    """Iris §4.2 contextual stream (Eq 3-4), support-only, independent of the query and the main
    transformer. sup_feat: (B,K*N,Cf) raw (pre-img_embed) encoder grid tokens. context_out:
    (B,K,D,H,W) support GT masks. Returns T_c: (B, iris_m, e)."""
    R = self.resolution
    F_s = self.img_embed(sup_feat).reshape(B * K, R, R, R, -1).permute(0, 4, 1, 2, 3)  # (B*K,e,R,R,R)
    r, side = self.iris_r, R * self.iris_r
    y_s = F.interpolate(context_out.reshape(B * K, 1, *context_out.shape[-3:]).float(),
                        size=(side, side, side), mode="trilinear", align_corners=False)
    shuffled = _pixel_shuffle_3d(F_s.contiguous(), r)             # (B*K, e/r^3, side,side,side)
    fused = self.iris_ctx_conv(torch.cat([shuffled, y_s.to(shuffled.dtype)], dim=1))
    Fhat_s = _pixel_unshuffle_3d(fused, r)                        # (B*K, e, R,R,R)
    kv = Fhat_s.flatten(2).transpose(1, 2).reshape(B, K * R ** 3, -1)   # (B,K*N,e), joint across K
    q = self.iris_ctx_query.unsqueeze(0).expand(B, -1, -1)
    for cross, selfattn, mlp, (n1, n2, n3) in zip(self.iris_ctx_cross, self.iris_ctx_self,
                                                  self.iris_ctx_mlp, self.iris_ctx_norms):
        q = q + cross(n1(q), kv, kv)[0]
        q = q + selfattn(n2(q), n2(q), n2(q))[0]
        q = q + mlp(n3(q))
    return q                                                       # T_c, (B, iris_m, e)
```

### 3. `forward()` wiring for `decoder="iris"`

```python
sup_feat, qry_feat = self._grid_tokens(feat_map, B, T, K)
if self.decoder_kind == "iris":
    T_c = self._iris_task_encode(sup_feat, context_out, B, K)
    qry_img_pre = self.img_embed(qry_feat)                        # (B,N,e), pre-transformer
    logit = self._decode_iris(qry_img_pre, T_c, fine)
    q, mask_support, mask_query, regs = self._attn(                # still runs, unused for logit
        sup_feat, qry_feat, self._occupancy(context_out), K, spacing=spacing,
        query_prior=query_prior, cascade_regs=cascade_regs)
else:
    q, mask_support, mask_query, regs = self._attn(...)
    logit = self._decode(q, fine)
return {"final_logit": logit, "mask_support": mask_support, "mask_query": mask_query,
       "registers": regs}
```

`_attn` itself needs **no changes** — no new return value, no widened `regs` condition. This is
a cleaner outcome than the first design round's plan.

### 4. Eq. 5 — dedicated bidirectional cross-attention (unchanged from the first round, `T`'s
   source is now `T_c` instead of the thinking-row output)

```python
def _decode_iris(self, F_q, T, fine):
    """Iris §5 mask decoding module. F_q: (B,N,e) query's pre-transformer image embedding.
    T: (B,iris_m,e) = T_c from _iris_task_encode. fine: query-only unpooled encoder stage maps."""
    t2, _ = self.iris_t2f(T, F_q, F_q)          # tokens attend image
    T2 = T + t2
    f2, _ = self.iris_f2t(F_q, T2, T2)          # image attends updated tokens
    Fq2 = F_q + f2
    ...
```
`iris_t2f`/`iris_f2t` are plain `nn.MultiheadAttention(e, a, batch_first=True)` — one block each
direction (paper states Eq. 5 as a single equation; depth unspecified).

### 5. Eq. 6 — plain skip-connected conv up-path + einsum read-out (unchanged from the first
   round)

```python
    B, N = Fq2.shape[0], Fq2.shape[1]
    R = self.resolution
    x = self.iris_token_proj(Fq2).transpose(1, 2).reshape(B, -1, R, R, R)
    for i, block in enumerate(self.iris_blocks):
        s = self._iris_sides[i]
        x = F.interpolate(x, size=(s, s, s), mode="trilinear", align_corners=False)
        skip = fine[self._iris_stage_order[i]]                  # RAW encoder feature, no projection
        x = block(torch.cat([x, skip], dim=1))                  # plain concat, every level alike
    mask_features = x                                            # (B, C_m, S, S, S), full res

    class_embed = self.iris_class_embed(T2.mean(dim=1))          # pool T_c's iris_m rows -> (B,C_m)
    logit = torch.einsum('bc,bcdhw->bdhw', class_embed, mask_features).unsqueeze(1)
    g = self.grid_size
    if logit.shape[-1] != g:
        logit = F.interpolate(logit, size=(g, g, g), mode="trilinear", align_corners=False)
    return logit
```
Deliberate differences from `_build_conv_decoder`: no 1×1×1 skip projection before concat, no
per-skip `InstanceNorm3d` z-score, no FiLM at the finest level (concat at every level alike), no
final head-conv/token-only residual — `mask_features` *is* the up-path's raw output.
`self.iris_blocks[i]` = two stacked `_ConvNormAct` layers (generic conv-norm-act primitive reuse,
not fusion-logic reuse) — the standard nnU-Net decoder block. `_iris_sides`/`_iris_stage_order`
are generic UNet skip-level geometry, computed exactly like `_build_conv_decoder`'s equivalents.

### 6. New params (`__init__`)

```python
elif self.decoder_kind == "iris":
    self._build_iris_decoder(e, int(image_size[0]), resolution, int(decoder_dim), a,
                             iris_pixelshuffle_r, iris_m, iris_ctx_layers)
```
New constructor args, all with paper-motivated defaults: `iris_pixelshuffle_r: int = 4`,
`iris_m: int = 10`, `iris_ctx_layers: int = 2`.

```python
def _build_iris_decoder(self, e, in_size, resolution, c_d, a, r, m, ctx_layers):
    assert e % (r ** 3) == 0, f"arch.iris_pixelshuffle_r={r} requires e % r^3 == 0 (e={e})"
    self.iris_r = r
    c_shuf = e // (r ** 3)
    self.iris_ctx_conv = nn.Conv3d(c_shuf + 1, c_shuf, 1)              # +1 = concatenated mask
    self.iris_ctx_query = nn.Parameter(torch.empty(m, e))
    nn.init.normal_(self.iris_ctx_query, std=0.02)
    self.iris_ctx_cross = nn.ModuleList([nn.MultiheadAttention(e, a, batch_first=True)
                                         for _ in range(ctx_layers)])
    self.iris_ctx_self = nn.ModuleList([nn.MultiheadAttention(e, a, batch_first=True)
                                        for _ in range(ctx_layers)])
    self.iris_ctx_mlp = nn.ModuleList([nn.Sequential(nn.Linear(e, 4 * e), nn.GELU(),
                                                     nn.Linear(4 * e, e))
                                       for _ in range(ctx_layers)])
    self.iris_ctx_norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(e) for _ in range(3)])
                                         for _ in range(ctx_layers)])

    self.iris_t2f = nn.MultiheadAttention(e, a, batch_first=True)
    self.iris_f2t = nn.MultiheadAttention(e, a, batch_first=True)
    stages = sorted(self.fine_stage, key=lambda st: self.encoder.fine_stage_size(in_size, st))
    self._iris_stage_order = [self.fine_stage.index(st) for st in stages]
    self._iris_sides = [self.encoder.fine_stage_size(in_size, st) for st in stages]
    chans = [self.encoder.fine_stage_channels(st) for st in stages]
    dims = [max(c_d // (2 ** i), 8) for i in range(len(stages))]
    self.iris_token_proj = nn.Linear(e, c_d)
    self.iris_blocks = nn.ModuleList()
    prev = c_d
    for i, st in enumerate(stages):
        self.iris_blocks.append(nn.Sequential(_ConvNormAct(prev + chans[i], dims[i]),
                                              _ConvNormAct(dims[i], dims[i])))
        prev = dims[i]
    self.iris_class_embed = nn.Linear(e, prev)          # C_m = final taper width, no extra knob
```

## Config

`configs/experiment/3d/model/m2_patchset_decoder.yaml`'s `decoder:` comment extends to mention
`iris`; new knobs get their own commented lines (`iris_pixelshuffle_r: 4`, `iris_m: 10`,
`iris_ctx_layers: 2`). New experiment file for a direct A/B against exp92's `conv` decoder:

```yaml
# configs/experiment/3d/experiment/93_iris_decoder.yaml
# @package _global_
# 93_iris_decoder — 92_multisource_synth with the literal Iris task-encoding + decoding modules
# (arch.decoder=iris, docs/methods/iris.md §4.2 + §5) swapped in for the progressive conv decoder.
#
#   python experiments/3d/train.py experiment=93_iris_decoder \
#     train.checkpoint=<a 92-line best.pt> train.resume_weights_only=true

defaults:
  - 92_multisource_synth
  - _self_

arch:
  decoder: iris

wandb:
  name: 93_iris_decoder
```

`train.resume_weights_only=true` required when resuming from a `conv`-decoder checkpoint — the
`iris_*` submodules have no counterpart in a `conv` state dict (and vice versa) — an expected
non-strict load, same as any other decode-head swap in this codebase.

## Interactions with existing options

- **`arch.fine_decode`** — hard prerequisite, same nesting as `conv`/`fine_filter`.
- **`_attn` and everything that lives inside it** (`seq_compress`, `pool_token`,
  `cascade_registers`, `register_routed`, `context_id_embed`, `mask_slots`, `transformer_rope`,
  `query_self_attn`) — **all fully unaffected**, since `_attn` runs completely unchanged for this
  decoder (its output is simply unused for the final logit). This is a stronger isolation
  guarantee than the first design round's plan.
- **`decoder_kind` (`fine_filter` | `conv`)** — untouched; those paths never call
  `_iris_task_encode`/`_decode_iris`.

## Backward compatibility

`decoder != "iris"` (all existing configs) allocates none of the new params and takes the
original `forward()` branch unchanged — existing checkpoints load and run byte-identical.

## Known deviations from the paper (accepted, not blocking)

- **Joint KV across K** support volumes for `Q`'s cross-attention, vs. Iris's own
  single-reference Eq. 3–4 (their multi-reference handling is an inference-time averaging
  strategy, not part of this equation) — motivated by consistency with how the rest of this
  model already handles K-shot context.
- **`F_q`'s feature level**: Iris taps a specific stride-4, `C=128` encoder level (§7) chosen to
  satisfy their own PixelShuffle channel constraint for `F_s`/`F_c` (Eq. 3–4). That constraint
  doesn't carry over to `F_q` here — `F_q` is simply the model's existing coarse `R³` grid image
  embedding.
- **Single cross-attention block** for Eq. 5, and **`K=1` class** throughout (no multi-class
  loop) — both already true of `PatchSet3D` generally, not new simplifications from this decoder.
- **`T'` → class-embedding collapse** (mean-pool over `T_c`'s `iris_m` rows) is `[assumed]` in the
  paper itself, not just here.
- **Main transformer runs but its output is unused** for this decoder's final logit — accepted
  per "attention pattern stays the same for now"; a future iteration could skip `_attn` entirely
  for `decoder="iris"` runs to save compute, out of scope here.

## Testing

- **No-op guarantee:** `decoder != "iris"` — `forward()`'s branch is unchanged, no `iris_*`
  attributes exist, output byte-identical to today.
- **PixelShuffle/Unshuffle round-trip:** `_pixel_unshuffle_3d(_pixel_shuffle_3d(x, r), r) == x`
  for a random tensor, several `(C,D,H,W,r)` combinations where `C % r**3 == 0`.
- **Shape/gradient:** `decoder="iris"` at tiny resolution — verify `_iris_task_encode` output
  shape `(B, iris_m, e)`, final output shape `(B,1,G,G,G)`, finite values, and gradient reaches
  `iris_ctx_conv`/`iris_ctx_query`/`iris_t2f`/`iris_f2t`/`iris_token_proj`/`iris_blocks`/
  `iris_class_embed`.
- **Divisibility assert:** constructing with an `e` not divisible by `iris_pixelshuffle_r**3`
  raises at construction, not at first forward.
- Keep tests minimal (repo guideline: tests only when necessary) — the above is the minimum to
  trust the no-op guarantee and the new modules' basic correctness.

## Logging

Record the change in `docs/logs.md` per repo convention.

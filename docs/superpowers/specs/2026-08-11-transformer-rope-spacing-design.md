# Spacing-aware 3D RoPE for the PatchSet3D transformer

Date: 2026-08-11

## Motivation

Experiment 36 (`36_colipri_spacing_aware_128`) makes the **frozen encoder** spacing-aware:
its RoPE positions are scaled by physical voxel spacing so a fixed anatomical distance maps
to a fixed rotary phase across the 1–4 mm training range (`_set_rope_scaled_grid` in
`src/models/primus_encoder.py`). But the **downstream transformer** is still spacing-blind:
`FourierPositionalEncoding` uses `uv = (ijk+0.5)/resolution` — pure token index
(`src/models/patchset_pfn.py:41`). So at 1 mm vs 4 mm the in-context attention sees identical
positions for the same `(i,j,k)`; physical scale reaches the attention stack only through the
encoder features.

This spec gives the transformer the **same** positional scheme as the encoder: 3D axial RoPE
whose positions scale with physical spacing.

## Decisions (settled during brainstorming)

- **RoPE-only.** When enabled, the additive Fourier PE (`self.pos`) is dropped, mirroring the
  encoder (eva runs `use_abs_pos_embed=False`, RoPE-only). Position enters only via the
  sample-axis `q·k` rotation.
- **Spacing units = `index × spacing/train_mm`** with `train_mm = 2` (the encoder's pretrain
  pitch). At 2 mm → identity integer positions `0..15` (so `theta=100` stays valid and it is
  bit-identical to a no-spacing RoPE baseline); 1 mm → `0..7.5` (interpolation); 4 mm → `0..30`
  (extrapolation). Exactly the encoder's scheme.
- **Opt-in / backward-compatible.** The shared transformer (`pfn_seg_2d.py`, used by
  `patchset3d`, `patchset_pfn`, `patchset_cnn`, 2D `ImagePFN`) gains an optional `rope=None`
  argument; `None` preserves current behavior everywhere.

## Design

### 1. RoPE builder — `src/rope.py`

Add, reusing the existing `_axis_splits` / `_rotate_half` / `apply_rope`:

```python
def build_3d_rope_freqs_from_positions(head_dim, positions, theta=100.0):
    """positions: (R, 3) float — arbitrary per-token (i,j,k), already spacing-scaled.
    Returns cos, sin each (R, head_dim)."""
```

Per axis `dim` from `_axis_splits(head_dim)`:
`inv_freq = 1/theta**(arange(0,dim,2)/head_dim)`, `freqs = outer(positions[:,ax], inv_freq)`,
`cat([freqs,freqs], -1)` → `cos/sin`; concat the 3 axis chunks → `(R, head_dim)`. Same math as
`build_3d_rope_freqs`, but explicit float positions (spacing scaling + thinking-row zeros work
directly). For `e=768, a=12`, `head_dim=64` → axis splits `[22,22,20]` (all even).

Sanity: with integer grid positions it must equal `build_3d_rope_freqs` for the same grid.

### 2. Opt-in RoPE in the shared transformer — `src/models/pfn_seg_2d.py`

- `TransformerEncoderStack.forward(x, sep, attn_mask=None, full_attn=False, rope=None)` — pass
  `rope` to each block.
- `TransformerEncoderLayer.forward(src, sep, attn_mask=None, full_attn=False, rope=None)`:
  applied **only in the sample-axis** (row) attention. After `qkv_row`, rotate `q` and `k`
  (not `v`) on the **full** row sequence, *then* apply the existing `[:sep]` slice / `attn_mask`.
  `rope=(cos,sin)` is reshaped `(1,1,r,d)` and broadcasts over the `b·c` batch and `a` heads.
  The feature-axis attention (2 img/mask cols, no spatial meaning) is left unrotated.

`rope=None` ⇒ byte-identical to today ⇒ 2D models and the conv/pfn variants are untouched.

### 3. PatchSet3D wiring — `src/models/patchset3d.py`

- Constructor: add `transformer_rope: bool = False`, `rope_theta: float = 100.0`. Store
  `self.rope_train_mm = float(getattr(self.encoder, "train_spacing_mm", 2.0))`. When
  `transformer_rope=True`, **do not build `self.pos`** and skip the additive term in `_tokens`.
- `forward(...)` threads `spacing` into `_attn`.
- `_attn(sup_feat, qry_feat, sup_occ, K, spacing=None)`: build positions
  `(thinking_rows + K·N + N, 3)` = `cat([zeros(thinking,3), ijk_base.repeat(K,1), ijk_base])`,
  matching the row order after `self.thinking(...)` (thinking prepended, then support, then
  query). Scale `positions * (spacing / rope_train_mm)` when `spacing is not None`, else identity
  (the train/eval loops only pass `spacing` in the spacing-aware regime, so this decouples the
  transformer's scaling from the encoder while giving identical behavior in practice — and lets
  it be tested with the lightweight conv encoder). Build `cos,sin` with the new helper (`head_dim = e//a`,
  `theta=rope_theta`) and pass `rope=(cos,sin)` to `self.transformer`. Thinking rows at
  `(0,0,0)` → no rotation.

### 4. Config plumbing

- `experiments/3d/train.py` `build_model`: add `"transformer_rope": a.get("transformer_rope",
  False)` and `"rope_theta": a.get("rope_theta", 100.0)` to the arch dict.
- New experiment `configs/experiment/3d/experiment/37_colipri_transformer_rope_128.yaml`
  extending exp 36 with `arch.transformer_rope: true` (inherits `encoder_spacing_aware: true`
  and `train_spacing_range: [1,4]`, so encoder and transformer are spacing-aware together).

## Compile & correctness notes

- `cos/sin` are runtime tensor inputs to the already-`dynamic=True`-compiled transformer → no
  recompile (they are values, not shapes; `r` varying with K is already dynamic).
- Rebuilt each forward (a few small outer products; negligible vs attention).
- Eval reloads `arch` from the checkpoint, so old checkpoints rebuild with
  `transformer_rope=False` and are unaffected.

## Testing

One focused unit test (`tests/` or a `test_*` next to the model):
1. `build_3d_rope_freqs_from_positions` on an integer grid equals `build_3d_rope_freqs`.
2. A `PatchSet3D` forward with `transformer_rope=True` returns finite logits of the right shape
   and differs from the `transformer_rope=False` forward (RoPE actually changes the output).

## Out of scope

- Applying RoPE to `patchset_cnn` / `patchset_pfn` / 2D `ImagePFN` (they keep additive PE).
- Making the additive Fourier PE itself spacing-aware (rejected in favor of RoPE-only).
- Feature-axis RoPE.

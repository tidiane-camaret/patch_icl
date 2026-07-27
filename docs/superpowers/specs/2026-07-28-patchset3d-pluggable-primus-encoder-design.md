# PatchSet3D: Pluggable (frozen) CoLiPri/Primus Encoder — Design

**Date:** 2026-07-28
**Status:** design (approved for spec)
**Env:** `.venv_thor` (thor, RTX A6000 sm_86, torch 2.5.1+cu121)

## Goal

Train the PatchSet3D in-context segmentation model with the **CoLiPri vision backbone**
(a frozen, pretrained nnUNet Primus-M) as its image encoder, instead of the from-scratch
`ConvEncoder3D`. This is the training follow-up to the frozen feature-similarity study, which
found CoLiPri's frozen features competitive with (even ahead of) the task-trained encoder on
intrinsic matching. Question this answers: does a pretrained encoder lift end-to-end in-context
Dice?

Non-goals: no new model family; no changes to the transformer / decoder / mask path; no
distillation; no multi-scale ViT features (single native grid only). YAGNI.

## Why this is clean

PatchSet3D's encoder is **image-only (1-channel)** — the context mask never passes through it.
The encoder maps `(B·T, 1, D, H, W) → (B·T, out_ch, R, R, R)`; masks are embedded separately
via `mask_embed` and enter the transformer as their own tokens (`src/models/patchset3d.py`
`_pack_tokens`, lines ~121-165). So CoLiPri (image-only) drops straight into the encoder slot
with **no mask-channel problem**, and everything downstream is untouched. `img_embed =
Linear(encoder.out_ch, e)` already reads `out_ch`, so a 864-dim encoder wires up automatically.

## Architecture

Make the encoder pluggable via `arch.encoder` (default `conv` → existing behavior unchanged):

```
arch.encoder ─┬─ "conv"   → ConvEncoder3D(1, enc_dims, resolution)         (unchanged)
              └─ "primus" → PrimusEncoder(sidecar, resolution, frozen)     (new)
```

`PrimusEncoder` satisfies the exact `ConvEncoder3D` contract: `forward(x) -> (B, out_ch, R,R,R)`,
attributes `.out_ch` and `.resolution`. Everything else in PatchSet3D is unchanged.

## Components

### 1. `PrimusEncoder(nn.Module)`  (new, in `src/models/patchset3d.py`)
- **Construction:** `PrimusEncoder(sidecar_path, resolution, frozen=True, device=...)`. Reads the
  CoLiPri extraction sidecar (`results/checkpoints/primus_colipri.json`): `primus_kwargs`,
  `weights`, `preproc`. Builds `dynamic_network_architectures.Primus(**primus_kwargs)`, loads the
  backbone `state_dict` (strict=False; the unused `up_projection` decoder is absent, as in the
  feature_sim adapter). `frozen=True` → `requires_grad_(False)`.
- **`out_ch`** = `primus_kwargs["embed_dim"]` (864).
- **`resolution`** = R (the PatchSet3D token grid).
- **`forward(x)`** `(B,1,D,H,W) -> (B,864,R,R,R)`: resample to `input_shape` (192³) if needed,
  reconstruct HU + apply CoLiPri norm (`preproc`), run the Primus ViT encoder
  (`down_projection` + `eva`, no decoder) → `(B,864,24,24,24)`, then `_down_to(R)`. Under
  `torch.no_grad()` when frozen (no ViT activation graph → cheap), plain forward when trainable.
  bf16 autocast, matching the training autocast.
- This reuses the exact encode + preprocess logic already validated in
  `experiments/3d/feature_sim/adapters.py::PrimusEncoderAdapter` (share a helper rather than
  duplicate: factor the `_preprocess` + `_encode` core so both call it, OR keep a single
  small implementation in `src/models/` that the adapter imports — decided at plan time).

### 2. `PatchSet3D.__init__`  (one branch)
```python
enc = arch.get("encoder", "conv")
if enc == "primus":
    self.encoder = PrimusEncoder(arch.primus_sidecar, resolution,
                                 frozen=arch.get("encoder_frozen", True), device=...)
else:
    self.encoder = ConvEncoder3D(1, tuple(enc_dims), resolution)
```
`img_embed = Linear(self.encoder.out_ch, e)` is unchanged (reads 864 automatically).

### 3. `build_model` / config
- `build_model` (`experiments/3d/train.py`) passes the new arch fields into the PatchSet3D
  `arch` dict: `encoder`, `encoder_frozen`, `primus_sidecar`.
- New `configs/experiment/3d/model/patchset3d_colipri.yaml` (or CLI overrides): `encoder: primus`,
  `encoder_frozen: true`, `primus_sidecar: results/checkpoints/primus_colipri.json`,
  `resolution: 24` (map CoLiPri's ViT tokens 1:1 to PatchSet3D cells).

### 4. Optimizer / trainable-param wiring
No change needed: the Muon/AdamW split and the trainable-param count already filter on
`p.requires_grad`, so a frozen encoder is excluded automatically and the head trains normally.

## Data flow (frozen)
```
loader (B,1,192³ @2mm)
  → PrimusEncoder [no_grad: resample→192³, HU-recon→CoLiPri norm, Primus ViT → 864×24³, _down_to R]
  → img_embed(864→e) (+ separate mask tokens) → transformer → decoder → logits (B,1,R³)
  → loss/backward into the HEAD only (CoLiPri fixed)
```

## Run configuration
- Run with `data.image_size=[192,192,192] data.use_crop=true data.crop_spacing_mm=2.0` so the
  loader delivers CoLiPri-native input and the encoder does not double-resample. (It still works
  at other sizes — the encoder resamples internally — just less in-distribution.)
- Cost: frozen training still pays the CoLiPri forward each step (~3.7 vol/s × (K+1) volumes/task,
  no backward through the ViT). At R=24 the transformer is 3.4× the R=16 cost.

## Error handling / edge cases
- **Missing sidecar / weights** → clear error at `PrimusEncoder` construction (mirror the
  adapter's messaging).
- **`arch.encoder=="primus"` without `primus_sidecar`** → explicit config error.
- **Trainable mode** (`encoder_frozen=false`) → forward runs with grad; VRAM/step-time rise
  (backward through the 300M ViT at 192³) — supported, not default.
- **Regression safety:** `arch.encoder` defaults to `conv`, so every existing patchset3d run is
  byte-for-byte unchanged. Verified by a short before/after run.

## Testing
- **Unit:** `PrimusEncoder` on a random-init Primus (no CoLiPri needed) returns `(B,864,R,R,R)`;
  `out_ch==864`; frozen → encoder params have `requires_grad=False` and produce no grad after a
  backward on a dummy head loss.
- **Integration:** `train.py model=patchset3d arch.encoder=primus …` runs a few steps; loss
  decreases; only head params receive gradients; overfit-one-batch sanity (loss → ~0).
- **Regression:** `arch.encoder=conv` (default) matches a pre-change short run.

## Open implementation details (resolved during the plan, not blockers)
1. Whether to factor the shared preprocess/encode core out of `PrimusEncoderAdapter` into a
   single `src/models/` helper, or have `PrimusEncoder` own it and the adapter import it.
2. Exact device handling when the frozen encoder runs under `no_grad` inside the training
   autocast (dtype of the downsampled features handed to `img_embed`).

# PatchSet3D Pluggable Frozen CoLiPri/Primus Encoder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let PatchSet3D use a frozen (or trainable) nnUNet Primus ViT — the CoLiPri backbone — as its image encoder, selected by config, with the transformer/decoder/mask path unchanged.

**Architecture:** PatchSet3D's encoder is image-only (masks are embedded separately), so an image-only ViT drops into the encoder slot. Add a `PrimusEncoder` that satisfies the `ConvEncoder3D` contract (`forward(B,1,D,H,W)->(B,out_ch,R,R,R)`, `.out_ch`, `.resolution`) and select it via `arch.encoder`. `img_embed = Linear(encoder.out_ch, e)` auto-wires the 864-dim output.

**Tech Stack:** PyTorch, `dynamic_network_architectures.Primus`, Hydra configs, `.venv_thor` (torch 2.5.1+cu121, A6000).

## Global Constraints

- Run all Python with `.venv_thor/bin/python` (this node's env).
- Default `arch.encoder="conv"` MUST leave every existing patchset3d run byte-for-byte unchanged (regression safety).
- CoLiPri sidecar lives at `results/checkpoints/primus_colipri.json` (`primus_kwargs`, `weights`, `preproc`), produced by `scripts/extract_colipri_backbone.py`. embed_dim=864, patch 8, input_shape 192³.
- Reuse `_down_to` from `src/models/patchset3d.py` and `CT_MEAN/CT_STD` from `src/totalseg_dataset.py`.
- Frozen encoder: params `requires_grad_(False)` and forward under `torch.no_grad()` (no ViT graph). The outer training loop already wraps the forward in bf16 autocast, so the encoder needs no autocast of its own.
- No pytest suite in this repo; each task's "test" is a small runnable smoke script executed with `.venv_thor/bin/python`, asserting behavior.

---

### Task 1: `PrimusEncoder` module

**Files:**
- Create: `src/models/primus_encoder.py`
- Test: `/tmp/test_primus_encoder.py`

**Interfaces:**
- Consumes: `dynamic_network_architectures.architectures.primus.Primus`; `_down_to` from `src.models.patchset3d`; `CT_MEAN,CT_STD` from `src.totalseg_dataset`.
- Produces: `PrimusEncoder(sidecar_path, resolution, frozen=True, device="cuda")` — an `nn.Module` with `.out_ch:int`, `.resolution:int`, and `forward(x:(B,1,D,H,W))->(B,out_ch,resolution,resolution,resolution)`.

- [ ] **Step 1: Write the module**

Create `src/models/primus_encoder.py`:

```python
"""Frozen (or trainable) nnUNet Primus ViT as a PatchSet3D image encoder.

PatchSet3D embeds context masks separately, so its encoder only ever sees the
image (1 channel). This wraps the Primus ViT encoder (down_projection + eva, no
segmentation decoder) to the same contract as ConvEncoder3D:
    forward(B,1,D,H,W) -> (B, out_ch, R, R, R), with .out_ch and .resolution.
Weights + arch + HU preprocessing come from the CoLiPri extraction sidecar.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# _down_to is defined at module top of patchset3d (before the class), so this import
# resolves even though patchset3d imports PrimusEncoder lazily inside its __init__.
from src.models.patchset3d import _down_to
from src.totalseg_dataset import CT_MEAN, CT_STD


class PrimusEncoder(nn.Module):
    def __init__(self, sidecar_path, resolution, frozen=True, device="cuda"):
        super().__init__()
        from dynamic_network_architectures.architectures.primus import Primus
        meta = json.load(open(sidecar_path))
        kw = dict(meta["primus_kwargs"])
        self.input_shape = tuple(kw["input_shape"])
        self.preproc = meta.get("preproc")
        self.resolution = int(resolution)
        self.out_ch = int(kw["embed_dim"])
        self.frozen = bool(frozen)
        self.primus = Primus(**kw)
        weights = meta.get("weights")
        if weights:
            sd = torch.load(weights, map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            missing, unexpected = self.primus.load_state_dict(sd, strict=False)
            print(f"[PrimusEncoder] loaded weights: {len(missing)} missing "
                  f"(up_projection decoder, unused), {len(unexpected)} unexpected")
        if self.frozen:
            for p in self.primus.parameters():
                p.requires_grad_(False)
        self.primus.to(device)

    def _preprocess(self, x):
        """(B,1,D,H,W) loader z-scored HU -> resampled to input_shape, encoder-normalised."""
        v = x.float()
        if self.preproc is not None:
            hu = v * CT_STD + CT_MEAN
            hu = hu.clamp(self.preproc["clip_min"], self.preproc["clip_max"])
            v = (hu - self.preproc["mean"]) / self.preproc["std"]
        if tuple(v.shape[-3:]) != self.input_shape:
            v = F.interpolate(v, size=self.input_shape, mode="trilinear", align_corners=False)
        return v

    def _encode(self, x):
        """Primus ViT encoder only (down_projection + eva) -> (B, out_ch, g, g, g)."""
        p = self.primus
        x = p.down_projection(x)
        B, C, W, H, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        if p.register_tokens is not None:
            x = torch.cat([p.register_tokens.expand(B, -1, -1), x], dim=1)
        x, keep = p.eva(x)
        assert keep is None, "patch dropping must be off for dense features"
        if p.register_tokens is not None:
            x = x[:, p.register_tokens.shape[1]:]
        return x.transpose(1, 2).reshape(B, self.out_ch, W, H, D)

    def forward(self, x):
        dev = next(self.primus.parameters()).device
        v = self._preprocess(x.to(dev))
        if self.frozen:
            with torch.no_grad():
                f = self._encode(v)
        else:
            f = self._encode(v)
        return _down_to(f.float(), self.resolution)
```

- [ ] **Step 2: Write the smoke test**

Create `/tmp/test_primus_encoder.py`:

```python
import sys; sys.path.insert(0, ".")
import torch
from src.models.primus_encoder import PrimusEncoder

dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = PrimusEncoder("results/checkpoints/primus_colipri.json", resolution=24,
                    frozen=True, device=dev)
assert enc.out_ch == 864, enc.out_ch
assert enc.resolution == 24
x = torch.randn(2, 1, 128, 128, 128)          # != input_shape -> exercises resample
f = enc(x)
assert f.shape == (2, 864, 24, 24, 24), f.shape
assert torch.isfinite(f).all()
# frozen -> no encoder grads
assert all(not p.requires_grad for p in enc.primus.parameters())
# downsample path: resolution 16
enc16 = PrimusEncoder("results/checkpoints/primus_colipri.json", resolution=16,
                      frozen=True, device=dev)
assert enc16(x).shape == (2, 864, 16, 16, 16)
print("OK PrimusEncoder")
```

- [ ] **Step 3: Run the test**

Run: `.venv_thor/bin/python /tmp/test_primus_encoder.py`
Expected: prints `[PrimusEncoder] loaded weights: 10 missing ...` then `OK PrimusEncoder`.

- [ ] **Step 4: Commit**

```bash
git add src/models/primus_encoder.py
git commit -m "feat(patchset3d): PrimusEncoder — frozen Primus ViT as a ConvEncoder3D-compatible encoder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Make PatchSet3D encoder pluggable + wire config

**Files:**
- Modify: `src/models/patchset3d.py` (PatchSet3D.`__init__` signature + encoder construction, ~lines 89-121)
- Modify: `experiments/3d/train.py` (`build_model`, ~lines 179-195)
- Create: `configs/experiment/3d/model/patchset3d_colipri.yaml`
- Test: `/tmp/test_patchset_primus.py`

**Interfaces:**
- Consumes: `PrimusEncoder` from Task 1.
- Produces: `PatchSet3D(..., encoder="conv"|"primus", encoder_frozen=True, primus_sidecar=None)`; `build_model` forwards `arch.encoder`, `arch.encoder_frozen`, `arch.primus_sidecar`.

- [ ] **Step 1: Add encoder params to PatchSet3D.__init__**

In `src/models/patchset3d.py`, add three params to the `__init__` signature (after `image_size=None,`):

```python
        image_size=None,
        encoder: str = "conv",
        encoder_frozen: bool = True,
        primus_sidecar: str = None,
```

- [ ] **Step 2: Branch the encoder construction**

In `src/models/patchset3d.py`, replace the single encoder line:

```python
        self.encoder = ConvEncoder3D(1, tuple(enc_dims), resolution)
```

with:

```python
        if encoder == "primus":
            if not primus_sidecar:
                raise ValueError("encoder='primus' requires arch.primus_sidecar")
            from src.models.primus_encoder import PrimusEncoder   # lazy: avoids import cycle
            self.encoder = PrimusEncoder(primus_sidecar, resolution,
                                         frozen=encoder_frozen, device="cpu")
        elif encoder == "conv":
            self.encoder = ConvEncoder3D(1, tuple(enc_dims), resolution)
        else:
            raise ValueError(f"unknown arch.encoder {encoder!r} (conv | primus)")
```

(device="cpu": the encoder is a submodule, so train.py's `net.to(DEVICE)` moves it with the model.)

- [ ] **Step 3: Thread the fields through build_model**

In `experiments/3d/train.py`, inside `build_model`'s `patchset3d` branch, add three keys to the `arch` dict (after `"image_size": list(cfg.data.image_size),`):

```python
            "image_size": list(cfg.data.image_size),
            "encoder": a.get("encoder", "conv"),
            "encoder_frozen": a.get("encoder_frozen", True),
            "primus_sidecar": a.get("primus_sidecar", None),
```

- [ ] **Step 4: Create the CoLiPri model config**

Create `configs/experiment/3d/model/patchset3d_colipri.yaml` (mirrors patchset3d.yaml; frozen CoLiPri encoder at R=24):

```yaml
# @package _global_
# PatchSet3D with a FROZEN CoLiPri (nnUNet Primus-M) image encoder:
#   python experiments/3d/train.py model=patchset3d_colipri \
#       data.image_size=[192,192,192] data.use_crop=true
# Run at 1.5mm crop (data.crop_spacing_mm=1.5, the default) — the frozen-feature A/B
# found 1.5mm beats 2mm for CoLiPri on organ-centered crops (more voxels-on-target).
model: patchset3d

arch:
  encoder: primus
  encoder_frozen: true
  primus_sidecar: results/checkpoints/primus_colipri.json
  resolution: 24               # map CoLiPri's 24^3 ViT tokens 1:1 to PatchSet3D cells
  enc_dims: [32, 32, 32, 32]   # ignored when encoder=primus (kept for schema parity)
  e: 256
  h: 512
  l: 6
  a: 4
  thinking_rows: 8
  residual_decay: 0.95
  fourier_bands: 8
  mask_patch_size: 8
  mask_patch_decode_size: 8    # 24*8 = 192^3 prediction
  context_id_embed: true
  max_context: 16
  full_attn: true
  query_self_attn: false       # R=24 (13824 tokens): the r×r query self-attn mask is prohibitive
  compile: true

train:
  batch_size: 2
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 0.01
  scheduler: cosine
  warmup_epochs: 1
  loss: bce_dice
  dice_weight: 1.0
  muon: true
  muon_lr_scale: 0.1
  muon_momentum: 0.96
  muon_wd: 0.1
  lawa_k: 10
  checkpoint: null
```

- [ ] **Step 5: Write the build test**

Create `/tmp/test_patchset_primus.py`:

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from src.models.patchset3d import PatchSet3D

m = PatchSet3D(resolution=24, e=256, h=512, l=6, a=4, mask_patch_size=8,
               mask_patch_decode_size=8, context_id_embed=True, full_attn=True,
               query_self_attn=False, image_size=[192, 192, 192],
               encoder="primus", encoder_frozen=True,
               primus_sidecar="results/checkpoints/primus_colipri.json").to("cuda").eval()
# img_embed must be wired to the 864-dim encoder output
assert m.img_embed.in_features == 864, m.img_embed.in_features
# forward: target + K=1 context, 192^3
B, K, S = 1, 1, 192
img = torch.randn(B, 1, S, S, S, device="cuda")
cin = torch.randn(B, K, 1, S, S, S, device="cuda")
cout = (torch.rand(B, K, S, S, S, device="cuda") > 0.7).float()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(img, context_in=cin, context_out=cout, mode="train")
assert out["final_logit"].shape == (B, 1, 192, 192, 192), out["final_logit"].shape
# conv default still builds (regression)
mc = PatchSet3D(resolution=16, image_size=[128, 128, 128])
assert mc.img_embed.in_features == mc.encoder.out_ch
print("OK PatchSet3D primus wiring")
```

- [ ] **Step 6: Run the test**

Run: `.venv_thor/bin/python /tmp/test_patchset_primus.py`
Expected: `OK PatchSet3D primus wiring`.

- [ ] **Step 7: Commit**

```bash
git add src/models/patchset3d.py experiments/3d/train.py configs/experiment/3d/model/patchset3d_colipri.yaml
git commit -m "feat(patchset3d): pluggable encoder (arch.encoder=conv|primus) + CoLiPri config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: End-to-end training smoke + regression

**Files:**
- Test: `/tmp/train_smoke.sh` (ad-hoc; not committed)

**Interfaces:**
- Consumes: the CoLiPri config from Task 2; the real train loader + data.

- [ ] **Step 1: Frozen-encoder training smoke (a few steps on real data)**

Run:
```bash
.venv_thor/bin/python experiments/3d/train.py \
  model=patchset3d_colipri \
  'data.image_size=[192,192,192]' data.use_crop=true \
  data.max_ds_len_train=8 train.batch_size=1 train.workers=4 \
  train.epochs=1 train.eval_every=1000 eval.n_subjects=2 \
  arch.compile=false wandb.project=null
```
Expected: builds, prints `[PrimusEncoder] loaded weights: 10 missing ...`, `Trainable params: <head-only, well under CoLiPri's 300M>M`, the tqdm train bar advances with a finite decreasing loss, run completes without error.

- [ ] **Step 2: Verify only the head trains (frozen encoder gets no grad)**

Create `/tmp/test_frozen_grads.py`:

```python
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from src.models.patchset3d import PatchSet3D

m = PatchSet3D(resolution=24, mask_patch_size=8, mask_patch_decode_size=8,
               full_attn=True, query_self_attn=False, image_size=[192, 192, 192],
               encoder="primus", encoder_frozen=True,
               primus_sidecar="results/checkpoints/primus_colipri.json").to("cuda").train()
B, K, S = 1, 1, 192
img = torch.randn(B, 1, S, S, S, device="cuda")
cin = torch.randn(B, K, 1, S, S, S, device="cuda")
cout = (torch.rand(B, K, S, S, S, device="cuda") > 0.7).float()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(img, context_in=cin, context_out=cout, mode="train")
    loss = out["final_logit"].float().sum()
loss.backward()
enc_grad = any(p.grad is not None for p in m.encoder.parameters())
head_grad = m.img_embed.weight.grad is not None
n_train = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
print(f"encoder got grad: {enc_grad} (want False) | head got grad: {head_grad} (want True) "
      f"| trainable {n_train:.1f}M")
assert not enc_grad and head_grad
print("OK frozen grads")
```

Run: `.venv_thor/bin/python /tmp/test_frozen_grads.py`
Expected: `encoder got grad: False ... | head got grad: True` then `OK frozen grads`.

- [ ] **Step 3: Regression — default conv path unchanged**

Run:
```bash
.venv_thor/bin/python experiments/3d/train.py \
  model=patchset3d data.max_ds_len_train=8 train.batch_size=1 train.workers=4 \
  train.epochs=1 train.eval_every=1000 eval.n_subjects=2 arch.compile=false wandb.project=null
```
Expected: builds a `ConvEncoder3D` encoder and completes exactly as before (no PrimusEncoder message, no error).

- [ ] **Step 4: Log + commit the docs**

Append a `docs/logs.md` entry summarizing the frozen-CoLiPri-encoder training path (config `model=patchset3d_colipri`, run at 1.5mm, frozen head-only training, expected step cost = CoLiPri forward × (K+1) volumes). Then:

```bash
git add docs/logs.md
git commit -m "docs: log patchset3d frozen CoLiPri encoder training path

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** PrimusEncoder (Task 1) ✓; pluggable encoder in PatchSet3D + build_model + config (Task 2) ✓; frozen-default + optimizer auto-exclusion via requires_grad (Task 3 Step 2 verifies) ✓; regression safety of conv default (Task 3 Step 3) ✓; run-config note re 1.5mm (config comment + log) ✓; error handling for missing sidecar / bad encoder name (Task 2 Steps 1-2 raise) ✓.

**Placeholder scan:** none — all code and test bodies are concrete.

**Type consistency:** `PrimusEncoder(sidecar_path, resolution, frozen, device)` with `.out_ch`/`.resolution`/`forward` used identically in Tasks 1-3; PatchSet3D params `encoder`/`encoder_frozen`/`primus_sidecar` match between the signature (Task 2 Step 1), build_model (Step 3), config (Step 4), and tests (Tasks 2-3).

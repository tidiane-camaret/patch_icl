# Iris — Architecture Reproduction Guide

**Paper:** *Show and Segment: Universal Medical Image Segmentation via In-Context Learning* (Gao et al., CVPR 2025 — arXiv:2503.19359)

This document reconstructs the Iris architecture in enough detail to reimplement it. Values stated explicitly in the paper are marked **[paper]**; values the paper leaves open, together with the reasoning behind a reproducible default, are marked **[assumed]**. Treat everything **[assumed]** as a knob to tune, not ground truth.

---

## 1. Core idea

Iris segments 3D medical volumes by conditioning on a reference `(image, mask)` pair rather than on a fixed set of trained classes. Its defining design choice is to **decouple task encoding from query inference**:

- A **task encoding module** compresses one reference pair into a small task embedding `T`.
- A **mask decoding module** consumes `T` plus the query image and predicts the mask.

Because `T` is computed once and reused across every query, cost scales as **O(k + m)** for `k` reference pairs and `m` query images, instead of **O(k·m·n)** for methods (UniverSeg, Tyche) that re-encode the reference set per query and run one pass per class. Iris also segments all `K` classes in a **single forward pass**.

---

## 2. Notation and global config

| Symbol | Meaning | Value |
|---|---|---|
| `x_q`, `x_s` | query / reference (support) image | `(B, 1, D, H, W)` |
| `y_s` | reference mask, binary per class | `{0,1}^(D×H×W)` |
| `D, H, W` | input volume dims | `128 × 128 × 128` **[paper]** |
| `C` | encoder feature channels at the task-encoding level | **[assumed]** (see §7) |
| `r` | downsample ratio of that feature level | **[assumed]** (see §7) |
| `d, h, w` | downsampled feature dims | `D/r, H/r, W/r` |
| `m` | number of learnable query tokens | `10` **[paper]** |
| `K` | number of target classes | task-dependent |

**Backbone [paper]:** 3D UNet with residual connections, 4 downsampling stages, base channel width 32, trained from scratch.

---

## 3. Module flow overview

```
         x_s ──┐                         x_q ──┐
               │  (shared encoder E)            │  (shared encoder E)
               ▼                                ▼
             F_s : (C,d,h,w)                  F_q : (C,d,h,w)
               │                                │
   ┌───────────┴───────────┐                    │
   ▼                        ▼                    │
Foreground stream     Contextual stream          │
 (Eq. 2)               (Eqs. 3–4 + attn)         │
   │  T_f:(1,C)           │  T_c:(m,C)            │
   └──────────┬───────────┘                      │
              ▼                                   │
     T = [T_f ; T_c] : (m+1, C)  per class        │
     stack K classes → (K(m+1), C) ──────────────►│
                                                   ▼
                                    Mask decoding module (Eqs. 5–6)
                                    bidirectional cross-attn + mask head
                                                   ▼
                                    ŷ_q : (K, D, H, W)
```

The encoder is the only heavy component and its weights are **shared** between reference and query paths.

---

## 4. Task encoding module

Input: reference features `F_s : (C, d, h, w)` and the full-resolution binary mask `y_s : (D, H, W)`. Two parallel streams, computed **per class**.

### 4.1 Foreground stream (Eq. 2)

```
T_f = Pool( Upsample(F_s) ⊙ y_s ) ∈ R^(1×C)
```

Step-by-step shapes:

```
Upsample(F_s)   (C,d,h,w) --trilinear ×r--> (C,D,H,W)
  ⊙ y_s         (C,D,H,W) ⊙ (1,D,H,W)  ->  (C,D,H,W)   # zero outside foreground
  Pool          masked average over FG voxels ->  (1,C)
```

**Why upsample before masking:** applying the mask at downsampled resolution can shrink or erase small structures. Restoring full resolution first preserves fine boundaries. This step is what the ablation credits with lifting small-object Dice from **62.13 → 78.92**.

**Pooling** is a foreground-masked average: `sum(features · mask) / sum(mask)` over spatial dims. **[assumed]** — the paper says "Pool"; mean pooling is the natural reading.

### 4.2 Contextual stream (Eqs. 3–4)

Fuses features with the mask at full spatial resolution, memory-efficiently, via sub-pixel (PixelShuffle) tricks.

```
F'_s = PixelShuffle(F_s)                                (C/r³, D, H, W)   # Eq. 3
     = Conv1³( Concat[F'_s, y_s] )                      (C/r³, D, H, W)
F̂_s  = PixelUnshuffle( · )                              (C,   d, h, w)    # Eq. 4
```

Then `m` learnable query tokens attend over the fused map to produce `T_c`:

```
Q (learned params)            (m, C)
flatten F̂_s                   (d·h·w, C)          # keys / values
cross-attention  Q ← F̂_s      (m, C)
self-attention   Q            (m, C)
                              -> T_c : (m, C)
```

**Attention config [assumed]:** number of cross/self-attention layers and heads is unspecified. A workable default: 2 blocks, each `{cross-attn, self-attn, MLP}`, 8 heads, pre-LN. Tune as needed.

### 4.3 Combine and stack

```
T   = [T_f ; T_c]                 (m+1, C)     # per class
T   = [T¹ ; T² ; … ; T^K]         (K(m+1), C)  # multi-class
```

Only this lightweight module repeats per class; the encoder pass is shared. For `K=1`, `T` is `(m+1, C)`.

---

## 5. Mask decoding module

Input: query features `F_q : (C, d, h, w)` and task tokens `T : (K(m+1), C)`. Query-based decoder in the Mask2Former family (paper ref [8]).

### 5.1 Bidirectional cross-attention (Eq. 5)

```
F'_q, T' = CrossAttn(F_q, T)
```

Shapes:

```
flatten F_q            (d·h·w, C)
tokens attend to image, image attends to tokens
F'_q                   (d·h·w, C) -> reshape (C, d, h, w)
T'                     (K(m+1), C)
```

Information flows both directions: task tokens condition the image features and image content refines the task tokens.

### 5.2 Mask prediction (Eq. 6)

```
ŷ_q = D(F'_q, T') ∈ {0,1}^(K×D×H×W)
```

Mechanics **[assumed]** (standard query-based head): the decoder up-path (with UNet skip connections) produces a per-voxel mask-feature map `(C_m, D, H, W)`; each class's token in `T'` is projected to a `C_m` embedding and dotted against the map to yield that class's logits; stack over `K`.

```
mask features (up-path)          (C_m, D, H, W)
per-class token -> embedding      (K, C_m)
einsum('kc,cdhw->kdhw')          -> (K, D, H, W)  logits
```

All `K` classes emerge from one pass — the key contrast with per-class methods.

---

## 6. Training

**Regime [paper]:** end-to-end, episodic, one-shot.

Per episode (Algorithm 1 in the paper):
1. Sample a dataset index `k`.
2. Sample a query pair `(x_q, y_q)` and a reference pair `(x_s, y_s)` from the **same** dataset.
3. If the mask is multi-class, split into binary masks.
4. `T = TaskEnc(E(x_s), y_s)`; `ŷ_q = Dec(E(x_q), T)`.
5. Backprop through E, task encoder, and decoder jointly.

**Loss [paper]:** `L_seg = L_dice + L_ce`.

**Regularization tricks [paper]:**
- Augment both query and reference images.
- Add random perturbation to query images to simulate imperfect references.
- Randomly drop classes in multi-class datasets to force independent class-wise task encoding.

**Optimization [paper]:**

| Setting | Value |
|---|---|
| Optimizer | Lamb |
| Base LR | 2e-3, exponential decay |
| Weight decay | 1e-5 |
| Iterations | 80,000 |
| Warm-up | 2,000 iters |
| Batch size | 32 |
| Patch size | 128³ |

**Preprocessing [paper]:** resample to isotropic `1.5 mm³`; CT clipped to HU `[-990, 500]`; MR/PET clipped at 2nd/98th percentiles; then per-volume z-score normalization.

**Augmentation [paper]:** scaling `0.9–1.1`, rotation `±10°`, translation, random/center crop to `128³`; multiplicative brightness `0.9–1.1`, additive brightness `σ=0.1`, gamma `0.8–1.2`, contrast `0.8–1.2`, Gaussian blur `σ=0.7–1.3`, Gaussian noise `σ<0.02`. For reference images, ensure annotated regions survive augmentation.

---

## 7. The two under-specified dimensions (`C`, `r`)

The paper never states `C` or which feature level feeds the task encoder, but the PixelShuffle in Eq. 3 constrains it. PixelShuffle by factor `r` in 3D requires the channel count divisible by `r³`.

- The bottleneck after 4 downsamplings sits at stride 16 (`8³`). A shuffle factor of 16 needs `C` divisible by `16³ = 4096`, which base-32 channels never reach. **So the task-encoding feature cannot be the bottleneck** — it must be a shallower / decoder-side, higher-resolution feature.

A self-consistent reproduction choice with `r = 4`:

```
F_s              (128, 32, 32, 32)     # C=128, r=4  (stride-4 feature)
PixelShuffle     (2,   128,128,128)    # C/r³ = 128/64 = 2
Concat mask      (3,   128,128,128)
Conv 1³          (2,   128,128,128)
PixelUnshuffle   (128, 32, 32, 32)
Q attn -> T_c    (10, 128)             # m=10
T_f              (1, 128)
T (per class)    (11, 128)
15-class ref     (165, 128)            # AMOS CT
```

Pick `C`/`r` so that (a) `C % r³ == 0`, (b) `d·h·w` is small enough for the token↔feature attention to be affordable, and (c) resolution is high enough to keep small structures. `r=4, C=128` satisfies all three at 128³.

---

## 8. Inference strategies

All four reuse cached task embeddings; only in-context tuning does gradient work at test time.

| Strategy | What it does | Cost |
|---|---|---|
| **One-shot** | Encode `T` from one reference, cache, reuse across all queries. | cheapest per query |
| **Context ensemble** | Encode `T` for several references, average. For **seen** classes, read from an EMA memory bank instead of encoding. | ~same as one-shot at test |
| **Object-level retrieval** | Per-class cosine match between query and pool embeddings; pick best reference **per class** (beats image-level averaging). | + milliseconds (vector compare) |
| **In-context tuning** | Freeze the model; optimize only `T` by gradient descent on the reference loss; cache the tuned `T`. | most expensive |

**EMA memory bank (Eq. 7)** for seen classes:

```
T_k ← α·T_k + (1-α)·T_k^new ,   α = 0.999
```

Maintained during training; at inference on seen classes, index `T_k` directly — no reference pair needed. This lets Iris act as a plain segmentation model on known classes and an in-context learner on novel ones.

**Sliding-window inference [paper]:** for volumes larger than 128³, slide a 128³ window with 50% overlap, average predictions in overlaps, compute metrics on the full reassembled volume.

---

## 9. Minimal reproduction checklist

1. **Encoder** — 3D residual UNet, 4 stages, base 32; expose a stride-4 feature (`C=128`) for the task encoder and the full up-path for the mask head.
2. **Foreground stream** — upsample→mask→masked-mean → `(1,C)`.
3. **Contextual stream** — PixelShuffle→concat mask→Conv1³→PixelUnshuffle; `m=10` learned tokens with cross+self attention → `(m,C)`.
4. **Task token** — concat to `(m+1,C)`; loop classes → `(K(m+1),C)`.
5. **Decoder** — bidirectional cross-attention (Eq. 5) + Mask2Former-style per-class mask head (Eq. 6), single pass for all `K`.
6. **Training** — episodic same-dataset reference/query sampling, binary-decompose multi-class masks, `L_dice+L_ce`, class dropout + query perturbation, Lamb, 80k iters, batch 32.
7. **EMA bank** — update `T_k` per seen class with `α=0.999`.
8. **Inference** — cache embeddings; implement the four strategies; sliding window at 50% overlap.

---

## 10. Known gaps to decide yourself

These are not in the paper and must be fixed in any reimplementation:

- Exact `C` and feature level for the task encoder (see §7 for a consistent choice).
- Attention depth/heads/norm in the contextual stream and the decoder.
- Decoder up-path channel schedule and the mask-head projection dim `C_m`.
- Whether pooling in Eq. 2 is mean vs. sum-normalized (mean assumed).
- Sampling weights across datasets during episodic training (paper samples a dataset index but does not give the distribution).

*Prepared as a reproduction reference. Cross-check equation numbers against arXiv:2503.19359 v1 before implementing.*
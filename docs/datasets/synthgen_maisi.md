# MAISI as an on-the-fly synthetic generator — pipeline & findings

Investigation into using NVIDIA **MAISI / NV-Generate-CTMR** (`rflow-ct`, a 30-step
rectified-flow mask→CT latent diffusion model) as a synthetic-data source for the
in-context 3D CT segmentation project. The framing throughout: we care more about
**covering a large prior distribution** (diverse, even OOD) than photorealism, and we want
to know whether we can **bank latents and cheaply render crops on the fly** during training.

Repo: `/home/dpxuser/repos/NV-Generate-CTMR`. All benches under
`experiments/3d/synth_task_generation/`. Node: **thor / RTX A6000** (Ampere sm_86),
torch 2.5.1+cu121, `.venv_thor`.

---

## 1. The pipeline (mask → CT)

`rflow-ct` renders a CT from a label mask:

```
mask source → binarize_labels (8-ch)
  → ControlNet + DiffusionUNet, 30-step rectified-flow loop (in latent space)
  → VAE SlidingWindowInferer decode (latent → HU image)
  → HU windowing + background/body-envelope crop
```

- **VAE**: 4× spatial downsample, 4 latent channels. A **128³** image ↔ a **(4, 32, 32, 32)**
  latent = **16× compression**. Whole-body 384³ ↔ (4, 96, 96, 128) ≈ 9 MB fp16.
- **Latent channels** = 4; autoencoder `num_channels` [64,128,256] (`config_network_rflow.json`).

### Per-module cost (whole-body 384³, `bench_pipeline_breakdown.py`)

| module | time | share |
|---|---:|---:|
| **VAE decode** | 98.3 s | **77 %** |
| 30-step diffusion loop | 13.7 s | 11 % |
| read/prep mask | 4.4 s | — |
| save | 6 s | — |

**Decode dominates at every scale.** The diffusion loop is comparatively cheap once the
latent is small. This is the single most important number: optimizing MAISI = optimizing
the VAE decoder.

---

## 2. Offline generation into the training dataloader (already wired)

Fast pipelined generator + a standalone in-context dataset that consumes the output. This
is the **pragmatic default in production** — an offline *image* bank, not an online render.

- `gen_maisi_fast.py` — threaded mask-prefetch + threaded QC/save, one compressed `.npz`
  per pair (`ct` f16 z-scored HU, `label` u8 MAISI-132 vocab, `spacing`). **11.1 s/vol**
  (1.42× over 15.8 s baseline). `torch.compile` **rejected** in the real path (MetaTensor
  dynamo guard-thrash → 2× slower); batch>1 **rejected** (per-vol slower, OOM@B4).
- `src/synth_gen_maisi_dataset.py` — `SynthGenMaisiDataset(TotalSegInContextDataset)`:
  native-only, standalone MAISI vocab (no TotalSeg remap), reads the flat `.npz` dir.
  Supports `use_crop`/`crop_spacing_mm` (organ-centred native crop → isotropic
  crop_spacing_mm/voxel). Reuses base context sampling / class-balancing / aug / collate.
- `data/maisi_classes.py` — MAISI vocab vendored from `label_dict.json`.
- Config: `configs/experiment/3d/dataset/synth_gen_maisi.yaml` (`dataset=synth_gen_maisi`;
  `paths.synth_gen_maisi`). **train/val classes must be MAISI names or `all`** — TotalSeg
  names won't match.

---

## 3. Can we bank *latents* and decode crops on the fly? (the core question)

### 3a. Naïve "store latents, decode per item" is NOT viable — decode is the cost

`bench_decode_sdedit.py`: per-item online cost = SDEdit(K re-noise/denoise) + decode ≈
**0.95–1.1 s**, ~85 % of it decode, **un-amortisable**. At batch16 × (1 target + 1 context)
= 32 decodes/step ⇒ **~27 s/step**. Caching latents buys 16× storage + cheap SDEdit
appearance jitter, but **not a cheap render** — the render is the cost whether the latent
is cached or freshly diffused.

### 3b. Full-body latent → crop-at-decode = the realism fix (`inspect_fullbody_crops.py`)

Per-crop conditioning (diffuse each small crop in isolation) gives *coherent but not
realistic* CT — flat soft tissue, graininess (small FOV < MAISI's 256 mm recommendation).
Fix: **run the 30-step loop ONCE on the full-body latent** (14.6 s offline), then crop 32³
latent windows and decode each.

- **Dramatically more realistic**: crisp bone (skull/vertebrae/ribs/pelvis), textured
  soft tissue (liver/kidney gain internal structure), correct placement, good mask
  alignment, 3D-coherent coronal views. **No seam artifacts** from decoding isolated 32³
  windows (the VAE decoder is local; interiors are clean).
- **Mechanism**: diffusion sees the global anatomy; we only crop at decode.

### 3c. Validated architecture

```
OFFLINE: bank of full-body latents (14.6 s each, ~9 MB fp16 @ 4×96×96×128)
ONLINE : random latent-window crop  +  compiled/TRT decode  →  (image, mask) pair
         (many crops per latent = spatial diversity for free;
          appearance diversity via different masks/seeds)
```

Because decode is un-amortisable per item, the practical online form is a **background GPU
worker** that decodes into a **reused ring buffer** which training samples over many
steps — decode paid at *refill rate*, not per item. Feasible: **~12–18 crops/s** (below).

---

## 4. How low can decode go? (the speed ladder)

Goal = cheap decode of small crops for diverse priors (realism optional ⇒ 96³, fp16,
custom kernels all fair). All at **96³** (latent 4×24³):

| decode 96³ | ms | throughput | notes |
|---|---:|---:|---|
| eager (cuDNN fallback) | 344 | 2.9/s | warns "cuDNN cannot be used for large non-batch-splittable conv" |
| **torch.compile** | **80** | **12.5/s** | ~**10×** whole-body-equivalent lever; free, in-env, shape-flexible → **pragmatic default** |
| **TensorRT fp16** | **55** | **18/s** | MAISI's own path; 24-min one-time build, static shape |
| distilled tiny decoder | *sub-10 (est.)* | *>100/s* | needs a training run; realism-optional ⇒ right tradeoff |

- **torch.compile** routes 3D convs to Triton, avoiding the cuDNN slow fallback — this is
  the big lever (`bench_vae_decode_opt.py`). Uses `torch.compile(ReconModel(ae.half(), sf),
  mode="default")` on a pure-fp16 model. **Batching is counterproductive under compile**
  (B≥2 recompiles to a slow dynamic kernel); max-autotune crashes
  (expandable_segments/cudagraph).
- **TensorRT** (`bench_vae_trt.py`): `monai.networks.trt_compile(ae, plan,
  submodule='decoder', args={'precision':'fp16'})`. **Static shape** — 128³ needs its own
  24-min build. **FP8 N/A on Ampere** (no FP8 tensor cores) ⇒ **55 ms is the ceiling for
  the stock decoder** on this hardware. Sub-10 ms requires a **distilled tiny 3D decoder**
  (TAESD-style).
- **TRT env** (additive to `.venv_thor`, reversible): `tensorrt-cu12==10.13.3.9`,
  `polygraphy 0.53.4`, `onnx 1.22.0`, `cuda-python 12.9.7` (replaced cu13 cuda-bindings to
  match cu121 torch; torch verified intact). `trt_compile` **silently no-ops** without
  polygraphy + `cuda.cudart`. The `tensorrt==10.13` meta-package pulls cu13 → install
  `tensorrt-cu12` explicitly.

**Recommendation**: build the latent-bank/ring-buffer on **torch.compile (80 ms)** —
zero build cost, no extra deps, shape-flexible. Swap in **TRT (55 ms)** only when
committing to a fixed 96³ crop size for a long run (1.5× throughput for the 24-min build).

---

## 5. Latent-space properties

### 5a. Geometric ops are coherent + ~equivariant (`inspect_latent_ops.py`)

Spatial ops on a 32³ latent window, decoded: **rot90 / rot45 / flip / avg_pool3d / zoom**
all decode coherently. Equivariance (decode∘op vs op∘decode): **rot90 MAE 38 HU, rot45
57 HU** — small non-equivariance is *desirable* diversity.

⇒ **Geometric aug can run in latent space + decode, no diffusion re-run** — free on-the-fly
aug on a latent bank. pool = cheap multi-scale/blur; zoom = scale aug.
**CAVEAT**: the latent op does **not** carry the mask — apply the *same* transform to the
label in image space to keep (image, mask) aligned.

### 5b. Additive latent noise = geometry-preserving appearance aug (`inspect_latent_perturb.py`)

Encode a real crop + add σ×latent-std Gaussian noise. **Organs stay in place ≤ 1σ; 2σ
breaks.** The noise decodes as coarse ~4-voxel blobs (the 4× decoder upsample). Unlike 5a,
this needs **no mask transform** — geometry is preserved.

### 5c. Real TotalSeg crops survive the VAE round-trip (`encdec_totalseg_crops.py`)

Deterministic encode (`encode().mu` → `decode_stage_2_outputs`, no diffusion/scale_factor)
of real dataloader crops (1.5 mm, aug off): **PSNR 31.8 dB, MAE 32.7 HU** (soft-tissue
31 HU). Error is **edge-localized** (bone/air boundaries); interiors clean.

⇒ **The MAISI latent space is valid for real data, not just synthetic** → real and
synthetic latents are **interchangeable** for in-context learning in latent space.
The ~33 HU softening hits thin/high-contrast structure (cortical bone, small vessels) —
the known thin-structure blind spot.

Intensity bridge (dataloader z-scored ↔ MAISI [0,1]):
`hu = z*505.8 − 167.3`, then `maisi01 = (hu + 1000) / 2000`.

### 5d. Resolution comfort zone (`bench_vae_resolution.py`)

Soft-tissue MAE by crop spacing: **1 mm = 35, 2 mm = 30, 3 mm = 37, 4 mm = 70, 5 mm = 61 HU**.
⇒ **1–3 mm is the VAE comfort zone** (MAISI trained ≤ 3 mm in-plane). At 4–5 mm error
~doubles (128³@5 mm spans a 640 mm whole-body cross-section — tiny structures/voxel + OOD).
Soft-tissue MAE is the honest metric (overall PSNR is dominated by growing air background).

---

## 6. Is the VAE latent a good *task* representation? (No — it's a renderer)

`compare_latent_vs_primus.py`: in-context prototype-matching (cosine nearest-centroid) +
**fg-retrieval@1** over 40 TotalSeg tasks (1.5 mm, K=4). Frozen features classify the target.

| space | fg-retrieval@1 |
|---|---:|
| **Primus** (CoLiPri ViT, 864×16³) | **0.355** |
| rawHU (1×32³) | 0.049 |
| vae32 (4×32³) | 0.022 |

**Primus retrieval is ~16× better than the VAE latent.** And on forward time
(`bench_encoder_fwd.py`, 128³): **MAISI VAE encode 482 ms** (8.8M params) vs **Primus ViT
1.8 ms** (144.9M params) — **Primus ~270× faster despite 16× more params**.

⇒ **Reconstruction fidelity ⟂ task usefulness.** Clean role division:

- **MAISI VAE = generator / renderer** (great decode input, poor task encoder, slow to encode).
- **Primus = the task encoder** (discriminative, fast).

Do **not** try to learn in-context tasks directly in the VAE latent space.

---

## 7. Bottom line

- **Latent-bank + crop-at-decode is empirically validated**: offline full-body latents →
  online random latent-crop + compiled/TRT decode → realistic, correctly-placed
  (image, mask) pairs, real & synthetic latents interchangeable.
- **Decode is the whole cost** (77 % whole-body, un-amortisable per item). torch.compile
  (80 ms/96³) is the free 10× lever; TRT (55 ms) is the stock-decoder floor on Ampere;
  sub-10 ms needs a distilled tiny decoder.
- **Serve it via a background worker → ring buffer** (decode paid at refill rate).
- **Two free latent aug families**: geometric ops (coherent + ~equivariant, needs mask
  co-transform) and additive noise ≤ ~1σ (geometry-preserving, no mask transform).
- **1–3 mm is the resolution comfort zone.**

## Gotchas

- **Colliding `scripts` packages**: `patch_icl` and `NV-Generate-CTMR` both ship a top-level
  `scripts` pkg. Build the dataset first (caches patch_icl `scripts`), then purge
  `scripts*` from `sys.modules` before importing the repo's `scripts.utils_infer`.
- **Hydra searchpath** uses `${oc.env:PWD}/configs` → **run from the patch_icl cwd**, not
  the MAISI repo, or config composition resolves wrong.
- `combine_label_or` / conditioning mask must be a **MONAI MetaTensor** (`binarize` calls
  `.as_tensor()`).
- RFlow `add_noise` device mismatch: `set_timesteps(..., device=dev)` and `t_start.to(dev)`.

## Files

`gen_maisi_fast.py`, `bench_maisi_gen.py`, `visualize_maisi_output.py`,
`bench_decode_sdedit.py`, `bench_pipeline_breakdown.py`, `bench_vae_decode_opt.py`,
`bench_vae_compiled_batch.py`, `inspect_crop_quality.py`, `inspect_fullbody_crops.py`,
`inspect_latent_ops.py`, `inspect_latent_perturb.py`, `encdec_totalseg_crops.py`,
`bench_vae_resolution.py`, `compare_latent_vs_primus.py`, `bench_encoder_fwd.py`,
`bench_vae_trt.py` — all in `experiments/3d/synth_task_generation/`.
`src/synth_gen_maisi_dataset.py`, `data/maisi_classes.py`,
`configs/experiment/3d/dataset/synth_gen_maisi.yaml`.

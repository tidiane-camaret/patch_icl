# query_prior_injection

Goal: feed the **full level-i prediction mask** (not just its centre-of-mass) as a spatial
prior into level i+1 of the PatchSet3D coarse→fine cascade
(`configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml`).

## Planned injection point (design, not yet built)

`src/models/patchset3d.py:524` — the query token's mask/occupancy column is currently
`qry_occ = sup_occ.mean(dim=1)` (the support-mean prior). Plan: add
`forward(..., query_prior=None)`; when given, build `qry_occ` from the warped previous-level
prediction through the existing `_down_to` / `_mask_tiles_3d` + `mask_embed` path instead of
the support-mean. No new params, token count stays R³ (RoPE/Fourier-PE untouched), falls back
to support-mean at level 0. The geometric warp (level i−1 augmented grid → level i augmented
grid) lives in `experiments/3d/cascade.py`, reusing the captured `GeoState.grid` /
`invert_geo_center` machinery.

## Step 1 — baseline: what the coarse level looks like on its own

Checkpoint: `.../3d_train/2026-08-31_66_train_spacing_range_3_6/best.pt`
(`experiment=57_organs_encoder_from_scratch data.context_size=1 data.crop_spacing_mm=6
data.train_spacing_range=[3,6]`; arch: `encoder=plainconv_ts`, e=768, l=4, a=12, R=16,
mask_patch_size/decode=8, feat_norm=self, encoder trainable). `load_state_dict`: 0 missing /
0 unexpected.

Predicted **at 6 mm, no augmentation**, one val case per class (K=1 context). Repro:
`.venv_blackwell/bin/python results/3d/query_prior_injection/predict_6mm_baseline.py`.
Figures in `baseline_6mm_no_prior/` — panels: context | target CT | GT (red) | pred (blue).

| # | class | subject | Dice | GT vox | pred vox |
|---|-------|---------|------|--------|----------|
| 00 | spleen | s0045 | 0.901 | 969 | 922 |
| 01 | kidney_right | s0045 | 0.917 | 562 | 590 |
| 02 | kidney_left | s0032 | 0.444 | 53 | 178 |
| 03 | gallbladder | s0045 | 0.077 | 43 | 512 |
| 04 | liver | s0032 | 0.765 | 4443 | 5116 |
| 05 | stomach | s0544 | 0.588 | 1421 | 2087 |
| 06 | pancreas | s0045 | 0.239 | 260 | 387 |
| 07 | adrenal_gland_right | s0095 | 0.084 | 19 | 159 |
| 08 | adrenal_gland_left | s0032 | 0.032 | 3 | 151 |
| 09 | lung_upper_lobe_left | s0021 | 0.799 | 3362 | 2762 |

### Observations

- **Localisation is reliable** — every prediction sits on the correct organ; no gross misses.
- **Systematic over-segmentation.** `pred vox > GT vox` on 8/10 cases; the blob has a
  roughly fixed floor of ~150–500 voxels regardless of the true size.
- **Large organs are already good** at 6 mm: spleen / kidney_right / liver / lung lobe score
  0.76–0.92 with near-matched volumes.
- **Sub-cell structures collapse to the floor.** gallbladder (43 vox → 512, Dice 0.08),
  adrenals (3–19 vox → ~150, Dice ≤0.08), a partial-FOV kidney_left (53 vox → 178). One
  6 mm mask cell is 8³·6³ mm³ effective, so these are ≤1 cell — nothing the coarse head can
  resolve.

### Takeaway for the prior work

The coarse level gives a **trustworthy location and a loose over-inclusive extent**. That is
exactly the signal a full-mask prior should carry into level i+1: the fine level re-crops on a
correct centre and can use the coarse mask shape as a soft starting region to tighten, while
the small-organ cases depend entirely on the finer spacing (crop_spacing 1.5 mm) to escape the
size floor — the prior mostly contributes the *where*, not the *what*, for those.

---

## Step 2 — placing the 6 mm prediction on the 3 mm augmented grid

Tool: `warp_probe.py` (`.venv_blackwell`). Builds a real 2-level cascade step
([6, 3] mm) with the exp57 GPU augmentor (flip p=0.5/axis, affine p=0.5 ±30°, no
elastic/deform), runs the level-0 forward, then tests warp methods that resample the 6 mm
volume onto the level-1 augmented grid. Fidelity = warp level-0 GT, compare to level-1 GT
(Dice + centroid offset); timing = wall clock around grid-build + `grid_sample`.

### Key structural finding

`run_cascade` passes the **same `geo_gen` seed at every level**, and `_geometric` builds the
flip/affine grid from that RNG on a T³ lattice → the augmentation grid is **identical across
levels**. Measured: `max|grid0 − grid1| = 0.0`, `flips_equal = True`, affine-fit residual
`3.6e-7` (pure affine). So the only thing separating level 0 and level 1 is the crop, and
`crop_geom` (starts/crop_sizes in native-CT voxels) is the shared frame to compose through.

### Methods & results (4 batches, K=1, aug ON)

| method | what it models | GT-warp Dice | COM Δ (mm) | ms / vol |
|--------|----------------|--------------|------------|----------|
| **M0** | crop-geom compose only | 0.37 ± 0.32 | 45.7 ± 54.8 | 0.3 |
| **M1** | + flip axis-reversal (both ends) | 0.60 ± 0.30 | 16.4 ± 20.6 | 0.5 |
| **M2** | + closed-form affine conjugation | **0.87 ± 0.06** | **2.7 ± 0.5** | 1.4 |

Chain for M2 (per level-1 output voxel `g1`, using the shared grid affine `R,t` fit from the
captured grid, and the shared flip record):

```
g1 --[R g1 + t]--> flip1 --> cropgeom1_fwd --> native-CT --> cropgeom0_inv --> flip0 --[R^-1(. - t)]--> g0
prior_on_grid1 = grid_sample(vol6, normalize(g0))
```

### Reading

- **Flips alone break M0** (case 02: no rotation, one flip → M0 0.86 vs M1/M2 0.93).
- **Rotation breaks M1** (case 00: ±30° rotation → M1 0.19 vs M2 0.90).
- **M2 is essentially exact.** COM Δ 2.7 mm ≈ half a 6 mm voxel — the discretisation floor.
  The Dice ceiling ~0.87 is the 6 mm → 3 mm resolution loss (blocky prior vs crisp fine GT),
  not misplacement. For a *prior* that is fine — it needs the right location and rough
  extent, which M2 delivers.
- **Cost is negligible:** 1.4 ms/vol vs ~800 ms for the encoder step. The per-b Python loop +
  `lstsq` (16³ strided fit) is the only overhead; batchable if it ever matters.

### Recommendation / next

Use **M2** as the `query_prior` placement. It is exact for affine-only aug (exp57/exp59
config). If elastic/`deform` are later enabled, M2 degrades in step with the affine-fit
residual → add **M3**: iterative fixed-point inversion of the nonlinear part of the shared
grid (grid0), seeded from the M2 affine. Everything else in the chain is unchanged.

---

## Step 3 — M3: nonlinear (deform) inversion

Added to `warp_probe.py`. Only needed when `augmentations.task.deform.p > 0` or
`elastic.p > 0` (both **off** in exp57/exp59 today). Exercised by forcing calibrated
diffeomorphic deform (`max_disp=0.15`, 6 control points, `deform.p=1`) on top of the flips +
±30° rotation.

**M3 = M2 + Newton inversion of the composed grid.** M2's closed-form affine inverse is
replaced by an LM-damped Newton solve of `Φ(x)=y`:

```
x0 = R^-1 (y - t)                        # affine-inverse seed
x <- x + (J(x) + 1e-2 I)^-1 (y - Φ(x))   # J = grid Jacobian (finite diff), clamp x to [-1.2,1.2]
```

`Φ` and its Jacobian are read from the captured grid by `grid_sample` (grid0 == grid1). The
`1e-2 I` damping matters: `_geometric` does `grid = (grid+phi).clamp(-1,1)`, so `J` is exactly
singular wherever the deform pushed the grid past the border — LM degrades to a damped
gradient step there (those output voxels have no true preimage anyway).

Earlier attempts that **failed**: affine-preconditioned fixed point (`x += R⁻¹(y−Φ(x))`) and
inverse-consistent displacement iteration (`w ← −ψ(z+w)`) both stalled at residual ≈ the
deform magnitude — at calibrated strength `‖∇ψ‖ ≳ 1`, so no first-order fixed point contracts.
Newton (second-order, uses `J`) is required.

### Results

Deform OFF (affine only) — M3 must not regress vs M2:

| method | GT-warp Dice | COM Δ (mm) | ms/vol | Newton resid |
|--------|--------------|------------|--------|--------------|
| M2 | 0.865 | 2.74 | 1.5 | – |
| M3 | 0.863 | 2.74 | 15.3 | 2.5e-7 |

Deform ON (`deform.p=1`, calibrated `max_disp=0.15`) — M2 collapses, M3 recovers:

| method | GT-warp Dice | COM Δ (mm) | ms/vol |
|--------|--------------|------------|--------|
| M0 | 0.06 | 63 | 0.3 |
| M1 | 0.11 | 46 | 0.5 |
| M2 | 0.15 | 42 | 1.4 |
| **M3** (3 it) | 0.76 | 6.6 | 6.9 |
| **M3** (5 it) | 0.81 | 4.3 | 10.1 |
| **M3** (8 it) | 0.82 | 3.4 | 15.3 |

### Reading

- **M3 reduces to M2 exactly when only affine fired** (Newton converges to 2.5e-7 in the
  affine case) — safe to always use, no regression.
- **Under forced calibrated deform, M3 restores the warp** (COM 4 mm vs M2's 42 mm; Dice 0.81
  vs 0.15). The residual gap to the deform-off ceiling (~0.86) is the extra 6→3 mm blur from
  the larger warp, not misplacement.
- **Knee at ~5 Newton iters** (0.81 Dice / 4 mm / 10 ms). 8 iters buys little.
- **Cost 10–15 ms/vol** vs ~800 ms/encoder step (~1–2 %). The per-b Python loop + 2 M
  batched 3×3 solves per iter dominate; batchable across b if it ever matters.

### Bottom line

- **Ship M2** for the `query_prior` placement — exp57/exp59 run affine-only aug, M2 is exact
  there at 1.4 ms.
- **Keep M3 available** behind the same code path (it *is* M2 when deform is off) for any
  future run that enables `deform`/`elastic`; use 5 Newton iters.

---

## Step 4 — coverage: 21 classes, weighted to low val/dice at 6 mm

`warp_probe.py` now targets an explicit class list (thin vessels, sub-cell organs, cervical
vertebrae, ribs — the bottom of `wandb/latest-run` val/dice — plus spleen/liver controls),
`tasks_per_class=3`, deform **off** (exp57/59 config → M2 operative, M3 ≡ M2). 63 cases.
Warp fidelity is GT-based, so it is independent of the model seeing the class.

### Global

| method | GT-warp Dice | COM Δ (mm) | ms/vol |
|--------|--------------|------------|--------|
| M0 | 0.19 | 42 | 0.4 |
| M1 | 0.31 | 21 | 0.5 |
| M2 | **0.56** | **4.8** | 1.8 |
| M3 | 0.56 | 4.7 | 10.7 |

### Per-class COM Δ (mm), M2, sorted worst-first

| class | gt6 vox | M2 Dice | M2 COM Δ |
|-------|--------:|--------:|---------:|
| iliac_artery_right | 73 | 0.50 | 15.7 |
| subclavian_artery_right | 33 | 0.45 | 9.4 |
| rib_left_10 | 66 | 0.44 | 7.9 |
| rib_right_5 | 28 | 0.38 | 6.5 |
| portal_vein_and_splenic_vein | 118 | 0.63 | 5.7 |
| common_carotid_artery_left | 17 | 0.36 | 5.6 |
| adrenal_gland_right | 3 | 0.13 | 4.7 |
| esophagus | 74 | 0.57 | 4.4 |
| iliopsoas_right | 798 | 0.56 | 3.8 |
| vertebrae_C5 | 53 | 0.59 | 3.7 |
| prostate | 91 | 0.78 | 3.6 |
| common_carotid_artery_right | 6 | 0.34 | 3.4 |
| adrenal_gland_left | 9 | 0.20 | 3.3 |
| atrial_appendage_left | 19 | 0.37 | 2.9 |
| gallbladder | 84 | 0.79 | 2.7 |
| pancreas | 359 | 0.81 | 2.6 |
| liver | 9877 | 0.94 | 2.6 |
| duodenum | 196 | 0.77 | 2.5 |
| spleen | 636 | 0.91 | 2.4 |
| vertebrae_C1 | 67 | 0.61 | 1.6 |

### Reading

- **Warp geometry is exact everywhere.** Affine-fit residual 3.6e-7. Compact structures land
  at COM Δ **2.4–4.7 mm ≈ half a 6 mm voxel — the discretisation floor — regardless of class
  difficulty or size**: adrenal_gland_right (3 voxels) 4.7 mm, liver (9877 voxels) 2.6 mm.
- **Elongated thin structures are the only outliers** (iliac/subclavian artery, ribs, portal
  vein: COM Δ 6–16 mm, Dice 0.4–0.5). The figures show this is **not misplacement** — the
  warped prediction sits on the fine GT where it exists — but a **FOV mismatch**: the 6 mm
  crop (768 mm FOV) and the 3 mm crop (384 mm FOV, re-centred on the predicted COM) contain
  *different sub-segments* of a long tube, so their masks and centroids genuinely differ.
  For a prior this is harmless — it still marks "the structure runs through here".
- **M0/M1 stay badly off** on the rotation/flip cases (0.19 / 42 mm, 0.31 / 21 mm) — the
  affine conjugation in M2 is doing real work on ~50 % of tasks.

### Conclusion (unchanged, now with coverage)

Ship **M2**. The warp is placement-accurate for every class type tried; the residual COM
error is the 6 mm voxel floor for compact organs and a crop-FOV artefact (not a warp fault)
for long vessels/ribs. M3 available for future deform-enabled runs.

---

## Step 5 — should the next-level crop centre come from the warped pred?

Question: instead of `invert_geo_center(COM(logit_i))` (map the coarse prob-weighted
centroid back through the aug inverse), warp the level-i prediction to native space and take
*its* COM there.

**For affine-only aug (exp57/exp59): identical, provably and empirically.** The
prob-weighted centroid is linear and every step of the inverse chain (aug grid, flip,
crop-geom) is affine, and `COM(Φ⁻¹·p) = Φ⁻¹·COM(p)` for affine Φ. Probe check over 63 cases:

| aug | affine-fit resid | `|invert_geo_center(COM) − COM(dewarped)|` |
|-----|------------------|--------------------------------------------|
| affine only (deform off) | 3.6e-7 | **0.000** native voxels (all 63) |
| + calibrated deform p=1  | 3.2e-1 | mean **15.6**, max 118 native voxels |

**Under nonlinear deform they diverge and the volume-COM is the correct one:**
`Φ⁻¹(COM(p_aug)) ≠ COM(Φ⁻¹(p_aug))`, so pushing a single centroid point through the deform
inverse is biased by how the deform redistributes mass; de-warping the volume first
(`_dewarp_native`, M3 Newton inverse) and then taking the COM is unbiased.

Not circular: the warp target is **native-CT space** (the frame the level-(i+1) crop is placed
in), which needs only level-i's own `crop_geom_i` + `geo_i` — known right after the level-i
forward. Warping onto the level-(i+1) grid *would* be circular (that grid's geometry depends
on the centre being computed).

### Decision

- **Keep `invert_geo_center(COM)`** for the centre — exact for affine aug (exp57/exp59) and
  cheaper (one point vs a volume `grid_sample` + Newton).
- **If `deform.p`/`elastic.p` > 0:** switch the centre to `COM(M3-dewarp(σ(logit_i) →
  level-i native crop))`, in lockstep with switching the prior placement M2 → M3. Same Newton
  machinery, one extra volume.

---

## Step 6 — implemented

**Model** (`src/models/patchset3d.py`): `forward(..., query_prior=None)`; when given, `_attn`
builds `qry_occ` from it via the new `_prior_occupancy` (same `_down_to` / `_mask_tiles_3d`
path as the support masks) instead of the support-mean. No new params, R³ token count
unchanged, checkpoint-compatible; `query_prior=None` is byte-identical to before.
`train_forward`/`predict`/`_native_logit` gained a pass-through kwarg.

**Cascade** (`experiments/3d/cascade.py`): `_warp_prior_m2` (affine-conjugation warp onto
level i's aug grid; grid+flips level-invariant, only `crop_geom` differs), `_warp_prior_cropgeom`
(no-aug val path), `_build_query_prior` (**detach** → sigmoid → interpolate to T → warp).
`run_cascade(..., query_prior=False)`: for i>0 builds the prior from level i-1's logit and
passes it to `_forward_level` (forwarded into `model()` only when non-None, so fake models in
tests are unaffected). `evaluate_cascade` + `train.py` thread `cfg.data.cascade_query_prior`.

**Config**: `data.cascade_query_prior` (default false); set `true` in
`59_organs_cascade_from_scratch.yaml`.

**Tests**: +3 in `tests/test_patchset3d.py`, +2 in `experiments/3d/tests/test_cascade.py`;
50 passed. End-to-end smoke on the 66_3_6 checkpoint (bf16, B=1): level 1 receives a
`(1,1,128³)` prior in [0,1], logits finite, both the M2 (train+aug) and crop-geom (val) warp
paths run.

---

## Step 7 — prior source modes

`data.cascade_query_prior` is now an enum (bool still works: `false`→`none`, `true`→`pred`):

| mode | prior | measures |
|------|-------|----------|
| `none` | support-mean fallback | baseline |
| `pred` | `σ(logit_{i-1})` detached, M2-warped | the real cascade |
| `gt_coarse` | level i−1's aug GT, M2-warped onto level i's grid | perfect-coarse-seg ceiling |
| `gt_fine` | level i's own aug GT (no warp) | perfect-prior ceiling |

Plus `data.cascade_query_prior_hard` (bool) — threshold the prior at 0.5 before the mask
embed. **Eval uses the same mode as train**, so `gt_*` runs report oracle ceilings, not
deployable Dice.

`gt_coarse` / `gt_fine` answer "is a good prior worth the training cost" before investing in
making the coarse stage produce one. Implementation: `_prior_mode` normalises the config;
`_build_query_prior(mode, hard, prev_logit, prev_label, cur_label, …)` picks the source
(`prev_label` = level i−1's `cur["label"]`, kept per loop iteration) → warp → optional hard.
`run_cascade` gained `query_prior_hard`; `train.py` + `evaluate_cascade` thread both keys;
`_assert_cascade_supported` validates the enum.

Smoke (66_3_6 ckpt, bf16): all four modes build a `(1,1,128³)` prior in `[0,1]`, logits
finite. `pred` mean 0.0041 vs `gt_coarse`/`gt_fine` ~0.0028 — the coarse level's known
over-segmentation, visible in the prior mass.

---

## Step 8 — profiled test run (exp59 cascade + query_prior=pred)

`experiment=59_organs_cascade_from_scratch`, weights-only warm-start from
`66_train_spacing_range_3_6/best.pt`, `cascade_spacings=[6,3]`, `cascade_query_prior=pred`,
`decoder=conv`, `train_classes=balanced`, `val_classes=all`. Blackwell RTX PRO 6000 (97 GiB),
`.venv_blackwell`. Capped to 40 train steps + one eval (`tasks_per_class=2` → 234 tasks).

### Compile-on OOMs — had to run eager at B=2

`arch.compile=true` (the config default) hits **CUDA OOM at ~94.5 GiB during the inductor
compile pass** — `decoder=conv` (~6× decode FLOPs) × 2 cascade levels × grad × dynamic-shape
compile exceeds the card. OOM'd at B=8 **and** B=4. Ran **`arch.compile=false`, B=2** to get a
profile. This config is **not launchable as written**; options: eager, B≤2, switch
`decoder=fine_filter`, gradient checkpointing, or narrower `plainconv_ts_features_per_stage` /
`fine_proj_dim`.

### Timing (eager, B=2)

| phase | cost |
|-------|------|
| build + weights load + measure_flops | ~9 s |
| first train step (cuDNN autotune) | 6.2 s |
| steady train step | **0.69 s/step** (0.35 s/item) — 40 steps in 31 s |
| eval warm-up (forkserver workers + first recrop I/O) | ~28 s |
| eval, per cascade task (2 fwд + recrop + stitch) | **~0.54 s/task** — 234 tasks in ~100 s |
| **total wall** | **~172 s** |

`query_prior` (M2 warp) itself is ~1.4 ms/vol — invisible at this step time. The step cost is
the second forward + the conv decoder; ~1.7× the single-level per-item cost (exp57 ≈ 0.21
s/item @ B=4). Full-scale eval (`tasks_per_class=10` × 117 ≈ 1170 tasks) ≈ **10 min/eval**,
recrop-I/O-bound — building `ct_raw_3mm.npy` caches would remove most of the stalls.

### Resource usage

| metric | train (B=2) | eval |
|--------|-------------|------|
| GPU util | 95–100 % | 0–100 % bursty (~40 % mean; recrop-I/O gaps) |
| GPU mem | **28.9 GiB flat** | 20–29 GiB |
| GPU power | 320–410 W | up to **534 W** (≈ TDP) |
| process RSS | ~27 GiB | **~44 GiB** (main + 22 forkserver eval workers mmap'ing npy) |
| process CPU | ~210 % | ~360 % (infer) → **>1000 %** (`_stitched_native_dice_multi` numpy scoring) |
| threads / procs | 107 / 13 | **914 / 23** |
| host RAM | ~62 GiB (6 %) | ~72 GiB (7 %) of ~1 TB |
| load1 | 4 → 8 | up to 13.8 (Dice scoring) |

`val_dice=0.4080` after 40 warm-start steps ≈ the checkpoint's 0.4094 — prior wiring intact.

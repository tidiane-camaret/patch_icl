# Change log

## 2026-08-20 — In-context dataloader v2

Added `src/incontext_dataset_v2.py` (`InContextDataset` engine + `LoadRequest`/
`LoadResult`/`VolumeProvider`) and `src/providers/totalseg.py` (`TotalSegProvider`
+ `crop_and_place`). Generic task-assembly separated from source I/O; single
raw_ct organ-crop load path; per-item state via dataclasses (no more
`_cur_rng`/`_last_crop_geom` side-channels). Gated behind `data.loader_v2`; the
v1 `TotalSegInContextDataset` is untouched. Spec:
docs/superpowers/specs/2026-08-20-incontext-dataloader-v2-design.md.

- **Intentional spacing divergence (v1 vs v2):** `TotalSegProvider._load_spacings`
  returns the TRUE native spacing from `spacings.json` and passes it to
  `crop_and_place`. v1's `use_crop` path instead substitutes 1.5mm-isotropic
  spacing for all subjects. As a result, v2 produces DIFFERENT crops than v1 for
  anisotropic subjects — this is intentional and correct (v1's substitution was a
  latent shortcut). Consumers comparing v1/v2 outputs or reusing v1-trained
  checkpoints should expect crop differences on anisotropic data.

## 2026-08-15 — GPU augmentation pipeline (batched, replaces CPU per-item augs)
- NEW `src/gpu_augment.py::GpuAugmentor` — batched on-device aug run in the train loop after batch.to(device), before model(). Own torch ops (no deps), non-differentiable. `_geometric` (shared per task group_size=T, or independent group_size=1), `_batched_intensity` (brightness/contrast/gamma/noise/blur/sharpness/low-res), `_batched_gin_ipa` (grouped conv, groups=N). Mode dispatch on `aug_mode` (0 real, 1 synth, 2 self_context).
- Dataset `defer_aug_to_gpu` (from `augmentations.gpu`): __getitem__/_get_synth_item skip apply_*_aug, emit RAW volumes + aug_mode; collate stacks aug_mode. train.py moves batch to device once + augments. Behind `augmentations.gpu` (default false → CPU path unchanged). Exact CPU repro is a non-goal; tests assert shape/range/K+1-sharing/eval-identity. See docs/superpowers/specs/2026-08-15-gpu-augmentation-pipeline-design.md.

## 2026-08-15 — GIN / IPA CPU aug cost benchmark
- NEW: `experiments/3d/bench_cpu_aug.py` — times the per-item training aug path (apply_task_aug + apply_intensity_aug×N) at experiment-42 shapes (image_size 128³, context_size 3 → N=4), loading the real resolved aug config (nnunet base ⊕ exp-42 overrides). Compares GIN off / gin / ipa; single torch thread by default (mimics one worker in a saturated 16-worker pool). `--size/--n/--iters/--threads`.
- RESULT (N=4, 128³, 1 thread, 30 iters): full per-item aug mean **off 438.6 ms → gin 1110.8 (+672) → ipa 1623.0 (+1184)**. Isolated intensity (only gin/ipa): gin 572 ms (~143/vol), ipa 975 ms (~244/vol; ≈2× gin = 2 GIN copies + blend). Baseline "off" bimodal (median 278 ≪ mean 439: task affine/elastic at p=0.2); gin/ipa rows forced p=1 so tight — median deltas +818/+1312 ms.
- TAKEAWAY: at p=1 GIN ~2.5×'s and IPA ~3.7×'s per-item aug; scale by chosen p (ipa @p=0.5 ≈ +590 ms/item). With 16 workers IPA@p=1 caps ~10 items/s ≈ 1.25 steps/s @B=8 before I/O/crop/resample — will bottleneck (exp-30 ran ~3.5 steps/s @B=1). Motivates moving these pure-torch conv3d/interpolate augs to a batched GPU stage.

## 2026-08-15 — GIN / IPA appearance augmentation (Ouyang et al. causality-inspired DG)
- NEW: `_gin_transform_3d` + `_ipa_blend_3d` in `src/augmentations.py`, and a config-gated `gin` block at the TOP of `apply_intensity_aug`. GIN = a stack of `n_layer` conv3d layers with FRESH random kernels+shifts each call (no learned state, `torch.no_grad`), leaky-relu between layers, blended with the source via random α, then Frobenius-norm matched to the input; clamped to `[CT_NORM_MIN, CT_NORM_MAX]`. IPA (`mode: ipa`) generates `ipa_copies` independent GIN warps of the same volume and mixes them with a smooth `ipa_control_points³`→trilinear-upsampled random field (spatial appearance blend). Applied INDEPENDENTLY per volume (apply_intensity_aug runs per volume at dataloader:1208), so the target and its K contexts get DIFFERENT appearance warps → trains shape/structure matching, not intensity matching.
- Adapted from `/home/dpxuser/repos/Causality-Medical-Image-Domain-Generalization` (`models/imagefilter3d.py` GINGroupConv3D, `ginipa.py`). Ported to CPU (stripped hardcoded `.cuda()`), single-channel CT (vs the reference's 3-channel replication), functional (no `nn.Module`). The reference's 738-line `AdvBias` bias-field module is NOT ported — it is used in ginipa.py purely as a random smooth-field source, replaced here by the coarse-control-point field pattern already in `_simulate_bias_field`.
- Config: `augmentations.intensity.gin` in `configs/augmentations/nnunet.yaml` — `p` (default 0.0 → no-op, existing runs unchanged), `mode` (gin|ipa), `n_layer`, `interm_channel`, `scale_pool`, `out_norm`, `ipa_copies`, `ipa_control_points`.
- VERIFIED: tests/test_gin_aug.py 5/5 (shape preserved; GIN changes the image; IPA multi-copy shape; determinism under seeded random+torch; output stays in CT norm range via apply_intensity_aug; p=0 → passthrough no-op).

## 2026-08-15 — self_context.synth_masks gains a "supervoxel" source (folds old p_synth SVs in)
- NEW: `data.self_context.synth_masks.sources` (list, default `[ellipse]`) — when the self-context synth branch fires (`synth_masks.p`), it now picks a source UNIFORMLY per item. `ellipse` = the existing random-rotated-ellipsoid target label; `supervoxel` = a supervoxel group from THIS subject's `label_synth_{method}` volume, placed on the target grid so it pairs with the real target image (real image ↔ anatomically-plausible blob label). This unifies the old top-level `p_synth`/`_get_synth_item` supervoxel path into the self_context clone+augment machinery. `data.synth_method` (e.g. seeds3d) still names the SV file; the old `p_synth` path is left untouched (inert at p_synth=0).
- Config: `data.self_context.synth_masks.supervoxel.{n_merge_min,n_merge_max}` merges randint(min,max) face-adjacent supervoxels into one label (reuses the adjacency cache + `_sample_merged_svs`). common._self_context forwards `sources`/`supervoxel` verbatim (synth_masks.p still split-specific). configs/experiment/3d/experiment/42_reg_to_all.yaml set to `sources: [ellipse, supervoxel]`, supervoxel merge 1..3.
- Impl (src/totalseg_dataloader_incontext): new `_supervoxel_label_on_grid(subj, crop_geom)` — crop path slices the native SV volume with the target's crop geom (starts/crop_sizes), picks a SV group present in the crop, resamples (`_resample_multiclass`) + `place_label` onto the T³ grid; fast path (crop_geom=None) uses `label_synth_{method}_{size}.npy`. Returns None (→ ellipse fallback) when the subject has no usable supervoxel. The synth-cache setup now builds adjacency / picks the base-label suffix on the EFFECTIVE merge max (old n_synth_merge_max ∪ the supervoxel source's). synth_coord logged from the SV centroid; synth_radii left NaN (ellipse-only).
- FIX (experiments/3d/plot_dataset_items.py `_merge_batches`): it cat'd every key of batches[0] across all batch-of-1 dicts, but ellipse-synth items carry `synth_radii_mm`/`synth_coord` while supervoxel/real items don't → `KeyError: 'synth_radii_mm'` on mixed-source batches. Now iterates the UNION of keys and pads batches missing an optional tensor key (NaN for float, 0 else), mirroring incontext_collate_fn's mixed-batch NaN-padding.
- VERIFIED: tests/test_synth_supervoxel.py 4/4 (SV label == SV occupancy on grid; crop-geom branch; no-SV → None; __getitem__ over both a SV subject and a no-SV subject → non-empty synth label + self-context clone) + test_synth_ellipsoid.py 5/5. _self_context parses the new shape (train forwards sources/supervoxel, eval p→0). plot_dataset_items on experiment=42 renders a mix of ellipsoid + irregular supervoxel blobs (results/3d/dataset_items.png), self-context cloned.

## 2026-08-14 — register_routed flex hangs in ptxas on cold cache → arch.register_flex escape hatch
- Symptom: `experiment=40_colipri_large_head ... arch.register_routed=true arch.thinking_rows=128` stuck at `train e0: 0/125`, GPU 0%. NOT the measure_flops fix — that completes (prints `predict GFLOPs: 4971.72`). faulthandler stack showed the real hang: flex_attention's Triton kernel stuck in `triton .../make_cubin → subprocess.communicate → _wait` (ptxas). Reproduced in isolation: flex first fwd+bwd = 8s with a WARM ~/.triton cache but **>10 min (killed) on a COLD node-local cache** — train.py forces `TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR` onto /tmp (node-local, cold) to dodge cross-node GLIBC poisoning, so the first batch recompiles the flex kernel from scratch and ptxas stalls on Blackwell/cu13. Earlier isolation was fast only because prior bench runs had warmed ~/.triton (head_dim=64 flex kernels).
- Fix: `arch.register_flex` (bool, default true) — when false, register_routed uses the dense r×r bool-mask SDPA (cuDNN/flash backend, no custom Triton kernel), which compiles/runs cleanly cold. Verified: `arch.register_flex=false` on the exp40 command trains epoch 0 → 125/125 and into epoch 1 on a cold cache (~0.8s/step at K=1). Dense is fine at small K (exp40 K=1, r=8320); heavy at large K (bench_attn_pattern.py). Threaded through build_model + build_register_block_mask gate; eval rebuilds from stored arch (old ckpts default true).

## 2026-08-14 — register_routed FlexAttention fast-path (block-diagonal + register border)
- Dense-masked SDPA was register_routed's worst case (O(r²) mem, no flash). Replaced with a FlexAttention `BlockMask` (`build_register_block_mask` in pfn_seg_2d.py: keep iff either endpoint is a register OR same image block) threaded through the transformer like `attn_mask`; the layer calls `flex_attention` when a block_mask is given, else the old SDPA branches. Guarded by `HAS_FLEX` — CPU / no-flex torch falls back to the dense bool mask (verified numerically equivalent, max 3.9e-3 @ bf16). flex is `torch.compile`d at import (its eager path materializes scores). Needs the CC/CXX shim (Triton) — added to bench_attn_pattern.py; already in train.py.
- "flex_attention called without torch.compile" warning: emitted ONLY by `measure_flops` (the startup FLOP probe), not the training loop. FlopCounterMode is a TorchDispatchMode → disables dynamo → the compiled flex runs eager (uncountable HOP + the warning). Fix: `_FLEX_ENABLED` switch in pfn_seg_2d (build_register_block_mask returns None when off → dense bool-mask SDPA, which IS traceable); measure_flops toggles it off around the count, restores in finally. Verified: 0 warnings in measure_flops AND in the normal fwd+bwd (which stays fused whether or not the transformer is torch.compiled). The reported predict-GFLOPs for register_routed is then the DENSE-equivalent (overstates true cost — sparse figure lives in bench_attn_pattern.py).
- FLOP note: FlopCounterMode can't introspect the fused flex kernel (nor masked-SDPA's real sparsity) — it reported IDENTICAL FLOPs for both patterns, hiding the point. Switched the bench to analytical sample-axis attention FLOPs (4·L·bc·e·r² full vs 4·L·bc·e·(T·N·(N+n_t)+n_t·r) routed; ratio ~T=K+1).
- Result (R=16, mask_patch=8, 128³, B=1, fwd+bwd, Blackwell) flex vs full_attn vs OLD dense-masked, at K=16: attn FLOPs **3518 vs 59593 GF (~17× = T fewer)**; RAM **25.1 vs 24.9 GB** (flash-like — the old 87.9 GB / 4.8 GB mask blowup GONE); latency **346 ms vs 840 ms** (flex now **2.4× faster than full_attn**, and 6.8× faster than the old 2336 ms dense-masked). At K=1 register_routed≈full_attn (45 vs 44 ms, no overhead); gap widens with K. RAM is now encoder-bound (T volumes), not attention-bound, for both patterns. Verdict: register-routing is strictly cheaper than full_attn with flex — the cost objection is removed.

## 2026-08-14 — PatchSet3D register_routed attention pattern + benchmark
- New `arch.register_routed` (bool, default false) in `PatchSet3D`: the thinking rows become the ONLY cross-image path — registers read every token and every token reads registers, while each image otherwise attends only within its own N-cell block (K support blocks + 1 query block). No direct ctx↔tgt token attention → blocks the ctx→tgt feature-matching shortcut. Built as an explicit r×r bool mask in `_attn` (reuses the existing `attn_mask` SDPA branch; no transformer change), so it drops off flash onto mem-efficient/math SDPA. Precedence: `full_attn` > `register_routed` > `query_self_attn` > default read-only. Plumbed via `build_model` (train.py); eval rebuilds it from the stored `arch` (old ckpts default false).
- Benchmark `experiments/3d/bench_attn_pattern.py`: register_routed vs full_attn, realistic R=16/mask_patch=8/128³, B=1, fwd+bwd bf16, sweep K∈{1,2,4,8,16}. Blackwell (97 GB): FLOPs identical (FlopCounterMode counts dense r² regardless of mask — the block-diagonal sparsity is NOT realized by dense-masked SDPA). Cost is pure flash-loss overhead: RAM 3.9/6.5/12.9/30.9/**87.9** GB vs full_attn 3.0/4.5/7.4/13.2/**24.9** GB (~3.5× at K=16); latency 65/120/255/719/**2336** ms vs 44/73/129/303/**840** ms (~2.8× at K=16). Mask alone is r²≈4.8 GB at K=16. Verified mask connectivity (registers all-to-all, per-image block-diagonal, zero cross-image) + fwd/bwd on a tiny CPU grid.

## 2026-08-12 — CoLiPri frozen encoder moved to shared NFS + config-referenced

The primus (CoLiPri) sidecar json + weights were loaded from a **CWD-relative** path
(`results/checkpoints/primus_colipri.{json,pt}`) baked into patchset3d checkpoints — so eval /
inference only worked when launched from the repo root, and not at all for users without the
repo. Moved both files to shared NFS `…/ANALYSIS_20251122/checkpoints/colipri/` (weights path
inside the json rewritten to absolute NFS) and referenced them from config:
- `cluster/nfs.yaml`: new `paths.colipri`.
- `eval.yaml`: `eval.primus_sidecar: ${paths.colipri}/primus_colipri.json`.
- `eval.py._build_model`: redirects an existing (primus) checkpoint's `arch.primus_sidecar`
  to `eval.primus_sidecar` — weight-free override, same pattern as `feat_norm`.
- Train/feature_sim configs (`model/patchset3d*.yaml`, `experiment/30`, `feature_sim.yaml`)
  and the producer `scripts/extract_colipri_backbone.py` now point at `${paths.colipri}` /
  the absolute NFS dir. Repo copies deleted.

Verified: `infer_cli.py` run from `/tmp` with the repo copies removed loads the encoder from
NFS and reproduces Dice 0.8546 (bladder, s0000/s0001).

Also added `experiments/3d/infer_cli.py`: a general argparse CLI (`--target`, `--context IMAGE
MASK` repeatable, `--checkpoint`, `--out`, `--gt`, `--crop-spacings`) that composes the eval cfg
internally (pins PWD so eval.yaml's `hydra.searchpath` resolves from any dir) and calls
predict_nifti.

Shared-env packaging (`scripts/sync_patchset_env.sh`): the repo is on a personal path other
users can't read, so the code is snapshotted into the shared `patchset` env. The script rsyncs
source files (py/yaml/json, ~3.6M) into `$ENV/share/patchset_infer/` preserving the tree (flat
sibling imports + __file__ logic resolve unchanged) and writes a `$ENV/bin/patchset-infer`
launcher that runs the env Python on the bundled CLI. Any env user then does
`conda activate patchset && patchset-infer --target … --context IMG MASK --checkpoint …`.
Re-run the sync script after code changes (no editable install possible — source is unreadable
to others). Verified from /tmp with PYTHONPATH unset: loads code from the bundle (not the repo),
encoder from NFS, reproduces Dice 0.8546.

## 2026-08-12 — `patchset` conda env for nifti inference + first real run

Created a minimal conda env `patchset` (python 3.11, torch 2.6.0+cu124) to run
`experiments/3d/infer_nifti.py` on the Ampere nodes (nero/thor/loki). Recipe pinned in
`requirements-patchset-infer.txt`. Key gotcha: `dynamic_network_architectures` (primus encoder
dep) has no torch pin, so a naive install pulls torch 2.13 + a cu13 cuDNN wheel; a cu13 cuDNN
inside cu124 torch raises `CUDNN_STATUS_NOT_INITIALIZED` specifically on the primus 8^3-kernel
patch-embed conv3d (small convs still pass, so it masquerades as a flaky GPU/env bug). Fix:
install torch first, then dna with `--no-deps`, keeping torch 2.6's bundled cuDNN 9.1.0.70.

First real run (`run_infer_heart.py`, s0000 target / s0001 context, +organ=urinary_bladder —
heart is out-of-FOV for these pelvic scans): Dice 0.8546, coarse-only 0.6733, cascade gain
+0.18. Confirms the nifti cascade path reproduces the eval accuracy path end-to-end.

Orientation fix: `load_nifti` canonicalises every input to RAS (to match training), which
previously meant the saved mask carried the RAS-canonical affine/grid — misaligned by voxel
index with a non-canonical input CT. Added `_to_original_orientation`: the model still runs in
RAS, but the returned/saved mask is reoriented back to the target file's stored orientation +
affine (no-op when the target is already RAS). Metrics stay in canonical space (Dice is
orientation-invariant). New test `test_predict_nifti_output_matches_target_orientation` uses an
LAS target to prove the round-trip.

## 2026-08-12 — coarse->fine cascade eval (spacing_cascade)

Added a real coarse->fine cascade to the 3D spacing sweep, complementing the existing
geometry-only `spacing_locator` containment metric (which never re-ran the model). New flag
`eval.spacing_cascade=true` (on top of `eval.spacing_sweep=[4,1.5]`): for each descending
consecutive pair, the coarse pass's soft prediction centroid is mapped back to native voxels
and used as the TARGET crop centre for a second, finer-spacing pass — measuring real Dice on
the crop the model itself localized. Reported alongside the GT-centred fine pass (oracle) so
the cascade->oracle gap isolates localization loss. Empty coarse predictions fall back to the
volume centre. Needs `model.train_forward` (soft prob); totalseg / use_crop / descending
sweep only (same guards as the locator). Adds one extra fine pass per descending pair.

Changes:
- `src/totalseg_dataloader_incontext.py`: `_pred_centers` override dict `{(subj,cls): center}`
  consulted by the TARGET load only (contexts stay GT-centred); `"volume_center"` sentinel for
  empty coarse preds. `_organ_crop_arrays` now stashes `_last_crop_geom` (starts, crop_sizes,
  out_sizes, pad_lo); the single-label crop path attaches it to the item as `crop_geom` (4,3),
  collated to `(B,4,3)` — the grid<->native map needed to invert a prediction.
- `experiments/3d/evaluate.py`: `_predicted_native_center` (prob centroid -> native voxels via
  crop_geom; empty -> "volume_center"); `evaluate_classes(pred_centers_out=...)` fills it on the
  coarse pass; `evaluate_spacing_sweep(cascade=True)` runs the extra predicted-crop fine pass
  (injecting `ds._pred_centers` before the loader iterates), tagging rows/cases `cascade_from`.
- `experiments/3d/eval.py`: `spacing_cascade` flag, validation, wandb/CSV columns, and a cascade
  summary block. Cascade rows are kept out of the base mean / per-spacing curve to avoid
  double-counting classes.

Cascade Dice is scored END-TO-END on the ORIGINAL native volume, not as an average of per-crop
Dice: the coarse pred is composited into the native volume (label.npy grid) and the fine pred
overwrites its region (finer replaces coarser), then Dice'd against the native GT
(`_stitched_native_dice`; per-sample preds captured bit-packed via
`evaluate_classes(pred_geom_out=)`, composited by `_write_native` = inverse of the crop). Each
cascade row also reports `coarse_only_dice` (same native score from the coarse pred alone) as the
no-refinement baseline; the printed/logged `mean_dice_cascade` + `mean_cascade_gain` use these.
Smoke (spleen/liver, jitter=0): cascade 0.90 vs coarse-only 0.70–0.81, gain +0.09…+0.20.

Cascade figures (`eval.cascade_figures=true`, needs spacing_cascade + save_figures): one 2x5
coarse->fine panel per class under `<out_dir>/figures/cascade/`. Columns (top=target,
bottom=1st context): (1) coarse img+GT, (2) coarse img + coarse pred + fine/oracle bboxes,
(3) fine img+GT, (4) fine img + fine pred, (5) coarse img + fine pred REFITTED into the coarse
frame + GT.
- `experiments/3d/evaluate.py`: `save_cascade_figure` (masked-foreground overlays + bbox
  rectangles), `_grid_centroid`, `_refit_into_coarse` (fine->native->coarse affine remap from
  both crop geometries — grids are centre-padded so a plain resize misaligns; `_refit_into_box`
  is the no-geometry fallback), `_save_cascade_pair` (pairs the two passes' caches per class +
  prints a geometry guardrail: refit(fine GT) vs coarse GT Dice, ~0.9 when correct);
  `evaluate_classes(figure_cache=, figure_classes=)` stashes one case's arrays per class (first
  seen, keyed (subj,cls)) including `crop_geom`; `evaluate_spacing_sweep(cascade_figures=)` runs
  the coarse+fine captures and emits the panels. Dummy driver: `experiments/3d/cascade_fig_demo.py`.

### 2026-08-12 — crop start clamp bug (found via cascade-figure geometry check)
`_organ_crop_arrays` (`src/totalseg_dataloader_incontext.py`) computed the crop start as
`lo=max(0, ideal-j); hi=max(lo, min(s-cs, ideal+j))` — it never capped `lo` at `s-cs`, so when
the organ centroid sits high on a near-full axis (`ideal > s-cs`, e.g. `cs==s`) the start
exceeded `s-cs`; the numpy slice `[start:start+cs]` then silently clipped short (organ cut off,
that axis stretched at resample, and the recorded `crop_sizes` no longer matched the real slice).
Fixed by clamping start into `[0, s-cs]` (`smax=max(0,s-cs); lo=min(max(0,ideal-j),smax);
hi=min(max(0,ideal+j),smax)`). This matches the documented "take what exists and pad" intent and
made the cascade refit's geometry guardrail jump from ~0.04 to ~0.9. IMPACT: triggers whenever
an organ centroid is in the upper half of an axis the crop spans fully (`cs==s`), which at COARSE
spacing (4mm, 512mm FOV ≥ most volumes) is the common case, not an edge case — so it pervasively
shifted/stretched large-FOV crops. Finer crops (1.5mm) are smaller than the volume so are mostly
unaffected. This is a latent bug in the shared train/eval data path (independent of the cascade
feature); the `spacing_1_to_4` checkpoint was trained with the buggy coarse crops, so re-eval /
retrain implications are the user's call.

## 2026-08-12 — nifti in-context cascade inference

Added `experiments/3d/infer_nifti.py::predict_nifti(cfg, target_path, context_pairs,
gt_path=None, out_path=None)`: runs the 4mm->1.5mm in-context cascade on arbitrary
nifti files (GT-free target; coarse crops on the volume centre, fine recenters on the
coarse prediction centroid). Reuses eval._build_model + evaluate._write_native /
_predicted_native_center + the newly extracted crop helpers (organ_crop_arrays /
place_image / place_label / resample_binary, refactored out of TotalSegInContextDataset,
behaviour-preserving). Returns the native-grid mask + optional Dice and coarse-only Dice.

## 2026-08-11 — PatchSet3D random token masking

Added SimMIM-style in-place token masking to `PatchSet3D` (`src/models/patchset3d.py`):
`arch.token_mask_ratio_support` / `arch.token_mask_ratio_query` (both default 0.0 = off)
randomly replace whole tokens (image + mask columns) with a learned `mask_token` during
training only (`self.training`), keeping the R³ token count so the compiled transformer and
RoPE are unaffected. `forward` now also returns `mask_support` / `mask_query` (None when off)
as the hook for a future auxiliary reconstruction loss. Eval/predict never mask.
Spec: docs/superpowers/specs/2026-08-11-patchset3d-token-masking-design.md.

## 2026-08-11 — self-context ceiling probe + exp40 large-head config (CoLiPri size-stall study)
Investigating why the frozen-CoLiPri in-context run stalls at val Dice ~0.60 with Dice strongly
correlated to object size (small anatomy → low Dice). Established via `plot_dataset_items` +
`measure_occupancy.py` that masks are NOT destroyed by aug (intact at native 128³) and — key arch
fact — the mask I/O is full 128³ (`mask_patch_size=8` tiles in, `mask_patch_decode_size=8` tiles
out, target pooled to the 128³ logit = identity), so there is NO coarse-grid target ceiling. The
only stream still at 16³ is the IMAGE token grid (one CoLiPri feature per 8³=12 mm patch); small
objects fill <35% of even their best image cell (`max_cell_occ`), so the bottleneck is image-side
matching/discriminability, not the mask decoder. Decoder-ceiling test: feed the target's own
image+mask as context (trivial matching) and see whether small-object Dice lifts.

Changes:
- `src/totalseg_dataloader_incontext.py`: `self_context` — after aug, with probability
  `self_context` (0..1; True→1.0) overwrite the K contexts with clones of the (augmented) target
  (bit-identical → trivial matching, leakage by design). `self_context_augs=true` re-augments each
  clone INDEPENDENTLY via the new PER-IMAGE aug (`augmentations.per_image` geometric + the
  already-per-image intensity) so target≠context by a controlled pose/appearance jitter — the
  pose-invariance training lever (drift probe showed the copy model is fragile to rigid pose).
  NB per_image, NOT task: `task` aug applies ONE shared transform across all K+1 volumes (preserves
  correspondence) and must not be used per-image. Verified: per_image off → mask IoU 1.0; per_image
  affine.p=1.0 → warped mask (IoU≈0.05). Default self_context=0.0.
- `src/augmentations.py`: new `apply_per_image_aug(image, mask, cfg)` — independent flip/affine/
  elastic on ONE volume (same schema as `task`), the per-image counterpart of the shared
  `apply_task_aug`.
- `configs/augmentations/nnunet.yaml`: new `augmentations.per_image` block (flip/affine/elastic,
  all p=0 by default → no-op). Raise `per_image.affine.p` to warp the self-context copy.
- `experiments/3d/common.py`: forward `data.self_context` + `data.self_context_augs` into both
  train and eval dataset builds.
- `configs/experiment/3d/dataset/totalseg.yaml`: declare `self_context: 0.0`, `self_context_augs: false`.
- `experiments/3d/probe_context_drift.py`: context-drift probe (warp context only by known
  translate/rotate/scale/elastic, Dice vs magnitude; mag0 == self-context ceiling sanity).
- `configs/experiment/3d/experiment/40_colipri_large_head.yaml`: exp40 — larger transformer
  (l=4/e=768/h=3072/a=12) on frozen CoLiPri, train_classes=all, class_balanced, crop_spacing=1.5,
  mask_occupancy_thr=0.1, raw_ct, spacing-aware encoder + transformer RoPE.
- `experiments/3d/measure_occupancy.py`: per-class target occupancy over an epoch (native + 16³
  grid), writes results/3d/exp35_occupancy_{items,per_class}.csv.
- `experiments/3d/probe_precision.py`: (from earlier) live per-module dtype probe.
Run (b) status: warm-started exp40 from the 0.604 cross-subject best.pt with `data.self_context=true`
(wandb `40_selfctx_ceiling`). Epoch 0 already val 0.776 (rising) — leans toward matching/encoder
being the limiter, but the size-resolved per-class verdict awaits convergence.

## 2026-08-11 — wire `encoder_precision` into the frozen primus/CoLiPri encoder + dtype probe
Sanity-checking the CoLiPri-features-for-in-context-seg study (exp 35). Traced + empirically
verified per-module precision: under the ambient bf16 autocast, the frozen CoLiPri ViT computes
its attention/MLP in **bf16** (params stored fp32); `eva`/transformer tails emit fp32 (LayerNorm is
autocast-fp32); loss is fp32. Train and eval are byte-identical in precision. Found `arch.encoder_precision`
was a **no-op for primus** (only wired to tap_ct), so the "is bf16 rounding of the frozen features
costing signal?" control was impossible.

Changes:
- `src/models/primus_encoder.py`: `precision` ctor arg + `_autocast_ctx()`; `_encode_batch` wraps the
  ViT forward in it, overriding the ambient autocast for the encoder region only (fp32 disables autocast
  → fp32 params compute in fp32; bf16/fp16 force that dtype). Off-cuda no-op.
- `src/models/patchset3d.py`: `build_model` already read `encoder_precision`; now threaded into `PrimusEncoder`.
- `configs/experiment/3d/model/patchset3d.yaml`: declare `arch.encoder_precision: bf16` (bf16|fp16|fp32)
  so it's overridable in struct mode.
- `experiments/3d/probe_precision.py`: new one-shot probe — synthetic batch, forward hooks, prints live
  per-module in/out dtypes for TRAIN and EVAL. `python experiments/3d/probe_precision.py experiment=35_colipri_enc_8_i_128`.
  Verified: default bf16 unchanged; `arch.encoder_precision=fp32` makes `down_projection`/`eva` fp32 while
  downstream stays bf16. A/B run deferred (to test later).

## 2026-08-11 — spacing-aware 3D RoPE for the PatchSet3D transformer (flag-gated)
Gave the downstream transformer the same positional scheme as the spacing-aware encoder (exp 36):
3D axial RoPE on the sample (cross-cell) axis, positions scaled by `spacing/train_mm` (2 mm), so a
fixed anatomical distance maps to a fixed rotary phase across the 1–4 mm range. RoPE-only when on
(drops the additive spacing-blind Fourier PE), mirroring the encoder (`use_abs_pos_embed=False`).
Spec: `docs/superpowers/specs/2026-08-11-transformer-rope-spacing-design.md`.

Changes:
- `src/rope.py`: new `build_3d_rope_freqs_from_positions(head_dim, positions, theta)` — RoPE tables
  from explicit float positions (spacing scaling + thinking-row zeros), consistent with the grid builder.
- `src/models/pfn_seg_2d.py`: opt-in `rope=None` arg on `TransformerEncoderStack/Layer.forward`;
  when set, rotates q,k in the sample-axis attention (before the `[:sep]` slice). `None` ⇒ byte-identical,
  so `patchset_cnn`/`patchset_pfn`/2D `ImagePFN` are untouched.
- `src/models/patchset3d.py`: `transformer_rope` / `rope_theta` args; `_tokens` skips the additive PE in
  RoPE-only mode; `_attn` builds the row-sequence RoPE (thinking=(0,0,0)) and threads `spacing` through.
- `experiments/3d/train.py`: `build_model` passes the two new arch keys.
- `configs/experiment/3d/experiment/37_colipri_transformer_rope_128.yaml`: exp 37 (extends 36, adds
  `arch.transformer_rope=true`) — both stages spacing-aware.
- Tests: `tests/test_rope_positions.py`, `tests/test_patchset3d_rope.py` (grid-equivalence, RoPE-only drops
  PE, spacing scales positions, 2 mm == no-spacing identity, forward+backward finite).

## 2026-08-10 — raw_ct pipeline: store ct_raw + normalise in the loader (flag-gated)
Implemented the raw-intensity store evaluated below. Opt-in, back-compatible, storage-neutral.

- `scripts/convert_to_npy.py --store-raw`: also writes native `ct_raw.npy` (int16 raw HU for
  CT — lossless, same 2 B/voxel as float16; float16 for MRI). For MRI also writes per-volume
  norm stats to `ct_stats.json` (like spacings.json). Normalisation now delegates to shared
  helpers so the written ct.npy is byte-identical to the loader's on-the-fly result.
- `src/totalseg_dataset.py`: shared `normalize_ct` (global pointwise — crop==whole),
  `mri_stats` (whole-volume percentile+z-score stats) + `normalize_mri` (apply stats to a
  crop; verified Δ=0 vs the old `_normalise_mri`, and crop-with-whole-stats == slice of whole).
- `TotalSegInContextDataset(raw_ct=False, modality="ct")`: when raw_ct, native reads
  (`_organ_crop_arrays`, synth crop) load `ct_raw.npy` via `_load_native_ct_mmap` (int16
  dtype guard against feeding raw HU as normalised) and normalise ONLY the crop. Pre-resized
  fast path stays normalised (unaffected). `_normalize_native` dispatches CT/MRI. Threaded
  through `common.build_dataset` + `make_eval_loader` (modality from `_source_root` is_mri),
  the more_labels subclass, and `get_incontext_loader`. Config key `data.raw_ct` (default
  false) in configs/.../dataset/totalseg.yaml.

Verified end-to-end (primus crimson-deluge-224, use_crop, 128³): production raw_ct loader
crop vs float16 crop max|Δ|=0.49 HU, labels identical; convert --store-raw writes int16 HU
(range e.g. [-1094, 1865] — even preserves >1573 bone the old clip dropped); eval.py
`data.raw_ct=true` vs false n=10 → Dice 0.7241 vs 0.7240. Enable for the whole dataset with:
`python scripts/convert_to_npy.py --store-raw` then train/eval with `data.raw_ct=true`.
Note: pre-resized files stay normalised, so raw_ct only changes native/use_crop reads.

## 2026-08-10 — raw-HU store vs float16 store: no signal loss for frozen encoders
Investigated storing raw CT intensities as .npy and delegating normalization to the
dataloader (to standardize the pipeline across encoders that de-normalize back to HU:
primus, tap_ct — see src/models/{primus,tapct}_encoder.py). Findings:

- **Storage**: identical. Raw CT is integer-valued int16 HU (verified: ct.nii.gz header
  dtype int16, get_fdata all-integer), so raw int16 = 2 B/voxel = the current float16 npy.
  Zero storage cost, strictly lossless. nii.gz is ~61% the size but slow to decompress.
- **Signal**: the storage clip [-1007, 1573] (CT_CLIP_MIN/MAX) is WIDER than both encoders'
  own clip windows (primus preproc [-1000,1000] per primus_colipri.json; tap_ct [-1008,822]),
  so the pre-clip is transparent — both re-clip tighter after de-norm. The ONLY real
  difference a raw store makes to these frozen encoders is removing float16 quantization,
  measured at max 0.49 HU / mean 0.066 HU per voxel (s0000; ≤0.24 HU p99).
- **Dataloading time**: +clip+z-score is ~5 ms on the 128³ use_crop crop (the 65 ms figure
  is the full ~9.5M-voxel native volume; use_crop only normalizes the crop). Hidden behind
  the ~90–150 ms/sample frozen-ViT encode and worker parallelism → no throughput impact.

Measured on the two trained checkpoints (test, 50 subj, benchmark 47 classes,
experiments/3d/eval.py): primus (crimson-deluge-224) baseline Dice 0.7009 / 89 ms /
1772 GFLOP; tap_ct (sweet-armadillo-237) baseline Dice 0.7015 / 149 ms / 5055 GFLOP.

Paired A/B (experiments/3d/rawcheck_ab.py): same deterministic eval crops, loader reads
float16 ct.npy vs int16 raw-HU ct_raw + in-loader normalize. Input differs by exactly the
float16 bound (max|Δ|=0.49 HU, identical=False). Result — no meaningful change:
  primus: ΔDice −0.000001 (max|Δ| 0.0015), pred voxel agreement 99.9995%
  tap_ct: ΔDice +0.000010 (max|Δ| 0.011),  pred voxel agreement 99.9952%
Conclusion: for these frozen encoders the raw-store proposal is a strict simplification/
storage-neutral win with no accuracy cost; the current float16-clip round-trip is lossless
in practice. (Caveat: the conv-from-scratch encoder expects z-scored input — a raw store
must still hand the model a normalized tensor, or feed each encoder its own norm.)

## Renamed data.spacing_range → data.train_spacing_range (2026-08-07)
The variable-spacing knob is train-only (train_loader wraps the sampler in
SpacingBatchSampler when it's set; make_eval_loader never reads it — eval always
crops at the fixed data.crop_spacing_mm). Renamed the config key to signpost that
scope. crop_spacing_mm kept as-is: it's the fixed eval spacing AND the train spacing
whenever train_spacing_range is null, so "spacing_eval" would have been wrong. Updated
totalseg.yaml (+ clarifying comments), experiments 36/37/38/39, model/patchset3d.yaml,
common.train_loader, and plot_dataset_items.py. SpacingBatchSampler's own `spacing_range`
constructor arg is unchanged (generic component param).

## Rebuilt .venv_thor as clean uv-managed venv (2026-08-06)

2026-08-06: **`.venv_thor` had drifted from uv tracking** (pip-installed on top: torch stayed at
2.5.1+cu121 while `uv.lock` had moved to 2.12.1; ~77 packages unknown to the lock; a mixed
CUDA-12/CUDA-13 nvidia runtime). Snapshotted the working set and rebuilt a fresh venv at
`.venv_thor_fresh` (uv venv, 200 pkgs) — original `.venv_thor` left untouched for safe swap.
Recipe: (1) `uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url
https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
--index-strategy unsafe-best-match`; (2) `uv pip install -r <clean freeze minus torch/nvidia/triton>`.
Dropped `mamba_ssm` and its `tilelang`→`nvidia-cutlass-dsl`→CUDA-13 dependency chain: `mamba_ssm`
2.3.2.post1 is **non-importable in the original venv too** (`triton.set_allocator` AttributeError on
triton 3.1.0), and the repo's only use is a guarded fallback import in
`experiments/3d/encoder_bench/encoders_standin.py` that already fails — so removing it is
functionally equivalent and yields a clean cu12-only env. Exact pins snapshot in
`requirements-thor.lock.txt`. Verified: torch runs a CUDA op on GPU; core libs (monai, nnunetv2,
transformers, cupy, hydra, wandb, totalsegmentator, lightning, marimo) all import.

2026-08-06 (follow-up): **Adopted the thor env into `pyproject.toml` so it is uv-lock-managed**
(no more `UV_NO_SYNC`). Changes: (1) pinned the `cu121` extra to `torch==2.5.1` / `torchvision==0.20.1`
(cu124/cu128 left free — nero/odin unaffected; lock kept `2.6.0+cu124` and `2.11.0+cu128`, only added
a `2.8.0+cu128` branch); (2) added a non-default `[dependency-groups] thor` (transformers,
totalsegmentator, cupy-cuda12x, cuda-bindings, nninteractive, fvcore, python-dotenv + dev tooling
pytest/ipykernel/ninja/wheel/uvloop/httptools/watchfiles). Regenerated `uv.lock` and ran
`uv sync --extra cu121 --group thor` into `.venv_thor_fresh` — re-resolve churned ~15 pkgs to latest
(notably `fury` 0.12→2.0, which swaps vtk→pygfx/wgpu; verified fury/pygfx still import). Daily use:
`UV_PROJECT_ENVIRONMENT=.venv_thor_fresh uv run --extra cu121 --group thor <cmd>` (extras can't be
defaulted per-machine, so `--extra cu121 --group thor` stay explicit).

## eval.py per-sample in_train (2026-08-05)

2026-08-05: **eval.py's per-sample `cases` table now logs `in_train`, matching train.py's
`val/samples`.** `in_train` is filled by resolving `cfg.data.train_classes` via
`_source_root` (guarded; falls back to None for non-totalseg sources) and passing
`train_classes` to `build_sample_table`. `soft_dice`/`loss` stay empty on the eval path:
soft-Dice needs a second (untimed) forward per batch (~2x eval time), so it was dropped —
eval reports only the timed `model.predict` Dice. (train.py's val step still logs soft_dice,
since it already runs that forward.)

2026-08-05: **Per-sample `spacing` column added to the shared sample table (train + eval).**
`evaluate_classes` now writes `case["spacing"] = float(batch["spacing"][i, 0])` when the
dataset reports spacing (the first-axis scalar the spacing-aware model consumes; the crop
path is isotropic), and `_SAMPLE_TABLE_COLS`/`build_sample_table` gained the `spacing`
column. Since both train.py's val step and eval.py route through `evaluate_classes`, both
`val/samples` and eval's `cases` tables get it; NaN for datasets that emit no spacing.

## More-labels eval wiring (2026-08-05)

2026-08-05: **Wired the extra TotalSegmentator `more_labels` classes into eval.**
New `src/totalseg_more_labels_dataset.py` (`TotalSegMoreLabelsDataset`, subclass of
`TotalSegInContextDataset`) roots at `totalseg_test_more_labels/`: class identity is
the task-qualified key `"{task}/{name}"` from `more_labels_classes.json` (329 unique
names collide across 37 tasks), subject→classes from `more_labels_subject_classes.json`
(no label.npy scan). `_load` loads CT from `ct.nii.gz` reproducing `convert_to_npy`'s
normalise + `_iso_resize` (aligns pixel-for-pixel with the pre-sized `{task}_64³.npy`
masks; verified by `experiments/totalseg_more_labels/check_more_labels_dataset.py`),
and the binary mask as `task_array == local_id`. New `data.source=totalseg_more_labels`
routes through `common.py`/`eval.py`; `resolve_more_labels_classes` exposes the 285
classes present in ≥2 subjects (`val_classes=all`). Run:
`python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse`.
Eval-only: no synth/aug.

2026-08-05: **Added a `use_crop` path so `more_labels` can be evaluated at a chosen
spacing.** The extra subjects' `ct.nii.gz` is the same 1.5 mm-isotropic scan as the main
tree (identical spacing+shape; task masks share that grid). New
`experiments/totalseg_more_labels/generate_crop_assets.py` writes per-subject native
`ct.npy` (float16, reproducing `convert_to_npy`'s `_normalise_ct`; ~0.9 GB/25 subj) + root
`spacings.json`, so the crop path mmaps+slices the CT (cheap under many workers) and reads
true native spacing. `TotalSegMoreLabelsDataset` now accepts `use_crop`/`crop_spacing_mm`/
`crop_jitter`: `_load_crop` crops the task mask (`==local_id`) + `ct.npy` at fixed physical
extent `T*crop_spacing_mm` → resampled to T³ (isotropic `crop_spacing_mm`/voxel), reusing
the base `_organ_crop_arrays`/`_place_*`/`_resample_binary`; centroids come from a
per-`{task}/{name}` cache built once from the native task masks (pickled). Also fixes the
reported `item["spacing"]` (was a 1 mm placeholder → now effective/crop spacing via the new
`spacings.json`). `common.build_dataset` threads the crop knobs; config gains
`data.crop_spacing_mm`. Verified: crops organ-centred (label mass at voxel ~32/64),
reported spacing tracks 1.5/2/4 mm, foreground shrinks with FOV. Run:
`python experiments/3d/eval.py dataset=totalseg_more_labels eval.model=medverse
data.use_crop=true data.crop_spacing_mm=2.0`.

## Exp 38: medverse variable-spacing with exp 36's exact optim (trial) (2026-08-03)

`configs/experiment/3d/experiment/38_medverse_varspacing_36optim_128.yaml` — exp 37 but with
exp 36's train/optim recipe (adamw 1e-4 / wd 0.01, eval_every 10) instead of medverse-native
adam 3e-5 / wd 0. A deliberate stress test: medverse was moved OFF adamw/1e-4 because its
unbounded head diverges (docs/logs.md 2026-07-29), so instability is expected. Verified every
optim key equals exp 36; oob_weight resolves to 10.0 in both (exp 36 via train.py's default),
so the NaN guard stays. Two params NOT copied from exp 36: `checkpoint` (kept orig_weights —
36's null would random-init medverse, a different experiment) and oob_weight (already matches).

## Medverse variable-spacing baseline for exp 36 (no spacing injection) (2026-08-03)

`configs/experiment/3d/experiment/37_medverse_varspacing_128.yaml` — the medverse baseline
against the spacing-aware CoLiPri run (exp 36). Trains medverse on exp 36's data/task/aug and
the same 1-4 mm per-batch crop spacing (`data.spacing_range=[1,4]`), but medverse never sees
the spacing signal: `train_epoch`'s medverse branch calls `train_forward(image, ctx_in,
ctx_out)` with no `spacing` (only spacing-aware patchset3d threads it). Isolates "inject
spacing into the frozen encoder" vs "train blind on the same variable-spacing crops".
Inherits exp 31 (Medverse-twin recipe: `orig_weights`, Adam 3e-5/wd0, bce_dice, oob_weight=10
— medverse NaNs under adamw/1e-4 or smooth_l1) and layers exp 36's deltas (spacing_range,
mask_downsample=occupancy@0.5, train_classes=balanced, exp-35 task elastic/affine). Eval, like
exp 36, is fixed at crop_spacing_mm=2, so both models compare at 2 mm. Config-only; the
SpacingBatchSampler loader is already model-agnostic.

## Spacing-aware RoPE for the frozen CoLiPri encoder + variable-spacing training (2026-08-03)

Lets the frozen Primus/CoLiPri ViT honour physical voxel spacing so we can train over a
range of spacings (e.g. 1-4 mm) at fixed image_size. Position is injected through RoPE's
`ref_feat_shape`: `ref[axis] = grid[axis] * spacing[axis] / train_pitch` (train_pitch = 2 mm
from the sidecar), making rotary phase proportional to physical distance instead of token
index. At 2 mm this is bit-for-bit identical to the native-grid identity table (strict
superset). Anisotropic spacing falls out per-axis. Verified: real compiled eva stays ONE
graph across 5 spacings (in-place `copy_` of the RoPE buffer preserves tensor identity —
never reassign, or torch.compile recompiles/goes stale).

- `src/models/primus_encoder.py`: `_set_rope_identity_grid` -> `_set_rope_scaled_grid`
  (spacing + train_mm args, in-place copy_; builds the table on the existing buffer's
  device/dtype — else the first grid-change build lands on CPU and breaks the cuda/bf16 eva
  matmul); `PrimusEncoder(spacing_aware=...)` (implies
  native_grid; train pitch from sidecar `preproc.spacing_mm`); `forward/_encode(_batch)`
  thread a per-batch `spacing`; encode-cache key includes spacing.
- `src/models/patchset3d.py`: `PatchSet3D(encoder_spacing_aware=...)` + `.spacing_aware`;
  `forward/predict/train_forward/_native_logit` thread `spacing`; conv path unchanged.
- `experiments/3d/common.py`: `SpacingBatchSampler` (one log-uniform spacing per batch so a
  single shared RoPE table serves the forward); `train_loader` uses it when
  `data.spacing_range` is set. `src/totalseg_dataloader_incontext.py`: `__getitem__` accepts
  `(idx, spacing)`, `_crop_mm` per-item override drives crop extent + reported spacing.
- `experiments/3d/train.py` + `evaluate.py`: pass per-batch `batch["spacing"][0]` to
  spacing-aware models only (medverse untouched). Configs: `arch.encoder_spacing_aware`,
  `data.spacing_range`; new `experiment=36_colipri_spacing_aware_128` (spacing_range=[1,4]).
- NOTE: 4 mm extrapolates RoPE past its trained max token position (1 mm is safe interp);
  narrow toward [1,3] if the coarse end regresses. Eval runs at fixed crop_spacing_mm.
- `experiments/3d/eval.py`: eval restores only `arch` from the checkpoint, not `data`, so
  `_warn_uninherited_data` prints (at eval start) any input-fidelity data param
  (image_size, crop_spacing_mm, use_crop, context_size, mask_downsample, mask_occupancy_thr,
  source) whose eval-config value differs from the checkpoint's stored training `data` —
  guarding against silent train/test drift (e.g. defaults 1.5 mm/nearest vs trained 2 mm/occupancy).

## Startup FLOPs logging split into encoder / transformer / total (2026-08-03)

`measure_flops` (`experiments/3d/evaluate.py`) now returns a `{total, encoder, transformer}`
dict of GFLOPs instead of a single float. The per-component shares come from
`FlopCounterMode.get_flop_counts()`, keyed by submodule class name (`PrimusEncoder`/`ConvEncoder3D`
and `TransformerEncoderStack`); the small img/mask embeds + decoder are the unreported remainder.
`encoder`/`transformer` are `None` for models without those submodules (e.g. medverse). Both
`train.py` and `eval.py` print the breakdown (`predict GFLOPs: X [encoder=… transformer=…]`) and
`train.py` logs `gflops_encoder` / `gflops_transformer` to wandb alongside `gflops`. Useful for
seeing how much of the total scales with `data.image_size` under `arch.encoder_native_grid`
(only the encoder share moves; the transformer stays fixed at `resolution`).

## PrimusEncoder gains opt-in `arch.encoder_native_grid` flag (2026-08-03)

Added `arch.encoder_native_grid: false` (default off) to `PatchSet3D` / `PrimusEncoder`.
When **false** (default): the frozen ViT always resamples its input to a fixed 192³ grid before
tokenizing — encoder FLOPs are independent of `data.image_size`.
When **true**: the encoder tokenizes at `image_size/8` (patch size 8) directly; a per-call
identity-RoPE grid is built to match the actual token count, so encoder FLOPs scale linearly
with `data.image_size`. The transformer/decode path is unchanged — `_down_to` pools the
native-grid features to the configured `resolution` before the attention stack, so the rest of
the model is unaffected.

Config: `configs/experiment/3d/model/patchset3d.yaml` → `arch.encoder_native_grid`.
Wired through `experiments/3d/train.py` arch dict and `PatchSet3D.__init__` (after
`encoder_stage`). Design doc: `docs/superpowers/specs/2026-08-03-primus-encoder-native-grid-design.md`.

## Occupancy mask downsampling: keep thin-structure signal under heavy downsampling (2026-07-30)

At coarse crops (e.g. crop_spacing_mm=4 → ~2.7× downsample from 1.5mm native), nearest-mode label
resize point-samples one native voxel per output voxel, so thin structures that don't hit the sample
point vanish — verified: a 2-voxel-thick tube (256 native fg) → **0 fg** under nearest at 4× down.
Stats on the 4mm regime: ~1% of (subject,class) masks drop below 1 voxel, ~5% below a 3³ block;
worst on thin vessels/distal ribs/adrenals.

Added `data.mask_downsample` (`nearest` default | `occupancy`) + `data.mask_occupancy_thr` (default
0.5) to `TotalSegInContextDataset`, threaded through `common.py` (build_dataset + make_eval_loader),
`get_incontext_loader`, and `dataset/totalseg.yaml`. Occupancy mode area-pools the binary mask to the
foreground FRACTION per output voxel and keeps voxels ≥ thr (thr→0 = keep every touched voxel /
dilate; 0.5 = majority). Applied at all three resize sites: `_load_crop`, `_load_crop_multi`, and the
synth slow-path resize, via shared helpers `_resample_binary` / `_resample_multiclass`.
Binary path GUARANTEES a non-empty mask (keeps the densest voxel) if the input had any fg; multiclass
(num_labels>1) has no hard guarantee at thr=0.5 — a small class split across cell boundaries can still
drop, so use a low thr there. Default stays `nearest` → existing runs unchanged.

## exp31 medverse bce_dice is unstable: clamp blinds the loss to the unbounded output → add out-of-bounds anchor (2026-07-29)

The real root cause of exp31's instability (supersedes the same-day `_st_clamp`-only entry below,
which was an incomplete diagnosis). Medverse's output head is a **plain, unbounded conv** (no
activation), but the `is_prob` bce_dice loss operates on `out.clamp(0,1)`. So **once the raw
output leaves [0,1] the loss saturates and can no longer see its magnitude** — there is no
gradient pressure keeping the output bounded. Two failure modes, same cause:

- **Original run (hard clamp on BOTH terms):** out-of-range → `torch.clamp` gives zero gradient →
  the run FROZE at all-background (val/loss stuck at 1.1099 = mean fg ~0.8%·−log eps + Dice≈1;
  grad L2 exactly 0.000 at that state).
- **After the first `_st_clamp` "fix" (Dice gradient restored):** the gradient flowed again but
  still nothing bounded the output, so resuming from the epoch-50 best.pt it DIVERGED —
  `logits|max|` climbed 0.79 → 199 → 1e5 → 1e8 → … → **3.7e24** over ~950 steps and overflowed to
  Inf → NaN → CUDA device-side assert (BCE `input∈[0,1]`), while the loss stayed a flat ~1.0 the
  whole time (clamp hid it). `_st_clamp` traded a freeze for an explosion.

Diagnostics that pinned it: a grad-norm comparison at the fixed e50 state showed BCE is INERT —
`bce_dice`, `dice_only`, and `sl1_dice` gave IDENTICAL grad norms (so the earlier "BCE 1/p
explosion" story was wrong); norms scaled inversely with fg fraction (soft-Dice `1/denominator`
on near-empty targets, amplified by the U-Net Jacobian) and are clipped by grad_clip anyway. The
divergence is the UNBOUNDED OUTPUT, not any single loss term. (Why patchset3d's bce_dice is fine:
it emits logits, and BCEWithLogits' restoring gradient `sigmoid(z)−y ∈ [−1,1]` bounds them; the
clamp-on-probability destroys that.)

Fix (train.py `build_loss`, is_prob bce_dice path): add an **out-of-bounds anchor**
`oob_w * mean((out − clamp(out,0,1))²)`, `oob_w = cfg.train.oob_weight` (default 10). It is
exactly 0 while the output is in [0,1] (never fights in-range learning) and only pulls it back
when it escapes. Verified over 700 real optimizer steps from best.pt: `logits|max|` stays ~[0,6]
(vs 3.7e24 without it) and running train dice holds/rises to ~0.257. `SmoothL3L1`-on-raw was too
weak (spiked to 1e4, and corrupted in-range values → dice fell to 0.095). exp31 config now sets
`train.oob_weight: 10.0`. Recommend a FRESH run from orig_weights (best.pt is a collapse-edge
state); the anchor also makes resume safe if desired.

## exp31 medverse: bce_dice collapse from torch.clamp zero-gradient trap → straight-through Dice (2026-07-29, SUPERSEDED)

NB superseded by the entry above — `_st_clamp` alone stops the freeze but not the divergence; the
out-of-bounds anchor is the actual fix. Kept for history.

exp31 (loss=bce_dice, run dark-capybara-204) trained healthily for 50 epochs (train dice
0.16→0.30, val 0.16→0.30) then collapsed at epoch 51-53: train dice 0.26→0.009, and from
epoch 53 on the loss was **frozen** — val/loss exactly 1.1099 for 57 straight epochs, dice
~0.007. Model stuck at all-background, permanently, never recovered.

Root cause (partial): the `is_prob` bce_dice loss fed Medverse's raw (unbounded, linear) conv head
through `prob = out.clamp(eps, 1-eps)` before `F.binary_cross_entropy`. `torch.clamp` has
**zero gradient outside [eps, 1-eps]**, so once the whole output sat in the dead zone the gradient
was exactly 0 → frozen. Fix attempted: `_st_clamp` straight-through clamp on the Dice term. This
removed the freeze but exposed the deeper problem (unbounded output) — see the entry above.

## Fix negative soft-Dice metric for medverse (clamp output to [0,1]) (2026-07-29)

Train `soft` and val `dice_soft` could go negative for medverse (reported soft=-0.05). The
plain-conv head dips slightly below 0 in background; summed over ~262k voxels `prob.sum()`
goes negative, driving the soft-Dice denominator negative and the coefficient >1 (so
`1-softdice` < 0). Metric-only artifact — the bce_dice LOSS already clamps to [eps,1-eps], so
training was fine (hard dice kept rising). Fix: `_to_prob` (train.py) and the eval prob
(evaluate.py) now `clamp(0,1)` the medverse output. No-op for the >=0.5 hard threshold; loss
unchanged. Verified: soft metric -0.11 -> 0.667 on the repro. The already-running process
keeps the old code, so its soft curves stay cosmetically off until relaunch (headline
val/dice from predict is unaffected).

## exp31 medverse: smooth_l1 collapses under class imbalance → switch to bce_dice (2026-07-29)

The first exp31 run (loss=smooth_l1) collapsed: val dice fell from zero-shot ~0.17 to 0.044
after ONE epoch and stayed ~0.01, while train loss sat flat at ~0.085 (= 50·SmoothL3L1, so
SmoothL3L1 ≈ 0.0017 ≈ foreground_fraction/3 → the model predicts background everywhere).

Root cause: `smooth_l1` (SmoothL3L1) is a per-voxel mean regression loss with no foreground
normalization. TotalSeg organs occupy <1% of the 128³/256mm crop, so for most classes
`SmoothL3L1(good mask) ≈ SmoothL3L1(all-zeros)` — no gradient to segment. Measured zero-shot
on real tasks: the loss "signal" (L_zeros − L_pred) tracks foreground fraction — liver
(fg 9%) signal 1.09 / dice 0.64; spleen/aorta/stomach (fg 1–1.6%) signal 0.06–0.13 / dice
0.3–0.4; everything <0.2% fg → signal ~0 / dice ~0. The imbalanced batch gradient pushes
everything to background, eroding even the large organs the pretrained model handled.
patchset3d (val 0.32) is immune because its bce_dice has a soft-Dice term (imbalance-robust).

Fix: exp31 loss `smooth_l1 → bce_dice` (plain BCE + soft-Dice on the [0,1] output; the Dice
term restores gradient — synthetic 0.13%-fg check: loss(good)=0.05 vs loss(zeros)=1.02).
NB `F.binary_cross_entropy` is autocast-UNSAFE and the train forward runs under bf16 autocast,
so build_loss computes the medverse BCE in fp32 under `torch.autocast(enabled=False)` (can't
use the autocast-safe with_logits variant — the output is already a probability, not a logit).
Also switched optim to Medverse-native Adam(3e-5)/no-wd (gentle finetune) from AdamW(1e-4,
wd 0.01). Killed the collapsing run; rerun: `python experiments/3d/train.py
experiment=31_medverse_colipri_task`. NB smooth_l1 worked in Medverse pretraining only
because its brain-ICL data has far higher foreground fraction than abdominal organs here.

## Log params + FLOPs in 3d/train.py (2026-07-29)

`experiments/3d/train.py` now logs, at startup, trainable/total param counts and the model's
compute: `Params: X.XM trainable / Y.YM total (Z%) | predict GFLOPs: G (K=…, size=…)`. GFLOPs
reuse `evaluate.measure_flops` (one `predict()` call, the same metric eval.py reports, so
train/eval are comparable); measured before any `torch.compile` (FLOP count is weight-
independent). Also written to the wandb run config as `params_trainable_M`, `params_total_M`,
`gflops`. Medverse@128³/K=1: 71.1M (100% trainable), 2362.56 GFLOPs.

## Fix: Medverse finetune loss double-activated its already-[0,1] output (2026-07-29)

Root-caused a Medverse finetune that trained/val'd at ~0.1 dice. **The Medverse net outputs
the segmentation map directly in [0,1]** — the released checkpoint has `loss_seg='smoothl3_l1'`
and no output activation (the output block is a plain conv; `predict()`/`demo.py` threshold the
*raw* output at 0.5). Our self-written training helpers instead treated the output as raw logits
and applied a sigmoid in both the loss and the metric:
- `_hard_dice` / train soft-dice / `evaluate.py` val soft-dice: `sigmoid(output)`. Since
  `output ≥ 0 ⇒ sigmoid(output) ≥ 0.5`, this **predicts foreground in every voxel** → dice ≈
  2·FG/(N+FG) ≈ 0.1 regardless of learning.
- `build_loss`: `smooth_l1` did `SmoothL3L1(sigmoid(output), y)` and `bce_dice` did
  `BCEWithLogits(output, y)` — both mis-scale an already-[0,1] map, corrupting gradients.

Verified on real in-context tasks with the zero-shot released model (at its `smoothl3_l1`
optimum): raw-output path → dice 0.81, `SmoothL3L1(out,y)=0.003`; sigmoid path → dice 0.14,
loss 0.041, foreground fraction 1.0. Output range measured ~[0,1] (min ~0, max ~1).

Fix (model-aware; patchset3d still emits logits and is unchanged): added
`model_output_is_prob(cfg)` + `_to_prob()` in `experiments/3d/train.py`. `build_loss`,
`_hard_dice`, and the train-epoch soft metric now skip the sigmoid for medverse; `bce_dice`
uses plain `binary_cross_entropy` (not `_with_logits`) on the [0,1] output.
`evaluate.py:evaluate_classes` gained `output_is_prob=False` (default keeps eval.py's logit
path byte-identical) and `validate_mean` passes it through. Re-verified through the real
trainer functions: fixed path → good cases dice ~0.95 / small loss, old sigmoid path ~0.14.

Also switched `experiment=31_medverse_colipri_task` loss `bce_dice → smooth_l1` (scale 50) —
Medverse's native pretraining objective on the raw [0,1] output.

## Medverse twin of the exp-30 CoLiPri task (2026-07-29)

Added `configs/experiment/3d/experiment/31_medverse_colipri_task.yaml`: the Medverse
counterpart to `experiment=30_colipri_encoder`, for A/B against the frozen-CoLiPri
PatchSet3D trained with `experiment=30_colipri_encoder data.crop_spacing_mm=2
data.train_classes=all`.

Same data recipe (totalseg, `nnunet` aug, `use_crop=true`, `crop_spacing_mm=2`,
`train_classes=all`, `val_classes=all`, no synth, `class_balanced=false`,
`max_ds_len_train=1000`, `context_size=1`) and same optim recipe (bce_dice, AdamW, cosine,
lr 1e-4, wd 0.01, warmup 1, 1000 epochs, eval split=test n_subjects=20). Two deliberate
differences: **model=medverse** (released weights, `checkpoint=orig_weights`, level=1 no AR)
and **input size 128³** (Medverse native, vs 192³). `crop_spacing_mm=2` is held constant, so
FOV = 128·2 = **256mm @ 2mm/vox** (the patchset3d run was 384mm @ 2mm/vox) — same voxel
resolution, smaller body crop. `batch_size=1` kept from exp30 (raise it; 128³ Medverse has
memory headroom). Run: `python experiments/3d/train.py experiment=31_medverse_colipri_task`.

## Crop path: pad thin-FOV axes instead of stretching (2026-07-28)

Fixed anisotropic (elongated) crops under `data.use_crop=true`. `_load_crop` /
`_load_crop_multi` sized the crop with `min(dim, round(phys_ref/spi))`, clamping any axis
whose native FOV was thinner than the fixed physical extent, then resampled to T³ —
stretching that axis. Example: **s0249 kidney_right**, native (285,285,69) @1.5mm, T=192,
`crop_spacing_mm=2` → phys_ref=384mm needs 256 vox/axis, but the 69-slice axis was clamped
to 69 (103.5mm) and stretched to 192 → effective 0.54mm/voxel (3.7× elongation), while
`item["spacing"]` still reported a flat [2,2,2].

Fix: new `_organ_crop_arrays` helper (shared by both crop loaders) slices `min(dim, target)`
real voxels but **symmetrically pads the shortfall** (air = `crop_ct.min()` for CT, 0 for
label) up to the fixed `target` extent before resampling. Every axis is now truly
`crop_spacing_mm`/voxel isotropic; objects keep their native aspect ratio. Verified on
s0249: crop kidney extent 48/36/18mm vs native 48/37.5/18mm (aspect deviation 0.03, 1 vox
of 2mm rounding). Visual check: `results/3d/crop_padfix_check.png` shows padded black bands
on thin axes instead of stretched anatomy. Not specific to `crop_spacing_mm=2` — also
affected the 1.5mm default for any scan thinner than phys_ref on some axis.

**Expected side-effect (not a bug):** limited-FOV / thin-slab scans now show large air
padding instead of stretched anatomy. E.g. s1286 small_bowel is a 60-slice slab (90mm
superior-inferior) → its 384mm-FOV crop is 77% Z padding, so it looks much "smaller" than a
full-body context in `plot_dataset_items` (which slices axis-0 → the padded coronal plane).
In-plane torso width stays consistent (~310–366mm across subjects), confirming uniform
2mm/voxel scale. Reviewed 2026-07-28: padding accepted as the physically-honest behavior
(vs. the old stretch that filled the frame but distorted aspect ratio).

## Frozen CoLiPri encoder training path (2026-07-28)

Validated end-to-end frozen-CoLiPri-encoder training via `model=patchset3d_colipri`
(smoke + unit test + regression, all pass). Key facts:

- **Config:** `experiments/3d/train.py model=patchset3d_colipri data.image_size=[192,192,192]
  data.use_crop=true` — `arch.encoder=primus`, `arch.encoder_frozen=true`,
  `arch.primus_sidecar=results/checkpoints/primus_colipri.json`, `arch.resolution=24`.
- **Run at 1.5mm crop** (default `data.crop_spacing_mm=1.5`; 192 vox × 1.5mm = 288mm FOV,
  24³ ViT tokens = 12mm/token). Use `crop_spacing_mm=2.0` to match CoLiPri's native 2mm
  training resolution.
- **Frozen head-only training:** encoder weights frozen (`requires_grad=False`),
  trainable params = **4.7M** (img/mask embed + pos + transformer + decoder; full CoLiPri
  ~300M frozen). Optimizer auto-excludes frozen params via `requires_grad`.
- **Step cost ≈ CoLiPri forward × (K+1) volumes** (K contexts + 1 target) per training step
  at `arch.compile=false`; expected ~0.39 vol/s × 2 vol/step ≈ 5s/step @192³. With compile
  on (`arch.compile=true`, the config default) expect a modest speedup on the head forward.
- Smoke: `[PrimusEncoder] loaded weights: 10 missing (up_projection decoder, unused), 0
  unexpected`; train bar advances, loss finite, run completes. Frozen-grad unit test:
  `encoder got grad: False | head got grad: True | trainable 4.7M`. Conv regression:
  ConvEncoder3D path unchanged (no PrimusEncoder message, 4.7M trainable, completes).

## Encoder feature-similarity study (2026-07-25)

Added `experiments/3d/feature_sim/` — transformer-free target<->context matching metrics
(prototype cosine -> AUROC/soft-Dice or AP; FG-match margin; top-1 retrieval), a
PatchSet3D encoder adapter (per-stage / concat / img_embed / transformer_q tiers, dense
`R'^3` grids + native-res point sampling), and a Hydra driver (`run.py`) that sweeps
(tier x resolution) over the shared eval loader and writes a tidy `feature_sim.csv` with
the model's real Dice per task. Spec:
docs/superpowers/specs/2026-07-25-patchset3d-encoder-feature-similarity-design.md.
SAM/DINO adapters and Dice-correlation analysis are phase 2.

## 3d train: unified `train.checkpoint` weight-source knob

Collapsed the three medverse weight-source knobs (`train.random_init`,
`train.base_ckpt`, and the path-resume role of `train.checkpoint`) into a single
`train.checkpoint` accepting: `orig_weights` (fine-tune from the released
`Medverse.ckpt`, now the medverse default), `random` (train from scratch), or a
`<path>` to our finetuned `best.pt` (warm-start via `load_finetuned`). `build_model`
now only special-cases `checkpoint == "random"`; `main()` loads weights only for an
actual path (sentinels are handled at construction). Patchset3d is unchanged
(`null`=fresh / `<path>`=resume). Dropped `base_ckpt` (was `null` everywhere;
`MEDVERSE_CKPT` remains a module constant). `MedverseModel`'s constructor API
(`random_init`/`ckpt_path`) is untouched. Design:
docs/superpowers/specs/2026-07-23-unified-train-checkpoint-design.md.

## anchor_synth3d: barycentric multi-anchor positioning + frame-relative size

Replaced the single-anchor offset placement (position `= centroid + offset·extent`,
size `= frac·image`) with an affine-invariant scheme over **4 landmark organs**.
Per item (subject-first): pick a target subject, choose 4 co-occurring anchor
classes present in it (contexts drawn from their co-occurrence set, target
excluded), draw shared **barycentric weights** (`Σwᵢ=1`, mildly affine via
`extrapolation=0.3`) and a shared `size_frac`. Position `= Σ wᵢ·centroidᵢ`; object
side `= size_frac · L` where `L` = mean pairwise centroid distance (orientation-
invariant frame length). Because the weights and `size_frac` are shared across the
K+1 scenes while `centroids`/`L` are per-subject, both the anatomical position and
the apparent size (fraction of the anatomy) are consistent across target and
contexts — fixing the orientation-dependent `extent` and the FOV-dependent absolute
size. Anchors are landmarks only; label = the drawn object. Validation groups by
object **shape** (`label_name`; `anchor_shapes(cfg)`). New geometry helpers in
`draw.py` (`affine_weights`, `frame_length`, `barycentric_center`); `offset_to_center`
removed. New knobs: `n_anchors`, `extrapolation`, `weight_concentration`,
`max_select_tries`, `object_size_frac_min/max`, `object_size_min_vox` (replace
`offset_range`/`object_size_min`/`object_size_max_frac`). Design + plan:
docs/superpowers/specs/2026-07-22-anchor-synth3d-barycentric-positioning-design.md,
docs/superpowers/plans/2026-07-23-anchor-synth3d-barycentric-positioning.md.
Verification (val, real data): overall zero-rate **0.9%** (was ~11%); per-shape
occupancy median blob≈1810, elongated≈780, tubular≈150 vox @128³. Apparent size is
consistent by construction; residual within-task voxel-size CV≈0.14 is dominated by
the intentional `scale_jitter` (±15%) — lower `scale_jitter` for tighter consistency.

## anchor_synth3d: decouple object size from the anchor

Object size was tied to the anchor (`size = scale_frac * mean(anchor extent)`), so
small anchors (thin vessels, small glands) forced `size≈3` and produced empty /
near-empty labels. Root-caused via an occupancy probe: 13% of train targets (6% val,
no aug) had zero occupancy — from tubular shapes rendering below the `alpha>0.5`
label threshold at small sizes, plus the new nearest-interp aug erasing sub-~50-voxel
objects. Fix: the anchor now sets **position only**; object side is drawn
independently `~U[object_size_min, object_size_max_frac·min(image_size)]` (default
**20–51 vox** at 128³), shared across the K+1 scenes with per-scene `scale_jitter`.
Removed `scale_frac`. Config knobs `object_size_min` / `object_size_max_frac` replace
it (common.py, dataset3d.py, yaml, tests, plot/analyze captions updated). Also made
`build_dataset` read `cfg.get("augmentations")` so minimal cfgs (wiring test) don't
require the key. The resolved per-object shape (from `mix`) is recorded in the spec
and item `meta["shapes"]` for per-shape analysis. Zero-rate went 13.0→0.4% (train),
6.1→0.7% (val). Per-shape occupancy (median vox @128³): blob ~6000, elongated ~2500,
tubular ~360 — tubes stay much smaller (thin caliber), so raise tube radius if more
balance is wanted.

## anchor_synth3d: apply the multiverseg augmentations

`AnchorSynth3DICLDataset` fully overrides `__getitem__`, so the parent's aug hook
never ran and `build_dataset` never passed `aug_cfg` for `anchor_synth3d` — the
`augmentations: multiverseg` config was loaded but dead. Wired it through:
`build_dataset` now forwards `aug_cfg=(cfg.augmentations if is_train else None)`,
and the dataset applies shared geometric task aug over the K+1 scenes + independent
per-volume intensity aug (mirroring the real-data path; mask nearest-interp keeps
the int64 object ids). val/test remain un-augmented. Built-in per-scene
scale/rotation jitter is unchanged. Verified: aug wired on train (None on val), and
on a fixed deterministic scene the aug'd image/mask differ from the un-augmented one.
`plot_dataset_items.py` builds through `build_dataset`, so `--split train` now shows
augmented anchor_synth3d items; its caption gains a `+ aug` tag (was mislabelled "no
task-aug").

## PatchSet3D: profile a train step + avg_pool3d resample optimization
- Profiled one fwd+bwd step (B=1, K=1, 128³, R=16, full_attn, compiled transformer) on an
  A6000. GPU is already saturated (100% SM, ~memory-bandwidth bound). Breakdown of CUDA time:
  flash attention fwd+bwd ~48% (backward is 3–4× the forward), 3D conv encoder
  (conv+GroupNorm+LeakyReLU at 128³) ~25%, adaptive_avg_pool3d ~5%, elementwise tail the rest.
- **Batch size does NOT scale throughput**: measured 7.84→8.69 samp/s from B=1→B=8 (+11% only);
  peak mem 4.0→27.6 GB (B=16 OOMs on 49 GB). The workload saturates the GPU at B=1, so raise
  batch only for gradient quality, not speed. (config comment updated accordingly.)
- **Optimization applied** (`src/models/patchset3d.py`): new `_down_to(f, R)` replaces
  `adaptive_avg_pool3d` in `ConvEncoder3D._resample` and the p==1 occupancy pool. When the
  source side is an integer multiple of R (128/16, 64/16 here) it uses a strided
  `avg_pool3d(k, k)` — numerically identical (maxdiff 0) but ~3× faster incl. backward at large
  strides. Net step: 127.6→122.2 ms/it (compiled); ~18% under the original eager path.
- Larger levers are structural/modeling, not applied: `full_attn=False` (query→support only)
  is ~11% faster (122→109 ms/it) but changes attention semantics; smaller R cuts the R³ token
  count quadratically in attention.

## pfn_seg_2d: manual attention for the tiny feature-axis (set-of-patches speedup)
- Traced where the attention time actually goes (patchset3d, R=16, full_attn). Surprise: the big
  all-to-all set attention (b·c=2 seqs × r=8200 tokens) is CHEAP under flash (~7.5 ms/step fwd).
  The hog is the FEATURE-axis attention — in the set-of-patches layout that's a seqlen-2 attention
  (img+mask columns) batched over b·r=8200 patches × 4 heads. Flash launches ~33k microscopic
  2×2 attention problems with ~zero useful work, costing ~2.5× the real set attention (~18.6 ms/step
  fwd). `full_attn=True` was NOT the problem — it correctly uses flash; the seqlen-2 axis was.
- Fix (`src/models/pfn_seg_2d.py`): `_small_seq_attn` (plain q·kᵀ→fp32 softmax→·v) replaces the
  fused SDPA on the feature axis when `c <= _SMALL_SEQ_ATTN` (=16). ~3× faster incl. backward,
  numerically equivalent (end-to-end parity maxdiff 0.008, bf16 noise). Guarded so ImagePFN's large
  2N patch feature-axis still uses flash; the set-of-patches models (patchset_cnn, patchset3d) get
  the win. Compiles cleanly (pure tensor ops, unlike the flash library call).
- Measured (patchset3d compiled, B=1): 122.5 → 95.4 ms/it (~22%). Combined with the earlier compile
  + avg_pool3d work, ~33% under the original eager path (143 → 95 ms/it).

## 2026-07-22 — PatchSet3D: single-level 3D set-of-patches in-context segmentation
- feat(patchset3d): new `PatchSet3D` model (`src/models/patchset3d.py`) — single-level,
  dense R³ token grid, no refine stage. `ConvEncoder3D` downsamples each volume to an
  R³ feature map; every patch of every volume (target + K contexts) becomes a token;
  the dimension-agnostic dual-axis transformer (reused verbatim from `pfn_seg_2d`)
  does in-context matching over that set. Prediction at R³ or tiled to (R·d)³ via
  `mask_patch_decode_size`; trilinear upsample to native resolution in predict/eval.
- feat(fourier): generalized `FourierPositionalEncoding(n_axes)` (`src/models/patchset_pfn.py`)
  — default n_axes=2 unchanged; PatchSet3D passes n_axes=3 for (i,j,k) lattice position.
- feat(grid-metrics): new `experiments/3d/grid_metrics.py` — `hard_sum`/`soft_sum`/`cos_sum`
  accumulate `dice_ds`/`dice_ds_soft`/`cossim` at the token-grid resolution; logged as
  `train/dice_ds@{Rd}` and `val/dice_ds@{Rd}` (patchset3d only). `target_like` pools the
  native GT to the logit grid for the training loss.
- feat(config): `configs/experiment/3d/model/patchset3d.yaml` — Hydra model-group config;
  selects `model=patchset3d`, sets arch (R=16, enc_dims 4×32, e=256, h=512, l=6, a=4)
  and train recipe (AdamW, lr=3e-4, cosine, bce_dice). Run:
  `python experiments/3d/train.py model=patchset3d`.
- fix(evaluate): `experiments/3d/evaluate.py` — GT pooling for grid metrics now calls
  `.cpu()` before `F.adaptive_avg_pool3d` to guard against label arriving on GPU.
- fix(patchset3d): `ConvEncoder3D.cbr()` — `Conv3d(..., bias=False)` (bias redundant with
  GroupNorm affine; prevents dtype mismatch under bfloat16 autocast).
- fix(train): `experiments/3d/train.py` — `net.to(DEVICE)` for `patchset3d` after
  `build_model`; model was created on CPU and never moved to GPU.
- fix(dataloader): `src/totalseg_dataloader_incontext.py` — fall back to target self-context
  when no context candidates found (prevents empty-stack crash at `eval.n_subjects=2`).
- smoke: `python experiments/3d/train.py model=patchset3d train.epochs=1 arch.resolution=8
  arch.mask_patch_decode_size=2 data.max_ds_len_train=8 eval.n_subjects=2 train.batch_size=2
  wandb.project=null` — completed, `val_dice=0.0002` (epoch 0, random init, expected).
- Deferred: refine (bbox/scatter), sim_prior query seed, Muon/LAWA optimizer.

## 2026-07-21 — omnisynth3d: `synth3d.classes` accepts "benchmark" keyword
- feat(omnisynth3d): `synth3d.classes` in the 3D pipeline now flows through
  `resolve_classes` (data/totalseg_classes.py) at all three call sites that read it —
  `experiments/3d/common.py:build_dataset`, `train.py` (val-class listing) and
  `eval.py` — so it accepts the string keywords `"benchmark"` (→ BENCHMARK_CLASSES,
  47 classes) and `"not_benchmark"` in addition to an explicit list or `[]` (= all cached
  classes). Previously the train/eval bank builds passed the raw value, so `"benchmark"`
  was mis-parsed as `tuple("benchmark")` → empty class pool.
  `configs/.../experiment/1_medverse_benchmark.yaml` now sets `synth3d.classes: benchmark`.

## 2026-07-21 — 3D config layout: train/eval entrypoints + dataset/model groups
- refactor(3d-config): `configs/experiment/3d/` reorganized into `train.yaml`/`eval.yaml`
  entrypoints that compose `dataset/{totalseg,omnisynth3d}` + `model/medverse` groups,
  overridable by an `optional experiment:` preset. `hydra.searchpath`
  (`file://${oc.env:PWD}/configs`, run from repo root) keeps the global cluster→paths
  chain. The 3 hydra entrypoints (train.py/eval.py/plot_dataset_items.py) now point at
  the 3D config root. Old flat files removed; synth3d block de-duplicated (one source in
  dataset/omnisynth3d.yaml). New usage: `train.py dataset=omnisynth3d`,
  `eval.py dataset=omnisynth3d eval.split=val`. Spec: 2026-07-21-3d-config-layout-design.md.

## 2026-07-21 — omniSynth 3D (TotalSegmentator organs on a 3D canvas)
- feat(omnisynth3d): extends omniSynth to compose 3D in-context scenes by painting bbox-cropped
  TotalSegmentator organs at random 3D positions on a D×H×W canvas. Emits the
  `TotalSegInContextDataset` contract (image/label/context_in/context_out/subject/label_name/
  spacing), so the 3D pipeline + `incontext_collate_fn` consume it unchanged. Selected via
  `data.source=omnisynth3d` (config group `configs/experiment/3d/dataset/omnisynth3d.yaml`,
  selected via `dataset=omnisynth3d`).
- Free placement only, native (canvas-relative) organ sizes, contour-based compositing +
  anti-overlap (mask, never bbox). target_mode ∈ {identical, class} (aug/3D-rotation deferred).
- Dataloading-optimized: one-time build script `scripts/synth3d/build_totalseg_tiles.py` crops
  every organ into per-class fp16 `[2,T,T,T]` tile caches (`<root>/T{D}/{split}/class_{lv}.pkl`);
  `TotalSegObjectBank` reads them with an LRU — no full-volume reads/cropping in the hot path.
  Cubic-canvas only. New modules: bank_common3d, bank_totalseg, render3d, dataset3d (+config
  OmniTotalSegConfig). 2D omniSynth path untouched. New 3D tests 24/24; full suite 62.
  Spec: 2026-07-21-omnisynth-3d-design.md; plan: 2026-07-21-omnisynth-3d.md.
- Pre-training TODO: on the first real-data build, visually spot-check tile intensity range /
  mask coverage — CT is normalized as clip(≥0)/per-volume-max (validate HU/air handling).

## 2026-07-16 — sim_prior experiment config (6_sim_prior)
- feat(patchset): sim_prior — max-cosine similarity query prior (PFENet-style) seeding the query
  mask token; single-level, zero params, off by default. Config 6_sim_prior (A/B vs 5). Targets
  small-object needle-in-haystack. Spec: 2026-07-16-similarity-prior-query-seed-design.md

## 2026-07-15 — flops_giga is now per-sample (batch-size invariant)
- `validate()` (`experiments/2d/evaluate.py`) divided the FlopCounterMode total by `img.shape[0]`,
  so `flops_giga` is per-sample instead of per-batch. It was measured on one val batch, so it
  scaled linearly with `eval.batch_size` — a bs=64→128 change doubled it (~3700→7500) with no
  real model change. **Break in scale vs prior logged runs** (which were per-batch @ bs=64):
  new numbers are ~old/64 (≈58 GFLOP/sample for the scatter PatchSetCNN).

## 2026-07-14 — PatchSetCNN full (unmasked) sample-axis attention
- New `full_attn` flag (`PatchSetCNN.__init__`, default `False`). When set, the sample-axis
  drops the read-only mask entirely → dense `r×r` attention: every row (thinking + support +
  query) attends to every row, so context representations become target-aware. Breaks the
  "context is read-only" invariant intentionally; **no GT leak** (query rows still carry only
  the support-mean occupancy prior, never GT).
- `pfn_seg_2d.TransformerEncoderLayer/Stack.forward` gain `full_attn: bool = False`. The
  sample-axis branch is now 3-way: explicit `attn_mask` > `full_attn` (unmasked full `k,v`) >
  default read-only slice `k[:, :, :sep]`. Cost ≈0 — full drops the `masked_fill`/mask tensor
  and re-enables the fused (flash) SDPA kernel, so it is marginally cheaper + lower-memory.
- Precedence: `full_attn` supersets `query_self_attn`'s connectivity, so in `_attn_core` the
  mask is only built when `query_self_attn and not full_attn`. Threaded through `build_model`
  (`experiments/2d/train.py`) into the checkpoint `arch` and exposed in `model/patchset_cnn.yaml`
  (default `false`). Verified: full ≠ read-only output, full_attn wins over qsa, 35 tests pass.

## 2026-07-13 — scatter sampling diagnostic + tuned params
- `experiments/2d/multilevel/plot_scatter_sampling.py`: loads a scatter PatchSetCNN checkpoint,
  runs its COARSE pass on the training source (omnisynth_medseg), and sweeps `sample_patches` params
  over the real coarse map — prints behavior metrics (core@uncertainty, GT-boundary recall, cluster
  adjacency, coverage) + a tier-colored figure (diverse sources × configs).
- Finding on scarlet-disco-163 (scatter, epoch 11): GT-boundary recall already strong (84%→93% at
  n_total=1024); cluster↔explore is driven by `temperature`/`floor` (not blur_sigma at large fg_core);
  uncertainty focus was diluted by the big `n_fg_core=256`. Also fixed the eval top-left dump
  (always-stochastic seeded sampling + wider blur, commit 21841c1).
- Tuned `3_omnisynth_medseg_scatter.yaml` sample block: `n_fg_core 256→64`, `n_boundary_core 0→64`,
  `temperature 1→0.5`, `blur_sigma 4→3` → uncertainty focus 36%→39%, cluster 0.40→0.55, boundary
  recall 91% @ n_total=1024 (clustered on boundary/uncertainty while still exploring ~25% of the grid).

## 2026-07-13 — scatter refine qualitative figure
- `_refine_scatter` (`src/models/patchset_cnn.py`) now also returns the sampler tier flags —
  `refine_is_core`/`refine_is_fg` (B,M) for the query and `refine_sup_idx`/`refine_sup_is_core`/
  `refine_sup_is_fg` (B,K,M) for the support (previously discarded). Additive — loss/metrics ignore them.
- `evaluate.py`: `save_scatter_figure` (2×3 tier-colored panel — target sampled cells | coarse | fused,
  ctx0 support cells) + a `validate()` branch keyed on `refine_idx`, logged to `figures_scatter/…`.
  Cells map Rf-grid→pixels; tiers colored boundary=red / fg-core=orange / neighbor=cyan (as plot_sampling).
- Fills the gap where scatter eval figures previously showed only the coarse pred (the bbox refine panel
  is skipped for scatter). Tests: tier-key shapes + figure smoke test.

## 2026-07-13 — scatter refine mode: unconstrained scatter sampling + config + checkpoint persistence
- Added `refine_mode="scatter"` to `PatchSetCNN`: coarse@T=32 → scatter-refine@Rf=64 sampling M=256 cells per image via boundary/foreground-aware Gumbel-top-k (`src/models/scatter_sampling.py`); per-level losses (coarse BCE+Dice + refine on gathered cells); fused native prediction via `composite_predictions`; `build_model` persists `arch["sample"]` (plain dict via `OmegaConf.to_container`) so `eval.py` rebuilds with zero drift; config `configs/experiment/2d/3_omnisynth_medseg_scatter.yaml` launches the run.

## 2026-07-13 — patchset_cnn refine: log coarse-only counterfactual + fix exp10 version bug
- Goal: measure precisely where the added refine level (2-level patchset_cnn) contributes to
  the final prediction on medsegbench (run ttt6kmnk, crashed at epoch 67/500).
- `experiments/2d/evaluate.py`: `refine_geometry` now also returns `coarse_nat` (coarse prob
  upsampled to native — the refine-OFF counterfactual) and `coarse_R` (coarse pooled to Rf).
  `validate` logs per-sample `dice_coarse` (native) + `dice_coarse@{Rf}` and their summaries,
  so the exact refine delta is `dice − dice_coarse` at matched resolution.
- `results/experiments/10_patchset_cnn_2_lvls.py`: fixed `get_latest_table`, which selected the
  artifact by LEXICAL version max (`"v9" > "v67"`) and so analyzed a stale epoch-9 table; now
  parses the integer suffix. Added analysis cells (micro-vs-macro, coarse-conditioned collapse,
  per-dataset stage decomposition, direct refine delta when the counterfactual is present).
- Finding: the refine ENCODER helps on its crop (+0.05 macro over coarse) but the single-bbox
  STITCH gives it back (−0.065 macro; worst on thin/multi-region targets: drive/chasedb1 −0.21,
  m2caiseg −0.10, dynamicnuclear −0.09). Net final native ≈ coarse. It cannot rescue coarse
  misses (coarse@32==0 → 86% still empty). Micro metric is a 42%-weight m2caiseg thermometer.

## 2026-07-13 — thor node: cu121 venv for RTX A6000 (driver 12.3)
- thor = RTX A6000 (Ampere sm_86), driver 545.29.02 → CUDA 12.3. The cu124 torch build
  (used on nero) is one driver-minor too new: torch warns "driver too old (found 12030)"
  and CUDA falls back to unavailable.
- Added a `cu121` extra to pyproject.toml (optional-dep + `pytorch-cu121` index + torch/
  torchvision sources + conflicts entry), mirroring the existing cu124/cu128/cpu pattern.
- Created env: `UV_PROJECT_ENVIRONMENT=.venv_thor uv sync --extra cu121` → torch 2.5.1+cu121,
  torchvision 0.20.1+cu121. Verified `torch.cuda.is_available()` + GPU matmul on the A6000.
- Run on thor with `.venv_thor/bin/python`.

## 2026-07-12 — preview_omnisynth: Hydra config like train.py
- Rewrote experiments/2d/synth/preview_omnisynth.py from argparse (--config <yaml>) to
  `@hydra.main` so it takes an experiment `--config-name` (e.g. 2_omnisynth_medseg_refine),
  composing the full training config chain.
- Builds the dataset via `common.build_dataset(cfg, split)` — the exact path the trainer
  uses — so the preview matches training (source, backgrounds, instCopy, cfg.paths.omniglot).
- Old flags are now native Hydra overrides: `--mode class` -> `synth.scene.target_mode=class`,
  `--context_size 1` -> `data.context_size=1`, `--image_size` -> `data.image_size`. Only
  split/n/out remain preview-specific, appended as `+preview.split=val +preview.n=6 +preview.out=...`.
- out paths resolve against repo root (robust to Hydra cwd). Verified: renders exp-2's
  medseg-on-image distribution to a PNG.
- Added `+preview.augment=true`: loads the experiment's aug_preset (cfg.aug_preset merged
  with cfg.aug) and runs each item through pfn_train.augment (mirroring train.py._augment_batch)
  — contexts get geometric+intensity, query image gets task-intensity only, query mask
  untouched. Lets the preview match what the model sees when a config sets `augment: true`.

## 2026-07-10 — train: opt-in batch augmentation for the unified 2D trainer

- **Batch augmentation is opt-in via `augment: true/false` config flag** (default off in
  `train_base.yaml`). When enabled, the unified `experiments/2d/train.py` applies per-batch
  augmentation via the existing `_augment_batch` helper during training, loading the
  `configs/augmentations/<aug_preset>` preset and gating on `cfg.get("augment", False)`
  in `train_epoch`. Only contexts are geometrically transformed; query (target) image is
  never augmented (it receives intensity transforms only via the `task`/`aug_preset` knobs).
  The default `augment: false` preserves all existing run behavior; `augment: true` opts in.
- Config changes: `train_base.yaml` now defines `augment: false` as a top-level key
  (sibling to `arch:/train:/eval:`) with a clarified comment; `2_omnisynth_medseg_refine.yaml`
  adds `augment: true` to enable batch aug for that experiment.
- Tests: two new config-level tests in `tests/test_train_augment.py`:
  `test_train_base_augment_defaults_false` (checks default is off) and
  `test_omnisynth_refine_opts_in` (checks omniSynth refine experiment is on).

## 2026-07-10 — refine: `arch.refine_memory` flag (cross-level memory token)

- **Opt-in cross-level memory for the refine pass: `arch.refine_memory` (default off).** When enabled
  in multi-resolution mode, the refine pass prepends a learnable memory token (type-only adapter,
  `mem_type`, no projection) to the sequence before attention. The memory attends to the detached
  coarse pass's thinking rows (`mem_type` + coarse thinking, `.detach()`), allowing the refine
  pass to selectively condition on coarse-level reasoning without backprop through the coarse
  encoder. Mirrors `multilevel/`'s `stage1_think` design but with shared weights.
- Wiring: `PatchSetCNN(..., refine_memory=...)` constructor (Task 1); `_attn(..., mem=...,
  return_think=False)` dual-purpose signature (Task 2 — produces thinking for refine, consumes
  memory token for coarse); config flag + checkpoint rebuild (Task 3 — `build_model` adds to
  `arch` dict, `train_base.yaml` exposes `arch.refine_memory: false`). Single-level mode is inert
  (constructor-only param, no refine forward).
- Tests: round-trip checkpoint via `build_model`, wiring, gradient flow to `mem_type`, detach
  verification. Both refine modes (`reencode`/`encode_once`) supported.

## 2026-07-10 — train: fix compile flag namespace mismatch (`train.compile` → `arch.compile`)

- The compile knob was **still dead**: the config defines it at `arch.compile` (`train_base.yaml:23`)
  but `train.py` read `cfg.train.get("compile", False)` — a key that exists nowhere, so it always
  resolved `False` and the model ran eager regardless of `arch.compile=true`. `pfn_seg.py:327` reads
  the correct `cfg.arch.compile`; `train.py` had drifted. Now reads `cfg.arch.get("compile", False)`.
- Also ported two things from `pfn_seg.py`: (1) compile Muon's `_newtonschulz5_batched` (pure tensor
  ops) alongside `model.transformer`, and (2) the node-local triton/inductor cache guard
  (`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` on `/tmp` keyed by hostname) to avoid the NFS GLIBC
  cache-poisoning issue. Scope is unchanged: only `model.transformer` + Newton–Schulz compile; the
  encoder/bbox crop/pool stay eager (graph-break).

## 2026-07-10 — train: wire the (previously dead) `train.compile` flag for patchset_cnn

- `experiments/2d/train.py` defined `train.compile` (via `train_base.yaml`) but never read it — the
  model always ran eager. Now, when `train.compile` is set, it compiles **only** `model.transformer`
  (`torch.compile(..., dynamic=True)`), mirroring `multilevel/train.py` which compiles just its
  trainable PatchSetPFNs. The transformer is pure tensor ops so it graph-compiles cleanly; the
  encoder + bbox crop/pool (`grid_sample`, `adaptive_avg_pool2d` with data-dependent windows) stay
  eager (they graph-break). UniverSeg has no `.transformer` so it is untouched (`hasattr` guard).
- Checkpoint reload made robust to the mid-key `_orig_mod.` prefix that submodule-compile leaves
  (`transformer._orig_mod.…`): warm-start (`train.py`) and `eval_incontext.py` now strip with
  `.replace("_orig_mod.", "")` instead of `removeprefix` (leading-only). Verified the stripped keys
  strict-load into a fresh model and the Muon `"transformer" in n` filter still selects the 2D
  matrices post-compile. (Local `torch.compile` couldn't be exercised — broken `torch._inductor`
  import in this env; runs on the GPU cluster where multilevel already compiles.)

## 2026-07-10 — refine: `arch.refine_mode` flag (reencode | encode_once)

- **New prototype for the 2-level refine pass, behind `arch.refine_mode`** (`patchset_cnn.py`,
  default `reencode` = unchanged behavior):
  - `reencode` (old): re-run the whole model (encoder + attention) on the upsampled crop — every
    stage recomputed at the finer scale. 2× encoder passes.
  - `encode_once` (new, à la `experiments/2d/multilevel/zoom_pipeline`): encode all K+1 images
    ONCE into native multi-scale maps, then run the attention half twice — on the full pooled
    features (coarse) and on the SAME maps cropped to each bbox and pooled back to the T grid
    (refine). ~half the encoder compute; grid_sample crop is differentiable so grad still reaches
    the encoder from the refine loss. Tradeoff: the refine pass reuses stem/shallow detail but
    does NOT recompute deep-stage semantics at the crop scale.
- Refactor enabling this: `ConvEncoder.forward` split into `encode_maps` (native multi-scale list)
  + `pool_maps`; `_segment` split into `_grid_tokens`/`_occupancy` feature build + a shared `_attn`
  (encoder-independent attention half, now reused by the single-level model and both refine passes);
  bbox selection factored into `_select_bbox`. New `bbox_refine.crop_pool_maps` crops native maps
  (origin/s rescaled per map resolution) and pools to the token grid.
- Verified: coarse head identical across modes; encoder stem runs 2× (reencode) vs 1× (encode_once)
  on B·T images; grad reaches encoder+decoder from both heads. Tests in
  `tests/test_patchset_cnn_refine.py` (12 passed). Wire via `arch.refine_mode` in the experiment cfg.

## 2026-07-10 — metrics: tag cossim/top-k with the token grid; resolution-tagged sample table

- **`cossim` and `top{k}` now carry their resolution**: logged as `cossim@{T}` / `top{k}@{T}`
  (val, `evaluate.py`) and `train/cossim@{T}` / `train/top{k}@{T}` (train), where T = the coarse
  token grid (`low_res`, e.g. 32). They were always computed on that grid but were previously
  untagged. `_select_metric` (`train.py`) and the train console top-k lookup were updated to
  find the tagged keys; the cossim fallback now matches `cossim@*`.
- **Per-sample wandb `val/samples` table columns are now resolution-tagged and model-adaptive**
  (built lazily per run in `validate()`, replacing the static `SAMPLE_COLS`):
  - always: `dice` (native hard — fused pred for refine models);
  - non-native (patchset/refine): `dice_ds@{T}`, `dice_ds_soft@{T}` (coarse grid);
  - refine: `dice@{Rf}`, `dice_soft@{Rf}`, `dice_fused@{Rf}` (per-sample, previously only
    aggregated). Native (UniverSeg) tables now carry just `dice` (no meaningless NaN columns).

## 2026-07-10 — metrics: `ds_metric_res` (@R pooled Dice) is now UniverSeg-only

- The fixed-resolution `dice_ds@R` / `dice_ds_soft@R` metrics (from `eval.ds_metric_res`) are
  now computed **only for native models (UniverSeg)**, in both `validate()` (`evaluate.py`) and
  `train_epoch` (`train.py`). Their purpose is to pool UniverSeg's native prediction to R×R so
  it is comparable to patchset_cnn's coarse grid. For non-native patchset_cnn/refine those
  `@R` metrics were always computed on the *coarse-upsampled* pred — confusing and redundant
  with the model's own `dice_ds@{token grid}` — so `ds_metric_res` is now ignored for them.
- Removed the dead `eval.ds_metric_res` from `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`
  (patchset/refine); it lives in `configs/experiment/2d/model/universeg.yaml` (`[32]`), the
  correct home now that it is UniverSeg-only. patchset_cnn/refine still report their native
  coarse grid as `dice_ds@{low_res}` / `dice_ds_soft@{low_res}`.

## 2026-07-10 — refine: `dice` scored on the fused prediction + checkpoint selects on it

- For multi-resolution refine `PatchSetCNN` models, the headline **`dice`** metric (native
  hard Dice) is now computed on the **fused** prediction of the last level (`rg["fused"]`, at
  full H×W) instead of the coarse `final_logit`. Applies in both `validate()` (`evaluate.py`)
  and `train_epoch` (`train.py`, `train/dice`). Non-refine models (UniverSeg, plain
  patchset_cnn) are unchanged.
- **Checkpoint selection** (`_select_metric`, `train.py`) for refine models now selects on
  native `dice` (the fused full-res hard Dice) rather than `dice_fused@{last res}`. Non-refine
  selection (cossim → dice) is unchanged.
- Scoped step-by-step: only `dice` moved to the fused pred for now. `dice_ds@R` /
  `dice_ds_soft@R`, cossim, and top-k still reflect the coarse pass; `dice_fused@R` /
  `dice@R` per-level metrics remain as diagnostics.

## 2026-07-10 — eval: coarse→fine refine qualitative figure (save_refine_figure)

- **`save_refine_figure`** added to `experiments/2d/evaluate.py`: 2×3 PNG panel showing the coarse→fine refinement for a refine `PatchSetCNN` checkpoint. Row 0 = target, row 1 = first context. Col 0: full frame + GT contour + res0 (coarse) pred heatmap + bbox rectangle. Col 1: bbox crop + GT contour + res1 (refine) pred heatmap. Col 2: full frame + GT contour + fused (place_window stitch) pred. All overlaid with lime GT contours and yellow/cyan bbox rectangles.
- **`_refine_overlay_ax`** helper: gray base + optional Reds heatmap (with optional extent for T×T→crop stretch) + optional lime GT contour + bbox `Rectangle` patches. `from matplotlib.patches import Rectangle` added to the imports.
- **Wired into `validate()`**: inside the gated `if figures and fig_key not in saved` block, immediately after `save_figure`, a `if rg is not None` branch writes `{ds}_l{lv}_refine.png` alongside the standard `{ds}_l{lv}.png` panel; optionally logs to `figures_refine/{ds}/label_{lv}` in wandb. Controlled by the existing `eval.save_figures` / `eval.max_figures` gates. Plain checkpoints (`rg is None`) are unaffected.
- **Verified**: unit tests (`tests/test_refine_figure.py`, 2 tests) pass; 1-epoch CPU smoke (omniSynth, bs=2, 8 samples, size=64) trains to `best.pt`, eval re-loads and emits `dice/mean=0.0308` + 4 `*_refine.png` panels in the eval out_dir alongside the standard `*.png` panels.

## 2026-07-10 — Config + docs: multi-resolution refine (arch.resolutions) wired end-to-end

- **`PatchSetCNN` refine reworked to multi-resolution per-level** (`src/models/patchset_cnn.py`, `src/models/bbox_refine.py`): constructor arg is now `resolutions: list[int]` (effective full-image resolutions), replacing the old `refine: bool` / `refine_crop: int` flags. Token count is constant at T = `resolutions[0]`² tokens; the refine crop is derived as `image_size · R0 / Rk` (e.g. 128 · 32/64 = 64 px for `resolutions=[32,64]`). Per-level losses: coarse `@resolutions[0]` + one refine loss per additional level, weighted by `train.refine_loss_weight`. Old `refine`/`refine_crop` arch flags removed from `configs/experiment/2d/model/patchset_cnn.yaml`.
- **`place_window` (replace-stitch)** added to `src/models/bbox_refine.py`: writes the refine crop back into the native-resolution canvas by replace (not additive). Used exclusively inside the `refine_geometry` helper in `experiments/2d/evaluate.py` for metric assembly; no forward-path fusion in the model.
- **`refine_geometry` helper** in `experiments/2d/evaluate.py`: assembles per-level logits into the fused canvas for metric-only reporting. `fuse_window` (additive, logit-space) remains in `bbox_refine` unused, reserved for a future fused loss.
- **Metrics**: `dice@{Rk}` / `dice_soft@{Rk}` per refine level logged in train and val; `dice_fused@{Rk}` (hard and soft) reported via `refine_geometry`. Checkpoint selection is on `dice_fused@{resolutions[-1]}`.
- **Bbox origins are detached** (`argmax` on `sigmoid(coarse).detach()`); gradients flow only through the two `_segment` calls.
- **New experiment leaf** `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`: rewritten to set `arch.resolutions: [32, 64]`, `train.refine_loss_weight: 1.0`, and `eval.ds_metric_res: [16, 32]`. Inherits `1_omnisynth_medseg` via defaults.
- **Old checkpoints**: any `best.pt` carrying `arch.refine` / `arch.refine_crop` (the previous additive-fusion design) will not rebuild under the new constructor — they are throwaway smoke artifacts from the prior plan; retrain.
- **Verified**: dry-run `--cfg job` confirms `model: patchset_cnn`, `resolutions: [32, 64]`, `refine_loss_weight: 1.0`, `ds_metric_res: [16, 32]` all compose. 1-epoch smoke train (64 samples, bs=4) completes without shape errors, logs `val dice_fused@64=...` in the console, and saves `best.pt` at `…/2026-07-10_dummy-dqibaa6e/best.pt`. `eval_incontext.py` reloads and prints `Loaded patchset_cnn (size=128, ctx=1)` with no rebuild error — `arch.resolutions` round-tripped correctly through the checkpoint.

## 2026-07-09 — PatchSetCNN refine mode wired into config + trainer

- **New `PatchSetCNN` mode** (`src/models/patchset_cnn.py`, `src/models/bbox_refine.py`): constructor args `refine: bool` (default `False`) and `refine_crop: int` (default `64`). When `refine=True`, after the coarse 32×32 pass the model crops a `refine_crop × refine_crop` bbox around the predicted foreground (detached `argmax`), runs a second shared-weight `_segment` forward at native resolution inside that crop, and additively fuses the upsampled crop logit back onto the coarse logit → `final_logit` at native `(B,1,H,W)` resolution. When `refine=False`, `final_logit` is the coarse `(B,1,R,R)` as before. Bbox ops live in `src/models/bbox_refine.py`; both modules committed and tested (8 tests pass).
- **`build_model` threading** (`experiments/2d/train.py`): the `patchset_cnn` branch's `arch` dict now includes `"refine": a.get("refine", False)` and `"refine_crop": a.get("refine_crop", 64)`. These are passed to `PatchSetCNN(image_size=..., **arch)` and saved verbatim in `best.pt` so `eval_incontext.py` rebuilds the refine model with zero drift. No train/eval loop changes — the loop already pools GT to `final_logit`'s spatial size; when `refine=True` that size is native `H×W` so the loss and all Dice metrics land at native resolution.
- **Config group default** (`configs/experiment/2d/model/patchset_cnn.yaml`): added `refine: false` and `refine_crop: 64` under `arch:` (backward-compatible; off by default so all existing runs are unaffected).
- **New runnable leaf** (`configs/experiment/2d/2_omnisynth_medseg_refine.yaml`): inherits `1_omnisynth_medseg` via `defaults`, flips `arch.refine: true`, and sets `eval.ds_metric_res: [16, 32]` to preserve coarse-grid Dice alongside the new native Dice. Run with `python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine`; override window with `arch.refine_crop=48` etc.
- **Checkpoint metric**: with `refine=True` the model predicts at native resolution, so `final_logit.shape == lbl.shape` — the loop's native-vs-coarse branch selects `metric="dice"` (not `"cossim"`), and the checkpoint is selected on hard Dice. `ds_metric_res=[16,32]` in the new leaf preserves pooled coarse Dice for comparison.
- **Verified**: dry-run `--cfg job` confirms `model: patchset_cnn`, `refine: true`, `refine_crop: 64`, `resolution: 32`, `ds_metric_res: [16, 32]` all present. 1-epoch smoke train (64 samples, bs=4) completes without shape errors and saves `best.pt`. `eval_incontext.py` reloads and prints `Loaded patchset_cnn (size=128, ctx=1)` with no rebuild error — `arch.refine/refine_crop` round-tripped correctly through the checkpoint.

- 2D trainer: renamed the universeg train metric `train/dice_ds_soft` → `train/dice_soft` (`train.py:331`). It's soft Dice at **native full resolution**, so the `ds` (=downsampled) prefix was a misnomer — it now parallels `train/dice` (native hard) and matches `pfn_seg.py`'s existing `train/dice_soft` (= soft Dice at the model's native pred resolution). Symmetry: universeg logs `dice`/`dice_soft` (native hard/soft); patchset_cnn logs `dice_ds@R`/`dice_ds_soft@R` (coarse hard/soft, correctly `@`-tagged). Scope was train-side only: `evaluate.py`'s `dice_ds_soft@R`/`dice_ds_soft@{low_res}` summaries, the `SAMPLE_COLS` `dice_ds_soft` column, and the `train.py:354` val-summary lookup all refer to genuinely-downsampled values and were left unchanged. Caveat: renames a logged wandb key, so old runs won't align on it.
- 2D configs: new `model/` Hydra group (`configs/experiment/2d/model/{patchset_cnn,universeg}.yaml`, each `# @package _global_` on line 1 so it sets the top-level `model:` string plus that model's `arch`/`train`/`eval` blocks). This makes `model=patchset_cnn|universeg` a real group override. New runnable leaf `1_omnisynth_medseg.yaml` (defaults: `omnisynth_train_base`, `model: patchset_cnn`, `_self_`) bakes the omniSynth-medseg-scene params held fixed across the sweep — `data.context_size=1`, `arch.resolution=32`, `train.topk_k=64`, `synth.source=medseg`, `synth.scene.{p_copy=0,placement=random,max_nb_objects=8,background=image}` — and runs either model: `python experiments/2d/train.py --config-name 1_omnisynth_medseg model=patchset_cnn|universeg <cli overrides>`. Existing `patchset_cnn_train.yaml`/`universeg_train.yaml` thinned to consume the same `model/` group (single source of truth for model defaults; behavior byte-for-byte unchanged — verified via `--cfg job`). Gotcha: the `@package _global_` directive MUST be the first line; a prose `# @package _global_ ...` comment elsewhere in the header is ignored and the group nests under `model.*` instead (caught in verification: `model:` resolved to a mapping and `lr` stayed at the base default).
- 2D trainer: qualitative per-sample val log (ported from the old universeg_train.py into the unified train.py). `validate` now builds a `wandb.Table` (`SAMPLE_COLS`: epoch, dataset, character, target_mode, target_pos, context_pos, transforms, dice) logged as `val/samples`; `_fmt_transforms` renders each aug placement as `r..,s..,dx..,dy..` (`-` / empty for identical/class). Needs per-sample provenance, so: `render_scene` now returns a 4th `info` dict (`target_cells` indices + aligned `target_transforms`) and `affine_jitter` returns `(bitmap, params)` with the sampled rotate/scale/dx/dy; `OmniSynthICLDataset` meta gains `target_cells`/`context_cells` as **(row, col) grid positions** (not flat indices — `divmod(cell, grid)`) plus `target_transforms`. instCopy (`is_copy`/`copy_slot`) kept alongside; copied context slots correctly report the query's cells. test_render.py updated for the 4-tuple/2-tuple returns. Verified: render tests pass, 1-epoch smoke run (train→val→table→ckpt) clean, copy vs non-copy context positions correct. The old train-Dice-monitoring part of universeg_train.py was NOT ported (train.py already has it, generalised with cossim/top-k).
- 2D trainer: added top-k patch-recall metric (`top{k}`, k=`cfg.train.topk_k` default 16). Per sample, recall of the GT-positive patches within the k highest-valued predicted patches: |gt_pos ∩ topk(pred)| / |gt_pos|, where gt_pos = patches with GT>0 capped to k. Reaches 1.0 exactly when ALL true patches are in the model's top-k — denominator is the (sparse) positive count, not k, so a model that found every true patch isn't penalised for GT having < k positives. Purely rank-based on the pred side (threshold/scale-free), complements cossim. Low-res only: gated on the same `logit.shape != lbl.shape` condition as cossim (at native res a patch is a pixel), so it never fires for UniverSeg. `common.topk_overlap` (per-sample) + train `_topk_sum` (batched, on-device scatter/gather); logged for train+val (`train/top{k}`, `top{k}/*`). Verified batched==per-sample, perfect ranking=1.0 (incl. n_pos>k rows), miss-1-of-5=0.8, empty-GT rows skipped. Caveat: when patch count ≤ k (e.g. R=4 → 16 patches, k=16) pred top-k = all patches → recall trivially 1.0; only informative when R×R > k (e.g. R=16 → 256).
- 2D trainer: log the full resolved Hydra config to wandb (`OmegaConf.to_container(cfg, resolve=True)`) instead of a hand-picked flat dict. All config (train.*, data.*, synth.*, arch.*, ...) is now captured as the single source of truth; removed the flat overlay + `scene`/`target_mode` helper vars that only fed it (no retro-compat kept). `ckpt_meta` unchanged — still saved into the checkpoint. Note: wandb keys are now nested (e.g. `train.lr`, `data.image_size`) rather than flat (`lr`, `image_size`); update any saved wandb filters/groupings accordingly.
- 2D trainer: skip cossim at native prediction resolution. cossim only exists because at low logit res the soft/hard Dice targets collapse; when the logit is already at native res (UniverSeg → logit==GT res) it's redundant with Dice. `train_epoch`/`validate` now compute cossim only when `logit.shape[-2:] != lbl.shape[-2:]` (validate returns empty cos dicts). Checkpoint metric follows: `cossim` at low res, native hard `dice` otherwise (`metric` var; drives `val/best_{metric}` and the "Done." print). train/cossim + train postfix `cos` only shown when computed. So UniverSeg (native) now checkpoints on hard Dice; patchset (low-res) unchanged.
- PatchSetCNN: added optional per-context-image identity embedding (`context_id_embed`, default False; `max_context`=16). A learned tag per context-image slot is added to all N patches of that image (both img/mask cols, image-major order); the target image gets its own learned `qry_id` tag. Previously the support was a fully permutation-invariant patch set with only within-image (i,j) position — so two context images with identical content (e.g. instCopy duplicates) produced byte-identical tokens and were indistinguishable to attention. Now they can be grouped/told apart. Verified: off→identical imgs give identical tokens (0.0), on→differ by the tag (0.088). Init: ctx_id/qry_id at σ=0.1 (small-norm rich-regime sweet spot for Adam per Ito et al. 2025 on learnable PE init; nn.Embedding default σ=1 = lazy regime, σ≲0.05 also hurts under Adam). Fourier (i,j) PE left as-is (fixed 2D sinusoid = the paper's best-performing "ground-truth" encoding for known grid structure). Wired through train.build_model + ckpt meta; enabled in patchset_cnn_train.yaml. No eval.py change (no patchset_cnn eval path yet).
- 2D trainer: added cosine-similarity metric (`cossim`) and made it the checkpoint-selection metric. Diagnosed why low-res patchset training shows near-zero dice: the loss/metric target is avg-pooled soft occupancy, and at R=4 a thin Omniglot glyph fills only ~3-9% of a 32px cell, so `dice_ds_soft` is pinned at its mean-occupancy ceiling (~0.05) and native hard `dice` is structurally 0 (prob fit to ~0.03 occupancy never reaches 0.5) — even for a PERFECT predictor. Cosine `Σ(p·g)/(‖p‖‖g‖)` is scale-invariant → real 0→1 signal (verified: perfect=1.0, untrained~0.24, converged copy-task=1.0), no threshold/hyperparam. common.cosine_sim + train `_cos_sum`; logged for train+val (`train/cossim`, `cossim/*`); dice metrics still logged. The copy task itself was learning fine (BCE 0.69→0.013 in ~50 steps) — only the metric was misleading.
- PatchSetCNN: decoupled token resolution R from enc_dims. ConvEncoder depth is now len(dims)-1 (architecture), producing features at the encoder's natural res H/2**(len-1); each scale is then resampled (area-pool down / bilinear up) to R×R. R no longer constrains enc_dims length or requires image_size = R × pow2. Default (128, R16, 4 dims) unchanged (resample is identity). ConvEncoder no longer takes image_size.
- omniSynth: instCopy copy-tasks — OmniSceneConfig.p_copy (default 0.9) / n_copy (default 1); train-only, isolated rng; copy slot(s) are an EXACT copy of the query scene; meta gains is_copy/copy_slot. Eval byte-identical.

## 2026-06-30 — BiomedParseDataset init: skip the ~1.8h NFS glob/stat via the store index
- Symptom: `eval.py data.source=biomedparse` "dataloading takes forever" — init never
  reached the eval loop. Cause was entirely in `BiomedParseDataset.__init__`, not decode:
  it rebuilt the sample index by globbing the raw PNG tree (`_discover_sources` at depths
  1–4 = 67s) and calling `os.path.exists` on the sibling image of every one of ~179k masks.
  Measured ~37ms/stat over this NFS mount → ~6600s (~1.8h) of pure stat() before iteration.
- The pre-resized store already ships a per-dataset `index_{S}.npz` (written by
  `scripts/datasets/biomedparse/to_npz.py`) recording image/mask paths relative to
  data_root in the exact discovery order. The fast pixel-path read those stores but the
  index was still rebuilt from scratch every run.
- Fix (`src/datasets/biomedparse.py`): when `use_npy`, discover datasets from the store
  tree (`_discover_stores`: ~41 dirs under `_npy/<split>`) and rebuild each sample index
  from its `index_{S}.npz` (`_index_from_npz`) — zero PNG globs, zero per-mask stats. The
  raw PNG-tree scan (`_discover_sources` + per-mask `os.path.exists`) is kept as the
  fallback when `use_npy=False` or no store exists for the requested size.
- Result: test-split init 179,284 samples / 41 datasets in **35.6s** (was >900s, timed out),
  all on the npy fast-path. Sample count + ordering match the PNG path (to_npz uses the
  identical `_collect` discovery, so `index_{S}.npz` rows align with the image/mask stacks).
  Self-test (PNG fallback) still passes; item shapes/ranges/binary masks spot-checked.

## 2026-06-29 — omniSynth: in-context Omniglot grid dataset wired into 2D pipeline
- New dataset `src/datasets/omniSynth/`: in-context 2D segmentation from Omniglot characters
  arranged on a grid. A task = one target character class; each scene is a 4x4 grid where
  k cells contain the target (mask = ink pixels of the target characters within those cells)
  and the remaining cells hold distractor characters from other classes.
- Wired as `data.source=omnisynth` in `experiments/2d/common.py:build_dataset`.
- Hydra config: `configs/experiment/2d/synth/omniglot.yaml` (under `@package synth`).
  Key knobs: `target_mode` (identical | aug | class — controls what "same target" means),
  `k_min`/`k_max` (number of target cells per scene), train=background alphabets,
  val/test=evaluation alphabets (split by `val_test_split`).
- `paths.omniglot` added to `configs/config.yaml` and both cluster configs
  (`configs/cluster/nfs.yaml`, `configs/cluster/meta.yaml`).
- Integration test: `src/datasets/omniSynth/test_integration.py` (run directly, no pytest).

## 2026-06-28 — OOD generalization study of imagepfn_zoom (hard_diverse checkpoint)
- Probed how the `2026-06-22_kind-durian-59` imagepfn_zoom checkpoint (trained on
  `synth=hard_diverse`) generalizes outside its training distribution, on the held-out
  val shape pool. In-dist anchor Dice = 0.648.
- New eval configs: `configs/experiment/2d/synth/{ood_appearance,ood_conditions}.yaml`.
  Appearance-only OOD (noise/contrast/texture pushed OOD, context correspondence kept
  in-dist so ctx_dice unchanged) → Dice 0.648→0.261: a CLEAN −0.39 generalization gap.
  Combined OOD → 0.071 but ctx_dice collapsed 0.164→0.045, so partly an artifact (no
  usable context), not pure generalization.
- Per-axis sweep (`results/controlsynth/sweep/run_sweep.sh`, 50 runs, num_tasks=2000):
  each knob varied around its single training point. Used realized ctx_dice to split
  info-preserving axes (drop = model brittleness) from ctx_dice-falling axes (drop partly
  inherent). Findings: genuine weaknesses are appearance knobs only —
  foreground_contrast (−0.30) ≫ noise (−0.19) ≈ texture (−0.14); support_query_scale is
  robust to 2.5× trained; region_size extrapolates UPWARD better than in-range; ambiguity
  axes handled gracefully. No catastrophic overfit (easy-side extrapolation flat-or-better).
- Recommendation: randomize live appearance knobs during training (currently pinned to
  single scalars in hard_diverse). Full writeup + figures: `results/controlsynth/sweep/SUMMARY.md`.

## 2026-06-24 — BiomedParse dataloading: benchmark + pre-resize pipeline
- Benchmarked src/datasets/biomedparse.py loading. Bottleneck is purely PNG decode:
  every image/mask is a 1024x1024 RGBA PNG; decode->128 grayscale = ~33 ms/img, all CPU
  (zlib inflate + RGBA->L; resize ~2 ms, NFS negligible — cold≈warm, header-open 0.36 ms).
  Each __getitem__ does up to 2*(K+1)=8 decodes. Real shuffled DataLoader throughput:
  2.7 / 22 / 33 / 104 img/s at num_workers 0 / 8 / 16 / 32 — fully decode-bound, scales
  linearly with cores. Corpus = 1.23M PNGs (MSD 605k, amos22 240k dominate).
- Fix: decode once, store pre-resized uint8. memmap'd .npy reload = 0.043 ms/img (~760x).
  At 128px the whole corpus is ~20 GB (fits the 995 GB RAM / OS page cache); one-time
  convert ≈ 20 min on 32 cores.
- New scripts/datasets/biomedparse/to_npz.py: parallel PNG->uint8, per-dataset memmap-able
  images_{S}.npy / masks_{S}.npy stacks + index_{S}.npz (paths rel to data_root). NOT a
  monolithic .npz like totalseg2d's — NpzFile can't be memmapped, so 32 persistent workers
  would each hold a full RAM copy; standalone .npy lets them share one OS-cached copy
  (COW-safe). Row order + resize semantics reproduce the dataset exactly (image L/BILINEAR/
  /255; mask L/NEAREST/>0); verified bit-exact (max diff 0.0) and row==image_idx aligned.
- BiomedParseDataset memmap fast-path (done): new use_npy/npy_root args; per dataset,
  _maybe_load_store() memmaps images_{S}.npy/masks_{S}.npy from <data_root>/_npy and uses
  them iff shapes match the discovered (n_imgs, n_masks) — else that dataset silently
  decodes PNGs (per-dataset fallback). Rows align by construction (same sorted-glob order):
  image row == img_idx, mask row == per-ds kept-mask counter (self._mask_row, populated
  beside self.mask_path, so no new COW-heavy dict). Verified bit-exact vs decode (max diff
  0.0 over images/labels/context). Throughput on ACDC+BreastUS+amos22 @128/K=3: 142 -> 0.20
  ms/item single-process (~700x); DataLoader now FASTEST at num_workers=0 (2822 img/s) and
  SLOWER with workers (1545 @16) — load is no longer the bottleneck, so worker IPC dominates
  (revisit worker count once heavy augmentation is in the pipeline). Store built at 128px
  for both splits (~5 GB, 41 dataset sources each).

## 2026-06-24 — eval.py: imagepfn_zoom backend
- experiments/2d/eval.py now evaluates zoom-chain checkpoints (model=imagepfn_zoom),
  mirroring the patchset_pfn (is_multilevel) backend: reads arch+sample+stage1 path from
  the best.pt, loads the frozen stage-1 ImagePFN + UniverSeg encoder, rebuilds the
  warm-started external-features ImagePFN hops (new helper build_zoom_chain — same shapes
  as train.py:build_zoom_models, no warm-start since trained weights are loaded), and
  drives them with run_zoom_chain. Prediction = final hop's refined_full (already native
  H, no upsample); scored with the same hard/soft Dice. Figure low-res panel uses the
  stage-1 coarse seed (R0×R0). Added "imagepfn_zoom" to the top-of-file src-precedence
  argv gate. Run: python experiments/2d/eval.py model=imagepfn_zoom eval.checkpoint=<best.pt>

## 2026-06-22 — Zoom-refinement variant (shared ImagePFN arch)
- New stage-2 refinement path selectable via arch.refine_arch=imagepfn_zoom (default
  patchset, unchanged). Instead of PatchSetPFN on scattered patches, a warm-started
  ImagePFN refines a contiguous square crop ("zoom"), so the refiner is the SAME arch as
  stage-1 — isolating refinement from the change of model class.
- experiments/2d/multilevel/bbox.py: max_sum_window / gt_window (s×s square of largest
  predicted mass / densest GT), crop_resize (batched grid_sample crop+resample),
  composite_window (write crop back). Unit-tested in test_bbox.py.
- src/models/pfn_seg_2d.py ImagePFN: backward-compatible use_external_features (consume
  precomputed features, no internal encoder) + forward(image_feats=, seed_query_mask=).
  Defaults reproduce prior behavior; test_imagepfn_modes.py + test_pipeline.py green.
- experiments/2d/multilevel/zoom_pipeline.py run_zoom_chain: frozen stage-1 coarse pred →
  encode maps ONCE → per hop crop-pool maps to the bbox (same encode-once features the
  PatchSetPFN chain uses), seed the query with the cropped coarse pred, composite the
  upsampled R0 output back at native H. test_zoom_pipeline.py.
- train.py: refine_arch switch (build_zoom_models warm-start, chain_fn dispatch in
  train_epoch, run_eval_zoom with native-H dice + in-bbox refine delta). Config
  configs/experiment/2d/multilevel_zoom.yaml (single 64px hop). Verified 1-epoch smoke run.
- bbox.max_sum_window / gt_window: empty maps (densest s×s window holds <0.5 mass — e.g. an
  all-background prediction or empty GT) CENTER the crop instead of collapsing to corner
  (0,0). Per-sample within a batch. Covered by test_bbox.py (empty + mixed-batch cases).

## 2026-06-22 — eval.py: allow eval-time override of pfn_seg token-grid resolution

- The pfn_seg_2d eval branch rebuilt ImagePFN using `resolution`/`input_patch_size`
  read *only* from the checkpoint's stored `arch`, so passing `arch.resolution=...`
  on the eval CLI was silently ignored.
- Added `eval.resolution` / `eval.input_patch_size` (default null → use ckpt) to
  eval_base.yaml; eval.py now prefers them over the checkpoint's arch and reflects
  the value actually used in the logged run_cfg. ImagePFN's learnable
  `pos_embed` is shaped (1,1,resolution²,e), so overriding to a resolution the
  weights weren't trained at fails loudly at load_state_dict (verified) — no silent
  wrong-weight eval. (patchset_pfn resolution comes from cfg.sample.resolutions, a
  separate mechanism — unchanged.)

## 2026-06-22 — Deterministic eval context for the real-image 2D datasets

- medsegbench / biomedparse / totalseg2d picked their K context pairs via
  `cow_index.sample_context`, which used the global `random` module. That is
  worker-safe (PyTorch reseeds stdlib `random` per worker, unlike numpy), but it
  also meant eval context drifted run-to-run and epoch-to-epoch — context-sampling
  variance leaking into eval Dice.
- `sample_context(cand, exclude, k, rng=None)` now takes an optional
  `random.Random`. Each dataset gained `deterministic` (default `split != "train"`,
  mirroring controlSynth) and passes `random.Random(idx)` on non-train splits, so a
  given eval target always draws the same context. Train keeps `rng=None` → global
  module → fresh, worker-distinct draws each epoch. controlSynth already had its own
  deterministic eval path (SeedSequence), so it was unchanged.

## 2026-06-22 — Fix UnboundLocalError in pfn_seg.run_eval

- `run_eval` accumulated val loss into `total_loss`/`nl` but referenced an
  undefined `val_loss` in the `tqdm.write` print (line 215), crashing the first
  eval. `val_loss = total_loss / max(nl, 1)` was only computed later (before the
  wandb.log). Moved that computation up before the print and removed the
  now-duplicate line.

## 2026-06-20 — Multilevel encoder: encode-once-pool-many (lossless ~2× speedup)

- **Problem**: `pipeline.run_chain` re-ran the FULL frozen encoder once per hop
  (`encode_grid(encoder, all_images, grid)` for each grid in resolutions[1:], e.g.
  32/64/128 → 3 forwards). But the encoder's stage maps are resolution-independent —
  `out_size` only feeds the final `adaptive_avg_pool2d`. So 2 of every 3 encoder passes
  were redundant.
- **Fix**: split both encoders into `encode_maps(images)` (resolution-independent stage
  maps) + `pool_maps(maps, out_size)` (pool/concat → reduce). `forward` now = pool∘encode
  (unchanged for single-resolution callers: ImagePFN, eval feature paths). `run_chain`
  calls `encode_maps` ONCE, then `pool_grid` per hop. Fallback to per-hop `encode_grid`
  for plain-callable encoders without `encode_maps` (e.g. test stubs).
- **Bit-exact**: UniverSeg and DINOv3 (incl. stage_l2norm and reduce=random) match the
  old per-hop output to max|Δ|=0 at every grid; unfitted-PCA guard preserved in pool_maps.
- **Speedup**: UniverSeg, 64 imgs, hops [32,64,128]: 4792 → 2232 ms (~2.1×) on CPU.
- **Tradeoff**: the 4 native-res stage maps are now held across the hop loop (stage0
  dominates, ~Bᵀ·64·H·W). Worth it vs 3× encoder compute; note if GPU mem-bound.
- Not the encoder context path (that's load-bearing, see prior analysis); this is the
  redundant re-encode across resolutions.

## 2026-06-20 — Unify encoder-feature normalization across pfn_seg / multilevel

- **Problem**: encoder (UniverSeg/DINOv3) features were normalized inconsistently.
  `ImagePFN` (pfn_seg stage-1) per-channel z-scored features by *context-row* stats
  (matching the TabPFN feature_sim backend in eval.py), but the multilevel pipeline's
  `encode_grid` fed *raw* encoder magnitudes straight into PatchSetPFN — no normalization.
  Worst for DINOv3 `feature_level=all`, whose 4 concatenated stages differ ~10–100× in
  magnitude (a known imbalance the `encoder_stage_l2norm` knob only partly addressed).
- **Fix**: single shared helper `standardize_by_context(feat, n_context)` in
  `src/models/pfn_seg_2d.py` — per-channel z-score using the first `n_context` rows'
  stats, applied to all rows (query standardized in the context's frame), clamp ±10.
  `ImagePFN`'s encoder path now calls it (raw-pixel path keeps its scalar normalization);
  `pipeline.encode_grid` now calls it too (signature gains `n_context`; `run_chain` passes K).
- **DINOv3 note**: per-channel z-score makes the cross-stage *magnitude* imbalance moot,
  so `encoder_stage_l2norm` is now largely redundant (it only rebalances channel *counts*,
  a representational weighting choice, not a scale bug). `encoder_imagenet_norm` is an
  input-side normalization (gray→RGB, ImageNet mean/std) and is orthogonal — unchanged.
- Verified: test_pipeline passes; helper gives ctx per-channel mean~0/std~1; both ImagePFN
  paths run. No retraining done — existing checkpoints will see a shifted feature frame at
  the multilevel chain, so chains should be re-trained/re-evaluated on the unified path.

## 2026-06-19 — controlSynth: fix context_copy & noise_level; qualitative panels

- **`context_copy_fraction` collapse FIXED** (`dataset.py`). At copy=1.0 UniverSeg gave
  Dice exactly 0. Diagnostics (UniverSeg forward) showed it is NOT pixel-identity: even a
  noised near-copy and mask-rolled variants collapsed, while real contexts scored 0.86.
  Root cause: copying the target *frame* makes the **background** match the query as well
  as the foreground, so the fg loses its distinctiveness for a context-matcher (the fg is
  normally "the only region consistent across contexts"). Redefined the knob: a fraction
  of contexts are now **pristine exemplars** — rendered with `shift_scale=0.1` (near-zero
  deformation) but a **fresh background** (`_make_subject(..., shift_scale=)`; copy logic
  moved out of `_apply_context_difficulty` → renamed `_apply_context_consistency`). Now
  non-degenerate (copy=1.0 → Dice ~0.80); a mild ease knob (the easy baseline is already
  context-saturated, so little headroom — eases more at harder operating points).
- **`noise_level` now monotone-correct** (0.0 easiest 0.80 → 1.0 hardest 0.68). The earlier
  baseline-dependent sign-flip resolved itself once the contrast redesign made the fg
  properly salient (images less OOD-pathological), so added noise only ever hurts. No
  separate noise code change needed.
- **Qualitative panels** (`synth_benchmark.py --plot`): per knob, renders the target
  (GT green / pred red contours) + K context images (fg mask overlay) at several knob
  values, annotated with mean UniverSeg Dice → `results/2d/synth_benchmark/<ts>/panels/
  <knob>.png`. Lets the difficulty curves be read against what the images look like.

## 2026-06-19 — controlSynth difficulty study: findings + foreground_contrast redesign

Four-round sensitivity study (UniverSeg baseline) consolidated in
`docs/datasets/controlSynth_difficulty_findings.md`. Headline: context-quality
(consistency/shift) and `region_size` are the real difficulty levers; the spec's
identification axis (`task_ambiguity`) is **inert** for a context-matcher (confirmed via
ambiguity×intensity and ambiguity×region_size grids — penalty ~−0.05 at every fg size);
`foreground_contrast` was inverted and `context_copy_fraction` collapses UniverSeg to 0.

**`foreground_contrast` redesign** (`src/datasets/controlSynth/appearance.py`,
`config.py`, `task.py`, `dataset.py`):
- Original bug: `gmm_fill` pushed *background* regions to the [0,1] extremes as contrast
  rose (bg saturation 5%→52%), leaving the fg a bland mid-grey blob → higher contrast was
  *harder* (Dice 0.60→0.23, inverted).
- Fix (a): background means now stay in a fixed central band [0.25,0.75]; the foreground
  is pushed `gap(contrast)` toward an extreme, so the **fg** owns the salient extremes.
- Fix (b, deeper): the fg's side is now a **task-level constant**
  (`task.make_base_geometry` records `meta["appearance_sign"]`, threaded through
  `dataset._make_subject` → `gmm_fill(..., fg_sign=)`). Previously the per-subject random
  extreme made high contrast push each subject's fg to an *independent* side, so the
  context no longer matched the target (the true cause of the inversion).
- `map_contrast_gap` range trimmed 0.55→0.40 (max gap 0.45) so extreme contrast stays
  clear of the background band rather than saturating into its noise tail.
- Result: correctly oriented axis — low contrast hardest (Dice 0.80, fg
  intensity-invisible → found via shape/context), rising to a ~0.85 plateau.

## 2026-06-19 — controlSynth difficulty benchmark (UniverSeg baseline)

New `experiments/2d/synth_benchmark.py` — measures how each controlSynth knob affects
task difficulty for the zero-training UniverSeg baseline, and surfaces *why* a value is
easy/hard (not just that it is).

- **Method = one-factor-at-a-time (OFAT)**: pin every knob at a moderate baseline
  (`BUILD_DEFAULTS`/`LIVE_DEFAULTS` in the script), sweep ONE knob across a value grid,
  so each knob's marginal effect on Dice is isolated. Morphology-specific knobs
  (thinness/tortuosity/branching → tubular; scattered_count/clustering → scattered) are
  swept with that morphology fixed; the rest use a clean `blob`. Live knobs reuse one
  frozen geometry bank (only the per-subject path changes) → cheap.
- **Per-subject record**: each evaluated subject logs its full param vector (all build +
  live knobs), realized stats (`fg_frac`, mean target↔context Dice `ctx_dice`), axis
  loadings, and Dice → `per_sample.csv` for arbitrary offline analysis.
- **Outputs** (→ `results/2d/synth_benchmark/<ts>/`): `per_sample.csv`, `summary.csv`
  (per knob×value: n, dice mean/median/std, fail_rate=Dice<0.1, fg_frac, ctx_dice),
  `difficulty_curves.png` (Dice + fail-rate vs value per knob), `morphology_difficulty.png`,
  and `report.txt` — knob sensitivity ranking (Dice spread, easiest→hardest value),
  per-knob Spearman(Dice, fg_frac)/Spearman(Dice, ctx_dice) to attribute difficulty to
  foreground-shrinkage vs. context-informativeness, and pooled global drivers.
- CLI: `--num_tasks/--subjects/--image_size/--context_size/--batch_size/--workers`,
  `--knobs <subset>`, `--quick` (smoke). Default 48 tasks × 24 subjects/task.
- **Plumbing**: `common.collate` now passes through per-element `meta` when the dataset
  provides it (controlSynth attaches its per-subject knob vector there); additive and
  guarded, so non-synth loaders are unaffected. `eval.py` gains an opt-in
  `eval.synth_csv` that dumps the same per-element synth-param rows from a normal eval
  run (mirrors the existing `patch_csv` pattern).
- Verified end-to-end with `--quick` (UniverSeg loads, sweeps run, all 5 artifacts
  written, CSV carries the full knob vector per subject).

## 2026-06-19 — Unify 2D dataset-source loading

Deduplicated the dataset-loading logic that had drifted across the `experiments/2d`
scripts (three near-identical loader builders with divergent source support: eval
`common.build_loader` did all 3 sources, `pfn_seg` did medsegbench+synthetic,
`multilevel/train` did medsegbench only).

- **`common.py`**: two new shared functions are now the single source of truth.
  - `build_dataset(cfg, split)` — dispatches on `cfg.data.source`
    (`medsegbench | biomedparse | synthetic`); folds in the controlSynth config
    wiring (previously duplicated in `common` *and* `pfn_seg._build_synth_dataset`).
    biomedparse/controlSynth imported lazily so the medsegbench path stays light.
  - `make_loader(ds, cfg, split, shuffle)` — shared subsample + `TaggedDataset`/
    `collate` + sampler policy: non-train splits subsample to `eval.max_per_label`;
    train uses `train.batch_size/workers` + optional `RandomSampler`
    (`data.max_train_samples`); val/test use `eval.batch_size/workers`.
  - `build_loader(cfg)` (eval) = `make_loader(build_dataset(cfg, cfg.data.split), …)`.
- **`pfn_seg.py` / `multilevel/train.py`**: `build_split_loader` collapses to
  `make_loader(build_dataset(cfg, split), cfg, split, shuffle)`. Removed the
  copy-pasted source dispatch, synth wiring, subsample block, and DataLoader assembly.
  Net effect: **every source is now available to every script** (multilevel/pfn_seg
  can train on biomedparse or synthetic; eval already could).
- **Configs**: the `synth:` block moved to a shared Hydra group
  `configs/experiment/2d/synth/default.yaml` (`@package synth`), pulled into
  `base`, `pfn_seg`, and `multilevel` via `defaults: [synth: default, _self_]`
  (`feature_sim` inherits it through `base`). Added `data.source` to `multilevel.yaml`.
  One place to edit synth defaults instead of three.
- Verified: all 4 configs compose with `cfg.synth` present; the unified path builds
  working train (bs=train) and val (bs=eval, difficulty-binned names) synthetic
  loaders and yields correctly-shaped batches.

## 2026-06-19 — controlSynth V1: difficulty-controlled synthetic in-context dataset

New package `src/datasets/controlSynth/` — a minimal on-the-fly procedural generator
for studying the impact of training on synthetic data (`experiments/2d/pfn_seg.py`).
Implements the *minimal generator* slice of `docs/datasets/controlSynth.md`.

- **Three orthogonal config axes** (`config.py`): `DiversityConfig` (num_tasks),
  `DifficultyBuildSpec` (frozen geometry), `DifficultyLiveConfig` (per-subject), plus
  documented monotone `[0,1]→param` mappings so a single difficulty knob can be swept
  with the rest pinned.
- **All five morphologies** (`shapes/`): blob/elongated/annular (`blob.py`), tubular via
  space colonization → caliber taper → vectorized capsule rasterization (`vessel.py`),
  scattered point process hardcore↔Poisson↔clustered (`scattered.py`); plus
  morphology-independent `boundary.py`, `area.py`, and `distractors.py` (geometry side of
  `task_ambiguity`).
- **No LMDB.** Base geometry is precomputed in RAM at dataset init (`geometry.GeometryBank`,
  the in-memory analog of the spec's store), deterministic in `master_seed`; only the cheap
  live path (deform → GMM fill → noise from a precomputed Perlin bank) runs per item.
  Vessels are therefore generated once per task, never per `__getitem__`.
- **Determinism** (`dataset.SynthICLDataset`): train draws fresh entropy (infinite
  subjects); val/test derive every subject seed from
  `(eval_seed_namespace, task_id, sample_index)` → byte-identical eval set. Task ids are
  split into disjoint train/val/test pools (held-out anatomies). Returns the MedSegBench
  4-key dict + `meta`, and exposes `.samples`, so existing `TaggedDataset`/`collate` work
  unchanged.
- **Integration**: `data.source=synthetic` switch in `pfn_seg.py:build_split_loader` and
  `common.py:build_loader`; a `synth:` config block in `configs/experiment/2d/pfn_seg.yaml`.
- **Difficulty-stratified metrics**: val `.samples` names carry the swept difficulty tag
  (`difficulty_tag()` from `build_spec.bin_factor`, e.g. `synth/blob/amb0.40`), so the
  existing per-dataset grouping in `run_eval` logs `dice/dataset/synth/blob/amb0.40` (and
  `dice_ds`/`dice_ds_soft` variants) with no eval-loop changes. `dice/mean` is unaffected.
- **Run naming**: `pfn_seg.py` now lets W&B auto-generate the run name (single
  `wandb.init(mode=…)` call) and saves checkpoints under `{date}_{wandb_run_name}` — the
  same convention as `multilevel/train.py`. The full `synth` config is logged to W&B so a
  difficulty sweep is comparable across runs by `config.synth.*`.
- **Verification** (`scripts/controlsynth_smoke.py`): visual grids per morphology
  (`results/controlsynth/*.png`), val determinism, and a DataLoader+ImagePFN forward all
  pass. End-to-end sanity: easy config reaches val Dice 0.85 in 8 epochs; hard config
  (small region + `task_ambiguity`) stays near 0 as expected.
- **Task diversity in one run**: `build.morphology` accepts a `{type: weight}` mixture
  (spans all five shape families); `mode=per_task_sampled` draws the factors listed in
  `build.sampled` (`{factor: [lo, hi]}`) per task, so a single run spans a difficulty
  range. The val grid then bins `bin_factor` into `n_bins` buckets → difficulty-response
  curve from one run (keys like `dice/dataset/synth/tubular/amb0.30`). NB: inline-dict
  yaml needs spaces (`{blob: 1, …}`); CLI overrides of `sampled` use `++` to replace the
  empty default (`'++synth.build.sampled={task_ambiguity:[0.0,0.8]}'`).
- **Deferred to later sub-projects** (per spec): LMDB store + precompute CLI,
  `eval_harness.py` + oracle UNets, clDice/NSD/size-stratified metrics, `mixed.py`
  (real+synth MixedDataLoader + curriculum). `resolve_difficulty` mode `binned` (a fixed
  labeled eval grid) still raises `NotImplementedError`; `fixed` and `per_task_sampled`
  are wired.
- *Known V1 limitation*: vessel realized area is dominated by the min-caliber floor
  (region_size weak for tubular), and thin vessels fragment under deformation — vessel
  Dice is indicative only until clDice lands.

## 2026-06-18 — Cheap channel-dim reduction for DINOv3 features

`DINOv3FeatureEncoder` gained an `encoder_reduce` knob (applied after pooling, so the
reported `feature_dim` — and thus the downstream `image_embed` input — shrinks):
- `none` (default), `grouppool:<d>` (adaptive_avg_pool1d over channels, zero params,
  no fit), `random:<d>` (frozen Gaussian / Johnson–Lindenstrauss projection, no fit),
  `pca:<d>` (PCA fit once on data, cached to disk).
- `encoder_stage_l2norm` L2-normalizes each stage map before the `all` concat, fixing
  the channel-count/scale imbalance between stages (zero params).
- PCA: `ensure_pca(image_iter)` loads the cached projection
  (`<hf_cache>/reductions/dinov3_<variant>_l<level>_<raw|l2>_pca<d>.pt`) or fits + caches
  it. `reduce_proj`/`reduce_mean`/`reduce_fitted` are registered buffers → for the
  ImagePFN path they ride in the checkpoint state_dict; the multilevel chain encoder
  (not in the checkpoint) re-loads from the disk cache at eval. Fit is wired into
  `pfn_seg.py` and `multilevel/train.py` (guarded by `needs_pca_fit`) and the
  `eval.py` chain branch (cache-hit, iterator untouched).
- Cost is negligible — all reductions add ~1–2 ms to the ~39 ms encode (B=64, 128px,
  bf16); the backbone dominates. The savings are downstream: `image_embed` input
  1920→d, smaller cached/transferred features, and less overfit on a 1920-wide input.
  Select via e.g. `arch.image_encoder=dinov3 arch.encoder_reduce=pca:256`.

## 2026-06-18 — Encoder benchmark: UniverSeg vs DINOv3 (FLOPs / VRAM / latency)

`experiments/2d/bench_encoders.py` times the frozen encoders under matched conditions
(same batch / image size / token grid, same precision, same compile) reporting forward
GFLOPs (FlopCounterMode), peak VRAM, ms/iter. RTX A4000, level=`all`, B=64.
- **128px → grid 16**, bf16: UniverSeg 260 GFLOPs / 657 MiB / 21.8 ms ;
  DINOv3-cnvnxt-base 642 GFLOPs / 658 MiB / 39.0 ms. DINOv3 ≈ 2.5× FLOPs, 1.8× latency.
- All four ConvNeXt variants (`all`, B=64, 128px, bf16): added `tiny`/`small` to `_DIMS`
  (both dims 96/192/384/768 → 1440ch; differ only in stage-2 depth 27 vs 9):

  | variant | feat_dim | GFLOPs | VRAM MiB | ms/iter |
  |---|---|---|---|---|
  | dinov3-tiny  | 1440 | 186  | 356  | 17.5 |
  | dinov3-small | 1440 | 363  | 439  | 27.3 |
  | dinov3-base  | 1920 | 642  | 658  | 39.1 |
  | dinov3-large | 2880 | 1436 | 1223 | 63.0 |
  | (universeg)  | 256  | 260  | 657  | 21.8 |

  **dinov3-tiny is both faster (17.5 vs 21.8 ms) and lighter (356 vs 657 MiB) than
  UniverSeg** while giving a 1440-dim feature — an attractive default. tiny vs small is
  same feature_dim at ~half the FLOPs (stage-2 depth).
- **256px → grid 32**, bf16: UniverSeg 1039 GFLOPs / **2613 MiB** / 86.7 ms ;
  DINOv3 2567 GFLOPs / **1606 MiB** / 142 ms. VRAM **crosses over**: UniverSeg keeps
  high-res 64-ch feature maps through all stages (activation-heavy), while DINOv3's stem
  downsamples /4 immediately → param-heavy but activation-light, so it scales better in
  memory with resolution despite more FLOPs.
- `compile bf16` == `eager bf16` to within noise: both encoders are `@torch.compiler.disable`
  (the pipeline runs them eager; adaptive_avg_pool with symbolic windows can't lower), so
  `torch.compile` graph-breaks at the encoder and gives no speedup. Compute lives in the
  frozen backbone, not in launch overhead.
- feature_dim asymmetry at `all`: UniverSeg 256 vs DINOv3 1920 — inherent to the concat;
  for a compute-matched A/B pick a single `feature_level` per encoder.

## 2026-06-18 — DINOv3 ConvNeXt feature encoder + `image_encoder` factory

Added `DINOv3FeatureEncoder` (`src/models/pretrained_encoders.py`) to study the impact
of encoder architecture/pretraining on the in-context segmentation model. Frozen
`facebook/dinov3-convnext-{base,large}-pretrain-lvd1689m` backbone (ConvNeXt CNN, DINOv3
SSL on LVD-1689M), matching `UniverSegFeatureEncoder`'s interface exactly:
`forward(images, out_size) → (N, feature_dim, out_size, out_size)`, so it drops into the
existing `image_encoder` injection seam in `ImagePFN` / the multilevel chain.
- Fully convolutional → size-agnostic (runs at native H, pooled to the token grid),
  same property the Strategy-A eval relies on. Stage maps: channels
  `[128,256,512,1024]` (base) at strides `[4,8,16,32]`; `level` picks a stage (0=highest
  res) or `"all"` → concat (feature_dim 1920 base / 2880 large).
- Adapts DINOv3's input contract: 1-ch grayscale → repeated ×3, optional ImageNet
  normalization (`encoder_imagenet_norm`, default on; off to let ImagePFN's own
  per-context standardization be the sole norm). Loaded `local_files_only` from the NFS
  HF cache (`…/ANALYSIS_20251122/checkpoints`).
- New `build_image_encoder(arch, device)` factory dispatches on `arch.image_encoder`
  (`patch`→none, `universeg`, `dinov3`/`dinov3-base`/`dinov3-large`) and returns
  `(encoder, feature_dim)`. Replaced all 5 hand-rolled encoder-construction branches
  (`pfn_seg.py`, `multilevel/train.py` ×2, `eval.py` ×3). The multilevel **chain**
  encoder defaults to `universeg` (back-compat) and is overridable via
  `arch.image_encoder=dinov3`.
- Select at train/eval time, e.g. `arch.image_encoder=dinov3 arch.feature_level=2`.
  Note feature_dim changes (UniverSeg `all`=256 vs DINOv3 `all`=1920), which sets the
  `image_embed`/PatchSetPFN input dim — a fresh checkpoint, not warm-startable from a
  UniverSeg one.

## 2026-06-18 — Strategy-A eval: encode at a different resolution, grids fixed

New `eval.encode_size` knob (`base.yaml`, `null` = checkpoint size) lets `pfn_seg_2d`
and `patchset_pfn` eval **feed images at a different size while keeping every token grid
fixed**. Only the frozen, fully-convolutional UniverSeg encoder runs at the new size,
then pools into the unchanged grids; the output is upsampled to native for scoring.
Decouples *encoder input resolution* from the baked patch/token grid.
- `eval.py`: model is built at the checkpoint's `model_size`; the loader serves
  `encode_size`. Stage-1 (and the pfn_seg model) get `patch_size` rescaled to
  `encode_size // resolution` so `Hp = H//P` stays == `resolution` (else the patch
  count ≠ `self.N` and the forward reshape crashes — see the H=256 probe). The chain
  builds `PatchSetPFN.mask_patch_size` from `model_size`, not the served size. Final
  prediction upsampled from the model grid (128) to native before Dice.
- `multilevel/pipeline.py`: `refine_level` takes the patch-tile size `p` from
  `model.mask_patch_size` instead of `label.shape[-1] // grid_res`, so a larger eval
  image is resized down inside `_mask_tiles` rather than producing a `p×p` that
  mismatches `mask_embed`. No-op at the training size (the two are equal).
- wandb config now logs **`encoder_input_size`** (served pixels) and **`model_size`**
  (the resolution the token grids were built at) instead of the ambiguous `image_size`,
  across all four backends. Run name appends the encoder size only when it differs:
  `patchset_pfn_s128e256_k3` (model 128, encoder 256) vs `patchset_pfn_s128_k3` default.
- `encode_size` must be divisible by the stage-1 resolution (16). Attention sequence
  lengths are identical to default; only the conv encode scale (and thus FLOPs/time)
  changes — this measures feature-scale robustness, not higher-res output (output grid
  stays 128). Caveat: feature normalization uses support stats, which shift off the
  training scale, so expect small ± rather than a guaranteed gain.
- Verified on `fast-microwave-31` / `busi val` (same 4 samples, seed 0):
  chain 128→`dice 0.9002`, 257 GFLOPs, 234 ms ; chain 256→`dice 0.9225`, 452 GFLOPs,
  532 ms. pfn_seg 128→`0.8318`, 85 GFLOPs ; pfn_seg 256→`0.8692`, 133 GFLOPs. Default
  (`encode_size=null`) is a bit-for-bit no-op.

Launch: `… model=patchset_pfn eval.checkpoint=<run>/best.pt
eval.stage1_checkpoint=<stage1>/best.pt eval.encode_size=256`

## 2026-06-18 — `patchset_pfn` (multilevel chain) backend in the 2D eval

`experiments/2d/eval.py` gains a 4th backend, `model=patchset_pfn`, so the multilevel
coarse→fine chain is benchmarked under the **same conditions** as the other models
(per-dataset Dice, inference ms/item, FLOPs). Final native-resolution Dice only.
- Reconstructs the full system the way `multilevel/train.py` does: frozen stage-1 ImagePFN
  (`load_stage1`, duplicated from train) + frozen `UniverSegFeatureEncoder` + a `ModuleList`
  of trained `PatchSetPFN` hops. `arch` + `sample` (ladder/budgets) are read from the
  checkpoint and injected into `cfg`; the chain runs via `pipeline.run_chain` with
  `cfg.sample.eval` and `eval_deterministic`. Prediction = `outputs[-1]["refined_grid"]`
  reshaped to native (final hop grid == image_size), i.e. exactly training's
  `dice_r{native}/mean`.
- State-dict load uses `.replace("_orig_mod.", "")` (compiled `ModuleList` buries the prefix
  mid-key). FLOPs counted over the whole chain (stage-1 + encoder + all hops).
- The stage-1 path isn't in older checkpoints → new `eval.stage1_checkpoint` fallback
  (`base.yaml`); `train.py` now also records `stage1_checkpoint` in `best.pt` for future runs.
- `src`-shadowing guard (patch_icl src must beat ic_segmentation's) extended to fire on
  `patchset_pfn` too. image_size + context_size synced from the checkpoint.
- Verified end-to-end on `fast-microwave-31` (ladder 16→32→64→128): `busi val` →
  `dice/mean=0.9002`, 285 ms/item, FLOPs logged.

Launch:
`.venv311/bin/python experiments/2d/eval.py model=patchset_pfn
eval.checkpoint=<run>/best.pt eval.stage1_checkpoint=<stage1>/best.pt
data.split=val eval.max_per_label=20`

## 2026-06-18 — Wired BiomedParse into the 2D eval + macro-average

`experiments/2d/eval.py` can now run on BiomedParse (non-breaking for MedSegBench):
- New `data.source: medsegbench | biomedparse` knob (`configs/experiment/2d/base.yaml`);
  `common.build_loader` branches on it. BiomedParse has only `train`/`test`, so pass
  `data.split=test`.
- `common.log_summary` now also emits **`{prefix}/macro`** — the mean over per-`(dataset,
  label_value)` cell means (for BiomedParse = `(dataset, target)`). Weights every cell equally
  so multi-label datasets can't dominate the headline like the per-sample micro-average lets
  them (the original m2caiseg 0.33-vs-0.55 weighting bug). Additive: `dice/mean` (micro) still
  printed alongside `dice/macro`.
- Verified end-to-end: `eval.py data.source=biomedparse data.split=test data.dataset=DRIVE
  data.image_size=128` → UniverSeg 0.385 Dice (DRIVE retinal vessel; thin tubular, low as
  expected); loader→model→metrics→summary all run, wandb offline.

Launch (all datasets, capped, macro-averaged):
`.venv311/bin/python experiments/2d/eval.py data.source=biomedparse data.split=test
data.image_size=128 eval.max_per_label=20`

## 2026-06-18 — BiomedParse: cells keyed by (dataset, modality, target)

`cell_of` now returns `(dataset, modality, target)` (was `(modality, target)`) — added
`dataset_of`. Keeps each source separate (no cross-dataset pooling): e.g. amos22 CT-liver and
another CT dataset's liver are distinct cells. Dataset key carries sublevels (`amos22/CT`,
`Radiography/COVID`, `MSD/Task01_*`), so those are already first-class cells. Real 6-dataset
subset: 26 → 27 cells. This is the unit a macro-averaged eval should weight equally.

## 2026-06-18 — BiomedParse dataloader speed: benchmark + ~3x lazy-path optimization

Benchmarked dataloading throughput vs MedSegBench (`experiments/2d/bench_dataloader.py`,
batch_size=32, workers=16, steady-state samples/s after warmup batch):

| image_size | MedSegBench | BiomedParse (before) | BiomedParse (after) |
|-----------:|------------:|---------------------:|--------------------:|
| 128        | ~2160       | ~28 (cold) / ~40     | **~99**             |
| 256        | ~580        | ~41                  | **~92**             |
| 512        | ~161        | ~41                  | **~89**             |

Root cause of the gap: MedSegBench loads pre-resized npz into RAM (per-item = just `/255`),
while BiomedParse decodes a **1024×1024 RGBA PNG per image off NFS**. Micro-bench: decode alone
= **39 ms/img** (irreducible for PNG); the old path added ~27 ms (np channel-mean + torch
interpolate). Optimizations in `src/datasets/biomedparse.py`:
- Decode→gray→resize **inside PIL** (`convert("L")` + `resize`) instead of full-array mean +
  `F.interpolate` (66→45 ms/img).
- **Cache the small resized tensors** (not the 1024² source) keyed by path — context images are
  reused heavily across samples, so hits skip the decode entirely. → ~2.5–3.5× overall.
- Cache budget is now **size-aware** (`cache_size=None` → ~64 MB/worker) so 512px × 16 workers
  doesn't OOM.

BiomedParse is now flat ~90/s (decode-bound). Still ~20× under MedSegBench because each unique
image is a fresh 1024² NFS decode. **Real fix if needed (e.g. for training): pre-resized packed
cache** (one npz/npy per source in RAM, mirroring MedSegBench's fast path) — deferred; for *eval*
~90/s overlapped with GPU forward is likely sufficient.

## 2026-06-18 — BiomedParseData dataloader: reconciled with real extracted layout

Inspected the extracted data at `.../ANALYSIS_20251122/data/biomedparse` (29 datasets;
amos22/MSD still populating) and fixed `src/datasets/biomedparse.py` to match reality — the
prototype's flat-path assumption would have found **zero** data. Findings + changes:
- **Double-nesting:** real layout is `<root>/<DATASET>/<DATASET>/{train,test,...}`, and three
  datasets add one sublevel — `amos22/amos22/CT/…` (modality), `MSD/MSD/Task01_*/…` (task),
  `Radiography/Radiography/{Normal,COVID,Viral_Pneumonia,Lung_Opacity}/…` (class). Added
  `_discover_sources` (depth-bounded glob, depths 1–4) + `_collapse_key` → dataset keys like
  `ACDC`, `amos22/CT`, `Radiography/COVID` (context never mixes across sublevels).
- **`absent.png` sentinel** ("target not present") in PanNuke/kits23/amos22 mask dirs — now
  skipped explicitly.
- `DATA_ROOT` corrected `biomedparse_datasets` → `biomedparse`.
- Filename parsing **confirmed correct** against real names: target = last `_`-token (`+`→space,
  e.g. `left+heart+ventricle`); modalities use `-` internally (`MRI-T1-Gd`) so `_`-split is safe.
- Self-test fixture rewritten to mirror double-nest + a modality sublevel + sentinel + orphan
  mask; passes (20 samples, 5 cells). Real-data smoke test over DRIVE/GlaS/ISIC/REFUGE/amos22/
  Radiography: 41,272 samples, **26 (modality×target) cells** (CT/X-Ray/dermoscopy/fundus/
  pathology), shapes/range/context all correct.
- **Still open:** wire into `experiments/2d/eval.py` `build_loader`; add macro-average over
  `cell_of` cells in the eval aggregation.

## 2026-06-17 — BiomedParseData in-context dataloader (prototype)

`src/datasets/biomedparse.py` (new). `BiomedParseDataset` mirrors `MedSegBenchDataset`'s
contract (`samples` = `(dataset, image_idx, target_int)` 3-tuples; `__getitem__` →
`image/label/context_in/context_out`) so it drops into the 2D eval via `common.TaggedDataset`/
`collate`. Reads the official on-disk layout `<root>/<DATASET>/{train,train_mask,test,test_mask}/*.png`
(populate with `huggingface-cli download microsoft/BiomedParseData`); each `*_mask/` PNG =
one (image, modality, site, target) eval unit, parsed from the filename
(`[IMAGE]_[MODALITY]_[SITE]_[TARGET].png`, `+`→space). PNGs loaded lazily with an LRU cache;
images grayscaled + resized (bilinear) and `/255`, masks `!=0`→fg (nearest). Context = K others
sharing `(dataset, target)`, with-replacement if scarce.
Exposes `modality_of/target_of/cell_of` → the `(modality × target)` grid for macro-averaging
(the point vs MedSegBench's flat micro-average; see prior entry). Ships a synthetic-fixture
`__main__` self-test (no download needed) — passes: 14 samples, 3 cells, shapes/dtype/range
asserted. **Follow-up:** one-line branch in `experiments/2d/eval.py` `build_loader` to select it.

## 2026-06-17 — 2d eval: UniverSeg MedSegBench Dice gap diagnosed (weighting, not model); benchmark research

Investigated why `experiments/2d/eval.py` reports UniverSeg Dice 0.33 on MedSegBench while
`ic_segmentation/scripts/eval.py` reports 0.55 on the same model+data. **Root cause: eval-set
sample weighting, not the model or preprocessing.** Both paths build the *same* `UniverSegBaseline`
(input_size=128) and read the *same* npz directory; per-(dataset,label) Dice agrees between them.
The divergence is aggregation:
- `src/datasets/medsegbench.py` emits one sample per **(image, label_value)**, so the 18-organ
  `m2caiseg` dataset alone = **5,501 / 13,237 samples (42%)** of the eval, mostly near-zero rare
  organs → drags the micro-average to 0.33.
- ic emits one **random label per image** and caps per-dataset → those empty classes are
  down-weighted → 0.55.
Reproduced on a matched capped subset: patch_icl **0.25** vs ic **0.42** (same ~1.6× ratio).
- **Ruled out (A/B tested):** image normalization. Swapping patch_icl's plain `/255` `_to_tensor`
  for ic's percentile-[1,99] clip gave **0.233 vs 0.249** on the identical subset — slightly worse,
  not the cause. Change reverted. (Note: both pipelines also pass through the wrapper's per-sample
  min-max, which masks most of the normalization difference.)
- **Secondary (open):** RGB dermoscopy datasets genuinely differ beyond weighting (isic2016
  0.50 vs 0.73, isic2018 0.49 vs 0.70) — likely the RGB→gray path (patch_icl averages channels to
  uint8 at load vs ic per-sample float) and/or small-N context-sampling variance.
- **Takeaways:** (1) score in-context eval with **macro-average per (modality × shape-class)** +
  per-task sample cap, not flat per-sample micro-average; (2) move to a diverse, balanced
  multi-modality 2D benchmark. Candidate datasets researched and specced in `docs/datasets.md`.

## 2026-06-17 — 2d/multilevel: N-level resolution chain (16→32→64→128)

Extended the 2-level (res-16→res-32) refinement into a configurable coarse-to-fine chain
with per-level weights. Spec/plan: `docs/superpowers/specs/2026-06-17-multilevel-resolution-chain-design.md`,
`docs/superpowers/plans/2026-06-17-multilevel-resolution-chain.md`.
- `pipeline.py`: new `composite_predictions`, `refine_level` (one hop: sample→gather→
  forward→composite), `run_chain` (driver: seeds from stage-1, loops `sample.resolutions`,
  upsamples + DETACHES `refined_grid`/`this_think` between levels, no_grads frozen
  stage-1/encoder). `build_patch_batch` retired.
- `patchset_pfn.py`: `forward(..., return_thinking=True)` returns post-transformer thinking
  pooled over columns → chained as the next level's memory.
- `train.py`: models are now an `nn.ModuleList` (one `PatchSetPFN` per hop, own
  `mask_patch_size` 4→2→1 and `stage1_proj`); per-hop weighted losses; `run_eval` loops a
  true-resolution Dice ladder (`dice_r16/r32/r64/r128/mean`, `dice/mean`=final native) +
  per-hop `refine/hop{L}/{delta_err,dice_delta,soft_dice_delta}`. Checkpoint on `dice/mean`.
  Asserts `resolutions[0] == stage-1 res`. Compile wraps each submodule.
- config: `sample.resolutions`, per-hop `n_total/n_fg_core/n_fg_core_ctx`, `train.loss_weights`.
- Training detached per-level (each hop refines the detached composite below it). Verified:
  unit tests green; full 4-level run trains (no NaN, ckpt saves); `resolutions=[16,32]`
  reduces exactly to the old single hop (`dice/mean==dice_r32/mean`).
- Fix: `train.checkpoint` warm-start used `removeprefix("_orig_mod.")`, but a compiled
  ModuleList saves keys as `"{L}._orig_mod...."` (prefix is mid-key, not leading) → it
  matched 0 tensors silently for default `compile=true` chain checkpoints. Switched to
  `replace("_orig_mod.","")` (verified 0/74 → 74/74) + a warning when 0 tensors load.

## 2026-06-16 — 2d/multilevel: dice/mean checkpoint selection + pfn_seg-matched coarse baseline

Stage-2 `train.py` now selects the best checkpoint on native `dice/mean` (the deployment
metric, pfn_seg-comparable) instead of `refine/uncertain/delta_err` (kept as diagnostic;
it's a local L1-reduction on boundary cells that can diverge from native Dice and ignores
certain-cell regression). `run_eval` returns `dice/mean` for selection.

Metric naming made resolution-honest (a `_r16`/`_r32` suffix now means the dice is
computed AT that resolution, verified: dice@r16 0.896 ≠ s1@128 0.782, dice@r32 0.093 ≠
s2@128 0.139):
  - `dice_r16/mean` = stage-1 pred @res-16 vs GT@16 (== pfn_seg low-res dice).
  - `dice_r32/mean` = refined map @res-32 vs GT@32 (new).
  - `dice/mean` = stage-2 refined 32→128 @128 (checkpoint/headline, pfn_seg-comparable).
  - `dice_s1/mean` = stage-1 16→128 @128 (pfn_seg headline baseline); `dice/margin_vs_s1`.
  - refine scopes are res-32 stage comparisons → suffixed by STAGE: `refine/{scope}/dice_s1`
    vs `dice_s2`, `soft_dice_s1/s2`, `refine/certain_err_s1/s2`. Also log per-sample
    improvements `refine/{scope}/dice_delta` and `soft_dice_delta` (s2-s1, >0=better,
    nanmean of paired diffs like delta_err).
Rationale: r16/r32 reserved strictly for true-resolution dices; where two maps are
compared at one shared resolution (native-128 pair, res-32 refine scopes) resolution
can't distinguish them, so they carry s1/s2.
Resolution numbers in the metric keys are derived, not hardcoded: R1 = stage-1 native
res (`round(stage1.N**0.5)`), R2 = `cfg.sample.grid_res`, H = `cfg.data.image_size`;
keys built as `f"dice_r{R1}/mean"` etc.
Fixed the coarse baseline: `dice_coarse/mean` was computing stage-1 via res-16→32→128
(double upsample); now uses the stage-1 res-16 map upsampled 16→128 DIRECTLY, exactly as
pfn_seg.py (same bilinear/align_corners/hard_dice), so it equals pfn_seg's stage-1 number.
Pipeline `coarse_predict` now also returns `coarse_lowres` (B,R1,R1) for this. Smoke-tested
1 epoch on busi: plumbing OK, coarse baseline 0.78.

## 2026-06-16 — 2d/multilevel: target n_fg_core sweep (weak lever)

prev_pred --stats, n_fg_core 64/128/192: fg captured 66.2%→69.1%→69.5% (saturates at
128), bnd→miss FLAT 31.5% throughout (fg_core takes budget from neighbors, never the
top-priority boundary core). Gain lands only on easy datasets with spare budget; hard
thin-structure datasets unchanged (tnbcnuclei fg→miss 51.9→53.4, monusac 52.8→54.5) —
already budget-saturated by their large boundary band, residual fg→miss is structural
(objects > 256 patches at res-32). Much weaker than the context lever (64→160 gave +9pts
on ds_gt) because prev_pred under-confidence already puts fg interior in the boundary
band (fg→core 60% at nfg=64), making the explicit fg quota partly redundant on target.
**Decision: keep context heavy (160), target light — leave target 64 or nudge to 96
(~+2pts free, keeps neighbor budget; >128 the neighbor tier vanishes for no gain).**

## 2026-06-16 — plot_sampling.py: --hist value-distribution diagnostic

Added `--hist` mode: per-cell value distribution (per-dataset + total) for GT and the
source map, with `%@0`/`%@1`/`%mid`, mean, and `|v-0.5|<tau` band fractions + a 10-bin
histogram. Run with `--source prev_pred` to get both GT and prediction in one pass.
**Findings (full val, res-32):** distribution is extremely bimodal. GT: 84.6%@0 /
7.8%mid / 7.6%@1 — the boundary band ≈ all fractional cells (`%mid`≈`|.5|<0.45`).
prev_pred: 75.6%@0 / 21.2%mid / 3.2%@1 — same mean (0.11) but `%@1` HALVED and `%mid`
2.7×: the stage-1 model is under-confident on fg (interiors predicted ~0.7-0.9), so
the 0.5-band is biased inward and mixes true edge with under-predicted interior →
mechanism behind the ~31% target boundary-miss floor. Fixed `tau` gives a ~70×-variable
boundary-core size across datasets (idrib 0.5% → tnbcnuclei 35% at |.5|<0.30), so the
boundary/fg/neighbor budget split is set by the data, not by config (tnbcnuclei's
boundary core alone > n_total).

**Quota-cap experiment (tested → REJECTED).** Added opt-in `--n_boundary_core` (caps the
tau band at the N cells closest to 0.5; default 0 = uncapped = unchanged). prev_pred
--stats, cap 0/96/64: capping WORSENS bnd→miss (31.5%→37.6%→40.8%) and barely moves
fg→miss (33.9%→33.2%) — net negative, even on the thin-structure datasets it targeted
(tnbcnuclei fg→miss 54→62, bnd→miss 62→64). Two reasons: (1) freed budget flows to
NEIGHBORS not fg_core (fixed 64), trading high-value boundary cells for low-value
neighbor fill; (2) on prev_pred the under-confident prediction puts fg interior (0.7-0.9)
INSIDE the tau band, so it covers boundary AND fg at once — capping loses both. The
variable tau allocation is a feature: high-boundary datasets need more boundary budget.
Kept the arg as a diagnostic only; real sampling.py unchanged.

## 2026-06-16 — 2d/multilevel: per-role fg quota (context n_fg_core)

Split `n_fg_core` by sampler role. Target and context both call `sample_patches`, but
with opposite goals: target ranks cells by the stage-1 `coarse_flat` (prev_pred,
uncertainty regime), context ranks by the TRUE GT mask fraction (`ctx_frac`). Context
wants a large share of the foreground sampled for class info, not boundary recall.
Added `sample.n_fg_core_ctx: 160` (`configs/experiment/2d/multilevel.yaml`);
`pipeline.py` context call now uses `s.get("n_fg_core_ctx", s.n_fg_core)` (target call
unchanged at 64). Backward-compatible — absent key falls back to `n_fg_core`. Motivated
by the ds_gt fg-coverage sweep: fg captured 72%→81% going 64→160, boundary miss flat
~28% (heavy fg costs the boundary nothing on the GT map). Diagnostic: `--source ds_gt`
in plot_sampling.py.

## 2026-06-16 — plot_sampling.py: fg-sourced neighbor diagnostic

`sample_patches` neighbor field now diffuses from `boundary_core ∪ fg_core` (was
boundary core only), and gained a `boundary_tier` arg + `--no_boundary` CLI flag to
disable the tau tier (tau→0). Added `[C] boundary coverage` block to `--stats`:
of true-boundary cells (`0<gt<1`), the % reaching core / neighbor / missed. Lets the
head-to-head run by invoking the script with different flags (no `--compare` mode):
`--n_fg_core 0` = boundary-only baseline, `--n_fg_core N` = union,
`--n_fg_core N --no_boundary` = pure fg-sourced neighbors. Spec:
`docs/superpowers/specs/2026-06-16-fg-sourced-neighbor-experiment-design.md`.
**Result (prev_pred, full val, pooled):** dropping the boundary tier (fg-sourced
neighbors only) raises bnd→miss 31.4%→47.8% — unanimous across all 35 datasets;
worst on compact/sharp objects (covid19radio 3.8→42.7, isic2016 5.7→40.1,
pandental 16.4→68.6). The `core_b ∪ fg_core` neighbor-union does NOT help the
boundary: union bnd→miss 31.6% ≈ baseline 31.4%, and union's bnd→neigh (4.7%) is
LOWER than baseline's (6.4%) — boundary is held up entirely by the boundary-core
tier. The fg_core quota's value is interior coverage (fg→miss 45.7%→30.6%), not
boundary. **Conclusion: boundary tier is load-bearing; fg-sourced neighbors are
not. The res-16 map (offset 0.5-band) is the ~31% boundary-miss floor, not the
sampler.**

## 2026-06-16 — 2d/multilevel: configurable mask-token form (arch.mask_prior)

Replaced `arch.coarse_prior` (bool) with `arch.mask_prior: false | scalar | patch` in
`PatchSetPFN` + pipeline. `false`/`scalar` keep the scalar mask-token (neutral support-mean
vs coarse-pred query prior). `patch` makes the mask-token a `p×p` mask tile
(`mask_embed=Linear(p²,e)`, `p=image_size//grid_res` auto): support tiles = native GT under
each cell (exact boundary geometry, richer than the avg-pooled fraction), query tiles =
upsampled coarse prior. New `pipeline._mask_tiles` helper; `qry_coarse` scalar kept as the
metrics baseline. Caveat: patch mode ties `mask_embed` to grid_res (not cross-resolution
generalizable); query tile carries no detail below the res-16 stage-1. Tests
(`test_patchset.py`, `test_pipeline.py`) updated + patch-mode cases added; 1-epoch smoke run
in patch mode verified (params +3840 = (p²-1)·e).

## 2026-06-16 — 2d/multilevel: sampling-procedure redesign + diagnostic

Designed a new stage-2 patch sampler (spec:
`docs/superpowers/specs/2026-06-16-multilevel-patch-sampling-design.md`): threshold
boundary core + fixed random-fg-core quota + blurred-field Gumbel-top-k neighbor fill,
replacing the fixed n_uncertain/n_certain split. Built `experiments/2d/multilevel/plot_sampling.py`
to visualize and measure it on MedSegBench val (`--stats`, `--sweep`), with a `--source
ds_gt|prev_pred` toggle that samples from either res-32 GT or the real frozen stage-1
prediction (res-16→32) at the correct resolutions.

Sweep findings (full val, 13,237 imgs, fg from true GT): tuned defaults `tau=0.30,
sigma=1.0, floor=0.005, n_fg_core=64` (M=256, grid=32) → fg→miss ~36%, matching the GT
oracle. `tau` cannot be read off the oracle (0.45 best for ds_gt but regresses under
prev_pred as the neighbor fill collapses on the misplaced predicted boundary).

Implemented: `sampling.py` now exposes `sample_patches` + `gaussian_blur` (old
`sample_patch_indices` retired); `pipeline.build_patch_batch` uses it for both query and
support paths with `qry_is_uncertain` = boundary core (excludes fg-core), plus a
`stochastic` flag (train True, eval `not eval_deterministic`); config `sample.*` replaced
`n_uncertain/n_certain` with `n_total/tau/n_fg_core/blur_sigma/floor/temperature/
eval_deterministic`; `train.py` threads `stochastic`. Run name now comes from wandb
(auto-generated); checkpoints save under `{date}_{run_name}` (e.g. `2026-05-22_deft-field-72`).
Tests (`test_sampling.py`, `test_pipeline.py`) updated and passing; 1-epoch smoke run of
`train.py` train+eval+checkpoint verified.

## 2026-06-16 — 2d/multilevel: configurable query sampling map (prev_pred | ds_gt)

Isolate the stage-2 patch-selection signal from the model. Added `sample.train` and
`sample.eval` (`prev_pred` | `ds_gt`) to `configs/experiment/2d/multilevel.yaml`.
`build_patch_batch` now takes `sampling_source`: `prev_pred` ranks query cells by the
stage-1 coarse prediction (current/deployable behaviour, default), `ds_gt` ranks by the
downsampled target GT (oracle upper-bound — leaks labels, eval numbers optimistic).
Only query selection changes; `qry_prior` and the eval fusion base stay on the coarse
map, so the model input is held fixed across the two modes. Run name gains a
`_smp-{train}-{eval}` tag. Files: `pipeline.py`, `train.py`, `multilevel.yaml`.

## 2026-06-15 — pfn_seg: fix CUDA grid-limit crash at resolution≥32

`src/models/pfn_seg_2d.py`. Running with `arch.resolution=32` crashed in the
sample-axis attention with `CUDA error: invalid configuration argument`.

- **Root cause**: the sample-axis (cross-image) attention flattens to a batch of
  `B·2·resolution²` independent tiny (seq=12) attention problems. Flash /
  mem-efficient SDPA launch one CUDA grid-Y block per batch element, and
  `gridDim.y` is hardware-capped at 65535. At R32 with `batch_size=32` the batch
  is `32·2·1024 = 65536`, exactly one over the cap (R16 was 16384, fine).
  Confirmed by minimal repro: flash SDPA OK at batch 65535, crashes at 65536.
- **Fix**: `batched_sdpa()` helper splits the batch into equal chunks under the
  cap when needed. Keeps the fused kernel (the math backend works but materializes
  the full score tensor at ~2× memory). `int()` pins the symbolic batch to a
  concrete value so the chunk count is a Python int that `torch.compile`
  (`dynamic=True`) unrolls statically — a plain `range(0, B, step)` over a
  symbolic `B` fails to compile.
- **Note (separate constraint, not a bug)**: R32 + UniverSeg encoder + 6 layers
  at `batch_size=32` OOMs on a 24 GB card (~23 GB needed); R32 has 4× the tokens
  of R16. Use a smaller `batch_size` (verified training proceeds end-to-end at
  `batch_size=8`) or gradient accumulation for the full run.

## 2026-06-14 — multilevel: log native `dice/mean` for direct comparison to pfn_seg

`experiments/2d/multilevel/train.py`. `run_eval` now also logs `dice/mean` — native-resolution
hard Dice over the whole val set, aggregated identically to `pfn_seg.py` (mean over all samples,
plus `dice/dataset/<name>`). The stage-2 "final" map = coarse res-32 map with sampled cells
overwritten by stage-2 preds, upsampled to native (128). `dice_coarse/mean` logs the coarse-only
baseline (≈ the stage-1 model's own dice/mean) so the lift is visible in one place. This makes the
2-stage pipeline directly comparable to the single-stage ImagePFN number in `results/report.md`.

## 2026-06-14 — multilevel: within-image spatial reasoning (query↔query attention)

`src/models/pfn_seg_2d.py`, `src/models/patchset_pfn.py`, `experiments/2d/multilevel/train.py`,
`configs/experiment/2d/multilevel.yaml`.

`arch.query_self_attn` (default true): lets stage-2 query patches attend to **each other**
(keyed by Fourier PE), restoring within-image spatial coherence that the patch-set form
otherwise lacks (each query was classified independently given the support). No label leak —
query mask-tokens carry the coarse prior, not GT. Implemented as an asymmetric sample-axis
mask: support rows attend only to the train set `[:sep_t]`; query rows attend to train set +
all queries. `TransformerEncoderLayer`/`Stack` gained an optional `attn_mask` (default None =
prior behavior, so `ImagePFN` stage-1 is untouched). Memory cost ~0 (+0.01 GB) since the
sample axis is batched over only 2 columns. Verified: a coupling test (perturbing one query's
feature moves another query's output only when enabled), ImagePFN regression, and compiled
(`torch.compile(dynamic=True)`) forward+backward — the dynamic (r×r) bool mask lowers cleanly.

## 2026-06-14 — multilevel: critical target-leak fix + soft-dice / full-image metrics

`experiments/2d/multilevel/pipeline.py`, `experiments/2d/multilevel/train.py`.

**Bug fix (critical):** `build_patch_batch` derived the target fraction `gt32` from
`all_masks[:, -1]`, but the query mask is zeroed there (so stage-1 doesn't see the answer)
— so the supervision target was **all zeros**. Symptoms: train loss collapsed to 0.0000,
a spurious large positive `Δerr`, and NaN dice. Now `gt32` is pooled from the real `label`.

**Metrics:** `run_eval` now reports, per scope `{uncertain (192), sampled (256), full (1024
via compositing stage-2 preds into the coarse map at sampled cells)}`: `delta_err`,
hard `dice_stage2/coarse`, and **soft** `soft_dice_stage2/coarse` (vs the coarse baseline).
Pipeline returns `qry_idx`, `coarse_full`, `gt_full` to support this. Robust `nanmean`
removes the `Mean of empty slice` warning. Headline for checkpointing stays
`refine/uncertain/delta_err`.

## 2026-06-14 — multilevel patch refinement (stage-2 PatchSetPFN)

New experiment `experiments/2d/multilevel/`. A frozen res-16 ImagePFN (stage 1) +
frozen UniverSeg encoder produce a coarse target prediction and res-32 features; we
sample 256 patches/image (192 closest-to-0.5 + 64 most-certain) — target→query,
context→support — and train `src/models/patchset_pfn.py:PatchSetPFN` (nanoTabPFN-shaped:
rows=patches, cols=[img|mask], 2-D Fourier PE, query-attends-to-support) to refine the
query patches. Metric: `refine/delta_err_uncertain` = |error| reduction on the uncertain
target region vs the stage-1 coarse value (64 certain patches kept as a regression check).
`arch.coarse_prior` toggles using the coarse pred as the query mask prior (else neutral 0).

**Stage-1 thinking memory** (`arch.use_stage1_thinking`, default true): `ImagePFN.forward`
gains `return_thinking` → its post-transformer thinking rows mean-pooled over columns
`(B, n_think, e1)`. `PatchSetPFN` projects these `e1→e` (+ a learned type token) and prepends
them as extra **support** rows (inside `sep`), so query patches attend to stage-1's latent
task summary via the existing sample-axis attention. `stage1_dim`=stage-1's `e`, read from
`stage1.thinking.tokens` at construction; new proj/type params are fresh (warm-start-safe).

Files: `src/models/patchset_pfn.py` (FourierPositionalEncoding + PatchSetPFN),
`experiments/2d/multilevel/{sampling,pipeline,train}.py` (+ `test_*.py`),
`configs/experiment/2d/multilevel.yaml`. Reuses the shared utils in `pfn_train.py`.
Run: `python experiments/2d/multilevel/train.py [arch.coarse_prior=false]`.
Spec/plan: `docs/superpowers/{specs,plans}/2026-06-14-multilevel-patch-refinement*.md`.

## 2026-06-14 — Factor shared training utils into pfn_train.py

`experiments/2d/pfn_train.py` (new), `experiments/2d/pfn_seg.py`.

Extracted `_newtonschulz5_batched`, `Muon`, `augment`, `lawa_average`, `soft_dice_loss` from
`pfn_seg.py` into a new shared module `pfn_train.py`. `pfn_seg.py` now imports them from there.
No behaviour change; prepares for reuse in `experiments/2d/multilevel/train.py`.

## 2026-06-14 — pfn_seg_2d: optional frozen pretrained image encoder (UniverSeg features)

`src/models/pretrained_encoders.py` (new), `src/models/pfn_seg_2d.py`,
`experiments/2d/pfn_seg.py`, `experiments/2d/eval.py`, `configs/experiment/2d/pfn_seg.yaml`.

New `arch.image_encoder: patch | universeg`. With `universeg`, the image path becomes a
frozen UniverSeg encoder → `resolution×resolution` feature grid → `Linear(feature_dim, e)`,
replacing the raw-pixel patchify+embed. Mirrors the `feature_sim` eval backend
(`encode_images` + `extract_features_batch`); `arch.feature_level` selects the level
(`all` → 256ch). The mask path is unchanged (raw P×P patches → Q×Q → `Linear(Q², e)`).

Design: the encoder is **injected** into `ImagePFN` (new `image_encoder`/`feature_dim` args),
not imported by it, so `pfn_seg_2d.py` stays torch-only and free of the `src`-package
shadowing (UniverSeg loads from its own `/home/dpxuser/repos/UniverSeg` checkout). It runs
under `no_grad` and is frozen — gradients reach `image_embed` but not the encoder; its params
are filtered out of the Muon/Adam groups via `requires_grad`. Image features are normalized
**per channel** with context-image stats (vs. the single-scalar norm on raw pixels). `eval.py`
rebuilds + injects the encoder when the checkpoint's arch says `universeg`.
`encoder_resize_to_input` (default false) gates resizing inputs to 128² before the fully-conv encoder.
The encoder's `forward` is wrapped in `torch.compiler.disable` (frozen/no_grad — nothing to compile):
under `torch.compile(dynamic=True)` its `adaptive_avg_pool2d` otherwise gets symbolic window sizes
inductor can't lower (`cannot determine truth value of Relational`). Dynamo graph-breaks there; the
transformer still compiles.

`pfn_seg.py` warm-start (`train.checkpoint`) is now **tolerant**: it loads only tensors whose
name+shape match the current model (`strict=False`), so a checkpoint from before these changes
still warm-starts. Switching to a pretrained encoder changes `image_embed` (Q²→feature_dim) and
adds frozen `image_encoder.*` weights — those keep their fresh/pretrained values; shared weights
(mask_embed, pos_embed, thinking, transformer, decoder) transfer. Logs loaded/skipped/fresh counts.

## 2026-06-14 — pfn_seg_2d: `resolution` param decouples output grid from encoder input dim

`src/models/pfn_seg_2d.py`, `experiments/2d/pfn_seg.py`, `experiments/2d/eval.py`,
`configs/experiment/2d/pfn_seg.yaml`.

Replaced `arch.patch_size` with `arch.resolution` (+ `arch.input_patch_size`, default 8).
`resolution` = patches per side = output grid `Hp`; effective patch size `P = image_size //
resolution` (128//16 = 8). Each native `P×P` patch is now resized to `Q×Q` (`input_patch_size`)
via `F.interpolate` in `patchify` before embedding, so `image_embed`/`mask_embed` always take a
fixed `Q²` input regardless of `P`. This lets the output resolution change (e.g. res 8→32, P
16→4) while the patch encoder stays fixed at 8×8.

`eval.py` derives `resolution`/`input_patch_size` from old `patch_size` checkpoints for
back-compat. Run name now `pfn_seg_R{res}q{Q}_...`.

## 2026-06-12 — pfn_seg_2d eval backend: fixes, config consolidation, warm-start, comparison

`experiments/2d/eval.py`, `experiments/2d/pfn_seg.py`, `configs/experiment/2d/base.yaml`,
`configs/experiment/2d/pfn_seg.yaml`. Deleted `configs/experiment/2d/pfn_seg_eval.yaml`.

Finished and validated the `pfn_seg_2d` eval backend added on 2026-06-11.

**`src` package-shadowing fixes** — `common.py` puts `/home/dpxuser/ic_segmentation` on
`sys.path`, whose `src` package shadows patch_icl's. Both are regular packages, so only
one wins per process. Two failures resolved (pfn_seg runs only):
- `ModuleNotFoundError: src.datasets` — `common`'s `from src.datasets.medsegbench` import
  resolved to ic_segmentation (no `datasets`). Fix: when `"pfn_seg"` is in `sys.argv`,
  pre-import patch_icl's `src.datasets.medsegbench` before `common`, caching the right `src`.
- `ModuleNotFoundError: src.models.pfn_seg_2d` — ic_segmentation's `src.models` lacks it.
  Fix: load `pfn_seg_2d.py` directly by file path via `importlib.util` (torch-only deps).
- Both guards key off `"pfn_seg"` in argv, so universeg / feature_sim runs are untouched.

**Config consolidation** — deleted the 3-line `pfn_seg_eval.yaml`; eval now reads
`base.yaml` with overrides. Added `eval.checkpoint: null` to `base.yaml` (Hydra rejects
undeclared CLI keys). New invocation:
```bash
python experiments/2d/eval.py model=pfn_seg_2d eval.checkpoint=results/2d/<run>.pt
```

**Warm-start / resume** — `pfn_seg.py` gains `train.checkpoint`: loads weights (bare
state_dict or `{"model": ...}`, strips `_orig_mod.`) into a fresh model before training.
Used to resume the pre-format-change `results/2d/best.pt` for one epoch (54 741 samples,
val Dice 0.4348), resaved in the new format as `results/2d/pfn_seg_resumed.pt`.

**Validation** — full val eval of `pfn_seg_resumed.pt`: mean Dice 0.4393 over 1141 samples
(matches trainer within sampling noise), 68.20 GFLOPs, ~3.6 ms/item. See `results/report.md`
for the pfn_seg_2d vs UniverSeg comparison (mean Dice 0.524 vs 0.334 over 35 datasets).

## 2026-06-11 — pfn_seg: augmentations, compile flag, progress bars

`experiments/2d/pfn_seg.py`, `src/models/pfn_seg_2d.py`, `configs/experiment/2d/pfn_seg.yaml`, `configs/augmentations/medsegbench.yaml` (new), `experiments/2d/plot_aug.py` (new).

**`torch.compile` made optional (`arch.compile: false`)**:
- `TransformerEncoderLayer.forward` had `@torch.compile(dynamic=True)` and `_newtonschulz5_batched` had `@torch.compile` baked in.  On first run, 6 layers × 2 (fwd+bwd) Triton compilations caused a silent multi-minute hang that looked like a freeze.
- Both decorators removed.  Compile is now opt-in: `arch.compile=true` wraps the full model and `_newtonschulz5_batched` via `torch.compile(dynamic=True)` after construction.

**GPU augmentations (`augment()` in `pfn_seg.py`)**:
- New `configs/augmentations/medsegbench.yaml` — canonical light-aug config: geometric (hflip p=0.5, vflip p=0.5, rotate ±20° p=0.5) and intensity (brightness Δ0.15, contrast ×[0.8–1.2], gamma [0.75–1.33], noise σ=0.04).
- Geometric ops applied **to context pairs only** (joint image+mask via `F.affine_grid`/`F.grid_sample`) so the query GT is never misaligned; intensity ops applied independently to all images.
- All ops are batched GPU tensor operations (`torch.where`, `flip`, `affine_grid`/`grid_sample`); no per-sample Python loops.
- `pfn_seg.yaml` gains a `defaults` list loading `augmentations/medsegbench@aug`; `hydra.main` config_path widened to `../../configs` so Hydra resolves the group.  Disable at runtime with `aug.enabled=false`.

**Progress bars**:
- `train_epoch`: `tqdm` per-batch with running `loss` postfix.
- `run_eval`: existing `tqdm` extended with running `dice` postfix.
- `main` epoch loop: outer `tqdm` with `loss / lr / best` postfix updated after each epoch and eval.
- All `print()` calls inside the loop replaced with `tqdm.write()`.

**Visualisation**: `experiments/2d/plot_aug.py` — loads a small multi-dataset batch and saves a 6-column grid (orig ctx, orig mask, aug ctx, aug mask, orig query, aug query) to `results/datasets/medsegbench_aug.png`.

## 2026-06-11 — ImagePFN: nanoTabPFN-style in-context segmentation model

New files:
- `src/models/pfn_seg_2d.py` — `ImagePFN` model
- `experiments/2d/pfn_seg.py` — training + eval script
- `configs/experiment/2d/pfn_seg.yaml` — config
- `docs/tabpfn/nanotabpfn.md` — paper + repo summary

`ImagePFN` adapts all modded-nanoTabPFN techniques to 2D image segmentation:
- **Dual-axis transformer**: feature-axis attention (spatial, within each image) + sample-axis
  attention (cross-image, per patch position). Tensor layout `(B, rows, cols, e)` where
  rows = images and cols = patches.
- **Asymmetric masking**: query image attends only to thinking + context rows; mirrors TabPFN's
  train/test split in sample-axis attention.
- **Thinking rows**: `n` learnable row tokens prepended to the sequence, broadcast across all
  patch positions; treated as train rows in sample-axis attention.
- **Residual decay**: input to block i scaled by `residual_decay^i` (default 0.95).
- **LowerPrecisionRMSNorm**: pre-norm RMSNorm that upcasts to fp32 for bf16/fp16 inputs.
- **LAWA**: rolling buffer of last `K=10` checkpoints, averaged at eval time only; training
  weights unchanged.
- **Muon optimizer**: applies to all 2D weight matrices inside `transformer`; AdamW + cosine
  LR schedule for everything else (embeddings, norms, decoder).
- Images are patchified (P=8, N=256 patches per image) and normalized using context-image
  statistics per batch (mirrors TabPFN per-column normalization).
- Decoder outputs per-patch logits for query row; upsampled to native resolution for BCE loss.

## 2026-06-11 — feature_sim.py fixes (vectorisation, ensemble averaging, timing)

`experiments/2d/feature_sim.py`

- **Vectorised feature extraction**: replaced `extract_features` + `_pool2d` (which assumed B=1
  via `.squeeze(0)` and was called in an O(B×K) Python loop) with `extract_features_batch`,
  which operates on full `(N, C, H, W)` tensors and concatenates on `dim=1`. The main eval
  loop now calls it once for targets `(B, C', os, os)` and once for contexts `(B*K, C', os, os)`,
  then reshapes; context mask downsampling likewise vectorised via a single
  `F.adaptive_{avg,max}_pool2d` on `(B*K, 1, H, W)`.

- **Probability averaging in `batch_tabpfn`**: was averaging logits before softmax
  (`softmax(sum_logits / n_estimators)`), which differs from the standard ensemble approach.
  Now applies softmax per estimator and accumulates probabilities, dividing at the end.

- **Split timing**: `inference_times` (a single conflated number) replaced by separate
  `encode_times` and `tabpfn_times` lists. Summary now logs `time/encode_ms`,
  `time/tabpfn_ms`, and `time/total_ms` to wandb and prints each separately.

- **balance_ratio warning**: prints a warning at run start when `balance_ratio` is set,
  because it disables batched TabPFN and falls back to serial per-sample inference.

- **FLOPs message**: clarified to say "per sample" (the measured value is for 1+K images,
  not for the full eval batch).

## 2026-05-29 — VoComniNNUNetEncoder (PlainConvUNet/VoComni)

`src/models/encoders/vocomni_nnunet.py` — new encoder wrapping PlainConvUNet
(nnUNet CNN backbone) from the VoComni_nnunet.pt checkpoint (supervised on 20K
CT volumes, 21 classes). 6 stages, channels 32→64→128→256→320→320, total_stride=32.
Loads only the `encoder.*` prefix from the full model checkpoint.
torch.compile default True (~2× speedup vs eager).

`run.py` — added `--encoder vocomni_nnunet`, `--vocomni_nnunet_ckpt`,
`--vocomni_nnunet_compile` args.

## 2026-05-29 — VoComniEncoder (SwinUNETR/VoCo)

`src/models/encoders/vocomni.py` — new encoder wrapping MONAI SwinUNETR with
VoCo/VoComni pretrained weights.  Returns 5 feature levels at strides 2–32.
`skip_channels=[fs, 2fs, 4fs, 8fs]`, `bot_features=16fs`, `total_stride=32`.
Checkpoint loading handles `state_dict`/`student`/`module.`/`backbone.` prefixes.
Supports feature_size 48 (Base, 72M), 96 (Large, 290M), 192 (Huge, 1.2B).

`experiments/feature_similarity/run.py` — added `--encoder vocomni`,
`--vocomni_ckpt`, `--vocomni_feature_size` CLI args; vocomni uses the same
image-only combined-batch path as threedino.

## 2026-05-27 — val_loader pre-filtered to val_items_per_class

`experiments/nninteractive/train.py`.

`val_loader` now wraps a `torch.utils.data.Subset` that pre-selects at most
`val_items_per_class` indices per class from `ds_val.samples` (deterministic,
class-ordered). Previously the full val dataset was loaded (1686 samples / 211
batches) and a batch-level guard inside `validate()` skipped batches only when
**all** items were saturated — so the encoder + attention still ran for items
that were already counted. With the Subset, the loader contains exactly the items
that will be scored (~1410 samples / ~177 batches for 47 classes × 30 items).

Also fixed: `torch.compile(mode="reduce-overhead")` replaced with
`mode="default"` to avoid CUDA-graph segfaults caused by `cascade_registers`
changing from `None` (level 0) to a tensor (levels 1+) across calls to the
shared attention module.

## 2026-05-27 — feature_level list support in extract_features

`experiments/multilevel/train.py`, `configs/experiment/nninteractive.yaml`.

`extract_features` now accepts a list of level indices in addition to `'all'` and a single
integer. This lets a run select any subset of encoder levels without computing or storing
the unused ones.

```
feature_level: [2, 3, 4]   # skip large early stages; concat levels 2+3+4 only
feature_level: 4            # bottleneck only
feature_level: all          # concat all num_stages outputs (original behaviour)
```

Hydra deserialises YAML lists as `omegaconf.ListConfig`, so the isinstance check uses
duck-typing (`not isinstance(level, str) and hasattr(level, "__iter__")`) rather than
`isinstance(level, (list, tuple))`.

`configs/experiment/nninteractive.yaml` updated to `feature_level: [2, 3, 4]` (drops
skip0=32ch@128³ and skip1=64ch@64³, which account for 94% of skip-feature VRAM at B=8 fp16).
`embed_dim` drops from 800 → 704 (128+256+320); auto-detected via dummy forward in `main()`.

---

## 2026-05-27 — torch.compile integration + CUDA graph + RoPE fixes

`experiments/nninteractive/train.py`, `experiments/multilevel/train.py`, `src/rope3d.py`.

### torch.compile integration

Added `compile_model` flag to both training scripts (config key `train.compile_model`).
When enabled on a `shared_weights=True` run, compiles `model.shared_level` (the single
`PatchICLAttention` instance) with `mode="reduce-overhead"` (CUDA graph capture).
The compiled module is kept in a local `compiled_attn` variable — `model.state_dict()`,
optimizer parameters, and checkpoint saving are unaffected.

Default: `compile_model: true` in `nninteractive.yaml`; `false` in `multilevel.yaml` (opt-in).

### CUDA graph cascade_regs fix

`cascade_regs = regs.detach()` shares storage with the CUDA graph's output pool.
Passing that tensor as input to the *next* level's CUDA graph (a sibling graph) triggers:

    RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run

Fix: `.clone()` after every cascade_regs assignment so the tensor lives in normal GPU
memory and is no longer a view of the CUDA graph output pool:

```python
cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
if compiled_attn is not None:
    cascade_regs = cascade_regs.clone()
```

Applied at all 4 call sites in each train.py (L0 + L1, training + validation).
`torch.compiler.cudagraph_mark_step_begin()` is also called before each compiled forward
as a step-boundary hint for the CUDA graph tree.

### RoPE real-space rotation fix

`src/rope3d.py` `_rotate()` previously used `torch.view_as_complex` / `view_as_real`,
which Torchinductor does not support:

    UserWarning: Torchinductor does not support code generation for complex operators.
    Performance may be worse than eager.

Replaced with equivalent real-space arithmetic:

```python
x0' = x0·cos − x1·sin
x1' = x0·sin + x1·cos
```

No correctness change (fp32 max diff 4.77e-7; fp16 9.77e-4 — normal fp16 rounding).
The new path is fully fuseable by Triton.

---

## 2026-05-27 — Attention compile benchmark (torch.compile on PatchICLAttention)

`/tmp/compile_bench.py` (inline benchmark, not committed).

Benchmarked `torch.compile` modes on `PatchICLAttention` fwd+bwd using `.venv311`
(Python 3.11 + triton 3.2.0 — required for max-autotune; Python 3.12 breaks triton).

Config: B=4, K=1, N=M=512 (dense 8³), embed_dim=800, dim=256, L=8.

| Method | Latency | Speedup | ΔVRAM |
|--------|--------:|--------:|------:|
| baseline eager | 87.8 ms | 1.00× | — |
| **compile reduce-overhead** | **13.8 ms** | **6.34×** | **−499 MB** |
| compile max-autotune | 17.2 ms | 5.11× | −499 MB |

`reduce-overhead` wins (CUDA graph capture); `max-autotune` spends 5 min tuning Triton
kernels for a slightly worse result. Recommendation: add
`torch.compile(model.shared_level, mode="reduce-overhead")` to `train.py`.

## 2026-05-27 — NNInteractive experiment

`experiments/nninteractive/train.py` (new), `configs/experiment/nninteractive.yaml` (new).

New training experiment: same multilevel PatchICLAttention stack as `experiments/multilevel/`
but driven by the frozen `NNInteractiveEncoder` (90 M params) instead of STUNetEncoder.

Key architectural difference: context images are encoded **with their ground-truth masks**
(`encode_context(encoder, ctx_imgs, ctx_masks)`) so the encoder features are already
mask-conditioned at the backbone level. Target images are encoded with a zero mask.
The downstream label injection in PatchICLAttention is kept unchanged.

`embed_dim=800` (feature_level=all, 5 stages) vs 992 for STUNet-base.
MultilevelICL 2.48 M trainable params (encoder frozen).

Usage:
    python experiments/nninteractive/train.py
    python experiments/nninteractive/train.py model.nnint_mask_injection=separate
    python experiments/nninteractive/train.py model.nnint_ckpt=/local/path/...

## 2026-05-27 — NNInteractive pretrained encoder wrapper

`src/models/encoders/nninteractive.py` (new).

`NNInteractiveEncoder` wraps the pretrained ResidualEncoder from the nnInteractive v1.0
checkpoint (383 M params, 8-channel input: 1 image + 7 interaction slots).
Implements the same `(skip_channels, bot_features, total_stride, forward(imgs, masks))`
interface as `STUNetEncoder` so it can drop in as an encoder swap.

Two mask injection modes:
- **ch1** — pack `[image, mask, 0, 0, 0, 0, 0, 0]` as the 8-channel input, using the
  model's native "current segmentation" channel.
- **separate** — image in ch0 only; mask encoded by a SAM-style 3-D CNN
  (`_Mask3DEncoder` from stunet.py) and fused additively at the bottleneck.

`num_stages` controls depth (default 6 → 32× stride; 5 → 16× recommended for 64³/128³).
Encoder weights are frozen by default. nnInteractive installed as editable package into `.venv`.

## 2026-05-27 — Fix seeds3d segfault (float32 → uint8)

`scripts/synth_labels/generate.py`.

`seeds3d` was crashing with a segfault (core dump) on every call, causing forked worker processes to die silently and the pool to hang indefinitely waiting for results. Root cause: `python_3d_seeds` (a pybind11 wrapper around OpenCV's SEEDS) requires **uint8** input normalised to `[0, 255]`; passing `float32` causes an out-of-bounds memory access in the C++ layer. Fix: normalise the volume with `((vol - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)` before calling `sv.iterate()`.

## 2026-05-26 — MRI normalisation fix + bbox cache parallelisation

`scripts/convert_to_npy.py`, `src/totalseg_dataloader_incontext.py`.

**MRI normalisation (`convert_to_npy.py` `_normalise_mri`)**:
- Lower clip bound changed from hard `0` to the 0.5th percentile of non-zero foreground voxels (matches the 2D zopt dataloader in `ic_segmentation`). Upper bound remains p99.5. Z-score step retained.
- Motivation: `0` is an arbitrary floor that can include background signal and biases the clip range for certain MRI protocols. Using p0.5 of foreground is protocol-agnostic.
- **Pre-resized MRI `.npy` files must be regenerated**: `python scripts/convert_to_npy.py --data .../totalsegmri --modality mri --size 128 128 128 --overwrite`

**Bbox cache parallelisation (`totalseg_dataloader_incontext.py`)**:
- Added module-level `_bbox_for_subject(root, subj)` (picklable worker) — same pattern as `_adj_for_subject`.
- `_load_or_build_bbox_cache` now uses `ProcessPoolExecutor(max_workers=16)` instead of a sequential loop. Critical for TotalSegMRI where native label volumes can exceed 1000 × 1280 × 1900 voxels.
- Existing `.bbox_cache_*.pkl` files are reused as-is; only fresh builds benefit.

**MRI crop size fix (`totalseg_dataloader_incontext.py` `_load_crop` / `_load_crop_multi`)**:
- CT dataset: perfectly isotropic at 1.5 mm → `T × sp_min = 192 mm` always (consistent).
- TotalSegMRI: spacing 0.17–28 mm, up to 120× anisotropy → old formula gave physical crops as small as **21 mm** (zoomed in).
- Fix: `phys_ref = max(T * sp_min, T * 1.5)` — clamps the physical crop to at least 192 mm regardless of how fine the in-plane voxels are. High-res subjects (0.2 mm in-plane) now crop ~960 voxels in-plane and downsample to 128, giving the same physical context as CT.

## 2026-05-26 — `not_benchmark` class splits in resolve_classes

`data/totalseg_classes.py`

- Added `"not_benchmark"` → CT classes in `ALL_CLASSES[:117]` that are **not** in `BENCHMARK_CLASSES` (complement train set for CT).
- Added `"benchmark_mri"` → `MRI_BENCHMARK_CLASSES`.
- Added `"not_benchmark_mri"` → MRI classes in `MRI_ALL_CLASSES` that are **not** in `MRI_BENCHMARK_CLASSES` (complement train set for MRI).
- Use in config: `data.train_classes=not_benchmark data.val_classes=benchmark` to train on held-out classes and evaluate on benchmark.

## 2026-05-25 — Augmentation benchmark and synth_equiv preset

`configs/augmentations/synth_equiv.yaml`, `experiments/multilevel/benchmark_aug.py`.

- **`synth_equiv.yaml`**: new aug preset matching synth-aug strength applied to real labels (task.affine ±25°, scale [0.90–1.50], translate ±0.20, elastic p=0.8; intensity BC+noise+blur matching synth ops but no gamma/sim-low-res). Use with `train.aug_preset=synth_equiv data.p_synth=0.0` to isolate augmentation strength from data-type effect.
- **`experiments/multilevel/benchmark_aug.py`**: comprehensive runtime benchmark comparing custom PyTorch, MONAI, and batchgenerators augmentation pipelines. Sweeps presets × K values, reports per-transform breakdown, worker throughput estimates. Key findings at 128³ (20 reps, n_workers=20):

  | Preset | ms/vol (K=1) | ms/vol (K=3) | batch(K=1, 8 items) |
  |--------|-------------|-------------|---------------------|
  | nnunet (no elastic) | 68 ms | 32 ms | 26 ms |
  | multiverseg | 152 ms | 112 ms | 65 ms |
  | synth_equiv | 216 ms | 233 ms | 102 ms |
  | synth pipeline (p_synth=1) | 419 ms | 465 ms | 143 ms |

  - **Custom PyTorch vs libraries**: affine ≈ MONAI (1.03–1.28× at 128³, batching advantage lost to grid_sample cost), elastic custom is 2–3.5× faster than MONAI, batchgenerators `augment_spatial` is 17–20× slower.
  - **Bottleneck**: affine (~128–140 ms/vol) + elastic (~150 ms/vol) dominate at 128³. nnunet avoids elastic (p=0) and is 3–4× faster.
  - **Synth pipeline**: independent per-copy aug is the training bottleneck at 128³; with 20 workers K=3 gives only 45 sps → 180ms/batch, likely bottlenecking GPU pipeline.
  - **Recommendation**: keep custom PyTorch (already optimal); consider `aug_preset=nnunet` for faster iterations; use `synth_equiv` + `p_synth=0.0` to test augmentation-strength hypothesis.
  - Output saved to `results/aug_benchmark/aug_benchmark_*.json`.

## 2026-05-25 — TotalSegMRI support (convert, synth labels, eval)

`data/totalseg_classes.py`, `data/benchmark_classes.py`, `scripts/convert_to_npy.py`, `scripts/synth_labels/generate.py`, `scripts/eval.py`, `configs/config.yaml`, `configs/cluster/nfs.yaml`, `configs/cluster/meta.yaml`.

- **4 MRI-only classes** appended to `ALL_CLASSES` (indices 118–121): `lung_left`, `lung_right`, `intervertebral_discs`, `vertebrae`. Zero-indexed at the tail so all existing CT label indices are unchanged.
- **`convert_to_npy.py`**: added `--modality {ct,mri}` flag. MRI normalisation clips to `[0, p99.5(foreground)]` then z-scores with foreground mean/std (per-volume, no global constants). Output is still named `ct.npy` so the dataloader needs no changes. Fixed `get_zooms()` tuple vs ndarray bug by using `[float(x) for x in ...]`.
- **`synth_labels/generate.py`**: added `--modality {ct,mri}` flag, routing to `paths.totalsegmri` and loading `mri.nii.gz` with per-volume normalisation as the NIfTI fallback.
- **`eval.py`**: added `--dataset {totalseg,totalsegmri}` flag. `resolve_totalseg_root()` now uses an exact-key regex (`(?<!\w)totalseg:` vs `totalsegmri:`) to prevent prefix collision. Defaults to `MRI_BENCHMARK_CLASSES` when `--dataset totalsegmri`.
- **`MRI_BENCHMARK_CLASSES`**: curated 18-class MRI benchmark covering organs, whole-lung, spine (individual + merged), vasculature, and bones.
- Config: `paths.totalsegmri` added to `config.yaml`, `nfs.yaml`, and `meta.yaml`; `data.dataset: totalseg` default added.

## 2026-05-25 — Shared weights, physical scale injection, and role embeddings for MultilevelICL

`experiments/multilevel/model.py`, `experiments/multilevel/train.py`, `experiments/feature_attention/model.py`, `configs/experiment/multilevel.yaml`, `src/totalseg_dataloader_incontext.py`, `scripts/convert_to_npy.py`.

**Shared weights (`model.shared_weights`)**:
- `MultilevelICL` gains `shared_weights: bool` — when true, a single `PatchICLAttention` instance is used for all spatial levels instead of one per level. Works only with `pos_encoding="rope3d"` (learned PE has grid-size-dependent embedding tables; an assert guards this).
- Checkpoint key remapping on load handles both transition directions: `levels.0.*` → `shared_level.*` (per-level → shared, uses L0 weights as init) and `shared_level.*` → `levels.{i}.*` (shared → per-level, broadcasts to all slots). Happens before the existing shape-filter step so no other changes are needed.

**Physical scale injection (`model.use_scale_embed`)**:
- `ContinuousScaleEncoding` added to `experiments/feature_attention/model.py`: maps `log(scale_mm)` → `dim`-dimensional embedding via log-spaced learnable sinusoidal frequencies (matches `patch_icl_v3`). Added to `PatchICLAttention` behind `use_scale_embed` flag (default off).
- `spacings.json` at the dataset root stores `{"s0000": {"spacing": [dx,dy,dz], "shape": [D,H,W]}, ...}` for all subjects. `convert_to_npy.py` now writes it incrementally (merged with any existing entries). A standalone header-only extraction ran in 16 s for 1228 subjects.
- `TotalSegInContextDataset._load_spacings()` reads `spacings.json` at init and converts to effective mm/voxel: pre-resized path scales by `max_native_shape / T`; crop path uses native spacing as-is. Falls back to 1 mm isotropic if the file is absent.
- `incontext_collate_fn` stacks `"spacing"` → `(B, 3)`. `process_batch` and `validate` compute `scale_mm = (image_size / grid_size) * mean(spacing)` per level and pass it to `model[i](...)`.
- Bug fix: `validate()` sparse-level model call was silently omitting `scale_mm=scale_mm`; fixed.

**Role embeddings (`model.use_role_embed`)**:
- `PatchICLAttention` gains three zero-initialised parameters when `use_role_embed=True`: `tgt_type_embed (1,1,dim)`, `ctx_type_embed (1,1,dim)`, and `ctx_idx_embed Embedding(max_context_size, dim)`. The type embeddings distinguish target from context tokens; `ctx_idx_embed[k]` is added to tokens from context image `k`, letting the model track which context image each token came from. Injected at step 2c (after scale encoding, before label injection), K inferred as `ctx.shape[1] // tgt.shape[1]`. Zero-init means the model starts identically to a checkpoint trained without this flag.
- New config keys: `use_scale_embed: false`, `use_role_embed: false`, `max_context_size: 8`.

## 2026-05-24 — MultilevelICL benchmark adapter

Added `src/benchmark_models/multilevel.py` — `MultilevelICLAdapter` wrapping `MultilevelICL` for use in `scripts/eval.py`.

- Loads a multilevel checkpoint and reconstructs both the frozen `STUNetEncoder` and `MultilevelICL` from the stored config.
- Runs N-level coarse-to-fine inference: L0 is a dense forward over the full grid; finer levels sample `NP` sparse patches guided by the previous level's upsampled prediction (`predicted_entropy` at eval time — no GT available for error-based modes).
- Upsamples the final grid prediction back to the input spatial size for fair Dice comparison against 128³ models.

## 2026-05-22 — Multilevel train/val loop generalised to N levels

`experiments/multilevel/train.py`, `experiments/multilevel/model.py`, `experiments/feature_attention/model.py`, `configs/experiment/multilevel.yaml`.

**`MultilevelICL` model changes:**
- `MaskCNN` — shared 3D ConvNet (same-padding, grid-size agnostic) that encodes a soft/binary mask into per-voxel embeddings, interpolated trilinearly to each level's resolution. Replaces the old scalar avg-pool label path when `mask_cnn_dim > 0`.
- `num_registers` — learnable register tokens appended to context K/V and cascaded between levels; `detach_cascade_registers` controls whether gradients cross level boundaries.
- `append_zero_attn` — adds a null K/V slot to cross-attention so target patches can "abstain" from retrieving context.
- `output_dim` added to `PatchICLAttention` (separate from `label_dim`): head always predicts a 1-dim binary mask regardless of label embedding size.

**Training loop:**
- Train and validation loops now handle any number of levels (previously hardcoded to 2); loss weights configurable per level via `train.loss_weights`.
- `_encode_ctx_labels` helper unifies CNN-embedded and scalar label injection paths.
- `soft_labels_train` / `soft_labels_eval` separate: avg-pool float labels during training, binarised at inference.
- New config keys: `mask_cnn_dim`, `soft_labels_train`, `soft_labels_eval`, `num_registers`, `append_zero_attn`, `detach_cascade_registers`.

**`experiments/feature_attention/train_dice46.py`** — standalone training script for `PatchICLAttention` that uses Dice loss instead of BCE (experimental variant).

## 2026-05-21 — 3D RoPE for PatchICLAttention + multilevel patch sampling rewrite

`src/rope3d.py`, `experiments/feature_attention/model.py`, `experiments/multilevel/model.py`, `experiments/multilevel/train.py`.

**3D RoPE (`src/rope3d.py`):**
- `build_rope_cache_3d(max_pos, dim, num_heads)` — builds a `(max_pos, dim//2)` frequency cache; `per_axis` rounded to the nearest multiple of `head_dim` (same fix as in `src/rope.py`).
- `apply_rope_3d(x, coords, cache)` — applies RoPE to a `(B, N, dim)` tensor given integer `(B, N, 3)` d/h/w coordinates; works for any token subset (dense or sparse, no grid-size assumption).

**`PatchICLAttention` gains `pos_encoding="rope3d"`:**
- RoPE applied inside `_mha()` directly to Q and K after projection; falls back gracefully to no PE if coords are not supplied.
- `rope_max_pos` parameter overrides the cache extent (default: `max(grid_size)`).

**`MultilevelICL` switches to RoPE:**
- All levels use `pos_encoding="rope3d"` with a shared cache keyed by `global_max_pos = max(max(grid_size) for all levels)`.
- Removed the old `coord_projs` (`nn.ModuleList` of `Linear(3, embed_dim)`) that injected normalised coordinates externally.

**Patch sampling rewrite (`experiments/multilevel/train.py`):**
- `_gumbel_topk(weights, n, temperature)` — stochastic top-n via Gumbel noise; normalises weights per batch item, replaces the old deterministic entropy/fg/border split.
- `sample_target_patches` now accepts a `mode` argument: `gt_previous_pred_error` (|pred − GT|), `gt_foreground_entropy_balanced` (0.5·GT + 0.5·H(GT)), `predicted_entropy` (H(pred)); controlled by `data.target_sampling`.
- `sample_context_patches` uses `gt_foreground_entropy_balanced` averaged across K context masks; replaces the old fg + border morphological sampling.
- `grid_coords_3d` replaces `make_coords_3d` — returns integer voxel coords (long) rather than normalised floats.

## 2026-05-23 — Quality benchmark for in-context segmentation (eval.py)

Added a segmentation-quality benchmark comparing native models against SOTA.

- `scripts/eval.py` — new evaluation script; argparse CLI; measures per-class and mean Dice on TotalSegmentator test split; saves `results/eval_*.json` and `results/eval_*.csv`.
- `data/benchmark_classes.py` — curated 16-organ class list covering a range of sizes and difficulty.
- `src/benchmark_models/base.py` — `InContextModel` ABC (`predict(target, context_imgs, context_masks) → binary mask`).
- `src/benchmark_models/native.py` — adapters for `ViTInContext3D` (`NativeViT`) and `ResEncInContext3D` (`NativeResEnc`), loading from checkpoint.
- `src/benchmark_models/medverse.py` — adapter for Medverse (local repo at `/nfs/norasys/notebooks/camaret/repos/Medverse`); applies per-volume min-max normalisation; adds channel dim to context masks.
- `src/benchmark_models/__init__.py` — `load_model(name, ...)` registry.

Methods included: native_vit, native_resenc, medverse. UniverSeg (2D-only) and Show&Segment (CVPR 2025, no public code) excluded.

## 2026-05-22 — Fix head-boundary misalignment in 3D RoPE

`src/rope3d.py`, `experiments/feature_attention/model.py`.

`build_rope_cache_3d` previously computed `per_axis = ((dim // 3) // 2) * 2`, which
for `dim=512` gives `per_axis=170` — not a multiple of `head_dim=64`. After the head
split, heads 2 and 5 straddled two axis blocks (d+h and h+w respectively), receiving
mixed-axis rotations.

Fix: added `num_heads` parameter; `per_axis` is now rounded down to the nearest
multiple of `head_dim`. For `dim=512, num_heads=8`: `per_axis=128` (64 pairs/axis),
heads 0–1→d, 2–3→h, 4–5→w, 6–7 unrotated. Default `num_heads=1` preserves old
behaviour for callers that don't pass it. Call in `PatchICLAttention.__init__` updated
to pass `num_heads=num_heads`.

## 2026-05-20 — Synth pipeline fixes in _get_synth_item

Three issues fixed in `TotalSegInContextDataset._get_synth_item`
(`src/totalseg_dataloader_incontext.py`):

- **`use_crop` ignored** (main bug): synth items always loaded the whole-body
  pre-resized 128³ image, giving ~16× worse effective resolution than real-data
  items when `use_crop=True`. Fix: when `use_crop=True`, load native `ct.npy` +
  native synth label file, compute the centroid of the picked supervoxels' union,
  and crop a `T³` patch around it with the same jitter logic as `_load_crop`.
- **`aug_cfg` not guarded**: `self.aug_cfg.synth` raised `AttributeError` when
  no augmentation config was provided (valid for debug runs). Fix: guard with
  `if self.aug_cfg is not None and self.aug_cfg.enabled`, falling back to
  unaug'd identity copies.
- **Slow-path CT preferred `ct.nii.gz` over `ct.npy`**: native numpy is faster
  and avoids the nibabel stack. Fix: check for `ct.npy` first, fall back to
  `ct.nii.gz` only if absent.

## 2026-05-20 — Soft GT for avgpool downsampling

In `process_batch` (`experiments/feature_attention/train.py`), the `> 0` threshold
on `gt_loss` was discarding the continuous coverage fraction produced by avgpool,
making `mask_pool: avg` identical to `max` in practice.

**Fix**: `gt_loss` is now the raw avgpool output (soft float in [0,1]); a separate
`gt_bin = (gt_loss > 0).float()` is kept for norm_dice. Context labels stay binary
(`> 0`) since the model binarizes them internally anyway (`label_dim=1` path uses
`nn.Embedding(2, dim)`). BCE with soft targets is valid and provides proportional
gradient weighting by patch coverage fraction.

## 2026-05-18 — Fix synth elastic augmentation bottleneck

`apply_synth_aug` was generating a full-resolution `(3, D, H, W)` noise field
and smoothing it with `_gaussian_smooth_3d_field` (sigma 8–15 → kernel size
33–61).  For 128³ volumes that's three depthwise conv3d passes on 6M elements
each call, done K+1 times per item.  This starved the DataLoader prefetch
queue and caused periodic training stalls whenever workers drew large sigma.

**Fix** (`src/augmentations.py`, `apply_synth_aug`): replaced the
full-res-randn + Gaussian-conv path with a coarse-grid approach — generate
displacement at `(D/sigma, H/sigma, W/sigma)` resolution and upsample
trilinearly.  Equivalent smooth frequency, ~60× cheaper for sigma=8–15.
The real-label path was unaffected (uses `apply_task_aug` where elastic p=0).

## 2026-05-18 — SegGPT-style random coloring + synthetic label support

**Random coloring** (`--random_coloring`, on by default):
- `TotalSegInContextDataset._apply_coloring`: finds label ids shared across all
  masks in a sample, samples one random RGB per id, returns `(3,D,H,W)` float32
  tensors for `label` and `context_out` (same palette across target + context).
- `PatchICLAttention` gains `label_dim` (1=binary discrete, 3=RGB). Label injection
  uses `nn.Linear(label_dim, C, bias=False)` for RGB so background (black=0) injects
  nothing; output head predicts `label_dim` values per patch.
- Training uses smooth-L1 loss for RGB; metrics use L2 norm of predicted colour as
  scalar probability for AUROC/norm-dice. Validation stays binary (val dataset
  never colorises).

## 2026-05-18 — Synthetic label support in feature_attention/train.py

Added `--synth_method`, `--synth_unions`, and `--p_synth` CLI args to
`experiments/feature_attention/train.py`.  These are forwarded to
`TotalSegInContextDataset` for the training split (val unchanged), enabling
the same supervoxel-based synthetic augmentation path already supported in
`scripts/train.py`.

## 2026-05-18 — PatchICLAttention improvements from TabPFN comparison

Rewrote `experiments/feature_attention/model.py` with five changes derived from
comparing against TabPFN v3's ICL stage:

- **K/V normalization (bug fix)**: cross-attention now pre-norms K and V via
  per-layer `kv_norms` (RMSNorm). Previous code normalised only Q, breaking
  pre-norm invariance as encoder features bled into K/V at each layer.
- **Context self-attention**: each layer now runs a full transformer block
  (SA + FFN) on context tokens before the cross-attention. This lets context
  patches interact across K images before being retrieved, analogous to how
  TabPFN's ICL train rows self-attend through train-only K/V.
- **Log-n query scaling**: cross-attention Q is pre-scaled by
  `log(M)/log(n_base)` before `F.scaled_dot_product_attention` (which applies
  `1/sqrt(D)` internally). Calibrates softmax temperature for large context
  sequences (M = K×8³ = 512+). Configurable via `log_n_base` (default 512).
- **Retrieval head K projection**: added `ret_k_proj` separate from `ret_q_proj`,
  decoupling the similarity space from the representation space (mirrors
  TabPFN's `ManyClassDecoder`).
- **Default input_norm → rmsnorm**: encoder features have per-channel scale
  that varies across the 4 encoder levels; normalising at input stabilises
  Q/K dot products.

Switched from `nn.MultiheadAttention` to manual projections +
`F.scaled_dot_product_attention` throughout, enabling Flash Attention and
giving full control over scaling and norming. `train.py` gains
`--no_ctx_self_attn`, `--no_log_n_scaling`, `--log_n_base` flags.

## 2026-05-15 — pluggable encoder + STU-Net

Refactored the encoder out of `resenc_in_context.py` into `src/models/encoders/`:

- `src/models/encoders/resenc.py` — `ResEncEncoder`: nnUNet `ResidualEncoderUNet`
  encoder wrapper (4 stages, 8× downsample, 2-channel image+mask input).
  Identical behaviour to the original inline encoder.

- `src/models/encoders/stunet.py` — `STUNetEncoder`: 6-stage STU-Net image encoder
  (`_ImageEncoder`, conv_blocks_context naming preserved for pretrained weight
  compatibility) + SAM-style 3-D mask encoder (`_Mask3DEncoder`, stride-2 CNN),
  fused at the bottleneck (additive or concat+proj).
  Supports `small | base | large | huge` variants.
  Pretrained TotalSegmentator weights loadable via `stunet_pretrained=<path>`.

- `src/models/resenc_in_context.py` — accepts `encoder_name` ("resenc" | "stunet")
  and builds encoder + decoder dynamically from `encoder.skip_channels`,
  `encoder.bot_features`, `encoder.total_stride`.  Decoder depth scales
  automatically (3 stages for ResEnc, 5 for STU-Net).

- `configs/config.yaml` — added `model.encoder`, `stunet_variant`,
  `stunet_pretrained`, `stunet_freeze`, `mask_fusion`.

- `scripts/train.py` — `build_model` passes new encoder params to
  `ResEncInContext3D`.

## 2026-05-18 — Feature-attention experiment

Added `experiments/feature_attention/` to study the impact of attention mechanism design on patch-level in-context segmentation.

**Motivation**: cosine-similarity retrieval (normed dice 0.324) vs. TabPFN in-context classifier (0.416) showed a clear gap. The goal is to understand which architectural decisions explain it and to train a lightweight learned attention module.

**Files created**:

- `experiments/feature_attention/model.py` — `PatchICLAttention`: learned cross-attention binary classifier. Target patches attend to context patches (K/V from context only, same train-only masking as TabPFN). Eight configurable decisions:
  - `label_injection` — how context binary labels enter tokens: `additive` (TabPFN-style token += label_embed) | `concat` | `gate` | `none`
  - `output_head` — `linear` | `mlp` | `retrieval` (cross-attention Q=tgt, K=label-conditioned ctx, V=scalar labels)
  - `pos_encoding` — `none` | `sinusoidal` (fixed 3D sin/cos) | `learned` (nn.Embedding per grid position)
  - `input_norm` — `none` | `rmsnorm` | `l2`
  - `num_layers`, `num_heads`, `ff_factor`, `dropout`
  - Zero-init residual output projections (stable early training, TabPFN-style)

- `experiments/feature_attention/train.py` — trains `PatchICLAttention` on TotalSegmentator train split with frozen STU-Net encoder. Class-balanced sampling, AUROC-based validation each epoch, checkpoint saved on best val AUROC. W&B logging to `patch_icl_3d_exps`.

- `experiments/feature_attention/run.py` — evaluation script mirroring `feature_similarity/run.py` (same metrics: soft dice, normed dice, AUROC; same visualisation; same W&B logging).

**Also added to `experiments/feature_similarity/run.py`**:
- `--method tabpfn` option using `TabPFNClassifier` as the prediction head
- `--mask_pool` option (`max` / `avg`) for GT downsampling
- `--feature_level all` to concatenate all encoder levels
- W&B logging (project `patch_icl_3d_exps`, per-sample metrics + figures)
- Avg inference time tracking

## 2026-05-15 — PatchICL-style token conditioning

Added three token-level enrichments to `ResEncInContext3D`, inspired by PatchICL v3:

- **Type embeddings** (`target_type_embed` / `context_type_embed`): learnable `(1,1,C)`
  offsets added to bottleneck tokens before Stage 1. Teach the model to distinguish
  target volumes (zero mask) from context volumes (real mask). Init to zero → no effect
  at initialisation.

- **Register tokens** (`num_registers`): `R` learnable global tokens appended to the
  Stage-2 KV sequence alongside the spatial context tokens. Give the target a set of
  global scratchpad slots to attend to. `RoPECrossAttn` updated to apply identity
  rotation (position 0) to register tokens, preserving backward compatibility.

- **Context-first layers** (`num_context_layers`): optional extra `RoPETransformerBlock`
  layers applied only to context tokens before Stage 1. Context enriches itself via
  self-attention before the joint Stage-1 pass — mirrors PatchICL v3's `context_layers`.

Config knobs added to `configs/config.yaml` (`model.num_registers`,
`model.num_context_layers`); both default to 0 (disabled) to preserve existing runs.

To train with STU-Net (requires 128³ volumes):

```bash
python scripts/train.py \
  model.encoder=stunet \
  model.stunet_variant=base \
  model.stunet_pretrained=/path/to/stunet_base.model \
  data.image_size=[128,128,128]
```

## ImagePFN checkpoint eval in experiments/2d/eval.py

Added a third backend (`cfg.model=pfn_seg_2d`) to the unified 2D eval script so a
trained `ImagePFN` checkpoint can be evaluated alongside `universeg` and
`universeg_featuresim`.

- `pfn_seg.py` now saves `{model, arch, image_size, context_size}` (was a bare
  `state_dict`) so eval rebuilds the model from the checkpoint alone — no arch
  config to keep in sync. `_orig_mod.` prefix (torch.compile) is stripped on load.
- `eval.py` loads the checkpoint up front, syncs `data.image_size` into cfg before
  building the loader, reconstructs `ImagePFN`, and runs the same recipe as the
  trainer's `run_eval`: bf16 forward → sigmoid → bilinear upsample to native →
  `hard_dice` (native = headline `d_native`, low-res = `d_ds`). Timing reuses the
  shared `inference_times` path.
- Invoked via `base.yaml` + overrides: `model=pfn_seg_2d eval.checkpoint=<path>`.
  (A short-lived `pfn_seg_eval.yaml` was added here then removed on 2026-06-12 — see
  the config-consolidation entry above.)

Note: existing pfn_seg checkpoints saved before this change lack the arch dict and
must be retrained/resaved to be eval-loadable (`train.checkpoint` warm-start, 2026-06-12).

## 2026-06-14 — pfn_seg eval: per-patch error analysis CSV

- Added opt-in `eval.patch_csv` (base.yaml, null = off). When set, the `pfn_seg_2d`
  eval path dumps one CSV row per low-res patch (the `Hp×Hp` grid,
  `Hp = image_size // patch_size`): `dataset, label_value, sample_idx, patch_i,
  patch_j, pred` (sigmoid), `gt` (avg-pool soft fraction), `error` (pred−gt, signed),
  `gt_size` (native fg pixel count), `ctx_dice` (mean native `hard_dice` of target
  GT vs each of the K context masks). Last two are per-sample, replicated per row.
- Zero cost unless `eval.patch_csv` is set; collection reuses tensors already in
  scope and writes once at end via stdlib `csv` (no new deps).

## 2026-06-14 — pfn_seg training: patch-level soft-target loss

- Switched the training objective from native-res binary BCE (on the bilinearly
  upsampled logit grid) to **patch-resolution soft-target supervision**, computed
  directly at the `Hp×Hp` head grid (no upsample):
  `target = adaptive_avg_pool2d(binary_mask, Hp)` (per-patch fg fraction ∈ [0,1]);
  `loss = BCE_with_logits(logits, target) + dice_weight · soft_dice_loss(σ(logits), target)`.
  Soft-target BCE is a proper scoring rule → calibrated to the true patch fraction;
  the soft-Dice term (new `soft_dice_loss`, smoothing eps=1.0) counters background
  imbalance. New config `train.dice_weight` (default 1.0).
- Inference paths (`run_eval`, `eval.py`) still upsample for native-res Dice
  *reporting* only — not a training signal. `dice_ds_soft` now mirrors the objective.
- Architecture unchanged → existing checkpoints resume via the warm-start path
  (`train.checkpoint=...`); verified strict state_dict load on
  `results/2d/pfn_seg_resumed.pt`.
- Soft context masks (symmetric input-side representation) deliberately deferred to
  isolate the loss-term effect first.

- `pfn_seg.py`: set node-local `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` (under
  `/tmp/<user>_compile_<hostname>`) before importing torch. `~/.triton` and
  `~/.cache` are on shared NFS, so a `cuda_utils.so` compiled on a GLIBC-2.34 node
  poisoned the cache for GLIBC-2.31 nodes (`GLIBC_2.34 not found` at compile time).
  Keying the cache by hostname makes each node compile its own artifacts.

## 2026-06-14 — pfn_seg patch error-driver analysis

- New script `experiments/2d/patch_error_drivers.py` analyses what drives per-patch
  error in a `patch_analysis.csv` dump, oriented to the refinement-sampling goal
  (separates inference-observable drivers from oracle/GT-only ones). Outputs a text
  report + figures (`patch_error_drivers.txt`, `err_gt_pred_heatmap.png`,
  `err_uncert_modality.png`) next to the CSV. Attaches a best-effort dataset→modality
  map (from MedSegBench identities; not in the data).
- Findings on the soft-loss run (`results/2d/pfn_seg_low_res_loss/.../`):
  - **Error concentrates in boundary patches**: 12.4% of patches (0.05<gt<0.95) hold
    72% of total error mass; pure-bg (81.7%) only 15.5%. Dominant driver = partial
    coverage (boundaryness 4·gt·(1−gt)).
  - **Findable without GT**: `pred_uncert`=4p(1−p) (≡|pred−0.5|) ranks error almost
    perfectly (Spearman ρ=0.988). Top-10/20/30% patches by uncertainty capture
    61/88/96% of error → validates uncertainty-based patch sampling for the refiner.
  - **Blind spot**: confidently-wrong patches (extreme pred, opposite gt) have ~0
    uncertainty → uncertainty sampling misses them; reserve a small quota.
  - Secondary: modality matters (hardest microscopy>dermoscopy>xray; easiest oct/ct/mri;
    hardest datasets robotool/tnbcnuclei/kvasir/brifiseg/monusac — thin/many-object).
    `gt_size` weak (+), `ctx_dice` weak & oracle-only, patch position negligible.
  - Error magnitude only ~⅓–½ predictable (observable R²=0.334, intrinsic R²=0.461):
    rankable, not precisely weightable. NB error≡pred−gt, so a model given both is
    circular (excluded) — drivers come from observable-only and intrinsic models.

## controlSynth: fix gmm_fill IndexError on warped-out foreground (2026-06-20)
- Crash: `synth=hard_diverse` raised `IndexError: index 41 out of bounds for axis 0
  with size 27` in `appearance.gmm_fill` (`lut[fg_label]=fg_mean`). `fg_label=2*num_labels+1`
  (=41 for hard_diverse's `num_labels=20`); the LUT was sized to `labels.max()+1` from
  only the labels *present* in the deformed map.
- Root cause: the elastic warp (`deform`, `support_query_shift=0.50`) can fold a thin/small
  foreground (e.g. tubular, ~1.2% area) entirely out of frame, so `fg_label` exceeds the
  warped map's max present label. Rare (~1/40000 subjects) but inevitable over a 20k-sample
  epoch. Reproduced end-to-end on task 1044 (tubular), `warped.max()=26`.
- Fix: size the LUT to every label `gmm_fill` *writes* (`fg_label` + `distractor_labels`),
  not just those present. Slots for absent labels stay 0 and are never read (`img=lut[label_map]`).
  A vanished-fg subject now renders as all-background with an empty mask — a valid degenerate
  sample, no crash.

## eval: log a checkpoint's training-data provenance (2026-06-20)
- Problem: when evaluating a trained `pfn_seg_2d` / `patchset_pfn` checkpoint, the run had
  no record of what the model was *trained on*. `cfg.data.*` (and thus `run_cfg.source`)
  reflect the *eval* dataset, so a synth-trained vs medsegbench-trained checkpoint were
  indistinguishable in W&B.
- Fix (train side): `pfn_seg.py` and `multilevel/train.py` now embed the full training data
  config in `best.pt` — `"data": OmegaConf.to_container(cfg.data, resolve=True)` plus
  `"synth"` (the controlSynth knob block, or `None` when not `source=synthetic`).
- Fix (eval side): `eval.py` reads `pfn_ckpt.get("data")` / `get("synth")` after loading the
  checkpoint, prints a one-line `Checkpoint trained on: source=…, dataset=…, …` summary, and
  logs them to the W&B run config as `train_data` / `train_synth` (both backends). Old
  checkpoints lacking the keys log `None` and print an "older checkpoint" note — no guessing.

## pfn_seg + multilevel: select best checkpoint on soft Dice at the model's output level (2026-06-20)
- pfn_seg (`experiments/2d/pfn_seg.py`): `run_eval` now returns `dice_ds_soft/mean`
  (low-res soft/shape Dice at the head's native patch grid Hp) instead of `dice/mean`
  (the native-res hard Dice after upsampling preds to image size). All three metrics
  are still logged; only checkpoint selection + the `[best]` message changed.
- multilevel (`experiments/2d/multilevel/train.py`): added per-resolution soft Dice
  `dice_soft_r{res}/mean` (continuous composite vs avg-pooled un-binarized GT, parallel
  to the existing hard `dice_r{res}`), with `dice_soft/mean` = the last computed level.
  Checkpoint selection switched from `dice/mean` (final hard, native) to `dice_soft/mean`.
  Eval line now prints both hard and soft rows; best/summary keys renamed accordingly.
- Rationale: select at the resolution the model is actually supervised at, on a
  threshold-free shape score, rather than an upsampled hard-thresholded metric.

## controlSynth realism: biomedparse mask shape/position/size stats (2026-06-20)
- New `scripts/biomedparse_shape_stats.py`: samples masks balanced across all biomedparse
  datasets (41 train dirs, ~100/ds, N≈3944) and measures position/size/shape. Writes
  `results/controlsynth/biomedparse_shape_stats.{txt,json,png}`. Re-measured synth
  hard_diverse with the same code (`mask_stats`) for a direct gap.
- REAL biomedparse (median / p5..p95):
  center offset 0.158 (p95 0.367; 37% off-center >0.2); area_frac 0.019 (0.0009..0.269,
  ~294x span, heavy small tail); eccentricity 0.81; solidity 0.91; extent 0.59;
  n_cc median 1 (69% single-component, only 31% multi); border-touch 6%.
- SYNTH hard_diverse gap (the "centered + same size + ragged" issue):
  * position too centered: offset p95 0.24 vs 0.37, only 9% off-center vs 37%; centroid
    std ~0.09 vs ~0.14.
  * size too narrow/large: area_frac 0.056 median, ~28x span (region_size floored at 0.30
    → no small-object tail); real is 0.019 median, 294x span.
  * BIGGEST gap = fragmentation: n_cc median 14, 87% multi-component, solidity 0.57,
    extent 0.25 — real masks are mostly single compact regions (median 1 cc, solidity 0.91).
  * border-touch 39% vs 6% — synth foregrounds run off-frame far too often (deformation).
- Next (not yet done): translate targets into generator knobs — random fg placement
  (centroid offset), log-distributed area with a small tail, reduce fragmentation
  (boundary/deformation), keep fg in-frame.

## controlSynth realism: generator + hard_diverse config (2026-06-20, implemented)
- Added 4 backward-compatible DifficultyBuildSpec knobs (defaults = original behavior;
  set only in hard_diverse.yaml so the difficulty-study presets are untouched):
  boundary_amp_scale, boundary_sigma_frac, boundary_keep_largest (shapes/boundary.py),
  position_jitter (new task.place_foreground). Knobs flow via geo_params (no signature change).
  * boundary fix = the big one: roughening shattered shapes (n_cc median ~12); gentler
    amplitude (0.5) + low-freq blur (sqrt(area)*0.4) + keep-largest (blob/elongated/annular
    only) → mostly single compact regions.
  * place_foreground translates the finished shape to centroid ~N(0.5, jitter), clamped to
    keep its bbox in-frame.
- hard_diverse.yaml: boundary_amp_scale=0.5, boundary_sigma_frac=0.4, boundary_keep_largest=true,
  position_jitter=0.15, region_size range [0.30,0.70]→[0.12,0.62] (moderate floor).
- Result (val masks, vs real biomedparse): n_cc med 12→2, solidity 0.57→0.83, area 0.050→0.028
  (real 0.019), offset mean 0.14→0.18 / p95 0.27→0.31 (real 0.37). Sample plot regenerated
  (results/controlsynth/hard_diverse_samples.png) — visibly compact + off-centre.
- CAVEAT (not resolved): border-touch ~34% vs real 6%. Mostly the post-placement deformation
  (shift=0.50, ~6px) pushing inset-2 shapes over the edge + sprawling tubular trees. Would
  need a larger placement inset (vs deform magnitude) or in-frame deformation to fix.

## controlSynth realism: within-(target+context)-set mask distance (2026-06-20)
- New `scripts/context_mask_distance.py`: per in-context set (target label + K context
  masks), averages pairwise overlay_dice, centroid_dist, area_logratio. Run on
  biomedparse (1500 sets) and synth hard_diverse (800 sets), res=128, K=3.
- REAL biomedparse: overlay_dice 0.25 macro (0.18 micro, median 0.10), centroid_dist 0.159,
  area_logratio 1.20 (~2.3x within-set area swing). Many cells ~0 overlap (structure
  relocates entirely across patients: LIDC/colon/polyp/disc).
- SYNTH hard_diverse: overlay_dice 0.27 (median 0.24), centroid_dist 0.105, area_logratio 0.57.
- Finding: synth context sets are too SELF-SIMILAR — real within-set spread comes from
  independent patients; synth derives all K+1 subjects from ONE base geometry varied only
  by per-subject elastic deform (support_query_shift). position_jitter is per-TASK (shared
  by the set) so adds no within-set positional variability. To match real, add per-SUBJECT
  position (centroid 0.105->~0.16) + scale (logratio 0.57->~1.2) jitter — but bounded, else
  the fg stops being the consistent in-context cue. NOT implemented.

## controlSynth realism: per-subject pose jitter (within-set spread) (2026-06-20, implemented)
- Added 2 backward-compat DifficultyLiveConfig knobs (default 0 = original; declared in
  default.yaml live, set in hard_diverse.yaml): support_query_translate (per-subject centroid
  shift std, fraction of image), support_query_scale (per-subject log2 zoom std). Applied via
  new deformation.jitter_pose (affine about fg centroid, clamped in-frame), after deform in
  dataset._make_subject, scaled by shift_scale so pristine contexts stay near-aligned.
- Rationale: support_query_shift only deforms ONE shared base -> synth (target+ctx) sets too
  self-similar. position_jitter was per-TASK (no within-set spread).
- Tuned to tr=0.05, sc=0.45 (closes ~80% of position gap, most of size gap without collapsing
  within-set overlap below real; pushing tr>=0.06 overshoots centroid + drops overlap to ~0.14):
    within-set overlay/centroid/area_logratio: 0.250/0.113/0.546 -> 0.168/0.146/1.042
    (real biomedparse 0.254/0.159/1.202).
  Single-mask shape stats unregressed (area med 0.022, n_cc med 1, solidity ~0.78, offset 0.18).
- Sample plot regenerated: contexts now vary in position+size within each set.

## 2D eval: qualitative figures for all backends (2026-06-21, implemented)
- Generalized experiments/2d/eval.py:save_figure to be backend-agnostic. New signature takes
  pred_native (H,W; soft for feat_sim/pfn_seg/multilevel, binary for universeg) plus an optional
  coarse grid (pred_lowres/gt_lowres). Row 0 = Target+GT | Target+Pred | (GT↓ | Pred↓ only when a
  low-res grid is supplied); Row 1 = K context overlays. Panels built dynamically, squeeze=False,
  so K=1 and the no-low-res (universeg) case both render.
- Prediction resolution per backend (figure source tensors): feat_sim pred_lowres=preds_all[b]
  at output_size, native via bilinear upsample; pfn_seg pred_lowres=preds_lowres[b] at Hp, native
  preds[b]; multilevel pred_lowres=refined_grid at Hg (captured as preds_grid before upsample,
  only shown when Hg!=H), native preds[b]; universeg native binary preds[b,0], no low-res.
- Replaced the feat_sim-only inline save with one unified block in the per-sample loop, gated by
  new config knobs eval.save_figures (bool, default false) + eval.max_figures (cap, default 50).
  Dedup stays one figure per (dataset, label_value). Previously only feat_sim emitted figures and
  always-on; now all four backends can, opt-in.

## 2D eval: wandb-named output folders (2026-06-21, implemented)
- eval.py no longer builds descriptive run names per backend; wandb.init(name=cfg.wandb.name)
  so a null name auto-generates (e.g. "deft-field-72"). out_dir now mirrors pfn_seg.py:
  {eval.out_dir}/{date}_{run_name}, so each run's figures/CSVs land in their own folder instead
  of all dumping into a shared results/2d/outputs. Added eval.figures_to_wandb (default true) to
  toggle wandb upload independently of local PNG saving.

## TotalSegmentator 2D cross-section manifest (2026-06-21, implemented)
- New scripts/build_totalseg_2d_manifest.py: picks ONE axial cross-section per subject that
  is densest in the most common classes, emits one (subject,class) task row per class present.
  Output: results/totalseg2d/manifest_128.csv (17621 rows, 1228 subjects, 110/121 classes;
  splits from meta.csv: 1082 train / 57 val / 89 test; ~14 tasks/subject).
- Axial axis = AXIS 2 of label_{S}x{S}x{S}.npy (verified, non-obvious): a true cross-section
  cuts <=2 vertebrae, only axis 2 satisfies this (axes 0/1 span 7-18 vertebrae = coronal/sagittal).
  Axis-2 slices are also organ-richest.
- Selection score = soft area-ramp: sum over classes (area>=noise_floor) of
  global_weight[class] * min(1, area/area_cap). global_weight = occurrences/max from
  label_stats.csv. Ramp rewards substantial presence (not grazing many organs by a few px) and
  is robust (top slices form one tight cluster; a hard min_area threshold flips between distant
  abdominal/pelvic slices near ties, and a tiny floor selects fragment-heavy edge slices).
  Defaults noise_floor=10, area_cap=100. Task emission decoupled: emit classes with
  area>=task_min_area (default 25), so slivers don't become degenerate targets nor sway selection.
- Selected slices land on the thoraco-abdominal belt (lungs, heart, aorta, autochthon, liver,
  spleen, kidneys, pancreas, stomach, IVC) — matching the global most-common classes.
- NOTE for the dataset step: ct_{S}x{S}x{S}.npy is float16 and pre-normalized (~z-scored, range
  ~[-1.7, 3.3]), NOT [0,1] — a TotalSeg2D dataset must min-max/clip-rescale per slice to match the
  MedSegBench [0,1] convention before serving to the 2D in-context pipeline.

## TotalSegmentator 2D slice export to npz (2026-06-21, implemented)
- New scripts/totalseg2d/to_npz.py: exports ONE axial cross-section per subject to a single
  npz for the 2D experiments. Two deliberate specs vs convert_to_npy.py:
  1. FIXED mm/pixel (default 2.0mm, size 256 -> 512mm FOV), centered on the label. Fixes the
     cross-subject scale drift from longest-axis->cube normalization (s0919 vs s1158 was ~4.5x
     frame-area mismatch in the 128^3 cubes; now ~1.4x, i.e. genuine anatomy only).
  2. RAW int16 HU, no normalization (same bytes as float16, exact). Clip/z-score/windowing is
     deferred to the dataloader for flexibility. Disk/time cost of deferring measured ~0:
     int16 raw == float16 size; ~0.5ms/slice to normalize at load.
- Reads native label.npy (already canonical, axis2=axial) + ct.nii.gz (for raw HU only) +
  spacings.json. Slice selection = soft area-ramp over global class freq (build_totalseg_2d_manifest),
  but areas in PHYSICAL units (voxel count x in-plane mm^2 -> output px at mm_per_px) so argmax is
  FOV-consistent and self-contained (supersedes the cube-based manifest's z for storage).
- Per-slice render: in-plane ndi.zoom to mm_per_px (Gaussian AA on image downsample, nearest for
  label) then center on label centroid into a size x size grid; image pad = AIR_HU (-1024).
- npz schema per split: {split}_images int16 (N,size,size), {split}_label uint8, {split}_subjects,
  {split}_z, {split}_spacing; scalars mm_per_px/size/air_hu + class_names. savez_compressed.
- Smoke: 12 test subjects in 11s, 0 errors; verified raw HU range, scale fix, centering visually.

## TotalSeg2D in-context dataloader (2026-06-21, implemented)
- New src/datasets/totalseg2d.py: TotalSeg2DDataset mirrors MedSegBenchDataset's interface
  (returns {image,label,context_in,context_out}; exposes .samples/.label_index) so it plugs into
  the 2D pipeline via common.build_dataset with data.source="totalseg2d".
- Reads totalseg2d_{stored_size}.npz (to_npz output). Differences vs medsegbench:
  * Images are raw int16 HU -> normalized HERE: clip to data.hu_window (default [-1000,1000]) then
    min-max to [0,1] (clipping also tames extreme out-of-FOV HU outliers, e.g. -8653 seen in data).
  * Loads the 256px export (512mm FOV) and resizes to data.image_size (bilinear img / nearest label),
    decoupling model res from FOV. The 128px export is a tighter 256mm FOV (clips bodies) — avoided.
  * data.min_area (default 16 px @ image_size): a (subject,class) pair is a sample only above this,
    so tiny slivers aren't degenerate targets.
  * Single "dataset" name; label_value = TotalSeg class index 1..117.
- Wired into experiments/2d/common.py build_dataset (new totalseg2d branch + error-msg update).
- Verified with configs/augmentations/medsegbench.yaml via pfn_train.augment: geometric (flip/rotate)
  hits context pairs only, intensity (brightness/contrast/gamma/noise) hits all, query mask untouched,
  outputs stay in [0,1]. Plot: results/totalseg2d/dataloader_aug_samples.png (orig vs aug).

## 2D aug: single shared config + strong preset (2026-06-22)
- De-duplicated the 2D aug config. The identical `aug:` block was inlined in both
  configs/experiment/2d/pfn_seg.yaml and multilevel.yaml; replaced each with `aug_preset: 2d`
  and a code-load (`OmegaConf.load(configs/augmentations/<preset>.yaml)` merged into cfg.aug in
  main(), mirroring the 3D experiments/multilevel/train.py pattern). New canonical file:
  configs/augmentations/2d.yaml (2D schema: enabled/geometric/intensity; the old medsegbench.yaml
  is now an unused orphan). CLI field override uses the +-prefix, e.g. +aug.enabled=false.
- Extended pfn_train.augment() with literature-backed, backward-compatible ops (off unless the
  preset sets the keys; base 2d.yaml takes the identical legacy path — affine theta reduces to the
  old rotate-only matrix):
  * task.invert: episode-wide intensity inversion (UniverSeg "task augmentation"). Intensity-only,
    shared across all K+1, so the query image can be touched without desyncing its GT.
  * geometric.scale/translate: folded into the existing rotate affine (one grid_sample); context only.
  * geometric.elastic: smooth displacement field (low-res random -> bilinear upsample = smoothing);
    context pairs, image bilinear / mask nearest.
  * intensity.bias_field: smooth multiplicative inhomogeneity exp(field*strength) (SynthICL); all images.
  * query_perturb: extra independent noise on the query slot only (Iris "imperfect reference").
  * geometric.crop: random-resized crop — sample an in-bounds sub-window (relative size
    s∈[min_scale,1], shift bounded by 1-s so it never leaves the image → no border padding)
    and resize to full H×W. Context only; lets each context show a different region.
    In 2d_strong: p=0.5, min_scale=0.75. Verified a solid context stays unpadded post-crop.
  Geometric stays context-only by design: the training target is read from the un-augmented batch,
  so moving the query would break query/GT alignment. class dropout (Iris) is dataset-level, N/A here.
- New preset configs/augmentations/2d_strong.yaml enables all of the above (use aug_preset=2d_strong).
- Verified: both presets through augment() preserve shape, keep masks binary, leave the query mask slot
  zero, and clamp images to [0,1]. A/B vs 2d.yaml still TODO (gains may be aug strength, not transform).

## COW-safe dataset indexes (2026-06-22, implemented)
- Cause of DataLoader worker RAM creep / periodic stalls: forked workers share memory
  copy-on-write; COW copies a page only on WRITE, and the only writes during iteration are
  CPython refcount bumps on PyObject headers. The big image/label numpy buffers are read-only
  (safe), but the per-sample `list[(ds,idx,lv)]` and the `dict[..]->list[tuple]` context-lookup
  structures get refcount-churned every __getitem__ -> pages fork per worker -> RSS climbs over
  the run (persistent_workers never resets).
- Fix: src/datasets/cow_index.py — SampleIndex stores the (ds_id,img_idx,label) triples as 3
  contiguous int32 arrays + a small ds_names list, but still behaves like list[(ds,idx,lv)]
  (len/index/iter/subset) so common.TaggedDataset/collate/eval are unchanged. build_candidate_index
  groups image idxs by (ds_id,label) into read-only int32 arrays (replaces label_index/group_index).
  sample_context picks K context idxs from a candidate array via the `random` module (positions, not
  objects). Workers now read only numpy buffers -> no per-element refcount COW.
- Applied to src/datasets/{medsegbench,totalseg2d,biomedparse}.py (__init__ build + __getitem__
  lookup; biomedparse diversity tags + self-test updated). common.make_loader eval-subsampling uses
  SampleIndex.subset (no list rebuild).
- Verified: SampleIndex unit (tuple iface + subset), biomedparse self-test, totalseg2d/medsegbench
  item correctness, and 2 epochs through 3 persistent workers via TaggedDataset+collate.
- NB: still the bigger lever for the user's 48-worker/64GB case is fewer workers (~8-12) — an
  all-in-RAM dataset with cheap __getitem__ is GPU-bound; this fix removes the per-worker creep that
  made the stalls recur every few epochs.

## 2026-06-22 — Log val loss in 2D training scripts
- experiments/2d/pfn_seg.py and experiments/2d/multilevel/train.py now compute and log
  `val/loss` in their `run_eval`, accumulated with the same objective each uses for train
  loss (pfn_seg: BCE + dice_weight*soft_dice on the avg-pooled patch target; multilevel:
  the per-hop `loss_weights`-weighted `patch_loss` sum over chain outputs). Logged to wandb
  alongside the dice metrics and added to the per-epoch eval summary line.

## 2026-06-22 — Shared train_base.yaml for 2D training configs
- New configs/experiment/2d/train_base.yaml holds params common to pfn_seg.yaml and
  2d/multilevel.yaml (full data block; arch-core e/h/l/a/feature_level/thinking_rows/
  residual_decay/compile; full train block; eval.batch_size/workers/out_dir; wandb;
  aug_preset). Mirrors the eval-side base.yaml pattern (defaults: [synth: default, _self_]).
- pfn_seg.yaml / 2d/multilevel.yaml now do `defaults: [train_base, _self_]` and keep only
  model-specific + differing keys (encoders/resolution for pfn_seg; sample block, mask_prior/
  stage1 thinking/loss_weights/stage1_checkpoint for multilevel; per-config max_per_label).
- eval.out_dir set to .../2d_train for BOTH training scripts (multilevel previously .../2d).
- Verified composition with `--cfg job`: resolved configs match prior values (out_dir aside).
  Eval-only base.yaml and its consumers (feature_sim.yaml etc.) are untouched.
- Renamed configs/experiment/2d/base.yaml → eval_base.yaml for symmetry with train_base.yaml.
  Updated refs: feature_sim.yaml (defaults), eval.py + synth_benchmark.py (config_name).
  Verified eval.py / feature_sim.py compose unchanged via `--cfg job`.
- eval.py universeg backend: dice_ds / dice_ds_soft now logged as NaN instead of a copy of
  the native dice. UniverSeg emits only a native-res binary mask (no low-res/soft coarse grid),
  so for binary pred+GT both low-res metrics were mathematically equal to dice — misleadingly
  implying a shape score. log_summary's NaN-filtered aggregation drops them. Other backends
  (pfn_seg/feature_sim/patchset_pfn) score a genuine coarse grid and are unchanged.

## 2026-06-22 — Train-accuracy metrics in 2D training (near-zero cost)
- New common.batch_dice_sums(prob, target): vectorised, on-device soft + hard Dice SUMS +
  valid-row COUNTS, same per-row semantics as soft_dice/hard_dice (hard binarises both at
  0.5; empty pred+GT rows skipped). Verified numerically equal to the per-sample helpers
  (eval-style GT binarization). Lets train accuracy accumulate on GPU and sync once/epoch.
- pfn_seg.py train_epoch now returns (loss, train_dice_soft, train_dice) computed from the
  logits/target the forward already produced (reuses the sigmoid feeding the dice loss) —
  no extra forward, no per-batch GPU→CPU stall. Logged as train/dice_soft and train/dice,
  mirroring val dice_ds_soft / dice_ds so the train↔val gap is directly readable.
- multilevel/train.py train_epoch returns per-hop dicts; logs train/dice_soft_r{grid}/mean
  and train/dice_r{grid}/mean (mirrors val dice_soft_r* naming). Measured on each hop's
  sampled query patches (what it trains on), so reads slightly above the full-grid val metric.
- universeg_train.py: train Dice was never logged (train_epoch only returned loss; the
  soft-dice term was folded into the loss, and log_summary ran only on val output). Added
  per-batch hard_dice accumulation over train predictions; train_epoch now returns
  (loss, train_dice), logged as train/dice and shown in the per-epoch console line + tqdm
  postfix. Monitoring only (under no_grad), independent of synth=omniglot / pretrained flags.
- New model src/models/patchset_cnn.py (PatchSetCNN): trainable single-stream conv
  encoder downsamples each image to an R×R grid (default 16), then ImagePFN's dual-axis
  TransformerEncoderStack reasons in-context over [img | mask-occupancy] columns. Mask
  token = scalar avg-pool occupancy (Linear(1→e)); query mask = context-mean prior. Head
  is a per-patch Linear → R×R logits (no decoder/upsampling); loss vs avg-pooled GT.
- Fused experiments/2d/{universeg_train.py + patchset_cnn trainer} → experiments/2d/train.py.
  One model-agnostic loop: both models called as model(img, context_in, context_out, mode)
  -> {"final_logit"}; GT is avg-pooled to the logit's spatial size (no-op for UniverSeg
  H×W, R×R downsample for PatchSetCNN). Model selected by cfg.model (universeg | patchset_cnn).
  New config configs/experiment/2d/patchset_cnn_train.yaml; universeg_train.yaml kept.
  Removed universeg_train.py.
- Removed ic_segmentation dependency entirely. universeg_baseline lived only in
  ic_segmentation/src/models and was reached via common.py putting ic_segmentation on
  sys.path, whose src/__init__.py shadowed patch_icl's src. Vendored universeg_baseline.py
  into src/models/, dropped the ic_segmentation sys.path insert in common.py, and removed
  the shadowing workarounds (eval.py importlib-by-path load of pfn_seg_2d → plain import;
  stale "cache before ic_segmentation" comments in pfn_seg.py/multilevel/train.py/
  plot_synth_samples.py). `src` is now unambiguously patch_icl.
- Low-res Dice metric: hard Dice at R×R is degenerate for thin omniglot strokes (avg-pooled
  GT binarized at 0.5 is ~empty → NaN). train.py now logs BOTH soft Dice (threshold-free,
  prob vs occupancy) and hard Dice (pred≥0.5, GT>0) for train and val, and selects the best
  checkpoint on soft (dice_soft/mean). Mirrors multilevel/train.py's dice_soft/dice pair.
- PatchSetCNN encoder reworked to UniverSeg per-stage widths + multi-scale concat:
  enc_dims default [64,64,64,64] (UniverSeg v1 encoder_blocks). ConvEncoder now keeps
  the stem + every stage feature map, avg-pools each to R×R, and concatenates along
  channels (out_ch = sum(enc_dims) = 256) instead of returning only the last stage.
  img_embed = Linear(sum(enc_dims) → e) adapts automatically; per-patch tokens now carry
  low- through high-level features. ~4.44M params at e=256.
- train.py metric consistency: `dice` (hard) is now computed at the ORIGINAL resolution
  — preds upscaled (bilinear) to H×W, thresholded ≥0.5, vs the full-res GT (>0). This is a
  no-op for native-res UniverSeg, so the headline `dice` is directly comparable across
  models/resolutions. The soft metric (prob vs pooled occupancy at the model's logit res)
  is renamed dice_soft → `dice_ds_soft` (downsampled). wandb: train/dice_ds_soft, train/dice,
  dice_ds_soft/*, dice/*; best-checkpoint still on dice_ds_soft/mean. NB: a 16×16 PatchSetCNN
  upscaled to 128 cannot recover sub-patch thin strokes, so its native `dice` is capped — it
  reads the achievable native accuracy, while dice_ds_soft reflects the training objective.
- PatchSetCNN switched from image-grid to SET-of-patches attention (PatchSetPFN layout).
  Motivation: the image-grid layout's cross-image (sample-axis) attention is position-
  locked — query patch (i,j) attends only to context patch (i,j) — which is wasteful when
  objects aren't spatially aligned across images (omniSynth chars in random cells). New
  layout: rows = thinking + support patches (K·N) + query patches (N); cols = [img|mask].
  Sample-axis attention is now full-set content matching (query patches attend to all
  support patches); the patch (i,j) is injected as a Fourier positional FEATURE
  (FourierPositionalEncoding reused from patchset_pfn) instead of being enforced by
  structure. Replaced the learned pos_embed; per-channel feature standardization now uses
  support-patch stats. Query mask token = support-mean occupancy prior.
  Supporting fix: pfn_seg_2d TransformerEncoderLayer feature-axis now uses batched_sdpa
  (was plain SDPA) so the b*rows grid stays under the CUDA gridDim.y cap (65535) when rows
  = all patches (e.g. B=64, R=16 → 65600 rows). No-op for existing small-batch ImagePFN use.
- PatchSetCNN: added query_self_attn flag (arch.query_self_attn, default false), mirroring
  PatchSetPFN. When true, query patches attend to each other (within-target spatial
  reasoning) in addition to the support set, via an (r×r) bool attn_mask passed to the
  sample-axis. Queries carry the prior (not GT) so query↔query attention leaks no labels.

## 2026-07-03 — omnisynth_train_base.yaml between train_base and the omniSynth leaves
- universeg_train.yaml and patchset_cnn_train.yaml duplicated every non-model key
  (data.source=omnisynth, the 50-epoch/batch-64 schedule, eval batch/workers/max_per_label).
  Factored the shared keys into a new intermediate config
  configs/experiment/2d/omnisynth_train_base.yaml (defaults: [train_base, override
  synth: omniglot, _self_]). Leaves now carry only model-specific keys: `model`, their
  `arch` block (patchset only; universeg ignores arch), `train.lr`, universeg `pretrained`.
- train_base.yaml deliberately left untouched so the medsegbench/ImagePFN configs
  (pfn_seg, multilevel, multilevel_zoom) keep inheriting source=medsegbench + the long
  200-epoch schedule + the transformer arch block.
- Side effect: the intermediate defaults `synth: omniglot`, so `synth.scene.*` overrides
  (e.g. synth.scene.p_copy / synth.scene.grid) now work without passing synth=omniglot.
  Verified both leaves resolve to their pre-refactor values via `train.py --cfg job`.

## 2026-07-03 — Store full PatchSetCNN arch in the checkpoint
- train.py:build_model previously saved only a partial patchset_cnn ckpt_meta
  (resolution/enc_dims/query_self_attn/context_id_embed/max_context), omitting the
  transformer knobs (e/h/l/a/thinking_rows/residual_decay/fourier_bands) — so a best.pt
  could not be rebuilt without also supplying the training config.
- Now build_model constructs a single `arch` dict (all PatchSetCNN kwargs except
  image_size), uses it to instantiate the model, AND returns it as ckpt_meta -> the
  checkpoint stores it under `arch` (nested, mirroring pfn_seg_2d). A focused eval script
  can rebuild via PatchSetCNN(image_size=ckpt["image_size"], **ckpt["arch"]). Verified the
  round-trip yields an identical state_dict (keys+shapes). image_size stays at the ckpt
  top level (unchanged). Nothing else consumes patchset_cnn checkpoints, so the shape
  change (flat keys -> nested `arch`) is safe.

## 2026-07-03 — Shared validate() + focused eval_incontext.py (universeg / patchset_cnn)
- Extracted the per-epoch eval loop into experiments/2d/evaluate.py as a single shared
  validate() used by BOTH train.py and the new eval script (metrics coherent by construction).
  It computes native `dice` plus, for low-res models, `dice_ds`/`dice_ds_soft`/`cossim`/`top{k}`,
  fills an adaptive per-sample table (source-aware `_sample_detail`), and gates figures /
  patch_csv / synth_csv / a one-shot FLOPs count behind opt-in args.
- train.py now imports validate/_target_like/_upsample_to from evaluate.py (dropped its own
  copies + _fmt_transforms/SAMPLE_COLS and 6 now-dead imports). Behavior unchanged: low-res
  models checkpoint on cossim, native on dice (verified by 1-epoch smoke runs).
- eval.py DRY: its save_figure/_overlay_ax/_heatmap_ax now live in evaluate.py (eval.py imports
  save_figure). Its 5-backend dispatch is otherwise untouched.
- New experiments/2d/eval_incontext.py: thin Hydra wrapper (reuses eval_base.yaml) that loads a
  train.py checkpoint, dispatches on model_name, rebuilds universeg or patchset_cnn (via
  **ckpt["arch"]), and runs the shared validate() with figures/CSVs/FLOPs on. Fails loudly on
  pre-arch patchset_cnn checkpoints. Injects the checkpoint's stored `synth` block into cfg.synth
  so eval reproduces the model's exact training distribution (eval_base defaults to the
  controlSynth synth schema; omniSynth models need the omniglot schema — without this the eval
  silently ran on default scene config and scored a misleading ~0.07 vs the correct ~0.86).

## 2026-07-03 — eval_incontext.py: CLI omniSynth overrides (OOD eval)
- Added configs/experiment/2d/eval_incontext.yaml (defaults: [eval_base, override synth: omniglot,
  _self_]) and pointed the wrapper's @hydra.main at it. Defaulting the synth group to omniglot lets
  `synth.scene.*` compose (eval_base's controlSynth default has no `scene` -> "not in struct").
- Wrapper now uses the checkpoint's stored synth as the BASE, then merges CLI `synth.*` overrides on
  top (read from HydraConfig.get().overrides.task) so they win. Unspecified keys still reproduce the
  training distribution. Enables OOD eval, e.g. `synth.scene.grid=2 synth.scene.target_mode=class`.

## 4_weaknesses.py — universeg vs patchset_cnn weakness analysis (omnisynth)
- Added marimo analysis cells (runs uixvcpny=universeg, kq1cent3=patchset_cnn). The two eval
  tables have different schemas: universeg has explicit cols; patchset packs everything into a
  `detail` string ("char mode= cells= tf="). Cells parse `detail`, unify into one long df, decode
  `transforms` -> rot/scale/dx/dy features, and build a paired merge on
  (dataset, character, target_pos, transforms).
- Findings: universeg mean/median Dice 0.76/0.91 (bimodal: 52% >0.9, 10% <0.1); patchset_cnn
  0.46/0.50, unimodal peak ~0.5, 0% samples >0.9. Universeg wins 82.6% of paired samples.
- Both models: Dice *increases* with |rotation| — the (0,5]deg bin is the hardest
  (uni 0.57, pat 0.41), rising to ~0.88/0.50 by 15-20deg. target_pos has ~no effect.
  Spearman(dice,|rot|): uni +0.35, pat +0.22; scale/translation weak.
- NB: marimo/wandb/seaborn live in `.venv` (py3.12), not `.venv311`. Verification figs:
  results/experiments/4w_fig{1_dist,2_rot,3_paired}.png.

## 4_weaknesses.py — driver analysis (what drives Dice for each model)
- Enriched the paired frame (`mg`): borrowed universeg's `context_pos` onto patchset (same
  seeded samples) to get target-context grid distance for both; parsed transforms; added
  per-character object size = mean ink fraction of the Omniglot glyph (via
  src/datasets/omniSynth/bank.py over val+test pools, keyed by class_id in "alphabet/class_id").
- Quantified with eta^2 (variance-explained). The two models are driven by DIFFERENT things:
    driver              universeg  patchset_cnn
    object size (ink)      0.008     0.094
    character id           0.050     0.242
    dataset/alphabet       0.015     0.107
    target |rotation|      0.174     0.073
    scale deviation        0.009     0.015
    translation            0.033     0.010
    tgt-ctx distance       0.001     0.001
- universeg: accuracy driven by AUGMENTATION (rotation); robust to object identity/size.
  patchset_cnn: driven by OBJECT IDENTITY/SIZE (character 0.24, size 0.09, dataset 0.11);
  fails on small/thin glyphs (ink-quintile Dice 0.37->0.51 vs universeg 0.72->0.81).
- target-context grid distance is negligible for both (eta^2 0.001; ~flat curve).
- Rotation paradox (more |rot| -> higher Dice) is NOT a size confound:
  spearman(|rot|, ink_frac)=0.004; independent. Since target rotation is independent of the
  context, this looks like a target-mask rendering/Dice-geometry effect (thin axis-aligned
  strokes at |rot|~0), not a target-context similarity effect. Open question.
- Cross-model per-character difficulty correlates only rho=0.35 (dataset-level 0.46): partly
  shared difficulty, but the models fail on different objects. patchset uniformly below diagonal.
- Figures: results/experiments/4w_fig{4_drivers,5_size_dist,6_charscatter}.png.

## omniSynth eval logging: self-contained context + real (x,y) positions
- evaluate.py `_sample_detail`: the newer source-agnostic `detail` schema was dropping
  `context_cells`. Added `ctx=` (context cells) and `sub=` (subject_index) so an
  omniSynth-only run is self-contained (context position + per-sample join key). Kept the
  free-form `tf=` last for easy parsing.
- New: real post-aug glyph positions (not just grid cell indices).
  - render.py `render_scene` now returns `info["target_positions"]`: per target cell, the
    ink centre-of-mass in full-canvas coords normalised to [0,1] (x right, y down), via
    `_centroid_xy` (falls back to cell centre if a placement has no ink). Captures cell +
    shift + rotate/scale + glyph asymmetry — the continuous counterpart of target_cells.
  - dataset.py meta gains `target_positions` + `context_positions` (aligned with the cells).
  - `_sample_detail` logs them as `pos=(x,y)...` / `cpos=...` (via `_fmt_positions`).
- Motivation: grid-cell tgt-ctx distance was too coarse (eta^2~0.001 in 4_weaknesses.py);
  continuous positions enable a real distance driver. Existing runs unaffected (re-run needed).
- Verified: render + dataset tests pass; deterministic val sample renders correct centroids
  (cell (0,1) center x0.75,y0.25 -> centroid 0.726,0.247 under dx-0.07 shift).

## Fixed-resolution downsampled Dice for native-res models (UniverSeg)
- New config knob `eval.ds_metric_res` (in universeg_train.yaml, default `[16]`, list-capable,
  null/[] disables). UniverSeg predicts at native full-res so dice_ds/dice_ds_soft are NaN;
  this also logs hard + soft Dice with GT and pred both avg-pooled to RxR (hard thresholded at
  >=0.5), matching patchset_cnn's low-res grid recipe.
- evaluate.py: `validate(..., ds_metric_res=None)` + `_as_res_list` helper. Per batch, pools
  prob_nat & lbl to each R (batched), accumulates per-(dataset,label), emits summary keys
  `dice_ds@{R}/*` and `dice_ds_soft@{R}/*` via log_summary. Sample table (SAMPLE_COLS) unchanged.
- train.py: `train_epoch` reads `cfg.eval.ds_metric_res`, accumulates on-device running sums per
  R (reuses _soft_sum/_hard_sum on pooled maps), returns them as a 6th dict element; main logs
  `train/dice_ds@{R}` + `train/dice_ds_soft@{R}`. Checkpoint selection metric unchanged (native dice).
- Cost (measured, B=64, 128->16 on GPU): +0.30 ms/batch = 0.12% of a train step, 0.28% of eval;
  ~0.05s/epoch train, ~0.15s/full val. Negligible.
- Verified: 1-epoch smoke run (grid=2, ds_metric_res=[16,32]) logs all four metric families
  clean; other validate/train_epoch callers unaffected (default None / distinct local funcs).

## omniSynth: larger, overlapping glyphs via cell_margin (wired + allows <0)
- Problem: chars filled only 80% of their cell (hardcoded 0.1 margin in bank._to_bitmap) and
  cells tiled the canvas non-overlapping -> glyphs looked far apart, little ink/training signal.
- cell_margin is now threaded through (was dead config): glyph size = (1-2*margin)*cell.
  margin>0 insets inside the cell (old look); 0 fills it; <0 makes the glyph LARGER than its
  cell and overflow into neighbours.
- bank.py: _to_bitmap renders the glyph at inner=(1-2*margin)*cell into a tile=max(cell,inner)
  (so margin>=0 stays exactly cell-sized -> byte-identical + tests pass). get_or_build_bank /
  OmniglotBank take cell_margin (in the cache key). dataset.py passes scene.cell_margin.
- render.py: cells no longer written as disjoint slices. Each glyph is pasted centred on its
  cell centre via _paste (np.maximum union, clipped to canvas) so oversized glyphs overlap
  instead of overwrite; order-independent. Target mask = union of target tiles; positions =
  centroid of the pasted (clipped) target ink. New _paste/_paste_centroid; dropped _centroid_xy.
- config omniglot.yaml: cell_margin -0.15 (glyph 1.3x cell, small overlap). Tunable; visual
  sweep saved to results/experiments/omnisynth_margins.png (ink fraction 0.046 -> ~0.14 at -0.15).
- Backward compat: OmniSceneConfig default stays 0.1 -> all existing tests unchanged. Added
  test_oversized_tiles_overflow_and_union. 22 omniSynth tests pass; 1-epoch train smoke clean.

## Consistent dice_ds@{R} naming for patchset_cnn (match universeg)
- Patchset's coarse-grid metrics were logged unqualified (dice_ds, dice_ds_soft) while universeg
  used the resolution suffix (dice_ds@32). Now both carry @{R}.
- evaluate.py validate: capture `low_res` = non-native model's logit side length; log its coarse
  Dice as `dice_ds@{low_res}` / `dice_ds_soft@{low_res}` (skip entirely for native universeg, which
  previously emitted all-NaN dice_ds/*). Resolution is auto-detected, no config needed for patchset.
- train.py train_epoch: detect non-native low_res, accumulate hard Dice at that grid, and return
  `dice_ds@{low_res}` + `dice_ds_soft@{low_res}` in the metrics dict; returns low_res. main skips the
  generic `train/dice_ds_soft` for non-native models (universeg keeps it = full-res soft). tqdm.write
  and eval_incontext.py print now look up the dice_ds(_soft) key dynamically (any @R).
- eval_incontext.py: also passes ds_metric_res through so standalone eval honors the knob.
- Per-sample SAMPLE_COLS table keeps generic `dice_ds`/`dice_ds_soft` columns (fixed schema); only
  the scalar summary/train metric names got the @R suffix. Legacy eval.py untouched (separate script).
- Verified: patchset smoke logs hard@16/soft@16; universeg smoke logs hard@32/soft@32 (ds_metric_res
  =[32]); checkpoint selection unchanged (cossim / native dice).

## omniSynth: fix border-cell glyph cropping from cell_margin<0
- Bug: with cell_margin<0, oversized glyph tiles in the 12/16 border cells overflowed the canvas
  and _paste clipped them -> 13% of glyphs cropped >1% at margin=-0.15 (up to 42%), 25% at -0.25.
  Interior overlap was fine; only canvas-edge glyphs were cut.
- Fix (render.py _clamp_center): nudge each glyph's paste centre inward so its square tile stays
  fully on-canvas (border glyphs shift <=~5px inward for -0.15). No-op for cell-sized tiles
  (margin>=0). Applied to targets and distractors; positions computed from the clamped paste.
- Verified: crop rate 13%/25% -> 0% at -0.15/-0.25; 22 omniSynth tests pass; glyphs now whole
  (results/experiments/omnisynth_crop_fixed.png vs omnisynth_crop.png).

- train.py (2d): added optional warm-start via `train.checkpoint` (already `null` in
  train_base.yaml). Tolerant load — keeps only name+shape-matching tensors, strips
  `_orig_mod.` prefix, accepts bare state_dict or `{"model": ...}` (mirrors pfn_seg.py).
  Enables retraining, e.g. `... train.checkpoint=.../best.pt`.

## 2026-07-06 — 3D eval: single multi-class loader
- `evaluate.evaluate_classes` now builds ONE dataset over all val classes
  (`common.make_eval_loader`) instead of one loader per class. The scan/bbox
  caches load once per eval instead of ~40×, and the "Loaded scan/bbox cache"
  spam is gone. Results grouped back per class via each sample's `label_name`;
  (rows, cases) shape unchanged, so eval.py + train.py val step are untouched.
- `make_loader(cfg, cls)` kept as a thin wrapper over `make_eval_loader([cls])`.
- Classes with no samples now yield an `{"class", "error": "no samples"}` row.

## 2026-07-06 — Medverse: train-from-scratch option
- Added `train.random_init` (default false). When true, `MedverseModel` builds the
  net from the checkpoint's `hyper_parameters` via `LightningModel(hparams)` and
  skips loading the pretrained `state_dict` -> fresh random weights, same arch.
- `base_ckpt: null` still means "stock pretrained Medverse.ckpt"; `random_init: true`
  is the way to ignore pretrained weights entirely. Wired in train.py build_model.

## 2026-07-06 — 3D train: build val loader once
- The multi-class val loader was rebuilt every eval epoch (dataset + scan/bbox
  caches reloaded each time). `evaluate_classes` now accepts an optional prebuilt
  `loader`; `validate_mean` threads it through; `main()` builds `val_loader` once
  via `make_eval_loader` before the epoch loop and reuses it. Eliminates the
  per-epoch "Loaded scan/bbox cache" reloads. eval.py (no loader arg) unchanged.

## 2026-07-07 — TODO: PatchSetCNN refinement-pass ideas (design only)
- Documented two orthogonal extensions in docs/TODO.md (no code yet):
  (1) patch-level `sampling` maps for context images — drop the sample-axis mask
  for full `r×r` attention (~free, breaks "context read-only"), decode all
  `(K+1)·N` rows via one shared 2-ch `(mask, sampling)` head, discard ctx masks;
  (2) high-res outputs via the Medverse "pool QK / hi-res V" trick — coarse `A`,
  per-patch `f×f` sub-cells folded into V channels, `A@V` + `pixel_shuffle(f)` →
  `R·f` map (only `AV` width + V mem scale by `f²`; scores unchanged; no V proj).
  They compose: hi-res read-out with `q_tok=s_tok` yields ctx maps at high res.
  Reference: Medverse MultiContextSpatialCrossAttention3D (/home/dpxuser/repos/Medverse).

## 2026-07-08 — omniSynth: random glyph placement option
- Added `scene.placement` (grid | random) to `OmniSceneConfig`. `grid` (default)
  keeps the existing cell-centred layout; `random` pastes each of the grid*grid
  glyphs at a uniform-random canvas centre (overlap allowed — union blending keeps
  every glyph). Only the centre choice changes in `render_scene`; k targets,
  samplers, masks, provenance, copy-task logic all unchanged.
- preview_omnisynth.py now prints `placement=`; draw with
  `scene.placement=random`. Samples: results/omnisynth_preview_{random,grid}.png.

## 2026-07-08 — omniSynth: max_nb_objects cap
- Added `scene.max_nb_objects` to `OmniSceneConfig` (0 = no cap). Caps total glyphs
  (targets + distractors) at `min(grid*grid, max_nb_objects)`; k is clamped to
  `[1, n_obj]`. `render_scene` fills a random subset of `n_obj` cells (first k of the
  permutation are targets) and skips the rest — so a cap yields a random subset of
  occupied grid cells (grid mode) or fewer scattered glyphs (random mode).
- preview_omnisynth.py prints `max_obj=`; e.g. `scene.max_nb_objects=4`. Samples:
  results/omnisynth_preview_{random,grid}_cap4.png.

## 2026-07-08 — omniSynth: surface placement + max_nb_objects in configs
- Added `scene.placement` and `scene.max_nb_objects` to the omniglot scene block
  (configs/experiment/2d/synth/omniglot.yaml), the single source both train.py and
  evaluate.py/eval_incontext.py pull via common.build_dataset (OmniSceneConfig(
  **dict(s.scene)) forwards every key — no whitelist). Set at CLI, e.g.
  `synth.scene.placement=random synth.scene.max_nb_objects=4`.

## 2026-07-08 — omniSynth: explore MedSegBench as an object source
- New script experiments/2d/synth/explore_medseg_objects.py: crops every connected
  component of each MedSegBench mask to its bbox and characterises it as a candidate
  omniSynth "object" (binary mask = ink/label + intensity patch = texture). Reports
  per-object bbox/area/fill/intensity/contrast (CSV + text + histograms) and a
  montage of extracted objects (img | masked | mask). Outputs: results/2d/medseg_objects/.
- Findings (train, 128, min_area=16, ≤120 imgs/ds, 20k objects across 35 ds):
  three object regimes — single big blob (busi/isic/kvasir/wbc/pandental,
  ~1 obj/img, bbox 0.24–0.98 of image), dense small instances (cellnuclei/monusac/
  nuset/tnbcnuclei/dynamicnuclear, 19–47 obj/img, bbox ~0.06–0.08), and thin vessel
  networks that shatter into tiny fragments (drive/chasedb1/bbbc010, low fill ~0.2).
  fill median 0.72 (blob-like); intensity+contrast span the full range (some ds have
  darker-than-bg objects, e.g. busi/isic contrast ≈ -60; dynamicnuclear img ≈ black).
- Implications for the bank: filter by min_area + maybe min_fill to drop vessel
  fragments; pick instance-rich datasets for many-object scenes; carry the intensity
  patch (not just binary) so pasted objects keep realistic texture/contrast.

## 2026-07-08 — omniSynth: MedSegBench object source
- New object bank src/datasets/omniSynth/bank_medseg.py (MedSegObjectBank), a drop-in
  alternative to OmniglotBank exposing the same task_ids/get/alphabet interface. A
  "class" = (dataset, label_value) [alphabet(cid)="<dataset>/label_<lv>"]; a rendition
  = one image's WHOLE binary mask for that label (all connected components kept, so
  multi-component objects stay intact), bbox-cropped + resized into a cell tile, stored
  as [2,tile,tile] float32: ch0 = intensity (0..1, zeroed outside mask), ch1 = binary
  mask. Classes split train/val/test by seeded permutation (novel-class, like Omniglot).
- render.py generalised to 2-channel renditions via _split(tile): glyph 2D bitmap ->
  img==mask; medseg [2,h,w] -> paste ch0 into image, ch1 into mask. affine_jitter is
  channel-aware (intensity kept continuous + re-masked, mask re-binarised). Glyph path
  and all 21 existing tests unchanged.
- Config: new OmniMedSegConfig + `medseg` block in synth/omniglot.yaml (source=omniglot
  default). Flip with synth.medseg.source=medseg (+ datasets/train_frac/... overrides).
  Wired through common.build_dataset, dataset.py (bank selector), preview_omnisynth.py.
- Tools: experiments/2d/synth/preview_medseg_bank.py (rendition montage) and
  explore_medseg_objects.py (object stats). Tests: test_bank_medseg.py + 2 render tests.
  Previews: results/omnisynth_medseg_{random,grid}.png, results/medseg_bank_preview.png.
- Usage: python experiments/2d/train.py synth=omniglot synth.medseg.source=medseg \
    synth.scene.placement=random synth.scene.max_nb_objects=6

## 2026-07-08 — omniSynth medseg: split by MedSegBench image splits
- Changed MedSegObjectBank splitting (superseding the earlier seeded novel-class split).
  The bank is now scoped to one split: the "train" bank reads each dataset's TRAIN
  images, the "val" bank its VAL images (no test set — val doubles as test). Config:
  replaced train_frac/val_test_split/datasets/master_seed with train_datasets and
  val_datasets ([] = all) selecting which datasets feed each split. Default (both [])
  = same classes in train and val, but drawn from different underlying images.
- get_or_build_medseg_bank / MedSegObjectBank take `split`; dataset.py passes its split.
  task_ids() returns the single scoped pool. Fixes the earlier empty-val edge case with
  few datasets. Updated omniglot.yaml medseg block, preview scripts, test_bank_medseg.py.

## 2026-07-08 — omniSynth medseg: canvas-relative object sizing
- Objects no longer uniformly resized to the cell. New OmniMedSegConfig.size_mode:
  - canvas (default): each object tile is scaled by canvas/source so it keeps its size
    relative to the full canvas (bbox that filled f of its source image -> ~f of the
    canvas), aspect ratio preserved, centred in a square tile. Per-placement aug then
    jitters around this. size_scale multiplies the preserved size.
  - cell: previous behaviour (every object resized to (1-2*margin)*cell, glyph-like).
- Wiring: MedSegObjectBank / get_or_build_medseg_bank take image_size (canvas side;
  defaults to source_size); dataset.py passes it; cache key includes it. No render
  changes — variable/oversized tiles were already handled by _clamp_center + union paste.
- Verified: tile px varies per object (busi med 30, isic 88, cellnuclei/drive ~full
  canvas since their whole-mask bbox spans the image) and rescales exactly with
  image_size (128->64 halves sizes). Previews: results/omnisynth_medseg_{canvas,cell}.png.

## 2026-07-08 — omniSynth: random background (dark objects stay visible)
- Problem: dark medseg object textures vanished on the black canvas. Fix mirrors
  controlSynth's non-black GMM background (bg_center~0.5 off the extremes).
- OmniSceneConfig.background: "black" (default, zero canvas, unchanged) | "random"
  (smooth low-freq grey field via upsampled random + gaussian noise; bg_intensity,
  bg_structure, bg_noise knobs). "black" draws no rng, preserving existing seeds.
- render.py: image now composited with premultiplied-alpha _composite (canvas*(1-mask)
  + img_tile) instead of np.maximum, so an object's true texture (bright OR dark)
  REPLACES the background under its mask rather than being hidden by a lighter bg.
  Label mask keeps its union paste. For a black canvas _composite == the old maximum,
  so the glyph path and all prior tests are unchanged (30 pass).
- Verified: busi target texture (0.005..0.46, incl. near-black) now visible over a
  0.41 grey bg. Preview: results/omnisynth_medseg_bg_random.png.

## 2026-07-08 — omniSynth: cheap anti-overlap for random placement
- Rejection sampling in render_scene: for random placement, each object is sampled
  first, then up to placement_tries candidate centres are drawn and the least-overlapping
  (vs a running boolean occupancy of already-placed masks) is kept — accepted early once
  overlap fraction <= placement_max_overlap. New OmniSceneConfig fields placement_tries
  (default 1 = fully random, no rejection; yaml sets 8) + placement_max_overlap (0.25).
  Helpers _tile_slices/_occupy/_overlap_frac/_place_random; O(objects*tries), tiny cost.
- Loop refactor: object sampled before its position (so its mask drives placement). Grid
  path unchanged incl. rng order (position uses no rng); random path rng order shifts by
  one (object sample now precedes the position draw) — deterministic, just different scenes.
- Verified on real medseg objects (5 datasets, size_scale 0.5, 6 obj/scene): overlapped-
  pixel fraction 0.101 (tries=1) -> 0.033 (4) -> 0.031 (8) -> 0.028 (16); ~3x less overlap
  at the tries=8 default. Previews: results/omnisynth_medseg_tries{1,8}.png. 31 tests pass.

## 2026-07-08 — omniSynth: BiomedParse object source
- Added source=biomedparse alongside medseg. New bank_biomedparse.py
  (BiomedParseObjectBank), same task_ids/get/alphabet interface + [2,tile,tile]
  rendition format. class = (dataset, target) with target parsed from the mask
  filename (_parse_mask_stem, '+'->space); alphabet(cid)="<ds_key>/<target>".
  Reads the pre-resized store <root>/<split>/<ds>/{images,masks}_{size}.npy +
  index_{size}.npz (reusing biomedparse._discover_stores/_index_from_npz); mask row
  -> image row via filename stem. split: train->train store, val/test->test store.
- Factored shared sizing/tiling into bank_common.py (make_object_tile + crop_to_tile);
  MedSegObjectBank refactored to use it (behaviour identical, its 4 tests still pass).
- Config: source now omniglot|medseg|biomedparse; added biomedparse_root. dataset.py
  branches on source; yaml documents it. common.build_dataset unchanged (OmniMedSegConfig
  carries the selector). 41 datasets / 109 (dataset,target) classes (test split).
- Verified: bank builds (train 6 / val 4 on a subset), scene renders
  (results/omnisynth_biomedparse_random.png), train config builds 36/35 classes.
  Tests: test_bank_biomedparse.py (3). Full suite 34 passed.
- Usage: python experiments/2d/train.py synth=omniglot synth.medseg.source=biomedparse \
    synth.scene.placement=random synth.scene.background=random

## 2026-07-08 — omniSynth: un-nest object source selector
- Moved the object-source selector from synth.medseg.source to a top-level
  synth.source (omniglot | medseg | biomedparse); the `medseg` block now holds only
  the real-object params (shared by medseg + biomedparse). Removed `source` from
  OmniMedSegConfig; added a `source` arg to OmniSynthICLDataset (selector uses it).
  common.build_dataset reads s.source; preview_omnisynth.py reads cfg.source. Updated
  omniglot.yaml + bank tests. All 34 tests pass; medseg/biomedparse/omniglot build
  end-to-end via synth.source.

## 2026-07-08 — eval: option to drop per-class metric keys
- Added eval.log_per_class (eval_incontext.yaml, default false there) to suppress the
  per-dataset (dice/dataset/*) and per-class (dice/class/*) breakdown keys in the wandb
  summary — one-per-letter/object-class entries that the per-sample table already covers.
  mean + macro aggregates and the console table are always kept.
- Threaded as per_group through validate() -> log_summary() (common.py); defaults True so
  train.py and other eval scripts are unchanged. eval_incontext.py passes
  cfg.eval.get('log_per_class', True). Verified: per_group=False yields only */mean + */macro
  (identical values), dropping all per-group keys.

## 2026-07-08 — eval: accurate inference timing (cuda sync + warmup)
- evaluate.py validate(): time/inference_ms now brackets the per-batch forward with
  torch.cuda.synchronize() (CUDA only), so it measures real GPU forward compute instead
  of async kernel-launch time. The first timed batch is dropped as warmup (cudnn.benchmark
  autotune / allocator / lazy init). No-op on CPU. FLOPs batch still excluded from timing.

## 2026-07-09 — omniSynth: real-image backgrounds (scene.background=image)
- New background mode: scene.background=image uses a random real full image from
  medseg/biomedparse as the canvas (objects composited over it, as before). Independent
  of the object source (e.g. biomedparse objects on medseg image backgrounds, or omniglot
  glyphs on real backgrounds). New scene fields: bg_source (medseg|biomedparse),
  bg_datasets ([]=all), bg_max_images (pool cap). Roots/source_size reused from the medseg
  block; split maps train->train, val/test->val (medseg)/test (biomedparse).
- bank_background.py (BackgroundBank + get_or_build_background_bank): pools full images
  per split (budget spread across datasets; medseg copies the kept slice, biomedparse
  keeps memmaps), samples one and resizes to the canvas on demand. Process-shared cache.
- render_scene/_make_background take an optional background_sampler; dataset.py builds the
  bank when background=image and threads sampler.sample. render stays PIL-free (sampler
  returns canvas-sized image). Objects still painted over via _composite (dark textures
  visible on any background).
- Tests: test_bank_background.py (2) + a render test; full suite 37 passed. Preview:
  results/omnisynth_bg_image.png. Usage: synth.scene.background=image
  synth.scene.bg_source=medseg [synth.scene.bg_datasets=[...]].

## 2026-07-09 — omniSynth: draw targets last (on top of distractors)
- render_scene now defers target image-compositing to after all distractors, so a
  distractor overlapping a target can no longer overwrite the target's texture in the
  image. Placement, occupancy (anti-overlap), rng order, the label mask and target
  positions/transforms are unchanged — only the target/distractor overlap pixels in the
  image differ (target wins). Targets keep their mutual paste order.
- Test: test_targets_drawn_over_distractors (oversized fully-overlapping tiles ->
  target texture 0.5 survives under 0.9 distractors). Render tests 13 passed.

## 2026-07-09 — 2D trainer: Muon + LAWA for patchset_cnn
- The `muon_*`/`lawa_k` keys in `train_base.yaml` were dead in the unified
  `experiments/2d/train.py` (only pfn_seg.py used them). Wired them in so patchset_cnn
  trains with the same recipe as pfn_seg.py: Muon (Newton-Schulz orthogonalized grads)
  on the transformer 2D weight matrices (`p.ndim==2 and "transformer" in n` → 8 tensors:
  qkv_col/qkv_row/mlp.0/mlp.2 per block), AdamW on everything else (encoder convs,
  embeds, pos, decoder, thinking, ctx-id), plus LAWA checkpoint averaging.
- Always-on for patchset_cnn, gated by `is_patchset = model_name=="patchset_cnn"`.
  universeg keeps its exact prior path (single AdamW, no LAWA) — its Muon group would be
  empty and both Muon and the LAWA queue are skipped, so its baseline is unchanged.
- Scheduler (cosine+warmup, per-batch) drives AdamW only; Muon LR is constant
  (`muon_lr_scale·lr`). `train_epoch` now takes an `optimizers` list (zero_grad/step loop).
- LAWA: deque(maxlen=lawa_k), push CPU state_dict each epoch; before `validate` average
  the queue into the model (eval + any best-checkpoint save use averaged weights), then
  restore raw weights so training continues from them. Epoch-1 (len≤1) averaging is a
  no-op via lawa_average's own guard.
- Verified on the real PatchSetCNN (CPU): param split 8 Muon / 31 AdamW, 3 train steps
  decrease loss, LAWA average changes weights and restore recovers them exactly.
- Comparison caveat: this changes the patchset_cnn recipe, so its current checkpoints are
  not a clean optimizer before/after vs universeg. Design doc:
  docs/superpowers/specs/2026-07-09-muon-lawa-patchset-cnn-design.md

## 2026-07-09 — omniSynth: fix empty masks for tiny objects
- Symptom: ~4/10 preview samples (medseg, canvas sizing, size_scale<1) had fully empty
  target+context masks — all from tiny/sparse classes (idrib microaneurysms, m2caiseg
  rare labels). min_mask_px filters the SOURCE mask, but the mask vanished downstream.
- Two vanishing points, both bilinear-resize + 0.5 threshold on sparse masks:
  1. bank_common.make_object_tile: canvas downsizing blurred tiny masks below 0.5.
     Fix: if the >=0.5 mask is empty, fall back to >0 (keep any coverage); crop_to_tile
     also drops any rendition whose tile mask is still empty (excludes hopeless classes).
  2. render.affine_jitter (target_mode=aug): the warp re-thresholded at 0.5 and could
     blur/push a tiny mask off-tile. Fix: >0 fallback, then if still empty return the
     un-jittered base (bank guarantees it is non-empty).
- Result: empty target masks 4/10 -> 0/200. Full suite 38 passed.

## 2026-07-14 — Per-level soft-Dice eps: empirical A/B on small objects
- Exp 4 (`4_loss_eps_per_lvl.yaml`, run 9j69mib5, eps {32:0.01, 64:0.1}) vs exp 3
  baseline (run 03ypf2pk, eps=1). A/B on the deterministic val set (3450 paired samples),
  using the per-sample columns logged by evaluate.py. B still training (ep34) vs A (ep39),
  so numbers are a lower bound.
- Result — gain is confined to small objects, exactly as the loss theory predicted:
    ≤32px final dice 0.246→0.339 (+0.09); complete-miss (dice==0) 64%→45%
    33-128px +0.03; ≥129px ≈0; MICRO final 0.701→0.727 (all from small buckets)
- Mechanism confirmed: coarse-grid survival identical (GT property) and cossim@32 (ranking)
  barely moves; the coarse OCCUPANCY jumps (dice_soft@32 0.167→0.237) — the loss now turns
  surviving small-object cells ON instead of suppressing them (eps=1 inverted ∂dice/∂p below
  g*≈0.37). Added the payoff cell + `artifacts/12_eps_ab_payoff.png` to
  results/experiments/12_small_occupancies.py; cached tables in `artifacts/12_eps_ab.csv`.

## 2026-07-14 — Notebook 12 refactor: 3-way occupancy analysis
- Extracted `get_latest_table` (+ SZ_EDGES/SZ_LABELS, add_szbin) to shared
  `results/experiments/nb_common.py`; notebook 11 now imports it.
- Rewrote `12_small_occupancies.py` (simplified): kept one compact g* loss-theory cell,
  replaced the eps-scheme design/per-stage cells with a 3-run occupancy comparison —
  patchset_eps1 (03ypf2pk) vs patchset_epslvl (9j69mib5, per-lvl eps) vs universeg (08zmho80).
  Reads one cached CSV (`artifacts/12_occ_runs.csv`); backfills universeg's missing GT size
  cols from the patchset table (model-independent). Fixed savefig to absolute __file__ paths
  (was FileNotFoundError under marimo's CWD).
- Findings (final trained dice by size): universeg leads ONLY ≤32px (0.362); the eps fix
  lifts patchset ≤32px 0.246→0.341 (gap to uv now −0.021), complete-miss 64%→45%. At 33-128px
  patchset already beats universeg 0.680 vs 0.571 (uv misses 15% completely). ≥129px all tie.
  Mechanism: cossim@32 flat, coarse occupancy dice_ds_soft@32 +0.069 / dice_coarse +0.081 —
  fix activates surviving cells, survival itself unchanged (0.277, GT property).

## PatchSetCNN: shaped mask token (mask_patch_size)
- Added `mask_patch_size` (p) to `PatchSetCNN` (src/models/patchset_cnn.py). Mask token was a
  single scalar (avg-pool occupancy) → `Linear(1,e)`. p>1 now resamples each patch's mask to a
  fixed p×p tile → `Linear(p²,e)`, so a *shaped* occupancy (which sub-region of the cell is
  foreground) reaches the transformer, not just the fraction. Mirrors experiments/2d/multilevel
  (`_mask_tiles`, `mask_prior=patch`, PatchSetPFN.mask_embed=Linear(p²,e)).
- New module helper `_mask_tiles` (bilinear-resize to grid*p, exact reshape to per-cell p²).
  Generalized `_occupancy` (grid + encode_once paths), `_attn` query-prior expand, and the
  scatter path (support = true-mask tiles, query = coarse-prob tiles). p=1 keeps the scalar
  avg-pool branch → default runs byte-identical.
- Wired `arch.mask_patch_size` through train.py build_model + configs/.../model/patchset_cnn.yaml.
  RESOLUTION-AGNOSTIC by design: each cell's native mask patch (image_size//resolution px, which
  varies with resolution) is resized to a FIXED p×p tile, so `mask_embed` is always `Linear(p²,e)`
  and shareable/transferable across resolutions. Default p=8; `1` = scalar avg-pool occupancy.
  Verified mask_embed.in stays 64 at resolution 32 (4px/cell) and 16 (8px/cell). Smoke-tested
  single-level / scatter / encode_once at p∈{1,8}.

## PatchSetCNN: tiled decoder (decode_patch) — reconstruct higher/original res
- Added `decode_patch` (d) to `PatchSetCNN`. Decoder head was `Linear(e,h)→GELU→Linear(h,1)`
  (one logit per query token → R×R prediction). Now `Linear(h,d²)`: each of the R² query tokens
  decodes a d×d block, tiled (inverse of `_mask_tiles`, verified exact round-trip) into an
  (R·d)×(R·d) map — a higher-res mask with NO upsampling stage. d = image_size//resolution
  reconstructs the ORIGINAL input resolution (e.g. 128/32=4 → full 128×128). d=1 = unchanged
  default (checkpoint-compatible).
- Guarded: tiled decode asserts on the scatter/flat_out path (d must be 1 there). Grid +
  encode_once/reencode refine paths tile fine (refine_logit becomes (B,1,R·d,R·d)).
- Loss/metrics are res-agnostic in train.py (_target_like pools GT to the logit size); with d>1
  the logit hits native res so it trains/monitors like a native model. NOTE a `dice_eps.{32}`
  entry no longer matches when the logit side becomes R·d (falls back to default 1.0).
- Wired arch.decode_patch through train.py build_model + model/patchset_cnn.yaml (default 1).

## 3D experiments: per-sample val/eval detail table (port of 2D sample table)
- `experiments/3d/evaluate.py`: `evaluate_classes` now enriches each per-case dict with
  `tgt_size`/`tgt_occ` (target GT fg voxels + fraction) and `ctx_size`/`ctx_occ` (mean over
  the K contexts) via `_occupancy_stats`, plus a source-adaptive `detail` string
  (`_sample_detail`: omniSynth3D → "mode=.. class=.. sub=..", totalseg → "").
- New `build_sample_table(cases, epoch=None)` → wandb.Table with fixed cols
  [epoch, class, subject, dice, time_ms, tgt_size, tgt_occ, ctx_size, ctx_occ, detail];
  shared by train + eval so both log the same schema (epoch=-1 for standalone eval).
- `train.py`: `validate_mean` returns cases; val step logs `val/samples` (mirrors 2D).
- `eval.py`: replaced the 4-col `cases` table with the full `build_sample_table`.
- Plumbing for `detail`: omniSynth3D dataset3d emits a `meta` dict (class_id, sample_index,
  resolved target_mode); `incontext_collate_fn` passes `meta` through when present. TotalSeg
  items carry no meta → empty detail (columns stay fixed across sources).

## 3D Medverse loss: hardcode paper's cubic smooth-L1, drop beta knob
- `experiments/3d/train.py` `SmoothL3L1`: removed the `beta` parameter. The general-beta
  linear branch (`n + β³/3 − β`) was only C1-continuous at β=1 (the sole value used);
  now hardcoded to the paper's (Hu et al. 2025) form: `L(n)=n³/3` for n<1 else `n−2/3`.
- `build_loss` calls `SmoothL3L1()` (no beta); dropped `smooth_l1_beta` from
  configs/experiment/3d/model/medverse.yaml. loss_scale (×50) unchanged.

## 3D training: log val loss + train/val soft-Dice
- `train.py` `train_epoch` now also accumulates soft Dice (1 − soft-Dice-loss on σ(logits))
  → logs `train/dice_soft` (alongside existing `train/dice` hard + `train/loss`).
- `evaluate.py` `evaluate_classes` gained optional `logits_fn`/`loss_fn`: when given (train.py
  val step passes `model.train_forward` + the training `loss_fn`), each case also gets
  `soft_dice` (σ(logits) vs GT, threshold-free) + per-sample `loss`; `_summarize` adds
  `mean_soft_dice`/`mean_loss`. Hard `dice` still comes from `model.predict` (benchmark
  inference), so `val/dice` is unchanged. eval.py passes neither → its path is byte-identical.
- `validate_mean` returns (mean_dice, mean_soft, mean_loss, rows, cases); `train.py` logs
  `val/dice_soft` + `val/loss`. Val soft/loss use a single-ROI logits forward (valid at
  image_size==sw_roi), self-consistent with the training criterion.
- `build_sample_table` gained `soft_dice` + `loss` columns (NaN for the eval.py benchmark).
- Added `soft_dice_binary(prob, target)` helper (eps=1e-6, matches train.py `_soft_dice`).

## PatchSet3D: torch.compile + Muon + LAWA (port from 2D trainer)
- `experiments/3d/train.py`: mirrored experiments/2d/train.py's optimization stack for the
  `patchset3d` model (medverse path unchanged).
  - **compile** (`arch.compile`): `torch.compile(net.transformer, dynamic=True)` after the
    checkpoint load — the conv encoder (adaptive_avg_pool3d / trilinear) stays eager. Also
    compiles the shared `pfn_train._newtonschulz5_batched`.
  - **Muon + AdamW** (`train.muon`, default true): Muon on transformer 2D weight matrices
    (Newton–Schulz orthogonalized grads) + AdamW on everything else. `train_epoch` now takes a
    list of `optimizers` and iterates zero_grad/step; the scheduler drives AdamW (optimizers[0])
    only, Muon is unscheduled. `muon_lr_scale`/`muon_momentum`/`muon_wd` config keys.
  - **LAWA** (`train.lawa_k`): per-epoch CPU state_dict pushed to a deque; at eval the queue is
    averaged into the model (`lawa_average`), evaluated + checkpointed, then raw weights restored.
  - Checkpoint save strips the `_orig_mod.` compile prefix so checkpoints stay compile-agnostic;
    resume also strips it. Reuses `Muon`/`lawa_average` from experiments/2d/pfn_train.py
    (2D dir appended to sys.path).
  - Node-local compile caches: set `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` on /tmp keyed by
    hostname BEFORE importing torch (cf. 2D trainer) — without this, compile hit the poisoned NFS
    Triton cache (`GLIBC_2.34 not found` from a cuda_utils.so built on a newer-GLIBC node).
- `configs/experiment/3d/model/patchset3d.yaml`: added `arch.compile: true` and
  `train.{muon, muon_lr_scale, muon_momentum, muon_wd, lawa_k}`.

## anchor_synth3d dataset

Added `data.source=anchor_synth3d` (`dataset=anchor_synth3d`): pulls K+1 real CT
scans that share an anchor organ and draws a synthetic blob at a consistent
anchor-relative position (offset normalized to anchor extent, small per-scene
scale/rotation jitter, contrast blended to local background). Anchor is a
landmark only — the label is the drawn object(s). New package
`src/datasets/anchor_synth/` (analytic shapes + placement); subclasses
`TotalSegInContextDataset` for the scan cache + fast-path loading. v1 = blob
objects only; organ objects and multi-anchor deferred. Spec:
docs/superpowers/specs/2026-07-22-anchor-synth3d-design.md.

## patchset3d eval support

`experiments/3d/eval.py` can now evaluate a `model=patchset3d` checkpoint. Added
a `patchset3d` branch to `_build_model` that reuses `train.build_model` to
instantiate `PatchSet3D` (used directly as the eval model — it provides the
`.predict` the shared eval loop needs) and loads the checkpoint's `model` state
dict (`_orig_mod.` stripped). Architecture is rebuilt from the checkpoint's new
`arch` field; `train.py` now stores `arch` (patchset3d only) alongside the state
dict so eval no longer needs the `arch.*` overrides re-supplied. Older
checkpoints without stored arch still work by passing `+model=patchset3d arch.l=...`.

    python experiments/3d/eval.py eval.model=patchset3d \
        eval.checkpoint=results/checkpoints/3d/<DATE>_<run>/best.pt \
        dataset=omnisynth3d eval.split=val

## 3D eval reproducibility fix (per-item RNG)

`experiments/3d/eval.py` on the totalseg source was not reproducible across models:
context selection (`random.shuffle(candidates)`) and organ-crop jitter
(`random.randint`, crop_jitter=32, use_crop=true) drew from the process-global
`random`. With `workers>0`, each worker's `random` is seeded by PyTorch as
`base_seed + worker_id`, and `base_seed` comes from the main-process torch RNG at
loader-spawn time — which is perturbed *differently by each model's construction*
(PatchSet3D random init vs medverse pretrained load). Result: two runs saw
different context (98.4% of samples) and different crops (50.3%), so paired
medverse-vs-patchset comparisons were not on identical inputs.

Fix: `TotalSegInContextDataset` gains `eval_seed`. When set, `__getitem__` seeds a
per-item `random.Random(hash((eval_seed, idx)))` (`self._cur_rng`) and routes the
context shuffle, pad-resample, multi-label extra-class shuffle, and crop jitter
(`_load_crop`/`_load_crop_multi`) through it — fully reproducible regardless of
worker count, iteration order, or model. `eval_seed=None` (training) keeps the
global `random`, so training stochasticity is unchanged. `make_eval_loader` passes
`eval_seed=cfg.eval.seed` (default 0), covering both the eval entrypoint and
train-time per-class validation.

## nb 20 (medverse vs patchset3d): all-TotalSeg-class support + per-class dice plot

The new eval runs cover all TotalSegmentator labels (not the earlier 47-class
subset), which broke `results/experiments/20_medverse_patchset3d_comp.py`:
- The shape-taxonomy assert crashed with `'tuple' object has no attribute 'isna'`
  (`D.shape` is the DataFrame `(rows, cols)` tuple, not the `shape` column). Fixed
  the reference, then replaced the hard assert with a catch-all: unmapped classes
  now fall into a new `"other"` shape family (added to `SHAPE_ORDER`) and are
  printed, instead of failing.
- Removed hardcoded `/47` class counts in cells 0b and 1 — derived from `len(...)`.
- Added cell 1b (per-class dice analysis + plot): per-class mean dice for every
  run; for a 2-run pair, a run-vs-run scatter coloured by shape (y=x reference)
  plus a diverging bar of the largest per-class gaps; for >2 runs, per-class dice
  sorted with one line per run. Generalises to any RUNS set.

## nb 20: data-driven morphology taxonomy (clustered, auto-labelled)

Replaced the hand-mapped 4-family SHAPE dict (compact/tubular/elongated/bone) with a
data-driven taxonomy clustered from real-mask geometry. Two reusable helpers added to
`results/experiments/totalseg_geometry_extract.py`:
- `load_or_build_geometry(pairs, cache, ...)` — cache-or-rebuild per-(subject,class)
  geometry; now shared by nb 20 cells 0 and 4 (single cache, no NFS double-read).
- `shape_families(geom, k=10)` — Ward clustering (scipy, numpy-standardised, no sklearn)
  on scale-invariant shape descriptors + thickness + fragmentation. Auto-labels each
  cluster `{thick|mid|thin}_{blob|tube|sheet}` (+ `frag` for multi-component): thickness
  tercile from surf/vol (primary axis, = the medverse<->patchset dice driver); the shape
  tag is the Westin coord (linearity/planarity/sphericity) the cluster stands out on vs
  other clusters (z-score argmax, so it isn't swamped by linearity being globally largest).

Cell 0 now sets N_SHAPE=10 and derives SHAPE/SHAPE_ORDER at runtime; classes with no
non-empty mask fall back to 'other'. Key finding: geometry does NOT support "bone" as a
morphology family — bones scatter by actual shape (femur/humerus->thick blob, ribs->thin
tube, vertebrae->mid sheet, flat bones->mid). k=10 has the best silhouette (0.331) in 8-13
with no singletons; example families: thick_blob1 = liver/heart/brain/lungs/spleen/bladder,
thin_tube* = ribs + arteries, mid_sheet = vertebrae.

## nb 21: reusable single 3D training-run val breakdown (21_totalseg_seeds3d.py)

New marimo notebook for the FINAL-epoch val/samples breakdown of one patch_icl_3d_exps run
(default RUN = d7fk2k9h, patchset3d + seeds3d synth). General/uncluttered: RUN is the only
knob (N_SHAPE tunes family granularity); class list + shape families are derived from the
logged data, nothing hardcoded. Aggregate learning curves deliberately omitted (live in W&B).
Reuses totalseg_geometry_extract.{load_or_build_geometry, shape_families} for morphology (same
taxonomy as nb 20; here clustered on the evaluated pairs). Caches samples, config+summary, and
geometry under artifacts/21_<id>_*.{csv,json}. Cells: (0) fetch+cache+cluster+header; (1)
per-class val dice ranked bar coloured by shape family (+table); (2) per-shape family breakdown
(macro/micro dice, complete-miss rate, median thickness); (3) per-sample dice vs geometry
drivers (thickness, volume, target/context occupancy). Finding for d7fk2k9h: dice falls
monotonically with thinness — thick_blob1 macro≈0.20 (miss 39%) → thin_tube≈0.005 (miss 87%);
macro val/dice≈0.083, 11/47 classes complete-miss.

## fix: log Hydra group choices to wandb (3D train/eval)

`dataset=`, `augmentations=`, `model=`, `cluster=`, `experiment=` config files are all
`# @package _global_`, so a group selection merges into `data:`/`paths:`/... and leaves no
key of its own in the composed cfg. wandb logs `OmegaConf.to_container(cfg)`, so the *choice*
(e.g. which dataset) was never recorded — only Hydra's `runtime.choices` holds it. Now log
`HydraConfig.get().runtime.choices` under a `hydra_choices` key in both experiments/3d/train.py
and eval.py, so dataset/augmentations/etc. are visible in wandb. (`model` was already visible
only because model/*.yaml sets an explicit scalar `model:` key.)

## bench: medverse vs patchset3d compute/memory profile (experiments/3d/bench_arch.py)

Investigated why medverse costs ~190s/epoch + 36GB vs patchset3d 50s + 17GB under
experiment=1_medverse_benchmark. Added experiments/3d/bench_arch.py: isolates *model*
compute (no dataloader) at the real runtime shapes (B=1, K=1, 128^3), reproducing
train_epoch's exact fwd/bwd call paths, with param-by-submodule + torch-profiler op
breakdown. Ran on loki (RTX 6000 Ada 48GB, .venv_nero torch 2.12+cu130).

Findings (B=1, K=1, 128^3):
- Params: medverse 71.1M (context_unet 35M / target_decoder 20M / target_encoder 16M)
  vs patchset3d 4.7M (transformer 4.0M / rest tiny). 15x.
- fwd+bwd: 185ms vs 63ms (2.9x, matches the 190/50s epoch ratio -> genuine model
  compute, not dataloader). fwd-only 72 vs 22ms.
- Op profile (self CUDA): medverse is 3D-conv bound — convolution_backward 34% +
  conv fwd 14%; plus avoidable overhead: nchwToNhwc layout transpose 11% and
  aten::copy_/.to() 15% (the Medverse forward repeatedly `.to(device)`s already-on-
  device tensors, cf. Medverse.py L113-118/133-138). patchset3d: flash-attn 16%,
  group_norm, conv only 10%.
- Root cause: medverse runs a native-res 3D U-Net triple (~370 conv3d over 128^3..16^3
  multiscale); patchset3d downsamples each volume to a 16^3 grid (~21 convs) and does
  in-context matching as a transformer over 16^3=4096 tokens. Conv cost scales with
  voxel count (128^3 vs 16^3 = 512x/channel) -> dominant driver. Not primarily
  inefficiency; a different compute regime.
- VRAM reconciled at B=4 (both runs used B=4): reserved medverse 38.5GB (~reported 36),
  patchset3d 18.3GB (~reported 17). Driver is activation memory x batch, NOT
  cudnn.benchmark (no effect measured) nor eval (medverse predict B=8 only 3GB).
  Isolated train peak_alloc: medverse 8.2G(B1)/15.3G(B2)/29.5G(B4); patchset3d
  3.0/5.9/11.7G.

Optimization levers for the medverse runs (time + VRAM):
- gradient checkpointing on U-Net stages (biggest VRAM lever; ~2-3x less activation mem).
- channels_last_3d memory format -> removes the nchwToNhwc transpose (~11%).
- drop redundant `.to(device)` copies in the Medverse fork forward (~15% copy_).
- torch.compile the medverse net (currently only patchset3d's transformer is compiled).

## bench: medverse optimization prototypes (channels_last / ckpt / compile)

Prototyped the medverse levers from the previous entry, applied *externally* (the NFS
fork is untouched): gradient checkpointing via a non-reentrant `torch.utils.checkpoint`
wrapper on every conv block of the 3 U-Nets (34 blocks), `channels_last_3d` on conv
weights, and `torch.compile` on the whole net. Toggles + a variant runner added to
experiments/3d/bench_arch.py (`--optims -B <n>`; `bench_medverse_variant`). loki RTX 6000 Ada.

Results (fwd+bwd, K=1, 128^3, alloc/reserved GB):
  B=1  baseline                178.6ms  8.20/9.50
       channels_last           260.8ms  8.19/9.20   <- HURTS (+46% time)
       gradient_checkpointing  235.8ms  3.93/5.27   (-52% alloc, +32% time)
       compile                 161.3ms  8.10/8.85   (-10% time, free)
       compile+ckpt            196.7ms  5.31/6.19
  B=4  baseline               1040.8ms 29.54/38.65  (~matches reported 36G)
       gradient_checkpointing 1279.2ms 13.45/21.41  (-54% alloc, +23% time)
       compile+ckpt            858.6ms 18.23/24.90  <- WINNER: -17% time AND -38% alloc
  B=8  gradient_checkpointing 2351.6ms 24.89/32.20  (B=8 now fits in <baseline-B=4 mem)

Takeaways:
- channels_last_3d is a LOSS here (my profile hypothesis was wrong) — forcing NHWC makes
  cudnn pick slower kernels and the 6D context reshapes fight the format. Drop it.
- gradient checkpointing = the memory lever: ~-50% activation mem for +23-32% time; lets
  B double (B=4->B=8) within the same budget. Verified EXACT: fp32 loss bit-identical to
  baseline; grad diff (9e-3) sits inside the baseline-vs-baseline conv-nondeterminism band
  (6.8e-3), i.e. it's cudnn reduction-order noise, not a checkpoint error.
- torch.compile the whole medverse net compiles cleanly (no fatal graph break) and, at the
  real B=4, compile+ckpt is strictly better than baseline on BOTH time (-17%) and memory
  (-38%). Recommended default for the medverse training runs. (compile costs a slow first
  batch; amortized over an epoch.)
Not yet wired into training — prototypes live in bench_arch.py; integrating as train.py /
medverse-adapter flags is the follow-up.

## feat: wire medverse compile + gradient-checkpointing into training

Promoted the two winning prototypes from the bench entry above into the training path as
config flags. `configs/experiment/3d/model/medverse.yaml` gains a `medverse:` block —
`grad_checkpoint` (false) and `compile` (false). experiments/3d/train.py applies them after
weight load (guarded `if not is_patchset`), mirroring the patchset3d compile block.
Implementation lives in the adapter src/benchmark_models/medverse.py:
- `enable_gradient_checkpointing()` — monkeypatches each U-Net conv block's `forward` with
  non-reentrant `torch.utils.checkpoint` IN PLACE (no wrapper submodule), so param names —
  and thus saved checkpoints — are unchanged; a no-op passthrough when grad is disabled so
  `predict()`/eval are untouched. Returns #blocks wrapped (34).
- `compile_net()` — `torch.compile(self.model.net)`; the `_orig_mod.` prefix it adds is
  stripped by train.py's existing save logic.
bench_arch.py now calls these adapter methods (drops its local `_Ckpt`) so the benchmark
exercises shipping code; channels_last stays bench-only (measured a loss).

Verified: (1) checkpointing grads exact — bit-identical fp32 loss, grad diff 9e-3 within the
baseline-vs-baseline conv-nondeterminism band; param name set unchanged (169). (2) compile+ckpt
runs together via the monkeypatch path (B=1 192ms / 5.3G, matches the wrapper prototype).
(3) full save->load roundtrip with BOTH opts on: 172 keys, no `_orig_mod.`/`.mod.` leakage,
reloads into a fresh eval model + predict() OK. (4) Hydra compose: medverse block defaults
false, overrides work, patchset3d has no medverse key.

Recommended run: `train.py experiment=1_medverse_benchmark medverse.compile=true
medverse.grad_checkpoint=true` — at B=4 that's ~858ms/step vs 1041 eager (-17%) and ~18GB
vs 30GB alloc (-38%).

## Val sample table: `in_train` flag (2026-07-24)

Added an `in_train` boolean column to the val/samples wandb table (experiments/3d).
`build_sample_table` now takes an optional `train_classes` set and tags each row with
`case["class"] in train_classes`. train.py resolves `cfg.data.train_classes` once (same
call as common.py's loader) and passes it in. With the default benchmark/not_benchmark
split the two class sets are disjoint, so val classes read False — meaningful only once
train/val overlap. No source guards yet (anchor_synth3d/omnisynth3d val "classes" are
shapes/tile-ids, so they currently read False across the board).

## 2026-07-24 — thor toolchain fix for torch.compile (broken usr-merge)

`medverse.compile=true` under `.venv_thor` failed with inductor `CppCompileError:
fatal error: algorithm … nicht gefunden`. Root cause is NOT torch/code: thor (like
odin) is not usr-merged — `/bin` is a real dir (not a symlink to `/usr/bin`) and
precedes `/usr/bin` on PATH, so bare `g++`→`/bin/g++` derives install prefix `/` and
looks for its C++ headers at the nonexistent `/include/c++/9`, silently dropping all
libstdc++ dirs (C compiles; only C++ headers vanish). `/usr/bin/g++` (prefix `/usr`)
compiles fine, and torch inductor honors `CXX` (verified `get_cpp_compiler()→/usr/bin/g++`
+ a real `torch.compile` inductor build).

Fix: `experiments/3d/train.py` header now auto-sets `CC=/usr/bin/gcc CXX=/usr/bin/g++`
when `/bin` is not a symlink and bare gcc/g++ resolve under `/bin/` (no-op on usr-merged
nodes; skipped if CC/CXX already set). Set before `import torch` so Triton picks it up too.
So `train.py experiment=1_medverse_benchmark medverse.compile=true` now works on thor with
no manual export. Memory: feedback-python-env updated with the thor gotcha.

## Feature-similarity run on real checkpoint (2026-07-25)

Ran experiments/3d/feature_sim/run.py on the trained PatchSet3D (arch.l=2,
2026-07-25_usual-puddle-174) over totalseg test (experiment=22_totalseg_train_test).
Fixes surfaced by the real run (node thor, RTX A6000, .venv_thor cu121):
- Device: metrics run on the features' device (GPU); retrieval/margin do large (n_fg x M)
  matmuls that are far faster there. metrics.py helper tensors made device-aware.
- Soft occupancy labels for dense mode (grid_labels threshold=None): at 16^3 a cell pools
  8^3 voxels, so thin structures never reach the old 0.5 threshold -> dense@16 was ~100%
  nan AUROC on the model's own operating resolution. Soft labels + soft_auroc/soft_dice
  match the model's soft-Dice training; nan rate ~0%. Point mode (native res) stays exact 0/1.
- wandb: logs a per-(task,tier,res) Table + mean auroc/margin/retrieval per (tier,res);
  project defaults to patch_icl_feature_similarity.
First observations (n<=4 smoke): dense@16 soft_auroc already high & discriminative
(thin aorta ~0.92-0.99); point@64 auroc ~0.90-0.99. soft_dice is a min-max-normalized
relative overlap proxy (cosine maps aren't calibrated); soft_auroc is the scale-free
separability headline.

- 2026-07-26: Added `experiments/3d/encoder_bench/` — compute/latency scaling benchmark for 3D encoders (7 zoo + Primus/SegMamba compute-only stand-ins). Sweeps encoder × input_size at best-optimized config (torch.compile/bf16/SDPA), writes CSV + log-y scaling-curve PNGs. Real full-depth architectures; sizes an encoder can't process yield honest `error:*`/skip rows. NOTE: SegMamba uses an O(L) pure-Python reference scan unless `mamba_ssm` is installed (absent on thor) — its latency at ≥64³ is not representative; install mamba_ssm for meaningful Mamba numbers. Run: `.venv_thor/bin/python experiments/3d/encoder_bench/run.py`.

- 2026-07-27: feature_sim now accepts a **pluggable frozen encoder** (`eval.model=primus`), first target the **CoLiPri** vision backbone (a stock nnUNet Primus-M). New `PrimusEncoderAdapter` (weights-pluggable; runs `down_projection`+`eva` only, skips the Primus decoder; resamples input to `input_shape`, HU-recon→CoLiPri norm=HU/1000). `run.py` gained `build_adapter(cfg)`; `real_dice`+transformer tiers are now PatchSet3D-only extras (None/skipped for a generic encoder). New `cost.py` logs frozen image-encode **FLOPs/VRAM/it-s** per encoder (adapter `cost_target` hook) → `encode_cost.csv`. `scripts/extract_colipri_backbone.py` pulls the 373 `image_encoder.backbone.*` tensors from `microsoft/colipri` `model.safetensors` (+ arch/preproc from its Hydra configs) into `results/checkpoints/primus_colipri.{pt,json}` — no `colipri` package install (hf_hub+safetensors only). Regression-verified: patchset3d path unchanged (836 rows, real_dice 836/836). Cost: Primus-M@192³ (CoLiPri native, 24³ tokens) ≈ 1993 GFLOPs / 1.27GB / 0.39 vol-s vs current conv encoder @128³ ≈ 573MB / 32 vol-s (~80× slower). Validation smoke run (8 subjects, some self-context leakage) end-to-end OK; a proper scale run is the next step. Run: `python experiments/3d/feature_sim/run.py eval.model=primus eval.primus_sidecar=results/checkpoints/primus_colipri.json 'feature_sim.tiers=[backbone]' 'feature_sim.resolutions=[16,24]'`.

- 2026-07-28: **CoLiPri eval perf.** Measured Primus-M@192³ encode batch-scaling: **flat ~3.8 vol/s from B=1→16** (VRAM 1.1→9.4GB) — compute-bound, NOT batch-starved, so batching gives zero speedup (only VRAM). Clean steady-state ~3.7 vol/s (cold cost-probe underreports 0.39–0.82). Real win instead: **native-encode cache** in PrimusEncoderAdapter (`_encode_native` keyed by storage-ptr+shape, `reset_cache()` per task in `_rows_for_task`) — the study re-encoded each volume once per resolution; now encodes to the native grid once and downsamples to all res → **2 encodes/sample instead of 4** (bit-identical output, verified). Full test split (6217 samples) ~1h now. Added a tqdm bar to feature_sim run. GPU during eval: ~100% util, ~2GB VRAM, ~298W.

- 2026-07-28: **`crop_spacing_mm` (data config).** The use_crop=True crop had a hard-coded 1.5mm output spacing (`phys_ref = T*1.5`). Made it configurable (`data.crop_spacing_mm`, default 1.5) so the organ-centered crop's FOV = image_size*crop_spacing_mm and output spacing = crop_spacing_mm. Set **2.0 to match CoLiPri's native 2mm training** (192³ → 384mm FOV, 16mm/ViT-token vs 12mm at 1.5mm — aligns rope/positional scale + FOV to CoLiPri's distribution). Threaded through TotalSegInContextDataset + common.build_dataset/make_eval_loader. First result comparison (5 subj, frozen CoLiPri backbone@24 vs trained patchset3d img_embed@48): CoLiPri competitive/ahead in aggregate (transfer 0.245 vs 0.228, wins 59% of classes), dominant on bones/skeleton (humerus/clavicula/cervical-vertebrae/skull), weaker on soft organs (spleen/kidney/pancreas/adrenals) + lung lobes; confounded by frozen-vs-trained + res/spacing/n. A/B at 2mm running to test if matching spacing lifts soft-tissue.

- 2026-07-27: **Dataloading profiling (exp 22) + optimization.** New `experiments/3d/bench_dataloading.py` drives the real train_loader (throughput vs image_size × workers × use_crop, plus `--profile` cProfile attribution). Findings: (1) pre-resized cache exists only for 64³/128³ — other image_sizes fall to a slow native-load+CPU-interpolate path (~1 it/s, 10-20× cliff); pre-generate via `scripts/convert_to_npy.py --size S S S`. (2) `use_crop=true` ~1.7× slower than false (34.6→20.1 it/s @16w) because the crop path loads native ct.npy/label.npy + float32 cast + CPU trilinear resample per item, which the pre-resized path skips. (3) Both paths were bottlenecked ~30-37% by **omegaconf `__getattr__`** in the per-item aug hot path. **Fix:** `TotalSegInContextDataset` now converts `aug_cfg` (DictConfig) → nested `SimpleNamespace` ONCE at init via `_to_ns` (plain-attr access; lists stay lists; getattr-with-default preserved) — in-process per-item CPU dropped ~2.2× (false 25.96→11.74s, crop 31.45→14.29s per 40 items), behavior-preserving (verified). Helps worker-bound regimes (false@8w +36%); `random.shuffle` (~115ms/item) is now the top cost. (4) **`random.shuffle` fix:** context sampling shuffled the ENTIRE candidate pool (~all subjects with the class, ~1000) just to take `context_size` of them — O(#subjects) RNG/item. Replaced with `_lazy_shuffle` (forward Fisher-Yates generator, O(consumed) work, still walks all candidates if context loads fail). In-process per-item CPU now **8.18s/40 (3.2× vs the original 25.96s)**, function calls 344k→61k; remaining cost is genuine `_load` I/O + aug compute. Distribution identical to shuffle-then-take; exact per-seed context picks shift once (eval determinism preserved). Multi-worker end-to-end throughput too noisy on the shared node to read cleanly — profile is the reliable signal.

- 2026-07-28: **PrimusEncoder eval encode-cache (fixes eval >> train with `arch.encoder=primus`).** With the frozen CoLiPri encoder, eval took far longer than a train epoch — not a bug, structural: the frozen 192³ ViT encode (~3.7 vol/s) is the wall-clock bottleneck (negligible for the conv encoder), and eval does more of them than training. Train epoch = `max_ds_len_train=1000` samples × 1 forward × T=2 vols = ~2000 encodes; eval (`val_classes=all` ~117 classes × ≤20 subjects ≈ ~1800 samples) runs **two** full forwards per sample — `model.predict` (hard Dice) + `train_forward` (soft-Dice/loss monitoring) — × T=2 ≈ ~7000 encodes (≈3–4× a train epoch), and with `use_crop=false` the 20 test subjects' images are re-encoded hundreds of times across classes. **Fix:** eval-only encode cache in `PrimusEncoder` (`src/models/primus_encoder.py`), active only when `frozen AND not self.training` (deterministic, no-aug, so features are reusable), keyed on a cheap per-row fingerprint (shape + rounded sum + hash of 512 strided samples), per-row miss-batched then reused; bounded (256 entries) + `reset_cache()`. Train path bypasses it entirely (aug → every volume unique; also confirmed the frozen ViT is **stochastic in train mode** — dropout on — and deterministic in eval). Verified: cache bit-exact vs eval recompute, distinct volumes don't collide, batched mixed rows map correctly, **19.4× speedup** on 20 repeats of one volume; end-to-end through `PatchSet3D.predict` reuses volumes across tasks (2 unique → no growth). Immediate CLI mitigations also available: `eval.n_subjects=4`, `train.eval_every=5`.

- 2026-07-28: **encoder_bench GFLOPs fix for conv_encoder3d.** GFLOPs column was blank for `conv_encoder3d` at every size (others only missed 256³ via OOM). Cause: `count_gflops` swallowed the exception (`profiling.py`, `except: return None`) → blank cell but `status=ok`. Real failure: fvcore counts FLOPs via `torch.jit.trace`, and in `_down_to` (`patchset3d.py`) the avg-pool kernel `k = src // R` derived from `f.shape[-1]` becomes a traced tensor under tracing → `F.avg_pool3d` rejects a tensor kernel_size. conv_encoder3d is the only encoder routed through `ConvEncoder3D._resample→_down_to`, so it alone failed everywhere. Fix: `k = int(src) // R` (runtime-identical; `int()` is a no-op on a real int) + the exception now prints instead of vanishing. Patched the CSV GFLOPs (fvcore MACs convention, matching other rows) computed **on CPU** (device-independent; GPU was busy training): 0.29/2.35/7.92/18.78/150.2 at 32/64/96/128/256³ — the cheapest encoder by far (~25× under resenc@64). Regenerated the 3 scaling PNGs.

- 2026-07-28: **Persistent CPU-backed encode cache for frozen PrimusEncoder (fixes crop-path val).** The earlier in-memory cache only helped `use_crop=false` (a subject's whole-volume resize is class-independent → ~20 unique crops). Under `use_crop=true` the crop is **organ-centered** (class-dependent) + jittered (idx-dependent), so every (subject,class,sample) crop is a distinct image → cache never hit → val re-encoded each case×label (~1800×T encodes/val, the frozen 192³ ViT dominating). Key realization: the encoder is **frozen**, so its output for a fixed eval crop is invariant *across epochs* — so the cache should persist across val calls. Rebuilt as `_EncodeCache` (LRU, **CPU-backed** so it holds the whole eval set without eating VRAM shared with training; ~14 MB/entry at R=16) + `_cached_encode(encode_fn, x, key_fn, cache)` helper (per-row miss-batching, dedupes repeats within a batch, stacks in order on x's device). `PrimusEncoder.forward` uses it only when `frozen AND not training` (train bypasses: aug makes each volume unique, and the ViT is stochastic in train mode via dropout). Net: **first val encodes each distinct crop once; every later val is a head-only pass** — directly kills "encode same case ×117 labels" across epochs. Also within a sample the `predict`+soft-monitoring double forward now share encodes. Added `eval.crop_jitter` config (default null=T//4; set **0** for centered crops so a (subject,class) crop is identical as target vs context → within-epoch reuse too); threaded into `make_eval_loader`. Floor: ~N_samples distinct encodes once (organ-centering is genuinely per-class) → keep `eval.n_subjects` modest for training val (4–6 → ~450–900 crops → cache fits RAM, first val ~2 min, rest free); full n_subjects=20 crop eval once via eval.py. TDD: 10 CPU tests (LRU eviction, miss-batch dedupe, cross-call persistence, forward frozen/train/trainable routing) — all green; real-Primus GPU smoke deferred (node busy training). `cache_max` default 4096 entries.

- 2026-07-28: **Fixed stale feature_sim metric tests.** `tests/test_feature_sim_metrics.py` and `test_feature_sim_sweep.py` referenced `soft_dice`, which was intentionally removed from `feature_sim/metrics.py` (the old min-max prototype-score Dice was replaced by `label_transfer`'s `transfer_dice`; `soft_auroc` is the dense separability headline). The tests were written against the earlier design and had been broken since first commit (metrics.py never shipped `soft_dice`), erroring at collection. Aligned them to the current contract: `prototype_cosine(mode="dense")` → `{"auroc"}`, `mode="point"` → `{"auroc","ap"}`; replaced the removed `soft_dice` test with `label_transfer` coverage (transfer_dice on separable features + nan-on-empty-target); sweep row schema now checks `rows[1]["transfer_dice"] is None` for point mode. 3d+feature_sim test group green (46). NOTE: a separate, pre-existing test-infra issue remains — `pytest tests/` can't run 2d and 3d suites in one process because both experiment dirs expose bare `train`/`evaluate`/`common` modules that collide in sys.modules (2d's `train.py` does `from evaluate import _target_like`, which resolves to 3d's evaluate once that's on sys.path). Each file passes in isolation; run per-experiment-dir. Not addressed here.

- 2026-07-28: **Relabeled feature_sim trace tag `encoder` → `transformer_input`.** The `'encoder'` tier in `PatchSet3DEncoderAdapter.transformer_trace` (→ `val/transfer_dice/encoder`, `val/retrieval/encoder`) was a misnomer: it hooks the transformer's *input* (`forward_pre_hook` on `self.transformer`), which is the img token **after** the trainable `img_embed` linear + Fourier `pos` (and ctx/qry id embeddings), not the frozen Primus/CoLiPri encoder output. That is why the curve *rose during training with a frozen encoder* — `img_embed`/`pos` are trained and reshape the cosine geometry, improving 1-NN label transfer. Renamed the tag to `transformer_input` (adapters.py) so the metric name matches what's measured; docstrings in `train.py::_feature_sim_trace` and `tau_sweep.py` updated to state explicitly it is NOT the frozen encoder output. Pure relabel — no new tier, no added compute (the trace still does one hooked forward). A separate true-frozen-encoder (dim 864) tier was scoped but deferred: measured +22 ms/task (~+22 s per feature_sim call at n_tasks=1000, vs ~43 s current) because label_transfer's cosine matmul scales with feature dim (864 vs e=256); since the backbone is frozen and the val subsample deterministic that value is constant across epochs and would be cache-once if added later.

- 2026-07-28: **Optional MLP `img_embed` for patchset3d (default off).** Coherence check of `model=patchset3d` + `encoder=primus`: `img_embed = nn.Linear(encoder.out_ch, e)` compresses the frozen CoLiPri Primus output (`embed_dim=864`) to `e=256` — a 3.4× rank bottleneck, whereas for the conv encoder (`out_ch=sum(enc_dims)=128`) the same `e=256` is a 2× expansion. `patchset3d_colipri.yaml` had inherited `e=256`/`h=512` from the conv default unchanged. Added `arch.img_embed_mlp` (bool, default **false** = unchanged single Linear); true builds `Linear(oc,oc)→GELU→Linear(oc,e)` to keep the full encoder width through a nonlinearity before compressing. Wired: `PatchSet3D.__init__(img_embed_mlp=False)` (patchset3d.py), `build_model` arch dict (train.py), and `img_embed_mlp: false` in both `model/patchset3d.yaml` + `model/patchset3d_colipri.yaml`. Verified via `bench_train_step.py` (new, this session) at l=2/res=24/192³/use_crop=true: false→2.1M trainable, 234 ms/step, 2.64 GB; true→2.8M trainable (+0.75M: the extra 864×864 layer), 232 ms/step, 2.74 GB — i.e. the MLP is compute/mem-negligible here (step stays COMPUTE-bound on the 24³ full-attn, not img_embed). Other coherence findings (not code changes): `resolution=24`=192/8 maps 1:1 to Primus's 24³ tokens (`_down_to` identity — correctly set); `grid_size=24×8=192`=image_size (decode reconstructs native); `crop_spacing_mm=1.5` vs Primus-trained 2mm is a *deliberate* A/B-backed choice (documented in colipri yaml), not a bug; `enc_dims` dead under primus; base `patchset3d.yaml`'s `query_self_attn=true` is a no-op when `full_attn=true`.

- 2026-07-28: **feature_sim: `PatchSet3DEncoderAdapter` now supports primus-encoder checkpoints.** The adapter was conv-only — `tiers()`/`_stage_feats`/`features`/`sample_features`/`cost_target` all index `enc.stem`/`enc.stages`, which `PrimusEncoder` lacks (it emits a single native token grid), so any `stage:*`/`concat`/`img_embed` tier on a primus-encoder checkpoint (e.g. `2026-07-28_warm-cloud-192`: encoder=primus, frozen, R=24, l=2) raised AttributeError. Added `self._is_primus = not hasattr(self.enc, "stages")` and a primus branch to each method: tiers `[backbone, img_embed]` (transformer tiers unchanged — they hook the full forward). `backbone` = the raw frozen encoder map via `self.enc(volumes)` (PrimusEncoder self-preprocesses w/ sidecar HU renorm + has its own frozen-eval cache); `img_embed` = the trainable 864→e=256 projection of it (isolated, no per-task norm/pos — matches the conv img_embed tier's intent). Extracted the img_embed projection into `_apply_img_embed` (conv path refactored to use it — behaviour-identical). `native_res(backbone|img_embed)=R` (encoder emits R³ regardless of input_res). Verified on warm-cloud-192: backbone dim 864 / img_embed dim 256 at res∈{16,24}, point-mode sample_features, cost_target, and transformer_trace (tags `transformer_input,L0,L1`) all run; **`backbone` matches the standalone `PrimusEncoderAdapter` at cosine 0.99999** (rel-L2 0.0036, pure bf16-vs-fp32-autocast) — so a primus-patchset run and the standalone frozen-primus run share the same encoder-feature reference, enabling a clean `backbone → img_embed → tf:L0 → tf:L1` correspondence trace **with real_dice coupling** in one run. Conv adapter tests unchanged (8/8). Run: `python experiments/3d/feature_sim/run.py eval.model=patchset3d eval.checkpoint=.../warm-cloud-192/best.pt 'feature_sim.tiers=[backbone,img_embed,transformer_layers]' 'feature_sim.resolutions=[16,24]'`. NB warm-cloud-192 is barely trained (best_val_dice≈0.022 @ epoch15) — treat gains as noise until a better primus checkpoint exists.

- 2026-07-28: **Added `experiment=30_colipri_encoder`.** Bakes the frozen-CoLiPri-Primus PatchSet3D config into a single experiment file (`configs/experiment/3d/experiment/30_colipri_encoder.yaml`) — equivalent to `experiment=22_totalseg_train_test model=patchset3d arch.encoder=primus arch.primus_sidecar=results/checkpoints/primus_colipri.json arch.l=2 arch.e=512 arch.resolution=24 arch.img_embed_mlp=true data.image_size=[192,192,192] data.use_crop=true`. Mirrors exp22's data/optim recipe (train_classes=benchmark, val_classes=all, p_synth=0, class_balanced=false, epochs=1000, bs=1, adamw lr=1e-4, cosine, bce_dice, eval split=test n=20) but adds `override /model: patchset3d` to the defaults and the arch block. **Added `use_crop=true` beyond the literal param list** (not optional here: at 192³ there's no pre-resized cache, so use_crop=false falls to the slow native-nii resize path ~0.8 it/s → stall; the crop path reads native npy and is compute-bound). `muon=true` inherited from the patchset3d model group; augmentations=nnunet inherited from exp22's override. Verified by hydra compose: model=patchset3d, encoder=primus/frozen, resolution=24/e=512/l=2/img_embed_mlp=true, decode grid 24·8=192=image_size, use_crop=true, groups {model:patchset3d, dataset:totalseg, augmentations:nnunet}. Arch already proven to build+step earlier this session (6.2M trainable, ~280 ms/step, 3.9 GB @ B=1).

- 2026-07-28: **Fixed `item['spacing']` metadata bug under `use_crop=true`.** `_load_spacings` hardcoded the crop-path effective spacing to `1.5mm/voxel`, so `item['spacing']` (from `_get_spacing`) always reported 1.5 regardless of `crop_spacing_mm`. With `crop_spacing_mm=2` the crop grabs `T*2mm=384mm` (`round(384/1.5)=256` native voxels) and resamples to 192³ → the returned tensor is genuinely **2.0mm/voxel**, but the reported spacing said 1.5 — a latent metadata lie (harmless today: no consumer in the 3D train/eval path reads `item['spacing']`, verified by grep). Root cause is that `_get_spacing` served double duty: (a) the **native** spacing used *inside* `_load_crop`/`_load_crop_multi` to size the crop (`crop_sizes=round(T*crop_spacing_mm/native)`, must stay 1.5 = true ct.npy/label.npy spacing), and (b) the **reported** output spacing in the item dict (should be `crop_spacing_mm`). These coincide only at `crop_spacing_mm=1.5`, which is why it went unnoticed. Fix decouples them: added `_reported_spacing(subj)` (returns `crop_spacing_mm` iso under use_crop, else the resized effective `_get_spacing`) and pointed both `item['spacing']` sites at it; `_load_crop*` still use `_get_spacing` (native, untouched → crop-size math unchanged, 256 voxels at csm=2). Verified: dataset with csm=2.0 now returns `spacing=[2,2,2]` with native `_get_spacing=[1.5,1.5,1.5]` (crop still 256→192³); csm=1.5 unchanged (`[1.5,1.5,1.5]`) so no regression on the exp30 default. Aside (measured this session): csm 1.5→2.0 raises mean test-set scan-volume coverage 49.5%→77.8% (median 41%→83%, fully-covered 11→30/89) and matches CoLiPri/Primus's 2mm pretraining spacing.

- 2026-07-28: **nb 30 (`30_colipri_encoder.py`): added train-vs-held-out generalisation analysis via `in_train`.** The samples table carries a bool `in_train` (class seen during training). New cell **1b** splits eval into seen (47 cls, macro 0.584) vs held-out (68 cls, macro 0.297) — RAW gap **+0.287**, but it is a **composition confound**, not a generalisation gap: held-out classes are systematically smaller (median tgt_size 4883 vs 13016) and the held-out set is 32% ribs (a known instance-ambiguity blind spot). Two controls exposed this: (a) within-shape-family split (gap flips sign in several families — mid_tube −0.14, thick_sheet −0.03), and — cleanest — (c) **matched lateral-mirror pairs** (same organ, trained side vs held-out side; n=16): mean trained 0.557 vs held-out **0.594**, delta **−0.037**, held-out side even *better* in 7/16 pairs (humerus −0.30, lung_upper_lobe −0.28). Conclusion: when morphology is matched the model generalises to unseen classes at parity; the headline gap is anatomy-mix, not seen-vs-unseen. Also enhanced cell 1 (per-class bars now hatch held-out classes + `*` label marker) and added the `in_train` column to its table. Figure: (a) per-family train/held-out bars + (c) mirror-pair scatter vs parity diagonal. Excluding ribs barely moves the raw gap (0.287→0.284), so ribs aren't the driver — the small thin-vessel held-out tail is. Verified headless via `app.run()`.

- 2026-07-28: **Investigated TotalSegmentator subtask GT masks + added nnU-Net→npy converter.** The local `totalseg` root only holds the `total` task (117-class `label.npy`, 1228 subj); the additional subtask masks (hip_implant, pleural/pericardial effusion, liver_lesions, lung_nodules, kidney_cysts, teeth, ...) live in **separate self-contained nnU-Net raw datasets** on Zenodo ("Training dataset for task X", all CC-BY-4.0, imagesTr+labelsTr) and GitHub `-weights` release assets (the ~230 MB assets are **model weights, not masks** — a common trap). Verified via HTTP-range read of the hip_implant zip central directory (no full download): 71 subjects named with **re-anonymised random hashes** (`imagesTr/DKzjB1Tzcsy3lf9f_0000.nii.gz`), disjoint from our `s0xxx` IDs — so these are new data sources, not extra masks on existing volumes. Subject counts (from release filenames): hip_implant 71, kidney_cyst 501, liver_lesions(CT) 842, lung_nodules ~1353 (partly LIDC-IDRI), breasts 1559, mediastinum 1786. Genuinely-new targets (no existing 117-class equivalent): hip_implant, effusion, liver_lesions, lung_nodules; kidney_cysts/liver_segments overlap existing classes. **Code:** appended `hip_implant` to `ALL_CLASSES` (index 122, appended at end so existing indices stay stable) and added `scripts/convert_nnunet_task.py` — converts an extracted nnU-Net subtask dataset (imagesTr/labelsTr/dataset.json) into the per-subject `ct.npy`/`label.npy`(+sized)/`meta.csv`/`spacings.json` layout, reusing `_iso_resize`+`_normalise_ct` from convert_to_npy and encoding each dataset.json label under its ALL_CLASSES index (so `TotalSegInContextDataset._load` resolves it with zero dataloader changes; `--map src=our_class` for name mismatches). All subjects written `split=test` (eval-only) by default. Verified: hip_implant resolves to idx 122, script `--help` runs. Not yet run on real data (needs the 9 GB Zenodo download + unzip).

- 2026-08-01: **use_crop dataloader: resample the real slice, pad the T³ output (fixes the crop_spacing_mm=4 slowdown).** Traced why exp30 (`experiment=30_colipri_encoder`) epoch time rose ~300s→450s when going from `crop_spacing_mm=2` to `crop_spacing_mm=4 mask_downsample=occupancy mask_occupancy_thr=0.3`. Root cause was `crop_spacing_mm`, not the mask mode: in `_organ_crop_arrays` the crop's fixed physical extent is `target=round(T*crop_spacing_mm/native_sp)` voxels (native TotalSeg is 1.5mm iso), so csm 2→4 doubles it per axis (256→512) → the array handed to `F.interpolate` grows **8×** (256³→512³, ~17M→134M voxels), and since native axes are only ~250–440 the extra bulk is mostly `np.pad` air. Model input stays 192³ so **GPU step time is unchanged** — the epoch just flipped from GPU-bound to dataloader-bound (measured: 608→2197 ms/crop for csm alone, occupancy adds only ~+24%; 2 crops/item at K=1). Fix: `_organ_crop_arrays` no longer pads the native slice — it returns the unpadded mmap slice plus `out_sizes` (= `crop_sizes*T/target`, the object's extent in the T³ grid) and `pad_lo` (centre offset). New `_place_image`/`_place_label` resample only the real slice to `out_sizes` then centre it in an air-/0-filled T³ tensor (or return it directly when `out_sizes==T`, no-pad fast path). Geometrically equivalent to sub-voxel level (both centre the object at T/2; only integer-size rounding + resample-grid differ). **Verified** against the old path on 8 real subjects: label Dice(old,new) 0.99 @csm2 / 0.96 @csm4, CT max |Δ|≤3.4 HU; per-crop speedup **1.7× @csm2, 12.3× @csm4** (1987→161 ms). End-to-end `__getitem__` smoke (liver/spleen, K=1) returns correct (1,192³)/(192³) tensors at ~190–230 ms/item. Touches only the real-class crop path (`_load_crop`/`_load_crop_multi`); synth crop path (`_get_synth_item`, crops exactly T³ at native res, no resample) and the non-crop fast path are unchanged.

- 2026-08-02: **PrimusEncoder early-exit: `arch.encoder_stage` truncates the frozen CoLiPri ViT to the first N of 16 EVA blocks.** Motivated by a latency trace showing CoLiPri (Primus-M) is an **isotropic** ViT — patch-embed (8³, negligible) → 16 identical EVA blocks over a fixed 13,824-token seq (24³, D=864, SwiGLU mlp 8/3), no hierarchical downsampling — so encoder cost is **linear in depth**. Measured on odin (RTX PRO 6000, fp32 eager) truncating `eva.blocks[:k]`: stage 4/8/12 = 25/50/75% of full latency (batch=1: 106/212/317 ms vs 422 ms full; batch=2 i.e. target+ctx: 213/424/635 vs 846 ms), stage 1 = 6.4% ≈ 1/16 confirming ~zero fixed overhead → **saving ≈ (16−k)/16**. Per-block MACs ≈0.43 T (attention ∝L² is 76%, MLP 19%) → ~13.9 TFLOP/full encode. **Impl:** `PrimusEncoder(..., encoder_stage=N)` loads full weights (so `load_state_dict` matches) then `_truncate_blocks` reassigns `eva.blocks` to a new `ModuleList` of the first N — dropping the tail so it's **never computed and its VRAM is freed** (Eva.forward_features just iterates `self.blocks` then the final `norm`, which still runs on the stage-k output = standard normed layer-k hidden state). null/≤0/≥depth = full encoder. Wired through `PatchSet3D(encoder_stage=)` and `train.py` arch dict; config `arch.encoder_stage: null` in `model/patchset3d.yaml`. Verified: stage 8→8 blocks, 4→4, 20→clamped 16, out shape (1,864,24³) unchanged; `arch.encoder_stage=8` hydra override resolves on exp30. Purpose: sweep encoder depth vs Dice (mid-stage tap 8–12 the natural compute/accuracy trade); early blocks texture/local, late semantic. NB reducing **tokens** (larger patch / lower `resolution`) attacks the 76% attention term super-linearly, vs truncation's linear gain.

- 2026-08-02: **exp30 perf: compile the frozen Primus encoder + restrict LAWA to trainable params.** Two fixes after auditing `experiments/3d/train.py`. **(a) Compile the frozen Primus eva stack.** The compile block only wrapped `net.transformer` (the l=2 read-out), leaving the frozen 145M-param / 16-block Primus ViT — the dominant per-step cost — running eager (the "encoder stays eager" comment only ever applied to the *conv* encoder's graph-break). Extended the `is_patchset and arch.compile` block to also `torch.compile(net.encoder.primus.eva, dynamic=True)` when `encoder=primus`: the eva stack is pure attention/MLP (compiles cleanly), and the interpolate/`_down_to` that would graph-break live outside it in `_preprocess`/`_encode_batch`. Measured (odin, bf16): **~1.6× on the encoder** (B=2 target+ctx 340→208 ms, saving ~132 ms/step); stacks with `arch.encoder_stage`. `dynamic=True` so target/context batch-size differences don't retrigger compilation; no-op for `encoder=conv`. **(b) LAWA averages only trainable weights.** `lawa_queue.append({... net.state_dict() ...})` snapshotted the FULL state_dict every epoch — including the ~145M-param / ~580 MB frozen encoder that never changes — kept `lawa_k=10` copies (~5.8 GB CPU RAM) and averaged/reloaded them into themselves. Now snapshot a trainable-key set once (post-compile so `_orig_mod.` prefixes match state_dict keys) and push/restore only those. Made shared `pfn_train.lawa_average` subset-aware (average/save only the queued keys, `load_state_dict(..., strict=False)`) — backward-compatible with the 2D trainers that queue full state_dicts (all keys present → identical). 3D restore now `strict=False`. **Verified** (stage=4 smoke): eva compiles + bf16 forward returns (1,1,192³); LAWA snapshot 37 trainable keys / 6.2M elems vs 43.4M full state_dict, frozen encoder weight fingerprint unchanged across average+restore. Config knob unchanged (`arch.compile: true` already on for exp30). Secondary bloat NOT fixed: best.pt still saves the frozen encoder (harmless; matches sidecar).

- 2026-08-02: **exp30 eval ~8x faster: single autocast forward in `evaluate_classes` (patchset3d val).** The per-epoch val ran two full forwards per batch — `model.predict` (hard Dice) AND `logits_fn=train_forward` (soft-Dice/loss) — but for patchset3d these are the *same* native forward (`predict` = `sigmoid(train_forward)≥0.5`), and both ran in **fp32** (no autocast), which also forced torch.compile to recompile the eva/transformer between the train (bf16) and eval (fp32) dtypes. Added two **opt-in** params to `evaluate_classes`: `reuse_logits` (derive the hard prediction from the same logits used for the soft metrics — one forward, no separate predict pass) and `autocast` (run the eval forward under bf16, matching training). `validate_mean` sets both **only for `model==patchset3d`** (`fast_eval`); medverse keeps the current fp32 double-forward path byte-identical, and `eval.py` (defaults off, `_eval_autocast(False)`=nullcontext) is unchanged. **Verified** on exp30 val: reuse@fp32 vs predict = **EXACT** (0/340M voxels); reuse@bf16 vs predict flips 0.19% of boundary voxels but **mean |ΔDice| 0.00000 / max 0.00001** (metric unchanged); warm per-batch **2283→291 ms (7.8x)**, ~336→43 s per eval. Encoder cache still amortizes epoch-2+ (head-only). Not touched (Fix 1, deferred): `eval.n_subjects=20`×117 classes = 1173 samples/epoch — reduce for cheaper routine monitoring.

- 2026-08-02: **train.py: opt-in per-step timing breakdown (data / image-encode / attention).** New `train.profile_timing` (default false) logs `train/time/{data,encode,attn}_ms` per epoch (+ a `tqdm.write` line). data-wait = perf_counter between steps (dataloader stall); encode/attn = CUDA events bracketing `net.encoder` and `net.transformer`, each called exactly once per patchset3d forward (hooks on the outer modules, so compatible with the compiled eva/transformer). Nearly free — the loop already syncs each step via `loss.item()`, so `elapsed_time` reads add no extra stall; OFF by default = zero overhead (no hooks, phase timers guarded by `prof`). patchset3d + CUDA only. Verified: 12-step profiled epoch fires all hooks, returns positive timings, prints the breakdown. NB epoch 0 numbers include torch.compile warmup (first forward compiles) — read steady-state from epoch 1+. Usage: `python experiments/3d/train.py experiment=30_colipri_encoder train.profile_timing=true`.

- 2026-08-04: **Converted the extra TotalSegmentator `more_labels` masks to .npy + built a lossless global index.** The 37 produced multilabel tasks (`totalseg_test_more_labels/s*/segmentations/{task}.nii.gz`, 25 test subjects) overlap heavily — ~85% of fg voxels are covered by 2+ tasks (max depth 5), and there are 362 (task, local_id) pairs / 329 unique names, exceeding uint8. So flattening into one `label.npy`-style volume is impossible losslessly. New `experiments/totalseg_more_labels/convert_more_labels.py` keeps each task its own array (Approach A): per subject writes `more_labels/{task}.npy` (uint8, native, canonical) + `more_labels/{task}_DxHxW.npy` (nearest iso-resize via the **same** `_iso_resize` as convert_to_npy → aligns with `ct_DxHxW.npy`/`label_DxHxW.npy`). At the data root: `more_labels_classes.json` (global index, contiguous global_id 1..362 ↔ task/local_id/name, all classes incl. cross-task duplicates like vertebrae_pp vs _refined — no curation) and `more_labels_subject_classes.json` ({subject: [global_id present with >0 voxels]}, so eval never picks a class a subject lacks). Excludes the 4 `*_auxiliary` label-merge pseudo-tasks + the 2 missing-model tasks (total_highres_test, covid). Parallel per-subject (mp.Pool, forkserver). **Verified:** native .npy byte-exact vs source .nii.gz (lossless); sized on the 64³ grid matching main `label_64x64x64.npy`; index contiguous; 25 subjects, 38/102/200 present-classes (min/med/max), 1260 .npy files (630 masks × native+64³), 0 errors, 2.2 min. NOT yet wired into `TotalSegInContextDataset` — that's the next step (bridge the dataloader to read these extra labels + index for eval).

- 2026-08-06: **feature_sim now threads per-crop physical spacing through every encode (spacing-aware checkpoints, exp 36).** `experiment=36_colipri_spacing_aware_128` trains a spacing-aware PatchSet3D (`arch.encoder_spacing_aware=true` ⇒ native_grid; the frozen CoLiPri ViT scales its RoPE by the batch's mm/voxel via `forward(..., spacing=)` → `PrimusEncoder._encode`). The real eval path (`evaluate.py`) already threads `batch["spacing"]` into `model.predict`, but the whole feature_sim study did **not**: `run.py` called `model.predict(image,cin,cout)` (real_dice) and every `adapter.features`/`transformer_query`/`transformer_pair_per_layer` / `train.py::_feature_sim_trace`'s `transformer_trace` with no spacing, so the encoder silently fell back to `train_spacing_mm` (2 mm) regardless of the crop. Harmless *only* at the exp-36 default (`crop_spacing_mm=2` == train pitch, so the fallback coincides), but the moment any other spacing is probed — the point of a spacing-aware model — features + real_dice decouple from the crop (wrong RoPE positions). **Fix:** added an optional `spacing=None` kwarg to `PatchSet3DEncoderAdapter.{features,sample_features,transformer_query,transformer_pair,transformer_pair_per_layer,transformer_trace}`, threaded into `self.enc(volumes, spacing=)` / `self.model(..., spacing=)` (model/encoder ignore it when not spacing-aware, so conv + non-spacing primus are byte-identical). Callers compute the per-task/batch spacing (`float(item["spacing"][0,0])` gated on `getattr(model,"spacing_aware",False)`) and splat it as `**sp` (empty dict when not spacing-aware, so the generic `PrimusEncoderAdapter`/medverse signatures never receive the kwarg): `run.py::_rows_for_task` (predict + all 6 feature calls), `train.py::_feature_sim_trace`, and the two one-off scripts `probe_transformer.py` / `tau_sweep.py`. Other exp-36 knobs verified already-correct: `compile` is irrelevant (feature_sim's `_load_patchset` loads the checkpoint un-compiled), and `encoder_stage`/`encoder_native_grid` are rebuilt from `ckpt["arch"]`. All five edited files `py_compile`-clean; no behavioural change at `crop_spacing_mm=2`, correct now at any spacing.

- 2026-08-06: **feature_sim forward path: opt-in bf16 autocast (default on) + torch.compile (default off); ~7x headroom measured.** feature_sim ran the model in **fp32 with no autocast** (unlike train/eval, which use bf16), and never compiled — leaving a large speedup on the frozen CoLiPri ViT untouched. Benchmarked the two compile-relevant forwards per task (`predict` + `transformer_pair_per_layer`, the `eva` stack + read-out `transformer` train.py compiles) on the real exp-36 checkpoint (`crimson-deluge-224`, spacing-aware, eva truncated 8/16, R=16, L=2), RTX 6000 Ada, encoder cache reset per iter, median of 10 (`experiments/3d/feature_sim/bench_compile.py`, new): **eager fp32 260 ms/task → compiled fp32 117 (2.23x); eager bf16 92 → compiled bf16 36 (2.54x); bf16-vs-fp32 eager 2.83x; compiled-bf16 vs current-eager-fp32 7.17x.** Compile+warmup one-time ~15 s (bf16) / ~25 s (fp32); corrected break-even ~260 / ~175 tasks (a real run does thousands, so trivially amortized). **Wiring** (`run.py`): `_fwd_ctx(cfg)` = `torch.autocast("cuda", bf16)` when `eval.autocast` (default **true**) else nullcontext, wrapping only the per-task `_rows_for_task` forwards; `_maybe_compile(adapter, model, cfg)` compiles `model.transformer` + the frozen `primus.eva` (dynamic=True) when `eval.compile` (default **false**), called AFTER `measure_encode_cost` so fvcore doesn't trace a compiled graph. **Metrics stay fp32**: `_metric_row` casts the (bf16) features up with `.float()` and runs the cosine matmuls / argmax under `torch.autocast(enabled=False)` — the similarity *ranking* must stay precise even though the *encode* is bf16. Config: `eval.autocast: true`, `eval.compile: false` in `feature_sim.yaml`. Verified end-to-end on liver/spleen (2 subj): both paths write 16 rows with finite metrics; compiled vs autocast-only differ ≤1e-3 (`retrieval_at1` identical, `real_dice` 0.94822 vs 0.94819) — expected compile numerics, negligible for the study. NB default autocast=true slightly perturbs vs the old fp32 feature_sim numbers (consistent with train/eval already being bf16); flip `eval.autocast=false` to reproduce the exact old path. Pre-existing unrelated bug surfaced: `count_gflops` fails on the spacing-aware primus encoder (`round(float(tensor))` in the cache-key under fvcore tracing) → `encode_gflops=None` (gracefully handled; not fixed here).

- 2026-08-06: **feature_sim: pretrained-encoder (eval.model=primus) path now honours eval.autocast + actionable tier check.** Two follow-ups to the autocast/compile wiring. **(a) eval.autocast now controls the generic `PrimusEncoderAdapter`.** Its `_autocast()` **hardcoded bf16** and ignored the flag, so a frozen-CoLiPri/Primus eval always ran bf16 — `eval.autocast=false` (wanting an exact fp32 reference) was silently a no-op. This adapter self-autocasts internally (its encode runs outside `_fwd_ctx`'s scope — used by the cost probe + standalone), so the fix lives in the adapter: added `PrimusEncoderAdapter(autocast=True)`, `_autocast()` returns `contextlib.nullcontext()` when off, and `build_adapter` passes `autocast=cfg.eval.get("autocast", True)`. Metrics stay fp32 either way (`_encode_native` `.float()`s the encode; `_metric_row` re-disables autocast). Verified on liver/spleen (frozen primus_colipri sidecar): bf16 2.8 s/batch vs fp32 6.84 s/batch (**2.4x**, confirming the flag really switches dtype), auroc 0.91124 vs 0.91083 (~4e-4, bf16-vs-fp32 encode), retrieval_at1 identical, real_dice empty (no segmenter). **(b) Actionable tier validation.** The conv-oriented config default `feature_sim.tiers=[stage:0,stage:1,stage:2,concat,img_embed,transformer_layers]` crashed a generic-encoder run mid-loop with a cryptic assert deep in `features()` (Primus only exposes `backbone`). `main` now checks planned tiers against `adapter.tiers() ∪ {transformer_q,transformer_layers}` and raises `ValueError: eval.model=primus encoder does not support tiers [...]; supported: ['backbone']. Set 'feature_sim.tiers=[backbone]'` before any work. No regression: conv/patchset3d default tiers are all in the supported set. Note the illustrative command `experiment=22_totalseg_train_test eval.model=primus eval.primus_sidecar=...` still needs `feature_sim.tiers=[backbone]` — now it says so instead of asserting.

- 2026-08-06: **feature_sim encode-cost probe fixed for the spacing-aware/native-grid primus encoder (`PatchSet3DEncoderAdapter.cost_target`).** Two bugs, both from routing the cost probe through the cached `enc(x)` forward. **(1) encode_gflops was always None** (`[count_gflops] FLOP analysis failed: TypeError: type Tensor doesn't define __round__`): fvcore uses `jit.trace`, under which `.shape`-derived python arithmetic becomes a traced Tensor — and `PrimusEncoder._preprocess`'s native_grid path calls `_native_target_shape` = `round(shape/patch)` → `round(Tensor)`. (Hit only on native_grid; the generic `PrimusEncoderAdapter` skips it via a fixed `input_shape`, which is why *its* FLOPs already worked.) **(2) encode_it_s was inflated** (~1059 it/s at 128³, implausible): the timing loop reuses one zeros input, so the frozen-eval cache (`_key`) hit after the first call → it timed cache lookups, not the encode. **Fix:** trace `adapter.enc._encode` (down_projection + eva, the real compute — matching the generic adapter's convention) on an **eagerly preprocessed** input, so the untraceable `round()`/resample runs once with concrete shapes outside the trace and the cache/`_key` is never touched. Verified on exp 36 (`crimson-deluge-224`, spacing-aware, eva 8/16, 128³→16³ grid): `encode_gflops` now **296.17** (was None), `encode_it_s` **19.9** (real ~50 ms/encode, was ~1059 cache-hit); warning gone. Conv path (`_stage_feats`) and the generic `PrimusEncoderAdapter.cost_target` unchanged (no regression). NB the FLOP count is `_encode` only (excludes the negligible final `_down_to(resolution)` avg-pool), consistent across both primus adapters.

- 2026-08-06: **feature_sim: per-eva-block depth sweep for frozen encoders (`feature_sim.tiers=[backbone_layers]`, `eval.model=primus`).** The generic `PrimusEncoderAdapter` only exposed a single `backbone` tier (final eva output), so a frozen Primus/CoLiPri run couldn't trace target↔context correspondence *along transformer depth* the way the conv path sweeps `stage:*` — the ViT analogue was missing. Added a **fan-out meta-tier** `backbone_layers` (mirroring the patchset3d `transformer_layers` design): one forward captures every eva block's token grid and `run.py::_rows_for_task` emits a row per block (tier `bb:L{i}`), for both dense and point modes, with the same prototype/retrieval/margin metrics. **Impl** (`adapters.py`): `PrimusEncoderAdapter.tiers()` now `["backbone","backbone_layers"]`; `n_layers` = eva depth; `_encode_layers(x)` **reimplements** `Eva.forward_features`' block loop (down_projection → register-token cat → `eva._pos_embed` → iterate `eva.blocks`, capturing each post-block grid) rather than forward-hooking, so it's robust when eva is `torch.compile`-wrapped (unwraps via `_orig_mod` — hooks on a compiled graph's submodules don't fire reliably); grids are post-block **pre** final-norm (standard intermediate features, so the last grid differs slightly from `backbone` which includes the norm). `_encode_native_layers` caches the per-layer list by (storage ptr, shape) — one encode per volume across the resolution sweep; `features_per_layer`/`sample_features_per_layer` `_down_to`/grid_sample each layer. **Verified** on exp-36 sidecar (`eval.model=primus`, 192³→24³ grid, full 16 blocks): 180 samples × 16 layers = 2880 rows, clean depth curve auroc 0.870→peak **0.953 @L8-9**→0.930, retrieval_at1 0.321→**0.580 @L8**→0.474 (mid-stack tap is best — the layer-selection signal this study exists for). `eval.compile=true` runs without error (backbone_layers bypasses the compiled eva via `_orig_mod`, so compile gives no speedup *for this tier* — only a warmup cost; the `backbone` tier still uses compiled eva). Config example added to `feature_sim.yaml`. NB the smoke run used `n_subjects=2` → self-context leakage warnings (few candidates), metrics inflated but the mechanism is exercised. Not added to the patchset3d-checkpoint primus path (`PatchSet3DEncoderAdapter`, `_is_primus`) — that adapter's `transformer_layers` already traces read-out depth; add `backbone_layers` there too if the frozen-encoder depth is wanted alongside a trained checkpoint.

- 2026-08-06: **tap-ct-b-3d (`experiments/encoders/tap_ct.py`) OOM on full-volume forward — root-caused to xformers-less O(L²) attention; fixed with SDPA.** The `fomofo/tap-ct-b-3d` ViT blocks use `MemEffAttention`, which needs **xformers** for O(L) memory; when xformers is absent (not installed in `.venv_thor`) it **silently falls back to explicit `q@k.T` softmax (O(L²))**. The processor always resizes in-plane to 224×224 and pads depth to a multiple of 4 (patch (4,8,8)), so token count is driven purely by depth: `N=(D_pad/4)·784+5`. Raw (179,192,294) → D_pad=180 → **35 285 tokens**, whose per-layer attention matrix alone is ~60 GB fp32 → OOM on the 48 GB RTX 6000 Ada (loki). **Benchmark** (`tap_ct_bench.py`, depth sweep, 3 configs): baseline O(L²) tops out at **D=48 / 9 413 tok (9 GB)** then OOMs by D=96. Swapping in PyTorch `F.scaled_dot_product_attention` (flash kernel, no xformers) makes memory **linear**: D=180 = 1.6 GB / 3.07 s, D=512 (100 k tok) = 3.85 GB — the OOM is gone entirely. **bf16 autocast** is the real latency lever: D=180 drops **3.07 s → 0.38 s (~8×)** at ≤1.3 GB. **torch.compile** (`tap_ct_compile.py`, max-autotune) gives negligible speed (**1.03×**, matmul-bound, already cuBLAS-optimal) but ~3.5× lower peak (1.33 → 0.38 GB); the Triton autotuner logs benign "out of resource: shared memory" warnings and falls back to cuBLAS — not worth the compile cost since memory is no longer the constraint. **Fix applied to `tap_ct.py`:** monkeypatch the attention module's `forward` to SDPA right after load (q not pre-scaled — SDPA applies the head-dim scale internally), so a whole-volume forward stays O(L). Note the sliding-window inferer path (roi [12,224,224]) never OOMs regardless of xformers; the SDPA patch is what makes the single full-volume `model.forward(x)` usable. Alternative fix if desired: `uv pip install xformers` (the model's native fast path).

- 2026-08-06: **tap-ct profiling script (`experiments/encoders/tap_ct_profile.py`) — time/VRAM/FLOPs for full-volume vs sliding-window, `--precision`/`--compile` params.** FLOPs via torch `FlopCounterMode` (sees SDPA/flash; counted once on eager since FLOPs are precision/compile-independent), time+VRAM under the requested config. All at D_pad=180 (35 285 tok) on RTX 6000 Ada (loki, `.venv_thor`, SDPA patch on). **FLOPs: full volume 51.9 T vs sliding (roi 12×224×224, ovl 0.75) 34.6 T** — the sliding window is *cheaper* despite ~4× token overlap, because full-volume attention is O(L²) at 35k tokens while windowing caps attention at 2 357-tok blocks; both run ~17 TFLOP/s in fp32. **Time / peak VRAM:** full — fp32 3009 ms/1.70 GB, bf16 361 ms/1.44 GB (**8.3×**), fp16 384 ms, bf16+compile 350 ms/1.33 GB. sliding — fp32 1987 ms/0.70 GB, bf16 435 ms/0.85 GB, fp16 353 ms, bf16+compile 306 ms/0.68 GB. Notes: in **fp32** the quadratic-attention volume forward is compute-bound so sliding is faster (1987<3009); in **bf16** per-window inferer overhead (57 windows) dominates the FLOP saving so full ≤ sliding (361<435) — `--compile` recovers most of that (sliding 435→306) by cutting launch overhead, but barely moves the full forward (361→350). bf16 autocast peak > fp32 for sliding (0.85>0.70, autocast keeps fp32 copies). Bottom line: memory is a non-issue post-SDPA (<2 GB either path); bf16 is the ~8× lever; compile only worth it for the many-small-window sliding path.

## 2026-08-06 — Bridge TotalSegInContext dataloader → tap-ct-b-3d features

`experiments/encoders/tapct_features.py`: converts a `TotalSegInContextDataset`
image tensor into TAP-CT `pixel_values`. The two pipelines are incompatible as-is;
four reconciliations:

1. **Intensity (mandatory).** Dataloader image is already z-scored
   `clip([-1007,1573]) -> (x+167.3)/505.8`; TAP expects raw HU. Invert to HU
   (`x*505.8 - 167.3`) before the processor, else TAP re-normalizes on top and every
   voxel collapses to ~0.27. Effectively lossless (low clip matches; >822 HU clipped
   by TAP anyway).
2. **Padding.** Both dataloader paths pad in *normalized* space → padded voxels
   invert to ~-167.3 HU (soft tissue), not air. With use_crop + T%8==0 the crop
   usually fills T³ (no pad); optional `pad_hu=-1024` resets any fill region.
3. **Orientation.** Dataloader is canonical RAS (axis0=L-R, axis2=S-I); TAP trained
   LPS with axial (S) axis first. `ras_to_lps_axial_first = flip(transpose(2,1,0),
   axis=(1,2))` so TAP's in-plane == the real axial plane.
4. **Resolution.** Stock processor upsamples in-plane to 224². Use
   `TAPCTProcessor(resize_dims=(T,T))` to keep a T³ cube native (T%8==0; pair with
   use_crop).

Verified: (1,64,64,64) dataloader cube → pixel_values (1,1,64,64,64), TAP-normalized
[-2.86,2.82], last_hidden_state (1,1024,768) + pooler (1,768). tokens=(T/4)(T/8)²+5.

## 2026-08-06 — tap-ct transfer-Dice sanity (liver, before wiring to feature_sim)

`experiments/encoders/tapct_sanity_transfer.py`: 1-NN label transfer (feature_sim
metric) on frozen tap-ct-b-3d features. use_crop=True, crop_spacing_mm=1.5,
image_size=(224,224,224), liver, context_size=1. Feature grid = last_hidden_state
reshaped (D/4,H/8,W/8)=(56,28,28)=43904 cells; masks reoriented RAS->LPS axial-first
and area-pooled to the grid (soft occupancy). Target s0013 vs a different-subject liver:
  cross-subject transfer_dice 0.433 | precision 0.336 | recall 0.610 | retrieval@1 0.617
  self-context transfer_dice 0.897 (plumbing upper bound)
Signal is real (retrieval@1 14x chance, dice 6x trivial-all-FG) but modest for an easy
organ; precision low (FP on non-liver), self-context <1.0 (bf16 + soft-label boundary
artifact + feature degeneracy in homogeneous interior). Bridge (tapct_features.py) verified
end-to-end. Not yet wired into feature_sim/run.py adapters. Next: ablate to_lps and grid
resolution before committing an adapter.

## 2026-08-06 — orientation verified (tap-ct bridge)

Probed original ct.nii.gz across s0000..s0010: ALL already RAS (as_closest_canonical
is a no-op), all 1.5mm isotropic. So ct.npy is true RAS: axis0=+R, axis1=+A, axis2=+S
(axial/craniocaudal stack = axis 2). Bridge's ras_to_lps_axial_first (transpose(2,1,0)
then flip(1,2)) yields (+S,+P,+L) — EXACTLY the frame tap_ct.py produces via .T ->
DICOMOrient('LPS') -> GetArrayFromImage (verified against its logged 179->depth,
(192,294)->224² in-plane). => to_lps=True is confirmed correct, not assumed. Bridge is
also more robust than the reference, which is only correct because originals happen to be
RAS (it blind-.T's and drops the direction matrix).

## 2026-08-06 — tap-ct transfer-Dice over BENCHMARK_CLASSES (20 subj, 1.5mm)

`experiments/encoders/tapct_benchmark.py` (CSV: tapct_benchmark_transfer.csv). Frozen
tap-ct-b-3d, use_crop, crop_spacing_mm=1.5, image_size=224³, crop_jitter=0 (each
subject-class crop encoded once), round-robin K=1 label transfer over 47 classes.

MACRO transfer_dice=0.218, retrieval_at1=0.332. Per-category dice: Muscles 0.348 >
Organs Abd/Pelvis 0.263 > Bones Limbs 0.240 > Organs Thorax 0.208 > Bones Spine 0.176 >
Vessels 0.140 > Bones Ribs 0.125. Best classes: liver 0.583, urinary_bladder 0.462,
kidney_right 0.452, hip_right 0.435. Worst: ribs 0.02-0.03, vertebrae_T6/atrial_appendage
0.027, common_carotid 0.031, pulmonary_vein/adrenal 0.06.

Signal is strongly SIZE/THICKNESS-dependent: at 12mm in-plane / 6mm depth cells a thin
structure (rib, vessel, small vertebra) is ~1 cell thick so a cell straddles fg+bg and
correspondence collapses -> motivates crop_spacing_mm↓ (finer cells) for thin classes.
Big blobby organs already usable (0.4-0.58). Same thick>thin pattern as patchset3d study.

## 2026-08-06 — tap-ct benchmark made Hydra-configurable

experiments/encoders/tapct_benchmark.py is now a @hydra.main driver reading
configs/experiment/3d/encoders/tap_ct.yaml (singular `experiment`, repo convention).
Exposed knobs: data.{root,classes,split,n_subjects,image_size,use_crop,crop_spacing_mm,
crop_jitter,context_size,mask_downsample,mask_occupancy_thr,eval_seed};
encoder.{precision,compile,compile_mode,to_lps,resize_native,pad_hu};
metric.{soft,tau}; out.{csv,wandb_project,wandb_name}. context_size K pools the next K
subjects (round-robin) as context. Reusable helpers dense_features()/occ_labels() moved
into tapct_features.py (grid dims derived from proc.resize_dims, not hardcoded T). Smoke
test (liver,spleen / 6 subj) OK; CSV -> results/3d_encoders/tap_ct_transfer.csv.
Override examples: data.crop_spacing_mm=1.0 ; encoder.precision=fp32 encoder.to_lps=false.

## 2026-08-06 — tap-ct benchmark QC plots

Added plot.{n_per_task,dir,thr} to configs/experiment/3d/encoders/tap_ct.yaml and
experiments/encoders/tapct_plot.py. n_per_task>0 saves per-sample 1x3 QC figures:
[target+GT] [context+GT] [target+transfer-pred], titled task | tgt/ctx subject ids |
spacing | dice/prec/rec/r@1. Drawn in the feature frame (reoriented iff encoder.to_lps),
axial slice = max-GT slice; pred = 1-NN transfer grid upsampled to full res, thresholded
at plot.thr. Verified liver/spleen figures render correctly (green/cyan GT, red pred).

## 2026-08-06 — soft + hard transfer metrics

feature_sim/metrics.py: added transfer_metrics() — one 1-NN pass scores BOTH soft
(fractional occupancy overlap, threshold-free) and hard (ctx/pred/GT binarised at thr,
real set-overlap Dice) + folds in retrieval_at1. Existing label_transfer left intact
(run.py). tap benchmark now reports soft_/hard_ {dice,precision,recall}, retrieval_at1,
and hard_frac (share of pairs with a non-nan hard Dice — <1 when a class's coarse cells
never reach thr). config metric.soft/tau -> metric.thr. Plot title shows soft+hard.
Smoke (liver/aorta/rib, 10 subj, thr=0.5): liver 0.544/0.567, aorta 0.379/0.502 (soft
under-rates tube boundaries), rib 0.029/0.059. hard>=soft for thick/tubular; both ~0 for
sub-cell ribs. macro reported over classes with any non-nan hard pair.

## 2026-08-06 — plot: add soft-pred overlay panel
tapct_plot.py now saves 1x4 figures: [target+GT] [context+GT] [target+pred hard@thr]
[target+pred soft]. Soft panel = jet heatmap of the upsampled 1-NN occupancy prediction
with per-pixel alpha=value (transparent where ~0), so the graded confidence the hard
threshold discards is visible. Verified on aorta.

## 2026-08-06 — wire tap_ct into feature_sim/run.py

Added TapCTEncoderAdapter (feature_sim/adapters.py): frozen tap-ct-b-3d as an EncoderAdapter,
tiers=[backbone]. Self-preprocesses via tapct_features bridge (de-norm HU / reorient / TAP
processor); encodes once at the ANISOTROPIC native grid (T/4,T/8,T/8), inverse-reorients the
LPS grid back to RAS (_inv_reorient = flip(2,3).permute(0,3,2,1)) so it aligns with
grid_labels' loader-frame mask pooling, caches (storage ptr,shape), then resamples to res^3.
Key fix: _down_to keys off shape[-1] (isotropic assumption) and no-ops on the anisotropic
grid -> features() uses F.interpolate to res^3 explicitly. run.py build_adapter gains a
tap_ct branch (+ _tapct_spec reading eval.tapct); feature_sim.yaml eval.model|tap_ct +
eval.tapct{precision,to_lps,resize_native,pad_hu}. cost.py: adapters can set
flops_traceable=False (TAP SDPA untraceable -> skip fvcore, still time/VRAM). Needs
data.image_size%8==0; tiers must be [backbone]. Smoke (liver,3 subj,res16/32) OK, uses the
same soft label_transfer/auroc/retrieval as colipri/primus -> directly comparable.
Run: python experiments/3d/feature_sim/run.py eval.model=tap_ct 'feature_sim.tiers=[backbone]' ...

## 2026-08-06 — tap_ct feature_sim: found+fixed feature-cache contamination bug

Compared TapCTEncoderAdapter (run.py path) vs standalone native-grid scoring
(tapct_compare_paths.py) on identical crops+pairing. Native path matched EXACTLY
(_inv_reorient verified: inv(occ_lps)==occ_ras, max diff 0.0), but adapter.features()
collapsed small organs (spleen 0.385->0.000). Localized: not resampling, not labels
(grid_labels==area), not align_corners -> the native-encode CACHE. features() keyed
_native_cache on `volumes.to(self.device)` (a throwaway temporary); its storage is freed and
reused across target/context calls -> cache HIT returns the WRONG volume's features. Fix:
key on the ORIGINAL caller-retained tensor (like PrimusEncoderAdapter); dense_features moves
to device internally so volumes stay CPU here. Also reverted the interpolate to
align_corners=False (cell-centered, matches grid_labels). Post-fix adapter tracks native:
spleen native 0.385 -> res48 0.421; liver 0.460 -> 0.475, degrading gracefully to res16.
run.py smoke (liver/spleen, 6 subj, 128^3, jitter=0): liver res32 td=0.590 r@1=0.632,
spleen 0.579/0.623 — sane (earlier ~0.9 was the contamination). No gaps remain.

## 2026-08-06 — tap_ct: compile + backbone_layers tier

- eval.compile=true now compiles the TAP ViT: _maybe_compile gains a tap_ct branch
  (torch.compile(adapter.model, dynamic=False) — static (1,1,T,T,T) shapes), guarded by
  cfg.eval.model=='tap_ct'. Config: eval.tapct.compile_mode. Needs CC/CXX=/usr/bin for inductor.
- TapCTEncoderAdapter now serves the backbone_layers tier: _encode_native_layers uses
  model(output_hidden_states=True).hidden_states (one forward -> every block's token grid),
  each reshaped+inverse-reoriented like backbone; features_per_layer/sample_features_per_layer
  mirror Primus. Emits bb:L0..L{n-1}. Smoke (liver,3 subj): correspondence peaks mid-stack
  (L6-7 td~0.807 @res32) then declines; bb:L11 ~= backbone. n_layers from vit.n_blocks.

## 2026-08-06 — tap_ct encode_gflops populated

fvcore can't trace SDPA (returned None + flooded). Replaced the flops_traceable skip with
TapCTEncoderAdapter.count_encode_flops (torch FlopCounterMode, display=False — counts flash/
SDPA). cost.py prefers adapter.count_encode_flops over fvcore. encode_cost.csv now: tap_ct
128^3 -> 3872.6 GFLOP (8192 tok); scales ~quadratically in tokens (~52 TFLOP at 224^3).

## 2026-08-06 — tap_ct: max_layers truncation (like Primus depth)

eval.tapct.max_layers=N physically truncates the TAP ViT to the first N of 12 blocks
(vit.blocks=blocks[:N], vit.n_blocks=N — the block loop runs ALL blocks so n alone saves
nothing; count assert needs n_blocks updated). backbone then returns block-N's normed
output. Measured (liver 3 subj, 128^3, res32): full 3872.6 GFLOP / 2.73 it/s / td 0.720;
max_layers=7 -> 2260.3 GFLOP (-42%) / 3.65 it/s / td 0.807. Strict win: less compute AND
better features (backbone@7 == bb:L6 full = mid-stack peak). Composes with backbone_layers
(fans out bb:L0..L{N-1}) and eval.compile. Config eval.tapct.max_layers (null=all).

## 2026-08-06 — tap_ct as a frozen PatchSet3D encoder (arch.encoder=tap_ct)

Wired frozen fomofo/tap-ct-b-3d as a PatchSet3D image encoder, mirroring the CoLiPri/Primus
path (exp 30/35). New src/models/tapct_encoder.py TapCTEncoder honours ConvEncoder3D's
contract — forward(B,1,D,H,W) -> (B,out_ch=768,R,R,R), .out_ch/.resolution — reusing the
feature-sim bridge (experiments/encoders/tapct_features.py: de-norm z-scored HU -> RAS->LPS
axial-first -> TAP processor) so the training encoder ≡ the feature-sim tap_ct probe. Encodes
at the native ANISOTROPIC token grid (patch (4,8,8): T/8 in-plane, T/4 axial), inverse-
reorients LPS->RAS, F.interpolate to R^3 (NOT _down_to — anisotropic). Frozen-only (bridge
runs under no_grad); kept in eval mode via a train() override for deterministic features;
reuses Primus's _EncodeCache/_cached_encode for head-only re-eval. encoder_stage early-exits
the transformer blocks (=feature-sim max_layers). NOT spacing-aware (learned pos-embeds are
interpolated to the grid; cell scale set by data.crop_spacing_mm). arch.encoder_precision
(bf16) threaded via train.py build_model; compile block compiles enc.model (dynamic=False).
PatchSet3D __init__ gains the tap_ct branch (ignores native_grid/sidecar). Config
experiment=39_tapct_enc_i_128 (inherits 35: 128^3, res=16, occ@0.5, balanced, nnUNet aug;
overrides encoder=tap_ct, encoder_stage=7, crop_spacing_mm=1.5). Smoke (64^3, K=2): train
forward + predict shapes correct, encoder no-grad while img_embed trains; eval cache path OK.

- 2026-08-07: **train.py profile_timing now also reports per-item (÷ batch size).** The
  `profile_timing` per-phase timers (data/encode/attn) summed whole-batch GPU time and
  divided only by step count → per-step ms, which isn't comparable across batch sizes.
  Added a `prof_items` counter (Σ batch sizes) and per-item metrics `time/{data,encode,
  attn}_ms_item` alongside the existing per-step `time/*_ms`; the tqdm line now prints
  both (`per-step: … || per-item (÷B): …`). Per-item normalizes throughput so a B-sweep
  is comparable, and exposes batching efficiency: on tap_ct (exp39) encode/item stays
  ~flat (101ms@B1 → 110ms@B8) = frozen ViT is compute-bound, batching gives no per-task
  encode speedup. NB compare at e1+; e0 per-item is inflated by torch.compile warmup
  (dynamic=True recompiles at the new batch shape). No-op unless train.profile_timing=true.

- plot_dataset_items.py: tightened row/col spacing (gridspec hspace/wspace=0.02,
  tight_layout h_pad/w_pad=0.2) for denser, more readable grids; row ylabel now
  appends per-item voxel spacing (`d0×d1×d2 mm`) when the batch carries a `spacing` key.
  Also: on the train split, when data.spacing_range is set, the loader now mirrors
  train_loader via SpacingBatchSampler(batch_size=1) so each row draws its own
  log-uniform crop spacing (plain plot loader ignored spacing_range → every row was
  fixed crop_spacing_mm). spacing_range is a loader-level knob (only train_loader/
  SpacingBatchSampler read it); crop_spacing_mm is the fixed fallback for all splits.

- 2026-08-07: **eval.spacing_sweep config key added (gated + off by default).** New `cfg.eval.spacing_sweep` parameter in `configs/experiment/3d/eval.yaml` enables multi-spacing evaluation sweeps on the same dataset. When set to a list like `[1.5, 2.5, 3.5]`, eval produces per-(class, spacing) rows in eval.csv/json and per-class wandb scalars tagged with spacing (`class/<c>/mean_dice@<s>`). All samples evaluated at each spacing with same GFLOPs/sample; wall time scales linearly with number of spacings. Figures are saved only on the first spacing to reduce I/O. Requires `data.use_crop=true` and totalseg source; pair with `crop_jitter=0` for a fully controlled sweep (centered crops, only spacing changes). Usage: `python experiments/3d/eval.py 'eval.spacing_sweep=[1.5,2.5,3.5]' eval.crop_jitter=0`.

- 2026-08-07: **eval.spacing_locator config key added (coarse→fine layered on spacing_sweep).** New `cfg.eval.spacing_locator` parameter in `configs/experiment/3d/eval.yaml` enables coarse→fine localization metric layered on `spacing_sweep`. For each descending consecutive pair (e.g. [4, 2] in `[4, 2]`), uses coarse (4 mm) prediction to place a fine-spacing (2 mm) bounding box and measures containment |GT ∩ box| / |GT|, plus an oracle box on GT centroid. Adds one soft-prob forward per non-final spacing; centroid via soft-prob-weighted voxel sum (hard-mask fallback) through `model.train_forward`. Produces per-(class, spacing-pair) columns in eval.csv/json + `class/<c>/containment@<s>` wandb scalars. Requires `spacing_sweep` with descending step (+ totalseg / use_crop). Default false. Usage: `python experiments/3d/eval.py 'eval.spacing_sweep=[4,2]' eval.spacing_locator=true eval.crop_jitter=0`.

- 2026-08-07: **spacing_sweep + spacing_locator now support `data.source=totalseg_more_labels`.** `TotalSegMoreLabelsDataset` subclasses `TotalSegInContextDataset`, inheriting the `(idx, spacing)` crop override (`__getitem__` → `_cur_crop_spacing`), `_reported_spacing` (reports the swept `crop_spacing_mm` under use_crop), and `_organ_crop_arrays` (sizes the FOV as `T*self._crop_mm`); its overridden `_load_crop` delegates the extent to that base helper. Only two things blocked it: the guard rejected the source, and `make_eval_loader`'s build_dataset branch ignored the `spacing` arg. Fix: `_assert_sweep_supported` now rejects only omnisynth3d/anchor_synth3d; `make_eval_loader` wraps the build_dataset-routed dataset in `SpacingBatchSampler([s,s])` when `spacing` is set (only more_labels ever reaches there with spacing, since omnisynth/anchor stay guarded out). No new params. Usage: `python experiments/3d/eval.py dataset=totalseg_more_labels 'eval.spacing_sweep=[4,1.5]' eval.spacing_locator=true eval.crop_jitter=0`.

- 2026-08-08: **Analysis notebook `results/experiments/37_patchset_spacing_locator.py`** (marimo) — pulls the patchset3d spacing-sweep+locator eval run (`tidiane/patch_icl_3d_eval/05kb6kcc`, `spacing_sweep=[4,1.5]`, `spacing_locator=true`) via the W&B API. Reuses `nb_common.get_latest_table(table_key="cases.table.json")` for the per-sample table (per-sample `spacing` → coarse 4 mm / fine 1.5 mm are two conditions in one table) and parses per-class locator `containment@4`/`containment_oracle@4` from `run.summary` (per-sample containment is not logged). Geometry + shape families from `totalseg_geometry_extract`; caches under `artifacts/37_05kb6kcc_*`. Focus = the FINE (@1.5 mm) dice distribution under three cuts, each with a morphology control: (1) trained vs held-out — raw held-out advantage (+0.186 macro) is an anatomy confound; matched lateral-mirror pairs collapse it to Δ=−0.023; (2) fine dice vs coarse containment — marginal ρ=−0.11 (low containment = large objects, containment↔log_volume=−0.18, not accuracy); (3) when the oracle fails (oracle<1 crop-size ceiling) — ceiling classes score HIGHER (0.597 vs 0.494, ceiling↔log_volume=+0.37) because oracle<1 marks large easy organs, so the fine window is not the accuracy bottleneck.

- 2026-08-08: **Analysis notebook `results/experiments/38_patchset_more_labels_failure.py`** (marimo) — failure analysis of patchset3d on `totalseg_more_labels` (eval run `tidiane/patch_icl_3d_eval/gcoroxrx`: 285 novel held-out hierarchical classes `task/structure`, `spacing_sweep=[4,1.5]`+locator). Sibling of nb 37 but **no** `totalseg_geometry_extract` (different dataset root + hierarchical names) — size drivers come straight from the cases table (`tgt_size`/`tgt_occ`/`ctx_occ`); per-class locator containment parsed from `run.summary` (regex handles the '/' in class names; NaN=empty coarse pred). Caches `artifacts/38_gcoroxrx_*`. Focus = the fine (@1.5mm) accuracy distribution: cell 1 by **task** (per-sample dice box per group, macro/miss table), cell 2 by **class** (dice histogram+ECDF, worst/best-20 bars coloured by size), cell 3 by **size** (dice vs tgt_size & ctx_occ, ρ≈+0.5; miss-rate/box per size sextile — two-tailed tiny+huge failure), cell 4 **localization vs segmentation** taxonomy (empty-coarse 6 / diffuse-ceiling 13 / localization 14 / segmentation 126 / ok 126 → segmentation on tiny novel structures is the bottleneck, containment 0.83, 42% of well-localized still fail). Findings in memory `project_more_labels_failure`.

- 2026-08-10: **Universal-coords invariance assessment** — `experiments/3d/universal_coords/coord_invariance.py` (+ `figs/`). Question: can the `coords` field (NN that maps every voxel→3D canonical body position; ChemoTox cohort, `coords_paths_chemotox.json`, 366 scans/220 patients) serve as a shared frame so a fixed coords-region generates the same anatomical synthetic label across subjects. Alignment: coords grid (~90×90×80, 4/4/8mm) and full-res totalseg share the same world origin — sample totalseg at coords-grid world points via affines (all coords voxels land in-volume). Three methods: (1) **centroid consistency ratio** = between-subject centroid spread / within-subject organ extent per totalseg label; majority <1 (position agrees tighter than the organ's own size). (2) **LOO nearest-centroid retrieval** of label identity from a coords centroid over 60 unique patients: top-1 **0.743**, top-5 **0.959** over 106 labels → coords IS a retrievable canonical body frame. (3) **synthetic-label round-trip**: apply LOO canonical centroid + coords-threshold to held-out subject, measure purity/Dice vs true label. Isotropic 25mm ball macro Dice 0.18 (purity high for big organs: liver 0.78/heart 0.71/lungs 0.67-0.70); anisotropic ellipsoid (k·within-std/axis) improves to Dice 0.27 @k=2. Conclusion: **position is invariant/retrievable; the coords-threshold shape is the weak link, not the coords field** — a good generator needs a per-label shape/extent model, not a fixed ball. CLI: `--extract [--unique] --n N`, `--analyze`, `--synth [--ellipsoid] --radius k`.

- 2026-08-10: **coords-driven ellipsoid transfer, cheap matchers benchmarked** — `experiments/3d/universal_coords/{transfer_methods.py, plot_ellipsoid_transfer.py}`. Idea: draw ellipsoid on ctx in IMAGE space -> its voxels' coords form an irregular cloud Q -> select tgt voxels whose coords match Q (uses both ctx & tgt coords fields as a dense correspondence). Four matchers, eval = Dice/purity of transferred tgt mask vs tgt totalseg organ, over 6 pairs × 8 organs on the coarse 90³ coords grid: base (analytic axis-aligned ellipsoid in coords, no cloud) 0.265 D / 17ms; gauss (Mahalanobis mean+full 3×3 cov) 0.264→**0.285** D when source=real-organ footprint / 24ms; bin (b=8mm coords-bin hashing, O(N), no tree) 0.272 / 32ms; knn (cKDTree radius τ=8) 0.278→0.290 / **540ms (20×)**. Findings: (1) for an ELLIPSOID source all matchers ≈ equal (~0.27) — the ctx ellipsoid maps to a near-ellipsoid coords blob so the cloud adds nothing; (2) cloud transfer only helps when source shape is non-ellipsoidal (real organ) and only +0.02 D; (3) Dice ceiling ~0.29 is matcher-agnostic → bottleneck is coords resolution (4/8mm) + cross-subject anatomy, not the algorithm. **Recommendation: bin-hashing default (cheap, non-parametric, arbitrary shapes); gauss if a smooth parametric region is wanted; knn not worth 20×.** To raise ceiling: finer coords / intensity-based local refinement.

- 2026-08-10: **next8 coords model CAN produce a finer map — coarseness is a stitch-grid choice, not a model limit.** Explored `experiments/3d/universal_coords/coords_predictor/next8.py` (PatchWork2 model `.../next8/model_patchwork.json`, applied via `model.apply_on_nifti(..., level="mixnohead", scale_to_original=False, sampling_factor=1)`). Config: `cropper.scheme.destvox_mm=[4,4,8]` → that is exactly why coords come out 4/4/8mm (90×90×80). But the pyramid (`patch_size=[32,32,32]`, `fov_mm=[400,400,800]`, `scale_fac=0.4`, `depth=4`) carries genuine detail per level 12.9/5.2/2.1/**0.83**mm in-plane (z 25.8/10.3/4.1/**1.65**mm); `mixnohead` already fuses fine levels, so 4/8mm is downsampling info the net holds. Resolution knob = `sampling_factor`(→`destshape_size_factor`, crop_generator.py:884 `dshape=round(dssf*psf*input_width/wperm+1)`, linear in dssf; supports `[dx,dy,dz,'mm']` target-voxel form) and/or `scale_to_original=True` (trilinear to input grid). Recipe to regenerate: set `sampling_factor=[2,2,4,'mm']` (→2/2/4mm, level-2 detail, real gain, ~8× voxels) or `[1,1,2,'mm']` (near finest; diminishing returns since coords were SUPERVISED at 4/8mm). **Blocker to actually re-run:** model saved under Keras 2; current `/software/anaconda3/envs/tensorflow` is TF2.18/Keras3 → `warpLayer`/`Conv3D` deserialize fails; also Blackwell GPU (sm_120) unsupported by TF2.18 (CUDA_ERROR_INVALID_HANDLE). Need a keras-2 TF env (CPU ok) to regenerate. Cheap stopgap w/o re-running: trilinear-upsample existing 4/8mm coords (smooth field) to kill the 90³ discretization in the transfer — no new detail but removes part of the 0.29-Dice grid penalty.

- 2026-08-10: **Option-1 (trilinear-upsample coords) does NOT lift the transfer Dice ceiling — the limit is coords fidelity, not grid quantization.** `experiments/3d/universal_coords/finer_transfer.py`: trilinear-upsample existing 4/8mm coords to 2/4mm (factor 2), sample sharper totalseg GT on the fine grid, sweep ellipsoid size K∈{0.5..3.0} × matchers {base(diag ellipsoid), gauss(full-cov Mahalanobis), bin(8mm hash)}, knn excluded; all three transfer the SAME ctx ellipsoid cloud. Result (Dice vs tgt organ, 3 pairs × 8 organs): peak at **K≈2.5 ~0.236**, x1 vs x2 essentially identical (base K2.5: 0.236→0.236; bin only helps at tiny K: K0.5 0.036→0.049). Matchers tied (base≈gauss; bin marginally higher at small K). CONCLUSION: upsampling adds no info (coords is smooth), so the ~0.24 ceiling is intrinsic 4/8mm coords fidelity + cross-subject anatomy, NOT the 90³ discretization — refutes the cheap-stopgap hope. Real lever = regenerate genuinely finer coords from the model (sampling_factor=[2,2,4,'mm']), which needs the keras-2 TF env. Also: ellipsoid K≈2.5 is the sweet spot (K=3 over-inflates → purity loss).

- 2026-08-10: **Free (label-agnostic) random-ellipsoid transfer works — the synthetic-label recipe.** `experiments/3d/universal_coords/random_ellipsoid_transfer.py`: draw an ellipsoid at RANDOM body position (center sampled from CT>-300 body mask on the coords grid), RANDOM radii 15-50mm and RANDOM orientation on ctx → coords cloud Q → transfer to tgt via bin/gauss. No label tie, so eval = totalseg label-histogram intersection (incl bg) ctx-region vs tgt-region. Over 160 samples × 4 pairs: **bin 0.773±0.206, gauss 0.770±0.217, random-placement baseline 0.522±0.219** → transfer is +0.25 over chance; empty-tgt ~1-2% (near FOV edges). bin≈gauss (tied). Figure random_ellipsoid_transfer.png: bin/gauss tgt regions near-identical, land on corresponding anatomy (iliac hist∩0.83, central 0.73); weak case = organ/body-wall boundary (0.28). Confirms: can generate position-consistent synthetic labels as free random blobs and transfer through coords with the cheap matchers. Caveat: baseline 0.52 is high because bg/fat (totalseg label 0) dominates and inflates intersection; use bg-excluded metric for sharper discrimination. Sweet spot from finer_transfer: none needed here (free size), but ellipsoid mm bounds set blob scale.

- 2026-08-10: **Random-ellipsoid transfer, bg-EXCLUDED metric on 16 pairs — chance collapses, signal is decisive.** Re-ran random_ellipsoid_transfer.py with label-hist intersection restricted to totalseg labels 1-117 (bg/fat=0 dropped, renormalized) and only scoring blobs whose ctx region is >=30% labeled anatomy (else it's fat, meaningless). 16 pairs × 40 samples → 345 scored (295 skipped as mostly-fat). **bin 0.626±0.262, gauss 0.615±0.276, random-placement chance 0.052±0.148.** vs the incl-bg version (0.773/0.522) the bg was inflating BOTH; bg-excluded shows transfer places free blobs on the SAME labeled anatomy ~0.62 organ-composition overlap vs ~0.05 chance — ~12× over chance. bin≈gauss. ~46% of free-position blobs skipped (land in fat/muscle outside the 117 classes). Confirms coords-transfer of free random ellipsoids is a strong position-consistent synthetic-label generator.

- 2026-08-10: **next8 coords model RUNS on nero — reproduced 4/8mm + generated genuine 2/2/4mm map.** Clean standalone runner `experiments/3d/universal_coords/coords_predictor/run_next8.py` (original next8.py needed NORA DPX_selectFiles + had undefined f1/f5). Env that works: `/software/anaconda3/envs/patchwork_minimal/bin/python` (TF 2.12 / **Keras 2.12** — Keras2 fixes the warpLayer/Conv3D deserialize that broke on odin's TF2.18/Keras3). GPU (A4000, Ampere) needs `LD_LIBRARY_PATH=/software/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib:/software/anaconda3/envs/tf215/lib/python3.10/site-packages/nvidia/cudnn/lib` (TF2.12 built cuda11.8/cudnn8; libs not on default path). Baseline (`--sampling 1`) reproduces original coords exactly: (90,90,80,3)@3.996/3.996/8.013mm. Finer: use SCALAR `--sampling 2` → (179,179,159,3)@2/2/4mm, 99% coverage, ~74s GPU. **BUG**: the `[dx,dy,dz,'mm']` sampling_factor form crashes (model.py:1066 `if sigma>0` on a vector sigma) — use scalar factor instead. Genuine-vs-interp check: model-2mm vs trilinear-up of 4mm map = mean|Δcoords| 3.7 (p95 7.3), +11% gradient magnitude, Δ concentrates at boundaries → real sub-4mm detail, but modest because coords is an intrinsically smooth position field (z-channel = smooth SI ramp). Next: batch-generate 2mm coords for the cohort (~74s/scan) and re-run transfer benchmark to see if real finer coords lifts the ~0.6 correspondence. Fig figs/model_finer_vs_upsample.png.

- 2026-08-10: **Genuinely finer coords do NOT lift transfer correspondence — resolution is exhausted as a lever.** Generated factor-3 (1.33/1.33/2.67mm) coords for 20 scans via batch_next8.py, then `finer_random_transfer.py` re-ran the free random-ellipsoid transfer (bg-excluded label-hist intersection) on 8 cross-patient pairs, same seed, comparing orig 4/8mm vs fine 1.33mm coords. Result: orig bin 0.676/gauss 0.649/chance 0.041 vs fine bin 0.633/gauss 0.626/chance 0.062 — FLAT (fine a hair lower, within ~1-2 SE). Third consistent line of evidence (with trilinear-upsample no-help + model-2mm-vs-interp modest) that the ~0.65 correspondence ceiling is intrinsic to the coords model's cross-subject accuracy (supervised at 4/8mm), NOT the output grid. Levers left: better/finer-TRAINED coords model, or post-hoc intensity/boundary refinement after transfer. Highest res that runs as-is on 16GB A4000 = factor 3 (factor 4 OOMs on GPU anti-alias conv). 20 finer maps in coords_predictor/output_batch/ (2.4GB).

## 2026-08-10 — Coords model on TotalSeg (loki): finest-level generation + ellipsoid-transfer Dice
- Loki (RTX 6000 Ada, 48GB) coords inference: factors 1-6 all fit (nero A4000 OOM'd >3). Time ~28-32s flat for f1-3, 45.7s(f4)/64.4s(f5, 0.8/0.8/1.6mm)/89.9s(f6). ~3-4x faster than nero at matched res.
- TotalSeg test set: uniformly 1.5mm iso, dims 99-454 (FOV up to ~680mm). Generated finest coords (sf=5, 0.8/0.8/1.6mm) for 20 test cases -> coords_predictor/output_totalseg/ (9.2GB, no OOM fallback even on 562^3). Scripts: coords_predictor/batch_totalseg.py.
- Resampled those onto each case's native CT/label grid (1.5mm iso, full-affine trilinear) -> coords_predictor/output_totalseg_1p5/. Now coords/ct/label co-registered voxel-for-voxel (label.npy is on the CT grid). Script: resample_totalseg_coords.py.
- Ellipsoid-transfer Dice (totalseg_ellipsoid_transfer.py, mirrors chemotox transfer_methods.py; 8 full-body pairs, 12 largest shared organs, ~3mm eval): base 0.191 / gauss 0.194 / bin 0.196. Matchers TIED (matcher-agnostic ceiling, as chemotox). vs ChemoTox coarse-grid 0.265/0.285/0.272 -> totalseg ~30% LOWER. Likely DOMAIN SHIFT (coords model trained on chemotox body-comp CT; totalseg = heterogeneous off-distribution cohort). Per-organ: thin/ambiguous worst (lung-lobe L13 0.06, L86 0.08), vertebrae/solid organs 0.20-0.29. Confirms: resolution not the lever, domain/training is.
- Note re 1.5mm iso native gen: scalar sampling_factor keeps [4,4,8] anisotropy (only [4/f,4/f,8/f] reachable); per-axis '[1.5,1.5,1.5,mm]' path exists in crop_generator but crashes at model.py:1066 `if sigma>0` on vector sigma (one-line bug). Resampling the finer maps is equivalent in info (interpolation shown to add no correspondence), so used that.

## 2026-08-10 — TotalSeg free-blob transfer viz + synthetic-task batch generator
- plot_totalseg_free_transfer.py: freely-positioned ellipsoid transfer (random body pos + radii 15-50mm + orientation, NOT organ-tied) on aligned 1.5mm coords; overlays ctx blob -> tgt bin/gauss, annotated with bg-excluded label-hist intersection (HI), plus GT contour (green = union of ctx-covered organs projected on tgt). Figs for pairs s0040->s0667, s0029->s0687. Big organs land inside GT (HI ~0.5-0.9); free-blob HI >> organ Dice (softer composition metric).
- free_synth_generator.py: batch generator of K+1 position-corresponding (subject,mask) in-context tasks over the 20 test cases. Reference free blob -> coords cloud Q -> bin-transfer to K contexts. Masks stored sparsely (flat nonzero idx) as npz/task + manifest.json -> output_synth_tasks/.
- CRITICAL: needs a validity guard. Without it, mean cross-subject HI 0.336 +/- 0.307 (min 0.0) — partial-body contexts (head/extremity) give non-empty but OFF-anatomy masks. Added --min_hi reject-resample guard: HI>=0.3 -> mean 0.673 +/- 0.151 (min 0.406), 30/30 tasks in 57 tries (vs 31). Guard is essential for a usable generator on a mixed-anatomy cohort.

## 2026-08-10 — All-scans coords generation for TotalSeg
- coords_predictor/batch_all_totalseg_coords.py: generates coords for ALL 1228 totalseg scans (all splits). Per scan: run next8 model on ct.nii.gz at sampling_factor=2 (2/2/4mm), then trilinear-resample onto the NATIVE ct grid (1.5mm iso) via full affines, cast float16, save <scan>/coords.npy shape (X,Y,Z,3). Co-registered with ct.npy/label.npy (all share native grid, verified) -> rides use_crop=true crop_spacing_mm=1.5 with no bridge. Resumable (skip existing), OOM fallback sf2->1. ~17-20s/scan, ~6-8h total, ~50GB float16.
- SPACING CHOICE (user): 1.5mm-iso float16 co-registered w/ label.npy. Rationale: correspondence is resolution-INVARIANT (proven), so fine gen wasted; must co-register with what use_crop reads (ct.npy/label.npy native 1.5mm). Generate coarse (sf2) + resample down is info-equivalent. Alternatives rejected: finest 0.8mm ~565GB no benefit; pre-resized 128^3 ~15GB but only fits the non-crop fast path.
- apply_on_nifti(ofname=None) returns bare ndarray (no affine) -> route gen through a temp nifti to recover the 2mm affine for resampling.

## 2026-08-11 — Coords quality on TotalSeg (balanced classes)
- coords_quality.py: for 50 random subjects, centroid of each BALANCED_CLASS (61) in canonical coords frame (coords.npy voxel-aligned with label.npy, stride 2 ~3mm), per-class between-subject spread vs within-subject extent, + LOO nearest-centroid class retrieval.
- RESULT: LOO retrieval top-1 0.386 / top-5 0.747 (chance 0.016) vs CHEMOTOX top-1 0.743 / top-5 0.959 -> accuracy ~halves off-distribution. Cross-subject centroid spread ~60-120mm (want 10-30mm). Only 1/61 classes tight (ratio<1), median ratio 2.22.
- STRUCTURE: best = large solid organs/cavities (liver 1.18, colon 1.23, heart 1.59); worst = small/repeated instances (cervical vertebrae C3 6.45/C7 5.98, upper ribs, thin vessels brachiocephalic_trunk 8.93/SVC 4.39, thyroid 5.30, adrenal 5.11) — within-extent tiny so ratio explodes but raw between (90-110mm) genuinely bad. Vertebra/rib families = instance ambiguity (places "a vertebra" not "which").
- Reconfirms DOMAIN SHIFT at the position level (not just downstream Dice). Frame usable for coarse large-anatomy positioning, noisy for thin/repeated/lateralized -> synth-gen should favor large free blobs on trunk anatomy + HI validity guard, not rely on coords to pin fine structures.

## 2026-08-11 — What stable info is in the coords maps? (body-axis diagnostic)
- shape_stability.py: 4 blob shape families (ellipsoid/metaball/noise/coords_ball) size-matched ~3000vox at SAME body positions, bin-hash transfer to K tgt, score same-region landing (bg-excl label-hist intersection HI). RESULT (40 refs, n=146/family): ALL TIED — HI 0.362-0.369, surv 0.73. Even coords_ball (shape defined in shared frame) no better => SHAPE is irrelevant to transfer stability; coords-map fidelity is the cap. Levers that matter = size, position, HI/survival guard, not shape.
- coords_axes.py: per subject, correlate each coords channel vs RAS world axes (from ct affine) over labelled-anatomy voxels. RESULT (30 subj): c0<->LR |corr|0.986, c1<->AP 0.976, c2<->SI 0.931; axis assignment 100/100/97% consistent, SIGN 100% consistent across subjects; linear R^2 = 0.989/0.983/0.975 (mean 0.982). SI ordering: 16 landmark organs head->foot reproduced at Spearman |rho|=0.93 (min 1.0).
- KEY: coords is ~98% an AFFINE transform of raw scanner RAS per subject -> only ~2% nonlinear body-normalization. Reconciles everything: BODY AXES / coarse position = excellent+stable (clean warped body-coordinate frame, reliable SI level/laterality/quadrant); FINE organ correspondence = weak (retrieval 0.39, 60-120mm spread) BECAUSE little nonlinear warp on top of affine. => coords good as COARSE localizer/positional prior (crop, region conditioning, large-region synth), not for pinning small structures.

## 2026-08-11 — Coords-function synth labels: Phase-0 + visual QA (pre-wiring)
- Design: synthetic label = smooth field f(coords) in [0,1], evaluated per subject -> cross-subject correspondence by construction (no matcher/transfer). Two tiers: hard bounded primitives + soft coords-Gaussians. Spec: docs/superpowers/specs/2026-08-11-coords-synth-labels-design.md. Field primitives in coords_synth_consistency.py (FIELDS, sample_params, LOCALIZED, coords_aabb).
- Phase-0 consistency sweep (coords_synth_consistency.py, 20 subj): cross-subject anatomy HI ~0.25-0.31 (~5-6x random chance); VARIANCE is the selector -> scale floor ~40mm (below it HI std doubles/triples + masks miss FOV). gaussian sigma>=40, slab/cyl >=40 stable.
- Visual QA (plot_coords_synth.py) CAUGHT A DESIGN BUG before wiring: unbounded primitives (half-space/full slab/long cylinder) FAIL on heterogeneous FOVs — a chest-anchored half-space hits lungs in a chest scan but SKULL 98% in a head-only scan (mean-pairwise HI averaged the bimodality away). FIX: (1) retire unbounded primitives, use LOCALIZED/bounded fields only (anisotropic ellipsoid, capped cylinder, gaussian); (2) FOV-aware grouping — precompute per-subject coords AABB, only group subjects whose covered region contains the anchor mu; (3) consistency backstop mean pairwise HI>=0.15 else redraw field. After fix: every montage row hits same anatomy across target+contexts (HI 0.53-0.83), skull-for-lung gone.
- Also: existing supervoxel synth path uses 1 subject + K+1 AUG COPIES (identical position) -> coords synth (K+1 different subjects, one field) is complementary: injects cross-subject POSITION signal supervoxels lack. Integration = new p_coords mode; hard labels drop into integer pipeline unchanged (Phase A), soft labels need float plumbing (Phase B).

## 2026-08-11 — self_context refactor into nested {p, augs.{intensity, per_image}} + translation-jitter drift result
- WHY: the "translation-alone" self_context run (l7awfrqg) was actually translation + INDEPENDENT intensity jitter — line 984 applied apply_intensity_aug to every context clone unconditionally when self_context_augs=true. Previous ceiling run (self_context=true, augs off) had context = exact clone of the augmented target -> zero intensity mismatch. So the val-ceiling drop 0.95->0.88 was confounded (translation AND appearance).
- REFACTOR: flat data.self_context (float) + data.self_context_augs (bool) -> nested data.self_context.{p, augs.{intensity, per_image}}. Two independent toggles so each aug family is isolatable for clean A/B. Dataloader ctor: self_context (=p) + self_context_intensity + self_context_per_image; getitem gates per_image via self_context_per_image, intensity via self_context_intensity (separately). common.py: _self_context(d) parser (nested form, scalar/bool fallback -> p only) used in build_dataset + make_eval_loader. totalseg.yaml nested block; nnunet.yaml comment updated. Verified: parser unit cases + hydra struct-mode compose of data.self_context.augs.per_image=true etc.
- DRIFT RESULT (probe_context_drift.py, 120 val samples, ceiling vs translate ckpt; CSVs results/3d/exp40_context_drift_{ceiling,translate}.csv): translation jitter LIFTS THE WHOLE POSE-DRIFT SURFACE, transferring to UNTRAINED transforms. translate_vox@16 0.356->0.769 (+0.413); rotate@20deg 0.373->0.664 (+0.290, rotation NOT trained); scale@1.15 0.557->0.785 (+0.228, scale NOT trained); elastic@16 0.657->0.790 (+0.133). Cost: exact-match ceiling (mag0) 0.951->0.866 (-0.085), worst on small objects. Interpretation: ceiling model solved self-context by position-locked copying (output=ctx mask at same voxel addr); pose jitter breaks the shortcut -> forces feature-based localization -> generalizes across transform types. H2 (pose-invariant matching bottleneck) is trainable; pose jitter is the lever. NEXT: isolate translation from intensity (self_context.augs.intensity=false rerun), then add rotation+scale, then cross-subject (p<1.0).
- SPLIT-AWARE p: data.self_context.p is now {train, eval} (was a single scalar) so you can train on self-context (p.train=1.0) but evaluate on real cross-subject contexts (p.eval=0.0). _self_context(d, split) resolves p.train for split=='train', p.eval for val/test; accepts scalar-p (both splits) and whole-block-scalar fallbacks. augs.{intensity,per_image} apply to both splits. Verified via parser cases + struct-mode compose of data.self_context.p.train/p.eval.

## 2026-08-11 — feat_norm arch flag (context|self|none) for the encoder-feature normalization
- CONTEXT: patchset3d _attn z-scored BOTH support and query by SUPPORT per-channel stats (mu/sig over dim=1, applied to sup_feat AND qry_feat) — inherited verbatim from patchset_cnn.py -> pfn_seg_2d.normalize_by_context ("matches TabPFN feature_sim backend"), never re-justified for frozen 3D CT. Puts the query in the context's feature frame; benign in self-context (support≈target clones) but distorts the query under cross-subject mismatch and is background-dominated for small objects — a suspected contributor to the size↔Dice / cross-subject stall (H2).
- ADDED arch.feat_norm (patchset3d.py): _feat_norm() dispatch. context = current (support stats on both); self = each side by its OWN stats (query decoupled); none = pass-through. assert-guarded; default 'context' (unchanged). Wired: build_model (train.py a.get feat_norm), model/patchset3d.yaml declares it. Weight-free — eval.py adds an eval-time override (eval.feat_norm, default null) applied on top of the ckpt's stored arch so ONE checkpoint sweeps all three (older archs lack the key -> default context).
- VERIFIED: 3 modes build/forward finite + mutually distinct outputs; bogus -> AssertionError; hydra struct compose of arch.feat_norm + eval.feat_norm. NEXT: eval-only sweep context|self|none on the base cross-subject ckpt (self_context.p.eval=0.0) to see if decoupling the query moves cross-subject Dice.

## 2026-08-11 — Per-sample table: ctx_cases + self_ctx columns (self-context provenance)
- WHY: with split-specific self_context.p (train self-context, eval cross-subject) the val/eval sample tables gave no way to tell which rows were self-context vs real cross-subject. Added per-context case-id provenance.
- Dataset (totalseg_dataloader_incontext.py): track ctx_subjects list alongside context_in/out — real path records each sampled ctx subject; empty-fallback + resample-pad record the target/reused subject; self_context block sets [subj]*K; synth path sets [subj]*K (K aug-copies of same subject). Emitted as item['context_subjects'] (list[str] len K). Collate passes it through (guarded -> backward-compatible).
- evaluate.py: evaluate_classes reads batch['context_subjects'] -> per case ctx_cases (';'-joined ids) + self_ctx (bool: all ctx == target case). _SAMPLE_TABLE_COLS + build_sample_table gain both columns (after 'subject'). Covers train.py val table AND eval.py (both go through evaluate_classes/build_sample_table; evaluate_spacing_sweep too; legacy validate() unused). Absent key -> ctx_cases='' self_ctx=None (non-totalseg sources).
- VERIFIED: collate pass-through + backward-compat; build_sample_table 16-col/16-val alignment with defaults.

## 2026-08-11 — plot_dataset_items.py: per-context case id + self-context markers
- Uses the new batch['context_subjects']: each context cell gets a badge with its case id; when a context's case id == the target case (self-context / leaked clone) the badge turns crimson with "SELF" + a crimson cell border, and the row ylabel gains [SELF-CTX] (or [part-self n/K] when only some contexts are self). No-op when the source doesn't emit context_subjects. Verified on totalseg train: self_context.p.train=1.0 -> all SELF; =0.0 -> distinct cross-subject ids, no markers (results/3d/dataset_items_{selfctx,cross}.png).
- GAP found: only augmentations/nnunet.yaml has the per_image block; the totalseg default preset (multiverseg_v2) lacks it, so data.self_context.augs.per_image=true silently no-ops there (and augmentations.per_image.* overrides raise struct errors). Needs per_image added to multiverseg_v2 (mirror nnunet) for the pose-jitter lever to work on the default preset.

## 2026-08-12 — chemotox converter: affine-based label resampling (fix zero-size crop crash)
- BUG: `python experiments/3d/eval.py dataset=chemotox ...` crashed in a DataLoader worker — `F.interpolate ... input (D:0, ...)` inside `_place_image`. Root cause: 13/366 chemotox subjects have `ct.npy` and `label.npy`/`bc.npy` on DIFFERENT shapes (e.g. 16258784 ct (259,259,218) vs label (517,517,304)). The crop path (`_organ_crop_arrays`) derives voxel crop indices from `label.npy.shape` then slices `ct.npy` at them → out-of-range → empty (0-length) crop → interpolate crash.
- WHY: the img and the totalseg/bclabels masks are NOT on one native grid. Inspecting affines: they differ in shape, spacing AND origin (16258784: img 0.758mm origin z=194 vs ts 0.424mm origin z=-119; 20813849 origins 343mm apart). They relate only through world coordinates. The old converter read ONE spacing from the img affine and per-axis `ndi.zoom`-resampled every label with it → misaligned/wrong-shaped labels for those subjects.
- FIX (scripts/convert_to_npy.py, chemotox path only; totalseg byte-identical): `load_raw` now returns nibabel IMAGES (each label carrying its own affine). `_convert_chemotox` resamples the CT to the target 1.5mm grid via `nibabel.processing.resample_to_output` (order=1), then resamples every label onto that exact grid via `resample_from_to((ct.shape, ct.affine), order=0)` — world-aligned, guaranteed same shape as ct. Removed the now-dead `_resample_to_spacing` helper. Verified on the 4 offenders + 1 good subject: ct/label/bc shapes all equal and every labeled voxel lands inside the CT body (lbl_in_body_frac=1.00). NB: CT shapes shift slightly (canonical RAS grid) vs the old ndi.zoom cache — internally consistent.
- DEFENSE-IN-DEPTH (totalseg_dataloader_incontext.py `_organ_crop_arrays`): assert ct_mm.shape == label_mm.shape with the culprit subject name — converts silent misalignment/interpolate-crash into a loud, actionable error.
- Dataloader smoke: instantiated TotalSegInContextDataset over the re-converted 5 subjects (incl. all offenders), iterated every crop item, all image/label 128^3, no crash. Tests updated (test_convert_chemotox = nib-image contract; test_convert_generalize = labels-on-different-native-grid resample-onto-ct-grid alignment); 9/9 chemotox+converter tests pass.
- OPERATOR ACTION REQUIRED: re-run the chemotox conversion with --overwrite (the on-disk cache still holds the old misaligned data), then delete the stale scan/bbox caches so they rebuild:
  `python scripts/convert_to_npy.py --source chemotox --out <chemotox> --target-spacing 1.5 --workers 32 --overwrite`
  `rm <chemotox>/.scan_cache_*.pkl <chemotox>/.bbox_cache_*.pkl <chemotox>/.bc_centroid_cache_*.pkl`

## 2026-08-13 — infer_nifti: multi-label context masks + batched cascade
- Extended predict_nifti (experiments/3d/infer_nifti.py) to segment MANY organs from one id-valued context mask (e.g. TotalSegmentator total_seg_total.nii.gz). New args: label_ids (None=single binary organ, original behavior; a list of ints or "all" = multi-label) and batch_size (label-tasks per model forward, default 8). Single-organ = the 1-label degenerate case (byte-identical return contract: bool pred, scalar dice).
- Design: for label L, each --context MASK contributes (mask == L) as that label's binary context (K>1 still works). The coarse->fine cascade runs per label, batched batch_size labels on the model's batch dim (target_b (B,1,T³), ctx_in_b (B,K,1,T³), ctx_out_b (B,K,T³)) — one model.predict per chunk. Coarse target crop is shared (volume centre); fine crop is per-label (its own predicted centroid) — both handled by the batch.
- Output: one id-valued uint8 mask on the target's original grid; overlaps resolved smaller-organ-wins (larger organs written first). --gt (id-valued) -> per-label dice + coarse-only dict + macro_dice. Memory bounded: per-label passes are kept as small T³ grids and stitched to a full-size native one label at a time (never n_labels full volumes at once).
- CLI (infer_cli.py): --labels 1,2,5|all + --batch-size; multi-label branch prints per-label + macro dice.
- VERIFIED: 10/10 tests (test_infer_nifti.py) incl. batched multi-label == per-label single-label loop + small-wins combine, and --labels all id resolution + per-label dice. Real GPU run (checkpoint 2026-08-11_40, s0000 target / s0001 context, labels bladder=1 + gluteus_max_left=2, batch 2): label 1 dice 0.8546 (IDENTICAL to the prior single-organ bladder run) + label 2 0.7667, macro 0.8106; output uint8 ids {0,1,2} on the target grid (shape/affine/orientation match). Re-synced env bundle (scripts/sync_patchset_env.sh) so patchset-infer exposes the new flags.

## 2026-08-13 — infer_nifti: propagate organ names/colors (Caret LabelTable) to predictions
- FINDING: TotalSegmentator masks (e.g. total_seg_total.nii.gz) store organ names + RGBA colors INSIDE the .nii.gz, as a NIfTI header extension (code 0) holding a Caret `<LabelTable>` (Key->name+color, 117 entries). Not a sidecar. ITK-SNAP/MRIcroGL/Workbench read it. Our multi-label predictions were dropping it (fresh Nifti1Image → 0 extensions → viewer shows bare ids).
- FIX (infer_nifti.py): in multi-label mode, attach a LabelTable to the output subset to the segmented ids. Source preference: the target GT's table if --gt has one, else the first context mask that carries one (helpers _read_caret_label_table / _caret_label_names / _subset_caret_label_table / _output_label_table). predict_nifti now also returns "label_names" {id:name}; infer_cli prints the table + per-label dice with names. Single-organ (binary) path unchanged.
- Note: ElementTree re-serializes CDATA as plain element text (`<Label ...>spleen</Label>`) — Caret/ITK-SNAP parse either form; colors preserved.
- VERIFIED: 13/13 tests (subset-carried, GT-preferred, context-fallback). Real 117-label table parses (spleen/liver/heart), subsets to [1,5,51], round-trips nibabel save/load, dropped ids absent. Real GPU run (labels 1,2; GT without table) → output .nii.gz carries {1:urinary_bladder,2:gluteus_maximus_left} + colors via context fallback; CLI prints names. Env bundle re-synced.

## 2026-08-13 — infer_nifti: LabelTable subset must preserve atlas structure (byte-surgery)
- BUG: multi-label predictions opened as a plain mask but NOT as an atlas in the viewer, unlike the source GT/context mask. Cause: _subset_caret_label_table re-serialized the Caret extension with ElementTree, which dropped the `<?xml?>` declaration, CDATA wrapping, and — critically — `<VolumeType><![CDATA[Label]]></VolumeType>` (the marker that makes a viewer treat the volume as a label/atlas). Same shape/dtype/intent as GT, only the extension XML differed.
- FIX: _subset_caret_label_table now does surgical byte deletion (regex removes only the non-kept `<Label ...>...</Label>` entries) leaving the rest of the original extension bytes intact — output is viewer-identical to the source atlas minus the dropped labels. No ElementTree round-trip. (_caret_label_names still uses ET for READ-only parsing.)
- Repaired the stale on-disk ANALYSIS_patchset/total_seg_total_00_pred.nii.gz in place (re-subset the atlas table to its 72 present ids); decl + VolumeType + names now present.
- VERIFIED: 14/14 tests (added test_multilabel_label_table_preserves_atlas_structure asserting decl/VolumeType/CDATA kept, dropped keys gone). Real GT subset to [1,5,51] keeps all atlas markers. Env bundle re-synced.

## 2026-08-13 — self_context synth_masks: purely-synthetic ellipsoid target labels
- NEW mode under data.self_context: when self-context fires, with prob `synth_masks.p` replace the real target label with a PURELY SYNTHETIC mask — a random rotated 3D ellipsoid pasted on the real CT's body — giving a geometric in-context task with no real anatomy. The image is untouched (real CT); only the label is synthetic, then the existing self-context clone+augment machinery produces the K contexts. Config: `data.self_context.synth_masks.{p, ellipse_min_mm=1, ellipse_max_mm=50, body_hu=-400}` (configs/experiment/3d/dataset/totalseg.yaml). synth_masks.p is SPLIT-SPECIFIC (p.train / p.eval, scalar = both), resolved per split in common._self_context via the new _split_scalar helper (also used for self_context.p) — so you can pure-synth-train (self_context.p.train=1 + synth_masks.p.train=1) and still eval on real cross-subject contexts, or run a synth eval (needs self_context.p.eval>0 too). Pure-synth training = self_context.p.train=1.0 + synth_masks.p.train=1.0.
- Generator (src/totalseg_dataloader_incontext.make_ellipsoid_label): per-axis radii ~ U(min_mm,max_mm) → voxels via item spacing; random ZYX-euler rotation; centroid rejection-sampled inside body (image > body_hu mapped to normalised CT units via CT_MEAN/CT_STD). Rasterized only within a local bbox (cheap). Never empty (centroid voxel always set → guards sub-voxel radius at coarse spacing). rng is `random` or the eval-seeded Random (determinism preserved).
- coords: item["synth_coord"] (3,) = the object's anatomical coords.npy value at its centroid. O(1) — mmaps coords.npy and reads ONE voxel, mapping the item-grid centroid back to native via crop geometry (use_crop: starts/crop_sizes/out_sizes/pad_lo) or shape ratio (no-crop). collate stacks it only when every item in the batch is synth.
- Per-sample table (evaluate.build_sample_table): two new columns populated only for synth samples (empty otherwise, backward-compatible). `synth_size` = mean ellipsoid diameter mm (from item["synth_radii_mm"], the sampled per-axis radii now returned by make_ellipsoid_label — generative size, independent of body clipping); `synth_coord` = "x,y,z" anatomical-coords string. Threaded batch->case in evaluate_classes.
- DATALOADING COST (measured, 128³, .venv_nero): ellipse gen adds ~7.4 ms/item typical (U(1,50)mm@1.5mm), ~0.5 ms worst-case small, ~52 ms worst-case (50mm@1mm, 100³ local box); coords mmap+read ~0.5 ms. All negligible vs the ~hundreds-of-ms/item the real-context native ct.npy crop loads already cost (self_context still loads then discards K real contexts — pre-existing, dominant; a future pure-synth optimization could skip that). Verdict: keep coords mapping (0.5 ms is free).
- VERIFIED: tests/test_synth_ellipsoid.py 5/5 (shape/dtype/non-empty + radii returned, centroid-in-body, radii scale with spacing, seed-deterministic, tiny-radius). End-to-end use_crop DataLoader (num_workers=2): label_name=synth_ellipse, synth_radii_mm + synth_coord (B,3) mapped through crop geom, contexts stacked; build_sample_table emits synth_size (mean diameter mm) + synth_coord columns (18-col schema, row-length match).

## 2026-08-13 — fix: pure-synth val Dice was nan (synth_ellipse dropped by per-class aggregation)
- BUG: training with self_context.synth_masks on the val split (self_context.p.eval=1 + synth_masks.p.eval=1) gave val_dice=nan every epoch while train_dice grew. Because synth relabels every target label_name='synth_ellipse', which is NOT one of the requested benchmark val classes. evaluate_classes built summary rows by iterating only the REQUESTED `classes`, so every class got a {'error':'no samples'} row (no mean_dice) and all synth cases in cases_by_class['synth_ellipse'] were dropped. validate_mean then had zero rows with 'mean_dice' -> nan. (train_epoch computes dice on the batch ungrouped, so it was unaffected.) Consequence: val_dice>best is never True (nan), so best.pt is NEVER saved for a pure-synth run — rerun after the fix to get a checkpoint.
- FIX (evaluate.evaluate_classes): after summarizing the requested classes, also summarize any EXTRA class keys that appeared in cases_by_class but weren't requested (e.g. 'synth_ellipse'). Backward-compatible: normal eval has every label_name in `classes`, so `extra` is empty and rows are unchanged. Now a pure-synth val reports val/dice/synth_ellipse and val_dice = that; a mixed eval (synth_masks.p.eval<1) averages the real-class rows + the synth_ellipse row.
- VERIFIED: drove the real evaluate_classes with a stub model over a synth val loader (all targets synth, requested classes liver/spleen): val_dice=0.0295 non-nan, 19 synth cases summarized (was nan pre-fix). Synth per-sample table columns (synth_size/synth_coord) flow through since the synth cases now reach all_cases.

## 2026-08-13 — synth mask: generic 'synth' name + shape in the detail column
- Renamed the synthetic label_name 'synth_ellipse' -> 'synth' (generic, forward-compatible for other shapes). The per-sample `detail` column carries the shape: "ellipse <rx>x<ry>x<rz>mm @(<x>,<y>,<z>)" (per-axis radii mm + anatomical coords), built in evaluate_classes from batch synth_radii_mm/synth_coord (those batch/collate keys stay). KEPT a numeric `synth_size` column = mean diameter (mm) for dice-vs-size plotting; dropped the free-text synth_coord column (coords now in detail). Table schema: 17 cols.
- VERIFIED: real evaluate_classes over a synth val loader -> class='synth', detail='ellipse 22x27x14mm @(34,-57,-380)', synth_size=42.1, val_dice=0.0295 (non-nan); build_sample_table row length == schema. test_synth_ellipsoid.py 5/5.

## 2026-08-13 — synth radii/coords logged per-item in mixed synth+real eval batches
- Collate `incontext_collate_fn` gated `synth_radii_mm`/`synth_coord` behind `all(... in b for b in batch)`, so any batch mixing synth + real samples (the norm under self_context.synth_masks.p.eval<1) dropped BOTH keys and the detail/coord columns went empty (only fully-synth batches logged — e.g. 8/617 synth rows in run u4bqyxww). Switched to `any(...)` + per-item NaN-pad: real rows -> NaN, synth-without-coords.npy -> NaN coord. `.float()` cast avoids float64 radii vs NaN-placeholder dtype clash. all-real batches still add no keys.
- evaluate.py per-sample detail now guards on `not torch.isnan(batch[...][i,0])` instead of key presence, so real rows keep empty detail and synth rows log "ellipse ...mm @(x,y,z)".
- VERIFIED: mixed batch [real, synth+coord, synth-no-coord] -> radii NaN/valid/valid, coord NaN/valid/NaN; per-item guard logs detail for items 1&2, @coord only for item 1; all-real batch adds no keys.

## 2026-08-16 — unify independent per-volume geometry under `per_image`
- Removed the dead `instance_elastic` block from `apply_task_aug` (src/augmentations.py): it was read via `getattr(cfg, "instance_elastic", None)` and never declared in any config → always a no-op. Its role (independent per-volume elastic on the real-context task) is now served by the single `apply_per_image_aug` path.
- Real-context aug path (src/totalseg_dataloader_incontext.py) now applies `apply_per_image_aug` per volume (target + all K contexts) using `aug_cfg.per_image`, AFTER the shared `apply_task_aug` and BEFORE `apply_intensity_aug`. So `per_image` (flip/affine/elastic, independent per volume) is now the one knob for independent geometric jitter, shared by both the real-context and self_context paths.
- Backward-compatible: `per_image` defaults to all p=0 (configs/augmentations/nnunet.yaml), so existing runs are a bit-for-bit no-op. Raising `per_image.affine.p` etc. now also reposes volumes (incl. the target) in the ordinary real-context path, not just self_context clones. Updated the per_image config comment accordingly.

## 2026-08-16 — unify intensity aug order across CPU + GPU backends
- Made `apply_intensity_aug` (src/augmentations.py, per-item CPU) and `_batched_intensity` (src/gpu_augment.py, batched GPU) apply the SAME op sequence, following the physical image-formation chain:
  `GIN → bias_field → brightness/contrast → gamma → inverted-gamma → sharpness → gaussian_noise → gaussian_blur → simulate_low_resolution`.
  Noise sits BEFORE blur (deliberate) so blur correlates it → reconstructed-CT noise texture (nnUNet convention), not white noise on the final image.
- Closed op-set gaps so both backends have the full set: added `sharpness` (unsharp mask) to the CPU path; added `bias_field` (new batched helper `_batched_bias_field`) and the inverted-gamma pass to the GPU path; moved GPU GIN from last → first. Also ported `brightness_contrast.preserve_range` (clip to per-volume min/max) and `gamma.retain_stats` (rescale to per-volume mean/std) to the GPU path so both honor the same config flags.
- Rationale: bias field is a multiplicative acquisition field — it must modulate the signal BEFORE the window transforms and degradations (CPU previously applied it dead-last, after noise; GPU didn't have it at all).
- KNOWN residual CPU/GPU divergences (pre-existing, NOT order-related, left as-is): GPU blur sigma and low-res scale are one draw per batched call (per-volume kernels/output-shapes aren't batchable); RNG streams differ (per-item `random` vs batched torch.Generator) so per-sample outputs are not bit-identical — only the op sequence + semantics are equivalent.
- VERIFIED (.venv_nero): whitebox source-order check confirms both functions reference ops in the identical canonical sequence; both run end-to-end on a full all-ops-p=1 config with finite, in-range, changed output (incl. new sharpness/bias/inverted-gamma). tests/test_gin_aug.py + test_gpu_augment.py: 21 passed, 1 failed (test_gin_ipa_run_on_cuda — Blackwell sm_120 vs .venv_nero ≤sm_90 arch mismatch, environmental, unrelated).

## 2026-08-16 — fix: 42_reg_to_all.yaml intensity keys were albumentations-style (silently unread) + sharpness crash
- The `augmentations.intensity` block in configs/experiment/3d/experiment/42_reg_to_all.yaml used albumentations-style key names the aug code never reads. Base preset is `nnunet` (via `override /augmentations: nnunet` in 30_colipri_encoder.yaml); Hydra deep-merges 42's keys, so the mis-named keys merged in as extra IGNORED keys and the real magnitudes silently fell back to nnunet defaults:
  - `brightness_limit: 0.2` (unread) → code read `brightness` = 0.0 (base) → brightness aug did NOTHING
  - `contrast_limit: 0.2` (unread) → `contrast_range` = [0.75,1.25] (base)
  - `sigma_min/sigma_max` (unread) → `sigma_range` = [0.5,1.0] (base)
  - `var_limit: 0.01` (unread) → `max_std` = 0.316 (base) → noise ~3× stronger std than intended
  - `sharpness.alpha_min/alpha_max` (unread) + NO `factor` → runtime ConfigAttributeError. Newly LIVE after the 2026-08-16 CPU-sharpness addition (CPU dataloader previously ignored sharpness so the missing key was never touched; GPU path already read sc.factor).
- Also inherited-active but not listed in 42: `gamma` (p=0.3), `simulate_low_resolution` (p=0.25) from nnunet base.
- FIX: rewrote 42's intensity block to the code's schema — `brightness: 0.2`, `contrast_range: [0.8, 1.2]`, `sigma_range: [0.1, 1.0]`, `gaussian_noise.max_std: 0.1` (from var_limit 0.01 → std √0.01), `sharpness.factor: 0.5` (code has no alpha range; single unsharp weight). Noise=variance and sharpness-factor value were user-confirmed.
- VERIFIED (.venv_nero): OmegaConf merge of nnunet base + fixed 42 block → code reads the intended values; apply_intensity_aug with all ops forced p=1 (incl sharpness) runs finite, no missing-key crash.

## 2026-08-16 — add diffeomorphic `deform` op (SVF + scaling-and-squaring) as elastic upgrade
- Motivation: inspected TotalSeg inter-subject deformation from GT labels (all subjects 1.5mm iso; experiments/3d/deform_stats/estimate.py). Findings: task scale [0.7,1.4] ≈ real ±1sd (0.72–1.38); rotation ±30° (std 17°) ≈ real per-axis std ~10–18° (aorta/kidney/bladder outliers are PCA-axis/anisotropy artifacts, not real tumbling); elastic α=0.12 (~11.5mm) ≈ crude 11mm constellation residual. Conclusion: magnitudes are roughly calibrated, but the *elastic model itself* is weak — one coarse Gaussian field added straight to the grid + `.clamp` can FOLD (negative Jacobian) → invalid warps and torn masks.
- New op `deform`: samples a smooth stationary velocity field on a coarse grid (`control_spacing`), upsamples, then integrates by scaling-and-squaring (`num_steps`, VoxelMorph VecInt) → GUARANTEED diffeomorphic (no folding, masks stay valid). `max_disp` = velocity std in normalized grid units, inherits elastic.alpha calibration (~0.12 ≈ 11.5mm @128³/1.5mm).
- Shared helper `_svf_displacement(shape, control_spacing, max_disp, num_steps, generator, device)` in src/augmentations.py (device/generator-aware so CPU + GPU call the identical code). Wired into `apply_task_aug` (one shared field across the K+1 task), `apply_per_image_aug` (independent per volume), `apply_synth_aug`, and the GPU `_geometric` (per-group) in src/gpu_augment.py.
- Config: added `deform` block (p=0 default → no-op) under task/per_image/synth in configs/augmentations/nnunet.yaml. Kept legacy `elastic` for back-compat; deform supersedes it (opt in: deform.p>0, elastic.p=0).
- Integration uses `padding_mode="border"` and NO in-loop clamp (clamping the composition would distort the diffeomorphism).
- Test: tests/test_deform_svf.py — Jacobian determinant strictly positive everywhere (interior) across seeds/magnitudes (diffeomorphism); contrast test shows legacy elastic folds at comparable magnitude; determinism; shared-field-across-task + mask-stays-binary; GPU `_geometric` deform branch runs. 5 passed. Regression: test_gpu_augment.py 16 passed, 1 pre-existing Blackwell CUDA failure (unrelated).
- NOT done (separate step-2 track): data-driven SDM — sample deformations from a PCA model of real fields (coords.npy / pairwise registration) instead of white-noise velocity.

## 2026-08-16 — deform: assess vs real inter-case deformation, set max_disp=0.15, example plot
- Assessed SVF `deform` aug against REAL inter-case nonrigid deformation of GT labels (experiments/3d/deform_stats/assess_svf.py, GPU via .venv_blackwell). Method: register ~24 same-organ case pairs (compact organs) — affine (removes pose/scale/rot = task.affine analog) then a diffeomorphic SVF fit on the aug's coarse control grid — and measure the fitted displacement in the SAME normalized grid units as `max_disp`.
- Result: real inter-case nonrigid deform RMS median 0.14 (p10–p90 0.10–0.20), p95 0.28. Aug max_disp=0.12 → RMS 0.11 / p95 0.205; max_disp=0.24 → RMS 0.21 / p95 0.374. → the old default 0.12 UNDERSHOOTS real by ~25%. Set default `max_disp` 0.12→0.15 (task + per_image) in configs/augmentations/nnunet.yaml to match real median. Jacobian character overlaps (aug 0.12: 0.09–7.9× local vol-change; real ~0.1–6.7×). num_steps=6 stays strictly diffeomorphic at ≤0.15; raise to 7–8 if max_disp≥0.2 (at 0.24 a few voxels fold).
- Example plot: experiments/3d/deform_stats/plot_examples.py → deform_examples.png (liver/spleen/kidney/aorta: original CT+GT contour | SVF-deformed +orig contour overlay | warped grid).
- FINDING from the plot (not yet fixed): `control_spacing` is in VOXELS, so correlation length is resolution-DEPENDENT. Calibration fit real deform at ~6 control points (R=48/cs=8); at a 128³ crop cs=8 gives 16 control points → deformation is too HIGH-FREQUENCY / rough vs smooth real inter-case deform (plot rendered at cs≈21=128/6 to match). Recommend reparameterizing `control_spacing`(voxels) → `control_points`(count, e.g. 6) for resolution invariance. max_disp (normalized units) IS resolution-invariant, so its 0.15 calibration is unaffected.

## 2026-08-16 — deform: reparameterize control_spacing(voxels) → control_points(count), resolution-invariant
- We take 128³ crops at various spacings (1–4mm), so smoothness must not depend on the mm resolution. `control_spacing` (voxels) tied correlation length to the crop's voxel count; replaced with `control_points` (nodes per axis, a COUNT) in `_svf_displacement` (src/augmentations.py) → correlation length is a fixed fraction of the crop, resolution-invariant. max_disp is normalized so it was already invariant (and bigger FOVs at coarse res naturally hold bigger-scale deformation, so fixed-normalized amplitude is the right invariant).
- Set default `control_points: 6` (task/per_image; = the calibration basis, correlation length ~1/6 of crop) and 8 (synth, more local). Updated call sites in src/augmentations.py (3) + src/gpu_augment.py, config nnunet.yaml, tests/test_deform_svf.py, and the two analysis scripts.
- Re-ran assess_svf.py with control_points=6: identical to before (RMS med real 0.14; aug 0.12→0.11, 0.24→0.21) — confirms 6 control points on R=48 is the same 6³ velocity grid the old cs=8 produced, so the max_disp=0.15 calibration transfers unchanged. Regenerated deform_examples.png (control_points=6). Tests: 21 passed, 1 pre-existing Blackwell CUDA fail.

## 2026-08-16 — new augmentation preset configs/augmentations/calibrated.yaml
- Self-contained `# @package _global_` preset baking in the deformation study findings; select via `override /augmentations: calibrated`.
- Geometric: diffeomorphic SVF `deform` ENABLED (task.deform.p=0.3, control_points=6, max_disp=0.15, num_steps=6) with legacy `elastic` OFF; affine scale [0.70,1.40] + rotation ±30° kept (validated vs real inter-subject ±1sd 0.72–1.38 / per-axis std ~10–18°); task translation 0 (organ-centered crop). per_image OFF by default (pose-invariance lever, realistic magnitudes preset for when enabled). Intensity = nnUNet canonical-order chain. `p` values flagged as a training lever, not a calibration result.
- Verified: OmegaConf loads; apply_task_aug with deform forced on runs at 128³, finite, masks stay binary.

## MAISI synthetic pairs -> in-context dataloader (synth_gen_maisi source)
- `experiments/3d/synth_task_generation/gen_maisi_fast.py` — fast pipelined MAISI rflow-ct generator: threaded mask-prep prefetch + threaded QC/save (`--skip_qc`), writes one compressed `.npz` per pair (`ct` f16 z-scored HU via normalize_ct, `label` u8 MAISI-132 vocab, `spacing`). No-compile 11.1s/vol (1.42x over 15.8s baseline); torch.compile REJECTED (MetaTensor dynamo guard-thrash → 2x slower in real path; only helped the plain-tensor microbench). Batch>1 REJECTED (per-vol slower, OOM@B4). Remaining limiter: mask-prep GPU resample contends with the main loop on default CUDA stream.
- `bench_maisi_gen.py` — GPU microbench (compile x batch); `visualize_maisi_output.py` — per-pair PNG viz of a nii dir.
- `data/maisi_classes.py` — MAISI 125-class vocab vendored from NV-Generate-CTMR/configs/label_dict.json (idx->name; 200=body kept).
- `src/synth_gen_maisi_dataset.py` — `SynthGenMaisiDataset(TotalSegInContextDataset)`: native-only, MAISI vocab standalone (no TotalSeg remap). Reads the flat `.npz` dir (one file = one subject). Overrides `_get_subjects` (hash train/val split), `_load_or_build_cache` (scan npz labels -> MAISI names + spacing), `_get_spacing`, `_load` (pre-normalised ct + binary mask==maisi_id, resize only if image_size != native). Reuses base context sampling / class-balancing / aug / collate.
- `experiments/3d/common.py` — build_dataset branch for `data.source=synth_gen_maisi` (needs `paths.synth_gen_maisi`). Smoke-tested: 128³ items, 108 non-empty classes, class-balanced.

### synth_gen_maisi: use_crop + effective-spacing fix + dataset preset
- Fixed reported spacing: resize path (use_crop=false) now reports native*shape/image_size (e.g. 384x384x512 -> 128^3 = [4.5,4.5,6.0]mm), not native 1.5. Earlier it mislabelled downsized whole-body crops as 1.5mm.
- `use_crop=true` implemented in SynthGenMaisiDataset: organ-centred native crop of extent image_size*crop_spacing_mm resampled to image_size^3 -> true isotropic crop_spacing_mm/voxel (verified [1.5,1.5,1.5]). npz-based bbox cache `_maisi_bbox_for_subject` (per-slice bincount centroids, parallel). Reuses module-level organ_crop_arrays/place_image/place_label/resample_binary.
- Wired use_crop/crop_spacing_mm/crop_jitter/mask_downsample + defer_aug_to_gpu through common.build_dataset's synth_gen_maisi branch.
- Added `configs/experiment/3d/dataset/synth_gen_maisi.yaml` (paths.synth_gen_maisi + data block; `dataset=synth_gen_maisi`). train_classes/val_classes MUST be MAISI names or "all" (TotalSeg names won't match). Plot: `plot_dataset_items.py dataset=synth_gen_maisi data.use_crop=true data.crop_spacing_mm=1.5`.

### on-the-fly latent-bank feasibility bench (bench_decode_sdedit.py)
- Goal: can we precompute a MAISI latent bank and cheaply decode crops on-the-fly during training (vs baking an image bank)? `experiments/3d/synth_task_generation/bench_decode_sdedit.py` measures the ONLINE cost (VAE decode + K-step SDEdit re-noise/denoise), reusing gen_maisi_fast.build_args + load_image_models.
- FINDING — decode is the wall, ~825 ms per 128^3 crop (latent 4x32^3), and it is a COMPUTE-BOUND HARD FLOOR: unchanged by num_splits (4→1), cudnn.benchmark, channels_last_3d, or batching (827 ms/item at B=1..8). cuDNN warns it falls back off the V8 path for these large non-batch-splittable 3D convs. Decode scales ~cubically: 64^3=220ms, 128^3=825ms, 192^3=2800ms, 256^3 OOMs single-shot (needs the sliding-window tiling the real pipeline uses).
- FINDING — the diffusion loop is cheap at 32^3 latent: 55 ms/step; full 30-step=1644 ms; SDEdit K=2=113ms / K=3=165ms / K=5=275ms. So replacing the 30-step gen loop with a K=2–5 SDEdit appearance-refresh saves ~1.4s and is nearly free vs decode.
- VERDICT — the "store latents, decode cheaply per item" hypothesis is FALSE: per-item online cost ≈ SDEdit(K)+decode ≈ 0.95–1.1 s, ~85% of it decode, un-amortisable. At batch16×(1 tgt+1 ctx)=32 decodes/step ⇒ ~27 s/step. Not viable as a training hot path. The latent bank still buys 16× storage + cheap SDEdit appearance aug, but NOT a cheap render — the render is the cost whether latents are cached or regenerated.
- Recommended use: (1) offline IMAGE bank (current npz) + image-space aug remains the pragmatic default; (2) if extra appearance diversity wanted, a background GPU worker runs SDEdit(K)+decode (~1 crop/s) into a reused ring buffer that training samples over many steps — decode paid at refill rate, not per item. Faster decode would need a distilled/lighter VAE decoder or TensorRT (out of scope).

### pushing the VAE decode — torch.compile is a ~10x lever (bench_vae_decode_opt.py, bench_vae_compiled_batch.py)
- Context: VAE decode is 77% of whole-body gen and ~825ms/128³ crop, on a cuDNN slow path (warns "cuDNN cannot be used for large non-batch-splittable conv"). Goal = cheap decode of small crops for DIVERSE priors (realism not required), so 96³/compile/precision all fair game.
- **torch.compile(mode="default") routes the 3D convs to Triton `triton_convolution3d` kernels, bypassing the aten/cuDNN fallback → the big win.** Decode-only, num_splits=1, pure fp16 (model.half, no autocast):
    - 96³ (latent 24³): baseline 341ms → pure-fp16 322ms → **compile 78.9ms** (4.3x; 10x vs the original 825ms num_splits=4 path)
    - 128³ (latent 32³): baseline 751ms → pure-fp16 671ms → **compile 546ms** (only 1.4x — compile helps far more at 96³; 96³ is the sweet spot)
- Precision: pure-fp16 (model.half, no autocast) beats autocast-fp16 by ~5-15% and halves peak mem; bf16 slightly slower than fp16. TF32 on.
- **Batching is COUNTERPRODUCTIVE under compile**: compiled B=1 @96³ = 78.6ms/item (12.7 crops/s); B≥2 triggers a dynamic-batch recompile to a slower kernel (~245ms/item). Serial B=1 (4×78=314ms) beats B=4 batch (991ms). Use fixed B=1. (Without compile, batching gives only ~15%.) max-autotune crashed on an expandable_segments/cudagraph teardown bug — default mode already wins, skip it.
- **Implication for on-the-fly:** 96³ compiled decode ≈ 79ms/item, ~12.7 crops/s per GPU stream. A background decode worker → ~760 fresh crops/min feeding a reused ring buffer = ample diversity without per-step cost. Fresh-crop-via-SDEdit ≈ 79ms decode + ~50-110ms K-step loop ≈ 130-190ms. Still not a pure per-training-item hot path (batch16×2=32 ⇒ ~2.5s/step decode-only) but the ring-buffer pattern is now very comfortable. Bonus: whole-body offline gen decodes 8 fixed 80³-latent tiles — compiling those should cut the 98s decode too.

### 96³ crop quality inspection (inspect_crop_quality.py)
- From one bank mask (generated MAISI npz, 384³, body=200 present), cropped 96³ windows centred on 4 organs (lung lobes, liver, brain) × 3 seeds, full 30-step mask→CT gen each. Result: `results/synth_task_gen/crop_quality.png`.
- VERDICT: usable diverse priors. Coherent CT, correct+placed anatomy (brain convincing; lungs/chest good), genuine per-seed appearance diversity, reasonable mask↔image alignment, body envelope respected (outside-body→-1000). Weak spots (expected at 144mm FOV < MAISI's 256mm rec): graininess/blur + flat low-contrast soft tissue; LIVER/abdomen is the weakest (homogeneous, muddled). Failure mode = "soft/less-detailed", not wrong-anatomy — fine for a realism-optional prior. FOV is the fidelity lever (gen larger + downsample, or 128³).
- 30-step loop dominates at 24³ latent (~1.6s; ~19 it/s); decode is the cheap part once compiled ⇒ run the loop ONCE offline (latent bank), online = SDEdit(few steps)+compiled decode. combine_label_or must be a MONAI MetaTensor (binarize uses .as_tensor()).

### full-body latent → crop → decode: realism SOLVED (inspect_fullbody_crops.py)
- Fix for the crop-scale softness: don't condition the diffusion on a tiny crop (loses global context). Instead run the 30-step loop ONCE on the full-body latent (mask 384³ → latent 1,4,96,96,128; 14.6s), then crop the LATENT at 32³ windows (=128³ image) at random in-body locations and decode each. `results/synth_task_gen/fullbody_crops.png`.
- RESULT: dramatically more realistic than per-crop conditioning — crisp bone (skull/vertebrae/ribs/pelvis), textured soft-tissue organs (liver/kidney now have internal structure, not flat gray), correct organ placement + good mask alignment, 3D-coherent coronal views (continuous spine). NO visible seam/edge artifacts from decoding isolated 32³ latent windows (VAE decoder is local; interiors clean). Mechanism: diffusion sees global anatomy, we only crop at decode.
- Cost model VALIDATED for the latent-bank plan: full-body latent 14.6s once (offline, ~9MB @ 4x96x96x128 fp16) → MANY 128³ crops per latent = spatial diversity for free; each crop = one decode (~546ms compiled). Appearance diversity via different masks/seeds. Recommended arch = offline bank of full-body latents + online/background random latent-window crop + compiled decode → realistic (image,mask) pairs. combine_label_or must be MetaTensor.

### latent-space ops: VAE decode is coherent + ~equivariant (inspect_latent_ops.py)
- Tested spatial ops on a 32³ latent crop of a full-body MAISI latent, decoding each. `results/synth_task_gen/latent_ops_{coherence,equivariance}.png`.
- COHERENCE: all coherent CT — rot90/rot180, rot45(interp), hflip, avg_pool2(→64³, clean smoothed), zoom×1.5(→192³)/×0.6(→76³, correctly scaled). Decoder robust to spatial latent manipulation; nothing degenerates.
- EQUIVARIANCE (decode∘op vs op∘decode, HU MAE): rot90=38, rot45=57 (~2-3% of 2000-HU range); residuals concentrated at high-gradient edges, interiors near-identical. So latent-rotation ≈ image-rotation + small edge jitter.
- IMPLICATIONS: geometric aug (rotate/flip/zoom/pool) can be done IN LATENT SPACE + decode, no diffusion re-run → free on-the-fly aug on a latent bank. The small non-equivariance is desirable diversity (rot + appearance jitter). pool=cheap multi-scale/blur; zoom=scale aug. CAVEAT: latent op does NOT carry the mask — must apply the same transform to the label in image space to keep (image,mask) aligned. Ops used: torch.rot90/flip (dims 2,3=axial), scipy.ndimage.rotate axes=(1,2) per-channel for arbitrary angle, F.avg_pool3d, F.interpolate trilinear for zoom.

### real TotalSeg crops survive the MAISI VAE round-trip (encdec_totalseg_crops.py)
- Built the REAL dataloader (source=totalseg, use_crop=true, crop_spacing_mm=1.5, p_synth=0, 128³, aug off), encode→decode 8 crops through the MAISI image VAE (deterministic encode().mu → decode_stage_2_outputs, no diffusion/scale_factor). `results/synth_task_gen/encdec_totalseg.png`.
- Intensity bridge: dataloader z-scored (hu=z*505.8-167.3) → MAISI [0,1] via (hu+1000)/2000.
- RESULT: PSNR 31.8 dB, MAE 32.7 HU (soft-tissue 31 HU), latent (1,4,32,32,32) = 16× compression. Reconstructions faithful; error EDGE-LOCALIZED (bone/air boundaries), interiors clean. ⇒ MAISI latent space is valid for REAL data, not just synthetic → real+synthetic latents interchangeable for in-context learning in latent. Caveat: ~33 HU softening hits thin/high-contrast structure (cortical bone, small vessels) — the known thin-structure blind spot.
- Gotcha: patch_icl and NV-Generate-CTMR both ship a top-level `scripts` pkg → build the dataset first, then purge `scripts*` from sys.modules before importing the repo's scripts.utils_infer. Also hydra searchpath uses ${oc.env:PWD}/configs → run from patch_icl cwd, not the MAISI repo.

### MAISI VAE resolution capability on real crops (bench_vae_resolution.py)
- Round-trip real TotalSeg 128³ crops at crop_spacing_mm {1,2,3,4,5} (latent (4,32,32,32) for ALL — same voxel grid, so this isolates content-frequency/FOV, not latent size). `results/synth_task_gen/vae_resolution.png`.
- soft-tissue MAE: 1mm=35, 2mm=30, 3mm=37, 4mm=70, 5mm=61 HU; PSNR 31/29/27/28/28 dB. → 1-3mm = VAE comfort zone (MAISI trained ≤3mm in-plane; soft MAE ~30-37 HU, faithful). At 4-5mm soft-tissue error ~DOUBLES (60-70 HU): 128³@5mm spans 640mm whole-body cross-section (tiny structures/voxel + OOD), diff lights up all bone/organ edges though gross anatomy still recon'd. Overall PSNR noisy (dominated by growing air background); soft MAE is the honest metric.
- Caveats: n=6 DIFFERENT crops/spacing (variance, not same anatomy); coarse crops pack more structure/voxel so error rise is partly content density not pure OOD.
- IMPLICATION: for in-context learning in this latent, stay at 1-3mm crops (faithful encode/decode); >3mm discards soft-tissue detail. Consistent with the project's 1.0-1.5mm sweet-spot [[project_colipri_selfctx_ceiling]].

### latent perturbation keeps organs in place (inspect_latent_perturb.py)
- Real TotalSeg 128³ crops (+GT label) at 1-4mm, encode→ add Gaussian noise σ×latent-std (σ=0.5,1,2)→ decode, GT contour (cyan) overlaid. `results/synth_task_gen/latent_perturb.png`. crop latent std ~0.67-0.92.
- FINDING: latent additive noise perturbs APPEARANCE not GEOMETRY — organs stay under the GT contour. 0.5σ = safe mild texture jitter (label stays valid); 1.0σ = strong but structures/boundaries mostly in place; 2.0σ = breaks into coarse blobby noise (label invalid). ⇒ the latent is SPATIALLY LOCAL.
- Noise is COARSE/structured (~4-voxel blobs) because decoder upsamples 4× (1 latent voxel → 4³ image patch) = organ-scale texture jitter, not pixel noise.
- IMPLICATION: additive latent noise ≤~0.5-1σ = geometry-preserving appearance-diversity aug that keeps GT labels aligned — no diffusion re-run, NO mask transform (unlike the geometric latent ops which need the mask co-transformed). 0.5σ safe operating point.
- (cosmetic: b["label_name"] key not present → title shows '?'.)

### VAE latent is a poor TASK representation (compare_latent_vs_primus.py)
- Q: is the MAISI VAE latent useful for in-context seg, or just a good renderer input? In-context prototype-matching + fg-retrieval@1 over 40 TotalSeg tasks (use_crop 1.5mm, K=4), comparing vae32 (4×32³), vae16 (4×16³, =Primus grid), primus (CoLiPri ViT 864×16³), rawHU (1×32³) — features frozen, cosine prototypes from context crops classify the target.
- RESULT (fg-retrieval@1, the clean metric): primus 0.355 | vae32 0.022 | vae16 0.037 | rawHU 0.049. proto-Dice: primus 0.074 | vae32 0.016 | vae16 0.019 | rawHU 0.025. → VAE latent matches organs across crops AT CHANCE and BELOW raw intensity; Primus ~16× better. vae16≈vae32 ⇒ NOT a spatial-res issue, it's channel SEMANTICS (latent encodes appearance, not identity).
- CONCLUSION: reconstruction fidelity (32 dB) ⟂ task usefulness — confirmed empirically. The VAE latent is a great RENDERER input, a poor TASK representation; frozen-latent organ classes aren't linearly/metrically separable (info present but entangled with appearance). Keep roles separate: VAE = data generator/renderer (validated); Primus/discriminative features = the task. Learning IN the latent would need a TRAINABLE head to re-extract semantics, not the frozen latent as-is.
- Caveats: absolute Dice low for ALL (crude 2-prototype matching + extreme fg imbalance, median fg=0.2%); retrieval@1 (fg-only, subsampled) is the robust signal; ranking unambiguous. Relates to [[project_feature_sim_findings]] (retrieval_at1 methodology).

### encoder forward time: Primus vs MAISI VAE encode (bench_encoder_fwd.py)
- 128³ crop, native precision, warmup+sync. MAISI VAE encode: 482ms/crop (B=1), 469 (B=4), 8.8M params, out 4×32³. Primus ViT: 1.8ms/crop (B=1), 1.7 (B=4), 144.9M params, out 864×16³. → Primus ~270× FASTER despite 16× more params.
- Why the speed/param inversion: VAE encode = dense FULL-RES 3D convs (128³→32³) on the cuDNN slow path (uncompiled; compile would ~10×→~50ms, still ~30× slower than Primus). Primus = 1 strided patch-embed conv → coarse 16³ token grid + bf16 flash-attn over 4096 tokens = ~2ms.
- COMBINED VERDICT: frozen VAE latent loses to Primus on BOTH axes — fg-retrieval@1 0.022 vs 0.355 (16× worse task usefulness) AND 482 vs 1.8ms (270× slower). No dimension where the VAE latent wins as a task encoder. Its validated role = generator/renderer (realistic crops, cheap COMPILED decode ~79ms/96³, coherent latent aug); task = Primus/discriminative features.

### TensorRT decoder spike: 55ms/96³ (bench_vae_trt.py)
- MAISI's own path (config_trt.json: trt_compile(autoencoder, submodule='decoder')). Wrapped the VAE decoder with monai.networks.trt_compile, fp16. Node: thor / RTX A6000 (Ampere sm_86), torch 2.5.1+cu121.
- 96³ decode ladder: eager fp16 344ms → torch.compile 80ms (4.3×) → **TRT fp16 54.6ms** (6.3× eager, 1.5× compile, ±0.2ms) ⇒ ~18 crops/s. TRT is the floor for the STOCK decoder on this GPU.
- Cost/limits: engine build = 24 min one-time, cached to /tmp/maisi_trt_engines/maisi_dec_96.decoder.plan, STATIC shape (128³ needs its own 24-min build). FP8 N/A on Ampere (no FP8 tensor cores) → 55ms ~ceiling for stock decoder; sub-10ms needs a DISTILLED tiny 3D decoder (TAESD-style, realism-optional makes it the right tradeoff).
- ENV (additive, .venv_thor, reversible): tensorrt-cu12 10.13.3.9, polygraphy 0.53.4, onnx 1.22.0, cuda-python 12.9.7 (replaced cu13 cuda-bindings→12.9.7 to match cu121 torch; torch verified intact). trt_compile also needs polygraphy + cuda.cudart or it silently no-ops ("TensorRT and/or polygraphy not available"). tensorrt==10.13 meta pulls cu13 → use tensorrt-cu12 explicitly.

### MAISI mask bank + fully-random GMM (SynthSeg) — cheap generator alternative (prototype_gmm_synth.py)
- New direction: instead of the diffusion+VAE render, SAMPLE a mask from MAISI's candidate-mask bank and PAINT it with a fully-random per-label Gaussian mixture (Billot et al. SynthSeg). ~Free, label-perfect, max/OOD contrast diversity — fits "cover a large distribution, realism optional".
- Bank inspected: `datasets/all_masks_flexible_size_and_spacing_4000.zip` = 5164 full-body label maps (`*_133combined_aug_wbdm.nii.gz`, 14GB unzipped) across 21 source datasets (HNSCC 1183, TCIA_Colon 510, StonyBrook 438, TotalSegV2 297...). `candidate_masks_*.json` (4135 entries) carries dim/spacing/top-bottom_region_index/**label_list** per mask ⇒ class→mask index with NO volume scan. `all_anatomy_size_conditions.json` (1429) = organ_size vectors. 124 unique label ids (MAISI-132 vocab; 0=air, 200=body envelope, ~111 labels/mask). Flexible size/spacing (dim 121–1024, spacing 0.25–3.2mm xy / 0.1–6.8mm z) ⇒ resample-on-load.
- `prototype_gmm_synth.py`: reads a few masks straight from the zip (gzip.decompress → nib.from_bytes), organ-centred 128³ crop, `paint_gmm` = per-label N(μ~U(0,1), σ~U(0,.15)) → gaussian_filter blur → low-freq multiplicative bias field (4³ upsampled) → global noise → per-vol minmax. `results/synth_task_gen/gmm_prototype.png` (6 sources × 3 seeds).
- VERDICT: works, classic SynthSeg look. Perfect mask↔image alignment (painted from mask), strong per-seed contrast diversity (region dark→bright across seeds = OOD priors), plausible partial-volume edges/shading. Trait: single Gaussian/label ⇒ organ interiors texture-flat (fine for realism-optional; add per-label low-freq field later if wanted). Air(0) currently random per seed (moot — dataloader z-scores).
- DECISIONS (this session): intensity prior = fully-random SynthSeg; mask access = one-time convert to compact .npy bank (like convert_to_npy.py); prototype-first (done). NEXT if approved: `synth_gmm_maisi` data.source parallel to synth_gen_maisi_dataset.py — class-balanced (label_list index), on-the-fly GMM paint, reuse base context sampling/crop/collate.

### GMM-synth vs real dataloader timing — is it worth it? (bench_dataloader_gmm.py)
- Q: does GMM-painting MAISI-bank masks beat loading real TotalSeg crops? Head-to-head, SAME crop machinery (organ_crop_arrays/place_label) so delta = REAL-CT-read+resize vs GMM-paint. Both K=4 (5 vols/item), 128³, 1.5mm, aug off. Real = TotalSegInContextDataset(use_crop=true, p_synth=0, class_balanced), NFS. GMM = 24-mask bank on local /tmp, per-label μ/σ LUT paint (vectorised, no python loop) + blur+bias+noise. loki, 32 cores.
- workers=0 per-item: REAL 2443.7 ms | GMM 763.9 ms (load 57 + **paint 744** ms across 5 vols = ~149 ms/vol) → **GMM 3.2× faster**. Real cost = 5× NFS mmap of 20MB ct.npy+10MB label.npy.
- workers=8 throughput: REAL **1.1 items/s** | GMM **9.0 items/s** → **8.5×**. Real barely scales (NFS-IO-bound, 8 workers saturate bandwidth); GMM near-linear (CPU-bound paint, local masks).
- VERDICT: worth it. GMM avoids the dominant real cost (per-context 20MB CT read) → small mask-crop read + CPU paint that PARALLELISES across workers, whereas real use_crop is NFS-bound and stalls ~1 item/s. Paint (149ms/vol) is GMM's only cost and is the optimisable lever (→ GPU via existing GpuAugmentor, or fewer labels).
- CAVEATS: (1) GMM masks on local /tmp vs real CT on NFS — part of the win is storage locality; a production mask bank likely on NFS too, but mask crops are ~few MB (mmap window) vs 20MB CT, and compress well (int labels). (2) 4D HNSCC mask (512,512,223,1) needs np.squeeze. (3) 24-mask bank native uint8 = 33–200MB each (512³+) → full 5164-mask bank must be resized/compressed, not stored native (~867GB otherwise).

### GPU GMM intensity synthesis stage (src/gpu_gmm_intensity.py + tests/test_gmm_intensity.py)
- SynthSeg-style intensity stage per spec: labels [N,D,H,W] int64 → images [N,1,D,H,W] f32. Ids 0=air(det. mu=0,sigma=0), 1..K organ slots, K+1 container. Cohort-shared mu/sigma draw (mu~U(0,255), VAR~U(0,5)→sigma=sqrt; once per call) + per-voxel per-subject noise ⇒ one "scanner" across a support/query cohort. Two gathers + randn + FMA, vectorised over N, no python loops.
- Two-RNG contract: cohort_gen (mu/sigma) vs subject_gen (noise) as separate torch.Generators → reproduce/ablate each level independently (support/query = same cohort state, advanced subject). background_mode zero|component; optional clamp (off by default so downstream gamma/norm sees true values); slot ids are blueprint not class (intended domain-randomization; means unsorted, no min-separation).
- `pack_label_ids(labels, container_id)` bridge: arbitrary anatomical ids (MAISI 1..132) → dense 0/1..K/K+1 via LUT gather (container→K+1). Intensity stage itself is id-agnostic.
- Tests: 11 §9 checks pass — shapes/dtype, bg=0 exact, shared-GMM across N (per-id subject-mean std <0.2), differs across cohort seeds, bitwise determinism, noise-resample holds GMM, near-delta contrast (within-id std <=2.5≈sqrt5), id-coverage assert, component-bg, clamp, pack_label_ids. (CPU/CUDA portable.)
- Timing (A6000, 128³, K=50): 0.16 ms/VOL flat for N=1..32 (N=32→5.06ms/call). vs CPU scipy paint 149ms/vol ⇒ **~930× faster** → moving GMM paint to GPU collapses the dataloader-bench bottleneck (the 149ms/vol that made GMM CPU-bound). Intensity synth is never the bottleneck; the placement/crop loop is.

### GMM intensity: K→L rename + GPU sample plot (plot_gmm_gpu.py)
- Renamed the organ-slot count K→**L** in gpu_gmm_intensity.py + test (K is reserved for the in-context sample count). ids: 0=air, 1..L organs, L+1 container.
- `plot_gmm_gpu.py`: samples bank masks → organ-crop 128³ → pack_label_ids (body 200→container L+1) → synthesize_intensities under several cohort seeds. `results/synth_task_gen/gmm_gpu_samples.png` (5 masks × [mask | 2 subjects same scanner | 2 more scanners]).
- Confirms spec invariants visually: cohort-sharing (2 subjects/1 scanner near-identical, only faint noise differs), domain randomization (new scanners reassign contrast arbitrarily, organ bright→dark across draws, unsorted means/no class signal), near-piecewise-constant clean profile (σ≈√5), perfect mask alignment. L=31–42 slots/crop.

### GMM intra-label texture: var_max knob vs downstream bias/noise (plot_gmm_varmax.py, plot_gmm_texture.py)
- Observation: GPU GMM samples show no intra-label (within-region) variation. This is spec-INTENDED: var_max=5 → σ=√var ≤ 2.24 on 0–255 → near-piecewise-constant. Not a bug.
- `plot_gmm_varmax.py` (results/synth_task_gen/gmm_varmax.png): sweep var_max {5,50,200,1000,4000}, means fixed per row → within-slot jitter grows with σ. Confirms mechanism, BUT it's WHITE (uncorrelated) noise, not structured texture.
- DECISION (user): texture source = downstream bias+noise (spec default), keep GMM flat (var_max≈5). `plot_gmm_texture.py` (results/synth_task_gen/gmm_texture.png): flat GMM → multiplicative log-normal bias field (smooth correlated gradients) → blur-correlated Gaussian noise (structured grain) → visible, spatially-correlated intra-label texture. Mirrors src/gpu_augment.py _batched_bias_field + noise/blur. NB those stages clamp to CT_NORM range → need normalize bridge (GMM 0–255 → norm) before reusing _batched_intensity (spec §5: normalization downstream).

### MAISI mask-bank cohort sampling + metadata + perturbation methods (inspected NV-Generate-CTMR/scripts)
- GOAL: sample a "cohort" = K+1 SIMILAR masks from the bank (later: add mask randomization). MAISI already implements this end-to-end.
- BANK METADATA (candidate_masks_*.json == database.json, per mask): `label_list` (anatomy ids present — the "contains organ X" key); `top_region_index`/`bottom_region_index` (4-d one-hot body-region markers 0=head,1=chest/thorax,2=abdomen,3=pelvis/lower; mask SPANS regions [top_idx..bottom_idx]); `dim`,`spacing` (FOV/resolution); `pseudo_label_filename`. Separately all_anatomy_size_conditions.json = 10-d organ_size vector per mask (conditioning for the DDPM mask GENERATOR sample_mask.py; unused by bank query but a candidate extra similarity feature).
- COHORT SAMPLING (scripts/sample.py + find_masks.py): `find_masks(body_region, anatomy_list, spacing, output_size, check_spacing_and_output_size, db_json, folder)` → pool of masks that (a) contain ALL anatomy_list, (b) span ALL body_region (top_idx≤r≤bottom_idx), (c) tumor-free unless a tumor id requested, (d) optional exact spacing+dim. If pool<num_img → `find_closest_masks` relaxes to CLOSEST by a FOV/dim/spacing diff metric (won't upsample >128vox) + resamples. `select_mask(pool, num_img)` = random.shuffle + take num_img, each tagged `if_aug=True`. ⇒ "draw K+1 similar masks" = select_mask(find_masks(query), K+1); "similar" = shared anatomy + body-region span + spacing/FOV band.
- FAST PERTURBATION METHODS (scripts/augmentation.py, torch/MONAI, GPU-capable): erode3d/dilate3d (F.conv3d morphology); augmentation_body (RandZoom 0.99–1.01, tiny); augmentation_tumor_{bone,liver,lung,pancreas,colon}+augmentation_tumor_only (RandAffine/Rand3DElastic/RandZoom warps constrained inside organ via dilate/erode); augmentation() dispatcher (routes by tumor id present else body); remove_tumors, remap_labels. NB built-in body perturbation is very mild (zoom±1%); heavy warps are tumor-only → for domain-randomization we'd use stronger (our own src/gpu_augment.py _geometric affine/elastic/SVF deform is stronger and already GPU-batched).
- DESIGN (maps onto patch_icl in-context): precompute class→mask index + per-mask (region_span, spacing, dim, label_list) from JSON (no volume scan). Cohort = pick target class (balanced) → masks containing it → optional narrow by region-span/spacing band → draw K+1. Each mask → one (image,mask) via GPU GMM paint; cohort-shared GMM params (done) + similar masks = one "scanner + anatomy family". Later: per-mask erode/dilate/affine/elastic = within-cohort mask randomization.

### Cohort-sampling design FINALIZED (decisions)
- organ_size finding: all_anatomy_size_conditions.json (1429, 10-d [gallbladder,liver,stomach,pancreas,colon,+5 tumors], -1=absent, L1-skip-absent match) is NOT keyed to bank masks (4135) — it's the DDPM mask-generator condition set. Bank masks have no organ_size → must COMPUTE per mask.
- DECISION cohort key = TIGHTEST: contains target class ∧ same body-region span [top_idx,bottom_idx] ∧ spacing/FOV band ∧ kNN by size vector.
- DECISION size vector = RICHER ALL-LABEL: normalized voxel-count per present class (~132 ids), computed during .npy bank conversion (no extra pass); subsumes MAISI 10-d; genuine size match for ANY target (not just the 10 abdominal). kNN = L1 skip-absent, take K+1 nearest or sample from top-M.
- Precompute per mask (fold into bank convert): label_list+region_span+spacing+dim (from JSON, free) + all-label size vector (from volume). Then cohort = class-balanced target → filtered pool → size-kNN → K+1. Each → GPU GMM paint w/ cohort-shared params. Later: per-mask erode/dilate/affine/elastic = within-cohort mask randomization (methods in NV augmentation.py + our src/gpu_augment.py _geometric).

### Cohort sampler + compact mask bank BUILT (build_gmm_mask_bank.py, src/gmm_cohort_sampler.py)
- `build_gmm_mask_bank.py`: extract MAISI candidate masks from zip → save uint8 .npy + precompute cohort index (label_list, body-region span, spacing, dim, per-class centroids, normalized all-label size_vec len-256). Flags: --native (no resample, full detail — user chose; raw uint8 33–200MB/mask) | --spacing S (iso resample), --random (else source-spread), --max_masks, --workers. index.pkl = {maxid, spacing, entries[], size_mat (N,256)}.
- RUN: 500 random native masks → /nfs/.../ANALYSIS_20251122/data/gmm_bank (persistent; user-chosen path). ~60GB. (earlier 200@3mm build validated the pipeline.)
- `src/gmm_cohort_sampler.py` CohortSampler: fuses the 3 similarity axes into ONE weighted distance (region-span idx + FOV mm + spacing mm + size_vec L1) → kNN around a random anchor → k+1 masks (top_m_factor>1 = sample from top-M for stochasticity). No empty-pool edge cases. class→masks index from label_list (no volume scan). cohort_stats() reports tightness (span set, fov std, size L1).
- DESIGN DECISION (cohort GMM indexing): the K+1 cohort masks are DIFFERENT subjects sharing the MAISI ANATOMICAL id convention (liver=1…, body=200), so the cohort-shared GMM is indexed by the SHARED id space (L=maxid=200, drawn once per cohort via cohort_gen), NOT pack_label_ids per-mask. ⇒ an organ keeps a CONSISTENT shade across the cohort ("one scanner"); domain randomization is ACROSS cohorts (new cohort_gen draw). pack_label_ids (per-mask dense slots, independent means) is for the later single-mask randomization stage, not cohorts. `plot_cohort.py` validates: rows=cohorts (one scanner), cols=k+1 subjects, green=target contour.

### Cohort sampler VALIDATED on 500-mask NFS bank (plot_cohort.py)
- Bank built: 500 random native masks → /nfs/.../ANALYSIS_20251122/data/gmm_bank (index.pkl size_mat (500,256)), 0 failed. 124 classes have >= k+1=5 masks.
- plot_cohort.py (results/synth_task_gen/cohort_samples.png): 4 cohorts × 5 subjects, shared-id GMM per row. Confirms: (1) each cohort member contains the target class (green contour present in all), (2) shared-id GMM → per-structure consistent shade across a row = one scanner, rows differ = different scanners, (3) cohorts reasonably tight — size_L1 0.20–0.57, fov_std 8–25mm; some sources lack region index → span (-1,-1), handled by the weighted distance.
- Pipeline now complete through selection+paint: bank+index → CohortSampler (weighted-distance kNN) → organ-crop → shared-id GPU GMM. NEXT: per-mask randomization stage (erode/dilate/warp for within-cohort diversity) + wire into synth_gmm_maisi dataloader. Full 5164-mask native bank ≈ user will scale later.

### Minimal GMM-synth dataloader WIRED to train.py (src/synth_gmm_maisi_dataset.py)
- `SynthGmmMaisiDataset`: each item = a COHORT (CohortSampler, k+1 similar masks sharing a target class) → organ-crop each around the class → CPU flat shared-id GMM paint (mu[lab]+sd[lab]*noise, cohort-shared mu/sd once per item indexed by shared MAISI id, per-vol noise) → fixed 0-255→z bridge (mean128/std74, cohort-preserving). Returns the TotalSegInContextDataset item dict (image 1,T³ / label T³ / context_in K,1,T³ / context_out K,T³ / spacing / label_name=MAISI name / aug_mode) → plugs into incontext_collate_fn + train.py unchanged. Paint on CPU (no train.py edits; GPU src/gpu_gmm_intensity is the drop-in fast path later). eval_seed → deterministic per-idx for val.
- Wiring: common.build_dataset branch (source=synth_gmm_maisi, needs paths.gmm_bank) + make_eval_loader routes it through build_dataset (like omnisynth3d, per-item label_name grouping) + train.py main() resolves train/val classes from bank ids→MAISI names (bypasses _source_root). Config `configs/experiment/3d/dataset/synth_gmm_maisi.yaml` (data.epoch_length, data.gmm.{var_max,background_mode}); paths.gmm_bank added to cluster/nfs.yaml.
- Smoke: `python experiments/3d/train.py dataset=synth_gmm_maisi model=medverse train.checkpoint=random train.epochs=1 data.epoch_length=8 train.batch_size=2` → train epoch (4 steps) + val (grouped by MAISI name over 124 classes) + best.pt save, exit 0. Dice~0 (random init/1 epoch = plumbing only). Per-item ~749ms single-thread (native masks over NFS ×5 crops); parallelizes over workers.
- NEXT: real training run (pretrained medverse, full epoch_length, more masks); optionally move paint to GPU (synthesize_intensities in the loop) + downstream bias/noise texture via gpu_aug.

## synth_gmm_maisi: honor train_spacing_range
- `SynthGmmMaisiDataset.__getitem__` now unpacks the `(idx, spacing)` tuple from `SpacingBatchSampler` (same contract as `TotalSegInContextDataset`): when `data.train_spacing_range=[lo,hi]` is set, each train batch crops+reports one log-uniform physical spacing; plain int idx → fixed `crop_spacing_mm`.
- `_crop_multiclass` takes `crop_mm` as an arg; returned `item['spacing']` reflects the per-item spacing. Eval path (`make_eval_loader`) still uses fixed `crop_spacing_mm`.
- Added `train_spacing_range: null` to `configs/experiment/3d/dataset/synth_gmm_maisi.yaml`. Smoke: `ds[(0,2.5)]` → spacing [2.5,2.5,2.5], `ds[0]` → [1.5,1.5,1.5].

## Cross-source train/eval: `data.val` override
- Added `common.eval_cfg(cfg)`: overlays `data.*` with an optional `data.val` block on the EVAL path only (empty/absent → unchanged, val=train source). image_size/context_size inherited from data.* (model geometry must match); val-source keys win via OmegaConf.merge.
- `train.py`: extracted `_resolve_classes_for(cfg, classes_key)` (per-source class resolution: anchor/omni/gmm/chemotox/totalseg); `main()` builds `vcfg=eval_cfg(cfg)`, resolves val_classes from vcfg, train_classes from cfg (train source), and passes `vcfg` to `make_eval_loader`. So a run can TRAIN on synth_gmm_maisi and VALIDATE on totalseg/chemotox — the OOD signal; `in_train` flag reads False for every val class under a cross-source override.
- `validate_mean`/`evaluate_classes` unchanged (loader is prebuilt from vcfg; only other cfg read is cfg.eval.split, shared).
- Usage: `dataset=synth_gmm_maisi +data.val.source=totalseg +data.val.val_classes=[liver,spleen] +data.val.use_crop=true`. Commented example in dataset/synth_gmm_maisi.yaml.
- Smoke: vcfg.source=totalseg, image_size inherited [128,128,128], val_classes=[liver,spleen], train_classes=124 MAISI names.

## Cohort distance structure (analyze_cohort_distance.py)
- Inter-subject pairwise L1 per CohortSampler component over the 500-subject index (metadata only; safe mid-mask-rewrite). results/synth_task_gen/cohort_distance.png.
- WEIGHTED contribution to cohort selection: fov 50.9% (dominates), size 30.9%, span 14.9%, spacing 3.3%. fov leads despite w_fov=0.02 being smallest — its raw L1 is ~405mm vs single digits for the rest. The `size` vector (w=3.0, intended to lead) is only 2nd; spacing ~unused.
- Components moderately redundant: Spearman span/fov/size ρ≈0.4–0.5 (bigger FOV≈wider span≈more anatomy); spacing independent (ρ≈0) but negligible. ~2 effective axes.
- MDS(combined weighted d): one dense abdomen cluster (most MAISI sources abdominal CT) + sparse whole-body/pelvis arm; position strongly predicted by SOURCE dataset → kNN cohorts pull same-source/same-FOV masks ("one patient family" proxy, but FOV/source-driven not composition-driven).
- Takeaways: to make size lead, normalize each component before weighting; spacing droppable; rare-class cohorts inherently homogeneous. Open Q (diversity→training) needs a tight/current/random cohort ablation on OOD totalseg val.

## synth_gmm cohort knobs (config)
- Exposed CohortSampler params as `data.cohort.*` in dataset/synth_gmm_maisi.yaml → SynthGmmMaisiDataset(cohort=...) → CohortSampler(**cohort). Wired in common.build_dataset.
- Knobs: w_span/w_fov/w_spacing/w_size (distance weights), top_m_factor (diversity: 1=tight k+1 nearest, higher=wider neighbourhood), min_masks_per_class.
- Defaults preserved (1.0/0.02/0.3/3.0, top_m_factor=2). Override e.g. `data.cohort.top_m_factor=1 data.cohort.w_fov=0.0`. Smoke: overrides reach cs.w / cs.top_m.
- Enables the tight/current/random cohort-diversity ablation (see project_cohort_distance_structure).

## Size factor: air-contamination in size_vec (cohort distance)
- Ran analyze_cohort_distance.py with size-only weights (--w_size 1, others 0): cohort_distance_sizeonly.png / _sizeorgans.png.
- BUG: size_vec[0] = air/foreground ratio (build_gmm_mask_bank: counts/fg includes counts[0]=air), range 0–9.6, = 72% of the size L1. Only 28% is organ composition (cols 1+, which correctly sum to 1). So w_size is mostly a FOV/coverage proxy; true composition signal ≈ 9% of total cohort distance.
- Clean organ composition (--drop_air): size L1 mean 0.46 (max 1.1) — subjects fairly similar; MDS still clusters by body-region span + source dataset → composition is a proxy for region/protocol, not an independent axis.
- FIX (pending): zero size_mat[:,0] at load in CohortSampler (fixes current + full bank, no rebuild); optionally size_vec[0]=0 in build for future. Added --drop_air/--w_* CLI to the study script.

## Size factor: shared-core restriction (analyze_cohort_distance --common_frac)
- Strict "common to ALL" masks = only `body` (degenerate). Shared core grows w/ threshold: >=95% -> 7 classes (aorta/esophagus/lungs-lower/autochthons/body), >=75% -> 42 (abdominal core: +liver/spleen/pancreas/kidneys/IVC/adrenals/stomach).
- Restricting size to shared-core (+drop_air): size L1 mean 0.46(124)->0.28(42-core)->0.13(7-core). Removing rare-organ presence/absence cuts SOURCE clustering (panel f) but the body-region gradient (panel e) persists -> clean composition mostly tracks body-region COVERAGE, not an independent axis.
- Added --common_frac to study script (implies drop_air; freq over label_lists). Figures: cohort_distance_core075.png / _core095.png.
- Recommend CohortSampler restrict size distance to shared-core (>=frac, computed at LOAD from the actual bank — full 5164 bank core differs from 500-subset) + drop air; default frac 0.75. Pending wire-in as data.cohort.size_common_frac.

## CohortSampler: clean composition (size) distance — APPLIED
- Size distance was contaminated by two coverage proxies: size_vec[0]=air/fg (72% of raw L1) and size_vec[200]=body/fg (mean 64% of fg, 18% of composition L1; ≈1−organ-coverage → encodes annotation completeness).
- Exact per-mask size: counts=bincount(labels); fg=counts[1:].sum() (incl body 200); size_vec=counts/fg.
- FIX in CohortSampler.__init__: drop ids {0,200}; restrict to organs present in >= size_common_frac of masks (shared core computed AT LOAD from the actual bank → full 5164 gets its own core); renormalize kept columns per mask to sum 1 (relative organ composition). New param size_common_frac (default 0.75) exposed as data.cohort.size_common_frac.
- Verified: frac=0.75 -> 41 organ cols, rowsum=1, air+body zeroed; frac=0.9 -> 7 cols; config override reaches ds.cs.size_ncols. Works on current 500 bank + will apply to full bank (no rebuild).

## CohortSampler: size_mode fraction|volume (spacing correction)
- Q: does size use spacing? Raw voxel counts, but per-mask fg normalization makes it a VOLUME FRACTION (voxel-volume cancels in the ratio) → scale-INVARIANT. Consequence (correct): large-patient-large-FOV and thin-patient-small-FOV with same proportions get pulled together by the size term; fov only loosely compensates (Spearman 0.85 w/ body vol, but fov bbox is 56% air).
- Body physical volume spans 30× (3.3–96.9 L) — a real patient-size signal ignored by fraction mode.
- FIX: added size_mode. 'fraction' (default, current renorm-to-1 relative composition) | 'volume' (organ physical volume in L = counts·prod(spacing); scale-AWARE, corr 0.91 w/ body size → separates large/thin; big-organ-weighted: liver/colon/bowel/lungs=63% of L1). Both recoverable from index (fg=prod(dim)/(1+air_frac)); no rebuild.
- volume L1 ~5.6 L vs fraction ~0.86 → use smaller w_size (~0.5) in volume mode. Exposed as data.cohort.size_mode. Verified both modes load.

## Rename size->by_class_size + fraction/volume distance study
- Renamed the cohort distance component `size`->`by_class_size` everywhere: CohortSampler (w_by_class_size, by_class_size_common_frac, by_class_size_mode, by_class_size_mat, by_class_size_ncols, w['by_class_size']), config data.cohort.*, analyze_cohort_distance.py. Dataset forwards **cohort so config keys match new param names.
- analyze_cohort_distance.py now builds components straight from a CohortSampler (reflects the real cleaned by_class_size: air+body dropped, shared-core, mode). New flags --by_class_size_mode/--by_class_size_common_frac; figures cohort_distance_fraction.png / _volume.png.
- Contribution @ default w (span1/fov0.02/spacing0.3/bcs3): FRACTION -> fov 59.7%, by_class_size 18.9% (cleaned raw L1 0.86), span 17.5%, spacing 3.9%. VOLUME -> by_class_size 60.4% (raw L1 5.6 L), fov 29.2% (weight not retuned).
- Key: fraction mode by_class_size is INDEPENDENT of fov (Spearman ~0.34) but weak; volume mode is strong but REDUNDANT with fov (~0.63, both scale). => fraction: raise w to use composition as orthogonal lever; volume: lower w_fov (avoid double-counting scale) + w_by_class_size ~0.5.

## Per-cohort randomness dial (replaces top_m_factor)
- CohortSampler: removed `top_m_factor` (fixed global band width); added `randomness` = per-cohort tightness r in [0,1], drawn each cohort via new `_draw_r`. Mapping: M = round((k+1) + r*(pool-(k+1))), take M nearest, sample k+1 uniformly (anchor forced in). r=0 -> k+1 nearest (tight); r=1 -> uniform random over the whole class pool.
- `randomness` accepts: float (fixed), [lo,hi] (r~U per cohort), or {p_tight,tight,loose} bimodal ("some similar, some full random"; sweep p_tight for the diversity->OOD ablation).
- Config data.cohort: top_m_factor -> randomness (default 0.3). Distance-contribution comment refreshed to full-bank numbers (fov 60 / bcs 19 / span 17 / spacing 4).
- Verified on full bank (4135), largest class pool: within-cohort by_class_size L1 mean rises 0.173 (r=0) -> 0.490 (r=0.3) -> 0.704 (r=1). Bimodal p=.5 gives highest std 0.306 spanning [0.00,1.20] = tight AND random cohorts in one stream, which fixed floats can't. Full-bank config-param distance study: cohort_distance_fullbank.png (structure stable vs 500-subset; 41 bcs cols kept).

## Fix: cohort override loading + OmegaConf container
- `data.cohort.*` only exists when the dataset group file is merged: use `dataset=synth_gmm_maisi` (defaults-list selector), NOT `data.source=synth_gmm_maisi` (plain string override -> `Key 'cohort' is not in struct`). With the group loaded, `data.cohort.randomness=...` is a plain override (no `+`).
- common.build_dataset now passes cohort via OmegaConf.to_container(resolve=True): dict(cohort) only converts the top level, leaving a `randomness` DictConfig/ListConfig that slips past CohortSampler's isinstance(dict)/(list,tuple) checks. Now native dict/list reach _draw_r.

## Fix: drop degenerate-spacing masks in full bank
- Full bank has 6 masks with a degenerate spacing axis (3 all-zero + 3 near-zero ~3e-8) -> ZeroDivisionError / giant target_size in organ_crop_arrays. Not in the 500-subset.
- CohortSampler now filters entries at load with a 0.05mm physical floor (well below any real CT; keeps legit 0.1-0.25mm high-res), row-aligning size_mat. Dropped: m00733, m02017, m02883, m03711, m03870, m04070. 4135 -> 4129 usable masks.

## Cross-subject-only eval guard (drop_self_ctx)
- `data.self_context.p.eval=0` stops the intentional self-context probe, but the context sampler still self-clones the target for candidate-less classes (leakage-inflated, warned) -> stray self_ctx=True rows leak into the mean.
- evaluate_classes gains `drop_self_ctx` (default False): excludes self_ctx=True cases from the per-class summary rows only; the per-sample `cases`/wandb table still logs every sample (flagged). train.py validate_mean auto-enables it when `_self_context(eval_cfg.data,"val")` p==0 (also filters those rows out of the val/loss mean). p.eval>0 (probe) keeps them.
- eval.py path unaffected (default False). To get honest cross-subject numbers: run with `data.self_context.p.eval=0`.

## eval.py drift check: add raw_ct + self_context to _FIDELITY_KEYS
- eval.py restores only `arch` from the checkpoint; all data.* comes from the eval config. `raw_ct` (intensity normalization) and `self_context` (how K contexts are built; p.eval>0 = self-context leakage) were NOT in the drift check, so they mismatched training silently. Added both to _FIDELITY_KEYS.
- _warn_uninherited_data now to_container-normalizes the checkpoint-side value too (nested self_context compares by content, not DictConfig-vs-dict). Still warn-only, never halts.
- Reminder of what IS auto-restored: model weights + full arch.* (patchset3d, rebuilt from ckpt["arch"]); only weight-free eval.feat_norm / eval.primus_sidecar override it. Everything else (all data.*, eval.* harness knobs) is eval-config authoritative.

## synth_gmm_maisi: occupancy mask downsampling (match totalseg training)
- SynthGmmMaisiDataset resized the crop->grid multiclass label with plain NEAREST (_crop_multiclass), so thin/small organs vanished at large FOV — a train(synth)/eval(totalseg occupancy thr=0.1) mismatch, and small-object cross-subject Dice was the weakest bucket.
- Added `mask_downsample` (nearest|occupancy, default occupancy) + `mask_occupancy_thr` (default 0.1). New `_resample_multiclass`: per present-id area-pool to foreground fraction, assign argmax id clearing thr (small ids survive), with a non-empty guard that keeps the TARGET class's densest voxel if thresholding erased it (no dead empty-target items). Paint is drawn FROM the resampled map, so image<->mask stay consistent (image is painted from the label here, unlike real-CT totalseg).
- Threaded from common.build_dataset (synth_gmm_maisi branch) + config keys in dataset/synth_gmm_maisi.yaml (mask_downsample: occupancy, mask_occupancy_thr: 0.1). Startup print shows the mode.

## synth_gmm_maisi: class_balanced sampling toggle
- sample_cohort already picked the target class uniformly (implicitly balanced) but had no toggle. Added `class_balanced` (CohortSampler + SynthGmmMaisiDataset + data.class_balanced config, default true), mirroring totalseg.
- true = uniform over usable bank classes (rare organs as often as common); false = rng.choices weighted by #bank masks per class (natural anatomical-frequency prior). Weights built inline in sample_cohort (not cached) so the dataset's post-init self.cs.classes filter can't stale them.
- Verified over 4000 draws: balanced hits 125/125 classes (max/mean 1.6 ~ sampling noise); frequency prior 124/125 (max/mean 1.9). Difference is modest because full-body MAISI masks make most classes co-occur — the flag mainly affects the genuinely rare classes.

## configs/experiment/3d/experiment/43_synth_gmm.yaml (collision-free synth-GMM run)
- Replaces the long override string (experiment=42_reg_to_all dataset=synth_gmm_maisi +data.val.* data.self_context.p.eval=0 augmentations=calibrated arch.*). Inherits 35_colipri_enc_8_i_128 (NOT 42), overrides /dataset=synth_gmm_maisi + /augmentations=calibrated, bakes cohort.randomness=[0,1] + totalseg val override + encoder_frozen=false.
- Collisions fixed: 42 injected train_classes=balanced / val_classes=not_balanced / raw_ct=true / p_synth / synth_method / self_context (all totalseg-oriented, wrong for the bank). Inheriting 35 drops raw_ct + self_context entirely (val self_context.p.eval defaults to 0 => cross-subject, no override needed). The dataset group merges BEFORE the experiment, so 35 still clobbers train_classes/crop_spacing_mm/mask_occupancy_thr/class_balanced/max_ds_len_train -> 43's _self_ re-asserts the synth values.
- Silent bug caught: 35 chain sets max_ds_len_train=1000, which caps the synth train epoch to 1000 via RandomSampler(num_samples=min(max_len,len(ds))) despite epoch_length=10000 (10x fewer samples/epoch). 43 sets max_ds_len_train=null.
- full_attn=true / register_routed=false are already the 35 defaults (those two CLI flags were redundant). 35 overrides calibrated's task.affine/elastic, so 43 restores the calibrated magnitudes (affine 0.90-1.10 p0.2, elastic alpha0.12 grid8; deform survives from calibrated).
- HEADS-UP: 35 arch = l=2/e=512/h=512/a=4 (small transformer), NOT 42's l=4/e=768/h=3072/a=12. Inheriting 35 is the smaller model by design.

## GPU-realize synth-GMM dataloading (occupancy resample + paint moved to GPU) — 2026-08-19

- Problem: `experiment=43_synth_gmm` (mask_downsample=occupancy) was data-starved. Profiled `SynthGmmMaisiDataset.__getitem__` (T=128, K+1=2): occupancy resample = **15.6 s/item (99.4%)**; load+crop 21 ms, GMM paint 77 ms, nearest resample (ref) 450 ms. So moving PAINT to GPU buys ~0.5% — the whole cost is the occupancy area-pool.
- Root cause: occupancy area-pools EVERY present label (mean 40, up to 67) of the full native crop (mean 8.9 M vox, up to 33 M) in a per-class Python loop of CPU `F.interpolate(mode="area")`. Totalseg's occupancy is cheap only because it does 1 binary mask/volume; synth pays 40x because paint needs the whole multiclass map. (In `__getitem__` the multiclass map feeds ONLY paint — the supervised mask is `lab==cls`.)
- Fix (user chose full-GPU): new `gpu_realize` mode. Worker ships the NATIVE uint8 multiclass crop + placement geometry (out_sizes/pad_lo) + the cohort GMM draw (mu/sd) instead of a painted image; occupancy resample + SynthSeg paint run batched on GPU in the train loop. Files:
  - `src/synth_gmm_maisi_dataset.py`: `gpu_realize` + `gpu_realize_max_native` params; `_native_crop()` (ships native uint8, nearest pre-downsample above the cap to bound H2D/mem at large crop FOV); `__getitem__` GPU branch returns native payload dict.
  - `src/gpu_synth_realize.py` (NEW): `synth_gpu_collate_fn` (keeps variable-shape native crops as a list-of-lists, stacks the rest) + `_occupancy_to_grid` (verbatim GPU port of `_resample_multiclass`+`place_label`) + `SynthRealizer` (fills image/label/context_in/context_out on device; paint == `_paint`: `(mu[lab]+sd[lab]*noise-128)/74`).
  - `experiments/3d/common.py`: dataset gets `gpu_realize` (TRAIN split only — val is a real source via eval_cfg) + `gpu_realize_max_native`; `train_loader` selects `synth_gpu_collate_fn` when source=synth_gmm_maisi & gpu_realize.
  - `experiments/3d/train.py`: builds `SynthRealizer` next to `GpuAugmentor`; `train_epoch` calls it (`if "native_lbls" in batch`) BEFORE GpuAugmentor, so painted volumes flow through geo/intensity aug unchanged.
- Config: `data.gpu_realize: true`, `gpu_realize_max_native: 256` (2xT) in 43_synth_gmm.yaml; both added (defaults false/256) to configs/.../dataset/synth_gmm_maisi.yaml.
- Verified: occupancy CPU==GPU bit-exact on the same native crop (target vox 2951==2951); realizer output shapes correct, image finite (bg -1.73), labels non-empty; **45 ms/item on GPU vs 15,600 ms CPU (~346x)**. `gpu_realize_max_native=256` caps native side (seen 39–256) so H2D stays ~16 MB/member even at crop_spacing 4 mm. CPU path unchanged when gpu_realize=false (plot_dataset_items / eval still use it).

### Decouple GMM-paint label (nearest, image-like) from supervised mask (occupancy) — 2026-08-19

- Insight (user): mask_occupancy_thr is low (0.1) specifically to GROW the supervised target mask so small/thin objects survive downsampling — but the multiclass label that drives GMM painting does NOT need enlarging; it can be treated like an image (nearest). Previously both the CPU path and the GPU realizer derived the mask from an all-class occupancy argmax, coupling them and paying a ~40-class occupancy loop.
- Change: paint and mask now resize by role.
  - paint_lab: full multiclass, NEAREST (image-like, drives per-voxel `mu[lab]` shade only; no enlargement).
  - mask: target-class BINARY under mask_downsample — "occupancy" area-pools the target fraction + threshold (low thr enlarges, == totalseg resample_binary), "nearest" = paint==cls. Non-empty guard.
  - `SynthGmmMaisiDataset._resample_multiclass`/`_crop_multiclass` -> `_resample_paint_mask`/`_crop_paint_mask` (return (paint, mask)); `src/gpu_synth_realize._occupancy_to_grid` -> `_resample_member` (returns (paint, mask)); SynthRealizer gains `mask_downsample` (threaded from cfg.data in train.py).
- Verified: paint AND mask bit-exact CPU==GPU; small target (native 4943 vox) mask occupancy(thr0.1)=6435 vox > nearest 4398 (grows small objects); 2057 rim voxels where mask=1 but paint!=cls (intended partial-volume hard cases — paint is genuinely decoupled). Realizer GPU work dropped 45->12 ms/item (nearest + 1 binary area-pool vs 40-class loop). CPU fallback ~840 ms/item (was 15.6 s).

### plot_dataset_items auto-disables gpu_realize — 2026-08-19

- `plot_dataset_items.py experiment=43_synth_gmm` crashed with KeyError 'image' in incontext_collate_fn: 43 sets data.gpu_realize=true, so build_dataset(split=train) yields NATIVE crops (painted on GPU in the train loop), which the plot's CPU collate can't stack. (Also: top-level `gpu_realize=false` fails struct validation — the key is nested: `data.gpu_realize=false`.)
- Fix: plot_dataset_items now force-sets cfg.data.gpu_realize=False before build_dataset (it's a CPU viz tool that needs painted image/label items), so the plot works with no manual override. Added `from omegaconf import OmegaConf`. Confirmed: realize=CPU, saved results/3d/dataset_items.png.

### Modular composable configs (m1/e1/d1) + 48_abdomen_ceiling — 2026-08-20

- Refactored away from chain-inheriting experiment yamls: added three independently-swappable Hydra groups for the frozen-CoLiPri PatchSet3D + v2-dataloader abdomen run:
  - `configs/experiment/3d/dataset/d1.yaml` — abdomen TotalSeg, `loader_v2=true`, 2mm crops. Minimal: v2 ignores the v1 probe keys (self_context/synth/p_synth/use_crop/raw_ct/num_labels), so they're omitted.
  - `configs/experiment/3d/model/m1.yaml` — PatchSet3D transformer head + Muon recipe. Only `train.*` keys train.yaml does NOT define survive the group (epochs/batch_size/eval_every/grad_clip are shadowed by train.yaml `_self_` → set them in the experiment layer).
  - `configs/experiment/3d/encoder/e1.yaml` — NEW `encoder/` group; frozen Primus/CoLiPri (stage 8, native-grid, spacing-aware, bf16). Experiment adds it via `- /encoder: e1` (non-override adds must precede `override` entries in the defaults list).
  - `configs/experiment/3d/experiment/48_abdomen_ceiling.yaml` — composes d1/m1/e1 + `override /augmentations: calibrated`, self-contained (no dep on other experiments). Verified via `train.py experiment=48_abdomen_ceiling --cfg job`: arch/train/data match the in-flight 42-based run.
- Divergence from the in-flight run (by choice): 48 uses PURE calibrated intensity aug (lighter, no sharpness), not 42's merged heavier intensity chain — so it's a fresh clean baseline, not bit-comparable to the current wandb 48_abdomen_ceiling. Also sets `eval.crop_jitter=0` (centered eval crops) vs the run's null (T//4 jitter).
- Ceiling caveat baked into the file's header: v2 makes it pure cross-subject, but task flips/affine-rotation/deform still fire on GPU; zero their `p` for a stricter geometric ceiling.

### synth_gmm through the v2 dataloader (cohort hook) — 2026-08-21

- Plugged `data.source=synth_gmm_maisi` into the generic v2 engine (`src/incontext_dataset_v2.py`) so it shares the engine's aug path, eval-seed RNG, and `(idx, spacing)` handling with totalseg-v2. Gated on `data.loader_v2=true`; `loader_v2=false` keeps the standalone `SynthGmmMaisiDataset` path unchanged.
- The synth source can't use the engine's independent per-subject `load()` (it samples a K+1 cohort of similar masks jointly + paints them with ONE cohort-shared GMM draw). Solution = an optional **cohort hook**: a provider that implements `assemble_task(rng, crop_spacing_mm) -> item_dict` bypasses the engine's target+context sampling; the engine then owns only aug + per-item RNG + tuple-idx unpacking.
  - `InContextDataset`: detects `hasattr(provider, "assemble_task")` → `cohort_mode` (len from `provider.epoch_length`, no per-subject `samples`); `__getitem__` delegates to the hook. Extracted the inline aug block into `_augment_stacks` / `_aug_active` (shared by both paths; the load-based totalseg path now stacks-then-augments identically).
  - `SynthGmmMaisiDataset.__getitem__` split: rng/nrng derivation stays, cohort-sample+shared-GMM+crop/paint body extracted to `assemble(rng, nrng, crop_mm)` (pure refactor, one impl, two entry points).
  - `src/providers/synth_gmm.py` (NEW): `SynthGmmProvider` wraps a `SynthGmmMaisiDataset`, exposes `classes`/`epoch_length`/no-op `subjects_for`, and `assemble_task` derives `nrng` deterministically from the engine's `rng`.
  - `common.build_dataset`: `synth_gmm_maisi + loader_v2` builds provider+engine (aug_cfg/defer_aug/eval_seed like totalseg-v2); eval already routes via build_dataset; train_loader's gpu_realize collate switch still fires (provider emits `native_lbls` in that mode).
- Both output modes supported: CPU-paint items get engine task+intensity aug (deferred to GPU when `augmentations.gpu=true`); gpu_realize native payloads carry no `image` and pass through aug-free to the downstream `SynthRealizer`.
- Verified on the real GMM bank (4129 masks, 125 classes, T=64, K=3): CPU-paint via engine is byte-identical to a direct `assemble_task` call (faithful routing) + eval-seed deterministic + correct shapes; `(idx, spacing)` reaches the hook as crop spacing; gpu_realize payload passes through untouched with aug enabled; load-based totalseg path dict unchanged after the aug refactor.

### Cross-source train/val under loader_v2 (synth_gmm_v2 dataset group) — 2026-08-22
- Goal: "use a different dataset for train and val in the v2 dataloader, like experiment 43".
- Finding: no code change needed. The `data.val` overlay (`common.eval_cfg`) is source-agnostic and already flows through loader_v2: `train_loader` builds `data.source` for the train split, while `make_eval_loader(eval_cfg(cfg), ...)` sees the merged `data.val.source` and routes it — totalseg-val lands in the v2 `TotalSegProvider` branch (common.py:479), synth-val (if any) in the build_dataset synth branch. `loader_v2` is inherited by the merge, so both loaders stay on the v2 engine.
- Added `configs/experiment/3d/dataset/synth_gmm_v2.yaml`: v2 twin of experiment 43 as a modular dataset group (TRAIN synth_gmm cohort-hook, VAL real totalseg `balanced`). Stack a v2 model/encoder on top like 48 does: `python experiments/3d/train.py dataset=synth_gmm_v2 +model=m1 +encoder=e1`.
- Verified (real GMM bank + totalseg): `dataset=synth_gmm_v2` builds TRAIN=`InContextDataset` cohort_mode=True (gpu_realize native payload) and VAL=`InContextDataset` cohort_mode=False over `TotalSegProvider` (61 balanced classes, painted 128^3 items) — both yield an item.

### 49_abdomen_synth_ceiling — synth-trained twin of 48 — 2026-08-22
- New experiment: same frozen-CoLiPri / PatchSet3D / calibrated-aug recipe and the SAME 10-organ REAL TotalSeg abdomen val as 48, but TRAIN exclusively on the GMM-synth MAISI bank (10 matching MAISI classes). Measures how close synth-only training gets to 48's real abdomen ceiling.
- Built on the `synth_gmm_v2` cross-source dataset group (train synth cohort-hook / val real totalseg, both loader_v2). Overrides: train_classes = 10 MAISI abdomen names (spaces); crop_spacing_mm=2 + train_spacing_range=null to match 48's fixed-2mm regime; data.val.val_classes = the 10 TotalSeg abdomen names at 2mm/occupancy(0.5).
- Verified compose+build: TRAIN=synth 10 classes (gpu_realize, cohort_mode) / VAL=totalseg 10 abdomen organs (TotalSegProvider), model=patchset3d, encoder_frozen. Run: `python experiments/3d/train.py experiment=49_abdomen_synth_ceiling`.

### Frozen TotalSegmentator nnU-Net encoder for PatchSet3D — 2026-08-23
- Goal: use TotalSegmentator pretrained weights as a frozen PatchSet3D image encoder.
- Inspected `/software/TotalSegmentator/weights`: all downloaded CT models are `PlainConvUNet` (default `nnUNetPlans`, NOT ResEnc). Chosen: **Dataset297** (total, all 118 labels, 3 mm, 5 encoder stages, features 32/64/128/256/320, checkpoint `fold_0/checkpoint_final.pth` key `network_weights`, encoder tensors prefixed `encoder.`).
- `src/models/encoders/nnunet_ts.py` (NEW): `NnUNetTSEncoder` — builds PlainConvUNet from `plans.json`, loads weights (strict=False; asserts no `encoder.` key unfilled), keeps `.encoder` (return_skips=True). Same contract as PrimusEncoder/ConvEncoder3D: `forward(B,1,D,H,W,spacing=None)->(B,out_ch,R,R,R)`, `.out_ch`/`.resolution`/`.train_spacing_mm`, frozen-eval LRU cache (reuses Primus `_EncodeCache`/`_cached_encode`). Re-applies pretrain `CTNormalization` (invert loader z-score → clip [-1004,1588] → dataset mean/std -50.4/503.4); spacing arg ignored (conv, no RoPE).
- Multi-scale concat: at 128^3 input the stages land 128/64/32/16/8^3. Default `stages={2,3,4}` → out_ch=704 (stage 3 native 16^3 anchor, stage 2 2x avg-pool down, stage 4 2x trilinear up); skips low-level/normalization-dominated stages 0-1. Configurable via `arch.nnunet_ts_stages`.
- Wiring: `patchset3d.py` new `encoder="nnunet_ts"` branch + kwargs `nnunet_ts_weights`/`nnunet_ts_stages`; plumbed through `experiments/3d/train.py` build_model. `paths.totalseg_nnunet` added to `configs/cluster/nfs.yaml`; staged the 3 needed files (plans/dataset json + 158M checkpoint) onto NFS at `checkpoints/totalseg_nnunet/Dataset297_total_3mm/3d_fullres` (weights were node-local `/software` only).
- Configs: `encoder/e2.yaml` (frozen nnunet_ts group, conv twin of e1) + `experiment/50_abdomen_synth_nnunet_ts.yaml` (inherits 49, `override /encoder: e2`). Caveat noted: encoder pretrained at 3 mm but 49's crops are 2 mm (kept for ceiling comparability); ablate `crop_spacing_mm=3`.
- Verified: standalone encoder builds + loads (128^3→(B,704,16,16,16)); config composes (nnunet_ts keys merged, primus keys dropped); full build_model resolves the NFS path and builds PatchSet3D (encoder 8.5M frozen / head 38.5M trainable, img_embed in=704). Run: `python experiments/3d/train.py experiment=50_abdomen_synth_nnunet_ts`.

### nnU-Net TS encoder: normalization check + 3 mm variant (exp 51) — 2026-08-23
- Verified the two-source normalization for `NnUNetTSEncoder` (train synth_gmm / val totalseg):
  - VAL (real CT): `normalize_ct` clips HU to [-1007,1573] then z-scores with CT_MEAN=-167.3/CT_STD=505.8. Encoder inverts (`x*CT_STD+CT_MEAN` → exact clipped HU), re-clips to nnU-Net [-1004,1588] (~no-op), applies pretrain (HU+50.4)/503.4 → frozen conv sees its exact pretrain CTNormalization. Correct (this is the path that measures transfer).
  - TRAIN (synth GMM): image is in the arbitrary `(mu[0..255]+noise-128)/74` space (random per-label appearance by design, GMM_MEAN=128/GMM_STD=74). Encoder round-trips through CT_MEAN/CT_STD into a sane normalized range (≈[-2,1.5]) — IDENTICAL treatment to the existing Primus encoder e1. Exact HU semantics irrelevant for random synth appearance; consistency with e1 keeps 49-vs-50 fair. No encoder change needed.
- `experiment/51_abdomen_synth_nnunet_ts_3mm.yaml` (NEW): inherits 50, overrides `data.crop_spacing_mm=3` + `data.train_spacing_range=null` + `data.val.crop_spacing_mm=3` so BOTH train and eval crops match Dataset297's 3 mm pretrain pitch (in-distribution conv receptive fields). Trade-off: FOV=384 mm, 16^3 cell=24 mm (coarser targets). 50(2mm) vs 51(3mm) isolates encoder-scale vs grid-resolution. Verified compose: train synth@3mm / val totalseg@3mm / encoder nnunet_ts.

### Real-data organ nnU-Net encoder at 1.5 mm (exp 52) — 2026-08-24
- Checked TotalSegmentator weight availability for a v3 "organs" 1.5 mm encoder (`totalsegmentator` v2.18.0, `map_tasks_config.py`): v3 organ part = task 831 `Dataset831_TotalSegmentator_part1_organs_1830subj` (version `v3.0.0-weights`) but that GitHub release tag does NOT exist yet (404; latest published = `v2.5.0-weights`). Falling back to the **v2** organ part = task 291 `Dataset291_TotalSegmentator_part1_organs_1559subj` (`v2.0.0-weights`, 1.5 mm, `class_map_5_parts["class_map_part_organs"]` = 24 classes) — downloadable now. Only `Dataset297_total_3mm` is staged locally under `paths.totalseg_nnunet`; the organ model must still be downloaded/unpacked there.
- `experiment/52_organs_real_nnunet_ts.yaml` (NEW): real-data twin of 50 / nnU-Net counterpart of 48. Inherits `48_abdomen_ceiling` (loader_v2 real cross-subject train+val, m1 PatchSet3D head, calibrated aug, 400 ep) + `override /encoder: e2`. Overrides: `arch.nnunet_ts_weights=${paths.totalseg_nnunet}/Dataset291_organs_1.5mm/3d_fullres` (placeholder folder name — must match the unpacked model), `data.crop_spacing_mm=1.5` (train+eval; match encoder pretrain pitch), and train/val classes = the 24 part1_organs list minus kidney_cyst_left/right → 22 real organ classes. `NnUNetTSEncoder` auto-derives out_ch/channel dims/pretrain spacing from the model's own plans.json+dataset.json, so `nnunet_ts_stages=[2,3,4]`/`resolution=16` are inherited unchanged and no dim edits are needed. Verified compose: source=totalseg, loader_v2, crop 1.5 mm, encoder nnunet_ts frozen, 22 train/22 val classes. Run: `python experiments/3d/train.py experiment=52_organs_real_nnunet_ts` (after staging the Dataset291 weights).

### NSD (Normalized Surface Dice) in 3D eval — 2026-08-24
- Added batched, GPU-native NSD@tol alongside Dice in `experiments/3d/evaluate.py`. `nsd_batch(pred,target,spacing,tol_mm)` extracts both surfaces via 6-connectivity erosion (`_surface_voxels`, padded min-of-face-neighbours = scipy `binary_erosion` default), dilates each by a physical ball of radius `tol` (`_ball_kernel`, a tiny fixed `conv3d` — no full distance transform, exploits the fixed tolerance), and scores the fraction of each surface within `tol` of the other. Both-empty→1.0, one-empty→0.0. Voxel-count convention.
- **Validated bit-for-bit (max|Δ|=0.0)** against `monai.metrics.compute_surface_dice` across isotropic + anisotropic spacings, tolerances {1,2,3.5} mm, batched. Initial 26-conn surface ran high (Δ up to 0.25 near tol≈shift·spacing); switching to 6-conn made it exact. GPU timing: ~31 ms/batch (B=8, 128³ worst-case random blobs) on Blackwell — untimed (computed OUTSIDE the `t0` inference-timing block, before the `.cpu()` move), so it never inflates reported ms/sample.
- Wiring: `evaluate.py` reads `cfg.eval.nsd_tolerance_mm` (via `cfg.get("eval")`, so train.py's val cfg without the key simply skips NSD — Dice-only val, no breakage), computes per-batch NSD on `pred.device` using `batch["spacing"][0]` (fallback isotropic `crop_spacing_mm`), stores `case["nsd"]`. `_summarize` adds `mean_nsd`/`std_nsd`; `build_sample_table` adds an `nsd` column (after `dice`). Sweep path inherits it via `evaluate_classes`. `eval.py` prints `nsd=` per class + overall `Mean NSD`, logs `class/<c>/mean_nsd` + `mean_nsd` to wandb, and adds `mean_nsd,std_nsd` CSV columns. Config: `eval.nsd_tolerance_mm: 2.0` in `configs/experiment/3d/eval.yaml` (null disables; ≈1.3 vox at 1.5 mm).
- Follow-up: moved hard **Dice** onto the GPU too (`dice_batch`, same smooth `(2·inter+1)/(union+1)`, batched (B,)). `evaluate_classes` now moves `label` to `pred.device` once and computes Dice + NSD there (both untimed, before the `.cpu()` move), replacing the per-sample CPU `dice_binary(pred[i],label[i])` (2 `.item()` syncs/sample) in the case dict + figure title. Verified bit-identical to `dice_binary` (max Δ=0.0 @4dp incl. empty/full masks). `dice_binary` stays for the legacy `validate()`, the cascade native-numpy stitch, and `infer_nifti.py`.

### TotalSegmentator as a context-free eval baseline (Route B) — 2026-08-24
- Goal: score the TotalSegmentator method on our own dataset. TS ingests one image and outputs all-organ masks (ignores context); our loader gives target/context crops + a class name. Inspected the exact organ-class pipeline (`repos/totalsegmentator`): `task=total` (v2) = 5-part ensemble [291-295] @1.5mm, `nnUNetTrainerNoMirroring`, 3d_fullres, folds[0]; with `roi_subset` of organ names only **part1 (task 291)** runs. Full path: rough 6mm crop -> canonical(RAS) -> resample 1.5mm(order1) -> nnUNetv2_predict (internal CTNormalization from plans, Gaussian sliding-window, no TTA) -> part->total label LUT -> postproc -> resample back(order0) -> undo_canonical/undo_crop.
- **Route B** (`src/benchmark_models/totalseg.py` `TotalSegModel`): context-free adapter. Builds one `nnUNetPredictor` from the staged `Dataset291_organs_1.5mm/3d_fullres` (same folder as encoder e2), `name2id` from dataset.json (own label space 1..24). `predict()` drops context, inverts the loader z-score back to HU (`hu=x*CT_STD+CT_MEAN`, clamp — the same invert as `encoders/nnunet_ts.py:_norm`; nnU-Net re-applies its own CTNorm, so no double-norm), calls `predict_single_npy_array` per crop with `props={spacing}` (crops are 1.5mm iso = train spacing -> identity resample, output on our grid), selects the `label_name` channel, binarizes. Sets `needs_label_names=True`; eval loop now forwards per-sample `label_names` to predict (mirrors `spacing_aware` sp_kw). Registered in `eval.py:_build_model` + `benchmark_models.load_model` as `eval.model=totalsegmentator`; weights from `eval.totalseg_weights` or the experiment's `arch.nnunet_ts_weights`.
- Validated (CPU, exp52 test crops): **laterality correct** (kidney_left 0.87/0.72, kidney_right 0.98/0.98 — a flip would zero one side), liver 0.95/0.69, i.e. the low cases are Route-B crop-FOV effects (organ partly outside our crop), not orientation bugs. Run: `python experiments/3d/eval.py experiment=52_organs_real_nnunet_ts eval.model=totalsegmentator`. NEXT = Route A (faithful full-pipeline per native subject via the `totalsegmentator()` API + resample onto GT grid) as a subset faithfulness check.

### TotalSegmentator faithful native baseline (Route A) — 2026-08-24
- `experiments/3d/eval_totalseg_native.py` (NEW): standalone Hydra script (config_name=eval) running the OFFICIAL `totalsegmentator()` pipeline once per test subject — native ct.nii.gz through TS's own rough-crop(6mm task298) + canonical + 1.5mm resample + sliding-window + CTNorm + postproc — `ml=True, task=total, roi_subset=val_classes` (organ subset -> only part1/task291 runs). Extracts every requested class from the single multilabel output (label ids = `class_map["total"]`) and scores vs our native GT `label.npy` (ALL_CLASSES encoding) with the SAME metrics (`evaluate.dice_batch`/`nsd_batch`, reused `_summarize`) and wandb/CSV as eval.py.
- Geometry: TS output canonicalized (`as_closest_canonical`) -> RAS, matching label.npy's grid (both = as_closest_canonical of the same acquisition; convert_nnunet_task.py:102-107). Classes extracted BY NAME on both sides (TS `class_map["total"]` vs GT `_ALL_CLASSES_IDX` differ in integer encoding). NSD uses per-subject native RAS spacing from spacings.json (nsd_batch handles anisotropy). Shape-mismatch subjects warn+skip.
- Official weights already cached at ~/.totalsegmentator (Datasets 291-295, 297/298/300 crop) -> runs offline. Validated (2 test subjects, CPU): liver 0.995/kidney_left 0.991/kidney_right 0.990/spleen 0.990/pancreas 0.972, NSD~1.0, 0 skips — near-ceiling as expected (faithful TS reproduces the TS-derived GT), confirming RAS alignment + correct laterality + by-name extraction. Run: `.venv_blackwell/bin/python experiments/3d/eval_totalseg_native.py experiment=52_organs_real_nnunet_ts` (needs totalsegmentator + GPU; not in .venv_nero).
- Route A (native, whole-organ RAS) vs Route B (`eval.py eval.model=totalsegmentator`, our 128^3 crops): metrics identical but geometry differs by design — A scores the full organ, B is crop-FOV limited; A-B isolates TS's rough-crop/full-FOV/own-resampling cost.

### TotalSegmentator part-model class sets in data/totalseg_classes.py — 2026-08-24
- Extracted the 5 `total` part-model class lists verbatim from `totalsegmentator.map_to_binary.class_map_5_parts` into `data/totalseg_classes.py`: `TS_SET_ORGANS` (24, Dataset291), `TS_SET_VERTEBRAE` (26), `TS_SET_CARDIAC` (18), `TS_SET_MUSCLES` (23), `TS_SET_RIBS` (26). Order preserved = each part model's own label ids. Verified: exact match to the repo, and the 5 sets partition `ALL_CLASSES[:117]` exactly (117 unique). `TS_PART_SETS` dict maps `ts_organs|ts_vertebrae|ts_cardiac|ts_muscles|ts_ribs` -> list; `resolve_classes()` now accepts these special strings (CT only) so a config can set `val_classes: ts_organs`.
- Note: `TS_SET_ORGANS` is the faithful 24 incl. kidney_cyst_left/right (~absent from standard CT label.npy). exp 52 deliberately uses those 24 minus the 2 cysts = 22; to switch exp52 to the constant, set `val_classes: ts_organs` and accept 2 zero-Dice cyst rows (or keep the manual 22-list).

### Why TS native (0.93) >> TS crop-based (0.46) — root cause traced — 2026-08-24
- Route A (eval_totalseg_native.py) 0.93 vs Route B (eval.py eval.model=totalsegmentator) 0.46 on exp52 organs. Traced both: SAME weights (md5-identical Dataset291 nnUNetTrainerNoMirroring; TS-home == staged), same CTNorm (plans), same 1.5mm, same fold0. Route A = official `totalsegmentator()` on native ct.nii.gz: rough-crop torso -> canonical -> 1.5mm -> **Gaussian sliding-window over the whole volume** -> LUT -> resample back. Route B = v2 loader crops a 192mm cube (128^3 @1.5mm) around the GT centroid -> `predict_single_npy_array` on **that single isolated tile** (body truncated to a 192mm cube padded with air).
- NOT the user's hypothesis (crop centering) and NOT FOV: per-class diag showed Dice does not track FOV-capture (colon 0.81 @0.55 capture; gallbladder 0.00, adrenal 0.00, spleen 0.46 all at ~1.0 capture). Decisive 3-way test (same 128-vox window): **A_full ~= A_win >> B_win**. i.e. cropping the OUTPUT of the full-volume pred is harmless (kidney 0.99/0.99, spleen 0.989/0.989, trachea 0.947/0.947, esophagus 0.883/0.883); feeding the SAME isolated crop as INPUT collapses context-dependent organs (spleen 0.989->0.088, adrenal 0.941->0.001, esophagus 0.883->0.000, trachea 0.947->0.000) while big/high-contrast ones survive (liver 0.995->0.952, kidney 0.990->0.917).
- Root cause: nnU-Net is a WHOLE-IMAGE segmenter; running it on an isolated organ-centered crop (torso truncated to a 192mm cube in air) is OOD -> it fails to localize small/thin/confusable organs that it identifies from the broader anatomical field. => Route B is NOT a fair "TS on crops" baseline; it handicaps TS with OOD inputs. Fair crop-level TS = run native + crop the OUTPUT (= A_win), not feed crops to TS. For model comparison use Route A (native, whole-organ) or A_win (native pred restricted to the eval crop window).

### CORRECTION: TS crop-vs-native gap was an AXIS-ORDER BUG, not context loss — 2026-08-24
- Supersedes the earlier "isolated-crop / whole-image OOD / context loss" conclusion for the Route A (0.93) vs Route B (0.46) gap — that was WRONG. Root cause = `TotalSegModel` (Route B) fed nnU-Net the crop in nibabel RAS **(x,y,z)** order, but `predict_single_npy_array` expects the SimpleITKIO **(z,y,x)** convention. The silent transpose collapsed lateralized/small organs (spleen, esophagus, trachea, thyroid, adrenal) while big central blobs (liver 0.95, kidney 0.96) partly overlapped their own reflection and survived — masking the bug.
- Trace that nailed it (spleen, s0311, same Dataset291 weights): sliding-window cube-size sweep (128->256^3) + Gaussian/overlap toggles ALL stayed ~0.08 (so NOT per-tile FOV nor blending — the user was right that each tile is only a 192mm cube). predict_single_npy_array on the FULL ct_raw.npy also gave spleen 0.094 / eso 0.000 while liver 0.96 / kidney 0.95. BUT ct_raw.npy saved as NIfTI -> TS file-based pipeline gave spleen 0.989 / eso 0.971 == ct.nii.gz -> data is fine, only the npy input path differed. Cube orientation test: identity spleen 0.088 vs `transpose(2,1,0)` spleen **0.990**, liver 0.952->0.995.
- Fix: `src/benchmark_models/totalseg.py` predict() now `hu[i].transpose(2,1,0)` before predict_single_npy_array and transposes the seg back (spacing isotropic -> order-invariant). Re-validated end-to-end (exp52 loader, 2 subjects): spleen 0.088->0.986, eso 0.000->0.958, trachea 0.363->0.985, adrenal 0.001->0.961, gallbladder 0.941, mean 0.46->**0.967** ~= Route A. Route A (eval_totalseg_native.py) was never affected (uses totalsegmentator() file pipeline). NOTE: the earlier decisive "A_win>>B_win" table was also corrupted by this same bug (B_win used npy path) — disregard it.

### e2 frozen encoder is NOT affected by the axis-order bug — 2026-08-24
- Checked whether the Route-B (x,y,z)-vs-pretrain-(z,y,x) axis bug also hurts e2 (NnUNetTSEncoder used for training exp52). `_encode_batch` feeds the loader crop straight to the conv encoder with NO transpose. But it is NOT a correctness bug, for two structural reasons: (1) no spatial misalignment — the encoder applies no transpose, so its (B,704,16^3) feature map stays axis-aligned with the target/context masks (all x,y,z); (2) it's a FROZEN feature extractor + trainable head, trained AND evaluated with the same convention, so orientation is a consistent reparametrization the head learns around (unlike Route B which read out nnU-Net's semantic organ labels, which require canonical orientation).
- Also empirically negligible: encoding the same 30 organ crops in (x,y,z) vs (z,y,x), pooled features are **0.997** cosine-identical and class retrieval@1 is identical (0.633). Reason it differs from Route B: e2 uses only the ENCODER stages {2,3,4} (low/mid-level, orientation-robust after nnU-Net's rotation/mirror pretrain DA), NOT the decoder+final classification layer that actually does orientation-sensitive organ labeling (what collapsed in Route B). Conclusion: exp52 training is valid; feeding (z,y,x) would be marginally more in-distribution but the effect is ~0 and would invalidate existing checkpoints for no gain — leave as is.

### Coarse Dataset298 6mm as Route B (reproduce native rough-seg step) — 2026-08-24
- Goal: reproduce the coarse step of the native TotalSegmentator pipeline (the 6mm Dataset298 "total" model that native runs on the whole volume, resample=6.0, to compute the crop bbox) as a Route B eval, so we can score it directly. Traced it in the installed pkg: `totalsegmentator/python_api.py:872-910` runs `nnUNet_predict_image(input, None, task_id=298, model=3d_fullres, folds=[0], trainer=nnUNetTrainer_4000epochs_NoMirroring, tta=False, resample=6.0, task_name=total, multilabel_image=True)`. eval_totalseg_native.py doesn't call it separately — it's internal to `totalsegmentator()`.
- **No architectural change needed.** The existing Route B `TotalSegModel` already wraps `nnUNetPredictor.predict_single_npy_array`, which internally resamples the crop from its spacing to the plans target (6mm) → predicts → resamples back — i.e. exactly the coarse forward. Faithfulness matches: folds=(0,), use_mirroring=False == tta=False/NoMirroring, checkpoint_final.pth. Just point weights at Dataset298 (labels = full 117 "total" map, target spacing [6,6,6], patch 64^3; all 22 exp52 organs present by name) and set 6mm crops.
- Added `_resolve_weights_dir` in `src/benchmark_models/totalseg.py`: `eval.totalseg_weights` now accepts a full `..._3d_fullres` path OR a short TS token (`298` / `total_6mm`), resolved from `$nnUNet_results` or `~/.totalsegmentator/nnunet/results` like the native pipeline (portable across the NFS/Blackwell node split; cache path is home-dir-specific).
- Recipe: `python experiments/3d/eval.py experiment=52_organs_real_nnunet_ts eval.model=totalsegmentator eval.totalseg_weights=298 data.crop_spacing_mm=6.0`. FOV = 128*6 = 768mm (whole-body-ish coarse view, matching native running 298 on the full volume; vs the 1.5mm/192mm organ-neighborhood crop of the default Route B). Dice is scored at 6mm, so NOT on the same geometry as 1.5mm Route B / native Route A — intended (coarse resolution).
- Smoke test (3 test subjects, n=1..3/class): **Mean Dice 0.853, Mean NSD 0.750**, 1.7s/sample. Big organs high (kidney_r 0.966, lung_lower_l 0.974, liver 0.958), small/thin low as expected at 6mm (gallbladder 0.44, adrenal_l 0.744, thyroid 0.739, prostate 0.705). Runs clean end-to-end, exit 0.

### arch.nnunet_ts_weights token resolver (train + eval encoder) — 2026-08-24
- Added `resolve_ts_weights_dir(spec)` in `src/models/encoders/nnunet_ts.py` and call it in `NnUNetTSEncoder.__init__` (was `Path(weights_dir)`). Now `arch.nnunet_ts_weights` accepts a full `..._3d_fullres` path (unchanged) OR a short TS token — numeric id (`298`) or name fragment (`total_6mm`) — resolved from `$nnUNet_results` / `~/.totalsegmentator/nnunet/results` like the native pipeline finds its coarse model. Mirrors the Route-B benchmark resolver `src/benchmark_models/totalseg.py::_resolve_weights_dir` (kept independent to avoid coupling the light Route-B model to the heavy encoder module).
- Flows through: `train.py build_model -> PatchSet3D(nnunet_ts_weights=...) -> NnUNetTSEncoder`. Checkpoints store the token in `arch`, so eval.py rebuilds + re-resolves it. Verified: token + full-path both resolve, encoder builds from `total_6mm` (out_ch 704, train_spacing_mm 6.0, fwd (1,704,16,16,16)).
- Usage (frozen 6mm encoder). NB pair with `data.crop_spacing_mm=6.0` to match the 6mm pretrain pitch (mirrors exp52's 1.5mm choice for the 291 organ encoder):
    `python experiments/3d/train.py experiment=52_organs_real_nnunet_ts arch.nnunet_ts_weights=total_6mm data.crop_spacing_mm=6.0`

### 6 mm image cache for v2 loader (train speedup at crop_spacing_mm=6) — 2026-08-24
- Problem: `train.py experiment=52 data.crop_spacing_mm=6` is ~2.3× slower/volume than 1.5mm. Trace: at 6mm `organ_crop_arrays` sets FOV=128*6=768mm > body, so `crop_sizes` clamps to the WHOLE native volume (~10.1M vox vs 2.1M at 1.5mm) and it reloads+resamples the full volume per item. Single-thread breakdown @6mm: load+slice 50ms, image trilinear 24ms, occupancy mask 40ms (=114ms vs 49ms @1.5mm). The image resample is fully REDUNDANT — class/center-independent at whole-body FOV, recomputed for every class×context.
- Fidelity check first (why not just precompute a nearest 6mm multilabel): occupancy(thr=0.5, exp52's resolved value via d1.yaml — NOT the 0.1 code default) vs nearest at 6mm on 25 test subjects, all 117 labels → mean Dice 0.610, 22 near-empty events; thin/tubular structures (ribs 0.14–0.50, adrenals 0.31–0.40, esophagus/thyroid 0.58, carotids, cervical vertebrae) diverge hard. Also occupancy@0.5 is per-class + threshold-specific → can't be baked into a stored argmax multilabel. So: cache the IMAGE only, keep the occupancy mask on-the-fly.
- Impl (image-only, zero mask fidelity loss):
  1. `scripts/convert_to_npy.py`: `_resample_to_spacing()` + a `--target-spacing S` branch in `_convert_totalseg` writing `ct_raw_{S:g}mm.npy` (raw int16 HU, native→S mm iso, scipy zoom order-1 + Gaussian AA). Reuses in-memory raw or loads ct_raw.npy/header. Generate: `python scripts/convert_to_npy.py --target-spacing 6 --workers 20` (skips existing; the `omni_tiles/` dir errors benignly — not a real subject).
  2. `src/providers/totalseg.py`: `crop_and_place_cached()` — crop geometry + occupancy mask computed on the full-res native label (byte-identical to `crop_and_place`), image cropped from the cache by mapping the native physical box → cache indices, resampled to out_sizes. `TotalSegProvider.load` uses `ct_raw_{crop_spacing:g}mm.npy` when present (pitch==crop_spacing so image is downsampled not upsampled), else falls back to the native `ct_raw.npy` path (lazy — native CT no longer loaded when cache hits).
- Verified (s0004 full torso 255×177×440, cache 64×44×110): masks BYTE-IDENTICAL native vs cached for liver/spleen/esophagus/adrenal(1 vox)/kidney/stomach; image MAE 0.012 (tiny; cache adds Gaussian AA vs on-the-fly plain trilinear). Timing: 165→26 ms/load = **6.3×**. Mixed cached/native subjects coexist (per-subject file detection). Only activates at the crop pitch the cache was built for; 1.5/2mm runs are untouched.

  - Smoke A/B (exp52 crop_spacing_mm=6, 40 train subj, 78 steps, B=8, 8 workers, profile_timing): WITH cache data-wait 33ms/step (4ms/item), epoch ~55s; NATIVE (caches renamed .off) data-wait 339ms/step (43ms/item), epoch ~77s. => data-wait 10.3× less; compute unchanged (~310ms/step); with cache the 33ms data-wait is fully HIDDEN behind compute (GPU never stalls) whereas native 339ms exceeds compute and stalls it. Loss curve byte-identical A vs B (same seed) — training numerically unchanged. NB the post-train eval crashed with cudaErrorInitializationError in an eval worker = known cu13 worker CUDA-init quirk, unrelated to the cache; training loop itself clean. Full-dataset gen: `python scripts/convert_to_npy.py --target-spacing 6 --workers 20`.

### crop_spacing_mm=6 FOV (768mm) vs body extent — GT clipping check (pre-cache) — 2026-08-24
- Concern (before committing to the 6mm image cache): if a body axis > 6*128=768mm, the organ-centred crop clips the body and loses GT. Measured over 1228 subjects (spacings.json shape*spacing): **51/1228 (4.2%) exceed 768mm, long (craniocaudal, axis2) only, up to 1277mm** (whole-body/head-to-thigh scans). Axes 0/1 never exceed (max 749/700).
- BUT crops are organ-CENTRED on the target class, so what matters is target-organ retention, not body coverage. Across the 51 clipping subjects × exp52's 22 classes = **997 (subject,class) target pairs, 0 lose ANY target GT** (retention 1.000, jitter=0) — every exp52 organ (incl. colon 233k vox, small_bowel) is compact enough (<768mm) to fit a 768mm window centred on its own centroid; the clipped-away part is distant anatomy absent from the binary target mask.
- This clipping is a property of crop_spacing=6 ITSELF (native path), not the cache. The cache stores the whole body and clips identically at crop time. Verified on clipping subject s0224 (500×500×932mm): sub-box crops (starts=[0,0,34/92/...]) give BYTE-IDENTICAL masks native-vs-cache for colon/small_bowel/liver/kidney/esophagus at jitter=32; img_MAE 0.028.
- Caveat: retention measured at jitter=0; the ±T//4 (=192mm) train jitter + start-clamp keeps crops in-bounds and 0/997 loss even for the most spread organs, so negligible. WOULD matter if train/val_classes included full-body structures (whole spine/aorta) at crop_spacing=6 — exp52 has none. Conclusion: crop_spacing=6 does not lose target GT for exp52; cache is safe.

### Fix: eval DataLoader CUDA-init abort at crop_spacing_mm=6 (forkserver) — 2026-08-24
- Symptom: `train.py experiment=52 data.crop_spacing_mm=6 ...` finishes the train epoch then aborts at the FIRST eval batch: `terminate ... c10::AcceleratorError: CUDA error: initialization error` (in ExchangeDevice), `DataLoader worker exited unexpectedly / killed by signal Aborted`. Recurred across every crop_spacing_mm=6 run. NOT the 6mm image cache (val subjects are mostly uncached; happened pre-cache too).
- Root cause: no `set_start_method` in train.py -> Linux default = FORK. `measure_flops` (and training) initialize CUDA in the parent BEFORE loaders run, so eval workers are forked from a CUDA-initialized parent and inherit its CUDA context; the first CUDA touch in the worker aborts (contexts can't cross fork). Confirmed: `eval.workers=0` (no forked workers) -> run completes, val_dice=0.0341.
- Fix: `experiments/3d/common.py` make_eval_loader — all 3 eval DataLoader `common` dicts now pass `multiprocessing_context=("forkserver" if nw>0 and DEVICE.type=="cuda" else None)`. forkserver workers spawn from a clean server that never touched CUDA -> no inherited context. Keeps eval parallel (no per-run flag). Train loader left as fork (works; forked earlier under lighter CUDA state). CPU / workers=0 paths unchanged (None = default).
- Verified: default eval.workers (parallel) run completes, eval 31/31, val_dice=0.0341 (identical to workers=0), Done. forkserver adds a one-time ~30s worker-spin-up on the first eval (module re-import), then 2.5 it/s. Applies to eval.py too (same helper).

# Overnight multi-patch cascade probe — progress notes

User directive (2026-10-13, ~23:50): don't standby, keep iterating for ~8h on:
1. Impact of smaller input size (T=64, T=32) on compute and accuracy.
2. Sampling patches at UNCERTAIN target-pred / context borders (p~0.5) instead of
   confident foreground (p~1).
3. Also eval on totalseg CT+MRI (the checkpoint's own training distribution), not
   just FLARE22.

Checkpoint: `.../2026-09-28_168_cascade_real_host_all_shapes/best.pt` (168's OWN
best.pt, not 167's — resumed+trained further under cascade+real-host, this is the
actual target checkpoint, not the one named in the earlier plan docs by mistake).
GPU: single RTX A4000 (16GB) on host `nero`. Env: `.venv_thor` (torch 2.5.1+cu121).

## What was built (committed on `multipatch-cascade`)

- `c797f2e` — `data.cascade_sample_mode="uncertainty"`: generalizes the target-side
  sampler (`_prob_weighted_points_multi`) from weight=p (peaked at confident
  foreground) to weight=`1-|2p-1|` (peaked at the decision boundary, p~0.5).
  Context/support side gets the GT-appropriate analogue: a new
  `src/providers/totalseg.py::_resolve_center` `"random_border"` mode (uniform on
  the GT mask's own boundary shell via `scipy.ndimage.binary_erosion`, shared by
  `NativeGridProvider`/FLARE22 via its existing import of `_resolve_center`).
  7 new tests, full `experiments/3d/tests/` suite 259/259 + the 3 ledgered
  pre-existing failures (unrelated, confirmed during the original SDD process).
- `ad0ed8e` — fixed `compare_multipatch_flare22.py`'s Hydra entry point (was
  pointed at the train root config with no `eval:` section at all — the bug the
  final review flagged as "pre-existing, not yet attempted"). Must be run from the
  REPO ROOT (`python experiments/3d/compare_multipatch_flare22.py ...`), not from
  `experiments/3d` — the eval config's hydra searchpath resolves off `$PWD`.
- `385df4b` — generalized the script to any `cfg.data.source` (flare22/multisource/
  totalseg), added `cascade_sample_mode_sweep`, JSON output (incremental, one file
  per run under `results/experiments/3d/multipatch_cascade_probe/`), and
  `run_multipatch_overnight.sh` (sequential 8-phase driver, single GPU).

## Required overrides discovered empirically (not in any doc beforehand)

- Must run from repo root (see above).
- `augmentations.enabled=false` — `_assert_cascade_supported` rejects
  `cascade_spacings` + augmentations without `augmentations.gpu=true`.
- `data.crop_spacing_mm` must equal `cascade_spacings[0]` (level-0 geometry guard).
- `eval.batch_size=1` — default 8 OOMs this 16GB card at native decode resolution
  (`arch.decoder=conv`'s FiLM/skip-fusion path is memory-heavy); confirmed 1 fits.
- `data.cascade_spacings` / `data.cascade_n_patches_sweep` / `cascade_sample_mode_sweep`
  need a `+` prefix (new keys, Hydra struct mode) UNLESS composed via an
  `experiment=NNN` config that already declares them.
- `data.image_size=[64,64,64]` (and presumably `[32,32,32]`) runs WITHOUT crashing
  against the frozen checkpoint — verified empirically, contradicting my own
  pre-run theoretical worry that the decode head's `fine_m`/`grid_size` (baked in
  at construction from the ORIGINAL `image_size=128`) would shape-mismatch. Root
  cause it doesn't: `_decode`'s fine-filter path already has an
  `F.interpolate` fallback for exactly this mismatch (projects each fine stage to
  its own native resolution, then interpolates to the expected decode size if they
  differ) — so a smaller encoder input is mechanically fine; the ENCODER does less
  work, the DECODE resolution stays fixed at 128 regardless of input T.
  Caveat: `data.image_size` is a GLOBAL key, so shrinking it shrinks the COARSE
  level's FOV too (4mm*128=512mm -> 4mm*64=256mm), confounding "cheaper/coarser
  encoder" with "sees less of the body at the localization step" — this is why
  the runner ALSO does a FOV-preserving proxy (hold `data.image_size=128`, only
  coarsen the FINE level's `crop_spacing_mm`, e.g. `cascade_spacings=[4,3]` instead
  of `[4,1.5]`) as the cleaner, confound-free version of the same question.

## Spot-checks before committing to the full sweep (liver only, 50 FLARE22 subjects)

| config | n_patches | dice | nsd |
|---|---|---|---|
| spacings=[4,1.5], T=128 (baseline) | 1 | 0.9406 | 0.8533 |
| spacings=[4,1.5], T=128 | 2 | 0.9369 | 0.8458 |
| spacings=[4,1.5], T=64 (image_size override) | 1 | 0.8769 | 0.6143 |

Single-organ liver is the "easy, already-working" case per the original design
spec's own prediction (large solid organ, closest to today's working regime) — a
small regression at n=2 and a real (not catastrophic) drop at T=64 are both
expected directions, not yet a verdict on the hypothesis; the full 13-class +
totalseg sweep is what actually tests it.

## Sweep status (updated each check-in — READ THIS SECTION FIRST on resume)

Launched `nohup bash experiments/3d/run_multipatch_overnight.sh` at 23:55:43,
pid 249158 (disowned, survives shell/session changes), log at
`/tmp/claude-1011/overnight_sweep.log`, JSON results accumulating under
`results/experiments/3d/multipatch_cascade_probe/*.json`.

8 phases, sequential (single GPU, `set -uo pipefail` — one phase's failure/OOM
does NOT stop the rest):
1. FLARE22 13-class, n=[1,2,4] x mode=[prob,uncertainty], spacings=[4,1.5] (headline)
2. FLARE22 13-class, n=[1,2], spacings=[4,3] (size proxy A, FOV-preserved)
3. FLARE22 13-class, n=[1,2], spacings=[4,6] (size proxy A, more extreme)
4. FLARE22 13-class, n=[1,2], image_size=[64,64,64] (size proxy B, FOV shrinks too)
5. FLARE22 13-class, n=[1,2], image_size=[32,32,32] (size proxy B, more extreme)
6. totalseg CT+MRI, n=[1,2,4] x mode=[prob,uncertainty], spacings=[4,1.5], eval_epoch_length=150
7. totalseg CT+MRI, n=[1,2], spacings=[4,3] (size proxy A, in-distribution)
8. totalseg CT+MRI, n=[1,2], image_size=[64,64,64] (size proxy B, in-distribution)

**(status line updated at each check-in below this point)**
- 23:56 — phase 1 started.
- 00:23 — phase 1, point 1/5 (n=1, mode=prob) done. Process alive (pid 249158),
  GPU 100% util, 6.6GB/16GB used, healthy. ~27 min for point 1 (forward + native
  scoring over 650 samples). Now on point 2/5 (n=2, prob). Pace estimate at this
  rate: phase 1 (5 pts) ~115min, phases 2-3 (650 samples, FOV-preserved size
  proxy) ~92min, phases 4-5 (650 samples, T=64/32 -- forward is faster but native
  scoring cost is independent of T) ~72min, phase 6 (totalseg, 150 samples not
  650) ~25min, phases 7-8 ~20min total. Rough total ~5.4h, comfortably within
  budget -- no changes needed yet.
- **02:55 — phase 1 (all 6 points: n=[1,2,4] x mode=[prob,uncertainty]) completed
  cleanly (exit 0).** Phase 2 started at 02:55:06.
- **~03:04 — the whole process (and everything else in the sandbox) died.**
  Discovered at the next check-in (which landed at system time 12:09 the next
  day -- a large gap, likely the ScheduleWakeup mechanism + environment both got
  interrupted by a container/environment restart, not a code bug): log file's
  mtime stopped exactly at 03:04:10, mid-tqdm-line (483/650 samples into phase
  2's n=1 point), NO Python traceback anywhere in the log, `nvidia-smi` showed a
  perfectly clean 0% util / 15MB used (GPU fully released, not an OOM-kill
  signature which usually leaves a dmesg/traceback trail), and ALL pids from the
  original launch (249158 driver, 249160 python, forkserver workers) were gone
  with NO zombie/orphan remnants. This reads as the sandbox environment itself
  being torn down and recreated under me, not a bug in the sweep -- `nohup`+
  `disown` protect a background process from losing its CONTROLLING TERMINAL,
  not from the whole container being recycled.
  **Phase 1's data survived** (already flushed to disk incrementally, which is
  exactly why the script writes JSON per-point rather than buffering until the
  end): `results/experiments/3d/multipatch_cascade_probe/20261013_235557_flare22.json`,
  78 rows (13 classes x 6 points), complete and valid.
  **Relaunched phases 2-8** via `run_multipatch_overnight_part2.sh` (identical
  logic, just skips the already-done phase 1), pid 4819, log at
  `/tmp/claude-1011/overnight_sweep_part2.log`. If this happens again, the fix is
  the same: check which phase's JSON files exist, write a new `_partN.sh` that
  skips completed phases, relaunch.

## HEADLINE RESULT (phase 1, FLARE22, 13 classes, macro Dice)

| n_patches | mode=prob | mode=uncertainty |
|---|---|---|
| 1 (baseline) | 0.7668 | 0.7668 (same point, reused) |
| 2 | 0.7461 (Δ-0.0207) | 0.7449 (Δ-0.0219) |
| 4 | 0.7245 (Δ-0.0423) | 0.7199 (Δ-0.0469) |

**Multi-patch sampling consistently HURTS on FLARE22, monotonically worse with
more windows (n), and "uncertainty" (border) sampling does NOT rescue it** --
same or very slightly worse than "prob" sampling at every n. Per-class pattern:
small/ambiguous organs (adrenal glands) degrade the MOST (adrenal_gland_left
-0.258 dice at n=4!); the two kidneys are the only classes with a small,
consistent IMPROVEMENT (+0.002 to +0.004) at every n; most large solid organs
(liver, spleen, stomach, IVC) show small-to-moderate degradation (-0.004 to
-0.03); aorta is roughly flat at n=2, degrades at n=4.

**Read with the right caveat**: FLARE22's 13 organs are mostly solid,
single-focus structures (per the design spec's own framing, this was always an
"unseen-acquisition-centre generalization check," NOT a diffuse/multi-focal-ROI
stress test -- the ORIGINAL motivating hypothesis for multi-patch sampling was
large/diffuse/multi-focal targets like ISLES22 stroke lesions or GNC kidney
cysts, which FLARE22 does not contain). This result says "zero-shot multi-
window attention doesn't help on ordinary solid organs under domain shift" (a
real, useful finding, and consistent with the spec's own flagged risk that the
checkpoint was never trained for multi-window attention) -- it is NOT yet a
verdict on the core hypothesis the whole plan was motivated by. The totalseg
CT+MRI run (phase 6, in-progress) tests the SAME solid-organ-dominated question
under the checkpoint's OWN training distribution instead of domain shift, which
is the next-most-informative data point already queued; actually testing the
diffuse/multi-focal hypothesis would need a source like ISLES22/GNC_kidney,
which is NOT in the current queue (user's own 3 directives were input-size,
uncertainty-sampling, and totalseg CT+MRI -- not a request to add a new eval
source) -- flagging this as a good candidate for a FOLLOW-UP sweep if time
remains after the current 8 phases, not assumed/added unilaterally.

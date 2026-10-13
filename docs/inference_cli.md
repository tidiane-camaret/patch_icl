# Nifti inference CLI

One command-line tool for running the in-context cascade on real nifti files (no repo
checkout needed — vendored into the shared `patchset` conda env): **`patchset-infer`**
(`experiments/3d/infer_cli.py`), with two mutually-exclusive modes:

- **`--target`** — one target, one process.
- **`--manifest`** — many targets, one process, one checkpoint load.

Both wrap `predict_nifti()` in `experiments/3d/infer_nifti.py`, which runs the same
coarse→fine v2 cascade (`cascade.run_cascade`) as training/eval, over whatever nifti
files you point it at. The target is GT-free (no mask needed) — only the context
image/mask pairs need real masks. See `docs/superpowers/specs/2026-08-12-nifti-
incontext-cascade-inference-design.md` for the underlying design rationale (why the
cascade is GT-free, where `T=128` comes from, etc.) — this doc is usage-only.

`--checkpoint` defaults to the current working checkpoint
(`168_cascade_real_host_all_shapes`, `_DEFAULT_CHECKPOINT` in `infer_cli.py`) — a
20-epoch probe (`best_val_dice=0.51` @ epoch 10), not a final/fully-converged
checkpoint. Pass `--checkpoint` explicitly to use a different one; update
`_DEFAULT_CHECKPOINT` when a better checkpoint replaces it as the working default.

## Setup

```bash
conda activate patchset
patchset-infer --help
```

No repo access required — the env has its own vendored copy of the code
(`$ENV/share/patchset_infer/`), kept in sync by `scripts/sync_patchset_env.sh`. If
you're working from a repo checkout instead, run it directly:

```bash
python experiments/3d/infer_cli.py --help
```

**Node requirement:** the `patchset` env is built for Ampere GPUs (nero A4000,
thor/loki A6000). Blackwell (odin, sm_120) needs a different torch build — not
covered by this env.

## `--target`: one target

Single binary organ, with Dice against a GT mask:

```bash
patchset-infer \
  --target tgt_ct.nii.gz \
  --context ctx_ct.nii.gz ctx_mask.nii.gz \
  --checkpoint /nfs/.../best.pt \
  --out pred.nii.gz \
  --gt gt_mask.nii.gz
```

Repeat `--context` for K>1 contexts. Without `--gt`, it just predicts (no Dice
reported) — the target never needs a mask either way.

### Multi-label mode

Pass `--labels` to segment many organs from one id-valued context mask (e.g. a
TotalSegmentator-style combined segmentation) in one run:

```bash
patchset-infer \
  --target tgt_ct.nii.gz \
  --context ctx_ct.nii.gz ctx_combined_mask.nii.gz \
  --checkpoint /nfs/.../best.pt \
  --labels 1,5,13 \
  --batch-size 4 \
  --gt gt_combined_mask.nii.gz \
  --out pred.nii.gz
```

- `--labels all` segments every non-zero id present in the context mask(s) — but
  that's every id the *context* has, not intersected with what the target actually
  contains. If the target doesn't have some of those organs in frame, Dice against
  an all-zero GT region for that label will be near-zero and drag the macro average
  down — not a bug, just not what you meant. Pass an explicit, intersected id list
  (or compute one from the target's own GT) for a meaningful macro Dice.
- Output is one id-valued `uint8` mask on the target grid; where two organs'
  predictions overlap, the smaller one wins. If a source mask carries a Caret
  `<LabelTable>` (organ names/colors, as TotalSegmentator masks do), it's
  propagated onto the prediction automatically.
- `--batch-size` controls how many labels run through *one model forward* at
  once — lower it if you hit `CUDA out of memory`. On a 16GB GPU, a 20-label run
  OOM'd at the default `--batch-size 8` (and at 6); `4` was the fastest setting
  that didn't OOM in that test (13.25GB peak vs. `2`'s 8.86GB). This is about how
  many labels share *one* forward pass, not the total label count — a 100-label
  run at `--batch-size 4` uses the same peak memory as a 20-label run at the same
  batch size, just more forward passes.

## `--manifest`: many targets

Every `patchset-infer` process call pays a real cost before touching your data: ~1-2s
of Python import overhead, CUDA context init, and GPU clock ramp-up on the first kernel
launches after an idle period. None of that scales with how much work you actually
do — it's paid once per *process*. Running N targets as N separate `--target` calls
pays it N times; `--manifest` builds the model **once** and reuses it across every
target instead:

```bash
patchset-infer \
  --manifest tasks.jsonl \
  --checkpoint /nfs/.../best.pt \
  --out-dir preds/
```

**Manifest format** — JSON Lines, one task object per line:

```jsonl
{"target": "s0000/ct.nii.gz", "context": [["s0001/ct.nii.gz", "s0001/mask.nii.gz"]], "gt": "s0000/gt.nii.gz"}
{"target": "s0004/ct.nii.gz", "context": [["s0001/ct.nii.gz", "s0001/mask.nii.gz"]]}
```

- `target` and `context` are required (`context` is a list of `[image, mask]`
  pairs, so K>1 contexts just means more pairs in the list).
- `gt` (optional) — report Dice for that task.
- `out` (optional) — explicit output path; otherwise derived as
  `<out-dir>/<parent-dir-name>_<stem>_pred.nii.gz` (e.g. `s0000_ct_pred.nii.gz`).
  The parent-dir prefix matters: nifti layouts that reuse the same filename across
  every subject (TotalSegmentator's own `<subject>/ct.nii.gz` layout, for one)
  would otherwise collide and silently overwrite each other's predictions under
  `--out-dir`.
- `labels` / `batch_size` (optional) — per-task overrides of `--labels`/`--batch-size`
  (single-target mode's flags don't apply in `--manifest` mode; set per task instead).
- `--gt-required` — error out if any task is missing `gt`, instead of silently
  omitting Dice for it.

Model/cascade flags (`--checkpoint`, `--crop-spacings`, `--gpu-realize-crop`, etc.)
are shared across every task in one `--manifest` run, since the whole point is one
process/one checkpoint for all of them — only `labels`/`batch_size` vary per task
(via the manifest), everything else is set once on the command line.

## Performance flags

Two flags, measured on a 20-label multi-class run on a 16GB Ampere GPU — worth
trying both, since they don't clearly stack:

| Flag | What it does | Measured effect |
|---|---|---|
| `--gpu-realize-crop` | Resamples re-crops on the GPU instead of CPU (the same mechanism `data.gpu_realize_crop` already uses in training) | **~25-35% faster, and far more consistent** run to run (CPU resampling is sensitive to NFS/page-cache state; GPU isn't). Output Dice differs from the CPU path by a small, stable margin (<2e-2) — a decimate-then-area-pool vs. direct-trilinear resample ordering difference, not a bug. Little benefit on single-organ runs (nothing to batch). |
| `--recrop-workers N` | Thread-pool fan-out for the per-task crop-load step (`provider.load`/`load_native_crop` release the GIL, so this is real parallelism) | **~2x faster wall time at `N=4`** — but only measured on the **plain CPU path** (`--gpu-realize-crop` *not* set). Showed no clear additional benefit stacked on top of `--gpu-realize-crop` in the same test, since that flag already removes most of the CPU-bound work this would parallelize. **Caveat**: if the native volume is huge (e.g. very fine in-plane spacing clamping the crop to the whole volume), concurrent threads can instead contend for CPU/BLAS resources and make things *slower* — the 2x win was measured on modest, page-cached volumes. Check `nproc`/load before assuming more workers helps. |

Both default off/1 (unchanged CLI behavior) — opt in explicitly.

## Other flags worth knowing about

- `--crop-spacings` (default `4,1.5`) — the cascade's coarse→fine field-of-view
  schedule in mm/voxel, **not** the input nifti's native voxel spacing (that's read
  from the affine automatically). Match whatever the checkpoint was trained with.
- `--query-prior` (default `pred`) — match `data.cascade_query_prior` from training.
- `--mask-downsample` (default `occupancy`) — match the checkpoint's training
  value. Getting this wrong doesn't error, it just silently changes what the model
  sees; the CLI prints a `[warn] eval data config NOT inherited from the
  checkpoint` block listing every data-fidelity key that differs from training —
  read it, it flags exactly this kind of drift.
- `--mask-occupancy-thr` — omit it; it auto-inherits the checkpoint's own training
  value. Only set it explicitly if you deliberately want to override.
- `--override KEY=VALUE` (repeatable) — escape hatch for any other Hydra override
  not covered by a dedicated flag. Note the `data` config group is struct-typed, so
  a brand-new key (not already read somewhere in the composed config) needs the
  `++` prefix, e.g. `--override ++data.some_new_key=value`.

## Checking output against training

The `[warn] eval data config NOT inherited from the checkpoint` block the CLI
prints on every run is not decorative — `predict_nifti` only restores the
checkpoint's stored `arch` automatically; everything under `data.*` (crop spacing,
mask downsample, cascade center mode, etc.) has to be re-supplied by the caller to
match training. Read that block before trusting a number from a new checkpoint.

## Distributing code changes

If you change anything under `experiments/3d/` or `src/` that `patchset-infer`
depends on, re-sync the vendored env copy so other users (who don't have repo access)
pick it up:

```bash
bash scripts/sync_patchset_env.sh                 # default env path
bash scripts/sync_patchset_env.sh /path/to/env    # custom env prefix
```

This refreshes the `patchset-infer` launcher and mirrors `.py`/`.yaml`/`.json` source
files into the env's `share/patchset_infer/` bundle. It doesn't touch the env's actual
Python dependencies (see `requirements-patchset-infer.txt` for those).

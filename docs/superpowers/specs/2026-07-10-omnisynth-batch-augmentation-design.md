# Batch augmentation for omniSynth (and any source) in the unified trainer — design

**Date:** 2026-07-10
**Component:** `experiments/2d/train.py` (+ config + a test)
**Status:** approved, pending implementation plan

## Goal

Let `configs/augmentations/` presets be applied to omniSynth training (and any
data source) run through the unified `experiments/2d/train.py`, reusing the
existing `pfn_train.augment()` function. Opt-in, so existing `train.py` runs are
unchanged.

## Background: how augmentation is wired today

The presets in `configs/augmentations/*.yaml` are consumed by a single,
source-agnostic function `augment(images, masks, K, cfg)` in
`experiments/2d/pfn_train.py`. It operates on `(B, T, 1, H, W)` float batches
(query at index `K = T-1`) with a two-tier scheme:

- **task** — one episode-wide, intensity-only op (e.g. invert); safe on the query.
- **geometric** — per-context image+mask jointly (hflip/vflip/rotate/scale/
  translate/crop/elastic); **never touches the query** (its GT is read from the
  un-augmented batch).
- **intensity** — per-image incl. query (brightness/contrast/gamma/noise/
  bias_field); masks unchanged.

It is applied by the **`pfn_seg.py`** and **`multilevel/train.py`** trainers,
which load the preset (`OmegaConf.load(.../configs/augmentations/{aug_preset}.yaml)`
→ `cfg.aug`) and call `if cfg.aug.enabled: augment(...)` inside the loop.

The **unified `train.py`** (universeg / patchset_cnn — the trainer omniSynth runs
through) never loads `cfg.aug` or calls `augment()`. The `aug_preset: 2d` key and
its "Loaded into cfg.aug in main()" comment in `train_base.yaml` are vestigial for
this trainer. So "apply augmentations to omniSynth" = port the `pfn_seg.py` wiring
into the unified trainer.

## Decisions

- **Reuse `augment()`** (not a per-dataset or refactored path). Source-agnostic,
  GPU-batched, already exercised by the `pfn_seg.py` route.
- **Generic**, not omniSynth-specific: wired into `train.py` so any source
  benefits; geometric-vs-intensity is a preset choice, not a code choice.
- **Opt-in**, default off: existing `train.py` runs stay byte-identical until a
  config opts in. A dedicated boolean `augment: false` is the gate (the existing
  always-present `aug_preset` key cannot serve as the trigger).

## Changes

### `experiments/2d/train.py`

**Import** `augment` from `pfn_train` (the file already imports `Muon,
lawa_average, soft_dice_loss` from it).

**In `main()`** — load the preset only when opted in (mirrors `pfn_seg.py`).
`train.py` has no repo-root constant yet, so add one at module scope next to the
existing `sys.path.insert` (line ~52), identical to `pfn_seg.py:47`:
```python
_ROOT = str(Path(__file__).resolve().parents[2])   # experiments/2d/train.py -> repo root
```
then in `main()`:
```python
if cfg.get("augment", False):
    _aug = OmegaConf.load(Path(_ROOT) / "configs" / "augmentations" / f"{cfg.aug_preset}.yaml")
    cfg.aug = OmegaConf.merge(_aug, cfg.aug) if cfg.get("aug", None) else _aug
```

**A pure helper** (unit-testable in isolation):
```python
def _augment_batch(img, cin, cout, aug_cfg):
    """Augment context pairs + query intensity via pfn_train.augment.

    img (B,1,H,W); cin/cout (B,K,1,H,W). Returns (img, cin, cout). The query GT
    (lbl) is never passed in, so it stays valid: augment() geometrically transforms
    contexts only; the query receives at most intensity/task ops."""
    K = cin.shape[1]
    imgs = torch.cat([cin, img.unsqueeze(1)], dim=1)              # (B,T,1,H,W), query at index K
    msks = torch.cat([cout, torch.zeros_like(img.unsqueeze(1))], dim=1)
    imgs, msks = augment(imgs, msks, K, aug_cfg)
    return imgs[:, K], imgs[:, :K], msks[:, :K]                   # img, cin, cout
```

**In `train_epoch`**, right after the `if batch is None: continue` check and after
moving `img/lbl/cin/cout` to device — train only, never val:
```python
if cfg.get("augment", False) and cfg.aug.get("enabled", True):
    img, cin, cout = _augment_batch(img, cin, cout, cfg.aug)
```

### `configs/experiment/2d/train_base.yaml`

- Add top-level `augment: false` (sibling to `aug_preset`) with a comment: gates
  the `pfn_train.augment` step in `train.py`; default off preserves current
  no-aug behavior.
- Fix the stale `aug_preset` comment to describe the real opt-in path (loaded into
  `cfg.aug` in `train.py` `main()` only when `augment: true`).

### `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`

- Add `augment: true`. It inherits `aug_preset: 2d`, so it uses `2d.yaml` by
  default; overridable via `aug_preset=…` or `+aug.<field>=…` on the CLI. Only
  this config is enabled; experiment 1 and all other configs remain opt-in.

### `docs/logs.md`

Append a change-log entry.

## Query-GT correctness

`augment()` transforms contexts geometrically and applies at most intensity/task
ops to the query (masks unchanged). `lbl` (the loss target, used for both the
coarse and refine losses) is never handed to `augment()`, so it remains aligned
with the query image. This is the same invariant `pfn_seg.py` relies on. Refine
mode needs no special handling: augmentation happens on the model inputs before
the forward, so both passes and their internal bbox crops operate on the
augmented batch.

## Testing (`tests/test_train_augment.py`, new)

1. **Opt-in default.** With no `augment` key (or `false`), `main()`'s load branch
   does not fire (no `aug` block added / gate is `False`) — existing baselines
   untouched.
2. **Split alignment (load-bearing).** `_augment_batch` with a geometric-only
   preset (`rotate.p=1`, all intensity `p=0`): the returned query image is
   bit-identical to the input query (augment left index `K` geometrically
   untouched AND the assemble/split indexing is correct), while at least one
   context image differs (aug actually ran). An off-by-one in the cat/split flips
   these.
3. **Shapes.** Returned `img` is `(B,1,H,W)`, `cin`/`cout` are `(B,K,1,H,W)`;
   `lbl` is not an argument (structurally uncorruptible).
4. **End-to-end smoke.** One tiny `PatchSetCNN` refine step with `augment` enabled
   runs forward+backward and yields a finite loss.

`augment()` itself is pre-existing and exercised by the `pfn_seg.py` route on the
identical `(B,T,1,H,W)` shape, so it is not re-tested — only the new `train.py`
wiring.

## Out of scope

- Changes to `pfn_train.augment` or the preset schema.
- New augmentation presets (the preset choice is deferred/config-driven; `2d.yaml`
  is the default).
- Augmentation inside `OmniSynthICLDataset` / the synth generator.
- Enabling aug on configs other than `2_omnisynth_medseg_refine.yaml`.
- Refactoring the shared aug step across the three trainers.

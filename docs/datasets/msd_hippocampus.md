# MSD Task04_Hippocampus

*Medical Segmentation Decathlon, Task 4 — T1 MRI, hippocampus anterior/posterior.* Downloaded
and inspected 2026-09-12 (see `docs/datasets/eval_strategy_report.md` — this is the report's
flagship "genuinely unseen anatomical class" pick: TotalSegmentator labels only a single
undifferentiated `brain` region, no hippocampus). Not yet wired into the harness (no
provider/converter/config — this pass is characterization only, mirroring the pre-integration
state nasalseg/flare22 were once in).

## 1. Access — fully open, zero friction

Public S3 bucket (`s3://msd-for-monai`, AWS Registry of Open Data, `eu-west-2`/`us-west-2`
mirrors), **no account, no login, plain HTTPS**:
```
curl -LO https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar
```
28 MB (yes, megabytes — see §3). License CC-BY-SA 4.0 (`dataset.json`), reference Vanderbilt
University Medical Center.

## 2. Composition

nnU-Net-style layout: `imagesTr/` + `labelsTr/` (paired, GT available) + `imagesTs/` (test,
**no GT** — held out by the original decathlon, not released).

| split | n | GT? |
|---|---:|---|
| train | **260** | yes |
| test | 130 | no (hidden) |

`imagesTr`/`labelsTr` hashed pairwise — **0 duplicates** across all 260 training volumes.

## 3. Geometry — VERIFIED, and the striking part

All 260 training volumes: orientation **RAS** (uniform), spacing **exactly 1×1×1 mm** (uniform,
isotropic). Shape:

| | min | median | max |
|---|---|---|---|
| shape (vox) | 31×40×24 | 35×50×36 | 43×59×47 |

**These are not full brain scans — they are already cropped to a small ROI around the
hippocampus itself.** A median volume is ~63k voxels total (vs. TotalSeg's ~12M, NasalSeg's
~1.4M). Physical FOV is on the order of 35–50 mm per axis. This is a fundamentally different
regime from every source currently in the harness (TotalSeg/FLARE22/NasalSeg all ship whole
organs/heads/torsos at their native anatomical scale) and has a direct design consequence: our
v2 crop path resamples a `T*crop_spacing_mm` physical box onto a 128³ grid, so filling any
reasonable fraction of that grid here needs a **sub-millimeter** `crop_spacing_mm` (something
like 0.3–0.4mm to get a ~40–50mm FOV) — every other source in the harness uses `crop_spacing_mm
>= 0.6`. Untested territory for the crop/resample pipeline (`organ_crop_arrays`'s antialiasing
and occupancy-mask assumptions were tuned against totalseg/FLARE22-scale decimation ratios, not
a sub-voxel-dominated upsample regime).

## 4. Labels

Two sub-structures, **both present in all 260 cases**:

| label | median mm³ | present |
|---|---:|---:|
| 1 = Anterior | 1725 | 260/260 |
| 2 = Posterior | 1578 | 260/260 |

No left/right split — each volume is a single hippocampus (left or right, undetermined from
the data alone; the decathlon doesn't document which, and unlike NasalSeg/FLARE22 there's no
laterality signal to check since the crop is already so tight there's no surrounding anatomy to
locate it against).

## 5. Intensity — NOT normalized, unlike our CT sources

Raw scanner T1 intensities, **not** clipped/z-scored at the source (unlike our CT
`fingerprint_1228` global frame): observed range up to ~1.7×10⁶ in raw units, wildly
non-uniform in scale across scans (MSD does not harmonize MRI intensity across tasks). Any
future provider for this source needs **per-volume** normalization (z-score per scan, the same
convention already used for `totalsegmri` via `normalize_mri`), not a fixed global window.

## 6. Fit as an eval set

- **Genuinely unseen class**: no hippocampus (or any brain substructure) exists in
  TotalSegmentator's ~117/238-class vocabulary — this is exactly the claim the report leads
  with (`Neuroverse3D` reports the ICL literature's largest gains, "exceeding 20 percentage
  points," on exactly this kind of target).
  Anterior/Posterior gives 2 held-out classes, not a single "hippocampus" class — a real if
  minor difference from how Medverse/Neuroverse3D would report it (they likely use PPMI's own
  richer brain-substructure atlas, not MSD's anterior/posterior split).
- **Tiny structures, tiny volumes**: 260 training cases is a healthy pool for K=1 in-context
  tasks (520 (subject, class) tasks), but the sub-2000mm³ label sizes and pre-cropped native
  volumes mean Dice will be sensitive to exactly the same small-object instability the report
  flags generally (§ Practical caveats: "tiny structures... make Dice unstable; report NSD
  alongside Dice").
- **Not a localization test**: because the volume IS already the ROI, there's no "find the
  hippocampus in a head" problem the way FLARE22 has a real "find the organ in a torso"
  problem — closer to NasalSeg's regime (object fills a large fraction of a small volume) than
  FLARE22's.

## 7. Integration (implemented, 2026-09-12)

Wired as an eval-only v2 source, same shape as ISLES22/Shifts-MS (`NativeGridProvider`
`MODALITY="mri"` — no new code needed in `native_grid.py`).

```bash
python scripts/convert_msd_hippocampus.py --workers 16   # .nii.gz -> native RAS .npy + mri_stats
python experiments/3d/eval.py dataset=msd_hippocampus eval.model=patchset3d eval.checkpoint=<ckpt>
```

| piece | file |
|---|---|
| converter | `scripts/convert_msd_hippocampus.py` |
| provider | `src/providers/msd_hippocampus.py` (`MsdHippocampusProvider`, `MODALITY="mri"`) |
| config | `configs/experiment/3d/dataset/msd_hippocampus.yaml` |
| dispatch | `experiments/3d/common.py`, `experiments/3d/eval.py` (same branches ISLES22/Shifts-MS use) |

**`crop_spacing_mm=0.5`** — resolved the §3 open question via the same round-trip
grid-occupancy sweep used for every other source (T=128, all 260 training shapes): 0.5mm is the
largest pitch with **zero** subjects clipped on any axis (0.45mm already clips 1/260 subjects
on one axis), giving 55-78% mean grid fill, 38-62% worst-case. This is the harness's first
sub-mm `crop_spacing_mm` — every other source needs `>=0.6`.

**Converted: 260/260 cases, 0 failures**, shapes match the source exactly (min/median/max
31x40x24 / 35x50x36 / 43x59x47, all spacing 1x1x1mm — already RAS so
`nib.as_closest_canonical` is a no-op safety net here, not a real fix like ISLES22/Shifts-MS
needed). `ct_raw.npy` float32 raw T1 (unnormalized); `label.npy` uint8 {0,1,2}.

**Found + fixed a general (not hippocampus-specific) latent bug** in the shared
`organ_crop_arrays` (`src/totalseg_dataloader_incontext.py`), surfaced here for the first time
because this is the first source where the target crop box (~64vox at 0.5mm) exceeds the
native volume on most axes: when `crop_size == dim` on an axis (`smax=0`), the start range
collapses to `lo==hi==0`, and `rng.randint(lo, hi)` — called with a real
`np.random.RandomState` in production — raises `ValueError: high <= 0` (numpy's `randint`
requires `high > low`; Python's `random.Random.randint`, used by the one existing test that
covers this crop-clamped case, is inclusive and tolerates `lo==hi`, which is why it was never
caught before). Fixed by skipping the `randint` call when `lo >= hi`. Regression test:
`test_organ_crop_smax_zero_with_numpy_rng_does_not_crash`
(`experiments/3d/tests/test_crop_helpers.py`). Full 3D suite re-run clean, no regressions.

Visual sanity check: `results/3d/msd_hippocampus_items.png` — clean, well-aligned elongated
hippocampus masks; a couple of rotated-crop rows show background corners entering frame
(expected task-level rotation aug on an already-tiny native volume).

Raw source stays at `/nfs/.../ANALYSIS_20251122/data/msd_hippocampus/Task04_Hippocampus/`;
converted native-grid `.npy` at `paths.msd_hippocampus`
(`/nfs/.../data/msd_hippocampus/npy/`).

## 8. First eval result (exp92 checkpoint, single-level, 2026-09-12)

```
python experiments/3d/eval.py dataset=msd_hippocampus eval.model=patchset3d \
  eval.checkpoint=.../3d_train/2026-09-11_92_multisource_synth/best.pt
```
**Mean Dice 0.4809, NSD 0.7193, n=260** (wandb `classic-gorge-77`) — by far the best OOD-MRI
result of the three integrated this session (ISLES22 0.0544, Shifts-MS 0.0632):
- Per-class: `hippocampus_anterior` 0.527±0.116 (nsd 0.720), `hippocampus_posterior`
  0.435±0.142 (nsd 0.719). Anterior scores consistently a bit higher, plausibly the larger of
  the two sub-structures.
- Qualitatively (`figures/hippocampus_anterior_hippocampus_001.png`, dice=0.523): a genuinely
  accurate, well-localized prediction closely matched to GT in shape and position — a real
  segmentation, not a near-miss/collapse pattern like ISLES22/Shifts-MS.
- Plausible reason for the gap: unlike the lesion datasets' scattered, variable-count,
  variable-size targets, hippocampus is a single well-defined compact structure per volume —
  closer to the model's trained regime (localize one structure, segment it) even though the
  class/modality/scale are all genuinely unseen. The class-OOD claim holds regardless
  (checked directly against `data/totalseg_classes.py` — no hippocampus/brain-substructure
  class exists there).

Not yet run: a matched cascade eval.

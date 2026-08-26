# NasalSeg

Head/sinus CT with 5 densely-annotated **air-filled cavities**. Inspected 2026-08-25 on NFS
(`paths` candidate: `.../ANALYSIS_20251122/data/nasalseg`), not yet wired into the harness.

Layout: `images/P###_img.nrrd`, `labels/P###_seg.nrrd` (+ the original `NasalSeg.zip`).
NRRD, so it needs `pynrrd` (present in `.venv_blackwell`), not nibabel.

## 1. Composition — 130 files, but only **107 unique cases**

**19 groups of byte-identical duplicates**, image *and* label (SHA-1 over the raw arrays):

```
P002/P003  P005/P006  P009/P010  P026/P027  P030/P031  P033/P034  P040/P041
P047/P048/P049   P053/P054   P060/P061/P062   P072/P073   P076/P077/P078/P079
P080/P081  P083/P084  P090/P091  P101/P102  P110/P111  P117/P118  P127/P128
```

23 redundant files. This is not a multi-annotator situation — the segmentations are
identical too, so there is nothing to gain by keeping them.

**This matters for in-context eval specifically**: a duplicate drawn as a context for its
own twin is self-context (an exact clone of the target), which inflates Dice toward the
trivial-matching ceiling with no `SELF` flag to catch it, because the case ids differ.
De-duplicate to 107 before building any dataset. → **535 (subject, class) tasks**.

## 2. Labels — index order VERIFIED, not assumed

All 5 present in all 130 files (dense). Names below are inferred geometrically, since the
files carry no label metadata:

| idx | name | evidence |
|---|---|---|
| 1 | `maxillary_sinus_right` | centroid −43.6 vox off midline (0/130 on the left); **stands 12.2 mm off the mid-sagittal plane** |
| 2 | `maxillary_sinus_left` | centroid +43.6 vox (130/130 left); 15.4 mm off midline |
| 3 | `nasal_cavity_right` | centroid −11.2 vox; **touches the midline (0.1 mm)** — septum-adjacent |
| 4 | `nasal_cavity_left` | centroid +10.4 vox (124/130 left); touches midline (0.4 mm) |
| 5 | `nasopharynx` | midline (+0.7 vox), strongly **posterior** (+54.9) and inferior (−8.5) |

The lateral/midline split is what separates the two pairs: nasal-cavity halves are separated
only by the septum, maxillary sinuses sit out in the cheeks. Volumes agree — the sinuses have
a much wider spread (hypoplasia/opacification), the cavities are tight.

| label | median mm³ | p5 | p95 | sphere-equiv ⌀ |
|---|---:|---:|---:|---:|
| maxillary_sinus_right | 12752 | 2525 | 23722 | 29.0 mm |
| maxillary_sinus_left | 12681 | 3087 | 25059 | 28.9 mm |
| nasal_cavity_right | 11043 | 8524 | 15973 | 27.6 mm |
| nasal_cavity_left | 11170 | 7753 | 16158 | 27.7 mm |
| nasopharynx | 13325 | 7671 | 22506 | 29.4 mm |

## 3. Geometry — clean, but 12 label headers are junk

Uniform and near-isotropic-in-plane: in-plane spacing **0.586 mm in all 130** (single value),
z **1.5 mm** (128 cases) / 1.51 (2). Anisotropy 2.56×. `space directions` exactly diagonal
(max off-diagonal 0.000e+00) in every file.

Volumes are **small**: median shape 152×187×51 = 1.4 M voxels, FOV 89×110×76 mm — ~20× fewer
voxels than a FLARE22 case. The whole dataset is 217 MB.

**Frame is LPS**, not RAS — all 130. Our converters (`convert_to_npy.py`, `convert_flare22.py`)
produce RAS via `nib.as_closest_canonical`, where anatomical left sits at the *lower* axis-0
index; LPS is the mirror. A converter must flip axes 0 and 1 or the model sees mirrored heads.

**12 cases have contradictory label headers** — image and label array shapes always match, but
the `_seg.nrrd` header disagrees with the `_img.nrrd` header:

* 9 cases (P025, P042, P074, P089, P100, P101, P108, P118, P129): seg z-direction is
  **negative** (−1.5) with a shifted origin, describing a physical extent that does not even
  overlap the image.
* 3 cases (P039, P096, P105): seg z-spacing is **1.0** where the image says 1.5.

These are **junk metadata, not real misalignment** — verified, not assumed. The labels mark
air-filled cavities, so mean HU under a correctly-aligned mask must be strongly negative. As-is
it is −760 to −930 HU with 93–99 % of voxels below −300 HU, indistinguishable from clean cases;
z-flipping the mask destroys it (−280 to −640 HU, 45–74 % air). So the arrays are index-aligned.

→ **A converter must take geometry from the image header and use the label array by index**,
never trusting the seg header. Assert only shape equality.

## 4. Storage

Already `int16`, integral, global HU range exactly **[−1024, 3071]** in all 130 — so int16 is
bit-exact and conversion to `.npy` is a straight copy (plus the LPS→RAS flip). Labels are
`int16` 0–5 → `uint8`.

## 5. Fit as an eval set

**Zero overlap with TotalSegmentator** — no nasal cavity, maxillary sinus, or nasopharynx
class exists there, so all 5 classes are strictly zero-shot for any TotalSeg-trained
checkpoint.

The sharper shift is **contrast polarity**: every target is an *air* cavity (≈ −840 HU)
bounded by bone, whereas every TotalSeg/FLARE22 organ we evaluate is soft tissue *brighter*
than its surround. Harder OOD probe than FLARE22 (acquisition + annotation-convention drift
only); it tests whether in-context segmentation reads the target's appearance off the context
mask or falls back on a learned soft-tissue prior.

Unresolved: whether NasalSeg publishes an official split (all files sit in one flat
directory here, no `meta.csv` equivalent). Currently every case is used for eval.

## 6. Integration (implemented)

Eval-only v2 source, same shape as FLARE22:

```bash
python scripts/convert_nasalseg.py --workers 16          # .nrrd -> native RAS .npy, de-duped
python experiments/3d/eval.py dataset=nasalseg eval.model=patchset3d
```

| piece | file |
|---|---|
| converter (de-dup + LPS→RAS) | `scripts/convert_nasalseg.py` |
| provider | `src/providers/nasalseg.py` (`NasalSegProvider`) |
| shared crop/resample base | `src/providers/native_grid.py` (`NativeGridProvider`, also backs FLARE22) |
| config | `configs/experiment/3d/dataset/nasalseg.yaml` |
| dispatch | `experiments/3d/common.py`, `experiments/3d/eval.py` |

**107 cases × 5 classes = 535 tasks.** The converter re-derives the duplicate groups by
hashing the raw arrays (not from a hardcoded list) and keeps the first id of each; the
mapping is written to `duplicates.json` so the drop is auditable.

**The LPS→RAS flip is verified, not assumed**: after conversion the anatomical *left*
structure sits at the *lower* axis-0 index — `maxillary_sinus_left − right` = −78 to −89
voxels, the same sign as FLARE22's `kidney_left − kidney_right` (−219). Alignment is
re-checked per case at convert time via contrast polarity (`frac_air_under_label`, min 83.5 %,
median 95.9 % of masked voxels below −300 HU).

### Settings, and why

**`crop_spacing_mm: 0.8`.** Unlike FLARE22 — where matching the native z spacing dominated
GT fidelity — the round-trip ceiling here is nearly flat across pitches, because the binding
constraint is structure complexity rather than the resample. So grid utilisation decides, and
the head FOV (89×110×76 mm) is *smaller* than a 128³ crop at 1.5 mm:

| pitch | crop FOV | median `out_sizes` | % GT inside crop |
|---:|---:|---|---:|
| 0.6 | 77 mm | [128, 128, 124] | 99.9 % |
| **0.8** | **102 mm** | **[111, 128, 93]** | **100 %** |
| 1.0 | 128 mm | [89, 110, 74] | 100 % |
| 1.5 | 192 mm | [59, 73, 49] | 100 % |

At 1.5 mm roughly four fifths of the grid would be air padding.

**`mask_downsample: occupancy`, `mask_occupancy_thr: 0.5`** — not optional here. The nasal
cavities are thin and convoluted (turbinates), and `nearest` loses them outright. Round-trip
Dice of a perfect predictor at pitch 0.8, with GT volume ratio in parentheses:

| mode | maxillary R/L | nasal_cavity R/L | nasopharynx |
|---|---:|---:|---:|
| occupancy @ 0.5 | 0.942 / 0.945 (1.02) | **0.859 / 0.859** (1.09) | 0.953 |
| occupancy @ 0.3 | 0.936 / 0.938 (1.09) | 0.849 / 0.844 (1.26) | 0.943 |
| occupancy @ 0.1 | 0.909 / 0.907 (1.20) | 0.802 / 0.788 (1.51) | 0.909 |
| nearest | 0.879 / 0.888 (1.00) | **0.670 / 0.687** (1.00) | 0.908 |

The nasal cavities are the binding class at ~0.86 — read their Dice against that ceiling,
not against 1.0. The same crop-space-vs-native-space caveat as FLARE22 applies (see
`docs/datasets/flare22.md` §7); `NasalSegProvider.native_meta()` supplies what a native-space
metric needs.

# Datasets — summary

Quick-reference index for this directory. For the full narrative (why a source was picked,
provenance decisions, eval numbers across checkpoints over time), see
`eval_expansion_status.md`; for the original tiered candidate survey, see
`eval_strategy_report.md`.

## In-context eval sources, integrated + eval'd

| dataset | modality | target class(es) | n | doc |
|---|---|---|---:|---|
| TotalSegmentator | CT | 117 organs (training vocabulary) | -- | primary training/eval source, no dedicated doc here |
| FLARE22 | CT | 13 abdominal organs (in-vocabulary subset) | 50 | `flare22.md` |
| NasalSeg | CT | 5 nasal/sinus cavities (air-filled, OOD class) | 107 unique cases | `nasalseg.md` |
| HU\_LWK1 | CT | L1-vertebra HU-measurement ROI (non-object) | 36 | `hu_lwk1.md` (local NFS cohort) |
| ISLES22 | MRI (DWI) | stroke lesion (OOD) | 250 | `isles22.md` |
| Shifts-MS | MRI (FLAIR) | MS lesion (OOD) | 46 | `shifts_ms.md` |
| MSD Hippocampus | MRI (T1) | hippocampus anterior/posterior (OOD) | 260 | `msd_hippocampus.md` |
| MSD Prostate | MRI (T2+ADC) | prostate PZ/TZ zones (OOD) | 32 cases (64 channel-split subjects) | `msd_prostate.md` |
| ATLAS v2.0 | MRI (T1) | chronic stroke lesion (OOD) | 654 | `atlas_v2.md` — ⚠ provenance: unofficial mirror |
| GNC\_705 | MRI (Dixon) | 14 kidney-lesion classes (OOD) | 610 labeled (subject, visit) cases | `gnc_kidney_lesions.md` (local restricted cohort) |

## Characterized, not yet wired into the harness

| dataset | modality | target class(es) | n | doc |
|---|---|---|---:|---|
| AMOS22 | CT+MRI | 15 abdominal organs (in-vocabulary control, not OOD) | 360 | `amos22.md` |
| ACDC | cine-MRI | LV/RV/myocardium (OOD) | 150 subjects × 2 phases | `acdc.md` |
| AutoPET III (Lite mirror) | PET/CT | whole-body tumor lesion (OOD modality: PET) | 1038 | `autopet_iii.md` |
| crossMoDA | MRI (ceT1) | vestibular schwannoma, cochlea (OOD) | 105 labeled | `crossmoda.md` |
| BraTS 2024 | MRI | tumor sub-regions / GTV, 3 tracks (OOD) | 2728 | `brats2024.md` — ⚠⚠ stronger provenance caveat (2 of 3 tracks admit circumventing the official DUA) |

## Synthetic data generators (training-data sources, not eval datasets)

| doc | purpose |
|---|---|
| `controlSynth.md` | 2D difficulty-controlled synthetic generator spec — task difficulty disentangled from diversity/quantity |
| `controlSynth_difficulty_findings.md` | Sensitivity study of controlSynth's difficulty knobs, against a frozen UniverSeg baseline |
| `synthgen_maisi.md` | MAISI (NV-Generate-CTMR) diffusion model as an on-the-fly 3D synthetic generator; latent-bank rendering investigation |

## Still gated, no path found

FeTA (Synapse team-join gate, no open mirror found) and PPMI/ADNI-PET (formal DUA) — see
`eval_expansion_status.md` for details.

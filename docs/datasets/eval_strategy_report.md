# Evaluation Datasets for In-Context / Few-Shot 3D Medical Image Segmentation: A Prioritized Guide

## TL;DR
- The most defensible evaluation strategy is to **reproduce the held-out splits already used by the two 3D in-context learning (ICL) papers closest to your work** — Medverse (arXiv:2509.09232, your 2D/3D ICL baseline family) and IRIS/"Show and Segment" (CVPR 2025) — and then layer on lesion, cross-modality, and non-human benchmarks that a fixed-class supervised foundation model structurally cannot address.
- To beat the "just train one supervised foundation model" critique, prioritize datasets whose targets are **genuinely absent from TotalSegmentator's label set** (brain substructures, cochlea/vestibular schwannoma, fetal tissues, mitochondria/cells), **arbitrary lesions** (ULS23, AutoPET, ISLES, MS lesions), and **non-human/microscopy domains** (mouse micro-CT, MitoEM) — these are where in-context learning "earns its keep."
- Beware contamination: TotalSegmentator, AMOS, MSD, and several abdominal CT sets are inside most foundation-model training distributions (and partly inside your own training pool), so use them only as *in-distribution reference points*, never as evidence of generalization.

## Key Findings

**1. Reproduce established ICL held-out suites first.** Reviewers find a reproduced split far more persuasive than a novel one. Medverse organizes its held-out sets into exactly the categories your critic cares about — unseen center (PPMI, FLARE22), unseen organ (NasalSeg), unseen species (Rosenhain mouse micro-CT), unseen modality (ADNI-PET) — using support-set size k=4, 8 repeated random samplings, reported as Dice. IRIS uses ACDC/SegTHOR/IVDM3Seg for domain shift and MSD-Pancreas-Tumor/Pelvic1K for unseen classes, one-shot, Dice.

**2. TotalSegmentator's label set is now enormous, so "unseen class" claims must be checked carefully.** The default `total` CT task covers 117 classes and `total_mr` covers 50; but across *all* tasks and subtasks the tool can now segment **up to 238 CT and 85 MR structures** (v2.6, Jan 2025; Wasserthal et al., arXiv:2512.15921). Subtasks add head/neck glands, cavities, muscles, oculomotor muscles, vertebrae C1–S1, ribs, appendicular bones, lung nodules, liver tumor, kidney cysts, cerebral hemorrhage, hip implants, breast, and liver segments. Structures that remain genuinely *unseen* include brain substructures (hippocampus, thalamus, cortex, putamen, ventricles), cochlea/vestibular schwannoma, fetal tissues, prostate zones, laryngeal cartilage subparts, and all microscopy/cell targets.

**3. Lesion/pathology datasets are the cleanest structural argument.** A fixed-class model has no output channel for an arbitrary lesion; ICL can define the target from a single example. ULS23 (universal CT lesion), AutoPET III/IV (FDG+PSMA PET/CT), ISLES'22 (stroke), MS-lesion sets, and BraTS are the field standards.

**4. Non-human and microscopy domains are the strongest ICL arguments.** Mouse micro-CT and MitoEM electron-microscopy volumes lie entirely outside any human-CT foundation model's competence, making them the sharpest demonstration of context-only generalization.

## Recommended datasets — summary table (priority tiers)

| Dataset | Modality | Cases | Labels | Argument | Access | Why this one |
|---|---|---|---|---|---|---|
| **Medverse held-out suite** (PPMI, FLARE22, NasalSeg, Rosenhain mouse µCT, ADNI-PET) | T1 MRI / CT / CT / µCT / PET | 220 / 50 / 130 / 40 / 589 | cortex, hippocampus, thalamus / liver, spleen, kidney / maxillary sinus, nasopharynx / mouse lung / lateral ventricle | (a)(c)(d) | mixed (open + registration for ADNI/PPMI) | Reproduces your closest 3D-ICL baseline's exact protocol (k=4, ×8, Dice) |
| **IRIS held-out suite** (ACDC, SegTHOR, IVDM3Seg, MSD-Pancreas-Tumor, Pelvic1K) | MRI / CT / MRI / CT / CT | ~100 / 40 / ~16 / 281 / ~150 | cardiac / thoracic OAR / disc / pancreas tumor / pelvic bone | (a)(b)(c) | open / registration | Reproduces CVPR 2025 ICL split for domain shift + unseen class |
| **crossMoDA** | MRI (ceT1→hrT2) | 105 + 105 + 137 | vestibular schwannoma, cochlea | (a)(c) | open (grand-challenge) | Unseen class + extreme modality gap in one benchmark |
| **ULS23** | CT | 38,693 train lesions; 775 test (284 pts) | universal lesion (lung, liver, kidney, pancreas, colon, bone, node) | (b) | open (grand-challenge) | The de-facto 3D universal-lesion benchmark |
| **AutoPET III/IV** | FDG & PSMA PET/CT | 1,611 studies | tumor lesions | (b)(c) | open (TCIA/challenge) | Multi-tracer, multi-center lesion generalization |
| **FeTA** | fetal brain MRI (incl. 0.55T) | ~120+ | 7 fetal tissues | (a)(e) | open (grand-challenge) | Atypical/developing anatomy, unseen tissues |
| **MitoEM / MitoEM 2.0** | electron microscopy | 2 large volumes + collection | mitochondria instances | (d)(f) | open | Organelle instance seg, wholly outside radiology (AP metric) |
| **ISLES'22** | MRI (DWI/ADC/FLAIR) | 250 public (400 total) | stroke lesion | (b)(c) | open (Zenodo) | Multi-vendor lesion, high variability |
| **ATLAS v2.0** | T1 MRI | 1,271 | chronic stroke lesion | (b) | open (registration) | Large chronic-lesion set |
| **CVPR MedSegFM / BiomedSegFM** | CT/MRI/3DUS/PET/microscopy | 35,792 vol, 68 sub-datasets | up to 243 classes | (b)(c)(d)(f) | open (challenge) | The 2025 5-modality generalist benchmark |
| **M&Ms / M&Ms-2** | cardiac cine-MRI | 375+ | LV, RV, myocardium | (c) | open (registration) | Multi-centre/vendor domain shift |
| **AMOS** (CT+MRI) | CT/MRI | 500 CT + 100 MRI | 15 abdominal organs | (c); in-distribution ref | open (Zenodo) | Cross-modality organs — but contamination risk |
| **Mouse micro-CT** (Rosenhain / subcutaneous-tumor) | µCT | 225 / 452 | organs / tumor | (d) | open (Nature Sci Data) | Species shift, strongest ICL case |

## Details

### (a) Held-out / unseen anatomical classes
TotalSegmentator's expanding label set means many "obvious" targets are actually covered, so verify before claiming novelty. Verified genuinely-unseen targets and their datasets:
- **Brain substructures** — TotalSegmentator labels only a single "brain" region; hippocampus, thalamus, cortex, putamen, and lateral ventricle are unseen. Sources: **PPMI** (used by Medverse; 220 T1 scans) and the neuroimaging sets used by **Neuroverse3D** (ICCV 2025, arXiv:2503.02410), which reports Dice gains "exceeding 20 percentage points for targets such as hippocampus, thalamus, lateral ventricle, and putamen" over other ICL models (context size 8; 43,674 scans / 19 datasets).
- **Cochlea & vestibular schwannoma** — **crossMoDA**; neither structure is in TotalSegmentator's label set.
- **Nasal/paranasal sub-structures** — **NasalSeg** (130 CT; maxillary sinus, nasopharynx), Medverse's "unseen organ" split. *Partial-overlap caution:* TotalSegmentator's `head_glands_cavities` subtask labels nasal_cavity, nasopharynx, oropharynx, hypopharynx — so pick NasalSeg targets outside that subtask (e.g., maxillary sinus) to keep the "unseen" claim clean.
- **Fetal brain tissues** — **FeTA** (7 tissues: eCSF, GM, WM, ventricles, cerebellum, brainstem, deep GM).
- **Intervertebral disc** — **IVDM3Seg** (via IRIS); note `total_mr` *does* include "intervertebral discs," so this is only unseen for a CT-trained model.

### (b) Pathology / lesion segmentation
- **ULS23** — 38,693 training lesions across chest-abdomen-pelvis CT (incl. pancreatic, colon, bone lesions); test = 775 lesions from 284 patients; published baseline "achieved an average Dice coefficient of 0.703 ± 0.240 on the challenge test set" (de Grauw et al., arXiv:2406.05231; *Medical Image Analysis*, 2025).
- **AutoPET III** — 1,611 co-registered PET/CT studies = 1,014 FDG (UKT Tübingen; 501 melanoma/lymphoma/lung-cancer + 513 negative controls) + 597 PSMA (LMU Munich prostate carcinoma); hidden test = 200 studies split 50/50 same-center vs cross-center. AutoPET IV adds interactive foreground/background clicks.
- **ISLES'22** — 400 multi-vendor MRI (250 public), acute/subacute stroke lesions.
- **ATLAS v2.0** — 1,271 T1 chronic-stroke MRIs with manual lesion masks.
- **MS lesion** — **MSLesSeg** (115 scans / 75 patients, T1/T2/FLAIR), **MS3SEG** (100 patients, tri-mask: ventricles / normal WMH / abnormal WMH), **MSSEG**.
- **BraTS** — brain-tumor sub-regions (edema, enhancing, non-enhancing); ~484+ multi-sequence MRI.
- **MSD tumor tasks** — liver tumor, lung tumor, pancreas tumor, colon cancer, hepatic vessel; **LiTS / KiTS** — liver & kidney tumor.
- *TotalSegmentator caveat:* `liver_tumor`, `lung_nodules`, and `kidney_cyst` subtasks exist, so those specific lesion types are partly seen; prefer metastatic nodes, pancreatic/colon lesions, stroke, and MS lesions for the cleanest argument.

### (c) Modality / domain shift
- **crossMoDA** — ceT1 → hrT2, an "extreme example of domain shift"; training = 105 annotated ceT1 + 105 unpaired hrT2, test = 137 hrT2; metrics DSC + ASSD. Best single benchmark for a clean cross-modality claim.
- **IRIS domain-shift set** — ACDC (cardiac MRI), SegTHOR (thoracic CT), IVDM3Seg (in/opposed/fat MRI phases).
- **M&Ms / M&Ms-2** — multi-centre, multi-vendor cardiac cine-MRI (375+ subjects, four vendors).
- **AMOS Task 2** — CT & MRI cross-modality abdominal organs.
- **PROMISE12** — multi-vendor prostate MRI.
- **FeTA 2024** — adds low-field (0.55T) data for scanner shift.

### (d) Species / non-human / non-radiology
- **Mouse micro-CT** — Rosenhain et al. (225 native + contrast-enhanced µCT with whole-body organ segmentations, *Nature Scientific Data*); subcutaneous-tumor µCT database (452 scans / 223 mice with 3-annotator masks). Medverse uses a 40-scan mouse set for "unseen species" (target: mouse lung).
- **MitoEM / MitoEM 2.0** — 3D mitochondria instance segmentation from EM (human + rat cortex, ~40K instances); MitoEM 2.0 adds multi-species/multi-modality vEM (FIB-SEM, SBF-SEM, ssSEM). Uses instance AP, not Dice.
- **CVPR MedSegFM microscopy track** — 286 training / 83 validation microscopy volumes.
- **Cell Tracking Challenge / EMPIAR** — additional public microscopy volumes worth mining.

### (e) Pediatric / atypical / post-surgical anatomy
- **FeTA** — fetal (developing) brain, normal + abnormal neurodevelopment; the most reliable public atypical-anatomy option.
- **Hip implant** — TotalSegmentator's `hip_implant` subtask has a public training set (Zenodo) but is therefore *seen*; source implants elsewhere if you need them unseen.
- Publicly available pediatric whole-body segmentation sets remain sparse; FeTA is the safest choice.

### (f) User-defined / arbitrary structures & interactive benchmarks
- **CVPR 2025 MedSegFM / BiomedSegFM** (interactive + text-guided tracks): 35,792 preprocessed 3D volumes from 68 sub-datasets across 5 modalities (CT 24,786; MRI 8,975; 3DUS 1,122; PET 623; Microscopy 286 training), up to 243 classes; coreset track = a random 10% of training cases.
- **IMed-361M** (CVPR 2025) — 6.4M images, 14 modalities, 204 targets, interactive.
- **Promptable baselines to compare against:** ScribblePrompt, nnInteractive, SAM-Med3D, MedSAM, SegVol, VISTA3D.

### Established ICL/universal benchmarks & protocols (2023–2026)
- **UniverSeg** (ICCV 2023) — MegaMedical training; held-out incl. PanDental, WBC, STARE, SpineWeb; 5 predictions/subject with random support sets, Dice, subject bootstrapping (1,000 reps).
- **IRIS / Show and Segment** (CVPR 2025) — 12 in-distribution + 7 held-out; one-shot; Dice.
- **Medverse** (arXiv:2509.09232) — 22 train + 5 held-out; k=4, 8 repeated samplings; **Dice + PSNR (no NSD)**.
- **Neuroverse3D** — neuroimaging ICL; variable 3D context sizes.
- **MultiverSeg / Tyche / ScribblePrompt / Neuralizer** — interactive/in-context variants; MultiverSeg reports on 12 evaluation datasets unseen during training.
- **Metric norm:** Dice + NSD/surface Dice (TotalSegmentator itself reports Dice + NSD@3mm); support size k typically 1–16; repeated support sampling 5–8× with mean ± std.

### Large aggregated collections to mine
- **Medical Segmentation Decathlon** — 10 tasks (brain, heart, hippocampus, liver, lung, pancreas, prostate, colon, hepatic vessel, spleen), CC-BY-SA 4.0.
- **AMOS**, **AbdomenCT-1K/FLARE**, **BTCV/BCV**, **CHAOS**, **WORD**, **RAOS**, **CT-ORG**, **SAROS**, **TotalSegmentator-MRI**, **MRSegmentator**.
- Portals: **TCIA**, **OpenNeuro**, **Grand Challenge**, **EMPIAR**, **Cell Tracking Challenge**.

## Practical caveats
- **Contamination / leakage:** TotalSegmentator was built from clinical-routine CT and is widely used to pseudo-label other datasets. Medverse's own training pool includes TotalSegmentator, AMOS22, RAOS, MSD, BraTS, ISLES2022, and ATLAS — so evaluating on these tests in-distribution behavior, not generalization. AMOS, MSD, BTCV, LiTS, and KiTS overlap heavily with most foundation-model training corpora. Maintain a strict wall between anything in *your* training pool and your eval sets, and state it explicitly.
- **Label-definition mismatch:** organ boundary conventions differ across datasets (e.g., MRSegmentator flags AMOS "bladder" as mis-annotated; "kidney" vs "kidney+cyst"; liver with/without vessels). Cross-dataset Dice can be confounded by definition, not model quality.
- **Tiny structures / class imbalance:** cochlea, vessels, and small lesions make Dice unstable; report NSD/surface Dice and lesion-wise detection alongside Dice.
- **Instance vs semantic:** MitoEM and cell datasets require instance metrics (AP), not Dice.
- **Metric comparability:** Medverse reports Dice + PSNR (not NSD); to compare directly to it, report Dice at k=4 with repeated samplings.

## Recommendations
1. **Tier 1 (do first):** Reproduce the **Medverse held-out suite** (PPMI, FLARE22, NasalSeg, mouse µCT, ADNI-PET) at k=4 with 8 repeated random support samplings, Dice. This directly rebuts the "supervised model is enough" critique across four generalization axes and is numerically comparable to your closest baseline. Add the **IRIS held-out suite** (ACDC, SegTHOR, IVDM3Seg, MSD-Pancreas-Tumor, Pelvic1K), one-shot, as a second established reference protocol.
2. **Tier 2 (lesions + cross-modality):** Add **ULS23**, **AutoPET III**, **crossMoDA**, and **ISLES'22** — arbitrary lesions and severe modality gaps that a fixed-class model has no channel for. Report both Dice and lesion-wise detection.
3. **Tier 3 (non-human / microscopy):** Add **mouse micro-CT** and **MitoEM** to show the extreme end of context-only generalization; use AP for the EM instance task.
4. **Reference (in-distribution only):** Use AMOS / MSD / TotalSegmentator to demonstrate you match supervised models on *seen* classes — but label these clearly as in-distribution controls, not generalization evidence.
5. **Benchmarks that would change the plan:** If your model trails task-specific nnU-Nets on Tier 1 unseen classes by more than ~10 Dice, the ICL-vs-supervised argument weakens for known anatomy — pivot emphasis toward lesions and non-human domains where no supervised competitor exists at all. Conversely, if you match or approach supervised baselines on lesions/species, make that your headline result, since that is precisely the regime a single direct foundation model cannot serve.

## Caveats
Some figures (notably IRIS per-dataset case counts) are deferred to that paper's supplementary material and are approximate here. Several challenge test sets (ULS23, AutoPET, crossMoDA, M&Ms) are hidden and are evaluated via submission portals or on public training splits only. Micro-CT and EM datasets require instance-aware metrics rather than Dice. Neuroimaging repositories (ADNI, PPMI, UK Biobank, OASIS, ADHD-200, GSP) require data-use agreements or application, unlike the openly downloadable challenge sets. Verify current licensing at each dataset page before publication, as terms change over time.
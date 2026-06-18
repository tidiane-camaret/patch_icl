# 2D in-context segmentation — evaluation benchmark candidates

Specs for a diverse, well-balanced, multi-modality / multi-shape **2D** benchmark to evaluate
in-context segmentation (UniverSeg-style) more clearly than raw MedSegBench.

## Why a new benchmark / protocol

The MedSegBench UniverSeg gap (0.33 vs 0.55, see `docs/logs.md` 2026-06-17) was **not** a model
or normalization issue — it was **sample weighting**. Our eval emits one sample per
`(image, label_value)`, so an 18-organ dataset (`m2caiseg`) supplied 42% of all eval samples
(mostly near-zero rare organs) and dominated the micro-average.

**Design principle (applies to any dataset below):** evaluate on a **(modality × shape-class)
grid**, sample N tasks per cell with K context each, and report **macro-Dice over cells** (not a
flat per-sample micro-average). This makes the headline number robust to per-task count imbalance.

Target axes of diversity:
- **Modality:** CT, MRI, X-ray, ultrasound, endoscopy, fundus, dermoscopy, microscopy,
  histopathology, (PET).
- **Shape class:** tubular (vessels/angiography), blob/instance (nuclei, cells), organ
  (CT/MRI abdominal, cardiac), lesion (dermoscopy, breast US), cavity/region (polyps, lungs).

---

## Candidate benchmarks

### 1. MegaMedical + Tyche held-out protocol  *(most directly comparable)*
- **What:** the assembled-from-public-sources task collection used by UniverSeg (MegaMedical, ~16
  modalities / 26 domains / 53 datasets) and the curated held-out eval used by **Tyche**
  (CVPR 2024, same MIT lineage): ACDC (cardiac MRI), AMOS (abdominal CT/MRI), BUID (breast US),
  BBBC003/BBBC038 (microscopy), BRATS / BrainDev (brain MRI), WBC (microscopy), PanDental (X-ray),
  STARE/fundus, etc.
- **2D:** yes (slice-based).
- **Balance:** balanced *by task* by construction; recognized reference protocol.
- **Pros:** maximal comparability to UniverSeg/Tyche; reviewers know it.
- **Cons:** not a single download — assembled via their scripts from many public datasets
  (licenses/access vary).
- **Links:** Tyche https://arxiv.org/abs/2401.13650 ; UniverSeg https://arxiv.org/abs/2304.06131

### 2. BiomedParseData / BiomedParse  *(best for clean balanced reporting)*
- **What:** Nature Methods 2024 (Microsoft). 9 modalities, **64 major object types / 82
  subtypes**, ontology = {histology, organ, abnormality}. Held-out test = **102,855
  image–mask–label triples**. Includes text labels.
- **2D:** yes (parses 2D images / slices).
- **Balance:** the ontology gives a ready modality × object-type grid → straightforward
  macro-averaging; no single dataset can dominate if you weight per cell.
- **Pros:** public, well-curated, structured; great for per-(modality,structure) reporting.
- **Cons:** prompt/text-conditioned design — adapt to our image+mask in-context format.
- **Links:** paper https://www.nature.com/articles/s41592-024-02499-w ;
  arXiv https://arxiv.org/html/2405.12971v1 ; code https://github.com/microsoft/BiomedParse

### 3. COSMOS 1050K (SAT)  *(explicit object-modality balance)*
- **What:** **18 modalities, 84 objects, 125 object-modality paired targets**, ~1.05M 2D images,
  ~6.03M masks.
- **2D:** yes.
- **Balance:** "object-modality paired targets" *is* our (modality × shape) axis → naturally
  supports balanced per-target eval.
- **Pros:** very broad; explicit target taxonomy aligned to our grid idea.
- **Cons:** large; build a balanced subsample/split.
- **Links:** SAT / arXiv https://arxiv.org/pdf/2312.17183

### 4. SA-Med2D-20M  *(largest native-2D pool)*
- **What:** OpenMedLab. **4.6M 2D images / 19.7M masks**, **10 modalities** (CT, MR, X-ray, US,
  endoscopy, PET, fundus, dermoscopy, microscopy, histopathology), 200+ categories.
- **2D:** yes (native 2D).
- **Balance:** raw distribution is very skewed → must subsample a balanced split ourselves.
- **Pros:** breadth of modalities and structures; single curated release.
- **Cons:** size + imbalance; do the cell-wise capping ourselves.
- **Links:** arXiv https://arxiv.org/pdf/2311.11969 ;
  card https://github.com/openmedlab/dataset/blob/main/SA-Med2D-20M.md

### 5. IMed-361M / IMIS-Bench  *(largest multimodal; interactive)*
- **What:** CVPR 2025 (uni-medical). **6.4M images / 273.4M masks, 14 modalities.**
- **2D:** yes.
- **Balance:** large and broad; benchmark is geared to **interactive** (click/box) prompting.
- **Pros:** scale and modality coverage.
- **Cons:** protocol is interactive, not image+mask in-context — more adaptation needed.
- **Links:** https://github.com/uni-medical/IMIS-Bench

### 6. MedSegBench  *(current — keep as a sanity baseline)*
- **What:** Scientific Data 2024. 35 datasets, ~12 modalities, native 2D `.npz`.
- **Issue:** not the data — the *protocol*. Fixable with macro-averaging per (dataset,label) +
  per-dataset/label caps; the multi-organ datasets (`m2caiseg`, `abdomenus`) otherwise dominate.
- **Links:** https://www.nature.com/articles/s41597-024-04159-2

---

## Recommendation

1. **For comparability** to UniverSeg/Tyche → adopt the **Tyche/MegaMedical held-out task
   protocol**.
2. **For a clean, balanced, turnkey benchmark** → **BiomedParseData** (or **COSMOS 1050K**), scored
   on an explicit **(modality × shape-class) grid with macro-Dice**.
3. Keep a **macro-averaged MedSegBench** as a cheap regression sanity check.

Regardless of choice, fix the *protocol* first: macro-average per cell + per-task sample cap.

## Suggested grid (starting point)

| Shape class \ Modality | CT/MRI | Ultrasound | Endoscopy | Fundus/Retina | Dermoscopy | Microscopy/Histo | X-ray |
|---|---|---|---|---|---|---|---|
| Organ / region | abdominal, cardiac | — | — | — | — | — | lungs (CXR) |
| Lesion / tumor | brain tumor (BRATS) | breast (BUID/busi) | — | — | skin (ISIC) | — | — |
| Tubular / vessel | — | — | — | vessels (DRIVE/STARE/CHASE) | — | — | angiography (DCA1/CHUAC) |
| Blob / instance | — | — | — | — | — | nuclei/cells (MoNuSAC, TNBC, cellnuclei, BBBC) | — |
| Cavity / structure | — | — | polyps (Kvasir, BKAI) | — | — | — | teeth (PanDental) |

Fill each non-empty cell with a few tasks, K context each; report macro-Dice over filled cells
plus a per-modality and per-shape breakdown.

## References

- Tyche (CVPR 2024) — https://arxiv.org/abs/2401.13650
- Show and Segment / Iris (CVPR 2025) — https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Show_and_Segment_Universal_Medical_Image_Segmentation_via_In-Context_Learning_CVPR_2025_paper.html
- BiomedParse (Nature Methods 2024) — https://www.nature.com/articles/s41592-024-02499-w · https://github.com/microsoft/BiomedParse
- SA-Med2D-20M — https://arxiv.org/pdf/2311.11969 · https://github.com/openmedlab/dataset/blob/main/SA-Med2D-20M.md
- COSMOS 1050K / SAT — https://arxiv.org/pdf/2312.17183
- IMIS-Bench / IMed-361M (CVPR 2025) — https://github.com/uni-medical/IMIS-Bench
- MedSegBench (Scientific Data 2024) — https://www.nature.com/articles/s41597-024-04159-2

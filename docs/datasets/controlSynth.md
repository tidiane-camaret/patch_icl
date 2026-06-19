# Difficulty-Controlled Synthetic Data Generator for In-Context Medical Segmentation

**Version:** 1.0 (spec)
**Purpose:** On-the-fly synthetic (target + k-context) generation for training *and*
reproducible evaluation of in-context segmentation models, with explicit, calibrated
control over **task difficulty** — disentangled from task *diversity* and *quantity*.

**Image size:** `128 × 128` (default)
**Distractor regions:** `16` (default `num_labels`)
**Precompute:** base geometries built offline → LMDB + LZ4 store
**Deformation:** CPU, in dataloader workers (`scipy.ndimage.map_coordinates`)

---

## 1. Motivation and scope

In-context learning (ICL) for segmentation conditions a prediction on a support set of
(image, mask) pairs rather than on fixed weights. Whether a model learns the *general*
in-context mechanism — versus shortcutting to in-weight memorization — is known to depend
on properties of the training task distribution. Two drivers recur in the literature:

1. **Task diversity** — the number of distinct rules/anatomies seen during training.
   There is evidence for a *threshold* effect: below some diversity the model learns
   task-specialized solutions; above it, a general in-context solution emerges.
2. **Task difficulty** — how hard an individual task is. When the in-weight task is too
   easy, a simplicity bias suppresses ICL; harder tasks push the model onto the context.

Existing synthetic generators (SynthSeg-family domain randomization; UniverSeg's random
shapes) expose difficulty-relevant knobs but **randomize them for diversity** and never
sweep a single one as a controlled variable while holding diversity and quantity fixed.
This generator is built specifically to make that controlled study possible, and to do so
at a dataloading throughput that does not bottleneck training.

### Non-goals (V1)

- 3D / volumetric data.
- Realistic intensity *appearance* matching a specific modality (we use randomized GMM
  intensities; realism is not the objective — controllability is).
- Learned/generative (GAN/diffusion) synthesis. This is a procedural generator by design,
  because procedural control is what makes difficulty calibratable.

---

## 2. Core design principle: three orthogonal axes

The API enforces a separation that most prior work conflates. Any one axis can be swept
while the other two are pinned.

| Axis | Knob(s) | Is it "difficulty"? |
|------|---------|---------------------|
| **Diversity** | `num_tasks` | No — number of distinct rules |
| **Quantity** | `epoch_length` | No — samples per epoch |
| **Difficulty** | per-task properties (below) | Yes |

If these are not separable in code, the central experiment (difficulty vs. generalization,
controlling for diversity) cannot be run. This separation is the deliverable's reason to exist.

---

## 3. Difficulty decomposes into two sub-axes

Grounded in the ICL mechanism (a similarity/look-up function + a knowledge-aggregation
"previous-token"-style head), a task is hard for two independent reasons.

### 3.1 Axis A — Identification difficulty (in-context-specific)

How hard it is to **infer the rule from the support set**. This axis controls whether the
task requires context at all. It is the scientifically central axis for ICL.

| Factor | Meaning | Build/Live | Hard direction |
|--------|---------|------------|----------------|
| `task_ambiguity` | distractor regions sharing an attribute (e.g. intensity) with the true foreground, so the rule cannot be read off a single image | **build** (geometry) + **live** (intensity assignment) | more/closer distractors |
| `support_query_shift` | deformation + intensity gap between context subjects and query | live | larger shift |
| `context_copy_fraction` | fraction of context entries that are near-copies of the query (instCopy lever) | live | lower fraction |
| `context_consistency` | probability a context pair is **not** label-corrupted (label-swap analog) | live | lower consistency |

At `task_ambiguity ≈ 0` the foreground is uniquely identifiable from appearance alone →
the task is solvable without context → it is **not** an ICL task. As ambiguity rises the
model is forced onto the support set. `task_ambiguity` is the primary knob of the whole
generator (confirmed as central in design discussion).

### 3.2 Axis B — Segmentation difficulty (rule-given)

Given the rule, how hard the **pixels** are. An oracle task-specific UNet's accuracy moves
with this axis.

| Factor | Meaning | Build/Live | Hard direction |
|--------|---------|------------|----------------|
| `shape_morphology` | typed family: `blob`, `elongated`, `tubular` (vessel-like), `annular`, `scattered` | **build** | tubular / scattered |
| `thinness` | min caliber of tubular structures (within-`tubular` driver) | **build** | thinner |
| `region_size` | target foreground area fraction (log-scaled), enforced at generation | **build** | smaller |
| `boundary_complexity` | contour roughness, morphology held fixed | **build** | rougher |
| `foreground_contrast` | min intensity-mean separation between foreground and neighbors | live | lower |
| `noise_level` | Gaussian + Perlin magnitude | live | higher |
| `texture_heterogeneity` | within-region intensity variance / sub-texture | live | higher |

### 3.3 Cross-axis coupling (must be measured, not assumed)

- `scattered` morphology loads on **both** axes: it raises segmentation difficulty (many
  small components) *and* identification difficulty (the support must teach "all such
  components, everywhere," not "the one region here").
- `foreground_contrast → 0` couples into Axis A: when the region is intensity-invisible,
  only shape/position identify it.

The calibration protocol (§9) is what *quantifies* these loadings via the oracle-vs-
in-context gap. Each task's record stores an `axis_loadings` field so analysis is honest
about which axis a knob actually moved.

---

## 4. Build vs. live parameter split (consequence of precompute-to-disk)

Because base geometry is frozen at build time, every parameter affecting **geometry** is
resolved in `precompute.py` and **cannot** be changed by the dataloader:

- **[build]:** `shape_morphology`, `thinness`, `tortuosity`, `branching_density`,
  `region_size`, `boundary_complexity`, scattered `count`/`clustering`, distractor layout
  for `task_ambiguity`.
- **[live]:** `support_query_shift` (deformation), `foreground_contrast`,
  `texture_heterogeneity`, `noise_level`, `context_copy_fraction`, `context_consistency`,
  and the *intensity* component of `task_ambiguity` (which distractor shares the fg mean).

Keeping this split explicit prevents silent bugs where a "difficulty sweep" secretly
requires rebuilding the store.

---

## 5. Configuration API

Three independent config objects, so experimental design is explicit in code.

```python
@dataclass
class DiversityConfig:
    num_tasks: int = 1000
    num_labels: int = 16          # distractor/background regions behind the foreground
    context_size: int = 3
    master_seed: int = 42
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1)  # train / val / test task pools

@dataclass
class DifficultyBuildSpec:
    """Build-time difficulty (frozen into the store)."""
    mode: str = "fixed"           # "fixed" | "binned" | "per_task_sampled"
    morphology: str | dict = "blob"   # type, or {type: weight} mixture, or per-bin spec
    thinness: float = 0.5             # [0,1] -> min caliber
    tortuosity: float = 0.5           # [0,1] -> angular perturbation variance (tubular)
    branching_density: float = 0.5    # [0,1] -> p_branch x max_depth (tubular)
    region_size: float = 0.15         # target foreground area fraction (log-mapped)
    boundary_complexity: float = 0.3  # [0,1] contour roughness
    scattered_count: int = 8          # n components (scattered)
    scattered_clustering: float = 0.0 # [0,1] Poisson(0) -> clustered(1)
    task_ambiguity: float = 0.0       # [0,1] geometry side: n + similarity of distractors
    # binning / sampling control
    n_bins: int = 1                   # for mode="binned"
    bin_factor: str = "task_ambiguity"  # which factor the bins sweep

@dataclass
class DifficultyLiveConfig:
    """Live difficulty (applied per subject in the dataloader)."""
    support_query_shift: float = 0.3
    foreground_contrast: float = 0.5
    texture_heterogeneity: float = 0.2
    noise_level: float = 0.3
    context_copy_fraction: float = 0.0
    context_consistency: float = 1.0
    task_ambiguity_intensity: float = 0.0  # live side of ambiguity

@dataclass
class SamplingConfig:
    epoch_length: int = 10000
    deterministic: bool = False        # False=train (infinite subjects), True=eval
    eval_seed_namespace: int | None = None
```

Each `[0,1]` factor has a **documented monotone mapping** to its underlying generation
parameter; midpoints are anchored so that defaults reproduce baseline behavior
(e.g. `noise_level: 0→σ=0.0, 1→σ=0.25`). Provide:

- `difficulty(scalar)` — drive all factors along one calibrated curve (coarse sweeps).
- factor-wise control — sweep one factor with all others pinned (ablations).
- `mode="binned"` — emit difficulty-bin labels for reproducible eval grids.

---

## 6. Determinism model

A **task** is `(task_seed, fg_label, build_difficulty)`.
A **subject** is `(task, subject_seed)`.

- **Train (`deterministic=False`):** subject seed drawn from fresh entropy → infinite
  subject diversity.
- **Eval (`deterministic=True`):** subject seed = deterministic function of
  `(eval_seed_namespace, task_seed, sample_index)` → byte-identical test set across runs.

All live transforms (deformation, intensity, noise) consume RNG **only** through the
subject seed — no hidden global RNG. This is what makes evaluation reproducible (the
original loader's `np.random.default_rng()` per call broke this).

Base geometry is deterministic in `(master_seed, task_id, build_difficulty)` and identical
whether produced offline (`precompute.py`) or lazily on a cache miss.

---

## 7. Train / eval split

The `num_tasks` anatomy seeds are partitioned into disjoint **train / val / test** pools
(dataset-level split, à la UniverSeg). Evaluation tasks are therefore **unseen anatomies**.
The eval harness crosses:

```
{held-out task pool} × {difficulty bins} × {fixed subject seeds}
```

Critically, eval bins sweep the **identification** and **segmentation** axes *separately*,
not only a joint scalar — that separation is the scientific payload.

---

## 8. Module layout

```
synth/
  shapes/
    __init__.py
    blob.py          # blob, elongated, annular
    vessel.py        # branching tree -> caliber -> rasterize   (EXPENSIVE; build-time only)
    scattered.py     # point process -> components              (build-time)
    boundary.py      # contour roughness, decoupled from morphology
    area.py          # enforce_area_fraction (scale-to-target)
    distractors.py   # task_ambiguity geometry injection
  appearance.py      # GMM intensity fill; noise-bank sampler (Perlin precomputed offline)
  deformation.py     # per-subject CPU warp (the cheap path)
  task.py            # Task dataclass; pure base-geometry function; difficulty resolution
  context.py         # context assembly; copy_fraction, consistency corruption
  precompute.py      # batch-generate + pack base geometries to LMDB; build noise bank + splits
  store.py           # packed read-only LMDB store + LRU decode cache
  dataset.py         # Dataset + collate (original interface preserved) + meta
  eval_harness.py    # held-out x bins x fixed seeds; oracle hooks; Dice + clDice + NSD
  mixed.py           # MixedDataLoader (real+synth) + optional curriculum schedule
```

---

## 9. Performance model (the precompute + CPU-deform contract)

Separate the **expensive, per-task, reusable** part from the **cheap, per-subject** part.

| Stage | Cost | Where | Reuse |
|-------|------|-------|-------|
| base geometry (vessel tree / scattered layout / distractors) | high (tens of ms) | `precompute.py`, offline | once per task, reused by every subject & context entry |
| decode packed record | ~0 on cache hit | `store.py` LRU | per task |
| per-subject deformation | ~1–3 ms | `dataset.py` worker (CPU) | per subject |
| intensity fill + noise | low (vectorized) | `dataset.py` worker | per subject |

**Hard rules:**

- No per-call branching-tree generation. Ever. It is build-time only.
- No per-call Perlin synthesis — sample/crop from a **precomputed noise bank** stored in
  the LMDB. (Per-call Perlin is the classic hidden bottleneck.)
- No Python pixel loops in the live path; everything vectorized (NumPy / SciPy).
- Masks stay `uint8` until the final tensor cast (one late copy in the dataset).
- One LMDB environment opened **lazily per worker** (LMDB handles do not survive `fork`).

**Latency budget** (128×128, k=3 → 4 subjects/sample): dominated by the cheap path; base
geometry contributes ~0 on cache hit and is amortized across thousands of reuses.

---

## 10. File-level specifications

### 10.1 `store.py`

```python
@dataclass
class GeometryRecord:
    label_map: np.ndarray   # uint8 [H,W]; full multi-region map; treat READ-ONLY
    fg_label: int
    meta: dict              # frozen build-time difficulty + realized values + axis_loadings

class GeometryStore:
    def __init__(self, path: str, max_decode_cache: int = 512, readonly: bool = True): ...
    def _env(self):                      # lazy per-worker LMDB env (fork-safe)
    def __len__(self) -> int: ...
    def get(self, task_id: int) -> GeometryRecord:   # LRU-cached decode; 0 cost on hit
    def task_ids(self, split: str) -> list[int]:     # "train" | "val" | "test"
    def noise_bank(self) -> np.ndarray:              # precomputed Perlin fields [N,H,W]
    def difficulty_table(self):                      # one row/task -> DataFrame for analysis
```

- Value = LZ4-compressed msgpack of `{label_map(uint8), fg_label(int), meta(dict)}`.
- Reserved keys hold the split table, the noise bank, and the difficulty table, so the
  store is self-describing and a test grid is reproducible from the file alone.

### 10.2 `precompute.py`

```python
def build_store(out_path, diversity: DiversityConfig,
                difficulty: DifficultyBuildSpec,
                image_size=(128,128), n_jobs=...): ...

def make_base_geometry(image_size, morphology, geo_params, rng):
    """Dispatch to morphology generator -> composite background (num_labels regions)
       + distractors -> enforce area -> apply boundary roughness.
       Returns (label_map uint8 [H,W], fg_label int, realized_meta dict)."""
```

- `resolve_difficulty(spec, task_id, rng)` supports the three modes:
  `fixed` (clean disentanglement run), `binned` (labeled eval grid),
  `per_task_sampled` (mixture / curriculum studies).
- **Record realized difficulty, not requested.** Generators won't hit `region_size` or
  `thinness` exactly; analysis must correlate Dice against what was actually generated.
- Build the Perlin noise bank and the split table here; write them as reserved keys.

### 10.3 `shapes/vessel.py` (anatomically branching — build-time core)

```python
def make_vessel_tree(image_size, params, rng) -> np.ndarray:  # uint8 mask
    """
    params: thinness, tortuosity, branching_density, region_size
    1. centerline = grow_tree(root, rng, persistence~tortuosity,
                              p_branch & max_depth ~ branching_density)
         recommended: SPACE COLONIZATION (natural retinal-looking branching,
         tunable by ~2 params); children inherit direction + bifurcation angle.
    2. caliber per node TAPERS with generation depth (Murray-law-ish);
         min radius at deepest generation == `thinness`.
    3. raster = rasterize_graph_with_caliber(...) via distance-transform threshold
         (vectorized; NO pixel loops).
    4. globally scale calibers to hit `region_size`; record realized area.
    """
def grow_tree(...): ...
def rasterize_graph_with_caliber(...): ...
```

V1 target: **anatomically vessel-like** branching trees (confirmed requirement).

### 10.4 `shapes/scattered.py` (build-time)

```python
def make_scattered(image_size, params, rng) -> np.ndarray:  # uint8 mask
    """
    params: count, size_dispersion, region_size, clustering
    1. centers via point process: hardcore (min-dist) <-> Poisson <-> clustered.
    2. stamp small blobs/ellipses; sizes ~ size_dispersion.
    3. scale to hit region_size; record realized area + component count.
    meta MUST flag dual-axis loading (segmentation + identification).
    """
```

V1: included (confirmed requirement).

### 10.5 `shapes/blob.py`, `boundary.py`, `area.py`, `distractors.py`

- `blob.py`: `blob`, `elongated` (eccentric ellipses), `annular` (ring/shell — interior is
  a built-in distractor).
- `boundary.py`: `set_boundary_complexity(mask, c)` — roughness independent of morphology.
- `area.py`: `enforce_area_fraction(mask, target)` — scale shape to target **before**
  deformation, so size and morphology don't couple through layout.
- `distractors.py`: `inject_distractors(label_map, fg_label, ambiguity, rng)` — adds regions
  that share an attribute with the foreground; count + similarity scale with `task_ambiguity`.

### 10.6 `appearance.py`

```python
def gmm_fill(label_map, fg, contrast, texture, ambiguity_intensity, rng) -> np.ndarray
def add_noise(img, level, noise_bank, rng) -> np.ndarray   # crops from precomputed bank
```

- Each region gets a Gaussian intensity mean; `foreground_contrast` sets min separation of
  the fg mean from neighbors; `task_ambiguity_intensity` chooses how many distractors share
  the fg mean (the live side of ambiguity).
- `texture_heterogeneity` adds within-region variance / sub-texture.

### 10.7 `deformation.py`

```python
def deform(mask_or_img, sigma_def, rng) -> np.ndarray
    # smoothed random displacement field -> scipy.ndimage.map_coordinates
    # CPU; scales with num_workers; leaves GPU for the model
```

`support_query_shift` controls the magnitude of the context↔query deformation gap.

### 10.8 `dataset.py` (live path; original interface preserved)

```python
class SynthICLDataset(Dataset):
    def __init__(self, store_path, split, context_size, image_size,
                 difficulty_live: DifficultyLiveConfig,
                 deterministic: bool, epoch_length, eval_seed_namespace=None): ...
    def _worker_store(self) -> GeometryStore:   # lazy LMDB open in worker
    def __getitem__(self, idx) -> dict: ...
```

`__getitem__` outline:

```
store = self._worker_store()
if deterministic:                       # EVAL
    task_id, subj_seed_base = eval_index_map(eval_seed_namespace, idx, store, split)
else:                                   # TRAIN
    task_id        = rng_choice(store.task_ids(split))
    subj_seed_base = fresh_entropy()

rec  = store.get(task_id)               # cache hit -> ~free
base = rec.label_map                    # uint8, read-only view
fg   = rec.fg_label

def make_subject(seed):
    r      = np.random.default_rng(seed)
    warped = deform(base, difficulty_live.support_query_shift, r)
    img    = gmm_fill(warped, fg, difficulty_live.foreground_contrast,
                      difficulty_live.texture_heterogeneity,
                      difficulty_live.task_ambiguity_intensity, r)
    img    = add_noise(img, difficulty_live.noise_level, store.noise_bank(), r)
    binseg = (warped == fg).astype(np.float32)
    return img, binseg

target_img, target_seg = make_subject(subj_seed_base)
ctx = [make_subject(subj_seed_base + 1 + i) for i in range(context_size)]
ctx = apply_context_difficulty(ctx, target=(target_img, target_seg),
                               copy_fraction=difficulty_live.context_copy_fraction,
                               consistency=difficulty_live.context_consistency)
```

**Output dict (unchanged from the original loader, plus `meta`):**

```python
{
  "image":            FloatTensor [1, H, W],
  "label":            FloatTensor [1, H, W],
  "context_in":       FloatTensor [k, 1, H, W],
  "context_out":      FloatTensor [k, 1, H, W],
  "target_case_id":   str,
  "context_case_ids": list[str],
  "label_id":         str,
  "meta": {                      # NEW — required for analysis
    "task_id": int, "fg": int, "subject_seed": int,
    "difficulty": { ...build fields..., ...live fields... },
    "axis": { "identification": float, "segmentation": float },
  },
}
```

`collate_fn` is preserved as-is, including the variable-`k` padding logic; it forwards
`meta` as a list.

### 10.9 `mixed.py`

- `MixedDataLoader(real_loader, synth_loader, synth_ratio, seed)` — preserved from the
  original loader (real + synthetic batch mixing).
- Optional `DifficultyCurriculumSchedule` — advance the live difficulty (or sample harder
  build-time bins) over training steps, for curriculum studies.

### 10.10 `eval_harness.py`

- Iterates `{held-out task pool} × {difficulty bins} × {fixed subject seeds}`.
- Logs **Dice + clDice + Normalized Surface Dice** (see §11).
- Provides **oracle hooks**: train/evaluate a task-specific UNet per setting to separate
  the two difficulty axes (see §9 calibration).

---

## 11. Evaluation metrics (V1 requirement, not optional)

For vessels and small scattered components, **Dice is fragile and misleading** — a 1-pixel
boundary slip on a thin structure tanks Dice even when the prediction is qualitatively
correct. Because tubular and scattered morphologies are in V1, the harness must log,
alongside Dice:

- **clDice** (centerline Dice) — standard for tubular/branching structures.
- **Normalized Surface Dice (NSD)** at a small tolerance — boundary-aware overlap.
- **Size-stratified Dice** — report by `region_size` bin, since Dice variance explodes at
  small sizes.

Without these, a `thinness` sweep mostly measures Dice's sensitivity to thin shapes rather
than model ability.

---

## 12. Calibration protocol

A knob is not a "difficulty" knob until shown to monotonically lower achievable accuracy.
For each factor, sweep it with all else pinned and measure two references:

1. **Oracle** task-specific UNet trained at that setting → measures **segmentation**
   difficulty (Axis B).
2. **In-context model** itself → the **gap** to oracle isolates **identification**
   difficulty (Axis A).

Expected behavior (a useful sanity table):

| Knob | Oracle accuracy | In-context gap |
|------|-----------------|----------------|
| `task_ambiguity` | ~flat | widens strongly |
| `noise_level` | drops | drops |
| `boundary_complexity` | drops | drops |
| `thinness` | drops (and Dice-fragile → use clDice) | drops |
| `region_size`↓ | drops | drops |
| `context_copy_fraction`↑ | ~flat | **narrows** (difficulty reducer) |
| `context_consistency`↓ | ~flat | widens |
| `scattered count`↑ | drops | widens (dual-axis) |

If a knob is non-monotone, recalibrate its `[0,1]→param` mapping before the main study.
This calibration table is itself a publishable artifact — per the field survey, no one has
produced a controlled difficulty-vs-generalization mapping for in-context medical
segmentation.

---

## 13. Defaults summary

| Parameter | Default |
|-----------|---------|
| `image_size` | `(128, 128)` |
| `num_labels` (distractors) | `16` |
| `context_size` | `3` |
| `num_tasks` | `1000` |
| splits (train/val/test) | `0.8 / 0.1 / 0.1` |
| store backend | LMDB + LZ4 + msgpack |
| deformation | CPU, `scipy.ndimage.map_coordinates`, in workers |
| Perlin noise | precomputed bank in store |
| eval metrics | Dice + clDice + NSD, size-stratified |

---

## 14. Open implementation choices (deferred to reference code, not spec)

- Vessel `grow_tree` algorithm: **space colonization** recommended (natural branching,
  ~2 tunable params); alternatives (random walk, L-system) are acceptable since this is
  build-time only and even a slow version is tolerable. Tune realism against rendered
  samples, not on paper.
- Scattered point process: hardcore vs. Poisson vs. clustered — exposed via `clustering`.

**Suggested first step:** build a tiny store (~50 tasks) and visually inspect the masks
(especially vessels and scattered) before scaling to `num_tasks=1000` and running
calibration.
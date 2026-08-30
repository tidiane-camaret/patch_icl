"""
GPU GMM intensity synthesis (SynthSeg-style): map an integer label map to a continuous
intensity volume by sampling each voxel from a Gaussian component selected by its label.
Component parameters (mu, sigma) are drawn ONCE per cohort and shared across all N
subjects in a call; only the per-voxel noise differs — mimicking one imaging protocol
across a support/query cohort.

Label id convention (from the placement stage):
  0        background / air        (deterministic component: mu=0, sigma=0)
  1 .. L   organ slots             (mu_l ~ U(0,255), var_l ~ U(0,var_max), per cohort)
  L+1      container fill          (same draw as an organ slot)
Ids are blueprint SLOT indices, not anatomical classes — two slots of the same class get
independent means, so intensity carries no class signal (intended domain randomization).
(L = number of organ slots; not to be confused with K, the in-context sample count.)

The whole stage is two gathers + one randn + one FMA, fully vectorized over N. See the
project spec (docs) for the full contract and validation checks.
"""
import numpy as np
import torch
from scipy.special import erf

# TotalSegmentator CT (docs/logs.md 2026-08-29 "Real-data intensity correlation analysis" +
# follow-ups): the between-subject redundancy structure discovered by
# analyze_totalseg_intensity.py's intensity-only clustering (avg-linkage, dist<=0.25,
# min_n=30) on the 116-class debiased fit, translated TotalSeg name -> MAISI id (all 91
# mapped 1:1, no name-based lookup at runtime -- these are frozen from that one analysis).
# A tighter cutoff than the first pass (dist<=0.4, which merged nearly the whole skeleton
# into one 54-id blob) resolves 18 anatomically/physiologically distinct sub-groups instead
# of 4 coarse ones -- most are bilateral left/right pairs of the SAME structure (rho 0.76-
# 0.97, e.g. autochthon_left/right at 0.97) or contiguous vertebral/rib runs, and the
# vasculature now splits into separate ARTERIAL (rho=0.90) and VENOUS (rho=0.88, +a smaller
# brachiocephalic-vein/SVC pair at 0.76) clusters instead of one mixed "vessels" group --
# finer and more physiologically informative than the 4-group summary. rho = MEASURED mean
# within-cluster correlation (not guessed). Order matches CT_GROUP_RHO below:
#   22 ribs (rho=0.83) | 11 arterial (0.90) | 10 cervical/upper-thoracic vertebrae C1-T3 (0.85)
#   8 lower vertebrae L1-5+T10-12 (0.79) | 7 abdominal organs+portal/splenic vein (0.82)
#   6 mid-thoracic vertebrae T4-T9 (0.84) | 4 gluteus max+medius (0.83) | 3 venous/IVC (0.88)
#   2-each: autochthon (0.97) | brachiocephalic-vein+SVC (0.76) | clavicula (0.92) |
#   gluteus minimus (0.85) | hip (0.94) | iliopsoas (0.86) | left lung (0.80) |
#   right lung upper+lower-lobe pair (0.82) | sacrum+vertebrae_S1 (0.82) | scapula (0.85)
# The remaining ~25 core classes (of 116) are left independent -- they were singletons in the
# real data (esophagus, stomach, colon, ... -- content-dependent GI/lumen classes, genuinely
# uncorrelated), as are all non-core MAISI ids outside TotalSegmentator's 116-class analysis.
#
# MEMBERSHIP IS FIXED (these exact ids, every cohort) -- NOT reshuffled. A reshuffled-
# membership version was tried first and rejected: docs/logs.md follow-up 5 shows it collapses
# the real block-diagonal structure (bone-bone high, bone-organ ~0) into one flat, anatomy-
# blind correlation bump applied to every id pair alike, which doesn't reproduce real
# correlations at all. Fixed membership matches the real per-tissue-type redundancy
# faithfully; it does not leak any real intensity VALUE (the shared value each group takes is
# still a fresh random draw every cohort) or a persistent brightness-to-organ mapping -- it
# only encodes "these ids are the same tissue type", true non-value-bearing structure.
CT_GROUP_MAISI_IDS = (
    (64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86),  # 22 ribs
    (6, 58, 59, 108, 109, 112, 113, 115, 119, 123, 124),                       # 11 arterial
    (47, 48, 49, 50, 51, 52, 53, 54, 55, 56),                                  # 10 vertebrae C1-T3
    (33, 34, 35, 36, 37, 38, 39, 40),                                          # 8 vertebrae L1-5+T10-12
    (3, 4, 5, 8, 9, 14, 17),                                                   # 7 abdominal organs
    (41, 42, 43, 44, 45, 46),                                                  # 6 vertebrae T4-T9
    (98, 99, 100, 101),                                                        # 4 gluteus max+medius
    (7, 60, 61),                                                              # 3 venous/IVC
    (104, 105),                                                              # 2 autochthon
    (111, 125),                                                              # 2 brachiocephalic vein+SVC
    (91, 92),                                                                # 2 clavicula
    (102, 103),                                                              # 2 gluteus minimus
    (95, 96),                                                                # 2 hip
    (106, 107),                                                              # 2 iliopsoas
    (28, 29),                                                                # 2 left lung
    (31, 32),                                                                # 2 right lung
    (97, 127),                                                               # 2 sacrum+vertebrae_S1
    (89, 90),                                                                # 2 scapula
)
CT_GROUP_RHO = (0.83, 0.90, 0.85, 0.79, 0.82, 0.84, 0.83, 0.88,
               0.97, 0.76, 0.92, 0.85, 0.94, 0.86, 0.80, 0.82, 0.82, 0.85)

# Cross-modality version: analyze_totalseg_intensity.py --dataset merged pools CT (1228 subj,
# raw HU) and TotalSegmentator-MRI (616 subj, per-subject clip+zscore) subject rows -- each
# dataset's per-class column standardized to ITS OWN between-subject mean/std first (so
# "correlated" means "co-varies the same WAY in a typical subject", not "same absolute
# value" -- CT and MRI share no absolute intensity unit). Only 46/122 classes are covered in
# BOTH datasets (MRI lacks per-rib/per-vertebra/per-lung-lobe splits), so this is necessarily
# coarser than CT_GROUP_MAISI_IDS above -- 6 groups (dist<=0.35, chosen for ~20 total
# clusters incl. singletons, matching CT_GROUP_MAISI_IDS's resolution target) instead of 18.
# rho = the POOLED measured correlation; CT-alone / MRI-alone agree closely for every group
# (docs/logs.md 2026-08-29, "single ~20-cluster fit for both" follow-up), confirming this is
# real shared anatomy, not one modality dominating the pool:
#   10 muscle (pooled 0.72, CT 0.70 / MRI 0.81): autochthon+gluteus x3+iliopsoas
#   8 abdominal organs (0.78, CT 0.79 / MRI 0.77): adrenals/kidneys/liver/pancreas/portal
#                                                   vein/spleen
#   4 arterial (0.85, CT 0.86 / MRI 0.81): aorta/heart/iliac artery
#   4 shoulder girdle (0.76, CT 0.76 / MRI 0.79): clavicula/scapula
#   3 pelvis (0.81, CT 0.80 / MRI 0.87): hip/sacrum
#   3 venous (0.88, CT 0.88 / MRI 0.87): iliac vena/IVC
# Notably does NOT include a "bone" cluster the way CT_GROUP_MAISI_IDS does: femur/humerus
# (long-bone shafts) fail to cluster with anything at this resolution in EITHER modality --
# only flat/girdle bone (clavicula/scapula/hip/sacrum) shows the redundancy, a distinction
# the CT-only per-level-vertebra/rib analysis couldn't surface (see docs/logs.md).
MERGED_GROUP_MAISI_IDS = (
    (98, 99, 100, 101, 102, 103, 104, 105, 106, 107),   # 10 muscle
    (1, 3, 4, 5, 8, 9, 14, 17),                          # 8 abdominal organs
    (6, 58, 59, 115),                                    # 4 arterial
    (89, 90, 91, 92),                                    # 4 shoulder girdle
    (95, 96, 97),                                        # 3 pelvis
    (7, 60, 61),                                         # 3 venous
)
MERGED_GROUP_RHO = (0.72, 0.78, 0.85, 0.76, 0.81, 0.88)

# Real intra-cohort (BETWEEN-MEMBER) intensity variance, TotalSegmentator CT (docs/logs.md
# 2026-08-29 "real intra-cohort variance analysis", analyze_intracohort_variance.py). Today
# mu[c]/sd[c] are drawn ONCE per cohort and shared verbatim by every member -- sd[c] models
# only within-scan voxel TEXTURE, and any organ with more than a few hundred voxels has its
# per-member mean converge to mu[c] (noise averages out), so there is close to zero actual
# member-to-member variability. Real CT splits a class's total spread into
# between_subj_std_hu (real subject-to-subject spread of the class MEAN -- what's missing)
# and within_subj_voxel_std_hu (within-one-scan texture -- what sd[c] already models); their
# RATIO is unit-free (survives synth intensities living on an arbitrary 0-255 scale, not real
# HU) and is NOT constant across classes -- it spans ~0.2 to ~1.6 and is driven by physiology,
# not sample size (corr(n_present, ratio)=0.03 across 115 classes). A coarse 6-family lookup
# (name-pattern grouping, MAISI ids translated via data.class_registry, no per-class tuning)
# already captures most of the real structure (pooled within-family ratio std 0.13 vs 0.29
# overall). Vascular/cardiac is the standout: ratio > 1 means real between-subject variability
# EXCEEDS within-scan texture -- IV-contrast bolus timing shifts a vessel's characteristic HU
# far more than its own lumen texture. Bone/muscle sit at the low end: trabecular/fiber
# texture WITHIN one scan dominates, real density is comparatively stable patient-to-patient.
CT_BETWEEN_WITHIN_DEFAULT = 0.48   # global median ratio across all 115 measured classes
CT_BETWEEN_WITHIN_GROUPS = (
    ((6, 7, 17, 58, 59, 60, 61, 108, 109, 110, 111, 112, 113, 115, 119, 123, 124, 125), 1.11),
    ((1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 62, 118, 126), 0.62),           # organ
    ((28, 29, 30, 31, 32), 0.61),                                                  # lung
    ((33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,   # bone
      52, 53, 54, 55, 56, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
      96, 97, 114, 120, 122, 127), 0.43),
    ((98, 99, 100, 101, 102, 103, 104, 105, 106, 107), 0.39),                      # muscle
    ((22, 57, 121), 0.30),                                                         # trachea/spinal_cord/other
)


def build_between_ratio_table(maxid, groups=CT_BETWEEN_WITHIN_GROUPS,
                              default=CT_BETWEEN_WITHIN_DEFAULT):
    """(maxid+1,) float32 array, indexed DIRECTLY by MAISI id (matches mu[id]/sd[id] -- no
    -1 offset needed, unlike CT_GROUP_INDICES which indexes into the shorter mu[1:] array).
    id 0 (background) is always 0 -- never perturbed between members."""
    table = np.full(maxid + 1, float(default), dtype=np.float32)
    table[0] = 0.0
    for ids, r in groups:
        for i in ids:
            table[i] = r
    return table


def resolve_between_ratio(spec, maxid):
    """spec: None (disabled, unchanged today's behavior) | 'ct' (CT_BETWEEN_WITHIN_GROUPS
    preset) | an explicit (maxid+1,)-length array-like. Returns None or a float32 array."""
    if spec is None:
        return None
    if isinstance(spec, str):
        assert spec == "ct", f"unknown between-ratio preset {spec!r}"
        return build_between_ratio_table(maxid)
    arr = np.asarray(spec, dtype=np.float32)
    assert arr.shape == (maxid + 1,), (arr.shape, maxid)
    return arr


def _gaussian_copula_latent_rho(target_corr):
    """Latent Gaussian correlation giving `target_corr` REALIZED Pearson correlation between
    two Uniform(0,1) variables built via a Gaussian copula (for jointly-normal (Z1,Z2) with
    correlation rho, Pearson_r(Phi(Z1),Phi(Z2)) ~= (6/pi)*arcsin(rho/2); inverted here so
    callers specify the correlation they actually want to see in the final draw, not the
    latent parameter)."""
    target_corr = np.clip(target_corr, -0.999, 0.999)
    return float(np.clip(2.0 * np.sin(target_corr * np.pi / 6.0), -0.999, 0.999))


def sample_grouped_uniform(n, lo, hi, group_indices, group_rho, rng):
    """`n` domain-randomized values, each marginally EXACTLY Uniform(lo,hi) -- identical
    per-slot statistics to full independence, so nothing about any single slot's appearance
    range changes and no real intensity VALUE is ever used -- but a FIXED subset of slots
    (`group_indices`) shares a latent factor every call, injecting the SAME-RANGE ambiguity
    real anatomy has (e.g. every rib sits in the same HU band in a real scan, so a model can't
    localize one rib by shade alone) that fully-independent-per-slot painting erases and makes
    segmentation artificially easy relative to real difficulty.

    Membership is FIXED (the same `group_indices` every call) -- see CT_GROUP_MAISI_IDS's
    docstring for why: reshuffling membership per call was tried and rejected, it collapses
    the real block-diagonal structure into one flat bump. Only the shared VALUE each group
    takes is randomized (fresh every call), never a real HU value or a persistent
    brightness-to-identity mapping -- fixed membership only encodes "these slots are the same
    tissue type", not what that tissue type looks like.

    Construction (Gaussian copula, exact marginals): combined_i = sqrt(rho_g)*z_g +
    sqrt(1-rho_g)*z_i, z_g shared per group / z_i iid N(0,1) -> combined_i ~ N(0,1) exactly
    regardless of grouping; u_i = Phi(combined_i) ~ U(0,1) exactly (probability integral
    transform) -- only the JOINT structure changes, never an individual slot's marginal law.

    group_indices: sequence of index arrays (0-based positions into the length-n output,
    e.g. CT_GROUP_MAISI_IDS converted to 0-based -- see CT_GROUP_INDICES); positions not in
    any group are independent (rho=0, i.e. today's default behavior).
    group_rho: TARGET realized Pearson correlation per group (same length as group_indices;
    internally converted to the latent Gaussian rho via _gaussian_copula_latent_rho).
    rng: a numpy Generator (e.g. the cohort-shared `nrng` in SynthGmmMaisiDataset.assemble).
    """
    assert len(group_indices) == len(group_rho), (group_indices, group_rho)
    combined = rng.standard_normal(n)                        # iid N(0,1); overwritten in-group
    for idx, r in zip(group_indices, group_rho):
        idx = np.asarray(idx)
        if idx.size < 2:
            continue
        latent_rho = _gaussian_copula_latent_rho(r)
        g = rng.standard_normal()
        combined[idx] = np.sqrt(latent_rho) * g + np.sqrt(1 - latent_rho) * combined[idx]
    u = 0.5 * (1.0 + erf(combined / np.sqrt(2.0)))           # Phi(combined), exact U(0,1)
    return lo + (hi - lo) * u


def maisi_ids_to_indices(group_maisi_ids):
    """CT_GROUP_MAISI_IDS (1-based MAISI ids, matching mu[id]) -> 0-based positions into the
    length-(maxid) array sample_grouped_uniform fills (mu[1:], so id i -> position i-1)."""
    return tuple(tuple(i - 1 for i in ids) for ids in group_maisi_ids)


CT_GROUP_INDICES = maisi_ids_to_indices(CT_GROUP_MAISI_IDS)
MERGED_GROUP_INDICES = maisi_ids_to_indices(MERGED_GROUP_MAISI_IDS)


def synthesize_intensities(
    labels: torch.Tensor,
    L: int,
    cohort_gen: torch.Generator,
    subject_gen: torch.Generator,
    mu_range: tuple[float, float] = (0.0, 255.0),
    var_max: float = 5.0,
    background_mode: str = "zero",
    clamp: tuple[float, float] | None = None,
) -> torch.Tensor:
    """labels [N,D,H,W] int64 → images [N,1,D,H,W] float32 (raw units, not normalized).

    L = number of organ slots (ids 1..L; L+1 = container). cohort_gen draws the shared
    mu/sigma ("the scanner"); subject_gen draws the per-voxel noise. Seeding either
    reproduces that level independently (support/query = same cohort_gen state, advanced
    subject_gen). background_mode: "zero" (hard air) or "component" (dark, mu~U(0,15)).
    clamp: optional (lo,hi), off by default so downstream gamma/normalization sees true
    values.
    """
    assert labels.dtype == torch.int64, f"labels must be int64, got {labels.dtype}"
    assert background_mode in ("zero", "component"), background_mode
    assert int(labels.max()) <= L + 1, (
        f"label id {int(labels.max())} exceeds L+1={L + 1} (organs 1..L + container L+1)")
    device = labels.device
    n_ids = L + 2                                    # 0=bg, 1..L organs, L+1 container

    # ---- cohort-level parameter draw (shared across all N subjects) ----
    mu = torch.empty(n_ids, device=device)
    sigma = torch.empty(n_ids, device=device)
    mu[1:] = torch.empty(n_ids - 1, device=device).uniform_(*mu_range, generator=cohort_gen)
    var = torch.empty(n_ids - 1, device=device).uniform_(0.0, var_max, generator=cohort_gen)
    sigma[1:] = var.sqrt()                           # paper specifies VARIANCE ~ U(0,var_max)

    if background_mode == "zero":
        mu[0] = 0.0
        sigma[0] = 0.0
    else:                                            # small dark air component
        mu[0] = torch.empty((), device=device).uniform_(0.0, 15.0, generator=cohort_gen)
        sigma[0] = 0.5 ** 0.5

    # ---- subject-level voxelwise sampling (independent noise per voxel & subject) ----
    noise = torch.randn(labels.shape, device=device, generator=subject_gen)
    img = mu[labels] + sigma[labels] * noise         # [N,D,H,W]; bg id 0 → 0+0*noise = 0
    if clamp is not None:
        img = img.clamp(*clamp)
    return img.unsqueeze(1).float()                  # [N,1,D,H,W]


def pack_label_ids(labels: torch.Tensor, container_id: int | None = None):
    """Remap arbitrary placement-stage ids (e.g. MAISI 1..132) to the dense slot scheme
    the intensity stage expects: 0 stays background, the `container_id` (if given and
    present) becomes L+1, and every other present nonzero id becomes an organ slot 1..L
    (ascending). Returns (packed int64 labels, L). Vectorized; no host sync beyond `unique`.

    A convenience bridge for banks whose ids are anatomical classes, not blueprint slots —
    the intensity stage itself is agnostic and only needs 0/1..L/L+1.
    """
    device = labels.device
    present = torch.unique(labels)
    present = present[present != 0]
    organ_ids = present[present != container_id] if container_id is not None else present
    organ_ids = organ_ids.sort().values
    L = int(organ_ids.numel())
    lut = torch.zeros(int(labels.max()) + 1, dtype=torch.int64, device=device)
    lut[organ_ids] = torch.arange(1, L + 1, dtype=torch.int64, device=device)
    if container_id is not None and (present == container_id).any():
        lut[container_id] = L + 1
    return lut[labels], L

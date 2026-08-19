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
import torch


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

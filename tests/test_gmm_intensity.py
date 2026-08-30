"""GPU GMM intensity synthesis: the §9 validation checks from the spec.

Covers the invariants that make it a valid SynthSeg-style generator: cohort-shared GMM
(same mu/sigma across N, different noise), bitwise determinism from the two seeds, id
coverage, and the near-piecewise-constant "clean" contrast profile (discrete near-delta
peaks per id). Runs on CUDA when available, else CPU (torch.Generator is device-typed).
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import torch

from src.gpu_gmm_intensity import (
    synthesize_intensities, pack_label_ids, sample_grouped_uniform,
    _gaussian_copula_latent_rho, maisi_ids_to_indices,
    CT_GROUP_MAISI_IDS, CT_GROUP_INDICES, CT_GROUP_RHO,
    MERGED_GROUP_MAISI_IDS, MERGED_GROUP_INDICES, MERGED_GROUP_RHO,
    CT_BETWEEN_WITHIN_GROUPS, CT_BETWEEN_WITHIN_DEFAULT,
    build_between_ratio_table, resolve_between_ratio,
)
import pytest

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gen(seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed); return g


def _blocky_labels(N, L, size=24):
    """N subjects, identical geometry: axis-partitioned slabs of ids 0..L+1 (all realized)."""
    D = H = W = size
    lab = torch.zeros(D, H, W, dtype=torch.int64)
    edges = torch.linspace(0, D, L + 3).long()       # L+2 bands: bg,1..L,container
    for i in range(L + 2):
        lab[edges[i]:edges[i + 1]] = i
    return lab.to(DEV).unsqueeze(0).expand(N, -1, -1, -1).contiguous()


def test_shapes_and_dtype():
    lab = _blocky_labels(3, 5)
    img = synthesize_intensities(lab, 5, _gen(0), _gen(1))
    assert img.shape == (3, 1, *lab.shape[1:])
    assert img.dtype == torch.float32 and img.device.type == DEV.type


def test_background_is_zero():
    lab = _blocky_labels(2, 5)
    img = synthesize_intensities(lab, 5, _gen(0), _gen(1))[:, 0]
    assert torch.count_nonzero(img[lab == 0]) == 0     # id 0 → exactly 0, no masking


def test_shared_gmm_across_subjects():
    """N>1 share one GMM draw: per-id voxel means across subjects match to ~sigma/sqrt(n)."""
    N, L = 8, 6
    lab = _blocky_labels(N, L)
    img = synthesize_intensities(lab, L, _gen(0), _gen(1))[:, 0]
    for i in range(1, L + 2):
        m = lab[0] == i
        per_subj_mean = img[:, m].mean(dim=1)          # (N,) mean over the id's voxels
        # tight spread across subjects (shared mu; noise sigma<=sqrt(5) over ~1000s voxels)
        assert per_subj_mean.std().item() < 0.2, (i, per_subj_mean.std().item())


def test_gmm_differs_across_calls():
    lab = _blocky_labels(1, 6)
    a = synthesize_intensities(lab, 6, _gen(0), _gen(1))[0, 0]
    b = synthesize_intensities(lab, 6, _gen(99), _gen(1))[0, 0]   # different cohort seed
    # per-id means should differ (independent scanner draw)
    for i in range(1, 8):
        m = lab[0] == i
        assert abs(a[m].mean().item() - b[m].mean().item()) > 0.5


def test_determinism_bitwise():
    lab = _blocky_labels(4, 6)
    a = synthesize_intensities(lab, 6, _gen(7), _gen(11))
    b = synthesize_intensities(lab, 6, _gen(7), _gen(11))
    assert torch.equal(a, b)


def test_noise_resample_holds_gmm():
    """Same cohort seed, different subject seed → same means, different realizations."""
    lab = _blocky_labels(1, 6)
    a = synthesize_intensities(lab, 6, _gen(7), _gen(1))[0, 0]
    b = synthesize_intensities(lab, 6, _gen(7), _gen(2))[0, 0]
    assert not torch.equal(a, b)
    for i in range(1, 8):
        m = lab[0] == i
        assert abs(a[m].mean().item() - b[m].mean().item()) < 0.2   # GMM fixed


def test_contrast_profile_near_delta():
    """Each realized id forms a tight peak: within-id std ~ sqrt(var) <= sqrt(5) ~ 2.24."""
    lab = _blocky_labels(1, 8)
    img = synthesize_intensities(lab, 8, _gen(3), _gen(4))[0, 0]
    for i in range(1, 10):
        m = lab[0] == i
        assert img[m].std().item() <= 2.5


def test_id_coverage_assert():
    lab = _blocky_labels(1, 3)
    lab[0, 0, 0, 0] = 99                                # id > L+1
    try:
        synthesize_intensities(lab, 3, _gen(0), _gen(1))
        assert False, "expected assertion on out-of-range id"
    except AssertionError as e:
        assert "exceeds" in str(e)


def test_background_component_mode():
    lab = _blocky_labels(1, 4)
    img = synthesize_intensities(lab, 4, _gen(0), _gen(1), background_mode="component")[0, 0]
    bg = img[lab[0] == 0]
    assert 0.0 <= bg.mean().item() <= 15.0 and bg.std().item() > 0   # dark, jittered


def test_clamp_option():
    lab = _blocky_labels(1, 4)
    img = synthesize_intensities(lab, 4, _gen(0), _gen(1), clamp=(0.0, 255.0))
    assert img.min() >= 0.0 and img.max() <= 255.0


def test_pack_label_ids():
    """Arbitrary anatomical ids → dense 0/1..L/L+1, container mapped to L+1."""
    lab = torch.tensor([[0, 5, 5, 40, 200]], dtype=torch.int64, device=DEV).view(1, 1, 1, 5)
    packed, L = pack_label_ids(lab, container_id=200)
    assert L == 2                                       # organ ids {5,40}
    assert packed.flatten().tolist() == [0, 1, 1, 2, 3]  # 200 → L+1 = 3
    # packed is a valid intensity-stage input
    synthesize_intensities(packed, L, _gen(0), _gen(1))


N_SLOTS = 200   # SynthGmmMaisiDataset's default maxid (mu[1:] has this many entries)


def test_grouped_uniform_marginal_is_exact():
    """Pooled over slots+draws, u is EXACTLY Uniform(0,1) regardless of grouping -- the whole
    point (inject correlation, not shifted/narrowed per-slot marginals -> no leaked values,
    no change to any individual slot's appearance range)."""
    rng = np.random.default_rng(0)
    draws = np.stack([sample_grouped_uniform(N_SLOTS, 0.0, 1.0, CT_GROUP_INDICES, CT_GROUP_RHO, rng)
                      for _ in range(3000)])
    assert abs(draws.mean() - 0.5) < 0.01
    assert abs(draws.std() - (1 / 12 ** 0.5)) < 0.01
    hist, _ = np.histogram(draws.ravel(), bins=10, range=(0, 1))
    assert (hist / hist.mean()).std() < 0.05                # flat deciles


@pytest.mark.parametrize("group_indices,group_rho", [
    (CT_GROUP_INDICES, CT_GROUP_RHO),
    (MERGED_GROUP_INDICES, MERGED_GROUP_RHO),
])
def test_grouped_uniform_achieves_target_correlation(group_indices, group_rho):
    """Redraw many times with FIXED membership: within-group achieved Pearson r matches the
    requested target; cross-group pairs land near 0. Checked for both presets (CT-only and
    the coarser cross-modality-validated 'merged' one)."""
    rng = np.random.default_rng(1)
    N = 3000
    draws = np.stack([sample_grouped_uniform(N_SLOTS, 0.0, 1.0, group_indices, group_rho, rng)
                      for _ in range(N)])

    for idx, target in zip(group_indices, group_rho):
        c = np.corrcoef(draws[:, idx], rowvar=False)
        off = c[~np.eye(len(idx), dtype=bool)]
        assert abs(off.mean() - target) < 0.03, (target, off.mean())

    a, b = group_indices[0][:5], group_indices[1][:5]
    cross = [np.corrcoef(draws[:, i], draws[:, j])[0, 1] for i in a for j in b]
    assert abs(np.mean(cross)) < 0.05


def test_merged_group_ids_no_overlap_and_maps_cleanly():
    """MERGED_GROUP_MAISI_IDS (cross-modality preset): no id in two groups, matches its
    precomputed 0-based MERGED_GROUP_INDICES."""
    assert len(MERGED_GROUP_MAISI_IDS) == len(MERGED_GROUP_RHO)
    all_ids = [i for group in MERGED_GROUP_MAISI_IDS for i in group]
    assert len(set(all_ids)) == len(all_ids)
    assert MERGED_GROUP_INDICES == maisi_ids_to_indices(MERGED_GROUP_MAISI_IDS)
    # the two presets need not be disjoint (different analyses), but each is internally clean
    for ids in MERGED_GROUP_MAISI_IDS:
        assert len(set(ids)) == len(ids)


def test_grouped_uniform_membership_is_fixed_not_reshuffled():
    """Membership must NOT reshuffle across calls -- a reshuffled-per-call version was tried
    and rejected (docs/logs.md): it collapses the real block-diagonal correlation structure
    into a flat, anatomy-blind bump applied to every id pair alike. Two independent calls with
    CT_GROUP_INDICES must group the exact same slots every time."""
    rng = np.random.default_rng(2)
    # sample_grouped_uniform doesn't expose membership directly, so probe it indirectly:
    # two ids in the SAME fixed group correlate at the target rho across many draws; if
    # membership reshuffled, that correlation would collapse to the diluted co-occurrence
    # value instead (see analyze_synth_gmm_intensity.py's follow-up-5 finding, ~0.17 not 0.70).
    N = 2000
    draws = np.stack([sample_grouped_uniform(N_SLOTS, 0.0, 1.0, CT_GROUP_INDICES, CT_GROUP_RHO, rng)
                      for _ in range(N)])
    i, j = CT_GROUP_INDICES[0][0], CT_GROUP_INDICES[0][1]        # two fixed "bone" ids
    r = np.corrcoef(draws[:, i], draws[:, j])[0, 1]
    assert abs(r - CT_GROUP_RHO[0]) < 0.05, r                    # matches target, not ~0.17


def test_maisi_ids_to_indices():
    idx = maisi_ids_to_indices(((5, 8, 12),))
    assert idx == ((4, 7, 11),)
    assert CT_GROUP_INDICES == maisi_ids_to_indices(CT_GROUP_MAISI_IDS)
    for ids in CT_GROUP_MAISI_IDS:
        assert len(set(ids)) == len(ids)                         # no duplicate ids within a group
    all_ids = [i for group in CT_GROUP_MAISI_IDS for i in group]
    assert len(set(all_ids)) == len(all_ids)                     # groups don't overlap


def test_between_ratio_table_no_overlap_and_covers_default():
    """CT_BETWEEN_WITHIN_GROUPS ids don't collide; anything not listed gets the global default;
    background (id 0) is always 0 (never perturbed between members)."""
    all_ids = [i for ids, _ in CT_BETWEEN_WITHIN_GROUPS for i in ids]
    assert len(set(all_ids)) == len(all_ids)
    table = build_between_ratio_table(200)
    assert table.shape == (201,)
    assert table[0] == 0.0
    listed = set(all_ids)
    for i in range(1, 201):
        if i not in listed:
            assert table[i] == CT_BETWEEN_WITHIN_DEFAULT
    for ids, r in CT_BETWEEN_WITHIN_GROUPS:
        for i in ids:
            assert table[i] == np.float32(r)


def test_resolve_between_ratio_none_ct_and_explicit():
    assert resolve_between_ratio(None, 200) is None
    ct = resolve_between_ratio("ct", 200)
    assert ct.shape == (201,) and ct.dtype == np.float32
    explicit = np.full(201, 0.7, dtype=np.float32)
    out = resolve_between_ratio(explicit, 200)
    assert np.array_equal(out, explicit)
    try:
        resolve_between_ratio(np.zeros(5), 200)
        assert False, "expected a shape-mismatch assertion"
    except AssertionError:
        pass


def test_between_ratio_realizes_target_member_spread():
    """The generative formula synth_gmm_maisi_dataset.assemble()/gpu_synth_realize use:
    mu_e = mu + ratio*sd*eps_e, eps_e ~ N(0,1) FRESH per member. Across many simulated
    members, std(mu_e) should match ratio*sd (the whole point: sd stays the within-scan
    texture scale, ratio*sd becomes the member-to-member spread)."""
    rng = np.random.default_rng(0)
    mu_c, sd_c, ratio_c = 100.0, 8.0, 1.11                       # a "vascular"-like class
    N = 20000
    eps = rng.standard_normal(N)
    mu_e = mu_c + ratio_c * sd_c * eps
    assert abs(mu_e.mean() - mu_c) < 0.2
    assert abs(mu_e.std() - ratio_c * sd_c) < 0.2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

"""GPU GMM intensity synthesis: the §9 validation checks from the spec.

Covers the invariants that make it a valid SynthSeg-style generator: cohort-shared GMM
(same mu/sigma across N, different noise), bitwise determinism from the two seeds, id
coverage, and the near-piecewise-constant "clean" contrast profile (discrete near-delta
peaks per id). Runs on CUDA when available, else CPU (torch.Generator is device-typed).
"""
import sys; sys.path.insert(0, ".")
import torch

from src.gpu_gmm_intensity import synthesize_intensities, pack_label_ids

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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

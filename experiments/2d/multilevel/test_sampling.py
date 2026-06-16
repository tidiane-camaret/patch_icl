import sys; sys.path.insert(0, "experiments/2d/multilevel")
import torch
from sampling import sample_patches, gaussian_blur, idx_to_ij, gather_grid

R = 8
N = R * R
KW = dict(n_total=20, tau=0.15, n_fg_core=4, blur_sigma=1.0, floor=0.01,
          grid_res=R, temperature=1.0)


def _make(vals):
    return torch.tensor(vals, dtype=torch.float32)


def test_shape_unique_and_fixed_count():
    torch.manual_seed(0)
    v = torch.rand(3, N)
    idx, is_core, is_fg = sample_patches(v, **KW)
    assert idx.shape == is_core.shape == is_fg.shape == (3, 20)
    for b in range(3):                       # indices unique per row
        assert len(set(idx[b].tolist())) == 20


def test_fixed_count_across_core_sizes():
    # row0: no boundary cell (all pure 0/1); row1: a few; row2: nearly all near 0.5
    v = torch.stack([
        torch.tensor([0.0, 1.0] * (N // 2)),                 # core_count ≈ 0
        torch.cat([torch.full((4,), 0.5), torch.zeros(N - 4)]),  # small core
        torch.full((N,), 0.5),                               # core_count == N
    ])
    idx, is_core, _ = sample_patches(v, **KW)
    assert idx.shape == (3, 20)
    for b in range(3):
        assert len(set(idx[b].tolist())) == 20               # always n_total, unique


def test_boundary_core_matches_threshold():
    torch.manual_seed(1)
    v = torch.rand(4, N)
    idx, is_core, is_fg = sample_patches(v, **KW)
    is_boundary = is_core & ~is_fg
    sel_v = v.gather(1, idx)
    d = (sel_v - 0.5).abs()
    assert torch.all(d[is_boundary] < KW["tau"])             # boundary ⇒ within tau


def test_fg_core_quota_and_values():
    # plenty of pure-fg cells available, none within tau (tau=0.15 → boundary is (0.35,0.65))
    v = torch.cat([torch.ones(3, 40), torch.zeros(3, N - 40)], dim=1)
    idx, is_core, is_fg = sample_patches(v, **KW)
    sel_v = v.gather(1, idx)
    for b in range(3):
        assert int(is_fg[b].sum()) == KW["n_fg_core"]        # exactly the quota
        assert torch.all(sel_v[b][is_fg[b]] >= 0.5)          # all foreground

    idx0, _, is_fg0 = sample_patches(v, **{**KW, "n_fg_core": 0})
    assert int(is_fg0.sum()) == 0                            # n_fg_core=0 ⇒ none


def test_proximity_bias():
    # single boundary core cell at center; neighbors should land adjacent more than chance.
    torch.manual_seed(0)
    v = torch.zeros(1, N)
    center = (R // 2) * R + (R // 2)
    v[0, center] = 0.5                                       # the only boundary cell
    kw = dict(KW, n_total=6, n_fg_core=0, blur_sigma=0.7, floor=1e-4)
    adj = 0
    cr, cc = R // 2, R // 2
    for _ in range(200):
        idx, is_core, _ = sample_patches(v, **kw)
        for j in idx[0].tolist():
            if j == center:
                continue
            r, c = divmod(j, R)
            if abs(r - cr) <= 1 and abs(c - cc) <= 1:
                adj += 1
    # 5 neighbors/draw × 200 draws = 1000 picks; 8 of 63 non-core cells are adjacent
    # (chance ≈ 13%). Proximity bias should clear that comfortably.
    assert adj / 1000 > 0.4, adj / 1000


def test_gaussian_blur_spreads_mass():
    x = torch.zeros(1, N); x[0, (R // 2) * R + R // 2] = 1.0
    out = gaussian_blur(x, R, 1.0)
    assert out.shape == (1, N)
    assert (out > 0).sum() > 1                                # mass spread to neighbors
    assert abs(out.sum().item() - 1.0) < 1e-4                 # normalized kernel conserves mass


def test_idx_to_ij_roundtrip():
    idx = torch.tensor([[0, 1, 33, 1023]])
    ij = idx_to_ij(idx, 32)
    assert ij.shape == (1, 4, 2)
    assert ij[0, 0].tolist() == [0, 0]
    assert ij[0, 2].tolist() == [1, 1]
    assert ij[0, 3].tolist() == [31, 31]


def test_gather_grid_features_and_values():
    x = torch.arange(2 * 8 * 3).float().reshape(2, 8, 3)
    idx = torch.tensor([[1, 4], [0, 7]])
    g = gather_grid(x, idx)
    assert g.shape == (2, 2, 3)
    assert torch.equal(g[0, 0], x[0, 1]) and torch.equal(g[1, 1], x[1, 7])
    vals = torch.arange(2 * 8).float().reshape(2, 8)
    gv = gather_grid(vals, idx)
    assert gv.shape == (2, 2) and gv[0, 0] == vals[0, 1] and gv[1, 1] == vals[1, 7]


if __name__ == "__main__":
    test_shape_unique_and_fixed_count()
    test_fixed_count_across_core_sizes()
    test_boundary_core_matches_threshold()
    test_fg_core_quota_and_values()
    test_proximity_bias()
    test_gaussian_blur_spreads_mass()
    test_idx_to_ij_roundtrip()
    test_gather_grid_features_and_values()
    print("ALL SAMPLING TESTS PASSED")

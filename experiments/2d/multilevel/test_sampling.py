import sys; sys.path.insert(0, "experiments/2d/multilevel")
import torch
from sampling import sample_patch_indices, idx_to_ij, gather_grid

def test_selects_closest_and_farthest_from_half():
    # N=8 values; distances to 0.5 are well separated.
    v = torch.tensor([[0.50, 0.49, 0.51, 0.7, 0.3, 0.95, 0.05, 0.99]])
    idx = sample_patch_indices(v, n_uncertain=3, n_certain=2)
    unc, cer = idx[:, :3], idx[:, 3:]
    # 3 closest to 0.5 are values {0.50,0.49,0.51} -> indices {0,1,2}
    assert set(unc[0].tolist()) == {0, 1, 2}, unc
    # 2 farthest from 0.5 are {0.05,0.99} -> indices {6,7}
    assert set(cer[0].tolist()) == {6, 7}, cer
    # disjoint
    assert len(set(idx[0].tolist())) == 5

def test_idx_to_ij_roundtrip():
    R = 32
    idx = torch.tensor([[0, 1, 33, 1023]])
    ij = idx_to_ij(idx, R)
    assert ij.shape == (1, 4, 2)
    assert ij[0, 0].tolist() == [0, 0]
    assert ij[0, 1].tolist() == [0, 1]
    assert ij[0, 2].tolist() == [1, 1]
    assert ij[0, 3].tolist() == [31, 31]

def test_gather_grid_features_and_values():
    x = torch.arange(2*8*3).float().reshape(2, 8, 3)   # (B,N,C)
    idx = torch.tensor([[1, 4], [0, 7]])
    g = gather_grid(x, idx)
    assert g.shape == (2, 2, 3)
    assert torch.equal(g[0, 0], x[0, 1]) and torch.equal(g[1, 1], x[1, 7])
    vals = torch.arange(2*8).float().reshape(2, 8)      # (B,N)
    gv = gather_grid(vals, idx)
    assert gv.shape == (2, 2)
    assert gv[0, 0] == vals[0, 1] and gv[1, 1] == vals[1, 7]

if __name__ == "__main__":
    test_selects_closest_and_farthest_from_half()
    test_idx_to_ij_roundtrip()
    test_gather_grid_features_and_values()
    print("ALL SAMPLING TESTS PASSED")

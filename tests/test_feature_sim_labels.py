# tests/test_feature_sim_labels.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from feature_sim.labels import grid_labels, sample_points


def _blob(S=16):
    m = torch.zeros(S, S, S)
    m[4:12, 4:12, 4:12] = 1.0
    return m


def test_grid_labels_occupancy_matches_threshold():
    m = _blob(16)
    g = grid_labels(m, res=8)                 # 16->8, each cell = 2^3 block
    assert g.shape == (8, 8, 8)
    # cell (2,2,2) covers voxels [4:6] fully inside the blob -> 1
    assert g[2, 2, 2] == 1.0
    # corner cell fully outside -> 0
    assert g[0, 0, 0] == 0.0


def test_sample_points_counts_and_labels():
    m = _blob(16)
    coords, labels = sample_points(m, n_fg=50, n_bg=70,
                                   generator=torch.Generator().manual_seed(0))
    assert coords.shape == (120, 3) and labels.shape == (120,)
    assert labels.sum() == 50 and (labels == 0).sum() == 70
    assert coords.min() >= -1.0 and coords.max() <= 1.0


def test_sample_points_band_restricts_bg_near_object():
    m = _blob(16)
    coords, labels = sample_points(m, n_fg=10, n_bg=40, band=2,
                                   generator=torch.Generator().manual_seed(0))
    # all BG points should fall within a 2-voxel shell of the blob: convert the
    # normalized (d,h,w) coord back to a voxel index and check dist to [4,12).
    bg = coords[labels == 0]
    idx = ((bg + 1) / 2 * (16 - 1)).round().long()    # (n_bg,3) voxel indices
    inside_core = ((idx >= 4) & (idx < 12)).all(dim=1)
    assert not inside_core.any()                       # band excludes the FG core
    near = ((idx >= 2) & (idx < 14)).all(dim=1)
    assert near.all()                                  # within a 2-voxel shell


def test_sample_points_axis_order_is_dhw():
    S = 16
    m = torch.zeros(S, S, S)
    m[4:8, 8:12, 12:16] = 1.0          # distinct extent per axis: d[4,8) h[8,12) w[12,16)
    coords, labels = sample_points(m, n_fg=40, n_bg=0,
                                   generator=torch.Generator().manual_seed(0))
    fg = coords[labels == 1]
    idx = ((fg + 1) / 2 * (S - 1)).round().long()   # -> voxel indices, (d,h,w)
    assert ((idx[:, 0] >= 4) & (idx[:, 0] < 8)).all()    # d axis
    assert ((idx[:, 1] >= 8) & (idx[:, 1] < 12)).all()   # h axis
    assert ((idx[:, 2] >= 12) & (idx[:, 2] < 16)).all()  # w axis

import torch
from src.models.patchset_pfn import FourierPositionalEncoding


def test_2d_default_unchanged():
    pe = FourierPositionalEncoding(e=16, num_bands=8)
    assert pe.proj.in_features == 4 * 8          # 2 axes * 2 (sin,cos) * bands
    ij = torch.zeros(2, 5, 2, dtype=torch.long)
    assert pe(ij, grid_res=16).shape == (2, 5, 16)


def test_3d_accepts_three_axes():
    pe = FourierPositionalEncoding(e=16, num_bands=8, n_axes=3)
    assert pe.proj.in_features == 6 * 8
    ijk = torch.zeros(2, 5, 3, dtype=torch.long)
    assert pe(ijk, grid_res=16).shape == (2, 5, 16)

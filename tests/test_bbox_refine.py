import sys; sys.path.insert(0, ".")
import torch
from src.models.bbox_refine import max_sum_window, gt_window, crop_resize, fuse_window, place_window


def test_max_sum_window_finds_blob():
    H, s = 16, 8
    prob = torch.zeros(2, 1, H, H)
    prob[0, 0, 2:6, 3:7] = 1.0            # blob near top-left
    prob[1, 0, 10:14, 9:13] = 1.0         # blob near bottom-right
    o = max_sum_window(prob, s)
    assert o.shape == (2, 2)
    assert (o >= 0).all() and (o[:, 0] <= H - s).all() and (o[:, 1] <= H - s).all()
    assert o[0, 0] <= 2 and o[0, 1] <= 3          # window covers top-left blob
    assert o[1, 0] >= 6 and o[1, 1] >= 5          # window shifted toward bottom-right blob


def test_max_sum_window_empty_centers():
    H, s = 16, 8
    prob = torch.zeros(1, 1, H, H)                # no mass → center, not corner (0,0)
    o = max_sum_window(prob, s)
    assert o[0, 0] == (H - s) // 2 and o[0, 1] == (H - s) // 2


def test_gt_window_matches_blob():
    H, s = 16, 8
    mask = torch.zeros(1, 1, H, H)
    mask[0, 0, 4:8, 4:8] = 1.0
    o = gt_window(mask, s)
    assert o[0, 0] <= 4 and o[0, 1] <= 4 and o[0, 0] >= 0 and o[0, 1] >= 0


def test_crop_resize_recovers_region():
    # crop the exact 8x8 region back to 8x8 (out=s) should reproduce it (bilinear, aligned cells)
    H, s = 16, 8
    x = torch.zeros(1, 1, H, H)
    x[0, 0, 4:12, 4:12] = 1.0
    o = torch.tensor([[4, 4]])
    y = crop_resize(x, o, s, out=s, mode="nearest")
    assert y.shape == (1, 1, s, s)
    assert y.min() > 0.5                          # every cell inside the all-ones region


def test_fuse_window_adds_into_window_only():
    H, s = 16, 8
    full = torch.zeros(2, 1, H, H)
    full[0, 0, 0:s, 0:s] = 1.0                    # non-zero window → additive, not replace
    full[1, 0, 8:16, 8:16] = 1.0
    patch = torch.full((2, 1, s, s), 2.0)
    o = torch.tensor([[0, 0], [8, 8]])
    out = fuse_window(full, patch, o, s)
    assert out.shape == (2, 1, H, H)
    assert full[0, 0, 0, 0] == 1.0                # input not mutated
    assert out[0, 0, 0:s, 0:s].eq(3.0).all()      # 1 + 2 == 3 (additive, not replace)
    assert out[0, 0, s:, s:].eq(0).all()          # outside window untouched
    assert out[1, 0, 8:16, 8:16].eq(3.0).all()


def test_place_window_replaces_not_adds():
    H, s = 16, 8
    full = torch.zeros(2, 1, H, H)
    full[0, 0, 0:s, 0:s] = 1.0                    # non-zero window → REPLACE (not add)
    full[1, 0, 8:16, 8:16] = 1.0
    patch = torch.full((2, 1, s, s), 2.0)
    o = torch.tensor([[0, 0], [8, 8]])
    out = place_window(full, patch, o, s)
    assert out.shape == (2, 1, H, H)
    assert full[0, 0, 0, 0] == 1.0                # input not mutated
    assert out[0, 0, 0:s, 0:s].eq(2.0).all()      # window overwritten to patch (2), not 1+2=3
    assert out[0, 0, s:, s:].eq(0).all()          # outside window untouched
    assert out[1, 0, 8:16, 8:16].eq(2.0).all()

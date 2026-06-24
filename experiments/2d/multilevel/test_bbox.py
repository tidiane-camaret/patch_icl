import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d/multilevel")
import torch
from bbox import max_sum_window, gt_window, crop_resize, composite_window


def test_max_sum_window_finds_blob():
    H, s = 16, 8
    prob = torch.zeros(2, 1, H, H)
    prob[0, 0, 2:6, 3:7] = 1.0          # blob near top-left → origin should cover it
    prob[1, 0, 10:14, 9:13] = 1.0       # blob near bottom-right
    o = max_sum_window(prob, s)
    assert o.shape == (2, 2)
    # origin must be in-bounds and the window must contain the blob's mass
    assert (o >= 0).all() and (o[:, 0] <= H - s).all() and (o[:, 1] <= H - s).all()
    assert o[0, 0] <= 2 and o[0, 1] <= 3            # window starts at/above-left of blob
    assert o[1, 0] >= 6 and o[1, 1] >= 5            # window shifted toward the blob


def test_max_sum_window_border_blob_clamps():
    H, s = 16, 8
    prob = torch.zeros(1, 1, H, H)
    prob[0, 0, 0:2, 0:2] = 1.0          # corner blob
    o = max_sum_window(prob, s)
    assert (o == 0).all()              # origin clamped to the corner, still in-bounds


def test_gt_window_matches_max_sum():
    H, s = 16, 8
    mask = torch.zeros(1, 1, H, H)
    mask[0, 0, 4:8, 4:8] = 1.0
    o = gt_window(mask, s)
    assert o.shape == (1, 2)
    assert (o >= 0).all() and (o <= H - s).all()


def test_crop_resize_roundtrip_identity():
    # cropping the full image (origin 0, s=H, out=H) returns the image unchanged
    x = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    origin = torch.zeros(2, 2, dtype=torch.long)
    y = crop_resize(x, origin, s=4, out=4, mode="nearest")
    assert y.shape == (2, 1, 4, 4)
    assert torch.allclose(y, x)


def test_crop_resize_picks_region():
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 4:8, 4:8] = 5.0
    origin = torch.tensor([[4, 4]])
    y = crop_resize(x, origin, s=4, out=4, mode="nearest")
    assert torch.allclose(y, torch.full((1, 1, 4, 4), 5.0))


def test_composite_window_writes_region_only():
    full = torch.zeros(1, 1, 8, 8)
    patch = torch.ones(1, 1, 4, 4)
    origin = torch.tensor([[2, 3]])
    out = composite_window(full, patch, origin, s=4)
    assert torch.allclose(full, torch.zeros(1, 1, 8, 8))     # input not mutated
    assert torch.allclose(out[0, 0, 2:6, 3:7], torch.ones(4, 4))
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask[0, 0, 2:6, 3:7] = True
    assert torch.allclose(out[~mask], torch.zeros_like(out)[~mask])


def test_empty_prediction_centers_crop():
    # An all-zero (empty) prediction must center the crop, not collapse to the corner (0,0).
    H, s = 16, 8
    c = (H - s) // 2                    # centered origin (H == W here)
    o = max_sum_window(torch.zeros(1, 1, H, H), s)
    assert torch.equal(o, torch.tensor([[c, c]]))
    # same for an empty GT mask
    assert torch.equal(gt_window(torch.zeros(1, 1, H, H), s), torch.tensor([[c, c]]))


def test_empty_centering_is_per_sample():
    # Mixed batch: empty sample centers; non-empty sample still tracks its blob.
    H, s = 16, 8
    prob = torch.zeros(2, 1, H, H)
    prob[1, 0, 10:14, 9:13] = 1.0       # only sample 1 has foreground
    o = max_sum_window(prob, s)
    assert torch.equal(o[0], torch.tensor([(H - s) // 2, (H - s) // 2]))   # empty → centered
    assert o[1, 0] >= 6 and o[1, 1] >= 5                                   # blob → tracked


if __name__ == "__main__":
    test_max_sum_window_finds_blob()
    test_max_sum_window_border_blob_clamps()
    test_empty_prediction_centers_crop()
    test_empty_centering_is_per_sample()
    test_gt_window_matches_max_sum()
    test_crop_resize_roundtrip_identity()
    test_crop_resize_picks_region()
    test_composite_window_writes_region_only()
    print("ALL BBOX TESTS PASSED")

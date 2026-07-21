import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.render3d import render_scene_3d

CANVAS = (16, 16, 16)


def _obj(intensity, mask_val, t=6):
    def s(rng):
        arr = np.zeros((2, t, t, t), dtype=np.float32)
        arr[0] = intensity
        arr[1] = mask_val
        return arr
    return s


def test_shapes_and_k_range():
    rng = np.random.default_rng(0)
    for _ in range(30):
        img, mask, k, info = render_scene_3d(
            rng, CANVAS, n_objects=5, k_min=2, k_max=4,
            target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
        assert img.shape == CANVAS and img.dtype == np.float32
        assert mask.shape == CANVAS
        assert 2 <= k <= 4
        assert len(info["target_centroids"]) >= 1


def test_mask_binary_and_only_targets():
    # distractors paint intensity 0.9 but must never enter the label mask.
    rng = np.random.default_rng(1)
    img, mask, k, _ = render_scene_3d(
        rng, CANVAS, n_objects=6, k_min=2, k_max=2,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0
    # every masked voxel shows the target intensity, not a distractor's
    assert np.allclose(img[mask > 0], 0.5)


def test_k_clamped_to_n_objects():
    rng = np.random.default_rng(2)
    _, _, k, _ = render_scene_3d(
        rng, CANVAS, n_objects=3, k_min=99, k_max=99,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
    assert k == 3


def test_centroids_in_unit_range():
    rng = np.random.default_rng(3)
    _, _, _, info = render_scene_3d(
        rng, CANVAS, n_objects=4, k_min=1, k_max=1,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
    for (z, y, x) in info["target_centroids"]:
        assert 0.0 <= z <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= x <= 1.0


def test_anti_overlap_reduces_union_deficit():
    big = _obj(1.0, 1.0, t=10)               # oversized tiles -> overlap likely

    def occupancy(tries):
        rng = np.random.default_rng(5)
        areas = unions = 0
        for _ in range(15):
            img, _, _, _ = render_scene_3d(
                rng, CANVAS, n_objects=6, k_min=1, k_max=1,
                target_sampler=big, distractor_sampler=big,
                tries=tries, max_overlap=0.0)
            unions += int((img > 0).sum())
            areas += 6 * 10 ** 3
        return areas / max(unions, 1)         # higher => more overlap
    assert occupancy(16) < occupancy(1)


def test_black_background_zero_off_object():
    rng = np.random.default_rng(6)
    img, mask, _, _ = render_scene_3d(
        rng, CANVAS, n_objects=1, k_min=1, k_max=1,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.5, 1.0))
    assert (img[mask == 0] == 0).all()


if __name__ == "__main__":
    test_shapes_and_k_range()
    test_mask_binary_and_only_targets()
    test_k_clamped_to_n_objects()
    test_centroids_in_unit_range()
    test_anti_overlap_reduces_union_deficit()
    test_black_background_zero_off_object()
    print("ALL RENDER3D TESTS PASSED")

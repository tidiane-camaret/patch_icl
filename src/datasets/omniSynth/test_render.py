import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.config import OmniSceneConfig
from src.datasets.omniSynth.render import render_scene, affine_jitter

CELL, GRID = 16, 4


def _const_sampler(value):
    # returns a full-cell bitmap of the given constant (1=target ink, etc.)
    return lambda rng: np.full((CELL, CELL), value, dtype=np.uint8)


def test_shapes_and_k_range():
    scene = OmniSceneConfig(k_min=2, k_max=5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        img, mask, k, _ = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(0))
        assert img.shape == (GRID * CELL, GRID * CELL) and img.dtype == np.float32
        assert mask.shape == img.shape
        assert 2 <= k <= 5


def test_mask_marks_exactly_k_cells():
    # target sampler fills its cell with 1s; mask must equal the painted target cells.
    scene = OmniSceneConfig(k_min=3, k_max=3)
    rng = np.random.default_rng(1)
    img, mask, k, _ = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(0))
    assert k == 3
    # exactly 3 of the 16 cells are fully masked, the rest empty
    cells_masked = 0
    for r in range(GRID):
        for c in range(GRID):
            block = mask[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
            s = block.sum()
            assert s == 0 or s == CELL * CELL          # whole-cell, never partial
            cells_masked += int(s > 0)
    assert cells_masked == 3


def test_mask_is_target_cells_not_distractors():
    # distractor sampler also paints 1s into the image, but must NOT be in the mask.
    scene = OmniSceneConfig(k_min=4, k_max=4)
    rng = np.random.default_rng(2)
    img, mask, k, _ = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(1))
    assert img.sum() == GRID * GRID * CELL * CELL        # every cell painted
    assert mask.sum() == 4 * CELL * CELL                 # only 4 target cells masked


def test_k_clamped_to_valid_range():
    scene = OmniSceneConfig(k_min=99, k_max=99)          # absurd -> clamp to n_cells
    rng = np.random.default_rng(3)
    _, _, k, _ = render_scene(rng, scene, GRID, CELL, _const_sampler(1), _const_sampler(0))
    assert k == GRID * GRID


def test_oversized_tiles_overflow_and_union():
    # A tile larger than the cell (margin<0 case) must overflow its cell and union-blend
    # into neighbours instead of being clipped or overwriting them.
    scene = OmniSceneConfig(k_min=1, k_max=1)
    big = 24                                             # > CELL (16) -> overflows the cell
    rng = np.random.default_rng(7)
    tgt = lambda rng: np.ones((big, big), dtype=np.uint8)
    img, mask, k, info = render_scene(rng, scene, GRID, CELL, tgt, _const_sampler(0))
    assert img.shape == (GRID * CELL, GRID * CELL)
    # the single target's ink spills beyond one cell: more than CELL*CELL painted pixels
    assert mask.sum() > CELL * CELL
    # union never exceeds 1 even where the oversized glyph overlaps neighbours
    assert img.max() == 1.0 and mask.max() == 1.0
    # centroid is reported in [0,1]
    (x, y), = info["target_positions"]
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_affine_jitter_preserves_shape_and_binary():
    scene = OmniSceneConfig()
    base = np.zeros((CELL, CELL), dtype=np.uint8)
    base[4:12, 4:12] = 1
    out, _ = affine_jitter(base, scene, np.random.default_rng(4))
    assert out.shape == (CELL, CELL) and out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 1})


def _obj_sampler(intensity, mask_val):
    """A [2,CELL,CELL] (intensity, mask) rendition sampler (medseg-style objects)."""
    def s(rng):
        t = np.zeros((2, CELL, CELL), dtype=np.float32)
        t[0] = intensity; t[1] = mask_val
        return t
    return s


def test_two_channel_objects_paste_intensity_to_image_mask_to_label():
    # medseg-style: target intensity 0.5 under a full-cell mask; distractor intensity
    # 0.25 but NOT a target -> its texture lands in the image, never in the mask.
    # (values chosen float32-exact so equality checks are clean.)
    scene = OmniSceneConfig(k_min=2, k_max=2)
    rng = np.random.default_rng(5)
    img, mask, k, _ = render_scene(rng, scene, GRID, CELL,
                                   _obj_sampler(0.5, 1.0), _obj_sampler(0.25, 1.0))
    assert k == 2
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() == 2 * CELL * CELL              # only the 2 target cells masked
    # image carries the pasted intensities (targets 0.5, distractors 0.25), not the mask
    assert set(np.unique(img)).issubset({0.0, 0.25, 0.5})
    assert (img == 0.25).sum() > 0                    # distractor texture present in image
    assert (img == 0.5).sum() == 2 * CELL * CELL      # target texture only in target cells


def test_affine_jitter_two_channel_keeps_intensity_and_binarizes_mask():
    scene = OmniSceneConfig()
    base = np.zeros((2, CELL, CELL), dtype=np.float32)
    base[0, 4:12, 4:12] = 0.4       # intensity
    base[1, 4:12, 4:12] = 1.0       # mask
    out, _ = affine_jitter(base, scene, np.random.default_rng(6))
    assert out.shape == (2, CELL, CELL) and out.dtype == np.float32
    assert set(np.unique(out[1])).issubset({0.0, 1.0})       # mask stays binary
    assert float(out[0][out[1] == 0].max(initial=0.0)) == 0.0  # intensity masked to object
    assert out[0].max() <= 1.0


def test_random_background_nonblack_and_dark_object_visible():
    # A near-black object (intensity 0 under a full-cell mask) must stay visible over a
    # random grey background: the object region is dark while the background is grey.
    scene = OmniSceneConfig(k_min=1, k_max=1, max_nb_objects=1, background="random",
                            bg_intensity=(0.4, 0.6), bg_noise=0.0, bg_structure=0.0)
    rng = np.random.default_rng(11)
    img, mask, k, _ = render_scene(rng, scene, GRID, CELL,
                                   _obj_sampler(0.0, 1.0), _obj_sampler(0.0, 1.0))
    assert img.min() >= 0.0 and img.max() <= 1.0
    bg = img[mask == 0]
    assert bg.size > 0 and bg.mean() > 0.2          # background is grey, not black
    assert float(img[mask > 0].max()) < 0.05        # dark object painted (~0), visible over grey


def test_image_background_used_and_objects_painted_over():
    # background="image": the canvas comes from background_sampler; a dark object (0) is
    # painted over it and stays visible; untouched pixels keep the background image.
    scene = OmniSceneConfig(k_min=1, k_max=1, max_nb_objects=1, background="image")
    bg = np.full((GRID * CELL, GRID * CELL), 0.7, dtype=np.float32)
    sampler = lambda rng: bg.copy()
    rng = np.random.default_rng(21)
    img, mask, k, _ = render_scene(rng, scene, GRID, CELL,
                                   _obj_sampler(0.0, 1.0), _obj_sampler(0.0, 1.0),
                                   background_sampler=sampler)
    assert np.allclose(img[mask == 0], 0.7)         # background image preserved off-object
    assert float(img[mask > 0].max()) < 0.05        # dark object painted over the image bg


def test_targets_drawn_over_distractors():
    # grid=2 with oversized (2*cell) tiles all clamp to the canvas centre -> every object
    # fully overlaps. The target (intensity 0.5) must survive under the 3 distractors
    # (intensity 0.9): wherever the target mask is set, the image shows the target texture.
    big = CELL * 2
    def obj(intensity):
        def s(rng):
            t = np.zeros((2, big, big), dtype=np.float32)
            t[0] = intensity; t[1] = 1.0
            return t
        return s
    scene = OmniSceneConfig(k_min=1, k_max=1)          # 1 target among the 4 (2x2) cells
    for seed in range(5):
        img, mask, k, _ = render_scene(np.random.default_rng(seed), scene, 2, CELL,
                                       obj(0.5), obj(0.9))
        assert mask.sum() > 0
        assert np.allclose(img[mask > 0], 0.5)         # target texture never overwritten


def test_placement_tries_reduces_overlap():
    # Rejection sampling (placement_tries>1) should place the same objects with less
    # mutual overlap than fully-random placement (tries=1). Measure overlap as
    # (sum of per-object mask areas - occupied area) / occupied area.
    big = CELL * 2                                    # oversized tiles -> overlap is likely
    def sampler(rng):
        return np.ones((big, big), dtype=np.uint8)    # img==mask, full tile

    def total_overlap(tries):
        scene = OmniSceneConfig(placement="random", k_min=1, k_max=1, max_nb_objects=6,
                                placement_tries=tries, placement_max_overlap=0.0)
        # sum of individual footprints vs union area on the image (all objects paint 1s)
        rng = np.random.default_rng(3)
        areas, unions = 0, 0
        for _ in range(20):
            img, mask, k, _ = render_scene(rng, scene, GRID, CELL, sampler, sampler)
            unions += int((img > 0).sum())
            areas += 6 * big * big                    # 6 full-tile objects (may clip at edges)
        return areas / max(unions, 1)                 # 1.0 => no overlap; higher => more overlap

    assert total_overlap(16) < total_overlap(1)       # fewer overlaps with rejection sampling


def test_black_background_unchanged():
    # Default black background: canvas stays zero where nothing is painted.
    scene = OmniSceneConfig(k_min=1, k_max=1, max_nb_objects=1)
    rng = np.random.default_rng(12)
    img, mask, k, _ = render_scene(rng, scene, GRID, CELL,
                                   _const_sampler(1), _const_sampler(0))
    assert (img[mask == 0] == 0).all()              # untouched background is black


if __name__ == "__main__":
    test_shapes_and_k_range()
    test_mask_marks_exactly_k_cells()
    test_mask_is_target_cells_not_distractors()
    test_k_clamped_to_valid_range()
    test_oversized_tiles_overflow_and_union()
    test_affine_jitter_preserves_shape_and_binary()
    test_two_channel_objects_paste_intensity_to_image_mask_to_label()
    test_affine_jitter_two_channel_keeps_intensity_and_binarizes_mask()
    test_random_background_nonblack_and_dark_object_visible()
    test_image_background_used_and_objects_painted_over()
    test_targets_drawn_over_distractors()
    test_placement_tries_reduces_overlap()
    test_black_background_unchanged()
    print("ALL RENDER TESTS PASSED")

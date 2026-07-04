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


if __name__ == "__main__":
    test_shapes_and_k_range()
    test_mask_marks_exactly_k_cells()
    test_mask_is_target_cells_not_distractors()
    test_k_clamped_to_valid_range()
    test_oversized_tiles_overflow_and_union()
    test_affine_jitter_preserves_shape_and_binary()
    print("ALL RENDER TESTS PASSED")

"""Unit tests for PrimusEncoder native-grid helpers (no model weights loaded)."""
import torch
from timm.layers import RotaryEmbeddingCat

from src.models.primus_encoder import _native_target_shape, _set_rope_scaled_grid


def test_native_target_shape_divisible_is_passthrough():
    assert _native_target_shape((128, 128, 128), 8) == (128, 128, 128)
    assert _native_target_shape((192, 192, 192), 8) == (192, 192, 192)


def test_native_target_shape_rounds_to_nearest_multiple():
    # 130 -> 128 (nearest), 132 -> 136 (ties/above), min is one patch
    assert _native_target_shape((130, 130, 130), 8) == (128, 128, 128)
    assert _native_target_shape((4, 4, 4), 8) == (8, 8, 8)


def _make_rope(grid):
    # Mirror Primus' construction: fixed feat_shape (bands=None), identity ref.
    dim = 24  # rope_dim; divisible by 4
    return RotaryEmbeddingCat(dim, in_pixels=False, feat_shape=list(grid),
                              ref_feat_shape=list(grid))


def test_set_rope_scaled_grid_rebuilds_for_new_grid():
    rope = _make_rope((24, 24, 24))
    _set_rope_scaled_grid(rope, (16, 16, 16), spacing_mm=2.0, train_mm=2.0)
    assert tuple(rope.feat_shape) == (16, 16, 16)
    assert tuple(rope.ref_feat_shape) == (16, 16, 16)          # spacing==train → identity ref
    # pos_embed rows == number of tokens in the new grid
    assert rope.pos_embed.shape[0] == 16 ** 3


def test_set_rope_scaled_grid_identity_at_train_pitch():
    # spacing == train pitch must reproduce the plain identity table (ref == grid).
    ref = _make_rope((24, 24, 24)); _set_rope_scaled_grid(ref, (16, 16, 16), 2.0, 2.0)
    got = _make_rope((24, 24, 24)); _set_rope_scaled_grid(got, (16, 16, 16), 5.0, 5.0)
    assert torch.equal(ref.pos_embed, got.pos_embed)


def test_set_rope_scaled_grid_scales_ref_with_spacing():
    rope = _make_rope((24, 24, 24))
    _set_rope_scaled_grid(rope, (16, 16, 16), spacing_mm=4.0, train_mm=2.0)  # 2x pitch
    assert list(rope.ref_feat_shape) == [32.0, 32.0, 32.0]
    # anisotropic spacing → per-axis ref
    _set_rope_scaled_grid(rope, (16, 16, 16), spacing_mm=[1.0, 2.0, 4.0], train_mm=2.0)
    assert list(rope.ref_feat_shape) == [8.0, 16.0, 32.0]


def test_set_rope_scaled_grid_inplace_and_stable():
    rope = _make_rope((24, 24, 24))
    _set_rope_scaled_grid(rope, (16, 16, 16), 2.0, 2.0)   # first build for this grid (assign)
    buf = rope.pos_embed
    _set_rope_scaled_grid(rope, (16, 16, 16), 4.0, 2.0)   # value change: in-place copy_
    assert rope.pos_embed is buf                          # same tensor object (compile-safe)
    emb = rope.pos_embed.clone()
    _set_rope_scaled_grid(rope, (16, 16, 16), 4.0, 2.0)   # same args again → identical
    assert torch.equal(emb, rope.pos_embed)

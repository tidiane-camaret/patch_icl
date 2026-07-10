import sys; sys.path.insert(0, ".")
import pytest
import torch
from src.models.patchset_cnn import PatchSetCNN


def _model(resolutions, H=32, refine_mode="reencode"):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolutions[0], enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, resolutions=resolutions, refine_mode=refine_mode)


def _batch(B=2, K=2, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_single_level_unchanged():
    m = _model([8])
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert set(out) == {"final_logit"}
    assert out["final_logit"].shape == (2, 1, 8, 8)
    assert torch.equal(out["final_logit"], m._segment(img, cin, cout))


def test_multi_level_heads_and_derived_crop():
    m = _model([8, 16])                       # image_size 32 → crop = 32*8/16 = 16
    assert m.refine_crops == [16]
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)     # coarse, T=8
    assert out["refine_logit"].shape == (2, 1, 8, 8)    # refine, same T
    assert out["refine_origin"].shape == (2, 2)
    assert out["refine_crop"] == 16
    assert out["resolutions"] == [8, 16]


def test_derived_crop_full_zoom():
    m = _model([8, 32])                       # crop = 32*8/32 = 8
    assert m.refine_crops == [8]


def test_invalid_resolutions_rejected():
    with pytest.raises(AssertionError):
        _model([8, 12])                       # 12 % 8 != 0


def test_grad_reaches_shared_weights_from_both_heads():
    m = _model([8, 16])
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    loss = out["final_logit"].mean() + out["refine_logit"].mean()   # both go through _segment
    loss.backward()
    assert m.decoder[0].weight.grad is not None
    assert m.encoder.stem[0].weight.grad is not None


def test_multi_level_returns_ctx_origin():
    m = _model([8, 16])                       # image_size 32 → crop = 16
    img, cin, cout = _batch(B=2, K=2, H=32)
    out = m(img, context_in=cin, context_out=cout)
    assert out["refine_ctx_origin"].shape == (2, 2, 2)          # (B, K, 2)
    # origins are in-bounds top-left px for a crop of size 16 on a 32 image
    assert (out["refine_ctx_origin"] >= 0).all()
    assert (out["refine_ctx_origin"] <= 32 - 16).all()


def test_single_level_has_no_ctx_origin():
    m = _model([8])
    img, cin, cout = _batch(H=32)
    assert "refine_ctx_origin" not in m(img, context_in=cin, context_out=cout)


# ── encode_once refine mode ────────────────────────────────────────────────

def test_default_refine_mode_is_reencode():
    assert _model([8, 16]).refine_mode == "reencode"


def test_invalid_refine_mode_rejected():
    with pytest.raises(AssertionError):
        _model([8, 16], refine_mode="bogus")


def test_encode_once_same_heads_and_shapes():
    m = _model([8, 16], refine_mode="encode_once")
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)     # coarse, T=8
    assert out["refine_logit"].shape == (2, 1, 8, 8)    # refine, same T
    assert out["refine_origin"].shape == (2, 2)
    assert out["refine_ctx_origin"].shape == (2, 2, 2)
    assert out["refine_crop"] == 16
    assert out["resolutions"] == [8, 16]


def test_encode_once_coarse_matches_single_pass():
    # The coarse head is a full-image _segment, so it must equal the plain single-level model.
    m = _model([8, 16], refine_mode="encode_once")
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert torch.allclose(out["final_logit"], m._segment(img, cin, cout), atol=1e-5)


def test_encode_once_grad_reaches_shared_weights_from_both_heads():
    m = _model([8, 16], refine_mode="encode_once")
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    (out["final_logit"].mean() + out["refine_logit"].mean()).backward()
    assert m.decoder[0].weight.grad is not None
    # encoder runs ONCE but the refine crop (grid_sample) is differentiable → grad still flows.
    assert m.encoder.stem[0].weight.grad is not None


# ── refine memory ──────────────────────────────────────────────────────

def test_refine_memory_default_off_no_param():
    m = _model([8, 16])
    assert m.refine_memory is False
    assert not hasattr(m, "mem_type")
    assert "mem_type" not in dict(m.named_parameters())


def test_refine_memory_on_creates_mem_type():
    m = PatchSetCNN(image_size=32, resolution=8, enc_dims=[16], e=32, h=64, l=1, a=2,
                    thinking_rows=1, resolutions=[8, 16], refine_memory=True)
    assert m.refine_memory is True
    assert m.mem_type.shape == (32,)
    assert "mem_type" in dict(m.named_parameters())

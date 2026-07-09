import sys; sys.path.insert(0, ".")
import pytest
import torch
from src.models.patchset_cnn import PatchSetCNN


def _model(resolutions, H=32):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolutions[0], enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, resolutions=resolutions)


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

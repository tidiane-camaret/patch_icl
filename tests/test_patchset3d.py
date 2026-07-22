import torch
from src.models.patchset3d import ConvEncoder3D, _mask_tiles_3d


def test_conv_encoder3d_shape():
    enc = ConvEncoder3D(1, (8, 8, 8), resolution=4)
    assert enc.out_ch == 24
    out = enc(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 24, 4, 4, 4)


def test_mask_tiles_3d_shape_and_occupancy():
    m = torch.zeros(2, 1, 8, 8, 8)
    m[:, :, :4, :4, :4] = 1.0
    tiles = _mask_tiles_3d(m, grid_res=4, p=2)   # 4^3 cells, 2^3 tile
    assert tiles.shape == (2, 64, 8)
    # cell (0,0,0) fully inside the ones block -> all-ones tile
    assert torch.allclose(tiles[0, 0], torch.ones(8))
from src.models.patchset3d import PatchSet3D


def _dummy_batch(B=2, K=2, S=16):
    image = torch.randn(B, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    return image, context_in, context_out


def test_patchset3d_forward_grid_shape():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                   thinking_rows=2, fourier_bands=4, mask_patch_decode_size=2)
    assert m.grid_size == 8
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 8, 8, 8)


def test_patchset3d_backward():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    out.mean().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0


def test_predict_and_train_forward_native_shape():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    img, cin, cout = _dummy_batch(S=16)
    logits = m.train_forward(img, cin, cout)
    assert logits.shape == (2, 1, 16, 16, 16)
    pred = m.predict(img, cin, cout)
    assert pred.shape == (2, 16, 16, 16)
    assert set(torch.unique(pred).tolist()) <= {0.0, 1.0}

# tests/test_feature_sim_adapters.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from src.models.patchset3d import PatchSet3D
from feature_sim.adapters import PatchSet3DEncoderAdapter


def _model():
    return PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                      thinking_rows=2, fourier_bands=4)


def _vols(B=2, S=16):
    return torch.randn(B, 1, S, S, S)


def test_tiers_and_native_res():
    ad = PatchSet3DEncoderAdapter(_model())
    ts = ad.tiers()
    assert "concat" in ts and "img_embed" in ts and "stage:0" in ts
    assert ad.R == 4
    # stem is full res, stage:1 is halved
    assert ad.native_res("stage:0", 16) == 16
    assert ad.native_res("stage:1", 16) == 8
    assert ad.native_res("concat", 16) == 16


def test_features_dense_shapes():
    ad = PatchSet3DEncoderAdapter(_model())
    v = _vols()
    f = ad.features(v, "concat", res=8)
    assert f.shape[0] == 2 and f.shape[2:] == (8, 8, 8)
    fs = ad.features(v, "stage:1", res=6)
    assert fs.shape[2:] == (6, 6, 6)
    fe = ad.features(v, "img_embed", res=4)
    assert fe.shape[1] == 32 and fe.shape[2:] == (4, 4, 4)   # e=32 channels


def test_sample_features_shape():
    ad = PatchSet3DEncoderAdapter(_model())
    v = _vols()
    coords = torch.rand(2, 20, 3) * 2 - 1
    s = ad.sample_features(v, "concat", coords)
    assert s.shape[:2] == (2, 20) and s.shape[2] == ad._concat_ch


def test_transformer_query_shape():
    m = _model(); ad = PatchSet3DEncoderAdapter(m)
    img = torch.randn(2, 1, 16, 16, 16)
    cin = torch.randn(2, 2, 1, 16, 16, 16)
    cout = (torch.rand(2, 2, 16, 16, 16) > 0.5).float()
    q = ad.transformer_query(img, cin, cout)
    assert q.shape == (2, ad.R ** 3, 32)                     # (B, N, e)


def test_sample_features_matches_dense_at_voxel():
    torch.manual_seed(0)
    ad = PatchSet3DEncoderAdapter(_model())
    v = torch.randn(1, 1, 16, 16, 16)
    dense = ad.features(v, "stage:0", res=16)[0]        # (C,16,16,16); stem is stride-1 -> native 16
    d, h, w = 2, 5, 9                                   # asymmetric voxel: transposed order would differ
    expected = dense[:, d, h, w]
    coord = torch.tensor([[d, h, w]], dtype=torch.float) / (16 - 1) * 2 - 1   # (z,y,x) normalized
    got = ad.sample_features(v, "stage:0", coord.unsqueeze(0))[0, 0]          # (C,)
    assert torch.allclose(got, expected, atol=1e-4)

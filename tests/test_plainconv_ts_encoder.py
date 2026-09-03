"""Shape / contract / normalization checks for PlainConvTSEncoder (encoder=plainconv_ts)."""
import torch

from src.models.encoders.plainconv_ts import PlainConvTSEncoder
from src.totalseg_dataset import CT_NORM_PRESETS


def test_out_ch_and_forward_shape():
    enc = PlainConvTSEncoder(resolution=16, n_stages=5, stages=(2, 3, 4),
                             frozen=False, device="cpu", precision="fp32")
    assert enc.out_ch == 128 + 256 + 320       # stages 2,3,4 of [32,64,128,256,320]
    assert enc.supports_fine and enc.n_fine_stages == 5
    assert enc.input_norm == "zscore"          # from-scratch default (differs from resenc_ts)
    out = enc(torch.randn(2, 1, 64, 64, 64))
    assert out.shape == (2, 704, 16, 16, 16)


def test_features_per_stage_override():
    enc = PlainConvTSEncoder(resolution=8, stages=(1, 2, 3),
                             features_per_stage=[16, 32, 64, 128, 160],
                             input_norm="passthrough", frozen=False,
                             device="cpu", precision="fp32")
    assert enc.n_stages == 5
    assert enc.stage_ch == [16, 32, 64, 128, 160]
    assert enc.out_ch == 32 + 64 + 128
    out = enc(torch.randn(2, 1, 64, 64, 64))
    assert out.shape == (2, 224, 8, 8, 8)


def test_fine_stage_geometry_and_taps():
    enc = PlainConvTSEncoder(resolution=16, n_stages=5, stages=(2, 3, 4),
                             frozen=False, device="cpu", precision="fp32")
    # stage divisors: [1, 2, 4, 8, 16]
    assert [enc.fine_stage_size(128, s) for s in range(5)] == [128, 64, 32, 16, 8]
    assert enc.fine_stage_channels(1) == 64
    coarse, fine = enc(torch.randn(3, 1, 64, 64, 64),
                       fine_rows=torch.tensor([0, 2]), fine_stage=(1,))
    assert coarse.shape == (3, 704, 16, 16, 16)
    assert fine[0].shape == (2, 64, 32, 32, 32)


def test_passthrough_norm_is_identity():
    enc = PlainConvTSEncoder(resolution=8, n_stages=4, stages=(1, 2, 3),
                             input_norm="passthrough", device="cpu", precision="fp32")
    x = torch.randn(2, 1, 16, 16, 16)
    assert torch.equal(enc._norm(x), x.float())


def test_reframe_norm_matches_manual_roundtrip():
    enc = PlainConvTSEncoder(resolution=8, n_stages=4, stages=(1, 2, 3),
                             input_norm="reframe", loader_ct_norm="fingerprint_1228",
                             target_ct_norm="d297", device="cpu", precision="fp32")
    ld, tg = CT_NORM_PRESETS["fingerprint_1228"], CT_NORM_PRESETS["d297"]
    x = torch.randn(1, 1, 8, 8, 8)
    hu = x.float() * ld.std + ld.mean
    want = (hu.clamp(tg.clip_lo, tg.clip_hi) - tg.mean) / tg.std
    assert torch.allclose(enc._norm(x), want, atol=1e-5)


def test_zscore_norm_path_runs():
    enc = PlainConvTSEncoder(resolution=8, n_stages=4, stages=(1, 2, 3), input_norm="zscore",
                             frozen=False, device="cpu", precision="fp32")
    out = enc(torch.randn(1, 1, 32, 32, 32))
    assert out.shape == (1, 64 + 128 + 256, 8, 8, 8)


def test_instance_norm_no_hu_inversion():
    from src.models.encoders.plainconv_ts import PlainConvTSEncoder
    enc = PlainConvTSEncoder(resolution=8, n_stages=4, stages=(1, 2, 3),
                             input_norm="instance", frozen=False,
                             device="cpu", precision="fp32")
    x = torch.randn(2, 1, 16, 16, 16) * 3.0 + 1.0
    flat = x.float().reshape(2, -1)
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    assert torch.allclose(enc._norm(x), (x.float() - mu) / (sig + 1e-8), atol=1e-6)

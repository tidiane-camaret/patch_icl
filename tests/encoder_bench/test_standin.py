import torch
from encoder_bench import registry as R
from encoder_bench.encoders_standin import PrimusStandin


def test_primus_forward_shapes():
    m = PrimusStandin(img_size=64, patch=8, embed_dim=96, depth=2, heads=3).eval()
    for size in (32, 64):                      # variable input -> pos-embed interpolates
        y = m(torch.zeros(1, 1, size, size, size))
        n = (size // 8) ** 3
        assert y.shape == (1, n, 96)


def test_primus_registered():
    assert "primus" in R.list_encoders()
    assert R.REGISTRY["primus"].family == "transformer"
    assert R.REGISTRY["primus"].size_multiple == 8

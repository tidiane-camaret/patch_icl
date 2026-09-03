"""Unit tests for the shared InputRenorm stem (src/models/encoders/_input_norm.py)."""
import pytest
import torch

from src.models.encoders._input_norm import InputRenorm, _INPUT_NORMS
from src.totalseg_dataset import CT_NORM_PRESETS


LD = CT_NORM_PRESETS["fingerprint_1228"]
TG = CT_NORM_PRESETS["d297"]


def test_enum_has_instance():
    assert _INPUT_NORMS == ("passthrough", "reframe", "zscore", "instance")


def test_passthrough_is_identity_float():
    m = InputRenorm("passthrough")
    x = torch.randn(2, 1, 8, 8, 8, dtype=torch.float64)
    out = m(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, x.float())


def test_reframe_matches_inline_math():
    m = InputRenorm("reframe", loader_spec=LD, target_spec=TG)
    x = torch.randn(3, 1, 8, 8, 8)
    hu = x.float() * LD.std + LD.mean
    want = (hu.clamp(TG.clip_lo, TG.clip_hi) - TG.mean) / TG.std
    assert torch.allclose(m(x), want, atol=1e-6)


def test_zscore_matches_inline_math():
    m = InputRenorm("zscore", loader_spec=LD)
    x = torch.randn(3, 1, 8, 8, 8)
    hu = x.float() * LD.std + LD.mean
    flat = hu.reshape(hu.shape[0], -1)
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    want = (hu - mu) / (sig + 1e-8)
    assert torch.allclose(m(x), want, atol=1e-6)


def test_instance_is_zscore_without_hu_inversion():
    m = InputRenorm("instance")
    x = torch.randn(3, 1, 8, 8, 8) * 5.0 + 2.0
    flat = x.float().reshape(x.shape[0], -1)
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    want = (x.float() - mu) / (sig + 1e-8)
    assert torch.allclose(m(x), want, atol=1e-6)
    # per-sample standardized: mean ~0, std ~1
    per = m(x).reshape(3, -1)
    assert torch.allclose(per.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(per.std(dim=1), torch.ones(3), atol=1e-3)


def test_instance_affine_has_params_and_defaults_identity():
    m = InputRenorm("instance", affine=True)
    names = [n for n, _ in m.named_parameters()]
    assert names == ["gamma", "beta"]
    x = torch.randn(2, 1, 8, 8, 8)
    base = InputRenorm("instance")
    assert torch.allclose(m(x), base(x), atol=1e-6)   # gamma=1, beta=0 at init


def test_affine_false_registers_no_state():
    m = InputRenorm("instance", affine=False)
    assert list(m.parameters()) == []
    assert list(m.buffers()) == []
    assert m.state_dict() == {}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        InputRenorm("bogus")


def test_zscore_requires_loader_spec():
    with pytest.raises(ValueError):
        InputRenorm("zscore")


def test_reframe_requires_both_specs():
    with pytest.raises(ValueError):
        InputRenorm("reframe", loader_spec=LD)


def test_affine_rejected_for_non_instance_modes():
    with pytest.raises(ValueError):
        InputRenorm("zscore", loader_spec=CT_NORM_PRESETS["fingerprint_1228"], affine=True)

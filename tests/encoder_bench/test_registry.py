import torch
from encoder_bench import registry as R


def test_trivial_encoders_registered():
    names = R.list_encoders()
    assert "conv_encoder3d" in names and "resenc" in names


def test_make_inputs_conventions():
    spec_single = R.REGISTRY["conv_encoder3d"]
    spec_mask = R.REGISTRY["resenc"]
    x = torch.zeros(1, 1, 8, 8, 8)
    assert len(R.make_inputs(spec_single, x)) == 1
    assert len(R.make_inputs(spec_mask, x)) == 2


def test_factories_build_and_run_tiny():
    x = torch.zeros(1, 1, 16, 16, 16)
    for name in ("conv_encoder3d", "resenc"):
        spec = R.REGISTRY[name]
        mod = spec.factory().eval()
        out = mod(*R.make_inputs(spec, x))
        # accept a tensor or a list/tuple of tensors
        t = out[0] if isinstance(out, (list, tuple)) else out
        assert torch.is_tensor(t)

import contextlib
import torch
from encoder_bench import registry as R
from encoder_bench import optimize as O


def test_apply_optimization_cpu_noop_compile():
    spec = R.REGISTRY["conv_encoder3d"]
    mod = spec.factory()
    out_mod, ctx = O.apply_optimization(mod, spec.opt_profile, torch.device("cpu"))
    assert isinstance(out_mod, torch.nn.Module)
    # on CPU autocast/compile are disabled -> ctx is a nullcontext
    with ctx:
        x = torch.zeros(1, 1, 16, 16, 16)
        y = out_mod(*R.make_inputs(spec, x))
    assert y is not None


def test_set_compiler_env_runs():
    O.set_compiler_env()  # must not raise

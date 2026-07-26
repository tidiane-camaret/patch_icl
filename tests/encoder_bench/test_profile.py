import torch
import torch.nn as nn
from encoder_bench import registry as R
from encoder_bench import profile as P


class _BoomModule(nn.Module):
    """Minimal module whose forward always raises RuntimeError."""

    def forward(self, x):
        raise RuntimeError("boom")


def test_profile_point_error_path():
    """Graceful error row when forward raises a non-OOM exception."""
    spec = R.EncoderSpec(
        name="boom", family="cnn",
        factory=_BoomModule, call="single",
    )
    row = P.profile_point(spec, input_size=16, device=torch.device("cpu"),
                          n_warmup=1, n_timed=2)
    assert row["status"] == "error:RuntimeError"
    assert row["fwd_bwd_ms"] is None


def test_profile_point_cpu_conv():
    spec = R.REGISTRY["conv_encoder3d"]
    row = P.profile_point(spec, input_size=16, device=torch.device("cpu"),
                          n_warmup=1, n_timed=2)
    assert row["status"] == "ok"
    assert row["params"] > 0
    assert row["fwd_bwd_ms"] is not None and row["fwd_bwd_ms"] > 0
    # gflops may be None if fvcore missing, else finite positive
    assert row["gflops"] is None or row["gflops"] > 0
    # VRAM is CUDA-only -> None on CPU
    assert row["train_vram_mb"] is None


def test_profile_point_divisibility_skip():
    spec = R.EncoderSpec(name="fake", family="cnn",
                         factory=lambda: torch.nn.Conv3d(1, 1, 3, padding=1),
                         call="single", size_multiple=32)
    row = P.profile_point(spec, input_size=48, device=torch.device("cpu"))
    assert row["status"] == "skip:divisible"
    assert row["fwd_bwd_ms"] is None


class _MultiOutputModule(nn.Module):
    """Module with TWO independent branches returning a list; backward from only out[0]
    would miss conv2 entirely (no shared path), so the all-outputs fix is needed."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, 3, padding=1)   # branch A
        self.conv2 = nn.Conv3d(1, 8, 3, padding=1)   # branch B — independent

    def forward(self, x):
        return [self.conv1(x), self.conv2(x)]


def test_profile_point_multi_output_backward():
    """Backward over all outputs: multi-output encoder gets status ok and fwd_bwd_ms,
    and both branches receive gradients (proving full-graph coverage)."""
    spec = R.EncoderSpec(
        name="multi_out", family="cnn",
        factory=_MultiOutputModule, call="single",
    )
    mod = _MultiOutputModule()
    row = P.profile_point(spec, input_size=16, device=torch.device("cpu"),
                          module=mod, n_warmup=1, n_timed=2)
    assert row["status"] == "ok"
    assert row["fwd_bwd_ms"] is not None
    # Both independent branches must have received gradients.
    assert mod.conv1.weight.grad is not None, "branch A grad missing"
    assert mod.conv2.weight.grad is not None, "branch B grad missing (multi-out bug)"

"""Apply the best-optimized config per encoder (channels_last, bf16, compile)."""
import contextlib
import os
from pathlib import Path

import torch


def set_compiler_env() -> None:
    """thor/odin: bare g++ resolves to /bin/g++ with a broken prefix; force /usr/bin."""
    if not Path("/bin").is_symlink():
        os.environ.setdefault("CC", "/usr/bin/gcc")
        os.environ.setdefault("CXX", "/usr/bin/g++")


def apply_optimization(module, opt_profile: dict, device):
    opt_profile = opt_profile or {}
    module = module.to(device)
    if opt_profile.get("channels_last") and device.type == "cuda":
        module = module.to(memory_format=torch.channels_last_3d)
    ctx = contextlib.nullcontext()
    if device.type == "cuda" and opt_profile.get("autocast") == "bf16":
        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    # Unconditional on CUDA: fixes the toolchain even for encoders that self-compile
    # internally (e.g. a wrapper calling torch.compile) without our compile branch.
    if device.type == "cuda":
        set_compiler_env()
    if device.type == "cuda" and opt_profile.get("compile"):
        # Isolate each (encoder, size) compile: clear dynamo's guard/cache state so this
        # module compiles from scratch. Otherwise guards from a prior size's compile of the
        # SAME forward code promote later sizes to symbolic shapes, and inductor fails to
        # lower shape-dependent ops (e.g. Primus's interpolate pos-embed -> LoweringException).
        torch._dynamo.reset()
        try:
            # dynamic=False: each input size gets its own static compile. The sweep
            # rebuilds the module per size, but dynamo's shape guards are keyed on the
            # shared forward code, so a second distinct shape would trigger automatic
            # dynamic shapes -- which breaks e.g. Primus's interpolate-pos-embed backward
            # lowering (symbolic-shape assertion). Static per-size is what we want here.
            module = torch.compile(module, mode=opt_profile["compile"], dynamic=False)
        except Exception:
            pass  # fall back to eager; benchmark still runs
    return module, ctx

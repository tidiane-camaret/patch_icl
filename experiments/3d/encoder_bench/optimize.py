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
    if device.type == "cuda" and opt_profile.get("compile"):
        set_compiler_env()
        try:
            module = torch.compile(module, mode=opt_profile["compile"])
        except Exception:
            pass  # fall back to eager; benchmark still runs
    return module, ctx

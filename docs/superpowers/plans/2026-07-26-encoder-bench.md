# Encoder compute/latency scaling benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `experiments/3d/encoder_bench/`, a standalone harness that measures how each 3D encoder's compute cost (fwd+bwd latency, training VRAM, FLOPs/params, throughput) scales with input size, each encoder at its best-optimized config, to find the crossover where transformer/Mamba overtake CNN.

**Architecture:** A registry maps encoder names to factories + a uniform call convention; a profiler measures one (encoder, input_size) point device-aware (CUDA events / CPU fallback); an optimizer applies the best per-encoder config (channels_last_3d, bf16 autocast, SDPA, torch.compile); a runner sweeps encoders × input sizes and writes CSV + scaling-curve PNGs (+ optional wandb). Roster = 7 existing encoders + 2 compute-only stand-ins (Primus, SegMamba).

**Tech Stack:** PyTorch 2.5.1+cu121, fvcore (FLOPs), matplotlib (curves), wandb (optional), pytest. Spec: `docs/superpowers/specs/2026-07-26-encoder-bench-design.md`.

## Global Constraints

- Run everything with **`.venv_thor/bin/python`** (torch 2.5.1+cu121, RTX A6000 sm_86). Tests: `.venv_thor/bin/python -m pytest`.
- **`torch.compile` on thor requires `export CXX=/usr/bin/g++ CC=/usr/bin/gcc`** (set automatically by `optimize.py` when `/bin` is not a symlink to `/usr/bin`) or inductor's C++ build fails.
- **Unit tests must be CPU-runnable** (tiny inputs, no CUDA required). CUDA-only metrics (VRAM, compile, throughput search) degrade gracefully to `None`/skipped off-GPU.
- **Weight loading is off by default.** Random/`meta` init gives identical compute. Encoders that need a checkpoint just to build the architecture (`nninteractive`, `threedino`) are gated: registered but skipped-with-logged-`NaN` when the checkpoint/dep is unavailable.
- Log a one-line summary of the finished harness to `docs/logs.md`.
- Follow existing repo style: short docstrings, `experiments/3d/` layout conventions (mirror `feature_sim/`).

## File Structure

- `experiments/3d/encoder_bench/__init__.py` — package marker.
- `experiments/3d/encoder_bench/registry.py` — `EncoderSpec` dataclass + `REGISTRY: dict[str, EncoderSpec]` + `register()` + `make_inputs`/call convention. Zoo + stand-in specs registered here (importing factories).
- `experiments/3d/encoder_bench/encoders_standin.py` — `PrimusStandin`, `SegMambaStandin` (compute-only nn.Modules).
- `experiments/3d/encoder_bench/optimize.py` — `apply_optimization(module, profile, device)` + `set_compiler_env()`.
- `experiments/3d/encoder_bench/profile.py` — `profile_point(spec, input_size, device, opt)` → metrics dict.
- `experiments/3d/encoder_bench/run.py` — sweep entry: CSV + PNG + optional wandb.
- `tests/encoder_bench/test_registry.py`, `test_profile.py`, `test_optimize.py`, `test_run.py`, `test_standin.py` — CPU unit tests.

---

### Task 1: Registry protocol + two trivial encoders

**Files:**
- Create: `experiments/3d/encoder_bench/__init__.py` (empty)
- Create: `experiments/3d/encoder_bench/registry.py`
- Test: `tests/encoder_bench/test_registry.py`

**Interfaces:**
- Produces:
  - `@dataclass EncoderSpec(name: str, family: str, factory: Callable[..., nn.Module], call: str, in_ch: int = 1, size_multiple: int = 1, requires_ckpt: bool = False, opt_profile: dict | None = None)` where `call ∈ {"single","img_mask"}`.
  - `REGISTRY: dict[str, EncoderSpec]`.
  - `register(spec: EncoderSpec) -> None`.
  - `make_inputs(spec: EncoderSpec, x: torch.Tensor) -> tuple[torch.Tensor, ...]` — `(x,)` for `"single"`, `(x, torch.zeros_like(x))` for `"img_mask"`.
  - `list_encoders() -> list[str]` (sorted names).
  - Registers `conv_encoder3d` (family `cnn`, call `single`) and `resenc` (family `cnn`, call `img_mask`).

- [ ] **Step 1: Write the failing test**

```python
# tests/encoder_bench/test_registry.py
import torch
from experiments.encoder_bench_path import *  # noqa  (see conftest note below)
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
```

Add `tests/encoder_bench/conftest.py` to put the package on `sys.path` (mirrors how `feature_sim` tests resolve imports):

```python
# tests/encoder_bench/conftest.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                                  # repo root (for `src`)
sys.path.insert(0, str(ROOT / "experiments" / "3d"))           # for `encoder_bench`
```

Delete the bogus `from experiments.encoder_bench_path import *` line — replace test imports with `from encoder_bench import registry as R` (conftest makes it importable).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'encoder_bench'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/3d/encoder_bench/registry.py
"""Encoder registry: name -> build recipe + uniform call convention for the bench."""
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from src.models.patchset3d import ConvEncoder3D
from src.models.encoders import ResEncEncoder


@dataclass
class EncoderSpec:
    name: str
    family: str                       # "cnn" | "transformer" | "mamba"
    factory: Callable[..., nn.Module]
    call: str = "single"              # "single" -> module(x); "img_mask" -> module(x, zeros)
    in_ch: int = 1
    size_multiple: int = 1            # input D=H=W must be divisible by this
    requires_ckpt: bool = False
    opt_profile: dict = field(default_factory=dict)


REGISTRY: dict[str, EncoderSpec] = {}


def register(spec: EncoderSpec) -> None:
    REGISTRY[spec.name] = spec


def list_encoders() -> list[str]:
    return sorted(REGISTRY)


def make_inputs(spec: EncoderSpec, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if spec.call == "img_mask":
        return (x, torch.zeros_like(x))
    return (x,)


# --- trivial, weights-free encoders ---------------------------------------
register(EncoderSpec(
    name="conv_encoder3d", family="cnn", call="single",
    factory=lambda: ConvEncoder3D(in_ch=1, dims=(32, 32, 32, 32), resolution=16),
    opt_profile={"autocast": "bf16", "channels_last": True, "compile": "reduce-overhead"},
))
register(EncoderSpec(
    name="resenc", family="cnn", call="img_mask",
    factory=lambda: ResEncEncoder(in_channels=1, features_per_stage=(32, 64, 128, 256)),
    opt_profile={"autocast": "bf16", "channels_last": True, "compile": "reduce-overhead"},
))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_registry.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/encoder_bench/__init__.py experiments/3d/encoder_bench/registry.py tests/encoder_bench/
git commit -m "feat(encoder-bench): registry + conv/resenc specs"
```

---

### Task 2: Profiler (device-aware measurement core)

**Files:**
- Create: `experiments/3d/encoder_bench/profile.py`
- Test: `tests/encoder_bench/test_profile.py`

**Interfaces:**
- Consumes: `EncoderSpec`, `make_inputs` from Task 1.
- Produces:
  - `count_params(module) -> int`
  - `count_gflops(module, inputs) -> float | None` (fvcore; `None` if unavailable)
  - `profile_point(spec, input_size: int, device: torch.device, module=None, n_warmup=3, n_timed=10) -> dict` returning keys: `encoder, family, input_size, params, gflops, fwd_bwd_ms, train_vram_mb, throughput_vol_s, status`. On unsupported size (`input_size % spec.size_multiple != 0`) or OOM → numeric fields `None`, `status` set to `"skip:divisible"` / `"oom"` / `"ok"`. `module` optional so a caller can pass a pre-optimized module.

- [ ] **Step 1: Write the failing test**

```python
# tests/encoder_bench/test_profile.py
import math
import torch
from encoder_bench import registry as R
from encoder_bench import profile as P


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'encoder_bench.profile'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/3d/encoder_bench/profile.py
"""Device-aware measurement of one (encoder, input_size) point."""
import time

import torch

from encoder_bench.registry import EncoderSpec, make_inputs


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_gflops(module, inputs) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None
    try:
        flops = FlopCountAnalysis(module, inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception:
        return None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_fwd_bwd(module, inputs, device, n_warmup, n_timed) -> float:
    """Median forward+backward wall-time (ms). Uses a scalar .sum() surrogate loss."""
    def one():
        for p in module.parameters():
            p.grad = None
        out = module(*inputs)
        t = out[0] if isinstance(out, (list, tuple)) else out
        t.float().sum().backward()
    for _ in range(n_warmup):
        one(); _sync(device)
    times = []
    for _ in range(n_timed):
        _sync(device); t0 = time.perf_counter()
        one(); _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _peak_vram_mb(device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def _throughput(module, spec, input_size, device) -> float | None:
    """Largest batch that fits (exponential search) -> volumes/sec, fwd-only no_grad."""
    if device.type != "cuda":
        with torch.no_grad():
            x = torch.zeros(1, spec.in_ch, input_size, input_size, input_size)
            t0 = time.perf_counter(); module(*make_inputs(spec, x)); dt = time.perf_counter() - t0
        return 1.0 / dt if dt > 0 else None
    best_b, bs = None, 1
    while True:
        try:
            with torch.no_grad():
                x = torch.zeros(bs, spec.in_ch, input_size, input_size,
                                input_size, device=device)
                _sync(device); t0 = time.perf_counter()
                module(*make_inputs(spec, x)); _sync(device)
                dt = time.perf_counter() - t0
            best_b = (bs, dt); bs *= 2
            del x
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); break
        if bs > 256:
            break
    if best_b is None:
        return None
    b, dt = best_b
    return b / dt if dt > 0 else None


def profile_point(spec: EncoderSpec, input_size: int, device, module=None,
                  n_warmup=3, n_timed=10) -> dict:
    row = {"encoder": spec.name, "family": spec.family, "input_size": input_size,
           "params": None, "gflops": None, "fwd_bwd_ms": None,
           "train_vram_mb": None, "throughput_vol_s": None, "status": "ok"}
    if input_size % spec.size_multiple != 0:
        row["status"] = "skip:divisible"
        return row
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    try:
        if module is None:
            module = spec.factory()
        module = module.to(device).train()
        row["params"] = count_params(module)
        x = torch.zeros(1, spec.in_ch, input_size, input_size, input_size, device=device)
        inputs = make_inputs(spec, x)
        row["gflops"] = count_gflops(module, inputs)
        row["fwd_bwd_ms"] = _time_fwd_bwd(module, inputs, device, n_warmup, n_timed)
        row["train_vram_mb"] = _peak_vram_mb(device)
        row["throughput_vol_s"] = _throughput(module.eval(), spec, input_size, device)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_profile.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/encoder_bench/profile.py tests/encoder_bench/test_profile.py
git commit -m "feat(encoder-bench): device-aware profiler"
```

---

### Task 3: Optimizer (best-config application)

**Files:**
- Create: `experiments/3d/encoder_bench/optimize.py`
- Test: `tests/encoder_bench/test_optimize.py`

**Interfaces:**
- Consumes: `opt_profile` dict from `EncoderSpec`.
- Produces:
  - `set_compiler_env() -> None` — exports `CC=/usr/bin/gcc CXX=/usr/bin/g++` when `/bin` is not a symlink to `/usr/bin` (thor/odin gotcha); no-op otherwise.
  - `apply_optimization(module, opt_profile: dict, device) -> tuple[nn.Module, contextmanager]` — moves to `channels_last_3d` if requested, `torch.compile(mode=...)` on CUDA only, and returns `(module, autocast_ctx)` where `autocast_ctx` is `torch.autocast(device.type, dtype=bf16)` when `opt_profile["autocast"]=="bf16"` and device is CUDA, else a `nullcontext()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/encoder_bench/test_optimize.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_optimize.py -v`
Expected: FAIL — `No module named 'encoder_bench.optimize'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/3d/encoder_bench/optimize.py
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
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and opt_profile.get("compile"):
        set_compiler_env()
        try:
            module = torch.compile(module, mode=opt_profile["compile"])
        except Exception:
            pass  # fall back to eager; benchmark still runs
    return module, ctx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_optimize.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/encoder_bench/optimize.py tests/encoder_bench/test_optimize.py
git commit -m "feat(encoder-bench): optimization application (compile/bf16/channels_last)"
```

---

### Task 4: Runner (sweep → CSV + plots)

**Files:**
- Create: `experiments/3d/encoder_bench/run.py`
- Test: `tests/encoder_bench/test_run.py`

**Interfaces:**
- Consumes: `REGISTRY`, `profile_point` (Task 2), `apply_optimization` (Task 3).
- Produces:
  - `sweep(encoders: list[str], input_sizes: list[int], device, out_dir: Path, n_warmup=3, n_timed=10) -> list[dict]` — for each (encoder, size): build via `spec.factory()`, apply optimization, call `profile_point(..., module=opt_module)`. Writes `<out_dir>/encoder_bench.csv`.
  - `plot_curves(rows: list[dict], out_dir: Path) -> list[Path]` — one PNG per metric in `("fwd_bwd_ms","train_vram_mb","gflops")`, x=input_size, one log-y line per encoder colored by family. Returns PNG paths. Rows with `None` metric are dropped from that curve.
  - `main()` — argparse: `--encoders` (default all), `--input_sizes` (default `32 64 96 128`), `--out_dir` (default `results/encoder_bench`), `--device`, `--wandb_project`. Wandb-gated like `feature_sim/run.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/encoder_bench/test_run.py
import csv
import torch
from pathlib import Path
from encoder_bench import run as RUN


def test_sweep_writes_csv(tmp_path):
    rows = RUN.sweep(["conv_encoder3d"], [16], torch.device("cpu"),
                     tmp_path, n_warmup=1, n_timed=2)
    csv_path = tmp_path / "encoder_bench.csv"
    assert csv_path.exists()
    with open(csv_path) as fh:
        r = list(csv.DictReader(fh))
    assert r and r[0]["encoder"] == "conv_encoder3d"
    assert {"fwd_bwd_ms", "train_vram_mb", "gflops", "throughput_vol_s"} <= set(r[0])


def test_plot_curves_creates_png(tmp_path):
    rows = [{"encoder": "a", "family": "cnn", "input_size": 16, "fwd_bwd_ms": 1.0,
             "train_vram_mb": None, "gflops": 2.0, "throughput_vol_s": 3.0,
             "params": 10, "status": "ok"},
            {"encoder": "a", "family": "cnn", "input_size": 32, "fwd_bwd_ms": 4.0,
             "train_vram_mb": None, "gflops": 8.0, "throughput_vol_s": 1.0,
             "params": 10, "status": "ok"}]
    pngs = RUN.plot_curves(rows, tmp_path)
    assert any(p.exists() for p in pngs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_run.py -v`
Expected: FAIL — `No module named 'encoder_bench.run'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/3d/encoder_bench/run.py
"""Sweep encoders x input sizes -> CSV + scaling-curve PNGs (+ optional wandb)."""
import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from encoder_bench import registry as R                     # noqa: E402
from encoder_bench.profile import profile_point             # noqa: E402
from encoder_bench.optimize import apply_optimization       # noqa: E402

_FIELDS = ["encoder", "family", "input_size", "params", "gflops",
           "fwd_bwd_ms", "train_vram_mb", "throughput_vol_s", "status"]
_PLOT_METRICS = ("fwd_bwd_ms", "train_vram_mb", "gflops")
_FAMILY_COLOR = {"cnn": "tab:blue", "transformer": "tab:red", "mamba": "tab:green"}


def sweep(encoders, input_sizes, device, out_dir, n_warmup=3, n_timed=10):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in encoders:
        spec = R.REGISTRY[name]
        for size in input_sizes:
            if size % spec.size_multiple != 0:
                rows.append(profile_point(spec, size, device)); continue
            try:
                mod, ctx = apply_optimization(spec.factory(), spec.opt_profile, device)
            except Exception as e:                      # ckpt/dep missing -> log + skip
                r = profile_point(spec, size, device, module=torch.nn.Identity())
                r["status"] = f"unavailable:{type(e).__name__}"; rows.append(r); continue
            with ctx:
                rows.append(profile_point(spec, size, device, module=mod,
                                          n_warmup=n_warmup, n_timed=n_timed))
            print(f"  {name}@{size}: {rows[-1]['status']}")
    with open(out_dir / "encoder_bench.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _FIELDS})
    return rows


def plot_curves(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); paths = []
    encoders = sorted({r["encoder"] for r in rows})
    for metric in _PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(6, 4))
        drew = False
        for enc in encoders:
            pts = sorted([r for r in rows if r["encoder"] == enc
                          and r.get(metric) is not None], key=lambda r: r["input_size"])
            if not pts:
                continue
            fam = pts[0]["family"]
            ax.plot([p["input_size"] for p in pts], [p[metric] for p in pts],
                    marker="o", label=enc, color=_FAMILY_COLOR.get(fam)); drew = True
        if not drew:
            plt.close(fig); continue
        ax.set_yscale("log"); ax.set_xlabel("input size (D=H=W)"); ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs input size"); ax.legend(fontsize=7)
        p = out_dir / f"scaling_{metric}.png"; fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig); paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--input_sizes", nargs="*", type=int, default=[32, 64, 96, 128])
    ap.add_argument("--out_dir", default="results/encoder_bench")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()
    encoders = args.encoders or R.list_encoders()
    device = torch.device(args.device)
    rows = sweep(encoders, args.input_sizes, device, Path(args.out_dir))
    pngs = plot_curves(rows, Path(args.out_dir))
    print(f"Done. {len(rows)} rows, {len(pngs)} plots -> {args.out_dir}")
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project,
                   config={"encoders": encoders, "input_sizes": args.input_sizes})
        wandb.log({"encoder_bench/table": wandb.Table(
            columns=_FIELDS, data=[[r.get(k) for k in _FIELDS] for r in rows])})
        for p in pngs:
            wandb.log({f"encoder_bench/{p.stem}": wandb.Image(str(p))})
        wandb.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_run.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/encoder_bench/run.py tests/encoder_bench/test_run.py
git commit -m "feat(encoder-bench): sweep runner + scaling-curve plots"
```

---

### Task 5: Primus transformer stand-in

**Files:**
- Create: `experiments/3d/encoder_bench/encoders_standin.py`
- Modify: `experiments/3d/encoder_bench/registry.py` (register `primus`)
- Test: `tests/encoder_bench/test_standin.py`

**Interfaces:**
- Produces: `PrimusStandin(in_ch=1, img_size=64, patch=8, embed_dim=384, depth=12, heads=6, mlp_ratio=4.0)` — `forward(x) -> (B, N, embed_dim)` tokens. Conv3d patch-embed (stride=patch) → learnable pos-embed sized to the token grid, **interpolated** to the actual token count so variable input sizes work → `depth` pre-norm transformer blocks using `F.scaled_dot_product_attention`. `size_multiple = patch`. Registered as family `transformer`, call `single`.

- [ ] **Step 1: Write the failing test**

```python
# tests/encoder_bench/test_standin.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_standin.py -v`
Expected: FAIL — `No module named 'encoder_bench.encoders_standin'`.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/3d/encoder_bench/encoders_standin.py
"""Compute-only, architecturally-faithful encoder stand-ins (no pretrained weights).

Primus: high-res-token pure-ViT (arxiv 2503.01835). SegMamba: CNN-stem + SSM blocks
(arxiv 2401.13560). Faithful block structure/dims for FLOPs/latency/VRAM; NOT for Dice.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
        self.heads = heads
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, C // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (B,heads,N,d)
        a = F.scaled_dot_product_attention(q, k, v)               # -> FlashAttention-2
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
        return x + self.mlp(self.n2(x))


class PrimusStandin(nn.Module):
    def __init__(self, in_ch=1, img_size=64, patch=8, embed_dim=384, depth=12,
                 heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv3d(in_ch, embed_dim, patch, stride=patch)
        g = img_size // patch
        self.pos = nn.Parameter(torch.zeros(1, embed_dim, g, g, g))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(embed_dim, heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)                                          # (B,C,g,g,g)
        pos = F.interpolate(self.pos, size=x.shape[-3:], mode="trilinear",
                            align_corners=False)
        x = (x + pos).flatten(2).transpose(1, 2)                   # (B,N,C)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
```

Append to `registry.py`:

```python
from encoder_bench.encoders_standin import PrimusStandin  # add near other imports

register(EncoderSpec(
    name="primus", family="transformer", call="single", size_multiple=8,
    factory=lambda: PrimusStandin(img_size=64, patch=8, embed_dim=384, depth=12, heads=6),
    opt_profile={"autocast": "bf16", "compile": "max-autotune"},
))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_standin.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/encoder_bench/encoders_standin.py experiments/3d/encoder_bench/registry.py tests/encoder_bench/test_standin.py
git commit -m "feat(encoder-bench): Primus transformer stand-in"
```

---

### Task 6: SegMamba SSM stand-in

**Files:**
- Modify: `experiments/3d/encoder_bench/encoders_standin.py` (add `SegMambaStandin`)
- Modify: `experiments/3d/encoder_bench/registry.py` (register `segmamba`)
- Test: `tests/encoder_bench/test_standin.py` (add cases)

**Interfaces:**
- Produces: `SegMambaStandin(in_ch=1, dims=(32,64,128,256), d_state=16)` — conv stem + strided downsample stages; at each stage a tri-orientation SSM block. `forward(x) -> (B, dims[-1], d, h, w)` bottleneck map. Uses `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` **if importable**, else a pure-PyTorch reference sequential scan (`_ref_scan`) so it runs anywhere (CPU tests). Registered family `mamba`, call `single`, `size_multiple = 8` (3 stride-2 stages).

- [ ] **Step 1: Write the failing test (append to test_standin.py)**

```python
def test_segmamba_forward_and_registered():
    from encoder_bench.encoders_standin import SegMambaStandin
    m = SegMambaStandin(dims=(8, 16, 32, 64)).eval()
    y = m(torch.zeros(1, 1, 32, 32, 32))
    assert y.shape[0] == 1 and y.shape[1] == 64            # bottleneck channels
    assert y.shape[-1] == 32 // 8                          # 3 stride-2 stages
    assert "segmamba" in R.list_encoders()
    assert R.REGISTRY["segmamba"].family == "mamba"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_standin.py::test_segmamba_forward_and_registered -v`
Expected: FAIL — `cannot import name 'SegMambaStandin'`.

- [ ] **Step 3: Write minimal implementation (append to encoders_standin.py)**

```python
def _ref_scan(u, delta, A, B, C):
    """Pure-PyTorch fallback selective scan. u,delta:(b,d,l) A:(d,n) B,C:(b,n,l)."""
    b, d, l = u.shape
    n = A.shape[1]
    dA = torch.exp(delta.unsqueeze(-1) * A)                    # (b,d,l,n)
    dB = delta.unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)  # (b,d,l,n)
    x = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(l):
        x = dA[:, :, t] * x + dB[:, :, t] * u[:, :, t].unsqueeze(-1)   # (b,d,n)
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, t]))          # (b,d)
    return torch.stack(ys, dim=-1)                                    # (b,d,l)


class _SSM3D(nn.Module):
    """Minimal selective-SSM over a flattened 3D volume (single scan orientation)."""
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim, self.n = dim, d_state
        self.in_proj = nn.Linear(dim, dim)
        self.dt = nn.Linear(dim, dim)
        self.A = nn.Parameter(-torch.rand(dim, d_state))
        self.B = nn.Linear(dim, d_state); self.C = nn.Linear(dim, d_state)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):                                      # x: (B,dim,D,H,W)
        B_, dim, D, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)                    # (B,L,dim)
        u = F.silu(self.in_proj(seq))
        delta = F.softplus(self.dt(seq)).transpose(1, 2)      # (B,dim,L)
        Bm, Cm = self.B(seq).transpose(1, 2), self.C(seq).transpose(1, 2)  # (B,n,L)
        y = _selective_scan(u.transpose(1, 2), delta, self.A, Bm, Cm)      # (B,dim,L)
        y = self.out(y.transpose(1, 2))                        # (B,L,dim)
        return y.transpose(1, 2).reshape(B_, dim, D, H, W)


def _selective_scan(u, delta, A, B, C):
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return selective_scan_fn(u.contiguous(), delta.contiguous(),
                                 A.contiguous(), B.contiguous(), C.contiguous(),
                                 None, None, None, False)
    except Exception:
        return _ref_scan(u, delta, A, B, C)


class SegMambaStandin(nn.Module):
    def __init__(self, in_ch=1, dims=(32, 64, 128, 256), d_state=16):
        super().__init__()
        def cbr(ci, co, s):
            return nn.Sequential(nn.Conv3d(ci, co, 3, stride=s, padding=1),
                                 nn.InstanceNorm3d(co), nn.SiLU())
        self.stem = cbr(in_ch, dims[0], 1)
        self.stages = nn.ModuleList()
        self.ssms = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.stages.append(cbr(dims[i], dims[i + 1], 2))
            self.ssms.append(_SSM3D(dims[i + 1], d_state))

    def forward(self, x):
        x = self.stem(x)
        for stage, ssm in zip(self.stages, self.ssms):
            x = stage(x); x = x + ssm(x)
        return x
```

Append to `registry.py`:

```python
from encoder_bench.encoders_standin import SegMambaStandin  # add near imports

register(EncoderSpec(
    name="segmamba", family="mamba", call="single", size_multiple=8,
    factory=lambda: SegMambaStandin(dims=(32, 64, 128, 256)),
    opt_profile={"autocast": "bf16", "channels_last": True},   # no compile: ssm graph-breaks
))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_standin.py -v`
Expected: PASS (all standin tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/encoder_bench/encoders_standin.py experiments/3d/encoder_bench/registry.py tests/encoder_bench/test_standin.py
git commit -m "feat(encoder-bench): SegMamba SSM stand-in (mamba_ssm optional)"
```

---

### Task 7: Register pretrained zoo encoders (weights-off + gated)

**Files:**
- Modify: `experiments/3d/encoder_bench/registry.py`
- Test: `tests/encoder_bench/test_registry.py` (add cases)

**Interfaces:**
- Produces registry entries: `stunet`, `vocomni_swin`, `vocomni_nnunet` (buildable weights-off), plus gated `nninteractive`, `threedino` (`requires_ckpt=True`). Factories mirror the real construction signatures (verified from source):
  - `STUNetEncoder(in_channels=1, variant="base", pretrained=None, freeze_encoder=False)` — call `img_mask`, family `cnn`.
  - `VoComniEncoder(ckpt_path=None, feature_size=48)` — call `single`, family `transformer`, `size_multiple=32`. Needs MONAI.
  - `VoComniNNUNetEncoder(ckpt_path=None, freeze_encoder=False, compile_model=False)` — call `img_mask`, family `cnn`, `size_multiple=32`.
  - `NNInteractiveEncoder(ckpt_dir=<path>, num_stages=6, device="cpu")` — call `img_mask`, family `cnn`, `requires_ckpt=True`. Factory reads `NNINT_CKPT` env var; raises if unset (runner catches → `unavailable` row).
  - `ThreeDINOEncoder(ckpt_path=<path>)` — call `single`, family `transformer`, `size_multiple=16`, `requires_ckpt=True`. Factory reads `THREEDINO_CKPT` env var.

- [ ] **Step 1: Write the failing test (append to test_registry.py)**

```python
def test_zoo_encoders_registered():
    names = set(R.list_encoders())
    assert {"stunet", "vocomni_swin", "vocomni_nnunet",
            "nninteractive", "threedino"} <= names
    assert R.REGISTRY["nninteractive"].requires_ckpt
    assert R.REGISTRY["threedino"].requires_ckpt
    assert R.REGISTRY["vocomni_swin"].size_multiple == 32


def test_weightsfree_zoo_factory_builds():
    # stunet builds with no checkpoint (pretrained=None); tiny forward on CPU
    spec = R.REGISTRY["stunet"]
    mod = spec.factory().eval()
    import torch
    out = mod(*R.make_inputs(spec, torch.zeros(1, 1, 32, 32, 32)))
    assert out is not None
```

(If MONAI is absent in the test env, `vocomni_swin`/`vocomni_nnunet` factories raise at *call* time, not import time — registration must not import MONAI at module load. `test_weightsfree_zoo_factory_builds` uses `stunet`, which has no heavy deps.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_registry.py -v`
Expected: FAIL — new encoders not in `list_encoders()`.

- [ ] **Step 3: Write minimal implementation (append to registry.py)**

```python
import os

def _stunet():
    from src.models.encoders.stunet import STUNetEncoder
    return STUNetEncoder(in_channels=1, variant="base", pretrained=None)

def _vocomni_swin():
    from src.models.encoders.vocomni import VoComniEncoder
    return VoComniEncoder(ckpt_path=None, feature_size=48)

def _vocomni_nnunet():
    from src.models.encoders.vocomni_nnunet import VoComniNNUNetEncoder
    return VoComniNNUNetEncoder(ckpt_path=None, freeze_encoder=False, compile_model=False)

def _nninteractive():
    from src.models.encoders.nninteractive import NNInteractiveEncoder
    ckpt = os.environ.get("NNINT_CKPT")
    if not ckpt:
        raise FileNotFoundError("NNINT_CKPT not set")
    return NNInteractiveEncoder(ckpt_dir=ckpt, num_stages=6, device="cpu")

def _threedino():
    from src.models.encoders.threedino import ThreeDINOEncoder
    ckpt = os.environ.get("THREEDINO_CKPT")
    if not ckpt:
        raise FileNotFoundError("THREEDINO_CKPT not set")
    return ThreeDINOEncoder(ckpt_path=ckpt)

_OPT = {"autocast": "bf16", "channels_last": True, "compile": "reduce-overhead"}
register(EncoderSpec("stunet", "cnn", _stunet, call="img_mask", opt_profile=_OPT))
register(EncoderSpec("vocomni_swin", "transformer", _vocomni_swin, call="single",
                     size_multiple=32, opt_profile={"autocast": "bf16"}))
register(EncoderSpec("vocomni_nnunet", "cnn", _vocomni_nnunet, call="img_mask",
                     size_multiple=32, opt_profile=_OPT))
register(EncoderSpec("nninteractive", "cnn", _nninteractive, call="img_mask",
                     requires_ckpt=True, opt_profile=_OPT))
register(EncoderSpec("threedino", "transformer", _threedino, call="single",
                     size_multiple=16, requires_ckpt=True, opt_profile={"autocast": "bf16"}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/test_registry.py -v`
Expected: PASS. (If `stunet` build pulls a missing dep in the test env, mark `test_weightsfree_zoo_factory_builds` with `pytest.importorskip` for that dep — but STU-Net is torch-only.)

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/encoder_bench/registry.py tests/encoder_bench/test_registry.py
git commit -m "feat(encoder-bench): register pretrained zoo encoders (weights-off + gated)"
```

---

### Task 8: End-to-end smoke run on thor + log

**Files:**
- Modify: `docs/logs.md`

**Interfaces:**
- Consumes: the full harness. No new code — validates the real sweep on GPU and records it.

- [ ] **Step 1: Run a small real sweep on thor GPU**

Run:
```bash
.venv_thor/bin/python experiments/3d/encoder_bench/run.py \
    --encoders conv_encoder3d resenc primus segmamba \
    --input_sizes 32 64 --out_dir results/encoder_bench_smoke
```
Expected: prints per-point `status: ok`, writes `results/encoder_bench_smoke/encoder_bench.csv` + `scaling_*.png`. Confirm `fwd_bwd_ms` / `train_vram_mb` are finite numbers (GPU) and `throughput_vol_s` > 0.

- [ ] **Step 2: Verify the CSV + plots**

Run:
```bash
.venv_thor/bin/python -c "import csv; r=list(csv.DictReader(open('results/encoder_bench_smoke/encoder_bench.csv'))); print(len(r),'rows'); print([ (x['encoder'],x['input_size'],x['status'],x['train_vram_mb']) for x in r])"
ls results/encoder_bench_smoke/*.png
```
Expected: 8 rows, statuses `ok`, PNGs present, VRAM populated.

- [ ] **Step 3: Run the full unit suite**

Run: `.venv_thor/bin/python -m pytest tests/encoder_bench/ -v`
Expected: all PASS.

- [ ] **Step 4: Log to docs/logs.md**

Append a dated one-liner:
```markdown
- 2026-07-26: Added `experiments/3d/encoder_bench/` — compute/latency scaling benchmark for 3D encoders (7 zoo + Primus/SegMamba stand-ins). Sweeps encoder × input_size, best-optimized config, writes CSV + scaling-curve PNGs. Run: `.venv_thor/bin/python experiments/3d/encoder_bench/run.py`.
```

- [ ] **Step 5: Commit**

```bash
git add docs/logs.md
git commit -m "docs(encoder-bench): log harness + smoke-run results"
```

---

## Self-Review notes

- **Spec coverage:** goal/scope (Task 1–4), isolation single-volume (make_inputs, Task 1), best-optimized-only (Task 3), metrics params/gflops/fwd_bwd/vram/throughput (Task 2), input sizes 32/64/96/128 + divisibility skip (Task 2/4), 9-encoder roster (Tasks 1/5/6/7), compute-only stand-ins + optional mamba kernel (Tasks 5/6), weights-off default + gated ckpt encoders (Task 7), CSV + PNG + wandb output (Task 4), thor env + CXX gotcha (Global Constraints, Task 3). All covered.
- **Divisibility of 96:** 96 % 32 == 0 and 96 % 16 == 0, so Swin/ViT encoders keep all four sizes. Only hypothetical coarser encoders skip.
- **Type consistency:** `EncoderSpec` fields, `make_inputs`, `profile_point` return keys, and `_FIELDS` in `run.py` all match across tasks.

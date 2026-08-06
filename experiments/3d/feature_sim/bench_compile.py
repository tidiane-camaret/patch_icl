"""Measure the time/compute gain of torch.compile on the feature_sim forward path.

feature_sim's per-task work that torch.compile can accelerate is the frozen Primus `eva`
stack + the read-out `transformer` — the same two modules train.py compiles. This times,
on the real eval loader + a real checkpoint, the dominant feature_sim forwards per task:

  predict                     (encode + transformer + decode)      -> real_dice
  transformer_pair_per_layer  (one hooked forward, all blocks)     -> transformer_layers tier

Encoder cache is reset each iteration so the (K+1) encodes actually run (the honest
per-task cost; within a task feature_sim hits the content cache). Reports eager vs compiled
in both fp32 (feature_sim's current regime) and bf16 autocast (train/eval regime), plus the
one-time compile+warmup wall cost and the break-even task count.

    python experiments/3d/feature_sim/bench_compile.py \
        experiment=36_colipri_spacing_aware_128 eval.checkpoint=/.../best.pt +bench.n_tasks=6
"""
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from common import DEVICE, make_eval_loader, _source_root          # noqa: E402
from data.totalseg_classes import resolve_classes                  # noqa: E402
from feature_sim.adapters import PatchSet3DEncoderAdapter          # noqa: E402
from feature_sim.run import _load_patchset                         # noqa: E402


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _reset_enc_cache(net):
    enc = getattr(net, "encoder", None)
    if enc is not None and hasattr(enc, "reset_cache"):
        enc.reset_cache()


@torch.no_grad()
def _task_work(net, adapter, task):
    """The compile-relevant feature_sim forwards for one task (cache reset first)."""
    _reset_enc_cache(net)
    image, cin, cout, sp = task
    net.predict(image, cin, cout, **sp)
    list(adapter.transformer_pair_per_layer(image, cin, cout, **sp))


def _time_regime(net, adapter, tasks, autocast, n_warmup, n_timed, label):
    """Median ms/task over n_timed reps; returns (median_ms, first_iter_s). first_iter_s
    captures compile+warmup on the first call after (re)compilation."""
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast and DEVICE.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    # First iteration (may trigger compilation) — timed separately.
    _sync(); t0 = time.perf_counter()
    with ctx:
        _task_work(net, adapter, tasks[0])
    _sync(); first_s = time.perf_counter() - t0
    with ctx:
        for _ in range(max(0, n_warmup - 1)):
            _task_work(net, adapter, tasks[0])
    _sync()
    times = []
    with ctx:
        for i in range(n_timed):
            task = tasks[i % len(tasks)]
            _sync(); t0 = time.perf_counter()
            _task_work(net, adapter, task)
            _sync(); times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    med = times[(len(times) - 1) // 2]
    print(f"  {label:22s} first={first_s:6.1f}s  median={med:7.1f} ms/task")
    return med, first_s


@hydra.main(config_path="../../../configs/experiment/3d",
            config_name="feature_sim", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    _, root, is_mri = _source_root(cfg)
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    loader = make_eval_loader(cfg, val_classes, split=cfg.eval.split)
    net = _load_patchset(cfg)
    adapter = PatchSet3DEncoderAdapter(net)
    spacing_aware = getattr(net, "spacing_aware", False)

    n_tasks = int(cfg.get("bench", {}).get("n_tasks", 6))
    n_warmup = int(cfg.get("bench", {}).get("warmup", 3))
    n_timed = int(cfg.get("bench", {}).get("timed", 10))

    # Materialise a handful of real tasks (B=1 tensors) up front.
    tasks = []
    for batch in loader:
        sp = ({"spacing": float(batch["spacing"][0, 0])}
              if spacing_aware and "spacing" in batch else {})
        for b in range(batch["image"].shape[0]):
            tasks.append((batch["image"][b:b + 1].to(DEVICE),
                          batch["context_in"][b:b + 1].to(DEVICE),
                          batch["context_out"][b:b + 1].to(DEVICE), sp))
            if len(tasks) >= n_tasks:
                break
        if len(tasks) >= n_tasks:
            break
    is_primus = getattr(adapter, "_is_primus", False)
    L = len(net.transformer.blocks)
    print(f"model: encoder={'primus' if is_primus else 'conv'} R={adapter.R} "
          f"L={L} blocks spacing_aware={spacing_aware} | {len(tasks)} tasks | "
          f"warmup={n_warmup} timed={n_timed}\n")

    print("EAGER:")
    e32, _ = _time_regime(net, adapter, tasks, False, n_warmup, n_timed, "fp32")
    e16, _ = _time_regime(net, adapter, tasks, True, n_warmup, n_timed, "bf16 autocast")

    # Compile the two heavy modules exactly as train.py does.
    net.transformer = torch.compile(net.transformer, dynamic=True)
    enc = getattr(net, "encoder", None)
    if is_primus and enc is not None and hasattr(enc, "primus"):
        enc.primus.eva = torch.compile(enc.primus.eva, dynamic=True)
        print("\nCOMPILED (transformer + frozen eva, dynamic=True):")
    else:
        print("\nCOMPILED (transformer only; conv encoder eager):")
    c32, w32 = _time_regime(net, adapter, tasks, False, n_warmup, n_timed, "fp32")
    c16, w16 = _time_regime(net, adapter, tasks, True, n_warmup, n_timed, "bf16 autocast")

    def _summ(tag, eager, comp, warm):
        gain = eager - comp                       # ms/task saved
        be = (warm / (gain / 1e3)) if gain > 0 else float("inf")   # warm is s, gain is ms
        print(f"  {tag:14s} eager {eager:7.1f} -> compiled {comp:7.1f} ms/task "
              f"| {eager / comp:5.2f}x | save {gain:6.1f} ms/task "
              f"| compile+warmup {warm:5.1f}s -> break-even {be:6.0f} tasks")

    print("\nSUMMARY (median ms/task, lower is better):")
    _summ("fp32", e32, c32, w32)
    _summ("bf16", e16, c16, w16)
    print(f"\n  bf16-vs-fp32 (eager): {e32 / e16:.2f}x   "
          f"best (compiled bf16 vs eager fp32): {e32 / c16:.2f}x")


if __name__ == "__main__":
    main()

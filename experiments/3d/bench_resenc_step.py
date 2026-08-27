"""Compute-only train-step microbenchmark for the PatchSet3D + from-scratch ResEnc recipe
(experiment 57). Random on-device inputs — exp57 is ~98% compute-bound (bench_train_step:
data 15 ms / compute 964 ms), so the loader is irrelevant to what we're tuning here.

Mirrors experiments/3d/train.py main()'s compile wiring, then times fwd / bwd / opt with
CUDA events split into encoder-fwd, transformer-fwd, decode-fwd and the lumped backward.

    .venv_blackwell/bin/python experiments/3d/bench_resenc_step.py --steps 30 -- \
        experiment=57_organs_encoder_from_scratch encoder=e3_resenc \
        arch.resenc_n_stages=4 'arch.nnunet_ts_stages=[2,3]'

Flags:
  --no-compile        skip every torch.compile (eager baseline)
  --channels-last     run the ResEnc encoder in channels_last_3d (monkeypatched _stage_feats)
  --batch N           override train.batch_size
  --steps / --warmup  timed / discarded iters
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT / "experiments" / "2d"))

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from common import DEVICE
import train as T
from grid_metrics import target_like


def _cfg(overrides):
    GlobalHydra.instance().clear()
    ov = ["wandb.project=null"] + list(overrides)
    with initialize(config_path="../../configs/experiment/3d", version_base="1.3"):
        return compose(config_name="train", overrides=ov)


class Ev:
    """Named CUDA-event stopwatch; .lap(tag) between start/stop, read cumulative ms."""
    def __init__(self):
        self.acc = {}
        self.pool = []

    def _pair(self):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.pool.append((e0, e1))
        return e0, e1

    def region(self, tag):
        e0, e1 = self._pair()
        return _Region(self, tag, e0, e1)

    def reduce(self):
        torch.cuda.synchronize()
        out = {}
        for tag, e0, e1 in self._laps:
            out[tag] = out.get(tag, 0.0) + e0.elapsed_time(e1)
        return out


class _Region:
    def __init__(self, ev, tag, e0, e1):
        self.ev, self.tag, self.e0, self.e1 = ev, tag, e0, e1

    def __enter__(self):
        self.e0.record(); return self

    def __exit__(self, *a):
        self.e1.record()
        self.ev.acc.setdefault(self.tag, []).append((self.e0, self.e1))


def _patch_channels_last(net):
    """Run the ResEnc residual stack in channels_last_3d: convert its params once and feed
    NDHWC activations. cuDNN then picks the tensor-core NDHWC conv kernels."""
    enc = net.encoder                       # ResEncTSEncoder
    inner = enc.encoder                     # ResidualEncoderUNet.encoder (or compiled wrapper)
    tgt = getattr(inner, "_orig_mod", inner)
    tgt.to(memory_format=torch.channels_last_3d)
    _orig = enc._stage_feats.__func__ if hasattr(enc._stage_feats, "__func__") else enc._stage_feats

    def _cl_stage_feats(x):
        v = enc._norm(x).contiguous(memory_format=torch.channels_last_3d)
        with enc._autocast_ctx():
            return enc.encoder(v)
    enc._stage_feats = _cl_stage_feats
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--torch-profile", action="store_true",
                    help="run torch.profiler for a few steps, print top CUDA ops + module table")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = _cfg(args.overrides)
    if args.batch:
        cfg.train.batch_size = args.batch
    if args.no_compile:
        cfg.arch.compile = False
        cfg.arch.compile_encoder = False
        cfg.arch.compile_decoder = False

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    import random as _r
    _r.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)  # same random init across configs
    model, name = T.build_model(cfg)
    assert name == "patchset3d", name
    net = model
    net.to(DEVICE).train()

    # ---- replicate train.py main() compile wiring ----
    if cfg.arch.get("compile", False) and hasattr(net, "transformer"):
        net.transformer = torch.compile(net.transformer, dynamic=True)
        msg = "compiled: transformer"
        msg += T._compile_encoder(net, cfg)
        if cfg.arch.get("compile_decoder", False):
            net._decode = torch.compile(net._decode, dynamic=True)
            msg += " + decode"
        print(msg)
    else:
        print("eager (no compile)")

    if args.channels_last:
        _patch_channels_last(net)
        print("channels_last_3d on ResEnc encoder")

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    loss_fn = T.build_loss(cfg)

    B = int(cfg.train.batch_size)
    K = int(cfg.data.context_size)
    S = list(cfg.data.image_size)
    n_tr = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    n_all = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"B={B} K={K} size={S} | enc={cfg.arch.encoder} n_stages={cfg.arch.resenc_n_stages} "
          f"stages={list(cfg.arch.nnunet_ts_stages)} | trainable={n_tr:.1f}M / {n_all:.1f}M")

    g = torch.Generator(device=DEVICE).manual_seed(0)
    img = torch.randn(B, *S, generator=g, device=DEVICE)
    cin = torch.randn(B, K, *S, generator=g, device=DEVICE)
    cout = (torch.rand(B, K, *S, generator=g, device=DEVICE) > 0.7).float()
    lbl = (torch.rand(B, *S, generator=g, device=DEVICE) > 0.7).float()

    # hooks for the fwd sub-phases (encoder / transformer / decode)
    ev = Ev()
    spans = {}

    def pre(tag):
        def _h(m, i):
            e0 = torch.cuda.Event(enable_timing=True); e0.record()
            spans[tag] = e0
        return _h

    def post(tag):
        def _h(m, i, o):
            e1 = torch.cuda.Event(enable_timing=True); e1.record()
            ev.acc.setdefault(tag, []).append((spans[tag], e1))
        return _h

    hs = []
    hs.append(net.encoder.register_forward_pre_hook(pre("enc_fwd")))
    hs.append(net.encoder.register_forward_hook(post("enc_fwd")))
    hs.append(net.transformer.register_forward_pre_hook(pre("attn_fwd")))
    hs.append(net.transformer.register_forward_hook(post("attn_fwd")))

    # NB: per-module backward timing via full_backward hooks is unreliable once the module
    # is torch.compile-wrapped (compiled-autograd reorders the hook firing), so we report
    # only the lumped backward. The forward-hook split (enc_fwd / attn_fwd) survives compile.

    def step():
        opt.zero_grad(set_to_none=True)
        with T._autocast():
            out = net(img, context_in=cin, context_out=cout, mode="train")
            logits = out["final_logit"].float()
            target = target_like(lbl.unsqueeze(1), logits)
            loss = loss_fn(logits, target)
        f_done = torch.cuda.Event(enable_timing=True); f_done.record()
        loss.backward()
        b_done = torch.cuda.Event(enable_timing=True); b_done.record()
        opt.step()
        return loss, f_done, b_done

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # deterministic-input logit stats — eyeball precision drift across configs (same seed -> same inputs)
    with torch.no_grad(), T._autocast():
        lg = net(img, context_in=cin, context_out=cout, mode="train")["final_logit"].float()
    print(f"logit stats: mean={lg.mean():+.5f} std={lg.std():.5f} "
          f"min={lg.min():+.4f} max={lg.max():+.4f} absmean={lg.abs().mean():.5f} "
          f"fg@0.5={(lg > 0.5).float().mean():.4f}")

    if args.torch_profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, with_stack=False) as prof:
            for _ in range(6):
                step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
        for h in hs:
            h.remove()
        return

    ev.acc.clear()
    fwd_ms = bwd_ms = opt_ms = 0.0
    t_wall = time.perf_counter()
    marks = []
    for _ in range(args.steps):
        s0 = torch.cuda.Event(enable_timing=True); s0.record()
        loss, f_done, b_done = step()
        s1 = torch.cuda.Event(enable_timing=True); s1.record()
        marks.append((s0, f_done, b_done, s1))
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t_wall) * 1e3 / args.steps
    for s0, f_done, b_done, s1 in marks:
        fwd_ms += s0.elapsed_time(f_done)
        bwd_ms += f_done.elapsed_time(b_done)
        opt_ms += b_done.elapsed_time(s1)
    n = args.steps
    fwd_ms /= n; bwd_ms /= n; opt_ms /= n
    sub = {k: sum(a.elapsed_time(b) for a, b in v) / n for k, v in ev.acc.items()}
    peak = torch.cuda.max_memory_allocated() / 1e9

    for h in hs:
        h.remove()
    print(f"\n{'step (wall)':<16}{wall:8.1f} ms   {1e3/wall:6.2f} it/s")
    print(f"{'  fwd':<16}{fwd_ms:8.1f} ms")
    print(f"{'    enc_fwd':<16}{sub.get('enc_fwd', 0):8.1f} ms")
    print(f"{'    attn_fwd':<16}{sub.get('attn_fwd', 0):8.1f} ms")
    print(f"{'    other_fwd':<16}{fwd_ms - sub.get('enc_fwd',0) - sub.get('attn_fwd',0):8.1f} ms  (decode + tokenize + occ + loss)")
    print(f"{'  bwd (lumped)':<16}{bwd_ms:8.1f} ms")
    print(f"{'  opt':<16}{opt_ms:8.1f} ms")
    print(f"peak mem: {peak:.2f} GB")


if __name__ == "__main__":
    main()

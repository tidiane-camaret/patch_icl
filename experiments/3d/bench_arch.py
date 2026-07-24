"""
Ad-hoc architecture benchmark: medverse vs patchset3d.

Isolates *model* compute (no dataloader) at the real runtime shapes used by
experiment=1_medverse_benchmark: B=1, K=1, 128^3. Reproduces the exact training
forward/backward call paths from experiments/3d/train.py::train_epoch and reports
param counts (by submodule), forward/backward wall time, peak activation memory,
and a torch-profiler operator breakdown.

    .venv_nero/bin/python experiments/3d/bench_arch.py
"""
import sys, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT / "experiments" / "2d"))

import torch
import torch.nn.functional as F

DEV = torch.device("cuda")


# ---------------------------------------------------------------------------
# Medverse optimization prototypes. Gradient checkpointing + compile go through the
# adapter methods (MedverseModel.enable_gradient_checkpointing / compile_net) so this
# benchmark exercises exactly what train.py ships. channels_last stays bench-only
# (measured a loss; kept here only to reproduce that result).
# ---------------------------------------------------------------------------

def apply_channels_last(net):
    """Store conv weights in channels_last_3d so cudnn uses NHWC kernels natively.
    Bench-only: measured *slower* here (the 6D context reshapes fight the format)."""
    net.to(memory_format=torch.channels_last_3d)
    return net


def human(n):
    for u in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000
    return f"{n:.1f}T"


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def submodule_params(net):
    return {name: count_params(m) for name, m in net.named_children()}


def make_inputs(B, K, S):
    img = torch.randn(B, 1, S, S, S, device=DEV)
    ctx_in = torch.randn(B, K, 1, S, S, S, device=DEV)
    ctx_out = (torch.rand(B, K, S, S, S, device=DEV) > 0.7).float()
    lbl = (torch.rand(B, S, S, S, device=DEV) > 0.7).float()
    return img, ctx_in, ctx_out, lbl


def build_medverse():
    from src.benchmark_models.medverse import MedverseModel
    m = MedverseModel(device=DEV, sw_roi_size=(128, 128, 128))
    return m, m.model.net


def build_patchset3d():
    from src.models.patchset3d import PatchSet3D
    arch = dict(resolution=16, enc_dims=[32, 32, 32, 32], e=256, h=512, l=6, a=4,
                thinking_rows=8, residual_decay=0.95, fourier_bands=8,
                mask_patch_size=8, mask_patch_decode_size=8, context_id_embed=True,
                max_context=16, full_attn=True, query_self_attn=True,
                image_size=[128, 128, 128])
    m = PatchSet3D(**arch).to(DEV)
    return m, m


def fwd_bwd(name, model, net, img, ctx_in, ctx_out, lbl):
    from grid_metrics import target_like
    bce = F.binary_cross_entropy_with_logits

    def step():
        if name == "medverse":
            logits = model.train_forward(img, ctx_in, ctx_out)   # (B,1,S,S,S)
            target = lbl.unsqueeze(1)
        else:
            out = model(img, context_in=ctx_in, context_out=ctx_out, mode="train")
            logits = out["final_logit"].float()
            target = target_like(lbl.unsqueeze(1), logits)
        p = torch.sigmoid(logits.float())
        inter = (p.flatten(1) * target.flatten(1)).sum(1)
        den = p.flatten(1).sum(1) + target.flatten(1).sum(1)
        dice = (1 - (2 * inter + 1e-6) / (den + 1e-6)).mean()
        loss = bce(logits.float(), target) + dice
        return loss, logits

    return step


def bench(name, reps=8):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, net = build_medverse() if name == "medverse" else build_patchset3d()
    net.train()
    total = count_params(net)
    print(f"Total params: {human(total)}")
    for sub, p in sorted(submodule_params(net).items(), key=lambda x: -x[1]):
        if p:
            print(f"   {sub:<24} {human(p):>8}  ({100*p/total:4.1f}%)")

    img, ctx_in, ctx_out, lbl = make_inputs(1, 1, 128)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    step = fwd_bwd(name, model, net, img, ctx_in, ctx_out, lbl)

    def one(train=True):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, logits = step()
        if train:
            loss.backward(); opt.step()
        return loss.item(), tuple(logits.shape)

    # warmup
    _, out_shape = one()
    torch.cuda.synchronize()

    # peak memory (fwd+bwd)
    torch.cuda.reset_peak_memory_stats()
    one()
    torch.cuda.synchronize()
    peak_fb = torch.cuda.max_memory_allocated() / 1e9

    # forward-only peak
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        step()
    torch.cuda.synchronize()
    peak_f = torch.cuda.max_memory_allocated() / 1e9

    # timing fwd+bwd
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        one()
    torch.cuda.synchronize()
    fb_ms = 1000 * (time.perf_counter() - t0) / reps

    # timing fwd-only
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            step()
    torch.cuda.synchronize()
    f_ms = 1000 * (time.perf_counter() - t0) / reps

    print(f"\nOutput logits shape: {out_shape}")
    print(f"Forward       : {f_ms:7.1f} ms   peak {peak_f:5.2f} GB")
    print(f"Forward+bwd   : {fb_ms:7.1f} ms   peak {peak_fb:5.2f} GB")
    return dict(name=name, params=total, f_ms=f_ms, fb_ms=fb_ms,
                peak_f=peak_f, peak_fb=peak_fb, out_shape=out_shape)


def bench_medverse_variant(label, opts, B=1, reps=8, compile_net=False):
    """Build medverse, apply the given optimization opts, and measure fwd+bwd time/peak.
    opts: subset of {"ckpt", "channels_last"}."""
    from grid_metrics import target_like  # noqa: F401 (kept for parity)
    bce = F.binary_cross_entropy_with_logits
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = True
    model, net = build_medverse()
    net.train()
    n_ck = model.enable_gradient_checkpointing() if "ckpt" in opts else 0
    chlast = "channels_last" in opts
    if chlast:
        apply_channels_last(net)
    if compile_net:
        model.compile_net()

    img, ctx_in, ctx_out, lbl = make_inputs(B, 1, 128)
    if chlast:
        img = img.to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)

    def step():
        logits = model.train_forward(img, ctx_in, ctx_out)
        target = lbl.unsqueeze(1)
        p = torch.sigmoid(logits.float())
        inter = (p.flatten(1) * target.flatten(1)).sum(1)
        den = p.flatten(1).sum(1) + target.flatten(1).sum(1)
        dice = (1 - (2 * inter + 1e-6) / (den + 1e-6)).mean()
        return bce(logits.float(), target) + dice

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = step()
        loss.backward(); opt.step()
        return loss.item()

    for _ in range(3):   # warmup (compile + cudnn benchmark)
        one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); one(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    resv = torch.cuda.max_memory_reserved() / 1e9
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        one()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / reps
    print(f"{label:<32} B={B}  {ms:7.1f} ms   alloc {peak:5.2f}G  reserved {resv:5.2f}G"
          f"  (ckpt {n_ck} blocks)")
    del model, net; torch.cuda.empty_cache()
    return dict(label=label, ms=ms, peak=peak, resv=resv)


def bench_optims(B=1):
    print(f"\n{'='*74}\nMEDVERSE OPTIMIZATION VARIANTS (B={B}, K=1, 128^3)\n{'='*74}")
    bench_medverse_variant("baseline", set(), B=B)
    bench_medverse_variant("channels_last", {"channels_last"}, B=B)
    bench_medverse_variant("gradient_checkpointing", {"ckpt"}, B=B)
    bench_medverse_variant("ckpt+channels_last", {"ckpt", "channels_last"}, B=B)
    bench_medverse_variant("compile", set(), B=B, compile_net=True)
    bench_medverse_variant("compile+channels_last", {"channels_last"}, B=B, compile_net=True)


def profile_ops(name, rows=15):
    from torch.profiler import profile, ProfilerActivity
    model, net = build_medverse() if name == "medverse" else build_patchset3d()
    net.train()
    img, ctx_in, ctx_out, lbl = make_inputs(1, 1, 128)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    step = fwd_bwd(name, model, net, img, ctx_in, ctx_out, lbl)

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = step()
        loss.backward(); opt.step()

    one(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            one()
        torch.cuda.synchronize()
    print(f"\n----- {name}: top ops by CUDA time -----")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--optims", action="store_true", help="benchmark medverse opt variants")
    ap.add_argument("-B", type=int, default=1, help="batch size for --optims")
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    if args.optims:
        bench_optims(B=args.B)
        sys.exit(0)
    res = [bench("patchset3d"), bench("medverse")]
    print(f"\n{'='*70}\nSUMMARY (B=1, K=1, 128^3)\n{'='*70}")
    print(f"{'model':<12}{'params':>9}{'fwd ms':>10}{'fwd+bwd ms':>12}{'peak fwd':>10}{'peak f+b':>10}")
    for r in res:
        print(f"{r['name']:<12}{human(r['params']):>9}{r['f_ms']:>10.1f}"
              f"{r['fb_ms']:>12.1f}{r['peak_f']:>9.2f}G{r['peak_fb']:>9.2f}G")
    if args.profile:
        for n in ("patchset3d", "medverse"):
            profile_ops(n)

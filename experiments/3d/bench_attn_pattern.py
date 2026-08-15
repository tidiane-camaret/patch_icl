"""
Attention-pattern benchmark for PatchSet3D: register-routed vs full_attn, over context size.

register_routed makes the thinking rows the only cross-image bus (each image self-attends
within its own N-cell block + reads/writes registers), which needs an explicit r×r bool
mask and so drops off the flash-attention path onto mem-efficient/math SDPA. full_attn is
the dense unmasked flash baseline. This measures what that mask costs in FLOPs, peak RAM
(the r×r mask is itself a real allocation) and fwd+bwd latency as K grows.

Realistic training config: resolution=16, mask_patch=8, B=1, bf16 autocast, fwd+bwd —
exactly the step experiments/3d/train.py::train_epoch runs the mask in. Each (K, pattern)
cell runs isolated; an OOM at high K (r ≈ (K+1)·4096, mask alone ~GB) is recorded, not fatal.

    .venv_nero/bin/python experiments/3d/bench_attn_pattern.py
    .venv_nero/bin/python experiments/3d/bench_attn_pattern.py --ks 1 2 4 8
"""
import sys, time, argparse, gc, os, shutil as _shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Broken-usr-merge nodes (thor, odin): bare gcc/g++ resolve to /bin/* and fail Triton's
# compile (flex_attention needs it). Point CC/CXX at /usr/bin before importing torch — same
# shim as experiments/3d/train.py. No-op on usr-merged nodes.
if not os.path.islink("/bin"):
    for _var, _tool in (("CC", "gcc"), ("CXX", "g++")):
        _abs = f"/usr/bin/{_tool}"
        _found = _shutil.which(_tool)
        if _var not in os.environ and _found and _found.startswith("/bin/") and os.path.exists(_abs):
            os.environ[_var] = _abs

import torch
import torch.nn.functional as F

DEV = torch.device("cuda")
S = 128            # native volume side (image_size)
RES = 16           # feature grid R (N = R³ tokens per image)
MASK_PATCH = 8     # p³ occupancy tile / decode tile -> grid_size = R·8 = 128
# transformer shape (must match build() below) for the analytical attention FLOPs
E, A, L, N_THINK = 256, 4, 6, 8


def attn_gflops(pattern, K):
    """Analytical sample-axis attention FLOPs across all L layers (fwd), the term that
    actually differs between patterns. FlopCounterMode can't introspect the fused flex
    kernel (nor the masked SDPA's real sparsity), so we count QK+AV = 4·seq_q·seq_k·d
    directly. b·c = 2 (b=1, img/mask cols); a·d folds to e.

      full_attn:       every row attends to all r rows          -> 4·L·(b·c)·e·r²
      register_routed: each of T image blocks attends within     -> 4·L·(b·c)·e·(T·N·(N+n_t)
                       itself (N over N+n_t); n_t registers see r    + n_t·r)
    The ratio is ~T (=K+1): register-routing is T× cheaper than dense full attention."""
    N, T, n_t, bc = RES ** 3, K + 1, N_THINK, 2
    r = n_t + T * N
    if pattern == "full_attn":
        f = 4 * L * bc * E * r * r
    else:
        f = 4 * L * bc * E * (T * N * (N + n_t) + n_t * r)
    return f / 1e9


def build(pattern):
    """PatchSet3D at the shipping R=16/mask_patch=8 config with the given attn pattern."""
    from src.models.patchset3d import PatchSet3D
    arch = dict(resolution=RES, enc_dims=[32, 32, 32, 32], e=256, h=512, l=6, a=4,
                thinking_rows=8, residual_decay=0.95, fourier_bands=8,
                mask_patch_size=MASK_PATCH, mask_patch_decode_size=MASK_PATCH,
                context_id_embed=True, max_context=16, image_size=[S, S, S],
                full_attn=(pattern == "full_attn"),
                register_routed=(pattern == "register_routed"))
    return PatchSet3D(**arch).to(DEV)


def make_inputs(B, K):
    img = torch.randn(B, 1, S, S, S, device=DEV)
    ctx_in = torch.randn(B, K, 1, S, S, S, device=DEV)
    ctx_out = (torch.rand(B, K, S, S, S, device=DEV) > 0.7).float()
    lbl = (torch.rand(B, S, S, S, device=DEV) > 0.7).float()
    return img, ctx_in, ctx_out, lbl


def loss_of(model, img, ctx_in, ctx_out, lbl):
    from grid_metrics import target_like
    out = model(img, context_in=ctx_in, context_out=ctx_out, mode="train")
    logits = out["final_logit"].float()
    target = target_like(lbl.unsqueeze(1), logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def measure(pattern, K, reps):
    """Return dict(flops, ram_gb, ms) for one (pattern, K); marks OOM cleanly."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build(pattern)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    img, ctx_in, ctx_out, lbl = make_inputs(1, K)

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = loss_of(model, img, ctx_in, ctx_out, lbl)
        loss.backward(); opt.step()

    try:
        gflops = attn_gflops(pattern, K)   # analytical sample-axis attention FLOPs (fwd)
        one()  # warmup (compiles flex on first call; allocator + cudnn)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        one()
        torch.cuda.synchronize()
        ram = torch.cuda.max_memory_allocated() / 1e9

        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            one()
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / reps
        res = dict(flops=gflops, ram_gb=ram, ms=ms)
    except torch.cuda.OutOfMemoryError:
        res = dict(flops=float("nan"), ram_gb=float("nan"), ms=float("nan"), oom=True)
    del model, opt, img, ctx_in, ctx_out, lbl
    gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()
    patterns = ["full_attn", "register_routed"]

    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"config: R={RES} mask_patch={MASK_PATCH} size={S}³  N={RES**3} tokens/image  "
          f"B=1 fwd+bwd bf16\n")
    results = {}
    for pat in patterns:
        for K in args.ks:
            r = 8 + (K + 1) * RES ** 3
            res = measure(pat, K, args.reps)
            results[(pat, K)] = res
            tag = "OOM" if res.get("oom") else (
                f"{res['flops']:9.1f} attnGF   {res['ram_gb']:6.2f} GB   {res['ms']:8.1f} ms")
            print(f"  {pat:<16} K={K:<3} r={r:>6}   {tag}")

    print(f"\n{'='*78}\nSUMMARY (B=1, R={RES}, mask_patch={MASK_PATCH}, {S}³, fwd+bwd)\n{'='*78}")
    print(f"{'pattern':<16}{'K':>4}{'r (rows)':>10}{'attnGFLOP':>11}{'RAM GB':>10}{'ms':>10}")
    for pat in patterns:
        for K in args.ks:
            r = 8 + (K + 1) * RES ** 3
            res = results[(pat, K)]
            if res.get("oom"):
                print(f"{pat:<16}{K:>4}{r:>10}{'OOM':>11}{'OOM':>10}{'OOM':>10}")
            else:
                print(f"{pat:<16}{K:>4}{r:>10}{res['flops']:>11.1f}"
                      f"{res['ram_gb']:>10.2f}{res['ms']:>10.1f}")


if __name__ == "__main__":
    main()

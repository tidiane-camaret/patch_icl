"""Probe the shape limit of fomofo/tap-ct-b-3d on the current GPU and test
memory/latency optimizations.

Key facts (see config + tapct_processor.py in the HF cache):
- The processor always resizes in-plane to 224x224 and pads depth to a multiple
  of 4. patch_size = (4, 8, 8) -> each 4-slice group becomes a 28x28 patch grid.
- So token count depends ONLY on depth D: N = (D/4)*28*28 + 1 cls + 4 registers.
- The blocks use `MemEffAttention`, which needs xformers for O(L) memory. If
  xformers is absent it FALLS BACK to explicit q@k.T softmax -> O(L^2) memory,
  which is what OOMs on large volumes (e.g. raw (179,192,294) -> D_pad=180).

Optimizations tested here:
  1. sdpa   : monkeypatch attention to torch F.scaled_dot_product_attention
              (flash / mem-efficient kernel) -> O(L) memory, no xformers needed.
  2. bf16   : autocast to bfloat16 -> ~half the activation memory + faster.
  3. compile: torch.compile on top of the sdpa path (latency).

Run:  .venv_thor/bin/python experiments/encoders/tap_ct_bench.py
"""
import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel

MODEL_ID = "fomofo/tap-ct-b-3d"
PATCH = (4, 8, 8)
INPLANE = 224  # processor resize target
TOK_PER_LAYER = (INPLANE // PATCH[1]) * (INPLANE // PATCH[2])  # 28*28 = 784


def n_tokens(depth: int) -> int:
    return (depth // PATCH[0]) * TOK_PER_LAYER + 5  # +cls +4 registers


def sdpa_forward(self, x, attn_bias=None):
    """Drop-in replacement for MemEffAttention.forward using PyTorch SDPA.

    SDPA applies the 1/sqrt(head_dim) scale internally, matching self.scale, so
    q is NOT pre-scaled here.
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def load_model(device, use_sdpa: bool):
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval().to(device)
    if use_sdpa:
        # Patch the actual attention class used by the loaded blocks.
        attn_cls = type(_first_attn(model))
        attn_cls.forward = sdpa_forward
    return model


def _first_attn(model):
    for m in model.modules():
        if m.__class__.__name__ in ("MemEffAttention", "Attention"):
            return m
    raise RuntimeError("no attention module found")


def make_input(depth, device, dtype):
    # Direct preprocessed-style tensor: (B, C, D, 224, 224), already normalized.
    return torch.randn(1, 1, depth, INPLANE, INPLANE, device=device, dtype=dtype)


def run_once(model, x, amp_dtype):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(x)
        else:
            out = model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    del out
    return dt, peak


def sweep(name, model, device, depths, dtype, amp_dtype):
    print(f"\n=== {name} ===")
    print(f"{'D_pad':>6} {'tokens':>8} {'peak_GB':>9} {'time_s':>8}  status")
    last_ok = None
    for d in depths:
        torch.cuda.empty_cache()
        x = make_input(d, device, dtype)
        try:
            # warmup + timed
            run_once(model, x, amp_dtype)
            dt, peak = run_once(model, x, amp_dtype)
            print(f"{d:>6} {n_tokens(d):>8} {peak:>9.2f} {dt:>8.3f}  ok")
            last_ok = d
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{d:>6} {n_tokens(d):>8} {'--':>9} {'--':>8}  OOM")
                torch.cuda.empty_cache()
                break
            raise
        finally:
            del x
            torch.cuda.empty_cache()
    if last_ok is not None:
        print(f"  -> max depth OK: {last_ok} ({n_tokens(last_ok)} tokens)")
    return last_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[12, 24, 48, 96, 144, 180, 240, 320, 400, 512])
    ap.add_argument("--target", type=int, default=180,
                    help="padded depth of the reported OOM case (179->180)")
    args = ap.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  {props.total_memory/1e9:.1f} GB")
    print(f"Target case: raw (179,192,294) -> D_pad={args.target}, "
          f"tokens={n_tokens(args.target)}")

    depths = args.depths

    # 1. baseline: naive O(L^2) attention, fp32
    m = load_model(device, use_sdpa=False)
    sweep("baseline fp32 (O(L^2) fallback)", m, device, depths, torch.float32, None)
    del m
    torch.cuda.empty_cache()

    # 2. sdpa, fp32
    m = load_model(device, use_sdpa=True)
    sweep("sdpa fp32", m, device, depths, torch.float32, None)

    # 3. sdpa + bf16 autocast (same model, just autocast)
    sweep("sdpa + bf16 autocast", m, device, depths, torch.float32, torch.bfloat16)
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

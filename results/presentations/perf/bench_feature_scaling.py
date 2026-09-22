"""How does per-stage (encode / transformer / decode) GFLOPs and wall-clock scale with
"feature size" -- interpreted as two independent axes:

  (A) resolution R: the token/feature-GRID size (R^3 cells/volume). Encoder always runs on
      the fixed image_size=128^3 input and only *resamples* its output to R^3 at the end
      (src/models/encoders/plainconv_ts.py::_down_to) -- so encoder compute should be ~flat
      across this sweep; only the transformer (attends over ~(K+1)*R^3 raw tokens for v1
      dense / ~(K+1)*(compress_m+1) tokens for v2) and the decode grid should move.
  (B) embedding width e: the transformer's channel width. Encoder feature width
      (features_per_stage) is independent of e (there's a projection layer in between), so
      encoder compute should also be ~flat here; transformer linear/MLP layers scale ~e^2,
      attention matmuls ~e (per fixed token count).

Plus a third axis specific to v2's design:
  (C) compress_m: v2's Stage-A compression target (tokens/volume fed to the R^3-scale
      transformer). At fixed R this directly tests the O(L) vs O(L^2) transition discussed
      in RESULTS.md's Follow-up section, as a continuous curve instead of two points.

Two model variants, both from real experiment configs (see bench_inference_compare.py):
  - v1 dense: 92_multisource_synth's arch (decoder=conv, seq_compress=False, full_attn=True
    -- genuine R^3-token dense self-attention, no compression)
  - v2 compressed: 101_patchset_v2_mask8_wide's arch (compress_m=128 default)

Per point: GFLOPs breakdown (evaluate.py::measure_flops) + wall-clock breakdown (CUDA-event
forward hooks on net.encoder / net.transformer, matching profile_stage_split.py).

    .venv_blackwell/bin/python results/presentations/perf/bench_feature_scaling.py
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

import torch

DEV = torch.device("cuda")
IMAGE_SIZE = (128, 128, 128)
K = 1
REPS = 10
WARMUP = 3


def make_inputs():
    D, H, W = IMAGE_SIZE
    img = torch.randn(1, 1, D, H, W, device=DEV)
    ctx_in = torch.randn(1, K, 1, D, H, W, device=DEV)
    ctx_out = (torch.rand(1, K, D, H, W, device=DEV) > 0.7).float()
    return img, ctx_in, ctx_out


def build_v1_dense(resolution=16, e=768, h=3072):
    """92_multisource_synth's arch verbatim, resolution/e/h parameterized."""
    from src.models.patchset3d import PatchSet3D
    arch = dict(
        resolution=resolution, e=e, h=h, l=4, a=12, thinking_rows=8, residual_decay=0.95,
        fourier_bands=8, transformer_rope=True, rope_theta=100.0,
        token_mask_ratio_support=0.1, token_mask_ratio_query=0.1,
        mask_patch_size=8, mask_patch_decode_size=8, mask_embed="linear",
        mask_slots=1, decode_source="img", context_id_embed=True, max_context=16,
        full_attn=True, query_self_attn=True, register_routed=False, register_flex=False,
        encoder="plainconv_ts", encoder_frozen=False, encoder_input_norm="instance",
        encoder_precision="bf16", encoder_spacing_aware=True, plainconv_ts_n_stages=5,
        plainconv_ts_features_per_stage=[32, 64, 256, 512], nnunet_ts_stages=[2, 3],
        enc_dims=[32, 32, 32, 32], img_embed_mlp=True, feat_norm="self",
        fine_decode=True, fine_stage=[0, 1], fine_proj_dim=96,
        decoder="conv", decoder_dim=64,
        image_size=list(IMAGE_SIZE),
    )
    return PatchSet3D(**arch).to(DEV)


def build_v2_compressed(resolution=16, e=768, h=3072, compress_m=128):
    """101_patchset_v2_mask8_wide's arch verbatim, resolution/e/h/compress_m parameterized."""
    from src.models.patchset3d_v2 import PatchSetV2
    arch = dict(
        resolution=resolution, e=e, h=h, l=4, a=12, thinking_rows=8, residual_decay=0.95,
        fourier_bands=8, mask_patch_size=8, mask_embed="linear", compress_m=compress_m,
        compress_layers=1, context_id_embed=True, max_context=16, cascade_registers=False,
        fine_stage=[0, 1, 2], decoder_dim=64, img_embed_mlp=True, feat_norm="context",
        encoder="plainconv_ts", encoder_frozen=False, encoder_input_norm="instance",
        encoder_precision="bf16", plainconv_ts_n_stages=5,
        plainconv_ts_features_per_stage=[32, 64, 128, 768], nnunet_ts_stages=[3],
        enc_dims=[32, 32, 32, 32],
        image_size=list(IMAGE_SIZE),
    )
    return PatchSetV2(**arch).to(DEV)


class StageTimer:
    def __init__(self, module):
        self.events = []
        self._start = None
        self.h1 = module.register_forward_pre_hook(self._pre)
        self.h2 = module.register_forward_hook(self._post)

    def _pre(self, module, args):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def _post(self, module, args, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append((self._start, e))

    def reset(self):
        self.events = []

    def total_ms(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.events)

    def remove(self):
        self.h1.remove()
        self.h2.remove()


def bench_point(build_fn, **kw):
    torch.cuda.empty_cache()
    model = build_fn(**kw)
    net = getattr(model, "model", model)
    net.eval()
    params = sum(p.numel() for p in net.parameters())

    from evaluate import measure_flops
    flops = measure_flops(model, IMAGE_SIZE, K, DEV)

    enc_timer = StageTimer(net.encoder)
    tfm_timer = StageTimer(net.transformer)

    img, ctx_in, ctx_out = make_inputs()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def call():
        with torch.no_grad(), ctx:
            return model.predict(img, ctx_in, ctx_out)

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    total_list, enc_list, tfm_list = [], [], []
    for _ in range(REPS):
        enc_timer.reset()
        tfm_timer.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        total_list.append(1000 * (time.perf_counter() - t0))
        enc_list.append(enc_timer.total_ms())
        tfm_list.append(tfm_timer.total_ms())

    total_ms = sum(total_list) / REPS
    enc_ms = sum(enc_list) / REPS
    tfm_ms = sum(tfm_list) / REPS
    other_ms = total_ms - enc_ms - tfm_ms
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    enc_timer.remove()
    tfm_timer.remove()
    del model, net
    torch.cuda.empty_cache()

    return dict(params=params, gflops_total=flops["total"], gflops_encoder=flops["encoder"],
               gflops_transformer=flops["transformer"], gflops_other=flops["other"],
               ms_total=total_ms, ms_encoder=enc_ms, ms_transformer=tfm_ms, ms_other=other_ms,
               peak_gb=peak_gb)


def run_sweep(sweep_name, build_fn, variant_name, param_name, values, fixed):
    print(f"\n{'='*80}\n{variant_name} -- sweep {param_name} = {values}  (fixed: {fixed})\n{'='*80}")
    rows = []
    for v in values:
        kw = dict(fixed)
        kw[param_name] = v
        torch.cuda.reset_peak_memory_stats()
        try:
            r = bench_point(build_fn, **kw)
            r[param_name] = v
            rows.append(r)
            print(f"  {param_name}={v:<6} params={r['params']/1e6:6.1f}M  "
                  f"GFLOPs[tot={r['gflops_total']:8.1f} enc={r['gflops_encoder']:7.1f} "
                  f"tfm={r['gflops_transformer']:8.1f} oth={r['gflops_other']:7.1f}]  "
                  f"ms[tot={r['ms_total']:6.2f} enc={r['ms_encoder']:5.2f} "
                  f"tfm={r['ms_transformer']:6.2f} oth={r['ms_other']:5.2f}]  "
                  f"peak={r['peak_gb']:.2f}GB")
        except Exception as exc:
            print(f"  {param_name}={v:<6} FAILED: {exc}")
            traceback.print_exc()
            rows.append({param_name: v, "error": str(exc)})
    return dict(sweep=sweep_name, variant=variant_name, param=param_name, fixed=fixed, rows=rows)


def main():
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"Conditions: B=1, K={K}, image_size={IMAGE_SIZE}, {REPS} reps ({WARMUP} warmup)")

    results = []

    # --- Sweep A: resolution R (token-grid size). v1's fine_decode requires image_size (128)
    # divisible by R -- restrict to divisors of 128 for v1; v2 has no such constraint but use
    # the same set for a direct comparison, plus fill in v2's finer original points too.
    r_values_v1 = [4, 8, 16, 32]
    r_values_v2 = [4, 8, 12, 16, 20, 24, 28, 32]
    results.append(run_sweep("resolution", build_v1_dense, "v1_dense", "resolution", r_values_v1,
                             fixed=dict(e=768, h=3072)))
    results.append(run_sweep("resolution", build_v2_compressed, "v2_compressed", "resolution",
                             r_values_v2, fixed=dict(e=768, h=3072, compress_m=128)))

    # --- Sweep B: embedding width e, resolution fixed at 16. e must be divisible by a=12
    # heads -- restrict to multiples of 12 (roughly doubling) instead of round numbers.
    e_values = [96, 192, 384, 768, 1536, 3072]
    results.append(run_sweep("width", build_v1_dense, "v1_dense", "e", e_values,
                             fixed=dict(resolution=16, h=3072)))
    results.append(run_sweep("width", build_v2_compressed, "v2_compressed", "e", e_values,
                             fixed=dict(resolution=16, h=3072, compress_m=128)))

    # --- Sweep C: compress_m (v2 only -- the O(L) vs O(L^2) transition as a curve) ---
    m_values = [8, 16, 32, 64, 128, 256, 512, 1024]
    results.append(run_sweep("compress_m", build_v2_compressed, "v2_compressed", "compress_m",
                             m_values, fixed=dict(resolution=16, e=768, h=3072)))

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "feature_scaling_results.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "conditions": {"B": 1, "K": K, "image_size": list(IMAGE_SIZE),
                                "reps": REPS, "warmup": WARMUP},
                  "sweeps": results}, f, indent=2)
    print(f"\nWrote {out_dir / 'feature_scaling_results.json'}")


if __name__ == "__main__":
    main()

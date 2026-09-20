"""
Fair inference-time / FLOPs comparison: medverse vs patchset3d vs patchset3d_v2.

"Fair" here means each model is benchmarked at the precision it was actually trained/
evaluated at, not an artificially-equalized one:
  - medverse: plain fp32, no autocast anywhere (MedverseModel has no internal autocast,
    and 69_medverse_varspacing_6_1_5.yaml / the released checkpoint were never trained
    under bf16 -- forcing bf16 on it would benchmark a regime it was never validated in).
  - patchset3d / patchset3d_v2: bf16 autocast (arch.encoder_precision=bf16, the setting
    every real training run in this repo uses for both classes), matching
    experiments/3d/evaluate.py::_eval_autocast's own bf16-CUDA-autocast convention.

Architectures used are each model's actual real-run configuration, not a toy default:
  - patchset3d:    97_iris_decoder_ct_only / 99's own arch (plainconv_ts, e=768, l=4, a=12,
                   decoder=iris) -- the wandb run x1gz71wj config, verbatim.
  - patchset3d_v2: 101_patchset_v2_mask8_wide's arch (plainconv_ts widened to 768,
                   mask_patch_size=8, feat_norm=context, img_embed_mlp=true).
  - medverse:      released weights, sw_roi_size=(128,128,128) -- matches
                   69_medverse_varspacing_6_1_5.yaml's single-forward (no cascade) setup.

All three at B=1, K=1, 128^3 (matches bench_arch.py's and 69's own convention).

    .venv_blackwell/bin/python results/presentations/perf/bench_inference_compare.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

import torch

DEV = torch.device("cuda")
IMAGE_SIZE = (128, 128, 128)
K = 1
REPS = 20
WARMUP = 5


def human(n):
    for u in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000
    return f"{n:.1f}T"


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def make_inputs():
    D, H, W = IMAGE_SIZE
    img = torch.randn(1, 1, D, H, W, device=DEV)
    ctx_in = torch.randn(1, K, 1, D, H, W, device=DEV)
    ctx_out = (torch.rand(1, K, D, H, W, device=DEV) > 0.7).float()
    return img, ctx_in, ctx_out


def build_medverse():
    from src.benchmark_models.medverse import MedverseModel
    m = MedverseModel(device=DEV, sw_roi_size=IMAGE_SIZE)
    return m


def build_patchset3d():
    """97_iris_decoder_ct_only / 99's own arch (wandb run x1gz71wj), verbatim."""
    from src.models.patchset3d import PatchSet3D
    arch = dict(
        resolution=16, e=768, h=3072, l=4, a=12, thinking_rows=8, residual_decay=0.95,
        fourier_bands=8, mask_patch_size=8, mask_patch_decode_size=8, mask_embed="linear",
        mask_slots=1, decode_source="img", context_id_embed=True, max_context=16,
        full_attn=True, query_self_attn=True, register_routed=False, register_flex=False,
        encoder="plainconv_ts", encoder_frozen=False, encoder_input_norm="instance",
        encoder_precision="bf16", plainconv_ts_n_stages=5,
        plainconv_ts_features_per_stage=[32, 64, 256, 512], nnunet_ts_stages=[2, 3],
        enc_dims=[32, 32, 32, 32], img_embed_mlp=True, feat_norm="self",
        fine_decode=True, fine_stage=[0, 1], fine_proj_dim=96,
        decoder="iris", decoder_dim=64, iris_pixelshuffle_r=4, iris_m=10, iris_ctx_layers=2,
        image_size=list(IMAGE_SIZE),
    )
    return PatchSet3D(**arch).to(DEV)


def build_patchset_v2():
    """101_patchset_v2_mask8_wide's own arch, verbatim."""
    from src.models.patchset3d_v2 import PatchSetV2
    arch = dict(
        resolution=16, e=768, h=3072, l=4, a=12, thinking_rows=8, residual_decay=0.95,
        fourier_bands=8, mask_patch_size=8, mask_embed="linear", compress_m=128,
        compress_layers=1, context_id_embed=True, max_context=16, cascade_registers=False,
        fine_stage=[0, 1, 2], decoder_dim=64, img_embed_mlp=True, feat_norm="context",
        encoder="plainconv_ts", encoder_frozen=False, encoder_input_norm="instance",
        encoder_precision="bf16", plainconv_ts_n_stages=5,
        plainconv_ts_features_per_stage=[32, 64, 128, 768], nnunet_ts_stages=[3],
        enc_dims=[32, 32, 32, 32],
        image_size=list(IMAGE_SIZE),
    )
    return PatchSetV2(**arch).to(DEV)


def time_predict(model, img, ctx_in, ctx_out, autocast: bool):
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast
          else torch.autocast(device_type="cuda", enabled=False))

    def call():
        with torch.no_grad(), ctx:
            return model.predict(img, ctx_in, ctx_out)

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = call()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / REPS
    peak = torch.cuda.max_memory_allocated() / 1e9
    return ms, peak, tuple(out.shape)


def bench_model(name, build_fn, native_autocast: bool):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_fn()
    net = getattr(model, "model", model)
    net.eval() if hasattr(net, "eval") else None
    total, trainable = count_params(net)
    print(f"Params: {human(total)} total / {human(trainable)} trainable")

    img, ctx_in, ctx_out = make_inputs()

    from evaluate import measure_flops
    flops = measure_flops(model, IMAGE_SIZE, K, DEV)
    print(f"FLOPs (predict, one sample): {flops['total']:.2f} GFLOPs "
          f"[encoder={flops['encoder']}, transformer={flops['transformer']}, "
          f"other={flops['other']}]")

    ms_native, peak_native, out_shape = time_predict(model, img, ctx_in, ctx_out,
                                                      autocast=native_autocast)
    prec_native = "bf16 (native)" if native_autocast else "fp32 (native)"
    print(f"Inference [{prec_native}]: {ms_native:7.1f} ms   peak {peak_native:5.2f} GB   "
          f"out {out_shape}")

    del model, net
    torch.cuda.empty_cache()
    return dict(name=name, params_total=total, params_trainable=trainable,
               gflops=flops["total"], gflops_encoder=flops["encoder"],
               gflops_transformer=flops["transformer"], gflops_other=flops["other"],
               ms_native=ms_native, peak_native_gb=peak_native, native_precision=prec_native,
               out_shape=list(out_shape))


def main():
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"Conditions: B=1, K={K}, image_size={IMAGE_SIZE}, {REPS} reps ({WARMUP} warmup)")

    results = [
        bench_model("medverse", build_medverse, native_autocast=False),
        bench_model("patchset3d (97/99, decoder=iris)", build_patchset3d, native_autocast=True),
        bench_model("patchset3d_v2 (101, mask8_wide)", build_patchset_v2, native_autocast=True),
    ]

    print(f"\n{'='*90}\nSUMMARY (B=1, K={K}, {IMAGE_SIZE[0]}^3)\n{'='*90}")
    hdr = f"{'model':<34}{'params':>9}{'GFLOPs':>10}{'native ms':>12}{'peak GB':>9}"
    print(hdr)
    for r in results:
        print(f"{r['name']:<34}{human(r['params_total']):>9}{r['gflops']:>10.1f}"
              f"{r['ms_native']:>12.1f}{r['peak_native_gb']:>9.2f}")

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "results.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "conditions": {"B": 1, "K": K, "image_size": list(IMAGE_SIZE),
                                "reps": REPS, "warmup": WARMUP},
                  "results": results}, f, indent=2)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

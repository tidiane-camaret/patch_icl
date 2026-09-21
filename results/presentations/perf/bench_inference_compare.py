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
  - patchset3d (iris):  97_iris_decoder_ct_only / 99's own arch (plainconv_ts, e=768, l=4,
                   a=12, decoder=iris) -- the wandb run x1gz71wj config, verbatim. NOTE:
                   decoder=iris skips the R^3-token main self-attention transformer
                   entirely (patchset3d.py:1108) -- see RESULTS.md "Follow-up" section.
  - patchset3d (conv):  92_multisource_synth's arch (resolved via Hydra compose), verbatim --
                   decoder=conv, which does NOT skip that transformer. This is the genuine
                   R^3=4096-raw-token dense self-attention path.
  - patchset3d_v2: 101_patchset_v2_mask8_wide's arch (plainconv_ts widened to 768,
                   mask_patch_size=8, feat_norm=context, img_embed_mlp=true).
  - patchset3d_v2 (103): 103_patchset_v2_cascade's arch (resolved via Hydra compose),
                   verbatim -- benchmarked single-level (K=1, no cascade re-crop/re-forward)
                   like every other model here, since cascade.py::run_cascade just calls this
                   same forward N times per task, it doesn't change per-call cost. Adds
                   mask_embed=conv (MaskConvEmbedV2), decode_layers=3, encoder_spacing_aware,
                   encoder_input_norm=zscore vs 101 -- see docs/logs.md 2026-09-20.
  - medverse:      released weights, sw_roi_size=(128,128,128) -- matches
                   69_medverse_varspacing_6_1_5.yaml's single-forward (no cascade) setup.

All five at B=1, K=1, 128^3 (matches bench_arch.py's and 69's own convention).

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


def build_patchset3d_conv():
    """92_multisource_synth's own arch (resolved via Hydra compose), verbatim: decoder=conv
    (NOT iris) -- 92's lineage (92 -> 89_multisource_cascade -> 88_cascade) selects
    model=m2_patchset_decoder, which never sets decoder=iris. Unlike the iris-decoder arch
    above, decoder=conv does NOT skip _attn (patchset3d.py:1108 only skips it for
    decoder_kind=='iris') -- this is the genuine R^3=4096-raw-token dense self-attention
    path (arch.seq_compress defaults off, full_attn=True), included to measure the real
    cost of that path instead of the iris config's dead branch. See docs/logs.md
    2026-09-20 and RESULTS.md's "Follow-up" section."""
    from src.models.patchset3d import PatchSet3D
    arch = dict(
        resolution=16, e=768, h=3072, l=4, a=12, thinking_rows=8, residual_decay=0.95,
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


def build_patchset_v2_103():
    """103_patchset_v2_cascade's own arch (resolved via Hydra compose), verbatim -- benchmarked
    single-level (K=1, no cascade re-crop/re-forward) like every other model here; the cascade
    wrapper (cascade.py::run_cascade) only calls this same forward N times per task with
    different crops, it doesn't change per-call cost. Differences from 101's arch: encoder
    widened further (plainconv_ts_features_per_stage [32,64,128,768] -> [32,64,256,768],
    nnunet_ts_stages [3] unchanged so still no concat), encoder_input_norm instance->zscore,
    encoder_spacing_aware=True (was unset/False), mask_embed=conv (MaskConvEmbedV2, not
    linear), decode_layers=3 (was 1) -- see docs/logs.md 2026-09-20 entries."""
    from src.models.patchset3d_v2 import PatchSetV2
    arch = dict(
        resolution=16, e=768, h=3072, l=4, a=12, thinking_rows=8, residual_decay=0.95,
        fourier_bands=8, mask_patch_size=8, mask_embed="conv", compress_m=128,
        compress_layers=1, context_id_embed=True, max_context=16, cascade_registers=False,
        fine_stage=[0, 1], decoder_dim=64, img_embed_mlp=True, feat_norm="context",
        decode_layers=3,
        encoder="plainconv_ts", encoder_frozen=False, encoder_input_norm="zscore",
        encoder_precision="bf16", encoder_spacing_aware=True, plainconv_ts_n_stages=5,
        plainconv_ts_features_per_stage=[32, 64, 256, 768], nnunet_ts_stages=[3],
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


def bench_model(name, build_fn, native_autocast: bool, extra_bf16: bool = False):
    """extra_bf16: also time an externally-bf16-autocast-wrapped pass, IN ADDITION to the
    native-precision one. Only meaningful for models with no internal autocast of their own
    (medverse) -- patchset3d/v2's encoders apply their own bf16 autocast unconditionally at
    construction time, so external wrapping there is already a no-op (they're bf16 natively)
    and forcing fp32 externally instead crashes (see module docstring)."""
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

    ms_bf16 = peak_bf16 = None
    if extra_bf16:
        ms_bf16, peak_bf16, _ = time_predict(model, img, ctx_in, ctx_out, autocast=True)
        print(f"Inference [bf16 (extra, not the trained precision)]: {ms_bf16:7.1f} ms   "
              f"peak {peak_bf16:5.2f} GB")

    del model, net
    torch.cuda.empty_cache()
    return dict(name=name, params_total=total, params_trainable=trainable,
               gflops=flops["total"], gflops_encoder=flops["encoder"],
               gflops_transformer=flops["transformer"], gflops_other=flops["other"],
               ms_native=ms_native, peak_native_gb=peak_native, native_precision=prec_native,
               ms_bf16_extra=ms_bf16, peak_bf16_extra_gb=peak_bf16,
               out_shape=list(out_shape))


def main():
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"Conditions: B=1, K={K}, image_size={IMAGE_SIZE}, {REPS} reps ({WARMUP} warmup)")

    results = [
        bench_model("medverse", build_medverse, native_autocast=False, extra_bf16=True),
        bench_model("patchset3d (97/99, decoder=iris)", build_patchset3d, native_autocast=True),
        bench_model("patchset3d (92_multisource_synth, decoder=conv)", build_patchset3d_conv,
                   native_autocast=True),
        bench_model("patchset3d_v2 (101, mask8_wide)", build_patchset_v2, native_autocast=True),
        bench_model("patchset3d_v2 (103, cascade arch, decode_layers=3)", build_patchset_v2_103,
                   native_autocast=True),
    ]

    print(f"\n{'='*90}\nSUMMARY (B=1, K={K}, {IMAGE_SIZE[0]}^3)\n{'='*90}")
    hdr = f"{'model':<34}{'params':>9}{'GFLOPs':>10}{'native ms':>12}{'peak GB':>9}{'bf16 ms':>10}"
    print(hdr)
    for r in results:
        bf16 = f"{r['ms_bf16_extra']:.1f}" if r["ms_bf16_extra"] is not None else "-"
        print(f"{r['name']:<34}{human(r['params_total']):>9}{r['gflops']:>10.1f}"
              f"{r['ms_native']:>12.1f}{r['peak_native_gb']:>9.2f}{bf16:>10}")

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "results.json", "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "conditions": {"B": 1, "K": K, "image_size": list(IMAGE_SIZE),
                                "reps": REPS, "warmup": WARMUP},
                  "results": results}, f, indent=2)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

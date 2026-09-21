"""Stage-level wall-clock breakdown for patchset3d (v1) vs patchset3d_v2, via forward
hooks (CUDA events) on net.encoder / net.transformer. Investigates why v1 (self-attention
over R^3=4096 raw tokens/volume, seq_compress=False) and v2 (compressed to compress_m=128
tokens/volume) show near-identical total inference time despite the huge token-count gap.

Answer (see RESULTS.md "Follow-up" section): v1's decoder=iris config skips the R^3-scale
self.transformer entirely (patchset3d.py:1108) -- self.transformer fires 0 times/call here,
confirmed by this script. v1's actual decode attention (_iris_task_encode/_decode_iris) is
already m-scale (iris_m=10 query rows cross-attending into the R^3 grid, cost O(m*R^3) not
O(R^6)) -- same complexity family as v2's compress_m, just m=10 vs m=128.

    .venv_blackwell/bin/python results/presentations/perf/profile_stage_split.py
"""
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
REPS = 12
WARMUP = 4


def make_inputs():
    D, H, W = IMAGE_SIZE
    img = torch.randn(1, 1, D, H, W, device=DEV)
    ctx_in = torch.randn(1, K, 1, D, H, W, device=DEV)
    ctx_out = (torch.rand(1, K, D, H, W, device=DEV) > 0.7).float()
    return img, ctx_in, ctx_out


def build_patchset3d():
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
    """92_multisource_synth's arch (decoder=conv, NOT iris) -- does not skip self.transformer.
    See bench_inference_compare.py::build_patchset3d_conv for the full explanation."""
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


class StageTimer:
    """CUDA-event start/end around a submodule's forward, accumulated across calls
    within one profiled forward pass."""

    def __init__(self, module, name):
        self.name = name
        self.events = []
        self.total_calls = 0
        self._start = None
        module.register_forward_pre_hook(self._pre)
        module.register_forward_hook(self._post)

    def _pre(self, module, args):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def _post(self, module, args, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append((self._start, e))
        self.total_calls += 1

    def reset(self):
        self.events = []

    def total_ms(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.events)


def profile(name, build_fn):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    torch.cuda.empty_cache()
    model = build_fn()
    net = getattr(model, "model", model)
    net.eval()

    enc_timer = StageTimer(net.encoder, "encoder")
    tfm_timer = StageTimer(net.transformer, "transformer")

    img, ctx_in, ctx_out = make_inputs()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def call():
        with torch.no_grad(), ctx:
            return model.predict(img, ctx_in, ctx_out)

    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    total_ms_list, enc_ms_list, tfm_ms_list = [], [], []
    for _ in range(REPS):
        enc_timer.reset()
        tfm_timer.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        total_ms_list.append(1000 * (time.perf_counter() - t0))
        enc_ms_list.append(enc_timer.total_ms())
        tfm_ms_list.append(tfm_timer.total_ms())

    total = sum(total_ms_list) / REPS
    enc = sum(enc_ms_list) / REPS
    tfm = sum(tfm_ms_list) / REPS
    other = total - enc - tfm
    print(f"total:       {total:7.2f} ms")
    print(f"  encoder:   {enc:7.2f} ms  ({100*enc/total:5.1f}%)  "
          f"calls/fwd={enc_timer.total_calls/REPS:.1f}")
    print(f"  transformer:{tfm:6.2f} ms  ({100*tfm/total:5.1f}%)  "
          f"calls/fwd={tfm_timer.total_calls/REPS:.1f}")
    print(f"  other:     {other:7.2f} ms  ({100*other/total:5.1f}%)  "
          f"(decode / iris task-encode / misc, not module-hooked)")

    del model, net
    torch.cuda.empty_cache()
    return dict(name=name, total=total, encoder=enc, transformer=tfm, other=other)


def main():
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"Conditions: B=1, K={K}, image_size={IMAGE_SIZE}, {REPS} reps ({WARMUP} warmup)")
    r1 = profile("patchset3d (decoder=iris -> self.transformer skipped, patchset3d.py:1108)",
                 build_patchset3d)
    r2 = profile("patchset3d (decoder=conv, 92_multisource_synth -> R^3=4096 tok/vol)",
                 build_patchset3d_conv)
    r3 = profile("patchset3d_v2 (compress_m=128 tok/vol)", build_patchset_v2)

    print(f"\n{'='*70}\nCOMPARISON\n{'='*70}")
    for r in (r1, r2, r3):
        print(f"{r['name']:<55} total={r['total']:7.2f}ms  "
              f"enc={r['encoder']:6.2f}ms  tfm={r['transformer']:6.2f}ms  other={r['other']:6.2f}ms")


if __name__ == "__main__":
    main()

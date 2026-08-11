"""Extract `fomofo/tap-ct-b-3d` embeddings for TotalSegInContextDataset items.

The dataloader (src/totalseg_dataloader_incontext.py) and the TAP-CT inference
script (experiments/encoders/tap_ct.py) preprocess CT very differently, so a
dataloader tensor cannot be fed to the TAP processor as-is. This module bridges
the two. Four reconciliations (see docs comments inline):

1. INTENSITY (mandatory). The dataloader image is ALREADY z-scored with the
   TotalSeg CT stats `clip([-1007,1573]) -> (x + 167.3) / 505.8`. TAP's processor
   expects raw HU (it clips [-1008,822] and z-scores with its own mean/std). We
   invert the dataloader normalization back to HU first; TAP then re-normalizes.
   Skipping this collapses every voxel to ~0.27 and features are meaningless.

2. PADDING. Both dataloader paths zero/min-pad in NORMALIZED space, so padded
   voxels invert to ~-167.3 HU (soft tissue), not air. With use_crop=True and a
   T divisible by 8 the crop usually fills T^3 exactly (no padding); when it does
   pad, pass pad_hu to reset the fill region to air.

3. ORIENTATION. convert_to_npy stores canonical RAS (axis0=L-R, axis2=S-I). TAP
   was trained LPS with the axial (S) axis first (patch depth=4 coarse; in-plane
   8x8 -> 224). ras_to_lps_axial_first() moves S to axis 0 and flips R->L, A->P
   so TAP applies its learned axial-plane features to the real axial plane.

4. RESOLUTION. TAP's stock processor always upsamples in-plane to 224^2. We build
   TAPCTProcessor(resize_dims=(T,T)) instead so a T^3 dataloader cube is fed at
   native resolution (requires T % 8 == 0; pair with use_crop for real detail).

Requires the SDPA attention patch (tap_ct_bench.load_model) so a whole-volume
forward does not OOM without xformers. See project_tap_ct_oom memory.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from tap_ct_bench import load_model  # SDPA-patched model loader

# Dataloader CT z-score constants (src/totalseg_dataset.py). Inverting these
# recovers HU (lossless for TAP: low clip -1007 ~= TAP's -1008; anything the
# dataloader kept above 822 HU is clipped by TAP anyway).
CT_MEAN = -167.3
CT_STD = 505.8


def denorm_to_hu(img_norm: torch.Tensor) -> torch.Tensor:
    """Invert the dataloader z-score to recover HU. Accepts any shape."""
    return img_norm.float() * CT_STD + CT_MEAN


def ras_to_lps_axial_first(hu: np.ndarray) -> np.ndarray:
    """RAS cube (axis0=R, axis1=A, axis2=S) -> LPS axial-first (S, P, L).

    Move the superior-inferior (axial) axis to position 0 so TAP treats the axial
    plane as in-plane, and flip R->L, A->P to match TAP's LPS training frame.
    """
    return np.ascontiguousarray(np.flip(hu.transpose(2, 1, 0), axis=(1, 2)))


def make_processor(T: int):
    """TAP processor with in-plane resize disabled (resize_dims=(T,T)).

    Keeps a T^3 dataloader cube at native resolution instead of upsampling to
    224^2. T must be divisible by 8 (in-plane patch) — and hence by 4 (depth).
    """
    assert T % 8 == 0, f"resize_dims=(T,T) needs T divisible by 8, got {T}"
    proc = AutoImageProcessor.from_pretrained(
        "fomofo/tap-ct-b-3d", trust_remote_code=True
    )
    proc.resize_dims = (T, T)
    return proc


def item_to_tap_input(
    img_norm: torch.Tensor, proc, to_lps: bool = True, pad_hu: float | None = None
) -> torch.Tensor:
    """Dataloader image (1,D,H,W) or (D,H,W) -> TAP pixel_values (1,1,D',T,T).

    pad_hu: if set, voxels at the dataloader's normalized-zero fill (~-167.3 HU)
    are reset to this HU before TAP normalization (air = -1024). Leave None when
    use_crop fills T^3 exactly (no padding).
    """
    hu = denorm_to_hu(img_norm).squeeze().cpu().numpy()  # (D,H,W)
    if pad_hu is not None:
        fill = 0.0 * CT_STD + CT_MEAN  # normalized-0 -> HU
        hu = np.where(np.isclose(hu, fill, atol=1.0), pad_hu, hu)
    if to_lps:
        hu = ras_to_lps_axial_first(hu)
    return proc(hu[None, None])["pixel_values"]  # TAP clip + z-score


@torch.no_grad()
def embed(model, pixel_values: torch.Tensor, device, precision: str = "bf16"):
    """Run the TAP model; return (last_hidden_state, pooler_output) on CPU.

    last_hidden_state: (1, N, 768) patch tokens (N = (D'/4)*(T/8)^2).
    pooler_output:     (1, 768) CLS token.
    """
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    from contextlib import nullcontext

    ctx = nullcontext() if precision == "fp32" else torch.autocast("cuda", dtype=dtypes[precision])
    with ctx:
        out = model(pixel_values.to(device))
    return out.last_hidden_state.float().cpu(), out.pooler_output.float().cpu()


def dense_features(model, proc, img_norm, device, to_lps=True, precision="bf16"):
    """Dataloader image (1,D,H,W) -> (dense feature rows (N,C), grid dims (gd,gh,gw)).

    Grid dims are derived from the processor, not assumed: in-plane = resize_dims/8
    (patch 8), depth = N/(gh*gw) (patch 4, after depth padding). Robust to non-224
    resize_dims and non-cube inputs. Tokens are row-major (D,H,W), so the reshape aligns
    with occ_labels' area-pooled occupancy grid.
    """
    pix = item_to_tap_input(img_norm, proc, to_lps=to_lps)
    lhs, _ = embed(model, pix, device, precision=precision)   # (1, N, C)
    rh, rw = proc.resize_dims
    gh, gw = rh // 8, rw // 8
    n, c = lhs.shape[1], lhs.shape[2]
    gd = n // (gh * gw)
    return lhs[0].reshape(gd, gh, gw, c).reshape(-1, c), (gd, gh, gw)


def occ_labels(mask, grid_dims, to_lps=True):
    """Reorient a (D,H,W) mask like the image and area-pool to `grid_dims` -> soft
    occupancy fraction per cell, flattened to (N,). Must use the same to_lps as the
    features so cells align."""
    m = mask.float().cpu().numpy()
    if to_lps:
        m = np.ascontiguousarray(np.flip(m.transpose(2, 1, 0), axis=(1, 2)))
    occ = F.interpolate(torch.from_numpy(m)[None, None], size=grid_dims, mode="area")[0, 0]
    return occ.reshape(-1)


if __name__ == "__main__":
    # Smoke test on a synthetic dataloader-normalized cube (no dataset needed):
    # verifies no double-normalization, correct token count, and shapes.
    T = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = make_processor(T)

    # Fake a dataloader item: z-scored CT in the dataloader's value range.
    img_norm = torch.randn(1, T, T, T) * 0.5  # ~ N(0, .5), within [-1.66, 3.44]
    pix = item_to_tap_input(img_norm, proc)
    print(f"dataloader image {tuple(img_norm.shape)} -> TAP pixel_values {tuple(pix.shape)}")
    print(f"  HU range after de-norm: [{denorm_to_hu(img_norm).min():.0f}, "
          f"{denorm_to_hu(img_norm).max():.0f}]  "
          f"TAP-normalized range: [{pix.min():.2f}, {pix.max():.2f}]")

    expected_tokens = (T // 4) * (T // 8) ** 2 + 5
    print(f"  expected tokens = (T/4)*(T/8)^2 + 5 = {expected_tokens}")

    if device.type == "cuda":
        model = load_model(device, use_sdpa=True)
        lhs, cls = embed(model, pix, device, precision="bf16")
        print(f"  last_hidden_state {tuple(lhs.shape)}  pooler {tuple(cls.shape)}")
    else:
        print("  (no CUDA: skipping forward pass)")

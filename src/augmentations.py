"""
3D data augmentations for in-context segmentation.

Three augmentation modes:
  - Task augmentation: geometric, same random params applied to ALL volumes in a
    task (query + every context entry) so the task stays consistent.
  - Within-task (intensity): independently sampled per volume to add intra-task
    visual diversity.
  - Synth augmentation: heavy geometric + intensity, independently sampled per
    copy so K+1 views of the same supervoxel diverge as much as possible.

All ops work on CPU tensors inside DataLoader workers.
Geometric ops batch the K+1 volumes into one grid_sample call for speed.

Shapes
------
  images : (N, 1, D, H, W)  float32  z-score values in [CT_NORM_MIN, CT_NORM_MAX] ≈ [-1.66, +3.44]
  masks  : (N, D, H, W)     int64

Usage
-----
  from src.augmentations import apply_task_aug, apply_intensity_aug, apply_synth_aug

  # task aug: query + all context batched together
  images, masks = apply_task_aug(images, masks, cfg.augmentations.task)

  # intensity aug: one volume at a time
  for i in range(N):
      images[i] = apply_intensity_aug(images[i], cfg.augmentations.intensity)

  # synth aug: call independently per copy
  image, mask = apply_synth_aug(image, mask, cfg.augmentations.synth)
"""

import math
import random
from typing import Tuple

import torch
import torch.nn.functional as F

from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_3d(rx: float, ry: float, rz: float) -> torch.Tensor:
    """ZYX Euler angles (radians) → 3×3 rotation matrix."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx,  cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0,  cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return Rz @ Ry @ Rx  # (3, 3)


def _make_affine_theta(
    rx: float, ry: float, rz: float,
    scale: float,
    tx: float, ty: float, tz: float,
) -> torch.Tensor:
    """Build a (1, 3, 4) affine matrix for F.affine_grid."""
    R = _rotation_matrix_3d(rx, ry, rz)
    A = R * scale                                           # (3, 3)
    t = torch.tensor([[tx], [ty], [tz]], dtype=torch.float32)
    return torch.cat([A, t], dim=1).unsqueeze(0)            # (1, 3, 4)


def _apply_grid(images: torch.Tensor, masks: torch.Tensor,
                grid: torch.Tensor, mask_mode: str = "nearest") -> Tuple[torch.Tensor, torch.Tensor]:
    """grid: (N, D, H, W, 3) in normalised coords [-1, 1].

    A float `masks` (soft partial-volume target) is warped with `mask_mode` (bilinear
    anti-aliases the boundary) and kept float in [0, 1]; an integer mask uses nearest +
    long exactly as before regardless of `mask_mode`."""
    images = F.grid_sample(images, grid, mode="bilinear",
                           padding_mode="border", align_corners=False)
    mask_soft = masks.is_floating_point()
    mode = "bilinear" if (mask_soft and mask_mode == "bilinear") else "nearest"
    masks_f = F.grid_sample(masks.unsqueeze(1).float(), grid, mode=mode,
                             padding_mode="zeros", align_corners=False).squeeze(1)
    return images, (masks_f.clamp(0.0, 1.0) if mask_soft else masks_f.long())


def _svf_displacement(shape, control_points, max_disp, num_steps,
                      generator=None, device=None):
    """Diffeomorphic displacement field via scaling-and-squaring of a smooth SVF.

    A stationary velocity field is sampled on a coarse grid of `control_points` nodes
    per axis, upsampled, then integrated by scaling-and-squaring (VoxelMorph VecInt) so
    the resulting warp is guaranteed invertible — no folding, so masks stay valid.
    Unlike the legacy `elastic` op it cannot tear labels.

    `control_points` is a COUNT (not a voxel spacing), so the correlation length is a
    fixed fraction of the volume → resolution-invariant across our fixed-size crops
    (we take 128³ crops at 1–4mm; smoothness must not depend on the mm spacing).
    shape: (D, H, W). `max_disp` is the velocity std in normalized [-1, 1] grid units
    (same semantics as elastic.alpha). Returns (1, D, H, W, 3) displacement in [-1, 1].
    """
    D, H, W = shape
    cp = max(int(control_points), 2)
    sd, sh, sw = min(cp, D), min(cp, H), min(cp, W)
    v = torch.randn(1, 3, sd, sh, sw, generator=generator, device=device) * max_disp
    v = F.interpolate(v, size=(D, H, W), mode="trilinear", align_corners=False)
    v = v.permute(0, 2, 3, 4, 1)                                    # (1,D,H,W,3) velocity

    base = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0),
                         (1, 1, D, H, W), align_corners=False)      # identity grid
    phi = v / (2 ** num_steps)
    for _ in range(num_steps):                                      # phi = phi ∘ (id+phi) + phi
        # padding_mode="border" handles the (rare) out-of-range sample; no clamp
        # here — clamping inside the integration would distort the diffeomorphism.
        warped = F.grid_sample(phi.permute(0, 4, 1, 2, 3), base + phi,
                               mode="bilinear", padding_mode="border",
                               align_corners=False).permute(0, 2, 3, 4, 1)
        phi = phi + warped
    return phi


# ---------------------------------------------------------------------------
# Task-level (geometric, shared params)
# ---------------------------------------------------------------------------

def apply_task_aug(
    images: torch.Tensor,   # (N, 1, D, H, W)
    masks: torch.Tensor,    # (N, D, H, W)
    cfg,                    # DictConfig or SimpleNamespace: cfg.flip / .affine / .elastic
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply geometric augmentations with one shared set of random params."""
    N = images.shape[0]
    mask_mode = getattr(cfg, "mask_interp", "nearest")   # "bilinear" -> soft-mask warp

    # --- Flips -----------------------------------------------------------
    fcfg = cfg.flip
    for vol_dim, mask_dim, p in [
        (2, 1, fcfg.p_d),
        (3, 2, fcfg.p_h),
        (4, 3, fcfg.p_w),
    ]:
        if random.random() < p:
            images = images.flip(vol_dim)
            masks  = masks.flip(mask_dim)

    # --- Affine ----------------------------------------------------------
    acfg = cfg.affine
    if random.random() < acfg.p:
        max_rad = acfg.max_angle_deg * math.pi / 180.0
        rx = random.uniform(-max_rad, max_rad)
        ry = random.uniform(-max_rad, max_rad)
        rz = random.uniform(-max_rad, max_rad)
        scale = random.uniform(acfg.scale_min, acfg.scale_max)
        tx = random.uniform(-acfg.max_translate, acfg.max_translate)
        ty = random.uniform(-acfg.max_translate, acfg.max_translate)
        tz = random.uniform(-acfg.max_translate, acfg.max_translate)

        theta = _make_affine_theta(rx, ry, rz, scale, tx, ty, tz)
        theta = theta.expand(N, -1, -1)                     # (N, 3, 4)
        grid  = F.affine_grid(theta, images.shape, align_corners=False)
        images, masks = _apply_grid(images, masks, grid, mask_mode)

    # --- Elastic ---------------------------------------------------------
    ecfg = cfg.elastic
    if random.random() < ecfg.p:
        _, _, D, H, W = images.shape
        gs = max(ecfg.grid_scale, 2)
        sd, sh, sw = max(D // gs, 2), max(H // gs, 2), max(W // gs, 2)

        # Small random field → upsample → add to identity grid
        disp = torch.randn(1, 3, sd, sh, sw) * ecfg.alpha
        disp = F.interpolate(disp, size=(D, H, W),
                             mode="trilinear", align_corners=False)
        disp = disp.permute(0, 2, 3, 4, 1).expand(N, -1, -1, -1, -1)

        theta_id = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)
        base_grid = F.affine_grid(theta_id, images.shape, align_corners=False)
        grid = (base_grid + disp).clamp(-1.0, 1.0)
        images, masks = _apply_grid(images, masks, grid, mask_mode)

    # --- Diffeomorphic deform (SVF, shared field, no folding) ------------
    dcfg = getattr(cfg, "deform", None)
    if dcfg is not None and random.random() < dcfg.p:
        _, _, D, H, W = images.shape
        phi = _svf_displacement((D, H, W), dcfg.control_points, dcfg.max_disp, dcfg.num_steps)
        theta_id = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)
        base_grid = F.affine_grid(theta_id, images.shape, align_corners=False)
        images, masks = _apply_grid(images, masks, (base_grid + phi).clamp(-1.0, 1.0), mask_mode)

    # Independent per-volume geometry (flip/affine/elastic) is applied by the
    # caller via apply_per_image_aug using cfg.per_image — see the real-context
    # and self_context paths in the dataloader.

    return images, masks


def apply_per_image_aug(image, mask, cfg):
    """Per-IMAGE geometric aug: an INDEPENDENT flip/affine/elastic on ONE volume.

    Unlike apply_task_aug — which draws ONE shared transform for all K+1 volumes so the
    target/context correspondence is preserved — this draws its own random transform for a
    single (image, mask), so the volume's pose changes relative to the others. Used to jitter
    a self-context copy's pose independently of the target (pose-invariance training lever).
    `cfg` mirrors the task schema: cfg.flip / cfg.affine / cfg.elastic. Shapes preserved:
    image (1, D, H, W), mask (D, H, W)."""
    img = image.unsqueeze(0)        # (1, 1, D, H, W)
    msk = mask.unsqueeze(0)         # (1, D, H, W)
    mask_mode = getattr(cfg, "mask_interp", "nearest")   # "bilinear" -> soft-mask warp

    fcfg = cfg.flip
    for vol_dim, mask_dim, p in [(2, 1, fcfg.p_d), (3, 2, fcfg.p_h), (4, 3, fcfg.p_w)]:
        if random.random() < p:
            img = img.flip(vol_dim); msk = msk.flip(mask_dim)

    acfg = cfg.affine
    if random.random() < acfg.p:
        max_rad = acfg.max_angle_deg * math.pi / 180.0
        rx, ry, rz = (random.uniform(-max_rad, max_rad) for _ in range(3))
        scale = random.uniform(acfg.scale_min, acfg.scale_max)
        tx, ty, tz = (random.uniform(-acfg.max_translate, acfg.max_translate) for _ in range(3))
        theta = _make_affine_theta(rx, ry, rz, scale, tx, ty, tz)
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        img, msk = _apply_grid(img, msk, grid, mask_mode)

    ecfg = cfg.elastic
    if random.random() < ecfg.p:
        _, _, D, H, W = img.shape
        gs = max(ecfg.grid_scale, 2)
        sd, sh, sw = max(D // gs, 2), max(H // gs, 2), max(W // gs, 2)
        disp = F.interpolate(torch.randn(1, 3, sd, sh, sw) * ecfg.alpha, size=(D, H, W),
                             mode="trilinear", align_corners=False).permute(0, 2, 3, 4, 1)
        base = F.affine_grid(torch.eye(3, 4).unsqueeze(0), img.shape, align_corners=False)
        img, msk = _apply_grid(img, msk, (base + disp).clamp(-1.0, 1.0), mask_mode)

    dcfg = getattr(cfg, "deform", None)
    if dcfg is not None and random.random() < dcfg.p:
        _, _, D, H, W = img.shape
        phi = _svf_displacement((D, H, W), dcfg.control_points, dcfg.max_disp, dcfg.num_steps)
        base = F.affine_grid(torch.eye(3, 4).unsqueeze(0), img.shape, align_corners=False)
        img, msk = _apply_grid(img, msk, (base + phi).clamp(-1.0, 1.0), mask_mode)

    return img.squeeze(0), msk.squeeze(0)   # (1, D, H, W), (D, H, W)


# ---------------------------------------------------------------------------
# Within-task (intensity, independent per volume)
# ---------------------------------------------------------------------------

def apply_intensity_aug(
    image: torch.Tensor,    # (1, D, H, W)  float32  values in [0, 1]
    cfg,                    # cfg.gaussian_noise / .gaussian_blur / .brightness_contrast / .gamma
) -> torch.Tensor:
    """Intensity augmentations sampled independently for each volume."""

    # --- GIN / IPA appearance transform ----------------------------------
    # Randomly-weighted nonlinear conv (GIN), optionally spatially blended
    # across independent copies (IPA). Applied first so downstream intensity
    # ops act on the warped appearance. Independent per volume (this fn runs
    # per volume), so target and its K contexts get different warps.
    gincfg = getattr(cfg, "gin", None)
    if gincfg is not None and random.random() < gincfg.p:
        scale_pool = tuple(getattr(gincfg, "scale_pool", (1, 3)))
        n_layer    = getattr(gincfg, "n_layer", 4)
        interm     = getattr(gincfg, "interm_channel", 2)
        out_norm   = getattr(gincfg, "out_norm", "frob")
        if getattr(gincfg, "mode", "gin") == "ipa":
            image = _ipa_blend_3d(
                image,
                n_copies=getattr(gincfg, "ipa_copies", 2),
                control_points=getattr(gincfg, "ipa_control_points", 4),
                n_layer=n_layer, interm_channel=interm,
                scale_pool=scale_pool, out_norm=out_norm,
            )
        else:
            image = _gin_transform_3d(image, n_layer, interm, scale_pool, out_norm)
        image = image.clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # Canonical intensity order (kept identical to the GPU path _batched_intensity):
    #   GIN → bias → brightness/contrast → gamma → inverted-gamma → sharpness
    #        → noise → blur → low-res
    # Physical image-formation chain: appearance warp, multiplicative field, global
    # window transforms, edge/resolution effects, then acquisition noise (before blur
    # so blur correlates it, matching reconstructed-CT noise texture).

    # --- Bias field (smooth multiplicative log-normal field) -------------
    # Right after the appearance warp so the window transforms and degradations
    # act on the field-modulated signal.
    bfcfg = getattr(cfg, "bias_field", None)
    if bfcfg is not None and random.random() < bfcfg.p:
        image = _simulate_bias_field(image, bfcfg.magnitude, getattr(bfcfg, "coarse", 4))

    # --- Brightness / contrast -------------------------------------------
    bccfg = cfg.brightness_contrast
    if random.random() < bccfg.p:
        brightness = random.uniform(-bccfg.brightness, bccfg.brightness)
        contrast   = random.uniform(bccfg.contrast_range[0], bccfg.contrast_range[1])
        if getattr(bccfg, "preserve_range", False):
            # nnUNet ContrastTransform preserve_range=True: clip to per-volume range.
            vol_min, vol_max = image.min(), image.max()
            image = (image * contrast + brightness).clamp_(vol_min, vol_max)
        else:
            image = (image * contrast + brightness).clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # --- Gamma -----------------------------------------------------------
    # Gamma requires [0,1] input: temporarily map z-score → [0,1] → back.
    gcfg = cfg.gamma
    if random.random() < gcfg.p:
        gamma = random.uniform(gcfg.range[0], gcfg.range[1])
        span  = CT_NORM_MAX - CT_NORM_MIN
        if getattr(gcfg, "retain_stats", False):
            # nnUNet p_retain_stats=1: rescale output to match input mean/std.
            mean_in = image.mean()
            std_in  = image.std()
        image = ((image - CT_NORM_MIN) / span).clamp_(0.0, 1.0).pow_(gamma)
        image = image * span + CT_NORM_MIN
        if getattr(gcfg, "retain_stats", False):
            std_out = image.std()
            if std_out > 1e-8:
                image = (image - image.mean()) / std_out * std_in + mean_in
            image = image.clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # --- Inverted gamma (separate darkening pass, gamma forced > 1) ------
    if random.random() < getattr(gcfg, "inverted_p", 0.0):
        gamma = random.uniform(1.0, gcfg.range[1])
        span  = CT_NORM_MAX - CT_NORM_MIN
        image = ((image - CT_NORM_MIN) / span).clamp_(0.0, 1.0).pow_(gamma)
        image = image * span + CT_NORM_MIN

    # --- Sharpness (unsharp masking) -------------------------------------
    scfg = getattr(cfg, "sharpness", None)
    if scfg is not None and random.random() < scfg.p:
        blurred = _separable_gaussian_blur_3d(image, sigma=1.0)
        image   = (image + scfg.factor * (image - blurred)).clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # --- Gaussian noise --------------------------------------------------
    ncfg = cfg.gaussian_noise
    if random.random() < ncfg.p:
        std = random.uniform(0.0, ncfg.max_std)
        image = (image + torch.randn_like(image).mul_(std)).clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # --- Gaussian blur (separable 1-D convolutions) ----------------------
    blcfg = cfg.gaussian_blur
    if random.random() < blcfg.p:
        sigma  = random.uniform(blcfg.sigma_range[0], blcfg.sigma_range[1])
        image  = _separable_gaussian_blur_3d(image, sigma)

    # --- Simulate low resolution -----------------------------------------
    # nnUNet SimulateLowResolutionTransform: downsample then upsample trilinear.
    lrcfg = getattr(cfg, "simulate_low_resolution", None)
    if lrcfg is not None and random.random() < lrcfg.p:
        D, H, W = image.shape[1:]
        scale = random.uniform(lrcfg.scale_min, lrcfg.scale_max)
        small = (max(1, int(D * scale)), max(1, int(H * scale)), max(1, int(W * scale)))
        x = F.interpolate(image.unsqueeze(0), size=small, mode="trilinear", align_corners=False)
        image = F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=False).squeeze(0)

    return image


def _separable_gaussian_blur_3d(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """image: (1, D, H, W).  Applies separable 3-D Gaussian blur in-place-ish."""
    radius = max(1, int(math.ceil(2.0 * sigma)))
    size   = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    k1d    = k1d / k1d.sum()

    x = image.unsqueeze(0)              # (1, 1, D, H, W)
    kd = k1d.view(1, 1, size, 1, 1)
    kh = k1d.view(1, 1, 1, size, 1)
    kw = k1d.view(1, 1, 1, 1, size)
    x = F.conv3d(x, kd, padding=(radius, 0, 0))
    x = F.conv3d(x, kh, padding=(0, radius, 0))
    x = F.conv3d(x, kw, padding=(0, 0, radius))
    return x.squeeze(0)                 # (1, D, H, W)


def _simulate_bias_field(image: torch.Tensor, magnitude: float, coarse: int = 4) -> torch.Tensor:
    """Smooth multiplicative field (log-normal) simulating MRI coil inhomogeneity or CT intensity drift."""
    _, D, H, W = image.shape
    field = torch.randn(1, 1, coarse, coarse, coarse) * magnitude
    field = F.interpolate(field, size=(D, H, W), mode="trilinear", align_corners=False)
    return (image * field.squeeze(0).exp()).clamp_(CT_NORM_MIN, CT_NORM_MAX)


def _gin_transform_3d(
    image: torch.Tensor,          # (1, D, H, W)
    n_layer: int = 4,
    interm_channel: int = 2,
    scale_pool: Tuple[int, ...] = (1, 3),
    out_norm: str = "frob",
) -> torch.Tensor:
    """GIN (Global Intensity Non-linear) appearance transform.

    A stack of `n_layer` conv3d layers with FRESH random kernels + shifts drawn
    each call (no learned state), leaky-relu between layers. The warped image is
    blended with the original via a random alpha, then Frobenius-norm matched to
    the input so overall energy is preserved. Adapted from GINGroupConv3D
    (Ouyang et al., causality-inspired domain generalization) — ported to CPU,
    single-channel, functional. Returns (1, D, H, W)."""
    x_in = image.unsqueeze(0)                      # (1, 1, D, H, W)
    with torch.no_grad():
        x = x_in
        ch_in = 1
        for li in range(n_layer):
            out_ch = 1 if li == n_layer - 1 else interm_channel
            k = scale_pool[random.randrange(len(scale_pool))]
            ker = torch.randn(out_ch, ch_in, k, k, k)
            shift = torch.randn(out_ch, 1, 1, 1)
            x = F.conv3d(x, ker, stride=1, padding=k // 2) + shift
            if li < n_layer - 1:
                x = F.leaky_relu(x)
            ch_in = out_ch                          # (1, 1, D, H, W) after last layer

        alpha = random.random()
        mixed = alpha * x + (1.0 - alpha) * x_in
        if out_norm == "frob":
            in_frob   = torch.norm(x_in.reshape(-1), p="fro")
            self_frob = torch.norm(mixed.reshape(-1), p="fro")
            mixed = mixed * (in_frob / (self_frob + 1e-5))
    return mixed.squeeze(0)                          # (1, D, H, W)


def _ipa_blend_3d(
    image: torch.Tensor,          # (1, D, H, W)
    n_copies: int = 2,
    control_points: int = 4,
    n_layer: int = 4,
    interm_channel: int = 2,
    scale_pool: Tuple[int, ...] = (1, 3),
    out_norm: str = "frob",
) -> torch.Tensor:
    """IPA (Inter-instance Pseudo-correlation Augmentation) — spatial blend.

    Generates `n_copies` independent GIN warps of the SAME volume and mixes them
    with a smooth random spatial field, so different regions take appearances
    from different copies (breaking spurious appearance correlations across the
    volume). Adapted from ginipa.py — the reference's AdvBias field is replaced
    by a coarse control-point field upsampled trilinearly (AdvBias is used there
    purely as a random smooth-field source). Returns (1, D, H, W)."""
    copies = [
        _gin_transform_3d(image, n_layer, interm_channel, scale_pool, out_norm)
        for _ in range(max(2, n_copies))
    ]
    _, D, H, W = image.shape
    cp = max(2, control_points)
    field = torch.randn(1, 1, cp, cp, cp)
    field = F.interpolate(field, size=(D, H, W), mode="trilinear",
                          align_corners=False).squeeze(0)          # (1, D, H, W)
    fmin, fmax = field.amin(), field.amax()
    field = (field - fmin) / (fmax - fmin + 1e-20)                 # → [0, 1]

    out = copies[0]
    for c in copies[1:]:
        out = out * (1.0 - field) + c * field
    return out


def _gaussian_smooth_3d_field(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise separable Gaussian blur for a (C, D, H, W) displacement field."""
    radius = max(1, int(math.ceil(2.0 * sigma)))
    size   = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1d    = torch.exp(-0.5 * (coords / sigma) ** 2)
    k1d    = k1d / k1d.sum()
    C = field.shape[0]
    x  = field.unsqueeze(0)                                          # (1, C, D, H, W)
    kd = k1d.view(1, 1, size, 1, 1).expand(C, 1, size, 1, 1).clone()
    kh = k1d.view(1, 1, 1, size, 1).expand(C, 1, 1, size, 1).clone()
    kw = k1d.view(1, 1, 1, 1, size).expand(C, 1, 1, 1, size).clone()
    x  = F.conv3d(x, kd, padding=(radius, 0, 0), groups=C)
    x  = F.conv3d(x, kh, padding=(0, radius, 0), groups=C)
    x  = F.conv3d(x, kw, padding=(0, 0, radius), groups=C)
    return x.squeeze(0)                                              # (C, D, H, W)


# ---------------------------------------------------------------------------
# Synth augmentation (independent per copy)
# ---------------------------------------------------------------------------

def apply_synth_aug(
    image: torch.Tensor,   # (1, D, H, W) float32
    mask:  torch.Tensor,   # (D, H, W)    int64
    cfg,                   # augmentations.synth config section
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heavy geometric + intensity augmentation sampled independently.
    Call once per copy so K+1 views of the same supervoxel diverge.
    """
    _, D, H, W = image.shape
    mask_mode = getattr(cfg, "mask_interp", "nearest")   # "bilinear" -> soft-mask warp

    # --- Flips (all 3 axes) ----------------------------------------------
    for img_dim, msk_dim, p in [
        (1, 0, cfg.flip_d),
        (2, 1, cfg.flip_h),
        (3, 2, cfg.flip_w),
    ]:
        if random.random() < p:
            image = image.flip(img_dim)
            mask  = mask.flip(msk_dim)

    # --- Affine ----------------------------------------------------------
    acfg = cfg.affine
    if random.random() < acfg.p:
        max_rad = acfg.max_angle_deg * math.pi / 180.0
        rx = random.uniform(-max_rad, max_rad)
        ry = random.uniform(-max_rad, max_rad)
        rz = random.uniform(-max_rad, max_rad)
        scale = random.uniform(acfg.scale_min, acfg.scale_max)
        tx = random.uniform(-acfg.max_translate, acfg.max_translate)
        ty = random.uniform(-acfg.max_translate, acfg.max_translate)
        tz = random.uniform(-acfg.max_translate, acfg.max_translate)
        theta = _make_affine_theta(rx, ry, rz, scale, tx, ty, tz)
        grid  = F.affine_grid(theta, (1, 1, D, H, W), align_corners=False)
        image, mask = _apply_grid(image.unsqueeze(0), mask.unsqueeze(0), grid, mask_mode)
        image = image.squeeze(0)   # (1, D, H, W)
        mask  = mask.squeeze(0)    # (D, H, W)

    # --- Elastic: coarse-grid displacement field (cheap, same smoothness) ---
    ecfg = cfg.elastic
    if random.random() < ecfg.p:
        alpha = random.uniform(*ecfg.alpha_range)   # voxels
        sigma = random.uniform(*ecfg.sigma_range)   # smoothing scale in voxels
        # Generate at 1/sigma resolution and upsample — equivalent smoothness,
        # avoids expensive full-res Gaussian conv3d (kernel size 33–61 for sigma 8–15).
        sd = max(2, round(D / sigma))
        sh = max(2, round(H / sigma))
        sw = max(2, round(W / sigma))
        disp = F.interpolate(
            torch.randn(1, 3, sd, sh, sw),
            size=(D, H, W), mode="trilinear", align_corners=False,
        ).squeeze(0)                                            # (3, D, H, W)
        # normalise to peak=alpha voxels then convert to [-1,1] grid coords
        mx    = disp.abs().amax().clamp(min=1e-6)
        disp  = disp / mx * alpha
        scale_n = torch.tensor([2.0 / D, 2.0 / H, 2.0 / W]).view(3, 1, 1, 1)
        disp_n  = (disp * scale_n).permute(1, 2, 3, 0).unsqueeze(0)  # (1,D,H,W,3)
        theta_id = torch.eye(3, 4).unsqueeze(0)
        base  = F.affine_grid(theta_id, (1, 1, D, H, W), align_corners=False)
        grid  = (base + disp_n).clamp(-1.0, 1.0)
        image, mask = _apply_grid(image.unsqueeze(0), mask.unsqueeze(0), grid, mask_mode)
        image = image.squeeze(0)
        mask  = mask.squeeze(0)

    # --- Diffeomorphic deform (SVF, no folding) --------------------------
    dcfg = getattr(cfg, "deform", None)
    if dcfg is not None and random.random() < dcfg.p:
        phi = _svf_displacement((D, H, W), dcfg.control_points, dcfg.max_disp, dcfg.num_steps)
        base = F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, D, H, W), align_corners=False)
        image, mask = _apply_grid(image.unsqueeze(0), mask.unsqueeze(0),
                                  (base + phi).clamp(-1.0, 1.0), mask_mode)
        image = image.squeeze(0)
        mask  = mask.squeeze(0)

    # --- Intensity: brightness / contrast --------------------------------
    bccfg = cfg.brightness_contrast
    if random.random() < bccfg.p:
        brightness = random.uniform(-bccfg.brightness, bccfg.brightness)
        contrast   = random.uniform(bccfg.contrast_range[0], bccfg.contrast_range[1])
        if getattr(bccfg, "preserve_range", False):
            vol_min, vol_max = image.min(), image.max()
            image = (image * contrast + brightness).clamp_(vol_min, vol_max)
        else:
            image = (image * contrast + brightness).clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # --- Intensity: sharpness (unsharp masking) --------------------------
    scfg = cfg.sharpness
    if random.random() < scfg.p:
        blurred = _separable_gaussian_blur_3d(image, sigma=1.0)
        image   = (image + scfg.factor * (image - blurred)).clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # --- Intensity: Gaussian blur ----------------------------------------
    blcfg = cfg.gaussian_blur
    if random.random() < blcfg.p:
        sigma = random.uniform(blcfg.sigma_range[0], blcfg.sigma_range[1])
        image = _separable_gaussian_blur_3d(image, sigma)

    # --- Intensity: Gaussian noise ---------------------------------------
    ncfg = cfg.gaussian_noise
    if random.random() < ncfg.p:
        mean = random.uniform(ncfg.mean_range[0], ncfg.mean_range[1])
        std  = random.uniform(ncfg.std_range[0],  ncfg.std_range[1])
        image = (image + mean + torch.randn_like(image) * std).clamp_(CT_NORM_MIN, CT_NORM_MAX)

    # --- Gamma -----------------------------------------------------------
    gcfg = getattr(cfg, "gamma", None)
    if gcfg is not None and random.random() < gcfg.p:
        gamma = random.uniform(gcfg.range[0], gcfg.range[1])
        span  = CT_NORM_MAX - CT_NORM_MIN
        image = ((image - CT_NORM_MIN) / span).clamp_(0.0, 1.0).pow_(gamma)
        image = image * span + CT_NORM_MIN

    # --- Simulate low resolution -----------------------------------------
    lrcfg = getattr(cfg, "simulate_low_resolution", None)
    if lrcfg is not None and random.random() < lrcfg.p:
        D, H, W = image.shape[1:]
        scale = random.uniform(lrcfg.scale_min, lrcfg.scale_max)
        small = (max(1, int(D * scale)), max(1, int(H * scale)), max(1, int(W * scale)))
        x = F.interpolate(image.unsqueeze(0), size=small, mode="trilinear", align_corners=False)
        image = F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=False).squeeze(0)

    return image, mask

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
  images : (N, 1, D, H, W)  float32  [0, 1]
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
                grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """grid: (N, D, H, W, 3) in normalised coords [-1, 1]."""
    images = F.grid_sample(images, grid, mode="bilinear",
                           padding_mode="border", align_corners=False)
    masks_f = F.grid_sample(masks.unsqueeze(1).float(), grid, mode="nearest",
                             padding_mode="zeros", align_corners=False)
    return images, masks_f.squeeze(1).long()


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
        images, masks = _apply_grid(images, masks, grid)

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
        images, masks = _apply_grid(images, masks, grid)

    return images, masks


# ---------------------------------------------------------------------------
# Within-task (intensity, independent per volume)
# ---------------------------------------------------------------------------

def apply_intensity_aug(
    image: torch.Tensor,    # (1, D, H, W)  float32  values in [0, 1]
    cfg,                    # cfg.gaussian_noise / .gaussian_blur / .brightness_contrast / .gamma
) -> torch.Tensor:
    """Intensity augmentations sampled independently for each volume."""

    # --- Brightness / contrast -------------------------------------------
    bccfg = cfg.brightness_contrast
    if random.random() < bccfg.p:
        brightness = random.uniform(-bccfg.brightness, bccfg.brightness)
        contrast   = random.uniform(bccfg.contrast_range[0], bccfg.contrast_range[1])
        image = (image * contrast + brightness).clamp_(0.0, 1.0)

    # --- Gamma -----------------------------------------------------------
    gcfg = cfg.gamma
    if random.random() < gcfg.p:
        gamma = random.uniform(gcfg.range[0], gcfg.range[1])
        image = image.clamp(0.0, 1.0).pow_(gamma)

    # --- Gaussian noise --------------------------------------------------
    ncfg = cfg.gaussian_noise
    if random.random() < ncfg.p:
        std = random.uniform(0.0, ncfg.max_std)
        image = (image + torch.randn_like(image).mul_(std)).clamp_(0.0, 1.0)

    # --- Gaussian blur (separable 1-D convolutions) ----------------------
    blcfg = cfg.gaussian_blur
    if random.random() < blcfg.p:
        sigma  = random.uniform(blcfg.sigma_range[0], blcfg.sigma_range[1])
        image  = _separable_gaussian_blur_3d(image, sigma)

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
        theta = _make_affine_theta(rx, ry, rz, scale, tx, ty, tz)   # (1, 3, 4)
        grid  = F.affine_grid(theta, (1, 1, D, H, W), align_corners=False)
        image, mask = _apply_grid(image.unsqueeze(0), mask.unsqueeze(0), grid)
        image = image.squeeze(0)   # (1, D, H, W)
        mask  = mask.squeeze(0)    # (D, H, W)

    # --- Elastic: Gaussian-smoothed displacement field -------------------
    ecfg = cfg.elastic
    if random.random() < ecfg.p:
        alpha = random.uniform(*ecfg.alpha_range)   # voxels
        sigma = random.uniform(*ecfg.sigma_range)   # voxels
        disp  = torch.randn(3, D, H, W)
        disp  = _gaussian_smooth_3d_field(disp, sigma)
        # normalise to peak=alpha voxels then convert to [-1,1] grid coords
        mx    = disp.abs().amax().clamp(min=1e-6)
        disp  = disp / mx * alpha
        scale_n = torch.tensor([2.0 / D, 2.0 / H, 2.0 / W]).view(3, 1, 1, 1)
        disp_n  = (disp * scale_n).permute(1, 2, 3, 0).unsqueeze(0)  # (1,D,H,W,3)
        theta_id = torch.eye(3, 4).unsqueeze(0)
        base  = F.affine_grid(theta_id, (1, 1, D, H, W), align_corners=False)
        grid  = (base + disp_n).clamp(-1.0, 1.0)
        image, mask = _apply_grid(image.unsqueeze(0), mask.unsqueeze(0), grid)
        image = image.squeeze(0)
        mask  = mask.squeeze(0)

    # --- Intensity: brightness / contrast --------------------------------
    bccfg = cfg.brightness_contrast
    if random.random() < bccfg.p:
        brightness = random.uniform(-bccfg.brightness, bccfg.brightness)
        contrast   = random.uniform(bccfg.contrast_range[0], bccfg.contrast_range[1])
        image = (image * contrast + brightness).clamp_(0.0, 1.0)

    # --- Intensity: sharpness (unsharp masking) --------------------------
    scfg = cfg.sharpness
    if random.random() < scfg.p:
        blurred = _separable_gaussian_blur_3d(image, sigma=1.0)
        image   = (image + scfg.factor * (image - blurred)).clamp_(0.0, 1.0)

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
        image = (image + mean + torch.randn_like(image) * std).clamp_(0.0, 1.0)

    return image, mask

"""Efficient batched 3D mask transforms (pure GPU/CPU tensor ops).

Two consumers, kept as two separate entry points:

  * ``goal_mask`` augmentation (``src/gpu_augment.py``) — rewrites a task's TARGET +
    every CONTEXT mask with the same op, redefining the segmentation goal
    ("segment the eroded organ", "segment the boundary", ...).
  * ``perturb_prior_mask`` — degrades ONLY the mask handed to the query mask-token
    prior (``experiments/3d/query_prior.py`` non-cascade, ``cascade._build_query_prior``
    cascade), so an oracle / GT prior looks like a realistic upstream segmentation.

Masks are float in [0, 1], shape ``(N, 1, D, H, W)`` or ``(N, D, H, W)`` (a channel
axis is added/removed transparently). All spatial radii/shifts are given in **mm** and
converted to voxels with the grid pitch (``spacing_mm``): an op is then identical across
cascade levels and invariant to ``data.cascade_crop_jitter`` (jitter moves the crop
centre, not the pitch). Morphology is discrete — radii round to whole voxels; use noise /
shift for sub-voxel effects.

Every randomised helper takes an explicit ``torch.Generator`` and is otherwise
deterministic in its parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "mm_to_vox", "dilate", "erode", "boundary", "sobel_edge", "translate",
    "add_gaussian_noise", "apply_goal_op", "GOAL_OPS", "perturb_prior_mask",
]


# --------------------------------------------------------------------------------------------
# shape + unit helpers
# --------------------------------------------------------------------------------------------

def _to_5d(m: torch.Tensor):
    """(N,D,H,W) | (N,1,D,H,W) -> (N,1,D,H,W) float + a restore fn to the original rank."""
    f = m.float()
    if f.dim() == 4:
        return f.unsqueeze(1), (lambda x: x.squeeze(1))
    if f.dim() == 5:
        return f, (lambda x: x)
    raise ValueError(f"mask must be 4D or 5D, got shape {tuple(m.shape)}")


def mm_to_vox(mm: float, spacing_mm: float) -> int:
    """Physical length (mm) -> whole voxels on a grid of pitch ``spacing_mm`` (>= 0)."""
    if spacing_mm <= 0:
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm}")
    return int(round(float(mm) / float(spacing_mm)))


# --------------------------------------------------------------------------------------------
# morphology (L-inf ball via max-pool by default; true ball via a conv kernel when ball=True)
# --------------------------------------------------------------------------------------------

_BALL_CACHE: dict = {}


def _ball_kernel(radius: int, device, dtype) -> torch.Tensor:
    key = (radius, device, dtype)
    k = _BALL_CACHE.get(key)
    if k is None:
        a = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        d, h, w = torch.meshgrid(a, a, a, indexing="ij")
        k = ((d * d + h * h + w * w) <= radius * radius).to(dtype)[None, None]
        _BALL_CACHE[key] = k
    return k


def dilate(m: torch.Tensor, radius_vox: int, *, ball: bool = False) -> torch.Tensor:
    """Grey-scale dilation by ``radius_vox`` (0 -> no-op). ``ball=False`` = cube/L-inf
    structuring element via ``max_pool3d`` (fastest); ``ball=True`` = Euclidean ball via
    one ``conv3d`` + threshold (isotropic, ~k^3 cost)."""
    r = int(radius_vox)
    if r <= 0:
        return m
    t, restore = _to_5d(m)
    k = 2 * r + 1
    if ball:
        out = (F.conv3d(t, _ball_kernel(r, t.device, t.dtype), padding=r) > 0.5).to(t.dtype)
    else:
        out = F.max_pool3d(t, kernel_size=k, stride=1, padding=r)
    return restore(out.clamp_(0.0, 1.0))


def erode(m: torch.Tensor, radius_vox: int, *, ball: bool = False,
         min_keep: float = 0.0) -> torch.Tensor:
    """Grey-scale erosion — dilation of the complement.

    ``min_keep`` in (0, 1]: never erode a sample below this fraction of its own foreground
    mass. Implemented as `r` one-voxel erosion steps that stop, per sample, before the next
    step would breach the floor — so a small object keeps a core instead of vanishing while
    a large one still gets the full radius. ``min_keep=0`` -> the plain single-shot erosion
    (byte-identical to before)."""
    r = int(radius_vox)
    if r <= 0:
        return m
    t, restore = _to_5d(m)
    if min_keep <= 0.0:
        return restore((1.0 - dilate(1.0 - t, r, ball=ball)).clamp_(0.0, 1.0))
    red = tuple(range(1, t.dim()))
    floor = float(min_keep) * t.sum(dim=red, keepdim=True).clamp_min(1e-6)
    cur = t
    for _ in range(r):
        nxt = (1.0 - dilate(1.0 - cur, 1, ball=ball)).clamp_(0.0, 1.0)
        keep = nxt.sum(dim=red, keepdim=True) >= floor
        cur = torch.where(keep, nxt, cur)
    return restore(cur)


def boundary(m: torch.Tensor, width_vox: int, *, ball: bool = False,
             min_keep: float = 0.0) -> torch.Tensor:
    """Soft surface band = morphological gradient ``dilate(w) - erode(w)`` (>=1 vox each
    side). Turns a filled mask into a shell target. ``min_keep`` is forwarded to the inner
    erosion, so a small object yields a filled-ish core band instead of an empty ring."""
    w = max(1, int(width_vox))
    t, restore = _to_5d(m)
    out = (dilate(t, w, ball=ball) - erode(t, w, ball=ball, min_keep=min_keep)).clamp_(0.0, 1.0)
    return restore(out)


_SOBEL_CACHE: dict = {}


def _sobel_kernels(device, dtype) -> torch.Tensor:
    key = (device, dtype)
    k = _SOBEL_CACHE.get(key)
    if k is None:
        s = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
        d = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
        kd = torch.einsum("i,j,k->ijk", d, s, s)
        kh = torch.einsum("i,j,k->ijk", s, d, s)
        kw = torch.einsum("i,j,k->ijk", s, s, d)
        k = torch.stack([kd, kh, kw])[:, None]           # (3,1,3,3,3)
        _SOBEL_CACHE[key] = k
    return k


def sobel_edge(m: torch.Tensor) -> torch.Tensor:
    """3D Sobel gradient magnitude, per-sample normalised to [0, 1]. On a binary mask this
    is a 1-2 vox soft edge; keeps interior 0 so it reads as an outline target."""
    t, restore = _to_5d(m)
    g = F.conv3d(t, _sobel_kernels(t.device, t.dtype), padding=1)   # (N,3,D,H,W)
    mag = g.pow(2).sum(dim=1, keepdim=True).sqrt()
    peak = mag.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return restore((mag / peak).clamp_(0.0, 1.0))


# --------------------------------------------------------------------------------------------
# geometric + stochastic perturbations
# --------------------------------------------------------------------------------------------

def translate(m: torch.Tensor, shift_vox) -> torch.Tensor:
    """Rigid (sub-voxel) translation. ``shift_vox``: (3,) or (N,3) in (d,h,w) voxels;
    content moves by +shift, out-of-frame filled with 0 (bilinear ``grid_sample``)."""
    t, restore = _to_5d(m)
    N, _, D, H, W = t.shape
    sv = torch.as_tensor(shift_vox, device=t.device, dtype=torch.float32).reshape(-1, 3)
    if sv.shape[0] == 1:
        sv = sv.expand(N, 3)
    if float(sv.abs().max()) == 0.0:
        return m
    theta = torch.zeros(N, 3, 4, device=t.device, dtype=torch.float32)
    theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1.0
    # grid_sample xyz order = (w,h,d); sample at (loc - shift) to move content by +shift.
    size = torch.tensor([W, H, D], device=t.device, dtype=torch.float32)
    theta[:, :, 3] = -2.0 * sv.flip(-1) / size
    grid = F.affine_grid(theta, [N, 1, D, H, W], align_corners=False)
    out = F.grid_sample(t, grid.to(t.dtype), mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return restore(out.clamp_(0.0, 1.0))


def add_gaussian_noise(m: torch.Tensor, std: float, gen: torch.Generator) -> torch.Tensor:
    """Additive per-voxel Gaussian on the soft mask, then clamp to [0, 1]."""
    if std <= 0:
        return m
    t, restore = _to_5d(m)
    noise = torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype) * float(std)
    return restore((t + noise).clamp_(0.0, 1.0))


# --------------------------------------------------------------------------------------------
# entry point 1 — goal-mask op (one op over target + all contexts of a task)
# --------------------------------------------------------------------------------------------

GOAL_OPS = ("dilate", "erode", "boundary", "sobel")


def apply_goal_op(masks: torch.Tensor, op: str, *, radius_vox: int = 0,
                  ball: bool = False, min_keep: float = 0.0) -> torch.Tensor:
    """Apply one goal op to a stack of masks (``(M,D,H,W)`` or ``(M,1,D,H,W)``), returning
    the same shape/rank. ``radius_vox`` is ignored by ``sobel``; ``min_keep`` (fraction of
    per-mask foreground that erosion must leave) applies to ``erode`` / ``boundary``."""
    if op == "dilate":
        return dilate(masks, radius_vox, ball=ball)
    if op == "erode":
        return erode(masks, radius_vox, ball=ball, min_keep=min_keep)
    if op == "boundary":
        return boundary(masks, radius_vox, ball=ball, min_keep=min_keep)
    if op == "sobel":
        return sobel_edge(masks)
    raise ValueError(f"unknown goal op {op!r} (choose from {GOAL_OPS})")


# --------------------------------------------------------------------------------------------
# entry point 2 — query-prior perturbation
# --------------------------------------------------------------------------------------------

def _rng_uniform(gen: torch.Generator, device, lo: float, hi: float) -> float:
    if hi <= lo:
        return float(lo)
    return float(lo + (hi - lo) * torch.rand((), generator=gen, device=device).item())


def _range(cfg, key, default=(0.0, 0.0)):
    v = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    if v is None:
        return (0.0, 0.0)
    if isinstance(v, (int, float)):
        return (0.0, float(v))
    return (float(v[0]), float(v[1]))


@torch.no_grad()
def perturb_prior_mask(prior: torch.Tensor, cfg, spacing_mm: float,
                       gen: torch.Generator) -> torch.Tensor:
    """Degrade a soft prior ``(B,1,D,H,W)`` in [0, 1]. One parameter draw per call, shared
    across the batch. ``cfg`` (dict / DictConfig / None) keys, all optional:

        p          : probability the whole perturbation fires (default 1.0 when cfg given)
        dilate_mm  : [lo, hi] random dilation radius (mm)
        erode_mm   : [lo, hi] random erosion radius (mm)
        shift_mm   : [lo, hi] random rigid translation magnitude (mm), random direction
        noise_std  : [lo, hi] additive-Gaussian sigma on the soft prior
        ball       : bool — Euclidean ball SE for the morphology (default False)
        erode_min_keep : fraction of foreground erosion must leave (default 0.0 = no floor)

    A no-op (returns ``prior`` unchanged) when ``cfg`` is None/empty, ``p`` misses, or every
    range is zero. Dilate and erode are mutually exclusive per call (net radius = the one
    that was drawn larger), so the prior is not just blurred symmetrically."""
    if cfg is None:
        return prior
    p = float(cfg.get("p", 1.0)) if hasattr(cfg, "get") else float(getattr(cfg, "p", 1.0))
    dev = prior.device
    if p < 1.0 and torch.rand((), generator=gen, device=dev).item() >= p:
        return prior
    ball = bool(cfg.get("ball", False)) if hasattr(cfg, "get") else bool(getattr(cfg, "ball", False))
    e_min_keep = (cfg.get("erode_min_keep", 0.0) if hasattr(cfg, "get")
                  else getattr(cfg, "erode_min_keep", 0.0))
    e_min_keep = float(e_min_keep or 0.0)

    d_lo, d_hi = _range(cfg, "dilate_mm")
    e_lo, e_hi = _range(cfg, "erode_mm")
    s_lo, s_hi = _range(cfg, "shift_mm")
    n_lo, n_hi = _range(cfg, "noise_std")

    out = prior
    d_mm = _rng_uniform(gen, dev, d_lo, d_hi)
    e_mm = _rng_uniform(gen, dev, e_lo, e_hi)
    if d_mm >= e_mm and d_mm > 0:
        out = dilate(out, mm_to_vox(d_mm, spacing_mm), ball=ball)
    elif e_mm > 0:
        out = erode(out, mm_to_vox(e_mm, spacing_mm), ball=ball, min_keep=e_min_keep)

    if s_hi > 0:
        mag = _rng_uniform(gen, dev, s_lo, s_hi) / float(spacing_mm)
        if mag > 0:
            v = torch.randn(3, generator=gen, device=dev)
            v = v / v.norm().clamp_min(1e-6) * mag
            out = translate(out, v)

    if n_hi > 0:
        out = add_gaussian_noise(out, _rng_uniform(gen, dev, n_lo, n_hi), gen)

    return out.clamp(0.0, 1.0)

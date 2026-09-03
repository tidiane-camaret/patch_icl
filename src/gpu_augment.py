"""Batched GPU augmentation for 3D in-context segmentation.

Runs in the training loop after batch.to(device), replacing the per-item CPU
augmentation in totalseg_dataloader_incontext. All ops are device/dtype-agnostic
and run under torch.no_grad(). See docs/superpowers/specs/2026-08-15-*.
"""
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from src.augmentations import _make_affine_theta, _svf_displacement
from src.mask_transforms import apply_goal_op, mm_to_vox
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX, DEFAULT_CT_NORM, resolve_ct_norm

REAL, SYNTH, SELF_CONTEXT = 0, 1, 2

from dataclasses import dataclass


@dataclass
class GeoState:
    """Captured geometry of one _geometric() call, for cascade COM inversion + replay.

    grid  : (G, D, H, W, 3) float32 sampling grid (affine+elastic+deform composed,
            grid_sample xyz convention) captured just before grid_sample — one row per
            group (the per-task target), not per volume. None when not captured / no augmentor.
    flips : (G, 3) bool — per-group axis flips applied before the warp (D, H, W order).
    """
    grid: "torch.Tensor | None"
    flips: "torch.Tensor"


def _stack_task(batch: dict) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """batch tensors -> (vols (B*T,1,D,H,W), masks (B*T,D,H,W), B, T). vol b*T+0 = target b."""
    img, ctx = batch["image"], batch["context_in"]          # (B,1,D,H,W),(B,K,1,D,H,W)
    lbl, cout = batch["label"], batch["context_out"]         # (B,D,H,W),(B,K,D,H,W)
    B, K = ctx.shape[0], ctx.shape[1]
    T = K + 1
    D, H, W = img.shape[-3:]
    vols = torch.cat([img.unsqueeze(1), ctx], dim=1).reshape(B * T, 1, D, H, W)
    masks = torch.cat([lbl.unsqueeze(1), cout], dim=1).reshape(B * T, D, H, W)
    # Preserve a soft (partial-volume fraction) target through aug; hard masks stay long
    # so non-soft runs are byte-identical.
    masks = masks.float() if masks.is_floating_point() else masks.long()
    return vols, masks, B, T


def _unstack_task(vols: torch.Tensor, masks: torch.Tensor, B: int, T: int, batch: dict) -> None:
    D, H, W = vols.shape[-3:]
    v = vols.reshape(B, T, 1, D, H, W)
    m = masks.reshape(B, T, D, H, W)
    batch["image"] = v[:, 0]
    batch["context_in"] = v[:, 1:]
    batch["label"] = m[:, 0]
    batch["context_out"] = m[:, 1:]


def _rand(gen, device, *shape):
    return torch.rand(*shape, generator=gen, device=device)


def _uniform(gen, device, lo, hi):
    return (lo + (hi - lo) * torch.rand((), generator=gen, device=device)).item()


def _per_vol_mask(gen, N, device, p):
    """(N,1,1,1,1) bool: which volumes this op fires on."""
    return (torch.rand(N, generator=gen, device=device) < p).view(N, 1, 1, 1, 1)


def _grouped_gaussian_blur(vols, sigma):
    """Separable 3D gaussian blur, same sigma for all volumes in `vols` (N,1,D,H,W)."""
    radius = max(1, int(math.ceil(2.0 * sigma)))
    size = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, dtype=vols.dtype, device=vols.device)
    k = torch.exp(-0.5 * (coords / sigma) ** 2); k = k / k.sum()
    x = vols
    for dim, view in ((2, (1, 1, size, 1, 1)), (3, (1, 1, 1, size, 1)), (4, (1, 1, 1, 1, size))):
        pad = [0, 0, 0]; pad[dim - 2] = radius
        x = F.conv3d(x, k.view(view), padding=(pad[0], pad[1], pad[2]))
    return x


def _gin_once(vols, cfg, gen):
    """One GIN warp per volume via grouped conv (groups=N, fresh random kernels)."""
    N = vols.shape[0]
    device, dtype = vols.device, vols.dtype
    scale_pool = list(getattr(cfg, "scale_pool", (1, 3)))
    n_layer = int(getattr(cfg, "n_layer", 4))
    interm = int(getattr(cfg, "interm_channel", 2))
    x = vols                                          # (N,1,D,H,W)
    ch_in = 1
    for li in range(n_layer):
        out_ch = 1 if li == n_layer - 1 else interm
        k = scale_pool[torch.randint(len(scale_pool), (1,), generator=gen, device=device).item()]
        ker = torch.randn(N * out_ch, ch_in, k, k, k, generator=gen, device=device, dtype=dtype)
        shift = torch.randn(N * out_ch, 1, 1, 1, generator=gen, device=device, dtype=dtype)
        xg = x.reshape(1, N * ch_in, *x.shape[-3:])
        xg = F.conv3d(xg, ker, padding=k // 2, groups=N) + shift.reshape(1, N * out_ch, 1, 1, 1)
        x = xg.reshape(N, out_ch, *x.shape[-3:])
        if li < n_layer - 1:
            x = F.leaky_relu(x)
        ch_in = out_ch
    alpha = torch.rand(N, 1, 1, 1, 1, generator=gen, device=device, dtype=dtype)
    mixed = alpha * x + (1.0 - alpha) * vols
    if getattr(cfg, "out_norm", "frob") == "frob":
        in_f = torch.norm(vols.reshape(N, -1), dim=1).view(N, 1, 1, 1, 1)
        self_f = torch.norm(mixed.reshape(N, -1), dim=1).view(N, 1, 1, 1, 1)
        mixed = mixed * (in_f / (self_f + 1e-5))
    return mixed


def _batched_gin_ipa(vols, cfg, gen, clamp=None):
    """GIN or IPA warp: per-volume grouped conv with optional smooth field blending."""
    if getattr(cfg, "mode", "gin") != "ipa":
        out = _gin_once(vols, cfg, gen)
    else:
        copies = [_gin_once(vols, cfg, gen) for _ in range(max(2, int(getattr(cfg, "ipa_copies", 2))))]
        N, _, D, H, W = vols.shape
        cp = max(2, int(getattr(cfg, "ipa_control_points", 4)))
        field = torch.randn(N, 1, cp, cp, cp, generator=gen, device=vols.device, dtype=vols.dtype)
        field = F.interpolate(field, size=(D, H, W), mode="trilinear", align_corners=False)
        fmin = field.amin(dim=(2, 3, 4), keepdim=True)
        fmax = field.amax(dim=(2, 3, 4), keepdim=True)
        field = (field - fmin) / (fmax - fmin + 1e-20)
        out = copies[0]
        for c in copies[1:]:
            out = out * (1.0 - field) + c * field
    lo, hi = clamp if clamp is not None else (CT_NORM_MIN, CT_NORM_MAX)
    return out.clamp(lo, hi)


def _batched_bias_field(vols, magnitude, coarse, gen, clamp=None):
    """Per-volume smooth multiplicative log-normal field (batched _simulate_bias_field)."""
    N, _, D, H, W = vols.shape
    field = torch.randn(N, 1, coarse, coarse, coarse, generator=gen,
                        device=vols.device, dtype=vols.dtype) * magnitude
    field = F.interpolate(field, size=(D, H, W), mode="trilinear", align_corners=False)
    lo, hi = clamp if clamp is not None else (CT_NORM_MIN, CT_NORM_MAX)
    return (vols * field.exp()).clamp(lo, hi)


def _batched_intensity(vols, cfg, gen, clamp=None):
    # Canonical intensity order — MUST stay identical to the CPU path
    # apply_intensity_aug (src/augmentations.py):
    #   GIN → bias → brightness/contrast → gamma → inverted-gamma → sharpness
    #        → noise → blur → low-res
    N = vols.shape[0]
    device = vols.device
    lo, hi = clamp if clamp is not None else (CT_NORM_MIN, CT_NORM_MAX)
    span = hi - lo

    # 1. GIN / IPA appearance warp
    gin = getattr(cfg, "gin", None)
    if gin is not None and getattr(gin, "p", 0) > 0:
        mask = _per_vol_mask(gen, N, device, gin.p)
        aug = _batched_gin_ipa(vols, gin, gen, clamp=clamp)
        vols = torch.where(mask, aug, vols)

    # 2. Bias field
    bf = getattr(cfg, "bias_field", None)
    if bf is not None and getattr(bf, "p", 0) > 0:
        mask = _per_vol_mask(gen, N, device, bf.p)
        aug = _batched_bias_field(vols, bf.magnitude, int(getattr(bf, "coarse", 4)), gen, clamp=clamp)
        vols = torch.where(mask, aug, vols)

    # 3. Brightness / contrast
    bc = getattr(cfg, "brightness_contrast", None)
    if bc is not None and bc.p > 0:
        mask = _per_vol_mask(gen, N, device, bc.p)
        bright = (-bc.brightness + 2 * bc.brightness *
                  torch.rand(N, 1, 1, 1, 1, generator=gen, device=device))
        c0, c1 = bc.contrast_range
        contrast = c0 + (c1 - c0) * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
        aug = vols * contrast + bright
        if getattr(bc, "preserve_range", False):
            vmin = vols.amin(dim=(1, 2, 3, 4), keepdim=True)
            vmax = vols.amax(dim=(1, 2, 3, 4), keepdim=True)
            aug = torch.minimum(torch.maximum(aug, vmin), vmax)
        else:
            aug = aug.clamp(lo, hi)
        vols = torch.where(mask, aug, vols)

    # 4. Gamma
    gc = getattr(cfg, "gamma", None)
    if gc is not None and gc.p > 0:
        mask = _per_vol_mask(gen, N, device, gc.p)
        g0, g1 = gc.range
        gamma = g0 + (g1 - g0) * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
        aug = ((vols - lo) / span).clamp(0, 1).pow(gamma) * span + lo
        if getattr(gc, "retain_stats", False):
            m_in = vols.mean(dim=(1, 2, 3, 4), keepdim=True)
            s_in = vols.std(dim=(1, 2, 3, 4), keepdim=True)
            m_out = aug.mean(dim=(1, 2, 3, 4), keepdim=True)
            s_out = aug.std(dim=(1, 2, 3, 4), keepdim=True)
            aug = ((aug - m_out) / (s_out + 1e-8) * s_in + m_in).clamp(lo, hi)
        vols = torch.where(mask, aug, vols)

    # 5. Inverted gamma (darkening pass, gamma forced > 1)
    if gc is not None and getattr(gc, "inverted_p", 0.0) > 0:
        mask = _per_vol_mask(gen, N, device, gc.inverted_p)
        g1 = gc.range[1]
        gamma = 1.0 + (g1 - 1.0) * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
        aug = ((vols - lo) / span).clamp(0, 1).pow(gamma) * span + lo
        vols = torch.where(mask, aug, vols)

    # 6. Sharpness (unsharp masking)
    sc = getattr(cfg, "sharpness", None)
    if sc is not None and getattr(sc, "p", 0) > 0:
        mask = _per_vol_mask(gen, N, device, sc.p)
        blur = _grouped_gaussian_blur(vols, sigma=1.0)
        aug = (vols + sc.factor * (vols - blur)).clamp(lo, hi)
        vols = torch.where(mask, aug, vols)

    # 7. Gaussian noise (before blur so blur correlates it)
    nc = getattr(cfg, "gaussian_noise", None)
    if nc is not None and nc.p > 0:
        mask = _per_vol_mask(gen, N, device, nc.p)
        if hasattr(nc, "max_std"):                       # intensity schema
            std = nc.max_std * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
            mean = 0.0
        else:                                            # synth schema
            m0, m1 = nc.mean_range
            s0, s1 = nc.std_range
            mean = m0 + (m1 - m0) * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
            std = s0 + (s1 - s0) * torch.rand(N, 1, 1, 1, 1, generator=gen, device=device)
        noise = mean + std * torch.randn(vols.shape, generator=gen, device=device, dtype=vols.dtype)
        aug = (vols + noise).clamp(lo, hi)
        vols = torch.where(mask, aug, vols)

    # 8. Gaussian blur
    bl = getattr(cfg, "gaussian_blur", None)
    if bl is not None and bl.p > 0:
        mask = _per_vol_mask(gen, N, device, bl.p)
        # NOTE: blur sigma is one draw per batched call (per-volume sigma would need
        # per-volume kernels) — intentional batching simplification.
        s0, s1 = bl.sigma_range
        sigma = _uniform(gen, device, s0, s1)
        aug = _grouped_gaussian_blur(vols, sigma)
        vols = torch.where(mask, aug, vols)

    # 9. Simulate low resolution
    lr = getattr(cfg, "simulate_low_resolution", None)
    if lr is not None and lr.p > 0:
        mask = _per_vol_mask(gen, N, device, lr.p)
        # NOTE: low-res scale is shared per call (per-volume scale isn't batchable —
        # differing output shapes) — intentional batching simplification.
        D, H, W = vols.shape[-3:]
        scale = _uniform(gen, device, lr.scale_min, lr.scale_max)
        small = (max(1, int(D * scale)), max(1, int(H * scale)), max(1, int(W * scale)))
        down = F.interpolate(vols, size=small, mode="trilinear", align_corners=False)
        aug = F.interpolate(down, size=(D, H, W), mode="trilinear", align_corners=False)
        vols = torch.where(mask, aug, vols)

    return vols


def _cfg_get(cfg, key, default=None):
    """Read `key` from a DictConfig (.get) or a plain namespace (getattr) uniformly."""
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _goal_range(cfg, op):
    """[lo, hi] mm for a goal op: `radius_mm` (dilate/erode) or `width_mm` (boundary)."""
    sub = cfg.get(op, None) if hasattr(cfg, "get") else getattr(cfg, op, None)
    key = "width_mm" if op == "boundary" else "radius_mm"
    v = None if sub is None else (sub.get(key, None) if hasattr(sub, "get") else getattr(sub, key, None))
    if v is None:
        return (1.0, 3.0)
    if isinstance(v, (int, float)):
        return (0.0, float(v))
    return (float(v[0]), float(v[1]))


def _goal_mask_transform(masks, group_size, cfg, gen, spacing_mm):
    """Rewrite each task's TARGET + CONTEXT masks with one shared goal op (dilate / erode /
    boundary / sobel), redefining the segmentation goal. `masks` is (N,D,H,W) task-major
    (row g*group_size = task g's target). One parameter draw per BATCH by default; set
    `cfg.per_task=true` for an independent draw per task (G extra pooling calls).

    Radii are mm -> voxels via `spacing_mm`, so the op is identical across cascade levels
    (run_cascade replays the same goal seed) and invariant to data.cascade_crop_jitter.
    No-op (returns `masks` unchanged) when cfg is None, p<=0, or spacing_mm is None."""
    if cfg is None or spacing_mm is None:
        return masks
    p = float(cfg.get("p", 0.0)) if hasattr(cfg, "get") else float(getattr(cfg, "p", 0.0))
    if p <= 0.0:
        return masks
    ops = list(cfg.get("ops", ("dilate", "erode")) if hasattr(cfg, "get")
               else getattr(cfg, "ops", ("dilate", "erode")))
    ball = bool(cfg.get("ball", False)) if hasattr(cfg, "get") else bool(getattr(cfg, "ball", False))
    per_task = bool(cfg.get("per_task", False)) if hasattr(cfg, "get") else bool(getattr(cfg, "per_task", False))
    device = masks.device
    masks = masks.float()                          # goal ops (boundary/sobel) return fractions
    G = masks.shape[0] // group_size

    def _draw_and_apply(chunk):
        if torch.rand((), generator=gen, device=device).item() >= p:
            return chunk
        op = ops[torch.randint(len(ops), (1,), generator=gen, device=device).item()]
        r_vox = 0
        if op != "sobel":
            lo, hi = _goal_range(cfg, op)
            r_mm = lo + (hi - lo) * torch.rand((), generator=gen, device=device).item()
            r_vox = max(1, mm_to_vox(r_mm, spacing_mm))
        # erode/boundary: min_keep floors how much foreground erosion may remove per mask,
        # so a small organ keeps a core instead of vanishing. Read from cfg.<op>.min_keep.
        mk = float(_cfg_get(_cfg_get(cfg, op), "min_keep", 0.0) or 0.0)
        return apply_goal_op(chunk, op, radius_vox=r_vox, ball=ball, min_keep=mk)

    if not per_task:
        return _draw_and_apply(masks)
    for g in range(G):
        sl = slice(g * group_size, (g + 1) * group_size)
        masks[sl] = _draw_and_apply(masks[sl])
    return masks


def _geometric(vols, masks, group_size, cfg, gen, *, capture=False):
    """Shared (group_size=T) or independent (group_size=1) flip/affine/elastic/deform.

    capture=True additionally returns (grid, flips): the composed sampling grid (just
    before grid_sample) and the flip record, one row per group (the per-task target row),
    for cascade COM inversion + replay.
    """
    N = vols.shape[0]
    device = vols.device
    G = N // group_size                              # number of groups
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"
    D, H, W = vols.shape[-3:]

    # --- flips: one decision per group, per axis (ax_i 0,1,2 -> D,H,W) ---
    fp = cfg.flip
    flip_rec = torch.zeros(G, 3, dtype=torch.bool, device=device)
    for g in range(G):
        sl = slice(g * group_size, (g + 1) * group_size)
        for ax_i, (vol_dim, mask_dim, p) in enumerate(
                [(2, 1, fp.p_d), (3, 2, fp.p_h), (4, 3, fp.p_w)]):
            if _rand(gen, device, 1).item() < p:
                vols[sl] = vols[sl].flip(vol_dim)
                masks[sl] = masks[sl].flip(mask_dim)
                flip_rec[g, ax_i] = True

    # --- affine: one theta per group (built with the existing helper) ---
    ac = cfg.affine
    thetas = []
    for g in range(G):
        if _rand(gen, device, 1).item() < ac.p:
            mr = ac.max_angle_deg * math.pi / 180.0
            rx, ry, rz = (_uniform(gen, device, -mr, mr) for _ in range(3))
            scale = _uniform(gen, device, ac.scale_min, ac.scale_max)
            tx, ty, tz = (_uniform(gen, device, -ac.max_translate, ac.max_translate)
                          for _ in range(3))
            thetas.append(_make_affine_theta(rx, ry, rz, scale, tx, ty, tz)[0])
        else:
            thetas.append(torch.eye(3, 4))
    theta = torch.stack(thetas).to(device)                       # (G,3,4)
    theta = theta.repeat_interleave(group_size, dim=0)           # (N,3,4)
    grid = F.affine_grid(theta, vols.shape, align_corners=False)  # (N,D,H,W,3)

    # --- elastic: one coarse displacement field per group, added to grid ---
    ec = getattr(cfg, "elastic", None)
    if ec is not None and ec.p > 0:
        gs = max(int(getattr(ec, "grid_scale", 8)), 2)
        sd, sh, sw = max(D // gs, 2), max(H // gs, 2), max(W // gs, 2)
        for g in range(G):
            if _rand(gen, device, 1).item() < ec.p:
                disp = torch.randn(1, 3, sd, sh, sw, generator=gen, device=device) * ec.alpha
                disp = F.interpolate(disp, size=(D, H, W), mode="trilinear",
                                     align_corners=False).permute(0, 2, 3, 4, 1)  # (1,D,H,W,3)
                sl = slice(g * group_size, (g + 1) * group_size)
                grid[sl] = (grid[sl] + disp).clamp(-1.0, 1.0)

    # --- deform: diffeomorphic SVF per group (guaranteed no folding) ---
    dc = getattr(cfg, "deform", None)
    if dc is not None and getattr(dc, "p", 0) > 0:
        for g in range(G):
            if _rand(gen, device, 1).item() < dc.p:
                phi = _svf_displacement((D, H, W), dc.control_points, dc.max_disp,
                                        dc.num_steps, generator=gen, device=device)
                sl = slice(g * group_size, (g + 1) * group_size)
                grid[sl] = (grid[sl] + phi).clamp(-1.0, 1.0)

    captured_grid = grid.detach().float().clone() if capture else None
    grid = grid.to(vols.dtype)
    vols = F.grid_sample(vols, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)
    # Soft (float) masks: bilinear warp (anti-aliased boundary) when cfg.mask_interp asks
    # for it, else nearest keeps the fractions but blocky. Hard (long) masks: nearest+long
    # exactly as before.
    mask_soft = masks.is_floating_point()
    mask_mode = ("bilinear" if mask_soft and getattr(cfg, "mask_interp", "nearest") == "bilinear"
                 else "nearest")
    m = F.grid_sample(masks.unsqueeze(1).float(), grid, mode=mask_mode,
                      padding_mode="zeros", align_corners=False)
    m = m.squeeze(1)
    out_masks = m.clamp(0.0, 1.0) if mask_soft else m.long()
    if capture:
        # Only the per-task target row (row 0 of each group) is ever read downstream
        # (run_cascade COM inversion); capturing all N rows clones ~384 MiB at exp59 scale.
        return vols, out_masks, captured_grid[::group_size].contiguous(), flip_rec
    return vols, out_masks


class GpuAugmentor:
    def __init__(self, aug_cfg, self_context_per_image: bool = False,
                 self_context_intensity: bool = False, seed: int = 0, ct_norm=None,
                 clamp_frame=None):
        # Intensity ops clamp to CT_NORM_MIN/MAX (the default CT frame) unless an explicit
        # clamp_frame (lo, hi) is given — the seam for a non-CT / multi-modality frame.
        self._clamp = None if clamp_frame is None else (float(clamp_frame[0]), float(clamp_frame[1]))
        if self._clamp is None and resolve_ct_norm(ct_norm) != DEFAULT_CT_NORM:
            raise NotImplementedError(
                "GpuAugmentor is pinned to the default CT frame (fingerprint_1228); "
                f"data.ct_norm={ct_norm!r} needs an explicit clamp_frame=(lo, hi).")
        self.cfg = aug_cfg
        self.self_context_per_image = bool(self_context_per_image)
        self.self_context_intensity = bool(self_context_intensity)
        self._seed = seed
        self._step = 0

    @torch.no_grad()
    def apply(self, batch: dict, *, geo_gen: torch.Generator,
              int_gen: torch.Generator, capture: bool = False,
              goal_gen: torch.Generator | None = None):
        """REAL-mode aug for the cascade runner: shared geometric over target+K contexts
        (geo_gen), then the goal-mask transform on target+contexts (goal_gen, seed held
        constant across cascade levels), then per-volume intensity (int_gen). Mutates
        `batch` in place; returns (batch, GeoState|None). Every task is assumed REAL
        (aug_mode==0) — run_cascade asserts it. Kept separate from __call__ so the
        non-cascade path stays byte-identical.
        """
        cfg = self.cfg
        vols, masks, B, T = _stack_task(batch)          # tensors already on device
        if capture:
            vols, masks, grid, flips = _geometric(
                vols, masks, group_size=T, cfg=cfg.task, gen=geo_gen, capture=True)
        else:
            vols, masks = _geometric(vols, masks, group_size=T, cfg=cfg.task, gen=geo_gen)
            grid = flips = None
        gm_cfg = _cfg_get(cfg, "goal_mask")
        if gm_cfg is not None and goal_gen is not None:
            sp = float(batch["spacing"][0, 0]) if "spacing" in batch else None
            masks = _goal_mask_transform(masks, group_size=T, cfg=gm_cfg, gen=goal_gen,
                                         spacing_mm=sp)
        vols = _batched_intensity(vols, cfg.intensity, int_gen, clamp=self._clamp)
        _unstack_task(vols, masks, B, T, batch)
        return batch, (GeoState(grid=grid, flips=flips) if capture else None)

    @torch.no_grad()
    def __call__(self, batch: dict, training: bool) -> dict:
        cfg = self.cfg
        if (not training) or cfg is None or not getattr(cfg, "enabled", False):
            return batch
        vols, masks, B, T = _stack_task(batch)
        device = vols.device
        gen = torch.Generator(device=device)
        gen.manual_seed(self._seed + self._step)
        self._step += 1

        # goal_mask ops (boundary/sobel) return fractional masks; promote the whole stack to
        # float up-front so the per-mode `masks[vidx] = m` write-backs don't truncate.
        _gm = _cfg_get(cfg, "goal_mask")
        if _gm is not None and float(_cfg_get(_gm, "p", 0.0)) > 0.0 and not masks.is_floating_point():
            masks = masks.float()

        modes = batch.get("aug_mode", torch.zeros(B, dtype=torch.long, device=device))
        modes = modes.to(device)
        for mode in (REAL, SYNTH, SELF_CONTEXT):
            task_sel = (modes == mode).nonzero(as_tuple=True)[0]
            if task_sel.numel() == 0:
                continue
            # volume indices for the selected tasks
            vidx = (task_sel.view(-1, 1) * T + torch.arange(T, device=device)).reshape(-1)
            v, m = vols[vidx], masks[vidx]

            if mode == SYNTH:
                v, m = _geometric(v, m, group_size=1, cfg=cfg.synth, gen=gen)
                v = _batched_intensity(v, cfg.synth, gen, clamp=self._clamp)
            else:  # REAL or SELF_CONTEXT
                v, m = _geometric(v, m, group_size=T, cfg=cfg.task, gen=gen)
                gm_cfg = _cfg_get(cfg, "goal_mask")
                if gm_cfg is not None:
                    sp = float(batch["spacing"][0, 0]) if "spacing" in batch else None
                    m = _goal_mask_transform(m, group_size=T, cfg=gm_cfg, gen=gen, spacing_mm=sp)
                if mode == REAL or self.self_context_intensity:
                    v = _batched_intensity(v, cfg.intensity, gen, clamp=self._clamp)
                if mode == SELF_CONTEXT and self.self_context_per_image:
                    # extra independent geo jitter on the K context volumes only
                    G = task_sel.numel()
                    ctx_local = torch.tensor([g * T + t for g in range(G) for t in range(1, T)],
                                             device=device)
                    cv, cm = _geometric(v[ctx_local], m[ctx_local],
                                        group_size=1, cfg=cfg.per_image, gen=gen)
                    v[ctx_local], m[ctx_local] = cv, cm

            vols[vidx], masks[vidx] = v, m

        _unstack_task(vols, masks, B, T, batch)
        return batch

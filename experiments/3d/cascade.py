"""N-level coarse->fine cascade for PatchSet3D (v2 pipeline).

run_cascade executes one N-level forward (level 0 = GT-centred, level i>0 = target
re-cropped on level i-1's predicted centre-of-mass); shared by the train loop
(experiments/3d/train.py train_epoch) and the v2 cascade val pass (evaluate_cascade).
PatchSet3D.forward stays single-level.

See docs/superpowers/specs/2026-08-30-cascade-training-patchset3d-design.md.
"""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.gpu_augment import GeoState
from src.gpu_realize_crop import realize_native_crops, _regroup
from src.incontext_dataset_v2 import LoadRequest
from src.totalseg_dataset import resolve_ct_norm
from src.totalseg_dataloader_incontext import incontext_collate_fn
from grid_metrics import target_like
from evaluate import _grid_centroid, _predicted_native_center


def invert_geo_center(centroid_dhw, grid_row, flips_row, crop_geom_row, T):
    """Map a centroid in a level's AUGMENTED T^3 grid back to a native crop centre.

    centroid_dhw : length-3 (d,h,w) in the augmented grid, or None (empty prob) -> None.
    grid_row     : (T,T,T,3) float sampling grid (grid_sample xyz convention) or None.
    flips_row    : (3,) bool, per-axis flip (D,H,W order) applied to the volume before the warp.
    crop_geom_row: (4,3) long [starts, crop_sizes, out_sizes, pad_lo].

    _geometric's forward order is flip(vol) THEN grid_sample(vol, grid) with a flip-free
    grid, so the inverse is: interpolate `grid` at the augmented centroid (-> coord in the
    FLIPPED volume), THEN undo the flip, THEN the crop-geom affine. align_corners=False
    throughout, matching _geometric's affine_grid/grid_sample. Identity (grid_row None,
    no flips) reproduces evaluate._predicted_native_center for the same centroid.
    """
    if centroid_dhw is None:
        return None
    g = [float(centroid_dhw[a]) for a in range(3)]                     # d,h,w (augmented out)

    if grid_row is not None:
        def _n(v):                                                    # voxel -> align_corners=False norm
            return (2.0 * v + 1.0) / max(1, T) - 1.0
        q = torch.tensor([[[[[_n(g[2]), _n(g[1]), _n(g[0])]]]]],
                         dtype=torch.float32, device=grid_row.device)  # (x=w,y=h,z=d)
        field = grid_row.detach().float().permute(3, 0, 1, 2).unsqueeze(0)            # (1,3,T,T,T)
        pre = F.grid_sample(field, q, mode="bilinear", padding_mode="border",
                            align_corners=False)[0, :, 0, 0, 0]        # (3,) = (x,y,z) norm value
        x, y, z = (float(v) for v in pre)
        g = [((z + 1.0) * T - 1.0) / 2.0,                             # d  (in the FLIPPED volume)
             ((y + 1.0) * T - 1.0) / 2.0,                             # h
             ((x + 1.0) * T - 1.0) / 2.0]                             # w

    flips = [bool(v) for v in (flips_row.tolist() if torch.is_tensor(flips_row) else flips_row)]
    for a in range(3):
        if flips[a]:
            g[a] = (T - 1) - g[a]                                     # undo flip: flipped[i]==pre[(T-1)-i]

    starts, crop_sizes, out_sizes, pad_lo = (crop_geom_row[r].tolist() for r in range(4))
    native = [int(round(starts[a] + (g[a] - pad_lo[a]) / max(1, out_sizes[a]) * crop_sizes[a]))
              for a in range(3)]
    return tuple(max(0, c) for c in native)


INT_OFFSET = 1_000_000  # keep intensity seeds clear of geo seeds across plausible step counts


@dataclass
class CascadeResult:
    logits: list                  # per level: (B,1,G,G,G)
    targets: list                 # per level: grid GT (target_like)
    geoms: list                   # per level: (B,4,3) target crop_geom
    centers: list                 # len N: centers[0] == [None]*B; centers[i] native COM|None per b
    hard_preds: list | None       # per level: (B,T,T,T) binary; only when want_hard_preds
    empty_frac: float             # fraction of (level>=1, b) COM inversions that hit the fallback
    figure_levels: list | None = None  # per level: {img,gt,ctx_img,ctx_gt} (B,T,T,T) np; only
    #                                    when want_figure_arrays (post-aug target + 1st context)
    prior_modes_used: list | None = None  # len N: [0]=None; [i>0]=query_prior mode this level
    #                                       actually ran (a fixed mode, or the per-step draw)


def _gen(seed_int, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed_int) & 0x7FFF_FFFF_FFFF_FFFF)
    return g


def _centroid_from_logit(logit_b1ghw, T, is_prob):
    """Per-b prob-weighted centroid (d,h,w) in the T^3 grid, or None when empty.

    logit upsampled to T^3 so the crop-geom affine (which assumes a T^3 prob) applies.
    """
    prob = logit_b1ghw.float().clamp(0, 1) if is_prob else torch.sigmoid(logit_b1ghw.float())
    up = F.interpolate(prob, size=(T, T, T), mode="trilinear", align_corners=False)
    out = []
    for b in range(up.shape[0]):
        out.append(_grid_centroid(up[b, 0].detach().cpu().numpy()))   # np(d,h,w) or None
    return out


def _to_device(batch, device):
    for k in ("image", "label", "context_in", "context_out"):
        batch[k] = batch[k].to(device, non_blocking=True)
    return batch


_RECROP_POOL = None
_RECROP_POOL_SIZE = 0
_RECROP_POOL_LOCK = threading.Lock()


def _recrop_pool(workers):
    """Process-lifetime ThreadPoolExecutor for the re-crop fan-out, grown on demand.
    Threads (not processes): provider.load is np.load(mmap) + F.interpolate/avg_pool3d,
    all GIL-releasing, so there is real parallelism and no tensor pickling. Sized once per
    run from data.cascade_recrop_workers and then reused every step."""
    global _RECROP_POOL, _RECROP_POOL_SIZE
    with _RECROP_POOL_LOCK:
        if _RECROP_POOL is None or _RECROP_POOL_SIZE < workers:
            if _RECROP_POOL is not None:
                _RECROP_POOL.shutdown(wait=False)
            _RECROP_POOL = ThreadPoolExecutor(max_workers=workers,
                                              thread_name_prefix="recrop")
            _RECROP_POOL_SIZE = workers
        return _RECROP_POOL


def _run_pool(fn, tasks, workers):
    """Map `fn` over `tasks`, returning results in task order.

    With `workers > 1` (and >1 task) the calls fan out over the process-lifetime re-crop
    thread pool; torch intra-op is pinned to 1 for the duration so the many small
    F.interpolate/avg_pool3d calls don't each spawn cpu_count() threads and oversubscribe
    (safe -- the caller is a hard sync point in the step: GPU idle, nothing else runs).
    Otherwise a plain serial list-comp. Result list is identical either way -- completion
    order is never observable."""
    if workers and workers > 1 and len(tasks) > 1:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            pool = _recrop_pool(min(int(workers), len(tasks)))
            return list(pool.map(fn, tasks))
        finally:
            torch.set_num_threads(n_threads)
    return [fn(t) for t in tasks]


def _recrop_level(provider, batch, centers, spacing, *, step, seed, level, jitter,
                  recrop_workers=1, realize_crop=False, mask_downsample="occupancy",
                  occ_thr=0.1, ct_spec=None, device=None):
    """Build one level-i v2 batch: target re-cropped on `centers[b]`, K contexts GT-centred,
    same subjects/classes as level 0.

    The B*(K+1) provider loads are independent (each carries its own per-(b[,k]) seeded RNG),
    so with recrop_workers > 1 they are fanned out over a thread pool. Results are reassembled
    in task order, so the collated batch is byte-identical to the serial (recrop_workers=1)
    path regardless of completion order.

    realize_crop: load NativeCrop payloads (provider.load_native_crop, from the RAM cache) and
    resample them on `device` via realize_native_crops, instead of the provider.load +
    incontext_collate_fn CPU path. Both branches return the same batch-dict keys
    (image/label/context_in/context_out/spacing/crop_geom + subjects/context_subjects/
    label_names/aug_mode); crop_geom flows through untouched from each row's target crop."""
    subs, ctxs, clss = batch["subjects"], batch["context_subjects"], batch["label_names"]
    sp = float(spacing)

    # Flat load list: per b, the target (k == -1) then its K contexts, in order.
    tasks = []
    for b in range(len(subs)):
        tasks.append((b, -1, subs[b], centers[b], f"{seed}_{step}_{level}_{b}"))
        for k, cs in enumerate(ctxs[b]):
            tasks.append((b, k, cs, None, f"{seed}_{step}_{level}_{b}_{k}"))

    if realize_crop:
        def _load_nc(t):
            b, _k, subj, center, rk = t
            return provider.load_native_crop(subj, clss[b], LoadRequest(
                rng=random.Random(rk), crop_spacing_mm=sp, center=center, jitter=jitter))

        flat = _run_pool(_load_nc, tasks, recrop_workers)
        members = _regroup(flat, len(subs), 1 + len(ctxs[0]))   # [target, ctx0..ctxK-1] per b
        out = realize_native_crops(members, T=batch["image"].shape[-1],
                                   mask_downsample=mask_downsample, occ_thr=occ_thr,
                                   ct_spec=ct_spec, device=device)
        out["subjects"] = list(subs)
        out["context_subjects"] = [list(c) for c in ctxs]
        out["label_names"] = list(clss)
        out["aug_mode"] = torch.zeros(len(subs), dtype=torch.long)
        return out

    def _load(t):
        b, _k, subj, center, rk = t
        return provider.load(subj, clss[b], LoadRequest(
            rng=random.Random(rk), crop_spacing_mm=sp, center=center, jitter=jitter))

    results = _run_pool(_load, tasks, recrop_workers)

    slots = [{"tgt": None, "cin": [], "cout": []} for _ in range(len(subs))]
    for (b, k, *_), r in zip(tasks, results):
        if k == -1:
            slots[b]["tgt"] = r
        else:
            slots[b]["cin"].append(r.image)
            slots[b]["cout"].append(r.label)

    items = []
    for b in range(len(subs)):
        tgt = slots[b]["tgt"]
        items.append({
            "image": tgt.image, "label": tgt.label,
            "context_in": torch.stack(slots[b]["cin"]),
            "context_out": torch.stack(slots[b]["cout"]),
            "subject": subs[b], "context_subjects": list(ctxs[b]),
            "label_name": clss[b], "spacing": tgt.spacing,
            "aug_mode": torch.tensor(0, dtype=torch.long), "crop_geom": tgt.crop_geom,
        })
    return incontext_collate_fn(items)


def _forward_level(model, batch, spacing, query_prior=None):
    sp = float(spacing) if getattr(model, "spacing_aware", False) else None
    kw = {"query_prior": query_prior} if query_prior is not None else {}
    out = model(batch["image"], context_in=batch["context_in"],
                context_out=batch["context_out"], mode="train", spacing=sp, **kw)
    return out["final_logit"].float()


# --------------------------------------------------------------------------------------------
# query_prior placement — resample level i-1's prediction onto level i's augmented grid.
# run_cascade reuses the SAME geo_gen seed at every level, so the augmentation grid (affine +
# flip + deform) is level-INVARIANT: one GeoState serves both ends and only the crop
# (crop_geom, native-CT voxel units) differs. M2 = closed-form affine conjugation, exact for
# affine-only aug (flips + rotation). See results/3d/query_prior_injection/README.md.
# --------------------------------------------------------------------------------------------

def _prior_lattice(T, dev):
    a = torch.arange(T, device=dev, dtype=torch.float32)
    d, h, w = torch.meshgrid(a, a, a, indexing="ij")
    return torch.stack([d, h, w], dim=-1)                     # (T,T,T,3) voxel (d,h,w)


def _fit_grid_affine(grid_row, T, dev, stride=8):
    """3x4 affine A (xyz) with grid ≈ [x,y,z,1] @ A.T, fit on a strided sub-lattice of the
    captured sampling grid (exact for affine-only aug; residual grows with deform/elastic)."""
    idx = torch.arange(0, T, stride, device=dev)
    a = (2.0 * idx.float() + 1.0) / T - 1.0
    d, h, w = torch.meshgrid(a, a, a, indexing="ij")
    base = torch.stack([w, h, d, torch.ones_like(d)], dim=-1).reshape(-1, 4)
    tgt = grid_row[idx][:, idx][:, :, idx].reshape(-1, 3).to(dev)
    return torch.linalg.lstsq(base, tgt).solution.T           # (3,4)


def _crop_compose(g_dst, cg_dst_b, cg_src_b):
    """(...,3) consumer native-crop voxel (d,h,w) -> producer native-crop voxel, through the
    shared native-CT frame. crop_geom row = [starts, crop_sizes, out_sizes, pad_lo]."""
    s1, k1, o1, p1 = (cg_dst_b[r] for r in range(4))
    s0, k0, o0, p0 = (cg_src_b[r] for r in range(4))
    pnt = s1 + (g_dst - p1) / o1 * k1                          # native-CT voxel
    return p0 + (pnt - s0) / k0 * o0                           # producer native-crop voxel


def _warp_prior_m2(vol_src, cg_src, cg_dst, geo, T):
    """M2 warp: vol_src (B,1,T,T,T) on the producer (level i-1) augmented grid -> resampled
    onto the consumer (level i) augmented grid. geo = the shared GeoState (grid + flips).
    Per-b closed-form: aug_dst -> [R g + t] -> flip -> cropgeom compose -> flip -> [R^-1(.-t)]
    -> aug_src, then one grid_sample."""
    B, dev = vol_src.shape[0], vol_src.device
    cg_src = cg_src.to(dev).float()
    cg_dst = cg_dst.to(dev).float()
    base = _prior_lattice(T, dev)
    out = []
    for b in range(B):
        A = _fit_grid_affine(geo.grid[b], T, dev)
        R, t = A[:, :3], A[:, 3]
        fl = [bool(v) for v in geo.flips[b]]
        n = (2.0 * base + 1.0) / T - 1.0                       # consumer output, normalized
        n = (n.flip(-1) @ R.T + t).flip(-1)                    # shared grid affine (fwd)
        for a in range(3):                                     # flip == negate in norm coords
            if fl[a]:
                n[..., a] = -n[..., a]
        g_dst = ((n + 1.0) * T - 1.0) / 2.0                    # consumer native-crop voxel
        g_src = _crop_compose(g_dst, cg_dst[b], cg_src[b])     # producer native-crop voxel
        n0 = (2.0 * g_src + 1.0) / T - 1.0
        for a in range(3):
            if fl[a]:
                n0[..., a] = -n0[..., a]
        n0 = (n0.flip(-1) - t) @ torch.linalg.inv(R).T        # invert shared grid affine
        out.append(F.grid_sample(vol_src[b:b + 1], n0[None], mode="bilinear",
                                 padding_mode="zeros", align_corners=False))
    return torch.cat(out, 0)


def _warp_prior_cropgeom(vol_src, cg_src, cg_dst, T):
    """No-aug prior warp (augmentor is None): crop-geom compose only — M2 with an identity
    grid, exact. Used by the val cascade pass."""
    B, dev = vol_src.shape[0], vol_src.device
    cg_src = cg_src.to(dev).float()
    cg_dst = cg_dst.to(dev).float()
    base = _prior_lattice(T, dev)
    out = []
    for b in range(B):
        g_src = _crop_compose(base, cg_dst[b], cg_src[b])
        n0 = ((2.0 * g_src + 1.0) / T - 1.0).flip(-1)
        out.append(F.grid_sample(vol_src[b:b + 1], n0[None], mode="bilinear",
                                 padding_mode="zeros", align_corners=False))
    return torch.cat(out, 0)


_PRIOR_MODES = ("none", "pred", "gt_coarse", "gt_fine")
_GT_ALIAS = {"gt": "gt_coarse"}                            # `gt` shorthand -> coarse-seg prior


def _prior_mode(query_prior):
    """Normalize data.cascade_query_prior (bool back-compat: False/None -> none, True -> pred)."""
    if query_prior is True:
        return "pred"
    if query_prior is False or query_prior is None:
        return "none"
    m = str(query_prior)
    if m not in _PRIOR_MODES:
        raise ValueError(f"cascade_query_prior={query_prior!r} not in {_PRIOR_MODES}")
    return m


@dataclass(frozen=True)
class PriorSpec:
    """Resolved data.cascade_query_prior. A scalar mode -> a single-mode spec; a
    mapping ``{modes: [...], p: [...], eval_mode: ...}`` -> a categorical mixture drawn
    once per level>0 per training step. `eval_mode` is the deterministic mode used when
    training=False (so val Dice stays comparable across epochs)."""
    modes: tuple
    weights: tuple
    eval_mode: str


def _norm_mode(m):
    """`gt` -> `gt_coarse`; validate against _PRIOR_MODES."""
    m = _GT_ALIAS.get(str(m), str(m))
    if m not in _PRIOR_MODES:
        raise ValueError(f"cascade_query_prior mode {m!r} not in {_PRIOR_MODES} (or 'gt')")
    return m


def _resolve_prior_spec(query_prior) -> PriorSpec:
    """Normalize data.cascade_query_prior into a PriorSpec.

    Accepts the historical scalar (bool | None | 'none'|'pred'|'gt_coarse'|'gt_fine') and a
    mapping form for a per-step random mixture:
        cascade_query_prior:
          modes: [pred, none, gt]     # `gt` == gt_coarse
          p:     [0.4,  0.4,  0.2]    # optional; omitted -> uniform
          eval_mode: pred             # optional; default = pred if in modes, else max-weight
    """
    if query_prior is None or isinstance(query_prior, (bool, str)):
        m = _prior_mode(query_prior)
        return PriorSpec((m,), (1.0,), m)
    # mapping (plain dict or OmegaConf DictConfig) — duck-typed to avoid an omegaconf import.
    if not hasattr(query_prior, "get") or "modes" not in query_prior:
        raise ValueError(
            f"cascade_query_prior={query_prior!r}: expected a mode string "
            f"({'|'.join(_PRIOR_MODES)}), a bool, or a mapping with a 'modes' list.")
    modes = tuple(_norm_mode(x) for x in query_prior["modes"])
    if not modes:
        raise ValueError("cascade_query_prior.modes is empty")
    p = query_prior.get("p", None)
    if p is None:
        weights = tuple(1.0 for _ in modes)
    else:
        weights = tuple(float(x) for x in p)
        if len(weights) != len(modes):
            raise ValueError(f"cascade_query_prior.p (len {len(weights)}) must match "
                             f".modes (len {len(modes)})")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"cascade_query_prior.p={list(weights)} must be non-negative with "
                         f"a positive sum")
    em = query_prior.get("eval_mode", None)
    if em is not None:
        em = _norm_mode(em)
        if em not in modes:
            raise ValueError(f"cascade_query_prior.eval_mode={em!r} must be one of "
                             f".modes {list(modes)}")
    else:
        em = ("pred" if "pred" in modes
              else modes[max(range(len(weights)), key=weights.__getitem__)])
    return PriorSpec(modes, weights, em)


def _build_query_prior(mode, hard, prev_logit, prev_label, cur_label,
                       prev_geo, cg_prev, cg_cur, T, is_prob):
    """Soft prior on level i's grid (B,1,T,T,T) for query_prior, per `mode`:
      pred       sigmoid(logit_{i-1}) detached, warped (M2 / crop-geom) onto level i's grid
      gt_coarse  level i-1's augmented GT warped the same way — perfect-coarse-seg ceiling
      gt_fine    level i's own augmented GT, already on this grid (no warp) — perfect-prior ceiling
    `hard` thresholds the result at 0.5. The pred prior is DETACHED (each level keeps its own
    loss). The geometric warp runs fp32 (autocast off) so the coordinate maths keep sub-voxel
    precision under bf16."""
    B = prev_logit.shape[0]
    with torch.autocast(device_type=prev_logit.device.type, enabled=False):
        if mode == "gt_fine":
            p = cur_label.detach().float().reshape(B, 1, *cur_label.shape[-3:]).clamp(0, 1)
        else:
            if mode == "pred":
                src = prev_logit.detach().float()
                src = src.clamp(0, 1) if is_prob else torch.sigmoid(src)
            else:                                            # gt_coarse
                src = prev_label.detach().float().reshape(B, 1, *prev_label.shape[-3:]).clamp(0, 1)
            if src.shape[-1] != T:
                src = F.interpolate(src, size=(T, T, T), mode="trilinear", align_corners=False)
            if prev_geo is not None and getattr(prev_geo, "grid", None) is not None:
                p = _warp_prior_m2(src, cg_prev, cg_cur, prev_geo, T)
            else:
                p = _warp_prior_cropgeom(src, cg_prev, cg_cur, T)
        return (p >= 0.5).float() if hard else p


def _hard_pred_native(logit_b1ghw, T, is_prob):
    prob = logit_b1ghw.float().clamp(0, 1) if is_prob else torch.sigmoid(logit_b1ghw.float())
    up = F.interpolate(prob, size=(T, T, T), mode="trilinear", align_corners=False)
    return (up >= 0.5).float().squeeze(1)                                 # (B,T,T,T)


def realize_cascade_level0(batch, *, T, mask_downsample, occ_thr, ct_spec, device):
    """native_crop_collate_fn batch -> standard collated batch dict, on `device`.

    Thin wrapper over realize_native_crops: resamples/normalizes/places the level-0
    NativeCrop payloads, then re-attaches the passthrough id lists + aug_mode from
    the collated batch so run_cascade's level-0 handling is byte-identical to the
    finished-tensor collate path (same keys: image/label/context_in/context_out/
    spacing/crop_geom + subjects/context_subjects/label_names/aug_mode)."""
    out = realize_native_crops(batch["native_crop"], T=T, mask_downsample=mask_downsample,
                               occ_thr=occ_thr, ct_spec=ct_spec, device=device)
    out["subjects"] = list(batch["subjects"])
    out["context_subjects"] = [list(c) for c in batch["context_subjects"]]
    out["label_names"] = list(batch["label_names"])
    out["aug_mode"] = batch.get(
        "aug_mode", torch.zeros(len(out["subjects"]), dtype=torch.long)).to(device)
    return out


def run_cascade(model, provider, batch, augmentor, spacings, *, device, training,
                step, seed, jitter=0, is_prob=False, want_hard_preds=False,
                recrop_workers=1, query_prior=False, query_prior_hard=False,
                want_figure_arrays=False, realize_crop=False,
                mask_downsample="occupancy", occ_thr=0.1, ct_spec=None):
    assert len(spacings) >= 2, "cascade needs >=2 spacings"
    assert int(batch["aug_mode"].max()) == 0, "run_cascade: v2 REAL tasks only (aug_mode==0)"
    N = len(spacings)
    T = batch["image"].shape[-1]
    B = batch["image"].shape[0]
    geo_seed = seed * 1_000_003 + step
    prior_spec = _resolve_prior_spec(query_prior)

    logits, targets, geoms = [], [], []
    centers = [[None] * B]
    prior_modes_used = [None]                                 # per level (index 0 = level 0)
    hard = [] if want_hard_preds else None
    figs = [] if want_figure_arrays else None
    empty_hits = empty_total = 0
    prev_logit = prev_geo = prev_label = None                 # for the query_prior warp

    cur = _to_device(dict(batch), device)
    for i in range(N):
        if i > 0:
            cur = _recrop_level(provider, batch, centers[i], spacings[i],
                                step=step, seed=seed, level=i, jitter=jitter,
                                recrop_workers=recrop_workers, realize_crop=realize_crop,
                                mask_downsample=mask_downsample, occ_thr=occ_thr,
                                ct_spec=ct_spec, device=device)
            cur = _to_device(cur, device)   # no-op for realize output (already on device)
        capture = augmentor is not None and i < N - 1
        if augmentor is not None:
            cur, geo = augmentor.apply(
                cur, geo_gen=_gen(geo_seed, device),
                int_gen=_gen(geo_seed + INT_OFFSET * (i + 1), device), capture=capture)
        else:
            geo = None

        prior = None
        if i > 0:
            # One mode per level>0: a fixed spec runs its single mode; a mixture draws once
            # per (seed, step, level) at train time and uses eval_mode at eval time.
            if len(prior_spec.modes) == 1:
                mode_i = prior_spec.modes[0]
            elif training:
                mode_i = random.Random(f"{seed}_{step}_{i}_qprior").choices(
                    prior_spec.modes, weights=prior_spec.weights, k=1)[0]
            else:
                mode_i = prior_spec.eval_mode
            prior_modes_used.append(mode_i)
            if mode_i != "none" and prev_logit is not None:
                cg_cur = cur["crop_geom"] if "crop_geom" in cur else batch["crop_geom"]
                prior = _build_query_prior(mode_i, bool(query_prior_hard), prev_logit,
                                           prev_label, cur["label"], prev_geo, geoms[i - 1],
                                           cg_cur, T, is_prob)

        logit = _forward_level(model, cur, spacings[i], query_prior=prior)
        tgt = target_like(cur["label"].unsqueeze(1).float(), logit)
        logits.append(logit); targets.append(tgt)
        geoms.append(cur["crop_geom"] if "crop_geom" in cur else batch["crop_geom"])
        prev_logit, prev_geo, prev_label = logit, geo, cur["label"]
        if want_hard_preds:
            hard.append(_hard_pred_native(logit, T, is_prob))
        if want_figure_arrays:
            # Post-aug target volume + 1st context, on this level's T³ grid (CPU numpy).
            figs.append({
                "img":     cur["image"][:, 0].detach().float().cpu().numpy(),
                "gt":      cur["label"].detach().float().cpu().numpy(),
                "ctx_img": cur["context_in"][:, 0, 0].detach().float().cpu().numpy(),
                "ctx_gt":  cur["context_out"][:, 0].detach().float().cpu().numpy(),
            })

        if i < N - 1:
            cens = _centroid_from_logit(logit, T, is_prob)
            row = []
            for b in range(B):
                empty_total += 1
                if geo is not None and geo.grid is not None:
                    gr = geo.grid[b]          # per-task target grid, geo.grid is (B, T, T, T, 3)
                    fl = geo.flips[b]         # geo.flips is (B, 3)
                else:
                    gr, fl = None, torch.zeros(3, dtype=torch.bool)
                nc = invert_geo_center(cens[b], gr, fl, geoms[i][b], T)
                if nc is None:
                    empty_hits += 1
                row.append(nc)
            centers.append(row)

    return CascadeResult(logits=logits, targets=targets, geoms=geoms, centers=centers,
                         hard_preds=hard,
                         empty_frac=(empty_hits / empty_total if empty_total else 0.0),
                         figure_levels=figs, prior_modes_used=prior_modes_used)


def evaluate_cascade(model, cfg, classes, *, loader, seed, is_prob,
                     fig_dir=None, cascade_figures=False):
    """v2 cascade val pass. Iterates the level-0 val `loader`, runs the N-level cascade with
    no aug, and returns (rows, cases) shaped like evaluate.evaluate_classes: per class a
    macro stitched-native Dice as `mean_dice` (+ `std_dice`, `mean_time_ms`, `n_samples`),
    plus per-spacing `dice_r{s:g}`; per-case `time_ms` is the wall time of the case's cascade
    batch divided by the batch size (whole-cascade cost, not a single forward).

    cascade_figures=True (needs fig_dir): save one coarse->fine panel per class under
    fig_dir/cascade/ — reuses evaluate.save_cascade_figure via _save_cascade_pair, one panel
    per consecutive spacing pair (so N-1 panels per class for N>2 levels). Captures the first
    sample seen per requested class.
    """
    from collections import defaultdict
    import numpy as np
    from common import _source_root                     # NOTE: _source_root lives in common.py
    from evaluate import _stitched_native_dice_multi, _save_cascade_pair

    spacings = [float(s) for s in cfg.data.cascade_spacings]
    N = len(spacings)
    _, root, _ = _source_root(cfg)
    pg_levels = [dict() for _ in range(N)]
    order = []                                       # (subj,cls) in loader order
    times = {}                                       # (subj,cls) -> per-sample cascade ms
    want_figs = bool(cascade_figures and fig_dir is not None)
    fig_want = set(classes) if want_figs else set()  # classes still needing a panel
    fig_cache = [dict() for _ in range(N)]           # level -> {(subj,cls): arrays} for figures

    model_net = getattr(model, "model", model)
    model_net.eval()
    dev = next(model_net.parameters()).device
    step = 0
    t_cascade = n_seen = 0.0
    for batch in loader:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            res = run_cascade(model, loader.dataset.provider, batch, augmentor=None,
                              spacings=spacings, device=dev,
                              training=False, step=step, seed=seed, jitter=0,
                              is_prob=is_prob, want_hard_preds=True,
                              recrop_workers=int(cfg.data.get("cascade_recrop_workers", 16)),
                              query_prior=cfg.data.get("cascade_query_prior", False),
                              query_prior_hard=bool(cfg.data.get("cascade_query_prior_hard", False)),
                              realize_crop=bool(cfg.data.get("gpu_realize_crop", True)),
                              mask_downsample=cfg.data.get("mask_downsample", "occupancy"),
                              occ_thr=float(cfg.data.get("mask_occupancy_thr", 0.1)),
                              ct_spec=resolve_ct_norm(cfg.data.get("ct_norm")),
                              want_figure_arrays=bool(fig_want))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        step += 1
        subs, clss = batch["subjects"], batch["label_names"]
        dt_ms = (time.perf_counter() - t0) * 1e3
        per_sample_ms = dt_ms / max(len(subs), 1)
        t_cascade += dt_ms; n_seen += len(subs)
        for b in range(len(subs)):
            key = (subs[b], clss[b])
            order.append(key)
            times[key] = round(per_sample_ms, 1)
            for li in range(N):
                hp = res.hard_preds[li][b].cpu().numpy().astype(bool)
                geom = res.geoms[li][b].cpu().numpy()
                pg_levels[li][key] = (np.packbits(hp), tuple(hp.shape), geom)
            if clss[b] in fig_want:                        # first sample of a requested class
                fig_want.discard(clss[b])
                T = res.hard_preds[0].shape[-1]
                for li in range(N):
                    fl = res.figure_levels[li]
                    lg = res.logits[li][b:b + 1].float()
                    prob = lg.clamp(0, 1) if is_prob else torch.sigmoid(lg)
                    prob = F.interpolate(prob, size=(T, T, T), mode="trilinear",
                                         align_corners=False)[0, 0].cpu().numpy()
                    fig_cache[li][key] = {
                        "img": fl["img"][b], "gt": fl["gt"][b],
                        "pred": res.hard_preds[li][b].cpu().numpy(),
                        "prob": prob, "geom": res.geoms[li][b].cpu().numpy(),
                        "ctx_img": fl["ctx_img"][b], "ctx_gt": fl["ctx_gt"][b],
                        "spacing": spacings[li],
                    }

    # Score once after the loop: each key is stitched independently of arrival order, so the
    # full cascade stitch and the per-level (per-resolution) stitches are order-invariant.
    stitched = _stitched_native_dice_multi(pg_levels, root)
    per_res_by_level = [_stitched_native_dice_multi([pg_levels[li]], root) for li in range(N)]

    mean_ms = round(t_cascade / n_seen, 1) if n_seen else float("nan")

    cases_by_class = defaultdict(list)
    for key in order:
        subj, cls = key
        case = {"class": cls, "subject": subj,
                "dice": round(float(stitched.get(key, float("nan"))), 4),
                "time_ms": times.get(key, float("nan"))}
        for li, s in enumerate(spacings):
            case[f"dice_r{s:g}"] = round(float(per_res_by_level[li].get(key, float("nan"))), 4)
        cases_by_class[cls].append(case)

    rows, all_cases = [], []
    for cls in list(classes) + [c for c in cases_by_class if c not in set(classes)]:
        cs = cases_by_class.get(cls, [])
        all_cases.extend(cs)
        if not cs:
            rows.append({"class": cls, "error": "no samples"}); continue
        dvals = [c["dice"] for c in cs if not np.isnan(c["dice"])]
        if not dvals:
            rows.append({"class": cls, "error": "no valid samples"}); continue
        row = {"class": cls,
               "mean_dice": round(sum(dvals) / len(dvals), 4),
               "std_dice": round(float(np.std(dvals)), 4),
               "mean_time_ms": mean_ms,
               "n_samples": len(cs)}
        for s in spacings:
            vals = [c[f"dice_r{s:g}"] for c in cs if not np.isnan(c[f"dice_r{s:g}"])]
            if vals:
                row[f"dice_r{s:g}"] = round(sum(vals) / len(vals), 4)
        rows.append(row)

    if want_figs:
        from pathlib import Path
        out = Path(fig_dir) / "cascade"
        for i in range(N - 1):
            _save_cascade_pair(fig_cache[i], fig_cache[i + 1],
                               spacings[i], spacings[i + 1], out)
        n_cls = len(set(classes)) - len(fig_want)
        print(f"  [cascade-fig] saved {n_cls * (N - 1)} panel(s) -> {out}")

    return rows, all_cases


def _cascade_loss(res: CascadeResult, loss_fn, weights):
    """Sum_i w_i * loss_fn(logit_i, target_i). Returns (total, [per-level floats])."""
    per = [loss_fn(res.logits[i], res.targets[i]) for i in range(len(res.logits))]
    w = list(weights) if weights is not None else [1.0] * len(per)
    total = sum(float(w[i]) * per[i] for i in range(len(per)))
    return total, [float(p.detach()) for p in per]

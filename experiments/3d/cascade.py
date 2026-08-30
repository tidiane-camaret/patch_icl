"""N-level coarse->fine cascade for PatchSet3D (v2 pipeline).

run_cascade executes one N-level forward (level 0 = GT-centred, level i>0 = target
re-cropped on level i-1's predicted centre-of-mass); shared by the train loop
(experiments/3d/train.py train_epoch) and the v2 cascade val pass (evaluate_cascade).
PatchSet3D.forward stays single-level.

See docs/superpowers/specs/2026-08-30-cascade-training-patchset3d-design.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.gpu_augment import GeoState
from src.incontext_dataset_v2 import LoadRequest
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
        q = torch.tensor([[[[[_n(g[2]), _n(g[1]), _n(g[0])]]]]], dtype=torch.float32)  # (x=w,y=h,z=d)
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


def _recrop_level(provider, batch, centers, spacing, *, step, seed, level, jitter):
    """Build one level-i v2 batch: target re-cropped on `centers[b]`, K contexts GT-centred,
    same subjects/classes as level 0. Runs on the calling thread (synchronous provider I/O)."""
    subs, ctxs, clss = batch["subjects"], batch["context_subjects"], batch["label_names"]
    items = []
    for b in range(len(subs)):
        tgt = provider.load(subs[b], clss[b], LoadRequest(
            rng=random.Random(f"{seed}_{step}_{level}_{b}"), crop_spacing_mm=float(spacing),
            center=centers[b], jitter=jitter))
        cin, cout = [], []
        for k, cs in enumerate(ctxs[b]):
            r = provider.load(cs, clss[b], LoadRequest(
                rng=random.Random(f"{seed}_{step}_{level}_{b}_{k}"), crop_spacing_mm=float(spacing),
                center=None, jitter=jitter))
            cin.append(r.image); cout.append(r.label)
        items.append({
            "image": tgt.image, "label": tgt.label,
            "context_in": torch.stack(cin), "context_out": torch.stack(cout),
            "subject": subs[b], "context_subjects": list(ctxs[b]),
            "label_name": clss[b], "spacing": tgt.spacing,
            "aug_mode": torch.tensor(0, dtype=torch.long), "crop_geom": tgt.crop_geom,
        })
    return incontext_collate_fn(items)


def _forward_level(model, batch, spacing):
    sp = float(spacing) if getattr(model, "spacing_aware", False) else None
    out = model(batch["image"], context_in=batch["context_in"],
                context_out=batch["context_out"], mode="train", spacing=sp)
    return out["final_logit"].float()


def _hard_pred_native(logit_b1ghw, T, is_prob):
    prob = logit_b1ghw.float().clamp(0, 1) if is_prob else torch.sigmoid(logit_b1ghw.float())
    up = F.interpolate(prob, size=(T, T, T), mode="trilinear", align_corners=False)
    return (up >= 0.5).float().squeeze(1)                                 # (B,T,T,T)


def run_cascade(model, provider, batch, augmentor, spacings, *, device, training,
                step, seed, jitter=0, is_prob=False, want_hard_preds=False):
    assert len(spacings) >= 2, "cascade needs >=2 spacings"
    assert int(batch["aug_mode"].max()) == 0, "run_cascade: v2 REAL tasks only (aug_mode==0)"
    N = len(spacings)
    T = batch["image"].shape[-1]
    B = batch["image"].shape[0]
    geo_seed = seed * 1_000_003 + step

    logits, targets, geoms = [], [], []
    centers = [[None] * B]
    hard = [] if want_hard_preds else None
    empty_hits = empty_total = 0

    cur = _to_device(dict(batch), device)
    for i in range(N):
        if i > 0:
            cur = _recrop_level(provider, batch, centers[i], spacings[i],
                                step=step, seed=seed, level=i, jitter=jitter)
            cur = _to_device(cur, device)
        capture = augmentor is not None and i < N - 1
        if augmentor is not None:
            cur, geo = augmentor.apply(
                cur, geo_gen=_gen(geo_seed, device),
                int_gen=_gen(geo_seed + INT_OFFSET * (i + 1), device), capture=capture)
        else:
            geo = None

        logit = _forward_level(model, cur, spacings[i])
        tgt = target_like(cur["label"].unsqueeze(1).float(), logit)
        logits.append(logit); targets.append(tgt)
        geoms.append(cur["crop_geom"] if "crop_geom" in cur else batch["crop_geom"])
        if want_hard_preds:
            hard.append(_hard_pred_native(logit, T, is_prob))

        if i < N - 1:
            cens = _centroid_from_logit(logit, T, is_prob)
            row = []
            for b in range(B):
                empty_total += 1
                if geo is not None and geo.grid is not None:
                    gr = geo.grid.view(B, -1, T, T, T, 3)[b, 0]         # target = row 0 of task
                    fl = geo.flips.view(B, -1, 3)[b, 0]
                else:
                    gr, fl = None, torch.zeros(3, dtype=torch.bool)
                nc = invert_geo_center(cens[b], gr, fl, geoms[i][b], T)
                if nc is None:
                    empty_hits += 1
                row.append(nc)
            centers.append(row)

    return CascadeResult(logits=logits, targets=targets, geoms=geoms, centers=centers,
                         hard_preds=hard,
                         empty_frac=(empty_hits / empty_total if empty_total else 0.0))


def evaluate_cascade(model, cfg, classes, *, loader, seed, is_prob):
    """v2 cascade val pass. Iterates the level-0 val `loader`, runs the N-level cascade with
    no aug, and returns (rows, cases) shaped like evaluate.evaluate_classes: per class a
    macro stitched-native Dice as `mean_dice`, plus per-spacing `dice_r{s:g}`.
    """
    from collections import defaultdict
    import numpy as np
    from common import _source_root                     # NOTE: _source_root lives in common.py
    from evaluate import _stitched_native_dice_multi

    spacings = [float(s) for s in cfg.data.cascade_spacings]
    N = len(spacings)
    _, root, _ = _source_root(cfg)
    pg_levels = [dict() for _ in range(N)]
    per_res = defaultdict(dict)                      # (subj,cls) -> {s: native dice}
    order = []                                       # (subj,cls) in loader order

    model_net = getattr(model, "model", model)
    model_net.eval()
    step = 0
    for batch in loader:
        with torch.no_grad():
            res = run_cascade(model, loader.dataset.provider, batch, augmentor=None,
                              spacings=spacings, device=next(model_net.parameters()).device,
                              training=False, step=step, seed=seed, jitter=0,
                              is_prob=is_prob, want_hard_preds=True)
        step += 1
        subs, clss = batch["subjects"], batch["label_names"]
        for b in range(len(subs)):
            key = (subs[b], clss[b])
            order.append(key)
            for li in range(N):
                hp = res.hard_preds[li][b].cpu().numpy().astype(bool)
                geom = res.geoms[li][b].cpu().numpy()
                pg_levels[li][key] = (np.packbits(hp), tuple(hp.shape), geom)
            # per-resolution native dice = single-level stitch (that level alone)
            for li, s in enumerate(spacings):
                dl = _stitched_native_dice_multi([pg_levels[li]], root)
                per_res[key][s] = float(dl.get(key, float("nan")))

    stitched = _stitched_native_dice_multi(pg_levels, root)

    cases_by_class = defaultdict(list)
    for key in order:
        subj, cls = key
        case = {"class": cls, "subject": subj,
                "dice": round(float(stitched.get(key, float("nan"))), 4)}
        for s in spacings:
            case[f"dice_r{s:g}"] = round(per_res[key].get(s, float("nan")), 4)
        cases_by_class[cls].append(case)

    rows, all_cases = [], []
    for cls in list(classes) + [c for c in cases_by_class if c not in set(classes)]:
        cs = cases_by_class.get(cls, [])
        all_cases.extend(cs)
        if not cs:
            rows.append({"class": cls, "error": "no samples"}); continue
        row = {"class": cls,
               "mean_dice": round(sum(c["dice"] for c in cs) / len(cs), 4),
               "n_samples": len(cs)}
        for s in spacings:
            vals = [c[f"dice_r{s:g}"] for c in cs if not np.isnan(c[f"dice_r{s:g}"])]
            if vals:
                row[f"dice_r{s:g}"] = round(sum(vals) / len(vals), 4)
        rows.append(row)
    return rows, all_cases


def _cascade_loss(res: CascadeResult, loss_fn, weights):
    """Sum_i w_i * loss_fn(logit_i, target_i). Returns (total, [per-level floats])."""
    per = [loss_fn(res.logits[i], res.targets[i]) for i in range(len(res.logits))]
    w = list(weights) if weights is not None else [1.0] * len(per)
    total = sum(float(w[i]) * per[i] for i in range(len(per)))
    return total, [float(p.detach()) for p in per]

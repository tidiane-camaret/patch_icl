"""Pure 3D scene composition for omniSynth 3D — the volumetric twin of render.py
(free-placement path only). Bank-free (samplers passed in), so it is unit-testable
with trivial samplers. Objects are pasted by their true contour (ch1 mask), and
anti-overlap operates on mask>0, never the bbox rectangle."""

import numpy as np


def render_scene_3d(rng, canvas, n_objects, k_min, k_max, target_sampler,
                    distractor_sampler, *, tries=1, max_overlap=1.0,
                    background="black", bg_kwargs=None):
    """Compose a free-placement 3D scene. Returns (image (D,H,W) float32,
    mask (D,H,W) float32 binary, k, info). info["target_centroids"] holds each
    target's mask centre-of-mass in [0,1] (z,y,x)."""
    D, H, W = canvas
    n_obj = max(1, int(n_objects))
    k = int(rng.integers(k_min, k_max + 1))
    k = max(1, min(k, n_obj))
    is_target = np.zeros(n_obj, dtype=bool)
    is_target[rng.permutation(n_obj)[:k]] = True

    image = _make_background_3d(D, H, W, background, rng, bg_kwargs)
    mask = np.zeros((D, H, W), dtype=np.float32)
    occ = np.zeros((D, H, W), dtype=bool)
    centroids = []
    target_paints = []
    for i in range(n_obj):
        if is_target[i]:
            res = target_sampler(rng)
            tile = res[0] if isinstance(res, tuple) else res
        else:
            tile = distractor_sampler(rng)
        vol_t, mask_t = _split_3d(tile)
        assert vol_t.shape[0] == vol_t.shape[1] == vol_t.shape[2], (
            f"render_scene_3d expects cubic tiles, got {vol_t.shape}")
        td = vol_t.shape[0]
        cz, cy, cx = _place_random_3d(occ, mask_t, td, D, H, W, rng, tries, max_overlap)
        _occupy_3d(occ, mask_t, cz, cy, cx)
        if is_target[i]:
            target_paints.append((vol_t, mask_t, cz, cy, cx))
            pasted = _paste_3d(mask, mask_t, cz, cy, cx)
            centroids.append(_paste_centroid_3d(pasted, cz, cy, cx, D, H, W))
        else:
            _composite_3d(image, vol_t, mask_t, cz, cy, cx)
    for vol_t, mask_t, cz, cy, cx in target_paints:      # targets over distractors
        _composite_3d(image, vol_t, mask_t, cz, cy, cx)
    return image, mask, k, {"k": k, "target_centroids": centroids}


def _split_3d(tile):
    """(vol, mask) from a [2,T,T,T] rendition or a bare 3D bitmap (vol==mask)."""
    if tile.ndim == 4:
        return tile[0].astype(np.float32), tile[1].astype(np.float32)
    t = tile.astype(np.float32)
    return t, t


def _slices_3d(td, th, tw, cz, cy, cx, D, H, W):
    oz, oy, ox = cz - td // 2, cy - th // 2, cx - tw // 2
    dz0, dy0, dx0 = max(0, oz), max(0, oy), max(0, ox)
    dz1, dy1, dx1 = min(D, oz + td), min(H, oy + th), min(W, ox + tw)
    if dz0 >= dz1 or dy0 >= dy1 or dx0 >= dx1:
        return None
    return ((slice(dz0, dz1), slice(dy0, dy1), slice(dx0, dx1)),
            (slice(dz0 - oz, dz1 - oz), slice(dy0 - oy, dy1 - oy),
             slice(dx0 - ox, dx1 - ox)))


def _composite_3d(canvas, vol_t, mask_t, cz, cy, cx):
    sl = _slices_3d(*vol_t.shape, cz, cy, cx, *canvas.shape)
    if sl is None:
        return
    cs, ts = sl
    canvas[cs] = canvas[cs] * (1.0 - mask_t[ts]) + vol_t[ts] * mask_t[ts]


def _paste_3d(label, mask_t, cz, cy, cx):
    """Union-paste a mask; returns (offset, sub_mask) of the written region or None."""
    sl = _slices_3d(*mask_t.shape, cz, cy, cx, *label.shape)
    if sl is None:
        return None
    cs, ts = sl
    sub = (mask_t[ts] > 0).astype(np.float32)
    np.maximum(label[cs], sub, out=label[cs])
    return cs, sub


def _occupy_3d(occ, mask_t, cz, cy, cx):
    sl = _slices_3d(*mask_t.shape, cz, cy, cx, *occ.shape)
    if sl is None:
        return
    cs, ts = sl
    np.logical_or(occ[cs], mask_t[ts] > 0, out=occ[cs])


def _overlap_frac_3d(occ, mask_t, cz, cy, cx):
    sl = _slices_3d(*mask_t.shape, cz, cy, cx, *occ.shape)
    if sl is None:
        return 1.0
    cs, ts = sl
    m = mask_t[ts] > 0
    tot = int(m.sum())
    if tot == 0:
        return 0.0
    return float(np.logical_and(m, occ[cs]).sum()) / tot


def _clamp_center_3d(cz, cy, cx, t, D, H, W):
    lo = t // 2
    return (min(max(cz, lo), D - (t - lo)),
            min(max(cy, lo), H - (t - lo)),
            min(max(cx, lo), W - (t - lo)))


def _place_random_3d(occ, mask_t, t, D, H, W, rng, tries, max_overlap):
    best, best_ov = None, 2.0
    for _ in range(max(1, tries)):
        cz, cy, cx = _clamp_center_3d(int(rng.integers(0, D)), int(rng.integers(0, H)),
                                      int(rng.integers(0, W)), t, D, H, W)
        if tries <= 1:
            return cz, cy, cx
        ov = _overlap_frac_3d(occ, mask_t, cz, cy, cx)
        if ov < best_ov:
            best, best_ov = (cz, cy, cx), ov
        if ov <= max_overlap:
            return cz, cy, cx
    return best


def _paste_centroid_3d(pasted, cz, cy, cx, D, H, W):
    if pasted is not None:
        (sz, sy, sx), sub = pasted
        zs, ys, xs = np.nonzero(sub)
        if zs.size:
            return ((sz.start + float(zs.mean())) / D,
                    (sy.start + float(ys.mean())) / H,
                    (sx.start + float(xs.mean())) / W)
    return (cz / D, cy / H, cx / W)


def _make_background_3d(D, H, W, background, rng, bg_kwargs):
    """"black" -> zeros (no rng touched). "noise" -> a base grey level + gaussian
    noise, so a dark object painted over it stays visible."""
    if background != "noise":
        return np.zeros((D, H, W), dtype=np.float32)
    kw = bg_kwargs or {}
    lo, hi = kw.get("bg_intensity", (0.2, 0.6))
    img = np.full((D, H, W), float(rng.uniform(lo, hi)), dtype=np.float32)
    noise = kw.get("bg_noise", 0.03)
    if noise > 0:
        img = img + rng.normal(0.0, noise, size=(D, H, W)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)

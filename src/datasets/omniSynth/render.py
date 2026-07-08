"""Pure scene composition + character samplers for omniSynth.

render_scene is bank-free (takes sampler callables) so it is unit-testable with
trivial samplers. The sampler factories encode target_mode:
  identical -> one fixed rendition reused everywhere (chosen via base_rng so it is
               shared across the query + all contexts of an item)
  aug       -> one fixed base rendition + independent affine jitter per placement
  class     -> a fresh random rendition of the target class per placement
"""

import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, zoom as nd_zoom


def render_scene(rng, scene, grid, cell_size, target_sampler, distractor_sampler,
                 background_sampler=None):
    """Compose a grid scene. Returns (image float32 [H,W], mask float32 [H,W], k, info).

    info carries per-scene provenance for logging/analysis:
      - "k": number of target cells
      - "target_cells": sorted list of target cell indices (row-major)
      - "target_transforms": list aligned with target_cells; each entry is the
        affine-jitter params dict (aug mode) or None (identical/class mode).
      - "target_positions": list aligned with target_cells of the real (x, y) glyph
        location = ink centre-of-mass in full-canvas coords, normalised to [0, 1]
        (x rightward, y downward). This reflects the actual post-aug position (cell +
        shift + rotation/scale + glyph asymmetry), unlike the discrete cell index.
    target_sampler may return a bare bitmap or a (bitmap, params) tuple; the latter
    surfaces the per-placement transform. Distractor params are not recorded."""
    n_cells = grid * grid
    max_obj = getattr(scene, "max_nb_objects", 0)
    n_obj = min(n_cells, max_obj) if max_obj else n_cells   # 0 -> no cap
    k = int(rng.integers(scene.k_min, scene.k_max + 1))
    k = max(1, min(k, n_obj))                          # clamp to [1, n_obj]
    cells = rng.permutation(n_cells)
    target_cells = set(cells[:k].tolist())
    filled_cells = set(cells[:n_obj].tolist())        # k targets first, then distractors

    H = W = grid * cell_size
    image = _make_background(H, W, scene, rng, background_sampler)   # black / grey field / image
    mask = np.zeros((H, W), dtype=np.float32)
    random = getattr(scene, "placement", "grid") == "random"
    occ = np.zeros((H, W), dtype=bool)             # union of placed object masks (anti-overlap)
    tries = int(getattr(scene, "placement_tries", 1))
    max_ov = float(getattr(scene, "placement_max_overlap", 1.0))
    sorted_targets, transforms, positions = [], [], []
    target_paints = []                             # (img_t, mask_t, cy, cx) — composited last
    for cell in range(n_cells):
        if cell not in filled_cells:                  # unfilled when max_nb_objects caps count
            continue
        # Sample the object first (so its mask can drive placement), then choose a centre:
        # grid -> cell centre; random -> a uniform draw, optionally rejection-sampled over
        # `tries` candidates to keep the least-overlapping one (cheap anti-overlap). Tiles
        # are cell-sized for margin>=0 but larger for margin<0 (overflow + overlap).
        is_target = cell in target_cells
        if is_target:
            res = target_sampler(rng)
            bm, params = res if isinstance(res, tuple) else (res, None)
        else:
            bm, params = distractor_sampler(rng), None
        img_t, mask_t = _split(bm)                    # glyph: img==mask; medseg: (intensity, mask)
        th = img_t.shape[0]
        if random:
            cy, cx = _place_random(occ, mask_t, th, H, W, rng, tries, max_ov)
        else:
            r, c = divmod(cell, grid)
            cy, cx = _clamp_center(r * cell_size + cell_size // 2,
                                   c * cell_size + cell_size // 2, th, H, W)
        _occupy(occ, mask_t, cy, cx)                  # record footprint for later placements
        if is_target:
            # Defer the image paint so targets land on top of every distractor (a distractor
            # overlapping a target must not overwrite the target's texture).
            target_paints.append((img_t, mask_t, cy, cx))
            pasted = _paste(mask, mask_t, cy, cx)     # union into the label
            sorted_targets.append(cell)
            transforms.append(params)
            positions.append(_paste_centroid(pasted, cy, cx, H, W))
        else:
            _composite(image, img_t, mask_t, cy, cx)  # paint distractor texture over the bg
    for img_t, mask_t, cy, cx in target_paints:       # targets last, over the distractors
        _composite(image, img_t, mask_t, cy, cx)
    info = {"k": k, "target_cells": sorted_targets, "target_transforms": transforms,
            "target_positions": positions}
    return image, mask, k, info


def _make_background(H, W, scene, rng, background_sampler=None):
    """Initial canvas. "black" -> zeros WITHOUT touching rng (keeps existing seeds).
    "image" -> a random real full image from background_sampler (resized to the canvas).
    "random" -> a smooth low-frequency grey field (base level + upsampled random field)
    plus gaussian noise, so a dark object painted over it stays visible."""
    mode = getattr(scene, "background", "black")
    if mode == "image" and background_sampler is not None:
        # Contract: background_sampler returns a canvas-sized (H, W) float image in [0, 1].
        return np.clip(np.asarray(background_sampler(rng), dtype=np.float32), 0.0, 1.0)
    if mode != "random":
        return np.zeros((H, W), dtype=np.float32)
    lo, hi = scene.bg_intensity
    img = np.full((H, W), float(rng.uniform(lo, hi)), dtype=np.float32)
    struct = getattr(scene, "bg_structure", 0.0)
    if struct > 0:
        k = max(2, H // 16)
        low = rng.uniform(-1.0, 1.0, size=(k, k)).astype(np.float32)
        field = nd_zoom(low, (H / k, W / k), order=3)
        field = field[:H, :W]
        if field.shape != (H, W):                    # pad if zoom rounded short
            pad = np.zeros((H, W), dtype=np.float32)
            pad[:field.shape[0], :field.shape[1]] = field
            field = pad
        img = img + struct * field / (np.abs(field).max() + 1e-6)
    noise = getattr(scene, "bg_noise", 0.0)
    if noise > 0:
        img = img + rng.normal(0.0, noise, size=(H, W)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _composite(canvas, img_tile, mask_tile, cy, cx):
    """Paint a premultiplied `img_tile` (intensity already zeroed outside its object)
    over `canvas` under `mask_tile`, clipped and centred at (cy, cx):
        canvas = canvas*(1 - mask) + img_tile
    so the object's real texture (bright OR dark) replaces the background where its
    mask is, instead of np.maximum (which would let a bright background hide it).
    Overlapping objects paint over one another (later wins under the overlap)."""
    th, tw = img_tile.shape
    oy, ox = cy - th // 2, cx - tw // 2
    dy0, dx0 = max(0, oy), max(0, ox)
    dy1, dx1 = min(canvas.shape[0], oy + th), min(canvas.shape[1], ox + tw)
    if dy0 >= dy1 or dx0 >= dx1:
        return
    si = img_tile[dy0 - oy:dy1 - oy, dx0 - ox:dx1 - ox]
    sm = mask_tile[dy0 - oy:dy1 - oy, dx0 - ox:dx1 - ox]
    region = canvas[dy0:dy1, dx0:dx1]
    region[:] = region * (1.0 - sm) + si


def _tile_slices(th, tw, cy, cx, H, W):
    """(canvas_y, canvas_x, tile_y, tile_x) slices for a th×tw tile centred at (cy, cx),
    clipped to an H×W canvas; None if the tile is fully off-canvas."""
    oy, ox = cy - th // 2, cx - tw // 2
    dy0, dx0 = max(0, oy), max(0, ox)
    dy1, dx1 = min(H, oy + th), min(W, ox + tw)
    if dy0 >= dy1 or dx0 >= dx1:
        return None
    return (slice(dy0, dy1), slice(dx0, dx1),
            slice(dy0 - oy, dy1 - oy), slice(dx0 - ox, dx1 - ox))


def _occupy(occ, mask_tile, cy, cx):
    """OR an object's mask footprint into the boolean occupancy canvas `occ`."""
    sl = _tile_slices(mask_tile.shape[0], mask_tile.shape[1], cy, cx, *occ.shape)
    if sl is None:
        return
    cy_s, cx_s, ty_s, tx_s = sl
    np.logical_or(occ[cy_s, cx_s], mask_tile[ty_s, tx_s] > 0, out=occ[cy_s, cx_s])


def _overlap_frac(occ, mask_tile, cy, cx):
    """Fraction of an object's mask that would land on already-occupied pixels."""
    sl = _tile_slices(mask_tile.shape[0], mask_tile.shape[1], cy, cx, *occ.shape)
    if sl is None:
        return 1.0
    cy_s, cx_s, ty_s, tx_s = sl
    m = mask_tile[ty_s, tx_s] > 0
    tot = int(m.sum())
    if tot == 0:
        return 0.0
    return float(np.logical_and(m, occ[cy_s, cx_s]).sum()) / tot


def _place_random(occ, mask_tile, th, H, W, rng, tries, max_overlap):
    """A random (clamped) centre for an object. tries<=1 -> one fully-random draw (no
    rejection); otherwise the least-overlapping of `tries` candidates, accepted early
    once its overlap with already-placed objects is <= max_overlap."""
    best, best_ov = None, 2.0
    for _ in range(max(1, tries)):
        cy, cx = _clamp_center(int(rng.integers(0, H)), int(rng.integers(0, W)), th, H, W)
        if tries <= 1:
            return cy, cx
        ov = _overlap_frac(occ, mask_tile, cy, cx)
        if ov < best_ov:
            best, best_ov = (cy, cx), ov
        if ov <= max_overlap:
            return cy, cx
    return best


def _clamp_center(cy, cx, th, H, W):
    """Nudge a glyph's paste centre inward so its square `th`-sized tile stays fully on the
    HxW canvas — keeps oversized (margin<0) border glyphs whole instead of clipping them at
    the edge. A no-op for cell-sized tiles (margin>=0), which already fit their cell."""
    lo = th // 2
    cy = min(max(cy, lo), H - (th - lo))
    cx = min(max(cx, lo), W - (th - lo))
    return cy, cx


def _paste(canvas, tile, cy, cx):
    """Union-paste (np.maximum) a square `tile` centred at (cy, cx), clipped to `canvas`.

    Overlaps merge instead of overwrite, so order is irrelevant and no glyph erases another.
    Returns (dy0, dx0, sub) of the in-bounds slice actually written (for centroid), or None
    when the tile lands fully off-canvas."""
    th, tw = tile.shape
    oy, ox = cy - th // 2, cx - tw // 2
    dy0, dx0 = max(0, oy), max(0, ox)
    dy1, dx1 = min(canvas.shape[0], oy + th), min(canvas.shape[1], ox + tw)
    if dy0 >= dy1 or dx0 >= dx1:
        return None
    sub = tile[dy0 - oy:dy1 - oy, dx0 - ox:dx1 - ox]
    region = canvas[dy0:dy1, dx0:dx1]
    np.maximum(region, sub, out=region)
    return dy0, dx0, sub


def _paste_centroid(pasted, cy, cx, H, W):
    """Real (x, y) position of a pasted glyph = ink centre-of-mass in full-canvas coords,
    normalised to [0, 1] (x rightward, y downward). Falls back to the cell centre when the
    glyph has no ink or landed off-canvas."""
    if pasted is not None:
        dy0, dx0, sub = pasted
        ys, xs = np.nonzero(sub)
        if ys.size:
            return ((dx0 + float(xs.mean())) / W, (dy0 + float(ys.mean())) / H)
    return (cx / W, cy / H)


def _zoom_to_size(img, scale, size):
    """Zoom by `scale` then center-crop/pad back to (size, size)."""
    z = nd_zoom(img, scale, order=1)
    out = np.zeros((size, size), dtype=img.dtype)
    # center-crop source / center-place into dest
    sy = max(0, (z.shape[0] - size) // 2)
    sx = max(0, (z.shape[1] - size) // 2)
    cropped = z[sy:sy + size, sx:sx + size]
    dy = (size - cropped.shape[0]) // 2
    dx = (size - cropped.shape[1]) // 2
    out[dy:dy + cropped.shape[0], dx:dx + cropped.shape[1]] = cropped
    return out


def _split(tile):
    """(image_tile, mask_tile) for a rendition. A 2D glyph bitmap is its own mask;
    a [2, tile, tile] medseg tile carries (intensity, mask) as separate channels."""
    if tile.ndim == 3:
        return tile[0], tile[1]
    return tile, tile


def affine_jitter(base, scene, rng):
    """Per-placement rotate/scale/translate jitter of a base tile.

    Handles a 2D glyph bitmap (returned binarized uint8) and a [2, tile, tile] medseg
    tile (channel 0 intensity kept continuous + re-masked, channel 1 mask re-binarized).
    Returns (tile, params) where params records the sampled transform: rotation (deg),
    scale (linear factor), and translation as a fraction of cell."""
    cell = base.shape[-1]
    angle = rng.uniform(-scene.aug_rotate, scene.aug_rotate)
    scale = 2.0 ** rng.uniform(-scene.aug_scale, scene.aug_scale)
    dy_frac = rng.uniform(-scene.aug_translate, scene.aug_translate)
    dx_frac = rng.uniform(-scene.aug_translate, scene.aug_translate)

    def _warp(plane):
        p = nd_rotate(plane.astype(np.float32), angle, reshape=False, order=1,
                      mode="constant", cval=0.0)
        p = _zoom_to_size(p, scale, cell)
        return nd_shift(p, (dy_frac * cell, dx_frac * cell), order=1,
                        mode="constant", cval=0.0)

    params = {"rotate": float(angle), "scale": float(scale),
              "dy": float(dy_frac), "dx": float(dx_frac)}
    if base.ndim == 3:                       # [2,tile,tile] = (intensity, mask)
        wm = _warp(base[1])
        m = (wm > 0.5).astype(np.float32)
        if not m.any():                      # tiny mask blurred below 0.5 by the warp ->
            m = (wm > 0).astype(np.float32)  # keep any coverage
        if not m.any():                      # warp pushed the object off-tile -> skip jitter
            return base.astype(np.float32).copy(), params
        return np.stack([_warp(base[0]) * m, m], 0).astype(np.float32), params
    wb = _warp(base)
    out = wb > 0.5
    if not out.any():
        out = wb > 0
    if not out.any():                        # degenerate warp -> keep the un-jittered glyph
        return base.astype(np.uint8).copy(), params
    return out.astype(np.uint8), params


def make_target_sampler(bank, class_id, scene, base_rng, mode=None):
    """`mode` overrides scene.target_mode (used when the scene specifies a mixture
    that the dataset resolves per item). Falls back to scene.target_mode."""
    rends = bank.get(class_id)
    mode = mode if mode is not None else scene.target_mode
    if mode == "class":
        return lambda rng: rends[rng.integers(len(rends))].copy()
    base = rends[base_rng.integers(len(rends))]       # fixed per item (shared across subjects)
    if mode == "identical":
        return lambda rng: base.copy()
    if mode == "aug":
        return lambda rng: affine_jitter(base, scene, rng)
    raise ValueError(f"unknown target_mode {mode!r} (identical | aug | class)")


def make_distractor_sampler(bank, pool, target_class):
    others = [c for c in pool if c != target_class]
    if not others:
        raise ValueError("distractor pool empty after excluding target class")

    def sample(rng):
        cid = others[rng.integers(len(others))]
        rends = bank.get(cid)
        return rends[rng.integers(len(rends))].copy()

    return sample

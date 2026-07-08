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


def render_scene(rng, scene, grid, cell_size, target_sampler, distractor_sampler):
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
    k = int(rng.integers(scene.k_min, scene.k_max + 1))
    k = max(1, min(k, n_cells))                       # clamp to [1, n_cells]
    cells = rng.permutation(n_cells)
    target_cells = set(cells[:k].tolist())

    H = W = grid * cell_size
    image = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    random = getattr(scene, "placement", "grid") == "random"
    sorted_targets, transforms, positions = [], [], []
    for cell in range(n_cells):
        # paste each glyph centred on its cell centre (grid) or a uniform-random canvas
        # position (random; may overlap — union blending keeps every glyph). Tiles are
        # cell-sized for margin>=0 but larger for margin<0 (overflow + overlap).
        if random:
            cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        else:
            r, c = divmod(cell, grid)
            cy, cx = r * cell_size + cell_size // 2, c * cell_size + cell_size // 2
        if cell in target_cells:
            res = target_sampler(rng)
            bm, params = res if isinstance(res, tuple) else (res, None)
            pcy, pcx = _clamp_center(cy, cx, bm.shape[0], H, W)  # keep oversized glyphs whole
            _paste(image, bm, pcy, pcx)
            pasted = _paste(mask, bm, pcy, pcx)
            sorted_targets.append(cell)
            transforms.append(params)
            positions.append(_paste_centroid(pasted, pcy, pcx, H, W))
        else:
            bm = distractor_sampler(rng)
            pcy, pcx = _clamp_center(cy, cx, bm.shape[0], H, W)
            _paste(image, bm, pcy, pcx)
    info = {"k": k, "target_cells": sorted_targets, "target_transforms": transforms,
            "target_positions": positions}
    return image, mask, k, info


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


def affine_jitter(base, scene, rng):
    """Per-placement rotate/scale/translate jitter of a base bitmap.

    Returns (bitmap uint8 {0,1}, params) where params records the sampled transform:
    rotation (deg), scale (linear factor), and translation as a fraction of cell."""
    cell = base.shape[0]
    img = base.astype(np.float32)
    angle = rng.uniform(-scene.aug_rotate, scene.aug_rotate)
    img = nd_rotate(img, angle, reshape=False, order=1, mode="constant", cval=0.0)
    scale = 2.0 ** rng.uniform(-scene.aug_scale, scene.aug_scale)
    img = _zoom_to_size(img, scale, cell)
    dy_frac = rng.uniform(-scene.aug_translate, scene.aug_translate)
    dx_frac = rng.uniform(-scene.aug_translate, scene.aug_translate)
    img = nd_shift(img, (dy_frac * cell, dx_frac * cell), order=1, mode="constant", cval=0.0)
    params = {"rotate": float(angle), "scale": float(scale),
              "dy": float(dy_frac), "dx": float(dx_frac)}
    return (img > 0.5).astype(np.uint8), params


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

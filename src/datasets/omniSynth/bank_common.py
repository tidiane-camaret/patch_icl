"""Shared object-tile construction for the real-image object banks (medseg,
biomedparse). Both turn an (image, binary mask) pair into the omniSynth rendition
format — a [2, tile, tile] float32 tile (ch0 = intensity zeroed outside the mask,
ch1 = binary mask) — with identical canvas-relative / cell sizing. Only the on-disk
loading differs between banks, so that lives in each bank; this is the common core.
"""

import numpy as np
from PIL import Image


def make_object_tile(im_crop, m_crop, *, cell_size, cell_margin, source_size,
                     image_size, size_mode, size_scale):
    """(intensity crop in [0,1], binary mask crop) -> [2, tile, tile] float32 rendition.

    size_mode="canvas": scale by canvas/source so the object keeps its size relative to
    the full canvas (aspect preserved), centred in a square tile. "cell": resize to the
    inner box (inner=(1-2*margin)*cell) so every object is uniformly cell-sized. Intensity
    is zeroed outside the mask so only the object's texture is pasted."""
    bh, bw = m_crop.shape
    if size_mode == "canvas":
        r = (image_size / source_size) * size_scale
        h2 = int(min(image_size, max(2, round(bh * r))))
        w2 = int(min(image_size, max(2, round(bw * r))))
        tile = max(h2, w2)
        off_y, off_x = (tile - h2) // 2, (tile - w2) // 2
    else:                                        # cell: uniform inner-box (glyph-like)
        inner = max(1, int(round(cell_size * (1.0 - 2.0 * cell_margin))))
        h2 = w2 = inner
        tile = max(cell_size, inner)
        off_y = off_x = (tile - inner) // 2

    m_res = np.asarray(Image.fromarray((m_crop * 255).astype(np.uint8))
                       .resize((w2, h2), Image.BILINEAR))
    m_r = m_res >= 128
    if not m_r.any():                            # tiny/sparse mask blurred below 0.5 under
        m_r = m_res > 0                          # downsizing -> keep any coverage (stay non-empty)
    im_r = np.asarray(Image.fromarray((im_crop * 255).astype(np.uint8))
                      .resize((w2, h2), Image.BILINEAR)).astype(np.float32) / 255.0
    im_r = im_r * m_r                            # keep texture only under the mask
    out = np.zeros((2, tile, tile), dtype=np.float32)
    out[0, off_y:off_y + h2, off_x:off_x + w2] = im_r
    out[1, off_y:off_y + h2, off_x:off_x + w2] = m_r.astype(np.float32)
    return out


def crop_to_tile(image01, mask_bool, min_px, **sizing):
    """Bbox-crop an object from a full (image01, boolean mask) and build its tile.
    Returns None when the mask is smaller than `min_px` (near-empty / degenerate)."""
    if int(mask_bool.sum()) < min_px:
        return None
    ys, xs = np.nonzero(mask_bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tile = make_object_tile(image01[y0:y1, x0:x1], mask_bool[y0:y1, x0:x1], **sizing)
    if tile[1].sum() == 0:                       # mask fully vanished under resize -> drop
        return None
    return tile

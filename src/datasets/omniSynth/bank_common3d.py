"""3D tile builders for the TotalSegmentator object bank — the volumetric twin of
bank_common.py. Turn an (intensity, mask) volume crop into a [2, T, T, T] fp16
rendition (ch0 = intensity zeroed outside the mask, ch1 = binary mask), centered in
a cube. Pure functions, no I/O."""

import numpy as np
from scipy.ndimage import zoom as nd_zoom


def make_object_tile_3d(vol_crop, m_crop, *, source_size, image_size,
                        size_scale=1.0, min_tile=2):
    """(intensity crop [0,1] (d,h,w), bool mask (d,h,w)) -> [2,T,T,T] float16.

    Scales by r = (image_size/source_size)*size_scale so the object keeps its size
    relative to the canvas (aspect preserved), then centers it in a cube of side
    T = max(scaled dims). Intensity is zeroed outside the mask."""
    r = (float(image_size) / float(source_size)) * float(size_scale)
    d, h, w = m_crop.shape
    d2 = int(min(image_size, max(min_tile, round(d * r))))
    h2 = int(min(image_size, max(min_tile, round(h * r))))
    w2 = int(min(image_size, max(min_tile, round(w * r))))

    if (d2, h2, w2) != (d, h, w):
        zf = (d2 / d, h2 / h, w2 / w)
        m_res = nd_zoom(m_crop.astype(np.float32), zf, order=1)
        v_res = nd_zoom(vol_crop.astype(np.float32), zf, order=1)
    else:
        m_res = m_crop.astype(np.float32)
        v_res = vol_crop.astype(np.float32)

    mb = m_res >= 0.5
    if not mb.any():                      # thin mask blurred below 0.5 under resize
        mb = m_res > 0                    # keep any coverage (stay non-empty)
    v_res = np.clip(v_res, 0.0, 1.0) * mb

    tile = max(d2, h2, w2)
    off = ((tile - d2) // 2, (tile - h2) // 2, (tile - w2) // 2)
    out = np.zeros((2, tile, tile, tile), dtype=np.float16)
    sl = tuple(slice(o, o + s) for o, s in zip(off, (d2, h2, w2)))
    out[(0,) + sl] = v_res.astype(np.float16)
    out[(1,) + sl] = mb.astype(np.float16)
    return out


def crop_to_tile_3d(vol01, mask_bool, min_vox, **sizing):
    """Bbox-crop an organ from a full (vol01, bool mask) and build its tile.
    Returns None when the mask is smaller than `min_vox` or vanishes under resize."""
    if int(mask_bool.sum()) < min_vox:
        return None
    zs, ys, xs = np.nonzero(mask_bool)
    z0, z1 = zs.min(), zs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    tile = make_object_tile_3d(vol01[z0:z1, y0:y1, x0:x1],
                               mask_bool[z0:z1, y0:y1, x0:x1], **sizing)
    if tile[1].sum() == 0:
        return None
    return tile

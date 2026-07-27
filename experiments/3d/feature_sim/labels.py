# experiments/3d/feature_sim/labels.py
"""FG/BG labeling for feature-similarity: R'^3 occupancy grids (dense) and
native-res point sampling (point mode). Coords are grid_sample-ready."""
import torch
import torch.nn.functional as F

from src.models.patchset3d import _down_to


def grid_labels(mask, res, threshold=0.5):
    """Downsample `mask` (D,H,W) or (1,D,H,W) to an occupancy grid at res^3.

    threshold=None returns the raw occupancy FRACTION in [0,1] (soft label, matching the
    model's soft-Dice target); a float returns the binary `occupancy >= threshold` grid.
    Coarse cells pool many voxels, so thin structures rarely reach 0.5 — use the soft
    fraction for dense metrics so they don't collapse to an all-background (nan) target."""
    m = mask.float()
    if m.dim() == 3:
        m = m.unsqueeze(0)                    # (1,D,H,W)
    occ = _down_to(m.unsqueeze(0), res).squeeze(0).squeeze(0)   # (res,res,res) fraction
    return occ if threshold is None else (occ >= threshold).float()


def _to_norm_coords(idx, shape):
    """Voxel indices (N,3) in (d,h,w) -> normalized [-1,1] coords, same axis order."""
    dims = torch.tensor(shape, dtype=torch.float)
    return (idx.float() / (dims - 1)) * 2 - 1


def _dilate(mask, band):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    d = F.max_pool3d(m, kernel_size=2 * band + 1, stride=1, padding=band)
    return d.squeeze(0).squeeze(0) > 0


def _pick(coords_pool, n, generator):
    k = coords_pool.shape[0]
    if k == 0:
        return coords_pool.new_zeros((0, 3))
    replace = k < n
    sel = torch.randint(k, (n,), generator=generator) if replace \
        else torch.randperm(k, generator=generator)[:n]
    return coords_pool[sel]


def sample_points(mask, n_fg, n_bg, band=None, generator=None):
    m = mask
    fg_idx = torch.nonzero(m > 0, as_tuple=False)
    bg_mask = (m == 0)
    if band is not None:
        bg_mask = bg_mask & _dilate(m > 0, band)
    bg_idx = torch.nonzero(bg_mask, as_tuple=False)
    fg = _pick(fg_idx, n_fg, generator)
    bg = _pick(bg_idx, n_bg, generator)
    idx = torch.cat([fg, bg], dim=0)
    labels = torch.cat([torch.ones(fg.shape[0]), torch.zeros(bg.shape[0])])
    return _to_norm_coords(idx, m.shape), labels

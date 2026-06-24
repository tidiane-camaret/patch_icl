"""
Square-bbox crop / window ops for the zoom-refinement chain.

Origins are top-left (row, col) integer corners of an s×s square that always fits inside
the H×W frame. crop_resize / composite_window are batched (per-sample origin) via
F.grid_sample so the whole batch of context+target crops is handled in one call.
"""

import torch
import torch.nn.functional as F


def _box_sum(x, s):
    """(B,1,H,W) → (B,1,H-s+1,W-s+1) summed value of every s×s window (stride 1)."""
    return F.avg_pool2d(x, kernel_size=s, stride=1) * (s * s)


def _argmax_origin(score):
    """(B,1,Hs,Ws) window scores → (B,2) top-left (row,col) of the max window."""
    B = score.shape[0]
    Ws = score.shape[-1]
    flat = score.reshape(B, -1).argmax(dim=1)
    return torch.stack([torch.div(flat, Ws, rounding_mode="floor"), flat % Ws], dim=1)


def _window_origin(score, s, H, W, eps=0.5):
    """(B,1,Hs,Ws) scores → (B,2) argmax top-left, per sample. For samples whose densest
    window holds essentially no mass (max <= eps) — e.g. an empty prediction — the argmax
    would degenerate to the corner (0,0); instead center the crop. eps=0.5 means "less than
    half a cell of probability/foreground in the best window" (a binary fg cell scores >=1)."""
    B = score.shape[0]
    origin = _argmax_origin(score)
    maxval = score.reshape(B, -1).amax(dim=1)                       # (B,)
    center = origin.new_tensor([(H - s) // 2, (W - s) // 2])        # (2,)
    return torch.where((maxval <= eps).unsqueeze(1), center, origin)


def max_sum_window(prob, s):
    """Top-left (B,2) of the s×s square with the largest summed value in `prob`.
    Empty maps (no mass) center the crop rather than collapsing to the corner."""
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    prob = prob.float()
    H, W = prob.shape[-2:]
    return _window_origin(_box_sum(prob, s), s, H, W)


def gt_window(mask, s):
    """Top-left (B,2) of the s×s square with the most foreground in `mask`.
    Empty masks center the crop rather than collapsing to the corner."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    H, W = mask.shape[-2:]
    return _window_origin(_box_sum(mask, s), s, H, W)


def crop_resize(x, origin, s, out, mode="bilinear"):
    """Crop each (N,C,H,W) image to its s×s bbox at `origin` (N,2) and resample to out×out.

    Uses F.grid_sample with align_corners=False (pixel-center convention)."""
    N, C, H, W = x.shape
    r0 = origin[:, 0].to(x.dtype).view(N, 1)
    c0 = origin[:, 1].to(x.dtype).view(N, 1)
    i = torch.arange(out, device=x.device, dtype=x.dtype) + 0.5     # (out,) cell centers
    # source pixel coords of each output cell, then normalize to [-1,1] (align_corners=False)
    rows = r0 + i.view(1, out) * (s / out)                          # (N,out)
    cols = c0 + i.view(1, out) * (s / out)                          # (N,out)
    ny = 2.0 * rows / H - 1.0                                       # (N,out)
    nx = 2.0 * cols / W - 1.0
    grid = torch.stack([nx.view(N, 1, out).expand(N, out, out),
                        ny.view(N, out, 1).expand(N, out, out)], dim=-1)   # (N,out,out,2)
    return F.grid_sample(x, grid, mode=mode, align_corners=False, padding_mode="border")


def composite_window(full, patch, origin, s):
    """Write patch (B,1,s,s) into a clone of full (B,1,H,W) at each origin (B,2). New tensor."""
    B = full.shape[0]
    out = full.clone()
    for b in range(B):                       # per-sample origin; B is small (batch dim)
        r0, c0 = int(origin[b, 0]), int(origin[b, 1])
        out[b, 0, r0:r0 + s, c0:c0 + s] = patch[b, 0]
    return out

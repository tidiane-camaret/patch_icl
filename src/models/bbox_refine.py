"""Square-bbox crop / fuse ops for PatchSetCNN's bbox-zoom refinement.

Origins are top-left (row, col) integer corners of an s×s square that always fits
inside the H×W frame. crop_resize is batched (per-sample origin) via F.grid_sample.
fuse_window ADDS a patch into the window (logit-space residual fusion), unlike a
hard-replace composite. Adapted from experiments/2d/multilevel/bbox.py.
"""

import torch
import torch.nn.functional as F


def _box_sum(x, s):
    """(B,1,H,W) → (B,1,H-s+1,W-s+1): summed value of every s×s window (stride 1)."""
    return F.avg_pool2d(x, kernel_size=s, stride=1) * (s * s)


def _window_origin(score, s, H, W, eps=0.5):
    """(B,1,Hs,Ws) window scores → (B,2) argmax top-left per sample. Windows holding
    essentially no mass (max ≤ eps, e.g. an empty prediction) center the crop instead of
    collapsing to the (0,0) corner. eps=0.5 ≈ 'less than half a cell of mass'."""
    B, Ws = score.shape[0], score.shape[-1]
    flat = score.reshape(B, -1)
    idx = flat.argmax(dim=1)
    origin = torch.stack([torch.div(idx, Ws, rounding_mode="floor"), idx % Ws], dim=1)
    center = origin.new_tensor([(H - s) // 2, (W - s) // 2])
    return torch.where((flat.amax(dim=1) <= eps).unsqueeze(1), center, origin)


def max_sum_window(prob, s):
    """Top-left (B,2) of the s×s square with the largest summed value in `prob`
    ((B,1,H,W) or (B,H,W)). Empty maps center the crop."""
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    prob = prob.float()
    H, W = prob.shape[-2:]
    return _window_origin(_box_sum(prob, s), s, H, W)


def gt_window(mask, s):
    """Top-left (B,2) of the s×s square with the most foreground in `mask`
    ((B,1,H,W) or (B,H,W)). Empty masks center the crop."""
    return max_sum_window(mask, s)


def crop_resize(x, origin, s, out, mode="bilinear"):
    """Crop each (N,C,H,W) image to its s×s bbox at `origin` (N,2) and resample to
    out×out via F.grid_sample (align_corners=False, border padding)."""
    N, C, H, W = x.shape
    r0 = origin[:, 0].to(x.dtype).view(N, 1)
    c0 = origin[:, 1].to(x.dtype).view(N, 1)
    i = torch.arange(out, device=x.device, dtype=x.dtype) + 0.5      # cell centers
    rows = r0 + i.view(1, out) * (s / out)
    cols = c0 + i.view(1, out) * (s / out)
    ny = 2.0 * rows / H - 1.0
    nx = 2.0 * cols / W - 1.0
    grid = torch.stack([nx.view(N, 1, out).expand(N, out, out),
                        ny.view(N, out, 1).expand(N, out, out)], dim=-1)
    return F.grid_sample(x, grid, mode=mode, align_corners=False, padding_mode="border")


def fuse_window(full, patch, origin, s):
    """Return a clone of full (B,1,H,W) with patch (B,1,s,s) ADDED into the s×s window at
    each origin (B,2). Additive (logit-space) fusion; input not mutated. Per-sample loop
    (B is the small batch dim)."""
    out = full.clone()
    for b in range(full.shape[0]):
        r0, c0 = int(origin[b, 0]), int(origin[b, 1])
        out[b, 0, r0:r0 + s, c0:c0 + s] += patch[b, 0]
    return out

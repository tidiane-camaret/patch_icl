"""Regression test for the save_eval_figure overlay-transparency bug (docs/logs.md 2026-09-12).

Found evaluating ISLES22 DWI: a bare `imshow(mask, cmap="Reds", alpha=...)` colors EVERY
pixel through the colormap (0 -> a near-white "Reds" color, not transparent), so background
pixels get tinted too -- invisible on CT's brighter background but glaringly obvious on
MRI's min-max-crushed dynamic range (the un-tinted "target" panel rendered visibly black next
to the falsely-brightened GT/pred panels showing the SAME background image). Fixed by routing
save_eval_figure's overlays through the already-correct `_overlay` helper (alpha-masked RGBA,
used elsewhere for the cascade figures).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import _overlay, save_eval_figure


def test_overlay_is_transparent_outside_the_mask():
    """`_overlay`'s RGBA layer must have alpha==0 wherever mask<=0.5 (background untouched)
    and alpha>0 only where mask>0.5 -- the property the old bare-imshow call lacked."""
    fig, ax = plt.subplots()
    base = np.random.RandomState(0).uniform(0, 1, (16, 16))
    mask = np.zeros((16, 16)); mask[4:8, 4:8] = 1
    _overlay(ax, base, [(mask, "red", 0.45)])
    overlay_img = ax.images[-1].get_array()
    alpha = np.asarray(overlay_img)[..., 3]
    assert np.all(alpha[mask <= 0.5] == 0.0)
    assert np.all(alpha[mask > 0.5] == 0.45)
    plt.close(fig)


def test_save_eval_figure_background_consistent_across_panels(tmp_path):
    """target/GT/pred panels share the SAME grayscale background (_norm2d(target_img[z])), so
    their RENDERED brightness must be comparable. Decodes the actual saved PNG (not a
    re-render through the already-fixed helper) so this fails against the pre-fix code, where
    a bare `imshow(mask, cmap="Reds", alpha=...)` tinted GT/pred's WHOLE frame (even
    non-lesion background) while "target" (no overlay at all) showed the true, much darker
    min-max-normalized image -- exactly the DWI failure mode that surfaced this bug."""
    from PIL import Image

    T = 16
    # A few large bright outliers dominating the min-max range (mirrors the DWI failure mode:
    # a handful of hyperintense voxels crush the rest of the slice toward black).
    img = np.full((T, T, T), 50.0, dtype=np.float32)
    img[0, 0, 0] = 5000.0
    gt = np.zeros((T, T, T)); gt[8, 4:8, 4:8] = 1
    pred = np.zeros((T, T, T)); pred[8, 5:7, 5:7] = 1

    out = tmp_path / "fig.png"
    save_eval_figure(target_img=img, gt=gt, pred=pred, ctx_img=img, ctx_gt=gt, out_path=out)
    arr = np.asarray(Image.open(out).convert("L"), dtype=np.float64)

    # 4 equal-width panels (context|target|GT|pred), left to right.
    w = arr.shape[1] // 4
    panel_mean = [arr[:, i * w:(i + 1) * w].mean() for i in range(4)]
    target_mean, gt_mean, pred_mean = panel_mean[1], panel_mean[2], panel_mean[3]
    # Same background + a SMALL (16/256 px) foreground overlay -> means must be close.
    # The bug inflated GT/pred's mean by tens of brightness levels (near-white tint over the
    # whole frame at alpha=0.45); a loose-but-real threshold catches that without being
    # sensitive to font/whitespace-cropping noise between panels.
    assert abs(target_mean - gt_mean) < 15, panel_mean
    assert abs(target_mean - pred_mean) < 15, panel_mean

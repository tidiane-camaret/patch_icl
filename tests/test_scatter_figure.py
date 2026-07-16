import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import numpy as np
from pathlib import Path


def test_save_scatter_figure_writes_png(tmp_path):
    from evaluate import save_scatter_figure
    H, Rf, M = 32, 16, 10
    rng = np.random.default_rng(0)
    tgt_image = rng.random((H, H)).astype("float32")
    tgt_gt = (rng.random((H, H)) > 0.5).astype("float32")
    coarse = rng.random((H, H)).astype("float32")
    fused = rng.random((H, H)).astype("float32")
    q_ij = np.stack([rng.integers(0, Rf, M), rng.integers(0, Rf, M)], axis=-1)
    q_core = rng.random(M) > 0.5
    q_fg = q_core & (rng.random(M) > 0.5)          # fg is subset of core
    s_ij = np.stack([rng.integers(0, Rf, M), rng.integers(0, Rf, M)], axis=-1)
    s_core = rng.random(M) > 0.5
    s_fg = s_core & (rng.random(M) > 0.5)
    out = tmp_path / "scatter.png"
    save_scatter_figure(tgt_image, tgt_gt, coarse, fused,
                        q_ij, q_core, q_fg,
                        tgt_image, tgt_gt, s_ij, s_core, s_fg,
                        Rf, out, title="smoke")
    assert out.exists() and out.stat().st_size > 0

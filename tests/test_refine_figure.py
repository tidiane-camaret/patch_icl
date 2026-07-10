import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import numpy as np
from pathlib import Path
from evaluate import save_refine_figure


def _img(n=16):
    rng = np.random.default_rng(0)
    return rng.random((n, n)).astype("float32")


def test_writes_png(tmp_path):
    p = tmp_path / "sub" / "case_refine.png"       # parent dir does not exist yet
    save_refine_figure(
        tgt_image=_img(), tgt_gt=(_img() > 0.5).astype("float32"),
        ctx_image=_img(), ctx_gt=(_img() > 0.5).astype("float32"),
        coarse_pred=_img(), fused_pred=_img(), refine_pred=_img(8),   # T=8 over a 16 image
        tgt_box=(2, 3, 8), ctx_box=(4, 4, 8), out_path=p, title="t")
    assert p.exists() and p.stat().st_size > 0


def test_border_clamped_box_ok(tmp_path):
    p = tmp_path / "border_refine.png"
    # box flush against the bottom-right corner (origin + size == H): must not raise
    save_refine_figure(
        tgt_image=_img(), tgt_gt=(_img() > 0.5).astype("float32"),
        ctx_image=_img(), ctx_gt=(_img() > 0.5).astype("float32"),
        coarse_pred=_img(), fused_pred=_img(), refine_pred=_img(8),
        tgt_box=(8, 8, 8), ctx_box=(0, 0, 8), out_path=p, title="t")
    assert p.exists() and p.stat().st_size > 0

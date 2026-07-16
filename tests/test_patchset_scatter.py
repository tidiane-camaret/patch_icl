import sys; sys.path.insert(0, ".")
import torch
from src.models.patchset_cnn import PatchSetCNN


def _model(resolutions, H=32, refine_mode="reencode"):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolutions[0], enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, resolutions=resolutions, refine_mode=refine_mode)


def _batch(B=2, K=2, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_attn_core_grid_path_matches_segment():
    # After the refactor, the single-level forward (which routes through _attn -> _attn_core)
    # must equal a direct _segment call bit-for-bit.
    m = _model([8])
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert torch.equal(out["final_logit"], m._segment(img, cin, cout))
    assert hasattr(m, "_attn_core")


def _scatter_model(resolutions=(8, 16), H=32):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolutions[0], enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, resolutions=list(resolutions),
                       refine_mode="scatter", sample={"n_total": 20, "n_fg_core": 4,
                                                      "n_fg_core_ctx": 4})


def test_scatter_forward_shapes():
    m = _scatter_model((8, 16), H=32)          # fine grid Rf=16
    img, cin, cout = _batch(B=2, K=2, H=32)
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)     # coarse at T=8
    assert out["refine_logit"].shape == (2, 20)         # M sampled cells
    assert out["refine_idx"].shape == (2, 20)
    assert out["refine_grid_res"] == 16
    assert int(out["refine_idx"].max()) < 16 * 16 and int(out["refine_idx"].min()) >= 0
    assert "refine_origin" not in out                    # scatter != bbox
    assert torch.isfinite(out["refine_logit"]).all()


def test_scatter_backward_runs():
    m = _scatter_model()
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    out["refine_logit"].sum().backward()                 # gradients flow
    assert any(p.grad is not None for p in m.parameters())


def test_scatter_reproducible_with_seed():
    # Scatter sampling is always stochastic (Gumbel neighbor fill) to avoid the deterministic
    # top-left tie dump; reproducibility comes from seeding (eval_incontext.py seeds upstream).
    m = _scatter_model().eval()
    img, cin, cout = _batch()
    with torch.no_grad():
        torch.manual_seed(0); a = m(img, context_in=cin, context_out=cout)["refine_idx"]
        torch.manual_seed(0); b = m(img, context_in=cin, context_out=cout)["refine_idx"]
    assert torch.equal(a, b)                              # same seed -> same sample


def test_scatter_returns_tier_keys():
    m = _scatter_model((8, 16), H=32)          # K=2, M=20, Rf=16
    img, cin, cout = _batch(B=2, K=2, H=32)
    out = m(img, context_in=cin, context_out=cout)
    B, K, M, Rf = 2, 2, 20, 16
    assert out["refine_is_core"].shape == (B, M)
    assert out["refine_is_fg"].shape == (B, M)
    assert out["refine_is_core"].dtype == torch.bool
    assert out["refine_is_fg"].dtype == torch.bool
    assert out["refine_sup_idx"].shape == (B, K, M)
    assert out["refine_sup_is_core"].shape == (B, K, M)
    assert out["refine_sup_is_fg"].shape == (B, K, M)
    assert out["refine_sup_idx"].dtype == torch.long
    assert int(out["refine_sup_idx"].max()) < Rf * Rf and int(out["refine_sup_idx"].min()) >= 0
    # fg-core is a subset of core (partition invariant)
    assert bool((out["refine_is_fg"] & ~out["refine_is_core"]).any()) is False

import sys; sys.path.insert(0, ".")
import torch
from src.models.patchset_cnn import PatchSetCNN


def _model(sim_prior=True, H=32, resolution=8, mask_patch_size=2):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=resolution, enc_dims=[16], e=32, h=64,
                       l=1, a=2, thinking_rows=1, mask_patch_size=mask_patch_size,
                       sim_prior=sim_prior)


def test_similarity_prior_shape_range_detached():
    m = _model()
    B, N, S, Cf, p2 = 2, m.N, 3 * m.N, m.encoder.out_ch, m.mask_patch_size ** 2
    torch.manual_seed(1)
    qry = torch.rand(B, N, Cf, requires_grad=True)
    sup = torch.rand(B, S, Cf)
    occ = (torch.rand(B, S, p2) > 0.5).float()
    prior, valid = m._similarity_prior(qry, sup, occ)
    assert prior.shape == (B, N)
    assert valid.shape == (B,) and valid.dtype == torch.bool
    assert float(prior.min()) >= 0.0 and float(prior.max()) <= 1.0
    assert not prior.requires_grad                       # detached input signal


def test_similarity_prior_matches_fg_exemplar():
    # A query cell whose feature EQUALS a foreground support cell's feature has cosine 1.0
    # (the global max), so after per-image min-max it must sit at the per-image maximum.
    m = _model()
    B, N, Cf, p2 = 1, m.N, m.encoder.out_ch, m.mask_patch_size ** 2
    torch.manual_seed(2)
    sup = torch.rand(B, N, Cf)                            # S == N here (K=1 worth)
    occ = torch.zeros(B, N, p2)
    occ[0, 5] = 1.0                                       # support cell 5 is foreground
    qry = torch.rand(B, N, Cf)
    qry[0, 9] = sup[0, 5]                                 # query cell 9 == fg support cell 5
    prior, valid = m._similarity_prior(qry, sup, occ)
    assert bool(valid[0]) is True
    assert torch.argmax(prior[0]).item() == 9            # cell 9 is the peak
    assert float(prior[0, 9]) > 0.99                      # normalized to the max (~1.0)


def test_similarity_prior_degenerate_no_fg():
    # No foreground support cells -> valid False (caller falls back to the flat mean prior).
    m = _model()
    B, N, Cf, p2 = 2, m.N, m.encoder.out_ch, m.mask_patch_size ** 2
    torch.manual_seed(3)
    qry = torch.rand(B, N, Cf)
    sup = torch.rand(B, N, Cf)
    occ = torch.zeros(B, N, p2)                           # all background
    prior, valid = m._similarity_prior(qry, sup, occ)
    assert bool(valid.any()) is False
    assert torch.isfinite(prior).all()                   # degenerate rows are finite (0), not -inf


def _batch(B=2, K=1, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_sim_prior_no_new_params():
    # Zero-parameter feature: enabling sim_prior must not change the parameter set
    # (existing checkpoints must load strict=True).
    n_off = sum(p.numel() for p in _model(sim_prior=False).parameters())
    n_on = sum(p.numel() for p in _model(sim_prior=True).parameters())
    assert n_off == n_on


def test_sim_prior_changes_output():
    # Same seed -> identical weights (sim_prior adds no params/RNG draw), so any output
    # difference is due to the prior actually being wired into _attn.
    off = _model(sim_prior=False)
    on = _model(sim_prior=True)
    img, cin, cout = _batch()
    with torch.no_grad():
        a = off(img, context_in=cin, context_out=cout)["final_logit"]
        b = on(img, context_in=cin, context_out=cout)["final_logit"]
    assert a.shape == b.shape
    assert not torch.allclose(a, b)


def test_sim_prior_forward_backward_smoke():
    m = _model(sim_prior=True)               # H=32, resolution=8, mask_patch_size=2
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 8, 8)     # one logit per token (decode=1)
    assert torch.isfinite(out["final_logit"]).all()
    out["final_logit"].sum().backward()                 # gradients still flow through the model
    assert any(p.grad is not None for p in m.parameters())

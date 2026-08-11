import torch
from src.models.patchset3d import ConvEncoder3D, _mask_tiles_3d


def test_conv_encoder3d_shape():
    enc = ConvEncoder3D(1, (8, 8, 8), resolution=4)
    assert enc.out_ch == 24
    out = enc(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 24, 4, 4, 4)


def test_mask_tiles_3d_shape_and_occupancy():
    m = torch.zeros(2, 1, 8, 8, 8)
    m[:, :, :4, :4, :4] = 1.0
    tiles = _mask_tiles_3d(m, grid_res=4, p=2)   # 4^3 cells, 2^3 tile
    assert tiles.shape == (2, 64, 8)
    # cell (0,0,0) fully inside the ones block -> all-ones tile
    assert torch.allclose(tiles[0, 0], torch.ones(8))
from src.models.patchset3d import PatchSet3D


def _dummy_batch(B=2, K=2, S=16):
    image = torch.randn(B, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).long()
    return image, context_in, context_out


def test_patchset3d_forward_grid_shape():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                   thinking_rows=2, fourier_bands=4, mask_patch_decode_size=2)
    assert m.grid_size == 8
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 8, 8, 8)


def test_patchset3d_backward():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    out.mean().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0


def test_predict_and_train_forward_native_shape():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    img, cin, cout = _dummy_batch(S=16)
    logits = m.train_forward(img, cin, cout)
    assert logits.shape == (2, 1, 16, 16, 16)
    pred = m.predict(img, cin, cout)
    assert pred.shape == (2, 16, 16, 16)
    assert set(torch.unique(pred).tolist()) <= {0.0, 1.0}


def test_token_masking_noop_when_ratios_zero():
    """Default ratios (0.0): masks are None even in train mode; logit shape unchanged."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    m.train()
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    assert out["mask_support"] is None and out["mask_query"] is None


def test_token_masking_active_in_train():
    """ratio>0 under train(): masks have right shape, ~right fraction, grad flows to mask_token."""
    torch.manual_seed(0)
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   token_mask_ratio_support=0.5, token_mask_ratio_query=0.5)
    m.train()
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)   # N=64, support M=128, query M=64
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    assert out["mask_support"].shape == (2, 128) and out["mask_support"].dtype == torch.bool
    assert out["mask_query"].shape == (2, 64)
    assert abs(out["mask_support"].float().mean().item() - 0.5) < 0.15
    out["final_logit"].mean().backward()
    assert m.mask_token.grad is not None and torch.isfinite(m.mask_token.grad).all()


def test_token_masking_off_in_eval():
    """Even with ratio>0, eval mode never masks (eval/predict reproducibility)."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   token_mask_ratio_support=0.5, token_mask_ratio_query=0.5)
    m.eval()
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["mask_support"] is None and out["mask_query"] is None


def test_token_masking_content_replaced():
    """Masked cells must carry mask_token content, not the original embedding.

    Strategy: use transformer_rope=True (no additive pos encoding) so _tokens output is
    purely img_embed(feat) / mask_embed(occ) for unmasked or mask_token for masked cells.
    Set mask_token to a known sentinel (all 99.0), then assert the masked position's
    image and mask columns equal 99.0 while the unmasked position differs.
    """
    torch.manual_seed(42)
    B, M, e = 1, 4, 32
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=e, h=64, l=2, a=2, thinking_rows=2,
                   transformer_rope=True)   # rope=True => self.pos is None, no additive PE
    with torch.no_grad():
        m.mask_token.fill_(99.0)

    # Build minimal token inputs: random feat/occ (B, M, enc_dim / p^3)
    feat = torch.randn(B, M, m.encoder.out_ch)
    occ = torch.rand(B, M, 1)
    ijk = m.ijk_base[:M].unsqueeze(0).expand(B, M, 3)

    # Mask only position 0; positions 1-3 unmasked
    mask = torch.zeros(B, M, dtype=torch.bool)
    mask[0, 0] = True

    toks = m._tokens(feat, occ, ijk, mask=mask)   # (B, M, 2, e)

    # Masked position: both image (col 0) and mask (col 1) must be 99.0
    assert toks[0, 0, 0].eq(99.0).all(), "masked image column was not replaced by mask_token"
    assert toks[0, 0, 1].eq(99.0).all(), "masked mask column was not replaced by mask_token"

    # Unmasked position: image column must differ (img_embed output is random, not 99.0)
    assert not toks[0, 1, 0].eq(99.0).all(), "unmasked position wrongly shows mask_token value"

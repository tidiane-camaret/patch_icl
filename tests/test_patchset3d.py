import torch
from src.models.patchset3d import ConvEncoder3D, MaskConvEmbed, _mask_tiles_3d


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


def test_query_prior_feeds_the_query_mask_token():
    """query_prior=None is deterministic + unchanged; a prior shifts the logits (query path)."""
    torch.manual_seed(0)
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   mask_patch_decode_size=2)
    m.eval()
    img, cin, cout = _dummy_batch(S=16)
    base = m(img, context_in=cin, context_out=cout)["final_logit"]
    assert torch.equal(base, m(img, context_in=cin, context_out=cout)["final_logit"])
    prior = torch.rand(img.shape[0], 1, 16, 16, 16)
    with_prior = m(img, context_in=cin, context_out=cout, query_prior=prior)["final_logit"]
    assert with_prior.shape == base.shape
    assert not torch.allclose(with_prior, base)


def test_query_prior_mask_patch_size_gt1():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   mask_patch_size=2, mask_patch_decode_size=2)
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout,
            query_prior=torch.rand(2, 1, 16, 16, 16))["final_logit"]
    assert out.shape == (2, 1, 8, 8, 8)


def test_query_prior_backward_reaches_all_params():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2)
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout,
            query_prior=torch.rand(2, 1, 16, 16, 16))["final_logit"]
    out.mean().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0


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


# ── mask_slots (gt/pred identity tag on the single mask column) ─────────────────────────
# c-axis is always (img, mask) — see docs/logs.md 2026-09-04 second/third entries for why a
# separate c-axis column per content type (the earlier _tokens_multi design) was replaced by
# an additive identity tag on the one shared mask column.

def test_mask_slots_default_is_legacy_shape():
    """mask_slots=1 (default) reproduces the exact pre-existing param shapes — no slot_pos
    module, mask_embed input unchanged — so old checkpoints keep loading unmodified."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   mask_patch_size=2)
    assert m.slot_pos is None
    assert m.mask_embed.in_features == 2 ** 3          # p³, unchanged by mask_slots
    assert not any("slot_pos" in n for n, _ in m.named_parameters())
    assert m.slot_layout == ("img", "mask") and m.slot_index == {"img": 0, "mask": 1}
    assert m.decode_source == "img" and m._decode_col == 0


def test_slot_layout_is_fixed_regardless_of_mask_slots():
    """The c-axis never grows: slot_layout/slot_index are ("img","mask") at every
    mask_slots value — only decode_source (img vs mask) changes which column is read."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   mask_slots=2)
    assert m.slot_layout == ("img", "mask")
    assert m.slot_index == {"img": 0, "mask": 1}
    assert m._decode_col == 0          # default decode_source="img"

    m2 = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                    thinking_rows=2, mask_slots=2, decode_source="mask")
    assert m2._decode_col == 1
    # mask_embed itself never widens — content identity rides as an additive tag, not an
    # extra input column (see docstring above and _tokens).
    assert m.mask_embed.in_features == m2.mask_embed.in_features == 1


def test_decode_source_invalid_raises():
    """"gt"/"pred" are mask-CONTENT identities (_MASK_CONTENT_TYPES), not c-axis columns —
    they are not valid decode_source values even at mask_slots=2."""
    for bad in ("bbox", "gt", "pred"):
        try:
            PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                      thinking_rows=2, mask_slots=2, decode_source=bad)
            assert False, f"decode_source={bad!r} should have raised (not in slot_layout)"
        except AssertionError as exc:
            assert "decode_source" in str(exc)


def test_decode_source_changes_the_readout():
    """decode_source picks a genuinely different column: 'mask' output must differ from the
    default 'img' output for the same weights/inputs (same seed -> same init)."""
    torch.manual_seed(7)
    kw = dict(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
             mask_slots=2)
    m_img = PatchSet3D(decode_source="img", **kw)
    torch.manual_seed(7)
    m_mask = PatchSet3D(decode_source="mask", **kw)
    # Same seed -> identical initialization for every shared param.
    assert torch.equal(m_img.mask_embed.weight, m_mask.mask_embed.weight)

    torch.manual_seed(0)
    img, cin, cout = _dummy_batch(S=16)
    m_img.eval(); m_mask.eval()
    out_img = m_img(img, context_in=cin, context_out=cout)["final_logit"]
    out_mask = m_mask(img, context_in=cin, context_out=cout)["final_logit"]
    assert not torch.allclose(out_img, out_mask)


def test_mask_slots2_forward_and_backward():
    """mask_slots=2: forward shape unchanged; backward reaches every param, including the
    (unwidened) mask_embed and the new slot_pos module."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   mask_slots=2, mask_patch_decode_size=2)
    assert m.mask_embed.in_features == 1                # p³=1, no presence bit
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout,
            query_prior=torch.rand(2, 1, 16, 16, 16))["final_logit"]
    assert out.shape == (2, 1, 8, 8, 8)
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"


def test_mask_slots_rejects_unsupported_count():
    for bad in (0, 3, -1):
        try:
            PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                      thinking_rows=2, mask_slots=bad)
            assert False, f"mask_slots={bad} should have raised"
        except AssertionError as exc:
            assert "mask_slots" in str(exc)


def test_slot_pos_separable_from_spatial_pos():
    """Content-type identity (slot_pos) and spatial position (pos) must be linearly
    independent signals: (1) their projection ranges only intersect at 0 (rank(joint) ==
    sum of ranks), and (2) a single fixed linear readout separates gt from pred with a
    clear margin at EVERY grid position — i.e. spatial position can't leak into / corrupt
    content identity. Regression guard for the design discussed 2026-09-03 (see
    docs/logs.md)."""
    torch.manual_seed(0)
    e, bands, R = 64, 8, 4
    m = PatchSet3D(resolution=R, enc_dims=(8, 8, 8), e=e, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=bands, mask_slots=2)

    Wp = m.pos.proj.weight.detach()          # (e, 2*3*bands)
    Ws = m.slot_pos.proj.weight.detach()     # (e, 2*1*bands)
    r_p = torch.linalg.matrix_rank(Wp)
    r_s = torch.linalg.matrix_rank(Ws)
    r_joint = torch.linalg.matrix_rank(torch.cat([Wp, Ws], dim=1))
    assert r_joint == r_p + r_s, "slot_pos and spatial pos ranges overlap — not separable"

    ijk = m.ijk_base.float()                 # (R³, 3), every grid cell
    pos_all = m.pos(ijk, R)                                        # (R³, e)
    s0 = m._slot_pos_vec(m._mask_content_index["gt"], ijk.device, ijk.dtype)
    s1 = m._slot_pos_vec(m._mask_content_index["pred"], ijk.device, ijk.dtype)
    w = (s1 - s0) / (s1 - s0).norm()
    score0 = (pos_all + s0) @ w
    score1 = (pos_all + s1) @ w
    margin = score1.min() - score0.max()
    assert margin > 0, "gt/pred not linearly separable across all spatial positions"


def test_tokens_content_type_tags_the_mask_column_only():
    """content_type changes the mask column (identity tag) but never the img column, and
    two different content_types produce two different mask columns for identical occ."""
    torch.manual_seed(1)
    B, M, e = 1, 5, 32
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=e, h=64, l=2, a=2, thinking_rows=2,
                   mask_slots=2, transformer_rope=True)   # self.pos is None
    feat = torch.randn(B, M, m.encoder.out_ch)
    occ = torch.rand(B, M, 1)
    ijk = m.ijk_base[:M].unsqueeze(0).expand(B, M, 3)

    gt_tok = m._tokens(feat, occ, ijk, content_type="gt")      # (B,M,2,e): [img, mask]
    pred_tok = m._tokens(feat, occ, ijk, content_type="pred")
    plain_tok = m._tokens(feat, occ, ijk, content_type=None)

    assert torch.equal(gt_tok[:, :, 0], pred_tok[:, :, 0]), \
        "content_type must not affect the img column"
    assert not torch.allclose(gt_tok[:, :, 1], pred_tok[:, :, 1]), \
        "gt and pred tags must produce different mask-column embeddings for the same occ"
    # Untagged (mask_slots=1 style) call = mask_embed(occ) with no slot_pos offset.
    assert torch.equal(plain_tok[:, :, 1], m.mask_embed(occ))


# ── mask_embed conv option (arch.mask_embed="conv") ──────────────────────────────────────

def test_mask_conv_embed_shape_and_backward():
    """MaskConvEmbed: (...,p³) -> (...,e), gradient reaches every param, for both a p that
    needs conv layers (8) and one that degenerates to no conv layers at all (1)."""
    for p in (8, 4, 1):
        m = MaskConvEmbed(p, e=32)
        occ = torch.rand(2, 5, p ** 3, requires_grad=True)
        out = m(occ)
        assert out.shape == (2, 5, 32)
        out.mean().backward()
        missing = [n for n, prm in m.named_parameters() if prm.requires_grad and prm.grad is None]
        assert not missing, f"p={p}: no grad reached {missing}"


def test_mask_conv_embed_p1_matches_plain_linear():
    """p=1 has no spatial extent: MaskConvEmbed must reduce to exactly nn.Linear(1,e) (no
    conv layers added, pool/flatten on an already-1x1x1 input are no-ops)."""
    torch.manual_seed(3)
    m = MaskConvEmbed(1, e=16)
    assert len(m.convs) == 0
    occ = torch.rand(4, 1)
    assert torch.allclose(m(occ), m.proj(occ), atol=1e-6)


def test_mask_conv_embed_is_cheaper_than_linear_at_production_scale():
    """Regression guard for the params/FLOPs claim (p=8, e=768): conv path has meaningfully
    fewer params than the flat Linear(512,768) it replaces."""
    p, e = 8, 768
    linear_params = p ** 3 * e + e
    conv_params = sum(prm.numel() for prm in MaskConvEmbed(p, e).parameters())
    assert conv_params < linear_params / 5, (
        f"conv path ({conv_params} params) should be well under 1/5 of the linear "
        f"path's {linear_params} — check base_ch/architecture didn't grow")


def test_mask_embed_default_is_linear():
    """arch.mask_embed default ('linear') reproduces the exact pre-existing module — old
    checkpoints keep loading unmodified."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   mask_patch_size=2)
    assert isinstance(m.mask_embed, torch.nn.Linear)
    assert m.mask_embed.in_features == 2 ** 3


def test_mask_embed_conv_forward_and_backward():
    """arch.mask_embed='conv' swaps in MaskConvEmbed and the full model still trains."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   mask_patch_size=2, mask_embed="conv")
    assert isinstance(m.mask_embed, MaskConvEmbed)
    img, cin, cout = _dummy_batch(S=16)
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"


def test_mask_embed_rejects_unknown_value():
    try:
        PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                  thinking_rows=2, mask_embed="mlp")
        assert False, "mask_embed='mlp' should have raised"
    except AssertionError as exc:
        assert "mask_embed" in str(exc)

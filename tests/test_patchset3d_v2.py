import torch
from src.models.encoders.factory import build_encoder
from src.models.patchset3d_v2 import PatchSetV2


def test_build_encoder_conv_matches_direct_construction():
    from src.models.patchset3d import ConvEncoder3D
    enc = build_encoder("conv", resolution=4, enc_dims=(8, 8, 8))
    assert isinstance(enc, ConvEncoder3D)
    assert enc.out_ch == 24
    out = enc(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 24, 4, 4, 4)


def _dummy_batch(B=2, K=2, S=16):
    image = torch.randn(B, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).float()
    return image, context_in, context_out


def _replicate_stage_b_internals(m, seq_in, B, K, T, cascade_regs):
    """Reproduces _stage_b's steps up through the transformer call, so a test can inspect
    the intermediate sequence _stage_b itself doesn't return -- used to verify `regs` reads
    the correct row offset, not just that it differs from its input."""
    per_vol = m.compress_m + 1
    seq = seq_in
    if m.context_id_embed:
        seq = m._apply_context_tags(seq, B, K, T, per_vol)
    seq, _ = m.thinking(seq, seq.shape[1])
    n_extra = 0
    if cascade_regs is not None:
        mem = m.cascade_proj(cascade_regs) + m.cascade_type
        mem = mem.unsqueeze(2).expand(-1, -1, seq.shape[2], -1)
        seq = torch.cat([mem, seq], dim=1)
        n_extra = mem.shape[1]
    seq = m.transformer(seq, 0, full_attn=True)
    return seq, n_extra


def test_tokens_all_shape():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    B, K, T = 2, 2, 3
    feat = torch.randn(B, T, m.N, m.encoder.out_ch)
    occ = torch.randn(B, T, m.N, 1)
    tok = m._tokens_all(feat, occ, B, T)
    assert tok.shape == (B, T, m.N, 2, 32)


def test_occupancy_shapes():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    _, cin, cout = _dummy_batch(S=16, K=2)
    occ = m._occupancy(cout)
    assert occ.shape == (2, 2, m.N, 1)
    prior = torch.rand(2, 1, 16, 16, 16)
    pocc = m._prior_occupancy(prior)
    assert pocc.shape == (2, 1, m.N, 1)


def test_pool_all_native_resolution_matters():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[1], image_size=[16, 16, 16])
    B, K, T = 1, 1, 2
    S = m.encoder.fine_stage_size(16, 1)          # stage-1 side: coarser than native 16
    assert S < 16
    Cf = m.encoder.fine_stage_channels(1)

    # Feature map with real spatial structure: two halves along the D axis at different values.
    fine_finest = torch.ones(B * T, Cf, S, S, S)
    fine_finest[:, :, S // 2:] = 5.0

    # A tiny, native-resolution foreground region sitting entirely in the LOW-value half, near
    # the region boundary -- small enough that coarse (S-resolution) downsampling of the mask
    # would blur it across the boundary into the HIGH-value half, while upsampling the FEATURE
    # to native first (this task's implementation) keeps the mask exact.
    context_out = torch.zeros(B, K, 16, 16, 16)
    context_out[:, :, 6:7, :2, :2] = 1.0           # native slice just before the D-axis midpoint

    pool = m._pool_all(fine_finest, context_out, None, B, K, T)
    assert pool.shape == (B, T, 32)

    # Replicate the WRONG (coarse-mask-first) ordering this task must NOT match: downsample the
    # mask to S first, then mask the still-coarse feature map, then reduce.
    mask_coarse = torch.nn.functional.interpolate(
        context_out.reshape(B * K, 1, 16, 16, 16).float(), size=(S, S, S),
        mode="trilinear", align_corners=False).reshape(B, K, 1, S, S, S)
    feat_for_wrong = fine_finest.reshape(B, T, Cf, S, S, S)[:, :K]
    mu = feat_for_wrong.mean(dim=(-3, -2, -1), keepdim=True)
    sig = feat_for_wrong.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
    feat_z = ((feat_for_wrong - mu) / sig).clamp(-10, 10)
    num_wrong = (feat_z * mask_coarse).sum(dim=(-3, -2, -1))
    den_wrong = mask_coarse.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    pooled_wrong_raw = (num_wrong / den_wrong).squeeze(1)          # (B,Cf), pre-projection

    # The CORRECT (native-upsample-first) raw pooled vector, computed the same way _pool_all
    # does internally, for a like-for-like comparison before pool_proj's learned weights:
    feat_native = torch.nn.functional.interpolate(
        fine_finest.float(), size=(16, 16, 16), mode="trilinear", align_corners=False
        ).reshape(B, T, Cf, 16, 16, 16)[:, :K]
    mu2 = feat_native.mean(dim=(-3, -2, -1), keepdim=True)
    sig2 = feat_native.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
    feat_z2 = ((feat_native - mu2) / sig2).clamp(-10, 10)
    mask_native = context_out.reshape(B, K, 1, 16, 16, 16).float()
    num_right = (feat_z2 * mask_native).sum(dim=(-3, -2, -1))
    den_right = mask_native.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    pooled_right_raw = (num_right / den_right).squeeze(1)

    assert not torch.allclose(pooled_right_raw, pooled_wrong_raw, atol=1e-3), (
        "native-resolution and coarse-resolution masking orders produced the same result -- "
        "this test setup doesn't actually distinguish upsample-before-mask from mask-before-upsample")

    # Now tie the REAL _pool_all output to the correct reference, and confirm it does NOT
    # match the wrong one -- this is what actually proves _pool_all follows the correct
    # (upsample-to-native-first) ordering, not just that the two references differ from
    # each other.
    right_via_proj = m.pool_proj(pooled_right_raw)
    wrong_via_proj = m.pool_proj(pooled_wrong_raw)
    assert torch.allclose(pool[:, 0], right_via_proj, atol=1e-4), (
        "_pool_all's real output for the context volume does not match the correct "
        "(upsample-to-native-first) reference computation")
    assert not torch.allclose(pool[:, 0], wrong_via_proj, atol=1e-3), (
        "_pool_all's real output matches the WRONG (mask-at-coarse-first) ordering")


def test_compress_and_assemble_shapes():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, compress_layers=2, fine_stage=[0],
                   image_size=[16, 16, 16])
    B, T = 2, 3
    tok = torch.randn(B, T, m.N, 2, 32)
    compressed = m._compress_all(tok, B, T)
    assert compressed.shape == (B, T, 3, 2, 32)
    pool = torch.randn(B, T, 32)
    seq = m._assemble_sequence(pool, compressed, B, T)
    assert seq.shape == (B, T * 4, 2, 32)          # per volume: 1 pool row + 3 compressed
    # first row of each volume's block is the (broadcast) pool row
    per_vol = 4
    for t in range(T):
        block = seq[:, t * per_vol:(t + 1) * per_vol]
        assert torch.allclose(block[:, 0, 0], pool[:, t])   # img col
        assert torch.allclose(block[:, 0, 1], pool[:, t])   # mask col (broadcast)


def test_stage_b_shape_and_cross_volume_mixing():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, compress_layers=1, fine_stage=[0],
                   context_id_embed=True, image_size=[16, 16, 16])
    B, K, T = 2, 2, 3
    per_vol = m.compress_m + 1
    seq = torch.randn(B, T * per_vol, 2, 32)
    seq_out, regs = m._stage_b(seq, B, K, T)
    n_think = m.thinking.n
    assert seq_out.shape == (B, n_think + T * per_vol, 2, 32)
    assert regs is None                          # cascade_registers=False by default

    start = n_think + K * per_vol
    target_block_a = seq_out[:, start:start + per_vol]

    seq_perturbed = seq.clone()
    seq_perturbed[:, :per_vol] += 5.0             # perturb context volume 0 only
    seq_out_b, _ = m._stage_b(seq_perturbed, B, K, T)
    target_block_b = seq_out_b[:, start:start + per_vol]
    assert not torch.allclose(target_block_a, target_block_b)


def test_stage_b_cascade_registers_roundtrip():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], cascade_registers=True,
                   image_size=[16, 16, 16])
    B, K, T = 2, 2, 3
    per_vol = m.compress_m + 1
    seq = torch.randn(B, T * per_vol, 2, 32)
    seq_out, regs = m._stage_b(seq, B, K, T)
    assert regs.shape == (2, m.thinking.n, 32)
    seq_out2, regs2 = m._stage_b(seq, B, K, T, cascade_regs=regs)
    assert seq_out2.shape[1] == seq_out.shape[1] + m.thinking.n

    # Position-sensitive check: reproduce the internal sequence _stage_b computed for the
    # SECOND call, then verify regs2 matches the correct (post-mem) thinking-row slice of
    # that tensor and does NOT match the wrong (mem-block) slice -- catches a regression to
    # the pre-fix `seq[:, :thinking.n]` bug, which "differs from regs" would not catch (both
    # slices differ from the raw input regs post-attention regardless of which is read).
    seq_internal, n_extra = _replicate_stage_b_internals(m, seq, B, K, T, regs)
    right_slice = seq_internal[:, n_extra:n_extra + m.thinking.n].mean(dim=2)
    wrong_slice = seq_internal[:, :m.thinking.n].mean(dim=2)
    assert torch.allclose(regs2, right_slice, atol=1e-5), (
        "_stage_b's returned regs does not match the correct (post-mem) thinking-row slice")
    assert not torch.allclose(regs2, wrong_slice, atol=1e-3), (
        "_stage_b's returned regs matches the WRONG (mem-block) slice -- position bug reintroduced")


def test_decode_shape_and_backward():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], decoder_dim=16,
                   image_size=[16, 16, 16])
    B = 2
    per_vol = m.compress_m + 1
    T_tok = torch.randn(B, per_vol, 32, requires_grad=True)
    F_q = torch.randn(B, m.N, 32, requires_grad=True)
    S = m.encoder.fine_stage_size(16, 0)
    fine = (torch.randn(B, m.encoder.fine_stage_channels(0), S, S, S),)
    logit = m._decode(T_tok, F_q, fine, B)
    assert logit.shape == (B, 1, S, S, S)          # stage-0 side == native (16) for a 16^3 input
    logit.mean().backward()
    assert T_tok.grad is not None and F_q.grad is not None
    assert m.iris_t2f.out_proj.weight.grad is not None
    assert m.iris_f2t.out_proj.weight.grad is not None


def test_forward_end_to_end_shape():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], decoder_dim=16,
                   image_size=[16, 16, 16])
    img, cin, cout = _dummy_batch(S=16, K=2)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 16, 16, 16)
    assert out["registers"] is None


def test_grid_size_matches_native_image_size():
    """train.py's is_patchset-gated logging reads net.grid_size directly (no getattr
    fallback) for val metric label suffixes -- must exist and reflect the native side."""
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], decoder_dim=16,
                   image_size=[16, 16, 16])
    assert m.grid_size == 16


def test_forward_backward():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    img, cin, cout = _dummy_batch(S=16, K=2)
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    out.mean().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0


def test_predict_and_train_forward_native_shape():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    img, cin, cout = _dummy_batch(S=16, K=2)
    logits = m.train_forward(img, cin, cout)
    assert logits.shape == (2, 1, 16, 16, 16)
    pred = m.predict(img, cin, cout)
    assert pred.shape == (2, 16, 16, 16)
    assert set(torch.unique(pred).tolist()) <= {0.0, 1.0}


def test_query_prior_changes_output():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    m.eval()
    img, cin, cout = _dummy_batch(S=16, K=2)
    base = m(img, context_in=cin, context_out=cout)["final_logit"]
    assert torch.equal(base, m(img, context_in=cin, context_out=cout)["final_logit"])
    prior = torch.rand(2, 1, 16, 16, 16)
    with_prior = m(img, context_in=cin, context_out=cout, query_prior=prior)["final_logit"]
    assert with_prior.shape == base.shape
    assert not torch.allclose(with_prior, base)


def test_cascade_regs_end_to_end():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16],
                   cascade_registers=True)
    m.eval()
    img, cin, cout = _dummy_batch(S=16, K=2)
    out1 = m(img, context_in=cin, context_out=cout)
    assert out1["registers"].shape == (2, m.thinking.n, 32)
    out2 = m(img, context_in=cin, context_out=cout, cascade_regs=out1["registers"])
    assert out2["final_logit"].shape == out1["final_logit"].shape
    assert not torch.allclose(out2["final_logit"], out1["final_logit"]), (
        "cascade_regs should influence the output -- if this passes trivially, cascade_regs "
        "may not be getting threaded through to _stage_b")


def test_query_prior_occupancy_path_alone_changes_output():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    m.eval()
    img, cin, cout = _dummy_batch(S=16, K=2)
    base = m(img, context_in=cin, context_out=cout)["final_logit"]
    prior = torch.rand(2, 1, 16, 16, 16)

    orig_pool_all = m._pool_all
    m._pool_all = lambda fine_finest, context_out, query_prior, B, K, T: orig_pool_all(
        fine_finest, context_out, None, B, K, T)
    try:
        occ_only = m(img, context_in=cin, context_out=cout, query_prior=prior)["final_logit"]
    finally:
        m._pool_all = orig_pool_all
    assert not torch.allclose(occ_only, base), (
        "query_prior's occupancy/mask-embed path alone should move the output even when "
        "_pool_all is forced to ignore query_prior")


def test_query_prior_pool_path_alone_changes_output():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    m.eval()
    img, cin, cout = _dummy_batch(S=16, K=2)
    base = m(img, context_in=cin, context_out=cout)["final_logit"]
    prior = torch.rand(2, 1, 16, 16, 16)

    fallback_occ = m._occupancy(cout).mean(dim=1, keepdim=True)
    orig_prior_occ = m._prior_occupancy
    m._prior_occupancy = lambda prior_: fallback_occ
    try:
        pool_only = m(img, context_in=cin, context_out=cout, query_prior=prior)["final_logit"]
    finally:
        m._prior_occupancy = orig_prior_occ
    assert not torch.allclose(pool_only, base), (
        "query_prior's _pool_all path alone should move the output even when the "
        "occupancy/mask-embed path is forced to ignore query_prior")


def test_forward_native_resize_when_decode_grid_not_native():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[1], decoder_dim=16, image_size=[16, 16, 16])
    img, cin, cout = _dummy_batch(S=16, K=2)
    out = m(img, context_in=cin, context_out=cout)
    assert out["final_logit"].shape == (2, 1, 16, 16, 16)


def test_build_model_dispatches_patchset_v2():
    import importlib.util
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "experiments_3d_train", "experiments/3d/train.py")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    cfg = OmegaConf.create({
        "model": "patchset3d_v2",
        "arch": {
            "resolution": 4, "enc_dims": [8, 8, 8], "e": 32, "h": 64, "l": 2, "a": 2,
            "thinking_rows": 2, "residual_decay": 0.95, "fourier_bands": 4,
            "compress_m": 3, "compress_layers": 1, "fine_stage": [0], "decoder_dim": 16,
            "encoder": "conv",
        },
        "data": {"image_size": [16, 16, 16]},
    })
    model, name = train_mod.build_model(cfg)
    assert name == "patchset3d_v2"
    from src.models.patchset3d_v2 import PatchSetV2
    assert isinstance(model, PatchSetV2)
    assert model.resolution == 4
    assert model.compress_m == 3
    assert len(model.transformer.blocks) == 2


def test_tokens_all_img_column_matches_img_embed_plus_pos():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    B, T = 1, 2
    feat = torch.randn(B, T, m.N, m.encoder.out_ch)
    occ = torch.randn(B, T, m.N, 1)
    tok = m._tokens_all(feat, occ, B, T)
    ijk = m.ijk_base.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
    expected_img = m.img_embed(feat) + m.pos(ijk, m.resolution)
    assert torch.allclose(tok[:, :, :, 0, :], expected_img, atol=1e-5)


def test_compress_all_no_cross_volume_leakage():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    B, T = 1, 3
    tok = torch.randn(B, T, m.N, 2, 32)
    base = m._compress_all(tok, B, T)
    tok_perturbed = tok.clone()
    tok_perturbed[:, 0] += 5.0          # perturb only volume 0's raw cells
    perturbed = m._compress_all(tok_perturbed, B, T)
    assert not torch.allclose(base[:, 0], perturbed[:, 0])   # volume 0 changed (expected)
    assert torch.allclose(base[:, 1], perturbed[:, 1])       # volumes 1, 2 must be UNCHANGED
    assert torch.allclose(base[:, 2], perturbed[:, 2])


def test_img_embed_mlp_true_uses_sequential():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16],
                   img_embed_mlp=True)
    assert isinstance(m.img_embed, torch.nn.Sequential)
    assert len(m.img_embed) == 3
    feat = torch.randn(2, 3, m.N, m.encoder.out_ch)
    out = m.img_embed(feat)
    assert out.shape == (2, 3, m.N, 32)


def test_img_embed_mlp_false_uses_plain_linear():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    assert isinstance(m.img_embed, torch.nn.Linear)


def test_feat_norm_none_is_identity():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16],
                   feat_norm="none")
    feat = torch.randn(2, 3, m.N, 8)
    out = m._feat_norm(feat, K=2)
    assert torch.equal(out, feat)


def test_feat_norm_context_matches_reference_computation():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16],
                   feat_norm="context")
    B, T, K, Cf = 2, 3, 2, 8
    feat = torch.randn(B, T, m.N, Cf) * 5.0 + 3.0
    out = m._feat_norm(feat, K)
    assert out.shape == feat.shape
    ctx = feat[:, :K]
    mu = ctx.mean(dim=(1, 2), keepdim=True)
    sig = ctx.std(dim=(1, 2), keepdim=True) + 1e-8
    expected = ((feat - mu) / sig).clamp(-10, 10)
    assert torch.allclose(out, expected, atol=1e-5)


def test_feat_norm_context_uses_context_stats_not_targets_own():
    """A regression bug that accidentally normalized the target by its OWN stats (i.e.
    behaved like "self" mode under a "context" label) would make this test's target land
    near-zero-mean; the real "context" behavior leaves it far from zero since it's
    normalized by the (very differently scaled) context group's stats instead."""
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16],
                   feat_norm="context")
    B, K, Cf = 2, 2, 8
    feat = torch.cat([
        torch.randn(B, K, m.N, Cf),                    # context: mean~0, std~1
        torch.randn(B, 1, m.N, Cf) + 100.0,             # target: mean~100, same scale
    ], dim=1)
    out = m._feat_norm(feat, K)
    tgt_mean = out[:, K:K + 1].mean(dim=(1, 2))
    # the target's z-score under context stats is so extreme it saturates the +/-10 clamp --
    # itself confirms it was normalized by the (very different) context stats, not its own
    assert (tgt_mean.abs() >= 9.9).all()
    ctx_mean = out[:, :K].mean(dim=(1, 2, 3))
    assert (ctx_mean.abs() < 0.5).all()


def test_feat_norm_self_normalizes_each_volume_by_its_own_stats():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16],
                   feat_norm="self")
    B, T, K, Cf = 2, 3, 2, 8
    feat = torch.stack(
        [torch.randn(B, m.N, Cf) * (t + 1) + t * 100.0 for t in range(T)], dim=1)
    out = m._feat_norm(feat, K)
    for t in range(T):
        assert out[:, t].mean(dim=1).abs().max() < 1e-3


def test_feat_norm_rejects_unknown_mode():
    import pytest
    with pytest.raises(AssertionError):
        PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                  fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16],
                  feat_norm="bogus")

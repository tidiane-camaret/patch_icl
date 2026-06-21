"""
Coarse → sample → assemble for the multilevel refinement task.

Given a data batch, a frozen stage-1 res-16 ImagePFN, and a frozen res-32 feature
encoder, build the support/query patch tensors consumed by PatchSetPFN.
"""

import torch
import torch.nn.functional as F
from omegaconf import ListConfig

from src.models.pfn_seg_2d import standardize_by_context
from sampling import sample_patches, idx_to_ij, gather_grid


def composite_predictions(coarse_flat, qidx, vals):
    """(B,N) dense map + (B,M) indices + (B,M) values → (B,N) with vals scattered in.

    Returns a NEW tensor (coarse_flat is not mutated); unsampled cells keep coarse value."""
    refined = coarse_flat.clone()
    refined.scatter_(1, qidx, vals)
    return refined


@torch.no_grad()
def coarse_predict(stage1, all_images, all_masks, K, grid_res):
    """Frozen stage-1 target prediction + thinking summary.

    Returns (coarse (B, grid_res, grid_res), coarse_lowres (B, R1, R1), thinking
    (B, n_think, e1)). coarse_lowres is the stage-1 prediction at its NATIVE resolution
    (R1, before any upsampling) — used so the coarse baseline Dice can be computed the
    same way pfn_seg.py does it (R1 → native, not R1 → grid_res → native). The thinking
    rows are the stage-1 transformer's post-attention latent summary, exposed via
    ImagePFN.forward(return_thinking=True)."""
    logits, think = stage1(all_images, all_masks, sep=K, return_thinking=True)  # (B,R1,R1), (B,n_think,e1)
    p_lowres = torch.sigmoid(logits.float())                                    # (B,R1,R1) native stage-1 res
    p = F.interpolate(p_lowres.unsqueeze(1), size=(grid_res, grid_res),
                      mode="bilinear", align_corners=False).squeeze(1)
    return p, p_lowres, think


def _grid_from_feat(feat, B, T, grid_res, n_context):
    """(B*T, Cf, R, R) pooled features → standardized (B, T, R², Cf), row-major cells."""
    Cf = feat.shape[1]
    feat = feat.flatten(2).transpose(1, 2).reshape(B, T, grid_res * grid_res, Cf)
    return standardize_by_context(feat, n_context)


@torch.no_grad()
def encode_grid(encoder, images, grid_res, n_context):
    """images (B, T, 1, H, W) → features (B, T, grid_res², Cf) in row-major cell order.

    Full encode+pool in one call (fallback path for encoders without encode_maps;
    run_chain prefers the encode-once path via pool_grid). Features are per-channel
    standardized using the first `n_context` rows' statistics (standardize_by_context),
    so the chain consumes them in the same normalized frame as ImagePFN's image path
    (rather than raw encoder magnitudes, which for DINOv3 'all' differ ~10–100× across
    stages)."""
    B, T, _, H, W = images.shape
    feat = encoder(images.reshape(B * T, 1, H, W), grid_res)  # (B*T, Cf, R2, R2)
    return _grid_from_feat(feat, B, T, grid_res, n_context)


@torch.no_grad()
def pool_grid(encoder, maps, B, T, grid_res, n_context):
    """Pool pre-encoded stage maps (encoder.encode_maps output) to grid_res → standardized
    (B, T, grid_res², Cf). The encode-once-pool-many path: the encoder ran once in
    run_chain; here we only pool/concat (resolution-independent maps) per hop."""
    feat = encoder.pool_maps(maps, grid_res)                 # (B*T, Cf, R2, R2)
    return _grid_from_feat(feat, B, T, grid_res, n_context)


def _grid_fractions(masks, grid_res):
    """masks (B, T, 1, H, W) → soft mask fraction per cell (B, T, grid_res²)."""
    B, T, _, H, W = masks.shape
    f = F.adaptive_avg_pool2d(masks.reshape(B * T, 1, H, W).float(), (grid_res, grid_res))
    return f.reshape(B, T, grid_res * grid_res)


def _mask_tiles(mask_hw, grid_res, p):
    """(B, 1, Hf, Wf) → (B, grid_res², p²): per-cell p×p mask tiles, row-major cell order.

    Resizes to grid_res*p first when needed (e.g. upsampling a coarse prior); for a
    native mask where Hf == grid_res*p it is an exact reshape (no interpolation)."""
    target = grid_res * p
    if mask_hw.shape[-1] != target or mask_hw.shape[-2] != target:
        mask_hw = F.interpolate(mask_hw.float(), size=(target, target),
                                mode="bilinear", align_corners=False)
    B = mask_hw.shape[0]
    return (mask_hw.reshape(B, 1, grid_res, p, grid_res, p)
                   .permute(0, 2, 4, 3, 5, 1)
                   .reshape(B, grid_res * grid_res, p * p))


def refine_level(model, batch, feats, coarse_grid, prev_think, grid_res, s,
                 source, stochastic, device):
    """One coarse-to-fine hop at grid_res. See module docstring / spec."""
    label       = batch["label"].to(device)        # (B,1,H,W)
    context_out = batch["context_out"].to(device)  # (B,K,1,H,W)
    B, K = context_out.shape[0], context_out.shape[1]
    N = grid_res * grid_res
    M = s.n_total

    gt_grid = F.adaptive_avg_pool2d(label.float(), (grid_res, grid_res)).reshape(B, N)
    ctx_frac_grid = _grid_fractions(context_out, grid_res).reshape(B, K, N)  # true masks

    # ── Query (target) patches ──
    sampling_map = gt_grid if source == "ds_gt" else coarse_grid
    qidx, q_is_core, q_is_fg = sample_patches(
        sampling_map, M, s.tau, s.n_fg_core, s.blur_sigma, s.floor, grid_res,
        temperature=s.temperature, stochastic=stochastic)
    qry_feat   = gather_grid(feats[:, -1], qidx)                 # (B,M,Cf)  target is last row
    qry_coarse = gather_grid(coarse_grid, qidx)
    qry_gt     = gather_grid(gt_grid, qidx)
    qry_ij     = idx_to_ij(qidx, grid_res)
    is_unc     = q_is_core & ~q_is_fg

    # ── Support (context) patches ──
    ctx_feat = feats[:, :K].reshape(B * K, N, feats.shape[-1])
    ctx_frac = ctx_frac_grid.reshape(B * K, N)
    sidx, _, _ = sample_patches(
        ctx_frac, M, s.tau, s.n_fg_core_ctx, s.blur_sigma, s.floor, grid_res,
        temperature=s.temperature, stochastic=stochastic)
    sup_feat = gather_grid(ctx_feat, sidx).reshape(B, K * M, feats.shape[-1])
    sup_ij   = idx_to_ij(sidx, grid_res).reshape(B, K * M, 2)

    # ── Mask-token: scalar or p×p tile ──
    if s.mask_prior == "patch":
        # p follows the model's baked tile size, not the input image — so a larger
        # eval image (Strategy A) is resized down inside _mask_tiles rather than
        # producing a p×p that mismatches mask_embed. At the training size these are
        # equal (p = image_size//grid_res), so this is a no-op for training.
        p = model.mask_patch_size
        ctx_tiles = torch.stack([_mask_tiles(context_out[:, k], grid_res, p) for k in range(K)],
                                dim=1).reshape(B * K, N, p * p)
        sup_label = gather_grid(ctx_tiles, sidx).reshape(B, K * M, p * p)
        coarse_tiles = _mask_tiles(coarse_grid.reshape(B, 1, grid_res, grid_res), grid_res, p)
        qry_prior = gather_grid(coarse_tiles, qidx)
    else:
        sup_label = gather_grid(ctx_frac, sidx).reshape(B, K * M)
        qry_prior = qry_coarse

    logits, this_think = model(sup_feat, sup_label, sup_ij, qry_feat, qry_prior, qry_ij,
                               grid_res, stage1_think=prev_think, return_thinking=True)
    refined_grid = composite_predictions(coarse_grid, qidx, torch.sigmoid(logits.float()))
    return {"refined_grid": refined_grid, "logits": logits, "qry_gt": qry_gt,
            "qry_coarse": qry_coarse, "qry_is_uncertain": is_unc, "qidx": qidx,
            "this_think": this_think, "gt_grid": gt_grid}


def _level_cfg(cfg, L):
    """Per-hop config namespace, reading list entries from cfg.sample at index L."""
    from types import SimpleNamespace
    s = cfg.sample
    pick = lambda v: v[L] if isinstance(v, (list, ListConfig)) else v
    return SimpleNamespace(
        n_total=pick(s.n_total), n_fg_core=pick(s.n_fg_core),
        n_fg_core_ctx=pick(s.get("n_fg_core_ctx", s.n_fg_core)),
        tau=pick(s.tau), blur_sigma=pick(s.blur_sigma), floor=pick(s.floor),
        temperature=pick(s.temperature), mask_prior=cfg.arch.mask_prior)


def run_chain(batch, stage1, encoder, models, cfg, source, stochastic, device):
    """Coarse-to-fine chain. Returns (outputs list per hop, coarse_lr (B,R0,R0))."""
    resolutions = list(cfg.sample.resolutions)
    image       = batch["image"].to(device)
    context_in  = batch["context_in"].to(device)
    context_out = batch["context_out"].to(device)
    B, K = context_in.shape[0], context_in.shape[1]

    all_images = torch.cat([context_in, image.unsqueeze(1)], dim=1)
    all_masks  = torch.cat([context_out, torch.zeros_like(image.unsqueeze(1))], dim=1)
    T = all_images.shape[1]

    R0 = resolutions[0]
    # Frozen stage-1 + encoder: no grad (only the per-level PatchSetPFNs train).
    with torch.no_grad():
        _, coarse_lr, think = coarse_predict(stage1, all_images, all_masks, K, R0)  # p_lowres @R0
    prev_dense = coarse_lr.reshape(B, R0 * R0)
    prev_think = think
    prev_res = R0

    # Encode the images ONCE (encoder stage maps are resolution-independent), then pool
    # to each hop's grid. Avoids re-running the full encoder per hop — the maps differ
    # between hops only in the final pooling size. Falls back to per-hop encode_grid for
    # plain-callable encoders without encode_maps (e.g. test stubs).
    H, W = all_images.shape[-2:]
    encode_maps = getattr(encoder, "encode_maps", None)
    img_maps = None
    if encode_maps is not None:
        with torch.no_grad():
            img_maps = encode_maps(all_images.reshape(B * T, 1, H, W))

    outputs = []
    for L, grid in enumerate(resolutions[1:]):
        coarse_grid = F.interpolate(prev_dense.reshape(B, 1, prev_res, prev_res),
                                    size=(grid, grid), mode="bilinear",
                                    align_corners=False).reshape(B, grid * grid)
        with torch.no_grad():
            feats = (pool_grid(encoder, img_maps, B, T, grid, K) if img_maps is not None
                     else encode_grid(encoder, all_images, grid, K))
        s = _level_cfg(cfg, L)
        hop = refine_level(models[L], batch, feats, coarse_grid, prev_think, grid, s,
                           source, stochastic, device)
        outputs.append(hop)
        prev_dense = hop["refined_grid"].detach()
        prev_think = hop["this_think"].detach()
        prev_res = grid
    return outputs, coarse_lr

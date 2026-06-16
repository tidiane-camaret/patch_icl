"""
Coarse → sample → assemble for the multilevel refinement task.

Given a data batch, a frozen stage-1 res-16 ImagePFN, and a frozen res-32 feature
encoder, build the support/query patch tensors consumed by PatchSetPFN.
"""

import torch
import torch.nn.functional as F

from sampling import sample_patches, idx_to_ij, gather_grid


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


@torch.no_grad()
def encode_grid(encoder, images, grid_res):
    """images (B, T, 1, H, W) → features (B, T, grid_res², Cf) in row-major cell order."""
    B, T, _, H, W = images.shape
    feat = encoder(images.reshape(B * T, 1, H, W), grid_res)  # (B*T, Cf, R2, R2)
    Cf = feat.shape[1]
    return feat.flatten(2).transpose(1, 2).reshape(B, T, grid_res * grid_res, Cf)


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


@torch.no_grad()
def build_patch_batch(batch, stage1, encoder, cfg, device, sampling_source="prev_pred",
                      stochastic=True):
    """Returns a dict of tensors on `device` for PatchSetPFN + metrics.

    sampling_source selects the query sampling map only (prior + fusion stay coarse):
      "prev_pred" : rank cells by the stage-1 coarse prediction (deployable).
      "ds_gt"     : rank cells by the downsampled target GT (oracle — leaks labels).
    stochastic gates the Gumbel neighbor fill (True in train; False for reproducible eval).

    Keys: sup_feat (B,K*M,Cf), sup_label (B,K*M), sup_ij (B,K*M,2),
          qry_feat (B,M,Cf), qry_prior (B,M), qry_ij (B,M,2),
          qry_gt (B,M), qry_coarse (B,M), qry_is_uncertain (B,M bool),
          qry_idx (B,M flat cell idx), coarse_full (B,N), coarse_lowres (B,R1,R1),
          gt_full (B,N), stage1_think (B, n_think, e1).
    M = n_total;  N = grid_res².  qry_is_uncertain = boundary core (excludes fg-core).
    """
    R2 = cfg.sample.grid_res
    M  = cfg.sample.n_total
    s  = cfg.sample  # tau / n_fg_core / blur_sigma / floor / temperature

    image       = batch["image"].to(device)         # (B,1,H,W)
    label       = batch["label"].to(device)         # (B,1,H,W)
    context_in  = batch["context_in"].to(device)    # (B,K,1,H,W)
    context_out = batch["context_out"].to(device)   # (B,K,1,H,W)
    B, K = context_in.shape[0], context_in.shape[1]

    # Stack: target is the LAST row; query mask is zeros (stage-1 fills its own prior).
    all_images = torch.cat([context_in, image.unsqueeze(1)], dim=1)             # (B,T,1,H,W)
    all_masks  = torch.cat([context_out, torch.zeros_like(image.unsqueeze(1))], dim=1)

    coarse, coarse_lowres, stage1_think = coarse_predict(stage1, all_images, all_masks, K, R2)
    coarse_flat = coarse.reshape(B, R2 * R2)                                    # (B,N)

    feats = encode_grid(encoder, all_images, R2)                               # (B,T,N,Cf)
    fracs = _grid_fractions(all_masks, R2)                                     # (B,T,N); context rows = real masks
    # The TARGET fraction MUST come from the real label: all_masks zeroes the query
    # mask (so stage-1 never sees the answer), so fracs[:, -1] would be all zeros.
    gt32  = F.adaptive_avg_pool2d(label.float(), (R2, R2)).reshape(B, R2 * R2)  # (B,N) real target GT

    # ── Query (target) patches: threshold core + fg-core + neighbor fill ─────
    sampling_map = gt32 if sampling_source == "ds_gt" else coarse_flat
    qidx, q_is_core, q_is_fg = sample_patches(
        sampling_map, M, s.tau, s.n_fg_core, s.blur_sigma, s.floor, R2,
        temperature=s.temperature, stochastic=stochastic)                       # (B,M)
    qry_feat   = gather_grid(feats[:, -1], qidx)                                # (B,M,Cf)
    qry_coarse = gather_grid(coarse_flat, qidx)                                 # (B,M) — metrics baseline
    qry_gt     = gather_grid(gt32, qidx)                                        # (B,M)
    qry_ij     = idx_to_ij(qidx, R2)                                            # (B,M,2)
    is_unc = q_is_core & ~q_is_fg                                               # boundary core only

    # ── Support (context) patches: rank by true mask fraction, batched over K ─
    ctx_feat = feats[:, :K].reshape(B * K, R2 * R2, feats.shape[-1])            # (B*K,N,Cf)
    ctx_frac = fracs[:, :K].reshape(B * K, R2 * R2)                             # (B*K,N)
    n_fg_core_ctx = s.get("n_fg_core_ctx", s.n_fg_core)   # heavier fg quota for support patches
    sidx, _, _ = sample_patches(
        ctx_frac, M, s.tau, n_fg_core_ctx, s.blur_sigma, s.floor, R2,
        temperature=s.temperature, stochastic=stochastic)                       # (B*K,M)
    sup_feat  = gather_grid(ctx_feat, sidx).reshape(B, K * M, feats.shape[-1])  # (B,K*M,Cf)
    sup_ij    = idx_to_ij(sidx, R2).reshape(B, K * M, 2)                        # (B,K*M,2)

    # ── Mask-token: scalar fraction, or p×p mask tile (arch.mask_prior) ──────
    # "false" is handled in the model (query prior → support-mean); pipeline passes
    # the coarse-derived prior for both "false" and "scalar".
    if cfg.arch.mask_prior == "patch":
        p = image.shape[-1] // R2                                              # auto p = H // grid_res
        ctx_tiles = torch.stack([_mask_tiles(context_out[:, k], R2, p) for k in range(K)],
                                dim=1).reshape(B * K, R2 * R2, p * p)           # (B*K,N,p²) native GT
        sup_label = gather_grid(ctx_tiles, sidx).reshape(B, K * M, p * p)      # (B,K*M,p²)
        coarse_tiles = _mask_tiles(coarse.unsqueeze(1), R2, p)                  # (B,N,p²) upsampled prior
        qry_prior = gather_grid(coarse_tiles, qidx)                            # (B,M,p²)
    else:
        sup_label = gather_grid(ctx_frac, sidx).reshape(B, K * M)              # (B,K*M) scalar
        qry_prior = qry_coarse                                                  # (B,M) scalar

    return {
        "sup_feat": sup_feat, "sup_label": sup_label, "sup_ij": sup_ij,
        "qry_feat": qry_feat, "qry_prior": qry_prior, "qry_ij": qry_ij,
        "qry_gt": qry_gt, "qry_coarse": qry_coarse, "qry_is_uncertain": is_unc,
        "qry_idx": qidx,                                                        # (B, M) flat cell idx
        "coarse_full": coarse_flat, "gt_full": gt32,                           # (B, N) full res-32 maps
        "coarse_lowres": coarse_lowres,                                         # (B, R1, R1) native stage-1 res
        "stage1_think": stage1_think,                                           # (B, n_think, e1)
    }

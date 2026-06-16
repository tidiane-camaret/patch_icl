"""
Coarse → sample → assemble for the multilevel refinement task.

Given a data batch, a frozen stage-1 res-16 ImagePFN, and a frozen res-32 feature
encoder, build the support/query patch tensors consumed by PatchSetPFN.
"""

import torch
import torch.nn.functional as F

from sampling import sample_patch_indices, idx_to_ij, gather_grid


@torch.no_grad()
def coarse_predict(stage1, all_images, all_masks, K, grid_res):
    """Frozen stage-1 target prediction + thinking summary.

    Returns (coarse (B, grid_res, grid_res), thinking (B, n_think, e1)). The thinking
    rows are the stage-1 transformer's post-attention latent summary, exposed via
    ImagePFN.forward(return_thinking=True)."""
    logits, think = stage1(all_images, all_masks, sep=K, return_thinking=True)  # (B,R1,R1), (B,n_think,e1)
    p = torch.sigmoid(logits.float())
    p = F.interpolate(p.unsqueeze(1), size=(grid_res, grid_res),
                      mode="bilinear", align_corners=False).squeeze(1)
    return p, think


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


@torch.no_grad()
def build_patch_batch(batch, stage1, encoder, cfg, device, sampling_source="prev_pred"):
    """Returns a dict of tensors on `device` for PatchSetPFN + metrics.

    sampling_source selects the query sampling map only (prior + fusion stay coarse):
      "prev_pred" : rank cells by the stage-1 coarse prediction (deployable).
      "ds_gt"     : rank cells by the downsampled target GT (oracle — leaks labels).

    Keys: sup_feat (B,K*M,Cf), sup_label (B,K*M), sup_ij (B,K*M,2),
          qry_feat (B,M,Cf), qry_prior (B,M), qry_ij (B,M,2),
          qry_gt (B,M), qry_coarse (B,M), qry_is_uncertain (B,M bool),
          qry_idx (B,M flat cell idx), coarse_full (B,N), gt_full (B,N),
          stage1_think (B, n_think, e1).
    M = n_uncertain + n_certain;  N = grid_res².
    """
    R2 = cfg.sample.grid_res
    n_unc, n_cer = cfg.sample.n_uncertain, cfg.sample.n_certain
    M = n_unc + n_cer

    image       = batch["image"].to(device)         # (B,1,H,W)
    label       = batch["label"].to(device)         # (B,1,H,W)
    context_in  = batch["context_in"].to(device)    # (B,K,1,H,W)
    context_out = batch["context_out"].to(device)   # (B,K,1,H,W)
    B, K = context_in.shape[0], context_in.shape[1]

    # Stack: target is the LAST row; query mask is zeros (stage-1 fills its own prior).
    all_images = torch.cat([context_in, image.unsqueeze(1)], dim=1)             # (B,T,1,H,W)
    all_masks  = torch.cat([context_out, torch.zeros_like(image.unsqueeze(1))], dim=1)

    coarse, stage1_think = coarse_predict(stage1, all_images, all_masks, K, R2)  # (B,R2,R2), (B,n_think,e1)
    coarse_flat = coarse.reshape(B, R2 * R2)                                    # (B,N)

    feats = encode_grid(encoder, all_images, R2)                               # (B,T,N,Cf)
    fracs = _grid_fractions(all_masks, R2)                                     # (B,T,N); context rows = real masks
    # The TARGET fraction MUST come from the real label: all_masks zeroes the query
    # mask (so stage-1 never sees the answer), so fracs[:, -1] would be all zeros.
    gt32  = F.adaptive_avg_pool2d(label.float(), (R2, R2)).reshape(B, R2 * R2)  # (B,N) real target GT

    # ── Query (target) patches: rank by the sampling map ─────────────────────
    sampling_map = gt32 if sampling_source == "ds_gt" else coarse_flat
    qidx = sample_patch_indices(sampling_map, n_unc, n_cer)                     # (B,M)
    qry_feat   = gather_grid(feats[:, -1], qidx)                                # (B,M,Cf)
    qry_coarse = gather_grid(coarse_flat, qidx)                                 # (B,M)
    qry_gt     = gather_grid(gt32, qidx)                                        # (B,M)
    qry_ij     = idx_to_ij(qidx, R2)                                            # (B,M,2)
    qry_prior  = qry_coarse if cfg.arch.coarse_prior else torch.zeros_like(qry_coarse)
    is_unc = torch.zeros(B, M, dtype=torch.bool, device=device)
    is_unc[:, :n_unc] = True

    # ── Support (context) patches: rank by true mask fraction, batched over K ─
    ctx_feat = feats[:, :K].reshape(B * K, R2 * R2, feats.shape[-1])            # (B*K,N,Cf)
    ctx_frac = fracs[:, :K].reshape(B * K, R2 * R2)                             # (B*K,N)
    sidx = sample_patch_indices(ctx_frac, n_unc, n_cer)                         # (B*K,M)
    sup_feat  = gather_grid(ctx_feat, sidx).reshape(B, K * M, feats.shape[-1])  # (B,K*M,Cf)
    sup_label = gather_grid(ctx_frac, sidx).reshape(B, K * M)                   # (B,K*M)
    sup_ij    = idx_to_ij(sidx, R2).reshape(B, K * M, 2)                        # (B,K*M,2)

    return {
        "sup_feat": sup_feat, "sup_label": sup_label, "sup_ij": sup_ij,
        "qry_feat": qry_feat, "qry_prior": qry_prior, "qry_ij": qry_ij,
        "qry_gt": qry_gt, "qry_coarse": qry_coarse, "qry_is_uncertain": is_unc,
        "qry_idx": qidx,                                                        # (B, M) flat cell idx
        "coarse_full": coarse_flat, "gt_full": gt32,                           # (B, N) full res-32 maps
        "stage1_think": stage1_think,                                           # (B, n_think, e1)
    }

"""
Zoom-refinement chain: same ImagePFN arch as stage-1, fed a contiguous square crop.

Stage-1 (frozen) predicts at R0 on the full image; each hop crops an s×s square (max
predicted mass for the target, densest GT for each context), pools the once-computed
encoder maps to that bbox, and runs a warm-started ImagePFN that corrects the cropped
coarse prediction. The R0×R0 output is upsampled to s×s and composited back. Hops chain
through the detached composite. See the design/plan in docs/superpowers.
"""

import torch
import torch.nn.functional as F

from bbox import composite_window, crop_resize, gt_window, max_sum_window
from pipeline import _grid_from_feat


def crop_pool_maps(maps, origin, s, out):
    """encode_maps list → (N, sum(C_i), out, out): each stage map cropped to the s×s bbox
    at `origin` (N,2) and resampled to out×out, then concatenated over channels."""
    return torch.cat([crop_resize(m.float(), origin, s, out, mode="bilinear") for m in maps],
                     dim=1)


@torch.no_grad()
def _coarse(stage1, all_images, all_masks, K):
    logits = stage1(all_images, all_masks, sep=K)
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.sigmoid(logits.float())            # (B, R0, R0) probability


def run_zoom_chain(batch, stage1, encoder, models, cfg, source, stochastic, device):
    """Coarse-to-fine zoom chain. Returns (outputs list per hop, coarse_lr (B,R0,R0))."""
    crop_sizes = list(cfg.sample.crop_sizes)
    H = cfg.data.image_size
    image       = batch["image"].to(device)             # (B,1,H,W)
    context_in  = batch["context_in"].to(device)        # (B,K,1,H,W)
    context_out = batch["context_out"].to(device)       # (B,K,1,H,W)
    label       = batch["label"].to(device)             # (B,1,H,W)
    B, K = context_in.shape[0], context_in.shape[1]

    all_images = torch.cat([context_in, image.unsqueeze(1)], dim=1)         # (B,T,1,H,W)
    all_masks  = torch.cat([context_out, torch.zeros_like(image.unsqueeze(1))], dim=1)
    T = all_images.shape[1]

    coarse_lr = _coarse(stage1, all_images, all_masks, K)                   # (B,R0,R0)
    R0 = coarse_lr.shape[-1]
    pred = F.interpolate(coarse_lr.unsqueeze(1), size=(H, H),
                         mode="bilinear", align_corners=False)             # (B,1,H,W)

    with torch.no_grad():
        maps = encoder.encode_maps(all_images.reshape(B * T, 1, H, H))

    outputs = []
    for L, s in enumerate(crop_sizes):
        tgt_o = max_sum_window(pred, s)                                     # (B,2)
        ctx_o = torch.stack([gt_window(context_out[:, k], s) for k in range(K)], dim=1)  # (B,K,2)
        origins = torch.cat([ctx_o, tgt_o.unsqueeze(1)], dim=1).reshape(B * T, 2)        # (B*T,2)

        with torch.no_grad():
            feat = crop_pool_maps(maps, origins, s, R0)                     # (B*T, Cf, R0, R0)
            image_feats = _grid_from_feat(feat, B, T, R0, K)               # (B,T,N,Cf) standardized

        # Mask images cropped to each bbox: context = true GT (nearest); query = coarse prior.
        # out=H: crops are upsampled back to native H so ImagePFN re-patchifies them at its native resolution.
        ctx_mask = crop_resize(context_out.reshape(B * K, 1, H, H),
                               ctx_o.reshape(B * K, 2), s, H, mode="nearest").reshape(B, K, 1, H, H)
        qry_prior = crop_resize(pred, tgt_o, s, H, mode="bilinear").unsqueeze(1)         # (B,1,1,H,W)
        masks_in = torch.cat([ctx_mask, qry_prior], dim=1)                 # (B,T,1,H,W)

        logits = models[L](None, masks_in, sep=K, image_feats=image_feats,
                           seed_query_mask=True)                           # (B,R0,R0)
        logits = logits.reshape(B, R0 * R0)
        qry_gt = crop_resize(label, tgt_o, s, R0, mode="bilinear").reshape(B, R0 * R0)   # soft GT

        patch = F.interpolate(torch.sigmoid(logits.float()).reshape(B, 1, R0, R0),
                              size=(s, s), mode="bilinear", align_corners=False)         # (B,1,s,s)
        pred = composite_window(pred, patch, tgt_o, s).detach()
        outputs.append({"logits": logits, "qry_gt": qry_gt, "refined_full": pred,
                        "origin": tgt_o, "crop_size": s})

    return outputs, coarse_lr

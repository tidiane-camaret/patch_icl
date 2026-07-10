"""PatchSetCNN: low-resolution in-context 2D segmentation (set-of-patches attention).

A trainable CNN encoder downsamples each image to an R×R feature grid; every patch
of every image becomes a token in a *set*, and a TabPFN-style dual-axis transformer
(reused from pfn_seg_2d) does content-based in-context matching over that set —
following PatchSetPFN's layout rather than ImagePFN's image-grid layout.

Why the set layout: in ImagePFN/the image-grid layout, cross-image attention is
position-locked (query patch (i,j) attends only to context patch (i,j)). When objects
are not spatially aligned across images (e.g. omniSynth characters in random grid
cells), that edge is mostly wasted. Here the rows are the *patches themselves* and the
cross-patch (sample-axis) attention lets each query patch attend to ALL support
patches; the patch's (i,j) grid cell is injected as a Fourier positional *feature*
(resolution-generalizable) rather than enforced by the attention structure.

Layout (mirrors PatchSetPFN): rows = thinking + support patches (K·N) + query patches
(N); cols = [img-token | mask-token]. Prediction is at the bottleneck resolution R
(no decoder/upsampling): the head emits one logit per query patch → R×R, and the loss
is taken against the avg-pooled GT mask (the trainer pools labels to the logit size).

Design choices (see docs/logs.md):
  - single-stream encoder: no UniverSeg support/target double-encoding and no
    cross-convolution — a plain conv stack shared over the K+1 images, multi-scale
    features concatenated.
  - mask token = scalar occupancy (avg-pool of the binary mask within each patch),
    embedded by Linear(1→e). Query patches' mask = mean of support occupancy (the
    TargetEncoder class-frequency prior).
  - feature-axis attention routes img↔mask within a patch; sample-axis attention is
    the full-set cross-patch matching (query patches attend to support patches only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patchset_pfn import FourierPositionalEncoding
from src.models.pfn_seg_2d import ThinkingRows, TransformerEncoderStack
from src.models.bbox_refine import crop_resize, gt_window, max_sum_window


class ConvEncoder(nn.Module):
    """Single-stream conv encoder with multi-scale feature concatenation.

    (B, in_ch, H, W) → (B, sum(dims), R, R). The encoder DEPTH is set purely by
    `len(dims)` — a full-res stem + (len(dims)-1) stride-2 stages — so the deepest
    map lands at H / 2**(len(dims)-1), the encoder's *natural* resolution. The token
    grid R is decoupled from that: every scale is resampled to R×R (area-pool to
    downsample, bilinear to upsample) and concatenated along channels, so each output
    patch token carries low- through high-level features (cf. UniverSeg's per-stage
    64-wide encoder blocks) regardless of how R compares to the encoder depth. `dims`
    are the per-stage channel widths. Each stage is a strided conv + a same-resolution
    conv (GroupNorm + LeakyReLU); one stream, no cross-conv.
    """
    def __init__(self, in_ch: int, dims: tuple[int, ...], resolution: int,
                 groups: int = 8):
        super().__init__()
        assert len(dims) >= 1, "dims needs at least a stem width"
        self.resolution = resolution
        n_down = len(dims) - 1                          # depth from architecture, not R

        def cbr(ci, co, stride):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=stride, padding=1),
                nn.GroupNorm(groups, co),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.stem = cbr(in_ch, dims[0], 1)
        self.stages = nn.ModuleList([
            nn.Sequential(cbr(dims[i], dims[i + 1], 2),
                          cbr(dims[i + 1], dims[i + 1], 1))
            for i in range(n_down)
        ])
        self.out_ch = sum(dims)                        # concatenated multi-scale width

    @staticmethod
    def _resample(f: torch.Tensor, R: int) -> torch.Tensor:
        """Resize an (B,C,h,w) feature map to R×R: area-pool when shrinking, bilinear
        when growing — so R can be smaller OR larger than the feature's native size."""
        if f.shape[-1] == R:
            return f
        if f.shape[-1] > R:
            return F.interpolate(f, size=(R, R), mode="area")
        return F.interpolate(f, size=(R, R), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.resolution
        feats = [self.stem(x)]                         # full-res stem features
        for stage in self.stages:
            feats.append(stage(feats[-1]))             # successively downsampled stages
        # Resample every scale to the token grid R×R and concat along channels.
        feats = [self._resample(f, R) for f in feats]
        return torch.cat(feats, dim=1)                 # (B, sum(dims), R, R)


class PatchSetCNN(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        resolution: int = 16,
        enc_dims: tuple[int, ...] = (64, 64, 64, 64),
        e: int = 256,
        h: int = 512,
        l: int = 6,
        a: int = 4,
        thinking_rows: int = 8,
        residual_decay: float = 0.95,
        fourier_bands: int = 8,
        query_self_attn: bool = False,
        context_id_embed: bool = False,
        max_context: int = 16,
        resolutions: list[int] | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        # `resolutions` = effective full-image resolutions per level (level 0 = coarse over the
        # full image). The token grid T is constant across levels and equals resolutions[0]; each
        # further level k crops the image to c_k = image_size*resolutions[0]/resolutions[k] px so
        # its T tokens resolve a finer effective resolution. None → single level = plain model.
        self.resolutions = [resolution] if resolutions is None else [int(r) for r in resolutions]
        assert len(self.resolutions) <= 2, \
            "multi-hop refinement (>2 levels) not implemented yet; use resolutions=[T] or [T, R1]"
        resolution = self.resolutions[0]                 # token grid T (drives the encoder)
        self.resolution = resolution
        self.N = resolution ** 2
        self.query_self_attn = query_self_attn
        self.context_id_embed = context_id_embed
        self.max_context = max_context
        # Derived per-level crop sizes (px in the image_size frame); level 0 is the full image.
        self.refine_crops = []
        for rk in self.resolutions[1:]:
            assert rk % resolution == 0 and (image_size * resolution) % rk == 0, \
                f"resolutions[k]={rk} must be a multiple of resolutions[0]={resolution} and " \
                f"divide image_size*resolutions[0]={image_size * resolution}"
            c = image_size * resolution // rk
            assert 0 < c <= image_size, f"derived crop {c} out of range for resolutions[k]={rk}"
            self.refine_crops.append(c)
        self.encoder = ConvEncoder(1, tuple(enc_dims), resolution)
        self.img_embed = nn.Linear(self.encoder.out_ch, e)
        self.mask_embed = nn.Linear(1, e)              # scalar occupancy → e
        self.pos = FourierPositionalEncoding(e, fourier_bands)   # (i,j) → e, added to both cols
        # Optional per-context-image identity: a learned tag added to all patches of a
        # given context image (shared across its N patches, both cols), so the otherwise
        # permutation-invariant patch set can be grouped by source image. The query image
        # gets its own learned tag. Without this, two context images with identical content
        # (e.g. instCopy duplicates) yield byte-identical tokens and are indistinguishable.
        if context_id_embed:
            self.ctx_id = nn.Embedding(max_context, e)     # support image slot → e
            self.qry_id = nn.Parameter(torch.zeros(e))     # the target image's tag
            # Small-norm init (σ≈0.1) puts this learnable identity embedding in the
            # rich/feature-learning regime — the Adam sweet spot from Ito et al. (2025),
            # "Learning interpretable positional encodings depends on initialization".
            # nn.Embedding's default (σ=1) is the lazy/memorization regime; σ≲0.05 also
            # hurts under Adam (adaptive LR jumps back to the kernel regime).
            nn.init.normal_(self.ctx_id.weight, std=0.1)
            nn.init.normal_(self.qry_id, std=0.1)
        self.thinking = ThinkingRows(thinking_rows, e)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
        self.decoder = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, 1))
        # Row-major (i,j) grid coords of the R×R patch lattice, shared by every image.
        ii = torch.arange(resolution).repeat_interleave(resolution)
        jj = torch.arange(resolution).repeat(resolution)
        self.register_buffer("ij_base", torch.stack([ii, jj], dim=-1), persistent=False)  # (N,2)

    def _tokens(self, feat, occ, ij):
        """feat (B,M,Cf); occ (B,M,1); ij (B,M,2) → (B,M,2,e) = [img-token | mask-token]."""
        p = self.pos(ij, self.resolution)              # (B,M,e) Fourier position feature
        img = self.img_embed(feat) + p
        msk = self.mask_embed(occ) + p
        return torch.stack([img, msk], dim=2)

    def _segment(self, image, context_in, context_out):
        """Coarse single-pass segmentation → (B,1,R,R) logits.

        image (B,1,H,W); context_in/out (B,K,1,H,W). Support = all K·N context patches
        (known mask occupancy); query = the N target patches (mask = support-mean prior)."""
        B, K = context_in.shape[0], context_in.shape[1]
        R, N = self.resolution, self.N
        H, W = image.shape[-2:]

        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)          # (B,T,1,H,W)
        T = imgs.shape[1]

        # ── encode all images → per-patch features (B,T,N,Cf) ───────────────────
        feat = self.encoder(imgs.reshape(B * T, 1, H, W))                  # (B*T,Cf,R,R)
        Cf = feat.shape[1]
        feat = feat.flatten(2).transpose(1, 2).reshape(B, T, N, Cf)        # (B,T,N,Cf)
        sup_feat = feat[:, :K].reshape(B, K * N, Cf)                       # (B,S,Cf)  S=K·N
        qry_feat = feat[:, K:].reshape(B, N, Cf)                           # (B,Q,Cf)  Q=N

        # ── support mask occupancy; query prior = support-mean ──────────────────
        occ = F.adaptive_avg_pool2d(context_out.reshape(B * K, 1, H, W), (R, R))
        sup_occ = occ.reshape(B, K * N, 1)                                # (B,S,1)
        qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, 1)        # (B,Q,1) prior

        # ── per-channel standardize features by SUPPORT-patch stats ─────────────
        mu = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        sup_feat = ((sup_feat - mu) / sig).clamp(-10, 10)
        qry_feat = ((qry_feat - mu) / sig).clamp(-10, 10)

        # ── per-patch (i,j) grid coords ─────────────────────────────────────────
        sup_ij = self.ij_base.repeat(K, 1).unsqueeze(0).expand(B, K * N, 2)   # (B,S,2)
        qry_ij = self.ij_base.unsqueeze(0).expand(B, N, 2)                    # (B,Q,2)

        # ── set-of-patches transformer: rows = [support | query], cols = [img|mask]
        sup_tok = self._tokens(sup_feat, sup_occ, sup_ij)                 # (B,S,2,e)
        qry_tok = self._tokens(qry_feat, qry_occ, qry_ij)                 # (B,Q,2,e)

        # Per-image identity: same tag for all N patches of a context image (image-major
        # order in sup_*), added to both cols; the target image gets its own tag. Lets the
        # attention group patches by source image (and tell apart identical context copies).
        if self.context_id_embed:
            assert K <= self.max_context, \
                f"context_size {K} exceeds max_context {self.max_context}"
            e_dim = sup_tok.shape[-1]
            ctx_emb = self.ctx_id(torch.arange(K, device=sup_tok.device))  # (K,e)
            ctx_emb = ctx_emb.repeat_interleave(N, dim=0)                  # (S,e) image-major
            sup_tok = sup_tok + ctx_emb.view(1, K * N, 1, e_dim)
            qry_tok = qry_tok + self.qry_id.view(1, 1, 1, e_dim)

        x = torch.cat([sup_tok, qry_tok], dim=1)                          # (B,S+Q,2,e)

        x, sep_t = self.thinking(x, K * N)
        # Default: every row attends to the train set (thinking + support) only. With
        # query_self_attn, query patches additionally attend to each other (within-target
        # spatial reasoning); they carry the prior (not GT), so this leaks no labels.
        attn_mask = None
        if self.query_self_attn:
            r = x.shape[1]
            attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
            attn_mask[:, :sep_t] = True           # all rows → thinking + support
            attn_mask[sep_t:, sep_t:] = True       # query patches → query patches
        x = self.transformer(x, sep_t, attn_mask=attn_mask)

        q = x[:, sep_t:, 0, :]                                            # (B,Q,e) query img-col
        logit = self.decoder(q).squeeze(-1).reshape(B, 1, R, R)
        return logit                                                      # (B,1,R,R)

    def _refine_forward(self, image, context_in, context_out):
        """Coarse pass over the full image + one bbox-zoom refine pass (SAME weights) → per-level
        heads. Crop the target on its densest predicted region and each context on its densest GT,
        resize crops to the encoder input, re-segment at the same T-token grid. No fusion — levels
        are supervised/metricked separately (the fused stitch is a metric only, built elsewhere)."""
        B, K = context_in.shape[0], context_in.shape[1]
        H, W = image.shape[-2:]
        c = self.refine_crops[0]                                          # derived crop (px)

        coarse = self._segment(image, context_in, context_out)           # (B,1,T,T)
        prob_up = F.interpolate(torch.sigmoid(coarse).detach(), size=(H, W),
                                mode="bilinear", align_corners=False)     # bbox selection only
        tgt_o = max_sum_window(prob_up, c)                               # (B,2) px origin
        ctx_o = torch.stack([gt_window(context_out[:, k], c) for k in range(K)], dim=1)  # (B,K,2)

        tgt_img = crop_resize(image, tgt_o, c, H, mode="bilinear")       # (B,1,H,W)
        ctx_img = crop_resize(context_in.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              c, H, mode="bilinear").reshape(B, K, 1, H, W)
        ctx_msk = crop_resize(context_out.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              c, H, mode="nearest").reshape(B, K, 1, H, W)

        refine = self._segment(tgt_img, ctx_img, ctx_msk)                # (B,1,T,T), same weights
        return {"final_logit": coarse, "refine_logit": refine,
                "refine_origin": tgt_o, "refine_ctx_origin": ctx_o,
                "refine_crop": c, "resolutions": self.resolutions}

    def forward(self, image, context_in, context_out, mode="train"):
        """image (B,1,H,W); context_in/out (B,K,1,H,W).

        Single level (len(resolutions)==1): {"final_logit": (B,1,T,T)} — the plain model.
        Multi level: per-level heads (final_logit=coarse, refine_logit, refine_origin,
        refine_crop, resolutions). `mode` is accepted for interface parity; unused."""
        if len(self.resolutions) == 1:
            return {"final_logit": self._segment(image, context_in, context_out)}
        return self._refine_forward(image, context_in, context_out)

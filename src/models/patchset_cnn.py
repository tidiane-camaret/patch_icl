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
from src.models.bbox_refine import crop_pool_maps, crop_resize, gt_window, max_sum_window
from src.models.scatter_sampling import sample_patches, idx_to_ij, gather_grid


DEFAULT_SAMPLE = dict(n_total=256, tau=0.30, blur_sigma=1.0, floor=0.005,
                      n_fg_core=64, n_fg_core_ctx=64, temperature=1.0, n_boundary_core=0)


def _mask_tiles(mask_hw: torch.Tensor, grid_res: int, p: int) -> torch.Tensor:
    """(B,1,Hf,Wf) → (B, grid_res², p²): per-cell p×p mask tile, row-major cell order.

    Resizes to grid_res*p first when needed (bilinear — e.g. upsampling a coarse prior or a
    native mask coarser than grid_res*p); for Hf == grid_res*p it is an exact reshape. Mirrors
    experiments/2d/multilevel/pipeline._mask_tiles, giving each patch a shaped mask token
    (p²-vector into mask_embed) instead of a single scalar occupancy."""
    target = grid_res * p
    if mask_hw.shape[-1] != target or mask_hw.shape[-2] != target:
        mask_hw = F.interpolate(mask_hw.float(), size=(target, target),
                                mode="bilinear", align_corners=False)
    B = mask_hw.shape[0]
    return (mask_hw.reshape(B, 1, grid_res, p, grid_res, p)
                   .permute(0, 2, 4, 3, 5, 1)
                   .reshape(B, grid_res * grid_res, p * p))


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

    def encode_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B,in,H,W) → list of native-resolution multi-scale maps [stem@H, stage1@H/2, …].

        Left un-pooled so a caller can crop them to a bbox and pool to a FINER effective grid
        (the encode-once refine path) instead of re-running the conv stack per scale."""
        feats = [self.stem(x)]                         # full-res stem features
        for stage in self.stages:
            feats.append(stage(feats[-1]))             # successively downsampled stages
        return feats

    def pool_maps(self, feats: list[torch.Tensor], R: int) -> torch.Tensor:
        """Resample every scale to R×R and concat along channels → (B, sum(dims), R, R)."""
        return torch.cat([self._resample(f, R) for f in feats], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_maps(self.encode_maps(x), self.resolution)


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
        full_attn: bool = False,
        context_id_embed: bool = False,
        max_context: int = 16,
        resolutions: list[int] | None = None,
        refine_mode: str = "reencode",
        refine_memory: bool = False,
        sample: dict | None = None,
        mask_patch_size: int = 8,
        mask_patch_decode_size: int = 1,
        sim_prior: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        # Two-level refine strategies (ignored when single-level):
        #   "reencode"    — re-run the whole model (encoder + attention) on the upsampled crop;
        #                   every stage is recomputed at the finer scale. 2× encoder passes.
        #   "encode_once" — encode all K+1 images ONCE, run the attention half twice: on the full
        #                   pooled features (coarse) and on the SAME maps cropped to each bbox and
        #                   pooled back (refine). ~half the encoder compute; refine reuses stem/
        #                   shallow detail but does NOT recompute deep-stage semantics at the crop.
        #   "scatter"     — encode once; pool to the fine grid Rf; sample M query cells from the
        #                   coarse prediction and M support cells from each context's true mask;
        #                   run the attention core on that sampled set.
        assert refine_mode in ("reencode", "encode_once", "scatter"), \
            f"bad refine_mode {refine_mode!r}"
        self.refine_mode = refine_mode
        self.sample = {**DEFAULT_SAMPLE, **(sample or {})}
        self.refine_memory = refine_memory
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
        # full_attn drops the sample-axis read-only mask entirely: every row (thinking +
        # support + query) attends to every row, so context representations become
        # target-aware. No label leak (query rows carry only the support-mean occupancy
        # prior, never GT). Supersets query_self_attn's connectivity, so it takes
        # precedence when both are set.
        self.full_attn = full_attn
        # Max-cosine similarity query prior (PFENet-style): when True, _attn seeds the query
        # mask token with a localized foreground-similarity prior instead of the flat
        # support-mean. Adds NO parameters (checkpoint-compatible); grid/single-level path only.
        self.sim_prior = bool(sim_prior)
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
        # Mask token input — RESOLUTION-AGNOSTIC. Each cell's native mask patch (image_size //
        # token-grid px, which varies with resolution) is resized to a FIXED p×p tile, so the
        # mask_embed is always Linear(p², e) and can be shared/transferred across resolutions.
        # p==1 → scalar avg-pool occupancy (Linear(1,e), fraction only); p>1 → a shaped occupancy
        # tile (which part of the cell is foreground), not just the fraction.
        self.mask_patch_size = int(mask_patch_size)
        assert self.mask_patch_size >= 1, "mask_patch_size must be >= 1"
        self.mask_embed = nn.Linear(self.mask_patch_size ** 2, e)   # occupancy tile p² → e
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
        # Cross-level memory: the refine pass attends to the coarse pass's (detached)
        # thinking rows, prepended as extra rows plus this learned type marker. Only
        # created when enabled, so default checkpoints gain zero parameters. Inert for
        # single-level models (no coarse pass to summarize).
        if refine_memory:
            self.mem_type = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.mem_type, std=0.02)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
        # Decoder head: each query token emits mask_patch_decode_size² logits. d==1 → one logit
        # per token (prediction at the token grid R, the original low-res behavior). d>1 → each
        # token decodes a d×d block that tiles back into an (R·d)×(R·d) map, so the transformer
        # reconstructs a HIGHER-res mask with no upsampling stage. d = image_size // resolution
        # reconstructs exactly the original input resolution (R·d = image_size).
        self.mask_patch_decode_size = int(mask_patch_decode_size)
        assert self.mask_patch_decode_size >= 1, "mask_patch_decode_size must be >= 1"
        self.decoder = nn.Sequential(nn.Linear(e, h), nn.GELU(),
                                     nn.Linear(h, self.mask_patch_decode_size ** 2))
        # Row-major (i,j) grid coords of the R×R patch lattice, shared by every image.
        ii = torch.arange(resolution).repeat_interleave(resolution)
        jj = torch.arange(resolution).repeat(resolution)
        self.register_buffer("ij_base", torch.stack([ii, jj], dim=-1), persistent=False)  # (N,2)

    def _tokens(self, feat, occ, ij, res=None):
        """feat (B,M,Cf); occ (B,M,1); ij (B,M,2) -> (B,M,2,e) = [img-token | mask-token].
        `res` is the grid resolution used to normalize the Fourier position (defaults to the
        token grid T; the scatter refine passes the fine grid Rf)."""
        res = self.resolution if res is None else res
        p = self.pos(ij, res)                          # (B,M,e) Fourier position feature
        img = self.img_embed(feat) + p
        msk = self.mask_embed(occ) + p
        return torch.stack([img, msk], dim=2)

    def _grid_tokens(self, feat_map, B, T, K):
        """(B*T,Cf,R,R) pooled feature map → (sup_feat (B,K·N,Cf), qry_feat (B,N,Cf)), row-major
        cells, image-major support order (context images first, target last)."""
        Cf = feat_map.shape[1]
        feat = feat_map.flatten(2).transpose(1, 2).reshape(B, T, self.N, Cf)  # (B,T,N,Cf)
        return (feat[:, :K].reshape(B, K * self.N, Cf),                       # (B,S,Cf)  S=K·N
                feat[:, K:].reshape(B, self.N, Cf))                           # (B,Q,Cf)  Q=N

    def _occupancy(self, context_out):
        """context_out (B,K,1,H,W) → support mask-token input (B,K·N,p²), image-major, row-major
        cells. p=1: scalar avg-pool occupancy per patch (unchanged default). p>1: each patch's
        mask resampled to a p×p tile. Query prior (= support-mean) is derived inside _attn."""
        B, K = context_out.shape[0], context_out.shape[1]
        H, W = context_out.shape[-2:]
        p = self.mask_patch_size
        if p == 1:
            occ = F.adaptive_avg_pool2d(context_out.reshape(B * K, 1, H, W), (self.resolution,) * 2)
            return occ.reshape(B, K * self.N, 1)
        tiles = torch.stack([_mask_tiles(context_out[:, k], self.resolution, p) for k in range(K)],
                            dim=1)                                     # (B,K,N,p²)
        return tiles.reshape(B, K * self.N, p * p)

    def _segment(self, image, context_in, context_out, mem=None, return_think=False):
        """Coarse single-pass segmentation → (B,1,R,R) logits.

        image (B,1,H,W); context_in/out (B,K,1,H,W). Support = all K·N context patches
        (known mask occupancy); query = the N target patches (mask = support-mean prior).
        mem/return_think are forwarded to _attn (cross-level memory injection)."""
        B, K = context_in.shape[0], context_in.shape[1]
        H, W = image.shape[-2:]
        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)          # (B,T,1,H,W)
        T = imgs.shape[1]
        feat_map = self.encoder(imgs.reshape(B * T, 1, H, W))              # (B*T,Cf,R,R) pooled
        sup_feat, qry_feat = self._grid_tokens(feat_map, B, T, K)
        return self._attn(sup_feat, qry_feat, self._occupancy(context_out), K,
                          mem=mem, return_think=return_think)

    def _similarity_prior(self, qry_feat, sup_feat, sup_occ):
        """Max-cosine similarity prior mask (PFENet, Tian et al. 2020) → (prior, valid).

        qry_feat (B,N,Cf), sup_feat (B,S,Cf), sup_occ (B,S,p²). For each query cell, the MAX
        cosine similarity between its feature and the FOREGROUND support-cell features
        (occupancy≥0.5), min-max normalized per image to [0,1]. `max` (not softmax-mean) is
        imbalance-robust — the whole point for a needle. Returns a DETACHED (B,N) prior and a
        (B,) bool `valid` marking images with ≥1 fg support cell (callers fall back to the flat
        support-mean prior for the rest)."""
        occ = sup_occ.mean(dim=-1)                                   # (B,S) scalar occupancy
        fg = occ >= 0.5                                              # (B,S) foreground cells
        q = F.normalize(qry_feat, dim=-1)
        s = F.normalize(sup_feat, dim=-1)
        sim = torch.bmm(q, s.transpose(1, 2))                       # (B,N,S) cosine
        neg_inf = torch.finfo(sim.dtype).min
        sim = sim.masked_fill(~fg.unsqueeze(1), neg_inf)            # keep only fg exemplars
        prior = sim.max(dim=-1).values                              # (B,N) max-cosine to any fg
        valid = fg.any(dim=-1)                                      # (B,)
        prior = prior.masked_fill(~valid.unsqueeze(1), 0.0)        # degenerate rows -> 0 (finite)
        lo = prior.amin(dim=1, keepdim=True)
        hi = prior.amax(dim=1, keepdim=True)
        prior = (prior - lo) / (hi - lo).clamp_min(1e-6)          # per-image min-max -> [0,1]
        return prior.detach(), valid

    def _attn(self, sup_feat, qry_feat, sup_occ, K, mem=None, return_think=False):
        """Grid path: full R x R query lattice, support-mean prior. Wraps _attn_core with the
        grid defaults so the coarse / single-level output is unchanged.

        When self.sim_prior is set the flat support-mean query prior is replaced by the localized
        max-cosine similarity prior (_similarity_prior), per-image, falling back to the mean for
        images with no foreground support cell. NB: this gate lives here in _attn, so it fires on
        EVERY grid-path pass — the single-level _segment AND the coarse/refine grid passes of the
        two-level refine modes (_refine_encode_once, _refine_scatter). The scatter/flat path
        (_attn_core called directly) is unaffected. The shipped 6_sim_prior config is single-level."""
        B, N = sup_feat.shape[0], self.N
        mean_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])  # (B,Q,p²) flat prior
        if self.sim_prior:
            prior, valid = self._similarity_prior(qry_feat, sup_feat, sup_occ)        # (B,N), (B,)
            prior_tile = prior.unsqueeze(-1).expand(B, N, sup_occ.shape[-1])          # uniform-fill p² tile
            qry_occ = torch.where(valid.view(B, 1, 1), prior_tile, mean_occ)          # fallback for no-fg images
        else:
            qry_occ = mean_occ
        sup_ij = self.ij_base.repeat(K, 1).unsqueeze(0).expand(B, K * N, 2)  # (B,S,2)
        qry_ij = self.ij_base.unsqueeze(0).expand(B, N, 2)                   # (B,Q,2)
        return self._attn_core(sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij,
                               self.resolution, K, self.N, mem=mem,
                               return_think=return_think, flat_out=False)

    def _attn_core(self, sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij, res, K,
                   ctx_count, mem=None, return_think=False, flat_out=False):
        """Set-of-patches attention half over an arbitrary support/query set.

        sup_feat (B,S,Cf), qry_feat (B,Q,Cf), sup_occ (B,S,1), qry_occ (B,Q,1); sup_ij/qry_ij
        are (·,2) grid coords normalized by `res`. ctx_count = patches per context image (N for
        the grid path, M for scatter) — used to broadcast the per-context id embedding.
        flat_out=False -> (B,1,res,res); flat_out=True -> (B,Q). return_think adds (B,n_think,e)."""
        B = sup_feat.shape[0]

        # per-channel standardize features by SUPPORT-patch stats
        mu = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        sup_feat = ((sup_feat - mu) / sig).clamp(-10, 10)
        qry_feat = ((qry_feat - mu) / sig).clamp(-10, 10)

        sup_tok = self._tokens(sup_feat, sup_occ, sup_ij, res)              # (B,S,2,e)
        qry_tok = self._tokens(qry_feat, qry_occ, qry_ij, res)             # (B,Q,2,e)

        if self.context_id_embed:
            assert K <= self.max_context, \
                f"context_size {K} exceeds max_context {self.max_context}"
            e_dim = sup_tok.shape[-1]
            ctx_emb = self.ctx_id(torch.arange(K, device=sup_tok.device))  # (K,e)
            ctx_emb = ctx_emb.repeat_interleave(ctx_count, dim=0)          # (K*ctx_count,e) image-major
            sup_tok = sup_tok + ctx_emb.view(1, K * ctx_count, 1, e_dim)
            qry_tok = qry_tok + self.qry_id.view(1, 1, 1, e_dim)

        sep = K * ctx_count
        rows = [sup_tok, qry_tok]
        if mem is not None:
            T1 = mem.shape[1]
            m = (mem + self.mem_type).unsqueeze(2).expand(mem.shape[0], T1, 2, mem.shape[-1])
            rows = [m] + rows                                              # [memory | support | query]
            sep += T1
        x = torch.cat(rows, dim=1)

        x, sep_t = self.thinking(x, sep)      # -> [thinking | memory | support | query]
        attn_mask = None
        if self.query_self_attn and not self.full_attn:
            r = x.shape[1]
            attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
            attn_mask[:, :sep_t] = True            # all rows -> thinking + support
            attn_mask[sep_t:, sep_t:] = True        # query -> query
        x = self.transformer(x, sep_t, attn_mask=attn_mask, full_attn=self.full_attn)

        q = x[:, sep_t:, 0, :]                                             # (B,Q,e) query img-col
        out = self.decoder(q)                                             # (B,Q,d²)
        d = self.mask_patch_decode_size
        if flat_out:
            assert d == 1, "tiled decode (mask_patch_decode_size>1) unsupported on the scatter/flat path"
            logit = out.squeeze(-1)                                       # (B,Q)
        elif d == 1:
            logit = out.reshape(B, 1, res, res)                          # one logit per token
        else:
            # tile each query cell's d×d block into an (res·d)×(res·d) map (row-major cells,
            # row-major within-tile), i.e. the inverse of _mask_tiles.
            logit = (out.reshape(B, res, res, d, d)
                        .permute(0, 1, 3, 2, 4)
                        .reshape(B, 1, res * d, res * d))
        if return_think:
            return logit, x[:, :self.thinking.n].mean(dim=2)             # (B,n_think,e)
        return logit

    def _select_bbox(self, coarse, context_out, c):
        """coarse logits (B,1,T,T) → (tgt_o (B,2), ctx_o (B,K,2)) top-left px crop origins in the
        image frame: target on its densest PREDICTED mass (detached — selection only, no grad),
        each context on its densest GT. Shared by both refine modes."""
        K = context_out.shape[1]
        H, W = context_out.shape[-2:]
        prob_up = F.interpolate(torch.sigmoid(coarse).detach(), size=(H, W),
                                mode="bilinear", align_corners=False)
        tgt_o = max_sum_window(prob_up, c)                               # (B,2) px origin
        ctx_o = torch.stack([gt_window(context_out[:, k], c) for k in range(K)], dim=1)  # (B,K,2)
        return tgt_o, ctx_o

    def _refine_forward(self, image, context_in, context_out):
        if self.refine_mode == "scatter":
            return self._refine_scatter(image, context_in, context_out)
        if self.refine_mode == "encode_once":
            return self._refine_encode_once(image, context_in, context_out)
        return self._refine_reencode(image, context_in, context_out)

    def _refine_reencode(self, image, context_in, context_out):
        """Coarse pass over the full image + one bbox-zoom refine pass (SAME weights) → per-level
        heads. Crop the target on its densest predicted region and each context on its densest GT,
        resize crops to the encoder input, re-segment at the same T-token grid (2× encoder passes).
        No fusion — levels are supervised/metricked separately (the fused stitch is a metric only)."""
        B, K = context_in.shape[0], context_in.shape[1]
        H, W = image.shape[-2:]
        c = self.refine_crops[0]                                          # derived crop (px)

        if self.refine_memory:
            coarse, coarse_think = self._segment(image, context_in, context_out,
                                                 return_think=True)       # (B,1,T,T), (B,n,e)
        else:
            coarse, coarse_think = self._segment(image, context_in, context_out), None
        tgt_o, ctx_o = self._select_bbox(coarse, context_out, c)

        tgt_img = crop_resize(image, tgt_o, c, H, mode="bilinear")       # (B,1,H,W)
        ctx_img = crop_resize(context_in.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              c, H, mode="bilinear").reshape(B, K, 1, H, W)
        ctx_msk = crop_resize(context_out.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              c, H, mode="nearest").reshape(B, K, 1, H, W)

        mem = coarse_think.detach() if coarse_think is not None else None
        refine = self._segment(tgt_img, ctx_img, ctx_msk, mem=mem)       # (B,1,T,T), same weights
        return {"final_logit": coarse, "refine_logit": refine,
                "refine_origin": tgt_o, "refine_ctx_origin": ctx_o,
                "refine_crop": c, "resolutions": self.resolutions}

    def _refine_encode_once(self, image, context_in, context_out):
        """Encode all K+1 images ONCE (native multi-scale maps), then run the attention half twice:
        on the full-image pooled features (coarse) and on the SAME maps cropped to each bbox and
        pooled back to the T grid (refine). ~half the encoder passes of _refine_reencode; the refine
        pass reuses stem/shallow detail but does NOT recompute deep-stage semantics at the crop."""
        B, K = context_in.shape[0], context_in.shape[1]
        R = self.resolution
        H, W = image.shape[-2:]
        c = self.refine_crops[0]                                          # derived crop (px)
        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)         # (B,T,1,H,W)
        T = imgs.shape[1]

        maps = self.encoder.encode_maps(imgs.reshape(B * T, 1, H, W))     # native multi-scale, ONCE

        # ── coarse: pool the full maps to the token grid ────────────────────────
        sup_c, qry_c = self._grid_tokens(self.encoder.pool_maps(maps, R), B, T, K)
        if self.refine_memory:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K,
                                              return_think=True)             # (B,1,T,T), (B,n,e)
        else:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K), None

        # ── bbox selection, then crop the SAME maps (image-major: contexts, then target) ──
        tgt_o, ctx_o = self._select_bbox(coarse, context_out, c)
        origins = torch.cat([ctx_o, tgt_o.unsqueeze(1)], dim=1).reshape(B * T, 2)  # (B*T,2)
        refine_feat = crop_pool_maps(maps, origins, c, R, H)             # (B*T,Cf,R,R)
        sup_r, qry_r = self._grid_tokens(refine_feat, B, T, K)
        ctx_msk = crop_resize(context_out.reshape(B * K, 1, H, W), ctx_o.reshape(B * K, 2),
                              c, H, mode="nearest").reshape(B, K, 1, H, W)
        mem = coarse_think.detach() if coarse_think is not None else None
        refine = self._attn(sup_r, qry_r, self._occupancy(ctx_msk), K, mem=mem)  # same weights
        return {"final_logit": coarse, "refine_logit": refine,
                "refine_origin": tgt_o, "refine_ctx_origin": ctx_o,
                "refine_crop": c, "resolutions": self.resolutions}

    def _refine_scatter(self, image, context_in, context_out):
        """Coarse pass at T + unconstrained scatter refine at the fine grid Rf.

        Encode once; pool to Rf. Sample M query cells from the coarse prediction (prev_pred) and
        M support cells/context from the true mask fraction; run the attention core on that set.
        Returns per-sampled-cell logits + their flat Rf-grid indices (scattered back downstream)."""
        assert len(self.resolutions) == 2, "scatter refine requires resolutions=[T, Rf]"
        B, K = context_in.shape[0], context_in.shape[1]
        H, W = image.shape[-2:]
        T, Rf = self.resolution, self.resolutions[-1]
        Nf = Rf * Rf
        s = self.sample
        M = int(s["n_total"])
        # Always stochastic (Gumbel neighbor fill), even in eval: deterministic (stochastic=False)
        # top-k ties on the flat proximity field beyond the blurred halo all resolve to the lowest
        # index → a top-left corner dump. eval is seeded upstream (eval_incontext.py sets
        # torch.manual_seed), so stochastic sampling stays reproducible while spreading neighbors.
        stoch = True

        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)          # (B,Timgs,1,H,W)
        Timgs = imgs.shape[1]
        maps = self.encoder.encode_maps(imgs.reshape(B * Timgs, 1, H, W))  # native multi-scale, ONCE

        # ── coarse at the token grid T ──
        sup_c, qry_c = self._grid_tokens(self.encoder.pool_maps(maps, T), B, Timgs, K)
        if self.refine_memory:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K,
                                              return_think=True)             # (B,1,T,T), (B,n,e)
        else:
            coarse, coarse_think = self._attn(sup_c, qry_c, self._occupancy(context_out), K), None

        # ── fine features at Rf ──
        fine = self.encoder.pool_maps(maps, Rf)                            # (B*Timgs,Cf,Rf,Rf)
        Cf = fine.shape[1]
        feat = fine.flatten(2).transpose(1, 2).reshape(B, Timgs, Nf, Cf)   # (B,Timgs,Nf,Cf)

        # ── query: sample from the coarse prediction upsampled to Rf (prev_pred) ──
        coarse_prob = torch.sigmoid(coarse).detach()                       # (B,1,T,T)
        q_map = F.interpolate(coarse_prob, size=(Rf, Rf), mode="bilinear",
                              align_corners=False).reshape(B, Nf)           # (B,Nf)
        qidx, q_is_core, q_is_fg = sample_patches(
            q_map, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
            temperature=s["temperature"], stochastic=stoch,
            n_fg_core=s["n_fg_core"], n_boundary_core=s["n_boundary_core"])
        qry_feat = gather_grid(feat[:, -1], qidx)                          # (B,M,Cf)  target is last
        qry_ij = idx_to_ij(qidx, Rf)                                       # (B,M,2)
        p = self.mask_patch_size
        if p == 1:
            qry_occ = gather_grid(q_map, qidx).unsqueeze(-1)             # (B,M,1) coarse-prob prior
        else:                                                            # p×p coarse-prob tile prior
            qry_occ = gather_grid(_mask_tiles(coarse_prob, Rf, p), qidx)  # (B,M,p²)

        # ── support: sample from each context's true mask fraction at Rf ──
        ctx_frac = F.adaptive_avg_pool2d(context_out.reshape(B * K, 1, H, W),
                                         (Rf, Rf)).reshape(B * K, Nf)       # (B*K,Nf)
        sidx, s_is_core, s_is_fg = sample_patches(
            ctx_frac, M, s["tau"], s["blur_sigma"], s["floor"], Rf,
            temperature=s["temperature"], stochastic=stoch,
            n_fg_core=s["n_fg_core_ctx"], n_boundary_core=s["n_boundary_core"])
        ctx_feat = feat[:, :K].reshape(B * K, Nf, Cf)
        sup_feat = gather_grid(ctx_feat, sidx).reshape(B, K * M, Cf)
        if p == 1:
            sup_occ = gather_grid(ctx_frac, sidx).reshape(B, K * M, 1)
        else:                                                            # p×p true-mask tile per patch
            ctx_tiles = torch.stack([_mask_tiles(context_out[:, k], Rf, p) for k in range(K)],
                                    dim=1).reshape(B * K, Nf, p * p)     # (B*K,Nf,p²)
            sup_occ = gather_grid(ctx_tiles, sidx).reshape(B, K * M, p * p)
        sup_ij = idx_to_ij(sidx, Rf).reshape(B, K * M, 2)

        mem = coarse_think.detach() if coarse_think is not None else None
        refine_logit = self._attn_core(sup_feat, qry_feat, sup_occ, qry_occ, sup_ij, qry_ij,
                                       Rf, K, M, mem=mem, flat_out=True)     # (B,M)
        return {"final_logit": coarse, "refine_logit": refine_logit, "refine_idx": qidx,
                "refine_grid_res": Rf, "resolutions": self.resolutions,
                # tier flags for the scatter qualitative figure (unused by loss/metrics)
                "refine_is_core": q_is_core, "refine_is_fg": q_is_fg,
                "refine_sup_idx": sidx.reshape(B, K, M),
                "refine_sup_is_core": s_is_core.reshape(B, K, M),
                "refine_sup_is_fg": s_is_fg.reshape(B, K, M)}

    def forward(self, image, context_in, context_out, mode="train"):
        """image (B,1,H,W); context_in/out (B,K,1,H,W).

        Single level (len(resolutions)==1): {"final_logit": (B,1,T,T)} — the plain model.
        Multi level: per-level heads (final_logit=coarse, refine_logit, refine_origin,
        refine_ctx_origin, refine_crop, resolutions). `mode` is accepted for interface
        parity; unused."""
        if len(self.resolutions) == 1:
            return {"final_logit": self._segment(image, context_in, context_out)}
        return self._refine_forward(image, context_in, context_out)

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patchset_pfn import FourierPositionalEncoding
from src.models.pfn_seg_2d import ThinkingRows, TransformerEncoderStack


class ConvEncoder(nn.Module):
    """Single-stream conv encoder with multi-scale feature concatenation.

    (B, in_ch, H, W) → (B, sum(dims), R, R). One feature map is produced per stage
    (stem at full res, then one per stride-2 stage); every map is avg-pooled to the
    bottleneck resolution R and concatenated along channels, so each output patch
    token carries low- through high-level features (cf. UniverSeg's per-stage 64-wide
    encoder blocks). `dims` are the per-stage channel widths. Each stage is a strided
    conv + a same-resolution conv (GroupNorm + LeakyReLU); one stream, no cross-conv.
    """
    def __init__(self, in_ch: int, dims: tuple[int, ...], image_size: int,
                 resolution: int, groups: int = 8):
        super().__init__()
        self.resolution = resolution
        n_down = int(round(math.log2(image_size / resolution)))
        assert 2 ** n_down * resolution == image_size, \
            f"image_size {image_size} must be resolution {resolution} × a power of 2"
        assert len(dims) == n_down + 1, \
            f"need len(dims)={n_down + 1} for {image_size}→{resolution}, got {len(dims)}"

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.resolution
        feats = [self.stem(x)]                         # full-res stem features
        for stage in self.stages:
            feats.append(stage(feats[-1]))             # successively downsampled stages
        # Pool every scale to R×R (last stage is already R) and concat along channels.
        feats = [f if f.shape[-1] == R else F.adaptive_avg_pool2d(f, (R, R)) for f in feats]
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
    ):
        super().__init__()
        self.image_size = image_size
        self.resolution = resolution
        self.N = resolution ** 2
        self.query_self_attn = query_self_attn
        self.encoder = ConvEncoder(1, tuple(enc_dims), image_size, resolution)
        self.img_embed = nn.Linear(self.encoder.out_ch, e)
        self.mask_embed = nn.Linear(1, e)              # scalar occupancy → e
        self.pos = FourierPositionalEncoding(e, fourier_bands)   # (i,j) → e, added to both cols
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

    def forward(self, image, context_in, context_out, mode="train"):
        """image (B,1,H,W); context_in/out (B,K,1,H,W). Returns {"final_logit": (B,1,R,R)}.

        Support = all K·N context patches (known mask occupancy); query = the N target
        patches (mask = support-mean prior). `mode` is accepted for interface parity
        with the UniverSeg baseline and is otherwise unused.
        """
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
        return {"final_logit": logit}

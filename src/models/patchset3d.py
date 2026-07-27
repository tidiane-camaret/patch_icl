"""PatchSet3D: low-resolution in-context 3D segmentation (set-of-patches attention).

3D analog of src/models/patchset_cnn.py's single-level path. A ConvEncoder3D
downsamples each volume to an R³ feature grid; every patch of every volume becomes a
token in a set, and the dimension-agnostic dual-axis transformer (pfn_seg_2d) does
content-based in-context matching over that set. Position is a Fourier feature of the
(i,j,k) cell, not a tensor axis, so the transformer core is reused verbatim.

Single level only: prediction at the token grid R (mask_patch_decode_size=1) or tiled
to (R·d)³ (d>1). Refine / sim_prior / Muon-LAWA are intentionally omitted
(see docs/superpowers/specs/2026-07-22-patchset3d-design.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patchset_pfn import FourierPositionalEncoding
from src.models.pfn_seg_2d import ThinkingRows, TransformerEncoderStack


def _down_to(f: torch.Tensor, R: int) -> torch.Tensor:
    """Resample a feature/mask volume to R^3. When the source side is an exact integer
    multiple of R, a strided avg_pool3d gives the identical result as adaptive_avg_pool3d
    but ~3x faster (incl. backward) at large strides (e.g. 128->16); adaptive is the
    fallback for non-divisible sides, trilinear the fallback for upsampling."""
    src = f.shape[-1]
    if src == R:
        return f
    if src > R:
        if src % R == 0:
            k = src // R
            return F.avg_pool3d(f, k, k)
        return F.adaptive_avg_pool3d(f, (R, R, R))
    return F.interpolate(f, size=(R, R, R), mode="trilinear", align_corners=False)


def _mask_tiles_3d(mask: torch.Tensor, grid_res: int, p: int) -> torch.Tensor:
    """(B,1,Df,Hf,Wf) -> (B, grid_res**3, p**3): per-cell p³ mask tile, row-major cells.

    Resizes to grid_res*p (trilinear) when needed; exact reshape when already there.
    3D analog of patchset_cnn._mask_tiles."""
    target = grid_res * p
    if mask.shape[-3:] != (target, target, target):
        mask = F.interpolate(mask.float(), size=(target, target, target),
                             mode="trilinear", align_corners=False)
    B = mask.shape[0]
    return (mask.reshape(B, 1, grid_res, p, grid_res, p, grid_res, p)
                .permute(0, 2, 4, 6, 3, 5, 7, 1)
                .reshape(B, grid_res ** 3, p ** 3))


class ConvEncoder3D(nn.Module):
    """Single-stream 3D conv encoder with multi-scale feature concat (3D ConvEncoder).

    (B,in_ch,D,H,W) -> (B, sum(dims), R,R,R). Depth = len(dims)-1 stride-2 stages after a
    full-res stem; every scale is resampled to R³ (adaptive_avg_pool3d down, trilinear up)
    and concatenated on channels."""
    def __init__(self, in_ch: int, dims: tuple[int, ...], resolution: int, groups: int = 8):
        super().__init__()
        assert len(dims) >= 1, "dims needs at least a stem width"
        self.resolution = resolution
        n_down = len(dims) - 1

        def cbr(ci, co, stride):
            return nn.Sequential(
                nn.Conv3d(ci, co, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(groups, co),
                nn.LeakyReLU(0.1, inplace=True),
            )

        self.stem = cbr(in_ch, dims[0], 1)
        self.stages = nn.ModuleList([
            nn.Sequential(cbr(dims[i], dims[i + 1], 2), cbr(dims[i + 1], dims[i + 1], 1))
            for i in range(n_down)
        ])
        self.out_ch = sum(dims)

    def _resample(self, f: torch.Tensor, R: int) -> torch.Tensor:
        return _down_to(f, R)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.stem(x)]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return torch.cat([self._resample(f, self.resolution) for f in feats], dim=1)


class PatchSet3D(nn.Module):
    def __init__(
        self,
        resolution: int = 16,
        enc_dims: tuple[int, ...] = (32, 32, 32, 32),
        e: int = 256,
        h: int = 512,
        l: int = 6,
        a: int = 4,
        thinking_rows: int = 8,
        residual_decay: float = 0.95,
        fourier_bands: int = 8,
        mask_patch_size: int = 1,
        mask_patch_decode_size: int = 1,
        context_id_embed: bool = False,
        max_context: int = 16,
        full_attn: bool = False,
        query_self_attn: bool = False,
        image_size=None,
        encoder: str = "conv",
        encoder_frozen: bool = True,
        primus_sidecar: str = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.N = resolution ** 3
        self.mask_patch_size = int(mask_patch_size)
        self.mask_patch_decode_size = int(mask_patch_decode_size)
        assert self.mask_patch_size >= 1 and self.mask_patch_decode_size >= 1
        self.full_attn = full_attn
        self.query_self_attn = query_self_attn
        self.context_id_embed = context_id_embed
        self.max_context = max_context
        self.image_size = image_size          # metadata only (unused in forward)

        if encoder == "primus":
            if not primus_sidecar:
                raise ValueError("encoder='primus' requires arch.primus_sidecar")
            from src.models.primus_encoder import PrimusEncoder   # lazy: avoids import cycle
            self.encoder = PrimusEncoder(primus_sidecar, resolution,
                                         frozen=encoder_frozen, device="cpu")
        elif encoder == "conv":
            self.encoder = ConvEncoder3D(1, tuple(enc_dims), resolution)
        else:
            raise ValueError(f"unknown arch.encoder {encoder!r} (conv | primus)")
        self.img_embed = nn.Linear(self.encoder.out_ch, e)
        self.mask_embed = nn.Linear(self.mask_patch_size ** 3, e)   # occupancy tile p³ -> e
        self.pos = FourierPositionalEncoding(e, fourier_bands, n_axes=3)
        if context_id_embed:
            self.ctx_id = nn.Embedding(max_context, e)
            self.qry_id = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.ctx_id.weight, std=0.1)
            nn.init.normal_(self.qry_id, std=0.1)
        self.thinking = ThinkingRows(thinking_rows, e)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
        self.decoder = nn.Sequential(nn.Linear(e, h), nn.GELU(),
                                     nn.Linear(h, self.mask_patch_decode_size ** 3))
        # (i,j,k) lattice, row-major over R³ (cell index n = i*R² + j*R + k)
        r = resolution
        ii = torch.arange(r).repeat_interleave(r * r)
        jj = torch.arange(r).repeat_interleave(r).repeat(r)
        kk = torch.arange(r).repeat(r * r)
        self.register_buffer("ijk_base", torch.stack([ii, jj, kk], dim=-1), persistent=False)  # (N,3)

    @property
    def grid_size(self) -> int:
        return self.resolution * self.mask_patch_decode_size

    def _grid_tokens(self, feat_map, B, T, K):
        """(B*T,Cf,R,R,R) -> (support (B,K·N,Cf), query (B,N,Cf)), image-major, row-major cells."""
        Cf = feat_map.shape[1]
        feat = feat_map.flatten(2).transpose(1, 2).reshape(B, T, self.N, Cf)
        return (feat[:, :K].reshape(B, K * self.N, Cf), feat[:, K:].reshape(B, self.N, Cf))

    def _occupancy(self, context_out):
        """context_out (B,K,D,H,W) -> support mask-token input (B, K·N, p³)."""
        B, K = context_out.shape[0], context_out.shape[1]
        p = self.mask_patch_size
        if p == 1:
            D, H, W = context_out.shape[-3:]
            occ = _down_to(context_out.reshape(B * K, 1, D, H, W).float(), self.resolution)
            return occ.reshape(B, K * self.N, 1)
        tiles = torch.stack([_mask_tiles_3d(context_out[:, k].unsqueeze(1).float(),
                                            self.resolution, p) for k in range(K)], dim=1)
        return tiles.reshape(B, K * self.N, p ** 3)

    def _tokens(self, feat, occ, ijk):
        pos = self.pos(ijk, self.resolution)
        img = self.img_embed(feat) + pos
        msk = self.mask_embed(occ) + pos
        return torch.stack([img, msk], dim=2)               # (B,M,2,e)

    def _tile_logits(self, out):
        """(B,N,d³) -> (B,1,Rd,Rd,Rd), inverse of _mask_tiles_3d (d=1 -> one logit per cell)."""
        B = out.shape[0]
        r, d = self.resolution, self.mask_patch_decode_size
        if d == 1:
            return out.reshape(B, 1, r, r, r)
        return (out.reshape(B, r, r, r, d, d, d)
                   .permute(0, 1, 4, 2, 5, 3, 6)
                   .reshape(B, r * d, r * d, r * d)
                   .unsqueeze(1))

    def _attn(self, sup_feat, qry_feat, sup_occ, K):
        B, N = sup_feat.shape[0], self.N
        qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])  # support-mean prior
        sup_ijk = self.ijk_base.repeat(K, 1).unsqueeze(0).expand(B, K * N, 3)
        qry_ijk = self.ijk_base.unsqueeze(0).expand(B, N, 3)

        mu = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        sup_feat = ((sup_feat - mu) / sig).clamp(-10, 10)
        qry_feat = ((qry_feat - mu) / sig).clamp(-10, 10)

        sup_tok = self._tokens(sup_feat, sup_occ, sup_ijk)   # (B,S,2,e)
        qry_tok = self._tokens(qry_feat, qry_occ, qry_ijk)   # (B,Q,2,e)

        if self.context_id_embed:
            assert K <= self.max_context, f"context_size {K} exceeds max_context {self.max_context}"
            e_dim = sup_tok.shape[-1]
            ctx_emb = self.ctx_id(torch.arange(K, device=sup_tok.device)).repeat_interleave(N, dim=0)
            sup_tok = sup_tok + ctx_emb.view(1, K * N, 1, e_dim)
            qry_tok = qry_tok + self.qry_id.view(1, 1, 1, e_dim)

        sep = K * N
        x = torch.cat([sup_tok, qry_tok], dim=1)
        x, sep_t = self.thinking(x, sep)
        attn_mask = None
        if self.query_self_attn and not self.full_attn:
            r = x.shape[1]
            attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
            attn_mask[:, :sep_t] = True
            attn_mask[sep_t:, sep_t:] = True
        x = self.transformer(x, sep_t, attn_mask=attn_mask, full_attn=self.full_attn)
        q = x[:, sep_t:, 0, :]                               # (B,Q,e) query img-col
        return self._tile_logits(self.decoder(q))           # (B,1,Rd,Rd,Rd)

    def forward(self, image, context_in, context_out, mode="train"):
        B, K = context_in.shape[0], context_in.shape[1]
        D, H, W = image.shape[-3:]
        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)     # (B,T,1,D,H,W)
        T = imgs.shape[1]
        feat_map = self.encoder(imgs.reshape(B * T, 1, D, H, W))       # (B*T,Cf,R,R,R)
        sup_feat, qry_feat = self._grid_tokens(feat_map, B, T, K)
        logit = self._attn(sup_feat, qry_feat, self._occupancy(context_out), K)
        return {"final_logit": logit}

    def _native_logit(self, image, context_in, context_out):
        dev = next(self.parameters()).device
        image = image.to(dev); context_in = context_in.to(dev); context_out = context_out.to(dev)
        logit = self.forward(image, context_in, context_out)["final_logit"].float()
        return F.interpolate(logit, size=image.shape[-3:], mode="trilinear", align_corners=False)

    def train_forward(self, target_img, context_imgs, context_masks):
        """Native-resolution logits (B,1,D,H,W) — used by the val soft-Dice / loss path."""
        return self._native_logit(target_img, context_imgs, context_masks)

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks):
        """Native binary mask (B,D,H,W) — used by the eval Dice path."""
        logit = self._native_logit(target_img, context_imgs, context_masks)
        return (torch.sigmoid(logit) >= 0.5).float().squeeze(1)

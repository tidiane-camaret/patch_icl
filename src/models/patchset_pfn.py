"""
PatchSetPFN: stage-2 patch-set in-context refinement model.

A nanoTabPFN-shaped transformer: rows = sampled patches, cols = [img-token | mask-token].
Reuses ImagePFN's dual-axis TransformerEncoderStack and ThinkingRows.

  - img-token  = Linear(feature_dim → e) on the patch's frozen-encoder feature
  - mask-token = Linear(mask_dim → e) on the patch's mask, per mask_prior:
                 false  : scalar; query prior = support-mean (neutral, TargetEncoder analog)
                 scalar : scalar; query prior = coarse prediction
                 patch  : p×p mask tile (p auto); support = native GT tile,
                          query = upsampled coarse-prior tile (no detail below the
                          stage-1 resolution, but boundary geometry is exact for support)
  - 2-D Fourier positional encoding of the patch's (i,j) grid cell, added to both
    tokens (resolution-generalizable: normalized coords + fixed frequencies)
  - optional stage-1 memory: the frozen stage-1 model's post-transformer thinking
    rows (mean-pooled over columns → (B, n_think, e1)) are projected to e and
    prepended as extra support rows, so query patches attend to them
  - sample-axis attention: query patches attend to thinking + (stage-1 memory +)
    support rows only
  - decoder reads each query's img-col → per-query logit
"""

import math

import torch
import torch.nn as nn

from src.models.pfn_seg_2d import ThinkingRows, TransformerEncoderStack


class FourierPositionalEncoding(nn.Module):
    """2-D Fourier features of normalized (i,j) → Linear → e. Resolution-generalizable."""
    def __init__(self, e: int, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        freqs = 2.0 ** torch.arange(num_bands).float()      # (L,) geometric: 1,2,4,...
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(4 * num_bands, e)

    def forward(self, ij: torch.Tensor, grid_res: int) -> torch.Tensor:
        # ij: (..., 2) integer cell coords on a grid_res×grid_res grid
        uv  = (ij.float() + 0.5) / grid_res                 # (...,2) in (0,1)
        ang = 2 * math.pi * uv.unsqueeze(-1) * self.freqs   # (...,2,L)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)   # (...,2,2L)
        feats = feats.flatten(-2)                           # (...,4L)
        return self.proj(feats)                             # (...,e)


class PatchSetPFN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        e: int = 256,
        h: int = 512,
        l: int = 6,
        a: int = 4,
        thinking_rows: int = 8,
        residual_decay: float = 0.95,
        fourier_bands: int = 8,
        mask_prior: str = "scalar",      # false | scalar | patch
        mask_patch_size: int = 1,        # p; mask-token input dim = p² (p=1 ⇒ scalar)
        stage1_dim: int | None = None,
        query_self_attn: bool = False,
    ):
        super().__init__()
        assert mask_prior in ("false", "scalar", "patch"), mask_prior
        self.mask_prior = mask_prior
        self.mask_patch_size = mask_patch_size if mask_prior == "patch" else 1
        self.query_self_attn = query_self_attn
        self.img_embed  = nn.Linear(feature_dim, e)
        self.mask_embed = nn.Linear(self.mask_patch_size ** 2, e)
        self.pos        = FourierPositionalEncoding(e, fourier_bands)
        self.thinking   = ThinkingRows(thinking_rows, e)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
        self.decoder    = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, 1))
        # Optional stage-1 thinking memory: project e1→e, plus a learned type token
        # marking these rows as stage-1 memory (they have no patch (i,j) location).
        self.stage1_dim = stage1_dim
        if stage1_dim is not None:
            self.stage1_proj = nn.Linear(stage1_dim, e)
            self.stage1_type = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.stage1_type, std=0.02)

    def _tokens(self, feat, mask, ij, grid_res):
        # feat (B,R,F); mask (B,R) scalar or (B,R,p²); ij (B,R,2) → (B,R,2,e)
        p   = self.pos(ij, grid_res)                        # (B,R,e)
        img = self.img_embed(feat) + p
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)                        # (B,R,1)
        msk = self.mask_embed(mask) + p
        return torch.stack([img, msk], dim=2)               # (B,R,2,e)

    def forward(self, sup_feat, sup_label, sup_ij,
                qry_feat, qry_prior, qry_ij, grid_res, stage1_think=None, return_thinking=False):
        B, S, _ = sup_feat.shape
        Q = qry_feat.shape[1]

        # Per-channel feature normalization using support statistics.
        mu  = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        sup_feat = ((sup_feat - mu) / sig).clamp(-10, 10)
        qry_feat = ((qry_feat - mu) / sig).clamp(-10, 10)

        # Query mask prior: coarse pred ("scalar"/"patch"), or support-mean ("false").
        if self.mask_prior == "false":
            m = sup_label.mean(dim=1, keepdim=True)          # (B,1) or (B,1,p²)
            qry_prior = m.expand(B, Q, *m.shape[2:])

        sup_tok = self._tokens(sup_feat, sup_label, sup_ij, grid_res)   # (B,S,2,e)
        qry_tok = self._tokens(qry_feat, qry_prior, qry_ij, grid_res)   # (B,Q,2,e)

        # Stage-1 memory rows: project (B,T1,e1)→(B,T1,e), broadcast to both cols.
        # Prepended to the support block so queries attend to them (inside sep).
        sep = S
        rows = [sup_tok, qry_tok]
        if stage1_think is not None and self.stage1_dim is not None:
            T1 = stage1_think.shape[1]
            s1 = self.stage1_proj(stage1_think) + self.stage1_type        # (B,T1,e)
            s1_tok = s1.unsqueeze(2).expand(B, T1, 2, s1.shape[-1])        # (B,T1,2,e)
            rows = [s1_tok] + rows
            sep += T1
        x = torch.cat(rows, dim=1)                                        # (B, (T1+)S+Q, 2, e)

        x, sep_t = self.thinking(x, sep)        # prepend thinking rows; sep_t = n_think + sep

        # Within-image spatial reasoning: let query patches attend to each other
        # (keyed by their Fourier positions) in addition to the train set. Support
        # rows still attend only to the train set [:sep_t]; queries carry the coarse
        # prior (not GT) so query↔query attention leaks no labels.
        attn_mask = None
        if self.query_self_attn:
            r = x.shape[1]
            attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
            attn_mask[:, :sep_t] = True          # all rows → train set (thinking+memory+support)
            attn_mask[sep_t:, sep_t:] = True      # queries → queries (within-image)
        x = self.transformer(x, sep_t, attn_mask=attn_mask)

        q = x[:, sep_t:, 0, :]                  # query rows, img-col → (B,Q,e)
        out = self.decoder(q).squeeze(-1)       # (B,Q)
        if return_thinking:
            think = x[:, :self.thinking.n].mean(dim=2)   # (B, n_think, e)
            return out, think
        return out

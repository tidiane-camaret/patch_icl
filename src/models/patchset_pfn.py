"""
PatchSetPFN: stage-2 patch-set in-context refinement model.

A nanoTabPFN-shaped transformer: rows = sampled patches, cols = [img-token | mask-token].
Reuses ImagePFN's dual-axis TransformerEncoderStack and ThinkingRows.

  - img-token  = Linear(feature_dim → e) on the patch's frozen-encoder feature
  - mask-token = Linear(1 → e) on the patch's mask value (support: true fraction;
                 query: coarse prediction if coarse_prior else support-mean prior)
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
        coarse_prior: bool = True,
        stage1_dim: int | None = None,
        query_self_attn: bool = False,
    ):
        super().__init__()
        self.coarse_prior = coarse_prior
        self.query_self_attn = query_self_attn
        self.img_embed  = nn.Linear(feature_dim, e)
        self.mask_embed = nn.Linear(1, e)
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

    def _tokens(self, feat, label, ij, grid_res):
        # feat (B,R,F), label (B,R), ij (B,R,2) → (B,R,2,e)
        p   = self.pos(ij, grid_res)                        # (B,R,e)
        img = self.img_embed(feat) + p
        msk = self.mask_embed(label.unsqueeze(-1)) + p
        return torch.stack([img, msk], dim=2)               # (B,R,2,e)

    def forward(self, sup_feat, sup_label, sup_ij,
                qry_feat, qry_prior, qry_ij, grid_res, stage1_think=None):
        B, S, _ = sup_feat.shape
        Q = qry_feat.shape[1]

        # Per-channel feature normalization using support statistics.
        mu  = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        sup_feat = ((sup_feat - mu) / sig).clamp(-10, 10)
        qry_feat = ((qry_feat - mu) / sig).clamp(-10, 10)

        # Query mask prior: coarse pred, or the support-mean fraction (TargetEncoder analog).
        if not self.coarse_prior:
            qry_prior = sup_label.mean(dim=1, keepdim=True).expand(B, Q)

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
        return self.decoder(q).squeeze(-1)      # (B,Q)

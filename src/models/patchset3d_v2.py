"""PatchSetV2: clean Iris-style in-context 3D segmentation.

See docs/superpowers/specs/2026-09-19-patchset-v2-design.md for the full design. Reuses
RowCrossAttention / TransformerEncoderStack / ThinkingRows from pfn_seg_2d.py and the
encoder classes via build_encoder, but is a fresh, minimal class -- not another branch on
PatchSet3D's own accreted knob surface.

Per forward call: target volume (image + prior/prediction mask) + K context volumes
(image + real GT mask), T = K+1 total. Every volume is tokenized, foreground-pooled at
native resolution, and compressed to `compress_m` tokens (Stage A, weight-shared across
volumes) via the same RowCrossAttention arch.seq_compress already uses in PatchSet3D. All
volumes' compressed sequences run through one shared self-attention stack (Stage B). The
target's own post-Stage-B tokens are Iris's task tokens T; they cross-attend (Eq 5) against
the target's raw, never-compressed per-cell grid, and a conv up-path with encoder-pyramid
skips (Eq 6) produces the final logits. No decompression step exists -- the target's raw
grid was never discarded in the first place.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.factory import build_encoder
from src.models.patchset3d import MaskConvEmbed, _ConvNormAct, _down_to, _mask_tiles_3d
from src.models.patchset_pfn import FourierPositionalEncoding
from src.models.pfn_seg_2d import RowCrossAttention, ThinkingRows, TransformerEncoderStack


class PatchSetV2(nn.Module):
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
        mask_embed: str = "linear",
        compress_m: int = 32,
        compress_layers: int = 1,
        context_id_embed: bool = True,
        max_context: int = 16,
        cascade_registers: bool = False,
        image_size=None,
        encoder: str = "conv",
        encoder_frozen: bool = True,
        primus_sidecar: str | None = None,
        nnunet_ts_weights: str | None = None,
        nnunet_ts_stages=(2, 3, 4),
        nnunet_ts_random_init: bool = False,
        resenc_n_stages: int = 5,
        plainconv_ts_n_stages: int = 5,
        plainconv_ts_features_per_stage=None,
        encoder_input_norm: str | None = None,
        encoder_stage: int | None = None,
        encoder_native_grid: bool = False,
        encoder_spacing_aware: bool = False,
        encoder_precision: str = "bf16",
        fine_stage=(0, 1),
        decoder_dim: int = 64,
    ):
        super().__init__()
        self.resolution = resolution
        self.N = resolution ** 3
        self.mask_patch_size = int(mask_patch_size)
        assert self.mask_patch_size >= 1
        self.compress_m = int(compress_m)
        assert self.compress_m >= 1 and compress_layers >= 1
        self.context_id_embed = bool(context_id_embed)
        self.max_context = int(max_context)
        self.cascade_registers = bool(cascade_registers)
        self.spacing_aware = bool(encoder_spacing_aware)

        self.encoder = build_encoder(
            encoder, resolution, in_ch=1, enc_dims=enc_dims, encoder_frozen=encoder_frozen,
            primus_sidecar=primus_sidecar, nnunet_ts_weights=nnunet_ts_weights,
            nnunet_ts_stages=nnunet_ts_stages, nnunet_ts_random_init=nnunet_ts_random_init,
            resenc_n_stages=resenc_n_stages, plainconv_ts_n_stages=plainconv_ts_n_stages,
            plainconv_ts_features_per_stage=plainconv_ts_features_per_stage,
            encoder_input_norm=encoder_input_norm, image_size=image_size,
            encoder_stage=encoder_stage, encoder_native_grid=encoder_native_grid,
            encoder_spacing_aware=encoder_spacing_aware, encoder_precision=encoder_precision)
        if not getattr(self.encoder, "supports_fine", False):
            raise ValueError(f"PatchSetV2 needs an encoder exposing unpooled stages; "
                             f"encoder={encoder!r} has none (use conv | nnunet_ts | "
                             f"resenc_ts | plainconv_ts)")

        self.fine_stage = tuple(int(st) for st in fine_stage)
        for st in self.fine_stage:
            if not 0 <= st < self.encoder.n_fine_stages:
                raise ValueError(f"fine_stage {st} out of range [0, {self.encoder.n_fine_stages})")
        if not image_size:
            raise ValueError("PatchSetV2 needs arch.image_size (from data.image_size)")
        stages = sorted(self.fine_stage,
                        key=lambda st: self.encoder.fine_stage_size(int(image_size[0]), st))
        self._stage_order = [self.fine_stage.index(st) for st in stages]   # coarse->fine, into `fine`
        self._stage_sides = [self.encoder.fine_stage_size(int(image_size[0]), st) for st in stages]
        self._stage_chans = [self.encoder.fine_stage_channels(st) for st in stages]
        self._finest_idx = self._stage_order[-1]      # index into `fine` for the finest stage
        self.pool_proj = nn.Linear(self._stage_chans[-1], e)

        self.compress_slots = nn.Parameter(torch.empty(self.compress_m, e))
        nn.init.normal_(self.compress_slots, std=0.02)
        self.compressor = nn.ModuleList(
            [RowCrossAttention(a, e, h) for _ in range(compress_layers)])

        oc = self.encoder.out_ch
        self.img_embed = nn.Linear(oc, e)
        assert mask_embed in ("linear", "conv"), f"mask_embed={mask_embed!r} — 'linear' or 'conv'"
        self.mask_embed = (MaskConvEmbed(self.mask_patch_size, e) if mask_embed == "conv"
                           else nn.Linear(self.mask_patch_size ** 3, e))
        self.pos = FourierPositionalEncoding(e, fourier_bands, n_axes=3)

        r = resolution
        ii = torch.arange(r).repeat_interleave(r * r)
        jj = torch.arange(r).repeat_interleave(r).repeat(r)
        kk = torch.arange(r).repeat(r * r)
        self.register_buffer("ijk_base", torch.stack([ii, jj, kk], dim=-1), persistent=False)

        if self.context_id_embed:
            self.ctx_id = nn.Embedding(self.max_context, e)
            self.qry_id = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.ctx_id.weight, std=0.1)
            nn.init.normal_(self.qry_id, std=0.1)

        self.thinking = ThinkingRows(thinking_rows, e)
        if self.cascade_registers:
            self.cascade_proj = nn.Linear(e, e)
            self.cascade_type = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.cascade_type, std=0.02)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)

    def _tokens_all(self, feat: torch.Tensor, occ: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """feat (B,T,N,Cf) raw encoder grid tokens, occ (B,T,N,p^3) mask/prior occupancy ->
        (B,T,N,2,e): img_embed + mask_embed + Fourier positional encoding, columns
        [img, mask]. No content-type tag (mask_slots) -- context_id_embed, added later on
        the assembled per-volume sequence, already distinguishes target from context rows
        (see docs/superpowers/specs/2026-09-19-patchset-v2-design.md)."""
        img = self.img_embed(feat)
        msk = self.mask_embed(occ)
        ijk = self.ijk_base.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        pos = self.pos(ijk, self.resolution)
        img = img + pos
        msk = msk + pos
        return torch.stack([img, msk], dim=3)

    def _occupancy(self, context_out: torch.Tensor) -> torch.Tensor:
        """context_out (B,K,D,H,W) -> (B,K,N,p^3), same _down_to/_mask_tiles_3d path
        PatchSet3D._occupancy uses, kept volume-major (not flattened) to match this
        class's (B,T,...) canonical shape."""
        B, K = context_out.shape[0], context_out.shape[1]
        p = self.mask_patch_size
        if p == 1:
            Dn, Hn, Wn = context_out.shape[-3:]
            occ = _down_to(context_out.reshape(B * K, 1, Dn, Hn, Wn).float(), self.resolution)
            return occ.reshape(B, K, self.N, 1)
        tiles = torch.stack([_mask_tiles_3d(context_out[:, k].unsqueeze(1).float(),
                                            self.resolution, p) for k in range(K)], dim=1)
        return tiles.reshape(B, K, self.N, p ** 3)

    def _prior_occupancy(self, prior: torch.Tensor) -> torch.Tensor:
        """(B,1,D,H,W) soft prior -> (B,1,N,p^3), same convention as _occupancy."""
        B = prior.shape[0]
        p = self.mask_patch_size
        prior = prior.reshape(B, 1, *prior.shape[-3:]).float()
        if p == 1:
            return _down_to(prior, self.resolution).reshape(B, 1, self.N, 1)
        return _mask_tiles_3d(prior, self.resolution, p).reshape(B, 1, self.N, p ** 3)

    def _pool_all(self, fine_finest: torch.Tensor, context_out: torch.Tensor,
                 query_prior: torch.Tensor | None, B: int, K: int, T: int) -> torch.Tensor:
        """Iris Eq 2 foreground-masked average, generalized to every volume (not
        support-only). fine_finest: (B*T,Cf,S,S,S), the finest requested fine_stage map for
        ALL T volumes. Upsamples to the volume's NATIVE (D,H,W) resolution -- never R, never
        S -- before masking: Iris's own ablation credits masking-after-upsampling with the
        small-object Dice gain (docs/methods/iris.md sec 4.1). Returns (B,T,e)."""
        Cf = fine_finest.shape[1]
        Dn, Hn, Wn = context_out.shape[-3:]
        feat_native = F.interpolate(fine_finest.float(), size=(Dn, Hn, Wn),
                                    mode="trilinear", align_corners=False
                                    ).reshape(B, T, Cf, Dn, Hn, Wn)
        sup_mask = context_out.reshape(B, K, 1, Dn, Hn, Wn).float()
        if query_prior is not None:
            qry_mask = query_prior.reshape(B, 1, 1, Dn, Hn, Wn).float()
        else:
            qry_mask = sup_mask.mean(dim=1, keepdim=True)
        mask = torch.cat([sup_mask, qry_mask], dim=1)          # (B,T,1,Dn,Hn,Wn)

        mu = feat_native.mean(dim=(-3, -2, -1), keepdim=True)
        sig = feat_native.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
        feat_z = ((feat_native - mu) / sig).clamp(-10, 10)

        num = (feat_z * mask).sum(dim=(-3, -2, -1))             # (B,T,Cf)
        den = mask.sum(dim=(-3, -2, -1)).clamp_min(1e-6)         # (B,T,1)
        return self.pool_proj(num / den)                        # (B,T,e)

    def _compress_all(self, tok: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """tok (B,T,N,2,e) -> (B,T,compress_m,2,e). Weight-shared per-volume compression
        (Stage A): compress_m learnable slots cross-attend into that volume's own N raw
        cells only (T folded into the batch dim -- no cross-volume mixing here; that's
        Stage B's job). Identical mechanism to PatchSet3D's arch.seq_compress Stage A."""
        e = tok.shape[-1]
        kv = tok.reshape(B * T, self.N, 2, e)
        q = self.compress_slots.unsqueeze(0).unsqueeze(2).expand(B * T, -1, 2, -1).contiguous()
        for layer in self.compressor:
            q = layer(q, kv)
        return q.reshape(B, T, self.compress_m, 2, e)

    def _assemble_sequence(self, pool: torch.Tensor, compressed: torch.Tensor,
                           B: int, T: int) -> torch.Tensor:
        """pool (B,T,e), compressed (B,T,compress_m,2,e) -> (B, T*(compress_m+1), 2, e),
        volume-major: each volume's block is [pool_row ; compress_m rows], contiguous, so a
        later slice by volume index recovers exactly that volume's tokens. pool is
        broadcast into both img and mask columns."""
        e = compressed.shape[-1]
        pool_tok = pool.unsqueeze(2).unsqueeze(3).expand(B, T, 1, 2, e)
        seq = torch.cat([pool_tok, compressed], dim=2)          # (B,T,compress_m+1,2,e)
        return seq.reshape(B, T * (self.compress_m + 1), 2, e)

    def _apply_context_tags(self, seq: torch.Tensor, B: int, K: int, T: int,
                            per_vol: int) -> torch.Tensor:
        """seq (B, T*per_vol, 2, e). Adds ctx_id[k] to each context volume's block, qry_id
        to the target's block -- lets Stage B's self-attention tell context rows from the
        target row. This also distinguishes GT-content rows from prediction-content rows
        without a separate mask_slots-style tag, since target vs. context identity implies
        which content type that volume's mask column holds (see
        docs/superpowers/specs/2026-09-19-patchset-v2-design.md)."""
        ctx = self.ctx_id(torch.arange(K, device=seq.device))         # (K,e)
        tags = torch.cat([ctx, self.qry_id.unsqueeze(0)], dim=0)      # (T,e)
        tags = tags.repeat_interleave(per_vol, dim=0)                  # (T*per_vol,e)
        return seq + tags.to(seq.dtype).view(1, -1, 1, seq.shape[-1])

    def _stage_b(self, seq: torch.Tensor, B: int, K: int, T: int,
                cascade_regs: torch.Tensor | None = None):
        """seq (B, T*per_vol, 2, e) -> (post-transformer sequence, registers). Full
        self-attention across every volume's tokens (no register_routed, no full_attn
        toggle -- this is the only mode) plus thinking rows and, if enabled, the previous
        cascade level's carried memory rows -- unchanged mechanism from
        PatchSet3D._attn."""
        per_vol = self.compress_m + 1
        if self.context_id_embed:
            seq = self._apply_context_tags(seq, B, K, T, per_vol)
        seq, _ = self.thinking(seq, seq.shape[1])
        n_extra = 0
        if cascade_regs is not None:
            assert self.cascade_registers, "cascade_regs given but cascade_registers=False"
            mem = self.cascade_proj(cascade_regs) + self.cascade_type
            mem = mem.unsqueeze(2).expand(-1, -1, seq.shape[2], -1)
            seq = torch.cat([mem, seq], dim=1)
            n_extra = mem.shape[1]
        # `sep` is unused downstream when full_attn=True (see TransformerEncoderLayer) --
        # passed as 0 for clarity that it has no effect here.
        seq = self.transformer(seq, 0, full_attn=True)
        regs = (seq[:, n_extra:n_extra + self.thinking.n].mean(dim=2)
                if self.cascade_registers else None)
        return seq, regs

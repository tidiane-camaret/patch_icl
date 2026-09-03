"""PatchSet3D: low-resolution in-context 3D segmentation (set-of-patches attention).

3D analog of src/models/patchset_cnn.py's single-level path. A ConvEncoder3D
downsamples each volume to an R³ feature grid; every patch of every volume becomes a
token in a set, and the dimension-agnostic dual-axis transformer (pfn_seg_2d) does
content-based in-context matching over that set. Position is a Fourier feature of the
(i,j,k) cell, not a tensor axis, so the transformer core is reused verbatim.

Single level only: prediction at the token grid R (mask_patch_decode_size=1) or tiled
to (R·d)³ (d>1). arch.fine_decode swaps the constant-tile head for a per-token dynamic
filter read against the query's own unpooled encoder stage (sub-patch detail; conv
encoders only). Refine / sim_prior / Muon-LAWA are intentionally omitted
(see docs/superpowers/specs/2026-07-22-patchset3d-design.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patchset_pfn import FourierPositionalEncoding
from src.models.pfn_seg_2d import (
    ThinkingRows, TransformerEncoderStack, build_register_block_mask)
from src.rope import build_3d_rope_freqs_from_positions


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
            k = int(src) // R   # int(): keep kernel a Python int so FLOP tracing (fvcore) works
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
    supports_fine = True          # exposes unpooled per-stage maps (see encode_with_fine)

    def __init__(self, in_ch: int, dims: tuple[int, ...], resolution: int, groups: int = 8):
        super().__init__()
        assert len(dims) >= 1, "dims needs at least a stem width"
        self.resolution = resolution
        self.dims = tuple(dims)
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

    def _stage_feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Per-stage maps at their native resolutions: [stem@D, stage1@D/2, ...]."""
        feats = [self.stem(x)]
        for stage in self.stages:
            feats.append(stage(feats[-1]))
        return feats

    @property
    def n_fine_stages(self) -> int:
        return len(self.dims)

    def fine_stage_channels(self, stage: int) -> int:
        return self.dims[stage]

    def fine_stage_size(self, in_size: int, stage: int) -> int:
        """Spatial side of `stage`'s native map: stride-1 stem, then one stride-2 per stage."""
        return int(in_size) // (2 ** int(stage))

    def forward(self, x: torch.Tensor, fine_rows=None, fine_stage=None):
        """(B,in,D,H,W) -> (B,out_ch,R,R,R), or (coarse, fine) when fine_rows is given.

        `fine_rows` selects which volumes keep an unpooled map and `fine_stage` is a
        sequence of stage indices (one map returned per stage); the rest are freed with the
        stage list, so only the requested rows cost memory beyond the peak this forward
        already reaches."""
        feats = self._stage_feats(x)
        coarse = torch.cat([self._resample(f, self.resolution) for f in feats], dim=1)
        if fine_rows is None:
            return coarse
        return coarse, tuple(feats[st].index_select(0, fine_rows.to(feats[st].device))
                             for st in fine_stage)


class _ConvNormAct(nn.Module):
    """3x3x3 conv -> InstanceNorm3d(affine) -> LeakyReLU, the nnU-Net decoder block style
    (matches the ResEnc encoder's norm/nonlin so the from-scratch stats stay consistent)."""
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1, bias=True)
        self.norm = nn.InstanceNorm3d(cout, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PatchSet3D(nn.Module):
    # Canonical physical c-axis column order for mask_slots>=2 (index 0 is always img).
    # See __init__'s self.slot_layout for how this becomes the per-instance source of
    # truth (mask_slots=1 uses a separate ("img","mask") layout — see there).
    _SLOT_LAYOUT = ("img", "gt", "pred")

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
        mask_slots: int = 1,
        decode_source: str = "img",
        context_id_embed: bool = False,
        max_context: int = 16,
        full_attn: bool = False,
        query_self_attn: bool = False,
        register_routed: bool = False,
        register_flex: bool = True,
        image_size=None,
        encoder: str = "conv",
        encoder_frozen: bool = True,
        primus_sidecar: str = None,
        nnunet_ts_weights: str = None,
        nnunet_ts_stages=(2, 3, 4),
        nnunet_ts_random_init: bool = False,
        resenc_n_stages: int = 5,
        plainconv_ts_n_stages: int = 5,
        plainconv_ts_features_per_stage=None,
        encoder_input_norm: str = None,
        img_embed_mlp: bool = False,
        encoder_stage: int = None,
        encoder_native_grid: bool = False,
        encoder_spacing_aware: bool = False,
        encoder_precision: str = "bf16",
        transformer_rope: bool = False,
        rope_theta: float = 100.0,
        feat_norm: str = "context",
        token_mask_ratio_support: float = 0.0,
        token_mask_ratio_query: float = 0.0,
        fine_decode: bool = False,
        fine_stage: int = 1,
        fine_proj_dim: int = 64,
        decoder: str = "fine_filter",
        decoder_dim: int = 64,
    ):
        super().__init__()
        self.resolution = resolution
        self.N = resolution ** 3
        self.mask_patch_size = int(mask_patch_size)
        self.mask_patch_decode_size = int(mask_patch_decode_size)
        assert self.mask_patch_size >= 1 and self.mask_patch_decode_size >= 1
        # mask_slots: c-axis mask-content columns. 1 (default) = today's single shared
        # column (support GT / query prior share one slot, unchanged params -> old
        # checkpoints load as-is). 2 = separate gt/pred columns (see _tokens_multi). Only
        # 1|2 for now; more slot types (bbox/point/scribble/...) are a later extension of
        # the same mechanism, not a new one -- see _tokens_multi's docstring.
        self.mask_slots = int(mask_slots)
        assert self.mask_slots in (1, 2), (
            f"arch.mask_slots={mask_slots} — only 1 (legacy) or 2 (gt/pred split) for now")
        # Canonical physical c-axis column layout — the ONE source of truth for every
        # slot-indexed thing: token-building order (_tokens_multi), slot_pos identity
        # index, and the decode readout column (decode_source below). Adding a slot type
        # later (bbox/point/scribble/...) is appending a name to _SLOT_LAYOUT; nothing
        # else's indexing scheme changes. mask_slots=1 keeps its own single merged
        # "mask" column (the untouched legacy _tokens path) rather than a name from
        # _SLOT_LAYOUT, since it genuinely isn't split into gt/pred.
        self.slot_layout = (self._SLOT_LAYOUT[:1 + self.mask_slots] if self.mask_slots >= 2
                            else ("img", "mask"))
        self.slot_index = {name: i for i, name in enumerate(self.slot_layout)}
        # Which column the decoder reads for the query row. Default "img" reproduces
        # today's exact behavior (_decode_col=0) for every mask_slots value. See
        # docs/logs.md 2026-09-04: support populates gt (mask_slots>=2) / mask
        # (mask_slots=1) with REAL content, so — unlike pred, which support never
        # populates — that column's OWN cross-context attention does genuine retrieval
        # against real support masks; it's the more direct readout channel.
        self.decode_source = str(decode_source)
        assert self.decode_source in self.slot_layout, (
            f"arch.decode_source={decode_source!r} not in this config's slot_layout "
            f"{self.slot_layout}")
        self._decode_col = self.slot_index[self.decode_source]
        self.full_attn = full_attn
        self.query_self_attn = query_self_attn
        # register_routed: each image attends only within its own N-cell block; the thinking
        # rows are the sole cross-image bus (registers read all tokens, all tokens read
        # registers). No direct ctx<->tgt token attention -> blocks the ctx->tgt feature
        # -matching shortcut. Needs an explicit r x r mask (disables flash SDPA).
        self.register_routed = register_routed
        # register_flex: use the FlexAttention BlockMask (skips masked blocks -> flash-like mem,
        # ~T× cheaper). Set False to force the dense r×r bool-mask SDPA instead — needed on nodes
        # where flex's Triton kernel hangs in ptxas on a cold compile cache (Blackwell/cu13 seen
        # to stall >10 min). Dense is fine at small K; heavy at large K (see bench_attn_pattern.py).
        self.register_flex = register_flex
        self.context_id_embed = context_id_embed
        self.max_context = max_context
        self.image_size = image_size          # metadata only (unused in forward)
        # True when the frozen encoder scales its RoPE by physical voxel spacing; gates
        # whether callers (train loop / eval loop) thread a per-batch `spacing` through.
        self.spacing_aware = bool(encoder_spacing_aware)

        if encoder == "primus":
            if not primus_sidecar:
                raise ValueError("encoder='primus' requires arch.primus_sidecar")
            from src.models.primus_encoder import PrimusEncoder   # lazy: avoids import cycle
            self.encoder = PrimusEncoder(primus_sidecar, resolution,
                                         frozen=encoder_frozen, device="cpu",
                                         encoder_stage=encoder_stage,
                                         native_grid=encoder_native_grid,
                                         spacing_aware=encoder_spacing_aware,
                                         precision=encoder_precision)
        elif encoder == "tap_ct":
            # Frozen fomofo/tap-ct-b-3d ViT. Weights fixed on HF (no sidecar); it always
            # tokenizes at the native anisotropic grid (image_size drives the token count)
            # and is not spacing-aware — the physical cell size is set by data.crop_spacing_mm.
            # encoder_stage early-exits the transformer blocks (like Primus). Needs image_size
            # divisible by 8. Ignores encoder_native_grid/encoder_spacing_aware (always native).
            from src.models.tapct_encoder import TapCTEncoder   # lazy: avoids import cycle
            if not image_size:
                raise ValueError("encoder='tap_ct' requires arch.image_size (from data.image_size)")
            self.encoder = TapCTEncoder(resolution, image_size, frozen=encoder_frozen,
                                        device="cpu", encoder_stage=encoder_stage,
                                        precision=encoder_precision)
        elif encoder == "nnunet_ts":
            # Frozen TotalSegmentator nnU-Net PlainConvUNet encoder (default: Dataset297,
            # total 3 mm). Multi-scale concat of nnunet_ts_stages resampled to R^3; input is
            # 1-channel (image only), spacing arg ignored (conv net). nnunet_ts_weights points
            # at the weights folder (plans.json + fold_0/checkpoint_final.pth) and is required
            # even when nnunet_ts_random_init=True — plans.json defines the architecture and the
            # CTNormalization stats; only the trained weights are dropped (He init instead).
            from src.models.encoders.nnunet_ts import NnUNetTSEncoder   # lazy: avoids import cycle
            if not nnunet_ts_weights:
                raise ValueError("encoder='nnunet_ts' requires arch.nnunet_ts_weights")
            # encoder_input_norm: None keeps each encoder's own default (nnunet_ts=reframe,
            # so a frozen pretrained encoder still converts loader-frame -> its plans frame).
            _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
            self.encoder = NnUNetTSEncoder(nnunet_ts_weights, resolution,
                                           stages=tuple(nnunet_ts_stages),
                                           frozen=encoder_frozen, device="cpu",
                                           precision=encoder_precision,
                                           random_init=nnunet_ts_random_init, **_in_norm)
        elif encoder == "resenc_ts":
            # From-scratch nnU-Net ResidualEncoderUNet (the ResEnc twin of nnunet_ts). No
            # plans.json / checkpoint: the architecture is the ResEnc M/L/XL recipe with
            # resenc_n_stages stages (base 32, x2, cap 320; blocks 1/3/4/6/6/...), He init.
            # Multi-scale concat of nnunet_ts_stages resampled to R^3; 1-channel image input,
            # spacing arg ignored. encoder_input_norm defaults to passthrough (the image is
            # already in the pipeline CT frame — see src/totalseg_dataset.CtNormSpec).
            from src.models.encoders.resenc_ts import ResEncTSEncoder   # lazy: avoids import cycle
            _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
            self.encoder = ResEncTSEncoder(resolution, n_stages=resenc_n_stages,
                                           stages=tuple(nnunet_ts_stages),
                                           frozen=encoder_frozen, device="cpu",
                                           precision=encoder_precision, **_in_norm)
        elif encoder == "plainconv_ts":
            # From-scratch nnU-Net PlainConvUNet (the PlainConv twin of resenc_ts). No
            # plans.json / checkpoint: width is plainconv_ts_features_per_stage if given,
            # else the same base=32/x2/cap=320 formula resenc_ts uses; n_conv_per_stage=2
            # throughout (nnU-Net's standard plain-conv schedule), He init. Multi-scale
            # concat of nnunet_ts_stages resampled to R^3; 1-channel image input, spacing
            # arg ignored. encoder_input_norm defaults to zscore (per-volume HU) here — this
            # encoder carries no plans-file CTNormalization stats to reframe into.
            from src.models.encoders.plainconv_ts import PlainConvTSEncoder   # lazy: avoids import cycle
            _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
            self.encoder = PlainConvTSEncoder(resolution, n_stages=plainconv_ts_n_stages,
                                              stages=tuple(nnunet_ts_stages),
                                              features_per_stage=plainconv_ts_features_per_stage,
                                              frozen=encoder_frozen, device="cpu",
                                              precision=encoder_precision, **_in_norm)
        elif encoder == "conv":
            self.encoder = ConvEncoder3D(1, tuple(enc_dims), resolution)
        else:
            raise ValueError(f"unknown arch.encoder {encoder!r} "
                             f"(conv | primus | tap_ct | nnunet_ts | resenc_ts | plainconv_ts)")
        # Encoder feature -> token width e. Default: a single Linear. When the encoder
        # width far exceeds e (e.g. frozen primus out_ch=864 -> e=256), that lone Linear
        # is a rank bottleneck; img_embed_mlp=True instead keeps the full encoder width
        # through a GELU before compressing (Linear(oc,oc) -> GELU -> Linear(oc,e)),
        # preserving more of the frozen features. Off by default (identical to before).
        oc = self.encoder.out_ch
        self.img_embed = (nn.Sequential(nn.Linear(oc, oc), nn.GELU(), nn.Linear(oc, e))
                          if img_embed_mlp else nn.Linear(oc, e))
        # occupancy tile p³ -> e, SHARED across every mask-content slot (gt, pred, ... all go
        # through this one Linear — no per-slot learned weights). mask_slots>=2 widens the
        # input by one presence scalar (1 = this slot's real content, 0 = placeholder for a
        # slot that doesn't apply to this row group; see _tokens_multi), so the shape is a
        # function of mask_patch_size and mask_slots-the-boolean-question ">=2", never of the
        # slot COUNT itself.
        mask_in = self.mask_patch_size ** 3 + (1 if self.mask_slots >= 2 else 0)
        self.mask_embed = nn.Linear(mask_in, e)
        # Positional encoding. Default: additive Fourier features of the (i,j,k) cell (spacing
        # -blind). transformer_rope=True instead applies 3D axial RoPE on the sample axis inside
        # the transformer (mirrors the encoder's RoPE) and drops the additive term (RoPE-only).
        # RoPE positions are scaled by physical spacing / rope_train_mm (the encoder's pretrain
        # pitch, 2 mm) so they track physical distance across the variable-spacing range.
        self.transformer_rope = bool(transformer_rope)
        self.rope_theta = float(rope_theta)
        # Encoder-feature normalization before img_embed (see _feat_norm):
        #   context = per-channel z-score by SUPPORT stats, applied to both (TabPFN/ICL default)
        #   self    = each side z-scored by its OWN stats (query decoupled from context)
        #   none    = no extra norm (rely on encoder LN + learned img_embed)
        assert feat_norm in ("context", "self", "none"), feat_norm
        self.feat_norm = feat_norm
        # SimMIM-style in-place token masking (training only; both default 0.0 = off). A masked
        # cell has BOTH its image and mask/occupancy columns replaced by mask_token, keeping the
        # R³ token count intact (compiled transformer + RoPE-by-index unaffected) and leaving the
        # cell in the sequence for a future reconstruction loss. See
        # docs/superpowers/specs/2026-08-11-patchset3d-token-masking-design.md.
        self.token_mask_ratio_support = float(token_mask_ratio_support)
        self.token_mask_ratio_query = float(token_mask_ratio_query)
        # row 0 = image col, row 1 = the ROW GROUP's active content col (gt for support,
        # pred for query when mask_slots>=2 — see _tokens_multi) — indexed by content TYPE,
        # not physical slot position, so this shape is also independent of mask_slots.
        self.mask_token = nn.Parameter(torch.zeros(2, e))
        nn.init.normal_(self.mask_token, std=0.02)
        self.head_dim = e // a
        self.rope_train_mm = float(getattr(self.encoder, "train_spacing_mm", 2.0))
        self.pos = None if self.transformer_rope else FourierPositionalEncoding(e, fourier_bands, n_axes=3)
        # mask_slots>=2: additive Fourier feature of the (fixed-capacity-normalized) slot
        # index, the ONLY thing distinguishing otherwise-identically-embedded mask columns
        # (independent of transformer_rope — RoPE only rotates the row/sample axis, giving
        # c-axis columns no identity of their own). Linear(2*fourier_bands, e): fixed shape
        # regardless of mask_slots, so a checkpoint trained at mask_slots=2 loads unchanged
        # at mask_slots=3+ later. Capacity is a fixed constant (not the live mask_slots) so
        # slot 0/1's encoding never shifts when a slot is added — see
        # tests/test_patchset3d.py::test_slot_pos_separable_from_spatial_pos.
        self.slot_pos = FourierPositionalEncoding(e, fourier_bands, n_axes=1) if self.mask_slots >= 2 else None
        self._SLOT_CAPACITY = 8
        if context_id_embed:
            self.ctx_id = nn.Embedding(max_context, e)
            self.qry_id = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.ctx_id.weight, std=0.1)
            nn.init.normal_(self.qry_id, std=0.1)
        self.thinking = ThinkingRows(thinking_rows, e)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
        # Decode head. Default (fine_decode=False): a per-token MLP emitting d^3 CONSTANTS per
        # cell. fine_decode=True picks between two heads via `decoder`:
        #   "fine_filter" — each token emits a dynamic FILTER dotted against the query volume's
        #                   own unpooled encoder features inside its cell, + a linear tile term.
        #                   Cheap, but cells never interact (per-cell-independent readout).
        #   "conv"        — a progressive coarse->fine conv decoder: project tokens to
        #                   decoder_dim, upsample the R^3 field, fuse each requested unpooled
        #                   stage (concat at coarse levels, per-voxel FiLM at the finest), a
        #                   3x3x3 conv per level, then a 1x1x1 head + a token-only residual.
        #                   Spatially coupled (conv receptive field); ~6x the FLOPs.
        # Both need an encoder exposing unpooled stages (conv | nnunet_ts | resenc_ts | plainconv_ts).
        self.fine_decode = bool(fine_decode)
        self.decoder_kind = str(decoder)
        # int or list: fine_filter sums the projected stages at the finest grid; conv uses
        # them as the skip pyramid (coarse->fine order is derived, any order accepted here).
        self.fine_stage = tuple(int(st) for st in (
            [fine_stage] if isinstance(fine_stage, int) else list(fine_stage)))
        self.fine_m = None
        if not self.fine_decode:
            self.decoder = nn.Sequential(nn.Linear(e, h), nn.GELU(),
                                         nn.Linear(h, self.mask_patch_decode_size ** 3))
        else:
            if not getattr(self.encoder, "supports_fine", False):
                raise ValueError(
                    f"arch.fine_decode needs an encoder exposing unpooled stages; "
                    f"arch.encoder={encoder!r} has none (use conv | nnunet_ts | resenc_ts | plainconv_ts)")
            for st in self.fine_stage:
                if not 0 <= st < self.encoder.n_fine_stages:
                    raise ValueError(f"arch.fine_stage {st} out of range "
                                     f"[0, {self.encoder.n_fine_stages})")
            if not image_size:
                raise ValueError("arch.fine_decode needs arch.image_size (from data.image_size)")
            sides = [self.encoder.fine_stage_size(int(image_size[0]), st)
                     for st in self.fine_stage]
            fine_side = max(sides)                        # readout runs at the FINEST stage
            if fine_side % resolution:
                raise ValueError(f"fine stage side {fine_side} is not divisible by "
                                 f"resolution {resolution}")
            self.fine_m = fine_side // resolution         # fine voxels per cell axis
            chans = [self.encoder.fine_stage_channels(st) for st in self.fine_stage]
            if self.decoder_kind == "fine_filter":
                # One 1x1x1 projection per stage, each applied at that stage's NATIVE
                # resolution; the projected maps are upsampled to the finest side and summed.
                # Trilinear upsampling is per-channel and a 1x1x1 conv is per-voxel, so they
                # commute: this is exactly Conv1x1(concat[f0, up(f1)]) without ever
                # materializing the concatenated map at full resolution.
                self.fine_proj = nn.ModuleList([nn.Conv3d(c, fine_proj_dim, 1) for c in chans])
                self.filter_head = nn.Linear(e, fine_proj_dim)
                self.tile_head = nn.Linear(e, self.fine_m ** 3)
                # The readout is linear in the fine features (1x1x1 conv then a channel
                # contraction), so the token-generated filter lives in the row space of the
                # projection: its rank is capped at min(fine_proj_dim, sum(C_f)). Anything
                # above that is dead width.
                if fine_proj_dim > sum(chans):
                    print(f"[PatchSet3D] arch.fine_proj_dim={fine_proj_dim} exceeds total fine "
                          f"channels {sum(chans)} (stages {self.fine_stage}); the filter rank is "
                          f"capped at {sum(chans)} — the extra width is unused.")
            elif self.decoder_kind == "conv":
                self._build_conv_decoder(e, int(image_size[0]), resolution, int(decoder_dim))
            else:
                raise ValueError(f"arch.decoder {self.decoder_kind!r} (fine_filter | conv)")
        # (i,j,k) lattice, row-major over R³ (cell index n = i*R² + j*R + k)
        r = resolution
        ii = torch.arange(r).repeat_interleave(r * r)
        jj = torch.arange(r).repeat_interleave(r).repeat(r)
        kk = torch.arange(r).repeat(r * r)
        self.register_buffer("ijk_base", torch.stack([ii, jj, kk], dim=-1), persistent=False)  # (N,3)

    def _build_conv_decoder(self, e: int, in_size: int, resolution: int, c_d: int):
        """Progressive coarse->fine conv decoder (arch.decoder=conv). Levels are the requested
        unpooled stages ordered by resolution (coarsest first); the token field is upsampled to
        each and fused — concat + 3x3x3 conv at the coarse levels, per-voxel FiLM + 3x3x3 conv
        at the finest. decoder_dim halves per level (>=8). A 1x1x1 head plus a token-only
        residual (upsampled token field -> 1x1x1) gives the head a token-only fallback while
        the from-scratch encoder's fine features are still noise (cf. fine_filter's tile term)."""
        stages = sorted(self.fine_stage,
                        key=lambda st: self.encoder.fine_stage_size(in_size, st))  # coarse->fine
        self._dec_stage_order = [self.fine_stage.index(st) for st in stages]        # into `fine`
        self._dec_sides = [self.encoder.fine_stage_size(in_size, st) for st in stages]
        chans = [self.encoder.fine_stage_channels(st) for st in stages]
        n = len(stages)
        dims = [max(c_d // (2 ** i), 8) for i in range(n)]                          # taper
        self.token_proj = nn.Linear(e, c_d)
        # per-skip instance norm (affine off) = the fine_filter path's per-(sample,channel)
        # z-score, so the raw from-scratch encoder activation scale can't blow up the fuse.
        self.dec_skip_norm = nn.ModuleList([nn.InstanceNorm3d(c, affine=False) for c in chans])
        self.dec_skip_proj = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_film = None
        prev = c_d
        for i in range(n):
            if i < n - 1:                                   # concat fuse
                self.dec_skip_proj.append(nn.Conv3d(chans[i], dims[i], 1))
                self.dec_blocks.append(_ConvNormAct(prev + dims[i], dims[i]))
            else:                                           # finest: per-voxel FiLM fuse
                self.dec_film = nn.Conv3d(prev, 2 * chans[i], 1)   # gamma, beta over the skip
                self.dec_skip_proj.append(None)
                self.dec_blocks.append(_ConvNormAct(chans[i], dims[i]))
            prev = dims[i]
        self.dec_head = nn.Conv3d(prev, 1, 1)
        self.dec_token_residual = nn.Conv3d(c_d, 1, 1)

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

    def _prior_occupancy(self, prior):
        """Cascade coarse-level prior (B,1,D,H,W) soft prob -> query mask-token input
        (B, N, p³), the same _down_to / _mask_tiles_3d path _occupancy uses for the
        support masks. Replaces the support-mean prior on the query when threaded in."""
        B = prior.shape[0]
        p = self.mask_patch_size
        prior = prior.reshape(B, 1, *prior.shape[-3:]).float()
        if p == 1:
            return _down_to(prior, self.resolution).reshape(B, self.N, 1)
        return _mask_tiles_3d(prior, self.resolution, p)      # (B, N, p³)

    def _tokens(self, feat, occ, ijk, mask=None):
        img = self.img_embed(feat)
        msk = self.mask_embed(occ)
        if mask is not None:                                # SimMIM in-place [MASK] replacement
            m = mask.unsqueeze(-1)                          # (B,M,1) bool
            # Cast to the operand dtype so torch.where does not promote img/msk to fp32
            # when mask_token is fp32 and the token stream runs under bf16 autocast.
            img = torch.where(m, self.mask_token[0].to(img.dtype), img)
            msk = torch.where(m, self.mask_token[1].to(msk.dtype), msk)
        else:
            # Keep mask_token in the compute graph (zero contribution) so it always
            # receives a gradient — required for optimizers that track all parameters.
            img = img + self.mask_token[0].sum() * 0.0
            msk = msk + self.mask_token[1].sum() * 0.0
        if self.pos is not None:                            # additive Fourier PE (non-RoPE mode)
            pos = self.pos(ijk, self.resolution)
            img = img + pos                                 # masked token keeps its position
            msk = msk + pos
        return torch.stack([img, msk], dim=2)               # (B,M,2,e)

    def _slot_pos_vec(self, idx: int, device, dtype):
        """(e,) additive Fourier feature for canonical slot index `idx` (self.slot_index[...],
        img's index 0 never calls this) — fixed-capacity normalized (self._SLOT_CAPACITY,
        not self.mask_slots), so it never shifts when a later run adds more slots."""
        v = self.slot_pos(torch.tensor([[idx]], device=device), self._SLOT_CAPACITY)  # (1,e)
        return v.reshape(-1).to(dtype)

    def _tokens_multi(self, feat, occ, active_slot: str, ijk, mask=None):
        """arch.mask_slots>=2 token builder -> (B,M,len(self.slot_layout),e).

        `occ` is this row group's real mask content (real GT for support, prior/pred for
        query — same occ tensors _tokens already took); `active_slot` ("gt"/"pred") says
        which named slot it belongs to. Every OTHER content slot gets a zero-occupancy,
        presence=0 placeholder through the SAME shared mask_embed (widened by one presence
        scalar in __init__) — no per-slot learned weights, so nothing here depends on
        mask_slots except which iteration is "active". Column order and slot identity both
        come from self.slot_layout / self.slot_index (__init__) — the single source of
        truth; nothing in this method hardcodes an index."""
        dtype, device = feat.dtype, feat.device
        zeros_occ = torch.zeros_like(occ)
        pos = self.pos(ijk, self.resolution) if self.pos is not None else None
        active_col = self.slot_index[active_slot]

        cols = []
        for name in self.slot_layout:
            if name == "img":
                c = self.img_embed(feat)
            else:
                present = (self.slot_index[name] == active_col)
                o = occ if present else zeros_occ
                p = o.new_full(o.shape[:-1] + (1,), 1.0 if present else 0.0)
                c = self.mask_embed(torch.cat([o, p], dim=-1))
                c = c + self._slot_pos_vec(self.slot_index[name], device, dtype)
            if pos is not None:
                c = c + pos
            cols.append(c)

        img_col = self.slot_index["img"]
        if mask is not None:                                # SimMIM in-place [MASK] replacement
            m = mask.unsqueeze(-1)                          # (B,M,1) bool
            cols[img_col] = torch.where(m, self.mask_token[0].to(dtype), cols[img_col])
            cols[active_col] = torch.where(m, self.mask_token[1].to(dtype), cols[active_col])
        else:
            # Keep mask_token in the compute graph (zero contribution) so it always
            # receives a gradient — required for optimizers that track all parameters.
            cols[img_col] = cols[img_col] + self.mask_token[0].sum() * 0.0
            cols[active_col] = cols[active_col] + self.mask_token[1].sum() * 0.0

        return torch.stack(cols, dim=2)          # (B,M,len(self.slot_layout),e)

    def _tile_logits(self, out, d=None):
        """(B,N,d³) -> (B,1,Rd,Rd,Rd), inverse of _mask_tiles_3d (d=1 -> one logit per cell).

        d defaults to mask_patch_decode_size; the fine decoder passes its own tile size."""
        B = out.shape[0]
        r = self.resolution
        d = self.mask_patch_decode_size if d is None else int(d)
        if d == 1:
            return out.reshape(B, 1, r, r, r)
        return (out.reshape(B, r, r, r, d, d, d)
                   .permute(0, 1, 4, 2, 5, 3, 6)
                   .reshape(B, r * d, r * d, r * d)
                   .unsqueeze(1))

    def _rope(self, K, spacing, device):
        """3D axial RoPE cos/sin for the row sequence [thinking, K·N support, N query].

        Thinking rows get position (0,0,0) (no rotation); support/query use the (i,j,k)
        lattice. Positions are scaled by spacing/rope_train_mm when a spacing is given, so
        adjacent cells sit `spacing/train` apart in physical units — the encoder's scheme."""
        n_think = self.thinking.n
        pos = torch.cat([torch.zeros(n_think, 3, device=device),
                         self.ijk_base.repeat(K, 1).float(),
                         self.ijk_base.float()], dim=0)               # (R,3)
        if spacing is not None:
            pos = pos * (float(spacing) / self.rope_train_mm)
        return build_3d_rope_freqs_from_positions(self.head_dim, pos, self.rope_theta)

    @staticmethod
    def _zscore(x, mu, sig):
        return ((x - mu) / sig).clamp(-10, 10)

    def _feat_norm(self, sup_feat, qry_feat):
        """Per-channel z-score of encoder features (dim=1 = token axis), by mode:
        context = SUPPORT stats on both (query in the context frame); self = each side by
        its OWN stats (query decoupled); none = pass-through."""
        if self.feat_norm == "none":
            return sup_feat, qry_feat
        mu = sup_feat.mean(dim=1, keepdim=True)
        sig = sup_feat.std(dim=1, keepdim=True) + 1e-8
        if self.feat_norm == "context":
            return self._zscore(sup_feat, mu, sig), self._zscore(qry_feat, mu, sig)
        # self: query normalized by its own stats
        qmu = qry_feat.mean(dim=1, keepdim=True)
        qsig = qry_feat.std(dim=1, keepdim=True) + 1e-8
        return self._zscore(sup_feat, mu, sig), self._zscore(qry_feat, qmu, qsig)

    def _sample_mask(self, B, M, ratio, device):
        """Random per-cell boolean mask (B,M) at the given ratio; None when not training or
        ratio<=0. Independent Bernoulli per cell (in-place SimMIM masking, not token-dropping)."""
        if not self.training or ratio <= 0.0:
            return None
        return torch.rand(B, M, device=device) < ratio

    def _attn(self, sup_feat, qry_feat, sup_occ, K, spacing=None, query_prior=None):
        B, N = sup_feat.shape[0], self.N
        dev = sup_feat.device
        mask_support = self._sample_mask(B, K * N, self.token_mask_ratio_support, dev)
        mask_query = self._sample_mask(B, N, self.token_mask_ratio_query, dev)
        if query_prior is not None:                          # cascade: coarse-level prediction
            qry_occ = self._prior_occupancy(query_prior).to(sup_occ.dtype)
        else:
            qry_occ = sup_occ.mean(dim=1, keepdim=True).expand(B, N, sup_occ.shape[-1])  # support-mean prior
        sup_ijk = self.ijk_base.repeat(K, 1).unsqueeze(0).expand(B, K * N, 3)
        qry_ijk = self.ijk_base.unsqueeze(0).expand(B, N, 3)

        sup_feat, qry_feat = self._feat_norm(sup_feat, qry_feat)

        if self.mask_slots >= 2:
            # support always carries real GT (never a prediction); query never carries real
            # GT (only ever a prior/prediction, or the support-mean fallback) — see forward's
            # docstring and cascade.py's context_out handling.
            sup_tok = self._tokens_multi(sup_feat, sup_occ, "gt", sup_ijk, mask=mask_support)
            qry_tok = self._tokens_multi(qry_feat, qry_occ, "pred", qry_ijk, mask=mask_query)
        else:
            sup_tok = self._tokens(sup_feat, sup_occ, sup_ijk, mask=mask_support)   # (B,S,2,e)
            qry_tok = self._tokens(qry_feat, qry_occ, qry_ijk, mask=mask_query)     # (B,Q,2,e)

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
        block_mask = None
        if not self.full_attn and self.register_routed:
            # Registers (thinking rows) are the only cross-image path: they read every token
            # and every token reads them; each image otherwise attends only within its own
            # N-cell block (K support blocks + 1 query block), so there is no direct ctx<->tgt
            # token attention to short-circuit into feature matching. flex_attention skips the
            # masked off-diagonal blocks; a dense r×r bool mask is the fallback without flex.
            n_t = self.thinking.n
            block_mask = (build_register_block_mask(n_t, N, K + 1, x.device)
                          if self.register_flex else None)
            if block_mask is None:
                r = x.shape[1]
                attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
                attn_mask[:n_t, :] = True                     # registers read all tokens
                attn_mask[:, :n_t] = True                     # all tokens read registers
                for m in range(K + 1):
                    s = n_t + m * N
                    attn_mask[s:s + N, s:s + N] = True        # within-image self-attention
        elif not self.full_attn and self.query_self_attn:
            r = x.shape[1]
            attn_mask = torch.zeros(r, r, dtype=torch.bool, device=x.device)
            attn_mask[:, :sep_t] = True
            attn_mask[sep_t:, sep_t:] = True
        rope = self._rope(K, spacing, x.device) if self.transformer_rope else None
        x = self.transformer(x, sep_t, attn_mask=attn_mask, full_attn=self.full_attn,
                             rope=rope, block_mask=block_mask)
        q = x[:, sep_t:, self._decode_col, :]      # (B,Q,e) query row, arch.decode_source col
        return q, mask_support, mask_query

    def _decode(self, q, fine=None):
        """Query tokens (B,N,e) -> logits (B,1,G,G,G), G = resolution*mask_patch_decode_size.

        fine_decode path: z-score + 1x1x1-project each requested unpooled stage at its own
        resolution, sum them at the finest grid, regroup into per-cell blocks, and score each
        fine voxel with its own cell's token-generated filter; the tile term adds the
        token-only constant the default head would emit."""
        if not self.fine_decode:
            return self._tile_logits(self.decoder(q))
        if self.decoder_kind == "conv":
            return self._decode_conv(q, fine)
        B, N = q.shape[0], q.shape[1]
        R, m = self.resolution, self.fine_m
        side = R * m
        p = None
        for conv, f in zip(self.fine_proj, fine):
            f = f.float()
            mu = f.mean(dim=(2, 3, 4), keepdim=True)
            sig = f.std(dim=(2, 3, 4), keepdim=True) + 1e-8
            pi = conv(self._zscore(f, mu, sig))               # (B,Cp,S_i,S_i,S_i)
            if pi.shape[-1] != side:                          # lift coarser stages to the
                pi = F.interpolate(pi, size=(side, side, side),   # finest stage's grid
                                   mode="trilinear", align_corners=False)
            p = pi if p is None else p + pi                   # (B,Cp,S,S,S)
        cp = p.shape[1]
        # Regroup to cells with the SAME permutation as _mask_tiles_3d, so cell order matches
        # ijk_base (n = i*R² + j*R + k) and _tile_logits is its exact inverse. Left as a view —
        # einsum consumes it without materializing a contiguous copy.
        cells = (p.reshape(B, cp, R, m, R, m, R, m)
                  .permute(0, 2, 4, 6, 1, 3, 5, 7))           # (B,R,R,R,Cp,m,m,m)
        w = self.filter_head(q).reshape(B, R, R, R, cp)
        out = torch.einsum("bijkc,bijkcxyz->bijkxyz", w, cells).reshape(B, N, m ** 3)
        logit = self._tile_logits(out + self.tile_head(q), d=m)    # (B,1,S,S,S)
        g = self.grid_size
        if logit.shape[-1] != g:
            logit = F.interpolate(logit, size=(g, g, g), mode="trilinear", align_corners=False)
        return logit

    def _decode_conv(self, q, fine):
        """arch.decoder=conv: progressive coarse->fine conv decoder over the query's unpooled
        stages. `fine` is in self.fine_stage order; self._dec_stage_order reindexes to
        coarse->fine. Returns (B,1,G,G,G)."""
        B, N = q.shape[0], q.shape[1]
        R = self.resolution
        t = self.token_proj(q).transpose(1, 2).reshape(B, -1, R, R, R)   # (B,C_d,R,R,R)
        x = t
        n = len(self.dec_blocks)
        for i in range(n):
            s = self._dec_sides[i]
            x = F.interpolate(x, size=(s, s, s), mode="trilinear", align_corners=False)
            skip = self.dec_skip_norm[i](fine[self._dec_stage_order[i]].to(x.dtype))
            if i < n - 1:                                   # concat fuse
                x = self.dec_blocks[i](torch.cat([x, self.dec_skip_proj[i](skip)], dim=1))
            else:                                          # finest: per-voxel FiLM fuse
                g, b = self.dec_film(x).chunk(2, dim=1)
                x = self.dec_blocks[i]((1.0 + g) * skip + b)
        logit = self.dec_head(x)
        tr = F.interpolate(t, size=x.shape[-3:], mode="trilinear", align_corners=False)
        logit = logit + self.dec_token_residual(tr)         # token-only fallback path
        gs = self.grid_size
        if logit.shape[-1] != gs:
            logit = F.interpolate(logit, size=(gs, gs, gs), mode="trilinear", align_corners=False)
        return logit

    def forward(self, image, context_in, context_out, mode="train", spacing=None,
                query_prior=None):
        """query_prior: optional (B,1,D,H,W) soft probability volume, already resampled onto
        THIS forward's grid frame (the cascade runner does the geometric warp). When given it
        replaces the support-mean prior on the query's mask token — see _attn / _prior_occupancy."""
        B, K = context_in.shape[0], context_in.shape[1]
        D, H, W = image.shape[-3:]
        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)     # (B,T,1,D,H,W)
        T = imgs.shape[1]
        x = imgs.reshape(B * T, 1, D, H, W)
        fine = None
        if self.fine_decode:
            # The query volume is the last of T, so its flat rows are b*T + K; only those
            # keep an unpooled map (one encoder pass, the rest freed with the stage list).
            rows = torch.arange(B, device=x.device) * T + K
            feat_map, fine = self._encode(x, spacing, fine_rows=rows)
        else:
            feat_map = self._encode(x, spacing)                        # (B*T,Cf,R,R,R)
        sup_feat, qry_feat = self._grid_tokens(feat_map, B, T, K)
        q, mask_support, mask_query = self._attn(
            sup_feat, qry_feat, self._occupancy(context_out), K, spacing=spacing,
            query_prior=query_prior)
        logit = self._decode(q, fine)
        return {"final_logit": logit, "mask_support": mask_support, "mask_query": mask_query}

    def _encode(self, x, spacing, fine_rows=None):
        """Dispatch to the encoder, passing per-batch `spacing` only when it accepts it
        (PrimusEncoder in spacing-aware mode); the conv encoder takes only the image.

        Always goes through Module.__call__ so forward hooks fire (train.py's
        profile_timing records encode GPU time there). Returns (coarse, fine) when
        fine_rows is given, else the coarse grid alone."""
        kw = {"spacing": spacing} if self.spacing_aware else {}
        if fine_rows is not None:
            kw.update(fine_rows=fine_rows, fine_stage=self.fine_stage)
        return self.encoder(x, **kw)

    def _native_logit(self, image, context_in, context_out, spacing=None, query_prior=None):
        dev = next(self.parameters()).device
        image = image.to(dev); context_in = context_in.to(dev); context_out = context_out.to(dev)
        if query_prior is not None:
            query_prior = query_prior.to(dev)
        logit = self.forward(image, context_in, context_out, spacing=spacing,
                             query_prior=query_prior)["final_logit"].float()
        return F.interpolate(logit, size=image.shape[-3:], mode="trilinear", align_corners=False)

    def train_forward(self, target_img, context_imgs, context_masks, spacing=None,
                      query_prior=None):
        """Native-resolution logits (B,1,D,H,W) — used by the val soft-Dice / loss path."""
        return self._native_logit(target_img, context_imgs, context_masks, spacing=spacing,
                                  query_prior=query_prior)

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks, spacing=None, query_prior=None):
        """Native binary mask (B,D,H,W) — used by the eval Dice path."""
        logit = self._native_logit(target_img, context_imgs, context_masks, spacing=spacing,
                                   query_prior=query_prior)
        return (torch.sigmoid(logit) >= 0.5).float().squeeze(1)

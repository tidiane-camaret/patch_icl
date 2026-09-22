"""
ImagePFN: in-context 2D image segmentation via dual-axis transformer.

Tensor layout throughout: (batch, rows, cols, e)
  rows = n_thinking + K context images + 1 query image
  cols = 2N = 2 × resolution²  (effective patch size P = image_size // resolution)
         first N cols  = image patch embeddings
         last  N cols  = mask  patch embeddings

Image and mask patches are kept as separate column groups (mirroring nanoTabPFN's
feature cols + label col design). Feature-axis attention can then explicitly route
information between image and mask representations. The decoder reads only from the
N image cols of the query row.

TargetEncoder trick: query mask cols are filled with the mean of context mask patches
before embedding (rather than zeros), providing a class-frequency prior — same as
nanoTabPFN's TargetEncoder padding for test rows.

Techniques from modded-nanoTabPFN:
  - Dual-axis attention: feature-axis (spatial, within image) + sample-axis (cross-image, asymmetric)
  - Thinking rows: learnable latent rows prepended to the sequence
  - Residual decay: scale input to block i by residual_decay^i
  - LowerPrecisionRMSNorm: pre-norm, fp32 upcast for bf16/fp16 inputs
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rope import apply_rope


def patchify(x: torch.Tensor, P: int, out: int | None = None,
             mode: str = "bilinear") -> torch.Tensor:
    """(B, 1, H, W) → (B, N, Q²), N = (H//P)*(W//P).

    Splits into native P×P patches; if ``out`` is given and differs from P, each
    patch is resized to out×out (Q=out) so the embedding input dim is decoupled
    from the effective patch size. With ``out=None`` (or out==P) Q=P (no resize).
    """
    B, C, H, W = x.shape
    nh, nw = H // P, W // P
    x = x.reshape(B, C, nh, P, nw, P).permute(0, 2, 4, 1, 3, 5)  # (B, nh, nw, C, P, P)
    if out is not None and out != P:
        x = x.reshape(B * nh * nw, C, P, P)
        x = F.interpolate(x, size=(out, out), mode=mode, align_corners=False)
        return x.reshape(B, nh * nw, C * out * out)
    return x.reshape(B, nh * nw, C * P * P)


def standardize_by_context(feat: torch.Tensor, n_context: int,
                           clamp: float = 10.0) -> torch.Tensor:
    """Per-channel z-score of encoder features using context-row statistics.

    `feat` is (B, T, N, C): rows 0..n_context-1 are the context (image, mask)
    pairs and the remaining rows are the query. mu/sig are computed over the
    context rows × cells, per channel, and applied to ALL rows — so the query is
    standardized in the *context's* feature frame (it never sees its own stats).
    This is the single normalization shared by ImagePFN's image path and the
    multilevel pipeline, and matches the TabPFN feature_sim backend
    (experiments/2d/eval.py: predict_tabpfn / batch_tabpfn). clamp<=0 disables the
    final clamp.
    """
    ctx = feat[:, :n_context]
    mu  = ctx.mean(dim=(1, 2), keepdim=True)            # (B,1,1,C)
    sig = ctx.std( dim=(1, 2), keepdim=True) + 1e-8
    feat = (feat - mu) / sig
    return feat.clamp(-clamp, clamp) if clamp and clamp > 0 else feat


# Flash / mem-efficient SDPA launch one CUDA grid-Y block per batch element, and
# gridDim.y is hardware-capped at 65535. The sample-axis attention flattens to a
# batch of B·2·resolution², which crosses the cap at resolution≥32 (B=32 → 65536),
# raising "CUDA error: invalid configuration argument". Splitting the batch into
# equal chunks that each stay under the cap keeps the fused kernel (math backend
# would materialize the full score tensor and cost ~2× memory). int() pins the
# (symbolic) batch to a concrete value so the loop count is a Python int and
# torch.compile can unroll it statically.
_SDPA_MAX_BATCH = 65535


# Feature-axis attention in the set-of-patches layout (patchset_cnn / patchset3d) has a
# tiny sequence (c = 2 img/mask columns) but a huge batch (b·r = every patch of every
# volume). Flash/SDPA is pathological there — it launches one fused-attention problem per
# (batch, head) with ~zero useful work, so it costs more than the real r=R³ set attention.
# A plain q·kᵀ→softmax→·v (fp32 scores, matching SDPA's internal upcast) is ~3× faster
# incl. backward and numerically equivalent. Only used when the seq is small; larger
# feature axes (ImagePFN's 2N patch columns) fall back to the fused kernel below.
_SMALL_SEQ_ATTN = 16


def _small_seq_attn(q, k, v):
    """Manual attention for tiny sequences (q,k,v = (B, heads, seq, d)). fp32 softmax."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    return (s.softmax(-1).to(v.dtype) @ v)


def batched_sdpa(q, k, v, attn_mask=None):
    """scaled_dot_product_attention that survives batches over the grid-Y cap."""
    B = q.shape[0]
    if B <= _SDPA_MAX_BATCH:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    n = int((B + _SDPA_MAX_BATCH - 1) // _SDPA_MAX_BATCH)
    cs = (B + n - 1) // n
    return torch.cat(
        [F.scaled_dot_product_attention(q[i * cs:(i + 1) * cs],
                                        k[i * cs:(i + 1) * cs],
                                        v[i * cs:(i + 1) * cs],
                                        attn_mask=attn_mask)
         for i in range(n)],
        dim=0,
    )


# FlexAttention path for structured sample-axis masks (patchset3d register_routed). A
# block-diagonal + register-border connectivity is dense-masked SDPA's worst case: O(r²)
# score memory, no flash kernel. flex_attention with a BlockMask skips the fully-masked
# off-diagonal image blocks, restoring flash-like memory and ~T× less work. It is compiled
# because its eager path materializes the score tensor (defeating the point). Guarded so
# CPU / a torch without flex fall back to the dense bool-mask SDPA branch.
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention_raw, create_block_mask as _create_block_mask)
    flex_attention = torch.compile(_flex_attention_raw, dynamic=False)
    HAS_FLEX = True
except Exception:                                  # pragma: no cover - torch/platform dependent
    _create_block_mask = None
    flex_attention = None
    HAS_FLEX = False


# Escape hatch: FlopCounterMode / other TorchDispatchModes disable dynamo, so the compiled
# flex kernel can't run there (it falls to an uncountable eager HOP that also warns). Setting
# this False makes build_register_block_mask return None -> the dense bool-mask SDPA path,
# which IS countable/traceable. measure_flops toggles it; the training loop leaves it True.
_FLEX_ENABLED = True


def build_register_block_mask(n_t: int, N: int, T: int, device):
    """BlockMask for register_routed sample-axis attention over r = n_t + T·N rows.

    Registers (first n_t rows) are all-to-all; each of the T image blocks (N rows) attends
    only within its own block. Returns None when flex is unavailable so the caller falls
    back to a dense r×r bool mask (identical connectivity, no block-skipping)."""
    if not HAS_FLEX or not _FLEX_ENABLED:
        return None
    r = n_t + T * N

    def mask_mod(b, h, q, kv):
        # keep iff either endpoint is a register, or both sit in the same image block
        return (q < n_t) | (kv < n_t) | (((q - n_t) // N) == ((kv - n_t) // N))

    return _create_block_mask(mask_mod, None, None, r, r, device=device, _compile=True)


class LowerPrecisionRMSNorm(nn.RMSNorm):
    """RMSNorm that upcasts to fp32 when the input is bf16/fp16.

    autocast(enabled=False) alone does NOT upcast an already-bf16 tensor -- it only stops
    the outer autocast context from casting further inside the block. Without an explicit
    x.float(), self.weight (fp32, never cast) and x (still bf16) mismatch, so PyTorch's
    fused RMSNorm kernel declines and falls back to an unfused path (numerically correct,
    just slower -- "Cannot dispatch to fused implementation" UserWarning). x.float() makes
    the upcast genuine; .to(x.dtype) restores the caller's expected dtype on the way out."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            with torch.amp.autocast("cuda", enabled=False):
                return super().forward(x.float()).to(x.dtype)
        return super().forward(x)


class MaskConvEmbedV2(nn.Module):
    """p³ occupancy tile -> e via strided Conv3d local feature extraction, WITHOUT the global
    average-pool `MaskConvEmbed` (patchset3d.py) uses before its final projection.

    That pool is provably lossy: AdaptiveAvgPool3d(1) computes sum/N over the final s³ spatial
    cells, which is permutation-invariant BY CONSTRUCTION -- two feature maps that differ only
    in WHICH of the final s³ cells holds a given value produce IDENTICAL pooled output,
    regardless of the conv weights (confirmed empirically, docs/logs.md 2026-09-20 "mask_embed
    expressiveness"). That's exactly the kind of within-cell positional detail
    arch.mask_patch_size>1 exists to preserve.

    This variant keeps the same strided-conv front end (same translation-equivariant local
    pattern detection / parameter sharing PatchSet3D's MaskConvEmbed has) but flattens the
    final s×s×s×co feature grid and feeds the FULL vector to the projection instead of
    pooling it first -- the read-out weight matrix can assign an independent weight to every
    remaining spatial position, the same guarantee nn.Linear(p³,e) has over the raw input.
    Not a hard guarantee of "exactly as expressive as Linear" for adversarial inputs (the
    strided convs still do a real, learned p³->s³·co dimensionality reduction upstream), but
    it removes the one structurally-guaranteed loss -- and for p=8 the flattened width (256)
    is actually smaller than the raw p³ (512), so the final Linear ends up CHEAPER than the
    plain-Linear(p³,e) path, not more expensive.

    v2-only: PatchSet3D's own MaskConvEmbed is left untouched (no v1 checkpoint depends on
    this class); only PatchSetV2's arch.mask_embed="conv" uses this."""

    def __init__(self, p: int, e: int, base_ch: int = 16, groups: int = 4):
        super().__init__()
        self.p = p
        convs, ci, co, s = [], 1, base_ch, p
        while s > 2:
            convs.append(nn.Sequential(
                nn.Conv3d(ci, co, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(groups, co), co),
                nn.LeakyReLU(0.1, inplace=True),
            ))
            ci, co, s = co, co * 2, (s + 1) // 2
        self.convs = nn.Sequential(*convs)
        self.final_ch = ci if convs else 1
        self.final_s = s
        self.proj = nn.Linear(self.final_ch * self.final_s ** 3, e)

    def forward(self, occ: torch.Tensor) -> torch.Tensor:
        *lead, _ = occ.shape                              # (..., p³) -> (..., e)
        x = self.convs(occ.reshape(-1, 1, self.p, self.p, self.p))
        x = x.flatten(1)
        return self.proj(x).reshape(*lead, -1)


class ThinkingRows(nn.Module):
    """Prepend n learnable row embeddings broadcast across all patch positions."""
    def __init__(self, n: int, e: int):
        super().__init__()
        self.n = n
        self.tokens = nn.Parameter(torch.empty(n, e))
        nn.init.normal_(self.tokens)

    def forward(self, x: torch.Tensor, sep: int) -> tuple[torch.Tensor, int]:
        b, r, c, e = x.shape
        think = self.tokens.unsqueeze(0).unsqueeze(2).expand(b, -1, c, -1)
        return torch.cat([think, x], dim=1), sep + self.n


class TransformerEncoderLayer(nn.Module):
    """
    Dual-axis transformer block (or single-axis, dual_axis=False).

    Feature-axis (col-axis): full self-attention across N patches within each image row.
    Sample-axis  (row-axis): cross-image attention per patch position.
      Both context rows and query row attend only to the train set
      (thinking rows + context images); query cannot attend to itself.

    dual_axis=False drops the feature-axis block entirely (qkv_col/norm1 not even
    allocated) -- for callers whose columns are already fused into one BEFORE the
    transformer (e.g. PatchSet3D's arch.dual_axis=False pixel-shuffle token fusion) and
    want zero further per-layer column-mixing capacity, not just a vacuous c=1
    self-attention that would still cost params without being able to mix anything.
    """
    def __init__(self, a: int, e: int, h: int, dual_axis: bool = True):
        super().__init__()
        assert e % a == 0
        self.a = a
        self.d = e // a
        self.dual_axis = dual_axis
        if self.dual_axis:
            self.qkv_col = nn.Linear(e, 3 * e)
            self.norm1 = LowerPrecisionRMSNorm(e)
        self.qkv_row = nn.Linear(e, 3 * e)
        self.norm2 = LowerPrecisionRMSNorm(e)
        self.norm3 = LowerPrecisionRMSNorm(e)
        self.mlp = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, e))

    def forward(self, src: torch.Tensor, sep: int, attn_mask: torch.Tensor | None = None,
                full_attn: bool = False,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None,
                block_mask=None) -> torch.Tensor:
        b, r, c, e = src.shape
        a, d = self.a, self.d

        if self.dual_axis:
            # ── Feature-axis: spatial attention within each image ───────────────
            # batched_sdpa (not plain SDPA) so the b*r grid stays under the gridDim.y cap
            # when r is large (e.g. set-of-patches layouts where rows = all patches).
            x = src.reshape(b * r, c, e)
            res = x
            x = self.norm1(x)
            qkv = self.qkv_col(x).reshape(b * r, c, 3, a, d).permute(2, 0, 3, 1, 4)
            # c is tiny in the set-of-patches layout (2 img/mask cols) → manual attention
            # beats the fused kernel's per-problem overhead; ImagePFN's large c falls back
            # to SDPA.
            if c <= _SMALL_SEQ_ATTN:
                x = _small_seq_attn(qkv[0], qkv[1], qkv[2])
            else:
                x = batched_sdpa(qkv[0], qkv[1], qkv[2])
            x = x.transpose(1, 2).reshape(b * r, c, e)
            src = (res + x).reshape(b, r, c, e)

        # ── Sample-axis: cross-image attention per patch position ───────────────
        # Default: every row (context + query) attends only to the train set
        # (thinking+context rows, k_t/v_t = [:sep]). `full_attn` drops that slice for a
        # dense unmasked r×r attention (every row, incl. thinking+support, attends to
        # every row) — same fused SDPA kernel, no mask tensor, so it is marginally
        # cheaper; it makes context representations target-aware (breaks read-only). With
        # an explicit attn_mask, the connectivity is an (r×r) bool table instead (e.g.
        # queries also attending to queries for within-image spatial reasoning).
        x = src.permute(0, 2, 1, 3).reshape(b * c, r, e)
        res = x
        x = self.norm2(x)
        qkv = self.qkv_row(x).reshape(b * c, r, 3, a, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if rope is not None:
            # 3D axial RoPE on the sample (row) axis: rotate q,k by each row's grid
            # position before the k[:sep] slice below. cos/sin are (r, d) -> broadcast
            # over the b*c batch and a heads. v is left unrotated (standard RoPE).
            cos, sin = rope
            cos = cos.to(q.dtype)[None, None]
            sin = sin.to(q.dtype)[None, None]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if block_mask is not None:
            # register_routed: block-diagonal + register border via flex (skips masked blocks).
            x = flex_attention(q, k, v, block_mask=block_mask)
        elif attn_mask is not None:
            x = batched_sdpa(q, k, v, attn_mask=attn_mask)
        elif full_attn:
            x = batched_sdpa(q, k, v)
        else:
            x = batched_sdpa(q, k[:, :, :sep, :], v[:, :, :sep, :])
        x = x.transpose(1, 2).reshape(b * c, r, e)
        # contiguous() here ensures the next layer's feature-axis reshape is a view
        src = (res + x).reshape(b, c, r, e).permute(0, 2, 1, 3).contiguous()

        # ── MLP ────────────────────────────────────────────────────────────────
        return src + self.mlp(self.norm3(src))


class RowCrossAttention(nn.Module):
    """Row-axis cross-attention: Q rows attend into a DIFFERENT (kv) row-set's K/V, then the
    Q-side output runs the same column-axis self-attention + MLP TransformerEncoderLayer uses.
    Q and KV row counts may differ (r_q != r_kv) -- the asymmetric primitive
    TransformerEncoderLayer's self-attention (single input, single row count) can't express.
    Used by PatchSet3D's compress (Stage A: q=learned slots, kv=raw cells) and expand
    (Stage C: q=raw query cells, kv=compressed rows) stages -- same module, opposite argument
    order. No RoPE support (see PatchSet3D's arch.seq_compress/arch.transformer_rope
    incompatibility assert). See docs/superpowers/specs/2026-09-14-patchset3d-sequence
    -compression-design.md."""

    def __init__(self, a: int, e: int, h: int):
        super().__init__()
        assert e % a == 0
        self.a = a
        self.d = e // a
        self.q_proj = nn.Linear(e, e)
        self.kv_proj = nn.Linear(e, 2 * e)
        self.qkv_col = nn.Linear(e, 3 * e)
        self.norm_q = LowerPrecisionRMSNorm(e)
        self.norm_kv = LowerPrecisionRMSNorm(e)
        self.norm_col = LowerPrecisionRMSNorm(e)
        self.norm_mlp = LowerPrecisionRMSNorm(e)
        self.mlp = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, e))

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        """q_in (B,r_q,c,e), kv_in (B,r_kv,c,e), same c and e -> (B,r_q,c,e)."""
        b, rq, c, e = q_in.shape
        rkv = kv_in.shape[1]
        a, d = self.a, self.d

        # -- Row-axis cross-attention: rq queries read rkv keys/values, per column --------
        qn = self.norm_q(q_in).permute(0, 2, 1, 3).reshape(b * c, rq, e)
        kn = self.norm_kv(kv_in).permute(0, 2, 1, 3).reshape(b * c, rkv, e)
        qh = self.q_proj(qn).reshape(b * c, rq, a, d).transpose(1, 2)          # (b*c,a,rq,d)
        kvh = self.kv_proj(kn).reshape(b * c, rkv, 2, a, d).permute(2, 0, 3, 1, 4)
        x = batched_sdpa(qh, kvh[0], kvh[1])                                    # (b*c,a,rq,d)
        x = x.transpose(1, 2).reshape(b * c, rq, e).reshape(b, c, rq, e).permute(0, 2, 1, 3)
        q_in = q_in + x                                                         # (b,rq,c,e)

        # -- Column-axis self-attention (img/mask mix) -- identical shape/logic to
        # TransformerEncoderLayer's feature-axis block, applied to the cross-attended Q ---
        y = q_in.reshape(b * rq, c, e)
        res = y
        y = self.norm_col(y)
        qkv = self.qkv_col(y).reshape(b * rq, c, 3, a, d).permute(2, 0, 3, 1, 4)
        if c <= _SMALL_SEQ_ATTN:
            y = _small_seq_attn(qkv[0], qkv[1], qkv[2])
        else:
            y = batched_sdpa(qkv[0], qkv[1], qkv[2])
        y = y.transpose(1, 2).reshape(b * rq, c, e)
        q_in = (res + y).reshape(b, rq, c, e)

        # -- MLP --
        return q_in + self.mlp(self.norm_mlp(q_in))


class DecodeCrossBlock(nn.Module):
    """One round of bidirectional cross-attention between a small task-token set T (m-scale,
    e.g. PatchSetV2's compress_m+1 rows) and a large per-cell grid F (N=R^3 rows): T reads F,
    then F reads the UPDATED T (sequential, not simultaneous -- see docs/methods/iris.md's
    Eq 5 and the fidelity discussion in docs/logs.md 2026-09-20), then each side gets its own
    MLP. Stacking L of these (PatchSetV2's arch.decode_layers) lets F accumulate context over
    several hops before the conv decoder -- mirroring PatchSet3D's fine_filter decode getting
    its query tokens from _attn's l dense self-attention layers, but staying m-scale per hop
    (O(N*m), never O(N*N) -- see the R^3-vs-m benchmark, docs/logs.md 2026-09-20)."""

    def __init__(self, e: int, a: int, h: int):
        super().__init__()
        self.t2f = nn.MultiheadAttention(e, a, batch_first=True)
        self.f2t = nn.MultiheadAttention(e, a, batch_first=True)
        self.mlp_t = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, e))
        self.mlp_f = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, e))
        self.norm_t1 = LowerPrecisionRMSNorm(e)
        self.norm_f1 = LowerPrecisionRMSNorm(e)
        self.norm_t2 = LowerPrecisionRMSNorm(e)
        self.norm_f2 = LowerPrecisionRMSNorm(e)

    def forward(self, T: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """T (B,r_t,e), F (B,r_f,e) -> updated (T,F), same shapes."""
        T = T + self.t2f(self.norm_t1(T), F, F)[0]
        F = F + self.f2t(self.norm_f1(F), T, T)[0]
        T = T + self.mlp_t(self.norm_t2(T))
        F = F + self.mlp_f(self.norm_f2(F))
        return T, F


class TransformerEncoderStack(nn.Module):
    def __init__(self, l: int, a: int, e: int, h: int, residual_decay: float,
                 dual_axis: bool = True):
        super().__init__()
        self.residual_decay = residual_decay
        self.blocks = nn.ModuleList(
            [TransformerEncoderLayer(a, e, h, dual_axis=dual_axis) for _ in range(l)])

    def forward(self, x: torch.Tensor, sep: int, attn_mask: torch.Tensor | None = None,
                full_attn: bool = False,
                rope: tuple[torch.Tensor, torch.Tensor] | None = None,
                block_mask=None) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = x * (self.residual_decay ** i)
            x = block(x, sep, attn_mask=attn_mask, full_attn=full_attn, rope=rope,
                      block_mask=block_mask)
        return x


class ImagePFN(nn.Module):
    """
    In-context 2D image segmentation.

    Given K context (image, mask) pairs and a query image, predicts a binary
    segmentation mask for the query via dual-axis in-context attention.

    Column layout per row: [img_0 … img_{N-1} | mask_0 … mask_{N-1}]  (2N total cols).
    Image and mask patches are separate column groups so feature-axis attention can
    explicitly route information between them.  The decoder reads only the N image
    cols of the query row, mirroring nanoTabPFN's output[:, sep:, :-1, :] design.

    TargetEncoder trick: query mask cols are filled with the mean of context mask
    patches before embedding — providing a class-frequency prior rather than zeros.

    Args:
        resolution: patches per side; output grid Hp = resolution, total
            N = resolution² patches. Effective patch size P = image_size // resolution.
        image_size: expected spatial resolution (H = W)
        input_patch_size: side length Q every patch is resized to before embedding.
            Each native P×P patch is interpolated to Q×Q so the embedding input dim
            (Q²) is fixed regardless of the effective patch size P.
        e: embedding dimension
        h: MLP hidden size
        l: number of transformer layers
        a: number of attention heads
        thinking_rows: prepended learnable row tokens
        residual_decay: per-layer decay factor (input to block i scaled by decay^i)
        image_encoder: optional frozen feature encoder (e.g. UniverSegFeatureEncoder).
            When given, the image path becomes encoder → resolution×resolution feature
            grid → Linear(feature_dim, e) instead of raw-pixel patchify. The mask path
            is unchanged (raw P×P patches resized to Q×Q). The encoder is injected (not
            imported here) so this module stays dependency-light.
        feature_dim: channel count of the encoder's pooled features; required when
            image_encoder is provided or use_external_features is True (sets the
            image_embed input dim).
    """
    def __init__(
        self,
        resolution: int = 16,
        image_size: int = 128,
        input_patch_size: int = 8,
        e: int = 256,
        h: int = 512,
        l: int = 6,
        a: int = 4,
        thinking_rows: int = 8,
        residual_decay: float = 0.95,
        image_encoder: nn.Module | None = None,
        feature_dim: int | None = None,
        use_external_features: bool = False,
    ):
        super().__init__()
        assert image_size % resolution == 0, "image_size must be divisible by resolution"
        P = image_size // resolution            # effective (native) patch size
        Q = input_patch_size                    # fixed embedding input size
        N = resolution ** 2
        self.patch_size = P
        self.input_patch_size = Q
        self.N = N

        self.image_encoder = image_encoder
        self.use_external_features = use_external_features
        if image_encoder is not None:
            assert feature_dim is not None, "feature_dim required with image_encoder"
            self.image_embed = nn.Linear(feature_dim, e)   # embed pretrained features
        elif use_external_features:
            # Features are computed outside (e.g. the zoom pipeline crop-pools encoder maps)
            # and passed to forward(image_feats=...); no internal encoder submodule.
            assert feature_dim is not None, "feature_dim required with use_external_features"
            self.image_embed = nn.Linear(feature_dim, e)
        else:
            self.image_embed = nn.Linear(Q * Q, e)         # embed raw pixel patches
        self.mask_embed  = nn.Linear(Q * Q, e)             # mask path always raw patches
        # Shared positional embedding applied to both image and mask col groups
        self.pos_embed   = nn.Parameter(torch.zeros(1, 1, N, e))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.thinking    = ThinkingRows(thinking_rows, e)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
        # Decode per image-patch position from query row (no spatial pooling)
        self.decoder     = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, 1))

    def forward(
        self,
        images: torch.Tensor | None,  # (B, K+1, 1, H, W) — last row is query; may be None if image_feats given
        masks:  torch.Tensor,  # (B, K+1, 1, H, W) — query mask is replaced below (unless seed_query_mask)
        sep:    int,           # K = number of context images
        return_thinking: bool = False,
        image_feats: torch.Tensor | None = None,   # (B,T,N,Cf) precomputed → skip encoding
        seed_query_mask: bool = False,             # keep query mask as passed (no context-mean)
    ):                         # (B, H//P, W//P) logits, or (logits, thinking) if return_thinking
        ref = images if images is not None else masks
        B, T, _, H, W = ref.shape
        P, N, Q = self.patch_size, self.N, self.input_patch_size
        Hp = H // P

        # ── Image cols ─────────────────────────────────────────────────────────
        if image_feats is not None:
            # Precomputed features (e.g. zoom pipeline crop-pooled encoder maps).
            img_p = standardize_by_context(image_feats, sep)
        elif self.image_encoder is not None:
            # Pretrained features: encode each image → Hp×Hp feature grid → (B,T,N,C).
            # Patch order (row-major over the grid) matches pos_embed and the decoder
            # reshape. Per-channel context-stat standardization (shared with the
            # multilevel pipeline via standardize_by_context).
            feat  = self.image_encoder(images.reshape(B * T, 1, H, W), Hp)  # (B*T, C, Hp, Hp)
            img_p = feat.flatten(2).transpose(1, 2).reshape(B, T, N, feat.shape[1])
            img_p = standardize_by_context(img_p, sep)
        else:
            # Raw pixels: native P×P patches each resized to Q×Q → (B,T,N,Q²).
            # Scalar (not per-channel) normalization by context stats — the Q² columns
            # are one grayscale patch, not independent feature channels.
            img_p = patchify(images.reshape(B * T, 1, H, W), P, out=Q).reshape(B, T, N, Q * Q)
            mu  = img_p[:, :sep].mean(dim=(1, 2, 3), keepdim=True)         # (B,1,1,1) scalar
            sig = img_p[:, :sep].std( dim=(1, 2, 3), keepdim=True) + 1e-8
            img_p = ((img_p - mu) / sig).clamp(-10, 10)

        # ── Mask cols (always raw P×P patches resized to Q×Q) ──────────────────
        mask_p = patchify(masks.reshape(B * T, 1, H, W), P, out=Q).reshape(B, T, N, Q * Q)

        # TargetEncoder trick: replace query mask patches with mean of context masks —
        # unless seed_query_mask, in which case the caller already put a real prior
        # (e.g. the cropped coarse prediction) in the query rows.
        if not seed_query_mask:
            ctx_mask_mean = mask_p[:, :sep].mean(dim=1, keepdim=True)        # (B, 1, N, Q²)
            mask_p = torch.cat(
                [mask_p[:, :sep], ctx_mask_mean.expand(B, T - sep, N, Q * Q)],
                dim=1,
            )                                                                 # (B, T, N, Q²)

        # Separate col groups; cat along col dim → (B, T, 2N, e)
        x_img  = self.image_embed(img_p)  + self.pos_embed   # (B, T, N, e)
        x_mask = self.mask_embed(mask_p)  + self.pos_embed   # (B, T, N, e)
        x = torch.cat([x_img, x_mask], dim=2)                # (B, T, 2N, e)

        # Thinking rows + dual-axis transformer
        x, sep_t = self.thinking(x, sep)        # sep_t = n_thinking + K
        x = self.transformer(x, sep_t)

        # Decode from image cols only (first N) of the query row
        query  = x[:, sep_t:, :N, :].squeeze(1)              # (B, N, e)
        logits = self.decoder(query).squeeze(-1).reshape(B, Hp, Hp)

        if return_thinking:
            # Post-transformer thinking rows, mean-pooled over the 2N columns →
            # a compact per-row latent summary of the coarse task. (B, n_think, e)
            think = x[:, :self.thinking.n].mean(dim=2)
            return logits, think
        return logits

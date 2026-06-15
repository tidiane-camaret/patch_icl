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

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LowerPrecisionRMSNorm(nn.RMSNorm):
    """RMSNorm that upcasts to fp32 when the input is bf16/fp16."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            with torch.amp.autocast("cuda", enabled=False):
                return super().forward(x)
        return super().forward(x)


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
    Dual-axis transformer block.

    Feature-axis (col-axis): full self-attention across N patches within each image row.
    Sample-axis  (row-axis): cross-image attention per patch position.
      Both context rows and query row attend only to the train set
      (thinking rows + context images); query cannot attend to itself.
    """
    def __init__(self, a: int, e: int, h: int):
        super().__init__()
        assert e % a == 0
        self.a = a
        self.d = e // a
        self.qkv_col = nn.Linear(e, 3 * e)
        self.qkv_row = nn.Linear(e, 3 * e)
        self.norm1 = LowerPrecisionRMSNorm(e)
        self.norm2 = LowerPrecisionRMSNorm(e)
        self.norm3 = LowerPrecisionRMSNorm(e)
        self.mlp = nn.Sequential(nn.Linear(e, h), nn.GELU(), nn.Linear(h, e))

    def forward(self, src: torch.Tensor, sep: int, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, r, c, e = src.shape
        a, d = self.a, self.d

        # ── Feature-axis: spatial attention within each image ──────────────────
        x = src.reshape(b * r, c, e)
        res = x
        x = self.norm1(x)
        qkv = self.qkv_col(x).reshape(b * r, c, 3, a, d).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = x.transpose(1, 2).reshape(b * r, c, e)
        src = (res + x).reshape(b, r, c, e)

        # ── Sample-axis: cross-image attention per patch position ───────────────
        # Default: every row (context + query) attends only to the train set
        # (thinking+context rows, k_t/v_t = [:sep]). With an explicit attn_mask, the
        # connectivity is given as an (r×r) bool table instead (e.g. queries also
        # attending to queries for within-image spatial reasoning).
        x = src.permute(0, 2, 1, 3).reshape(b * c, r, e)
        res = x
        x = self.norm2(x)
        qkv = self.qkv_row(x).reshape(b * c, r, 3, a, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if attn_mask is None:
            x = F.scaled_dot_product_attention(q, k[:, :, :sep, :], v[:, :, :sep, :])
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(b * c, r, e)
        # contiguous() here ensures the next layer's feature-axis reshape is a view
        src = (res + x).reshape(b, c, r, e).permute(0, 2, 1, 3).contiguous()

        # ── MLP ────────────────────────────────────────────────────────────────
        return src + self.mlp(self.norm3(src))


class TransformerEncoderStack(nn.Module):
    def __init__(self, l: int, a: int, e: int, h: int, residual_decay: float):
        super().__init__()
        self.residual_decay = residual_decay
        self.blocks = nn.ModuleList([TransformerEncoderLayer(a, e, h) for _ in range(l)])

    def forward(self, x: torch.Tensor, sep: int, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = x * (self.residual_decay ** i)
            x = block(x, sep, attn_mask=attn_mask)
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
            image_encoder is provided (sets the image_embed input dim).
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
        if image_encoder is not None:
            assert feature_dim is not None, "feature_dim required with image_encoder"
            self.image_embed = nn.Linear(feature_dim, e)   # embed pretrained features
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
        images: torch.Tensor,  # (B, K+1, 1, H, W) — last row is query
        masks:  torch.Tensor,  # (B, K+1, 1, H, W) — query mask is replaced below
        sep:    int,           # K = number of context images
        return_thinking: bool = False,
    ):                         # (B, H//P, W//P) logits, or (logits, thinking) if return_thinking
        B, T, _, H, W = images.shape
        P, N, Q = self.patch_size, self.N, self.input_patch_size
        Hp = H // P

        # ── Image cols ─────────────────────────────────────────────────────────
        if self.image_encoder is not None:
            # Pretrained features: encode each image → Hp×Hp feature grid → (B,T,N,C).
            # Patch order (row-major over the grid) matches pos_embed and the decoder
            # reshape. Normalize per channel using context-image statistics.
            feat  = self.image_encoder(images.reshape(B * T, 1, H, W), Hp)  # (B*T, C, Hp, Hp)
            img_p = feat.flatten(2).transpose(1, 2).reshape(B, T, N, feat.shape[1])
            mu  = img_p[:, :sep].mean(dim=(1, 2), keepdim=True)            # (B,1,1,C) per-channel
            sig = img_p[:, :sep].std( dim=(1, 2), keepdim=True) + 1e-8
        else:
            # Raw pixels: native P×P patches each resized to Q×Q → (B,T,N,Q²).
            img_p = patchify(images.reshape(B * T, 1, H, W), P, out=Q).reshape(B, T, N, Q * Q)
            mu  = img_p[:, :sep].mean(dim=(1, 2, 3), keepdim=True)         # (B,1,1,1) scalar
            sig = img_p[:, :sep].std( dim=(1, 2, 3), keepdim=True) + 1e-8
        img_p = ((img_p - mu) / sig).clamp(-10, 10)

        # ── Mask cols (always raw P×P patches resized to Q×Q) ──────────────────
        mask_p = patchify(masks.reshape(B * T, 1, H, W), P, out=Q).reshape(B, T, N, Q * Q)

        # TargetEncoder trick: replace query mask patches with mean of context masks
        ctx_mask_mean = mask_p[:, :sep].mean(dim=1, keepdim=True)           # (B, 1, N, Q²)
        mask_p = torch.cat(
            [mask_p[:, :sep], ctx_mask_mean.expand(B, T - sep, N, Q * Q)],
            dim=1,
        )                                                                     # (B, T, N, Q²)

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

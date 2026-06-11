"""
ImagePFN: in-context 2D image segmentation via dual-axis transformer.

Tensor layout throughout: (batch, rows, cols, e)
  rows = n_thinking + K context images + 1 query image
  cols = 2N = 2 × (image_size // patch_size)²
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


def patchify(x: torch.Tensor, P: int) -> torch.Tensor:
    """(B, 1, H, W) → (B, N, P²), N = (H//P)*(W//P)."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // P) * (W // P), C * P * P)


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

    def forward(self, src: torch.Tensor, sep: int) -> torch.Tensor:
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
        # All rows (context + query) attend to the same k_t/v_t (thinking+context
        # rows only) — single SDPA call covers both; avoids the split+cat.
        x = src.permute(0, 2, 1, 3).reshape(b * c, r, e)
        res = x
        x = self.norm2(x)
        qkv = self.qkv_row(x).reshape(b * c, r, 3, a, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t, v_t = k[:, :, :sep, :], v[:, :, :sep, :]
        x = F.scaled_dot_product_attention(q, k_t, v_t).transpose(1, 2).reshape(b * c, r, e)
        # contiguous() here ensures the next layer's feature-axis reshape is a view
        src = (res + x).reshape(b, c, r, e).permute(0, 2, 1, 3).contiguous()

        # ── MLP ────────────────────────────────────────────────────────────────
        return src + self.mlp(self.norm3(src))


class TransformerEncoderStack(nn.Module):
    def __init__(self, l: int, a: int, e: int, h: int, residual_decay: float):
        super().__init__()
        self.residual_decay = residual_decay
        self.blocks = nn.ModuleList([TransformerEncoderLayer(a, e, h) for _ in range(l)])

    def forward(self, x: torch.Tensor, sep: int) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = x * (self.residual_decay ** i)
            x = block(x, sep)
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
        patch_size: side length P; images split into N = (H//P)² patches, 2N total cols
        image_size: expected spatial resolution (H = W)
        e: embedding dimension
        h: MLP hidden size
        l: number of transformer layers
        a: number of attention heads
        thinking_rows: prepended learnable row tokens
        residual_decay: per-layer decay factor (input to block i scaled by decay^i)
    """
    def __init__(
        self,
        patch_size: int = 8,
        image_size: int = 128,
        e: int = 256,
        h: int = 512,
        l: int = 6,
        a: int = 4,
        thinking_rows: int = 8,
        residual_decay: float = 0.95,
    ):
        super().__init__()
        P = patch_size
        N = (image_size // P) ** 2
        self.patch_size = P
        self.N = N

        self.image_embed = nn.Linear(P * P, e)
        self.mask_embed  = nn.Linear(P * P, e)
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
    ) -> torch.Tensor:         # (B, H//P, W//P) logits
        B, T, _, H, W = images.shape
        P, N = self.patch_size, self.N
        Hp = H // P

        # Patchify: (B, T, N, P²)
        img_p  = patchify(images.reshape(B * T, 1, H, W), P).reshape(B, T, N, P * P)
        mask_p = patchify(masks .reshape(B * T, 1, H, W), P).reshape(B, T, N, P * P)

        # Normalize image patches with context-image statistics
        mu  = img_p[:, :sep].mean(dim=(1, 2, 3), keepdim=True)
        sig = img_p[:, :sep].std( dim=(1, 2, 3), keepdim=True) + 1e-8
        img_p = ((img_p - mu) / sig).clamp(-10, 10)

        # TargetEncoder trick: replace query mask patches with mean of context masks
        ctx_mask_mean = mask_p[:, :sep].mean(dim=1, keepdim=True)           # (B, 1, N, P²)
        mask_p = torch.cat(
            [mask_p[:, :sep], ctx_mask_mean.expand(B, T - sep, N, P * P)],
            dim=1,
        )                                                                     # (B, T, N, P²)

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
        return logits

"""
PatchICLAttention: learned in-context binary classifier for patch-level segmentation.

Each context patch (feature vector + binary label) is a training example.
Cross-attention lets target patches attend to context patches, optionally
conditioned on context labels, then a head decodes probabilities.

Architecture decisions (all configurable):
  label_injection : "additive" | "concat" | "gate" | "none"
  output_head     : "linear" | "mlp" | "retrieval"
  pos_encoding    : "none" | "sinusoidal" | "learned" | "rope3d"
  input_norm      : "none" | "rmsnorm" | "l2"
  ctx_self_attn   : bool — context tokens self-attend before each cross-attn block
  log_n_scaling   : bool — scale cross-attn queries by log(n_ctx)/log(n_base)

rope3d notes
------------
When pos_encoding="rope3d", integer (d, h, w) coordinates are passed as optional
tgt_coords / ctx_coords to forward().  RoPE is applied inside _mha() directly to
Q and K after projection, so any token subset (dense or sparse) works without
needing the tokens to fill a complete grid.  If coords are not supplied the model
falls back to no positional encoding for that call.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rope3d import build_rope_cache_3d, apply_rope_3d


class ContinuousScaleEncoding(nn.Module):
    """Maps a per-sample physical patch size (mm) to a dim-dimensional embedding.

    Uses log-spaced sinusoidal functions with learnable frequencies, identical to
    patch_icl_v3's ContinuousScaleEncoding.  Input is log(scale_mm) so the model
    sees a roughly linear signal across the typical range (1–100 mm/patch).
    """

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.arange(half).float() * -(math.log(10000.0) / max(half - 1, 1)))
        self.freqs = nn.Parameter(freqs)

    def forward(self, scale_mm: torch.Tensor) -> torch.Tensor:
        """scale_mm: (B,) → (B, dim)"""
        log_s = scale_mm.clamp(min=1e-3).log().unsqueeze(1)   # (B, 1)
        args  = log_s * self.freqs.unsqueeze(0)                # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class PatchICLAttention(nn.Module):
    """
    Args
    ----
    embed_dim       : raw encoder feature dimension (e.g. 1504 for STU-Net all-levels)
    dim             : internal transformer working dimension (projected from embed_dim)
    num_heads       : MHA heads (dim must be divisible by num_heads)
    num_layers      : stacked cross-attention + FFN blocks
    ff_factor       : FFN hidden dim = dim * ff_factor
    label_injection : how context binary labels enter the context token representation
                        additive  — token += label_embed(y)
                        concat    — token = proj([token; label_embed(y)])
                        gate      — token = token * sigmoid(proj(label_embed(y)))
                        none      — labels only appear in the output head
    output_head     : how final target embeddings decode to probabilities
                        linear    — Linear(dim, 1) + sigmoid
                        mlp       — RMSNorm → Linear → GELU → Linear + sigmoid
                        retrieval — cross-attention(Q=tgt, K=ctx, V=ctx_labels)
    pos_encoding    : spatial position encoding for the D×H×W patch grid
                        none       — no positional information
                        sinusoidal — fixed 3D sinusoidal encoding (dense grids only)
                        learned    — nn.Embedding(D*H*W, dim) (dense grids only)
                        rope3d     — 3D RoPE applied inside Q/K projections;
                                     pass tgt_coords / ctx_coords (B, N, 3) int to
                                     forward().  Works for any token subset.
    input_norm      : applied to raw encoder features before the input projection
                        none    — use encoder features as-is
                        rmsnorm — RMSNorm per token (at embed_dim)
                        l2      — unit-norm per token
    grid_size       : (D, H, W) of the output patch grid (for pos encoding)
    dropout         : attention dropout probability (applied during training only)
    ctx_self_attn   : if True, context tokens self-attend (full transformer block)
                        before each cross-attention, letting context patches interact
                        across K context images before being retrieved
    log_n_scaling   : if True, scale cross-attention queries by log(n_ctx)/log(n_base)
                        to compensate for softmax flattening over large context sequences
    log_n_base      : reference context size for log-n normalisation (default 512 = 1×8³)
    """

    def __init__(
        self,
        embed_dim:       int,
        dim:             int   = 256,
        num_heads:       int   = 8,
        num_layers:      int   = 1,
        ff_factor:       int   = 2,
        label_injection: str   = "additive",
        output_head:     str   = "linear",
        pos_encoding:    str   = "none",
        input_norm:      str   = "none",
        grid_size: tuple[int, int, int] = (16, 16, 16),
        dropout:         float = 0.0,
        ctx_self_attn:   bool  = True,
        log_n_scaling:   bool  = True,
        log_n_base:      int   = 512,
        label_dim:       int   = 1,
        soft_labels:     bool  = False,
        rope_max_pos:    int   = 0,  # override max_pos for rope3d cache (0 = use max(grid_size))
        output_dim:      int   = 0,  # output head size; 0 = same as label_dim (default)
        num_registers:   int   = 0,  # learnable context summary tokens, cascaded between levels
        append_zero_attn: bool = False,  # add null K/V slot to cross-attention
        use_scale_embed: bool  = False,  # inject ContinuousScaleEncoding after input_proj
        use_role_embed:  bool  = False,  # inject target/context type + context-index embeddings
        max_context_size: int  = 8,      # upper bound on K for ctx_idx_embed table
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert label_injection in ("additive", "concat", "gate", "none")
        assert output_head in ("linear", "mlp", "retrieval")
        assert pos_encoding in ("none", "sinusoidal", "learned", "rope3d")
        assert input_norm in ("none", "rmsnorm", "l2")

        self.embed_dim          = embed_dim   # raw encoder dim (input only)
        self.dim                = dim         # internal working dim
        self.num_heads          = num_heads
        self.head_dim           = dim // num_heads
        self.num_layers         = num_layers
        self.label_injection    = label_injection
        self.output_head_type   = output_head
        self.pos_enc_type       = pos_encoding
        self.input_norm_type    = input_norm
        self.grid_size          = grid_size
        self.dropout            = dropout
        self.ctx_self_attn_flag = ctx_self_attn
        self.log_n_scaling      = log_n_scaling
        self.log_n_base         = log_n_base
        self.label_dim          = label_dim
        self.soft_labels        = soft_labels
        self.output_dim         = output_dim if output_dim > 0 else label_dim
        self.num_registers      = num_registers
        self.append_zero_attn   = append_zero_attn
        self.use_scale_embed    = use_scale_embed
        self.scale_encoder      = ContinuousScaleEncoding(dim) if use_scale_embed else None
        self.use_role_embed     = use_role_embed
        if use_role_embed:
            # Zero-init so they start as identity; model learns to use them.
            self.tgt_type_embed  = nn.Parameter(torch.zeros(1, 1, dim))
            self.ctx_type_embed  = nn.Parameter(torch.zeros(1, 1, dim))
            self.ctx_idx_embed   = nn.Embedding(max_context_size, dim)
            nn.init.zeros_(self.ctx_idx_embed.weight)
        else:
            self.tgt_type_embed = self.ctx_type_embed = self.ctx_idx_embed = None
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, dim) * 0.02)
        else:
            self.register_tokens = None

        # ---- Input normalization (at embed_dim, before projection) ------
        self.input_norm = nn.RMSNorm(embed_dim) if input_norm == "rmsnorm" else None

        # ---- Input projection: embed_dim → dim --------------------------
        self.input_proj = nn.Linear(embed_dim, dim, bias=False)

        # ---- Label injection (operates at dim) --------------------------
        # label_dim=1: binary discrete path — nn.Embedding(2, dim) maps {0,1} → dim.
        # label_dim>1: continuous RGB path — bias-free Linear(label_dim, dim) so black
        #   (all-zero background) injects nothing and foreground scales with its color.
        if label_injection in ("additive", "gate", "concat"):
            if label_dim == 1 and not soft_labels:
                self.label_embed = nn.Embedding(2, dim)
            else:
                # soft_labels=True or label_dim>1: bias-free linear so 0 → zero injection
                self.label_embed = nn.Linear(label_dim, dim, bias=False)
            if label_injection == "gate":
                self.gate_proj = nn.Linear(dim, dim)
            elif label_injection == "concat":
                self.concat_proj = nn.Linear(2 * dim, dim)

        # ---- Positional encoding (at dim) -------------------------------
        if pos_encoding == "learned":
            N = grid_size[0] * grid_size[1] * grid_size[2]
            self.pos_embed = nn.Embedding(N, dim)
        elif pos_encoding == "rope3d":
            _max_pos = rope_max_pos if rope_max_pos > 0 else max(grid_size)
            rope_cache = build_rope_cache_3d(max_pos=_max_pos, dim=dim, num_heads=num_heads)
            self.register_buffer("rope_cache_3d", rope_cache)  # not a param

        ff_dim = dim * ff_factor

        # ---- Cross-attention layers (target Q, context K/V) -------------
        # Using manual projections + F.scaled_dot_product_attention for full control
        # over K/V normalization and log-n query scaling. Q/K/V use bias=False;
        # out_proj and FFN retain bias (standard practice).
        self.q_projs    = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])
        self.k_projs    = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])
        self.v_projs    = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])
        self.out_projs  = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.attn_norms = nn.ModuleList([nn.RMSNorm(dim) for _ in range(num_layers)])  # pre-norm for Q
        self.kv_norms   = nn.ModuleList([nn.RMSNorm(dim) for _ in range(num_layers)])  # pre-norm for K/V

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.RMSNorm(dim) for _ in range(num_layers)])

        # ---- Context self-attention blocks (optional) -------------------
        if ctx_self_attn:
            self.ctx_q_projs   = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])
            self.ctx_k_projs   = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])
            self.ctx_v_projs   = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])
            self.ctx_out_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
            self.ctx_sa_norms  = nn.ModuleList([nn.RMSNorm(dim) for _ in range(num_layers)])
            self.ctx_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim),
                )
                for _ in range(num_layers)
            ])
            self.ctx_ffn_norms = nn.ModuleList([nn.RMSNorm(dim) for _ in range(num_layers)])

        # ---- Output head (at dim) ---------------------------------------
        _out = self.output_dim
        if output_head == "linear":
            self.head = nn.Linear(dim, _out)
        elif output_head == "mlp":
            self.head = nn.Sequential(
                nn.RMSNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, _out),
            )
        elif output_head == "retrieval":
            self.ret_q_proj = nn.Linear(dim, dim)
            self.ret_k_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-init all residual output projections for stable early training."""
        for proj in self.out_projs:
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        for ffn in self.ffns:
            nn.init.zeros_(ffn[-1].weight)
            if ffn[-1].bias is not None:
                nn.init.zeros_(ffn[-1].bias)
        if self.ctx_self_attn_flag:
            for proj in self.ctx_out_projs:
                nn.init.zeros_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
            for ffn in self.ctx_ffns:
                nn.init.zeros_(ffn[-1].weight)
                if ffn[-1].bias is not None:
                    nn.init.zeros_(ffn[-1].bias)

    # ------------------------------------------------------------------
    # Low-level attention helper
    # ------------------------------------------------------------------

    def _mha(
        self,
        q:        torch.Tensor,    # (B, Sq, dim) — already pre-normed
        k:        torch.Tensor,    # (B, Sk, dim) — already pre-normed
        v:        torch.Tensor,    # (B, Sk, dim) — already pre-normed (same as k)
        q_proj:   nn.Linear,
        k_proj:   nn.Linear,
        v_proj:   nn.Linear,
        out_proj: nn.Linear,
        q_scale:  float = 1.0,
        q_coords: torch.Tensor | None = None,  # (B, Sq, 3) int — for rope3d
        k_coords: torch.Tensor | None = None,  # (B, Sk, 3) int — for rope3d
        zero_attn: bool = False,
    ) -> torch.Tensor:
        """Project Q/K/V, apply RoPE if configured, run SDPA, project output."""
        B, Sq, _ = q.shape
        Sk = k.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = q_proj(q)   # (B, Sq, dim) — RoPE applied before head split
        K = k_proj(k)   # (B, Sk, dim)
        V = v_proj(v)

        if self.pos_enc_type == "rope3d":
            if q_coords is not None:
                Q = apply_rope_3d(Q, q_coords, self.rope_cache_3d)
            if k_coords is not None:
                K = apply_rope_3d(K, k_coords, self.rope_cache_3d)

        Q = Q.view(B, Sq, H, D).transpose(1, 2)   # (B, H, Sq, D)
        K = K.view(B, Sk, H, D).transpose(1, 2)   # (B, H, Sk, D)
        V = V.view(B, Sk, H, D).transpose(1, 2)   # (B, H, Sk, D)

        if q_scale != 1.0:
            Q = Q * q_scale

        if zero_attn:
            zero_k = torch.zeros(B, H, 1, D, device=K.device, dtype=K.dtype)
            zero_v = torch.zeros(B, H, 1, D, device=V.device, dtype=V.dtype)
            K = torch.cat([K, zero_k], dim=2)
            V = torch.cat([V, zero_v], dim=2)

        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)   # (B, H, Sq, D)
        return out_proj(out.transpose(1, 2).reshape(B, Sq, self.dim))

    # ------------------------------------------------------------------
    # Positional encoding helpers
    # ------------------------------------------------------------------

    def _sinusoidal_pos(self, N: int, device: torch.device) -> torch.Tensor:
        """Fixed 3D sinusoidal encoding for D×H×W regular grid."""
        D, H, W = self.grid_size
        d = torch.linspace(0, 1, D, device=device)
        h = torch.linspace(0, 1, H, device=device)
        w = torch.linspace(0, 1, W, device=device)
        gd, gh, gw = torch.meshgrid(d, h, w, indexing="ij")
        coords = torch.stack([gd, gh, gw], dim=-1).reshape(N, 3)   # (N, 3)

        dim_per_axis = self.dim // 6
        if dim_per_axis == 0:
            return torch.zeros(N, self.dim, device=device)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(dim_per_axis, device=device).float()
            / max(dim_per_axis - 1, 1)
        )
        parts = []
        for ax in range(3):
            args = coords[:, ax:ax + 1] * freqs.unsqueeze(0)   # (N, dim_per_axis)
            parts += [torch.sin(args), torch.cos(args)]
        pos = torch.cat(parts, dim=-1)                          # (N, 6*dim_per_axis)
        if pos.shape[-1] < self.dim:
            pos = F.pad(pos, (0, self.dim - pos.shape[-1]))
        return pos[:, :self.dim]

    def _apply_pos(
        self, tgt: torch.Tensor, ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tgt: (B, N, dim)  ctx: (B, M, dim)  with M = K*N
        N, M = tgt.shape[1], ctx.shape[1]
        if self.pos_enc_type == "sinusoidal":
            tgt_pos = self._sinusoidal_pos(N, tgt.device)          # (N, dim)
            ctx_pos = self._sinusoidal_pos(N, ctx.device).repeat(M // N, 1)[:M]
            return tgt + tgt_pos, ctx + ctx_pos
        elif self.pos_enc_type == "learned":
            idx = torch.arange(N, device=tgt.device)
            tgt_pos = self.pos_embed(idx)                           # (N, dim)
            ctx_pos = tgt_pos.repeat(M // N, 1)[:M]
            return tgt + tgt_pos, ctx + ctx_pos
        return tgt, ctx

    # ------------------------------------------------------------------

    def forward(
        self,
        tgt_feat:   torch.Tensor,              # (B, N, embed_dim)
        ctx_feat:   torch.Tensor,              # (B, M, embed_dim)  M = K*N
        ctx_labels: torch.Tensor,              # (B, M) or (B, M, label_dim)
        tgt_coords: torch.Tensor | None = None,  # (B, N, 3) int — d,h,w for rope3d
        ctx_coords: torch.Tensor | None = None,  # (B, M, 3) int — d,h,w for rope3d
        cascade_registers: torch.Tensor | None = None,  # (B, R_cas, dim) from prev level
        scale_mm:   torch.Tensor | None = None,  # (B,) physical mm per patch at this level
    ) -> torch.Tensor:                         # (B, N) or (B, N, label_dim)
        B = tgt_feat.shape[0]
        M = ctx_feat.shape[1]

        # 1. Input normalization (at embed_dim)
        if self.input_norm_type == "l2":
            tgt_feat = F.normalize(tgt_feat, dim=-1)
            ctx_feat = F.normalize(ctx_feat, dim=-1)
        elif self.input_norm is not None:
            tgt_feat = self.input_norm(tgt_feat)
            ctx_feat = self.input_norm(ctx_feat)

        # 2. Project embed_dim → dim
        tgt = self.input_proj(tgt_feat)   # (B, N, dim)
        ctx = self.input_proj(ctx_feat)   # (B, M, dim)

        # 2b. Scale embedding: add physical patch-size signal to all tokens
        if self.scale_encoder is not None and scale_mm is not None:
            s = self.scale_encoder(scale_mm).unsqueeze(1)  # (B, 1, dim)
            tgt = tgt + s
            ctx = ctx + s

        # 2c. Role embeddings: target/context type + per-context-image index
        if self.use_role_embed:
            tgt = tgt + self.tgt_type_embed                    # broadcast over (B, N, dim)
            ctx = ctx + self.ctx_type_embed                    # broadcast over (B, K*NP, dim)
            NP = tgt.shape[1]
            K  = ctx.shape[1] // NP
            k_idx = torch.arange(K, device=ctx.device).repeat_interleave(NP)  # (K*NP,)
            ctx = ctx + self.ctx_idx_embed(k_idx).unsqueeze(0)  # (B, K*NP, dim)

        # 3. Label injection into context tokens (at dim)
        if self.label_injection != "none":
            if self.label_dim == 1 and not self.soft_labels:
                lbl_emb = self.label_embed((ctx_labels > 0).long())    # (B, M, dim)
            elif self.label_dim == 1:
                lbl_emb = self.label_embed(ctx_labels.unsqueeze(-1).float())  # (B, M, dim)
            else:
                lbl_emb = self.label_embed(ctx_labels.float())         # (B, M, dim)
            if self.label_injection == "additive":
                ctx = ctx + lbl_emb
            elif self.label_injection == "concat":
                ctx = self.concat_proj(torch.cat([ctx, lbl_emb], dim=-1))
            elif self.label_injection == "gate":
                gate = torch.sigmoid(self.gate_proj(lbl_emb))
                ctx = ctx * gate

        # Prepend register tokens to context (cascade from prev level + own learnable)
        R_own    = self.num_registers
        R_cas    = cascade_registers.shape[1] if cascade_registers is not None else 0
        R_prefix = R_cas + R_own
        if R_prefix > 0:
            pieces = []
            if cascade_registers is not None:
                pieces.append(cascade_registers)
            if R_own > 0:
                pieces.append(self.register_tokens.expand(B, -1, -1))
            pieces.append(ctx)
            ctx = torch.cat(pieces, dim=1)  # (B, R_prefix + M, dim)
            if ctx_coords is not None:
                zero_c = torch.zeros(B, R_prefix, 3, dtype=ctx_coords.dtype, device=ctx_coords.device)
                ctx_coords = torch.cat([zero_c, ctx_coords], dim=1)
            M = M + R_prefix

        # 4. Positional encoding
        tgt, ctx = self._apply_pos(tgt, ctx)

        # Log-n query scaling
        q_scale = (
            math.log(max(M, 1)) / math.log(max(self.log_n_base, 2))
            if self.log_n_scaling else 1.0
        )

        # 5. Per-layer blocks
        for i in range(self.num_layers):
            # 5a. Context self-attention
            if self.ctx_self_attn_flag:
                ctx_n = self.ctx_sa_norms[i](ctx)
                ctx = ctx + self._mha(ctx_n, ctx_n, ctx_n,
                                      self.ctx_q_projs[i], self.ctx_k_projs[i],
                                      self.ctx_v_projs[i], self.ctx_out_projs[i],
                                      q_coords=ctx_coords, k_coords=ctx_coords)
                ctx = ctx + self.ctx_ffns[i](self.ctx_ffn_norms[i](ctx))

            # 5b. Cross-attention: target Q attends to context K/V
            ctx_kv = self.kv_norms[i](ctx)
            attn_out = self._mha(
                self.attn_norms[i](tgt), ctx_kv, ctx_kv,
                self.q_projs[i], self.k_projs[i], self.v_projs[i], self.out_projs[i],
                q_scale=q_scale, q_coords=tgt_coords, k_coords=ctx_coords,
                zero_attn=self.append_zero_attn,
            )
            tgt = tgt + attn_out
            tgt = tgt + self.ffns[i](self.ffn_norms[i](tgt))

        # Extract own register outputs before decoding (for cascading to the next level)
        reg_out = ctx[:, :R_own].clone() if R_own > 0 else None

        # 6. Output head
        if self.output_head_type in ("linear", "mlp"):
            out = torch.sigmoid(self.head(tgt))   # (B, N, output_dim)
            pred = out.squeeze(-1) if self.output_dim == 1 else out
            return (pred, reg_out) if R_own > 0 else pred

        # retrieval: uses real context patches only, not register prefix
        ctx_for_ret = ctx[:, R_prefix:] if R_prefix > 0 else ctx
        q = self.ret_q_proj(tgt)                                            # (B, N, dim)
        k = self.ret_k_proj(ctx_for_ret)                                    # (B, M_orig, dim)
        scores  = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim)          # (B, N, M_orig)
        weights = F.softmax(scores, dim=-1)                                 # (B, N, M_orig)
        if self.label_dim == 1:
            v = ctx_labels.unsqueeze(-1)                      # (B, M_orig, 1)
            pred = (weights @ v).squeeze(-1).clamp(0, 1)     # (B, N)
            return (pred, reg_out) if R_own > 0 else pred
        pred = (weights @ ctx_labels).clamp(0, 1)             # (B, N, label_dim)
        return (pred, reg_out) if R_own > 0 else pred

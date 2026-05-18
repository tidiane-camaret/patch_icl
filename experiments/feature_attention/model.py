"""
PatchICLAttention: learned in-context binary classifier for patch-level segmentation.

Each context patch (feature vector + binary label) is a training example.
Cross-attention lets target patches attend to context patches, optionally
conditioned on context labels, then a head decodes probabilities.

Architecture decisions (all configurable):
  label_injection : "additive" | "concat" | "gate" | "none"
  output_head     : "linear" | "mlp" | "retrieval"
  pos_encoding    : "none" | "sinusoidal" | "learned"
  input_norm      : "none" | "rmsnorm" | "l2"
  ctx_self_attn   : bool — context tokens self-attend before each cross-attn block
  log_n_scaling   : bool — scale cross-attn queries by log(n_ctx)/log(n_base)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchICLAttention(nn.Module):
    """
    Args
    ----
    embed_dim       : feature dimension C (must match encoder output)
    num_heads       : MHA heads (embed_dim must be divisible by num_heads)
    num_layers      : stacked cross-attention + FFN blocks
    ff_factor       : FFN hidden dim = embed_dim * ff_factor
    label_injection : how context binary labels enter the context token representation
                        additive  — token += label_embed(y)
                        concat    — token = proj([token; label_embed(y)])
                        gate      — token = token * sigmoid(proj(label_embed(y)))
                        none      — labels only appear in the output head
    output_head     : how final target embeddings decode to probabilities
                        linear    — Linear(C, 1) + sigmoid
                        mlp       — RMSNorm → Linear → GELU → Linear + sigmoid
                        retrieval — cross-attention(Q=tgt, K=ctx, V=ctx_labels)
    pos_encoding    : spatial position encoding for the D×H×W patch grid
                        none       — no positional information
                        sinusoidal — fixed 3D sinusoidal encoding
                        learned    — nn.Embedding(D*H*W, C)
    input_norm      : applied to raw features before all processing
                        none    — use encoder features as-is
                        rmsnorm — RMSNorm per token
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
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert label_injection in ("additive", "concat", "gate", "none")
        assert output_head in ("linear", "mlp", "retrieval")
        assert pos_encoding in ("none", "sinusoidal", "learned")
        assert input_norm in ("none", "rmsnorm", "l2")

        self.embed_dim          = embed_dim
        self.num_heads          = num_heads
        self.head_dim           = embed_dim // num_heads
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

        # ---- Input normalization ----------------------------------------
        self.input_norm = nn.RMSNorm(embed_dim) if input_norm == "rmsnorm" else None

        # ---- Label injection --------------------------------------------
        if label_injection in ("additive", "gate"):
            self.label_embed = nn.Embedding(2, embed_dim)
            if label_injection == "gate":
                self.gate_proj = nn.Linear(embed_dim, embed_dim)
        elif label_injection == "concat":
            self.label_embed = nn.Embedding(2, embed_dim)
            self.concat_proj = nn.Linear(2 * embed_dim, embed_dim)

        # ---- Positional encoding ----------------------------------------
        if pos_encoding == "learned":
            N = grid_size[0] * grid_size[1] * grid_size[2]
            self.pos_embed = nn.Embedding(N, embed_dim)

        ff_dim = embed_dim * ff_factor

        # ---- Cross-attention layers (target Q, context K/V) -------------
        # Using manual projections + F.scaled_dot_product_attention for full control
        # over K/V normalization and log-n query scaling. Q/K/V use bias=False;
        # out_proj and FFN retain bias (standard practice).
        self.q_projs    = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
        self.k_projs    = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
        self.v_projs    = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
        self.out_projs  = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.attn_norms = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_layers)])  # pre-norm for Q
        self.kv_norms   = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_layers)])  # pre-norm for K/V

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, embed_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_layers)])

        # ---- Context self-attention blocks (optional) -------------------
        # Each layer: context tokens self-attend (full SA + FFN block) before
        # the cross-attention, so context patches can interact across K images.
        if ctx_self_attn:
            self.ctx_q_projs   = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
            self.ctx_k_projs   = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
            self.ctx_v_projs   = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)])
            self.ctx_out_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
            self.ctx_sa_norms  = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_layers)])
            self.ctx_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, embed_dim),
                )
                for _ in range(num_layers)
            ])
            self.ctx_ffn_norms = nn.ModuleList([nn.RMSNorm(embed_dim) for _ in range(num_layers)])

        # ---- Output head ------------------------------------------------
        if output_head == "linear":
            self.head = nn.Linear(embed_dim, 1)
        elif output_head == "mlp":
            self.head = nn.Sequential(
                nn.RMSNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
        elif output_head == "retrieval":
            # Separate Q and K projections decouple the similarity space from
            # the representation space (mirrors TabPFN's ManyClassDecoder).
            self.ret_q_proj = nn.Linear(embed_dim, embed_dim)
            self.ret_k_proj = nn.Linear(embed_dim, embed_dim)

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
        q:        torch.Tensor,    # (B, Sq, C) — already pre-normed
        k:        torch.Tensor,    # (B, Sk, C) — already pre-normed
        v:        torch.Tensor,    # (B, Sk, C) — already pre-normed (same as k)
        q_proj:   nn.Linear,
        k_proj:   nn.Linear,
        v_proj:   nn.Linear,
        out_proj: nn.Linear,
        q_scale:  float = 1.0,
    ) -> torch.Tensor:
        """Project Q/K/V, run scaled dot-product attention, project output."""
        B, Sq, _ = q.shape
        Sk = k.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = q_proj(q).view(B, Sq, H, D).transpose(1, 2)   # (B, H, Sq, D)
        K = k_proj(k).view(B, Sk, H, D).transpose(1, 2)   # (B, H, Sk, D)
        V = v_proj(v).view(B, Sk, H, D).transpose(1, 2)   # (B, H, Sk, D)

        if q_scale != 1.0:
            Q = Q * q_scale

        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)   # (B, H, Sq, D)
        return out_proj(out.transpose(1, 2).reshape(B, Sq, self.embed_dim))

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

        dim_per_axis = self.embed_dim // 6
        if dim_per_axis == 0:
            return torch.zeros(N, self.embed_dim, device=device)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(dim_per_axis, device=device).float()
            / max(dim_per_axis - 1, 1)
        )
        parts = []
        for ax in range(3):
            args = coords[:, ax:ax + 1] * freqs.unsqueeze(0)   # (N, dim_per_axis)
            parts += [torch.sin(args), torch.cos(args)]
        pos = torch.cat(parts, dim=-1)                          # (N, 6*dim_per_axis)
        if pos.shape[-1] < self.embed_dim:
            pos = F.pad(pos, (0, self.embed_dim - pos.shape[-1]))
        return pos[:, :self.embed_dim]

    def _apply_pos(
        self, tgt: torch.Tensor, ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tgt: (B, N, C)  ctx: (B, M, C)  with M = K*N
        N, M = tgt.shape[1], ctx.shape[1]
        if self.pos_enc_type == "sinusoidal":
            tgt_pos = self._sinusoidal_pos(N, tgt.device)          # (N, C)
            ctx_pos = self._sinusoidal_pos(N, ctx.device).repeat(M // N, 1)[:M]
            return tgt + tgt_pos, ctx + ctx_pos
        elif self.pos_enc_type == "learned":
            idx = torch.arange(N, device=tgt.device)
            tgt_pos = self.pos_embed(idx)                           # (N, C)
            ctx_pos = tgt_pos.repeat(M // N, 1)[:M]
            return tgt + tgt_pos, ctx + ctx_pos
        return tgt, ctx

    # ------------------------------------------------------------------

    def forward(
        self,
        tgt_feat:   torch.Tensor,  # (B, N, C)
        ctx_feat:   torch.Tensor,  # (B, M, C)  M = K*N
        ctx_labels: torch.Tensor,  # (B, M) float binary labels
    ) -> torch.Tensor:             # (B, N) probabilities in [0, 1]
        M = ctx_feat.shape[1]

        # 1. Input normalization
        if self.input_norm_type == "l2":
            tgt = F.normalize(tgt_feat, dim=-1)
            ctx = F.normalize(ctx_feat, dim=-1)
        elif self.input_norm is not None:
            tgt = self.input_norm(tgt_feat)
            ctx = self.input_norm(ctx_feat)
        else:
            tgt, ctx = tgt_feat, ctx_feat

        # 2. Label injection into context tokens
        ctx_lab = (ctx_labels > 0).long()   # (B, M) int
        if self.label_injection == "additive":
            ctx = ctx + self.label_embed(ctx_lab)
        elif self.label_injection == "concat":
            ctx = self.concat_proj(torch.cat([ctx, self.label_embed(ctx_lab)], dim=-1))
        elif self.label_injection == "gate":
            gate = torch.sigmoid(self.gate_proj(self.label_embed(ctx_lab)))
            ctx = ctx * gate

        # 3. Positional encoding
        tgt, ctx = self._apply_pos(tgt, ctx)

        # Log-n query scaling: F.scaled_dot_product_attention uses 1/sqrt(D) internally;
        # we additionally pre-scale Q by log(M)/log(n_base) so softmax doesn't flatten
        # uniformly as the context sequence grows (matches TabPFN's SoftmaxScalingMLP idea).
        q_scale = (
            math.log(max(M, 1)) / math.log(max(self.log_n_base, 2))
            if self.log_n_scaling else 1.0
        )

        # 4. Per-layer blocks
        for i in range(self.num_layers):
            # 4a. Context self-attention: let context patches interact before retrieval.
            #     Each context image's patches can now attend to patches from other
            #     context images (and themselves), building a richer representation.
            if self.ctx_self_attn_flag:
                ctx_n = self.ctx_sa_norms[i](ctx)
                ctx = ctx + self._mha(ctx_n, ctx_n, ctx_n,
                                      self.ctx_q_projs[i], self.ctx_k_projs[i],
                                      self.ctx_v_projs[i], self.ctx_out_projs[i])
                ctx = ctx + self.ctx_ffns[i](self.ctx_ffn_norms[i](ctx))

            # 4b. Cross-attention: target Q attends to context K/V.
            #     Both target (Q) and context (K/V) are independently pre-normed,
            #     fixing the pre-norm invariance broken by the previous single-norm design.
            ctx_kv = self.kv_norms[i](ctx)
            attn_out = self._mha(
                self.attn_norms[i](tgt), ctx_kv, ctx_kv,
                self.q_projs[i], self.k_projs[i], self.v_projs[i], self.out_projs[i],
                q_scale=q_scale,
            )
            tgt = tgt + attn_out
            tgt = tgt + self.ffns[i](self.ffn_norms[i](tgt))

        # 5. Output head
        if self.output_head_type in ("linear", "mlp"):
            return torch.sigmoid(self.head(tgt).squeeze(-1))   # (B, N)

        # retrieval: separate Q/K projections into similarity space (mirrors
        # TabPFN's ManyClassDecoder), V = scalar binary labels
        q = self.ret_q_proj(tgt)                                      # (B, N, C)
        k = self.ret_k_proj(ctx)                                      # (B, M, C)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim)  # (B, N, M)
        weights = F.softmax(scores, dim=-1)                           # (B, N, M)
        v = ctx_labels.unsqueeze(-1)                                  # (B, M, 1)
        return (weights @ v).squeeze(-1).clamp(0, 1)                  # (B, N)

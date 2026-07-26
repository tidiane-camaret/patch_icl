"""Compute-only, architecturally-faithful encoder stand-ins (no pretrained weights).

Primus: high-res-token pure-ViT (arxiv 2503.01835). SegMamba: CNN-stem + SSM blocks
(arxiv 2401.13560). Faithful block structure/dims for FLOPs/latency/VRAM; NOT for Dice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
        self.heads = heads
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, C // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (B,heads,N,d)
        a = F.scaled_dot_product_attention(q, k, v)               # -> FlashAttention-2
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
        return x + self.mlp(self.n2(x))


class PrimusStandin(nn.Module):
    def __init__(self, in_ch=1, img_size=64, patch=8, embed_dim=384, depth=12,
                 heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv3d(in_ch, embed_dim, patch, stride=patch)
        g = img_size // patch
        self.pos = nn.Parameter(torch.zeros(1, embed_dim, g, g, g))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(embed_dim, heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)                                          # (B,C,g,g,g)
        pos = F.interpolate(self.pos, size=x.shape[-3:], mode="trilinear",
                            align_corners=False)
        x = (x + pos).flatten(2).transpose(1, 2)                   # (B,N,C)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def _ref_scan(u, delta, A, B, C):
    """Pure-PyTorch fallback selective scan. u,delta:(b,d,l) A:(d,n) B,C:(b,n,l)."""
    b, d, l = u.shape
    n = A.shape[1]
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))  # (b,d,l,n)
    dB = delta.unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)  # (b,d,l,n)
    x = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(l):
        x = dA[:, :, t] * x + dB[:, :, t] * u[:, :, t].unsqueeze(-1)   # (b,d,n)
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, t]))          # (b,d)
    return torch.stack(ys, dim=-1)                                    # (b,d,l)


class _SSM3D(nn.Module):
    """Minimal selective-SSM over a flattened 3D volume (single scan orientation)."""
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim, self.n = dim, d_state
        self.in_proj = nn.Linear(dim, dim)
        self.dt = nn.Linear(dim, dim)
        self.A = nn.Parameter(-torch.rand(dim, d_state))
        self.B = nn.Linear(dim, d_state); self.C = nn.Linear(dim, d_state)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):                                      # x: (B,dim,D,H,W)
        B_, dim, D, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)                    # (B,L,dim)
        u = F.silu(self.in_proj(seq))
        delta = F.softplus(self.dt(seq)).transpose(1, 2)      # (B,dim,L)
        Bm, Cm = self.B(seq).transpose(1, 2), self.C(seq).transpose(1, 2)  # (B,n,L)
        y = _selective_scan(u.transpose(1, 2), delta, self.A, Bm, Cm)      # (B,dim,L)
        y = self.out(y.transpose(1, 2))                        # (B,L,dim)
        return y.transpose(1, 2).reshape(B_, dim, D, H, W)


def _selective_scan(u, delta, A, B, C):
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except Exception:
        return _ref_scan(u, delta, A, B, C)          # kernel absent: expected fallback
    try:
        return selective_scan_fn(u.contiguous(), delta.contiguous(),
                                 A.contiguous(), B.contiguous(), C.contiguous(),
                                 None, None, None, False)
    except Exception as e:                            # kernel present but failed: surface it
        import warnings
        warnings.warn(f"mamba_ssm selective_scan_fn failed ({e}); using slow reference scan")
        return _ref_scan(u, delta, A, B, C)


class SegMambaStandin(nn.Module):
    def __init__(self, in_ch=1, dims=(32, 64, 128, 256), d_state=16):
        super().__init__()
        def cbr(ci, co, s):
            return nn.Sequential(nn.Conv3d(ci, co, 3, stride=s, padding=1),
                                 nn.InstanceNorm3d(co), nn.SiLU())
        self.stem = cbr(in_ch, dims[0], 1)
        self.stages = nn.ModuleList()
        self.ssms = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.stages.append(cbr(dims[i], dims[i + 1], 2))
            self.ssms.append(_SSM3D(dims[i + 1], d_state))

    def forward(self, x):
        x = self.stem(x)
        for stage, ssm in zip(self.stages, self.ssms):
            x = stage(x); x = x + ssm(x)
        return x

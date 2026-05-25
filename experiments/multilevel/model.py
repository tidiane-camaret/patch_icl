"""
MultilevelICL: coarse-to-fine in-context segmentation.

Wraps one PatchICLAttention per spatial level.  Training logic (sampling,
feature extraction, loss) lives in train.py; this module only holds parameters.

Positional encoding
-------------------
All levels use pos_encoding="rope3d".  RoPE is applied inside Q/K projections
via explicit integer (d, h, w) coordinates passed at call time, so both dense
levels (full 8³ grid) and sparse levels (NP sampled patches from 16³) share the
same PE mechanism without any grid-size assumption.

The rope cache is keyed by (max_pos, dim) and uses the same theta frequencies
regardless of which resolution the tokens come from — coordinates just index into
the cache at their actual voxel position.  rope_max_pos is set to the maximum
spatial extent across all levels so that every valid coordinate is in range.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from experiments.feature_attention.model import PatchICLAttention


class MaskCNN(nn.Module):
    """3D ConvNet encoding a soft/binary mask grid into per-voxel embeddings.

    Same-padding convolutions make it grid-size agnostic — shared across all
    spatial levels in MultilevelICL.
    """

    def __init__(self, out_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, out_dim, 1),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """mask: (B, 1, D, H, W) float → (B, D*H*W, out_dim)"""
        out = self.net(mask.float())           # (B, out_dim, D, H, W)
        return out.flatten(2).transpose(1, 2)  # (B, N, out_dim)


class MultilevelICL(nn.Module):
    """Container for per-level (or shared) PatchICLAttention instances.

    Args
    ----
    embed_dim      : raw encoder feature dimension (shared across levels)
    level_cfgs     : list of dicts — kwargs forwarded to PatchICLAttention for each level,
                     plus a required 'grid_size' key for the spatial resolution.
    shared_weights : if True, a single PatchICLAttention is used for all levels.
                     Requires pos_encoding="rope3d" (learned PE has grid-size-dependent
                     embedding tables and cannot be shared across resolutions).
    """

    def __init__(self, embed_dim: int, level_cfgs: list[dict], mask_cnn_dim: int = 0,
                 num_registers: int = 0, append_zero_attn: bool = False,
                 shared_weights: bool = False, use_scale_embed: bool = False,
                 use_role_embed: bool = False, max_context_size: int = 8):
        super().__init__()

        self._num_levels   = len(level_cfgs)
        self.shared_weights = shared_weights

        # Shared max_pos: covers the largest coordinate across all levels so the
        # same theta frequencies are valid everywhere.
        global_max_pos = max(max(cfg["grid_size"]) for cfg in level_cfgs)

        # Optional shared mask CNN (None = scalar label injection).
        self.mask_cnn     = MaskCNN(out_dim=mask_cnn_dim) if mask_cnn_dim > 0 else None
        self.mask_cnn_dim = mask_cnn_dim
        label_dim         = mask_cnn_dim if mask_cnn_dim > 0 else 1

        def _build(cfg: dict) -> PatchICLAttention:
            return PatchICLAttention(
                embed_dim=embed_dim,
                grid_size=tuple(cfg["grid_size"]),
                dim=cfg["dim"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                ff_factor=cfg.get("ff_factor", 2),
                label_injection=cfg.get("label_injection", "additive"),
                output_head=cfg.get("output_head", "linear"),
                pos_encoding=cfg.get("pos_encoding", "rope3d"),
                input_norm=cfg.get("input_norm", "rmsnorm"),
                dropout=cfg.get("dropout", 0.0),
                ctx_self_attn=cfg.get("ctx_self_attn", True),
                log_n_scaling=cfg.get("log_n_scaling", True),
                log_n_base=cfg.get("log_n_base", 512),
                soft_labels=cfg.get("soft_labels", True),
                label_dim=label_dim,
                output_dim=1,  # head always predicts binary mask regardless of label_dim
                num_registers=num_registers,
                append_zero_attn=append_zero_attn,
                rope_max_pos=global_max_pos,
                use_scale_embed=use_scale_embed,
                use_role_embed=use_role_embed,
                max_context_size=max_context_size,
            )

        if shared_weights:
            assert level_cfgs[0].get("pos_encoding", "rope3d") != "learned", \
                "shared_weights requires rope3d pos_encoding — learned PE has grid-size-dependent tables"
            # Single module used for every level; levels ModuleList is empty (no extra params).
            self.shared_level = _build(level_cfgs[0])
            self.levels       = nn.ModuleList()
        else:
            self.shared_level = None
            self.levels       = nn.ModuleList([_build(cfg) for cfg in level_cfgs])

    def __len__(self) -> int:
        return self._num_levels

    def __getitem__(self, idx: int) -> PatchICLAttention:
        return self.shared_level if self.shared_weights else self.levels[idx]

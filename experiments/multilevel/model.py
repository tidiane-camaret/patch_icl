"""
MultilevelICL: coarse-to-fine in-context segmentation.

Wraps one PatchICLAttention per spatial level.  Training logic (sampling,
feature extraction, loss) lives in train.py; this module only holds parameters.
"""

from __future__ import annotations

import torch.nn as nn

from experiments.feature_attention.model import PatchICLAttention


class MultilevelICL(nn.Module):
    """Container for per-level PatchICLAttention instances.

    Args
    ----
    embed_dim   : raw encoder feature dimension (shared across levels)
    level_cfgs  : list of dicts — kwargs forwarded to PatchICLAttention for each level,
                  plus a required 'grid_size' key for the spatial resolution at that level.

    Sparse levels (index > 0) receive only NP sampled tokens rather than the full
    D×H×W grid, so the grid-based sinusoidal PE in PatchICLAttention can't be used.
    Instead, MultilevelICL holds one learned coord_proj per sparse level: a bias-free
    Linear(3, embed_dim) that maps normalised (d, h, w) coordinates of the sampled
    patches into embed_dim space.  train.py gathers coords at sampled positions and
    adds the projected PE to the features before calling forward on the sparse level.
    """

    def __init__(self, embed_dim: int, level_cfgs: list[dict]):
        super().__init__()
        self.levels = nn.ModuleList([
            PatchICLAttention(
                embed_dim=embed_dim,
                grid_size=tuple(cfg["grid_size"]),
                dim=cfg["dim"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                ff_factor=cfg.get("ff_factor", 2),
                label_injection=cfg.get("label_injection", "additive"),
                output_head=cfg.get("output_head", "linear"),
                # Sparse levels use "none" — PE is injected externally via coord_projs.
                pos_encoding=cfg.get("pos_encoding", "none"),
                input_norm=cfg.get("input_norm", "rmsnorm"),
                dropout=cfg.get("dropout", 0.0),
                ctx_self_attn=cfg.get("ctx_self_attn", True),
                log_n_scaling=cfg.get("log_n_scaling", True),
                log_n_base=cfg.get("log_n_base", 512),
                soft_labels=cfg.get("soft_labels", True),
            )
            for cfg in level_cfgs
        ])
        # Learned (d, h, w) → embed_dim projection for sparse levels (index ≥ 1).
        # Level 0 is always dense and uses PatchICLAttention's own pos_encoding.
        self.coord_projs = nn.ModuleList([
            nn.Linear(3, embed_dim, bias=False)
            for _ in level_cfgs[1:]
        ])

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, idx: int) -> PatchICLAttention:
        return self.levels[idx]

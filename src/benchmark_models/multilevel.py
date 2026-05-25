"""
Adapter for MultilevelICL (experiments/multilevel/).

The model predicts at a coarse final resolution (default 32³) via a cascade
of PatchICLAttention layers over frozen STUNetEncoder features.  The adapter
upsamples the final grid prediction back to the input spatial size for fair
Dice comparison against other 128³ models.

Checkpoint format (saved by experiments/multilevel/train.py):
    {"epoch": int, "model": state_dict, "config": dict, "val_dice": float}
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.benchmark_models.base import InContextModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import shared utilities from the experiment's train module to avoid duplication
from experiments.multilevel.train import (  # noqa: E402
    encode_image_only,
    extract_features,
    downsample_mask,
    gather_patches,
    grid_coords_3d,
    sample_target_patches,
    sample_context_patches,
    _encode_ctx_labels,
)
from experiments.multilevel.model import MultilevelICL
from src.models.encoders.stunet import STUNetEncoder


class MultilevelICLAdapter(InContextModel):
    """
    Wraps MultilevelICL for the quality benchmark.

    Inference mirrors experiments/multilevel/train.py:validate() but handles
    a batched input tensor and upsamples the final prediction to the target
    volume's spatial size for fair Dice comparison.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device = None,
        stunet_pretrained: str = None,  # override the path stored in the checkpoint
        sampling_temperature: float = None,  # override; None = use checkpoint config
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg  = OmegaConf.create(ckpt["config"])

        self.cfg  = cfg
        self.temp = sampling_temperature if sampling_temperature is not None \
                    else float(cfg.data.sampling_temperature)
        self.NP   = int(cfg.data.n_patches_l1)
        self.resolutions = [tuple(r) for r in cfg.data.resolutions]
        self.level       = cfg.model.feature_level
        self.mask_pool   = cfg.data.mask_pool
        self.soft_labels_eval = bool(cfg.model.soft_labels_eval)
        self.target_sampling  = cfg.data.target_sampling

        # --- Encoder (frozen) ---
        pretrained = stunet_pretrained or cfg.model.stunet_pretrained
        self.encoder = STUNetEncoder(
            in_channels=1,
            variant=cfg.model.stunet_variant,
            pretrained=pretrained,
            freeze_encoder=True,
        ).to(self.device).eval()
        self._num_encoder_levels = len(self.encoder.skip_channels) + 1

        # --- Determine embed_dim via dummy forward ---
        with torch.inference_mode():
            dummy     = torch.zeros(1, 1, *cfg.data.image_size, device=self.device)
            feats     = encode_image_only(self.encoder, dummy)
            dummy_ds  = extract_features(feats, self.level, self.resolutions[0],
                                         self._num_encoder_levels)
        embed_dim = dummy_ds.shape[1]

        # --- MultilevelICL ---
        level_cfgs = [
            {
                "grid_size":       res,
                "dim":             cfg.model.dim,
                "num_heads":       cfg.model.num_heads,
                "num_layers":      cfg.model.num_layers,
                "ff_factor":       cfg.model.ff_factor,
                "label_injection": cfg.model.label_injection,
                "output_head":     cfg.model.output_head,
                "pos_encoding":    cfg.model.pos_encoding
                                   if (i == 0 or cfg.model.pos_encoding == "rope3d")
                                   else "none",
                "input_norm":      cfg.model.input_norm,
                "dropout":         cfg.model.dropout,
                "ctx_self_attn":   cfg.model.ctx_self_attn,
                "log_n_scaling":   cfg.model.log_n_scaling,
                "log_n_base":      cfg.model.log_n_base,
                "soft_labels":     cfg.model.soft_labels_train,
            }
            for i, res in enumerate(self.resolutions)
        ]
        mask_cnn_dim    = int(getattr(cfg.model, "mask_cnn_dim",    0)     or 0)
        num_registers   = int(getattr(cfg.model, "num_registers",   0)     or 0)
        append_zero_attn = bool(getattr(cfg.model, "append_zero_attn", False))
        shared_weights  = bool(getattr(cfg.model, "shared_weights",  False))
        use_scale_embed = bool(getattr(cfg.model, "use_scale_embed", False))
        use_role_embed  = bool(getattr(cfg.model, "use_role_embed",  False))
        max_context_size = int(getattr(cfg.model, "max_context_size", 8))

        self.num_registers  = num_registers
        self.mask_pool      = cfg.data.mask_pool

        self.model = MultilevelICL(
            embed_dim=embed_dim,
            level_cfgs=level_cfgs,
            mask_cnn_dim=mask_cnn_dim,
            num_registers=num_registers,
            append_zero_attn=append_zero_attn,
            shared_weights=shared_weights,
            use_scale_embed=use_scale_embed,
            use_role_embed=use_role_embed,
            max_context_size=max_context_size,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks):
        """
        Args:
            target_img    : (B, 1, D, H, W)
            context_imgs  : (B, K, 1, D, H, W)
            context_masks : (B, K, D, H, W) binary int64

        Returns:
            (B, D, H, W) binary int64 — upsampled to input spatial size
        """
        B, K = context_imgs.shape[:2]
        D, H, W = target_img.shape[2:]
        amp = self.device.type == "cuda"

        target_img    = target_img.to(self.device)
        context_imgs  = context_imgs.to(self.device)
        context_masks = context_masks.to(self.device).float()

        # Encode target and all context images
        tgt_feats = encode_image_only(self.encoder, target_img)
        ctx_flat  = context_imgs.reshape(B * K, 1, *context_imgs.shape[3:])
        ctx_feats = encode_image_only(self.encoder, ctx_flat)

        # Encode context masks once at finest resolution (MaskCNN path)
        mask_cnn_vol = None
        if self.model.mask_cnn is not None:
            finest_res = self.resolutions[-1]
            mask_in = downsample_mask(
                context_masks.reshape(B * K, *context_masks.shape[2:]),
                finest_res, self.mask_pool,
            ).unsqueeze(1)
            if not self.soft_labels_eval:
                mask_in = (mask_in > 0).float()
            with torch.autocast(device_type=self.device.type, enabled=amp):
                emb = self.model.mask_cnn(mask_in)
            mask_cnn_vol = emb.transpose(1, 2).reshape(
                B * K, self.model.mask_cnn_dim, *finest_res
            )

        grid_preds   = []
        cascade_regs = None
        C = None

        for i, res in enumerate(self.resolutions):
            N = res[0] * res[1] * res[2]

            tgt_feat_i  = extract_features(tgt_feats, self.level, res, self._num_encoder_levels)
            ctx_feat_if = extract_features(ctx_feats, self.level, res, self._num_encoder_levels)
            if C is None:
                C = tgt_feat_i.shape[1]
            ctx_feat_i = ctx_feat_if.reshape(B, K, C, *res)
            ctx_mask_i = downsample_mask(
                context_masks.reshape(B * K, *context_masks.shape[2:]),
                res, self.mask_pool,
            ).reshape(B, K, *res)

            coords = grid_coords_3d(res, self.device)

            if i == 0:
                tgt_f   = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
                ctx_f   = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K * N, C)
                ctx_lbl = _encode_ctx_labels(ctx_mask_i, None, self.soft_labels_eval,
                                             mask_cnn_vol, res)

                tgt_crds = coords.unsqueeze(0).expand(B, -1, -1)
                ctx_crds = coords.unsqueeze(0).expand(B, -1, -1).repeat(1, K, 1)

                with torch.autocast(device_type=self.device.type, enabled=amp):
                    result = self.model[0](tgt_f, ctx_f, ctx_lbl,
                                           tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                           cascade_registers=cascade_regs)
                if isinstance(result, tuple):
                    cascade_regs = result[1]
                    pred = result[0].float()
                else:
                    pred = result.float()
                grid_pred = pred

            else:
                prev_res = self.resolutions[i - 1]
                prev_up  = F.interpolate(
                    grid_preds[-1].detach().reshape(B, 1, *prev_res),
                    size=res, mode="trilinear", align_corners=False,
                ).reshape(B, N)

                tgt_idx = sample_target_patches(
                    prev_up, prev_up, self.NP, self.temp, "predicted_entropy"
                )
                ctx_idx = sample_context_patches(ctx_mask_i, self.NP, self.temp)

                tgt_flat_f = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
                ctx_flat_f = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K, N, C)

                tgt_sparse = gather_patches(tgt_flat_f, tgt_idx)
                ctx_pieces = [gather_patches(ctx_flat_f[:, k], ctx_idx) for k in range(K)]
                ctx_sparse = torch.cat(ctx_pieces, dim=1)
                ctx_lbl    = _encode_ctx_labels(ctx_mask_i, ctx_idx, self.soft_labels_eval,
                                                mask_cnn_vol, res)

                tgt_crds = coords[tgt_idx.reshape(-1)].reshape(B, self.NP, 3)
                ctx_crds = coords[ctx_idx.reshape(-1)].reshape(B, self.NP, 3)
                ctx_crds = ctx_crds.unsqueeze(1).expand(-1, K, -1, -1).reshape(B, K * self.NP, 3)

                with torch.autocast(device_type=self.device.type, enabled=amp):
                    result = self.model[i](tgt_sparse, ctx_sparse, ctx_lbl,
                                           tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                           cascade_registers=cascade_regs)
                if isinstance(result, tuple):
                    cascade_regs = result[1]
                    pred = result[0].float()
                else:
                    pred = result.float()
                grid_pred = prev_up.clone()
                grid_pred.scatter_(1, tgt_idx, pred)

            grid_preds.append(grid_pred)

        # Upsample final grid prediction to input spatial size
        final_res = self.resolutions[-1]
        pred_vol  = grid_preds[-1].reshape(B, 1, *final_res)
        pred_full = F.interpolate(pred_vol, size=(D, H, W),
                                  mode="trilinear", align_corners=False).squeeze(1)

        return (pred_full > 0.5).long()

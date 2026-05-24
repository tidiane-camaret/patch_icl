"""
Adapters for the two native in-context models: ViTInContext3D and ResEncInContext3D.
Both are instantiated with default hyperparameters from configs/config.yaml.
Pass a checkpoint path to load trained weights.
"""

import torch

from src.benchmark_models.base import InContextModel
from src.vit_in_context import ViTInContext3D
from src.models.resenc_in_context import ResEncInContext3D


def _load_state_dict(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    sd = torch.load(ckpt_path, map_location=device)
    # Strip torch.compile prefix if present
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)


class NativeViT(InContextModel):
    """ViTInContext3D with default config.yaml hyperparameters."""

    def __init__(
        self,
        ckpt_path: str,
        image_size: tuple = (64, 64, 64),
        device: torch.device = None,
        # Transformer
        patch_size: tuple = (8, 8, 8),
        embed_dim: int = 256,
        depth_stage1: int = 6,
        depth_stage2: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTInContext3D(
            image_size=image_size,
            in_channels=1,
            num_classes=2,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth_stage1=depth_stage1,
            depth_stage2=depth_stage2,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        ).to(self.device)
        _load_state_dict(self.model, ckpt_path, self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks):
        target_img   = target_img.to(self.device)
        context_imgs = context_imgs.to(self.device)
        context_masks = context_masks.to(self.device)
        logits = self.model(target_img, context_imgs, context_masks)
        return (logits.argmax(1) == 1).long()


class NativeResEnc(InContextModel):
    """ResEncInContext3D with default config.yaml hyperparameters."""

    def __init__(
        self,
        ckpt_path: str,
        image_size: tuple = (64, 64, 64),
        device: torch.device = None,
        # Encoder
        encoder_name: str = "resenc",
        features_per_stage: tuple = (32, 64, 128, 256),
        # STUNet (ignored when encoder_name="resenc")
        stunet_variant: str = "base",
        stunet_pretrained: str = None,
        stunet_freeze: bool = False,
        stunet_num_stages: int = None,
        mask_fusion: str = "additive",
        # Transformer
        rope_theta: float = 100.0,
        depth_stage1: int = 6,
        depth_stage2: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_registers: int = 0,
        num_context_layers: int = 0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResEncInContext3D(
            image_size=image_size,
            in_channels=1,
            num_classes=2,
            encoder_name=encoder_name,
            features_per_stage=features_per_stage,
            stunet_variant=stunet_variant,
            stunet_pretrained=stunet_pretrained,
            stunet_freeze=stunet_freeze,
            stunet_num_stages=stunet_num_stages,
            mask_fusion=mask_fusion,
            rope_theta=rope_theta,
            depth_stage1=depth_stage1,
            depth_stage2=depth_stage2,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_registers=num_registers,
            num_context_layers=num_context_layers,
        ).to(self.device)
        _load_state_dict(self.model, ckpt_path, self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks):
        target_img    = target_img.to(self.device)
        context_imgs  = context_imgs.to(self.device)
        context_masks = context_masks.to(self.device)
        logits = self.model(target_img, context_imgs, context_masks)
        return (logits.argmax(1) == 1).long()

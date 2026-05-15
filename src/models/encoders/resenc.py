"""nnUNet ResidualEncoder wrapper for in-context segmentation.

Encodes [image, mask] concatenated as a 2-channel input.
"""

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class ResEncEncoder(nn.Module):
    """4-stage nnUNet residual encoder (nnUNetResEncM config).

    Input:  image [B, 1, D, H, W] + mask [B, 1, D, H, W]  (mask = 0 for target)
    Output: [s0, s1, s2, s3] — s3 is the bottleneck (8× spatial downsample).

    skip_channels : [f0, f1, f2]  (high-res → low-res, excluding bottleneck)
    bot_features  : f3
    total_stride  : 8
    """

    total_stride: int = 8

    def __init__(
        self,
        in_channels: int = 1,
        features_per_stage: tuple[int, ...] = (32, 64, 128, 256),
        num_classes: int = 2,
    ):
        super().__init__()
        assert len(features_per_stage) == 4
        _unet = ResidualEncoderUNet(
            input_channels=in_channels + 1,
            n_stages=4,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=(1, 2, 2, 2),
            n_blocks_per_stage=(1, 3, 4, 6),
            num_classes=num_classes,
            n_conv_per_stage_decoder=(1, 1, 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
        )
        self._enc = _unet.encoder
        self.skip_channels = list(features_per_stage[:-1])
        self.bot_features = features_per_stage[-1]

    def forward(self, imgs: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
        """Returns [s0, s1, s2, s3]; s3 is bottleneck."""
        return self._enc(torch.cat([imgs, masks], dim=1))

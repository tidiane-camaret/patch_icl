"""
Pretrained image feature encoders for ImagePFN's image path.

Currently provides the UniverSeg encoder (frozen), mirroring the `feature_sim`
eval backend (experiments/2d/eval.py: encode_images + extract_features_batch).
The encoder is injected into ImagePFN rather than imported by it, so
pfn_seg_2d.py stays torch-only and free of the `src`-package shadowing that
common.py introduces.

UniverSeg lives at a fixed checkout path (its own top-level `universeg` package,
so importing it does not collide with either `src` namespace).
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_UNIVERSEG_PATH = "/home/dpxuser/repos/UniverSeg"


class UniverSegFeatureEncoder(nn.Module):
    """
    Frozen UniverSeg encoder → pooled feature grid per image.

    forward(images, out_size): (N, 1, H, W) → (N, feature_dim, out_size, out_size).

    Replicates encode_images (run enc_blocks on a dummy support, collect the
    target feature map at each scale) and extract_features_batch (adaptive-avg-pool
    each selected level to out_size and concat on the channel dim).

    Args:
        level: encoder stage 0..3 (0 = highest res), -1 = bottleneck, or "all"
            to concatenate all four levels (feature_dim = 4 × 64 = 256).
        input_size: resolution to resize inputs to before encoding (UniverSeg is
            trained at 128). Only applied when resize_to_input is True.
        resize_to_input: if True, bilinear-resize inputs to input_size² before
            encoding; if False (default), encode at the image's native resolution
            (UniverSeg is fully convolutional, so it runs at other sizes too).
    """

    def __init__(self, level="all", input_size: int = 128, resize_to_input: bool = False):
        super().__init__()
        if _UNIVERSEG_PATH not in sys.path:
            sys.path.append(_UNIVERSEG_PATH)
        from universeg import universeg

        self.model = universeg(pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.level = level
        self.input_size = input_size
        self.resize_to_input = resize_to_input
        self.feature_dim = 4 * 64 if str(level) == "all" else 64

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> list[torch.Tensor]:
        B, _, H, W = images.shape
        target  = images.unsqueeze(1)                                   # (B, 1, 1, H, W)
        dummy_s = torch.zeros(B, 1, 2, H, W, device=images.device, dtype=images.dtype)
        feats = []
        for i, block in enumerate(self.model.enc_blocks):
            target, dummy_s = block(target, dummy_s)
            feats.append(target[:, 0])                                  # (B, 64, H', W')
            if i < len(self.model.enc_blocks) - 1:
                target  = F.max_pool2d(target[:, 0], 2).unsqueeze(1)
                dummy_s = F.max_pool2d(dummy_s[:, 0], 2).unsqueeze(1)
        return feats

    # Run eager: the encoder is frozen + no_grad (nothing to compile), and under
    # torch.compile(dynamic=True) its adaptive_avg_pool2d gets symbolic window sizes
    # that inductor cannot lower. Dynamo graph-breaks here; the transformer still
    # compiles. The decorator is a no-op when the model isn't compiled.
    @torch.compiler.disable
    @torch.no_grad()
    def forward(self, images: torch.Tensor, out_size: int) -> torch.Tensor:
        if self.resize_to_input and (images.shape[-1] != self.input_size
                                     or images.shape[-2] != self.input_size):
            images = F.interpolate(images, size=(self.input_size, self.input_size),
                                   mode="bilinear", align_corners=False)
        feats = self._encode(images)
        size = (out_size, out_size)
        if str(self.level) == "all":
            maps = [F.adaptive_avg_pool2d(f.float(), size) for f in feats]
        else:
            idx  = int(self.level) % len(feats)
            maps = [F.adaptive_avg_pool2d(feats[idx].float(), size)]
        return torch.cat(maps, dim=1)                                  # (B, feature_dim, out_size, out_size)

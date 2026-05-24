from abc import ABC, abstractmethod
import torch


class InContextModel(ABC):
    """
    Protocol for in-context segmentation models in the benchmark.

    All models receive K reference (image, mask) pairs at inference time and
    predict a binary mask for the target image — no gradient, no fine-tuning.
    """

    @abstractmethod
    def predict(
        self,
        target_img: torch.Tensor,     # (B, 1, D, H, W) float32, z-scored
        context_imgs: torch.Tensor,   # (B, K, 1, D, H, W) float32, z-scored
        context_masks: torch.Tensor,  # (B, K, D, H, W) int64 binary
    ) -> torch.Tensor:                # (B, D, H, W) int64 binary
        ...

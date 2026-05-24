"""
Registry of in-context segmentation models for the benchmark.

Usage:
    from src.benchmark_models import load_model
    model = load_model("medverse")
    model = load_model("multilevel", ckpt_path="results/multilevel/.../best.pt")
    model = load_model("native_resenc", ckpt_path="results/checkpoints/resenc_in_context_best.pt")
    model = load_model("native_vit",    ckpt_path="results/checkpoints/vit_in_context_best.pt")
"""

import torch

from src.benchmark_models.base import InContextModel  # noqa: F401

_ALL_MODELS = ["native_vit", "native_resenc", "medverse", "multilevel"]


def load_model(
    name: str,
    ckpt_path: str = None,
    image_size: tuple = (128, 128, 128),
    device: torch.device = None,
    **kwargs,
) -> "InContextModel":
    """
    Instantiate a benchmark model by name.

    Args:
        name       : one of "native_vit", "native_resenc", "medverse", "multilevel"
        ckpt_path  : checkpoint path (required for all except medverse)
        image_size : spatial size used at training time (ignored by multilevel)
        device     : inference device (auto-detected if None)
        **kwargs   : forwarded to the model constructor
    """
    if name == "native_vit":
        from src.benchmark_models.native import NativeViT
        if ckpt_path is None:
            raise ValueError("ckpt_path is required for native_vit")
        return NativeViT(ckpt_path=ckpt_path, image_size=image_size, device=device, **kwargs)

    if name == "native_resenc":
        from src.benchmark_models.native import NativeResEnc
        if ckpt_path is None:
            raise ValueError("ckpt_path is required for native_resenc")
        return NativeResEnc(ckpt_path=ckpt_path, image_size=image_size, device=device, **kwargs)

    if name == "medverse":
        from src.benchmark_models.medverse import MedverseModel
        return MedverseModel(device=device, **kwargs)

    if name == "multilevel":
        from src.benchmark_models.multilevel import MultilevelICLAdapter
        if ckpt_path is None:
            raise ValueError("ckpt_path is required for multilevel")
        return MultilevelICLAdapter(ckpt_path=ckpt_path, device=device, **kwargs)

    raise ValueError(
        f"Unknown model {name!r}. Choose from: {', '.join(_ALL_MODELS)}"
    )

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

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_UNIVERSEG_PATH = "/home/dpxuser/repos/UniverSeg"

_DINOV3_CACHE = (
    "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
    "ANALYSIS_20251122/checkpoints"
)
# ImageNet stats DINOv3 was normalized with (preprocessor_config.json)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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


class DINOv3FeatureEncoder(nn.Module):
    """
    Frozen DINOv3 ConvNeXt backbone → pooled feature grid per image.

    Drop-in replacement for UniverSegFeatureEncoder with the same interface:
    forward(images, out_size): (N, 1, H, W) → (N, feature_dim, out_size, out_size).

    Model is `facebook/dinov3-convnext-{variant}-pretrain-lvd1689m`, a ConvNeXt CNN
    (DINOv3 self-supervised pretraining on LVD-1689M). Fully convolutional, so it
    runs at the image's native resolution and is pooled to out_size — matching how
    UniverSegFeatureEncoder is used in ImagePFN / the multilevel pipeline.

    Stage feature maps (channels, stride): [128 @/4, 256 @/8, 512 @/16, 1024 @/32]
    for the base variant. `level` selects a stage (0 = highest res .. 3 = deepest),
    or "all" to concat all four (feature_dim = 128+256+512+1024 = 1920).

    DINOv3 expects 3-channel ImageNet-normalized RGB; medical inputs here are
    1-channel grayscale, so the channel is repeated ×3 and (optionally) ImageNet-
    normalized. Set imagenet_norm=False to skip it when ImagePFN's own per-context
    standardization is preferred as the sole normalization.

    Cheap channel-dim reduction (applied AFTER pooling, so feature_dim shrinks before
    the downstream image_embed). `reduce`:
        "none"          no reduction (feature_dim = raw)
        "grouppool:<d>" adaptive-avg-pool the channels to d (zero params, no fit)
        "random:<d>"    fixed Gaussian (Johnson–Lindenstrauss) projection (no fit)
        "pca:<d>"       PCA projection fit once on data; call ensure_pca()/fit_pca()
                        before use (random/none/grouppool need no fitting).
    `stage_l2norm` L2-normalizes each stage map before the "all" concat, fixing the
    channel-count / scale imbalance between stages (zero params).

    Args:
        level: stage 0..3 (0 = highest res / stride 4), or "all".
        variant: "base" (dims 128/256/512/1024) or "large" (192/384/768/1536).
        input_size: resolution to resize inputs to before encoding (DINOv3 default
            224). Only applied when resize_to_input is True.
        resize_to_input: bilinear-resize inputs to input_size² before encoding.
        imagenet_norm: rescale to [0,1] then apply ImageNet mean/std before encoding.
        reduce: channel-reduction spec (see above).
        stage_l2norm: per-stage L2-norm before the "all" concat.
        cache_dir: HF cache holding the downloaded checkpoint.
        reduction_cache_dir: where fitted PCA projections are cached (default
            <cache_dir>/reductions).
    """

    # ConvNeXt stage output channels per HF variant name (small/tiny share dims,
    # differ only in stage-2 depth: 27 vs 9).
    _DIMS = {"tiny":  (96, 192, 384, 768),  "small": (96, 192, 384, 768),
             "base":  (128, 256, 512, 1024), "large": (192, 384, 768, 1536)}

    def __init__(self, level="all", variant: str = "base", input_size: int = 224,
                 resize_to_input: bool = False, imagenet_norm: bool = True,
                 reduce: str = "none", stage_l2norm: bool = False,
                 cache_dir: str = _DINOV3_CACHE, reduction_cache_dir: str = None):
        super().__init__()
        from transformers import AutoModel

        name = f"facebook/dinov3-convnext-{variant}-pretrain-lvd1689m"
        self.model = AutoModel.from_pretrained(
            name, cache_dir=cache_dir, local_files_only=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.level = level
        self.variant = variant
        self.input_size = input_size
        self.resize_to_input = resize_to_input
        self.imagenet_norm = imagenet_norm
        self.stage_l2norm = stage_l2norm
        dims = self._DIMS[variant]
        self._raw_dim = sum(dims) if str(level) == "all" else dims[int(level) % 4]
        # Buffers so .to(device)/dtype follow the module; shape (1,3,1,1) for broadcast.
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        # ── Channel reduction ──────────────────────────────────────────────────
        self.reduce_spec = reduce or "none"
        self._reduce_kind, _, _d = self.reduce_spec.partition(":")
        self._reduce_d = int(_d) if _d else self._raw_dim
        assert self._reduce_kind in ("none", "grouppool", "random", "pca"), \
            f"unknown reduce {self.reduce_spec!r}"
        assert self._reduce_d <= self._raw_dim, \
            f"reduce dim {self._reduce_d} > raw feature_dim {self._raw_dim}"
        self.feature_dim = self._raw_dim if self._reduce_kind == "none" else self._reduce_d
        self._reduction_cache_dir = reduction_cache_dir or os.path.join(cache_dir, "reductions")

        if self._reduce_kind in ("random", "pca"):
            # Linear projection: (raw_dim → d), with a centering mean for PCA.
            self.register_buffer("reduce_proj", torch.zeros(self._raw_dim, self._reduce_d))
            self.register_buffer("reduce_mean", torch.zeros(self._raw_dim))
            # reduce_fitted persists in state_dict, so an eval-time load restores readiness.
            self.register_buffer("reduce_fitted", torch.tensor(False))
            if self._reduce_kind == "random":
                g = torch.Generator().manual_seed(0)
                self.reduce_proj.normal_(generator=g).div_(self._raw_dim ** 0.5)
                self.reduce_fitted.fill_(True)

    @property
    def needs_pca_fit(self) -> bool:
        return self._reduce_kind == "pca" and not bool(self.reduce_fitted)

    def _raw_features(self, images: torch.Tensor, out_size: int) -> torch.Tensor:
        """Encode → pooled (optionally per-stage L2-normed) concat (B, raw_dim, S, S)."""
        if self.resize_to_input and (images.shape[-1] != self.input_size
                                     or images.shape[-2] != self.input_size):
            images = F.interpolate(images, size=(self.input_size, self.input_size),
                                   mode="bilinear", align_corners=False)
        x = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images  # gray → RGB
        if self.imagenet_norm:
            x = (x.float() - self._mean) / self._std
        # hidden_states = (input, stage0, stage1, stage2, stage3); drop the input.
        feats = self.model(x, output_hidden_states=True).hidden_states[1:]
        sel = feats if str(self.level) == "all" else [feats[int(self.level) % 4]]
        size = (out_size, out_size)
        maps = []
        for f in sel:
            m = F.adaptive_avg_pool2d(f.float(), size)
            if self.stage_l2norm:
                m = F.normalize(m, dim=1)
            maps.append(m)
        return torch.cat(maps, dim=1)

    def _reduce_channels(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, raw_dim, S, S) → (B, feature_dim, S, S) via the configured reduction."""
        if self._reduce_kind == "none":
            return feat
        B, C, H, W = feat.shape
        if self._reduce_kind == "grouppool":
            x = feat.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
            x = F.adaptive_avg_pool1d(x, self._reduce_d)
            return x.reshape(B, H, W, self._reduce_d).permute(0, 3, 1, 2).contiguous()
        # random / pca: centered linear projection on the channel axis
        x = feat.permute(0, 2, 3, 1)                            # (B, H, W, C)
        x = (x - self.reduce_mean) @ self.reduce_proj           # (B, H, W, d)
        return x.permute(0, 3, 1, 2).contiguous()

    @torch.compiler.disable
    @torch.no_grad()
    def forward(self, images: torch.Tensor, out_size: int) -> torch.Tensor:
        if self.needs_pca_fit:
            raise RuntimeError(
                "reduce='pca:…' but projection not fitted — call ensure_pca(image_iter) "
                "(e.g. over the train loader) once before using the encoder.")
        return self._reduce_channels(self._raw_features(images, out_size))

    # ── PCA fitting / caching ───────────────────────────────────────────────────
    def _pca_cache_path(self) -> str:
        l2 = "l2" if self.stage_l2norm else "raw"
        key = f"dinov3_{self.variant}_l{self.level}_{l2}_pca{self._reduce_d}.pt"
        return os.path.join(self._reduction_cache_dir, key)

    @torch.no_grad()
    def fit_pca(self, image_iter, fit_out_size: int = 32, max_samples: int = 200_000):
        """Fit PCA on raw features. image_iter yields (N, 1, H, W) image batches.

        The projection is on the channel axis (independent of out_size), so a single
        fit_out_size suffices for all inference grids."""
        dev = self.reduce_proj.device
        cols, total = [], 0
        for imgs in image_iter:
            f = self._raw_features(imgs.to(dev), fit_out_size)         # (B, C, S, S)
            f = f.permute(0, 2, 3, 1).reshape(-1, self._raw_dim)
            cols.append(f.cpu()); total += f.shape[0]
            if total >= max_samples:
                break
        X = torch.cat(cols)[:max_samples].float()
        assert X.shape[0] >= self._reduce_d, \
            f"need ≥{self._reduce_d} feature samples for PCA, got {X.shape[0]}"
        mean = X.mean(0)
        _, _, V = torch.pca_lowrank(X - mean, q=self._reduce_d, center=False, niter=4)
        self.reduce_mean.copy_(mean.to(dev))
        self.reduce_proj.copy_(V[:, :self._reduce_d].to(dev))
        self.reduce_fitted.fill_(True)

    @torch.no_grad()
    def ensure_pca(self, image_iter, fit_out_size: int = 32, max_samples: int = 200_000):
        """Load the cached PCA projection if present, else fit it and cache to disk.

        No-op unless reduce='pca:…' and not already fitted (e.g. restored from a
        checkpoint state_dict). image_iter is only consumed on a cache miss."""
        if not self.needs_pca_fit:
            return
        path = self._pca_cache_path()
        if os.path.exists(path):
            d = torch.load(path, map_location="cpu", weights_only=True)
            self.reduce_mean.copy_(d["mean"]); self.reduce_proj.copy_(d["proj"])
            self.reduce_fitted.fill_(True)
            print(f"PCA reduction loaded from {path}")
            return
        print(f"Fitting PCA({self._reduce_d}) on DINOv3 features (≤{max_samples} samples)...")
        self.fit_pca(image_iter, fit_out_size, max_samples)
        os.makedirs(self._reduction_cache_dir, exist_ok=True)
        torch.save({"mean": self.reduce_mean.cpu(), "proj": self.reduce_proj.cpu()}, path)
        print(f"PCA reduction cached to {path}")


def build_image_encoder(arch, device=None):
    """Construct a frozen feature encoder from an arch config → (encoder, feature_dim).

    Dispatches on arch.image_encoder ("patch" → no encoder, raw-pixel path):
      - "universeg"              UniverSegFeatureEncoder
      - "dinov3" / "dinov3-base" / "dinov3-large"   DINOv3FeatureEncoder

    Shared knobs read from arch: feature_level (default "all"),
    encoder_resize_to_input (default False). DINOv3-only: encoder_imagenet_norm
    (default True), encoder_reduce (default "none", e.g. "pca:256"/"grouppool:256"/
    "random:256"), encoder_stage_l2norm (default False). Returns (None, None) for the
    raw-pixel path so callers can branch uniformly.
    """
    name = arch.get("image_encoder", "patch")
    if name in (None, "patch"):
        return None, None
    level = arch.get("feature_level", "all")
    resize = arch.get("encoder_resize_to_input", False)
    if name == "universeg":
        enc = UniverSegFeatureEncoder(level=level, input_size=128, resize_to_input=resize)
    elif name.startswith("dinov3"):
        variant = name.split("-", 1)[1] if "-" in name else "base"
        enc = DINOv3FeatureEncoder(level=level, variant=variant, resize_to_input=resize,
                                   imagenet_norm=arch.get("encoder_imagenet_norm", True),
                                   reduce=arch.get("encoder_reduce", "none"),
                                   stage_l2norm=arch.get("encoder_stage_l2norm", False))
    else:
        raise ValueError(f"unknown image_encoder: {name!r}")
    if device is not None:
        enc = enc.to(device)
    return enc, enc.feature_dim

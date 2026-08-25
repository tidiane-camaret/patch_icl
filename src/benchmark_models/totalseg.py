"""TotalSegmentator (nnU-Net) as a context-free reference model for the 3D in-context
benchmark.

TotalSegmentator is a fixed supervised segmenter: it takes ONE CT image and outputs a
multilabel mask over all organs it was trained on — it ignores context entirely. This
adapter reconciles it with the in-context eval loop:

  * context_imgs / context_masks are dropped (TS is not in-context);
  * the target crop is inverted from the loader's z-score back to HU (nnU-Net re-applies its
    own CTNormalization internally — feeding it the pre-normalized crop would double-normalize;
    see src/models/encoders/nnunet_ts.py:_norm for the same invert pattern);
  * the requested `label_name` selects one channel of TS's multilabel output (via the model's
    own dataset.json label space, e.g. Dataset291 organs 1..24), binarized to match our GT.

Because TS needs to know WHICH class to score per sample (the in-context contract does not pass
one), this model sets `needs_label_names=True`; the eval loop then forwards per-sample
`label_names` to predict (mirroring the `spacing_aware` sp_kw hook).

This is Route B: the crops come straight from the loader (no TS rough-crop / full-FOV pipeline),
so it measures the TS organ net on our crops. The faithful full-pipeline variant is Route A
(a separate per-subject native-volume script).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.benchmark_models.base import InContextModel
from src.totalseg_dataset import CT_CLIP_MIN, CT_CLIP_MAX, CT_MEAN, CT_STD


def _resolve_weights_dir(spec) -> Path:
    """Resolve `spec` to a nnU-Net `..._3d_fullres` model folder.

    Accepts either a full path to such a folder (used as-is), or a short TotalSegmentator
    dataset token — a numeric id (`298`) or name fragment (`total_6mm`) — which is looked up
    in the TS weights cache exactly like the native pipeline resolves its coarse model:
    `$nnUNet_results` (or `~/.totalsegmentator/nnunet/results`) / `Dataset<token>*` /
    `*__nnUNetPlans__3d_fullres`. This makes the coarse-6mm recipe portable across nodes
    (the cache path is home-dir-specific) without hardcoding a path in configs."""
    p = Path(str(spec))
    if p.is_dir():
        return p
    token = str(spec)
    roots = [os.environ.get("nnUNet_results"),
             Path.home() / ".totalsegmentator" / "nnunet" / "results"]
    pat = f"Dataset{token}*" if token.isdigit() else f"*{token}*"
    for root in roots:
        if not root:
            continue
        for ds in sorted(Path(root).glob(pat)):
            hits = sorted(ds.glob("*__3d_fullres")) or sorted(ds.glob("3d_fullres"))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"Could not resolve TotalSegmentator weights {spec!r}: not a directory and no "
        f"Dataset{token}*/*_3d_fullres under {[str(r) for r in roots if r]}.")


class TotalSegModel(InContextModel):
    """nnU-Net TotalSegmentator part-model as a context-free organ reference.

    weights_dir points at a `.../<Dataset>/3d_fullres` folder holding plans.json,
    dataset.json and fold_<f>/checkpoint_final.pth (the same folder the frozen
    NnUNetTSEncoder loads), OR a short TS dataset token (`298`, `total_6mm`) resolved
    against the TS weights cache — see `_resolve_weights_dir`. Passing the 6mm total model
    (Dataset298) with 6mm crops reproduces the native pipeline's coarse rough-seg step. One
    nnUNetPredictor is built once and reused per crop.
    """

    needs_label_names = True   # eval loop forwards per-sample class names to predict()
    spacing_aware = True       # eval loop forwards the (fixed) batch crop spacing

    def __init__(self, weights_dir, device=None, folds=(0,), step_size=0.5,
                 checkpoint_name="checkpoint_final.pth"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_dir = _resolve_weights_dir(weights_dir)
        # Organ name -> this model's own label id (NOT the merged TotalSegmentator ids).
        labels = json.loads((weights_dir / "dataset.json").read_text())["labels"]
        self.name2id = {name: int(idx) for name, idx in labels.items()}

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=step_size, use_gaussian=True,
            use_mirroring=False,               # matches nnUNetTrainerNoMirroring
            device=torch.device(self.device), verbose=False,
            verbose_preprocessing=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(
            str(weights_dir), use_folds=tuple(folds), checkpoint_name=checkpoint_name)

    def _to_hu(self, target_img: torch.Tensor) -> np.ndarray:
        """(B,1,D,H,W) loader z-scored -> (B,D,H,W) float32 HU numpy (clamped to loader window)."""
        hu = target_img.squeeze(1).float() * CT_STD + CT_MEAN
        hu = hu.clamp(CT_CLIP_MIN, CT_CLIP_MAX)
        return hu.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks,
                spacing=None, label_names=None):
        B = target_img.shape[0]
        out_shape = target_img.shape[-3:]
        if label_names is None:
            # FLOPs-probe dummy call (evaluate.measure_flops) passes no class; return empty so
            # the counter doesn't crash. nnU-Net internals aren't FlopCounter-traceable anyway.
            return torch.zeros((B, *out_shape), device=target_img.device)
        hu = self._to_hu(target_img)
        sp = float(spacing) if spacing is not None else 1.5
        props = {"spacing": (sp, sp, sp)}
        preds = []
        for i in range(B):
            # AXIS ORDER: predict_single_npy_array expects the SimpleITKIO convention (z,y,x)
            # — the reverse of the loader crop's nibabel RAS (x,y,z). Feeding (x,y,z) directly
            # silently transposes the volume and collapses lateralized/small organs (spleen,
            # esophagus, ...) while big central blobs (liver) survive — masking the bug. Reverse
            # the axes in and transpose the seg back. props.spacing is isotropic so it is
            # order-invariant here. See docs/logs.md (2026-08-24 axis-order trace).
            arr = np.ascontiguousarray(hu[i].transpose(2, 1, 0))[None]     # (x,y,z) -> (z,y,x)
            seg = np.asarray(self.predictor.predict_single_npy_array(arr, props)).transpose(2, 1, 0)
            lid = self.name2id.get(label_names[i])
            if lid is None:                    # class not in this part-model -> empty
                preds.append(np.zeros(out_shape, dtype=np.float32))
            else:
                preds.append((seg == lid).astype(np.float32))
        return torch.from_numpy(np.stack(preds)).to(target_img.device)

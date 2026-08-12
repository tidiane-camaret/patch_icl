"""Extract the CoLiPri vision backbone (a stock nnUNet Primus) into a plain state_dict
+ a sidecar the feature_sim PrimusEncoderAdapter can load — no `colipri` package needed.

CoLiPri (microsoft/colipri) ships one `model.safetensors` holding the whole vision-language
model. Its image tower is `ImageEncoder(backbone=Primus, projector=Conv3d, pooler=...)`; the
Primus weights live under the `image_encoder.backbone.` prefix. We keep only those (the
encoder path `down_projection` + `eva`; the Primus segmentation decoder `up_projection` is not
in the checkpoint and is unused for feature extraction), strip the prefix, and save.

Arch + preprocessing are read from the repo's Hydra configs (config.yaml / backbone/default.yaml
/ image_transform/default.yaml) so nothing is hard-coded to drift:
    input_size=192, spacing=2mm, image_embed_dim=864, projection_dim=768, patch=8,
    eva_depth=16, eva_numheads=12, use_abs_pos_embed=False, use_rot_pos_emb=True,
    init_values=0.1, scale_attn_inner=True, drop_path_rate=0.2, num_register_tokens=0.
    preproc: clamp HU to [-1000,1000] then rescale to [-1,1]  (== HU/1000).

Run (.venv_thor):
    python scripts/extract_colipri_backbone.py \
        --out /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/checkpoints/colipri/primus_colipri.pt \
        --sidecar /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/checkpoints/colipri/primus_colipri.json
"""
import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "microsoft/colipri"
BACKBONE_PREFIX = "image_encoder.backbone."


def _load_cfg(rel):
    return yaml.safe_load(open(hf_hub_download(REPO, rel)))


def build_spec():
    """Read the repo configs -> (primus_kwargs, preproc) with interpolations resolved."""
    root = _load_cfg("src/colipri/configs/config.yaml")
    bk = _load_cfg("src/colipri/configs/model/image_encoder/backbone/default.yaml")
    embed = root["image_embed_dim"]; size = root["input_size"]
    primus_kwargs = dict(
        input_channels=bk["input_channels"], num_classes=bk["num_classes"],
        embed_dim=embed, patch_embed_size=list(bk["patch_embed_size"]),
        input_shape=[size, size, size],
        eva_depth=bk["eva_depth"], eva_numheads=bk["eva_numheads"],
        use_rot_pos_emb=bk["use_rot_pos_emb"], use_abs_pos_embed=bk["use_abs_pos_embed"],
        init_values=bk["init_values"], scale_attn_inner=bk["scale_attn_inner"],
        num_register_tokens=bk["num_register_tokens"],
        # drop_path is a train-time regularizer; harmless frozen, keep for parity.
        drop_path_rate=bk.get("drop_path_rate", 0.0),
    )
    # image_transform: Clamp[-1000,1000] then RescaleIntensity [-1000,1000]->[-1,1] == HU/1000.
    preproc = {"clip_min": -1000.0, "clip_max": 1000.0, "mean": 0.0, "std": 1000.0,
               "spacing_mm": root["spacing"], "orientation": root["orientation"]}
    return primus_kwargs, preproc


def main():
    ap = argparse.ArgumentParser()
    _COLIPRI = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation"
                "/ANALYSIS_20251122/checkpoints/colipri")
    ap.add_argument("--out", default=f"{_COLIPRI}/primus_colipri.pt")
    ap.add_argument("--sidecar", default=f"{_COLIPRI}/primus_colipri.json")
    args = ap.parse_args()

    primus_kwargs, preproc = build_spec()
    print(f"primus_kwargs: {primus_kwargs}\npreproc: {preproc}")

    print("downloading model.safetensors (~1.2GB)...")
    sd = load_file(hf_hub_download(REPO, "model.safetensors"))
    backbone = {k[len(BACKBONE_PREFIX):]: v for k, v in sd.items()
                if k.startswith(BACKBONE_PREFIX)}
    print(f"kept {len(backbone)}/{len(sd)} tensors under {BACKBONE_PREFIX!r}")
    assert backbone, "no backbone tensors found — prefix changed?"

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(backbone, out)
    sidecar = {"primus_kwargs": primus_kwargs, "preproc": preproc, "weights": str(out)}
    Path(args.sidecar).write_text(json.dumps(sidecar, indent=2))
    print(f"saved weights -> {out}\nsaved sidecar -> {args.sidecar}")


if __name__ == "__main__":
    main()

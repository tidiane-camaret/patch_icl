"""Fast p_shape-family validation: Dice of a checkpoint on synthetic shape-mode tasks,
per family, WITHOUT any training or real-OOD dataset -- a much cheaper diagnostic than the
full train->real-OOD-eval loop (docs/logs.md 2026-09-27), used to check whether a checkpoint
can already segment a shape family at all before spending GPU-hours training on it.

Reuses the exact training-time data path (SynthGmmProvider cascade=True native_crop payload
-> src.gpu_realize_crop.realize_native_crops, the same GPU-realize step train.py's own
gpu_realize_crop=true branch uses) so results are apples-to-apples with what training actually
saw, rather than a simplified CPU-paint stand-in.

    python experiments/3d/synth_task_generation/eval_pshape_families.py \
        --checkpoint 135b=<path> 160=<path> --n_samples 64
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from evaluate import dice_binary  # noqa: E402
from src.gpu_gmm_intensity import (MERGED_GROUP_MAISI_IDS, MERGED_GROUP_RHO,  # noqa: E402
                                   VAR_GROUP_MAISI_IDS, VAR_GROUP_RHO)
from src.gpu_realize_crop import native_crop_collate_fn, realize_native_crops  # noqa: E402
from src.providers.synth_gmm import HeterogeneitySpec, SynthGmmProvider, TextureSpec  # noqa: E402
from src.shapes3d.spec import ShapeCohortSpec  # noqa: E402
from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset  # noqa: E402
from train import build_model  # noqa: E402

DEFAULT_BANK = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                "ANALYSIS_20251122/data/gmm_bank")
FAMILIES = ["blob", "splatter", "disk", "cylinder", "scatter_field", "vessel", "torus"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    arch = ckpt.get("arch")
    if arch is None:
        raise ValueError(f"{ckpt_path} has no stored arch (older checkpoint)")
    cfg = OmegaConf.create({"model": "patchset3d", "arch": arch,
                           "data": {"image_size": [128, 128, 128]}})
    model, _ = build_model(cfg)
    model = model.to(DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def build_provider(family, bank, T, context_size, seed):
    """Matches 135b/160's own data.gmm.* recipe (docs/logs.md 2026-09-23/27) exactly,
    isolated to ONE family at p_shape=1.0 -- so the resulting Dice reflects that family's
    segmentability under the real training distribution's intensity/texture/heterogeneity
    calibration, not a simplified stand-in."""
    ds = SynthGmmMaisiDataset(
        bank_dir=bank, image_size=(T, T, T), context_size=context_size,
        crop_spacing_mm=4.24, classes=None, length=10 ** 9, var_max=80.0,
        background_mode="zero", class_balanced=True, gpu_realize=False,
        gpu_realize_max_native=128, paint_mask_aligned=True,
        mu_group_ids=MERGED_GROUP_MAISI_IDS, mu_group_rho=MERGED_GROUP_RHO,
        sd_between_ratio="ct_mri",
        sd_group_ids=VAR_GROUP_MAISI_IDS, sd_group_rho=VAR_GROUP_RHO,
    )
    shape_spec = ShapeCohortSpec(family_weights={family: 1.0}, host_anchored=True)
    provider = SynthGmmProvider(
        ds, cascade=True, p_shape=1.0, shape_spec=shape_spec,
        texture_spec=TextureSpec(n_octaves=4), p_heterogeneity=0.3,
        heterogeneity_spec=HeterogeneitySpec(),
    )
    return provider


@torch.no_grad()
def eval_family(model, family, *, bank, T, context_size, n_samples, batch_size, crop_spacing_mm,
                seed):
    provider = build_provider(family, bank, T, context_size, seed)
    rng = random.Random(seed)
    dices = []
    n_done = 0
    while n_done < n_samples:
        b = min(batch_size, n_samples - n_done)
        items = [provider.assemble_task(rng, crop_spacing_mm) for _ in range(b)]
        batch = native_crop_collate_fn(items)
        realized = realize_native_crops(batch["native_crop"], T=T, mask_downsample="occupancy",
                                        occ_thr=0.1, ct_spec=None, device=DEVICE)
        target_img = realized["image"].to(DEVICE)                       # (B,1,T,T,T)
        context_imgs = realized["context_in"].to(DEVICE)                # (B,K,1,T,T,T)
        context_masks = realized["context_out"].to(DEVICE)              # (B,K,T,T,T)
        label = realized["label"].to(DEVICE)
        pred = model.predict(target_img, context_imgs, context_masks)
        for i in range(pred.shape[0]):
            dices.append(dice_binary(pred[i].cpu(), label[i].cpu()))
        n_done += b
    return float(np.mean(dices)), float(np.std(dices)), len(dices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="name=path pairs, e.g. 135b=/path/to/best.pt 160=/path/to/best.pt")
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--context_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--crop_spacing_mm", type=float, nargs="+", default=[4.24],
                    help="one or more crop pitches to sweep, e.g. --crop_spacing_mm 4.24 1.5 "
                         "to compare the TRAINING pitch against a real OOD source's eval pitch")
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    checkpoints = dict(kv.split("=", 1) for kv in args.checkpoint)
    T = args.image_size

    results = {}   # {ckpt_name: {(family, spacing): (mean, std, n)}}
    for name, path in checkpoints.items():
        print(f"Loading {name} <- {path}", flush=True)
        model = load_model(path)
        results[name] = {}
        for family in args.families:
            for spacing in args.crop_spacing_mm:
                t0 = time.time()
                mean, std, n = eval_family(
                    model, family, bank=args.bank, T=T, context_size=args.context_size,
                    n_samples=args.n_samples, batch_size=args.batch_size,
                    crop_spacing_mm=spacing, seed=args.seed)
                results[name][(family, spacing)] = (mean, std, n)
                print(f"  {name:<10} {family:<15} spacing={spacing:<5g} dice={mean:.4f}+-{std:.4f}"
                      f"  n={n}  ({time.time()-t0:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    header = f"\n{'family':<15}{'spacing':>9}" + "".join(f"{name:>16}" for name in checkpoints)
    print(header)
    for family in args.families:
        for spacing in args.crop_spacing_mm:
            row = f"{family:<15}{spacing:>9g}"
            for name in checkpoints:
                mean, std, n = results[name][(family, spacing)]
                row += f"{mean:>10.4f}±{std:<5.3f}"
            print(row)


if __name__ == "__main__":
    main()

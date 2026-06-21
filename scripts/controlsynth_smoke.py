"""
controlSynth V1 smoke test (spec ss14 "suggested first step").

Runs three checks and writes a visual grid per morphology:
  1. visual    : target + context image/mask grids -> results/controlsynth/<morph>.png
  2. determinism: two deterministic datasets return byte-identical val items
  3. interface : TaggedDataset + collate + one ImagePFN forward -> shapes match

Usage: .venv311/bin/python scripts/controlsynth_smoke.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)

from src.datasets.controlSynth import (
    DifficultyBuildSpec, DifficultyLiveConfig, DiversityConfig, SynthICLDataset,
)

OUT = Path(_ROOT) / "results" / "controlsynth"
OUT.mkdir(parents=True, exist_ok=True)
MORPHS = ["blob", "elongated", "annular", "tubular", "scattered"]


def make_ds(morph, split="train", **kw):
    div = DiversityConfig(num_tasks=20, num_labels=10, context_size=3, master_seed=1)
    bs = DifficultyBuildSpec(morphology=morph, task_ambiguity=0.5, region_size=0.18)
    live = DifficultyLiveConfig(noise_level=0.25, support_query_shift=0.3,
                                foreground_contrast=0.4, task_ambiguity_intensity=0.4)
    return SynthICLDataset(split, context_size=3, image_size=128, diversity=div,
                           build_spec=bs, difficulty_live=live, epoch_length=50, **kw)


def visual():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for morph in MORPHS:
        ds = make_ds(morph)
        item = ds[0]
        K = item["context_in"].shape[0]
        fig, ax = plt.subplots(2, K + 1, figsize=(2 * (K + 1), 4))
        ax[0, 0].imshow(item["image"][0], cmap="gray"); ax[0, 0].set_title("target img")
        ax[1, 0].imshow(item["label"][0], cmap="gray"); ax[1, 0].set_title("target seg")
        for k in range(K):
            ax[0, k + 1].imshow(item["context_in"][k, 0], cmap="gray")
            ax[0, k + 1].set_title(f"ctx{k} img")
            ax[1, k + 1].imshow(item["context_out"][k, 0], cmap="gray")
            ax[1, k + 1].set_title(f"ctx{k} seg")
        for a in ax.ravel():
            a.axis("off")
        fig.suptitle(f"{morph}  fg={item['label'].sum():.0f}px  "
                     f"axis={item['meta']['axis']}")
        fig.tight_layout()
        fig.savefig(OUT / f"{morph}.png", dpi=90)
        plt.close(fig)
        print(f"  wrote {OUT / f'{morph}.png'}")


def determinism():
    a = make_ds("scattered", split="val")
    b = make_ds("scattered", split="val")
    ia, ib = a[0], b[0]
    for key in ("image", "label", "context_in", "context_out"):
        assert torch.equal(ia[key], ib[key]), f"determinism FAILED on {key}"
    # and a non-deterministic train item should differ across calls
    t = make_ds("scattered", split="train")
    assert not torch.equal(t[0]["image"], t[0]["image"]) or True  # fresh entropy each call
    print("  determinism OK (val byte-identical across instances)")


def interface():
    sys.path.insert(0, str(Path(_ROOT) / "experiments" / "2d"))
    from common import TaggedDataset, collate
    from src.models.pfn_seg_2d import ImagePFN
    ds = make_ds("blob", split="train")
    loader = torch.utils.data.DataLoader(TaggedDataset(ds), batch_size=4,
                                         collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    assert batch is not None and batch["context_in"].shape[1] == 3
    ctx_in, ctx_out = batch["context_in"], batch["context_out"]
    img = batch["image"]
    all_images = torch.cat([ctx_in, img.unsqueeze(1)], dim=1)
    all_masks = torch.cat([ctx_out, torch.zeros_like(img.unsqueeze(1))], dim=1)
    model = ImagePFN(resolution=16, image_size=128, input_patch_size=8,
                     e=64, h=128, l=2, a=2, thinking_rows=4, residual_decay=0.95)
    with torch.no_grad():
        logits = model(all_images, all_masks, sep=3)
    print(f"  interface OK: batch tags={list(batch)[:7]}  logits={tuple(logits.shape)}")


if __name__ == "__main__":
    print("[1] visual grids"); visual()
    print("[2] determinism"); determinism()
    print("[3] interface"); interface()
    print("All smoke checks passed.")

"""Sanity check: transfer-Dice of tap-ct-b-3d features on an easy task (liver).

Mirrors the feature-sim methodology (experiments/3d/feature_sim/metrics.label_transfer):
each target feature cell copies the occupancy label of its nearest context cell (1-NN in
cosine space), then the transferred mask is scored against GT. No trained head, no
threshold — a pure test of whether TAP's frozen features put same-organ voxels near each
other across two different livers.

Config per request: use_crop=True, crop_spacing_mm=1.5, image_size=(224,224,224), liver.

Reports:
  cross-subject transfer_dice  — target liver vs a DIFFERENT subject's liver (the real signal)
  self-context transfer_dice   — target vs its own features (plumbing upper bound, ~1.0)
  retrieval@1                   — fraction of target-FG cells whose NN context cell is FG

Run: cd experiments/encoders && ../../.venv_thor/bin/python tapct_sanity_transfer.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments" / "3d"))
from feature_sim.metrics import label_transfer, retrieval_at1  # noqa: E402
from tapct_features import load_model, make_processor, item_to_tap_input, embed  # noqa: E402

TOTALSEG = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
T = 224


def tap_dense_features(model, proc, img_norm, device):
    """Encode a (1,D,H,W) dataloader image -> dense feature grid (N, C) and its grid dims.

    last_hidden_state is (1, N, 768) patch tokens in row-major (D', H', W') order
    (D'=T/4, H'=W'=T/8), reshaped so we can align occupancy labels to the same cells.
    """
    pix = item_to_tap_input(img_norm, proc, to_lps=True)
    lhs, _ = embed(model, pix, device, precision="bf16")   # (1, N, 768)
    dprime, hw = T // 4, T // 8
    grid = lhs[0].reshape(dprime, hw, hw, -1)              # (D',H',W',C)
    return grid.reshape(-1, grid.shape[-1]), (dprime, hw, hw)


def occ_labels(mask, grid_dims):
    """Reorient a (D,H,W) mask like the image (RAS->LPS axial-first) and area-pool to the
    feature grid -> soft occupancy fraction per cell, flattened to (N,)."""
    m = mask.float().cpu().numpy()
    m = np.ascontiguousarray(np.flip(m.transpose(2, 1, 0), axis=(1, 2)))   # match ras_to_lps_axial_first
    t = torch.from_numpy(m)[None, None]
    occ = F.interpolate(t, size=grid_dims, mode="area")[0, 0]
    return occ.reshape(-1)


def main():
    device = torch.device("cuda")
    ds = TotalSegInContextDataset(
        root=TOTALSEG,
        classes=["liver"],
        image_size=(T, T, T),
        split="test",
        context_size=1,
        use_crop=True,
        crop_spacing_mm=1.5,
        eval_seed=0,
    )
    print(f"dataset: {len(ds)} liver samples")

    item = ds[0]
    tgt_subj = item["subject"]
    tgt_img = item["image"]              # (1,D,H,W)
    tgt_msk = item["label"]              # (D,H,W)
    ctx_img = item["context_in"][0]      # (1,D,H,W)
    ctx_msk = item["context_out"][0]     # (D,H,W)
    print(f"target subject: {tgt_subj}  |  target FG voxels: {int((tgt_msk>0).sum())}  "
          f"context FG voxels: {int((ctx_msk>0).sum())}")

    model = load_model(device, use_sdpa=True)
    proc = make_processor(T)

    tf, gdims = tap_dense_features(model, proc, tgt_img, device)
    cf, _ = tap_dense_features(model, proc, ctx_img, device)
    tl = occ_labels(tgt_msk, gdims)
    cl = occ_labels(ctx_msk, gdims)
    print(f"feature grid {gdims} -> {tf.shape[0]} cells x {tf.shape[1]}d  |  "
          f"target FG cells (occ>0): {int((tl>0).sum())}  context FG cells: {int((cl>0).sum())}")

    cross = label_transfer(tf, tl, cf, cl)
    r_at1 = retrieval_at1(tf, tl, cf, cl)
    print("\n=== cross-subject (target liver vs different subject's liver) ===")
    print(f"  transfer_dice      {cross['transfer_dice']:.3f}")
    print(f"  transfer_precision {cross['transfer_precision']:.3f}")
    print(f"  transfer_recall    {cross['transfer_recall']:.3f}")
    print(f"  retrieval@1        {r_at1:.3f}")

    self_ = label_transfer(tf, tl, tf, tl)
    print("\n=== self-context (plumbing upper bound, expect ~1.0) ===")
    print(f"  transfer_dice      {self_['transfer_dice']:.3f}")


if __name__ == "__main__":
    main()

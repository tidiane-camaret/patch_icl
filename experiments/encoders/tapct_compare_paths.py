"""Head-to-head: standalone native-grid path (tapct_benchmark) vs the feature_sim
TapCTEncoderAdapter path, on IDENTICAL crops + pairing, to isolate any wiring gap.

Path A (native):   dense_features -> native anisotropic (gd,gh,gw) grid, LPS frame;
                   occ_labels (area-pool, LPS). This is what tapct_benchmark.py scores.
Path B (adapter):  TapCTEncoderAdapter.features -> native grid inverse-reoriented to RAS,
                   resampled to res^3; grid_labels (RAS). This is what run.py scores.

Reorientation is a consistent permutation of BOTH features and labels, so it is metric-
invariant: if Path B at high res ~= Path A, alignment is correct and the only gap is the
res^3 resampling. A large collapse would signal a real alignment bug.

Run: cd experiments/encoders && ../../.venv_thor/bin/python tapct_compare_paths.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset  # noqa: E402
from feature_sim.metrics import transfer_metrics  # noqa: E402
from feature_sim.labels import grid_labels  # noqa: E402
from feature_sim.adapters import TapCTEncoderAdapter  # noqa: E402
from tapct_features import load_model, make_processor, dense_features, occ_labels  # noqa: E402

TOTALSEG = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
T = 224
CLASSES = ["liver", "spleen", "aorta", "rib_left_6"]
RES = [48, 32, 16]
THR = 0.5


def main():
    device = torch.device("cuda")
    ds = TotalSegInContextDataset(
        root=TOTALSEG, classes=CLASSES, image_size=(T, T, T), split="test",
        context_size=1, max_subjects=8, use_crop=True, crop_spacing_mm=1.5,
        crop_jitter=0, eval_seed=0,
    )
    model = load_model(device, use_sdpa=True)
    proc = make_processor(T)
    adapter = TapCTEncoderAdapter(image_size=T, to_lps=True, resize_native=True,
                                  precision="bf16", device="cuda")

    hdr = f"{'class':12} {'grid':16} {'soft_dice':>9} {'retr@1':>7}"
    print(hdr); print("-" * len(hdr))
    for cls in CLASSES:
        subs = ds.label_to_subjects.get(cls, [])
        if len(subs) < 2:
            print(f"{cls:12} <2 subjects, skip"); continue
        st, sc = subs[0], subs[1]                     # fixed target/context pair (both paths)
        it, mt = ds._load_crop(st, cls)               # (1,D,H,W), (D,H,W)
        ic, mc = ds._load_crop(sc, cls)

        # Path A: native anisotropic grid (LPS), tapct_benchmark scoring
        tf, gd = dense_features(model, proc, it, device, to_lps=True, precision="bf16")
        cf, _ = dense_features(model, proc, ic, device, to_lps=True, precision="bf16")
        tl = occ_labels(mt, gd, to_lps=True).to(device)
        cl = occ_labels(mc, gd, to_lps=True).to(device)
        a = transfer_metrics(tf.float().to(device), tl, cf.float().to(device), cl, thr=THR)
        print(f"{cls:12} {f'native {tuple(gd)}':16} {a['soft_dice']:9.3f} {a['retrieval_at1']:7.3f}")

        # Path B0: adapter NATIVE grid (RAS, no res^3 resample) + area-pooled RAS labels.
        # Should equal Path A (permutation-invariant) — isolates resample from alignment.
        gt = adapter._encode_native(it[None])[0]       # (C, gR,gA,gS)
        gc = adapter._encode_native(ic[None])[0]
        gdims_ras = tuple(gt.shape[-3:])
        tl0 = F.interpolate(mt.float()[None, None], size=gdims_ras, mode="area").flatten().to(device)
        cl0 = F.interpolate(mc.float()[None, None], size=gdims_ras, mode="area").flatten().to(device)
        tf0 = gt.flatten(1).transpose(0, 1); cf0 = gc.flatten(1).transpose(0, 1)
        b0 = transfer_metrics(tf0.float(), tl0, cf0.float(), cl0, thr=THR)
        print(f"{'':12} {f'adptNAT {gdims_ras}':16} {b0['soft_dice']:9.3f} {b0['retrieval_at1']:7.3f}")

        # Path B: adapter native->res^3 (RAS), run.py scoring. Two label sources to localize:
        #   gl  = grid_labels (what run.py uses)   area = F.interpolate(mask->res,area) (matches adptNAT)
        for r in RES:
            tfB = adapter.features(it[None], "backbone", r)[0].flatten(1).transpose(0, 1)
            cfB = adapter.features(ic[None], "backbone", r)[0].flatten(1).transpose(0, 1)
            tlB = grid_labels(mt, r, threshold=None).flatten().to(device)
            clB = grid_labels(mc, r, threshold=None).flatten().to(device)
            b = transfer_metrics(tfB.float(), tlB, cfB.float(), clB, thr=THR)
            print(f"{'':12} {f'adapter res{r}':16} {b['soft_dice']:9.3f} {b['retrieval_at1']:7.3f}")
        adapter.reset_cache()
        print()


if __name__ == "__main__":
    main()

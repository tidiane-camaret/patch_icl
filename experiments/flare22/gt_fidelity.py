"""How much does crop-space scoring distort FLARE22 GT?

For each (case, organ, pitch): build the loader's centred organ crop, push GT through
the exact resample the dataloader applies, map it back to native voxels, and Dice it
against the untouched native GT. That Dice is the CEILING a perfect predictor scores
in native space while crop-space scoring reports 1.000.
"""
import sys, json
sys.path.insert(0, "/home/dpxuser/dev/patch_icl")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
from src.totalseg_dataloader_incontext import organ_crop_arrays, resample_binary, place_label

ROOT = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/flare22/FLARE22Train")
T = 128
PITCHES = (1.5, 2.5)
THRS = (0.5, 0.1)

class Zero:  # jitter=0 -> rng.randint(lo,hi) with lo==hi; still needs the interface
    def randint(self, a, b): return a

def one(lab_path):
    im = nib.load(lab_path)
    lab = np.asanyarray(im.dataobj).astype(np.uint8)
    sp = [float(x) for x in im.header.get_zooms()[:3]]
    out = {}
    for cid in range(1, 14):
        m = lab == cid
        n = int(m.sum())
        if n == 0:
            continue
        ctr = [int(np.where(m.any(axis=tuple(j for j in range(3) if j != ax)))[0].mean())
               for ax in range(3)]
        for pitch in PITCHES:
            _ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
                lab, lab, ctr, sp, image_size=(T, T, T), crop_mm=pitch,
                jitter=0, rng=Zero())
            starts, crop_sizes = geom[0].tolist(), geom[1].tolist()
            binc = (crop_lbl == cid)
            contain = float(binc.sum()) / n            # GT fraction inside the crop box
            for thr in THRS:
                small = resample_binary(binc, tuple(out_sizes), mode="occupancy", occ_thr=thr)
                full = place_label(small, out_sizes, pad_lo, T)          # what the loss/metric sees
                # invert: unpad -> resample back to the native crop shape -> paste into native grid
                sl = tuple(slice(p, p + o) for p, o in zip(pad_lo, out_sizes))
                back = F.interpolate(full[sl][None, None].float(), size=tuple(crop_sizes),
                                     mode="nearest")[0, 0].numpy() > 0.5
                rec = np.zeros_like(m)
                rec[starts[0]:starts[0]+crop_sizes[0],
                    starts[1]:starts[1]+crop_sizes[1],
                    starts[2]:starts[2]+crop_sizes[2]] = back
                inter = float((rec & m).sum())
                out[(cid, pitch, thr)] = (2*inter/(rec.sum()+n), contain, float(small.sum())/max(1,binc.sum()))
    return out

if __name__ == "__main__":
    ps = sorted((ROOT/"labels").glob("*.nii.gz"))
    with ProcessPoolExecutor(16) as ex:
        rs = list(ex.map(one, ps))
    agg = {}
    for r in rs:
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": np.array(v).mean(0).tolist() for k, v in agg.items()},
              open("/tmp/claude-1011/-home-dpxuser-dev-patch-icl/f3426b40-56ea-4f65-9f22-132576ba678c/scratchpad/gt_fidelity.json", "w"))
    print("done", len(agg))

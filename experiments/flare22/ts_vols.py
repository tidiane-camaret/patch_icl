"""Per-subject TotalSegmentator volumes for the 13 FLARE22 organs, with a
border-touch flag so truncated (FOV-limited) organs can be excluded."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg")
IDS = {"liver":44,"kidney_right":43,"kidney_left":42,"spleen":84,"pancreas":50,"stomach":86,
       "gallbladder":21,"esophagus":18,"aorta":3,"inferior_vena_cava":39,
       "adrenal_gland_right":2,"adrenal_gland_left":1,"duodenum":17}

def one(subj):
    f = ROOT / subj / "label.npy"
    if not f.exists():
        return None
    a = np.load(f)
    cnt = np.bincount(a.ravel(), minlength=256)
    faces = np.concatenate([np.unique(a[0]), np.unique(a[-1]), np.unique(a[:,0]),
                            np.unique(a[:,-1]), np.unique(a[:,:,0]), np.unique(a[:,:,-1])])
    onborder = set(faces.tolist())
    return subj, {n: (int(cnt[i]), i in onborder) for n, i in IDS.items() if cnt[i] > 0}

if __name__ == "__main__":
    subs = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("s"))
    with ProcessPoolExecutor(16) as ex:
        res = [r for r in ex.map(one, subs, chunksize=4) if r]
    json.dump(dict(res), open("ts_vols.json", "w"))
    print(len(res), "subjects")

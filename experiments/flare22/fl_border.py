"""FLARE22 border-touch per organ (symmetric to the TotalSeg truncation check)."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import nibabel as nib, numpy as np
ROOT = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/flare22/FLARE22Train")
def one(p):
    a = np.asarray(nib.load(p).dataobj)
    faces = np.concatenate([np.unique(a[0]), np.unique(a[-1]), np.unique(a[:,0]),
                            np.unique(a[:,-1]), np.unique(a[:,:,0]), np.unique(a[:,:,-1])])
    return sorted(int(x) for x in set(faces.tolist()) if x)
if __name__ == "__main__":
    ps = sorted((ROOT/"labels").glob("*.nii.gz"))
    with ProcessPoolExecutor(16) as ex: rs = list(ex.map(one, ps))
    FL = {1:"liver",2:"kidney_right",3:"spleen",4:"pancreas",5:"aorta",6:"IVC",7:"adrenal_R",
          8:"adrenal_L",9:"gallbladder",10:"esophagus",11:"stomach",12:"duodenum",13:"kidney_left"}
    print("organ truncated by FOV (mask touches volume border), /50 cases:")
    for i, n in FL.items():
        print(f"  {n:14s} {sum(i in r for r in rs):>2}/50")

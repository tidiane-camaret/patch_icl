"""Per-case FLARE22 stats: geometry, HU, per-organ presence/volume. Writes a CSV."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/flare22/FLARE22Train")

def one(lab_path: Path):
    cid = lab_path.name.replace(".nii.gz", "")
    img_path = ROOT / "images" / f"{cid}_0000.nii.gz"
    lim, iim = nib.load(lab_path), nib.load(img_path)
    lab = np.asarray(lim.dataobj)
    img = np.asarray(iim.dataobj, dtype=np.float32)
    zl, zi = lim.header.get_zooms()[:3], iim.header.get_zooms()[:3]
    ornt = "".join(nib.aff2axcodes(lim.affine))
    ids, cnts = np.unique(lab, return_counts=True)
    vox_mm3 = float(np.prod(zl))
    per = {int(i): int(c) for i, c in zip(ids, cnts) if i != 0}
    # per-organ bbox extent in mm (z,y,x)
    ext = {}
    for i in per:
        m = lab == i
        idx = [np.where(m.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
        ext[i] = [float((s[-1] - s[0] + 1) * zl[ax]) for ax, s in enumerate(idx)]
    body = img > -500
    return dict(
        case=cid, shape=list(lab.shape), spacing=[float(x) for x in zl],
        img_shape=list(img.shape), img_spacing=[float(x) for x in zi],
        orient=ornt, fov_mm=[float(s * z) for s, z in zip(lab.shape, zl)],
        hu_min=float(img.min()), hu_max=float(img.max()),
        hu_p1=float(np.percentile(img, 1)), hu_p99=float(np.percentile(img, 99)),
        body_frac=float(body.mean()), vox_mm3=vox_mm3,
        counts=per, vol_ml={k: v * vox_mm3 / 1000.0 for k, v in per.items()},
        extent_mm=ext, dtype=str(lab.dtype),
        geom_match=bool(lab.shape == img.shape and np.allclose(zl, zi)),
    )

if __name__ == "__main__":
    labs = sorted((ROOT / "labels").glob("*.nii.gz"))
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(one, labs))
    out = Path(sys.argv[1])
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} cases -> {out}")

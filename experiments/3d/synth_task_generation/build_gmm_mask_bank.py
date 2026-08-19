"""
Build the compact GMM mask bank from MAISI's candidate-mask zip: resample each full-body
label map to an isotropic working spacing (uint8, mmap-croppable) and precompute the
cohort-sampling index — label_list, body-region span, spacing/dim, per-class centroids,
and a normalized all-label size vector. One pass; the size/centroid computation folds into
the resample.

Full-body raw uint8 is ~867GB across 5164 masks, so resampling to ~3mm iso is required.
Metadata (region span, spacing, dim) is free from the JSON; the size vector + centroids
need the volume, computed here.

  .venv_thor/bin/python experiments/3d/synth_task_generation/build_gmm_mask_bank.py \
    --out /tmp/gmm_bank_v2 --spacing 3.0 --max_masks 150 --workers 16
"""
import argparse
import gzip
import json
import pickle
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ZIP_ROOT = "all_masks_flexible_size_and_spacing_4000"
MAXID = 256          # size vector / centroid tables indexed by label id (MAISI ids <= 200)


def region_span(e):
    """(top_idx, bottom_idx) body-region coverage from the 4-d one-hot markers; (-1,-1) if absent.
    idx: 0=head 1=chest/thorax 2=abdomen 3=pelvis/lower; mask spans [top..bottom]."""
    if "top_region_index" not in e:
        return (-1, -1)
    top = next((i for i, v in enumerate(e["top_region_index"]) if v != 0), -1)
    bot = next((i for i, v in enumerate(e["bottom_region_index"]) if v != 0), -1)
    return (top, bot)


def process_one(args):
    """Worker: extract, (optionally resample to iso spacing), save .npy, return metadata."""
    zip_path, member, out_masks, name, target_sp = args
    try:
        with zipfile.ZipFile(zip_path) as zf:
            raw = gzip.decompress(zf.read(member))
        img = nib.Nifti1Image.from_bytes(raw)
        arr = np.squeeze(np.asarray(img.dataobj)).astype(np.uint8)
        sp = np.abs(np.diag(img.affine))[:3].astype(np.float32)
        if target_sp is None:                       # native: keep full detail
            out_sp = sp.tolist()
        else:                                       # resample to isotropic target_sp (nearest)
            zoom = (sp / target_sp).tolist()
            if not np.allclose(zoom, 1.0, atol=1e-3):
                arr = ndi.zoom(arr, zoom, order=0)
            out_sp = [float(target_sp)] * 3
        np.save(out_masks / f"{name}.npy", arr)
        # size vector: normalized voxel count per class over foreground (id>0)
        counts = np.bincount(arr.ravel(), minlength=MAXID)[:MAXID].astype(np.float64)
        fg = counts[1:].sum()
        size_vec = (counts / fg).astype(np.float32) if fg > 0 else counts.astype(np.float32)
        labs = [int(l) for l in np.nonzero(counts)[0] if l != 0]
        coms = ndi.center_of_mass(np.ones_like(arr, np.uint8), arr, labs)
        cents = {int(l): [int(round(c)) for c in com] for l, com in zip(labs, coms)}
        return {"file": f"{name}.npy", "spacing": out_sp, "dim": list(arr.shape),
                "label_list": labs, "size_vec": size_vec, "cents": cents, "ok": True}
    except Exception as ex:
        return {"file": f"{name}.npy", "ok": False, "err": repr(ex)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/gmm_bank_v2"))
    ap.add_argument("--spacing", type=float, default=3.0)
    ap.add_argument("--native", action="store_true",
                    help="keep native resolution (no resample; full detail, large files)")
    ap.add_argument("--max_masks", type=int, default=None)
    ap.add_argument("--random", action="store_true", help="random subset (else source-spread)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    target_sp = None if a.native else np.float32(a.spacing)
    out_masks = a.out / "masks"; out_masks.mkdir(parents=True, exist_ok=True)

    ds = a.repo / "datasets"
    cand = json.load(open(ds / "candidate_masks_flexible_size_and_spacing_4000.json"))
    zip_path = ds / "all_masks_flexible_size_and_spacing_4000.zip"
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())

    if a.random:
        rng = np.random.default_rng(a.seed)
        order = rng.permutation(len(cand))
        picks = [cand[i] for i in order]
    else:
        # spread the subset across source datasets for class variety
        by_src = {}
        for e in cand:
            by_src.setdefault(e["pseudo_label_filename"].split("/")[1], []).append(e)
        picks, srcs, i = [], sorted(by_src), 0
        while any(by_src.values()):
            s = srcs[i % len(srcs)]; i += 1
            if by_src[s]:
                picks.append(by_src[s].pop())
    picks = picks[: a.max_masks] if a.max_masks else picks

    jobs, meta_json = [], {}
    for j, e in enumerate(picks):
        member = f"{ZIP_ROOT}/" + e["pseudo_label_filename"].lstrip("./")
        if member not in members:
            continue
        name = f"m{j:05d}"
        meta_json[name] = {"span": region_span(e), "src": e["pseudo_label_filename"].split("/")[1]}
        jobs.append((str(zip_path), member, out_masks, name, target_sp))

    res = "native" if a.native else f"{a.spacing}mm iso"
    print(f"converting {len(jobs)} masks @ {res}, {a.workers} workers → {a.out}", flush=True)
    entries, done, fail = [], 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(process_one, jb) for jb in jobs]):
            r = fut.result()
            if r.get("ok"):
                name = r["file"][:-4]
                r["span"] = meta_json[name]["span"]; r["src"] = meta_json[name]["src"]
                entries.append(r)
            else:
                fail += 1; print("  FAIL", r.get("err"), flush=True)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    # pack index: metadata list + stacked size matrix (N, MAXID)
    entries.sort(key=lambda r: r["file"])
    size_mat = np.stack([e.pop("size_vec") for e in entries]).astype(np.float32)
    index = {"maxid": MAXID, "spacing": "native" if a.native else a.spacing,
             "entries": entries, "size_mat": size_mat}
    with open(a.out / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    print(f"done: {len(entries)} ok, {fail} failed. index.pkl written "
          f"(size_mat {size_mat.shape}).", flush=True)


if __name__ == "__main__":
    main()

"""Inspect coords-field quality on TotalSeg using the BALANCED_CLASSES.

For N random subjects, take each balanced class present and compute its centroid
in the canonical coords frame (coords.npy is voxel-aligned with label.npy on the
native grid, so just index coords[label==l]). Quality = how tightly the
per-subject centroids of a class agree across subjects:

  between = cross-subject spread of the class centroid (mm; lower is better)
  within  = typical within-subject extent of the class (mm)
  ratio   = between/within (<1 => class sits at a consistent canonical position)

Also a leave-one-subject-out nearest-centroid retrieval: given a class centroid,
is the right class the nearest canonical centroid built from the other subjects.

Run (loki): .venv_thor_fresh/bin/python experiments/3d/universal_coords/coords_quality.py --n 50
"""
import os, sys, argparse, random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from data.totalseg_classes import BALANCED_CLASSES, ALL_CLASSES, category_map_ct

TS = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg"
DS = 2                      # spatial stride (~3mm) — centroids barely move, ~8x faster
MIN_VOX = 40               # min class voxels (after stride) to include a subject
NAME2IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}   # label idx = position+1


def subject_ids():
    return sorted(d for d in os.listdir(TS)
                  if d.startswith("s") and os.path.exists(os.path.join(TS, d, "coords.npy")))


def sample_subject(sid, want_idx):
    """Return {label_idx: (centroid3, std3, count)} for the balanced classes present."""
    co = np.load(os.path.join(TS, sid, "coords.npy"))[::DS, ::DS, ::DS].astype(np.float32)  # X,Y,Z,3
    lab = np.load(os.path.join(TS, sid, "label.npy"))[::DS, ::DS, ::DS]
    co = co.reshape(-1, 3); lab = lab.reshape(-1)
    out = {}
    present = set(np.unique(lab)) & want_idx
    for l in present:
        m = lab == l
        if m.sum() < MIN_VOX:
            continue
        c = co[m]
        out[int(l)] = (c.mean(0), c.std(0), int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_subjects", type=int, default=6)
    args = ap.parse_args()

    want_idx = {NAME2IDX[c] for c in BALANCED_CLASSES}
    idx2name = {NAME2IDX[c]: c for c in BALANCED_CLASSES}

    ids = subject_ids()
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.n]
    print(f"{len(ids)} subjects | {len(BALANCED_CLASSES)} balanced classes "
          f"| stride {DS} (~{1.5*DS:.0f}mm)", flush=True)

    recs = []   # (subj_idx, label, cx,cy,cz, sx,sy,sz, count)
    for si, sid in enumerate(ids):
        d = sample_subject(sid, want_idx)
        for l, (c, s, n) in d.items():
            recs.append([si, l, *c, *s, n])
        print(f"[{si+1}/{len(ids)}] {sid}: {len(d)} balanced classes", flush=True)
    recs = np.array(recs, np.float32)
    subj = recs[:, 0].astype(int); lab = recs[:, 1].astype(int)
    cen = recs[:, 2:5]; wstd = recs[:, 5:8]
    n_subj = subj.max() + 1

    # Per-class between/within spread, grouped by category.
    rows = []
    for l in np.unique(lab):
        m = lab == l
        if m.sum() < args.min_subjects:
            continue
        between = np.linalg.norm(cen[m].std(0))
        within = np.linalg.norm(wstd[m].mean(0))
        rows.append((idx2name[l], category_map_ct.get(idx2name[l], "?"),
                     int(m.sum()), between, within, between / (within + 1e-6)))

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r[1]].append(r)
    print(f"\n{'class':26s} {'cat':28s} {'n':>3} {'between':>8} {'within':>7} {'ratio':>6}")
    for cat in sorted(by_cat):
        for name_, _, ns, b, w, rt in sorted(by_cat[cat], key=lambda x: x[5]):
            flag = "  tight" if rt < 1 else ""
            print(f"{name_:26s} {cat:28s} {ns:3d} {b:8.1f} {w:7.1f} {rt:6.2f}{flag}")
        cr = [x[5] for x in by_cat[cat]]
        print(f"{'  -> '+cat+' median ratio':56s} {np.median(cr):6.2f}\n")

    all_ratio = [r[5] for r in rows]
    print(f"OVERALL: {len(rows)} classes  median between/within ratio = {np.median(all_ratio):.2f} "
          f"| {sum(r<1 for r in all_ratio)}/{len(rows)} tight (<1)")

    # LOO nearest-centroid class retrieval.
    top1 = top5 = tot = 0
    for s in range(n_subj):
        tr, te = subj != s, subj == s
        if te.sum() == 0:
            continue
        book_l, book_c = [], []
        for l in np.unique(lab[tr]):
            book_l.append(l); book_c.append(cen[tr & (lab == l)].mean(0))
        book_l, book_c = np.array(book_l), np.stack(book_c)
        for c, gt in zip(cen[te], lab[te]):
            order = book_l[np.argsort(np.linalg.norm(book_c - c, axis=1))]
            top1 += order[0] == gt; top5 += gt in order[:5]; tot += 1
    print(f"\nLOO nearest-centroid class retrieval over {tot} (subj,class) queries:")
    print(f"  top-1 = {top1/tot:.3f}   top-5 = {top5/tot:.3f}  (chance ~{1/len(rows):.3f})")


if __name__ == "__main__":
    main()

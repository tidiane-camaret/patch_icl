"""Convert the extra TotalSegmentator `more_labels` masks to .npy + build a global index.

The extra labels (experiments/totalseg_more_labels) are *multilabel* files: each
`segmentations/{task}.nii.gz` holds several classes (one local id per voxel) and
different tasks overlap heavily (~85% of fg voxels are covered by 2+ tasks). To keep
every class losslessly we DO NOT flatten into one volume — each task stays its own
array, and a global index maps global_id <-> (task, local_id, name).

Per subject written (Approach A):
  more_labels/{task}.npy              — uint8, native res, canonical orientation
  more_labels/{task}_DxHxW.npy        — uint8, iso-resized (nearest), only with --size
                                        (same _iso_resize as convert_to_npy → aligns
                                         with ct_DxHxW.npy / label_DxHxW.npy)

At the data root:
  more_labels_classes.json            — global index: every (task, local_id) pair for
                                        the produced tasks, all 329 names / 362 entries
  more_labels_subject_classes.json    — {subject: [global_id, ...]} present (>0 voxels),
                                        so eval never picks a class a subject lacks

Usage
-----
  python experiments/totalseg_more_labels/convert_more_labels.py [--size 64 64 64] \
         [--workers N] [--overwrite]
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.convert_to_npy import _iso_resize  # identical resize → aligns with label_DxHxW
from totalsegmentator.map_to_binary import class_map

DATA = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/totalseg_test_more_labels")

# Auxiliary label-merge pseudo-tasks (not standalone models) and tasks whose weights
# could not be fetched — none of these were produced, but guard against stray files.
EXCLUDE_TASKS = {
    "kidney_cysts_auxiliary", "appendicular_bones_auxiliary",
    "renal_arteries_auxiliary", "face_mr_auxiliary",
    "total_highres_test", "covid",
}


def _produced_tasks(data: Path) -> list[str]:
    """Union of task names that exist as a mask file in any subject, minus EXCLUDE_TASKS."""
    tasks: set[str] = set()
    for subj in data.iterdir():
        seg = subj / "segmentations"
        if not seg.is_dir():
            continue
        for f in seg.glob("*.nii.gz"):
            tasks.add(f.name[:-len(".nii.gz")])
    return sorted(t for t in (tasks - EXCLUDE_TASKS) if t in class_map)


def _build_index(tasks: list[str]) -> tuple[list[dict], dict[tuple[str, int], int]]:
    """Global index over every (task, local_id) pair, ordered by (task, local_id).

    Returns (classes, lookup) where classes is the JSON list and lookup maps
    (task, local_id) -> global_id (1-based).
    """
    classes: list[dict] = []
    lookup: dict[tuple[str, int], int] = {}
    gid = 0
    for task in tasks:
        for local_id in sorted(class_map[task]):        # int keys, 1-based
            gid += 1
            name = class_map[task][local_id]
            classes.append({"global_id": gid, "task": task,
                            "local_id": int(local_id), "name": name})
            lookup[(task, int(local_id))] = gid
    return classes, lookup


def convert_subject(args: tuple) -> tuple[str, str, dict | None]:
    """Convert every produced task mask for one subject.

    Returns (subject, status, {task: [present_local_ids]}); present ids have >0 voxels.
    """
    subj_dir, tasks, overwrite, size = args
    subj_dir = Path(subj_dir)
    subj = subj_dir.name
    seg_dir = subj_dir / "segmentations"
    out_dir = subj_dir / "more_labels"
    size_str = f"{size[0]}x{size[1]}x{size[2]}" if size else None

    try:
        out_dir.mkdir(exist_ok=True)
        present: dict[str, list[int]] = {}
        for task in tasks:
            src = seg_dir / f"{task}.nii.gz"
            if not src.exists():
                continue                                  # task not gated for this subject
            npy_native = out_dir / f"{task}.npy"
            npy_sized  = out_dir / f"{task}_{size_str}.npy" if size_str else None

            need_native = overwrite or not npy_native.exists()
            need_sized  = size_str is not None and (overwrite or not npy_sized.exists())

            arr = spacing = None
            if need_native or need_sized:
                img = nib.as_closest_canonical(nib.load(str(src)))
                arr = np.asanyarray(img.dataobj).astype(np.uint8)
                spacing = tuple(float(x) for x in nib.affines.voxel_sizes(img.affine)[:3])

            if need_native:
                np.save(npy_native, arr)
            if need_sized:
                np.save(npy_sized, _iso_resize(arr, size, order=0, aa=False, spacing=spacing))

            # Presence recorded from native (nearest-resize can only drop, never add, ids)
            ref = arr if arr is not None else np.load(npy_native, mmap_mode="r")
            ids = np.unique(ref)
            present[task] = [int(i) for i in ids if i > 0]
        return subj, "ok", present
    except Exception:
        return subj, traceback.format_exc(), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA), help="totalseg_test_more_labels root")
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count()))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--size", nargs=3, type=int, metavar=("D", "H", "W"), default=None,
                        help="also write iso-resized more_labels/{task}_DxHxW.npy")
    args = parser.parse_args()

    data = Path(args.data)
    size = tuple(args.size) if args.size else None
    tasks = _produced_tasks(data)
    classes, lookup = _build_index(tasks)
    n_names = len({c["name"] for c in classes})
    print(f"{len(tasks)} tasks | {len(classes)} global classes ({n_names} unique names)"
          f" | size={size or 'native only'} | workers={args.workers}")

    # Global index — written up front, independent of per-voxel presence (all classes).
    idx_path = data / "more_labels_classes.json"
    with open(idx_path, "w") as f:
        json.dump({"version": 1, "n_classes": len(classes), "tasks": tasks,
                   "classes": classes}, f, indent=2)
    print(f"Wrote {idx_path.name}")

    subjects = sorted(p for p in data.iterdir()
                      if p.is_dir() and (p / "segmentations").is_dir())
    tasks_arg = [(str(s), tasks, args.overwrite, size) for s in subjects]
    total = len(subjects)

    subject_classes: dict[str, list[int]] = {}
    done = ok = errors = 0
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for subj, status, present in pool.imap_unordered(convert_subject, tasks_arg, chunksize=1):
            done += 1
            if status == "ok":
                ok += 1
                gids = sorted(lookup[(t, lid)]
                              for t, ids in present.items() for lid in ids)
                subject_classes[subj] = gids
            else:
                errors += 1
                print(f"\n[ERROR] {subj}:\n{status}")
            rate = done / (time.time() - t0)
            print(f"\r  {done}/{total}  ok={ok}  err={errors}  {rate:.1f} subj/s", end="", flush=True)

    sc_path = data / "more_labels_subject_classes.json"
    with open(sc_path, "w") as f:
        json.dump(subject_classes, f, indent=2)
    print(f"\nWrote {sc_path.name} ({len(subject_classes)} subjects)")
    print(f"Done in {(time.time()-t0)/60:.1f} min — ok={ok} errors={errors}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()

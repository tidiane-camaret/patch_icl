"""
Build a 2D cross-section manifest from TotalSegmentator for the 2D experiments.

Goal: pick ONE axial cross-section per subject that is densest in the *most common*
classes across the whole dataset, then emit one row per (subject, class) present in
that slice. Downstream, each row is a binary in-context segmentation task.

Selection logic
---------------
- Axial axis = axis 2 of the pre-resized `label_{S}x{S}x{S}.npy` volumes. This is the
  true cross-sectional plane: a single axis-2 slice cuts through <=2 vertebrae, whereas
  axes 0/1 span 7-18 vertebrae levels (i.e. they are coronal/sagittal). Verified on the
  data; axis-2 slices are also the organ-richest.
- Global class weight = occurrences / max_occurrences, read from `label_stats.csv`
  (so spinal_cord/aorta/liver/... weigh ~1, rare classes weigh ~0).
- Per subject, score each axis-2 slice z with a soft area-ramp:
      score(z) = sum over classes present (area >= --noise_floor) of
                 weight[class] * min(1, area / --area_cap)
  and take argmax_z. The ramp rewards classes that are *substantially* present (a class
  contributes its full weight once it exceeds area_cap, partial below), so the score
  is not dominated by a slice that merely grazes many organs by a few pixels. The
  noise_floor drops single-voxel speckle. This reliably lands on a thoraco-abdominal
  cross-section dense in the most common organs (liver, spleen, kidneys, pancreas,
  stomach, aorta, IVC, ...), and is robust: the top slices form one tight cluster rather
  than flipping between distant regions (which a hard threshold does near ties).
- Task emission is decoupled from selection: a (subject, class) task row is emitted for
  every class in the chosen slice with area >= --task_min_area, so tiny slivers (which
  make degenerate segmentation targets) are dropped without affecting which slice wins.

Output: a long-format CSV, one row per (subject, chosen-slice, class):
    subject_id, split, size, z, depth, slice_score, n_classes_in_slice,
    class_name, class_idx, area, area_frac

Usage
-----
python scripts/build_totalseg_2d_manifest.py                       # all subjects, size 128
python scripts/build_totalseg_2d_manifest.py --task_min_area 50 --workers 16
python scripts/build_totalseg_2d_manifest.py --size 64 --out results/totalseg2d/manifest_64.csv
"""

import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from data.totalseg_classes import ALL_CLASSES  # noqa: E402

AXIAL_AXIS = 2  # verified: axis-2 is the cross-sectional (axial) plane


# ── Root / split / weight helpers ─────────────────────────────────────────────

def resolve_totalseg_root(arg_root: str | None) -> str:
    """Find the totalseg root from --totalseg_root or the cluster config files."""
    if arg_root:
        return arg_root
    pattern = r"(?<!\w)totalseg\s*:\s*(.+)"
    for cfg_path in [ROOT / "configs/cluster/nfs.yaml", ROOT / "configs/cluster/meta.yaml"]:
        if cfg_path.exists():
            m = re.search(pattern, cfg_path.read_text())
            if m:
                root = m.group(1).strip()
                if Path(root).is_dir():
                    return root
    raise RuntimeError("Cannot auto-detect totalseg root. Pass --totalseg_root explicitly.")


def load_splits(root: Path) -> dict[str, str]:
    """subject_id -> split ('train'/'val'/'test') from meta.csv (semicolon-delimited)."""
    splits: dict[str, str] = {}
    with open(root / "meta.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            splits[row["image_id"]] = row["split"]
    return splits


def load_weights(root: Path) -> np.ndarray:
    """Global class weight by label index (1..117): occurrences / max_occurrences."""
    name2idx = {n: i + 1 for i, n in enumerate(ALL_CLASSES)}
    occ: dict[int, int] = {}
    with open(root / "label_stats.csv") as f:
        for row in csv.DictReader(f):
            if row["label_id"] in name2idx:
                occ[name2idx[row["label_id"]]] = int(row["occurrences"])
    w = np.zeros(len(ALL_CLASSES) + 1, dtype=np.float64)
    mx = max(occ.values())
    for idx, c in occ.items():
        w[idx] = c / mx
    return w


# ── Per-subject selection ─────────────────────────────────────────────────────

def select_slice(label_path: Path, weight: np.ndarray, noise_floor: int,
                 area_cap: int, task_min_area: int):
    """Return (z, depth, score, [(class_idx, area), ...]) for the best axial slice.

    Selection score uses a soft area ramp (see module docstring). The returned class
    list is filtered to task_min_area (what to emit), independent of the score.
    """
    lab = np.load(label_path)
    depth = lab.shape[AXIAL_AXIS]
    best = (-1.0, 0)  # (score, z)
    for z in range(depth):
        sl = np.take(lab, z, axis=AXIAL_AXIS)
        vals, cnts = np.unique(sl, return_counts=True)
        score = float(sum(weight[int(v)] * min(1.0, c / area_cap)
                          for v, c in zip(vals, cnts) if v != 0 and c >= noise_floor))
        if score > best[0]:
            best = (score, z)
    score, z = best
    sl = np.take(lab, z, axis=AXIAL_AXIS)
    vals, cnts = np.unique(sl, return_counts=True)
    present = [(int(v), int(c)) for v, c in zip(vals, cnts) if v != 0 and c >= task_min_area]
    return z, depth, score, present


def _worker(args):
    sid, label_path, weight, noise_floor, area_cap, task_min_area = args
    try:
        z, depth, score, present = select_slice(
            Path(label_path), weight, noise_floor, area_cap, task_min_area)
        return sid, z, depth, score, present, None
    except Exception as e:  # noqa: BLE001
        return sid, None, None, None, None, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--totalseg_root", default=None, help="Dataset root (auto-detected if absent)")
    p.add_argument("--size", type=int, default=128, help="Pre-resized volume size (uses label_{S}x{S}x{S}.npy)")
    p.add_argument("--noise_floor", type=int, default=10,
                   help="Min pixels for a class to enter the selection score (drops speckle)")
    p.add_argument("--area_cap", type=int, default=100,
                   help="Area (px) at which a class contributes its full weight to the score")
    p.add_argument("--task_min_area", type=int, default=25,
                   help="Min pixels for a class in the chosen slice to be emitted as a task")
    p.add_argument("--splits", nargs="*", default=None,
                   help="Only include these subject splits (e.g. test val). Default: all")
    p.add_argument("--max_subjects", type=int, default=None, help="Cap subjects (debug)")
    p.add_argument("--workers", type=int, default=8, help="Process pool size")
    p.add_argument("--out", default=None, help="Output CSV (default results/totalseg2d/manifest_{size}.csv)")
    args = p.parse_args()

    root = Path(resolve_totalseg_root(args.totalseg_root))
    weight = load_weights(root)
    splits = load_splits(root)
    label_name = f"label_{args.size}x{args.size}x{args.size}.npy"

    # Subjects = directories with the pre-resized label present.
    subjects = []
    for d in sorted(root.glob("s*")):
        if not d.is_dir():
            continue
        sid = d.name
        split = splits.get(sid, "unknown")
        if args.splits and split not in args.splits:
            continue
        if (d / label_name).exists():
            subjects.append((sid, split, d / label_name))
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]

    print(f"Root      : {root}")
    print(f"Label file: {label_name}   axial_axis={AXIAL_AXIS}")
    print(f"Scoring   : noise_floor={args.noise_floor}  area_cap={args.area_cap}  "
          f"task_min_area={args.task_min_area}")
    print(f"Subjects  : {len(subjects)}"
          + (f"  (splits={args.splits})" if args.splits else ""))

    out_path = Path(args.out) if args.out else ROOT / f"results/totalseg2d/manifest_{args.size}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [(sid, str(path), weight, args.noise_floor, args.area_cap, args.task_min_area)
             for sid, _, path in subjects]
    split_of = {sid: split for sid, split, _ in subjects}

    rows = []
    class_subject_count = np.zeros(len(ALL_CLASSES) + 1, dtype=int)
    n_done = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            sid, z, depth, score, present, err = fut.result()
            if err is not None:
                n_err += 1
                print(f"  [skip] {sid}: {err}")
                continue
            n_done += 1
            for cidx, area in present:
                class_subject_count[cidx] += 1
                rows.append((
                    sid, split_of[sid], args.size, z, depth, round(score, 4), len(present),
                    ALL_CLASSES[cidx - 1], cidx, area, round(area / (args.size * args.size), 5),
                ))
            if n_done % 200 == 0:
                print(f"  ...{n_done}/{len(tasks)} subjects")

    rows.sort(key=lambda r: (r[0], -r[9]))  # by subject, then descending area
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "split", "size", "z", "depth", "slice_score",
                    "n_classes_in_slice", "class_name", "class_idx", "area", "area_frac"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} (subject,class) rows for {n_done} subjects "
          f"({n_err} errors) -> {out_path}")

    # Coverage summary: how many subjects contribute a task for each class (top 30).
    order = np.argsort(-class_subject_count)
    print(f"\nTop classes by #subjects contributing a task (of {n_done}):")
    for cidx in order[:30]:
        if class_subject_count[cidx] == 0:
            break
        print(f"  {ALL_CLASSES[cidx - 1]:<32s} {class_subject_count[cidx]:>5d}")
    n_covered = int((class_subject_count > 0).sum())
    print(f"\nClasses with >=1 task: {n_covered}/{len(ALL_CLASSES)}")


if __name__ == "__main__":
    main()

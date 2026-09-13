"""Inspect the GNC_705 kidney-lesion cohort and select a smaller, balanced, label-rich subset.

GNC_705 (`/nfs/data/nii/data0/GNC/GNC_705/`) is a local restricted-access National Cohort MRI
release, NOT a public download -- 30,385 subjects x up to 2 visits (~49,220 total (subject,
visit) cases). Each visit has a 4-channel Dixon MRI (opp/in/fat/water composed into one 4D
.nii under `links/{subj}/{visit}/3D_GRE_TRA_4/*_W_COMPOSED_*.nii`) and, for a small fraction of
visits, hand-drawn kidney-lesion ROI masks directly under `data/{subj}/{visit}/*.nii.gz`.

Key findings this script surfaces (see docs/datasets/gnc_kidney_lesions.md for the full writeup):
- Only 610/49,220 visits (1.24%) carry ANY of the 14 canonical lesion classes -- the labeled
  cohort is already small; no labeled subject has labels at more than one visit.
- Label files are TIGHT ROI CROPS, not full-volume masks: same voxel spacing/orientation as the
  full image but a translated origin. The origin difference is an exact integer voxel offset
  (verified via affine diff), so a converter can composite by array placement -- no resampling.
- 100% of labeled (subject, visit) pairs have a matching COMPOSED 4-channel image file.
- Class names are NOT a clean nesting hierarchy: when e.g. `Hyper_R` and `mask_hyper.cyst.R`
  share the identical crop bounding box, the `mask_*` one is a voxel subset of the same lesion
  instance (a cystic component); but when the crop boxes differ entirely (as they do for
  `Hypo_R` vs `mask_hypo_R`, or `Complex_L` vs `mask_complex.cyst.L`), they are spatially
  distinct -- i.e. two different lesion instances in the same kidney, not two annotations of one
  lesion. Any per-class in-context task design should treat each of the 14 canonical names as an
  independent lesion-instance class, not assume a subregion relationship.

Usage:
    python scripts/inspect_gnc_kidney.py --scan                 # rebuild the file-presence census (~5 min, NFS-latency-bound)
    python scripts/inspect_gnc_kidney.py --analyze               # print the census summary (needs --scan output)
    python scripts/inspect_gnc_kidney.py --subset --cap 60       # write the balanced subset CSV
"""
import argparse
import csv
import glob
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/nfs/data/nii/data0/GNC/GNC_705"
DATA_ROOT = os.path.join(ROOT, "data")
LINKS_ROOT = os.path.join(ROOT, "links")

CACHE_DIR = "results/3d/gnc_kidney"
SCAN_JSON = os.path.join(CACHE_DIR, "scan_result.json")
SUBSET_CSV = "docs/datasets/gnc_kidney_lesions_subset.csv"

CANONICAL = [
    "Hyper_R.nii.gz", "Hyper_L.nii.gz", "Hypo_R.nii.gz", "Hypo_L.nii.gz",
    "Complex_R.nii.gz", "Complex_L.nii.gz",
    "mask_hyper.cyst.R.nii.gz", "mask_hyper.cyst.L.nii.gz",
    "mask_hyper.R.nii.gz", "mask_hyper.L.nii.gz",
    "mask_hypo_R.nii.gz", "mask_hypo_L.nii.gz",
    "mask_complex.cyst.R.nii.gz", "mask_complex.cyst.L.nii.gz",
]


def _scan_subject(subj):
    """List (not read) label files for one subject -- NFS metadata calls are latency-bound,
    so this is safe to fan out across a large thread pool (see main())."""
    subj_dir = os.path.join(DATA_ROOT, subj)
    out = {}
    try:
        visits = os.listdir(subj_dir)
    except OSError:
        return subj, out
    for v in visits:
        vdir = os.path.join(subj_dir, v)
        try:
            files = os.listdir(vdir)
        except OSError:
            continue
        present = sorted(set(files) & set(CANONICAL))
        all_niigz = [f for f in files if f.endswith(".nii.gz") or f.endswith(".nii")]
        out[v] = {"labels": present, "all_niigz": all_niigz}
    return subj, out


def do_scan(workers=128):
    os.makedirs(CACHE_DIR, exist_ok=True)
    subjects = sorted(os.listdir(DATA_ROOT))
    print(f"n_subjects={len(subjects)}", flush=True)
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_subject, s): s for s in subjects}
        for i, fut in enumerate(as_completed(futs), 1):
            subj, out = fut.result()
            results[subj] = out
            if i % 5000 == 0:
                print(f"{i}/{len(subjects)} in {time.time()-t0:.1f}s", flush=True)
    with open(SCAN_JSON, "w") as f:
        json.dump(results, f)
    print(f"DONE {len(results)} subjects in {time.time()-t0:.1f}s -> {SCAN_JSON}")


def _load_scan():
    with open(SCAN_JSON) as f:
        return json.load(f)


def do_analyze():
    data = _load_scan()
    n_subjects = len(data)
    n_visits = sum(len(v) for v in data.values())
    print(f"n_subjects={n_subjects} n_visits={n_visits}")

    class_visit_count = Counter()
    visits_with_any_label = 0
    for subj, visits in data.items():
        for vk, info in visits.items():
            present = set(info["labels"])
            if present:
                visits_with_any_label += 1
            for c in present:
                class_visit_count[c] += 1
    print(f"visits with >=1 canonical label: {visits_with_any_label} / {n_visits}")
    print("\nper-class presence (visits == subjects; no subject has labels at 2 visits):")
    for c in CANONICAL:
        print(f"  {c:32s} {class_visit_count[c]:4d}")


def _subj_classes(data):
    out = {}
    for subj, visits in data.items():
        classes = set()
        for vk, info in visits.items():
            classes |= set(info["labels"])
        if classes:
            out[subj] = sorted(classes)
    return out


def do_subset(cap=60):
    data = _load_scan()
    subj_classes = _subj_classes(data)
    global_count = Counter()
    for cs in subj_classes.values():
        for c in cs:
            global_count[c] += 1

    def score(cs):
        return sum(1.0 / global_count[c] for c in cs)

    ordered = sorted(subj_classes.items(), key=lambda kv: -score(kv[1]))
    selected = []
    running = Counter()
    for s, cs in ordered:
        if any(running[c] < min(cap, global_count[c]) for c in cs):
            selected.append(s)
            for c in cs:
                running[c] += 1

    print(f"CAP={cap}: selected {len(selected)} / {len(subj_classes)} labeled subjects "
          f"(of {len(data)} total subjects)")
    print("\npost-selection per-class subject counts (selected / total-available):")
    for c in sorted(global_count, key=lambda c: global_count[c]):
        print(f"  {c:32s} {running[c]:4d} / {global_count[c]:4d}")

    rows = []
    for s in selected:
        for v, info in data[s].items():
            if info["labels"]:
                rows.append((s, v, ";".join(info["labels"])))
    os.makedirs(os.path.dirname(SUBSET_CSV), exist_ok=True)
    with open(SUBSET_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "visit", "classes"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {SUBSET_CSV}")


def do_check_images():
    """Verify every labeled (subject, visit) has a matching COMPOSED 4-channel image."""
    data = _load_scan()
    subj_classes = _subj_classes(data)
    missing = []
    n_checked = 0
    for subj in subj_classes:
        for v, info in data[subj].items():
            if not info["labels"]:
                continue
            n_checked += 1
            hits = glob.glob(os.path.join(LINKS_ROOT, subj, v, "3D_GRE_TRA_*", "*COMPOSED*.nii"))
            if not hits:
                missing.append((subj, v))
    print(f"labeled (subject,visit) missing a COMPOSED image: {len(missing)} / {n_checked}")
    for m in missing[:20]:
        print("  missing:", m)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scan", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--subset", action="store_true")
    p.add_argument("--check-images", action="store_true")
    p.add_argument("--cap", type=int, default=60)
    p.add_argument("--workers", type=int, default=128)
    args = p.parse_args()

    if args.scan:
        do_scan(workers=args.workers)
    if args.analyze:
        do_analyze()
    if args.subset:
        do_subset(cap=args.cap)
    if args.check_images:
        do_check_images()

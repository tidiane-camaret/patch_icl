"""Prepare TotalSegmentator data for nnUNet v2."""
import argparse
import csv
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def _orthonormalize_affine(affine):
    """SVD-orthonormalize the rotation part while preserving voxel sizes."""
    R = affine[:3, :3]
    voxel_sizes = np.sqrt((R ** 2).sum(axis=0))
    U, _, Vt = np.linalg.svd(R / voxel_sizes)
    fixed = affine.copy()
    fixed[:3, :3] = (U @ Vt) * voxel_sizes
    return fixed


def prepare_subject(subject_id, totalseg_path, img_out_dir, lbl_out_dir, label_map):
    subj_path = totalseg_path / subject_id
    seg_dir = subj_path / "segmentations"

    img_dst = img_out_dir / f"{subject_id}_0000.nii.gz"
    lbl_dst = lbl_out_dir / f"{subject_id}.nii.gz"
    if img_dst.exists() and img_dst.stat().st_size > 0 and lbl_dst.exists() and lbl_dst.stat().st_size > 0:
        return

    ref = nib.load(subj_path / "ct.nii.gz")
    affine = _orthonormalize_affine(ref.affine)
    label_vol = np.zeros(ref.shape, dtype=np.uint8)

    for cls, idx in label_map.items():
        if cls == "background":
            continue
        seg_file = seg_dir / f"{cls}.nii.gz"
        if seg_file.exists():
            mask = nib.load(seg_file).get_fdata(dtype=np.float32) > 0.5
            label_vol[mask] = idx

    if not img_dst.exists():
        nib.save(nib.Nifti1Image(ref.get_fdata(dtype=np.float32), affine), img_dst)
    if not lbl_dst.exists():
        nib.save(nib.Nifti1Image(label_vol, affine), lbl_dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--totalseg", default="/work/dlclarge2/ndirt-SegFM3D/data/totalseg")
    parser.add_argument("--out", default="results/nnUNet/nnUNet_raw/Dataset001_totalseg_ct")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    totalseg_path = Path(args.totalseg)
    out_path = Path(args.out)

    for split in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (out_path / split).mkdir(parents=True, exist_ok=True)

    # class list from one subject — sorted for deterministic indices
    classes = sorted(
        p.stem.replace(".nii", "")
        for p in (totalseg_path / "s0000" / "segmentations").glob("*.nii.gz")
    )
    label_map = {"background": 0, **{cls: i + 1 for i, cls in enumerate(classes)}}
    print(f"{len(classes)} classes")

    split_map = {}
    with open(totalseg_path / "meta.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            split_map[row["image_id"]] = row["split"]

    all_subjects = sorted(p.name for p in totalseg_path.iterdir() if p.is_dir())
    train_subjects = [s for s in all_subjects if split_map.get(s, "train") in ("train", "val")]
    test_subjects  = [s for s in all_subjects if split_map.get(s, "train") == "test"]
    print(f"Train: {len(train_subjects)}  Test: {len(test_subjects)}")

    for subjects, img_dir, lbl_dir, desc in [
        (train_subjects, out_path / "imagesTr", out_path / "labelsTr", "train"),
        (test_subjects,  out_path / "imagesTs", out_path / "labelsTs", "test"),
    ]:
        fn = partial(prepare_subject,
                     totalseg_path=totalseg_path,
                     img_out_dir=img_dir,
                     lbl_out_dir=lbl_dir,
                     label_map=label_map)
        with Pool(args.workers) as pool:
            list(tqdm(pool.imap_unordered(fn, subjects), total=len(subjects), desc=desc))

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": label_map,
        "numTraining": len(train_subjects),
        "file_ending": ".nii.gz",
        "dataset_name": out_path.name,
    }
    with open(out_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print("Done — dataset.json written.")


if __name__ == "__main__":
    main()



"""
        
export nnUNet_raw="/work/dlclarge2/ndirt-SegFM3D/patch_icl/results/nnUNet/nnUNet_raw"                                                                           
export nnUNet_preprocessed="/work/dlclarge2/ndirt-SegFM3D/patch_icl/results/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/work/dlclarge2/ndirt-SegFM3D/patch_icl/results/nnUNet/nnUNet_results"

#naming convention 
mv $nnUNet_raw/totalseg_ct $nnUNet_raw/Dataset001_totalseg_ct

nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity

nnUNetv2_train 1 3d_fullres 0 -p nnUNetPlannerResEncM -num_gpus 2
"""
import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from totalsegmentator.map_to_binary import class_map

    # extract ct only
    ct_class_map = {
        task: classes 
        for task, classes in class_map.items() 
        if not task.startswith("mr_") and not task.endswith("_mr") and not task in ["total", "total_v1", "total_v3", "total_highres_test", "lung_vessels_LEGACY", "coronary_arteries_LEGACY"]
    }

    # every task and how many classes it has
    for task, classes in ct_class_map.items():
        print(f"{task:30s} {len(classes)} {list(classes.values())[:10]}")

    import json
    json.dump(ct_class_map, open("totalseg_ct_tasks.json", "w"), indent=2)
    return ct_class_map, json


@app.cell
def _():
    # downlaod all weights to .totalsegmentator/
    # totalseg_download_weights -t all
    return


@app.cell
def _():
    import nibabel as nib
    from pathlib import Path
    s0000_path = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg/s0000/ct.nii.gz"
    # plot middle slices
    img = nib.load(s0000_path)
    import matplotlib.pyplot as plt
    plt.imshow(img.get_fdata()[:, :, img.shape[2] // 2], cmap='gray')
    plt.show()
    return Path, nib, plt, s0000_path


@app.cell
def _(Path, ct_class_map, s0000_path):

    from totalsegmentator.python_api import totalsegmentator

    labels_path = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg/s0000/more_labels"
    ct_tasks = list(ct_class_map.keys()) 
    ct_imgs= [s0000_path]  # Add more paths to CT images as needed
    for ta in ct_tasks:                         # task-outer: model loaded once per task
        for img_ in ct_imgs:                     # image-inner
            print(f"Processing {img_} with task {ta}")
            if Path(labels_path)/(ta + ".nii") in Path(labels_path).iterdir():
                print(f"Output for task {ta} already exists. Skipping.")
                continue
            out = Path(labels_path)/ta
            try:
                totalsegmentator(img_, out, task=ta, ml=True, 
                             license_number="aca_630UZ7HI75J5MF")
            except Exception as e:
                print(f"Error processing {img_} with task {ta}: {e}")
    return ct_tasks, labels_path


@app.cell
def _(Path, ct_tasks, labels_path, nib, plt, s0000_path):
    # for each task, print nb of mask voxels and plot img + mask overlay on 3 axes at mask center
    for ta_ in ct_tasks:
        print(f"Task: {ta_}")

        mask_file = Path(labels_path) / (ta_ + ".nii")
        if not Path(mask_file).exists():
            print(f"Mask file {mask_file} does not exist. Skipping.")
            continue
        mask = nib.load(mask_file)
        mask_data = mask.get_fdata()
        nb_voxels = (mask_data > 0).sum()
        print(f"{ta_:30s} {mask_file.name:30s} {nb_voxels}")
        if nb_voxels > 0:
            # find center of mass of mask
            from scipy.ndimage import center_of_mass
            com = center_of_mass(mask_data)
            com = [int(c) for c in com]
            img_data = nib.load(s0000_path).get_fdata()
            # plot img + mask overlay on 3 axes at mask center
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_data[com[0], :, :], cmap='gray')
            axs[0].imshow(mask_data[com[0], :, :], cmap='jet', alpha=0.5)
            axs[1].imshow(img_data[:, com[1], :], cmap='gray')
            axs[1].imshow(mask_data[:, com[1], :], cmap='jet', alpha=0.5)
            axs[2].imshow(img_data[:, :, com[2]], cmap='gray')
            axs[2].imshow(mask_data[:, :, com[2]], cmap='jet', alpha=0.5)
            plt.savefig(f"totalseg_{ta_}_overlay.png")
    return


@app.cell
def _():
    return


@app.cell
def _(Path, json, nib, plt):
    data_root = Path(
            "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/totalseg_test_more_labels"
        )
    exp_dir = Path("/home/dpxuser/dev/patch_icl/experiments/totalseg_more_labels")
    subject_tasks = json.load(open(exp_dir / "totalseg_test_subject_tasks.json"))
    from scipy.ndimage import center_of_mass
    for subj, info in subject_tasks.items():
        if info.get("tasks") is None:
            continue
        ct_path = data_root / subj / "ct.nii.gz"
        seg_dir = data_root / subj / "segmentations"
        if not ct_path.exists() or not seg_dir.exists():
            continue
        img_data = nib.load(ct_path).get_fdata()     # load once per subject
        for ta_ in info["tasks"]:
            mask_file = seg_dir / f"{ta_}.nii.gz"     # single ml mask per task
            if not mask_file.exists():
                continue
            mask_data = nib.load(mask_file).get_fdata()
            nb_voxels = (mask_data > 0).sum()
            print(f"{subj} {ta_:30s} {nb_voxels}")
            if nb_voxels > 0:
                com = [int(c) for c in center_of_mass(mask_data > 0)]
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(img_data[com[0], :, :], cmap='gray')
                axs[0].imshow(mask_data[com[0], :, :], cmap='jet', alpha=0.5)
                axs[1].imshow(img_data[:, com[1], :], cmap='gray')
                axs[1].imshow(mask_data[:, com[1], :], cmap='jet', alpha=0.5)
                axs[2].imshow(img_data[:, :, com[2]], cmap='gray')
                axs[2].imshow(mask_data[:, :, com[2]], cmap='jet', alpha=0.5)
                fig.suptitle(f"{subj} / {ta_}")
                plt.show()
    return


@app.cell
def _():
    # === OPTIMIZED generation (use this instead of the plain loop above) =========
    # Hardware: 48 cores, 48 GB RAM, RTX 6000 Ada 48 GB.
    #   1. Crop-seg memoization (the real win). 23/45 tasks first run a rough 6mm
    #      `total` seg only to crop to their region; unpatched that reruns for every
    #      (task, subject). We memoize the intermediate (file_out=None) nnUNet_predict_
    #      image calls so each subject's rough seg is computed ONCE and reused by all
    #      its crop tasks.
    #   2. Drop the *_auxiliary entries: they are label-merge helpers, not standalone
    #      models (totalsegmentator(task=...) errors "Unknown task").
    # NOTE on resampling: TS's CPU resample_img only parallelizes over a 4th/time axis
    # (=1 for 3D CT), so nr_thr_resamp does nothing here. GPU resampling needs BOTH
    # cupy AND cucim (resampling.py gate) and TS itself notes the gain is marginal /
    # sometimes negative — so we leave threads at the default and don't rely on it.
    #
    # Everything lives inside `_run` (an underscore = marimo cell-local name) so this
    # cell exports NO globals and can't clash with the plain generation cell above
    # (marimo forbids defining the same variable in two cells).
    def _run():
        import json
        from pathlib import Path
        import totalsegmentator.nnunet as ts_nnunet          # patch target (see below)
        from totalsegmentator.python_api import totalsegmentator

        # Auxiliary label-merge pseudo-tasks that cannot be run on their own.
        drop_tasks = {
            "kidney_cysts_auxiliary", "appendicular_bones_auxiliary",
            "renal_arteries_auxiliary", "face_mr_auxiliary",
        }

        # --- install crop-seg memo (python_api does a local `from ...nnunet import
        #     nnUNet_predict_image` at call time, so patch the nnunet module attr) ---
        if not getattr(ts_nnunet.nnUNet_predict_image, "_crop_memoized", False):
            orig = ts_nnunet.nnUNet_predict_image
            crop_cache: dict = {}

            def memo(file_in, file_out, task_id, *args, **kwargs):
                # Only intermediate segs (crop / body / vertebrae-body) pass
                # file_out=None; the real task prediction passes the output path and
                # is never cached.
                if file_out is not None or not isinstance(file_in, (str, bytes)):
                    return orig(file_in, file_out, task_id, *args, **kwargs)
                key = (str(file_in), task_id, kwargs.get("model"),
                       str(kwargs.get("resample")), kwargs.get("trainer"),
                       tuple(kwargs.get("folds") or ()))
                if key not in crop_cache:
                    crop_cache[key] = orig(file_in, file_out, task_id, *args, **kwargs)
                return crop_cache[key]

            memo._crop_memoized = True
            memo._cache = crop_cache
            ts_nnunet.nnUNet_predict_image = memo
            print("Installed crop-seg memoization patch.")

        data_root = Path(
            "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/totalseg_test_more_labels"
        )
        exp_dir = Path("/home/dpxuser/dev/patch_icl/experiments/totalseg_more_labels")
        subject_tasks = json.load(open(exp_dir / "totalseg_test_subject_tasks.json"))

        # subject-outer so the memoized rough seg for a subject is reused immediately
        # by all its crop tasks (also works task-outer, but this keeps the cache tiny).
        for subj, info in subject_tasks.items():
            if info.get("tasks") is None:
                continue
            ct = data_root / subj / "ct.nii.gz"
            if not ct.exists():
                print(f"skip {subj}: no ct.nii.gz")
                continue
            for ta in info["tasks"]:
                if ta in drop_tasks:
                    continue
                out = data_root / subj / "segmentations" / f"{ta}.nii.gz"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                print(f"Processing {subj} with task {ta}")
                try:
                    totalsegmentator(str(ct), out, task=ta, ml=True, quiet=True,
                                     license_number="aca_630UZ7HI75J5MF")
                except Exception as e:
                    print(f"Error processing {subj} with task {ta}: {e}")

    _run()
    return


if __name__ == "__main__":
    app.run()

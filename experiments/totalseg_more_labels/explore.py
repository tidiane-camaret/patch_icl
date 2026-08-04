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
        if not task.startswith("mr_") and not task.endswith("_mr") and not task in ["total", "total_v1", "total_v3", "lung_vessels_LEGACY", "coronary_arteries_LEGACY"]
    }

    # every task and how many classes it has
    for task, classes in ct_class_map.items():
        print(f"{task:30s} {len(classes)} {list(classes.values())[:10]}")

    import json
    json.dump(ct_class_map, open("totalseg_ct_tasks.json", "w"), indent=2)
    return (ct_class_map,)


@app.cell
def _(ct_class_map):
    # Join all the task names with a space
    task_list = " ".join(ct_class_map.keys())

    # Format it into the bash command (fixing the missing semicolon before 'done')
    bash_command = f"for t in {task_list}; do totalseg_download_weights -t $t; done"

    print(bash_command)
    return


@app.cell
def _():
    import nibabel as nib
    s0000_path = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg/s0000/ct.nii.gz"
    # plot middle slices
    img = nib.load(s0000_path)
    import matplotlib.pyplot as plt
    plt.imshow(img.get_fdata()[:, :, img.shape[2] // 2], cmap='gray')
    plt.show()
    return (s0000_path,)


@app.cell
def _(ct_class_map, s0000_path):
    from pathlib import Path
    from totalsegmentator.python_api import totalsegmentator

    labels_path = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg/s0000/more_labels"
    ct_tasks = list(ct_class_map.keys()) 
    ct_imgs= [s0000_path]  # Add more paths to CT images as needed
    for ta in ct_tasks:                         # task-outer: model loaded once per task
        for img in ct_imgs:                     # image-inner
            out = Path(labels_path)/ta
            totalsegmentator(img, out, task=ta, ml=True, quiet=True,
                             license_number="aca_630UZ7HI75J5MF")
    return


if __name__ == "__main__":
    app.run()

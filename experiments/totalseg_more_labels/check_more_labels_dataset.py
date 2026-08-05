"""Sanity check for TotalSegMoreLabelsDataset (the plan's correctness gate).

Proves (a) the CT it loads from ct.nii.gz aligns pixel-for-pixel with the main
tree's ct_{size}.npy (so it aligns with the pre-resized more_labels masks), and
(b) its binary label equals (more_labels/{task}_{size}.npy == local_id).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.totalseg_more_labels_dataset import TotalSegMoreLabelsDataset

DATA = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data")
MORE = DATA / "totalseg_test_more_labels"
MAIN = DATA / "totalseg"
SIZE = (64, 64, 64)


def main():
    with open(MORE / "more_labels_classes.json") as f:
        idx = json.load(f)
    with open(MORE / "more_labels_subject_classes.json") as f:
        sc = json.load(f)
    gid_to = {int(c["global_id"]): c for c in idx["classes"]}

    # a class present in >=2 subjects (context-viable), on a subject that also
    # exists in the main tree (for the CT-alignment reference).
    cnt = Counter(g for v in sc.values() for g in v)
    gid = next(g for g, k in cnt.items()
               if k >= 2 and (MAIN / next(s for s, v in sc.items() if g in v)
                              / f"ct_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy").exists())
    c = gid_to[gid]
    key = f"{c['task']}/{c['name']}"
    subj = next(s for s, v in sc.items() if gid in v
                and (MAIN / s / f"ct_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy").exists())

    ds = TotalSegMoreLabelsDataset(root=MORE, classes=[key], image_size=SIZE, split="test")
    img, lbl = ds._load(subj, key)

    ref_ct = np.load(MAIN / subj / f"ct_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy").astype(np.float32)
    assert img.shape == (1, *SIZE), img.shape
    assert np.allclose(img[0].numpy(), ref_ct, atol=1e-2), \
        f"CT misaligned: max|diff|={np.abs(img[0].numpy()-ref_ct).max()}"

    task_arr = np.load(MORE / subj / "more_labels" / f"{c['task']}_{SIZE[0]}x{SIZE[1]}x{SIZE[2]}.npy")
    exp = (task_arr == c["local_id"]).astype(np.int64)
    assert np.array_equal(lbl.numpy(), exp), "label != (task_arr == local_id)"

    # end-to-end item: same keys as the base dataset, with a matching-class context.
    item = ds[0]
    for k in ("image", "label", "context_in", "context_out", "label_name", "subject"):
        assert k in item, k
    assert item["label_name"] == ds.samples[0][1]
    print(f"OK  subj={subj}  class={key}  fg={int(lbl.sum())}  "
          f"ctx={item['context_in'].shape}")


if __name__ == "__main__":
    main()

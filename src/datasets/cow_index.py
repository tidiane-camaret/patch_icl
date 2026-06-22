"""
Copy-on-write-safe index structures for the in-context datasets.

Problem: a forked DataLoader worker shares the parent's memory copy-on-write. COW
copies a page only on *write*, and the only writes during iteration are CPython
refcount bumps on PyObject headers. Large numpy image/label buffers are raw memory
that workers only read — already safe. But a `list` of millions of
`(ds_name, idx, label)` tuples, and a `dict` of lists of tuples for context lookup,
get their refcounts bumped on every access → the OS forks a page at a time → worker
RSS creeps up over the run (with persistent_workers it never resets), eventually
causing swap/page-cache stalls.

Fix: keep the per-sample triples and the per-(dataset, label) candidate lists as
contiguous int32 numpy arrays (one buffer, one refcount, read-only in workers). The
arrays carry no per-element PyObjects, so iterating them in a worker writes nothing.

`SampleIndex` still behaves like the old `list[(ds_name, idx, label)]` (len / index /
iterate) so common.TaggedDataset, collate and the eval scripts work unchanged.
"""

import random

import numpy as np


class SampleIndex:
    """A list-of-(ds_name, sample_idx, label_value) replacement backed by int arrays.

    Construct from three parallel sequences of ints plus the small list of dataset
    names. Indexing/iteration materializes a transient tuple (worker-local, GC'd),
    so external consumers are unchanged while the bulk storage stays COW-safe.
    """

    __slots__ = ("ds_ids", "img_idxs", "label_values", "ds_names")

    def __init__(self, ds_ids, img_idxs, label_values, ds_names):
        self.ds_ids = np.ascontiguousarray(ds_ids, dtype=np.int32)
        self.img_idxs = np.ascontiguousarray(img_idxs, dtype=np.int32)
        self.label_values = np.ascontiguousarray(label_values, dtype=np.int32)
        self.ds_names = list(ds_names)

    def __len__(self) -> int:
        return int(self.ds_ids.shape[0])

    def __getitem__(self, i: int):
        return (self.ds_names[int(self.ds_ids[i])],
                int(self.img_idxs[i]), int(self.label_values[i]))

    def __iter__(self):
        ids, ims, lvs, names = self.ds_ids, self.img_idxs, self.label_values, self.ds_names
        for i in range(ids.shape[0]):
            yield (names[int(ids[i])], int(ims[i]), int(lvs[i]))

    def subset(self, keep) -> "SampleIndex":
        """Return a new SampleIndex with only rows `keep` (used for eval subsampling)."""
        keep = np.asarray(keep, dtype=np.int64)
        return SampleIndex(self.ds_ids[keep], self.img_idxs[keep],
                           self.label_values[keep], self.ds_names)


def build_candidate_index(ds_ids, img_idxs, label_values):
    """Group image indices by (ds_id, label_value) into read-only int32 arrays.

    Returns dict[(ds_id, label_value)] -> np.ndarray(image_idxs). The dict has one
    key per (dataset, class) cell (hundreds at most), so it is small; the per-cell
    values are numpy arrays, so context lookup in a worker touches no per-element
    PyObjects. Mirrors the old `label_index` / `group_index` dicts.
    """
    from collections import defaultdict
    cells = defaultdict(list)
    for did, img, lv in zip(ds_ids, img_idxs, label_values):
        cells[(int(did), int(lv))].append(int(img))
    return {k: np.asarray(v, dtype=np.int32) for k, v in cells.items()}


def sample_context(cand: np.ndarray, exclude: int, k: int, rng=None) -> np.ndarray:
    """Pick k context image indices from `cand`, dropping the target's own `exclude`.

    Returns an int array (length k, or fewer/empty if no candidates). Picks
    positions (not objects), so seeding behaves like the old random.sample /
    random.choices path.

    `rng`: a `random.Random` instance for reproducible draws (e.g. seeded from the
    sample idx during deterministic eval). When None, uses the global `random`
    module — which PyTorch reseeds per worker, giving fresh, worker-distinct draws
    each epoch for training.
    """
    if exclude is not None and cand.shape[0]:
        cand = cand[cand != exclude]
    n = int(cand.shape[0])
    if n == 0:
        return cand[:0]
    r = rng if rng is not None else random
    pos = r.sample(range(n), k) if n >= k else r.choices(range(n), k=k)
    return cand[pos]

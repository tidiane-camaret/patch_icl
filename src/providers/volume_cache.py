"""Process-lifetime RAM cache of TotalSegmentator native volumes.

Holds ct_raw.npy (fp16) + label.npy (uint8) per subject as READ-ONLY numpy
arrays, preloaded once in the main process before the DataLoader forks its
workers, so every fork shares the buffers copy-on-write. Consumers must slice +
.contiguous() a small copy out and never mutate the cached arrays.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_CACHE: dict[str, dict[str, dict[str, np.ndarray]]] = {}   # str(root) -> {subject -> {"ct_raw","label"}}


def _load_one(root: Path, s: str, image_filename: str):
    cp, lp = root / s / image_filename, root / s / "label.npy"
    if not (cp.exists() and lp.exists()):
        return s, None
    ct = np.load(cp)                       # materialize into RAM (not mmap)
    lb = np.load(lp)
    ct.flags.writeable = False
    lb.flags.writeable = False
    return s, {"ct_raw": ct, "label": lb}


def get_cache(root, subjects, *, max_subjects=None, workers=16,
             image_filename: str = "ct_raw.npy") -> dict:
    """See plan Task 1 Interfaces. Idempotent per str(root); tops up missing subjects.

    `image_filename` names the raw-image file inside each subject dir ("ct_raw.npy" for
    CT sources; "mri_raw.npy" for the totalsegmri modality — see TotalSegProvider). The
    returned per-subject dict key stays "ct_raw" regardless (an internal cache-slot label,
    not a file name)."""
    key = str(root)
    root = Path(root)
    store = _CACHE.setdefault(key, {})
    want = list(dict.fromkeys(subjects))                    # de-dup, keep order
    if max_subjects is not None:
        want = want[: int(max_subjects)]
    todo = [s for s in want if s not in store]
    if max_subjects is not None:
        todo = todo[: max(0, int(max_subjects) - len(store))]
    if todo:
        t0, n, nbytes = time.perf_counter(), 0, 0
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            for s, payload in ex.map(lambda s: _load_one(root, s, image_filename), todo):
                if payload is not None:
                    store[s] = payload
                    n += 1
                    nbytes += payload["ct_raw"].nbytes + payload["label"].nbytes
        print(f"[volume_cache] loaded {n} subjects ({nbytes / 1e9:.1f} GB resident) "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)
    return store


def clear_cache() -> None:
    _CACHE.clear()

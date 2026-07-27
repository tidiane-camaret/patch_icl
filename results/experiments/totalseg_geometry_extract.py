"""
Geometric descriptors of real TotalSegmentator masks, for correlating shape/size
against segmentation performance.

Each (subject, class) mask is analysed at the 128³ whole-body resolution the
`use_crop=false` models actually saw (`label_128x128x128.npy`), cropped to its
bounding box for speed. Descriptors capture "how much solid bulk vs thin surface"
a structure has — the axis that dominates in-context Dice at low resolution.

Descriptors per mask
--------------------
  volume            fg voxel count (== logged tgt_size)
  n_components       26-connected components (fragmentation)
  largest_cc_frac    fraction of volume in the largest component
  bbox_fill          volume / bbox volume (how densely it fills its box)
  elongation         PCA std ratio s1/s2  (>1 elongated)
  flatness           PCA std ratio s2/s3  (>1 flat/sheet-like)
  linearity          (l1-l2)/l1  (~1 line/tube)
  planarity          (l2-l3)/l1  (~1 sheet)
  sphericity_pca     l3/l1  (~1 isotropic blob)
  thick_max/p90/mean interior EDT (voxels); thickness of the structure
  surface            exposed-face count (boundary area proxy)
  sphericity_iso     ideal-sphere-area / surface  (1 = sphere)
  surf_to_vol        surface / volume  (thinness; inverse of thickness)

Usage
-----
    from totalseg_geometry_extract import geometry_for_pairs, TOTALSEG_ROOT
    # pairs: DataFrame (or iterable) with columns ['subject', 'class']
    G = geometry_for_pairs(pairs)                 # -> DataFrame, one row per pair

CLI (writes a CSV):
    python totalseg_geometry_extract.py --pairs cases.csv --out geom.csv
    python totalseg_geometry_extract.py --classes liver,aorta --split val --out geom.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

# patch_icl src on path for the class→label-id map.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from src.totalseg_dataset import _ALL_CLASSES_IDX  # noqa: E402

# Default NFS root (cluster=nfs). Override via geometry_for_pairs(root=...) / --root.
TOTALSEG_ROOT = Path(
    "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
    "ANALYSIS_20251122/data/totalseg"
)

FEATURES = [
    "volume", "n_components", "largest_cc_frac", "bbox_fill",
    "elongation", "flatness", "linearity", "planarity", "sphericity_pca",
    "thick_max", "thick_p90", "thick_mean_fg", "surface", "sphericity_iso",
    "surf_to_vol",
]
# heavy-tailed features to log1p before correlation/plotting
LOG_FEATURES = ["volume", "surface", "thick_max", "thick_p90", "thick_mean_fg",
                "surf_to_vol", "n_components"]


def descriptors(m: np.ndarray) -> dict:
    """Geometric descriptors of a boolean 3D mask (already cropped to bbox+pad)."""
    V = int(m.sum())
    out = {"volume": V}
    if V == 0:
        return out

    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    out["n_components"] = int(n)
    sizes = np.bincount(lab.ravel())[1:]
    out["largest_cc_frac"] = float(sizes.max() / V)
    out["bbox_fill"] = float(V / m.size)

    coords = np.argwhere(m).astype(np.float64)
    coords -= coords.mean(0)
    if len(coords) >= 3:
        ev = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1]  # l1>=l2>=l3
        ev = np.clip(ev, 1e-9, None)
        l1, l2, l3 = ev
        s1, s2, s3 = np.sqrt(ev)
        out["elongation"] = float(s1 / s2)
        out["flatness"] = float(s2 / s3)
        out["linearity"] = float((l1 - l2) / l1)
        out["planarity"] = float((l2 - l3) / l1)
        out["sphericity_pca"] = float(l3 / l1)
    else:
        for k in ("elongation", "flatness", "linearity", "planarity", "sphericity_pca"):
            out[k] = np.nan

    edt = ndimage.distance_transform_edt(m)
    out["thick_max"] = float(edt.max())
    out["thick_p90"] = float(np.percentile(edt[m], 90))
    out["thick_mean_fg"] = float(edt[m].mean())

    pad = np.pad(m, 1)
    faces = sum(np.sum(pad ^ np.roll(pad, 1, axis=ax)) for ax in range(3))
    A = float(faces / 2)
    out["surface"] = A
    out["sphericity_iso"] = float((np.pi ** (1 / 3)) * ((6 * V) ** (2 / 3)) / A) if A > 0 else np.nan
    out["surf_to_vol"] = float(A / V)
    return out


def geometry_for_pairs(pairs, root: str | Path = TOTALSEG_ROOT,
                       size: int = 128) -> pd.DataFrame:
    """Descriptors for each (subject, class) pair, from label_{size}³.npy.

    `pairs`: DataFrame with 'subject'/'class' columns (or iterable of (subject, class)).
    Loads each subject's merged label volume once. Returns one row per pair
    (columns: subject, class, + FEATURES). Missing masks yield volume=0 rows.
    """
    root = Path(root)
    if isinstance(pairs, pd.DataFrame):
        pairs = pairs[["subject", "class"]].drop_duplicates()
    else:
        pairs = pd.DataFrame(list(pairs), columns=["subject", "class"]).drop_duplicates()
    by_subj = pairs.groupby("subject")["class"].apply(list).to_dict()

    rows = []
    fname = f"label_{size}x{size}x{size}.npy"
    for subj, classes in by_subj.items():
        full = np.load(root / subj / fname)   # uint8, all classes merged
        for cls in classes:
            idx = _ALL_CLASSES_IDX.get(cls)
            m = full == idx if idx is not None else np.zeros_like(full, dtype=bool)
            if m.sum() == 0:
                rows.append({"subject": subj, "class": cls, "volume": 0})
                continue
            nz = np.argwhere(m)
            lo, hi = nz.min(0), nz.max(0) + 1
            sl = tuple(slice(max(0, l - 1), h + 1) for l, h in zip(lo, hi))
            d = descriptors(m[sl])
            d["subject"], d["class"] = subj, cls
            rows.append(d)
    return pd.DataFrame(rows)


def load_or_build_geometry(pairs, cache, root: str | Path = TOTALSEG_ROOT,
                           size: int = 128) -> pd.DataFrame:
    """Return per-(subject,class) geometry, reading `cache` if it covers all `pairs`.

    Rebuilds via `geometry_for_pairs` (reading label_{size}³.npy on `root`) and rewrites
    the cache when it is missing or does not cover every requested pair. Shared by callers
    that want geometry without duplicating the cache dance (notebook cells 0 and 4).
    """
    if isinstance(pairs, pd.DataFrame):
        pairs = pairs[["subject", "class"]].drop_duplicates()
    else:
        pairs = pd.DataFrame(list(pairs), columns=["subject", "class"]).drop_duplicates()
    cache = Path(cache)
    need = set(pairs.itertuples(index=False, name=None))
    if cache.exists():
        G = pd.read_csv(cache)
        if need <= set(zip(G.subject, G["class"])):
            return G
    G = geometry_for_pairs(pairs, root=root, size=size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    G.to_csv(cache, index=False)
    return G


# ── data-driven morphology families ──────────────────────────────────────────
# Scale-invariant shape descriptors + thickness + fragmentation, clustered (Ward) into
# `k` families. Auto-labelled `{thickness-tercile}_{blob|tube|sheet}` (with a `frag`
# override), so the taxonomy generalises to any class set / resolution instead of a hand map.
SHAPE_FEATURES = ["linearity", "planarity", "sphericity_pca", "elongation", "flatness",
                  "surf_to_vol", "thick_p90", "bbox_fill", "largest_cc_frac", "n_components"]
_SHAPE_LOGF = ["thick_p90", "n_components"]
_WESTIN = {"linearity": "tube", "planarity": "sheet", "sphericity_pca": "blob"}


def shape_families(geom: pd.DataFrame, k: int = 10):
    """Cluster classes into `k` morphology families from per-class median geometry.

    Returns (class2shape: dict[class -> family], order: list[family] thick→thin). Families
    are auto-named `{thick|mid|thin}_{blob|tube|sheet}`: the thickness tercile comes from
    surf_to_vol (low = thick); the shape tag is the Westin coord (linearity/planarity/
    sphericity) the cluster stands out on *relative to other clusters* (z-score argmax, so
    it is not swamped by linearity being globally largest). Fragmented clusters (low
    largest_cc_frac or ≥3 components) are tagged `frag`. Collisions get a thickness-ordered
    suffix. Ward + numpy standardisation — no sklearn dependency.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    per = geom[geom.volume > 0].groupby("class").median(numeric_only=True)
    X = per[SHAPE_FEATURES].copy()
    for c in _SHAPE_LOGF:
        X[c] = np.log1p(X[c])
    X = X.fillna(X.median())
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    lab = fcluster(linkage(Z.values, method="ward"), k, criterion="maxclust")

    cen = per.assign(_cl=lab).groupby("_cl").median(numeric_only=True)
    tbin = pd.cut(cen.surf_to_vol.rank(method="first"), 3, labels=["thick", "mid", "thin"])
    w = cen[list(_WESTIN)]
    axis = ((w - w.mean()) / w.std(ddof=0).replace(0, 1)).idxmax(axis=1).map(_WESTIN)
    frag = (cen.largest_cc_frac < 0.7) | (cen.n_components >= 3)
    cen["name"] = np.where(frag, "frag", tbin.astype(str) + "_" + axis)
    for lbl, idxs in pd.Series(cen.index, index=cen["name"]).groupby(level=0):
        if len(idxs) > 1:  # disambiguate duplicate labels, thick→thin
            for j, cl in enumerate(cen.loc[idxs.values].surf_to_vol.sort_values().index, 1):
                cen.loc[cl, "name"] = f"{lbl}{j}"
    name = cen["name"].to_dict()
    class2shape = {c: name[l] for c, l in zip(per.index, lab)}
    order = cen.sort_values("surf_to_vol").name.tolist()
    return class2shape, order


def _subjects_for_split(root: Path, split: str) -> list[str]:
    meta = pd.read_csv(root / "meta.csv", sep=";")
    return sorted(meta.loc[meta.split.str.strip() == split, "image_id"].str.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", help="CSV with subject,class columns")
    ap.add_argument("--classes", help="comma-separated class list (with --split)")
    ap.add_argument("--split", default="val", help="split for --classes (default val)")
    ap.add_argument("--root", default=str(TOTALSEG_ROOT))
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)

    if a.pairs:
        pairs = pd.read_csv(a.pairs)[["subject", "class"]].drop_duplicates()
    elif a.classes:
        subjects = _subjects_for_split(root, a.split)
        classes = a.classes.split(",")
        pairs = pd.DataFrame([(s, c) for s in subjects for c in classes],
                             columns=["subject", "class"])
    else:
        ap.error("pass --pairs or --classes")

    G = geometry_for_pairs(pairs, root=root, size=a.size)
    G.to_csv(a.out, index=False)
    print(f"wrote {a.out}  {G.shape}")


if __name__ == "__main__":
    main()

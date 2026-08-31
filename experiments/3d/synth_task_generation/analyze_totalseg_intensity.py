"""
Real-data intensity distribution analysis for TotalSegmentator CT, per organ class --
step 1 towards replacing the independent per-class-slot uniform GMM draw
(src/gpu_gmm_intensity.py, mu_c ~ U(0,255) i.i.d.) with a joint model that captures the
real between-subject CORRELATION across classes (e.g. IV-contrast phase brightens
aorta/kidney/liver together; scanner/kVp shifts everything together) -- a la SynthSeg but
data-calibrated instead of independently randomized.

Stage 1 (extract): for every subject, per-class mean/var of RAW HU (ct_raw.npy) under
label.npy, BEFORE any normalization or intensity augmentation -- one np.bincount pass per
subject (weighted sum + sum-of-squares + voxel count). Cached to
results/synth_task_gen/totalseg_intensity_stats.npz so re-analysis is instant.

For totalseg CT this also adds a "body" pseudo-class (disable with --no_body): label.npy has
no class for the soft-tissue envelope, but MAISI's id 200 ("body" container fill) paints it
around every organ, so it had no real-HU calibration. Fed by each subject's pred_body.npy
(TotalSegmentator `body` task, produced by gen_body_masks.py, on the label.npy grid),
restricted to voxels inside the envelope but outside every one of the 117 labels. Cached
separately as totalseg_intensity_stats_body.npz.

Stage 2 (analyze): by default uses ALL classes with >= --min_n subjects present (i.e. the
whole 122-class ALL_CLASSES vocabulary, not just one part-set) -- pass --classes to restrict
to an explicit list (e.g. --classes ts_organs). TotalSegmentator subjects come from
disjoint scan protocols (chest-only, abdomen-only, angio, whole-body, ...), so no subject has
every class: correlation is estimated PAIRWISE-complete (each class pair uses whichever
subjects have BOTH present, min --min_pair_n), and PCA runs on that correlation matrix
(eigh) rather than SVD of a dense subject x class matrix -- there is no single dense matrix
to decompose once classes span disjoint subject pools. Reports:
  - per-class between-subject mean/std of the raw-HU class mean (the marginals the current
    code replaces with U(0,255) i.i.d.)
  - the cross-class correlation matrix (hierarchically clustered for the plot)
  - how much of the between-subject spread is a SHARED low-rank factor (e.g. one
    "brightness/contrast" axis) vs per-class-independent
  - a k-factor Gaussian fit (mean_c + std_c * loadings_c @ z, z ~ N(0,I_k)), per-class R^2 =
    communality = sum_k loadings[k,c]^2, saved to totalseg_intensity_factors.npz.
  - a DE-BIAS pass (on by default, --no_dedupe to skip): raw PCA above implicitly weights a
    body part by how many labels the vocabulary gives it (every vertebra/rib is its own class,
    so bone gets ~50 "votes" vs ~1 for a solitary organ). Classes are grouped by INTENSITY
    CORRELATION ALONE (avg-linkage clustering on 1-corr distance, --cluster_dist cutoff -- no
    class-name/anatomy lookup, so it generalizes to any label vocabulary), each group collapsed
    to one representative signal, PCA refit on that, then the count-unbiased loadings are
    broadcast back so every real class still gets a row -> totalseg_intensity_factors{tag}
    _debiased.npz (same shape as the raw file; this is the one to actually use downstream).

  .venv_blackwell/bin/python experiments/3d/synth_task_generation/analyze_totalseg_intensity.py
  .venv_blackwell/bin/python experiments/3d/synth_task_generation/analyze_totalseg_intensity.py --recompute --k_factors 3

Outputs:
  results/synth_task_gen/totalseg_intensity_stats.npz       (raw per-subject sum/sumsq/count)
  results/synth_task_gen/totalseg_intensity_factors.npz     (fitted k-factor model)
  results/synth_task_gen/totalseg_intensity_class_table.csv (per-class summary)
  results/synth_task_gen/totalseg_intensity_{corr,scree,loadings}.png
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.totalseg_classes import ALL_CLASSES  # noqa: E402

_DATA_ROOT = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                   "ANALYSIS_20251122/data")
# modality drives normalization: CT's raw HU is one absolute physical unit shared by every
# subject/scanner (comparable as-is); MRI has no such absolute unit -- ct_raw.npy is
# arbitrary per-scan gain, so it must be put in a comparable per-subject frame FIRST (the
# same clip+zscore normalize_mri() applies at train time, via each subject's ct_stats.json)
# before pooling across subjects means anything. label.npy uses the shared CT ALL_CLASSES
# encoding for both (totalseg_mri.yaml), so the rest of the pipeline is unchanged.
DATASETS = {
    "totalseg": dict(root=_DATA_ROOT / "totalseg", modality="ct"),
    "totalsegmri": dict(root=_DATA_ROOT / "totalsegmri", modality="mri"),
}
OUT = Path(__file__).resolve().parents[3] / "results" / "synth_task_gen"
N_BINS = len(ALL_CLASSES) + 1  # +1 for background id 0
# Optional extra pseudo-class "body" (--with_body, CT only): the soft-tissue envelope that
# ALL_CLASSES has no label for but MAISI's id 200 ("body" container fill) does -- fed by each
# subject's pred_body.npy (TotalSegmentator `body` task, on the label.npy grid) restricted to
# voxels INSIDE the envelope but OUTSIDE every one of the 117 labels (fat/muscle/skin/
# connective tissue -- exactly what id 200 paints around the organs). Stored as one extra
# column at index N_BINS in the stats arrays; class id/name = N_BINS / "body".
BODY_COL = N_BINS


def _subject_stats(args):
    """One np.bincount pass: (sum, sumsq, count) of intensity per label id, for one subject.
    CT: raw HU as-is. MRI: clip+zscore via that subject's own ct_stats.json entry first (no
    absolute unit otherwise) -- `stats` is None for CT, the subject's stats dict for MRI.

    With `with_body` (CT only) one extra column is appended (index BODY_COL) for the "body"
    pseudo-class: HU over pred_body.npy voxels that carry NO label (envelope minus every one
    of the 117 classes). 0/0/0 if pred_body.npy is missing or off-grid."""
    root, subj, stats, with_body = args
    d = Path(root) / subj
    try:
        ct = np.asarray(np.load(d / "ct_raw.npy", mmap_mode="r"), dtype=np.float64)
        lbl = np.asarray(np.load(d / "label.npy", mmap_mode="r")).ravel().astype(np.int64)
    except (FileNotFoundError, EOFError, ValueError, OSError):
        return subj, None
    if stats is not None:
        ct = np.clip(ct, stats["clip_lo"], stats["clip_hi"])
        ct = (ct - stats["mean"]) / stats["std"]
    ct = ct.ravel()
    s = np.bincount(lbl, weights=ct, minlength=N_BINS)[:N_BINS]
    ss = np.bincount(lbl, weights=ct * ct, minlength=N_BINS)[:N_BINS]
    c = np.bincount(lbl, minlength=N_BINS)[:N_BINS]
    if with_body:
        sb = ssb = cb = 0.0
        if stats is None:                            # CT only: pred_body is on the label grid
            try:
                body = np.load(d / "pred_body.npy", mmap_mode="r").ravel().astype(bool)
            except (FileNotFoundError, EOFError, ValueError, OSError):
                body = None
            if body is not None and body.shape == lbl.shape:
                v = ct[body & (lbl == 0)]            # envelope minus every labeled class
                sb, ssb, cb = float(v.sum()), float((v * v).sum()), float(v.size)
        s = np.append(s, sb); ss = np.append(ss, ssb); c = np.append(c, cb)
    return subj, (s, ss, c)


def extract(dataset="totalseg", recompute=False, workers=32, with_body=False):
    """Stage 1: subjects x (N_BINS [+1 if with_body]) sum/sumsq/count matrices, cached per
    dataset. `with_body` adds the pred_body.npy-fed "body" pseudo-class column and writes a
    distinct `_body`-tagged cache so it never collides with the plain run."""
    root, modality = DATASETS[dataset]["root"], DATASETS[dataset]["modality"]
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"{dataset}_intensity_stats{'_body' if with_body else ''}.npz"
    if cache.exists() and not recompute:
        z = np.load(cache, allow_pickle=True)
        print(f"extract[{dataset}]: loaded cache {cache} ({len(z['subjects'])} subjects)")
        return z["subjects"], z["sums"], z["sumsqs"], z["counts"]

    mri_stats = {}
    if modality == "mri":
        import json
        with open(root / "ct_stats.json") as f:
            mri_stats = json.load(f)

    subjects = sorted(p.name for p in root.iterdir()
                       if p.is_dir() and (p / "label.npy").exists()
                       and (p / "ct_raw.npy").exists()
                       and (modality != "mri" or p.name in mri_stats))
    nb = N_BINS + (1 if with_body else 0)
    print(f"extract[{dataset}]: {len(subjects)} subjects ({modality}), {workers} workers"
          f"{', +body pseudo-class' if with_body else ''}")
    sums = np.zeros((len(subjects), nb))
    sumsqs = np.zeros((len(subjects), nb))
    counts = np.zeros((len(subjects), nb))
    idx = {s: i for i, s in enumerate(subjects)}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_subject_stats, (root, s, mri_stats.get(s), with_body))
                for s in subjects]
        for fut in as_completed(futs):
            subj, res = fut.result()
            if res is not None:
                i = idx[subj]
                sums[i], sumsqs[i], counts[i] = res
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(subjects)}", flush=True)
    np.savez(cache, subjects=np.array(subjects), sums=sums, sumsqs=sumsqs, counts=counts)
    print(f"extract[{dataset}]: wrote {cache}")
    return np.array(subjects), sums, sumsqs, counts


def _pairwise_corr(M, names, min_pair_n):
    """Pairwise-complete Pearson correlation of M's columns (n_subj, n_names) -- each pair
    uses whichever rows have BOTH columns non-NaN; pairs with fewer than min_pair_n such rows
    are unreliable and zeroed (neutral, not spuriously +-1). Scale-invariant per column, so
    M can be raw HU or standardized -- same corr either way."""
    df = pd.DataFrame(M, columns=names)
    corr = df.corr(min_periods=min_pair_n).to_numpy()
    off = corr[~np.eye(len(corr), dtype=bool)]
    stats = dict(mean_abs_r=float(np.nanmean(np.abs(off))), max_r=float(np.nanmax(off)),
                 min_r=float(np.nanmin(off)), n_unreliable_pairs=int(np.isnan(off).sum()) // 2)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr, stats


def _fit_pca(corr, k_factors):
    """PCA via eigh on a correlation matrix (not SVD-of-data: pairwise corr has no single
    underlying dense data matrix once columns come from partly-disjoint subject pools, and
    trace(corr) == n regardless, so eigval/n is still a valid explained-variance-ratio; small
    negative eigenvalues from non-PSD pairwise estimation are clipped to 0).

    loadings[:,c] = eigvec scaled by sqrt(eigenvalue), so sum_k loadings[k,c]^2 = the fraction
    of class c's (standardized) variance the k factors explain (its "communality" / R^2)."""
    n = corr.shape[0]
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(-eigval)
    eigval, eigvec = eigval[order], eigvec[:, order]
    eigval_c = np.clip(eigval, 0, None)
    explained = eigval_c / n
    k = min(k_factors, n)
    loadings = (eigvec[:, :k] * np.sqrt(eigval_c[:k])).T                # (k, n)
    r2_per_class = np.clip((loadings ** 2).sum(0), 0, 1)
    resid_std = np.sqrt(1 - r2_per_class)
    return dict(explained=explained, eigvec=eigvec, loadings=loadings,
                resid_std=resid_std, r2_per_class=r2_per_class, k=k)


def _cluster_classes(corr, names, dist_thresh):
    """Group classes by INTENSITY CORRELATION ALONE (avg-linkage on 1-corr distance, cut at
    dist_thresh) -- no class name / anatomy lookup, so a body part with many near-duplicate
    labels (every vertebra, every rib) collapses to one cluster the same way it would for any
    other label vocabulary (e.g. MAISI ids). Returns {cluster_id: [class names]}."""
    d = np.clip((1 - corr + (1 - corr).T) / 2, 0, None)
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(d, checks=False), method="average")
    cid = fcluster(Z, t=dist_thresh, criterion="distance")
    groups = {}
    for c, name in zip(cid, names):
        groups.setdefault(int(c), []).append(name)
    return groups


def _dedup_signal(Zstd, names, groups):
    """Collapse each intensity-cluster to ONE representative per-subject signal (nanmean of
    its standardized members present in that subject) -- removes the "how many labels does
    this body part happen to have" vote-weight from the PCA input. Returns (dedup signal
    (n_subj, n_clusters), cluster label per column, and a (n_names,) column index mapping
    each ORIGINAL class to its cluster's column -- for broadcasting the debiased fit back)."""
    idx_of = {n: i for i, n in enumerate(names)}
    cluster_ids = sorted(groups)
    out = np.full((Zstd.shape[0], len(cluster_ids)), np.nan)
    labels = []
    class_cluster_col = np.zeros(len(names), dtype=int)
    for gi, cid in enumerate(cluster_ids):
        members = groups[cid]
        cols = [idx_of[m] for m in members]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN row (subject
            out[:, gi] = np.nanmean(Zstd[:, cols], axis=1)             # has none of this cluster)
        labels.append(members[0] if len(members) == 1 else f"{members[0]}+{len(members)-1}more")
        for m in members:
            class_cluster_col[idx_of[m]] = gi
    return out, np.array(labels), class_cluster_col


def class_mean_matrix(sums, sumsqs, counts, min_voxels=20):
    """(sum,sumsq,count) -> (mean_hu, voxel_var, present, frac_present, class_names), each
    (n_subj, n_classes) except class_names/frac_present (n_classes,). Shared by analyze() and
    analyze_merged() so both datasets build this identically. A trailing "body" column (arrays
    wider than N_BINS, from extract(with_body=True)) is picked up as one extra class."""
    has_body = sums.shape[1] > N_BINS
    class_ids = np.arange(1, N_BINS + (1 if has_body else 0))
    class_names = np.array(ALL_CLASSES + (["body"] if has_body else []))
    present = counts[:, class_ids] >= min_voxels                      # (n_subj, n_classes)
    frac_present = present.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_hu = sums[:, class_ids] / counts[:, class_ids]            # NaN where count==0
        voxel_var = sumsqs[:, class_ids] / counts[:, class_ids] - mean_hu ** 2
    mean_hu = np.where(present, mean_hu, np.nan)
    voxel_var = np.where(present, np.clip(voxel_var, 0, None), np.nan)
    return mean_hu, voxel_var, present, frac_present, class_names


def analyze(subjects, sums, sumsqs, counts, min_frac=0.75, min_voxels=20, k_factors=3,
            classes=None, tag="", min_n=30, min_pair_n=15, dedupe=True, cluster_dist=0.4,
            prefix="totalseg_intensity", unit="HU"):
    """Stage 2: per-class summary + correlation + PCA factor fit over 'core' classes."""
    n_subj = len(subjects)
    mean_hu, voxel_var, present, frac_present, class_names = class_mean_matrix(
        sums, sumsqs, counts, min_voxels)

    # per-class table (all classes, sorted by coverage)
    rows = []
    for j, name in enumerate(class_names):
        col = mean_hu[:, j]
        vcol = voxel_var[:, j]
        n_present = int(present[:, j].sum())
        rows.append((name, n_present, frac_present[j],
                     float(np.nanmean(col)) if n_present else np.nan,
                     float(np.nanstd(col)) if n_present else np.nan,
                     float(np.sqrt(np.nanmean(vcol))) if n_present else np.nan))
    rows.sort(key=lambda r: -r[1])
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"{prefix}_class_table{tag}.csv"
    with open(csv_path, "w") as f:
        f.write(f"class,n_present,frac_present,mean_{unit},between_subj_std_{unit},within_subj_voxel_std_{unit}\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f}\n")
    print(f"analyze: wrote {csv_path}")
    print(f"\n{'class':<28}{'n':>6}{'frac':>7}{'mean_'+unit:>10}{'std_'+unit+'(between)':>17}{'voxel_std(within)':>19}")
    for r in rows[:15]:
        print(f"{r[0]:<28}{r[1]:>6}{r[2]:>7.2f}{r[3]:>10.1f}{r[4]:>17.1f}{r[5]:>19.1f}")

    # --- core classes: an explicit list (e.g. ts_organs), else ALL classes with enough data -
    if classes:
        core = np.isin(class_names, classes)
        missing = set(classes) - set(class_names[core])
        if missing:
            print(f"analyze: WARNING classes not found: {sorted(missing)}")
    else:
        core = frac_present >= min_frac
    n_present = present.sum(0)
    dropped = core & (n_present < min_n)
    core = core & (n_present >= min_n)                                  # else corr is noise
    core_names = class_names[core]
    if dropped.any():
        print(f"analyze: dropping {dropped.sum()} classes with <{min_n} subjects present: "
              f"{sorted(class_names[dropped].tolist())}")
    print(f"\nanalyze: {core.sum()} core classes "
          f"(coverage {n_present[core].min()}-{n_present[core].max()} subjects each)")

    row_by_name = {r[0]: r for r in rows}
    mu = np.array([row_by_name[n][3] for n in core_names])              # between-subj mean HU
    sd = np.array([row_by_name[n][4] for n in core_names])              # between-subj std HU
    M_core = mean_hu[:, core]                                            # (n_subj, n_core) raw HU
    Zstd = (M_core - mu) / np.where(sd > 0, sd, 1.0)                     # standardized

    # --- pairwise-complete correlation (NOT a dense intersection: with 100+ classes spanning
    # disjoint TotalSegmentator annotation parts -- organs/vertebrae/cardiac/muscles/ribs --
    # requiring every class present in the SAME subject collapses the pool to ~0.
    corr, cstats = _pairwise_corr(Zstd, core_names, min_pair_n)
    print(f"analyze: cross-class correlation (pairwise-complete, min {min_pair_n} co-occurring "
          f"subjects/pair): mean|r|={cstats['mean_abs_r']:.3f}, max r={cstats['max_r']:.3f}, "
          f"min r={cstats['min_r']:.3f}; {cstats['n_unreliable_pairs']} pairs too sparse -> set to 0")

    fit = _fit_pca(corr, k_factors)
    explained, eigvec, loadings, resid_std, r2_per_class, k = (
        fit["explained"], fit["eigvec"], fit["loadings"], fit["resid_std"], fit["r2_per_class"], fit["k"])
    cum = np.cumsum(explained)
    print("analyze: PCA explained variance ratio (of between-subject, cross-class spread):")
    for kk in range(min(6, len(explained))):
        print(f"  PC{kk+1}: {explained[kk]:.3f}  (cumulative {cum[kk]:.3f})")
    pc1_loadings = eigvec[:, 0]
    frac_pos = (pc1_loadings > 0).mean()
    print(f"analyze: PC1 loading sign pattern: {frac_pos:.0%} positive "
          f"({'one shared factor (brightness/contrast)' if max(frac_pos, 1-frac_pos) > 0.8 else 'mixed -- not a single common factor'})")
    order = np.argsort(-np.abs(pc1_loadings))
    print("  top |loading| classes on PC1:")
    for i in order[:8]:
        print(f"    {core_names[i]:<28} {pc1_loadings[i]:+.3f}")
    print(f"\nanalyze: k={k}-factor fit -> R^2={r2_per_class.mean():.3f} of between-subject "
          f"cross-class variance explained (per-class range {r2_per_class.min():.2f}-{r2_per_class.max():.2f})")

    factors_path = OUT / f"{prefix}_factors{tag}.npz"
    np.savez(factors_path,
             core_classes=core_names, mean_hu=mu, std_hu=sd,
             loadings=loadings, resid_std_standardized=resid_std,
             explained_variance_ratio=explained, k_factors=k)
    print(f"analyze: wrote {factors_path} "
          f"(mean_c, std_c, loadings[{k},{core.sum()}], resid_std for a k={k}-factor draw:\n"
          f"    z ~ N(0, I_{k})  (one draw per cohort, like today's cohort_gen)\n"
          f"    mu_c = mean_c + std_c * (loadings[:,c] @ z + resid_std[c] * eps_c),  eps_c ~ N(0,1))")

    _plot(corr, core_names, explained, pc1_loadings, eigvec[:, 1] if k > 1 else None, tag=tag, prefix=prefix)

    if not dedupe or core.sum() < 4:
        return dict(core_names=core_names, mean=mu, std=sd, loadings=loadings,
                    resid_std=resid_std, explained=explained)

    # --- de-bias: PCA above lets a body part's PC-share scale with how many labels the
    # vocabulary happens to give it (bone = 1 label/vertebra/rib = ~50 votes; an organ = 1).
    # Group classes by INTENSITY CORRELATION ALONE (no anatomy lookup -> generalizes to any
    # label set), collapse each group to one representative signal, refit PCA on that -- then
    # broadcast the (now count-unbiased) loadings back so every real class still gets a row.
    groups = _cluster_classes(corr, core_names, cluster_dist)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    big = sorted((g for g in groups.values() if len(g) > 1), key=lambda g: -len(g))
    print(f"\nanalyze: intensity-only clustering (avg-linkage, 1-corr dist<={cluster_dist}) -> "
          f"{len(groups)} clusters from {core.sum()} classes (top sizes {sizes[:10]})")
    for members in big[:6]:
        shown = [str(m) for m in members[:6]] + (["..."] if len(members) > 6 else [])
        print(f"    cluster of {len(members)}: {shown}")

    Zc, cluster_labels, class_cluster_col = _dedup_signal(Zstd, core_names, groups)
    corr_c, cstats_c = _pairwise_corr(Zc, cluster_labels, min_pair_n)
    fit_c = _fit_pca(corr_c, k_factors)
    cum_c = np.cumsum(fit_c["explained"])
    print(f"analyze: DEDUPED ({len(cluster_labels)} intensity-clusters) explained variance "
          f"(mean|r|={cstats_c['mean_abs_r']:.3f}):")
    for kk in range(min(6, len(fit_c["explained"]))):
        print(f"  PC{kk+1}: {fit_c['explained'][kk]:.3f}  (cumulative {cum_c[kk]:.3f})")

    # broadcast: every class in a cluster gets that cluster's (count-unbiased) loading vector.
    loadings_db = fit_c["loadings"][:, class_cluster_col]                # (k_c, n_core)
    r2_db = np.clip((loadings_db ** 2).sum(0), 0, 1)
    resid_std_db = np.sqrt(1 - r2_db)
    print(f"analyze: de-biased per-class fit -> R^2={r2_db.mean():.3f} "
          f"(vs {r2_per_class.mean():.3f} un-debiased; per-class range {r2_db.min():.2f}-{r2_db.max():.2f})")

    debiased_path = OUT / f"{prefix}_factors{tag}_debiased.npz"
    np.savez(debiased_path,
             core_classes=core_names, mean_hu=mu, std_hu=sd,
             loadings=loadings_db, resid_std_standardized=resid_std_db,
             explained_variance_ratio=fit_c["explained"], k_factors=fit_c["k"],
             cluster_of_class=class_cluster_col, cluster_labels=cluster_labels,
             cluster_dist=cluster_dist)
    print(f"analyze: wrote {debiased_path} (same shape/use as the non-debiased file, "
          f"one row per real class, but loadings come from the {len(cluster_labels)}-cluster "
          f"dedup fit above instead of the raw {core.sum()}-class fit)")

    _plot(corr_c, cluster_labels, fit_c["explained"], fit_c["eigvec"][:, 0],
          fit_c["eigvec"][:, 1] if fit_c["k"] > 1 else None, tag=f"{tag}_dedup", prefix=prefix)

    return dict(core_names=core_names, mean=mu, std=sd, loadings=loadings, resid_std=resid_std,
                explained=explained, loadings_debiased=loadings_db, resid_std_debiased=resid_std_db)


def _plot(corr, names, explained, pc1, pc2, tag="", prefix="totalseg_intensity"):
    n = len(names)
    # hierarchical-clustering leaf order so correlated blocks (contrast-phase organs, lungs,
    # bone, ...) show up as visible blocks instead of the arbitrary alphabetical ALL_CLASSES
    # order -- purely cosmetic, doesn't touch the fitted numbers.
    if n > 3:
        d = 1 - corr
        d = np.clip((d + d.T) / 2, 0, None)
        np.fill_diagonal(d, 0)
        leaf = leaves_list(linkage(squareform(d, checks=False), method="average"))
    else:
        leaf = np.arange(n)
    corr_o, names_o = corr[np.ix_(leaf, leaf)], names[leaf]

    fig, ax = plt.subplots(figsize=(0.22 * n + 2, 0.22 * n + 2))
    im = ax.imshow(corr_o, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(names_o, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(names_o, fontsize=5)
    ax.set_title("Cross-class correlation of per-subject mean HU (core classes, clustered)")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_corr{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    k = min(15, len(explained))
    ax.bar(range(1, k + 1), explained[:k], color="tab:blue", alpha=0.7, label="per-PC")
    ax.plot(range(1, k + 1), np.cumsum(explained[:k]), "o-", color="tab:red", label="cumulative")
    ax.set_xlabel("principal component"); ax.set_ylabel("explained variance ratio")
    ax.set_title("PCA scree: between-subject cross-class HU spread")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_scree{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.25 * n + 2, 4))
    order = np.argsort(-np.abs(pc1))
    ax.bar(range(n), pc1[order], color="tab:blue", alpha=0.8, label="PC1 (brightness/contrast?)")
    if pc2 is not None:
        ax.bar(range(n), pc2[order], color="tab:orange", alpha=0.5, label="PC2")
    ax.set_xticks(range(n)); ax.set_xticklabels(names[order], rotation=90, fontsize=5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("loading")
    ax.set_title("PCA loadings per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_loadings{tag}.png", dpi=150)
    plt.close(fig)
    print(f"analyze: wrote {OUT}/{prefix}_{{corr,scree,loadings}}{tag}.png")


def analyze_merged(min_voxels=20, min_n=30, min_pair_n=15, k_factors=5, cluster_dist=0.25,
                   tag="_merged"):
    """Does a SHARED clustering explain intensity variance pooled across CT and MRI?

    CT's raw HU and MRI's per-subject z-score are already each a comparable-WITHIN-modality
    frame (extract() handles that per-dataset), but not comparable ACROSS modalities (CT bone
    ~200 HU means nothing next to MRI bone's z-score). So each dataset's per-class column is
    standardized to its OWN between-subject mean/std first (removing absolute-scale and
    modality-specific offset), THEN the two subject pools are stacked row-wise on their common
    classes -- the pooled correlation this measures is "do these two tissue types co-vary the
    same WAY (same sign/similar magnitude) in a typical CT subject as in a typical MRI
    subject", not "do they have the same absolute intensity". Only classes covered at
    min_n in BOTH modalities are eligible (mostly organs/muscle/bone at TotalSeg-MRI's coarser
    granularity -- MRI lacks per-rib/per-vertebra/per-lobe splits, so CT's fine subdivisions
    can't participate here)."""
    ct_subj, ct_sums, ct_sumsq, ct_counts = extract(dataset="totalseg")
    mri_subj, mri_sums, mri_sumsq, mri_counts = extract(dataset="totalsegmri")

    ct_mean, _, ct_present, _, class_names = class_mean_matrix(ct_sums, ct_sumsq, ct_counts, min_voxels)
    mri_mean, _, mri_present, _, _ = class_mean_matrix(mri_sums, mri_sumsq, mri_counts, min_voxels)

    ct_n, mri_n = ct_present.sum(0), mri_present.sum(0)
    common = (ct_n >= min_n) & (mri_n >= min_n)
    common_names = class_names[common]
    print(f"analyze_merged: {common.sum()} classes covered in BOTH totalseg (CT, {len(ct_subj)} "
          f"subj) and totalsegmri (MRI, {len(mri_subj)} subj): {sorted(common_names.tolist())}")
    if common.sum() < 4:
        print("analyze_merged: too few shared classes for a meaningful clustering, stopping")
        return

    def _standardize(mean_hu, present, common):
        M = mean_hu[:, common]
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0)
        return (M - mu) / np.where(sd > 0, sd, 1.0)

    Z_ct = _standardize(ct_mean, ct_present, common)
    Z_mri = _standardize(mri_mean, mri_present, common)
    Z_pooled = np.concatenate([Z_ct, Z_mri], axis=0)
    modality = np.array(["CT"] * len(Z_ct) + ["MRI"] * len(Z_mri))

    corr_ct, _ = _pairwise_corr(Z_ct, common_names, min_pair_n)
    corr_mri, _ = _pairwise_corr(Z_mri, common_names, min_pair_n)
    corr_pooled, cstats = _pairwise_corr(Z_pooled, common_names, min_pair_n)
    print(f"analyze_merged: pooled ({len(Z_pooled)} rows) mean|r|={cstats['mean_abs_r']:.3f} "
          f"(CT-alone {np.abs(corr_ct[~np.eye(len(corr_ct), dtype=bool)]).mean():.3f}, "
          f"MRI-alone {np.abs(corr_mri[~np.eye(len(corr_mri), dtype=bool)]).mean():.3f} -- "
          f"on this SAME shared class set, for reference)")

    fit = _fit_pca(corr_pooled, k_factors)
    cum = np.cumsum(fit["explained"])
    print("analyze_merged: pooled PCA explained variance:")
    for kk in range(min(6, len(fit["explained"]))):
        print(f"  PC{kk+1}: {fit['explained'][kk]:.3f}  (cumulative {cum[kk]:.3f})")
    order = np.argsort(-np.abs(fit["eigvec"][:, 0]))
    print("  top |loading| classes on pooled PC1:")
    for i in order[:8]:
        print(f"    {common_names[i]:<24} {fit['eigvec'][i, 0]:+.3f}")

    groups = _cluster_classes(corr_pooled, common_names, cluster_dist)
    big = sorted((g for g in groups.values() if len(g) > 1), key=lambda g: -len(g))
    print(f"analyze_merged: {len(groups)} intensity-only clusters (pooled, dist<={cluster_dist}) "
          f"from {common.sum()} shared classes:")
    for members in big:
        print(f"    cluster of {len(members)}: {[str(m) for m in members]}")

    # does this pooled clustering agree with each modality's OWN clustering on the same
    # classes? If a pooled cluster's members are ALSO tightly correlated within CT alone AND
    # within MRI alone, that's real shared structure; if only one modality drives it, the
    # pooling is being dominated by whichever dataset has more/tighter classes here.
    print("\nanalyze_merged: per-cluster within-modality agreement check "
          "(mean r within CT alone / within MRI alone, for pooled clusters of size>=3):")
    for members in big:
        if len(members) < 3:
            continue
        idx = [int(np.where(common_names == m)[0][0]) for m in members]
        r_ct = corr_ct[np.ix_(idx, idx)]
        r_mri = corr_mri[np.ix_(idx, idx)]
        mask = ~np.eye(len(idx), dtype=bool)
        print(f"    {[str(m) for m in members]}: CT={r_ct[mask].mean():+.3f}  "
              f"MRI={r_mri[mask].mean():+.3f}")

    factors_path = OUT / f"totalseg_mri_merged_intensity_factors{tag}.npz"
    np.savez(factors_path, common_classes=common_names, corr_pooled=corr_pooled,
             corr_ct=corr_ct, corr_mri=corr_mri, explained_variance_ratio=fit["explained"],
             modality=modality)
    print(f"\nanalyze_merged: wrote {factors_path}")
    _plot(corr_pooled, common_names, fit["explained"], fit["eigvec"][:, 0],
          fit["eigvec"][:, 1] if fit["k"] > 1 else None, tag=tag, prefix="totalseg_mri_merged_intensity")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS) + ["merged"], default="totalseg",
                    help="totalseg (CT, raw HU) | totalsegmri (MRI, per-subject clip+zscore "
                         "via ct_stats.json -- no absolute intensity unit otherwise) | merged "
                         "(pool both, per-class-standardized within each first; see "
                         "analyze_merged)")
    p.add_argument("--recompute", action="store_true", help="ignore the stats cache, rescan all subjects")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--min_frac", type=float, default=0.0,
                    help="class must be present (>=min_voxels) in this fraction of subjects to be "
                         "'core'; default 0.0 = ALL classes (still subject to --min_n)")
    p.add_argument("--min_voxels", type=int, default=20,
                    help="min full-res voxel count for a class to count as 'present' in a subject")
    p.add_argument("--min_n", type=int, default=30,
                    help="drop a class from correlation/PCA if fewer than this many subjects have it")
    p.add_argument("--min_pair_n", type=int, default=15,
                    help="a class PAIR's correlation is zeroed if fewer than this many subjects "
                         "have both classes present (pairwise-complete correlation)")
    p.add_argument("--k_factors", type=int, default=3)
    p.add_argument("--classes", type=str, default=None,
                    help="comma-separated explicit class list (overrides --min_frac auto-selection); "
                         "'ts_organs' expands to data.totalseg_classes.TS_SET_ORGANS")
    p.add_argument("--tag", type=str, default="", help="output filename suffix, e.g. _ts_organs")
    p.add_argument("--no_dedupe", action="store_true",
                    help="skip the intensity-clustering de-bias pass (raw per-class fit only)")
    p.add_argument("--cluster_dist", type=float, default=0.4,
                    help="avg-linkage cut on 1-corr distance for the de-bias clustering "
                         "(lower = fewer/tighter clusters, e.g. 0.15 merges only near-duplicates "
                         "like left/right vertebrae; higher merges more loosely-related classes)")
    p.add_argument("--no_body", action="store_true",
                    help="totalseg CT only: skip the extra 'body' pseudo-class (HU of each "
                         "subject's pred_body.npy minus every one of the 117 labels -- the "
                         "soft-tissue envelope MAISI id 200 paints but label.npy has no class "
                         "for). Included by default; writes a separate _body-tagged stats cache.")
    args = p.parse_args()

    classes = None
    if args.classes == "ts_organs":
        from data.totalseg_classes import TS_SET_ORGANS
        classes = TS_SET_ORGANS
    elif args.classes:
        classes = [c.strip() for c in args.classes.split(",")]

    if args.dataset == "merged":
        analyze_merged(min_voxels=args.min_voxels, min_n=args.min_n, min_pair_n=args.min_pair_n,
                       k_factors=args.k_factors, cluster_dist=args.cluster_dist,
                       tag=(args.tag or "_merged"))
    else:
        with_body = (args.dataset == "totalseg") and not args.no_body
        subjects, sums, sumsqs, counts = extract(dataset=args.dataset, recompute=args.recompute,
                                                 workers=args.workers, with_body=with_body)
        modality = DATASETS[args.dataset]["modality"]
        analyze(subjects, sums, sumsqs, counts,
                min_frac=args.min_frac, min_voxels=args.min_voxels, k_factors=args.k_factors,
                classes=classes, tag=args.tag, min_n=args.min_n, min_pair_n=args.min_pair_n,
                dedupe=not args.no_dedupe, cluster_dist=args.cluster_dist,
                prefix=f"{args.dataset}_intensity",
                unit=("HU" if modality == "ct" else "z"))

"""General CLI for nifti in-context cascade inference (patchset3d v2 cascade).

Runnable by anyone who can activate the shared `patchset` conda env — no repo checkout
needed (the code is vendored into the env; see docs). Predicts a target organ mask from
one or more context (image, binary-mask) nifti pairs for the same organ, via the
coarse->fine v2 cascade (cascade.run_cascade: COM re-crop + query-mask prior per level).
GT-free for the target; pass --gt to also report Dice.

Two modes, mutually exclusive:

  --target (single target, one process):
    patchset-infer \
        --target tgt_ct.nii.gz \
        --context ctx_ct.nii.gz ctx_mask.nii.gz \
        --checkpoint /nfs/.../best.pt \
        --out pred.nii.gz [--gt gt_mask.nii.gz]

  --manifest (many targets, ONE process/ONE checkpoint load for all of them -- every
  process pays ~1-2s of Python import + CUDA init + GPU clock ramp-up before touching any
  data, a cost that doesn't scale with work done; N separate --target calls pay it N
  times, --manifest pays it once):
    patchset-infer --manifest tasks.jsonl --checkpoint /nfs/.../best.pt \
        --out-dir preds/ [--gt-required]

  Manifest format: JSON Lines, one task object per line --
    {"target": "tgt.nii.gz", "context": [["ctx_img.nii.gz", "ctx_mask.nii.gz"], ...]}
  Optional per-task keys: "gt" (report Dice), "out" (write prediction here -- otherwise
  derived from --out-dir), "labels" (multi-label mode, same syntax as --labels below),
  "batch_size" (per-task override of --batch-size).

Repeat --context for K>1 contexts (single-target mode only; manifest mode's context list
is per-task). Extra Hydra overrides pass through via --override.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo / vendored-bundle root (…/experiments/3d -> root)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
# eval.yaml's hydra.searchpath is file://${oc.env:PWD}/configs, so the cluster/dataset/
# augmentations groups only resolve when PWD points at the bundle root. Pin it here so the
# CLI works from any working directory (and from the vendored env copy).
os.environ["PWD"] = str(ROOT)

# Current working checkpoint (2026-10-13) -- update this when a better one replaces it.
_DEFAULT_CHECKPOINT = (
    "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/"
    "results/patch_icl/3d_train/2026-09-28_168_cascade_real_host_all_shapes/best.pt"
)

from hydra import compose, initialize_config_dir  # noqa: E402
from infer_nifti import predict_nifti  # noqa: E402


def _build_cfg(args):
    """Compose the same cfg surface as experiments/3d/eval.py, with inference overrides."""
    cfg_dir = str(ROOT / "configs" / "experiment" / "3d")
    overrides = [
        f"eval.model={args.model}",
        f"eval.checkpoint={args.checkpoint}",
        f"eval.feat_norm={args.feat_norm}",
        # ++: the base eval `data` group is struct and lacks these v2-cascade keys.
        f"++data.cascade_spacings=[{args.crop_spacings}]",
        f"++data.cascade_query_prior={args.query_prior}",
        f"data.mask_downsample={args.mask_downsample}",
        # mask_occupancy_thr: omitted here when the user didn't pass --mask-occupancy-thr, so
        # _warn_uninherited_data (called from predict_nifti) auto-inherits the checkpoint's own
        # training value instead of silently forcing an unrelated constant. See main()'s
        # explicit_data_keys, which tells that check this WAS a deliberate CLI choice when set.
        *([f"data.mask_occupancy_thr={args.mask_occupancy_thr}"]
          if args.mask_occupancy_thr is not None else []),
        # ++: not in the base eval `data` group; only added when requested, so omitting the
        # flag leaves predict_nifti's own cfg.data.get("gpu_realize_crop", False) default in
        # effect rather than always forcing an explicit value either way.
        *(["++data.gpu_realize_crop=true"] if args.gpu_realize_crop else []),
        f"++data.cascade_recrop_workers={args.recrop_workers}",
        "wandb.project=null",
        *(args.override or []),
    ]
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        return compose(config_name="eval", overrides=overrides)


def _task_label(target: Path) -> str:
    """Parent-dir + stem, e.g. 's0000_ct' -- nifti layouts (incl. TotalSegmentator's own
    <subject>/ct.nii.gz) commonly reuse the SAME filename across every subject, so the
    filename alone collides both as a printed label and (critically) as a default output
    filename -- target.stem alone would make every task in a --manifest run silently
    overwrite the same <out-dir>/ct_pred.nii.gz."""
    return f"{target.parent.name}_{target.stem.replace('.nii', '')}"


def _load_manifest(path: Path) -> list[dict]:
    tasks = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i+1}: invalid JSON ({e})") from e
            if "target" not in task or "context" not in task:
                raise ValueError(f"{path}:{i+1}: task needs 'target' and 'context'")
            tasks.append(task)
    if not tasks:
        raise ValueError(f"{path}: no tasks (empty or all comments/blank)")
    return tasks


def _parse_label_ids(spec):
    if not spec:
        return None
    return ("all" if str(spec).strip().lower() == "all"
            else [int(x) for x in str(spec).split(",") if x.strip()])


def build_parser():
    ap = argparse.ArgumentParser(
        prog="patchset-infer", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, default=None,
                    help="single-target mode: target CT .nii.gz to segment")
    ap.add_argument("--context", action="append", nargs=2,
                    metavar=("IMAGE", "MASK"),
                    help="single-target mode: context image + binary mask (same organ); "
                         "repeatable for K>1")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="many-targets mode: JSON-Lines task list instead of --target/"
                         "--context, one model load for all of them -- see module "
                         "docstring for the schema. Mutually exclusive with --target.")
    ap.add_argument("--out-dir", dest="out_dir", type=Path, default=None,
                    help="--manifest mode only: write each task's prediction to "
                         "<out-dir>/<parent-dir>_<stem>_pred.nii.gz when the task has no "
                         "explicit 'out'; omit to skip writing those")
    ap.add_argument("--gt-required", dest="gt_required", action="store_true",
                    help="--manifest mode only: error out if any task is missing 'gt' "
                         "(default: Dice just omitted for tasks without one)")
    ap.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT,
                    help="trained patchset3d best.pt. Defaults to "
                         "168_cascade_real_host_all_shapes (the current working checkpoint) "
                         "-- a 20-epoch probe (best_val_dice=0.51 @ epoch 10), not a final/"
                         "fully-converged checkpoint. Pass --checkpoint explicitly to use "
                         "anything else.")
    ap.add_argument("--out", type=Path, default=None,
                    help="single-target mode only: write the predicted mask here "
                         "(.nii.gz, on the target grid)")
    ap.add_argument("--gt", type=Path, default=None,
                    help="single-target mode only: optional target GT mask .nii.gz -> "
                         "report Dice + coarse-only Dice (binary in single-organ mode, "
                         "id-valued when --labels is given)")
    ap.add_argument("--labels", default=None,
                    help="single-target mode only: multi-label mode -- read each "
                         "--context MASK as an id-valued TotalSegmentator mask and "
                         "segment these ids (comma-separated, e.g. 1,2,5) or 'all' for "
                         "every non-zero id present. Omit for the default single "
                         "binary-mask mode. Output is one id-valued mask (smaller "
                         "organs win overlaps). In --manifest mode, set per-task via "
                         "the \"labels\" key instead.")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=8,
                    help="label-tasks per model forward in multi-label mode (default 8; "
                         "lower it if the GPU runs out of memory). In --manifest mode, "
                         "this is the default for any task without its own \"batch_size\" key.")
    ap.add_argument("--crop-spacings", dest="crop_spacings", default="4,1.5",
                    help="comma-separated coarse->fine CROP resolutions in mm/voxel — the "
                         "cascade's field-of-view schedule (T*mm per pass), NOT the input's "
                         "native voxel spacing (that is read from the nifti affine). "
                         "Match the checkpoint's training crops; default 4,1.5")
    ap.add_argument("--query-prior", dest="query_prior", default="pred",
                    help="v2 cascade query-mask prior mode fed to level i>0: none|pred "
                         "(default pred). Match how the checkpoint was trained "
                         "(data.cascade_query_prior).")
    ap.add_argument("--model", default="patchset3d", help="eval.model (default patchset3d)")
    ap.add_argument("--feat-norm", dest="feat_norm", default="self",
                    help="patchset3d feature-norm mode (default self)")
    ap.add_argument("--mask-downsample", dest="mask_downsample", default="occupancy",
                    help="context-mask downsample: occupancy|nearest (default occupancy)")
    ap.add_argument("--mask-occupancy-thr", dest="mask_occupancy_thr", type=float,
                    default=None, help="occupancy threshold; default = the checkpoint's own "
                    "training value (auto-inherited, see eval._warn_uninherited_data)")
    ap.add_argument("--gpu-realize-crop", dest="gpu_realize_crop", action="store_true",
                    help="resample re-crops on the GPU instead of CPU (same mechanism "
                         "training's data.gpu_realize_crop already uses). Off by default; "
                         "worth it for multi-label runs with many organs (~25-35%% faster, "
                         "measured on a 20-label run) -- output Dice differs from the CPU "
                         "path by <2e-2 (decimate+area-pool vs. direct trilinear ordering), "
                         "not a bug. Single-organ runs see little speedup (nothing to batch).")
    ap.add_argument("--recrop-workers", dest="recrop_workers", type=int, default=1,
                    help="thread-pool size for the per-task crop-load fan-out (default 1, "
                         "serial). provider.load is GIL-releasing (np.load/mmap + "
                         "F.interpolate), so this is real parallelism, not GIL-bound -- "
                         "measured ~2x faster wall time at 4 on a 20-label run WITHOUT "
                         "--gpu-realize-crop (CPU resample is the bottleneck it parallelizes). "
                         "Showed no clear benefit ON TOP of --gpu-realize-crop in the same "
                         "test (that flag already removes most of the CPU-bound work this "
                         "would parallelize) -- try both, don't assume stacking helps.")
    ap.add_argument("--override", action="append", metavar="KEY=VALUE",
                    help="extra Hydra override, passed through to the cfg (repeatable)")
    return ap


def _run_single(args):
    contexts = [(Path(img), Path(msk)) for img, msk in args.context]

    print(f"target     : {args.target}")
    for i, (img, msk) in enumerate(contexts):
        print(f"context[{i}] : {img}  mask={msk}")
    print(f"model      : {args.model}   crop_spacings=[{args.crop_spacings}]   "
          f"checkpoint={args.checkpoint}\n")

    label_ids = _parse_label_ids(args.labels)
    if label_ids is not None:
        print(f"labels     : {label_ids}   batch_size={args.batch_size}")

    cfg = _build_cfg(args)
    explicit_data_keys = ({"data.mask_occupancy_thr"}
                          if args.mask_occupancy_thr is not None else set())
    res = predict_nifti(cfg, args.target, contexts, label_ids=label_ids,
                        batch_size=args.batch_size, gt_path=args.gt, out_path=args.out,
                        explicit_data_keys=explicit_data_keys)

    timing = res.get("timing") or {}
    per_label_ms = timing.get("per_label_ms") or {}

    print(f"\n  pred nonzero voxels : {int((res['pred'] > 0).sum())}  shape={res['pred'].shape}")
    if res.get("labels") is not None:                       # multi-label
        names = res.get("label_names") or {}
        print(f"  labels segmented    : {res['labels']}")
        if names:
            print(f"  label table         : {', '.join(f'{k}={names[k]}' for k in res['labels'] if k in names)}")
        for lab in res["labels"]:
            nm = f" ({names[lab]})" if lab in names else ""
            parts = []
            if res["dice"] is not None:
                parts.append(f"dice={res['dice'][lab]:.4f}  coarse={res['coarse_only_dice'][lab]:.4f}")
            if lab in per_label_ms:
                parts.append(f"time={per_label_ms[lab]:.0f}ms")
            if parts:
                print(f"    label {lab:>3}{nm} : {'  '.join(parts)}")
        if res["dice"] is not None:
            print(f"  macro dice          : {res['macro_dice']:.4f}")
    elif res["dice"] is not None:                           # single-organ
        print(f"  dice                : {res['dice']:.4f}")
        print(f"  coarse_only_dice    : {res['coarse_only_dice']:.4f}")
        print(f"  gain (fine-coarse)  : {res['dice'] - res['coarse_only_dice']:+.4f}")
    if res["pred_path"] is not None:
        print(f"  written             : {res['pred_path']}")
    if timing:
        print(f"  cascade time        : {timing['cascade_s']:.1f}s"
              f"  (total incl. model load/IO: {timing['total_s']:.1f}s)")
    return 0


def _run_manifest(args):
    from eval import _build_model  # local: only needed on this path

    tasks = _load_manifest(args.manifest)
    print(f"manifest   : {args.manifest}  ({len(tasks)} tasks)")
    print(f"checkpoint : {args.checkpoint}")
    if args.gt_required:
        missing = [i for i, t in enumerate(tasks) if "gt" not in t]
        if missing:
            raise ValueError(f"--gt-required: tasks missing 'gt' at lines {missing}")

    t_setup0 = time.perf_counter()
    cfg = _build_cfg(args)
    explicit_data_keys = ({"data.mask_occupancy_thr"}
                          if args.mask_occupancy_thr is not None else set())
    model = _build_model(cfg)
    if hasattr(model, "eval"):
        model.eval()
    t_setup = time.perf_counter() - t_setup0
    print(f"setup (cfg compose + model load, ONCE) : {t_setup:.2f}s\n")

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    results, errors = [], []
    t_batch0 = time.perf_counter()
    for i, task in enumerate(tasks):
        target = Path(task["target"])
        contexts = [(Path(a), Path(b)) for a, b in task["context"]]
        gt = Path(task["gt"]) if task.get("gt") else None
        if task.get("out"):
            out = Path(task["out"])
        elif args.out_dir is not None:
            out = args.out_dir / f"{_task_label(target)}_pred.nii.gz"
        else:
            out = None
        label_ids = _parse_label_ids(task.get("labels"))
        batch_size = int(task.get("batch_size", args.batch_size))

        t0 = time.perf_counter()
        try:
            res = predict_nifti(cfg, target, contexts, label_ids=label_ids,
                                batch_size=batch_size, gt_path=gt, out_path=out,
                                explicit_data_keys=explicit_data_keys, model=model)
        except Exception as e:  # noqa: BLE001 -- one bad task shouldn't kill the run
            dt = time.perf_counter() - t0
            print(f"[{i+1}/{len(tasks)}] {_task_label(target)} : ERROR ({dt:.1f}s) -- {e}")
            errors.append({"index": i, "target": str(target), "error": str(e)})
            continue
        dt = time.perf_counter() - t0
        dice = res.get("dice")
        if isinstance(dice, dict):
            dice_str = f"macro={res.get('macro_dice', float('nan')):.4f}"
        elif dice is not None:
            dice_str = f"dice={dice:.4f}"
        else:
            dice_str = "dice=n/a"
        print(f"[{i+1}/{len(tasks)}] {_task_label(target)} : {dice_str}  ({dt:.2f}s, "
              f"cascade={res['timing']['cascade_s']:.2f}s)")
        results.append({"index": i, "target": str(target), "dice": dice,
                        "timing": res["timing"], "pred_path": str(res["pred_path"])
                        if res["pred_path"] else None})

    t_batch = time.perf_counter() - t_batch0
    n_ok = len(results)
    print(f"\n{n_ok}/{len(tasks)} tasks OK ({len(errors)} errors)")
    print(f"setup (once)        : {t_setup:.2f}s")
    print(f"manifest wall time  : {t_batch:.2f}s  ({t_batch/max(n_ok,1):.2f}s/task avg)")
    if n_ok:
        cascade_sum = sum(r["timing"]["cascade_s"] for r in results)
        print(f"sum(cascade_s)      : {cascade_sum:.2f}s  "
              f"({cascade_sum/n_ok:.2f}s/task avg -- the part that doesn't amortize)")
    dices = [r["dice"] for r in results if isinstance(r["dice"], (int, float))]
    if dices:
        print(f"mean dice (single-organ tasks) : {sum(dices)/len(dices):.4f}  (n={len(dices)})")
    return 1 if errors else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.target is not None) == (args.manifest is not None):
        raise SystemExit("patchset-infer: pass exactly one of --target (single-target "
                         "mode, needs --context too) or --manifest (many-targets mode)")
    if args.manifest is not None:
        return _run_manifest(args)
    if not args.context:
        raise SystemExit("patchset-infer: --target mode needs at least one --context")
    return _run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""
Config-driven 3D in-context eval — the harness twin of experiments/3d/train.py,
mirroring experiments/2d/eval_incontext.py. Evaluates one model (default medverse)
on the TotalSegmentator test split over a class list, reporting per-class Dice,
mean inference time, and GFLOPs. Shares the loader (common.make_eval_loader) and
eval loop (evaluate.evaluate_classes) with the rest of the 3D harness.

    python experiments/3d/eval.py
    python experiments/3d/eval.py eval.model=native_resenc \
        eval.checkpoint=results/checkpoints/resenc_in_context_best.pt
    python experiments/3d/eval.py dataset=omnisynth3d eval.split=val   # eval on omniSynth-3D
"""

import datetime
import json
import random
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling common/evaluate (dir '3d')

from data.totalseg_classes import resolve_classes
from src.benchmark_models import load_model
from common import DEVICE, _source_root
from evaluate import measure_flops, evaluate_classes, evaluate_spacing_sweep, build_sample_table


# Data params that change what the model actually sees at inference. eval.py restores only
# `arch` from the checkpoint — data.* comes from the eval config — so drift on these keys
# silently produces plausible-but-wrong numbers instead of an error. Warn on any mismatch.
_FIDELITY_KEYS = ("image_size", "crop_spacing_mm", "use_crop", "context_size",
                  "mask_downsample", "mask_occupancy_thr", "source",
                  # raw_ct changes intensity normalization; self_context changes how the K
                  # contexts are built (p.eval>0 => self-context leakage). Neither is restored
                  # from the checkpoint, so both drift silently without this. See docs/logs.md.
                  "raw_ct", "self_context")


def _warn_uninherited_data(cfg: DictConfig) -> None:
    """Warn about eval-config data params that differ from the checkpoint's training data.

    The checkpoint stores the full training `cfg.data` but eval.py does NOT restore it (only
    `arch`): the eval config stays authoritative. So a run trained at crop_spacing_mm=2 /
    occupancy masks but evaluated with the loader defaults (1.5 / nearest) reports a
    train-test mismatch as if nothing were wrong. This prints those uninherited differences up
    front so the user can re-supply matching data.* overrides."""
    ckpt_path = cfg.eval.get("checkpoint")
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_data = ckpt.get("data")
    if not train_data:
        print("  [warn] checkpoint has no stored `data` (older run) — cannot check eval-config "
              "drift; ensure data.* matches training manually.")
        return
    drift = []
    for k in _FIDELITY_KEYS:
        ev = cfg.data.get(k)
        tr = train_data.get(k)
        # Normalize both sides to plain containers so nested keys (self_context) compare by
        # content, not DictConfig-vs-dict identity (which would flag a spurious mismatch).
        if isinstance(ev, (DictConfig, ListConfig)):
            ev = OmegaConf.to_container(ev, resolve=True)
        if isinstance(tr, (DictConfig, ListConfig)):
            tr = OmegaConf.to_container(tr, resolve=True)
        if tr != ev:
            drift.append((k, tr, ev))
    if drift:
        print("  [warn] eval data config NOT inherited from the checkpoint (only `arch` is); "
              "these differ from training and change what the model sees:")
        for k, tr, ev in drift:
            print(f"         - {k}: train={tr!r}  eval={ev!r}")
        print("         Re-supply matching data.* overrides for a faithful eval.\n")


def _assert_sweep_supported(cfg: DictConfig) -> None:
    """Fail fast when eval.spacing_sweep is set on an unsupported config.

    The per-spacing crop override only takes effect on the totalseg direct-build path
    with use_crop=true (the resized path ignores _cur_crop_spacing; build_dataset-routed
    sources build their own datasets). Reject anything else with a clear message rather
    than silently producing a single-spacing result."""
    if not cfg.data.get("use_crop"):
        raise ValueError(
            "eval.spacing_sweep requires data.use_crop=true — the crop-spacing override is a "
            "no-op on the pre-resized path (it reports fixed voxel spacing).")
    source = cfg.data.get("source", "totalseg")
    if source in ("omnisynth3d", "anchor_synth3d"):
        raise ValueError(
            f"eval.spacing_sweep is unsupported for data.source={source!r} (routed through "
            "build_dataset with no per-item spacing override). Supported: totalseg (direct "
            "build) and totalseg_more_labels (a TotalSegInContextDataset subclass).")
    if cfg.eval.get("spacing_locator") or cfg.eval.get("spacing_cascade"):
        which = "spacing_locator" if cfg.eval.get("spacing_locator") else "spacing_cascade"
        sweep = cfg.eval.get("spacing_sweep")
        sl = list(sweep) if sweep else []
        if len(sl) < 2 or not any(sl[i + 1] < sl[i] for i in range(len(sl) - 1)):
            raise ValueError(
                f"eval.{which} requires eval.spacing_sweep with at least one "
                f"descending step (e.g. [4, 2]); got {sl!r}.")


def _build_model(cfg: DictConfig):
    """Instantiate the eval model. Medverse is inference-only (no checkpoint needed);
    other models need eval.checkpoint. Handles medverse ckpt/sw_roi that load_model
    cannot forward through its `ckpt_path` param."""
    name = cfg.eval.model
    image_size = tuple(cfg.data.image_size)
    if name == "medverse":
        from src.benchmark_models.medverse import MedverseModel
        mk = {}
        if cfg.eval.get("sw_roi_size"):
            mk["sw_roi_size"] = tuple(cfg.eval.sw_roi_size)
        if cfg.eval.get("medverse_ckpt"):
            mk["ckpt_path"] = cfg.eval.medverse_ckpt
        model = MedverseModel(device=DEVICE, **mk)
        # Fine-tuned checkpoint from experiments/3d/train.py (state under "model").
        if cfg.eval.get("checkpoint"):
            ckpt = torch.load(cfg.eval.checkpoint, map_location=DEVICE, weights_only=False)
            model.load_finetuned(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"  Loaded fine-tuned medverse weights from {cfg.eval.checkpoint}")
        return model
    if name == "patchset3d":
        # PatchSet3D is used directly as the eval model (it provides .predict, the only
        # method the shared eval loop needs). The architecture is rebuilt from the
        # checkpoint's stored `arch` when present (new checkpoints), else from cfg.arch
        # (re-supply the same model=patchset3d arch.* overrides used at training).
        from train import build_model
        if not cfg.eval.get("checkpoint"):
            raise ValueError("eval.checkpoint is required for patchset3d")
        ckpt = torch.load(cfg.eval.checkpoint, map_location=DEVICE, weights_only=False)
        arch = ckpt.get("arch")
        from omegaconf import open_dict
        with open_dict(cfg):
            # build_model dispatches on top-level cfg.model, which is unset unless the
            # user passed +model=patchset3d — pin it so the arch-from-checkpoint path
            # (eval.model=patchset3d only) still builds a PatchSet3D, not medverse.
            cfg.model = "patchset3d"
            if arch is not None:
                cfg.arch = OmegaConf.create(arch)   # rebuild from the stored arch
            elif "arch" not in cfg:
                raise ValueError(
                    "checkpoint has no stored arch (older run); re-supply the training "
                    "arch, e.g. +model=patchset3d arch.l=2")
            # feat_norm is weight-free -> allow an eval-time override on top of the stored
            # arch so one checkpoint sweeps context|self|none (older archs lack the key).
            fn = cfg.eval.get("feat_norm")
            if fn is not None:
                cfg.arch.feat_norm = fn
            # Redirect the frozen-encoder sidecar to a config path (weight-free arch metadata):
            # checkpoints bake in a CWD-relative results/checkpoints/... path, but the CoLiPri
            # weights live on shared NFS. Only redirect an EXISTING sidecar (primus checkpoints).
            sc = cfg.eval.get("primus_sidecar")
            if sc is not None and cfg.arch.get("primus_sidecar") is not None:
                cfg.arch.primus_sidecar = sc
        model, _ = build_model(cfg)
        model = model.to(DEVICE)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd)
        model.eval()
        print(f"  Loaded PatchSet3D from {cfg.eval.checkpoint} (arch l={cfg.arch.l})")
        return model
    if name == "totalsegmentator":
        # Context-free nnU-Net TotalSegmentator organ reference (Route B). No checkpoint: the
        # weights folder is eval.totalseg_weights, or the experiment's frozen-encoder weights
        # (arch.nnunet_ts_weights, e.g. e2's Dataset291 organs) so `experiment=52 ...
        # eval.model=totalsegmentator` reuses the already-staged model.
        from src.benchmark_models.totalseg import TotalSegModel
        wdir = cfg.eval.get("totalseg_weights") or cfg.get("arch", {}).get("nnunet_ts_weights")
        if not wdir:
            raise ValueError("eval.model=totalsegmentator needs eval.totalseg_weights "
                             "(or an experiment providing arch.nnunet_ts_weights)")
        print(f"  TotalSegmentator weights: {wdir}")
        return TotalSegModel(wdir, device=DEVICE)
    return load_model(name, ckpt_path=cfg.eval.get("checkpoint"),
                      image_size=image_size, device=DEVICE)


@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.eval.seed)
    torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 tensor cores for fp32 matmuls

    source = cfg.data.get("source", "totalseg")
    if source == "anchor_synth3d":
        from common import resolve_anchor_classes
        root = cfg.paths.get("totalseg")
        classes = resolve_anchor_classes(cfg.anchor_synth, root)
    elif source == "omnisynth3d":
        # Classes come from the omniSynth3D tile-cache pool (the label_names the
        # dataset emits), not label_stats.csv. Build the bank once to list them.
        from src.datasets.omniSynth.bank_totalseg import get_or_build_totalseg_bank
        s3 = cfg.synth3d
        root = s3.get("tiles_root") or cfg.paths.totalseg
        bank = get_or_build_totalseg_bank(
            root, tuple(s3.get("size", cfg.data.image_size)),
            cfg.eval.split, tuple(resolve_classes(s3.get("classes") or (),
                                                  totalseg_root=cfg.paths.get("totalseg"))))
        classes = [bank.alphabet(c) for c in bank.task_ids()]
    elif source == "totalseg_more_labels":
        from data.totalseg_classes import resolve_more_labels_classes
        root = cfg.paths.get("totalseg_more_labels")
        classes = resolve_more_labels_classes(root, cfg.data.val_classes)
    else:
        _, root, is_mri = _source_root(cfg)
        classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    image_size = tuple(cfg.data.image_size)
    K = cfg.data.context_size
    model_name = cfg.eval.model

    print(f"Device      : {DEVICE}")
    print(f"Model       : {model_name}")
    print(f"Data root   : {root}  (source={cfg.data.get('source')}, split={cfg.eval.split})")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"K={K}  image_size={image_size}  n_subjects<={cfg.eval.n_subjects}\n")

    _warn_uninherited_data(cfg)
    model = _build_model(cfg)
    print(f"  Measuring FLOPs (K={K}, size={image_size})...")
    flops = measure_flops(model, image_size, K, DEVICE)
    gflops = flops["total"]
    _brk = "  ".join(f"{k}={flops[k]:.2f}" for k in ("encoder", "transformer")
                     if flops[k] is not None)
    print(f"  GFLOPs: {gflops:.2f}{('  [' + _brk + ']') if _brk else ''}\n")

    # ── wandb / output dir ───────────────────────────────────────────────────
    wb_on = bool(cfg.wandb.get("project"))
    # `_global_`-packaged group selections leave no key in cfg; log Hydra's runtime choices
    # so dataset=/augmentations=/... are visible in wandb (cf. train.py).
    from hydra.core.hydra_config import HydraConfig
    # val_classes is the spec (e.g. "benchmark" or an explicit list); keep it alongside the
    # resolved `classes` so the wandb run records which eval regime produced the numbers.
    val_classes = cfg.data.get("val_classes")
    if isinstance(val_classes, (DictConfig, ListConfig)):
        val_classes = OmegaConf.to_container(val_classes, resolve=True)
    spacing_sweep = cfg.eval.get("spacing_sweep")
    if isinstance(spacing_sweep, (DictConfig, ListConfig)):
        spacing_sweep = OmegaConf.to_container(spacing_sweep, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project, name=cfg.wandb.name,
        mode="online" if wb_on else "disabled",
        config={"model": model_name, "source": cfg.data.get("source"),
                "split": cfg.eval.split, "K": K, "image_size": list(image_size),
                "n_subjects": cfg.eval.n_subjects, "classes": list(classes),
                "gflops": round(gflops, 2),
                # Eval-fidelity knobs that change what the model sees / how Dice is scored.
                "val_classes": val_classes,
                "mask_downsample": cfg.data.get("mask_downsample"),
                "mask_occupancy_thr": cfg.data.get("mask_occupancy_thr"),
                "crop_spacing_mm": cfg.data.get("crop_spacing_mm"),
                "nsd_tolerance_mm": cfg.eval.get("nsd_tolerance_mm"),
                "spacing_sweep": spacing_sweep,
                "spacing_locator": bool(cfg.eval.get("spacing_locator")),
                "spacing_cascade": bool(cfg.eval.get("spacing_cascade")),
                "hydra_choices": dict(HydraConfig.get().runtime.choices)},
    )
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or model_name
    out_dir = Path(cfg.eval.out_dir) / f"{datetime.date.today():%Y-%m-%d}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (out_dir / "figures") if cfg.eval.get("save_figures", True) else None

    # Training class set for the table's `in_train` zero-shot flag (resolved like train.py).
    # Guarded: sources whose classes don't come from a totalseg root (omnisynth/anchor) can't
    # be resolved via _source_root here -> in_train falls back to None (its pre-change value).
    try:
        _, _troot, _is_mri = _source_root(cfg)
        train_classes = set(resolve_classes(cfg.data.get("train_classes"), _troot, is_mri=_is_mri))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not resolve train_classes for the `in_train` flag ({exc}); "
              "leaving it unset.")
        train_classes = None

    # ── per-class eval (shared loop, also used by train.py's val step) ────────
    # No logits_fn: soft-Dice would need a second (untimed) forward per batch — ~2x eval time —
    # so eval leaves `soft_dice`/`loss` empty and reports only the timed model.predict Dice.
    sweep = cfg.eval.get("spacing_sweep")
    locator = bool(cfg.eval.get("spacing_locator"))
    cascade = bool(cfg.eval.get("spacing_cascade"))
    if sweep:
        _assert_sweep_supported(cfg)
        spacings = list(sweep)
        tag = ("  (+ coarse->fine locator)" if locator else "") + \
              ("  (+ coarse->fine cascade)" if cascade else "")
        print(f"  Spacing sweep: {spacings} mm  ({len(spacings)}x eval time){tag}\n")
        cascade_figures = cascade and bool(cfg.eval.get("cascade_figures", False))
        rows, all_cases = evaluate_spacing_sweep(model, cfg, classes, spacings,
                                                 fig_dir=fig_dir, locator=locator,
                                                 cascade=cascade, cascade_figures=cascade_figures)
    else:
        rows, all_cases = evaluate_classes(model, cfg, classes, fig_dir=fig_dir)
    # Full per-sample detail table (mirrors experiments/2d eval.py's sample table): one row
    # per case with Dice, timing, GT/context occupancy stats, per-sample spacing + source-
    # adaptive `detail`, and an `in_train` flag. epoch stays -1 (build_sample_table's sentinel).
    case_table = build_sample_table(all_cases, train_classes=train_classes) if wb_on else None
    for row in rows:
        cls = row["class"]
        sp = row.get("spacing")
        casc = row.get("cascade_from")
        sp_str = f" @{sp:g}mm" if sp is not None else ""
        sp_str += f" (cascade<-{casc:g}mm)" if casc is not None else ""
        # Distinct wandb key for cascade rows so they don't overwrite the GT-centred fine pass
        # at the same spacing.
        sp_key = (f"@{sp:g}" if sp is not None else "") + (f"_casc{casc:g}" if casc is not None else "")
        if "error" in row:
            print(f"  {cls:<35s}{sp_str}  ERROR: {row['error']}")
            continue
        row["gflops"] = round(gflops, 2)
        cont_str = (f"  cont={row['mean_containment']:.3f} (orc={row['mean_containment_oracle']:.3f})"
                    if "mean_containment" in row else "")
        nsd_str = f"  nsd={row['mean_nsd']:.3f}" if "mean_nsd" in row else ""
        print(f"  {cls:<35s}{sp_str}  dice={row['mean_dice']:.3f} ± {row['std_dice']:.3f}"
              f"{nsd_str}  {row['mean_time_ms']:.0f}ms/sample  n={row['n_samples']}{cont_str}")
        if wb_on:
            # Only mean_dice per class; std_dice / mean_time_ms are inferable from the
            # per-sample `cases` table, so we don't duplicate them as scalar series.
            wandb.log({f"class/{cls}/mean_dice{sp_key}": row["mean_dice"]})
            if "mean_nsd" in row:
                wandb.log({f"class/{cls}/mean_nsd{sp_key}": row["mean_nsd"]})
            if "mean_containment" in row:
                wandb.log({f"class/{cls}/containment{sp_key}": row["mean_containment"],
                           f"class/{cls}/containment_oracle{sp_key}": row["mean_containment_oracle"]})

    # Cascade rows are extra (predicted-crop) fine passes; keep them out of the base mean and
    # per-spacing curve so they don't double-count classes, then summarise them separately.
    valid = [r for r in rows if "mean_dice" in r and r.get("cascade_from") is None]
    cascade_rows = [r for r in rows if "mean_dice" in r and r.get("cascade_from") is not None]
    if valid:
        mean_dice = sum(r["mean_dice"] for r in valid) / len(valid)
        mean_ms   = sum(r["mean_time_ms"] for r in valid) / len(valid)
        nsd_vals  = [r["mean_nsd"] for r in valid if "mean_nsd" in r]
        mean_nsd  = sum(nsd_vals) / len(nsd_vals) if nsd_vals else None
        nsd_line  = f"  |  Mean NSD: {mean_nsd:.4f}" if mean_nsd is not None else ""
        print(f"\n  Mean Dice: {mean_dice:.4f}{nsd_line}  |  Mean time: {mean_ms:.1f} ms/sample  "
              f"|  GFLOPs: {gflops:.2f}")
        if wb_on:
            log = {"mean_dice": round(mean_dice, 4), "mean_time_ms": round(mean_ms, 1),
                   "gflops": round(gflops, 2), "cases": case_table}
            if mean_nsd is not None:
                log["mean_nsd"] = round(mean_nsd, 4)
            wandb.log(log)
        if sweep:
            # Aggregate curve: mean Dice over classes at each spacing (GT-centred passes).
            print("  spacing -> mean_dice:")
            for s in spacings:
                vs = [r["mean_dice"] for r in valid if r.get("spacing") == s]
                if vs:
                    md = sum(vs) / len(vs)
                    print(f"    {s:g}mm : {md:.4f}  (n_classes={len(vs)})")
                    if wb_on:
                        wandb.log({f"mean_dice@{s:g}": round(md, 4)})
        if locator:
            print("  pair (coarse->fine) : mean_containment (oracle, gap, n, empty):")
            for r in valid:
                if "mean_containment" in r:
                    gap = r["mean_containment_oracle"] - r["mean_containment"]
                    print(f"    {r['class']:<28s} {r['spacing']:g}->{r['locator_to']:g}mm : "
                          f"{r['mean_containment']:.3f}  (orc {r['mean_containment_oracle']:.3f}, "
                          f"gap {gap:.3f}, n={r['n_locator']}, empty={r['n_locator_empty']})")
        if cascade_rows:
            # Stitched-native cascade Dice (coarse composited + fine overwrite, vs native GT)
            # against the coarse-only native baseline; the gain isolates the fine refinement.
            print("  cascade (coarse->fine, STITCHED native dice) : mean_dice (coarse-only, gain):")
            casc_dices, gains = [], []
            for r in cascade_rows:
                co = r.get("coarse_only_dice")
                has_co = co is not None and co == co  # not None / not NaN
                gain_str = f", gain {r['mean_dice'] - co:+.3f}" if has_co else ""
                co_str = f"{co:.3f}" if has_co else "n/a"
                print(f"    {r['class']:<28s} {r['cascade_from']:g}->{r['spacing']:g}mm : "
                      f"{r['mean_dice']:.3f}  (coarse-only {co_str}{gain_str}, n={r['n_samples']})")
                casc_dices.append(r["mean_dice"])
                if has_co:
                    gains.append(r["mean_dice"] - co)
            if casc_dices:
                mcd = sum(casc_dices) / len(casc_dices)
                gain_txt = f"  |  mean gain over coarse-only: {sum(gains)/len(gains):+.4f}" if gains else ""
                print(f"    mean cascade Dice (stitched native): {mcd:.4f}{gain_txt}")
                if wb_on:
                    wandb.log({"mean_dice_cascade": round(mcd, 4)})
                    if gains:
                        wandb.log({"mean_cascade_gain": round(sum(gains) / len(gains), 4)})

    # ── save outputs ─────────────────────────────────────────────────────────
    (out_dir / "eval.json").write_text(json.dumps(
        {"model": model_name, "config": OmegaConf.to_container(cfg.eval, resolve=True),
         "rows": rows}, indent=2))
    sweep_col = ",spacing" if sweep else ""
    loc_col = ",locator_to,mean_containment,mean_containment_oracle,mean_loc_err_mm" if locator else ""
    casc_col = ",cascade_from,coarse_only_dice" if cascade else ""
    nsd_col = ",mean_nsd,std_nsd" if any("mean_nsd" in r for r in rows) else ""
    csv = [f"model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples{nsd_col}{sweep_col}{loc_col}{casc_col}"]
    csv += [f"{model_name},{r['class']},{r['mean_dice']},{r['std_dice']},"
            f"{r.get('mean_time_ms','')},{r.get('gflops','')},{r['n_samples']}"
            + (f",{r.get('mean_nsd','')},{r.get('std_nsd','')}" if nsd_col else "")
            + (f",{r.get('spacing','')}" if sweep else "")
            + (f",{r.get('locator_to','')},{r.get('mean_containment','')},"
               f"{r.get('mean_containment_oracle','')},{r.get('mean_loc_err_mm','')}" if locator else "")
            + (f",{r.get('cascade_from','')},{r.get('coarse_only_dice','')}" if cascade else "")
            for r in rows if "mean_dice" in r]
    (out_dir / "eval.csv").write_text("\n".join(csv) + "\n")
    print(f"  Saved -> {out_dir}")
    run.finish()


if __name__ == "__main__":
    main()

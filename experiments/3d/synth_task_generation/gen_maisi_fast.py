"""
Fast pipelined MAISI (NV-Generate-CTMR) rflow-ct paired generator -> .npz.

Wraps the stock LDMSampler with three throughput levers established by
bench_maisi_gen.py (batching was rejected — it OOMs and doesn't help):

  1. torch.compile on the diffusion-UNet + ControlNet  (~1.6x on the 30-step loop)
  2. threaded prefetch of mask prep (CPU/disk) so it overlaps the GPU diffusion/decode
  3. threaded QC + save (CPU/IO) off the critical path; --skip_qc drops QC entirely

Each accepted pair is written as ONE compressed .npz (cf. scripts/convert_to_npy.py):
    ct     float16  (H,W,D)  z-scored HU  (normalize_ct; identical to the real-data loader)
    label  uint8    (H,W,D)  MAISI 132-class vocabulary (NOT TotalSegmentator ids)
    spacing float32 (3,)     mm/voxel
    hu_min/hu_max            HU range before normalization (for de-norm/debug)

Run inside the NV-Generate-CTMR repo (needs scripts.*), e.g.:
  MONAI_DATA_DIRECTORY=./temp_work_dir \
  /home/dpxuser/dev/patch_icl/.venv_thor/bin/python \
    /home/dpxuser/dev/patch_icl/experiments/3d/synth_task_generation/gen_maisi_fast.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --num 50 \
    --out /nfs/.../synth_task_gen/npz --compile --skip_qc
"""

import argparse
import json
import queue
import sys
import threading
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------- loading
def build_args(repo: Path, env_file: str, infer_file: str) -> Namespace:
    """Reconstruct the args Namespace inference.py builds, from the three configs."""
    args = Namespace()
    env = json.load(open(repo / "configs" / env_file))
    net = json.load(open(repo / "configs/config_network_rflow.json"))
    inf = json.load(open(repo / "configs" / infer_file))
    root = str(Path(__import__("os").environ.get("MONAI_DATA_DIRECTORY", "./temp_work_dir")))
    for d in (env, net, inf):
        for k, v in d.items():
            # dataset paths are relative to MONAI_DATA_DIRECTORY (mirrors inference.py:97)
            if isinstance(v, str) and "datasets/" in v:
                v = str(Path(root) / v)
            setattr(args, k, v)
    for k in ["trained_autoencoder_path", "trained_diffusion_path", "trained_controlnet_path",
              "trained_mask_generation_autoencoder_path", "trained_mask_generation_diffusion_path"]:
        setattr(args, k, str(repo / getattr(args, k)))
    if "autoencoder_tp_num_splits" in inf:
        args.autoencoder_def["num_splits"] = inf["autoencoder_tp_num_splits"]
        args.mask_generation_autoencoder_def["num_splits"] = inf["autoencoder_tp_num_splits"]
    return args


def load_sampler(args, device, do_compile: bool):
    """Load the 5 nets (mirrors inference.py) and return a ready LDMSampler."""
    import monai
    from scripts.sample import LDMSampler
    from scripts.utils import define_instance

    ae = define_instance(args, "autoencoder_def").to(device).eval()
    ck = torch.load(args.trained_autoencoder_path, weights_only=False)
    ae.load_state_dict(ck["unet_state_dict"] if "unet_state_dict" in ck else ck)

    dm = define_instance(args, "diffusion_unet_def").to(device).eval()
    ckd = torch.load(args.trained_diffusion_path, weights_only=False)
    dm.load_state_dict(ckd["unet_state_dict"], strict=False)
    scale_factor = ckd["scale_factor"].to(device)

    cn = define_instance(args, "controlnet_def").to(device).eval()
    ckc = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(cn, dm.state_dict())
    cn.load_state_dict(ckc["controlnet_state_dict"], strict=False)

    mask_ae = define_instance(args, "mask_generation_autoencoder").to(device).eval()
    mask_ae.load_state_dict(torch.load(args.trained_mask_generation_autoencoder_path, weights_only=True))
    mask_dm = define_instance(args, "mask_generation_diffusion").to(device).eval()
    ckm = torch.load(args.trained_mask_generation_diffusion_path, weights_only=False)
    mask_dm.load_state_dict(ckm["unet_state_dict"])
    mask_sf = ckm["scale_factor"]

    if do_compile:
        # only the two nets in the 30-step loop benefit; AE decode is single-shot (bench: no gain)
        dm = torch.compile(dm, mode="default")
        cn = torch.compile(cn, mode="default")

    latent_shape = [args.latent_channels] + [s // 4 for s in args.output_size]
    ldm = LDMSampler(
        args.body_region, args.anatomy_list, args.all_mask_files_json,
        args.all_anatomy_size_conditions_json, args.all_mask_files_base_dir,
        args.label_dict_json, args.label_dict_remap_json,
        ae, dm, cn, define_instance(args, "noise_scheduler"), scale_factor,
        mask_ae, mask_dm, mask_sf, define_instance(args, "mask_generation_noise_scheduler"),
        device, latent_shape, args.mask_generation_latent_shape,
        args.output_size, args.output_dir, args.controllable_anatomy_size,
        image_output_ext=args.image_output_ext, label_output_ext=args.label_output_ext,
        spacing=args.spacing, modality=args.modality,
        num_inference_steps=args.num_inference_steps,
        mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
        random_seed=args.random_seed,
        autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=args.autoencoder_sliding_window_infer_overlap,
        cfg_guidance_scale=args.cfg_guidance_scale,
    )
    return ldm


# ----------------------------------------------------------------------------- normalization
def get_normalizer():
    """normalize_ct from patch_icl if importable, else a byte-identical local fallback."""
    try:
        from src.totalseg_dataset import normalize_ct
        return normalize_ct
    except Exception:
        CLIP_MIN, CLIP_MAX, MEAN, STD = -1007.0, 1573.0, -167.3, 505.8  # mirror src.totalseg_dataset
        def _norm(hu):
            hu = np.clip(hu, CLIP_MIN, CLIP_MAX)
            return (hu - MEAN) / STD
        return _norm


# ----------------------------------------------------------------------------- pipeline
def select_masks(ldm, num):
    """Replicate LDMSampler.sample_multiple_images' Path-B mask selection."""
    from scripts.find_masks import find_masks

    candidates = find_masks(ldm.body_region, ldm.anatomy_list, ldm.spacing, ldm.output_size,
                            True, ldm.all_mask_files_json, ldm.data_root)
    need_resample = False
    if len(candidates) < num:
        candidates = ldm.find_closest_masks(num)
        need_resample = True
    # select_mask returns *all* shuffled candidates; cap at num so --num is honored
    return ldm.select_mask(candidates, num)[:num], need_resample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_batch.json")
    ap.add_argument("--out", type=Path, required=True, help="output dir for .npz pairs")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--compile", action="store_true", help="torch.compile diffusion-UNet + ControlNet")
    ap.add_argument("--skip_qc", action="store_true", help="skip is_outlier quality check (~2.5s/vol)")
    ap.add_argument("--prefetch", action="store_true", default=True, help="threaded mask-prep prefetch")
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--save_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args_cli = ap.parse_args()

    sys.path.insert(0, str(args_cli.repo))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # patch_icl root for normalize_ct
    from scripts.augmentation import augmentation

    device = torch.device("cuda")
    args = build_args(args_cli.repo, args_cli.env_file, args_cli.infer_file)
    args.random_seed = args_cli.seed
    args_cli.out.mkdir(parents=True, exist_ok=True)
    normalize_ct = get_normalizer()

    t_load = time.time()
    ldm = load_sampler(args, device, args_cli.compile)
    print(f"[load] models ready in {time.time()-t_load:.1f}s  (compile={args_cli.compile})")

    t_sel = time.time()
    selected, need_resample = select_masks(ldm, args_cli.num)
    print(f"[masks] selected {len(selected)} in {time.time()-t_sel:.1f}s  need_resample={need_resample}")

    modality = int(ldm.modality_tensor.flatten()[0])
    do_qc = not args_cli.skip_qc and (1 <= modality <= 7)

    # ---- prep (CPU/disk + some GPU resample): read mask -> resample -> augment
    def prep(item):
        cl, top, bot, sp = ldm.read_mask_information(item["mask_file"])
        if need_resample:
            cl = ldm.ensure_output_size_and_spacing(cl)
        if item["if_aug"]:
            cl = augmentation(cl, ldm.output_size, ldm.random_seed)
        return cl, top, bot, sp

    # ---- save (CPU/IO): optional QC then compressed npz
    save_q: "queue.Queue" = queue.Queue(maxsize=args_cli.save_workers + 2)
    counters = {"saved": 0, "qc_fail": 0}
    clock = {"qc": 0.0, "save": 0.0}
    lock = threading.Lock()

    def save_worker():
        while True:
            job = save_q.get()
            if job is None:
                save_q.task_done(); break
            img_hu, mask, sp_mm = job
            if do_qc:
                t = time.time()
                ok = ldm.quality_check_ct(img_hu[None, None], mask[None, None],
                                          perform_quality_check=True)
                with lock: clock["qc"] += time.time() - t
                if not ok:
                    with lock: counters["qc_fail"] += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            ct = normalize_ct(img_hu.astype(np.float32)).astype(np.float16)
            t = time.time()
            np.savez_compressed(args_cli.out / f"sample_{ts}.npz",
                                ct=ct, label=mask.astype(np.uint8),
                                spacing=np.asarray(sp_mm, np.float32),
                                hu_min=np.float32(img_hu.min()), hu_max=np.float32(img_hu.max()))
            with lock:
                clock["save"] += time.time() - t
                counters["saved"] += 1
            save_q.task_done()

    savers = [threading.Thread(target=save_worker, daemon=True) for _ in range(args_cli.save_workers)]
    for s in savers:
        s.start()

    # ---- prep prefetch thread -> prep_q
    prep_q: "queue.Queue" = queue.Queue(maxsize=2)

    def prep_worker():
        for item in selected:
            prep_q.put(prep(item))
        prep_q.put(None)

    if args_cli.prefetch:
        threading.Thread(target=prep_worker, daemon=True).start()

    # ---- main GPU loop
    print(f"[run] generating {len(selected)} pairs  qc={do_qc}  prefetch={args_cli.prefetch}  "
          f"save_workers={args_cli.save_workers}")
    t_gpu = {"diff_decode": 0.0}
    t0 = time.time()
    for i in range(len(selected)):
        cl, top, bot, sp = prep_q.get() if args_cli.prefetch else prep(selected[i])
        if cl is None:
            break
        tg = time.time()
        imgs, mask = ldm.sample_one_pair(cl, top, bot, sp, ldm.modality_tensor)
        torch.cuda.synchronize()
        t_gpu["diff_decode"] += time.time() - tg
        img_hu = imgs.squeeze().float().cpu().numpy()
        mask_np = mask.squeeze().to(torch.uint8).cpu().numpy()
        save_q.put((img_hu, mask_np, ldm.spacing))
        if (i + 1) % 5 == 0 or i == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(selected)}  {el/(i+1):.1f}s/vol  gpu={t_gpu['diff_decode']/(i+1):.1f}s")

    for _ in savers:
        save_q.put(None)
    save_q.join()
    for s in savers:
        s.join()

    wall = time.time() - t0
    n = counters["saved"]
    print(f"\n[done] {n} pairs in {wall:.1f}s = {wall/max(n,1):.2f}s/vol")
    print(f"       gpu(diff+decode)/vol={t_gpu['diff_decode']/max(n,1):.2f}s  "
          f"qc_total={clock['qc']:.1f}s  save_total={clock['save']:.1f}s  qc_fail={counters['qc_fail']}")
    print(f"       peak GPU mem={torch.cuda.max_memory_allocated()/1024**3:.1f}GB  out={args_cli.out}")


if __name__ == "__main__":
    main()

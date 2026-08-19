"""
Per-module wall-time breakdown of ONE end-to-end MAISI rflow-ct generation.

Times every stage of LDMSampler's Path-B (DB-mask) pipeline separately, so we can
see where the ~112 s/vol (whole-body 384³) actually goes. Reuses gen_maisi_fast's
loader. The diffusion-loop vs VAE-decode split is captured from the INFO logs that
utils_infer.run_controlnet_conditioned_image_dm already emits.

  MONAI_DATA_DIRECTORY=./temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/bench_pipeline_breakdown.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --infer_file config_infer_wholebody.json --num 2
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_maisi_fast import build_args, load_sampler, select_masks  # noqa: E402


class _Grab(logging.Handler):
    """Capture the two timing lines utils_infer logs (diff loop, VAE decode)."""
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.diff = None
        self.decode = None

    def emit(self, record):
        m = record.getMessage()
        if "Latent features generation time" in m:
            self.diff = float(re.search(r"([\d.]+) seconds", m).group(1))
        elif "VAE decoding time" in m:
            self.decode = float(re.search(r"([\d.]+) seconds", m).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--num", type=int, default=2)
    args_cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    grab = _Grab()
    logging.getLogger().addHandler(grab)

    sys.path.insert(0, str(args_cli.repo))
    from scripts.augmentation import augmentation

    device = torch.device("cuda")
    args = build_args(args_cli.repo, args_cli.env_file, args_cli.infer_file)
    args.random_seed = 0

    t = time.time()
    ldm = load_sampler(args, device, do_compile=False)
    t_load = time.time() - t
    print(f"\n[one-time] full sampler load (5 nets + DB): {t_load:.1f}s", flush=True)

    t = time.time()
    selected, need_resample = select_masks(ldm, args_cli.num)
    t_select = time.time() - t
    print(f"[one-time] mask select ({len(selected)} masks, need_resample={need_resample}): "
          f"{t_select:.1f}s  ({t_select/max(len(selected),1):.1f}s/vol amortised)", flush=True)

    rows = []
    for i, item in enumerate(selected[:args_cli.num]):
        st = {}
        t = time.time(); cl, top, bot, sp = ldm.read_mask_information(item["mask_file"]); torch.cuda.synchronize(); st["read_mask"] = time.time() - t
        if need_resample:
            t = time.time(); cl = ldm.ensure_output_size_and_spacing(cl); torch.cuda.synchronize(); st["resample_mask"] = time.time() - t
        if item["if_aug"]:
            t = time.time(); cl = augmentation(cl, ldm.output_size, ldm.random_seed); torch.cuda.synchronize(); st["augment_mask"] = time.time() - t
        grab.diff = grab.decode = None
        t = time.time(); imgs, mask = ldm.sample_one_pair(cl, top, bot, sp, ldm.modality_tensor); torch.cuda.synchronize(); st["image_stage_total"] = time.time() - t
        st["  diffusion_loop(30)"] = grab.diff
        st["  vae_decode(sliding)"] = grab.decode
        st["  binarize+bg+overhead"] = st["image_stage_total"] - (grab.diff or 0) - (grab.decode or 0)
        img_hu = imgs.squeeze().float().cpu().numpy(); mask_np = mask.squeeze().to(torch.uint8).cpu().numpy()
        t = time.time(); ldm.quality_check_ct(img_hu[None, None], mask_np[None, None], perform_quality_check=True); st["quality_check"] = time.time() - t
        t = time.time()
        import io; buf = io.BytesIO()
        np.savez_compressed(buf, ct=img_hu.astype(np.float16), label=mask_np, spacing=np.asarray(ldm.spacing, np.float32))
        st["save_npz(compress)"] = time.time() - t
        st["=VOL TOTAL"] = sum(v for k, v in st.items() if not k.startswith(" ") and k != "image_stage_total") + st["image_stage_total"] - st.get("image_stage_total", 0)
        rows.append(st)
        print(f"  vol {i} done", flush=True)

    # average, print table
    keys = [k for k in rows[-1].keys()]
    print(f"\n=== per-module wall time (mean of {len(rows)} vols, whole-body {ldm.output_size}@{ldm.spacing}) ===", flush=True)
    print(f"{'module':>26} {'sec':>9} {'%vol':>7}", flush=True)
    vol_total = np.mean([sum(r[k] for k in ("read_mask", "resample_mask", "augment_mask",
                                            "image_stage_total", "quality_check", "save_npz(compress)")
                             if k in r) for r in rows])
    for k in keys:
        if k in ("=VOL TOTAL", "image_stage_total"):
            continue
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        mv = float(np.mean(vals))
        pct = 100 * mv / vol_total if not k.startswith(" ") else 100 * mv / vol_total
        print(f"{k:>26} {mv:>9.2f} {pct:>6.1f}%", flush=True)
    print(f"{'-'*44}", flush=True)
    print(f"{'VOL TOTAL (excl. 1-time)':>26} {vol_total:>9.2f} {100.0:>6.1f}%", flush=True)
    print(f"\n[note] '  '-indented rows are sub-parts of image_stage_total.", flush=True)
    print(f"[peak GPU mem] {torch.cuda.max_memory_allocated()/1024**3:.1f} GB", flush=True)


if __name__ == "__main__":
    main()

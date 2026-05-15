"""Download STU-Net pretrained checkpoints from HuggingFace and extract state dicts.

Source: https://huggingface.co/ziyanhuang/STU-Net

Usage
-----
    # Download STU-Net-B (default) and extract state dict
    python scripts/download_stunet.py

    # Download a specific variant
    python scripts/download_stunet.py --variant small
    python scripts/download_stunet.py --variant large

    # Download only (skip extraction)
    python scripts/download_stunet.py --no-extract

    # Custom output directory
    python scripts/download_stunet.py --out /path/to/checkpoints/stunet

Output
------
    <out>/<variant>.model       — raw nnUNet checkpoint
    <out>/<variant>_statedict.pt — plain state dict (loadable with weights_only=True)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


REPO_ID = "ziyanhuang/STU-Net"

# Filename as stored on HuggingFace per variant (4000-epoch TotalSegmentator checkpoints)
_HF_FILENAMES = {
    "small": "small_ep4k.model",
    "base":  "base_ep4k.model",
    "large": "large_ep4k.model",
    "huge":  "huge_ep4k.model",
}


def download(variant: str, out_dir: Path) -> Path:
    filename = _HF_FILENAMES[variant]
    print(f"Downloading {REPO_ID}/{filename} …", flush=True)
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=out_dir,
    )
    dst = out_dir / filename
    print(f"  saved → {dst}")
    return dst


def extract(raw_ckpt: Path, out_dir: Path, variant: str) -> Path:
    out_pt = out_dir / f"{variant}_statedict.pt"
    extract_script = Path(__file__).parent / "extract_stunet_weights.py"
    cmd = [sys.executable, str(extract_script),
           "--input",  str(raw_ckpt),
           "--output", str(out_pt)]
    subprocess.run(cmd, check=True)
    return out_pt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(_HF_FILENAMES), default="base",
                    help="STU-Net variant to download (default: base)")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: results/checkpoints/stunet)")
    ap.add_argument("--no-extract", action="store_true",
                    help="Skip state-dict extraction; keep raw checkpoint only")
    args = ap.parse_args()

    # Resolve output directory
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        # Best-effort: find the project root (two levels up from this script)
        project_root = Path(__file__).parent.parent
        out_dir = project_root / "results" / "checkpoints" / "stunet"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ckpt = download(args.variant, out_dir)

    if not args.no_extract:
        state_dict_pt = extract(raw_ckpt, out_dir, args.variant)
        print(f"\nReady to use:")
        print(f"  stunet_pretrained: {state_dict_pt}")
    else:
        print(f"\nRaw checkpoint: {raw_ckpt}")
        print("Run extract_stunet_weights.py to produce a plain state dict.")


if __name__ == "__main__":
    main()

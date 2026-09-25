"""Strip a patchset3d checkpoint down to encoder + decoder weights only.

Keeps `encoder.*` (feature backbone) and the conv-decoder readout
(`token_proj.*`, `dec_skip_norm.*`, `dec_skip_proj.*`, `dec_blocks.*`,
`dec_film.*`, `dec_head.*`, `dec_token_residual.*`). Drops everything that
constitutes the in-context reasoning core under ablation: `transformer.*`,
`img_embed.*`, `mask_embed.*`, `mask_token`, `ctx_id.*`, `qry_id`,
`thinking.*`, `cascade_proj.*`, `cascade_type`, `pool_proj.*`, `pool_type`.

Loaded with `train.checkpoint_allow_partial=true`: the dropped keys fall
into `missing_keys` and get PatchSet3D's normal random init, identically
for any `arch.l`/`arch.dual_axis` — used to give two attention-mechanism
variants (e.g. dual_axis=True vs dual_axis=False) a symmetric encoder/
decoder-only warm start instead of an asymmetric partial transformer
transfer.

Usage
-----
    python scripts/strip_checkpoint_encoder_decoder.py \\
        --input  .../2026-09-22_108_cascade_register_varspacing_synth03_texture_pool/best.pt \\
        --output .../108_encdec_only.pt
"""

import argparse
from pathlib import Path

import torch

KEEP_PREFIXES = (
    "encoder.",
    "token_proj.",
    "dec_skip_norm.",
    "dec_skip_proj.",
    "dec_blocks.",
    "dec_film.",
    "dec_head.",
    "dec_token_residual.",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Full training checkpoint (.pt)")
    ap.add_argument("--output", required=True, help="Output path for the stripped state-dict")
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    kept = {k: v for k, v in sd.items() if k.startswith(KEEP_PREFIXES)}
    dropped = sorted(set(sd) - set(kept))

    print(f"total keys   : {len(sd)}")
    print(f"kept keys    : {len(kept)}")
    print(f"dropped keys : {len(dropped)}")
    for k in dropped:
        print(f"  - {k}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": kept}, out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

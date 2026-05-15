"""Extract STU-Net encoder weights from an nnUNet v1 checkpoint.

Works without a nnUNet v1 installation by registering stub modules that
satisfy pickle's class resolution during torch.load.

Usage
-----
    python scripts/extract_stunet_weights.py \\
        --input  /tmp/stunet_base.model \\
        --output results/checkpoints/stunet/base_statedict.pt

The output is a plain state-dict .pt file loadable with torch.load + weights_only=True.
"""

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub import system — intercepts any 'nnunet.*' import at pickle time
# ---------------------------------------------------------------------------

class _StubNNModule(nn.Module):
    """Placeholder for any nnunet nn.Module subclass.

    __init__ is intentionally a no-op: pickle reconstructs the object via
    __new__ + __setstate__, so __init__ is never called.  __setstate__
    (inherited from nn.Module) restores __dict__ including _modules,
    _parameters, _buffers — enough to call .state_dict() afterwards.
    """
    def __init__(self, *args, **kwargs):
        pass


class _NNUNetStubFinder:
    """sys.meta_path finder that auto-creates stub modules and classes for
    any nnunet.* package encountered during unpickling."""

    def find_module(self, fullname, path=None):
        if fullname.startswith("nnunet"):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__loader__ = self
        mod.__package__ = fullname
        mod.__path__    = []
        sys.modules[fullname] = mod
        return mod

    def find_class(self, fullname: str, name: str):
        mod = self.load_module(fullname)
        if not hasattr(mod, name):
            stub = type(name, (_StubNNModule,), {"__init__": lambda self, *a, **kw: None})
            setattr(mod, name, stub)
        return getattr(mod, name)


def _install_stubs() -> None:
    sys.meta_path.insert(0, _NNUNetStubFinder())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True,
                    help="Path to the STU-Net .model checkpoint (nnUNet v1 pickle)")
    ap.add_argument("--output", required=True,
                    help="Output path for the plain state-dict .pt file")
    args = ap.parse_args()

    _install_stubs()

    print(f"Loading {args.input} …", flush=True)
    obj = torch.load(args.input, map_location="cpu", weights_only=False)

    # nnUNet v1 training checkpoints are dicts with a 'state_dict' key.
    # Raw torch.save(model) pickles are the model object itself.
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj)
    else:
        state = obj.state_dict()

    enc_keys  = [k for k in state if k.startswith("conv_blocks_context.")]
    dec_keys  = [k for k in state if k.startswith(("conv_blocks_localization.", "upsample_layers.", "seg_outputs."))]
    print(f"  total keys   : {len(state)}")
    print(f"  encoder keys : {len(enc_keys)}")
    print(f"  decoder keys : {len(dec_keys)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"\nSaved → {out}")
    print("Sample encoder keys:")
    for k in sorted(enc_keys)[:6]:
        print(f"  {k}")


if __name__ == "__main__":
    main()

"""
One-shot precision probe: report the LIVE dtype each PatchSet3D module runs at, in
train and eval, so a "which precision is each module operating at?" sanity check is
empirical rather than a code trace.

Dtypes depend only on tensor shapes + the ambient autocast context, not on data, so
this feeds the model a synthetic batch of the config's shapes (no dataset scan → fast).
It builds the SAME model as experiments/3d/train.py (build_model) and wraps the forward
in the SAME autocast contexts (train._autocast / evaluate._eval_autocast).

    python experiments/3d/probe_precision.py experiment=35_colipri_enc_8_i_128
"""

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT / "experiments" / "2d"))

from common import DEVICE
from train import build_model, build_loss, model_output_is_prob, _autocast
from evaluate import _eval_autocast
from grid_metrics import target_like


def _dt(x):
    """Best-effort dtype label for a hook input/output (tensor, tuple, or dict)."""
    if torch.is_tensor(x):
        return str(x.dtype).replace("torch.", "")
    if isinstance(x, (tuple, list)):
        return ",".join(_dt(t) for t in x if torch.is_tensor(t)) or "-"
    if isinstance(x, dict):
        return ",".join(f"{k}:{_dt(v)}" for k, v in x.items() if torch.is_tensor(v)) or "-"
    return "-"


def _probe(net, cfg, phase, autocast_ctx):
    """Run one synthetic forward under autocast_ctx, printing per-module in/out dtypes."""
    B = 1
    K = int(cfg.data.context_size)
    D, H, W = tuple(cfg.data.image_size)
    g = torch.Generator(device="cpu").manual_seed(0)
    image = torch.randn(B, 1, D, H, W, generator=g).to(DEVICE)
    context_in = torch.randn(B, K, 1, D, H, W, generator=g).to(DEVICE)
    context_out = (torch.rand(B, K, D, H, W, generator=g) > 0.7).float().to(DEVICE)

    # Hook the modules that span the precision boundaries we care about.
    named = {}
    for name in ("encoder", "img_embed", "mask_embed", "transformer", "decoder"):
        if hasattr(net, name):
            named[name] = getattr(net, name)
    enc = getattr(net, "encoder", None)
    if enc is not None and hasattr(enc, "primus"):          # CoLiPri/primus internals
        p = enc.primus
        if hasattr(p, "down_projection"):
            named["encoder.primus.down_projection"] = p.down_projection
        if hasattr(p, "eva"):
            named["encoder.primus.eva"] = p.eva

    rows, handles = [], []

    def mk(tag):
        def hook(_m, inp, out):
            rows.append((tag, _dt(inp[0] if inp else None), _dt(out)))
        return hook

    # Deterministic order: encoder internals first, then the named-child order above.
    order = ["encoder.primus.down_projection", "encoder.primus.eva", "encoder",
             "img_embed", "mask_embed", "transformer", "decoder"]
    for tag in order:
        if tag in named:
            handles.append(named[tag].register_forward_hook(mk(tag)))

    is_prob = model_output_is_prob(cfg)
    loss_fn = build_loss(cfg)
    with torch.no_grad(), autocast_ctx:
        out = net(image, context_in=context_in, context_out=context_out, mode="train")
        logits = out["final_logit"]
        target = target_like(torch.zeros(B, 1, D, H, W, device=DEVICE), logits.float())
        loss = loss_fn(logits, target)
    for h in handles:
        h.remove()

    print(f"\n=== {phase}  (net.training={net.training}) ===")
    print(f"{'module':<34} {'in':<18} {'out':<18}")
    print("-" * 70)
    for tag, din, dout in rows:
        print(f"{tag:<34} {din:<18} {dout:<18}")
    print("-" * 70)
    print(f"{'final_logit (pre-loss)':<34} {'':<18} {_dt(logits):<18}")
    print(f"{'loss (' + cfg.train.get('loss', 'smooth_l1') + ')':<34} "
          f"{'':<18} {_dt(loss):<18}  is_prob={is_prob}")


@hydra.main(config_path="../../configs/experiment/3d", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.get("model", "medverse") != "patchset3d":
        raise SystemExit("probe_precision targets model=patchset3d (CoLiPri encoder).")
    if DEVICE.type != "cuda":
        print("WARNING: no CUDA — bf16 autocast is a no-op on CPU; every module will "
              "report fp32, which is NOT representative of a GPU run.")
    model, _ = build_model(cfg)
    net = getattr(model, "model", model)
    net.to(DEVICE)

    enc = getattr(net, "encoder", None)
    enc_kind = cfg.arch.get("encoder", "conv")
    print(f"Device: {DEVICE} | encoder={enc_kind} | "
          f"arch.encoder_precision={cfg.arch.get('encoder_precision', 'bf16')!r} "
          f"(overrides the ambient autocast for the frozen ViT region)")
    if enc is not None and enc_kind == "primus":
        pd = next(enc.primus.parameters()).dtype
        print(f"Frozen CoLiPri param storage dtype: {str(pd).replace('torch.', '')} "
              f"| frozen={getattr(enc, 'frozen', '?')}")

    net.train()
    _probe(net, cfg, "TRAIN", _autocast())
    net.eval()
    _probe(net, cfg, "EVAL", _eval_autocast(cfg.get("model") == "patchset3d"))


if __name__ == "__main__":
    main()

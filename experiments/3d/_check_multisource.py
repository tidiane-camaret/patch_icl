"""Integration check for dataset=multisource_ct_mri (real NFS data, no GPU).

    python experiments/3d/_check_multisource.py

Builds the train + eval datasets via Hydra compose, pulls items, and asserts:
  - regime frequencies are ~1/3 each;
  - every `cross` task draws a both-modality class and is genuinely cross-modality
    (F7 regime-conditional class draw -> genuine-cross rate ~= regime_p[cross]);
  - a CT-only union class never yields an MRI slot;
  - the eval dataset is deterministic (idx -> identical item across two builds);
  - both-modality classes all appear in the eval pass, overall coverage >= 90%.
Exit code 0 = all pass.
"""
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir

from common import build_dataset, resolve_multisource_classes


def _cfg(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                              version_base=None):
        return compose(config_name="train", overrides=overrides)


def main():
    base = ["experiment=81_multisource_ct_mri", "cluster=nfs"]
    cfg = _cfg(base)

    # --- train dataset: regime mix + cross-modality + CT-only guard ---
    ds = build_dataset(cfg, "train")
    prov = ds.provider
    print(f"union classes: {len(prov.classes)}  both-modality: "
          f"{sum(len(v) == 2 for v in prov._avail.values())}")
    assert ds.cohort_mode, "expected cohort_mode for multisource"

    rng = __import__("random").Random(0)
    regimes = Counter()
    cross_genuine = 0
    for _ in range(600):
        it = prov.assemble_task(rng, 3.0)
        m = it["meta"]
        regimes[m["regime"]] += 1
        if m["regime"] == "cross":
            # F7: `cross` draws its class from both-modality classes only, so every
            # cross task is genuinely cross-modality with no fallback collapse.
            assert len(prov._avail[it["label_name"]]) == 2, m
            assert m["tgt_mod"] != m["ctx_mod"] and not m["fallback"], m
            cross_genuine += 1
        if prov._avail[it["label_name"]] == ["ct"]:
            assert it["modality"] == "ct" and m["ctx_mod"] == "ct", m
    tot = sum(regimes.values())
    print("train regime mix:", dict(regimes),
          f"  genuine-cross: {cross_genuine}/{regimes['cross']} "
          f"({cross_genuine / tot:.2%} of all tasks)")
    for r in ("ct", "mri", "cross"):
        assert abs(regimes[r] / tot - 1 / 3) < 0.06, (r, regimes)
    # F7: genuine cross-modality rate ~= regime_p[cross] (~1/3), not the old ~6.7%.
    assert cross_genuine == regimes["cross"], (cross_genuine, regimes)
    assert cross_genuine / tot > 0.25, (cross_genuine, tot)

    # --- eval dataset: determinism + class coverage ---
    ev1 = build_dataset(cfg, "val")
    ev2 = build_dataset(cfg, "val")
    assert ev1.cohort_mode and len(ev1) == cfg.data.source_mix.eval_epoch_length
    for idx in (0, 1, 7, 123, len(ev1) - 1):
        a, b = ev1[idx], ev2[idx]
        assert a["subject"] == b["subject"]
        assert a["context_subjects"] == b["context_subjects"]
        assert a["meta"] == b["meta"]
        assert torch.equal(a["context_in"], b["context_in"]), idx
    seen = Counter(ev1[i]["label_name"] for i in range(len(ev1)))
    print(f"eval covers {len(seen)}/{len(ev1.provider.classes)} classes")

    # Config-resolved val classes with no subjects in EITHER sub-provider: these are
    # dropped by MultiSourceProvider (e.g. kidney_cyst_left/right are legitimately
    # absent from CT label.npy). Informational — not asserted equal.
    val_classes = resolve_multisource_classes(cfg, "val")
    print("eval classes with no provider subjects:",
          sorted(set(val_classes) - set(ev1.provider.classes)))

    # hard: every both-modality class must be sampled at least once
    both = [c for c in ev1.provider.classes if len(ev1.provider._avail[c]) == 2]
    missing_both = [c for c in both if seen[c] == 0]
    assert not missing_both, (
        f"{len(missing_both)} both-modality eval classes never sampled: {missing_both[:10]}")

    # soft: overall class coverage >= 90%
    ratio = len(seen) / len(ev1.provider.classes)
    assert ratio >= 0.90, f"eval class coverage {ratio:.3f} < 0.90 (raise eval_epoch_length)"

    print("OK")


if __name__ == "__main__":
    main()

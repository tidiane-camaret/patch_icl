import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
from feature_sim.run import plan_sweep


def test_plan_sweep_mode_and_budget():
    rows = plan_sweep(["concat", "stage:0"], [16, 64], budget=48 ** 3, R=16)
    modes = {(r["tier"], r["res"]): r["mode"] for r in rows}
    assert modes[("concat", 16)] == "dense"       # 16^3 <= 48^3
    assert modes[("concat", 64)] == "point"       # 64^3 > 48^3
    assert len(rows) == 4


def test_plan_sweep_transformer_q_pinned_to_R():
    rows = plan_sweep(["transformer_q"], [16, 64], budget=10 ** 9, R=16)
    assert len(rows) == 1
    assert rows[0] == {"tier": "transformer_q", "res": 16, "mode": "dense"}


def test_rows_for_task_reads_batch_keys_and_shapes():
    import torch
    from omegaconf import OmegaConf
    from src.models.patchset3d import PatchSet3D
    from feature_sim.adapters import PatchSet3DEncoderAdapter
    from feature_sim.run import _rows_for_task
    from common import DEVICE

    torch.manual_seed(0)
    S, K = 16, 2
    # Model must live on DEVICE: _rows_for_task moves inputs to DEVICE (as _load_patchset
    # moves the model in production), so a CPU model would device-mismatch on a GPU node.
    model = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2,
                       thinking_rows=2, fourier_bands=4).to(DEVICE)
    adapter = PatchSet3DEncoderAdapter(model)
    gt = torch.zeros(1, S, S, S); gt[0, 4:12, 4:12, 4:12] = 1.0
    item = {
        "image": torch.randn(1, 1, S, S, S),
        "label": gt,
        "context_in": torch.randn(1, K, 1, S, S, S),
        "context_out": (torch.rand(1, K, S, S, S) > 0.5).float(),
        "label_names": ["liver"],
    }
    cfg = OmegaConf.create({"feature_sim": {"n_fg": 20, "n_bg": 20, "band": None}})
    plan = [{"tier": "concat", "res": 4, "mode": "dense"},
            {"tier": "concat", "res": 8, "mode": "point"}]
    gen = torch.Generator().manual_seed(0)
    rows = list(_rows_for_task(adapter, model, item, cfg, plan, input_res=S, gen=gen))
    assert len(rows) == 2
    assert all(r["class"] == "liver" for r in rows)          # "?" would mean the wrong batch key
    assert rows[0]["mode"] == "dense" and rows[1]["mode"] == "point"
    assert all("auroc" in r and "margin" in r and "retrieval_at1" in r for r in rows)
    # mode-specific columns are None (not "") so wandb.Table types them as optional Number:
    # dense has transfer_* but no ap; point has ap but no transfer_dice.
    assert rows[0]["ap"] is None and rows[1]["transfer_dice"] is None

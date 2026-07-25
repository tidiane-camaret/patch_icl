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

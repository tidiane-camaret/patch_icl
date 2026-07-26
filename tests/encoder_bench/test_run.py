import csv
import torch
from pathlib import Path
from encoder_bench import run as RUN


def test_sweep_writes_csv(tmp_path):
    rows = RUN.sweep(["conv_encoder3d"], [16], torch.device("cpu"),
                     tmp_path, n_warmup=1, n_timed=2)
    csv_path = tmp_path / "encoder_bench.csv"
    assert csv_path.exists()
    with open(csv_path) as fh:
        r = list(csv.DictReader(fh))
    assert r and r[0]["encoder"] == "conv_encoder3d"
    assert {"fwd_bwd_ms", "train_vram_mb", "gflops", "throughput_vol_s"} <= set(r[0])


def test_plot_curves_creates_png(tmp_path):
    rows = [{"encoder": "a", "family": "cnn", "input_size": 16, "fwd_bwd_ms": 1.0,
             "train_vram_mb": None, "gflops": 2.0, "throughput_vol_s": 3.0,
             "params": 10, "status": "ok"},
            {"encoder": "a", "family": "cnn", "input_size": 32, "fwd_bwd_ms": 4.0,
             "train_vram_mb": None, "gflops": 8.0, "throughput_vol_s": 1.0,
             "params": 10, "status": "ok"}]
    pngs = RUN.plot_curves(rows, tmp_path)
    assert any(p.exists() for p in pngs)

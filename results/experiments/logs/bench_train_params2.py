"""Follow-up: is the ~70 img/s ceiling GPU-compute or CPU/eager-scatter bound?

Compares, at B=64: (a) eager full_attn, (b) eager full_attn=false, (c) compiled
transformer (as the real run does). Prints GPU util sampled during the timed loop.
"""
import sys, time, threading, subprocess
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose

_ROOT = Path("/home/dpxuser/dev/patch_icl")
sys.path.insert(0, str(_ROOT / "experiments" / "2d")); sys.path.insert(0, str(_ROOT))
from train import build_model, _autocast
from pfn_train import Muon
import pfn_train

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
CFG_DIR = str(_ROOT / "configs" / "experiment" / "2d")


def make(overrides, compile_tf):
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="4_loss_eps_per_lvl", overrides=overrides)
    m, _, _ = build_model(cfg); m = m.to(DEVICE)
    if compile_tf and hasattr(m, "transformer"):
        m.transformer = torch.compile(m.transformer, dynamic=True)
        pfn_train._newtonschulz5_batched = torch.compile(pfn_train._newtonschulz5_batched)
    muon_p = [p for n, p in m.named_parameters() if p.requires_grad and p.ndim == 2 and "transformer" in n]
    adam_p = [p for n, p in m.named_parameters() if p.requires_grad and not (p.ndim == 2 and "transformer" in n)]
    opts = [torch.optim.AdamW(adam_p, lr=3e-4)]
    if muon_p: opts.append(Muon(muon_p, lr=3e-5, momentum=0.96, weight_decay=0.1))
    return cfg, m, opts


def run(tag, overrides, compile_tf, B=64):
    cfg, model, opts = make(overrides, compile_tf)
    H, K = cfg.data.image_size, cfg.data.context_size
    def step():
        img = torch.randn(B,1,H,H,device=DEVICE); cin = torch.randn(B,K,1,H,H,device=DEVICE)
        cout = (torch.rand(B,K,1,H,H,device=DEVICE)>0.7).float()
        for o in opts: o.zero_grad(set_to_none=True)
        with _autocast():
            out = model(img, context_in=cin, context_out=cout, mode="train")
            loss = out["final_logit"].float().pow(2).mean()
            if out.get("refine_logit") is not None: loss = loss + out["refine_logit"].float().pow(2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        for o in opts: o.step()
    for _ in range(6): step()          # warmup (compile traces here)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    utils=[]; stop=threading.Event()
    def poll():
        while not stop.is_set():
            try:
                u=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split("\n")[0]
                utils.append(int(u))
            except Exception: pass
            time.sleep(0.05)
    th=threading.Thread(target=poll); th.start()
    t0=time.perf_counter(); N=12
    for _ in range(N): step()
    torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/N
    stop.set(); th.join()
    peak=torch.cuda.max_memory_allocated()/1e9
    gu=sum(utils)/len(utils) if utils else -1
    print(f"{tag:>28}  it/s={1/dt:6.2f}  img/s={B/dt:6.1f}  peak={peak:5.1f}GB  gpu_util~{gu:4.0f}%")
    del model, opts; torch.cuda.empty_cache()


print("B=64, K=1, image_size=128\n")
run("eager full_attn=true", ["arch.full_attn=true","data.source=medsegbench"], False)
run("eager full_attn=false", ["arch.full_attn=false","data.source=medsegbench"], False)
run("compiled full_attn=true", ["arch.full_attn=true","arch.compile=true","data.source=medsegbench"], True)

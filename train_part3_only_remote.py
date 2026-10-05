import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import pandas as pd, torch
import config as CFG
CFG.EPOCHS = 30  # CRITICAL: keep all methods comparable at 30 epochs
import lib_models as M
import lib_train as T

train_tbl = pd.read_csv("/root/breast_sci_out/splits/train_breast.csv")
val_tbl = pd.read_csv("/root/breast_sci_out/splits/val_breast.csv")
machine_to_group, n_groups = M.build_machine_groups(train_tbl)
CKPT = "/root/breast_sci_out/checkpoints"

DEFAULT_KD = CFG.LAMBDA_KD
DEFAULT_P = CFG.MASK_PROB
JOBS = [
    ("P2_zero", "model_P2_zero_seed42.pt", {}, "missing-token=zero"),
    ("P2_dup",  "model_P2_dup_seed42.pt",  {}, "missing-token=duplicate"),
    ("P2",      "model_P2_kd0.1_seed42.pt", {"LAMBDA_KD": 0.1}, "lambda_KD=0.1"),
    ("P2",      "model_P2_kd0.5_seed42.pt", {"LAMBDA_KD": 0.5}, "lambda_KD=0.5"),
    ("P2",      "model_P2_p0.3_seed42.pt",  {"MASK_PROB": 0.3}, "mask_prob=0.3"),
    ("P2",      "model_P2_p0.7_seed42.pt",  {"MASK_PROB": 0.7}, "mask_prob=0.7"),
]
for method, fname, overrides, tag in JOBS:
    CFG.LAMBDA_KD = DEFAULT_KD
    CFG.MASK_PROB = DEFAULT_P
    for k, v in overrides.items():
        setattr(CFG, k, v)
    p = f"{CKPT}/{fname}"
    if os.path.exists(p):
        print(f"[skip] {tag}", flush=True); continue
    print(f"=== {tag} (method={method} KD={CFG.LAMBDA_KD} p={CFG.MASK_PROB} epochs={CFG.EPOCHS}) ===", flush=True)
    model = T.train_model(train_tbl, val_tbl, method, 42, machine_to_group)
    torch.save(model.state_dict(), p)
    print("[save]", p, flush=True)
    del model; torch.cuda.empty_cache()
print("PART3_ABLATION_SENSITIVITY DONE", flush=True)

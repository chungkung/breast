import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import pandas as pd, torch
import config as CFG
CFG.EPOCHS = 30
import lib_models as M
import lib_train as T

train_tbl = pd.read_csv("/root/breast_sci_out/splits/train_breast.csv")
val_tbl = pd.read_csv("/root/breast_sci_out/splits/val_breast.csv")
machine_to_group, n_groups = M.build_machine_groups(train_tbl)
CKPT = "/root/breast_sci_out/checkpoints"

# Part 3b: hyperparameter sensitivity (lambda1, KD_T) + B1 multi-seed
JOBS = [
    ("P2", "model_P2_lambda1_0.1_seed42.pt", {"LAMBDA_MASKED": 0.1}, "lambda1=0.1", 42),
    ("P2", "model_P2_lambda1_1.0_seed42.pt", {"LAMBDA_MASKED": 1.0}, "lambda1=1.0", 42),
    ("P2", "model_P2_T1.0_seed42.pt", {"KD_T": 1.0}, "KD_T=1.0", 42),
    ("P2", "model_P2_T4.0_seed42.pt", {"KD_T": 4.0}, "KD_T=4.0", 42),
    ("B1", "model_B1_seed2024.pt", {}, "B1 multi-seed", 2024),
    ("B1", "model_B1_seed7.pt", {}, "B1 multi-seed", 7),
]
DEFAULTS = {}
for k in ("LAMBDA_MASKED", "LAMBDA_KD", "MASK_PROB", "KD_T"):
    DEFAULTS[k] = getattr(CFG, k)
for method, fname, overrides, tag, seed in JOBS:
    for k in DEFAULTS:
        setattr(CFG, k, DEFAULTS[k])
    for k, v in overrides.items():
        setattr(CFG, k, v)
    p = f"{CKPT}/{fname}"
    if os.path.exists(p):
        print(f"[skip] {tag}", flush=True); continue
    print(f"=== {tag} (method={method} seed={seed} epochs={CFG.EPOCHS}) ===", flush=True)
    model = T.train_model(train_tbl, val_tbl, method, seed, machine_to_group)
    torch.save(model.state_dict(), p)
    print("[save]", p, flush=True)
    del model; torch.cuda.empty_cache()
print("PART3B DONE", flush=True)

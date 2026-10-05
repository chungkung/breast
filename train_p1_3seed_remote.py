import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import pandas as pd, torch
import config as CFG
CFG.EPOCHS = 30  # match all other methods
import lib_models as M
import lib_train as T
import lib_common as C

train_tbl = pd.read_csv("/root/breast_sci_out/splits/train_breast.csv")
val_tbl = pd.read_csv("/root/breast_sci_out/splits/val_breast.csv")
machine_to_group, n_groups = M.build_machine_groups(train_tbl)

for seed in (2024, 7):
    p = f"/root/breast_sci_out/checkpoints/model_P1_seed{seed}.pt"
    if os.path.exists(p):
        print("[skip]", p, flush=True); continue
    model = T.train_model(train_tbl, val_tbl, "P1", seed, machine_to_group)
    torch.save(model.state_dict(), p)
    print("[save]", p, flush=True)
    del model; torch.cuda.empty_cache()
print("P1_3SEED DONE", flush=True)

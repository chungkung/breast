import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import pandas as pd, torch
import config as CFG
CFG.EPOCHS = 30  # match all other methods (trained via --epochs 30)
import lib_models as M
import lib_train as T
import lib_common as C

train_tbl = pd.read_csv("/root/breast_sci_out/splits/train_breast.csv")
val_tbl = pd.read_csv("/root/breast_sci_out/splits/val_breast.csv")

for method in ("B0_CC", "B0_MLO"):
    col = "cc_image_id" if method == "B0_CC" else "mlo_image_id"
    sub = train_tbl[train_tbl[col].notna()]
    sub_val = val_tbl[val_tbl[col].notna()]
    for seed in (2024, 7):
        p = f"/root/breast_sci_out/checkpoints/model_{method}_seed{seed}.pt"
        if os.path.exists(p):
            print("[skip]", p, flush=True); continue
        model = T.train_model(sub, sub_val, method, seed, None)
        torch.save(model.state_dict(), p)
        print("[save]", p, flush=True)
        del model; torch.cuda.empty_cache()
print("B0_3SEED DONE", flush=True)

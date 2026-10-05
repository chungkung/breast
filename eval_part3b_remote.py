import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")
CKPT = "/root/breast_sci_out/checkpoints"

def eval_model(model):
    model.eval()
    loader = C.make_loader(test_tbl, train=False)
    ys = []; pf = []; pc = []; pm = []
    with torch.no_grad():
        for b in loader:
            cc = b["cc"].to(DEVICE); mlo = b["mlo"].to(DEVICE)
            y = b["y"].numpy()
            lg = model(cc, mlo, b["pcc"].to(DEVICE), b["pmlo"].to(DEVICE))
            pf.append(torch.sigmoid(lg).cpu().numpy())
            lg = model(cc, mlo, b["pcc"].to(DEVICE), torch.zeros_like(b["pmlo"].to(DEVICE)))
            pc.append(torch.sigmoid(lg).cpu().numpy())
            lg = model(cc, mlo, torch.zeros_like(b["pcc"].to(DEVICE)), b["pmlo"].to(DEVICE))
            pm.append(torch.sigmoid(lg).cpu().numpy())
            ys.append(y)
    y = np.concatenate(ys); pf = np.concatenate(pf); pc = np.concatenate(pc); pm = np.concatenate(pm)
    a_f = average_precision_score(y, pf); a_c = average_precision_score(y, pc); a_m = average_precision_score(y, pm)
    r_f = roc_auc_score(y, pf)
    D = a_f - 0.5 * (a_c + a_m)
    return a_f, a_c, a_m, r_f, D

print(f"{'checkpoint':32s} {'Full':>7s} {'CC':>7s} {'MLO':>7s} {'AUROC':>7s} {'D':>7s}", flush=True)

JOBS = [
    ("P2", "model_P2_seed42.pt",             "P2 default (l1=0.5, T=2.0)"),
    ("P2", "model_P2_lambda1_0.1_seed42.pt", "lambda1=0.1"),
    ("P2", "model_P2_lambda1_1.0_seed42.pt", "lambda1=1.0"),
    ("P2", "model_P2_T1.0_seed42.pt",        "KD_T=1.0"),
    ("P2", "model_P2_T4.0_seed42.pt",        "KD_T=4.0"),
    ("B1", "model_B1_seed2024.pt",           "B1 seed 2024"),
    ("B1", "model_B1_seed7.pt",              "B1 seed 7"),
]
for m, f, tag in JOBS:
    p = f"{CKPT}/{f}"
    if not os.path.exists(p):
        print(f"{f:32s} MISSING ({tag})", flush=True); continue
    mo = M.build_model(m, CFG).to(DEVICE)
    mo.load_state_dict(torch.load(p, map_location=DEVICE))
    a_f, a_c, a_m, r_f, D = eval_model(mo)
    print(f"{f:32s} {a_f:7.4f} {a_c:7.4f} {a_m:7.4f} {r_f:7.4f} {D:7.4f}", flush=True)
    del mo; torch.cuda.empty_cache()

print("EVAL_PART3B DONE", flush=True)

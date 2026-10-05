import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")

def pred(model, tbl):
    model.eval()
    loader = C.make_loader(tbl, train=False)
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            lg = model(b["cc"].to(DEVICE), b["mlo"].to(DEVICE),
                       b["pcc"].to(DEVICE), b["pmlo"].to(DEVICE))
            ys.append(b["y"].numpy())
            ps.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

for method in ("B0_CC", "B0_MLO"):
    auprcs, aurocs = [], []
    for seed in (42, 2024, 7):
        p = f"/root/breast_sci_out/checkpoints/model_{method}_seed{seed}.pt"
        if not os.path.exists(p):
            print(f"{method} seed={seed} MISSING", flush=True); continue
        model = M.build_model(method, CFG).to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        y, prob = pred(model, test_tbl)
        auprc = average_precision_score(y, prob)
        auroc = roc_auc_score(y, prob)
        auprcs.append(auprc); aurocs.append(auroc)
        print(f"{method} seed={seed}: AUPRC={auprc:.4f} AUROC={auroc:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    if auprcs:
        print(f"{method} MEAN: AUPRC={np.mean(auprcs):.4f} +-{np.std(auprcs):.4f}  "
              f"AUROC={np.mean(aurocs):.4f}  (seeds {[f'{x:.4f}' for x in auprcs]})", flush=True)
print("B0_EVAL DONE", flush=True)

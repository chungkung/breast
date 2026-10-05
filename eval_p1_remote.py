import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")

def pred(model, tbl, force_cc=True, force_mlo=True):
    model.eval()
    loader = C.make_loader(tbl, train=False)
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            pcc = b["pcc"].to(DEVICE) * (1.0 if force_cc else 0.0)
            pmlo = b["pmlo"].to(DEVICE) * (1.0 if force_mlo else 0.0)
            lg = model(b["cc"].to(DEVICE), b["mlo"].to(DEVICE), pcc, pmlo)
            ys.append(b["y"].numpy())
            ps.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

print("seed | AUPRC_full AUROC_full | CC    MLO   | D", flush=True)
auprcs, aurocs, ds = [], [], []
for seed in (42, 2024, 7):
    p = f"/root/breast_sci_out/checkpoints/model_P1_seed{seed}.pt"
    if not os.path.exists(p):
        print(seed, "MISSING", flush=True); continue
    model = M.build_model("P1", CFG).to(DEVICE)
    model.load_state_dict(torch.load(p, map_location=DEVICE))
    yf, pf = pred(model, test_tbl, True, True)
    yc, pc = pred(model, test_tbl, True, False)
    ym, pm = pred(model, test_tbl, False, True)
    full = average_precision_score(yf, pf); auroc = roc_auc_score(yf, pf)
    cc = average_precision_score(yc, pc); mlo = average_precision_score(ym, pm)
    D = full - 0.5*(cc+mlo)
    auprcs.append(full); aurocs.append(auroc); ds.append(D)
    print(f"{seed} | {full:.4f} {auroc:.4f} | {cc:.4f} {mlo:.4f} | {D:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
print(f"MEAN: AUPRC={np.mean(auprcs):.4f} +-{np.std(auprcs):.4f}  "
      f"AUROC={np.mean(aurocs):.4f}  D={np.mean(ds):.4f} +-{np.std(ds):.4f}  "
      f"seeds={[f'{x:.4f}' for x in auprcs]}", flush=True)
print("P1_EVAL DONE", flush=True)

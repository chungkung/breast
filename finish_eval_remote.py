import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, brier_score_loss
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

val_tbl = pd.read_csv("/root/breast_sci_out/splits/val_breast.csv")
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

def fit_T(y, p):
    z = np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1-1e-7))
    def obj(T):
        q = z / T
        s = 1/(1+np.exp(-q))
        return -np.mean(y*np.log(np.clip(s,1e-9,1-1e-9)) + (1-y)*np.log(np.clip(1-s,1e-9,1-1e-9)))
    return minimize_scalar(obj, bounds=(0.3, 6.0), method="bounded").x

def cal(p, T):
    z = np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1-1e-7))
    return 1/(1+np.exp(-z/T))

print("method | Full  CC    MLO   D     | T    ECE_raw ECE_cal Brier_raw Brier_cal", flush=True)
for method in ("B2", "P1", "P2", "P3"):
    p = f"/root/breast_sci_out/checkpoints/model_{method}_seed42.pt"
    if not os.path.exists(p):
        print(method, "no ckpt", flush=True); continue
    model = M.build_model(method, CFG).to(DEVICE)
    model.load_state_dict(torch.load(p, map_location=DEVICE))
    yf, pf = pred(model, test_tbl, True, True)
    yc, pc = pred(model, test_tbl, True, False)
    ym, pm = pred(model, test_tbl, False, True)
    full = average_precision_score(yf, pf); cc = average_precision_score(yc, pc)
    mlo = average_precision_score(ym, pm); D = full - 0.5*(cc+mlo)
    yv, pv = pred(model, val_tbl, True, True)
    T = fit_T(yv, pv)
    pf_cal = cal(pf, T)
    print(f"{method:6s} {full:.4f} {cc:.4f} {mlo:.4f} {D:.4f} | {T:.2f} "
          f"{C.expected_calibration_error(yf,pf):.4f} {C.expected_calibration_error(yf,pf_cal):.4f} "
          f"{brier_score_loss(yf,pf):.4f} {brier_score_loss(yf,pf_cal):.4f}", flush=True)
    del model; torch.cuda.empty_cache()
print("FINISH_EVAL DONE", flush=True)

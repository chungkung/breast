import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

val_tbl = pd.read_csv("/root/breast_sci_out/splits/val_breast.csv")
CKPT = "/root/breast_sci_out/checkpoints"
PREDS = "/root/breast_sci_out/results/preds"

def run_val_pfull(model, method):
    model.eval()
    loader = C.make_loader(val_tbl, train=False)
    ys = []; pfs = []
    with torch.no_grad():
        for b in loader:
            cc = b["cc"].to(DEVICE); mlo = b["mlo"].to(DEVICE)
            pcc = b["pcc"].to(DEVICE); pmlo = b["pmlo"].to(DEVICE)
            y = b["y"].numpy()
            if method in ("B0_CC", "B0_MLO"):
                p = torch.sigmoid(model(cc, mlo, pcc, pmlo)).cpu().numpy(); pf = p
            else:
                pf = torch.sigmoid(model(cc, mlo, pcc, pmlo)).cpu().numpy()
            pfs.append(pf); ys.append(y)
    return np.concatenate(ys), np.concatenate(pfs)

def op_from_thresholds(y_test, p_test, thr_spec, thr_sens):
    pred_spec = (p_test >= thr_spec).astype(int)
    tp = ((pred_spec == 1) & (y_test == 1)).sum(); fp = ((pred_spec == 1) & (y_test == 0)).sum()
    fn = ((pred_spec == 0) & (y_test == 1)).sum(); tn = ((pred_spec == 0) & (y_test == 0)).sum()
    sens = tp / (tp + fn + 1e-9); ppv = tp / (tp + fp + 1e-9); npv = tn / (tn + fn + 1e-9)
    pred_sens = (p_test >= thr_sens).astype(int)
    tn2 = ((pred_sens == 0) & (y_test == 0)).sum(); fp2 = ((pred_sens == 1) & (y_test == 0)).sum()
    spec = tn2 / (tn2 + fp2 + 1e-9)
    return sens, spec, ppv, npv

methods = [("B2", "model_B2_seed42.pt", "s42"), ("B2", "model_B2_seed2024.pt", "s2024"),
           ("B2", "model_B2_seed7.pt", "s7"),
           ("P2", "model_P2_seed42.pt", "s42"), ("P2", "model_P2_seed2024.pt", "s2024"),
           ("P2", "model_P2_seed7.pt", "s7")]

rows = {}
for method, fn, stag in methods:
    m = M.build_model(method, CFG).to(DEVICE)
    m.load_state_dict(torch.load(f"{CKPT}/{fn}", map_location=DEVICE))
    yv, pv = run_val_pfull(m, method)
    del m; torch.cuda.empty_cache()
    thr_spec = C._threshold_for_specificity(yv, pv, CFG.FIXED_SPECIFICITY)
    thr_sens = C._threshold_for_sensitivity(yv, pv, CFG.FIXED_SENSITIVITY)
    d = np.load(f"{PREDS}/{fn.replace('.pt', '.npz')}")
    yt, pt = d["y"], d["p_full"]
    sens, spec, ppv, npv = op_from_thresholds(yt, pt, thr_spec, thr_sens)
    base = fn.split("_seed")[0].replace("model_", "")
    rows.setdefault(base, []).append((sens, spec, ppv, npv))
    print(f"{base} {stag}: thr_spec={thr_spec:.4f} thr_sens={thr_sens:.4f} -> "
          f"Sens@90Spec={sens:.4f} Spec@80Sens={spec:.4f} PPV={ppv:.4f} NPV={npv:.4f}", flush=True)

print("\n--- 3-seed means (Table 2) ---", flush=True)
for base in ("B2", "P2"):
    arr = np.array(rows[base]); mm = arr.mean(0)
    print(f"{base}: Sens@90Spec={mm[0]:.4f} Spec@80Sens={mm[1]:.4f} PPV={mm[2]:.4f} NPV={mm[3]:.4f}", flush=True)

print("OPVAL_DONE", flush=True)

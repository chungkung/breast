import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score
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

print("method seed | Full  CC    MLO   D", flush=True)
rows = {}
for method in ("B2", "P2"):
    ds = []
    for seed in (42, 2024, 7):
        p = f"/root/breast_sci_out/checkpoints/model_{method}_seed{seed}.pt"
        if not os.path.exists(p):
            print(method, seed, "no ckpt", flush=True); continue
        model = M.build_model(method, CFG).to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        yf, pf = pred(model, test_tbl, True, True)
        yc, pc = pred(model, test_tbl, True, False)
        ym, pm = pred(model, test_tbl, False, True)
        full = average_precision_score(yf, pf); cc = average_precision_score(yc, pc)
        mlo = average_precision_score(ym, pm); D = full - 0.5*(cc+mlo)
        ds.append(D)
        print(f"{method} {seed} | {full:.4f} {cc:.4f} {mlo:.4f} {D:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    rows[method] = ds
    print(f"{method} D mean={np.mean(ds):.4f} +-{np.std(ds):.4f}  seeds={ds}", flush=True)
if "B2" in rows and "P2" in rows:
    mb, mp = np.mean(rows["B2"]), np.mean(rows["P2"])
    print(f"MEAN D: B2={mb:.4f} P2={mp:.4f}  reduction={(mb-mp)/mb*100:.1f}%", flush=True)
print("E4_3SEED DONE", flush=True)

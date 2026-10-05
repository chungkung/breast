import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")
CKPT = "/root/breast_sci_out/checkpoints"
OUT = "/root/breast_sci_out/preds"
os.makedirs(OUT, exist_ok=True)

def pred(model, tbl, fc=True, fm=True):
    model.eval()
    loader = C.make_loader(tbl, train=False)
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            pcc = b["pcc"].to(DEVICE) * (1.0 if fc else 0.0)
            pmlo = b["pmlo"].to(DEVICE) * (1.0 if fm else 0.0)
            lg = model(b["cc"].to(DEVICE), b["mlo"].to(DEVICE), pcc, pmlo)
            ys.append(b["y"].numpy()); ps.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

def load(m, s):
    mo = M.build_model(m, CFG).to(DEVICE)
    mo.load_state_dict(torch.load(f"{CKPT}/model_{m}_seed{s}.pt", map_location=DEVICE))
    return mo

# save predictions per seed
for method in ("B2", "P2"):
    for s in (42, 2024, 7):
        f = f"{OUT}/{method}_s{s}.npz"
        if os.path.exists(f):
            continue
        mo = load(method, s)
        y, pf = pred(mo, test_tbl, True, True)
        _, pc = pred(mo, test_tbl, True, False)
        _, pm = pred(mo, test_tbl, False, True)
        np.savez(f, y=y, pf=pf, pc=pc, pm=pm)
        print("saved", f, flush=True)
        del mo; torch.cuda.empty_cache()

# patient mapping
pid_col = "patient_id" if "patient_id" in test_tbl.columns else None
pids = test_tbl[pid_col].values if pid_col else np.arange(len(test_tbl))
uniq = np.unique(pids)
pid_to_idx = defaultdict(list)
for i, p in enumerate(pids):
    pid_to_idx[p].append(i)

def ensemble(method):
    pf = pc = pm = None
    for s in (42, 2024, 7):
        d = np.load(f"{OUT}/{method}_s{s}.npz")
        pf = d["pf"] if pf is None else pf + d["pf"]
        pc = d["pc"] if pc is None else pc + d["pc"]
        pm = d["pm"] if pm is None else pm + d["pm"]
        y = d["y"]
    return y, pf/3, pc/3, pm/3

yb, bf, bc, bm = ensemble("B2")
yp, pf_, pc_, pm_ = ensemble("P2")
assert np.array_equal(yb, yp)

def boot_idx(rng):
    sel = rng.choice(uniq, len(uniq), replace=True)
    idx = []
    for p in sel:
        idx.extend(pid_to_idx[p])
    return np.array(idx)

def run(metric):
    rng = np.random.RandomState(0)
    ds = []
    for _ in range(2000):
        idx = boot_idx(rng)
        yy = yb[idx]
        if metric == "AUPRC":
            a = average_precision_score(yy, bf[idx]); b = average_precision_score(yy, pf_[idx])
        elif metric == "AUROC":
            a = roc_auc_score(yy, bf[idx]); b = roc_auc_score(yy, pf_[idx])
        elif metric == "CC":
            a = average_precision_score(yy, bc[idx]); b = average_precision_score(yy, pc_[idx])
        elif metric == "MLO":
            a = average_precision_score(yy, bm[idx]); b = average_precision_score(yy, pm_[idx])
        elif metric == "D":
            a = average_precision_score(yy, bf[idx]) - 0.5*(average_precision_score(yy, bc[idx]) + average_precision_score(yy, bm[idx]))
            b = average_precision_score(yy, pf_[idx]) - 0.5*(average_precision_score(yy, pc_[idx]) + average_precision_score(yy, pm_[idx]))
        ds.append(b - a)
    ds = np.array(ds)
    print(f"{metric}: mean={ds.mean():+.4f} CI=[{np.percentile(ds,2.5):+.4f}, {np.percentile(ds,97.5):+.4f}]", flush=True)

for m in ("AUPRC", "AUROC", "CC", "MLO", "D"):
    run(m)
print("CI_CORRECT DONE", flush=True)

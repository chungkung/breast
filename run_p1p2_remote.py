import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
import lib_train as T
from lib_train import DEVICE

test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")
CKPT = "/root/breast_sci_out/checkpoints"


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


def load(method, seed):
    model = M.build_model(method, CFG).to(DEVICE)
    p = f"{CKPT}/model_{method}_seed{seed}.pt"
    model.load_state_dict(torch.load(p, map_location=DEVICE))
    return model


# ============================================================
# Part 1: B1 late-fusion missing-view degradation (seed 42)
# ============================================================
print("=== Part 1: B1 degradation ===", flush=True)
m = load("B1", 42)
yf, pf = pred(m, test_tbl, True, True)
yc, pc = pred(m, test_tbl, True, False)
ym, pm = pred(m, test_tbl, False, True)
b1_full = average_precision_score(yf, pf)
b1_cc = average_precision_score(yc, pc)
b1_mlo = average_precision_score(ym, pm)
b1_D = b1_full - 0.5 * (b1_cc + b1_mlo)
print(f"B1: full={b1_full:.4f} CC={b1_cc:.4f} MLO={b1_mlo:.4f} D={b1_D:.4f} "
      f"AUROC_full={roc_auc_score(yf, pf):.4f}", flush=True)
del m; torch.cuda.empty_cache()

# ============================================================
# Part 2: paired patient-bootstrap CIs for B2 vs P2 (3 seeds)
# ============================================================
print("=== Part 2: paired bootstrap CI ===", flush=True)
pid_col = "patient_id" if "patient_id" in test_tbl.columns else None
if pid_col:
    pids = test_tbl[pid_col].values
    uniq = np.unique(pids)
else:
    pids = np.arange(len(test_tbl))
    uniq = pids.copy()


def collect(method, seeds=(42, 2024, 7)):
    fulls, ccs, mlos = [], [], []
    for s in seeds:
        m = load(method, s)
        yf, pf = pred(m, test_tbl, True, True)
        yc, pc = pred(m, test_tbl, True, False)
        ym, pm = pred(m, test_tbl, False, True)
        fulls.append(pf); ccs.append(pc); mlos.append(pm)
        del m; torch.cuda.empty_cache()
    return yf, np.mean(fulls, 0), np.mean(ccs, 0), np.mean(mlos, 0)


yb, bf, bc, bm = collect("B2")
yp, pf_, pc_, pm_ = collect("P2")
assert np.array_equal(yb, yp)


def boot_pair(a_f, a_c, a_m, b_f, b_c, b_m, y, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(uniq)
    out = {}
    # deltas: P2 - B2
    d_auprc, d_auroc, d_D, d_cc, d_mlo = [], [], [], [], []
    for _ in range(n_boot):
        sel = rng.choice(uniq, n, replace=True)
        idx = np.isin(pids, sel)
        if idx.sum() < 2:
            continue
        yy = y[idx]
        a_auprc = average_precision_score(yy, a_f[idx])
        b_auprc = average_precision_score(yy, b_f[idx])
        d_auprc.append(b_auprc - a_auprc)
        d_auroc.append(roc_auc_score(yy, b_f[idx]) - roc_auc_score(yy, a_f[idx]))
        a_D = average_precision_score(yy, a_f[idx]) - 0.5 * (
            average_precision_score(yy, a_c[idx]) + average_precision_score(yy, a_m[idx]))
        b_D = average_precision_score(yy, b_f[idx]) - 0.5 * (
            average_precision_score(yy, b_c[idx]) + average_precision_score(yy, b_m[idx]))
        d_D.append(b_D - a_D)
        d_cc.append(average_precision_score(yy, b_c[idx]) - average_precision_score(yy, a_c[idx]))
        d_mlo.append(average_precision_score(yy, b_m[idx]) - average_precision_score(yy, a_m[idx]))
    for name, arr in [("AUPRC_full", d_auprc), ("AUROC_full", d_auroc), ("D", d_D),
                      ("AUPRC_CC", d_cc), ("AUPRC_MLO", d_mlo)]:
        out[name] = (np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5))
        print(f"  delta {name}: mean={np.mean(arr):+.4f}  CI=[{np.percentile(arr,2.5):+.4f}, "
              f"{np.percentile(arr,97.5):+.4f}]", flush=True)
    return out


boot_pair(bf, bc, bm, pf_, pc_, pm_, yb)
print("=== CI DONE ===", flush=True)

# ============================================================
# Part 3: missing-token ablation + hyperparameter sensitivity
# ============================================================
train_tbl = pd.read_csv("/root/breast_sci_out/splits/train_breast.csv")
val_tbl = pd.read_csv("/root/breast_sci_out/splits/val_breast.csv")
machine_to_group, n_groups = M.build_machine_groups(train_tbl)

DEFAULT_KD = CFG.LAMBDA_KD
DEFAULT_P = CFG.MASK_PROB
JOBS = [
    ("P2_zero", "model_P2_zero_seed42.pt", {}, "missing-token=zero"),
    ("P2_dup",  "model_P2_dup_seed42.pt",  {}, "missing-token=duplicate"),
    ("P2",      "model_P2_kd0.1_seed42.pt", {"LAMBDA_KD": 0.1}, "lambda_KD=0.1"),
    ("P2",      "model_P2_kd0.5_seed42.pt", {"LAMBDA_KD": 0.5}, "lambda_KD=0.5"),
    ("P2",      "model_P2_p0.3_seed42.pt",  {"MASK_PROB": 0.3}, "mask_prob=0.3"),
    ("P2",      "model_P2_p0.7_seed42.pt",  {"MASK_PROB": 0.7}, "mask_prob=0.7"),
]
for method, fname, overrides, tag in JOBS:
    CFG.LAMBDA_KD = DEFAULT_KD
    CFG.MASK_PROB = DEFAULT_P
    for k, v in overrides.items():
        setattr(CFG, k, v)
    p = f"{CKPT}/{fname}"
    if os.path.exists(p):
        print(f"[skip] {tag}", flush=True); continue
    print(f"=== {tag} (method={method} KD={CFG.LAMBDA_KD} p={CFG.MASK_PROB}) ===", flush=True)
    model = T.train_model(train_tbl, val_tbl, method, 42, machine_to_group)
    torch.save(model.state_dict(), p)
    print("[save]", p, flush=True)
    del model; torch.cuda.empty_cache()
print("ABLATION_SENSITIVITY DONE", flush=True)

import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

SMOKE = os.environ.get("SMOKE") == "1"

test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")
pids_all = test_tbl.patient_id.values.astype(np.int64)
CKPT = "/root/breast_sci_out/checkpoints"
OUT = "/root/breast_sci_out/results/preds"
os.makedirs(OUT, exist_ok=True)

# (method_for_build, checkpoint_filename, tag)
CHECKPOINTS = [
    ("B2","model_B2_seed42.pt","B2 s42"), ("B2","model_B2_seed2024.pt","B2 s2024"), ("B2","model_B2_seed7.pt","B2 s7"),
    ("P1","model_P1_seed42.pt","P1 s42"), ("P1","model_P1_seed2024.pt","P1 s2024"), ("P1","model_P1_seed7.pt","P1 s7"),
    ("P2","model_P2_seed42.pt","P2 s42"), ("P2","model_P2_seed2024.pt","P2 s2024"), ("P2","model_P2_seed7.pt","P2 s7"),
    ("P3","model_P3_seed42.pt","P3 s42"),
    ("B0_CC","model_B0_CC_seed42.pt","B0_CC s42"), ("B0_CC","model_B0_CC_seed2024.pt","B0_CC s2024"), ("B0_CC","model_B0_CC_seed7.pt","B0_CC s7"),
    ("B0_MLO","model_B0_MLO_seed42.pt","B0_MLO s42"), ("B0_MLO","model_B0_MLO_seed2024.pt","B0_MLO s2024"), ("B0_MLO","model_B0_MLO_seed7.pt","B0_MLO s7"),
    ("B1","model_B1_seed42.pt","B1 s42"), ("B1","model_B1_seed2024.pt","B1 s2024"), ("B1","model_B1_seed7.pt","B1 s7"),
    ("P2_zero","model_P2_zero_seed42.pt","P2_zero"),
    ("P2_dup","model_P2_dup_seed42.pt","P2_dup"),
    ("P2","model_P2_kd0.1_seed42.pt","P2 kd0.1"),
    ("P2","model_P2_kd0.5_seed42.pt","P2 kd0.5"),
    ("P2","model_P2_lambda1_0.1_seed42.pt","P2 l1=0.1"),
    ("P2","model_P2_lambda1_1.0_seed42.pt","P2 l1=1.0"),
    ("P2","model_P2_T1.0_seed42.pt","P2 T1.0"),
    ("P2","model_P2_T4.0_seed42.pt","P2 T4.0"),
    ("P2","model_P2_p0.3_seed42.pt","P2 p0.3"),
    ("P2","model_P2_p0.7_seed42.pt","P2 p0.7"),
]

if SMOKE:
    CHECKPOINTS = CHECKPOINTS[:9]  # B2 x3, P1 x3, P2 x3

def run_preds(model, method):
    model.eval()
    loader = C.make_loader(test_tbl, train=False)
    ys=[]; pfs=[]; pcs=[]; pms=[]; pids_l=[]; macs=[]
    idx = 0
    with torch.no_grad():
        for b in loader:
            n = int(b["y"].size(0))
            cc=b["cc"].to(DEVICE); mlo=b["mlo"].to(DEVICE)
            pcc=b["pcc"].to(DEVICE); pmlo=b["pmlo"].to(DEVICE)
            y = b["y"].numpy()
            if method in ("B0_CC","B0_MLO"):
                p = torch.sigmoid(model(cc,mlo,pcc,pmlo)).cpu().numpy()
                pf=pc=pm=p
            else:
                pf = torch.sigmoid(model(cc,mlo,pcc,pmlo)).cpu().numpy()
                pc = torch.sigmoid(model(cc,mlo,pcc,torch.zeros_like(pmlo))).cpu().numpy()
                pm = torch.sigmoid(model(cc,mlo,torch.zeros_like(pcc),pmlo)).cpu().numpy()
            pfs.append(pf); pcs.append(pc); pms.append(pm); ys.append(y)
            pids_l.append(pids_all[idx:idx+n]); macs.append(b["machine"].numpy())
            idx += n
    y = np.concatenate(ys); pf=np.concatenate(pfs); pc=np.concatenate(pcs); pm=np.concatenate(pms)
    pid = np.concatenate(pids_l); mac = np.concatenate(macs)
    assert len(y) == len(test_tbl), f"length mismatch {len(y)} vs {len(test_tbl)}"
    return y, pid, mac, pf, pc, pm

print(f"{'ckpt':34s} {'Full':>7s} {'CC':>7s} {'MLO':>7s} {'AUROC':>7s} {'D':>7s}", flush=True)
for method, fn, tag in CHECKPOINTS:
    p = f"{CKPT}/{fn}"
    out_npz = f"{OUT}/{fn.replace('.pt','.npz')}"
    if os.path.exists(out_npz):
        print(f"{fn:34s} SKIP (already saved)", flush=True); continue
    if not os.path.exists(p):
        print(f"{fn:34s} MISSING ({tag})", flush=True); continue
    mo = M.build_model(method, CFG).to(DEVICE)
    mo.load_state_dict(torch.load(p, map_location=DEVICE))
    y, pid, mac, pf, pc, pm = run_preds(mo, method)
    np.savez_compressed(out_npz, y=y, pid=pid, machine=mac, p_full=pf, p_cc=pc, p_mlo=pm)
    a_f=average_precision_score(y,pf); a_c=average_precision_score(y,pc); a_m=average_precision_score(y,pm)
    r_f=roc_auc_score(y,pf); D=a_f-0.5*(a_c+a_m)
    print(f"{fn:34s} {a_f:7.4f} {a_c:7.4f} {a_m:7.4f} {r_f:7.4f} {D:7.4f}", flush=True)
    del mo; torch.cuda.empty_cache()

print("UNIFIED_EVAL DONE", flush=True)

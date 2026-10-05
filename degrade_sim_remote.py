import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
import config as CFG
import lib_common as C
import lib_models as M
from lib_train import DEVICE

# Distribution-shift simulation: real "missing" views are often corrupted rather than
# cleanly absent. We feed a Gaussian-noise-corrupted version of the would-be-missing view
# (still marked present) and check whether P2's advantage over B2 is maintained.
test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")
CKPT = "/root/breast_sci_out/checkpoints"

# Fixed RNG seed for the corruption noise. Both methods are re-seeded with this
# value before their run, so B2 and P2 consume an IDENTICAL Gaussian-noise stream
# per condition (fair comparison) and the numbers are reproducible.
NOISE_SEED = 0

def load(m, s=42):
    mo = M.build_model(m, CFG).to(DEVICE)
    mo.load_state_dict(torch.load(f"{CKPT}/model_{m}_seed{s}.pt", map_location=DEVICE))
    mo.eval()
    return mo

def run(mo, noise_cc=0.0, noise_mlo=0.0, fc=True, fm=True):
    loader = C.make_loader(test_tbl, train=False)
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            cc = b["cc"].to(DEVICE)
            mlo = b["mlo"].to(DEVICE)
            if noise_cc > 0:
                cc = cc + torch.randn_like(cc) * noise_cc
            if noise_mlo > 0:
                mlo = mlo + torch.randn_like(mlo) * noise_mlo
            pcc = b["pcc"].to(DEVICE) * (1.0 if fc else 0.0)
            pmlo = b["pmlo"].to(DEVICE) * (1.0 if fm else 0.0)
            lg = mo(cc, mlo, pcc, pmlo)
            ys.append(b["y"].numpy()); ps.append(torch.sigmoid(lg).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return average_precision_score(y, p), roc_auc_score(y, p)

for m in ("B2", "P2"):
    mo = load(m)
    # Re-seed identically before each method so B2 and P2 consume the SAME
    # noise stream per condition. The loader order is deterministic
    # (shuffle=False), so a matching seed gives a matching noise sequence.
    torch.manual_seed(NOISE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(NOISE_SEED)
    print(f"===== {m} =====", flush=True)
    for name, (nc, nm, fc, fm) in {
        "Full (clean)":              (0.0, 0.0, True, True),
        "CC-only (MLO->token)":      (0.0, 0.0, True, False),
        "MLO-only (CC->token)":      (0.0, 0.0, False, True),
        "CC + MLO corrupted (s=0.3)":(0.0, 0.3, True, True),
        "MLO + CC corrupted (s=0.3)":(0.3, 0.0, True, True),
        "CC-only + MLO corrupted":   (0.0, 0.3, True, False),
        "MLO-only + CC corrupted":   (0.3, 0.0, False, True),
    }.items():
        a, r = run(mo, nc, nm, fc, fm)
        print(f"  {name:32s} AUPRC={a:.4f} AUROC={r:.4f}", flush=True)
    del mo; torch.cuda.empty_cache()
print("DEGRADE_SIM DONE", flush=True)

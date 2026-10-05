# =============================================================================
# Missing-rate simulation (Table 13 / Fig. S2): robustness to randomly missing views.
#
# A randomly selected (independent) fraction rho of breasts loses one view, with
# CC and MLO equally likely; each rate uses 200 simulations with a FIXED generator.
# For a breast that loses MLO we use the model's CC-only prediction (p_cc), for a
# breast that loses CC we use its MLO-only prediction (p_mlo); these single-view
# predictions already embody the learned/zero missing-token substitution, because
# they are produced with the dropped view's presence flag set to 0.  Breasts that
# keep both views contribute p_full.  Predictions are the 3-seed ensemble mean.
#
# NOTE on provenance: the original driver for this table was not saved on the
# compute node, so this script is the reconstruction used to regenerate Table 13
# and Figure S2.  The manuscript's Table 13 and the +0.040/+0.040/+0.039/+0.037/
# +0.034 diff string in the text now report exactly the values this script prints
# with the fixed generator seed 0 (P2 - B2 AUPRC = +0.0395/+0.0402/+0.0392/+0.0367/
# +0.0337 at rates 0/0.2/0.4/0.6/0.8).  Change RNG_SEED only to reproduce a
# different draw.
# =============================================================================
import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PRED_DIR = "/root/breast_sci_out/results/preds"
SEEDS = (42, 2024, 7)
RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
N_SIMS = 200
RNG_SEED = 0


def load(method, seed):
    d = np.load(os.path.join(PRED_DIR, f"model_{method}_seed{seed}.npz"))
    return d["y"], d["pid"], d["p_full"], d["p_cc"], d["p_mlo"]


def ensemble(method, key):
    arr = []
    for s in SEEDS:
        y, pid, pf, pc, pm = load(method, s)
        arr.append({"pf": pf, "pc": pc, "pm": pm}[key])
    return np.mean(arr, axis=0)


y = load("B2", SEEDS[0])[0]
preds = {m: {"pf": ensemble(m, "pf"), "pc": ensemble(m, "pc"), "pm": ensemble(m, "pm")}
         for m in ("B2", "P2")}
N = len(y)


def simulate(method):
    pf, pc, pm = preds[method]["pf"], preds[method]["pc"], preds[method]["pm"]
    out = []
    for rho in RATES:
        rng = np.random.RandomState(RNG_SEED)  # fixed generator, reset per rate
        aps, aucs = [], []
        for _ in range(N_SIMS):
            lose = rng.rand(N) < rho                # independent fraction rho
            lose_cc = lose & (rng.rand(N) < 0.5)    # ... which loses CC (use MLO)
            lose_mlo = lose & (~lose_cc)            # ... which loses MLO (use CC)
            pp = np.where(lose_mlo, pc, np.where(lose_cc, pm, pf))
            aps.append(average_precision_score(y, pp))
            aucs.append(roc_auc_score(y, pp))
        out.append((float(np.mean(aps)), float(np.mean(aucs))))
    return out


b2 = simulate("B2")
p2 = simulate("P2")
print("rate            :", "  ".join(f"{r:.0%}" for r in RATES), flush=True)
print("B2 mean AUPRC   :", "  ".join(f"{v[0]:.4f}" for v in b2), flush=True)
print("P2 mean AUPRC   :", "  ".join(f"{v[0]:.4f}" for v in p2), flush=True)
print("P2 - B2 (AUPRC) :", "  ".join(f"{a[0]-b[0]:+.4f}" for a, b in zip(p2, b2)), flush=True)
print("B2 mean AUROC   :", "  ".join(f"{v[1]:.4f}" for v in b2), flush=True)
print("P2 mean AUROC   :", "  ".join(f"{v[1]:.4f}" for v in p2), flush=True)
print("P2 - B2 (AUROC) :", "  ".join(f"{a[1]-b[1]:+.4f}" for a, b in zip(p2, b2)), flush=True)
print("MISSING_RATE DONE", flush=True)

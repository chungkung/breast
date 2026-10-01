# =============================================================================
# SUPERSEDED -- do not use for the manuscript's primary statistics.
#
# Reason 1 (sidedness): pvals = (diffs[n] <= 0).mean() is a ONE-sided tail
#   probability (numpy.random.RandomState(0)). The canonical script
#   unified_metrics_remote.py uses a TWO-sided probability. Mixing the two
#   within one Benjamini-Hochberg family is not valid.
# Reason 2 (inputs): this script reads /root/breast_sci_out/preds/, which holds
#   only 6 files (B2 and P2, three seeds each). The manuscript's statistics use
#   results/preds/*.npz (29 files).
#
# The three AUROC p-values this script yields (0.0005 / 0.0060 / 0.0010) were
# carried into an earlier draft; they are NOT the values in the current
# manuscript. Retained only as a record of the one-sided computation.
# =============================================================================
import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = "/root/breast_sci_out/preds"
test_tbl = pd.read_csv("/root/breast_sci_out/splits/test_breast.csv")
pids = test_tbl["patient_id"].values
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
    return y, pf / 3, pc / 3, pm / 3

yb, bf, bc, bm = ensemble("B2")
yp, pf_, pc_, pm_ = ensemble("P2")
assert np.array_equal(yb, yp)
y = yb

def boot_idx(rng):
    sel = rng.choice(uniq, len(uniq), replace=True)
    idx = []
    for p in sel:
        idx.extend(pid_to_idx[p])
    return np.array(idx)

# name -> (b_pred, p_pred, metric) ; metric "auroc" or "auprc"
comparisons = [
    ("full AUROC",  bf, pf_, "auroc"),
    ("full AUPRC",  bf, pf_, "auprc"),
    ("CC AUROC",    bc, pc_, "auroc"),
    ("CC AUPRC",    bc, pc_, "auprc"),
    ("MLO AUROC",   bm, pm_, "auroc"),
    ("MLO AUPRC",   bm, pm_, "auprc"),
]

n_boot = 2000
diffs = {name: np.zeros(n_boot) for name, _, _, _ in comparisons}
rng = np.random.RandomState(0)
for i in range(n_boot):
    idx = boot_idx(rng)
    yy = y[idx]
    for name, a, b, met in comparisons:
        ai, bi = a[idx], b[idx]
        if met == "auroc":
            d = roc_auc_score(yy, bi) - roc_auc_score(yy, ai)
        else:
            d = average_precision_score(yy, bi) - average_precision_score(yy, ai)
        diffs[name][i] = d
    if (i + 1) % 400 == 0:
        print("  boot %d/%d" % (i + 1, n_boot), flush=True)

names = [c[0] for c in comparisons]
pvals = np.array([(diffs[n] <= 0).mean() for n in names])

# Benjamini-Hochberg
order = np.argsort(pvals)
sorted_p = pvals[order]
m = len(pvals)
q = np.empty(m)
run = 1.0
for i in range(m - 1, -1, -1):
    cand = sorted_p[i] * m / (i + 1)
    if cand < run:
        run = cand
    q[i] = run
q_orig = np.empty(m)
for rank, orig in enumerate(order):
    q_orig[orig] = q[rank]

print("=== 6 primary P2-vs-B2 p-values (patient-level paired bootstrap, 2000x, 3-seed ensemble) ===", flush=True)
print("%-12s %-12s %-12s %-8s %-24s" % ("comparison", "raw_p", "BH_q", "sig", "CI"), flush=True)
for i, n in enumerate(names):
    d = diffs[n]
    ci = "[%+.4f, %+.4f]" % (np.percentile(d, 2.5), np.percentile(d, 97.5))
    print("%-12s %-12.4f %-12.4f %-8s %-24s" % (n, pvals[i], q_orig[i], "YES" if q_orig[i] < 0.05 else "no", ci), flush=True)
print("DONE_6PVALS", flush=True)

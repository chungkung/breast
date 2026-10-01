# =============================================================================
# CANONICAL statistics script for the six primary P2-vs-B2 comparisons.
#
# This is the script that produced the p- and q-values reported in the manuscript
# (Section 3.5 and Table 2 caption):
#     full-view AUPRC  p = 0.030  q = 0.036
#     CC-only  AUPRC   p = 0.231  q = 0.231
#     MLO-only AUPRC   p = 0.003  q = 0.006
#     full-view AUROC  p < 0.001  q = 0.003
#     CC-only  AUROC   p = 0.011  q = 0.017
#     MLO-only AUROC   p < 0.001  q = 0.003
#
# Conventions (ONE convention for all six comparisons):
#   * input          : results/preds/model_<method>_seed<seed>.npz  (3-seed ensemble mean)
#   * resampling     : 2,000 patient-level paired bootstrap resamples
#   * RNG            : numpy.random.default_rng(0)
#   * sidedness      : TWO-sided, p = 2 * min(P(delta > 0), P(delta < 0))
#   * multiplicity   : Benjamini-Hochberg at 0.05 (see bh_correction.py)
#
# Do NOT mix these values with compute_6_pvals_remote.py, which uses a one-sided
# tail probability and a different prediction directory. See that file's header.
# =============================================================================
import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import lib_common as C

PRED_DIR = "/root/breast_sci_out/results/preds"
RES_DIR = "/root/breast_sci_out/results"
N_BOOT = 2000
RNG_SEED = 0

# ---- load all saved predictions ----
FILES = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".npz"))
print(f"Loaded {len(FILES)} prediction files", flush=True)

def load(fn):
    d = np.load(os.path.join(PRED_DIR, fn))
    return dict(y=d["y"], pid=d["pid"], machine=d["machine"],
                pf=d["p_full"], pc=d["p_cc"], pm=d["p_mlo"])

DATA = {fn[:-4]: load(fn) for fn in FILES}

def ap(y, p): return average_precision_score(y, p)
def roc(y, p): return roc_auc_score(y, p)

# ---- per-checkpoint metrics ----
rows = []
for name, d in DATA.items():
    y = d["y"]
    row = dict(name=name,
        Full_AP=ap(y, d["pf"]), CC_AP=ap(y, d["pc"]), MLO_AP=ap(y, d["pm"]),
        Full_AUC=roc(y, d["pf"]),
        D=ap(y, d["pf"]) - 0.5*(ap(y, d["pc"]) + ap(y, d["pm"])),
        pF1=C.pfbeta(y, d["pf"]), Brier=C.all_metrics(y, d["pf"])["Brier"],
        ECE=C.all_metrics(y, d["pf"])["ECE"])
    rows.append(row)
per_ckpt = pd.DataFrame(rows)
per_ckpt.to_csv(f"{RES_DIR}/unified_per_checkpoint.csv", index=False)
print("\n===== per-checkpoint (Full/CC/MLO AUPRC, AUROC, D) =====", flush=True)
print(per_ckpt.to_string(index=False), flush=True)

# ---- patient-level bootstrap machinery (multiplicity-corrected) ----
def make_booter(pids):
    pats, pat_inv = np.unique(pids, return_inverse=True)
    n_pat = len(pats)
    pat_rows = {}
    for i, p in enumerate(pat_inv):
        pat_rows.setdefault(int(p), []).append(i)
    pat_rows = {p: np.array(v, dtype=int) for p, v in pat_rows.items()}
    rng = np.random.default_rng(RNG_SEED)
    def gen():
        bp = rng.integers(0, n_pat, n_pat)
        return np.concatenate([pat_rows[int(p)] for p in bp])
    return gen

def paired_ci(pids, y, p_a, p_b, metric=ap):
    """95% CI + mean of metric(p_a)-metric(p_b) under patient bootstrap."""
    gen = make_booter(pids)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = gen()
        out[b] = metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])
    lo, hi = np.percentile(out, [2.5, 97.5])
    pval = 2 * min(float((out > 0).mean()), float((out < 0).mean()))
    return out.mean(), lo, hi, pval

# ---- seed groups ----
def seeds_of(method):
    return [f"model_{method}_seed{s}" for s in (42, 2024, 7)]

def mean3(names, key):
    vals = [DATA[n][key] for n in names if n in DATA]
    return None if not vals else np.mean(vals, axis=0)

def ap_mean3(names, key):
    """AUPRC of the seed-mean prediction (ensemble-of-3)."""
    ps = [DATA[n][key] for n in names if n in DATA]
    if not ps: return float("nan")
    y = DATA[names[0]]["y"]
    return ap(y, np.mean(ps, axis=0))

print("\n===== 3-seed summary (metric-mean | ensemble-mean AUPRC) =====", flush=True)
summary = []
for m in ["B0_CC","B0_MLO","B1","B2","P1","P2"]:
    ns = seeds_of(m)
    if not all(n in DATA for n in ns):
        print(f"{m}: missing seeds {[n for n in ns if n not in DATA]}", flush=True); continue
    y = DATA[ns[0]]["y"]
    def metric_mean(key):
        return float(np.mean([ap(y, DATA[n][key]) for n in ns]))
    row = dict(method=m,
        Full_AP_mean=metric_mean("pf"), Full_AP_ens=ap_mean3(ns, "pf"),
        CC_AP_mean=metric_mean("pc"), CC_AP_ens=ap_mean3(ns, "pc"),
        MLO_AP_mean=metric_mean("pm"), MLO_AP_ens=ap_mean3(ns, "pm"),
        AUC_mean=float(np.mean([roc(y, DATA[n]["pf"]) for n in ns])))
    summary.append(row)
    print(row, flush=True)
pd.DataFrame(summary).to_csv(f"{RES_DIR}/unified_seed_means.csv", index=False)

# ---- paired bootstrap: P2 vs B2, P2 vs P1, P1 vs B2 (seed 42 & ensemble-of-3) ----
print("\n===== paired bootstrap (patient-level, 2000x) =====", flush=True)
y = DATA["model_B2_seed42"]["y"]
pids = DATA["model_B2_seed42"]["pid"]
pa_rows = []
def report(tag, pa, pb, key):
    m, lo, hi, pv = paired_ci(pids, y, pa, pb)
    ma, loa, hia, pva = paired_ci(pids, y, pa, pb, metric=roc)
    print(f"{tag:28s} AP mean={m:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  p={pv:.4f}", flush=True)
    print(f"{'':28s} AUC mean={ma:+.4f}  95%CI=[{loa:+.4f},{hia:+.4f}]  p={pva:.4f}", flush=True)
    pa_rows.append(dict(tag=tag, AP_mean=m, AP_lo=lo, AP_hi=hi, AP_p=pv,
                        AUC_mean=ma, AUC_lo=loa, AUC_hi=hia, AUC_p=pva))

# seed 42 (single-seed, the direct P2 vs P1 vs B2 comparison)
for key, label in [("pf","Full"),("pc","CC"),("pm","MLO")]:
    report(f"P2-P1 {label} (seed42)", DATA["model_P2_seed42"][key], DATA["model_P1_seed42"][key], key)
    report(f"P2-B2 {label} (seed42)", DATA["model_P2_seed42"][key], DATA["model_B2_seed42"][key], key)
    report(f"P1-B2 {label} (seed42)", DATA["model_P1_seed42"][key], DATA["model_B2_seed42"][key], key)

# ensemble-of-3 (average prediction across seeds)
for key, label in [("pf","Full"),("pc","CC"),("pm","MLO")]:
    p2e = mean3(seeds_of("P2"), key); p1e = mean3(seeds_of("P1"), key); b2e = mean3(seeds_of("B2"), key)
    report(f"P2-P1 {label} (ens3)", p2e, p1e, key)
    report(f"P2-B2 {label} (ens3)", p2e, b2e, key)
    report(f"P1-B2 {label} (ens3)", p1e, b2e, key)
pd.DataFrame(pa_rows).to_csv(f"{RES_DIR}/unified_paired_bootstrap.csv", index=False)

print("\nUNIFIED_METRICS DONE", flush=True)

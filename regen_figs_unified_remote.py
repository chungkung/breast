import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
import lib_common as C

PRED = "/root/breast_sci_out/results/preds"
FIG = "/root/breast_sci_out/figures"
os.makedirs(FIG, exist_ok=True)

def load(name):
    d = np.load(f"{PRED}/{name}.npz")
    return dict(y=d["y"], pid=d["pid"], machine=d["machine"],
                p_full=d["p_full"], p_cc=d["p_cc"], p_mlo=d["p_mlo"])

def seeds(m): return [f"model_{m}_seed{s}" for s in (42, 2024, 7)]

def ap(y, p): return average_precision_score(y, p)
def roc(y, p): return roc_auc_score(y, p)

# metric-mean (mean of per-seed AUPRC/AUROC) -- matches the tables
def metric_mean(m, key):
    y = load(seeds(m)[0])["y"]
    return float(np.mean([ap(y, load(n)[key]) for n in seeds(m)]))

# ensemble (3-seed averaged prediction) -- only for ROC/PR curves
def ens_pred(m, key):
    return np.mean([load(n)[key] for n in seeds(m)], axis=0)

# ---------- Fig2: ROC + PR (B2 vs P2, 3-seed ensemble, labeled) ----------
y = load("model_B2_seed42")["y"]
b2 = ens_pred("B2", "p_full"); p2 = ens_pred("P2", "p_full")
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
fpr, tpr, _ = roc_curve(y, b2); ax[0].plot(fpr, tpr, label=f"B2 (AUROC {roc(y,b2):.3f})")
fpr, tpr, _ = roc_curve(y, p2); ax[0].plot(fpr, tpr, label=f"P2 (AUROC {roc(y,p2):.3f})")
ax[0].plot([0,1],[0,1],"k--",lw=0.8); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC (3-seed ensemble)"); ax[0].legend(loc="lower right")
pre, rec, _ = precision_recall_curve(y, b2); ax[1].plot(rec, pre, label=f"B2 (AUPRC {ap(y,b2):.3f})")
pre, rec, _ = precision_recall_curve(y, p2); ax[1].plot(rec, pre, label=f"P2 (AUPRC {ap(y,p2):.3f})")
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-Recall (3-seed ensemble)"); ax[1].legend(loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/Fig2_ROC_PR.png", dpi=150); plt.close(fig)
print("Fig2 done", flush=True)

# ---------- Fig3: missing-view degradation (metric-mean, matches Table 4) ----------
cond = ["p_full", "p_cc", "p_mlo"]; labels = ["Full", "CC-only", "MLO-only"]
b2v = [metric_mean("B2", k) for k in cond]
p2v = [metric_mean("P2", k) for k in cond]
x = np.arange(3); w = 0.35
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(x - w/2, b2v, w, label="B2"); ax.bar(x + w/2, p2v, w, label="P2")
for i, (a, b) in enumerate(zip(b2v, p2v)):
    ax.text(i - w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("AUPRC (3-seed mean)"); ax.set_ylim(0, 0.20)
ax.set_title("Missing-view robustness"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/Fig3_missing_view.png", dpi=150); plt.close(fig)
print("Fig3 done", flush=True)

# ---------- Fig4: per-device AUPRC (seed 42, matches Table 5) ----------
db = load("model_B2_seed42"); dp = load("model_P2_seed42")
y = db["y"]; machine = db["machine"]; b2 = db["p_full"]; p2 = dp["p_full"]
groups = np.unique(machine[machine >= 0])
rows = []
for g in groups:
    m = machine == g
    n = int(m.sum()); pos = int(y[m].sum())
    if n < 2 or pos == 0 or pos == n:
        continue
    rows.append((g, n, pos, ap(y[m], b2[m]), ap(y[m], p2[m])))
rows = sorted(rows, key=lambda r: r[0])
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(rows)); w = 0.38
ax.bar(x - w/2, [r[3] for r in rows], w, label="B2")
ax.bar(x + w/2, [r[4] for r in rows], w, label="P2")
ax.set_xticks(x); ax.set_xticklabels([f"dev{r[0]}\n(n={r[1]})" for r in rows], fontsize=8)
ax.set_ylabel("AUPRC"); ax.set_title("Per-device AUPRC (seed 42, random split)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/Fig4_site_generalization.png", dpi=150); plt.close(fig)
print("Fig4 done", flush=True)

# ---------- Fig5: method ladder (metric-mean full-view AUPRC, matches Table 8) ----------
methods = [
    ("B0_MLO", metric_mean("B0_MLO", "p_full")),
    ("B0_CC", metric_mean("B0_CC", "p_full")),
    ("B1", metric_mean("B1", "p_full")),
    ("B2", metric_mean("B2", "p_full")),
    ("P1", metric_mean("P1", "p_full")),
    ("P2", metric_mean("P2", "p_full")),
    ("P3*", ap(y, load("model_P3_seed42")["p_full"])),
]
fig, ax = plt.subplots(figsize=(7, 4.5))
names = [m[0] for m in methods]; vals = [m[1] for m in methods]
colors = ["#999999","#999999","#4C72B0","#4C72B0","#DD8452","#C44E52","#55A868"]
ax.bar(names, vals, color=colors)
for i, v in enumerate(vals): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Full-view AUPRC (3-seed mean)"); ax.set_ylim(0, 0.20)
ax.set_title("Method comparison (P3 single seed 42)")
fig.tight_layout(); fig.savefig(f"{FIG}/Fig5_ablation.png", dpi=150); plt.close(fig)
print("Fig5 done", flush=True)

# ---------- Fig6: calibration / reliability (seed 42, matches Table 7) ----------
T = {"B2": 0.88, "P2": 0.43}
def ece(y, p): return C.expected_calibration_error(y, p, n_bins=10)
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
for i, m in enumerate(["B2", "P2"]):
    dd = load(f"model_{m}_seed42"); y = dd["y"]; p_raw = dd["p_full"]
    eps = 1e-9
    pc = np.clip(p_raw, eps, 1-eps)
    p_scaled = 1/(1 + np.exp(-np.log(pc/(1-pc))/T[m]))
    print(f"{m}: ECE raw={ece(y,p_raw):.4f} scaled={ece(y,p_scaled):.4f}  "
          f"Brier raw={np.mean((y-p_raw)**2):.4f} scaled={np.mean((y-p_scaled)**2):.4f}", flush=True)
    for j, (p, tag) in enumerate([(p_raw, "raw"), (p_scaled, "scaled")]):
        axx = ax[j, i]
        bins = np.linspace(0, 1, 11); mids=[]; obs=[]
        for b in range(10):
            mm = (p >= bins[b]) & (p < bins[b+1])
            if mm.sum() == 0: continue
            mids.append((bins[b]+bins[b+1])/2); obs.append(y[mm].mean())
        axx.bar(mids, obs, width=0.08, alpha=0.6, label="observed")
        axx.plot([0,1],[0,1],"k--",lw=0.8)
        axx.set_title(f"{m} {tag}   (ECE {ece(y,p):.4f})", fontsize=9)
        axx.set_xlabel("predicted probability"); axx.set_ylabel("observed frequency")
fig.suptitle("Reliability diagrams before/after temperature scaling (seed 42)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig6_calibration.png", dpi=150); plt.close(fig)
print("Fig6 done", flush=True)

print("REGEN_FIGS_UNIFIED DONE", flush=True)

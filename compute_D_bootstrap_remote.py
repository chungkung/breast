# =============================================================================
# D significance test (Section 3.5, "degradation gap" D = AUPRC_full - 0.5*(CC+MLO)).
#
# Reconstructs the descriptive test quoted in the manuscript as
#     "does not reach significance under a patient-level paired bootstrap applied
#      to the per-seed metric differences ... uncorrected p = 0.122".
#
# Procedure (matches that sentence):
#   * input    : results/preds/model_<method>_seed<seed>.npz  for B2 and P2,
#                seeds {42, 2024, 7}  (p_full, p_cc, p_mlo, y, pid)
#   * per-seed reduction  r_s = D(B2, seed s) - D(P2, seed s)
#   * resampling          : 2,000 patient-level paired bootstrap resamples; for
#                           each resample the three per-seed reductions are
#                           recomputed and AVERAGED (seed-mean reduction)
#   * RNG                 : numpy.random.RandomState(0)  (this is what reproduces
#                           the manuscript's 0.122 exactly; default_rng(0) would
#                           give 0.1145 instead)
#   * sidedness           : reports BOTH tails. The manuscript's "0.122" is the
#                           ONE-sided P(reduction <= 0); the TWO-sided value
#                           (matching the convention of unified_metrics_remote.py)
#                           is 2x that (= 0.244). Both are printed.
# =============================================================================
import os, sys
sys.path.insert(0, "/root/breast_sci_out")
import numpy as np
from sklearn.metrics import average_precision_score

PRED_DIR = "/root/breast_sci_out/results/preds"
N_BOOT = 2000
RNG_SEED = 0
SEEDS = (42, 2024, 7)


def load(method, seed):
    d = np.load(os.path.join(PRED_DIR, f"model_{method}_seed{seed}.npz"))
    return d["y"], d["pid"], d["p_full"], d["p_cc"], d["p_mlo"]


def D_of(method, seed, idx):
    y, pid, pf, pc, pm = load(method, seed)
    return (average_precision_score(y[idx], pf[idx])
            - 0.5 * (average_precision_score(y[idx], pc[idx])
                     + average_precision_score(y[idx], pm[idx])))


y, pid = load("B2", SEEDS[0])[0], load("B2", SEEDS[0])[1]
uniq = np.unique(pid)

# point per-seed reductions (full data)
reds = [D_of("B2", s, np.arange(len(y))) - D_of("P2", s, np.arange(len(y))) for s in SEEDS]
print("per-seed D reduction (B2 - P2):", ["%+.4f" % r for r in reds],
      " seed-mean = %+.4f" % np.mean(reds), flush=True)

rng = np.random.RandomState(RNG_SEED)
boot = []
for _ in range(N_BOOT):
    sel = rng.choice(uniq, len(uniq), replace=True)
    idx = np.isin(pid, sel)
    if idx.sum() < 2:
        continue
    boot.append(np.mean([D_of("B2", s, idx) - D_of("P2", s, idx) for s in SEEDS]))
boot = np.array(boot)

two_sided = 2 * min(float((boot > 0).mean()), float((boot < 0).mean()))
one_sided = float((boot <= 0).mean())
print(f"D bootstrap: mean=%+.4f  95%% CI=[%+.4f, %+.4f]" %
      (boot.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)), flush=True)
print(f"p (two-sided) = {two_sided:.4f}   p (one-sided, P(reduction<=0)) = {one_sided:.4f}", flush=True)
print("D_BOOTSTRAP DONE", flush=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benjamini-Hochberg FDR correction for the six primary P2-vs-B2 comparisons.

Input : six two-sided raw p-values from the patient-level paired bootstrap
        (2,000 resamples) on three-seed-averaged (ensemble) predictions.
        Canonical script: unified_metrics_remote.py (numpy.random.default_rng(0),
        input results/preds/*.npz, two-sided p = 2*min(P(d>0), P(d<0))).
Output: raw p -> adjusted q -> significant (q < 0.05).

Raw p-values (two-sided, reproducible; values < 0.001 mean 0/2000 resamples
attained the opposite sign, entered here at the resolution floor 0.001):
    full-view AUROC  p < 0.001   (q = 0.003)
    full-view AUPRC  p = 0.030   (q = 0.036)
    CC-only  AUROC   p = 0.011   (q = 0.017)
    CC-only  AUPRC   p = 0.231   (q = 0.231)
    MLO-only AUROC   p < 0.001   (q = 0.003)
    MLO-only AUPRC   p = 0.003   (q = 0.006)

Five of six comparisons have q < 0.05; CC-only AUPRC does not.

NOTE: an earlier draft mixed one-sided AUROC p-values (0.0005/0.0060/0.0010,
from compute_6_pvals_remote.py) with these two-sided AUPRC p-values. That mix
is not a valid single family; the manuscript now reports this consistent
two-sided family throughout.
"""

import numpy as np

ALPHA = 0.05

# (name, raw p-value)
comparisons = [
    ("full-view AUROC", 0.0010),   # p < 0.001 (0/2000), entered at resolution floor
    ("full-view AUPRC", 0.0300),
    ("CC-only AUROC",   0.0110),
    ("CC-only AUPRC",   0.2310),
    ("MLO-only AUROC",  0.0010),   # p < 0.001 (0/2000), entered at resolution floor
    ("MLO-only AUPRC",  0.0030),
]

def benjamini_hochberg(pvals, alpha=ALPHA):
    """Return adjusted q-values (BH) for the input p-values.

    q_i = min_{j >= i} ( p_(j) * m / j ), where p_(1) <= ... <= p_(m) are the
    sorted p-values.  A test is rejected iff q_i < alpha.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)          # ascending
    sorted_p = pvals[order]
    q = np.empty(m)
    running = np.inf
    # sweep from the largest p downward, carrying the minimum ratio
    for i in range(m - 1, -1, -1):
        ratio = sorted_p[i] * m / (i + 1)
        running = min(running, ratio)
        q[i] = running
    # map back to original order
    q_orig = np.empty(m)
    for rank, orig in enumerate(order):
        q_orig[orig] = q[rank]
    return q_orig

def main():
    names = [c[0] for c in comparisons]
    pvals = np.array([c[1] for c in comparisons])
    qvals = benjamini_hochberg(pvals, ALPHA)

    print("Benjamini-Hochberg FDR correction (alpha = %.2f, m = %d)" % (ALPHA, len(pvals)))
    print("%-16s %-10s %-10s %-8s" % ("comparison", "raw p", "BH q", "sig"))
    print("-" * 48)
    for n, p, q in zip(names, pvals, qvals):
        print("%-16s %-10.4f %-10.4f %-8s" % (n, p, q, "YES" if q < ALPHA else "no"))

if __name__ == "__main__":
    main()

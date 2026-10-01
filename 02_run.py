"""02 — Training + evaluation orchestration for E1-E6 (SCI manuscript pipeline).

Run AFTER 01_audit_split.py has produced splits/*.csv.

Flow (flags gate the expensive optional parts):
  E2 baselines   B0_CC / B0_MLO / B1 / B2        (--baselines)
  E3 main        B2 vs P2 (locked test, bootstrap Delta-AUPRC)   [core]
  E4 degradation full / CC-only / MLO-only for B2 & P2           [core]
  E5 machine     per-machine AUPRC + worst group (P3 optional)   (--train-p3)
  E6 ablation    B2 -> P1 -> P2 -> P3 ladder (+ zero-token ctrl)  [core]

Primary metric everywhere: breast-level AUPRC (protocol).  pF1 is reported only
as an RSNA parity metric.

NOTE ON OPERATING THRESHOLDS -- the values in the manuscript's Table 2 do NOT
come from this script.  evaluate_method() below calls C.all_metrics(y, p) on the
TEST predictions, so sensitivity-at-90%-specificity, specificity-at-80%-
sensitivity, PPV and NPV are thresholded on the test set here.  The manuscript
reports thresholds fitted on the VALIDATION set and then applied to the test set,
per seed, averaged over the three seeds; that computation is in
recompute_opvals_remote.py, and it reproduces the published operating points
(B2 0.3960/0.5209/0.0739/0.9867; P2 0.4786/0.5928/0.0859/0.9885) exactly.
The seed-42 artefacts of this script (results/Table2_main_results.csv) are
therefore an earlier, test-thresholded variant.

Examples:
  python 02_run.py                       # core: B2, P1, P2 (+ ablations), 3 seeds
  python 02_run.py --baselines --train-p3
  python 02_run.py --epochs 5 --seeds 42 # quick smoke test
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend (supercomputer / no display)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve,
                             precision_recall_curve)

import config as CFG
import lib_common as C
import lib_models as M
import lib_train as T
from lib_train import DEVICE

AUPRC = average_precision_score


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, tbl, method, force_cc: bool = True, force_mlo: bool = True):
    """Predict on a table. Returns (y, p, patients) in table order."""
    model.eval()
    loader = C.make_loader(tbl, train=False)
    ys, ps = [], []
    for b in loader:
        pcc = b["pcc"].to(DEVICE) * (1.0 if force_cc else 0.0)
        pmlo = b["pmlo"].to(DEVICE) * (1.0 if force_mlo else 0.0)
        logit = model(b["cc"].to(DEVICE), b["mlo"].to(DEVICE), pcc, pmlo)
        ys.append(b["y"].numpy())
        ps.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), tbl.patient_id.values


def eval_conditions(model, tbl, method):
    """Predict under Full / CC-only / MLO-only, aligned on the same patients."""
    out = {}
    for cond, (fcc, fmlo) in {"Full": (True, True), "CC-only": (True, False),
                              "MLO-only": (False, True)}.items():
        y, p, pat = predict(model, tbl, method, fcc, fmlo)
        out[cond] = dict(y=y, p=p, pat=pat)
    return out


# --------------------------------------------------------------------------- #
# Model cache
# --------------------------------------------------------------------------- #
def ckpt_path(method, seed):
    return os.path.join(CFG.CHECKPOINT_DIR, f"model_{method}_seed{seed}.pt")


def train_or_load(train_tbl, val_tbl, method, seed, machine_to_group, force=False):
    p = ckpt_path(method, seed)
    if os.path.exists(p) and not force:
        print(f"[load] {p}")
        model = M.build_model(method, CFG).to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()
        return model
    model = T.train_model(train_tbl, val_tbl, method, seed, machine_to_group)
    torch.save(model.state_dict(), p)
    print(f"[save] {p}")
    return model


# --------------------------------------------------------------------------- #
# E3: main comparison + per-seed bootstrap
# --------------------------------------------------------------------------- #
def evaluate_method(train_tbl, val_tbl, test_tbl, method, seeds, machine_to_group, force):
    per_seed = []
    models = {}
    for s in seeds:
        model = train_or_load(train_tbl, val_tbl, method, s, machine_to_group, force)
        models[s] = model
        y, p, pat = predict(model, test_tbl, method)
        m = C.all_metrics(y, p)
        auprc_ci = C.bootstrap_ci(y, p, pat, AUPRC, CFG.BOOTSTRAP, s)
        auc_ci = C.delong_auc_ci(y, p)
        per_seed.append(dict(seed=s, y=y, p=p, pat=pat, metrics=m,
                             auprc_ci=auprc_ci, auc_ci=auc_ci))
    return models, per_seed


def summarize(per_seed):
    """Aggregate per-seed results into a summary dict (mean AUPRC etc.)."""
    auprc = np.mean([r["metrics"]["AUPRC"] for r in per_seed])
    auprc_sd = np.std([r["metrics"]["AUPRC"] for r in per_seed])
    auroc = np.mean([r["metrics"]["AUROC"] for r in per_seed])
    brier = np.mean([r["metrics"]["Brier"] for r in per_seed])
    ece = np.mean([r["metrics"]["ECE"] for r in per_seed])
    sens = np.mean([r["metrics"]["Sens_at_90Spec"] for r in per_seed])
    spec = np.mean([r["metrics"]["Spec_at_80Sens"] for r in per_seed])
    ppv = np.mean([r["metrics"]["PPV"] for r in per_seed])
    npv = np.mean([r["metrics"]["NPV"] for r in per_seed])
    pF1 = np.mean([r["metrics"]["pF1"] for r in per_seed])
    return dict(AUPRC=auprc, AUPRC_SD=auprc_sd, AUROC=auroc, pF1=pF1, Brier=brier,
                ECE=ece, Sens_at_90Spec=sens, Spec_at_80Sens=spec, PPV=ppv, NPV=npv)


# --------------------------------------------------------------------------- #
# E4: missing-view degradation
# --------------------------------------------------------------------------- #
def degradation_summary(conds):
    d = {}
    for cond in ("Full", "CC-only", "MLO-only"):
        d[cond] = AUPRC(conds[cond]["y"], conds[cond]["p"])
    d["D"] = d["Full"] - 0.5 * (d["CC-only"] + d["MLO-only"])
    return d


def paired_condition_ci(condsA, condsB, cond):
    """Paired bootstrap CI of AUPRC(B) - AUPRC(A) in one viewing condition."""
    a = condsA[cond]; b = condsB[cond]
    return C.paired_delta_ci(a["y"], a["p"], b["p"], a["pat"], AUPRC, CFG.BOOTSTRAP, 42)


# --------------------------------------------------------------------------- #
# E5: machine-group (worst-group) evaluation
# --------------------------------------------------------------------------- #
def machine_group_eval(model, tbl, method):
    y, p, _ = predict(model, tbl, method)
    mach = tbl.machine_id.values if "machine_id" in tbl.columns else None
    if mach is None:
        return None
    rows = []
    for mid in np.unique(mach):
        m = mach == mid
        auprc = AUPRC(y[m], p[m]) if (len(np.unique(y[m])) > 1) else float("nan")
        rows.append(dict(machine_id=int(mid), n=int(m.sum()), pos=int(y[m].sum()),
                         AUPRC=auprc))
    df = pd.DataFrame(rows).sort_values("AUPRC", na_position="last")
    weighted = np.nansum(df.AUPRC * df.n) / df.n.sum() if len(df) else float("nan")
    macro = df.AUPRC.mean() if len(df) else float("nan")
    worst = df.dropna(subset=["AUPRC"]).iloc[0] if len(df.dropna(subset=["AUPRC"])) else None
    return dict(table=df, weighted=weighted, macro=macro, worst=worst)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_roc_pr(yB, pB, yP, pP):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for (y, p, lab, c) in [(yB, pB, "B2 (baseline)", "#4C72B0"),
                           (yP, pP, "P2 (ours)", "#DD8452")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{lab}  AUROC={roc_auc_score(y,p):.3f}", color=c)
        pre, rec, _ = precision_recall_curve(y, p)
        ax[1].plot(rec, pre, label=f"{lab}  AUPRC={AUPRC(y,p):.3f}", color=c)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="Figure 2a. ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Figure 2b. Precision-Recall")
    for a in ax:
        a.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.FIG_DIR, "Fig2_ROC_PR.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_degradation(dB, dP):
    import matplotlib.pyplot as plt
    conds = ["Full", "CC-only", "MLO-only"]
    x = np.arange(len(conds)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w / 2, [dB[c] for c in conds], w, label="B2 (baseline)", color="#4C72B0")
    ax.bar(x + w / 2, [dP[c] for c in conds], w, label="P2 (ours)", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Breast-level AUPRC")
    ax.set_title(f"Figure 3. Missing-View Degradation\n"
                 f"(D: B2={dB['D']:.3f}, P2={dP['D']:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.FIG_DIR, "Fig3_missing_view.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_machine(mach_res):
    import matplotlib.pyplot as plt
    if not mach_res:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in mach_res.items():
        df = r["table"]
        ax.plot(df.machine_id.astype(str), df.AUPRC, "o-", label=f"{name}")
    ax.set_xlabel("machine_id"); ax.set_ylabel("Breast-level AUPRC")
    ax.set_title("Figure 4. Per-Machine Performance (worst-group stress test)")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.FIG_DIR, "Fig4_per_device_auprc.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_calibration(ys_ps):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (y, p) in ys_ps.items():
        xs, ys_ = [], []
        bins = np.linspace(0, 1, 11)
        for i in range(10):
            m = (p >= bins[i]) & (p < bins[i + 1])
            if m.sum() == 0:
                continue
            xs.append(p[m].mean()); ys_.append(y[m].mean())
        ax.plot(xs, ys_, "o-", label=f"{name}  ECE={C.expected_calibration_error(y,p):.3f}")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Figure 6. Calibration (Reliability Diagram)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.FIG_DIR, "Fig6_calibration.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_ablation(ablation_rows):
    import matplotlib.pyplot as plt
    names = [r[0] for r in ablation_rows]
    vals = [r[1] for r in ablation_rows]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(names, vals, color="#4a7")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel("Breast-level AUPRC (locked test)")
    ax.set_title("Figure 5. Incremental Ablation (B2 → P1 → P2 → P3)")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.FIG_DIR, "Fig5_ablation.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="B2,P1,P2")
    ap.add_argument("--seeds", default="42,2024,7")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if checkpoint exists")
    ap.add_argument("--skip-precache", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="processes for DICOM precache")
    ap.add_argument("--baselines", action="store_true", help="also train B0_CC/B0_MLO/B1")
    ap.add_argument("--train-p3", action="store_true", help="train P3 (GroupDRO) for E5/H3")
    args = ap.parse_args()

    if args.epochs:
        CFG.EPOCHS = args.epochs
    methods = [m for m in args.methods.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    # ---- load frozen splits ----
    def load(name):
        p = os.path.join(CFG.SPLIT_DIR, f"{name}_breast.csv")
        if not os.path.exists(p):
            print(f"[FATAL] missing {p}. Run 01_audit_split.py first.")
            sys.exit(1)
        return pd.read_csv(p)

    train_tbl = load("train")
    val_tbl = load("val")
    test_tbl = load("test")
    print(f"Loaded splits: train={len(train_tbl)} val={len(val_tbl)} test={len(test_tbl)} "
          f"(test pos={int((test_tbl.cancer==1).sum())})")

    machine_to_group, n_groups = M.build_machine_groups(train_tbl)
    print(f"machine groups: {n_groups} (GroupDRO {'enabled' if n_groups > 1 else 'disabled'})")

    if not args.skip_precache:
        C.precache(pd.concat([train_tbl, val_tbl, test_tbl]), workers=args.workers)

    results = {}        # method -> {'models': {seed: model}, 'per_seed': [...], 'summary': {...}}
    conds = {}          # method -> condition predictions (from seed 42)
    n_seeds = len(seeds)

    # ---- train + evaluate each requested method ----
    for method in methods:
        print(f"\n===== Training/evaluating {method} ({n_seeds} seeds) =====")
        models, per_seed = evaluate_method(train_tbl, val_tbl, test_tbl, method,
                                           seeds, machine_to_group, args.force)
        results[method] = dict(models=models, per_seed=per_seed,
                               summary=summarize(per_seed))
        # condition predictions for the primary seed (first seed)
        conds[method] = eval_conditions(models[seeds[0]], test_tbl, method)
        # free fragmentation between methods (prevents cumulative GPU OOM on B2->P2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- optional baselines B0/B1 ----
    if args.baselines:
        for method in ("B0_CC", "B0_MLO"):
            sub = train_tbl[train_tbl.cc_image_id.notna()] if method == "B0_CC" \
                else train_tbl[train_tbl.mlo_image_id.notna()]
            sub_val = val_tbl[val_tbl.cc_image_id.notna()] if method == "B0_CC" \
                else val_tbl[val_tbl.mlo_image_id.notna()]
            sub_test = test_tbl[test_tbl.cc_image_id.notna()] if method == "B0_CC" \
                else test_tbl[test_tbl.mlo_image_id.notna()]
            print(f"\n===== Baseline {method} =====")
            models, per_seed = evaluate_method(sub, sub_val, sub_test, method,
                                               seeds, machine_to_group, args.force)
            results[method] = dict(models=models, per_seed=per_seed,
                                   summary=summarize(per_seed))
        # B1 uses full tables (prob-average of both views)
        print(f"\n===== Baseline B1 =====")
        models, per_seed = evaluate_method(train_tbl, val_tbl, test_tbl, "B1",
                                           seeds, machine_to_group, args.force)
        results["B1"] = dict(models=models, per_seed=per_seed, summary=summarize(per_seed))

    # ---- optional P3 (GroupDRO) ----
    if args.train_p3 and n_groups > 1:
        print(f"\n===== Training P3 (GroupDRO over {n_groups} machines) =====")
        models, per_seed = evaluate_method(train_tbl, val_tbl, test_tbl, "P3",
                                           seeds, machine_to_group, args.force)
        results["P3"] = dict(models=models, per_seed=per_seed, summary=summarize(per_seed))
        conds["P3"] = eval_conditions(models[seeds[0]], test_tbl, "P3")

    # ==================================================================== #
    # Report: E3 (main comparison)
    # ==================================================================== #
    print("\n\n================ E3 MAIN COMPARISON (locked test) ================")
    for method, r in results.items():
        s = r["summary"]
        print(f"{method:8s} AUPRC={s['AUPRC']:.4f}+/-{s['AUPRC_SD']:.4f}  "
              f"AUROC={s['AUROC']:.4f}  pF1={s['pF1']:.4f}  Brier={s['Brier']:.4f}  "
              f"ECE={s['ECE']:.4f}")

    if "B2" in results and "P2" in results:
        # paired delta AUPRC (P2 - B2) per seed + summary
        deltas = []
        for seed in seeds:
            b2 = [r for r in results["B2"]["per_seed"] if r["seed"] == seed][0]
            p2 = [r for r in results["P2"]["per_seed"] if r["seed"] == seed][0]
            d, lo, hi = C.paired_delta_ci(b2["y"], b2["p"], p2["p"], b2["pat"],
                                          AUPRC, CFG.BOOTSTRAP, seed)
            deltas.append((d, lo, hi))
            print(f"  seed {seed}: Delta-AUPRC (P2-B2) = {d:.4f}  95%CI [{lo:.4f}, {hi:.4f}]")
        d_mean = np.mean([x[0] for x in deltas])
        d_lo = np.mean([x[1] for x in deltas])
        d_hi = np.mean([x[2] for x in deltas])
        h1_pass = (d_lo > 0) and (d_mean >= 0.02)
        print(f"  SUMMARY Delta-AUPRC = {d_mean:.4f}  95%CI [{d_lo:.4f}, {d_hi:.4f}]  "
              f"-> H1 {'PASS' if h1_pass else 'not met'}")

    # ==================================================================== #
    # Report: E4 (missing-view degradation)
    # ==================================================================== #
    print("\n================ E4 MISSING-VIEW DEGRADATION ================")
    if "B2" in conds and "P2" in conds:
        dB = degradation_summary(conds["B2"])
        dP = degradation_summary(conds["P2"])
        for name, d in (("B2", dB), ("P2", dP)):
            print(f"  {name}: Full={d['Full']:.4f} CC-only={d['CC-only']:.4f} "
                  f"MLO-only={d['MLO-only']:.4f}  D={d['D']:.4f}")
        rel_reduction = (dB["D"] - dP["D"]) / (dB["D"] + 1e-9)
        print(f"  D(B2)-D(P2) = {dB['D']-dP['D']:.4f}  "
              f"relative reduction = {100*rel_reduction:.1f}%  "
              f"-> H2 {'PASS' if rel_reduction >= 0.20 else 'not met'}")
        for cond in ("CC-only", "MLO-only"):
            d, lo, hi = paired_condition_ci(conds["B2"], conds["P2"], cond)
            print(f"  {cond}: Delta-AUPRC(P2-B2) = {d:.4f}  95%CI [{lo:.4f}, {hi:.4f}]")
        fig_degradation(dB, dP)

    # ==================================================================== #
    # Report: E5 (machine-group worst-case)
    # ==================================================================== #
    print("\n================ E5 MACHINE-GROUP STRESS TEST ================")
    mach_res = {}
    if "machine_id" in test_tbl.columns:
        for method in results:
            if method in ("B0_CC", "B0_MLO"):
                continue
            r = machine_group_eval(results[method]["models"][seeds[0]], test_tbl, method)
            if r:
                mach_res[method] = r
                w = r["worst"]
                print(f"  {method}: weighted={r['weighted']:.4f} macro={r['macro']:.4f}  "
                      f"worst group={int(w['machine_id'])} AUPRC={w['AUPRC']:.4f} "
                      f"(n={int(w['n'])}, pos={int(w['pos'])})")
        fig_machine(mach_res)
        if "P2" in mach_res and "P3" in mach_res:
            wp2 = mach_res["P2"]["worst"]["AUPRC"]
            wp3 = mach_res["P3"]["worst"]["AUPRC"]
            print(f"  H3 worst-group Delta(P3-P2) = {wp3-wp2:.4f}  "
                  f"-> {'PASS (point est >0)' if wp3 > wp2 else 'not met'}")

    # ==================================================================== #
    # Report: E6 (incremental ablation ladder)
    # ==================================================================== #
    print("\n================ E6 ABLATION (incremental) ================")
    ablation_rows = []
    for method, r in results.items():
        ablation_rows.append((method, r["summary"]["AUPRC"]))
    _order = {"B0_MLO": 0, "B0_CC": 1, "B1": 2, "B2": 3, "P1": 4, "P2": 5, "P3": 6}
    ablation_rows.sort(key=lambda x: _order.get(x[0], 99))
    for name, v in ablation_rows:
        print(f"  {name:10s} AUPRC={v:.4f}")
    if len(ablation_rows) >= 2:
        fig_ablation(ablation_rows)

    # ==================================================================== #
    # Figures (ROC/PR, calibration)
    # ==================================================================== #
    if "B2" in results and "P2" in results:
        b2 = results["B2"]["per_seed"][0]
        p2 = results["P2"]["per_seed"][0]
        fig_roc_pr(b2["y"], b2["p"], p2["y"], p2["p"])
        fig_calibration({"B2 (baseline)": (b2["y"], b2["p"]),
                         "P2 (ours)": (p2["y"], p2["p"])})

    # ==================================================================== #
    # Export Table2 (main results) + Table3 (per-seed)
    # ==================================================================== #
    os.makedirs(CFG.RESULT_DIR, exist_ok=True)
    rows = []
    for method, r in results.items():
        s = r["summary"]
        # CI from the primary seed bootstrap
        p0 = r["per_seed"][0]
        rows.append(dict(
            Model=method, AUPRC=round(s["AUPRC"], 4), AUPRC_SD=round(s["AUPRC_SD"], 4),
            AUPRC_CI=f"[{p0['auprc_ci'][1]:.3f}, {p0['auprc_ci'][2]:.3f}]",
            AUROC=round(s["AUROC"], 4), pF1=round(s["pF1"], 4),
            Sens_at_90Spec=round(s["Sens_at_90Spec"], 4),
            Spec_at_80Sens=round(s["Spec_at_80Sens"], 4),
            PPV=round(s["PPV"], 4), NPV=round(s["NPV"], 4),
            Brier=round(s["Brier"], 4), ECE=round(s["ECE"], 4)))
    tab2 = pd.DataFrame(rows)
    tab2.to_csv(os.path.join(CFG.RESULT_DIR, "Table2_main_results.csv"), index=False)

    per_seed_rows = []
    for method, r in results.items():
        for x in r["per_seed"]:
            per_seed_rows.append(dict(Model=method, seed=x["seed"],
                                      AUPRC=round(x["metrics"]["AUPRC"], 4),
                                      AUROC=round(x["metrics"]["AUROC"], 4),
                                      Brier=round(x["metrics"]["Brier"], 4),
                                      ECE=round(x["metrics"]["ECE"], 4)))
    pd.DataFrame(per_seed_rows).to_csv(
        os.path.join(CFG.RESULT_DIR, "Table3_per_seed.csv"), index=False)

    print("\nWrote Table2_main_results.csv and Table3_per_seed.csv under", CFG.RESULT_DIR)
    print("Figures saved under", CFG.FIG_DIR)
    print("\nALL DONE.")


if __name__ == "__main__":
    main()

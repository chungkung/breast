"""01 — Data audit + patient-level frozen split (RSNA breast cancer detection).

Run FIRST on the supercomputer.  Needs only pandas + numpy (no torch), so it is
fast and safe on the metadata level.  It decides which manuscript claims
(H1-H4, E5/P3) are feasible *before* any expensive GPU training.

Outputs (under OUT_DIR):
  splits/train_breast.csv   (dev: training)
  splits/val_breast.csv     (dev: early-stop / threshold selection)
  splits/test_breast.csv    (LOCKED — evaluated exactly once at the end)
  results/audit_report.csv  (full audit table)
  results/audit_report.md   (human-readable audit + gate results)

Usage:
  python 01_audit_split.py [--train-csv PATH] [--test-frac 0.2] [--val-frac 0.2] [--seed 42]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import config as CFG
import lib_common as C


def audit(tbl: pd.DataFrame) -> dict:
    """Return a flat dict of audit statistics for the breast-level table."""
    n_pat = tbl.patient_id.nunique()
    n_breast = len(tbl)
    n_pos_breast = int((tbl.cancer == 1).sum())
    pos_pat = tbl[tbl.cancer == 1].patient_id.nunique()

    has_cc = tbl.cc_image_id.notna()
    has_mlo = tbl.mlo_image_id.notna()
    both = has_cc & has_mlo
    only_cc = has_cc & ~has_mlo
    only_mlo = ~has_cc & has_mlo

    # view completeness among CANCER breasts (protocol gate) and overall
    pos = tbl[tbl.cancer == 1]
    pos_both = (pos.cc_image_id.notna() & pos.mlo_image_id.notna()).mean() if len(pos) else float("nan")

    out = dict(
        patients=int(n_pat),
        breasts=int(n_breast),
        positive_breasts=int(n_pos_breast),
        positive_patients=int(pos_pat),
        prevalence_breast=float(n_pos_breast / n_breast if n_breast else float("nan")),
        prevalence_patient=float(pos_pat / n_pat if n_pat else float("nan")),
        both_views=int(both.sum()),
        cc_only=int(only_cc.sum()),
        mlo_only=int(only_mlo.sum()),
        both_view_frac=float(both.mean()),
        cancer_both_view_frac=float(pos_both),
    )
    if "laterality" in tbl.columns:
        out["L_breasts"] = int((tbl.laterality == "L").sum())
        out["R_breasts"] = int((tbl.laterality == "R").sum())
    if "density" in tbl.columns:
        out["density_counts"] = tbl.density.value_counts().to_dict()
    if "machine_id" in tbl.columns:
        out["n_machines"] = int(tbl.machine_id.nunique())
    if "site_id" in tbl.columns:
        out["n_sites"] = int(tbl.site_id.nunique())
    return out


def machine_group_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Per-machine_id statistics (for E5 / GroupDRO feasibility)."""
    if "machine_id" not in tbl.columns:
        return pd.DataFrame()
    rows = []
    for mid, g in tbl.groupby("machine_id"):
        rows.append(dict(
            machine_id=int(mid),
            breasts=len(g),
            positives=int((g.cancer == 1).sum()),
        ))
    out = pd.DataFrame(rows).sort_values("positives", ascending=False)
    return out.reset_index(drop=True)


def format_md_table(df: pd.DataFrame, float_fmt="{:.4f}") -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check_gates(aud: dict, mach: pd.DataFrame, test_tbl: pd.DataFrame) -> dict:
    gates = {}
    n_test_pos = int((test_tbl.cancer == 1).sum())
    gates["test_positives_ge_100"] = n_test_pos >= CFG.MIN_TEST_POSITIVES
    gates["cancer_view_completeness_ge_80"] = aud["cancer_both_view_frac"] >= CFG.MIN_VIEW_COMPLETENESS
    gates["machine_groups_ge_3"] = (not mach.empty) and (len(mach) >= CFG.MIN_MACHINE_GROUPS)
    n_ok_groups = int((mach.positives >= CFG.MIN_MACHINE_POSITIVES).sum()) if not mach.empty else 0
    gates["machine_groups_ge30_pos_ge3"] = n_ok_groups >= CFG.MIN_MACHINE_GROUPS
    gates["n_machine_groups_with_ge30_pos"] = n_ok_groups
    gates["n_test_positives"] = n_test_pos
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=CFG.TRAIN_CSV)
    ap.add_argument("--out", default=CFG.OUT_DIR)
    ap.add_argument("--test-frac", type=float, default=0.25,
                    help="25%% gives >=100 locked-test positives on RSNA (0.20 -> 89).")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of the dev set reserved for validation (early stop)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.train_csv):
        print(f"[FATAL] train.csv not found at {args.train_csv}")
        print("        Set RSNA_DIR (env) or pass --train-csv.")
        sys.exit(1)

    print(f"Reading {args.train_csv} ...")
    raw = pd.read_csv(args.train_csv)
    print(f"Raw rows (images): {len(raw)}")
    print(f"Columns present:  {list(raw.columns)}")

    # Detect label column name (cancer) and required keys.
    for col in ("cancer", "patient_id", "laterality", "view", "image_id"):
        if col not in raw.columns:
            print(f"[FATAL] required column '{col}' missing from train.csv")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Cohort assembly + exclusions (metadata-level)
    # ------------------------------------------------------------------ #
    n0 = raw.image_id.nunique()
    n_dup = len(raw) - n0
    if n_dup:
        print(f"[QC] dropping {n_dup} duplicate image_id rows (keeping first)")
        raw = raw.drop_duplicates(subset="image_id", keep="first")

    breast = C.build_breast_table(args.train_csv)
    print(f"Breast-level samples (patient_id + laterality): {len(breast)}")

    # Drop breasts that have NO image at all (cannot be modeled).
    no_img = breast.cc_image_id.isna() & breast.mlo_image_id.isna()
    if no_img.any():
        print(f"[QC] dropping {int(no_img.sum())} breasts with no CC and no MLO image")
        breast = breast[~no_img].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Audit report
    # ------------------------------------------------------------------ #
    aud = audit(breast)
    mach = machine_group_table(breast)

    print("\n================ AUDIT REPORT ================")
    for k in ("patients", "breasts", "positive_breasts", "positive_patients",
              "prevalence_breast", "prevalence_patient", "both_views",
              "cc_only", "mlo_only", "both_view_frac", "cancer_both_view_frac"):
        if k in aud:
            v = aud[k]
            print(f"  {k:28s}: {v:.4f}" if isinstance(v, float) else f"  {k:28s}: {v}")
    if "n_machines" in aud:
        print(f"  {'n_machines':28s}: {aud['n_machines']}")
    if "n_sites" in aud:
        print(f"  {'n_sites':28s}: {aud['n_sites']}")

    print("\n--- machine_id distribution (positives) ---")
    if mach.empty:
        print("  machine_id column ABSENT -> E5/GroupDRO NOT feasible")
    else:
        print(mach.head(20).to_string(index=False))

    # ------------------------------------------------------------------ #
    # Frozen patient-level split: 80% dev (train+val) / 20% locked test
    # ------------------------------------------------------------------ #
    split = C.patient_level_split(breast, frac=args.test_frac, seed=args.seed)
    test_tbl = split[split.split == "test"].reset_index(drop=True)
    dev_tbl = split[split.split == "train"].reset_index(drop=True)

    # Within dev: train / val (patient-level again)
    dev_split = C.patient_level_split(dev_tbl, frac=args.val_frac, seed=args.seed + 1)
    train_tbl = dev_split[dev_split.split == "train"].reset_index(drop=True)
    val_tbl = dev_split[dev_split.split == "test"].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Gate checks
    # ------------------------------------------------------------------ #
    gates = check_gates(aud, mach, test_tbl)

    print("\n================ SPLIT ================")
    print(f"  train breasts: {len(train_tbl)}  (pos {int((train_tbl.cancer==1).sum())})")
    print(f"  val   breasts: {len(val_tbl)}  (pos {int((val_tbl.cancer==1).sum())})")
    print(f"  test  breasts: {len(test_tbl)}  (pos {int((test_tbl.cancer==1).sum())})  [LOCKED]")
    leak = len(set(train_tbl.patient_id) & set(test_tbl.patient_id)) + \
           len(set(val_tbl.patient_id) & set(test_tbl.patient_id))
    print(f"  patient leakage across splits: {leak} (must be 0)")

    print("\n================ FEASIBILITY GATES ================")
    for k, v in gates.items():
        if k == "n_test_positives":
            flag = "INFO"
        else:
            flag = "PASS" if v else "FAIL"
        print(f"  [{flag}] {k}: {v}")

    # ------------------------------------------------------------------ #
    # Persist
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)

    for name, t in (("train", train_tbl), ("val", val_tbl), ("test", test_tbl)):
        p = os.path.join(args.out, "splits", f"{name}_breast.csv")
        t.to_csv(p, index=False)
        print(f"Wrote {p}")

    # audit report
    aud_df = pd.DataFrame([{k: v for k, v in aud.items() if not isinstance(v, dict)}])
    aud_path = os.path.join(args.out, "results", "audit_report.csv")
    aud_df.to_csv(aud_path, index=False)
    if not mach.empty:
        mach.to_csv(os.path.join(args.out, "results", "machine_group_table.csv"), index=False)

    # markdown report
    md = []
    md.append("# RSNA Data Audit & Split Report\n")
    md.append("## Audit\n")
    md.append(format_md_table(aud_df))
    md.append("\n## machine_id distribution (top groups by positives)\n")
    md.append(format_md_table(mach.head(20)))
    md.append("\n## Gates\n")
    gates_df = pd.DataFrame([{"gate": k, "value": v} for k, v in gates.items()])
    md.append(format_md_table(gates_df))
    md.append("\n## Split\n")
    split_summary = pd.DataFrame([
        {"split": "train", "breasts": len(train_tbl), "positives": int((train_tbl.cancer == 1).sum())},
        {"split": "val", "breasts": len(val_tbl), "positives": int((val_tbl.cancer == 1).sum())},
        {"split": "test", "breasts": len(test_tbl), "positives": int((test_tbl.cancer == 1).sum())},
    ])
    md.append(format_md_table(split_summary))
    md_path = os.path.join(args.out, "results", "audit_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {md_path}")

    print("\nDONE. Review the gates above before starting training (02_run.py).")


if __name__ == "__main__":
    main()

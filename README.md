# Missing-View Robust Breast Cancer Detection via Masked Multi-View Distillation

Official code and locked predictions for the paper:

> **Missing-View Robust Breast Cancer Detection via Masked Multi-View Distillation**

This repository reproduces all tables and statistics of the paper on the
[RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
dataset. The method (`P2`) combines a **shared learned missing token**, **random
single-view masking**, and **fixed-weight cross-view self-distillation** in a
dual-view Transformer over a shared ResNet-50 backbone.

## Repository layout

```
config.py                  central configuration (paths, hyper-parameters, seeds)
lib_common.py              DICOM loading, preprocessing, metrics, bootstrap helpers
lib_models.py              ResNet-50 + 2-layer/8-head Transformer (B0/B1/B2/P1/P2/DRO-abl)
lib_train.py               training loops, focal loss, masking, Bernoulli-KL distillation
01_audit_split.py          cohort construction + frozen patient-level split (screening)
02_run.py                  training + evaluation orchestration
recompute_opvals_remote.py validation-set operating thresholds (Table 2 operating points)
finish_eval_remote.py      temperature scaling / calibration (Table 8)
degrade_sim_remote.py      Gaussian-noise corrupted-view stress test (Table 12)
unified_metrics_remote.py  CANONICAL statistics: per-checkpoint + bootstrap (Section 3.5)
compute_6_pvals_remote.py  SUPERSEDED one-sided variant (kept for provenance)
bh_correction.py           Benjamini-Hochberg FDR correction
regen_figs_v2.py           figure generation (reliability diagrams, per-device plots)
requirements.txt           Python dependencies
splits/                    frozen train/val/test breast CSVs (patient_id, laterality, ...)
results/preds/             locked per-breast predictions (29 .npz: y, pid, machine, p_full/p_cc/p_mlo)
results/                   seed means, per-checkpoint metrics, paired bootstrap, cohort audit
```

## Fast path: reproduce the statistics without retraining

The locked predictions in `results/preds/` reproduce the paper's numbers directly
(no GPU, no image download needed):

```bash
pip install -r requirements.txt

# 1) Six primary comparisons + paired bootstrap (two-sided, default_rng(0))
python unified_metrics_remote.py      # -> results/unified_seed_means.csv,
                                      #    results/unified_per_checkpoint.csv,
                                      #    results/unified_paired_bootstrap.csv

# 2) Benjamini-Hochberg correction of the six comparisons
python bh_correction.py
```

> `unified_metrics_remote.py` hard-codes `results/preds/` under its working
> directory; run it from the repository root (or adjust the two `PRED_DIR` /
> `RES_DIR` constants at the top of the file).

Expected primary results (P2 vs B2, three-seed ensemble, 2,000 patient-level
paired bootstrap resamples, **two-sided**):

| comparison | mean Δ | 95% CI | p | BH q |
|---|---|---|---|---|
| full-view AUPRC | +0.0404 | [+0.0039, +0.0808] | 0.030 | 0.036 |
| CC-only  AUPRC  | +0.0224 | [-0.0129, +0.0604] | 0.231 | 0.231 |
| MLO-only AUPRC  | +0.0492 | [+0.0178, +0.0884] | 0.003 | 0.006 |
| full-view AUROC | +0.0560 | [+0.0234, +0.0896] | <0.001 | 0.003 |
| CC-only  AUROC  | +0.0574 | [+0.0134, +0.1044] | 0.011 | 0.017 |
| MLO-only AUROC  | +0.0595 | [+0.0255, +0.0948] | <0.001 | 0.003 |

Five of six comparisons have q < 0.05; CC-only AUPRC does not.

## Full path: reconstruct data, retrain, and re-evaluate

1. Download the RSNA dataset (`train.csv`, `train_images/`) and set
   `RSNA_DIR` to the extracted folder (see `config.py`).
2. `python 01_audit_split.py` — rebuild the frozen split (or reuse `splits/`).
3. `python 02_run.py` — train `B0_CC`, `B0_MLO`, `B1`, `B2`, `P1`, `P2` (three
   seeds 42 / 2024 / 7).
4. `python recompute_opvals_remote.py` — Table 2 operating points
   (validation-set thresholds applied to the test set).
5. `python finish_eval_remote.py` — Table 8 temperature scaling.
6. `python degrade_sim_remote.py` — Table 12 corrupted-view stress test.

## Statistical convention

All six primary p-values are **two-sided** bootstrap tail probabilities
(`p = 2 * min(P(Δ>0), P(Δ<0))`) computed by `unified_metrics_remote.py` with
`numpy.random.default_rng(0)`. With 2,000 resamples the smallest attainable
non-zero value is 0.001, so "p < 0.001" denotes 0/2000 resamples with the
opposite sign. An earlier draft mixed a one-sided AUROC family
(`compute_6_pvals_remote.py`) into the same BH correction; that file is retained
for provenance and is **not** the canonical script.

## Model weights

Trained checkpoints (~3.2 GB across 29 models) are not stored in this repository.
They are available from the corresponding author upon request. The locked
predictions in `results/preds/` are sufficient to reproduce every table and
statistic without them.

## Data

The RSNA Screening Mammography Breast Cancer Detection dataset is publicly
available on [Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
and is **not redistributed** here. The cohort, split, and per-breast predictions
included in this repository are derived artifacts intended solely for
reproducing the paper's analyses.

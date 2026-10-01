# RSNA Data Audit & Split Report

## Audit

| patients | breasts | positive_breasts | positive_patients | prevalence_breast | prevalence_patient | both_views | cc_only | mlo_only | both_view_frac | cancer_both_view_frac | L_breasts | R_breasts | n_machines | n_sites |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11913.0000 | 23826.0000 | 492.0000 | 486.0000 | 0.0206 | 0.0408 | 23826.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 11913.0000 | 11913.0000 | 10.0000 | 2.0000 |

## machine_id distribution (top groups by positives)

| machine_id | breasts | positives |
|---|---|---|
| 49 | 9178 | 235 |
| 48 | 4148 | 85 |
| 29 | 4042 | 78 |
| 21 | 4000 | 75 |
| 170 | 382 | 10 |
| 93 | 776 | 5 |
| 216 | 772 | 3 |
| 190 | 52 | 1 |
| 197 | 8 | 0 |
| 210 | 468 | 0 |

## Gates

| gate | value |
|---|---|
| test_positives_ge_100 | True |
| cancer_view_completeness_ge_80 | True |
| machine_groups_ge_3 | True |
| machine_groups_ge30_pos_ge3 | True |
| n_machine_groups_with_ge30_pos | 4 |
| n_test_positives | 117 |

## Split

| split | breasts | positives |
|---|---|---|
| train | 14294 | 294 |
| val | 3574 | 81 |
| test | 5958 | 117 |
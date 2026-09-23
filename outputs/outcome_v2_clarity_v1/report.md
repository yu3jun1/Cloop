# Outcome-v2 formal development experiment

Run: `outcome_v2_clarity_v1` · Completed: 2026-09-23 13:19 UTC  
Protocol: 5-fold patient-level CV, seeds 7/17/29, train-fold PCA64, piecewise-exponential hazards at `[0,90,180,365,730]` days. Original train and validation patients form the development cohort; the registered formal test split remains sealed. Values below are mean ± sample SD across five fold means (the three seeds are averaged within each fold).

## Outcome input ablation

| Evaluator | C-index ↑ | IPCW Brier@365 ↓ | Survival NLL ↓ |
|---|---:|---:|---:|
| O0 clinical only | 0.6067 ± 0.0695 | 0.2193 ± 0.0503 | 3.0029 ± 0.1391 |
| O1 single-state residual | 0.6572 ± 0.0643 | 0.2122 ± 0.0652 | 3.0004 ± 0.1432 |
| O1-matched | 0.6027 ± 0.0467 | 0.2754 ± 0.0931 | 3.4441 ± 0.1121 |
| OT observed transition | 0.6098 ± 0.0505 | 0.2744 ± 0.0902 | 3.4404 ± 0.1152 |

O1−O0: mean ΔC-index `+0.05045`, ΔBrier `−0.00713`, ΔNLL `−0.00247`; all three improved in 3/5 folds. OT−O1-matched: mean ΔC-index `+0.00710` (4/5 folds improved), ΔBrier `−0.00095` (1/5), ΔNLL `−0.00370` (4/5). Gate B passes by the registered rule of at least two primary metrics improving in a majority of folds. The gain is small and not consistent for Brier.

## Frozen OT downstream comparison

| Future-state source | C-index ↑ | IPCW Brier@365 ↓ | Survival NLL ↓ |
|---|---:|---:|---:|
| Observed target MRI | 0.6098 ± 0.0505 | 0.2744 ± 0.0902 | 3.4404 ± 0.1152 |
| Baseline predicted | 0.5987 ± 0.0425 | 0.2737 ± 0.0891 | 3.4435 ± 0.1144 |
| RRT predicted | 0.5982 ± 0.0409 | 0.2751 ± 0.0903 | 3.4443 ± 0.1138 |
| Ensemble predicted | 0.5993 ± 0.0409 | 0.2736 ± 0.0888 | 3.4428 ± 0.1147 |
| RRT+Ensemble predicted | 0.5982 ± 0.0409 | 0.2754 ± 0.0905 | 3.4446 ± 0.1135 |

Predicted−observed mean gaps (C-index, Brier, NLL): Baseline `−0.01113, −0.00075, +0.00317`; RRT `−0.01158, +0.00071, +0.00398`; Ensemble `−0.01044, −0.00079, +0.00247`; RRT+Ensemble `−0.01158, +0.00097, +0.00426`. The observed row is an evaluator reference, not an achievable model upper bound.

Against OP-Baseline, OP-RRT has mean ΔC-index `−0.00046`, ΔBrier `+0.00146`, and ΔNLL `+0.00081`; there is no evidence of a downstream RRT gain in this run. OP-Ensemble improves Brier and NLL in 4/5 folds, but the mean changes are only `−0.00004` and `−0.00070`, with C-index improving in 2/5 folds. RRT+Ensemble does not improve on RRT in the fold-mean survival metrics.

Ensemble disagreement in PCA space was compared with absolute per-sample NLL difference from the observed transition. Mean Spearman correlation across 15 fold/seed tasks was `−0.061` for Ensemble and `−0.047` for RRT+Ensemble (positive in 5/15 tasks for each). Selective Risk@80 was lower than Risk@100 in 6/15 and 4/15 tasks, respectively. This run therefore does not establish useful uncertainty–error ranking.

## Integrity and interpretation

- 75 Outcome tasks, 60 complete fold-specific world entries (180 trained members), and 60 OP tasks completed; 2880 prediction records have unique IDs.
- All four OP variants use the same matched sample hash and frozen OT state hash within each fold/seed. Five PCA states are 64-dimensional and fitted on training-fold eligible landmarks.
- Validation folds are patient-disjoint. No prediction record belongs to a formal-test patient.
- Gate B passed, but the downstream RRT and RRT+Ensemble comparisons did not support Planner integration. The existing Planner remains unchanged.
- OP is factual conditional forecasting under recorded actions and MRI intervals. These observational prognostic comparisons do not identify causal treatment effects.

Machine-readable details: `run.json`, `metrics.json`, `models.pt`, and `predictions.jsonl`. Execution log: `logs/outcome_v2_clarity_v1.log` at the repository root.

# Cloop downstream evaluation with the CLARITY outcome architecture

- Protocol: `development_reuse`
- Evidence scope: development comparison; source worlds selected on source validation
- Dynamics are frozen; each variant has its own independently trained outcome head.
- H2 Cox-risk C-index is primary; H2 IPCW Brier@365 is the companion metric.

## Main full-cohort results

| Method | H | Patients/fold | Events/fold | Latent MSE | C-index (risk) | IPCW Brier@365 | TD-AUC@365 |
|---|---:|---|---|---:|---:|---:|---:|
| baseline | 1 | [22, 21, 19, 20, 20] | [11, 11, 11, 11, 10] | 0.9628 ± 0.1925 (k=5) | 0.5019 ± 0.0724 (k=5) | 0.2876 ± 0.0957 (k=5) | 0.4770 ± 0.1426 (k=5) |
| baseline | 2 | [14, 13, 10, 8, 15] | [9, 8, 8, 4, 7] | 0.9764 ± 0.1498 (k=5) | 0.4699 ± 0.1028 (k=5) | 0.2694 ± 0.1227 (k=5) | 0.4466 ± 0.1986 (k=4) |
| baseline | 3 | [11, 7, 6, 4, 7] | [6, 5, 5, 1, 3] | 0.9574 ± 0.3198 (k=5) | 0.3633 ± 0.1852 (k=4) | 0.3108 ± 0.2234 (k=5) | 0.4309 ± 0.0647 (k=3) |
| ensemble | 1 | [22, 21, 19, 20, 20] | [11, 11, 11, 11, 10] | 0.9091 ± 0.1918 (k=5) | 0.4801 ± 0.0641 (k=5) | 0.2831 ± 0.0935 (k=5) | 0.5089 ± 0.1452 (k=5) |
| ensemble | 2 | [14, 13, 10, 8, 15] | [9, 8, 8, 4, 7] | 0.8839 ± 0.1563 (k=5) | 0.4423 ± 0.1527 (k=5) | 0.2711 ± 0.1212 (k=5) | 0.4411 ± 0.1860 (k=4) |
| ensemble | 3 | [11, 7, 6, 4, 7] | [6, 5, 5, 1, 3] | 0.8065 ± 0.3235 (k=5) | 0.3692 ± 0.2020 (k=4) | 0.3127 ± 0.2450 (k=5) | 0.3634 ± 0.1366 (k=3) |
| rrt | 1 | [22, 21, 19, 20, 20] | [11, 11, 11, 11, 10] | 0.9496 ± 0.1748 (k=5) | 0.5260 ± 0.1047 (k=5) | 0.2796 ± 0.0911 (k=5) | 0.5677 ± 0.1147 (k=5) |
| rrt | 2 | [14, 13, 10, 8, 15] | [9, 8, 8, 4, 7] | 0.9411 ± 0.1398 (k=5) | 0.4880 ± 0.0910 (k=5) | 0.2869 ± 0.1492 (k=5) | 0.4995 ± 0.2122 (k=4) |
| rrt | 3 | [11, 7, 6, 4, 7] | [6, 5, 5, 1, 3] | 0.9047 ± 0.3015 (k=5) | 0.3369 ± 0.2020 (k=4) | 0.3195 ± 0.2027 (k=5) | 0.3696 ± 0.1942 (k=3) |
| rrt_ensemble | 1 | [22, 21, 19, 20, 20] | [11, 11, 11, 11, 10] | 0.9084 ± 0.1765 (k=5) | 0.5288 ± 0.1211 (k=5) | 0.2935 ± 0.0853 (k=5) | 0.4873 ± 0.1783 (k=5) |
| rrt_ensemble | 2 | [14, 13, 10, 8, 15] | [9, 8, 8, 4, 7] | 0.8704 ± 0.1561 (k=5) | 0.4800 ± 0.0956 (k=5) | 0.2907 ± 0.1545 (k=5) | 0.4734 ± 0.2289 (k=4) |
| rrt_ensemble | 3 | [11, 7, 6, 4, 7] | [6, 5, 5, 1, 3] | 0.7990 ± 0.3140 (k=5) | 0.2739 ± 0.0799 (k=4) | 0.3181 ± 0.2011 (k=5) | 0.3745 ± 0.3015 (k=3) |

## Pre-specified paired comparisons

- `rrt_ensemble_minus_baseline` at H2: ΔC-index 0.0101 ± 0.0975 (k=5); ΔBrier 0.0213 ± 0.0427 (k=5); C-index improved in 2/5 folds, Brier improved in 1/5 folds.
- `rrt_minus_baseline` at H2: ΔC-index 0.0182 ± 0.0828 (k=5); ΔBrier 0.0174 ± 0.0363 (k=5); C-index improved in 3/5 folds, Brier improved in 2/5 folds.
- `ensemble_minus_baseline` at H2: ΔC-index -0.0276 ± 0.0966 (k=5); ΔBrier 0.0017 ± 0.0060 (k=5); C-index improved in 2/5 folds, Brier improved in 2/5 folds.
- `rrt_ensemble_minus_ensemble` at H2: ΔC-index 0.0377 ± 0.0753 (k=5); ΔBrier 0.0197 ± 0.0475 (k=5); C-index improved in 3/5 folds, Brier improved in 2/5 folds.

## Uncertainty (separate inference-only analysis)

Only ensemble variants are included. Probability disagreement is not a clinical confidence interval and selective results do not replace the full-cohort main table.

| Method | H | Requested coverage | Actual coverage | Retained/fold | Events/fold | Selective Brier@365 | Random-retention Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| ensemble | 1 | 0.60 | 0.5883 ± 0.0128 (k=5) | 12.0000 ± 0.7071 (k=5) | 5.8667 ± 0.6912 (k=5) | 0.2462 ± 0.0832 (k=5) | 0.2825 ± 0.0898 (k=5) |
| ensemble | 1 | 0.80 | 0.7848 ± 0.0170 (k=5) | 16.0000 ± 0.7071 (k=5) | 8.2667 ± 0.5963 (k=5) | 0.2735 ± 0.0964 (k=5) | 0.2832 ± 0.0937 (k=5) |
| ensemble | 1 | 1.00 | 1.0000 ± 0.0000 (k=5) | 20.4000 ± 1.1402 (k=5) | 10.8000 ± 0.4472 (k=5) | 0.2830 ± 0.0934 (k=5) | 0.2830 ± 0.0934 (k=5) |
| ensemble | 2 | 0.60 | 0.5620 ± 0.0429 (k=5) | 6.8000 ± 1.9235 (k=5) | 3.9333 ± 1.4414 (k=5) | 0.2536 ± 0.1523 (k=5) | 0.2712 ± 0.1234 (k=5) |
| ensemble | 2 | 0.80 | 0.7810 ± 0.0215 (k=5) | 9.4000 ± 2.4083 (k=5) | 5.4000 ± 1.7385 (k=5) | 0.2678 ± 0.1502 (k=5) | 0.2714 ± 0.1227 (k=5) |
| ensemble | 2 | 1.00 | 1.0000 ± 0.0000 (k=5) | 12.0000 ± 2.9155 (k=5) | 7.2000 ± 1.9235 (k=5) | 0.2707 ± 0.1214 (k=5) | 0.2707 ± 0.1214 (k=5) |
| ensemble | 3 | 0.60 | 0.5377 ± 0.0360 (k=5) | 3.8000 ± 1.4832 (k=5) | 2.1333 ± 1.3864 (k=5) | 0.3133 ± 0.2508 (k=5) | 0.3080 ± 0.2367 (k=5) |
| ensemble | 3 | 0.80 | 0.7145 ± 0.0305 (k=5) | 5.0000 ± 1.8708 (k=5) | 2.8000 ± 1.5563 (k=5) | 0.2952 ± 0.1973 (k=5) | 0.3110 ± 0.2384 (k=5) |
| ensemble | 3 | 1.00 | 1.0000 ± 0.0000 (k=5) | 7.0000 ± 2.5495 (k=5) | 4.0000 ± 2.0000 (k=5) | 0.3101 ± 0.2372 (k=5) | 0.3101 ± 0.2372 (k=5) |
| rrt_ensemble | 1 | 0.60 | 0.5883 ± 0.0128 (k=5) | 12.0000 ± 0.7071 (k=5) | 6.6667 ± 0.6667 (k=5) | 0.3114 ± 0.0918 (k=5) | 0.2925 ± 0.0812 (k=5) |
| rrt_ensemble | 1 | 0.80 | 0.7848 ± 0.0170 (k=5) | 16.0000 ± 0.7071 (k=5) | 8.5333 ± 0.1826 (k=5) | 0.2939 ± 0.0673 (k=5) | 0.2933 ± 0.0852 (k=5) |
| rrt_ensemble | 1 | 1.00 | 1.0000 ± 0.0000 (k=5) | 20.4000 ± 1.1402 (k=5) | 10.8000 ± 0.4472 (k=5) | 0.2935 ± 0.0851 (k=5) | 0.2935 ± 0.0851 (k=5) |
| rrt_ensemble | 2 | 0.60 | 0.5620 ± 0.0429 (k=5) | 6.8000 ± 1.9235 (k=5) | 3.7333 ± 1.1402 (k=5) | 0.2513 ± 0.1490 (k=5) | 0.2898 ± 0.1545 (k=5) |
| rrt_ensemble | 2 | 0.80 | 0.7810 ± 0.0215 (k=5) | 9.4000 ± 2.4083 (k=5) | 5.1333 ± 1.5384 (k=5) | 0.2454 ± 0.1246 (k=5) | 0.2908 ± 0.1552 (k=5) |
| rrt_ensemble | 2 | 1.00 | 1.0000 ± 0.0000 (k=5) | 12.0000 ± 2.9155 (k=5) | 7.2000 ± 1.9235 (k=5) | 0.2900 ± 0.1540 (k=5) | 0.2900 ± 0.1540 (k=5) |
| rrt_ensemble | 3 | 0.60 | 0.5377 ± 0.0360 (k=5) | 3.8000 ± 1.4832 (k=5) | 2.4000 ± 1.1402 (k=5) | 0.3643 ± 0.2191 (k=5) | 0.3188 ± 0.2004 (k=5) |
| rrt_ensemble | 3 | 0.80 | 0.7145 ± 0.0305 (k=5) | 5.0000 ± 1.8708 (k=5) | 2.8000 ± 1.2383 (k=5) | 0.3325 ± 0.2041 (k=5) | 0.3206 ± 0.2026 (k=5) |
| rrt_ensemble | 3 | 1.00 | 1.0000 ± 0.0000 (k=5) | 7.0000 ± 2.5495 (k=5) | 4.0000 ± 2.0000 (k=5) | 0.3198 ± 0.2023 (k=5) | 0.3198 ± 0.2023 (k=5) |

## Limitations

- The development-reuse source worlds were selected using their source-fold validation cohorts.
- Outcome fit features are in-sample world forecasts whereas report features are out-of-sample.
- Single-token global latents preserve the CLARITY computation but not its original multi-token MRI representation.
- This is factual prognostic evaluation and does not establish causal treatment benefit.

# CLARITY outcome checkpoint and predicted-post diagnostics

- Source run: `next_stage_clarity_outcome_v1` (read-only)
- Diagnostic horizons: [2]
- Fixed derangements per task: 20
- Outcome heads were not retrained; uncertainty filtering was not run.

## A. Training and checkpoint audit

| Variant | H | Available | Epoch 0 | Trained checkpoint | Fit n | Stop n | Checkpoint consistent |
|---|---:|---:|---:|---:|---|---|---:|
| baseline | 1 | 15 | 4 | 11 | 64–66 | 16–17 | 15/15 |
| baseline | 2 | 15 | 10 | 5 | 35–43 | 8–11 | 15/15 |
| baseline | 3 | 15 | 1 | 14 | 18–25 | 4–7 | 15/15 |
| ensemble | 1 | 15 | 4 | 11 | 64–66 | 16–17 | 15/15 |
| ensemble | 2 | 15 | 6 | 9 | 35–43 | 8–11 | 15/15 |
| ensemble | 3 | 15 | 0 | 15 | 18–25 | 4–7 | 15/15 |
| rrt | 1 | 15 | 2 | 13 | 64–66 | 16–17 | 15/15 |
| rrt | 2 | 15 | 6 | 9 | 35–43 | 8–11 | 15/15 |
| rrt | 3 | 15 | 0 | 15 | 18–25 | 4–7 | 15/15 |
| rrt_ensemble | 1 | 15 | 2 | 13 | 64–66 | 16–17 | 15/15 |
| rrt_ensemble | 2 | 15 | 6 | 9 | 35–43 | 8–11 | 15/15 |
| rrt_ensemble | 3 | 15 | 0 | 15 | 18–25 | 4–7 | 15/15 |

Across all tasks, 41/180 selected epoch 0 and 139/180 selected a trained checkpoint. Checkpoint/log inconsistencies: 0.
At the diagnostic horizon(s), 28/60 selected epoch 0.

Selecting epoch 0 means the initialized head won the recorded stop-loss comparison; it does not mean that optimization was skipped.
In 41/41 epoch-0 tasks, train loss improved while the last stop loss was worse than initialization.

## B. Fixed-head predicted-post diagnostic

Positive shuffle ΔC and ΔB mean shuffling made the metric worse. Positive observed ΔC and ΔB mean observed-post substitution improved the metric relative to normal predicted post.

| Variant | H | Checkpoint | Normal C-index | Normal Brier | Shuffle ΔC | Shuffle ΔB | Observed C-index | Observed Brier | Mean abs Δprobability | Reproduced |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2 | trained (n=5) | 0.4977 ± 0.1034 (k=4) | 0.3262 ± 0.1554 (k=4) | -0.0617 ± 0.0734 (k=4) | -0.0125 ± 0.0120 (k=4) | 0.4985 ± 0.1188 (k=4) | 0.3121 ± 0.1456 (k=4) | 0.054048 ± 0.032920 (k=4) | 5/5 |
| baseline | 2 | initialization (n=10) | 0.4759 ± 0.1256 (k=5) | 0.2564 ± 0.1108 (k=5) | 0.0036 ± 0.1216 (k=5) | 0.0022 ± 0.0089 (k=5) | 0.4710 ± 0.0621 (k=5) | 0.2622 ± 0.1126 (k=5) | 0.019289 ± 0.002672 (k=5) | 10/10 |
| ensemble | 2 | trained (n=9) | 0.5118 ± 0.0890 (k=4) | 0.3235 ± 0.1493 (k=4) | -0.0102 ± 0.0314 (k=4) | -0.0045 ± 0.0081 (k=4) | 0.4740 ± 0.0889 (k=4) | 0.3216 ± 0.1514 (k=4) | 0.046662 ± 0.017028 (k=4) | 9/9 |
| ensemble | 2 | initialization (n=6) | 0.4014 ± 0.2220 (k=3) | 0.3044 ± 0.1334 (k=3) | -0.1039 ± 0.1886 (k=3) | -0.0044 ± 0.0035 (k=3) | 0.5227 ± 0.1700 (k=3) | 0.3046 ± 0.1283 (k=3) | 0.017705 ± 0.002451 (k=3) | 6/6 |
| rrt | 2 | trained (n=9) | 0.5221 ± 0.0858 (k=4) | 0.3194 ± 0.1500 (k=4) | 0.0046 ± 0.0529 (k=4) | 0.0132 ± 0.0378 (k=4) | 0.4548 ± 0.0653 (k=4) | 0.3169 ± 0.1448 (k=4) | 0.072815 ± 0.072862 (k=4) | 9/9 |
| rrt | 2 | initialization (n=6) | 0.4867 ± 0.0871 (k=4) | 0.2372 ± 0.1096 (k=4) | 0.0222 ± 0.1828 (k=4) | -0.0013 ± 0.0050 (k=4) | 0.4875 ± 0.1543 (k=4) | 0.2371 ± 0.1112 (k=4) | 0.017379 ± 0.003134 (k=4) | 6/6 |
| rrt_ensemble | 2 | trained (n=9) | 0.5007 ± 0.0634 (k=4) | 0.3254 ± 0.1543 (k=4) | -0.0301 ± 0.0794 (k=4) | 0.0024 ± 0.0147 (k=4) | 0.4377 ± 0.0590 (k=4) | 0.3207 ± 0.1503 (k=4) | 0.069670 ± 0.068813 (k=4) | 9/9 |
| rrt_ensemble | 2 | initialization (n=6) | 0.4912 ± 0.1369 (k=3) | 0.2412 ± 0.1353 (k=3) | -0.0303 ± 0.1797 (k=3) | -0.0038 ± 0.0042 (k=3) | 0.5674 ± 0.1510 (k=3) | 0.2407 ± 0.1348 (k=3) | 0.018542 ± 0.004350 (k=3) | 6/6 |

## Interpretation boundaries

- Each diagnostic condition uses the task's own saved checkpoint; epoch-0 tasks are initialization sensitivity, not learned post utilization.
- A shuffled post creates an unnatural pre/post/condition combination and is a sensitivity test, not a causal intervention.
- Observed post is evaluated with a head trained on predicted latents and may be distribution shifted; it is a reference, not a strict upper bound.
- Permutations are repeated perturbations of the same patients, not independent samples. They are averaged inside each task before seeds and folds are aggregated.
- These are development-reuse diagnostics and do not open or redefine the formal test set.

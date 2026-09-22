# Outcome diagnostics E0-E5

This patch implements the six development-only Outcome Head diagnostics from
`cloop_outcome_six_diagnostic_experiments.md`.

It deliberately does **not** modify the formal `brainiac_main_v1_provenance`
run. The runner reads that run only to recover its audited cache and registered
patient split, combines the original train+validation patients into a
development cohort, and leaves the formal test patients untouched.

Files:

- `src/cloop/outcome_diagnostics.py` — standalone runner and model definitions.
- `configs/outcome_diagnostics.yaml` — registered diagnostic defaults.
- `tests/test_outcome_diagnostics.py` — PCA, fold, shuffling, and residual-fusion tests.

Run:

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python

"$PY" -m pytest -q tests/test_outcome_diagnostics.py

CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop.outcome_diagnostics \
  --config configs/default.yaml \
  --paths configs/server.yaml \
  --diag-config configs/outcome_diagnostics.yaml
```

For a cheaper first pass:

```bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop.outcome_diagnostics \
  --config configs/default.yaml \
  --paths configs/server.yaml \
  --diag-config configs/outcome_diagnostics.yaml \
  --seeds 17
```

The diagnostic run writes only the usual flat artifacts under
`outputs/outcome_diag_v1/`:

- `run.json`
- `models.pt`
- `metrics.json`

E5 requires the matching E0 fold/seed checkpoint. The runner orders registered
experiments as E0→E5 and skips completed tasks on rerun.

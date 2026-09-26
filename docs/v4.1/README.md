# Cloop v4.1

This directory contains the v4.1 modification plan and the execution notes for
the corresponding code.

## Isolation from earlier versions

- Code entry points remain under `src/cloop/v4/`, as required by the plan.
  The v4.1 implementations are isolated in `*_v4_1.py` and `outcome.py`.
- Configuration lives at `configs/v4_1/next_experiment.yaml`.
- New artifacts are written below
  `outputs/v4_1/next_experiment_v4_1/`.
- Existing `src/cloop/v1` through `src/cloop/v3` and
  `outputs/v1` through `outputs/v4` are read-only inputs/history.

## Run

```bash
PYTHONPATH=src python -m cloop.v4.next_experiment \
  --experiment-config configs/v4_1/next_experiment.yaml \
  --stage outcome
```

Run `--stage outcome` before `--stage planner`; the dynamics stage is an
independent ablation. The `all` stage runs every experiment in dependency
order.

The outcome stage uses patient-level out-of-fold world rollouts for training.
The planner evaluates every ensemble trajectory with the survival model,
scores mean risk plus outcome-risk uncertainty, and reports open-loop,
closed-loop, MPC, and uncertainty-aware MPC separately.

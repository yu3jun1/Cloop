# CLARITY Loop

CLARITY Loop is a research framework for training and evaluating recursive-rollout (RRT) ensemble dynamics from frozen longitudinal MRI latents, linking predicted states to a separately trained censored-survival head, and testing receding-horizon replanning. It is intentionally offline by default and does not execute treatment.

The implementation keeps three evidence levels separate:

1. **Factual forecasting** conditions on recorded actions and recorded time intervals and scores future latent/outcome predictions.
2. **Observed replay** exposes only information available at each recorded observation and measures treatment-set agreement and system behavior. A recommendation never drives the historical next MRI.
3. **Independent synthetic closed loop** sends recommended actions to a non-medical toy environment whose transition code never calls the learned world model.

RRT, ensembles, and MPC are general methods. D1/D2 are internal controlled baselines and are not presented as a full reproduction of CLARITY. Planning uses an observational prognostic proxy, not an identified causal treatment effect.

## Install and verify

Use the existing server environment; do not replace its PyTorch build merely to install this package.

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python
"$PY" -m pip install -e ".[dev]"
"$PY" -m pytest -q
"$PY" -m cloop smoke --device cpu
```

`smoke` uses deterministic, in-memory, non-medical data. It needs no API key, no clinical files, and no network.

## Real-data workflow

Global options must precede the subcommand.

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python

"$PY" -m cloop --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 doctor
"$PY" -m cloop --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 prepare

CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop --config configs/default.yaml \
  --paths configs/server.yaml --run brainiac_v1 train --suite dynamics \
  --variants baseline rrt ensemble rrt_ensemble --seeds 7 17 29

CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop --config configs/default.yaml \
  --paths configs/server.yaml --run brainiac_v1 train --suite outcome --seeds 7 17 29

"$PY" -m cloop --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 evaluate --suite all --split validation
"$PY" -m cloop --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 freeze-protocol
"$PY" -m cloop --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 evaluate --suite all --split test
"$PY" -m cloop --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 report
```

For a complete-epoch recovery, repeat the matching training command with `--resume`. `last.pt` includes the model, optimizer, early-stopping state, data generator, and Python/NumPy/Torch RNG states. It is removed after the selected training suite finishes.

The optional legacy regression protocol must be a separate run:

```bash
"$PY" -m cloop --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_legacy --protocol legacy_stage1 prepare
```

## Independent synthetic experiment

```bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop --config configs/default.yaml \
  --paths configs/server.yaml --run synthetic_v1 synthetic --seeds 7 17 29
```

Its `A0/A1/A2` actions and state variables have no medical meaning. Results test feedback, replanning, perturbation recovery, constraints, and inference budgets only.

## Artifacts and privacy boundary

Each run directory is flat and contains at most:

- `run.json` — resolved configuration, provenance, patient split IDs, audit, protocol state;
- `models.pt` — best model states and train-only preprocessing;
- `last.pt` — the sole temporary resume point;
- `metrics.json` — training history and aggregate metrics;
- `predictions.jsonl` — optional idempotent per-window/per-decision records;
- `report.md` — the one human-readable report.

No per-patient directories, epoch checkpoints, branch tensors, API keys, raw prompts, MRI arrays, or online calls are created by default. External LLM payloads are constrained to catalog IDs and sanitized summaries; only the offline catalog and fake provider are required in v1.

See [`clarity_loop_from_scratch_experiment_design.md`](clarity_loop_from_scratch_experiment_design.md) for the registered protocol, metric definitions, data-quality rules, and interpretation boundaries.

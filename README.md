# CLARITY Loop — 版本目录

本仓库按实验阶段保存代码、配置、测试、说明、输出和日志。以下路径从仓库根目录执行。
各运行目录中的 `run.json`、`models.pt`、`metrics.json` 等仍保持扁平结构。

| 版本 | 对应历史 | 代码 | 配置 | 说明 | 结果 |
|---|---|---|---|---|---|
| v1 | `ff5cc8f` 起的基础 CLARITY Loop，含后续修复及正式运行 | `src/cloop/v1/` | `configs/v1/` | [v1 README](docs/v1/README.md) | `outputs/v1/brainiac_main_v1_provenance/` |
| v1_1 | `cedd406`，Outcome E0–E5 诊断 | `src/cloop/v1_1/` | `configs/v1_1/` | [v1_1 README](docs/v1_1/README.md) | `outputs/v1_1/outcome_diag_v1/` |
| v2 | `3b4a627`，Outcome-v2 | `src/cloop/v2/` | `configs/v2/` | [v2 README](docs/v2/README.md) | `outputs/v2/outcome_v2_clarity_v1/` |
| v3 | 当前下一阶段正式实验，尚未提交 | `src/cloop/v3/` | `configs/v3/` | [v3 README](docs/v3/README.md) | `outputs/v3/next_stage_v1/`、`outputs/v3/next_stage_toy_v1/` |

对应测试位于 `tests/<版本>/`；历史日志位于 `logs/v1/` 和 `logs/v2/`。
`configs/brainiac_latent_provenance.json` 保留在根配置目录，因为其路径参与历史运行签名。
基础配置 `configs/v1/server.yaml` 中的 `paths.output_root` 仍指向仓库的 `outputs/`，
由运行代码追加版本号和 run 名。历史结果文件仅迁移目录，v3 的 `run.json`
另记录目录迁移校验信息；已完成实验的指标和模型未重新计算。

## 运行入口

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python
"$PY" -m pytest -q
"$PY" -m cloop smoke --device cpu
"$PY" -m cloop.v1_1.outcome_diagnostics --help
"$PY" -m cloop.v2.outcome_v2 --help
"$PY" -m cloop.v3.next_stage --help
"$PY" -m cloop.v3.next_stage_report --help
```

`cloop` 和 `python -m cloop` 默认进入 v1；各阶段的正式执行命令及实验边界见相应 README。

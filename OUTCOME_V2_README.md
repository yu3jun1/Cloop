# Outcome-v2（CLARITY-inspired transition evaluator）

本轮实验使用独立入口 `src/cloop/outcome_v2.py`、配置
`configs/outcome_v2.yaml`、运行目录 `outputs/outcome_v2_clarity_v1/` 和日志
`logs/outcome_v2_clarity_v1.log`。旧的 E0–E5 diagnostics、正式 run、日志和
模型文件不会被写入。输出目录中的 `run.json` 使用
`cloop_outcome_v2_clarity_v1` schema；重新运行会跳过已完成任务。若配置或数据
签名变化，必须选择新的 `outcome_v2_` 前缀 run 名。

使用仓库登记的 Python 环境：

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python

CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop.outcome_v2 \
  --config configs/default.yaml --paths configs/server.yaml \
  --v2-config configs/outcome_v2.yaml --stage outcome

CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop.outcome_v2 \
  --config configs/default.yaml --paths configs/server.yaml \
  --v2-config configs/outcome_v2.yaml --stage world

CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop.outcome_v2 \
  --config configs/default.yaml --paths configs/server.yaml \
  --v2-config configs/outcome_v2.yaml --stage op
```

`--stage all` 依次运行三阶段。Outcome 阶段训练 O0、O1、内部临床对照
O0-matched、O1-matched 和 OT-observed。O1-matched 与 OT 使用相同 target
landmark、标签和 target history。fold 内 PCA 仅使用训练患者的有效 MRI landmark。
OT 的临床分支来自同 fold/seed 的 O0-matched；MRI transition residual 的末层
零初始化。所有阶段按患者做 5-fold CV（预注册 3-fold fallback），seeds 为
7、17、29。

`world` 阶段必须等所有 OT fold/seed checkpoint 完成，并且 Gate B 通过。
四种 Dynamics 在各 fold 的训练患者上独立训练，不能借用旧正式 run 的
checkpoint，否则旧训练/验证划分会污染 CV 评估。`op` 阶段将同一个冻结 OT
应用到 factual action、factual MRI interval 条件下的四种预测 transition。
`models.pt` 记录每 fold 的 PCA 均值、主成分、解释方差、训练患者 hash，
以及冻结 OT state hash；`metrics.json` 给出配对指标、Gate B、每种 OP 的
predicted-minus-observed gap 与可靠性诊断；`predictions.jsonl` 保存可核对的
fold/seed/landmark 记录。

这轮结果只属于观察性预后评估。Planner 接入取决于实验 Gate B 与 OP 结果，
此入口不会提前改动现有 Planner。

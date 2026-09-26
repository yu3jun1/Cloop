# 下一阶段实验代码

本轮使用独立入口 `python -m cloop.v3.next_stage`、配置
`configs/v3/next_stage.yaml`、输出 `outputs/v3/next_stage_v1/`。原有
`cloop`、Outcome-v2 代码和历史输出仅作为只读来源。若实验配置或来源模型
发生变化，请改用新的 `next_stage_` run 名；程序会检查来源模型 SHA-256，
拒绝在同一 run 中混用不同来源。

## 运行

使用仓库现有 Python 环境：

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python

# 可先只运行一个 fold/seed 检查环境；重复命令会跳过已完成任务
"$PY" -m cloop.v3.next_stage --stage dynamics --fold 0 --seed 7 --device cpu

# 其后运行完整的开发折实验
"$PY" -m cloop.v3.next_stage --stage dynamics
"$PY" -m cloop.v3.next_stage --stage outcome

# 独立非医疗 toy 环境；输出 outputs/v3/next_stage_toy_v1/
"$PY" -m cloop.v3.next_stage --stage planner

# 全部开发折和 toy 结果齐备后生成跨折报告
"$PY" -m cloop.v3.next_stage_report
```

各阶段也可用 `--stage all` 顺序执行。`--fold`、`--seed` 用于局部运行；
正式汇总时需完成配置里的全部折和种子。默认使用
`configs/v1/default.yaml`、`configs/v1/server.yaml` 和 `configs/v3/next_stage.yaml`。
如需 GPU，可使用 `CUDA_VISIBLE_DEVICES=7` 以及 `--device cuda:0`。

## CLARITY outcome 后续诊断

对 `next_stage_clarity_outcome_v1` 的训练/checkpoint 和预测 post latent
利用情况，使用独立的只读诊断入口：

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python
CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop.v3.clarity_outcome_diagnostics --device cuda:0
```

配置固定在 `configs/v3/clarity_outcome_diagnostics.yaml`。程序核对全部
训练日志与实际保存的 head 哈希，再对预先指定的 H2 运行 normal predicted
post、20 次患者内无固定点重排以及 observed post 替换。normal 不能复现原
指标时，该任务不会继续解释扰动结果。源目录保持只读，输出写入
`outputs/v3/next_stage_clarity_outcome_diagnostics_v1/`；epoch 0 与训练后
checkpoint 分层报告，且不运行 uncertainty 筛选或重新训练 outcome。

## 实验协议

- **动力学**：只读复用 Outcome-v2 按患者分折训练的 baseline、RRT、
  ensemble、RRT+ensemble 模型，并加入 persistence。对各折未见患者的事实
  action 与 MRI 时间间隔计算 MSE、CosSim、同一起点相对 H1 的 drift。
  Ensemble 另输出分歧与预测误差相关、误差三分位、分歧与真实 latent
  变化的相关，以及分歧与观察性生存结局的时间依赖 AUC。
- **Outcome**：每个 horizon 在相同的患者、目标时间点、标签、clinical
  与历史信息上比较 O0 终态 piecewise hazard、O1 轨迹 Transformer、
  O2 轨迹加 ensemble variance。三者均使用同一冻结的
  `rrt_ensemble` 事实条件预测轨迹。action token 表示进入当前状态前的
  action，时间从起始 MRI 起算；padding 被 attention mask 排除。
  训练折按患者再分成拟合与早停集合，外层验证折只作比较。
  输出 C-index、IPCW Brier、0–365 天积分 Brier、365 天时间依赖 AUC、
  生存 NLL 和校准三分位；缺少可比事件或删失支持时返回 null 和原因。
- **Planner**：沿用仓库已实现的独立 toy 环境比较 fixed、greedy、MPC
  和带 uncertainty penalty 的 MPC，运行结果写入新的
  `next_stage_toy_v1`。toy action 与真实治疗无关。

`outputs/v3/next_stage_v1/` 中的 `run.json` 记录来源哈希和已完成任务，
`models.pt` 只保存本轮 outcome 模型，`metrics.json` 保存聚合结果，
`predictions.jsonl` 保存可核对的事实预测。目录保持扁平。原正式测试集
不会进入本轮 fold 训练或评估。

## 解释边界

Outcome-v2 先前结果没有证明 RRT 的下游 survival 增益，也没有证明
ensemble disagreement 能稳定排序误差；本轮指标应检验这些假设，不能
预先把它们写成结论。折内 world 模型对训练折患者的预测是样本内特征，
因此 outcome 结果仍属探索性观察性分析。O0 是终态 hazard 结构对照，
不是完整复刻 CLARITY。历史病例不提供反事实治疗效果；planner 的真实
医疗收益不能由这些结果推出。

## 历史运行迁移

已完成的 v3 运行仅移动了文件位置。`run.json` 保留原实验签名和原实现哈希，
`layout_migration` 单独记录迁移前清单哈希及当前路径与导入适配后的运行签名。
复用已完成任务时会核验当前来源模型与迁移签名；如实际代码或配置变更，
必须使用新的 run 名。历史指标和模型文件不因目录整理而重算。

原始设计依据：[下一阶段实验设计](Cloop_next_experiment_design.md)、[下一阶段代码设计](Cloop_next_stage_code_design.md)。
正式结果见[跨折报告](../../outputs/v3/next_stage_v1/report.md)。

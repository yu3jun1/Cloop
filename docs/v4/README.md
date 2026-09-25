# Cloop v4 实验代码

v4 使用独立入口 `python -m cloop.v4.next_experiment`、配置
`configs/v4/next_experiment.yaml` 和输出目录
`outputs/v4/next_experiment_v1/`。v2 的患者分折 world checkpoint 是只读来源，
原正式测试集不会进入本轮训练或评价。

## 主要改动

- Dynamics 保留 baseline、ensemble、RRT、RRT+ensemble。Drift 严格计算为同一
  起点的 `L2(error_H) - L2(error_H1)`；MSE、CosSim 继续单独报告。
- O0–O3 依次比较终态、轨迹、轨迹加 delta latent、轨迹加 delta latent 与
  uncertainty。轨迹模型使用 attention pooling，不再使用 last-token pooling。
- O3 的 ensemble disagreement 不直接作为风险输入：它通过单调可靠性 gate、
  独立 uncertainty embedding 和 mean/max/slope 摘要进入编码器；辅助头用真实
  factual latent MSE 校准 disagreement。
- Dual-head 同时输出 piecewise-exponential survival 和固定时点 adverse-outcome
  value risk。value 只在“时点前已观察事件”或“随访超过该时点”的可辨识标签上
  训练；它是观察性预后代理，不是治疗效果。
- Planner 使用训练折 historical action catalog，按
  `score = predicted value risk + lambda * normalized disagreement` 做软惩罚，
  不按 uncertainty threshold 删除分支。

## 运行

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python

# 先做一个 fold/seed 的快速链路验证
"$PY" -m cloop.v4.next_experiment --stage dynamics --fold 0 --seed 7 --device cpu
"$PY" -m cloop.v4.next_experiment --stage outcome --fold 0 --seed 7 --device cpu
"$PY" -m cloop.v4.next_experiment --stage planner --fold 0 --seed 7 --device cpu

# 完整开发折实验；命令可重复执行，已完成任务会跳过
"$PY" -m cloop.v4.next_experiment --stage dynamics
"$PY" -m cloop.v4.next_experiment --stage outcome
"$PY" -m cloop.v4.next_experiment --stage planner
```

也可以用 `--stage all` 顺序执行。Planner 阶段依赖同 fold/seed 已完成的 dynamics
uncertainty scale 和 O3 checkpoint，因此单独运行时须保持上述顺序。

## 输出和指标

输出保持扁平结构：`run.json` 记录来源与实现哈希，`models.pt` 保存 v4 outcome
模型，`metrics.json` 保存 dynamics/outcome/planning 汇总，`predictions.jsonl`
保存可审计的逐样本预测与 attention 权重。

除 C-index、IBS、time-dependent AUC、校准分箱外，v4 还输出 disagreement–error
相关、error calibration 和 coverage–risk curve。Coverage–risk 仅作为选择性预测
诊断，不参与主模型筛样。临床 planner 的累计 predicted risk 和 disagreement
均属于 self-model 指标，不能解释为真实或反事实治疗收益。

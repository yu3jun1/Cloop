# Cloop 下一阶段完整实验设计方案

> 归档说明：这是该阶段的历史设计稿。目录结构和命令示例保留了原设计时的写法；当前可运行路径请参阅[ v3 README ](README.md)。


## 1. 当前研究定位

Cloop 当前目标不是简单替换 CLARITY 的 survival
predictor，而是构建一个能够支持长期递归预测、状态不确定性建模以及未来闭环规划的
medical world model。

当前已经完成或正在实现：

-   Frozen MRI latent 输入；
-   One-step dynamics baseline；
-   Recursive Rollout Training (RRT)；
-   Ensemble dynamics；
-   独立 outcome head；
-   Receding-horizon planner 框架。

当前限制：

-   Policy Agent 尚未接入 LLM；
-   暂不进行真实治疗方案自动生成；
-   暂不声称治疗因果效果。

因此下一阶段重点应该集中于：

1.  证明 RRT 是否缓解 long-horizon rollout error；
2.  证明 Ensemble 是否提供有效 uncertainty；
3.  设计 trajectory-aware outcome evaluation；
4.  验证闭环 replanning 的必要性（不依赖 LLM）。

------------------------------------------------------------------------

# 2. 总体实验假设

## Hypothesis 1

Recursive Rollout Training 可以缓解 teacher forcing 导致的 exposure
bias。

验证：

训练阶段： ground-truth latent 输入

推理阶段： previous prediction latent 输入

RRT 应降低 long-horizon drift。

## Hypothesis 2

Ensemble dynamics 可以提供有效 epistemic uncertainty。

验证：

ensemble disagreement 是否与：

-   latent prediction error；
-   disease progression uncertainty；

相关。

## Hypothesis 3

Trajectory-conditioned outcome model 比单状态 outcome model 更适合作为
world model evaluator。

验证：

疾病风险不仅由最终 latent 决定，而由整个演化过程决定。

------------------------------------------------------------------------

# 3. 实验阶段规划

# Phase 1: Dynamics Evaluation

## 目标

验证 RRT 对 world model prediction 的贡献。

## 方法比较

  Method              Recursive training   Ensemble
  ------------------- -------------------- ----------
  Persistence         No                   No
  Baseline Dynamics   No                   No
  RRT                 Yes                  No
  Ensemble            No                   Yes
  RRT + Ensemble      Yes                  Yes

## 输入

当前：

-   MRI latent z0
-   treatment action
-   time interval
-   clinical context

## 输出

未来 latent：

z1, z2, z3

------------------------------------------------------------------------

## Metrics

### 1. Latent MSE

计算：

-   MSE@1
-   MSE@2
-   MSE@3

重点观察：

随着 horizon 增加，error growth。

------------------------------------------------------------------------

### 2. Cosine similarity

评价 latent representation consistency。

------------------------------------------------------------------------

### 3. Drift accumulation

定义：

error(horizon)

绘制：

error vs prediction horizon。

预期：

RRT 曲线增长更慢。

------------------------------------------------------------------------

# Phase 2: Ensemble Uncertainty Evaluation

## 目标

验证 ensemble disagreement 是否具有预测价值。

## 输出

对于 ensemble:

mean latent:

μ

uncertainty:

σ

------------------------------------------------------------------------

## Metrics

### 1. Uncertainty-error correlation

计算：

Spearman:

corr(disagreement, prediction error)

如果有效：

高 disagreement 应对应高预测误差。

------------------------------------------------------------------------

### 2. Calibration

按照 uncertainty 分组：

low uncertainty

medium uncertainty

high uncertainty

比较：

真实 latent error。

------------------------------------------------------------------------

### 3. Risk uncertainty analysis

分析：

高 uncertainty patient 是否：

-   survival 更差；
-   progression 更快。

注意：

这里只做关联分析，不宣称 causal effect。

------------------------------------------------------------------------

# Phase 3: Outcome Model Redesign

## 动机

CLARITY survival predictor 主要处理：

(z_pre, z_post)

但是 Cloop 的核心贡献是 recursive trajectory。

因此 outcome model 应升级为 trajectory-aware evaluator。

------------------------------------------------------------------------

# Proposed Outcome Model

## 输入

Trajectory:

\[z0,z1,...,zH\]

Treatment history:

\[a0,a1,...\]

Uncertainty:

\[u0,u1,...\]

Clinical:

patient context

------------------------------------------------------------------------

## Architecture

推荐：

Temporal Transformer + Piecewise Hazard Head

结构：

trajectory tokens

↓

Transformer encoder

↓

trajectory embedding

↓

hazard prediction

↓

survival curve

------------------------------------------------------------------------

## 输出

piecewise hazard:

λ1, λ2,...λK

得到：

S(t)

------------------------------------------------------------------------

# Outcome Ablation

必须保留 CLARITY-compatible baseline。

## O1

CLARITY style:

final latent -\> survival head

## O2

Trajectory survival:

\[z0,...,zH\] -\> survival

## O3

Trajectory + uncertainty:

\[z0,...,zH\]+ensemble variance -\> survival

------------------------------------------------------------------------

## Metrics

### Survival

-   C-index
-   Integrated Brier Score
-   time-dependent AUC

### Calibration

-   calibration curve
-   predicted vs observed survival

------------------------------------------------------------------------

# Phase 4: Planner Evaluation (No LLM)

## 当前限制

Policy Agent 未接入 LLM。

因此不评价：

-   自动治疗推荐；
-   临床方案生成。

------------------------------------------------------------------------

## 替代方案

使用：

1.  Historical action catalog

或者：

2.  replay actions

进行 world-model planning evaluation。

------------------------------------------------------------------------

# 4. Closed-loop Evaluation

核心问题：

固定计划 vs 动态重规划。

------------------------------------------------------------------------

## Baselines

### B1 Fixed Plan

初始状态规划一次：

z0

↓

action sequence

之后不更新。

------------------------------------------------------------------------

### B2 Greedy Replanning

每一步：

重新选择最低风险 action。

------------------------------------------------------------------------

### B3 MPC

滚动窗口：

observe

↓

simulate

↓

choose first action

↓

next observation

------------------------------------------------------------------------

# Evaluation Metrics

## 1. Replanning frequency

衡量系统是否响应状态变化。

------------------------------------------------------------------------

## 2. Risk improvement

比较：

fixed plan risk

vs

replanned risk

定义：

ΔRisk

------------------------------------------------------------------------

## 3. Robustness

加入：

-   latent noise；
-   missing observation；
-   prediction perturbation。

评价：

planner 是否恢复。

------------------------------------------------------------------------

# Phase 5: Synthetic Closed-loop Environment

由于真实医疗数据无法验证 counterfactual treatment effect。

建立独立 toy environment。

目的：

验证：

-   feedback；
-   replanning；
-   uncertainty-aware control。

不用于证明临床有效性。

------------------------------------------------------------------------

# 5. 最终论文实验结构

## Table 1

Dynamics prediction

指标：

MSE / CosSim / Drift

证明：

RRT。

------------------------------------------------------------------------

## Table 2

Uncertainty

指标：

Calibration / Error correlation

证明：

Ensemble。

------------------------------------------------------------------------

## Table 3

Outcome

比较：

CLARITY survival head

vs

Trajectory survival head

证明：

trajectory representation。

------------------------------------------------------------------------

## Table 4

Closed-loop planning

比较：

Fixed

Greedy

MPC

MPC+uncertainty

证明：

adaptive rollout。

------------------------------------------------------------------------

# 6. 当前最优开发顺序

## Step 1

完成：

RRT + Ensemble dynamics benchmark

## Step 2

实现：

trajectory outcome model

不要先接 LLM。

## Step 3

完成：

offline MPC evaluation

## Step 4

最后：

接入 LLM policy agent。

------------------------------------------------------------------------

# 7. 预期贡献表述

最终贡献不应定义为：

"better survival prediction"

而应定义为：

"An uncertainty-aware recursive medical world model that improves
long-horizon disease trajectory modeling and enables adaptive
closed-loop planning."

# Cloop v4 下一阶段实验代码设计方案

## 目标

基于 v3 实验结果，v4 不再单纯优化 latent MSE，而重点解决：

1.  RRT + Ensemble 中 RRT 不提升 MSE 的问题；
2.  trajectory outcome model 无法充分利用疾病演化信息的问题；
3.  uncertainty 没有有效连接 survival / planning 的问题。

总体目标：

    Uncertainty-aware recursive world model

                |

    Trajectory representation

                |

    Survival + Value evaluator

                |

    Closed-loop planning

------------------------------------------------------------------------

# 1. RRT重新定位

v3结果显示：

-   Ensemble 提升 long-horizon latent prediction；
-   RRT主要改善 drift，而不是最终 MSE。

因此论文中不再声明：

> RRT improves prediction accuracy

而声明：

> RRT improves recursive rollout consistency under autoregressive
> deployment.

代码保留：

-   Ensemble dynamics
-   RRT training

评价增加：

## Drift metric

\[ Drift_H = \|\|z_H-`\hat `{=tex}z_H\|\| - \|\|z_1-`\hat `{=tex}z_1\|\|
\]

比较：

-   baseline
-   ensemble
-   RRT
-   RRT+ensemble

目标：

证明 RRT 降低长期误差累积。

------------------------------------------------------------------------

# 2. Outcome model v4设计

## 当前问题

v3:

    z0,z1,...,zH

    Transformer

    last token pooling

    hazard

问题：

1.  最后状态占主导；
2.  progression velocity 丢失；
3.  trajectory没有充分用于风险预测。

------------------------------------------------------------------------

# 3. 新Trajectory Encoder

## 输入token

每个时间点：

    z_t
    +
    Δz_t
    +
    previous action
    +
    time interval
    +
    uncertainty

其中：

\[ `\Delta `{=tex}z_t=z_t-z\_{t-1} \]

原因：

疾病风险更多与变化速度相关，而不是单独状态。

------------------------------------------------------------------------

## Pooling修改

删除：

    last hidden state

改为：

    attention pooling

学习不同时间点的重要性。

例如：

首次progression可能比最后MRI更重要。

------------------------------------------------------------------------

# 4. Dual-head Outcome Model

结构：

    Trajectory Encoder

            |

    trajectory embedding

            |

    -------------------

    |                 |

    Survival Head    Value Head

------------------------------------------------------------------------

## Survival Head

保留：

piecewise exponential survival。

输出：

\[ `\lambda`{=tex}\_1,`\lambda`{=tex}\_2,...,`\lambda`{=tex}\_K \]

Loss:

    survival NLL
    +
    ranking loss

评价：

-   C-index
-   IBS
-   time-dependent AUC

------------------------------------------------------------------------

## Value Head

新增：

用于planner。

预测：

\[ V(z,a) \]

作用：

评价candidate rollout。

区别：

survival回答：

> 患者未来风险是多少？

value回答：

> 该治疗轨迹是否值得选择？

------------------------------------------------------------------------

# 5. Uncertainty重新设计

## 当前问题

v3：

    trajectory + uncertainty

    concat

    survival

问题：

uncertainty不是risk。

------------------------------------------------------------------------

# 5.1 Uncertainty embedding

来自ensemble:

\[ u_t=Var(z_t) \]

提取：

    mean uncertainty

    max uncertainty

    uncertainty slope

生成：

uncertainty embedding。

------------------------------------------------------------------------

# 5.2 Uncertainty attention gating

不要直接concat。

使用：

\[ attention=f(u_t) \]

让模型降低低可靠trajectory区域权重。

------------------------------------------------------------------------

# 5.3 Uncertainty calibration loss

增加辅助任务：

预测未来latent error：

\[ L_u=\|`\hat `{=tex}e-u\| \]

目标：

让uncertainty真正代表prediction confidence。

------------------------------------------------------------------------

# 6. Planner连接

不用LLM时：

使用：

-   historical action catalog
-   replay actions

流程：

    candidate action

          |

    world model rollout

          |

    trajectory

          |

    value head

          |

    select action

------------------------------------------------------------------------

# 7. Uncertainty如何进入planner

不采用：

    uncertainty > threshold

    discard trajectory

原因：

高uncertainty可能代表：

-   rare progression
-   high-risk patient

改：

soft penalty:

\[ Score=Risk+`\lambda `{=tex}U \]

其中：

Risk来自value head。

U来自ensemble disagreement。

------------------------------------------------------------------------

# 8. 实验设计

## Experiment 1 Dynamics

比较：

    baseline
    ensemble
    rrt
    rrt+ensemble

指标：

-   MSE@1/2/3
-   CosSim
-   Drift

目标：

证明：

Ensemble提升accuracy。

RRT提升recursive stability。

------------------------------------------------------------------------

## Experiment 2 Outcome Ablation

比较：

### O0

CLARITY style:

    final latent -> survival

### O1

trajectory:

    z0...zH

### O2

trajectory + velocity:

    z + Δz

### O3

trajectory + velocity + uncertainty

指标：

-   C-index
-   IBS
-   AUC
-   calibration

------------------------------------------------------------------------

## Experiment 3 Uncertainty

验证：

### Prediction calibration

uncertainty vs latent error

### Survival calibration

with uncertainty vs without uncertainty

### Selective prediction

额外实验：

discard high uncertainty samples

输出：

coverage-risk curve

不是主模型。

------------------------------------------------------------------------

## Experiment 4 Closed-loop planning

比较：

    fixed plan

    greedy

    MPC

    MPC + uncertainty penalty

指标：

-   cumulative predicted risk
-   robustness
-   uncertainty-aware selection

------------------------------------------------------------------------

# 9. 开发顺序

## Step 1

实现 trajectory encoder：

-   Δlatent
-   attention pooling

## Step 2

实现 dual-head outcome：

-   survival
-   value

## Step 3

实现 uncertainty embedding:

-   uncertainty token
-   calibration loss

## Step 4

连接MPC

## Step 5

接入LLM policy

------------------------------------------------------------------------

# 最终目标

Cloop v4 从：

    future latent prediction

升级为：

    recursive disease simulation

    +

    trajectory-based prognosis

    +

    uncertainty-aware decision evaluation

    +

    adaptive closed-loop planning

核心贡献不再是单次预测误差，而是长期递归疾病模拟和闭环决策能力。

# Cloop 下一阶段实验代码设计方案

> 归档说明：这是该阶段的历史设计稿。目录结构和命令示例保留了原设计时的写法；当前可运行路径请参阅[ v3 README ](README.md)。


版本：v1.0

## 1. 文档目标

本文针对当前 Cloop 仓库状态，设计下一阶段实验代码实现方案。

目标不是直接实现完整 LLM treatment agent，而是在当前框架下完成：

1.  RRT 是否缓解 recursive rollout distribution shift；
2.  Ensemble dynamics 是否提供有效 uncertainty；
3.  trajectory-aware outcome model 是否能够更好连接 latent rollout 与
    survival；
4.  offline closed-loop replanning 是否优于 open-loop planning。

当前阶段不评价真实治疗推荐准确性，不声明治疗因果效果。

------------------------------------------------------------------------

# 2. 当前代码结构调整建议

当前：

    src/cloop/

    data.py
    world.py
    outcome.py
    planner.py
    policy.py
    metrics.py
    engine.py

建议扩展：

    src/cloop/

    world.py
        - OneStepDynamics
        - RRTDynamics
        - EnsembleWorldModel

    trajectory.py       # 新增
        - rollout trajectory processing
        - positional/time encoding

    outcome.py          # 重构
        - StateOutcomeHead
        - TrajectorySurvivalHead
        - UncertaintyAwareOutcomeHead

    metrics.py
        - dynamics metrics
        - survival metrics
        - uncertainty metrics

    planner.py
        - MPC planner
        - replay planner

    experiments/
        dynamics_eval.py
        outcome_eval.py
        planner_eval.py

------------------------------------------------------------------------

# 3. Phase 1: Dynamics Benchmark

## 3.1 目标

比较：

-   baseline one-step dynamics
-   RRT dynamics
-   ensemble dynamics
-   RRT ensemble

验证 long horizon prediction。

------------------------------------------------------------------------

# 3.2 world.py 修改

增加统一接口：

``` python
class RolloutDynamics(nn.Module):

    def rollout(
        self,
        z0,
        actions,
        delta_days,
        clinical,
        history
    ):
        pass
```

所有 dynamics 使用同一接口。

------------------------------------------------------------------------

## 3.3 Rollout 输出格式

新增：

``` python
@dataclass
class Trajectory:

    states:
        Tensor

        shape:
        [B,H+1,D]

    actions:
        [B,H,A]

    uncertainty:
        [B,H+1]

    timestamps:
        [B,H+1]
```

------------------------------------------------------------------------

# 3.4 Dynamics evaluation

新增：

metrics.py

## Latent MSE

``` python
def latent_mse(
    pred,
    target
):

    return ((pred-target)**2).mean()
```

计算：

    MSE@1
    MSE@2
    MSE@3

------------------------------------------------------------------------

## Cosine similarity

``` python
cosine_similarity(
    pred,
    target
)
```

------------------------------------------------------------------------

## Drift

定义：

    drift_h =
    distance(pred_h,target_h)
    -
    distance(pred_1,target_1)

评价 error accumulation。

------------------------------------------------------------------------

# 4. Phase 2: Ensemble uncertainty

## 4.1 Ensemble输出

修改：

world.py

增加：

``` python
def predict_distribution():

    states = []

    for member in ensemble:
        states.append(
            member.rollout()
        )

    states=torch.stack(states)

    mean=states.mean(0)

    variance=states.var(0)

    return mean,variance
```

------------------------------------------------------------------------

# 4.2 Uncertainty metrics

新增：

metrics.py

## Error-Uncertainty correlation

``` python
spearmanr(
    uncertainty,
    prediction_error
)
```

验证：

ensemble disagreement 是否反映真实错误。

------------------------------------------------------------------------

## Calibration

按照 uncertainty 分桶：

    low
    medium
    high

计算：

    mean prediction error

------------------------------------------------------------------------

# 5. Phase 3: Trajectory Outcome Model

## 5.1 设计目标

替代当前：

    latent
     |
    MLP
     |
    survival

改为：

    z0,z1,...,zH

    +
    action history

    +
    uncertainty

    +
    clinical

            |

    Trajectory Transformer

            |

    Hazard head

            |

    Survival curve

------------------------------------------------------------------------

# 5.2 新文件

outcome.py

新增：

``` python
class TrajectorySurvivalHead(nn.Module):

    def __init__(
        self,
        latent_dim,
        action_dim,
        clinical_dim,
        hidden_dim
    ):
        pass
```

------------------------------------------------------------------------

# 5.3 输入

trajectory:

    [B,H,D]

action:

    [B,H,A]

uncertainty:

    [B,H,1]

clinical:

    [B,C]

------------------------------------------------------------------------

# 5.4 Token设计

每个时间点：

    token_t =
    concat(
    z_t,
    action_t,
    uncertainty_t,
    time_embedding
    )

得到：

    [B,H,E]

------------------------------------------------------------------------

# 5.5 Transformer

``` python
encoder_layer =
nn.TransformerEncoderLayer(
    d_model=hidden_dim,
    nhead=8
)


encoder =
nn.TransformerEncoder(
    encoder_layer,
    num_layers=3
)
```

------------------------------------------------------------------------

# 5.6 Survival output

保持当前 piecewise exponential:

输出：

    lambda_1
    lambda_2
    ...
    lambda_K

计算：

    survival(t)

------------------------------------------------------------------------

# 5.7 Outcome ablation

必须实现三个版本：

## O0

CLARITY compatible:

    z_final

    ->

    SurvivalHead

## O1

trajectory:

    [z0,z1,...zH]

    ->

    TrajectoryHead

## O2

trajectory + uncertainty:

    [z0...zH]+sigma

    ->

    TrajectoryHead

------------------------------------------------------------------------

# 6. Phase 4: Offline MPC Planner

## 6.1 当前不用LLM

Policy来源：

    historical action catalog

例如：

    action_1
    action_2
    action_3

------------------------------------------------------------------------

# 6.2 Planner接口

planner.py

新增：

``` python
class MPCPlanner:


    def plan(
        self,
        current_state,
        candidate_actions
    ):

        rollout trajectories

        evaluate outcome

        return best_action
```

------------------------------------------------------------------------

# 6.3 比较方法

## Fixed plan

只规划一次。

    z0

    plan

    execute all

------------------------------------------------------------------------

## Greedy

每一步重新选择：

    argmin risk

------------------------------------------------------------------------

## MPC

预测H步：

    simulate

    evaluate

    execute first action

    repeat

------------------------------------------------------------------------

# 7. Planner Evaluation

## Metric 1

Risk reduction

定义：

    Risk_fixed

    -

    Risk_MPC

------------------------------------------------------------------------

## Metric 2

Plan revision

记录：

    action_t
    action_t+1

变化。

------------------------------------------------------------------------

## Metric 3

Robustness

加入：

latent noise:

    z=z+epsilon

重新规划。

评价：

risk deviation。

------------------------------------------------------------------------

# 8. 实验运行入口设计

新增：

    experiments/

------------------------------------------------------------------------

## dynamics

运行：

``` bash
python experiments/dynamics_eval.py \
--variant baseline rrt ensemble rrt_ensemble
```

输出：

    outputs/dynamics/
    metrics.json

------------------------------------------------------------------------

## outcome

运行：

``` bash
python experiments/outcome_eval.py
```

输出：

    outputs/outcome/
    cindex
    brier
    auc

------------------------------------------------------------------------

## planner

运行：

``` bash
python experiments/planner_eval.py
```

输出：

    outputs/planner/
    risk
    revision
    robustness

------------------------------------------------------------------------

# 9. 推荐实现顺序

## Step 1

不要改 planner。

完成：

RRT + Ensemble dynamics evaluation。

目标：

证明：

long horizon improvement。

------------------------------------------------------------------------

## Step 2

实现：

TrajectorySurvivalHead。

目标：

解决 latent improvement 无法传递到 survival 的问题。

------------------------------------------------------------------------

## Step 3

加入 uncertainty feature。

目标：

证明 ensemble 不只是 ensemble averaging。

------------------------------------------------------------------------

## Step 4

实现 offline MPC。

目标：

证明闭环比 open-loop 更适合动态疾病过程。

------------------------------------------------------------------------

# 10. 最终论文实验映射

贡献1：

RRT

对应：

Dynamics Table

指标：

MSE/CosSim/Drift

------------------------------------------------------------------------

贡献2：

Ensemble uncertainty

对应：

Calibration Table

指标：

uncertainty-error correlation

------------------------------------------------------------------------

贡献3：

Trajectory outcome model

对应：

Survival Table

指标：

C-index/Brier

------------------------------------------------------------------------

贡献4：

Closed-loop planner

对应：

Planning Table

指标：

risk reduction / robustness

------------------------------------------------------------------------

# 11. 当前阶段明确不做

-   LLM treatment generation
-   online reinforcement learning
-   causal treatment effect estimation
-   automatic clinical recommendation

这些应该作为后续阶段。

# Cloop v4.1 代码修改设计方案

## 目标

基于 Cloop v4
实验代码审查结果，本版本不是继续增加模型复杂度，而是针对当前 v4
的核心问题进行修正。

主要问题：

1.  Outcome model 的预测目标与 closed-loop planning 目标不完全一致；
2.  trajectory encoder 没有充分利用时间信息；
3.  uncertainty calibration 没有真正影响 outcome/planning；
4.  value head 与 survival head 学习重复风险，没有形成真正 value
    function；
5.  planner 当前不是完整 closed-loop rollout；
6.  world model feature 训练和 evaluation 存在分布差异。

v4.1 目标：

    Recursive world model

            |

    Trajectory distribution

            |

    Calibrated outcome evaluator

            |

    Closed-loop planning

------------------------------------------------------------------------

# 1. 修改总体原则

## 不修改

保留：

-   Ensemble Dynamics
-   RRT Dynamics
-   BrainIAC latent
-   patient-level split

原因：

当前 dynamics 已经证明有效。

## 优先修改

重点：

1.  outcome representation；
2.  uncertainty integration；
3.  planner evaluation protocol。

------------------------------------------------------------------------

# 2. Outcome Model 修改

## 2.1 当前问题

当前：

    predicted trajectory

    [z0,z1,...zH]

    Transformer

    attention pooling

    survival/value

存在：

-   缺少绝对时间位置；
-   缺少 observed/predicted 区分；
-   trajectory 与 endpoint 信息混合；
-   value head 和 survival head 重复。

------------------------------------------------------------------------

# 2.2 新输入设计

每个 timestep token:

    token_t =

    latent z_t

    +

    delta latent

    +

    normalized velocity

    +

    previous action

    +

    delta time

    +

    cumulative time

    +

    observation flag

    +

    terminal flag

其中：

## delta latent

\[ `\Delta `{=tex}z_t=z_t-z\_{t-1} \]

## velocity

\[ v_t= `\frac{z_t-z_{t-1}}`{=tex} {`\Delta `{=tex}t_t+`\epsilon`{=tex}}
\]

避免把 latent change 误认为 progression rate。

------------------------------------------------------------------------

# 2.3 增加时间编码

新增：

``` python
TemporalEmbedding
```

输入：

    relative time
    absolute time

例如：

``` python
time_embedding = MLP(
    [
        log1p(delta_days),
        log1p(cumulative_days)
    ]
)
```

加入 token:

``` python
token =
concat(
    latent,
    delta_latent,
    velocity,
    action,
    time_embedding
)
```

------------------------------------------------------------------------

# 2.4 增加 observed / predicted embedding

原因：

当前 outcome 输入：

全部来自 world rollout。

但是：

真实临床：

    observed MRI

    +

    future simulation

应该区分。

新增：

``` python
state_source_embedding
```

其中：

    0:
    observed


    1:
    predicted

------------------------------------------------------------------------

# 2.5 保留 endpoint shortcut

不要完全依赖 Transformer。

修改：

当前：

    trajectory embedding

    ↓

    survival

改：

    trajectory encoder

            |

    trajectory embedding

            |

    concat

            |

    final latent embedding

            |

    survival head

代码：

``` python
joint_embedding = concat(
    trajectory_embedding,
    final_state_embedding
)
```

原因：

CLARITY 已经证明 post-treatment latent 有价值。

trajectory 应该作为增强，而不是替代。

------------------------------------------------------------------------

# 3. Outcome Head 修改

## 3.1 删除独立 value BCE

当前：

    survival head

    value head

两个任务：

预测同一个固定 horizon risk。

修改：

第一阶段：

删除：

``` python
value_loss
```

使用：

\[ Value(z,a)=1-S(t) \]

即：

``` python
value = 1-survival_probability
```

------------------------------------------------------------------------

## 3.2 保留 Value Interface

为了未来 MPC：

保留 API：

``` python
model.value()
```

但是：

实现：

``` python
return 1-survival()
```

而不是独立网络。

------------------------------------------------------------------------

## 3.3 后续如果需要独立value

必须满足：

加入：

-   counterfactual reward；
-   treatment response label；
-   offline RL objective。

否则不能称为 treatment value。

------------------------------------------------------------------------

# 4. Survival Loss 修改

## 当前：

    survival NLL

    +

    ranking loss

    +

    value BCE

    +

    uncertainty calibration

修改：

v4.1:

    survival NLL
    +
    optional ranking
    +
    uncertainty auxiliary

------------------------------------------------------------------------

# 4.1 Ranking loss处理

当前问题：

ranking pair 没有限制患者来源。

修改：

增加：

``` python
patient_id
```

只允许：

不同患者 pair。

避免：

同一患者不同 horizon 产生伪排序。

------------------------------------------------------------------------

# 4.2 Validation loss

当前：

batch mean average。

修改：

累计：

``` python
total_loss / total_samples
```

避免最后 batch 权重异常。

------------------------------------------------------------------------

# 5. Uncertainty 修改方案

## 5.1 当前问题

当前：

uncertainty:

    latent variance

    ↓

    feature/gate

但是：

calibrated error:

没有进入 prediction。

------------------------------------------------------------------------

# 5.2 修改 uncertainty pipeline

改为：

    ensemble members

            |

    latent distribution

            |

    trajectory uncertainty

            |

    outcome distribution

            |

    survival uncertainty

------------------------------------------------------------------------

# 5.3 Outcome uncertainty propagation

不要：

    mean latent

    ↓

    outcome

改：

对于每个 ensemble member:

``` python
for member in ensemble:

    trajectory_i

    risk_i = outcome(trajectory_i)
```

得到：

\[ {R_1,R_2,...R_M} \]

输出：

mean risk:

\[ `\bar `{=tex}R \]

uncertainty:

\[ Var(R) \]

------------------------------------------------------------------------

# 5.4 新指标

增加：

## Outcome disagreement

不是：

latent disagreement

而是：

\[ `\sigma`{=tex}\_R\^2 = Var(P(T)) \]

更符合：

    world uncertainty

            |

    clinical uncertainty

------------------------------------------------------------------------

# 5.5 Selective prediction

保留：

coverage-risk。

但是改：

当前：

    latent error

改：

    survival prediction error

    or

    Brier score

评价：

uncertainty 是否帮助选择可靠预测。

------------------------------------------------------------------------

# 6. Cross-fitting 修改

## 当前问题

world model feature:

训练集生成。

导致：

outcome training:

看到过于理想的 rollout。

------------------------------------------------------------------------

## 修改

采用：

patient-level cross fitting。

例如：

5 folds:

fold0:

world train:

train folds

outcome feature:

held-out fold

得到：

out-of-fold trajectory feature。

然后：

训练 outcome。

------------------------------------------------------------------------

# 7. Planner 修改

## 7.1 当前问题

当前：

一次：

    initial state

    ↓

    plan

不是完整 closed-loop。

------------------------------------------------------------------------

# 7.2 新流程

实现：

``` python
for stage in horizon:

    current_state

    candidate actions

    rollout

    select action

    observe next state

    update belief
```

------------------------------------------------------------------------

# 7.3 新baseline

重新定义：

## Open-loop

一次生成：

    a1,a2,a3

执行。

------------------------------------------------------------------------

## Closed-loop

每阶段：

重新规划。

------------------------------------------------------------------------

## MPC

滚动窗口。

------------------------------------------------------------------------

# 7.4 Uncertainty-aware MPC

score:

\[ Score= Risk+ `\lambda `{=tex}`\sigma`{=tex}\_R \]

其中：

\[ `\sigma`{=tex}\_R \]

来自 outcome risk variance。

不是 latent variance。

------------------------------------------------------------------------

# 8. 新实验设计

## Experiment 1

Dynamics

比较：

    baseline
    ensemble
    rrt
    rrt+ensemble

指标：

-   MSE
-   CosSim
-   exposure gap
-   drift

------------------------------------------------------------------------

## Experiment 2

Outcome ablation

O0:

terminal latent

O1:

trajectory

O2:

trajectory + delta latent

O3:

trajectory + uncertainty propagation

指标：

-   C-index
-   IBS
-   AUC
-   calibration

------------------------------------------------------------------------

## Experiment 3

Uncertainty

比较：

latent uncertainty

vs

risk uncertainty

指标：

-   error correlation
-   selective prediction

------------------------------------------------------------------------

## Experiment 4

Planning

比较：

-   open-loop
-   closed-loop
-   MPC
-   uncertainty MPC

指标：

-   predicted risk
-   replanning rate
-   uncertainty reduction

------------------------------------------------------------------------

# 9. 文件修改列表

## src/cloop/v4/trajectory.py

修改：

-   TemporalEmbedding
-   source embedding
-   terminal shortcut
-   uncertainty propagation

------------------------------------------------------------------------

## src/cloop/v4/outcome.py

修改：

-   remove independent value loss
-   survival-derived value

------------------------------------------------------------------------

## src/cloop/v4/planner.py

修改：

-   closed-loop execution
-   risk uncertainty penalty

------------------------------------------------------------------------

## src/cloop/v4/next_experiment.py

修改：

-   cross fitting
-   ensemble member outcome evaluation
-   new metrics

------------------------------------------------------------------------

## src/cloop/v4/metrics.py

新增：

-   outcome uncertainty calibration
-   survival selective prediction

------------------------------------------------------------------------

# 最终目标

v4.1 不追求更大的模型，而解决当前最大逻辑断点：

当前：

    latent prediction

    ↓

    survival prediction

升级为：

    uncertain disease trajectory

    ↓

    calibrated future risk distribution

    ↓

    adaptive closed-loop planning

这才与 Cloop 的核心贡献一致。

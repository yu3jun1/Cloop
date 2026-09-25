# Cloop Outcome-v2 实验设计

> 归档说明：这是该阶段的历史设计稿。目录结构和命令示例保留了原设计时的写法；当前可运行路径请参阅[ v2 README ](README.md)。

## ——固定 E5 作为单状态 MRI baseline，采用 CLARITY-inspired Transition-aware Frozen Outcome Evaluator

**项目：** CLARITY Loop / Cloop  
**版本定位：** Outcome-v2 重新设计  
**核心研究目标：** 将 Outcome Model 从“需要持续创新和调参的研究对象”降级为**固定、共享、可解释的预后评估器**，使论文主要贡献重新聚焦于：

\[
\boxed{
\text{RRT}
+
\text{Ensemble Learning}
+
\text{Closed-loop / Receding-horizon Replanning}
}
\]

本设计不再把 Outcome Head 本身作为主要创新点，而是尽量采用与 CLARITY 思路一致的 transition-aware evaluator，并通过固定共享 Outcome Model，公平比较不同 Dynamics World Model 的 downstream 表现。

---

# 1. 设计动机

前一阶段已经完成 E0–E5 Outcome diagnostics：

```text
E0  Clinical + mask + history
E1  MRI-only raw768
E2  Raw768 MRI + clinical early fusion
E3  Shuffled MRI negative control
E4  PCA64 MRI + clinical early fusion
E5  PCA64 MRI residual on top of clinical baseline
```

诊断结果支持以下工作结论：

1. BrainIAC MRI latent 并非完全没有 prognostic signal；
2. raw 768-D MRI 输入在当前 survival cohort 上存在明显过拟合；
3. PCA64 显著缓解高维过拟合；
4. early fusion 会干扰 clinical baseline；
5. E5 的 residual fusion 更稳定，并可作为 single-state MRI baseline；
6. Outcome Model 不应继续成为主要 architecture search 对象。

因此，本轮 Outcome-v2 的设计目标从：

> “继续寻找最强 survival architecture”

改为：

> “建立一个固定、共享、transition-aware 的 Outcome Evaluator，用于公平评价 Baseline / RRT / Ensemble / RRT+Ensemble 生成的 future states。”

---

# 2. Outcome-v2 的研究定位

论文中 Outcome Model 的角色应定义为：

\[
\boxed{
\text{Frozen prognostic evaluator}
}
\]

而不是：

\[
\boxed{
\text{主要方法创新}
}
\]

最终系统结构：

\[
\text{BrainIAC state}
\rightarrow
\text{World Model}
\rightarrow
\text{Frozen Outcome Evaluator}
\rightarrow
\text{Planner}
\]

其中 World Model 分别为：

```text
Baseline
RRT
Ensemble
RRT + Ensemble
```

而 Outcome Evaluator：

\[
G_\psi
\]

在所有 Dynamics variants 中保持完全相同。

因此 downstream performance 的差异更容易归因于：

\[
\boxed{
\text{future-state prediction quality}
}
\]

而不是 Outcome Model 本身的变化。

---

# 3. 本轮不再继续优化的内容

本轮明确不再大规模搜索：

```text
raw 768-D MRI early fusion
PCA32 / PCA64 / PCA128 grid
different residual hidden sizes
large dropout search
large weight-decay search
different hazard-bin counts
Cox vs piecewise-exponential model search
Two-Way Transformer architecture search
BrainIAC LoRA
joint Dynamics-Outcome training
end-to-end MRI encoder tuning
```

其中：

\[
PCA64
\]

固定为 MRI reduction，因为上一轮已验证其显著缓解过拟合。

---

# 4. 本轮核心实验问题

Outcome-v2 只回答三个问题。

## Q1：当前单状态 MRI 是否能在 clinical baseline 之上提供增量 prognostic value？

使用 E5-style single-state residual。

## Q2：Observed MRI transition 是否比 single-state MRI 更具有 prognostic value？

比较：

\[
z_{t+1}
\]

与：

\[
[z_t,z_{t+1},z_{t+1}-z_t].
\]

## Q3：不同 World Model 能否保留 observed transition 中与 survival 相关的信息？

固定同一个 Outcome Evaluator，将 observed target：

\[
z_{t+1}^{obs}
\]

替换为：

\[
\hat z_{t+1}^{variant}.
\]

比较：

```text
Baseline
RRT
Ensemble
RRT+Ensemble
```

---

# 5. 实验组总体设计

本轮只保留四类核心实验：

```text
O0   Clinical-only baseline
O1   E5-style single-state MRI residual
OT   Observed transition-aware Outcome
OP   Predicted transition-aware Outcome
```

其中 OP 再细分为：

```text
OP-Baseline
OP-RRT
OP-Ensemble
OP-RRT+Ensemble
```

---

# 6. O0：Clinical-only Baseline

## 6.1 输入

\[
x_{O0}=[c,m,h].
\]

其中：

- \(c\)：static clinical/genomic features；
- \(m\)：clinical missingness mask；
- \(h\)：当前 landmark 的 treatment history。

不使用 MRI。

## 6.2 作用

O0 是最基础 prognostic baseline，用于判断 MRI 是否提供 clinical/history 之外的增量信息。

---

# 7. O1：E5-style Single-state MRI Residual

## 7.1 MRI 表征

固定：

\[
e_t=PCA_{64}(z_t).
\]

PCA 只能在 training fold fit。

## 7.2 Outcome 结构

Clinical branch：

\[
q_c=G_c(c,m,h).
\]

MRI residual branch：

\[
r_z=R_z(e_t).
\]

最终 hazard logits：

\[
\boxed{
q=q_c+r_z
}
\]

推荐 MRI residual branch 最后一层做 zero initialization：

\[
W_{last}=0,\qquad b_{last}=0.
\]

因此模型初始化时：

\[
q=q_c.
\]

这样 MRI 只能学习 clinical baseline 之外的增量修正。

## 7.3 作用

O1 不作为最终主 Outcome 模型，而作为：

\[
\boxed{
\text{single-state MRI baseline}
}
\]

用于回答：

> MRI 单点状态是否有增量 prognostic value？

---

# 8. OT：Observed Transition-aware Outcome

OT 是本轮主 Outcome Model。

## 8.1 样本定义

每个样本需要一个连续 transition：

\[
t\rightarrow t+1.
\]

输入 target survival label 必须对应：

\[
t+1.
\]

即：

\[
T=T_{t+1},
\qquad
\delta=\delta_{t+1}.
\]

Treatment history 也使用 target：

\[
h_{t+1}.
\]

## 8.2 MRI 表征

使用同一个 train-fold PCA：

\[
e_t=PCA_{64}(z_t),
\]

\[
e_{t+1}=PCA_{64}(z_{t+1}).
\]

显式构造 transition：

\[
\Delta e_t=e_{t+1}-e_t.
\]

## 8.3 Transition representation

定义：

\[
u^{obs}_t
=
[e_t,e_{t+1},\Delta e_t].
\]

维度：

\[
64+64+64=192.
\]

## 8.4 Outcome architecture

Clinical branch：

\[
q_c=G_c(c,m,h_{t+1}).
\]

Transition residual：

\[
r_{trans}
=
R_{trans}(u^{obs}_t).
\]

最终：

\[
\boxed{
q=q_c+r_{trans}
}
\]

输出与当前一致的 piecewise exponential hazards：

\[
\lambda_{1:4}.
\]

---

# 9. 为什么采用 CLARITY-inspired 而不是直接复制 Two-Way Attention

CLARITY 的核心 Outcome 思路是：

\[
\text{pre-state}
+
\text{post-state}
+
\text{state change}
\rightarrow
\text{survival}.
\]

本设计保留这一核心思想：

\[
[e_t,e_{t+1},e_{t+1}-e_t].
\]

但当前 Cloop 的 MRI representation 是单个：

\[
768D
\]

global latent vector，而不是适合 token-level cross-attention 的多 token MRI representation。

因此本轮采用：

\[
\boxed{
\text{low-dimensional pre/post/delta residual evaluator}
}
\]

以减少 Outcome architecture 自身成为 confounder。

---

# 10. OT 与 O1 必须做 matched comparison

OT 只能用于有连续 transition 的患者/landmark，因此 OT cohort 小于普通 O1 cohort。

为了公平评价 transition 的增量价值，必须构建：

```text
O1-matched
```

O1-matched 与 OT 使用完全相同：

```text
patient IDs
target landmarks
survival labels
clinical/history
```

区别仅在：

O1-matched：

\[
e_{t+1}
\]

OT：

\[
[e_t,e_{t+1},e_{t+1}-e_t].
\]

真正的核心 comparison：

\[
\boxed{
OT-O1_{matched}
}
\]

而不是全 cohort O1 vs OT。

---

# 11. OT 的训练原则

OT 使用：

\[
\boxed{
\text{真实 observed transition}
}
\]

训练。

即：

\[
(z_t,z_{t+1}^{obs}).
\]

OT 不与 Dynamics 联合训练。

训练完成后：

\[
\boxed{
G_\psi\text{ permanently frozen}
}
\]

后续所有 predicted-state experiments 共享这一套 Outcome 参数。

---

# 12. OP：Predicted Transition-aware Outcome

OP 使用已经训练好的 frozen OT Outcome Evaluator。

不重新训练 Outcome Model。

---

# 13. OP 的 World Model 输入

起点：

\[
z_t^{obs}.
\]

使用 factual：

\[
a_t^{obs}
\]

和 factual MRI interval：

\[
\Delta t_t^{obs}.
\]

得到：

\[
\hat z_{t+1}
=
F_\theta
(
z_t,
a_t,
c,
h_t,
\Delta t_t
).
\]

诊断阶段仍使用 recorded action，不使用 Planner 推荐 action。

因此：

\[
\boxed{
OP\text{ 是 factual conditional forecasting evaluation}
}
\]

不是 counterfactual treatment efficacy evaluation。

---

# 14. OP transition representation

使用 OT 同一个 PCA：

\[
e_t=PCA_{64}(z_t),
\]

\[
\hat e_{t+1}
=
PCA_{64}(\hat z_{t+1}).
\]

然后：

\[
\Delta\hat e_t
=
\hat e_{t+1}-e_t.
\]

输入 frozen evaluator：

\[
u^{pred}_t
=
[e_t,\hat e_{t+1},\Delta\hat e_t].
\]

最终：

\[
\boxed{
G_\psi(u^{pred}_t,c,m,h_{t+1})
}
\]

---

# 15. 四种 Dynamics 的公平比较

所有 OP variant 必须保持以下完全一致：

```text
same patients
same source MRI
same target landmark
same factual treatment
same Δt
same PCA
same Outcome Evaluator
same survival labels
same evaluation code
```

只改变：

\[
F_\theta.
\]

实验为：

```text
OP-Baseline
OP-RRT
OP-Ensemble
OP-RRT+Ensemble
```

这是本轮最核心的 World Model downstream comparison。

---

# 16. 主要实验表 1：Outcome input ablation

使用 observed MRI：

| Outcome input | C-index ↑ | Brier@365 ↓ | NLL ↓ |
|---|---:|---:|---:|
| O0 Clinical-only |  |  |  |
| O1 Single-state MRI |  |  |  |
| O1-matched |  |  |  |
| OT Observed transition |  |  |  |

最重要比较：

\[
O1-O0
\]

和：

\[
OT-O1_{matched}.
\]

---

# 17. 主要实验表 2：Dynamics downstream comparison

固定 frozen OT：

| Future-state source | C-index ↑ | Brier@365 ↓ | NLL ↓ |
|---|---:|---:|---:|
| Observed target MRI |  |  |  |
| Baseline predicted |  |  |  |
| RRT predicted |  |  |  |
| Ensemble predicted |  |  |  |
| RRT+Ensemble predicted |  |  |  |

其中 Observed target MRI 是真实 transition 下的 evaluator reference，而不是严格意义的 achievable model upper bound。

---

# 18. 关键 downstream gap 指标

定义：

\[
\Delta C_{gap}
=
C_{pred}-C_{obs},
\]

\[
\Delta BS_{gap}
=
BS_{pred}-BS_{obs},
\]

\[
\Delta NLL_{gap}
=
NLL_{pred}-NLL_{obs}.
\]

更接近 0 表示 predicted transition 更接近 observed transition 的 prognostic utility。

不同 Dynamics 比较时，重点看：

\[
\boxed{
\text{谁的 predicted-to-observed gap 最小}
}
\]

---

# 19. Primary hypotheses

## H1：MRI 有增量预后信息

\[
O1>O0.
\]

已有 E5 结果支持该方向，本轮作为复现/确认。

## H2：Transition 比单点 MRI 更有 prognostic value

\[
\boxed{
OT>O1_{matched}
}
\]

这是采用 transition-aware Outcome 的必要前提。

## H3：RRT 更好地保留 transition prognostic signal

核心比较：

\[
OP\text{-RRT}
\]

vs

\[
OP\text{-Baseline}.
\]

## H4：Ensemble 提供额外 robustness / reliability information

核心比较：

\[
OP\text{-RRT+Ensemble}
\]

vs

\[
OP\text{-RRT}
\]

以及 uncertainty–error association。

本设计不预设某一种方法必须获胜，只进行统一 evaluator 下的公平比较。

---

# 20. 数据协议

继续使用 development cohort：

```text
original train
+
original validation
```

原正式 test 不再参与 Outcome architecture selection。

---

# 21. Cross-validation

固定：

\[
5\text{-fold patient-level CV}
\]

如 event support 不足，可使用预注册 fallback：

\[
3\text{-fold}.
\]

要求：

- patient ID 不跨 fold；
- event/censoring 尽量 stratify；
- 所有 Outcome ablation 使用同一 folds；
- seeds 使用 `7, 17, 29`。

---

# 22. PCA 规则

每个 fold：

\[
PCA_{64}
\]

只能在 training-fold eligible MRI landmarks 上 fit。

必须记录：

```text
PCA mean
PCA components
explained variance
training patient hash
```

OT 和 OP 在同一个 fold 中必须共享完全相同的 PCA。

---

# 23. Outcome family 固定

继续使用：

\[
\boxed{
\text{Piecewise Exponential Survival}
}
\]

edges：

\[
[0,90,180,365,730].
\]

不在本轮更换 Cox / DeepSurv / discrete-time model。

理由：

> 本轮研究问题是 Dynamics 对 prognostic transition 的保真度，而不是 survival model family 的比较。

---

# 24. Training protocol

Outcome training 固定使用当前注册配置：

```text
optimizer
learning rate
weight decay
grad clip
max epochs
patience
checkpoint = validation survival NLL
```

不做大规模 hyperparameter search。

---

# 25. Clinical branch 是否加入 landmark day

Landmark day 不再作为本轮主研究问题。

推荐：

### Primary

保持：

\[
[c,m,h]
\]

与 E5 一致。

### Supplemental

只做一个小型 ablation：

\[
[c,m,h]
\]

vs

\[
[c,m,h,d_t].
\]

如果 day 在多数 folds 有稳定增益，再纳入最终 evaluator。

不要把 landmark-day 扩展成另一套完整主实验。

---

# 26. Transition-specific diagnostics

除 survival metrics 外，需要记录：

\[
\|e_{t+1}-e_t\|_2
\]

\[
\|\hat e_{t+1}-e_t\|_2
\]

\[
\|\hat e_{t+1}-e_{t+1}\|_2.
\]

---

# 27. Prognostic transition error

定义：

\[
E_{latent}
=
\|\hat e_{t+1}-e_{t+1}\|_2.
\]

再计算：

\[
\operatorname{Spearman}
(
E_{latent},
E_{outcome}
).
\]

其中：

\[
E_{outcome}
\]

可以使用 per-sample survival NLL 或 365-day prediction error。

目的：

> 判断 Dynamics latent error 是否真正对应 downstream prognostic degradation。

---

# 28. Ensemble reliability

对于 Ensemble 与 RRT+Ensemble：

计算 member-wise transition：

\[
\hat e_{t+1}^{(m)}.
\]

定义 disagreement：

\[
U_t
=
\frac1M
\sum_m
\|
\hat e_{t+1}^{(m)}
-
\bar e_{t+1}
\|^2.
\]

评估：

\[
\operatorname{Spearman}(U,E_{outcome}).
\]

以及 selective prediction：

```text
Risk@100
Risk@80
high/low uncertainty error ratio
```

这部分直接服务于：

\[
\boxed{
\text{Ensemble learning contribution}
}
\]

---

# 29. OP 与原 O2 的关系

原 O2：

\[
\hat z_{t+k}
\rightarrow
O1.
\]

新的 OP：

\[
[z_t,\hat z_{t+1},\hat z_{t+1}-z_t]
\rightarrow
OT.
\]

因此 OP 是：

\[
\boxed{
\text{transition-aware downstream evaluation}
}
\]

相比原 O2 更贴近 CLARITY-style Outcome interpretation，也更贴近 treatment planning。

---

# 30. Planner 接口修改条件

只有当：

\[
OT>O1_{matched}
\]

且 predicted transition evaluation 显示稳定 downstream value 时，才将 OT 接入 Planner。

如果 observed transition 本身没有增量价值，则 Planner 保留 E5 single-state evaluator，不增加复杂度。

---

# 31. Planner 最终形式

若 transition route 成立，则 candidate action：

\[
a_t^{(j)}
\]

经过 World Model：

\[
\hat z_{t+1}^{(j)}
=
F_\theta(z_t,a_t^{(j)},c,h_t,\Delta t).
\]

构造：

\[
e_t,
\qquad
\hat e_{t+1}^{(j)},
\qquad
\Delta\hat e_t^{(j)}
=
\hat e_{t+1}^{(j)}-e_t.
\]

Outcome cost：

\[
C^{(j)}
=
1-
S_\psi
\left(
365
\mid
e_t,
\hat e_{t+1}^{(j)},
\Delta\hat e_t^{(j)},
c,
h
\right).
\]

Planner 在 candidate branches 中进行搜索。

---

# 32. Outcome evaluator 中是否直接输入 action

Primary Outcome 不直接输入 candidate action。

即不使用：

\[
G(z_t,z_{t+1},a_t).
\]

Action 只通过：

\[
a_t
\rightarrow
\hat z_{t+1}
\]

进入 Outcome。

原因：

1. 避免 action double-count；
2. observational treatment assignment 存在 confounding；
3. 保持 Outcome evaluator 更接近 prognostic evaluator；
4. 更容易归因 Dynamics 的贡献。

---

# 33. Evidence boundary

本轮所有真实数据 Outcome 分析仍属于：

\[
\boxed{
\text{observational prognostic evaluation}
}
\]

不能解释为：

\[
\boxed{
\text{identified causal treatment effect}
}
\]

即使某 candidate action 在 Planner 中获得更低 predicted risk，也不能写成：

> 该治疗被证明可以提高患者 survival。

更准确的描述是：

> Under the learned observational world model and frozen prognostic evaluator, the candidate induces a predicted state trajectory associated with lower estimated risk.

---

# 34. 结果判定 Gate

## Gate A：Single-state MRI

如果：

\[
O1>O0
\]

方向在多数 folds 稳定，则保留 MRI residual baseline。

## Gate B：Observed transition

如果：

\[
OT>O1_{matched}
\]

在至少两个 primary metrics 上多数 folds 改善，则 transition-aware Outcome 成为主 evaluator。

否则停止 transition route。

## Gate C：Predicted transition

如果至少一种 learned dynamics 的 OP 优于 Baseline OP，说明 Dynamics quality 能传播到 downstream prognostic performance。

## Gate D：RRT contribution

主要比较：

\[
OP\text{-RRT}
\]

vs

\[
OP\text{-Baseline}.
\]

## Gate E：Ensemble contribution

比较：

\[
OP\text{-RRT+Ensemble}
\]

vs

\[
OP\text{-RRT}
\]

以及 reliability metrics。

---

# 35. 推荐实验执行顺序

```text
Stage 1
O0
O1

Stage 2
构建 matched transition cohort
O1-matched
OT-observed

Stage 3
Freeze OT evaluator

Stage 4
OP-Baseline
OP-RRT
OP-Ensemble
OP-RRT+Ensemble

Stage 5
Ensemble reliability

Stage 6
如果 Gate B–E 成立
将 frozen OT 接入 Planner
```

---

# 36. 推荐 Artifact Keys

建议：

```text
outcome_v2/O0/fold0/seed7
outcome_v2/O1/fold0/seed7
outcome_v2/O1_matched/fold0/seed7
outcome_v2/OT_observed/fold0/seed7

outcome_v2/OP_baseline/fold0/seed7
outcome_v2/OP_rrt/fold0/seed7
outcome_v2/OP_ensemble/fold0/seed7
outcome_v2/OP_rrt_ensemble/fold0/seed7
```

paired summaries：

```text
paired/O1_vs_O0
paired/OT_vs_O1_matched
paired/OP_rrt_vs_OP_baseline
paired/OP_rrt_ensemble_vs_OP_rrt
```

---

# 37. 推荐主图

## Figure A：Outcome input ablation

```text
Clinical
   ↓
Single-state MRI residual
   ↓
Observed transition
```

展示：

```text
C-index
Brier
NLL
```

## Figure B：Downstream World Model comparison

固定 frozen OT：

```text
Observed
Baseline
RRT
Ensemble
RRT+Ensemble
```

展示 predicted-to-observed gap。

## Figure C：Uncertainty vs downstream error

对于：

```text
Ensemble
RRT+Ensemble
```

展示：

\[
U
\]

vs

\[
Outcome\ error.
\]

---

# 38. 推荐论文叙事

Outcome 部分不作为主创新，而应描述为：

> We adopt a CLARITY-inspired transition-aware prognostic evaluator that summarizes the pre-treatment state, the future state, and their latent change. The evaluator is trained only on observed transitions and subsequently frozen for all world-model comparisons.

然后把核心贡献放回：

> We investigate whether recursive rollout training and ensemble learning improve the fidelity of future state transitions and preserve their downstream prognostic information under recursive planning.

因此：

\[
\boxed{
\text{Outcome = controlled evaluator}
}
\]

\[
\boxed{
\text{RRT + Ensemble = main methodological contribution}
}
\]

---

# 39. 最终研究主线

本轮实验最终形成如下证据链：

\[
\boxed{
\text{MRI adds prognostic information}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Observed state transitions are prognostically informative}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{RRT predicts those transitions more faithfully}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Ensemble provides robustness / reliability information}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{Frozen transition-aware evaluator supports receding-horizon planning}
}
\]

这条路线比继续围绕 Outcome architecture 本身做大量优化更符合 Cloop 的核心贡献定位。

# Cloop 当前实现：建议修改项清单

> 归档说明：这是该阶段的历史设计稿。目录结构和命令示例保留了原设计时的写法；当前可运行路径请参阅[ v1 README ](README.md)。


**基于仓库：** `https://github.com/yu3jun1/Cloop`  
**检查版本：** `ff5cc8f693cb7555abe717be337518f3df63f46b`  
**对照文档：** `clarity_loop_from_scratch_experiment_design.md` v1.1

本文只整理“值得修改的点”，按优先级分组，目标是在正式运行 `main_v1` 前收紧实验协议、数据 provenance、配置一致性和测试覆盖。

---

## P0：正式主实验前建议必须修改

### 1. 修正 `update_history()` 中 completed interval count 的语义

当前代码：

```python
count = history[..., 2 * action_dim : 2 * action_dim + 1] + 0.1
```

而设计文档描述的是“已完成区间数”。当前实现实际得到：

```text
0 → 0.1 → 0.2 → 0.3 ...
```

与计数语义不一致。

**建议：**

若本意是区间计数：

```python
count = history[..., 2 * action_dim : 2 * action_dim + 1] + 1.0
```

若本意是缩放后的计数，则必须显式定义：

```text
scaled_interval_count = completed_interval_count / 10
```

并把缩放规则写入配置和 `run.json`。

**原因：** 该 history 会同时进入 Dynamics、Outcome Head 和 Planner。一旦正式训练后再修改，会导致主实验需要整体重跑。

---

### 2. 增加 Frozen BrainIAC latent provenance 验证

当前 `build_main_cache()` 会直接记录：

```python
"encoder": {
    "name": config["data"]["encoder"],
    "frozen": True
}
```

这只是程序声明，并不能证明当前 latent 的真实来源。

**建议增加 latent provenance manifest，例如：**

```json
{
  "encoder": "brainiac",
  "frozen": true,
  "adapter": null,
  "checkpoint": "/home/.../BrainIAC.ckpt",
  "checkpoint_sha256": "...",
  "output_dim": 768,
  "extraction_protocol": "brainiac_mean_v1",
  "source_commit": "...",
  "num_timepoints": 596
}
```

`prepare` 时至少强制验证：

```text
encoder == brainiac
frozen == true
adapter == null
output_dim == 768
```

并将 provenance hash 写入：

```text
run.json
cache metadata
```

**原因：** 正式论文必须能追溯 representation 的来源。目录名和 config 不能作为 frozen encoder 的充分证据。

---

### 3. 补齐 `mri_day` 的时间质量审计

当前 `main_v1` 正确使用 `mri_day`，而不是 Timepoint 编号差。但旧 CLARITY timeline 构造中存在：

- interior interpolation
- leading `next_day - 100`
- trailing `previous_day + 100`

等 MRI 日期插补。

Cloop 当前仅保存：

```python
"quality_flags": {"time_quality": "unknown"}
```

没有逐 timepoint 标记 observed / imputed。

**建议：**

最好让每个 timeline node 记录：

```json
{
  "mri_day": 123,
  "mri_day_source": "observed"
}
```

或：

```json
{
  "mri_day": 123,
  "mri_day_source": "imputed_interior"
}
```

至少区分：

```text
observed
imputed_interior
imputed_leading
imputed_trailing
unknown
```

`prepare` 中报告：

```text
总 timepoint 数
observed time 数
各类 imputed time 数
涉及 imputation 的 transition 数
涉及 imputation 的 H2/H3 window 数
```

必要时增加：

```text
all valid timeline
vs
observed-time-only subset
```

敏感性分析。

**原因：**

`mri_day` 同时决定：

\[
\Delta t
\]

以及治疗事件落入哪个：

\[
(d_t,d_{t+1}]
\]

区间，所以它同时影响 time input 和 action input。

---

### 4. 明确 `legacy_stage1` 的定位

设计文档把 `legacy_stage1` 描述成旧 Stage 1 回归检查，但当前 Cloop 使用的是新的 `OneStepDynamics`，并未严格复刻旧 MyMeWM architecture。

当前 Cloop 大致是：

```text
action Linear + SiLU
time Linear + SiLU
initial_hidden(z)
GRUCell
2-layer residual MLP
```

旧 Stage 1 还有不同的 LayerNorm、time MLP、GRU 用法、residual MLP 和初始化方式。

因此当前：

```bash
--protocol legacy_stage1
```

不能用于严格 numeric regression。

**建议二选一：**

方案 A（推荐）：

```python
LegacyOneStepDynamics
```

严格复制旧 Stage 1 architecture，仅用于回归验证。

方案 B：

把名称改为：

```text
legacy_data_protocol
```

明确它只复用旧 trajectory/action vocabulary，而不声称复现旧模型数值。

---

## P1：正式主实验前强烈建议修改

### 5. 清理“配置允许但实现忽略”的字段

当前存在若干配置项，看起来可切换，但代码行为实际上固定。例如：

```yaml
data:
  action_alignment: timestamps
```

当前 `main_v1` 实际始终走 timestamp alignment。

类似字段还包括：

```text
unknown_interval_policy
planner.interval_source
planner.require_supported_actions
planner.abstain_when_no_valid_action
```

**建议采用：**

\[
oxed{	ext{要么实现，要么拒绝}}
\]

例如：

```python
if protocol == "main_v1" and action_alignment != "timestamps":
    raise ConfigError(...)
```

```python
if planner["interval_source"] != "train_median":
    raise ConfigError(...)
```

这样 `run.json` 中 resolved config 才与实际行为完全一致。

---

### 6. 增加正式的 D3 → D4 paired improvement

当前 Dynamics 汇总主要围绕 Baseline 做 RI，但 Stage 1 的关键对照之一是：

\[
	ext{Ensemble}

ightarrow
	ext{RRT+Ensemble}
\]

它回答：

> 在相同 ensemble 条件下，RRT 是否仍有增量收益？

**建议正式输出：**

```json
{
  "baseline_vs_rrt": {},
  "ensemble_vs_rrt_ensemble": {},
  "baseline_vs_rrt_ensemble": {}
}
```

论文主分析优先解释：

```text
D1 → D2
D3 → D4
```

---

### 7. 明确 cosine 指标是 similarity 还是 distance

当前代码使用：

```python
torch.nn.functional.cosine_similarity(...)
```

这是：

\[
\cos(\hat z,z)
\]

越大越好。

但设计文档只写：

```text
Cos@1 / Cos@2 / Cos@3
```

容易和 cosine distance 混淆。

**建议：**

直接命名为：

```text
CosSim@1
CosSim@2
CosSim@3
```

若以后需要 distance，则另定义：

\[
CosDist=1-CosSim
\]

不要继续使用含义不明确的 `Cos@k`。

---

### 8. `structural_rule_violation_rate` 未启用规则时不能固定为 0

当前 replay summary 中：

```python
"structural_rule_violation_rate": 0.0
```

但配置为：

```yaml
clinical_rules_path: null
toxicity_model: disabled
```

此时真实含义是“未评价”，而不是“零违规”。

**建议改成：**

```json
{
  "structural_rule_violation_rate": null,
  "structural_rule_violation_reason": "clinical_rules_disabled"
}
```

---

### 9. 增强 resume 测试

当前主要验证 RNG state 能保存与恢复，但设计文档要求完整 epoch 边界 resume。

**建议增加集成测试：**

```text
Run A:
epoch 1 → 4 连续训练

Run B:
epoch 1 → 2
保存 last.pt
重新启动
resume epoch 3 → 4
```

比较：

```text
final model state
optimizer state
best epoch
validation history
```

在 deterministic CPU 环境下应一致或在严格浮点容差内一致。

---

### 10. 增加真正 H=3 MPC 的 replanning 测试

当前已有“新状态可改变 greedy action”的测试，但最终论文主线是：

```text
H=3 MPC + re-observation + replanning
```

**建议构造测试：**

```text
s0 下 MPC 计划 [A, B, C]
只执行 A
得到新观测 s1
重新 MPC
新的第一动作变成 D
```

并验证：

```text
D != 原 fixed plan 中的 B
```

这样可以直接证明：

\[
	ext{receding-horizon replanning}

eq
	ext{固定计划执行}
\]

---

## P2：接入真实 LLM Policy 前修改

### 11. 增加统一 `PolicyAgent` factory

目前接口层已经有：

```text
PolicyAgent
LLMProvider
CatalogPolicy
LLMPolicy
FakeProvider
```

但正式 pipeline 中 `_planner_for_method()` 当前仍限制 `policy.kind == catalog`。

**建议增加：**

```python
def build_policy(config, action_codec) -> PolicyAgent:
    ...
```

以后可统一支持：

```text
catalog
fake
openai
anthropic
local
http_compatible
```

Planner、World Model、Outcome 不应知道底层 LLM 类型。

---

### 12. 修正 imagined node 的 `PolicyRequest`

当前 MPC 深度 >1 时，新的候选请求仍使用根 observed state 的：

```text
observation_day
state_source
```

CatalogPolicy 不依赖这些字段，所以当前离线实验不受影响；但未来 LLMPolicy 会受影响。

**建议 imagined node 使用：**

```python
PolicyRequest(
    observation_day=root_day + node.elapsed_days,
    state_source="imagined",
    executed_history_summary=...,
    ...
)
```

未来如果加入经过验证的 imagined-state summary，也应在这里传递。

不要直接把 768-D latent 提供给 LLM 并声称其理解 MRI。

---

### 13. 真正 enforce LLM 调用预算

配置已有：

```yaml
max_calls_per_decision: 2
```

未来接入真实 provider 时需要真正限制：

```text
每次 decision 最大 API calls
每层 beam 的调用预算
timeout
retry
fallback
```

否则 H=3 beam search 会迅速放大 LLM 调用数。

---

## P2：数据与 action 表示建议收紧

### 14. 明确第一版 action granularity

当前 `ActionCodec` 主要编码：

```text
category
agent
```

剂量、fractions、cycle length 等不进入 dynamics action vector。

这与当前 MVP 的“粗粒度治疗集合”是一致的，不需要马上扩展。

**建议仅明确记录：**

```text
action_granularity = category_agent_set
```

并在报告中明确：

> 当前不评价 dose optimization。

以后扩展剂量级 action space 时作为独立 protocol v2。

---

### 15. 正式统计 unknown action interval 的覆盖损失

当前：

```text
action_known=False
```

会正确阻止 RRT window 跨越未知治疗区间。

建议正式 report 增加：

```text
known transitions
unknown/excluded transitions
因 unknown action 丢失的 H1/H2/H3 windows
按 train/validation/test 分开
```

否则主实验实际使用了多少 longitudinal data 不够透明。

---

### 16. 在 Planning 前先报告 Catalog Coverage

Planner 只能从 train catalog 里选择治疗。

因此正式 P0–P5 前应先报告：

\[
CatalogCoverage=P(a_{actual}\in A_{train})
\]

以及 Top-N candidate recall。

如果 catalog coverage 很低，那么 action-match F1 的上限本身就低，不能把问题全部归因于 planner。

---

## P3：当前阶段不建议扩大的内容

### 17. 暂时不要重新加入 MRI-CORE LoRA

当前主链路已经包含：

```text
RRT
+ Ensemble
+ Outcome
+ MPC
+ Replanning
```

MRI-CORE LoRA 保留为 representation analysis 即可。

---

### 18. 暂时不要让 Outcome loss 联合训练 Dynamics

当前：

```text
World Model 单独训练
Outcome Head 单独训练
Outcome 冻结后评价 imagined states
```

对于第一版更利于归因。

这样才能回答：

> RRT 改善 rollout 后，是否自然改善 downstream outcome prediction？

Joint training 可以留作后续扩展。

---

### 19. 暂时不要开放 LLM 自由生成新治疗组合

第一版让 LLM 只选择 catalog IDs 是合理的。

应先验证：

\[
	ext{World Model}
+
	ext{Reliability}
+
	ext{Replanning}
\]

本身是否成立，再扩展 action space。

---

### 20. Synthetic 继续只作为机制验证

Synthetic 当前的定位是正确的：

\[
	ext{action}

ightarrow
	ext{独立环境状态变化}

ightarrow
	ext{新观测}

ightarrow
	ext{replanning}
\]

它用于验证闭环机制，不作为临床疗效证据。

---

# 推荐修改顺序

## Step 1：先修小型 protocol 问题

优先修改：

```text
history count
dead config
cosine naming
structural violation null
```

这些修改成本低，但能消除协议歧义。

---

## Step 2：修数据 provenance

完成：

```text
Frozen BrainIAC manifest
checkpoint / extraction provenance
MRI-day time-quality audit
unknown-action coverage audit
```

然后重新执行：

```bash
cloop doctor
cloop prepare
```

重点检查 `run.json` 是否已经能够完整描述正式实验数据协议。

---

## Step 3：处理 legacy regression

若需要核对旧实验：

```text
实现 LegacyOneStepDynamics
```

只需先跑：

```text
legacy Baseline seed17
legacy RRT seed17
```

确认方向和数量级即可。

若不需要 numeric regression，则把 `legacy_stage1` 改名为仅表示数据协议。

---

## Step 4：补关键测试

建议增加：

```text
true resume test
H3 MPC replanning test
dead-config rejection test
provenance mismatch rejection test
time-quality audit test
```

然后运行：

```bash
pytest -q
cloop smoke --device cpu
```

---

## Step 5：再开始正式 `main_v1`

建议顺序：

```text
doctor
prepare
↓
Dynamics D0–D4
↓
validation dynamics/reliability
↓
Outcome O0/O1/O2
↓
validation planning P0–P5
↓
freeze-protocol
↓
test evaluation
```

---

# 最终优先级汇总

## 正式 `main_v1` 前必须解决

```text
P0-1  history count 语义
P0-2  BrainIAC latent provenance
P0-3  MRI-day / time-quality audit
P0-4  legacy_stage1 定位
```

## 正式 `main_v1` 前强烈建议解决

```text
P1-1  dead config
P1-2  D3→D4 paired RI
P1-3  cosine naming
P1-4  structural violation null
P1-5  true resume test
P1-6  H3 MPC replanning test
```

## LLM 接入前解决

```text
P2-1  Policy factory
P2-2  imagined PolicyRequest
P2-3  LLM call budget
```

## 当前不要扩展

```text
MRI-CORE LoRA
joint outcome-dynamics training
自由治疗生成
K=5 主实验
更多 encoder
复杂 toxicity model
```

---

# 修改完成后的理想状态

正式实验开始前，应达到：

\[
oxed{
	ext{数据来源可追溯}
+
	ext{配置与真实执行完全一致}
+
	ext{RRT/Ensemble 可回归验证}
+
	ext{闭环 replanning 可独立测试}
}
\]

当前阶段最重要的不是继续增加功能，而是保证已有每个模块的语义、数据来源和评价协议都能被准确解释。

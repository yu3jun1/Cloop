# Cloop：正式 `main_v1` 前四个必须修改项的实施建议

> 归档说明：这是该阶段的历史设计稿。目录结构和命令示例保留了原设计时的写法；当前可运行路径请参阅[ v1 README ](README.md)。


**目标仓库：** `https://github.com/yu3jun1/Cloop`  
**基于检查版本：** `df4dcb852b8f81d8d4768cd85bcdfafbc3cad627`  
**目标：** 不再扩展新模块，只收紧数据时间来源、legacy numeric regression、协议冻结和闭环 replanning 指标四个关键问题。

---

# 0. 修改范围总览

建议只修改以下文件，不扩大仓库结构：

```text
src/cloop/
├── config.py
├── data.py
├── engine.py
├── metrics.py
└── planner.py

tests/
├── test_data.py
├── test_pipeline.py
├── test_policy_planner.py
└── test_metrics.py

README.md
clarity_loop_from_scratch_experiment_design.md
```

如果要重新生成 clinical timeline，还需要修改上游数据准备脚本。建议不要把上游原始 Excel 解析逻辑搬入 Cloop；Cloop 继续只消费已经整理好的 timeline。

四个必须项：

```text
M1. 真正恢复逐 timepoint 的 MRI 日期 provenance
M2. 完整对齐 legacy Stage 1 的随机种子和 normalizer
M3. freeze-protocol 后锁死模型并验证 models.pt hash
M4. 修正 replanning metric，使其真正测量计划修订
```

---

# 1. M1：真正生成逐 timepoint 的 `mri_day_source`

## 1.1 当前问题

Cloop 当前已经支持：

```text
observed
imputed_interior
imputed_leading
imputed_trailing
unknown
```

并在 `build_main_cache()` 中统计这些来源。

问题是当前 CLARITY 的 clinical timeline 生成逻辑虽然会进行 MRI 日期插补，但原始 timeline node 通常只保存：

```json
{
  "tp_id": "TP3",
  "mri_day": 212,
  "actions": {}
}
```

并没有保存：

```json
"mri_day_source": "..."
```

因此 Cloop 的审计框架可能会把绝大多数实际数据标成 `unknown`。

---

## 1.2 目标

每个 MRI timepoint 必须显式拥有：

```json
{
  "mri_day": 212,
  "mri_day_source": "observed"
}
```

或：

```json
{
  "mri_day": 212,
  "mri_day_source": "imputed_interior"
}
```

不得因为 `mri_day` 是一个有效数字就默认它是 observed。

---

## 1.3 推荐修改位置

优先修改产生 `clinical_latest.json` 的上游 timeline 构造代码。

不要在 Cloop 内根据日期数值反推 provenance，因为数值本身无法区分真实记录日期和插补日期。

---

## 1.4 上游生成逻辑建议

同步维护：

```python
day_sources = {}
```

示例：

```python
days = {tp: tp_to_day.get(tp, None) for tp in tp_candidates}
day_sources = {}

for idx, tp in enumerate(tp_candidates):
    if days[tp] is not None:
        days[tp] = max(days[tp], 0)
        day_sources[tp] = "observed"
        continue

    prev_tp = ...
    next_tp = ...

    if prev_tp is not None and next_tp is not None:
        pv, nv = days[prev_tp], days[next_tp]
        days[tp] = max((pv + nv) // 2, 0)
        day_sources[tp] = "imputed_interior"

    elif prev_tp is None and next_tp is not None:
        days[tp] = max(days[next_tp] - 100, 0)
        day_sources[tp] = "imputed_leading"

    elif prev_tp is not None and next_tp is None:
        days[tp] = max(days[prev_tp] + 100, 0)
        day_sources[tp] = "imputed_trailing"

    else:
        days[tp] = 0
        day_sources[tp] = "imputed_leading"
```

生成 timeline：

```python
timeline.append({
    "tp_id": f"TP{tpn}",
    "mri_day": int(day),
    "mri_day_source": day_sources[tpn],
    "state": {"days_since_dx": int(day)},
    "actions": {},
})
```

---

## 1.5 Cloop 侧增加正式协议开关

在 `configs/default.yaml`：

```yaml
data:
  require_mri_day_provenance: true
```

在 `config.py`：

```python
if data["require_mri_day_provenance"] is not True:
    raise ConfigError(
        "main_v1 requires data.require_mri_day_provenance=true"
    )
```

在 `build_main_cache()` 完成 time audit 后：

```python
unknown_count = time_quality_report["timepoint_sources"]["unknown"]

if (
    config["data"]["protocol"] == "main_v1"
    and config["data"]["require_mri_day_provenance"]
    and unknown_count > 0
):
    raise DataError(
        f"main_v1 requires explicit mri_day_source for every usable timepoint; "
        f"found {unknown_count} unknown timepoints"
    )
```

第一版正式协议建议直接要求：

```text
unknown == 0
```

---

## 1.6 建议审计字段

`run.json.audit.time_quality` 至少保存：

```json
{
  "total_timepoints": 0,
  "timepoint_sources": {
    "observed": 0,
    "imputed_interior": 0,
    "imputed_leading": 0,
    "imputed_trailing": 0,
    "unknown": 0
  },
  "total_transitions": 0,
  "transitions_involving_imputation": 0,
  "windows": {
    "H2": {
      "eligible": 0,
      "involving_imputation": 0
    },
    "H3": {
      "eligible": 0,
      "involving_imputation": 0
    }
  }
}
```

正式数值必须由实际数据产生。

---

## 1.7 测试

新增：

```text
test_main_v1_requires_explicit_mri_day_source
test_unknown_mri_day_source_is_rejected
test_imputed_window_counts_match_manual_example
```

---

## 1.8 完成标准

```text
[ ] 上游 timeline 每个可用 TP 都写 mri_day_source
[ ] Cloop main_v1 中 unknown timepoint = 0
[ ] doctor/prepare 能审计来源数量
[ ] 相关统计进入 run.json
[ ] 单元测试覆盖 unknown 拒绝行为
```

---

# 2. M2：完整对齐 legacy Stage 1 的 seed 与 normalizer

## 2.1 当前问题

`LegacyOneStepDynamics` 已经基本复刻旧 MyMeWM Stage 1 architecture，但当前训练 seed 规则仍有差异。

当前 Cloop：

```python
member_seed = training_seed * 1000 + member_index
```

即：

```text
single model seed=17 → actual seed=17000
```

旧 Stage 1：

```python
member_seed = seed if members == 1 else seed * 1000 + member_index
```

即：

```text
Baseline seed17 → actual seed17
RRT seed17      → actual seed17
Ensemble member0 → 17000
Ensemble member1 → 17001
...
```

---

## 2.2 修改 seed 规则

推荐抽成：

```python
def resolve_member_seed(
    training_seed: int,
    member_count: int,
    member_index: int,
) -> int:
    return (
        training_seed
        if member_count == 1
        else training_seed * 1000 + member_index
    )
```

在 `_train_world_member()`：

```python
recursive, member_count = variant_spec(
    variant,
    int(config["world"]["ensemble_size"]),
)

member_seed = resolve_member_seed(
    int(training_seed),
    member_count,
    int(member_index),
)
```

---

## 2.3 保存实际 member seed

在 `models.pt` 和 `metrics.json` 中显式保存：

```json
{
  "members": [
    {
      "member": 0,
      "member_seed": 17,
      "best_epoch": 21
    }
  ]
}
```

ensemble 则保存 17000、17001 等。

---

## 2.4 Legacy normalizer 完全对齐

旧 Stage 1 的 normalizer 使用：

```python
values = np.concatenate(...).astype(np.float64)
mean = values.mean(axis=0)
std = values.std(axis=0)
std = np.where(std < min_std, 1.0, std)

return mean.astype(np.float32), std.astype(np.float32)
```

当前 Cloop 是 PyTorch float32 statistics。

建议增加：

```python
@dataclass(frozen=True)
class LegacyNormalizer:
    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, values: Tensor, min_std: float = 1e-6):
        array = values.detach().cpu().numpy().astype(np.float64)

        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std = np.where(std < min_std, 1.0, std)

        return cls(
            mean=torch.from_numpy(mean.astype(np.float32)),
            std=torch.from_numpy(std.astype(np.float32)),
        )
```

`fit_preprocessing()`：

```python
if cache["protocol"] == "legacy_stage1":
    latent_norm = LegacyNormalizer.fit(...)
else:
    latent_norm = Normalizer.fit(...)
```

---

## 2.5 Regression 验收不要要求 bitwise 完全一致

必须一致：

```text
split IDs
window counts
action_dim
latent_dim
training horizon counts
member seeds
model parameter names
```

指标只要求接近旧结果，不要把旧历史 MSE 写成 pytest 硬断言。

---

## 2.6 测试

```python
assert resolve_member_seed(17, 1, 0) == 17
assert resolve_member_seed(17, 5, 0) == 17000
assert resolve_member_seed(17, 5, 4) == 17004
```

并增加：

```text
legacy normalizer vs NumPy reference
```

测试。

---

## 2.7 完成标准

```text
[ ] legacy single model seed17 真正使用 seed17
[ ] ensemble seed 规则与旧 Stage 1 一致
[ ] member_seed 写入 metrics/models
[ ] legacy normalizer 用 float64 statistics → float32
[ ] regression 测试通过
```

---

# 3. M3：`freeze-protocol` 后锁死模型并校验 `models.pt` hash

## 3.1 当前问题

当前 `freeze_protocol()` 会设置：

```text
protocol_frozen = true
```

但训练入口主要检查 `test_revealed`，理论上仍可能：

```text
freeze
→ retrain / force-task
→ test
```

这会破坏 validation freeze 的意义。

---

## 3.2 新增统一 guard

在 `engine.py` 增加：

```python
def _assert_run_mutable(manifest: dict[str, Any]) -> None:
    if manifest.get("stage_states", {}).get("protocol_frozen"):
        raise EngineError(
            "protocol is frozen; model/preprocessing changes require a new run"
        )
    if manifest.get("test_revealed"):
        raise EngineError(
            "test has already been revealed; changes require a new run"
        )
```

调用于：

```text
train_dynamics
train_outcome
任何 force-task model replacement
任何 future encoder adaptation
```

已 frozen 的 run 也不应重新拟合 preprocessing。

---

## 3.3 Freeze 时保存 `models.pt` SHA-256

```python
models_path = artifacts.path("models.pt")

if not models_path.exists():
    raise EngineError("models.pt is required before protocol freeze")

models_sha256 = sha256_file(models_path)
manifest["frozen_models_sha256"] = models_sha256
```

冻结签名建议包含：

```python
payload = {
    "config_signature": manifest["config_signature"],
    "data_signature": manifest["data_signature"],
    "uncertainty_scales": manifest.get("uncertainty_scales"),
    "models_sha256": models_sha256,
}
```

---

## 3.4 Test 前验证 frozen integrity

增加：

```python
def _assert_frozen_integrity(
    manifest: dict[str, Any],
    artifacts: RunArtifacts,
) -> None:
    if not manifest["stage_states"].get("protocol_frozen"):
        raise EngineError("protocol is not frozen")

    expected = manifest.get("frozen_models_sha256")
    if not expected:
        raise EngineError("frozen model hash is absent")

    current = sha256_file(artifacts.path("models.pt"))

    if current != expected:
        raise EngineError(
            "models.pt changed after protocol freeze; use a new run"
        )
```

在：

```text
evaluate_dynamics(test)
evaluate_outcome(test)
evaluate_replay(test)
```

统一调用。

---

## 3.5 Freeze 前要求 `last.pt` 不存在

```python
if artifacts.path("last.pt").exists():
    raise EngineError(
        "cannot freeze while last.pt exists; finish or clear interrupted training"
    )
```

不要自动删除 `last.pt`。

---

## 3.6 允许的操作

Freeze 后允许：

```text
report
read metrics
重复只读 evaluation
```

禁止：

```text
train
force-task
重新拟合 preprocessing
重新选择 uncertainty scale
```

---

## 3.7 测试

新增：

```text
test_freeze_blocks_dynamics_training
test_freeze_blocks_force_task
test_modified_models_pt_breaks_test_evaluation
test_report_still_works_after_freeze
```

---

## 3.8 完成标准

```text
[ ] freeze 后所有 model-changing 操作失败
[ ] freeze 保存 models.pt hash
[ ] test 前验证 models.pt hash
[ ] last.pt 存在时不能 freeze
[ ] validation scale frozen 后不再变化
```

---

# 4. M4：修正 `replanning_action_change_rate`

## 4.1 当前问题

当前 metric 实际比较：

\[
a_t^{recommended}
\]

与：

\[
a_{t-1}^{recommended}
\]

它回答的是连续两个 visit 当前推荐是否不同，而不是：

> 新观测是否修改了之前对当前阶段的未来计划？

---

## 4.2 正确定义

在时间 \(t\)：

```text
MPC imagined plan:
[a_t, a_{t+1}^{old}, a_{t+2}^{old}]
```

执行第一步后得到：

```text
s_{t+1}^{obs}
```

重新规划：

```text
[a_{t+1}^{new}, a_{t+2}^{new}, ...]
```

真正应该比较：

\[
a_{t+1}^{new}
\]

和：

\[
a_{t+1}^{old}
\]

即：

\[
PlanRevision_t =
1[a_{t+1}^{new} != a_{t+1}^{old}]
\]

---

## 4.3 保留两个指标

### A. `sequential_action_change_rate`

把当前 metric 改名为：

```text
sequential_action_change_rate
```

它描述相邻 visit 的当前推荐是否改变。

### B. `plan_revision_rate`

新增：

```text
plan_revision_rate
```

定义为：

```text
新观测后的当前推荐
vs
上一轮 imagined plan 中原本为这个阶段规划的动作
```

---

## 4.4 Replay 记录需要增加字段

每条 decision 建议增加：

```json
{
  "recommended_action": "A_x",
  "previously_planned_action_for_this_stage": "A_y",
  "plan_revision_evaluable": true,
  "plan_revised": true
}
```

没有 previous future action 时：

```json
{
  "previously_planned_action_for_this_stage": null,
  "plan_revision_evaluable": false,
  "plan_revised": null
}
```

---

## 4.5 `evaluate_replay()` 的实现建议

每个患者维护：

```python
previous_future_action: str | None = None
```

每个 visit：

```python
decision = planner.plan(state)
current_action = decision.recommended_action

planned_from_previous = previous_future_action
```

如果可比较：

```python
plan_revised = (
    current_action.action_id != planned_from_previous
)
```

然后保存当前 imagined plan 的第二个 action：

```python
if decision.status == "recommend" and len(decision.imagined_plan) >= 2:
    previous_future_action = decision.imagined_plan[1].action_id
else:
    previous_future_action = None
```

---

## 4.6 `_replay_summary()` 修改

新增：

```python
revision_rows = [
    row for row in rows
    if row.get("plan_revision_evaluable")
]

plan_revision_rate = (
    np.mean([row["plan_revised"] for row in revision_rows])
    if revision_rows
    else None
)
```

同时输出：

```text
plan_revision_evaluable_count
plan_revision_evaluable_fraction
```

---

## 4.7 不同 planning method 的处理

### `fixed_plan`

不做重新规划，所以：

```json
{
  "plan_revision_rate": null,
  "plan_revision_reason": "fixed_plan_does_not_replan"
}
```

不能写 0。

### `greedy`

H=1，没有 previous future action：

```json
{
  "plan_revision_rate": null,
  "plan_revision_reason": "no_prior_future_action_in_h1_policy"
}
```

### MPC 方法

以下适合报告：

```text
mpc_ensemble
mpc_rrt_ensemble
mpc_rrt_ensemble_unc
```

---

## 4.8 不要把 Plan Revision Rate 解释成越高越好

它是行为指标，不是疗效指标。

它用于证明：

\[
	ext{系统确实在利用新观测修订未来计划}
\]

不是证明：

\[
	ext{修订越多越好}
\]

真正的性能仍需结合 synthetic environment 的：

```text
environment cost
optimality gap
perturbation recovery
```

以及真实数据上的描述性评价。

---

## 4.9 测试

### Test A：不应误报 revision

旧计划：

```text
[A, B, C]
```

新 plan：

```text
[B, D, ...]
```

应：

```text
plan_revised = False
```

### Test B：真正 revision

旧计划：

```text
[A, B, C]
```

新 plan：

```text
[D, ...]
```

应：

```text
plan_revised = True
```

### Test C：greedy

```text
plan_revision_evaluable = False
```

### Test D：fixed plan

```text
plan_revision_rate = null
```

---

## 4.10 完成标准

```text
[ ] 旧 metric 改名 sequential_action_change_rate
[ ] 新增 plan_revision_rate
[ ] replay record 保存 previously_planned_action
[ ] P3/P4/P5 正确计算 revision
[ ] P1/P2 不伪造 revision rate
[ ] 单元测试覆盖 false-positive / false-negative
```

---

# 5. 推荐实施顺序

## Step 1：M2 Legacy seed + normalizer

完成后：

```bash
pytest tests/test_world.py tests/test_data.py -q
```

## Step 2：M3 Freeze integrity

完成后：

```bash
pytest tests/test_pipeline.py -q
```

## Step 3：M4 Plan Revision Metric

完成后：

```bash
pytest tests/test_policy_planner.py tests/test_metrics.py -q
```

## Step 4：M1 MRI time provenance

完成 timeline 重建后：

```bash
cloop doctor
cloop prepare
```

人工检查：

```text
run.json.audit.time_quality
```

必须确认：

```text
unknown == 0
```

---

# 6. 正式验收命令

```bash
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python

"$PY" -m pytest -q

"$PY" -m cloop smoke --device cpu

"$PY" -m cloop   --config configs/default.yaml   --paths configs/server.yaml   --run brainiac_v1 doctor

"$PY" -m cloop   --config configs/default.yaml   --paths configs/server.yaml   --run brainiac_v1 prepare
```

检查：

```text
latent provenance verified
mri_day unknown = 0
split 固定
action catalog 仅 train
normalizer 仅 train
```

然后至少做一次 legacy：

```text
Baseline seed17
RRT seed17
```

确认：

```text
window counts
seed protocol
Long MSE 数量级
RRT 改善方向
```

合理后，再正式执行：

```text
D0–D4
→ validation
→ O0/O1/O2
→ P0–P5 validation
→ freeze-protocol
→ test
```

---

# 7. 四项修改完成后的目标状态

完成 M1–M4 后，第一版研究框架应具备：

\[
oxed{
	ext{每个时间点的时间来源可追溯}
}
\]

\[
oxed{
	ext{legacy Stage 1 的 architecture / seed / normalization 可回归验证}
}
\]

\[
oxed{
	ext{validation freeze 后模型内容不可再改变}
}
\]

\[
oxed{
	ext{plan revision 指标真正测量新观测导致的未来计划修订}
}
\]

做到这四点后，继续增加新功能的优先级应明显低于直接开始正式 `main_v1` 实验。

# Cloop：相同 CLARITY Outcome 架构、独立训练的下游验证实验

**文档版本：1.0｜定位：替代此前不断扩展的 v4/v4_1/v4.2 修改路线**

本方案以已核查的 Cloop 提交 `f24ed763daf9e078180f86bb3ac2d784b0bdc2ad` 和 CLARITY 提交 `dadb82241a24f5ec5e4e4dc994e3116fd4a9da04` 为依据。本文是待实施的代码设计，不代表这些新增入口已存在，也不代表已经重新训练或验证了真实数据结果。源码依据列于文末。

> **本轮只回答一个主问题：在使用相同 CLARITY outcome 架构、但为各自 dynamics 分别训练权重时，RRT、Ensemble 以及组合方法，能否改善下游生存预测？**
>
> 不要求四组共享冻结的 outcome checkpoint。不增加新的 trajectory Transformer、value head、variance-to-Brier loss 或 planner。Uncertainty 是第二张独立实验表，不用于替代主表的全样本预后评价。

---

## 1. 最终要实施的实验

### 1.1 四组主实验

| 实验名 | 上游 dynamics | Outcome 架构 | Outcome 权重 | 主输入聚合 |
|---|---|---|---|---|
| `baseline` | 现有 one-step 单模型 | 同一个 CLARITY `SurvivalModule` 定义 | 本组独立训练 | 单模型预测 |
| `rrt` | 现有 RRT 单模型 | 同上 | 本组独立训练 | 单模型预测 |
| `ensemble` | 现有 one-step ensemble | 同上 | 本组独立训练 | 先平均 latent |
| `rrt_ensemble` | 现有 RRT ensemble | 同上 | 本组独立训练 | 先平均 latent |

对于方法 \(v\) 和预测跨度 \(H\)：

\[
\hat Z^{v}=F_v(z_s,a_{s:t-1},\Delta t_{s:t-1},c,h_s),\qquad t=s+H,
\]

\[
(r^v,\ell^v)=G_{\phi_{v,H}}(z_s,\bar z_t^v,c_{s:t}),\quad
p^v_{365}=\sigma(\ell^v).
\]

- `G` 的计算结构及核心超参数对四组相同。
- `phi[v,H]` 分别训练；每个 fold/seed/H 下从相同初始化开始，不共享训练后的权重。
- dynamics 完成训练后冻结；本轮仅训练 outcome。**冻结的是 dynamics，不是四组共用同一 outcome。**
- 一组 ensemble 只训练一个 outcome，不为每个成员另训练一个 head。
- `H` 是 MRI transition 数量；`365` 是从目标 landmark 起算的预后时间，两者不是同一个 horizon。

### 1.2 预先确定主比较

必须报告：

1. `rrt_ensemble - baseline`：组合是否具有下游收益。
2. `rrt - baseline`：RRT 在单模型上的收益。
3. `ensemble - baseline`：集成收益。
4. `rrt_ensemble - ensemble`：RRT 是否有超出集成的额外价值。

建议预先指定 **H2 的 Cox-risk C-index 为主指标，H2 的 IPCW Brier@365 为关键伴随指标**；H1 为短期对照，H3 为长跨度探索。该选择是本方案的建议，不是 CLARITY 原文规定。应在运行新实验前锁定，不能看到结果后改主指标。若更关注概率预测，也可预先交换两者的主次，但所有方法必须遵循同一规则。

### 1.3 本轮允许支持的结论

如在相同样本上取得较一致的改善，可以表述：

> 在保持 outcome 架构不变、允许其分别适配上游预测的条件下，改进后的 dynamics 提高了完整系统的事实条件预后预测能力。

不能自动推出：所有收益均由 MSE 下降单独造成、RRT 每个组件均必不可少、推荐方案具有因果治疗收益，或已经超过完整官方 CLARITY 系统。

---

## 2. 仓库依据与复用范围

以下“当前实现”来自源码；“本轮处理”是新设计，两者明确区分。

| 当前文件/实现 | 已核查内容 | 本轮处理 |
|---|---|---|
| `src/cloop/v3/next_stage.py` | `_source()`、`_fold_bundle()`、`_world()`、`_refs()`；按 H 选每患者第一个合格目标；复用 v2 world | 复用数据与多步 rollout 流程，不复用其 outcome 结构消融 [S1] |
| `src/cloop/v2/outcome_v2.py` | `_train_world()`、`_world_from_entry()`；四种 dynamics；训练与 checkpoint 结构 | 读取旧权重用于开发实验；确认实验可直接调用训练函数，无需重写 RRT [S2] |
| `src/cloop/v1/data.py` | `DynamicsDataset`、`collate_dynamics`、`DataBundle` 及 latent/action/history 数据 | 保留预处理与字段语义；不重做 MRI encoder |
| `src/cloop/v1/metrics.py` | `harrell_c_index`、`ipcw_brier`、`spearman`、`reliability_summary` | 复用单项指标，避免把旧 summary 的风险定义错接到新 head [S3] |
| `src/cloop/v3/trajectory_metrics.py` | `time_dependent_auc` 等 | 复用一年 time-dependent AUC；不沿用整条 survival curve 假设 |
| `src/cloop/v1/artifacts.py` | 六种平面结果文件、`RunArtifacts(version="v3")`、原子写入 | 复用，不建立新 artifact 平台 [S4] |
| `src/cloop/v4/outcome.py` | 逐成员预测 outcome 的思想；也包含本轮不需要的辅助 loss | 只借鉴逐成员前向逻辑；**不要直接导入它的模型或 loss** [S5] |
| 官方 `Predictor/models/survival_module.py` | 双向 attention，pre/pred/delta/condition 融合，risk 与 survival-logit 双输出 | 原样迁入独立文件，所有组使用同一类 [S6] |
| 官方 `Predictor/models/full_model.py` | full model 调用 `SurvivalModule`，Cox + 一年 masked BCE；实际传入 two-way layers=2 | 用于核对接法和核心架构设置，不搬入 LLM/MRI 在线编码器 [S7] |

### 2.1 现有 baseline 不是完整 CLARITY

Cloop 的 `baseline` 是当前 Cloop one-step dynamics。把 outcome 改为官方类之后，仍不能把该行直接改名为“Official CLARITY”。原因包括 latent 形式、条件编码、训练流程等并未全部复刻。

本轮正确标题是：

> **CLARITY outcome architecture 下的 dynamics 受控比较。**

未来增加官方 CLARITY 参考行时，必须在统一患者、标签和可用信息下重新评价，不能直接把论文分数抄入同一主表。本轮不以完整官方复现为开发阻塞条件。

### 2.2 Ensemble 与 RRT 的名称按已有实现解释

v2 中 ensemble 是多个独立随机种子训练的成员；不能仅凭名称称其为完整贝叶斯后验或经过校准的随机疾病分布。当前 RRT 路线使用可用多步 window 的递归 rollout 及 terminal latent loss；本轮不悄悄改成另一种 scheduled sampling 或多阶段 loss。[S2]

---

## 3. 只新增四个生产代码文件

```text
src/cloop/v3/
├── clarity_survival_module.py       # 上游 SurvivalModule 源码，保留来源/许可说明
├── clarity_downstream_head.py       # 输入适配、Cox+BCE、成员输出聚合
├── clarity_downstream_metrics.py    # 组合现有指标、配对汇总、selective 评价
└── clarity_downstream.py            # prepare/train/evaluate/uncertainty/report 入口

configs/v3/clarity_downstream.yaml

tests/v3/test_clarity_downstream.py

outputs/v3/next_stage_clarity_outcome_v1/
├── run.json
├── models.pt
├── last.pt                         # 可选：缓存 forecast 张量，不是新的目录树
├── metrics.json
├── predictions.jsonl
└── report.md
```

不修改旧 v1/v2/v3/v4 的训练入口；不覆盖任何既有输出。直接使用 `RunArtifacts(..., version="v3")`，无需修改允许版本集合。

`clarity_downstream.py` 需使用自己的 config loader，不能把新增配置直接交给 v3 原 `load_experiment_config()`；原函数会拒绝未知字段。[S1]

---

## 4. 数据划分：先快速验证，再做一次独立确认

本轮不建立 nested cross-fitting 系统。采用以下两个清楚命名的协议，代码主体完全相同。

### 4.1 默认：`development_reuse`

用于快速回答新 head 是否出现下游收益，尽量复用已有计算。

对原 v2 的每折：

- `T`：该 world 的训练患者。
- `E`：该折原 validation 患者。
- 从 `T` 内按固定患者 ID 规则分出 outcome-fit `A` 和 outcome-stop `B`，约 80%/20%。
- 四个 dynamics 各自为 `A/B/E` 生成 forecast；四个 outcome 分别在 `A` 拟合、在 `B` 早停、在 `E` 报告。
- 同一折内，`A/B/E` 不随 variant 改变；推荐也不随训练 seed 改变。
- 不从其他外层折借用 world 生成 `A` 的特征，不使用 v4_1 的 `_cross_fitted_outcome_rows()`。

**必须保留的解释边界：**

源码中 `_train_world()` 用 `_world_validation()` 在原 validation 的 H2/H3 MSE 上选 checkpoint。因此 `E` 虽然没有用于该 world 的梯度拟合，却参与过上游选模，不能把本轮复用这些权重的分数称为全管线完全未触碰的外层测试结果。[S2]

此外，`A/B` 的 world forecast 是样本内特征，`E` 是样本外特征；这是一种明确的两阶段训练协议，可能有分布差异。四组统一遵循该协议，**不等于已经消除了所有适配问题**。本轮先如实报告，不为此重新引入复杂交叉拟合。

`run.json` 必须保存：

```json
{
  "protocol": "development_reuse",
  "evidence_scope": "development comparison; source worlds selected on source validation",
  "world_train_feature_mode": "in_sample",
  "formal_test_untouched": true
}
```

### 4.2 确认阶段：`original_holdout`

只有开发路线锁定后执行，不把它变成每次修改都必须重跑的前置条件。

使用基础 run 已登记的 `train / validation / test` 患者划分：

1. normalizer、action codec 等只在 `train` 拟合。
2. 四组 world 在 `train` 拟合，`validation` 选 checkpoint。
3. 每组 outcome 在自己的 `train` forecast 上拟合，`validation` 早停。
4. 所有设置锁定后，四组在相同 `test` 患者上一次性评价。

可调用现有 `_build_fold_bundle()`、`_train_world()`、`_world_from_entry()`；不要走依赖旧 OT Gate B 的 `run_worlds()` 总入口。新的确认权重写入新 run，不覆盖 v2 checkpoint。

这只是普通 train/validation/test，不需要 nested CV。`test` 不得用于筛选 head 架构、loss 权重、uncertainty 阈值或最优 seed。若旧 test 已被实际查看，应如实记录，不能继续标为 sealed。

### 4.3 最低限度断言

```python
assert not (fit_ids & stop_ids)
assert not ((fit_ids | stop_ids) & report_ids)
assert same_ids_for_all_variants
assert normalizer_fit_ids <= permitted_preprocessing_train_ids
```

确认实验另外检查整个上游训练/选模、outcome 训练/选模均未使用 final-test IDs。不需要开发通用依赖追踪框架，直接在 manifest 保存这些 ID 列表或可核对 hash 即可。

---

## 5. 样本定义：不改标签，只改预测来源

### 5.1 对齐到 v3 的 H1/H2/H3

每个 H 使用 `_refs(bundle, split, horizon=H, labels=True)`，每患者每 H 只取第一个合格 window。[S1]

假设窗口为 `s → t=s+H`：

- `pre_latent = tr.latents[s]`：真实起点。
- `pred_latent = mean_m rollout_m[t]`：四组自己的递归预测终点。
- `true_post_latent = tr.latents[t]`：只用于 latent MSE、参考诊断，不能作为预测 head 的主输入。
- `time = tr.survival_time[t]`。
- `event = tr.survival_event[t]`。
- factual actions、MRI intervals、clinical、history 的取值规则对四组完全一致。

主 head 的 pre 始终是同一个真实起点，不给某一组偷偷补入真实中间 MRI。本轮将所有方法统一定义为“起点—预测终点”的跨度对，不混用“最后相邻两步”的接口。

### 5.2 生存时间锚点

本轮保留现有 **目标 landmark 之后的预后预测**：标签从 `t` 起算，而非从 `s` 起算。这是研究预测 MRI 的预后可用性，不是从起点执行治疗计划后的总体价值。

不把 `survival_time[t]` 改成 `survival_time[s]` 后继续使用终点信息；不把后续真实 MRI 存在的样本选择解释为所有起点患者都能到达该终点。

主实验使用原始正的剩余随访时间计算 Cox 与 C-index，不因为旧 piecewise head 的 `[0,90,180,365,730]` edges 而自动截断到 730 天。新 head 没有这组 hazard bins。此变化要写入 run；不要直接与旧表的数值视为严格同指标比较。

### 5.3 每个 H 独立训练 head

推荐 `head_scope = per_horizon`：`G[v,fold,seed,H]` 独立训练。

理由很具体：每个训练 cohort 内每患者只有一条记录，避免把同患者 H1/H2/H3 的不同剩余生存时间一起放进 Cox risk set，也避免为了共享 head 编写复杂加权或分层 Cox。

最多 `4 variants × 5 folds × 3 seeds × 3 horizons = 180` 个小型 head，不是 180 个 MRI/LLM 模型。首次 smoke test 只运行一个 fold、一个 seed 的 H1/H2。

H2/H3 患者子集不同，所以不同 H 的指标升降不能单独解释成误差累积趋势。报告每 H 的患者数、事件数和一年可判定标签数。小样本 H3 不承担唯一成败判断。

### 5.4 Forecast row 的最小字段

| 字段 | Shape/类型 | 用途 |
|---|---|---|
| `patient_id`, `start_index`, `target_index`, `horizon` | 标量 | 匹配患者和窗口 |
| `pre` | `[D]` | 真正的起点 latent |
| `post_mean` | `[D]` | 本组预测均值 |
| `post_members` | `[M,D]` | uncertainty 推理；单模型 M=1 |
| `true_post` | `[D]` | latent MSE / 参考实验，不送主 head |
| `condition` | `[Ccond]` | 同一事实条件 |
| `time`, `event` | 标量 | 终点 landmark 的生存标签 |
| `latent_mse`, `latent_disagreement` | 标量 | 诊断 |

使用 Python dict 即可，不引入新的患者数据库或轨迹存储层。`feature_cache` 可直接保存 `last.pt`，按 `fold/seed/variant/split/H` 建字典。

`record_id` 必须包含 `variant/fold/seed/H/patient_id/start/target`；推理模式不同还需包含 `mean_latent` 或 `mean_probability`，防止覆盖。

---

## 6. CLARITY outcome 接入：保留真实网络，不造一个“同名 head”

### 6.1 上游迁入

将固定提交的 `Predictor/models/survival_module.py` 复制为：

```text
src/cloop/v3/clarity_survival_module.py
```

保留三类：

```python
TwoWayCrossAttentionLayer
TwoWayTransformer
SurvivalModule
```

保留 upstream 来源与许可证要求。不要把两个输出改成 piecewise hazards，也不要加一个独立 planning value 网络。

**官方的 risk head + survival-logit head 要保留。** 这两个输出分别接受 Cox 排序与一年生存 BCE 监督，不是此前 v4 中额外学习重复一年风险的 value head。

源码注释有些地方称输出 head 为“single linear”，但实际实现是 `Linear → GELU → Dropout → Linear`。以代码为准，不能按注释简化。[S6]

### 6.2 显式超参数

四组使用同一个配置：

```python
SurvivalModule(
    latent_dim=bundle.latent_dim,   # 当前 expected=768；运行时核对，不能照抄默认767
    num_modalities=1,               # 这里表示本轮的单个全局token，不冒充4种MRI模态
    hidden_dim=128,
    attention_dim=128,
    num_twoway_layers=2,
    num_heads=4,
    dropout=0.3,
    condition_dim=condition_dim,
)
```

这些是本轮统一设置；其中 two-way layers=2 对应已核查 full model 的调用，不能把 `SurvivalModule` 类默认 layers=1 错称为当前 full model 实际设置。dropout 等以本轮配置固定，不声称它们必然等于每一份官方运行脚本的覆盖参数。[S6][S7]

### 6.3 单个全局 latent 如何变成官方接口

Cloop rollout 的状态是 `[B,D]`；官方接口接受 `[B,P,D]`。本轮使用：

```python
pre_tokens = pre[:, None, :]          # [B,1,D]
post_tokens = post_mean[:, None, :]   # [B,1,D]
```

- 不把 ensemble 维 M 当成 token 维 P。
- 不把 H 个时间点塞进 P 并声称还是同一个 pair-wise outcome。
- 不复制一个 latent 32 次冒充 BrainIAC 空间 token。
- 不把同为 768 维当作官方 checkpoint 可直接兼容的依据；本轮 head 从头训练。

单 token 模式下 attention 的 token 选择性会退化，但网络仍按同一类计算 pre/pred 交互。它是基于现有 global latent 的合理受控适配，**不是对官方多 token MRI 表示能力的完整复刻**。未来有真实多 token latent 才扩展 P；本轮不新建 token 生成器。

### 6.4 条件向量保持简单、各组完全相同

默认使用现有结构化信息，不新接 MedGemma/LLM：

\[
c_{s:t}=[c,\;mask_c,\;h_t,\;\log(1+\text{elapsed days})/\log(366)].
\]

因此：

```python
condition_dim = 2 * bundle.clinical_dim + bundle.history_dim + 1
```

`history[t]` 表示本次事实窗口终点已有的治疗历史；时间为原始 MRI 时间差。所有组共享同一值。结构化条件并不等于官方 MedGemma 的 drug/clinical embedding；这项差异应在实验设置中写明。

不要把 `true_post`、生存标签、预测误差、uncertainty 或未来新增临床变量放入主条件。

### 6.5 Adapter 的接口

```python
class ClarityOutcomeAdapter(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, **head_config):
        super().__init__()
        self.head = SurvivalModule(
            latent_dim=latent_dim, num_modalities=1,
            condition_dim=condition_dim, **head_config,
        )

    def forward(self, pre, post, condition):
        if pre.ndim != 2 or post.shape != pre.shape:
            raise ValueError("pre/post must be matching [B,D]")
        if condition.ndim != 2 or condition.shape[0] != pre.shape[0]:
            raise ValueError("condition must be [B,Ccond]")
        risk, logit = self.head(pre[:, None], post[:, None], condition)
        return risk.squeeze(-1), logit.squeeze(-1)
```

实际实现需再检查 finite、device、维度；`B=1` 时只 `squeeze(-1)`，不要无参数 `squeeze()`。

---

## 7. 训练目标：相同 Cox + 一年 BCE，分别适配各组 forecast

### 7.1 主 loss

\[
L=0.5L_{\mathrm{Cox}}+0.5L_{\mathrm{BCE365}}.
\]

不加：latent reconstruction loss、pairwise ranking、value BCE、variance calibration、trajectory consistency 或规划 loss。Dynamics 已冻结，所以 outcome 训练期间的 latent loss也不能改变 dynamics。

一年 BCE 沿用官方实际公式而非有歧义的注释：[S7][S8]

```python
y_alive = (time > 365).float()
valid = (time > 365) | ((time <= 365) & (event == 1))
```

提前删失者不进入这一 BCE；不能标为死亡。`time==365,event==0` 也按此约定排除。一年后发生事件的患者，在一年生存标签中为 1。

这是为了保持相同训练目标，不是声称 masked BCE 已自动解决全部概率校准问题。主概率评价仍使用删失感知的 IPCW Brier。

### 7.2 Cox risk set 与 ties

本轮每 H 的患者数较小，建议 outcome 使用 **full-cohort loss**，每个 epoch 对该 H 的全部 fit 患者做一次更新。Forecast 提取可用 batch=32，这个 batch 与 Cox 的 risk set 无关。

原官方 `NLLDeepSurvLoss` 会给时间加随机微扰处理 ties。[S9] 为使验证 loss 确定且输入顺序无关，本轮建议使用明确的 Breslow ties 版本（参考代码见附录 A）。这属于统一的数值/训练协议选择，**不声称逐行复刻官方训练器**；四组必须使用同一个实现。

不要随机拆成小 batch 后简单把各 batch Cox loss 当成全 cohort loss。若显存不足，可对 head 前向分块但保留计算图，合并所有 risk 后计算一次 Cox；本轮单 token、小 cohort 通常无需这一优化。

### 7.3 为什么分开 H 的小 head 更简单

每 H 每患者一条记录，训练和 validation 可以直接使用等患者权重；不需要 `1 / patient_window_count` 的跨 H 权重。不要沿用 v3 原跨 H rows 的权重后又给每 H 单独训练，造成重复加权。

### 7.4 初始化和早停

同一 fold/seed/H 下，先生成一份初始化 `init_state`，四组分别复制：

```python
init_state = deepcopy(new_head(seed).state_dict())
for variant in variants:
    head = new_head(seed)
    head.load_state_dict(init_state)
    optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    # 分别训练，每组有独立optimizer和最佳checkpoint
```

建议首版使用：full-cohort，max_epochs=300，patience=40，dropout=0.3，grad_clip=1.0。这些是待验证的统一起始配置，不保证最优。与旧 mini-batch 训练不同，每 epoch 仅一次更新，所以不能机械复用“60 epochs”并认为更新预算不变。

在 stop 集上 `head.eval()`，计算完整 cohort 的 `0.5*Cox+0.5*BCE`；记录 epoch0，以及逐 epoch 的 train/stop 分量、梯度范数、输出方差。保存最优 stop loss；不能用 report/test C-index 选 checkpoint。

若某 H 的 fit 集无事件或 stop 集没有任何可用监督，标记 `unavailable` 并给原因；不根据某组结果临时换患者。所有组使用相同可评价任务集合。评价 C-index 无可比 pair 时返回 null，不补 0.5。

### 7.5 主训练与评价的最短流程

以下是流程伪代码，不是当前已存在的入口：

```python
for fold in source_folds:
    bundle = load_same_fold_bundle(fold)
    reference_rows = build_label_only_reference_rows(bundle)  # 与variant无关
    fit_ids, stop_ids = choose_patient_split_once(reference_rows, fold)

    for seed in seeds:
        for variant in ["baseline", "rrt", "ensemble", "rrt_ensemble"]:
            world = load_own_world(fold, seed, variant)
            world.eval().requires_grad_(False)
            rows = extract_own_forecasts(world, bundle, horizons=[1, 2, 3])
            assert_label_and_window_alignment(rows, reference_rows)

            for H in horizons:
                fit, stop, report = select_by_ids_and_horizon(rows, H)
                head = initialize_same_architecture(seed, H)
                train_head_on_own_forecasts(head, fit, stop)
                evaluate_on_own_forecasts(head, report)
                save_model_and_predictions(variant, fold, seed, H)
```

不能把 `outcome_world` 固定成 `rrt_ensemble` 后只改变结果名；每组必须使用其自己的 forecast 来训练与评价自己的 head。

---

## 8. 指标：正确区分官方两个输出

### 8.1 必须实现的主表字段

```python
risk_score, survival_logit = head(pre, predicted_post, condition)
survival365 = sigmoid(survival_logit)
```

| 指标 | 输入 | 含义 |
|---|---|---|
| `c_index_risk` | 原始 `risk_score`，越大越危险 | Cox 分支的生存排序，主排序指标 |
| `ipcw_brier365` | `survival365`，越大越可能存活 | 一年概率预测误差 |
| `td_auc365` | `1-survival365` | 一年事件风险排序，次指标 |
| `cox_partial_nll` | risk/time/event | 诊断，不称完整 survival distribution NLL |
| `bce_identifiable` | logit 与可判定一年标签 | 诊断，不冒充总体删失校正概率误差 |
| `latent_mse_same_cohort` | 该 outcome cohort 的预测/真实终态 | 同 cohort 的上游证据 |

必须分别调用 `harrell_c_index(times, events, risk_scores)` 和 `ipcw_brier(...)`。**不能直接复用把 `1-survival_probability` 作为唯一 risk 的旧 `survival_summary()` 来生成主 C-index。**

### 8.2 不要虚构完整生存曲线

官方 head 的 survival 分支只有一个一年 logit。主实验不报告 IBS，也不报告 piecewise survival NLL；不能把这一 logit当成 90/180/365 天整条曲线。

未来若通过 Cox risk 配合仅在训练集拟合的基线累计 hazard 构造曲线，可以另做明确实验，但它不是本轮必需部分。

### 8.3 不沿用自动翻转的 AUC 工具

已核查的官方 `compute_auc()` 包含根据评价标签检查方向并可能翻转预测的逻辑。[S8] 本轮应使用 Cloop 的明确方向 `time_dependent_auc`，不自动翻转、不平滑跨 epoch AUC。生存概率与事件风险的方向在代码中一次定义，不能由评价结果决定。

### 8.4 IPCW 参考数据

- 删失分布在对应训练/拟合患者的标签上估计，四组共享同一个 reference。
- 不把 report/test 标签用于拟合 censoring reference。
- follow-up support 不足或权重无定义时输出 null 与原因。
- 主表对所有符合预先定义的患者评价；不能删掉“模型不好”的病例。

沿用 Cloop 当前 IPCW 实现时保留其假设和返回原因。不能因为某方法指标无定义而把它记成 0，也不能重选一个只对某方法有利的阈值。

### 8.5 汇总与比较

开发协议：先在每折内平均三个 seed 的指标，再跨折报告均值、样本 SD 和有效折数。不要把同一患者三个 seed 当成三名独立患者，也不要直接拼接不同 fold 的原始 Cox risk 后计算一个不经说明的 pooled C-index。

配对比较必须使用同一 fold/seed/H 的样本 key；保存每折差值及改善折数。确认 test 上可以按患者成组重采样做 paired bootstrap，三个 seed 的配对变化在每次重采样内先汇总。区间不能替代小样本限制，也不能承诺显著性。

主表推荐：

| Method | H | N/events | Same-cohort latent MSE | C-index(risk) | IPCW Brier@365 | TD-AUC@365 |
|---|---|---|---|---|---|---|
| baseline | 1/2/3 | … | … | … | … | … |
| rrt | 1/2/3 | … | … | … | … | … |
| ensemble | 1/2/3 | … | … | … | … | … |
| rrt_ensemble | 1/2/3 | … | … | … | … | … |

本轮重新提取同 cohort MSE；不要把旧全-window dynamics MSE 与新每患者一个 window 的 outcome 指标拼在一行后解释为逐例传递。

---

## 9. Uncertainty：不改主 head，不加新 loss

### 9.1 先完成四组主表，再跑 forward-only 消融

仅对 `ensemble` 与 `rrt_ensemble`：复用本组已训练好的 outcome，设为 eval，对每个成员终态分别前向。

\[
(r_m,\ell_m)=G_{\phi_v}(z_s,\hat z_t^{(m)},c_{s:t}),\quad p_m=\sigma(\ell_m).
\]

\[
\bar p=\frac1M\sum_m p_m,\quad
u_p=\sqrt{\frac1M\sum_m(p_m-\bar p)^2}.
\]

- 主表：`mean_latent`，即 `G(pre, mean(post_members), condition)`。
- 附加表：`mean_probability`，即 `mean(sigmoid(member_logits))`。
- 附加 C-index 可使用同一个 head 下成员 risk-score 的均值；明确这是 risk-score 聚合，不是生存概率。
- 不平均 logit 后当作平均概率；一般 `sigmoid(mean(logit)) != mean(sigmoid(logit))`。
- 不新增成员专属 head，不混合不同 seed/fold 的 Cox 原始 risk。

本消融有意检验“推理时聚合顺序变化”；主 head 是在 mean-latent 特征上训练的，因此逐成员输入可能有分布偏移。要如实称为 inference-only ablation，不能说它是已经训练校准的完整概率模型。只有这个小实验显示值得继续时，再设计额外的成员输入训练，不把它放入首版。

### 9.2 Uncertainty 的定义边界

`u_p` 叫 **dynamics-induced survival-probability disagreement**。它不等于全部临床不确定性，也没有覆盖 outcome 参数不确定性、共同偏差和全部事件随机性。

不再使用：

```python
smooth_l1_loss(risk_variance, (predicted_risk - event_target)**2)
```

不把 uncertainty 直接作为死亡概率，不强制高 uncertainty 患者有更差预后。

### 9.3 选择预测只作为第二张表

对同一套 prediction：

1. 按 `u_p` 从低到高保留预先声明的覆盖水平，如 100%、80%、60%。
2. 排序时保留全体合格患者，包括提前删失者；不能先按结局可判定性筛人。
3. 在被保留集合上计算 IPCW Brier；分母是被保留集合人数，删失权重 reference 仍来自训练 cohort。
4. 报告相同数量随机保留的参考误差，避免把任意小子集变化误认为 uncertainty 有效。
5. 同时报告实际覆盖率、保留事件数、删失数以及权重支持情况。

两种阈值模式明确区分：

- `rank_curve`：评价集上仅按 uncertainty 排名生成描述性 coverage curve，不用标签选阈值；不是部署阈值验证。
- `fixed_threshold`：在 outcome-stop 集的 uncertainty 上取预定分位数，冻结阈值后作用到 report/test；报告实际而非名义 coverage。这是最小部署式检查，不声称额外独立校准集已存在。

若选择与删失机制相关，边际 censoring-KM 的假设可能不足；现阶段将 selective 结果作为可靠性探索，不能凭一条下降曲线宣布正式校准成功。主表仍然是不拒绝任何人的全 cohort 结果。

### 9.4 最少要保存的 uncertainty 输出

```text
member_survival365   [M]
member_risk_score    [M]
survival365_mean_probability
survival365_mean_latent
probability_std
latent_disagreement
```

单模型可以记录 `uncertainty_available=false`；不能把 0 当作“完全确定”。M=1 的经验方差为0只是定义，不是医学置信度。

---

## 10. 各文件具体职责和接口

### 10.1 `clarity_survival_module.py`

原样承载上游类。记录 source commit、blob SHA `21e984ee04745348b8fc5151e4650a63616a4567`。不要在四个 variant 下各拷贝不同版本。

### 10.2 `clarity_downstream_head.py`

实现：

```python
ClarityOutcomeAdapter(latent_dim, condition_dim, head_config)
cox_breslow_nll(risk, time, event)
fixed_time_survival_targets(time, event, tau=365.0)
outcome_loss(risk, survival_logit, time, event, ...)
summarize_member_logits(member_risk, member_survival_logit)
```

输入维度严检；risk/logit 必须是 `[N]`；成员输出必须是 `[M,N]`。附录 A 给出后四个辅助函数的可运行参考。

### 10.3 `clarity_downstream_metrics.py`

实现小函数，不重新实现全部生存指标库：

```python
evaluate_cohort(prediction_rows, censor_reference, tau)
paired_comparisons(task_metrics)
selective_ipcw(prediction_rows, censor_reference, coverages, seed)
build_report(metrics, run)
```

`evaluate_cohort()` 内分别调用已有 Harrell、IPCW Brier、TD-AUC。`selective_ipcw()` 选择样本后调用同一个 IPCW Brier，不能使用 v4_1 排除早期删失后普通平方误差的函数。

### 10.4 `clarity_downstream.py`

实现：

```python
load_downstream_config(path)
prepare_rows(world, bundle, split, refs, feature_batch_size)
train_one_head(fit_rows, stop_rows, spec, seed)
evaluate_one_head(head, report_rows, reference_rows, spec)
run_uncertainty(head, report_rows, stop_rows, reference_rows, spec)
run_stage(config, spec, stage, fold=None, seed=None, horizon=None, variant=None)
```

`prepare_rows()` 调用以下已存在接口：

```python
rollout = world.rollout(
    batch["z0"], batch["actions"], batch["deltas"],
    batch["context"], batch["clinical_mask"], batch["history0"],
    batch["step_mask"],
)
# rollout.states: [M,B,H+1,D]
member_post = rollout.states[:, :, H, :]
mean_post = member_post.mean(0)
```

四组均在 `torch.no_grad()` 下生成特征，保存 CPU 张量；不让 outcome 梯度意外回传至 world。

`_source()` 和 `_fold_bundle()` 可用于 `development_reuse`；`original_holdout` 使用基础 split 和训练集预处理，不能把原折 normalizer/world 生硬移过去。

### 10.5 不需要新增的内容

无需 `TrajectoryValueMPC`、`Observer`、belief update、nested feature cache service、通用 callback 系统、新 LLM agent、肿瘤 progression 标签生成器。当前没有这些模块，不影响主实验成立。

---

## 11. 配置建议

以下是**新增配置规范**，需由新的 loader 实现，不能假定现有程序已经接受这些键。

```yaml
clarity_downstream:
  run: next_stage_clarity_outcome_v1
  protocol: development_reuse
  base_run: brainiac_main_v1_provenance
  world_run: outcome_v2_clarity_v1

  variants: [baseline, rrt, ensemble, rrt_ensemble]
  seeds: [7, 17, 29]
  horizons: [1, 2, 3]
  primary_horizon: 2
  head_scope: per_horizon
  pair_mode: observed_start_predicted_endpoint
  landmark: first_eligible_per_patient_per_horizon
  label_anchor: target_landmark
  administrative_censor_days: null

  feature_batch_size: 32
  token_mode: single_global_token
  condition_mode: clinical_mask_target_history_elapsed
  expected_latent_dim: 768
  main_aggregation: mean_latent
  train_outcome_independently: true
  freeze_dynamics: true

  head:
    hidden_dim: 128
    attention_dim: 128
    num_twoway_layers: 2
    num_heads: 4
    dropout: 0.3

  training:
    full_cohort_loss: true
    lr: 0.001
    weight_decay: 0.0001
    max_epochs: 300
    patience: 40
    grad_clip: 1.0
    outcome_split_seed: 1701
    early_stop_metric: cox_bce
    cox_ties: breslow
    cox_weight: 0.5
    bce_weight: 0.5
    probability_horizon_days: 365.0

  uncertainty:
    enabled: true
    retrain_head: false
    compare_probability_aggregation: true
    score: probability_std
    coverages: [1.0, 0.8, 0.6]
    threshold_mode: rank_curve
    random_reference_repeats: 200

  evaluation:
    primary_metric: c_index_risk
    companion_metric: ipcw_brier365
    report_td_auc365: true
    aggregate_seeds_within_fold: true
    paired_bootstrap_repeats: 0  # 确认test时按预定协议另设，例如1000
```

配置中的 `max_epochs` 是统一预算，不是对某组承诺更长训练。修改 lr/head 超参数时给所有组相同搜索机会；不能只对组合调参后拿未调 baseline 比较。

---

## 12. 运行入口和顺序

以下命令是新增入口的目标用法，实施前不可视为已可执行。

### 12.1 首轮 smoke test

```bash
PYTHONPATH=src python -m cloop.v3.clarity_downstream \
  --config configs/v1/default.yaml \
  --paths configs/v1/server.yaml \
  --experiment-config configs/v3/clarity_downstream.yaml \
  --stage all --fold 0 --seed 7 --horizon 1 --device cpu
```

顺序为 `prepare → train → evaluate → uncertainty → report`。先检查四组 head 独立、输出方向正确、样本 key 一致，再开 GPU 完整跑。

### 12.2 完整开发实验

```bash
PYTHONPATH=src python -m cloop.v3.clarity_downstream \
  --config configs/v1/default.yaml --paths configs/v1/server.yaml \
  --experiment-config configs/v3/clarity_downstream.yaml \
  --stage all --device cuda:0
```

阶段要求：

| stage | 操作 |
|---|---|
| `prepare` | 读取来源，生成四组 forecast 缓存和共同 labels/IDs |
| `train` | 各组各 H 独立 head；禁止后台重训旧 world |
| `evaluate` | 主表 mean-latent 全 cohort 评价 |
| `uncertainty` | 对 ensemble 两组做成员前向和独立可靠性表，不更新参数 |
| `report` | 生成同样本配对汇总；缺失任务明确列出 |
| `all` | 按上述顺序串行执行 |
| `world` | 只用于 `original_holdout`，按现有训练函数重训确认用四组 world |

`original_holdout` 的 final-test 评价必须额外传 `--evaluate-test`。默认 `all` 只能生成 validation 开发报告，不能自动反复读取正式测试结果。

### 12.3 不增加重训练瓶颈

- 未改变 source world/config 时，`prepare` forecast 可复用。
- 改 head 配置时新建 run，不能混用旧 metrics。
- 修改 uncertainty 阈值分析无需重新训练 head，但必须标明设置来源，不能在 test 结果上反复择优。

---

## 13. 结果文件规范：保持现有扁平结构

### `run.json`

保存：源码 commit、官方 head commit/blob、配置、protocol、source world file hash、患者划分、样本 key hash、head 架构描述、已完成 task。记录是否使用了正式测试，是否 source world 在报告 cohort 上选过 checkpoint。

### `models.pt`

```python
{
    "schema_version": "cloop_clarity_outcome_v1",
    "heads": {
        "baseline/fold0/seed7/H1": {
            "state": ..., "best_epoch": ...,
            "init_state_hash": ..., "architecture_hash": ...,
            "fit_ids_hash": ..., "stop_ids_hash": ...,
            "world_key": "baseline/fold0/seed7",
        },
        # 每组保存自己的head，不能指向同一训练后state
    },
    "worlds": {}  # 只有original_holdout新训练时才写
}
```

`architecture_hash` 和同 fold/seed/H 的 `init_state_hash` 应对四组一致；最终参数 hash 不要求一致，也不能硬性要求必须不同。

### `predictions.jsonl`

```json
{
  "record_id": "clarity_out:development_reuse:baseline:fold0:seed7:H2:patient_key:start:target:mean_latent",
  "kind": "factual_downstream",
  "variant": "baseline",
  "fold": 0,
  "seed": 7,
  "horizon": 2,
  "patient_id": "patient_key",
  "label_anchor": "target_landmark",
  "time": 420.0,
  "event": 1,
  "risk_score": 0.12,
  "survival_logit365": 0.8,
  "survival365": 0.689974,
  "aggregation": "mean_latent",
  "latent_mse": 0.7,
  "head_key": "baseline/fold0/seed7/H2"
}
```

上述数值只是 schema 示例，不是实验结果。保存原始浮点精度，不用手写舍入值参与复算。

### `metrics.json` / `report.md`

分为 `main`、`paired`、`uncertainty`、`training`、`limitations`。主表不与 selective 子集表混排。训练日志至少保存 train/stop Cox、BCE、总 loss，best epoch、实际事件数；不再只有一个总 validation 数字。

---

## 14. 最小测试与验收

### 14.1 必须通过的单元/集成测试

| 测试 | 验收条件 |
|---|---|
| 官方 head 迁入 | 同参数、相同 state_dict、eval 模式下，适配器与原类输出一致 |
| 同架构独立训练 | 四组参数 shape 相同、初始 state 相同、optimizer 不共享；训练一组不修改另一组 |
| Forecast 来源 | 改变 variant 实际读取对应 world key；不是四次复用 `rrt_ensemble` |
| 维度 | `[M,B,H+1,D]` 中 M/P/H 不混淆；B=1、M=1 正常 |
| Leakage | `true_post/time/event` 不进入主 head 的 feature；报告患者不参与 head 训练 |
| 标签 | 早期删失被 mask；一年后事件算一年存活；边界遵循官方公式 |
| Cox | 风险方向、常数平移、ties 置换、极值稳定、无事件返回逻辑正确 |
| 指标接线 | 主 C-index 接 risk head；Brier 接 sigmoid(survival logit) |
| 概率聚合 | 逐成员平均概率，不错误平均 logits；dropout 已 eval |
| Selective | 不先删除早期删失者；actual coverage 以全体合格患者为分母 |
| Artifact | 新 run 不覆盖旧结果；相同记录不被不同 aggregation 覆盖 |

附录 A 辅助函数已在本次文档制作中通过 14 项 CPU 张量测试；这些不包含完整训练、源模块迁入 parity、真实数据或仓库集成测试。后四类工程测试仍需实现时运行。

### 14.2 一轮实验是否完成

完成的标准不是“组合必须赢”，而是：

- 预定四组、相同 H/cohort 均有独立 head 与可追溯结果。
- 四组 Cox/BCE/输入条件一致，无针对性调参偏置。
- 主表同时展示下游与同 cohort latent 指标。
- `rrt_ensemble vs ensemble` 的结果没有被省略。
- uncertainty 未通过删样本伪装成全体性能提升。
- 开发结果与独立确认结果的证据范围明确。

---

## 15. 如何根据结果决定下一步

| 观察 | 支持的判断 | 下一步 |
|---|---|---|
| 组合相对 baseline 的下游指标稳定改善 | 上游改进在同 outcome 架构下有下游价值 | 做锁定设置的独立确认，再考虑论文主表 |
| ensemble 改善，但组合不优于 ensemble | 集成有价值，RRT 额外贡献未确定 | 研究RRT本身，不再通过换head掩盖 |
| latent MSE改善，下游所有指标仍无收益 | 当前同架构下未见下游价值 | 可加一个真实目标latent参考组定位，不先扩展planner |
| C-index不变但Brier改善 | 概率准确性收益，非排序收益 | 如实报告，不要求所有指标同时改善 |
| uncertainty识别高误差但均值预测不变 | 可靠性识别有价值，非全样本精度收益 | 报告coverage代价，独立于主表 |
| 所有方法表现很差 | 可能是表示/任务/head适配问题，也可能样本不足 | 先排查标签、condition、单token适配与训练；不预设更大网络能解决 |

可选真实-target参考必须独立命名：可在真实 pre/post 上训练同架构 head 并在真实 post 上评价，以评估这套表示的预后可用性。它使用不同输入信息，不是主比较的“官方baseline”或必然上界。**这一参考不是四组主实验的前置条件。**

---

## 16. 给开发者的执行摘要

> 在 Cloop 当前提交上新增一个 v3 独立入口。复用 v3 的 split/window/rollout 和 v2 的四组 world。迁入固定版本的 CLARITY `SurvivalModule`，四组使用相同架构、相同初始化规则和 Cox+BCE 目标，但各自训练自己的 head。每个 H 独立训练；主输入为真实起点和本组预测终点均值，保持条件和终点生存标签完全一致。主表报告 raw Cox-risk C-index 与一年 IPCW Brier，不能伪造 IBS 或完整 survival NLL。完成后，对 ensemble 两组用各自已有 head 做逐成员前向，单独报告概率平均和 uncertainty 的可靠性分析。不要引入 planner、value head、uncertainty训练loss或跨外层折借模型。旧world复用只作为开发证据；路线锁定后用普通train/validation/test做一次独立确认。

---

## 附录 A：可运行的损失与成员聚合参考

下列代码是**本方案新增的参考实现**，不是从仓库直接抽取的已有文件。Cox 使用确定性 Breslow ties，与官方随机时间微扰实现的数值协议不同，四组须统一使用。它不包含数据加载或完整 trainer。

```python
"""Standalone reference utilities proposed for the new experiment (not repo code)."""
from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def validate_survival_inputs(risk: Tensor, time: Tensor, event: Tensor) -> None:
    if risk.ndim != 1 or time.shape != risk.shape or event.shape != risk.shape:
        raise ValueError("risk/time/event must be matching [N] tensors")
    if risk.numel() == 0:
        raise ValueError("empty survival cohort")
    if not bool(torch.isfinite(risk).all() and torch.isfinite(time).all()):
        raise ValueError("non-finite risk/time")
    if not bool((time > 0).all()):
        raise ValueError("time must be positive")
    if not bool(((event == 0) | (event == 1)).all()):
        raise ValueError("event must be binary")


def cox_breslow_nll(risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
    """Full-cohort Cox partial NLL, deterministic Breslow handling of ties."""
    validate_survival_inputs(risk, time, event)
    observed = event.bool()
    if not bool(observed.any()):
        return risk.sum() * 0.0
    result = risk.sum() * 0.0
    for day in torch.unique(time[observed]):
        deaths = observed & (time == day)
        at_risk = time >= day
        result = result + deaths.sum() * torch.logsumexp(risk[at_risk], dim=0)
        result = result - risk[deaths].sum()
    return result / observed.sum()


def fixed_time_survival_targets(time: Tensor, event: Tensor, tau: float = 365.0):
    if time.ndim != 1 or time.shape != event.shape or time.numel() == 0:
        raise ValueError("time/event must be non-empty matching [N] tensors")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not bool(torch.isfinite(time).all() and (time > 0).all()):
        raise ValueError("invalid follow-up time")
    if not bool(((event == 0) | (event == 1)).all()):
        raise ValueError("event must be binary")
    labels = (time > tau).to(torch.float32)
    valid = (time > tau) | ((time <= tau) & (event == 1))
    return labels, valid


def outcome_loss(risk: Tensor, survival_logit: Tensor, time: Tensor, event: Tensor,
                 tau: float = 365.0, w_cox: float = 0.5, w_bce: float = 0.5):
    if survival_logit.shape != risk.shape:
        raise ValueError("risk and survival_logit must be matching [N] tensors")
    if not bool(torch.isfinite(survival_logit).all()):
        raise ValueError("non-finite survival_logit")
    if not (math.isfinite(w_cox) and math.isfinite(w_bce)) or min(w_cox, w_bce) < 0:
        raise ValueError("loss weights must be finite and non-negative")
    if w_cox + w_bce == 0:
        raise ValueError("at least one loss weight must be positive")
    cox = cox_breslow_nll(risk, time, event)
    labels, valid = fixed_time_survival_targets(time, event, tau)
    bce = F.binary_cross_entropy_with_logits(survival_logit[valid], labels[valid]) \
        if bool(valid.any()) else survival_logit.sum() * 0.0
    return {
        "total": w_cox * cox + w_bce * bce,
        "cox_partial_nll": cox,
        "bce_identifiable": bce,
        "n_events": int(event.sum()),
        "n_identifiable": int(valid.sum()),
    }


def summarize_member_logits(member_risk: Tensor, member_survival_logit: Tensor):
    """Inputs [M,B]. One shared head; do not confuse M with token dimension."""
    if member_risk.ndim != 2 or member_risk.shape != member_survival_logit.shape:
        raise ValueError("member outputs must be matching [M,B] tensors")
    if min(member_risk.shape) < 1:
        raise ValueError("empty ensemble/batch")
    if not bool(torch.isfinite(member_risk).all() and torch.isfinite(member_survival_logit).all()):
        raise ValueError("non-finite member output")
    member_survival = torch.sigmoid(member_survival_logit)
    return {
        "member_survival": member_survival,
        "mean_survival": member_survival.mean(0),
        "mean_risk_score": member_risk.mean(0),
        "probability_variance": member_survival.var(0, unbiased=False),
        "probability_std": member_survival.var(0, unbiased=False).sqrt(),
    }
```

### 本次实际执行的参考测试

14 项 CPU 检查包括：正确风险排序的 Cox loss 更低；Cox 常数平移不变；ties 输入置换不变；ties 的零分数解析值；极大 risk 的 loss/gradient 有限；无事件可微零值；一年标签 mask；生存标签方向；两个输出分支均获得梯度；全提前删失零监督；M=1 方差为零；平均概率不同于平均 logit 后 sigmoid；概率方差在定义范围内；无效标签报错。

这些测试只验证附录代码的局部数值行为，不等同于对原始 MRI 数据、官方 head 权重或新实验入口的验证。

---

## 附录 B：源码依据

本文的新实验选择、配置与代码接口属于设计建议；以下链接支持“当前仓库实际做了什么”，不是为预期实验提升背书。

- **[S1] Cloop v3 数据、world 加载、window、旧 outcome 入口**：
  <https://github.com/yu3jun1/Cloop/blob/f24ed763daf9e078180f86bb3ac2d784b0bdc2ad/src/cloop/v3/next_stage.py>
- **[S2] Cloop v2 world 训练、validation checkpoint 选择、模型加载**：
  <https://github.com/yu3jun1/Cloop/blob/f24ed763daf9e078180f86bb3ac2d784b0bdc2ad/src/cloop/v2/outcome_v2.py>
- **[S3] Cloop 指标与 censoring reference**：
  <https://github.com/yu3jun1/Cloop/blob/f24ed763daf9e078180f86bb3ac2d784b0bdc2ad/src/cloop/v1/metrics.py>
- **[S4] Cloop 平面 artifact 接口**：
  <https://github.com/yu3jun1/Cloop/blob/f24ed763daf9e078180f86bb3ac2d784b0bdc2ad/src/cloop/v1/artifacts.py>
- **[S5] Cloop v4_1 outcome 实现（仅用于识别不再沿用的部分）**：
  <https://github.com/yu3jun1/Cloop/blob/f24ed763daf9e078180f86bb3ac2d784b0bdc2ad/src/cloop/v4/outcome.py>
- **[S6] 官方 CLARITY SurvivalModule**：
  <https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/models/survival_module.py>
- **[S7] 官方 full model 的模块调用与 Cox/BCE**：
  <https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/models/full_model.py>
- **[S8] 官方一年标签函数与 AUC 实现**：
  <https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/utils/metrics.py>
- **[S9] 官方 Cox loss 与 ties 实现**：
  <https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/losses/auton_survival_loss.py>
- **[S10] Cloop v3 time-dependent AUC 等工具**：
  <https://github.com/yu3jun1/Cloop/blob/f24ed763daf9e078180f86bb3ac2d784b0bdc2ad/src/cloop/v3/trajectory_metrics.py>

**结束语：先用同构、独立训练的 outcome 把四组下游结果做清楚。后续是否需要新 head 或闭环系统，由这轮证据决定，而不是把它们设为本轮必须完成的工程任务。**

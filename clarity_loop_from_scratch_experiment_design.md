# 从空白仓库搭建可靠闭环医疗世界模型：实验代码设计

**版本：** v1.1 · 2026-09-21  
**建议项目名：** `CLARITY_Loop`；Python 包名：`cloop`  
**本阶段目标：** 不依赖在线 LLM，先建立可训练、可评估、可回放、可做独立模拟闭环验证的最小版本；Policy Agent 保持可插拔，后续可替换不同 LLM。工程上不引入逐样本记录，默认只使用 JSON/YAML、JSONL 与 PyTorch checkpoint。

> 本文是新仓库的工程实施规范，不是已经完成的代码或新的实验结果。下文 `cloop` 命令均为需要实现的命令。现有仓库仅作为数据、实验协议和接口参考，不作为新项目必须导入的运行时依赖。服务器路径来自仓库记录，未直接访问服务器，必须通过 `doctor` 验证。

> **v1.1 变更：** 删除 v1.0 中的数据库持久化设计。所有实验产物改为单层 `run.json / models.pt / last.pt / metrics.json / predictions.jsonl / report.md`，其中 `last.pt` 与 `predictions.jsonl` 均可按阶段省略。

---

## 1. 希望实现的框架

### 1.1 研究主线

最终研究问题是：**能否利用 RRT 训练的独立 Ensemble World Model，在获得新患者观测后重新规划下一阶段治疗，而不一直沿用初始状态下的计划？**

三个模块的职责保持清楚：

- **Recursive Rollout Training（RRT）：** 缓解真实状态训练与预测状态递归输入之间的不匹配。
- **Ensemble Dynamics：** 为同一候选动作序列输出多条独立预测轨迹，提供模型分歧信号。
- **闭环规划器：** 短期向前模拟、比较候选、仅提交当前动作；收到新观测后重新编码和规划。

```text
当前可见信息：MRI latent + 截至当前的临床信息 + 已执行治疗历史
                                  |
                                  v
                         PatientState / StateBuilder
                                  |
                                  v
                     可插拔 PolicyAgent.propose()
                 默认：训练集动作目录；将来：LLM 提议器
                                  |
                        结构校验 / 支持度检查
                                  |
                                  v
                    RRT-trained Ensemble World Model
                   同一动作分支、各 member 独立递归
                                  |
                       +----------+----------+
                       |                     |
                       v                     v
                  OutcomeEvaluator       Disagreement
                       |                     |
                       +----------+----------+
                                  |
                                  v
                    Receding-Horizon Planner（H≤3）
                    排序、必要时拒绝、只输出当前一步
                                  |
                    外部执行后获得真实的新观测
                                  |
                                  +----> 重新建状态，再规划
```

**重要定义：** MPC／滚动时域规划可以在一次规划中搜索完整的短期动作序列，但只执行第一步。它不要求第一版就实现“每个 imagined state 都让 LLM 生成一棵条件策略树”。新观测到来后重新规划，已经构成观测反馈闭环。复杂的分支自适应策略树留作后续扩展，避免第一版同时引入过多变量。

### 1.2 本阶段实现范围

| 现在实现 | 暂不实现 |
|---|---|
| 冻结 BrainIAC latent 的读取与标准化 | 新一轮 MRI-CORE LoRA 训练 |
| One-step / RRT / Ensemble / RRT+Ensemble | Survival、Dynamics、Encoder 的联合端到端微调 |
| 轻量、独立训练的生存／结局评估头 | 直接复刻 CLARITY 全部多模态 Transformer |
| 离线动作目录 Agent；LLM 接口和 mock 测试 | 默认调用在线 LLM、默认上传患者信息 |
| 贪心、固定短期计划、MPC、带分歧惩罚 MPC | 在线强化学习、真实患者自动执行治疗 |
| 真实数据事实预测与逐观测回放 | 用回顾性数据虚构推荐治疗的真实生存收益 |
| 独立、动作可影响状态的合成环境闭环实验 | 将模型自身 rollout 当独立环境证明自己有效 |

**决策粒度：** 第一版以一次 MRI 观测及其后一个计划区间为一个阶段，动作是粗粒度治疗集合，不是逐日处方或剂量调整。数据区间中多次更换治疗被聚合时，应标记为复合暴露；不能声称恢复了区间内每一次真实临床决策。

**第一版保留完整模块链路，但不声称已经复现或超过完整 CLARITY。** 只有在相同数据、观测权限、任务定义及评价协议下运行官方或经验证的 CLARITY baseline，才能作出正式方法优劣比较。

### 1.3 必须区分的三类验证

1. **事实预测：** 给定历史实际治疗和实际间隔，预测真实未来 MRI latent／结局。可评价预测误差。
2. **逐观测回放：** 每次只向系统暴露截至当前的信息，再与记录中的后续治疗作描述性比较。历史下一张 MRI 是历史真实治疗下的结果，不是新推荐的结果。
3. **独立合成环境：** 推荐动作真正进入独立环境，环境返回下一状态与代价。可测试反馈、重规划和误差恢复机制，但不能外推为临床疗效。

第一版必须分别输出三类结果，不能把它们混成一个“闭环治疗有效性”指标。

---

## 2. 仓库目录与组件

### 2.1 建议目录：扁平 Python 包 + 两个配置文件

```text
CLARITY_Loop/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── default.yaml           # 方法、训练、评价、Policy 默认配置
│   └── server.yaml            # 服务器路径；不包含密钥
├── src/cloop/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py              # 严格配置加载、覆盖与签名
│   ├── types.py               # PatientState / Action / Rollout / Decision
│   ├── data.py                # 数据导入、时间对齐、split、window、codec
│   ├── world.py               # OneStepDynamics、RRT rollout、Ensemble
│   ├── outcome.py             # 生存/结局评估头
│   ├── policy.py              # CatalogPolicy、LLMPolicy、provider 协议
│   ├── planner.py             # 约束、beam search、MPC、评分
│   ├── synthetic.py           # 独立可控闭环环境
│   ├── metrics.py             # Dynamics / reliability / survival / policy 指标
│   ├── artifacts.py           # JSON/JSONL/PT 原子读写；不使用逐样本记录
│   ├── engine.py              # train/evaluate/replay/synthetic 调度
│   └── cli.py                 # 唯一命令入口
├── tests/
│   ├── test_data.py
│   ├── test_world.py
│   ├── test_outcome.py
│   ├── test_policy_planner.py
│   └── test_pipeline.py
└── outputs/                   # 不提交 Git
    └── brainiac_v1/
        ├── run.json           # resolved config、split、provenance、状态与审计
        ├── models.pt          # 已完成模型的 best 权重与预处理器
        ├── last.pt            # 唯一恢复点；完整结束后可删除
        ├── metrics.json       # 训练摘要 + 全部汇总指标
        ├── predictions.jsonl  # 可选：逐窗口预测/逐步决策；一个文件
        └── report.md          # 唯一人工阅读报告
```

大型输入继续保留在已有 `/data/...` 路径；新项目只建立一个外部 `cache_root`，默认每个数据协议保存一个紧凑 `.pt` 缓存。不要复制 MRI，不要复制旧仓库的 `outputs`。

### 2.2 为什么不用逐样本记录

当前数据量和实验规模不需要 本地记录文件/MySQL/PostgreSQL。持久化只使用：

- YAML：人工维护配置；
- `run.json`：配置、provenance、split、数据审计、阶段状态；
- `models.pt` / `last.pt`：模型与恢复状态；
- `metrics.json`：聚合指标和紧凑训练历史；
- `predictions.jsonl`：需要逐样本分析时保存预测与决策；
- `report.md`：最终摘要。

所有写入采用 `tmp -> fsync/close -> rename` 的原子替换。`predictions.jsonl` 数据量很小，可按 `record_id` 读入内存、去重后整体重写；不需要逐样本记录索引。正式 run 完成后若不再需要恢复，删除 `last.pt`，最终通常只保留 5 个文件。

### 2.3 目录与产物约束

- 不按 `seed/member/epoch/horizon/patient/timepoint` 创建子目录。
- 不建立 `logs/`、`reports/`、`checkpoints/`、`tmp_outputs/` 多层树。
- 不每个 epoch 保存 checkpoint；只保留 `models.pt` 中的 best 和一个 `last.pt`。
- epoch 训练曲线只在 `metrics.json.training` 中保存必要数值；默认终端每 epoch 一行，不额外生成 `.log`。
- 逐窗口或逐决策结果统一进入一个 `predictions.jsonl`；若当前 suite 不需要逐样本分析，可关闭该文件。
- 不保存每个 branch 的 latent tensor、每个 candidate 的临时 checkpoint 或每个 patient 的单独文件。
- 调试导出必须由显式 `--debug-export` 开启，并写到操作员指定的临时位置，不进入默认正式 run。

### 2.4 模块依赖方向

```text
config / types / artifacts
          |
          v
 data     world     outcome     policy     synthetic
    \       |          |          |           /
     \      +----------+----------+          /
      \                |                    /
       +----------> planner <--------------+
                       |
                       v
                    engine
                       |
                       v
                      cli
```

`world.py` 不导入 `policy.py`；`policy.py` 不控制训练器；`planner.py` 不直接读取完整患者未来时间线；`engine.py` 是唯一组织数据、训练和环境交互的层。`artifacts.py` 只负责文件格式和原子写入，不包含实验逻辑。

---

## 3. 当前阶段的比较实验与指标

### 3.1 实验 A：重建 RRT + Ensemble 的 Dynamics 证据

| ID | 方法 | 训练 horizon | member 数 | 主要作用 |
|---|---|---:|---:|---|
| D0 | Persistence：始终预测当前 latent | 不训练 | 0 | 排除“只保持状态不变就足够” |
| D1 | Baseline | 1 | 1 | 一步训练对照 |
| D2 | RRT | 最大可用，封顶 3 | 1 | 验证递归训练收益 |
| D3 | Ensemble | 1 | 5 | 验证集成及分歧信号 |
| D4 | RRT + Ensemble | 最大可用，封顶 3 | 5 | 拟用于主框架的 dynamics |

**指标：** normalized latent MSE@1/2/3、Long MSE、同 seed 配对 RI、patient-macro MSE；辅助报告 Cos@1/2/3。D3/D4 额外报告逐 horizon Spearman、高低分歧三分位误差比、Risk@100、Risk@80、实际保留覆盖率。

主 split seed=17，训练 seeds=[7,17,29]。首轮开发先用 seed=17 跑通，再扩为完整 12 个 variant-seed 组合。[S1–S4]

**解释边界：** D1/D2 是新仓库内部的公平对照，不等于完整 CLARITY。RRT 不仅改变递归输入，还改变训练终点的 horizon 分布；不能将该比较写成“只改变输入而其他监督完全相同”。

### 3.2 实验 B：把预测 latent 接回结局任务

| ID | 输入或评价路径 | 目的 |
|---|---|---|
| O0 | 仅临床信息与历史摘要 → OutcomeHead | 测试是否根本不需要 MRI |
| O1 | 真实当前 latent + 同样上下文 → OutcomeHead | 校验结局头本身是否有预测能力 |
| O2-D1…D4 | 各 dynamics 的预测未来 latent → **同一个冻结的 O1** | 测试 dynamics 误差对结局预测的影响 |

**指标：** C-index、365 天 IPCW Brier、分段风险率生存 NLL；样本数、事件数、删失数、不可估计原因必须一起报告。O2 按预测 horizon 分开计算，标签的生存起点是目标 MRI 的时间点，而不是最初 MRI。

本阶段 O1 与 world model 分开训练，避免所有模型同时变化而无法归因。O2 是“已随访到目标 MRI 的患者中的未来状态结局预测”，不是无条件的整体生存预测。

### 3.3 实验 C：离线 Agent 下的规划器对照

| ID | World Model | 决策方式 | 分歧惩罚 | 主要对比 |
|---|---|---|---:|---|
| P0 | 无 | 训练集频率最高的有效动作 | 0 | 无模型基线 |
| P1 | D4 | 初始观测下生成短期固定计划 | 0 | 固定计划参考；不标为官方 CLARITY |
| P2 | D4 | 每次新观测后，一步贪心 | 0 | 重观测但不做多步规划 |
| P3 | D3 | 每次新观测后，H=3 MPC | 0 | 无 RRT 的 Ensemble MPC |
| P4 | D4 | 每次新观测后，H=3 MPC | 0 | 检验 RRT 对 MPC 的增量 |
| P5 | D4 | 每次新观测后，H=3 MPC | 验证集固定的 λU | 检验分歧惩罚的增量 |

第一轮六组均使用同一个 `CatalogPolicy`、同一训练集动作目录、同一结局头、同一有效动作过滤规则。P3/P4/P5 保持相同搜索预算。P1/P2 的信息或计算预算差异必须单独报告。

**真实数据回放指标：** 下一阶段已记录治疗的 action-set precision/recall/F1/Jaccard、目录覆盖率、候选召回率、拒绝率、结构规则违规率、重规划改动作比例、耗时和 world-model 调用数。它们衡量一致性及系统行为，**不是治疗优劣的因果证据**。

**独立合成环境指标：** 真实环境累计代价／回报、与已知环境最优策略的期望价值差、约束违规率、扰动后恢复代价、拒绝率、推理预算。允许讨论模拟闭环机制，不能称为真实患者生存获益。

### 3.4 现阶段不扩大的实验

不默认增加 MRI-CORE LoRA、第三个 encoder、K=5、十几个 LLM、大规模超参数搜索或全组合消融。已有 MRI-CORE 结果保留为背景材料；先完成 BrainIAC 上整条链路。

开发验收只要求数据与计算正确、没有信息泄漏、能产生有效结果。**不设置“RI 必须大于某值”之类自动成功阈值。**

---

## 4. 参考基线与设计边界

### 4.1 本次核对的源版本

- MyMeWM：`b5e3b979e46e6ec254d72aac46fbc352b311bed6`。
- CLARITY：`dadb82241a24f5ec5e4e4dc994e3116fd4a9da04`。

本文中，已有路径、字段和旧实验事实标注来源；新模块、损失选择、接口和执行顺序均为本次**拟实现设计**。

### 4.2 不能误写为创新的已有能力

CLARITY 公开代码已经有候选生成与世界模型评分分离、无密钥时的离线规则生成，以及 MC-dropout 风险分歧和 `w_unc` 评分惩罚。[S8–S9]

因此差异应表述为：

> 在明确的患者观测反馈协议下，研究 RRT、独立 member 递归 Ensemble，以及分歧感知重规划的组合与增量效果。

不要写成“CLARITY 完全没有反馈／不确定性／离线 Agent”。未来正式 baseline 至少需要加入 **每个真实观测都重新运行的 CLARITY**，不能只和故意不更新观测的弱固定计划比较。

### 4.3 现有 BrainIAC 结果只作回归参考

旧仓库 Long MSE 均值：Baseline 0.9721、RRT 0.8616、Ensemble 0.9182、RRT+Ensemble 0.8533。RRT+Ensemble 的 reliability 并不全面优于 Ensemble-only，尤其 H3 高低分歧误差比约为 0.974。[S4]

新仓库不把这些数值写成测试断言。时间对齐、词表或上下文协议变化后，需要重新训练，报告新结果；旧数值不能直接拼到新主实验表里。

---

## 5. 数据与服务器路径

### 5.1 已记录的路径与拟新建位置

| 用途 | 路径 | 状态 |
|---|---|---|
| Python | `/home/tanyuejun/miniconda3/envs/py310/bin/python` | 旧运行记录 [S5] |
| 旧 MyMeWM 项目 | `/home/tanyuejun/CLARITY_HAUWM_Minimal` | 旧记录 [S1] |
| CLARITY 项目 | `/home/tanyuejun/CLARITY` | 旧记录 [S1] |
| 主要 clinical timeline | `/data/tanyuejun/CLARITY/clinical/MU_Glioma_Post/clinical_latest.json` | 实际产物 provenance [S6] |
| timeline 另一记录位置 | `/home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json` | README [S1]；不能假定与上一个内容相同 |
| MRI NIfTI | `/data/tanyuejun/CLARITY/dataset/MU-Glioma-Post` | provenance [S6] |
| Frozen BrainIAC latent | `/data/tanyuejun/CLARITY_HAUWM_Minimal/latents/brainiac` | README [S1] |
| 已构造 BrainIAC trajectories | `/home/tanyuejun/CLARITY_HAUWM_Minimal/data/trajectories/brainiac` | README 的项目根目录与相对路径 [S1] |
| BrainIAC 权重 | `/home/tanyuejun/CLARITY/BrainIAC-main/src/checkpoints/BrainIAC.ckpt` | README [S1]；第一版训练不必加载 |
| 新项目根目录 | `/home/tanyuejun/CLARITY_Loop` | **拟新建**；不要覆盖旧项目 |
| 新缓存根目录 | `/data/tanyuejun/CLARITY_Loop/cache` | **拟新建** |

首选读取现有冻结 latent；其来源与 encoder 冻结状态必须通过 metadata 检查。**从空白仓库搭建代码，不意味着必须重新提取所有 MRI。**

`doctor` 对配置中的每个必需路径执行 exists/readable 检查；对两个 timeline 路径比较文件哈希。两者不同则要求显式选择，不能静默切换。新增目录与源目录相同、互为可写覆盖目标时直接报错。

### 5.2 新缓存只保留一份必要内容

建议 `brainiac_main.pt` 内容：

```python
{
    "schema_version": "cloop_data_v1",
    "source": {"timeline_sha256": "...", "latent_manifest_sha256": "..."},
    "encoder": {"name": "brainiac", "frozen": True, "latent_dim": 768},
    "protocol": "main_v1",
    "patients": [
        {
            "patient_id": "...",
            "timepoint_ids": ["TP1", "TP2", "..."],
            "mri_days": tensor_T,
            "latents_raw": tensor_T_D,
            "clinical_raw": {"...": "..."},
            "events": ["标准化的治疗事件；保留时间与来源标志"],
            "labels": {"survival_time": tensor_T, "event": tensor_T, "valid": tensor_T},
            "quality_flags": {"...": "..."},
        }
    ],
    "audit": {"excluded_counts": {}, "warnings": []},
}
```

只保存张量和内置 Python 类型，便于 `torch.load(..., weights_only=True)`。不要在缓存中 pickle 自定义类。缓存不包含从全体患者拟合的 normalizer 或动作频次目录；它们必须在每次患者划分后仅使用 train 拟合。

### 5.3 数据契约：事实标签不能进入 Policy

`types.py` 至少定义以下概念：

```python
@dataclass(frozen=True)
class PatientState:
    patient_key: str             # 内部研究标识；发往外部 LLM 时移除
    timepoint_key: str
    observed_day: float
    z: Tensor                   # [D]，处于明确版本的 normalized latent space
    clinical: Tensor            # [C]，截至当前可见的上下文
    clinical_mask: Tensor       # [C]，缺失不是正常值
    history: Tensor             # [Hc]，只包含已经执行的治疗历史
    source: Literal["observed", "imagined"]
    state_version: int

@dataclass(frozen=True)
class FactualTarget:
    next_z: Tensor
    actual_action: "Action"
    actual_delta_days: float
    survival_time: float | None
    event: int | None
```

接口实现时可调整字段类型，但必须保持两个对象分离。`PolicyAgent`、`Planner` 只能接收 `PatientState/PolicyRequest`，不能拿到 `FactualTarget` 或完整 `patient["timeline"]`。

---

## 6. 数据准备：先审计时间语义，再构建窗口

### 6.1 时间与治疗对齐是实施前置条件

旧 MyMeWM 将动作按 `source` 节点聚合；但 CLARITY 的 `extract_clinicial.py` 中，部分 interval therapy 被写入区间 `(上一 MRI 日, 当前 MRI 日]` 对应的**当前节点**，并带有 `interval_start_day/interval_end_day`。[S1, S7]

这意味着不能只根据字段位于 source 还是 destination，就断言它属于哪一段治疗。也不能反过来宣称服务器现有数据一定错误：服务器实际 JSON 是否经过同样处理，需要读取文件核查。

**实现两个明确协议，不静默修补：**

| 协议 | 用途 | 动作处理 |
|---|---|---|
| `legacy_stage1` | 重建旧 RRT/Ensemble 结果 | 读取旧 trajectory 的 actions/delta_days，保留原 source 约定与旧词表，记录 legacy 标志 |
| `main_v1` | 新框架正式实验 | 优先用明确事件时间重建实际区间暴露；不依赖节点摆放位置猜测 |

`main_v1` 对实际区间 `(d_t,d_{t+1}]` 构建动作：

1. 收集该患者各节点保存的治疗事件；保留原始时间、类别、agent 和 provenance。
2. 对同一事件的重复或跨窗口片段去重／合并；没有 event_id 时使用规范化药物、类别、原始 start/end 和其他可用属性构造指纹，不把两次独立治疗错误合并。
3. 以明确区间与当前 MRI 区间的交集确定暴露；点事件按 `(d_t,d_{t+1}]` 归属。
4. 缺失时间、`duration_unknown`、`assigned_reason` 只表示不确定归属，不表示整段一直用药。
5. 无法确定的 interval 不用于 action-conditioned 主指标；仍可保留在状态编码或符合条件的结局训练中，分别统计覆盖率。

**连续性规则：** 某个相邻区间因动作不明被排除后，RRT 窗口不能跨过这个断点；不得把两段不连续轨迹直接拼起来。缺少中间 MRI 时，只能在完整事件记录足以重建合并区间暴露、且真实起止日明确的条件下建立较长单个转移，并记录该事实。患者 ID 与 split 始终不变。

**特别区分：** 在事实 dynamics 训练中，将“后来被记录、但实际发生于两个 MRI 之间”的治疗作为监督输入，并不等同于把未来状态泄漏给模型。它是该区间的事实 action。可是部署／回放的 Policy 在区间开始时不能偷看这份实际 action，它只能提出候选。

### 6.2 不允许新增不可追溯的时间插补

时间排序使用 `mri_day`，TP 编号只用于标识。缺失、相同或逆序时间都要审计。旧临床脚本存在插补及截断逻辑；若 JSON 只保存汇总插补数而没有逐 timepoint 标志，不得称所有间隔都是真实未插补测量。[S7]

新代码不再自行补“100 天”。主协议记录 `time_quality=verified/legacy_imputed/unknown`；必要时做高质量子集敏感性分析。没有足够信息时报告限制，而不是根据 TP1/TP2 猜天数。

### 6.3 临床与生存字段

已在 CLARITY 数据代码中确认的路径包括：[S7, S10]

```text
patients[pid].context_static
patients[pid].timeline[i].tp_id
patients[pid].timeline[i].mri_day
patients[pid].timeline[i].actions
patients[pid].timeline[i].state.progression
patients[pid].timeline[i].survival.survival_from_tp_days
patients[pid].timeline[i].survival.event_indicator
patients[pid].timeline[i].survival.censoring_rule
```

第一版不需要读取原始 Excel。若上述字段缺失，`doctor/prepare` 输出具体缺失字段，不生成伪造 survival label。

临床字段使用白名单：例如年龄、性别、WHO grade、IDH/MGMT 等已有结构化变量。基因组结果被视为基线已知属于需要声明的假设；有检测日期时必须执行可见性过滤。未知值单独编码，不用 0 代替。

不要直接信任 `occurred_up_to_tp` 就等于当时可知：旧处理脚本在部分缺失 progression time 情形使用患者级 progression 标志。第一版默认不使用这类未经时间审计的进展状态作为输入；只有可靠发生日期≤当前观测日时才开放。[S7]

### 6.4 生存标签质量规则

- `T<=0`、不合法 event、缺失标签：排除该 survival 样本，但不连带删掉可用 dynamics 样本。
- `death_shifted_to_L_plus_1`：属于旧处理脚本的人为修正规则，主生存评价默认排除，单独计数。
- `no_death_last_mri`：属于代理删失终点；可以在明确披露条件下使用，但不能声称是真正最后临床随访时间。
- 同一患者各节点的 `mri_day + survival_from_tp_days` 应在同一死亡／删失规则下基本一致；不一致时隔离并审计。
- survival 的终点与 event 字段只进入 label 对象，不进入 state、action history、prompt 或检索索引。

以上规则来自对旧代码行为的核查，不表示已核验服务器 JSON 中各种情况的实际数量。[S7]

### 6.5 患者划分与标准化

优先保留旧 Stage 1 的多时间点患者集合，使用 SHA256 排序：

```python
ranked = sorted(patient_ids, key=lambda p: sha256(f"{split_seed}:{p}".encode()).hexdigest())
```

比例 70%/15%/15%，小样本保底规则参照旧实现。[S2]

划分在 survival-label 筛选之前固定。某个患者缺失 survival 标签，只减少该任务覆盖，不重新分配患者。不要因为选择 epoch、encoder、动作过滤或 outcome 模型而重抽 test。

只用 train patients 拟合：latent 均值方差、临床标准化、动作词表、动作目录频次、历史摘要尺度。latent 标准差过小的维度记录数量与处理策略，不能在 validation/test 重新拟合。

`legacy_stage1` 为对照可保留旧 metadata 词表，但要标记“旧全数据词表约定”；`main_v1` 使用 train-only 词表和 UNK。新主实验的 action_dim 由数据决定，不硬编码成 28。

### 6.6 旧统计用于核对，不用于强制改数据

旧 BrainIAC 记录为 train/validation/test=107/23/24 位患者；H1 窗口 262/60/70，H2 为155/37/46，H3 为79/21/28。[S13]

若 `legacy_stage1` 导入不能重建这些数量，要解释差异再训练。`main_v1` 的严格时间对齐可能减少样本，允许数量变化，但必须记录每一步过滤及变化原因。

## 7. World Model 与 RRT 训练

### 7.1 单步模型：先保留已验证的小架构

`world.py` 定义一个单步函数，所有 variant 共用：

\[
\hat z_{t+1}^{(m)}=z_t+F_{\theta_m}(z_t,e_a(a_t),e_\Delta(\Delta t),e_c(c_t),e_h(h_t)).
\]

`legacy_stage1` 关闭 clinical/history 输入，保留旧 action projection、time MLP、单 token GRU、residual MLP 的尺寸及初始化。[S3]

`main_v1` 加入轻量 clinical/history projection；四个 variant 同时打开，不能只给最终模型增加上下文。建议：hidden_dim=128、action_embed_dim=32、time_embed_dim=16、clinical/history projection 各16维。

时间编码沿用：

\[
 e_\Delta=\mathrm{MLP}\left(\frac{\log(1+\Delta t)}{\log(1+365)}\right).
\]

真实输入必须是正时间间隔。padding 的 0 仅通过有效步 mask 处理，不能把时间异常自动 clamp 后继续训练。

历史摘要首版不做复杂序列编码器，可使用：上一步已执行动作 multi-hot、累计已完成区间的动作暴露摘要、已完成区间数、截至当前的累计时间。更新函数唯一且无参数：

\[
h_{t+1}=\mathrm{update\_history}(h_t,a_t,\Delta t).
\]

历史区间暴露是数据代理量，不解释为精确累计药物剂量。不要用 `num_cycles` 等未来完整疗程数字作为已经完成的暴露量。

### 7.2 Batch 与 rollout 契约

```python
# Batch
z0:       Tensor[B, D]
actions:  Tensor[B, K, A]
deltas:   Tensor[B, K]
context:  Tensor[B, C]
history0: Tensor[B, Hc]
target:   Tensor[B, D]           # 对应每个样本自身终点
horizons: LongTensor[B]

# WorldModel.rollout(...) -> Rollout
states:    Tensor[M, B, K + 1, D]
step_mask: BoolTensor[B, K]
```

所有 member 从同一个观测 latent 开始，但下一步保持自己的状态。动作和时间序列在同一分支上对所有 member 一致。

```python
states = z0.unsqueeze(0).repeat(M, 1, 1)
for j in range(K):
    next_states = stack([
        member(states[m], actions[:, j], deltas[:, j], context, history)
        for m, member in enumerate(members)
    ])
    states = where(active_at_step(j), next_states, states)
    history = masked_update(history, actions[:, j], deltas[:, j])
    save_in_memory(states)
```

这是实现约定的伪代码，不是需逐行照抄的完整模块。

**禁止：** 在递归中 detach 预测 latent；把 ensemble mean 回灌给全部成员；拿真实下一 MRI 替换训练中的预测中间状态；为了生成终点 target 而把未来 MRI 暴露给 Agent。

### 7.3 训练窗口与 loss

每个有后续观测的起点生成一个训练样本。

\[
k_t=\begin{cases}
1,&\text{Baseline / Ensemble}\\
\min(3,T-1-t),&\text{RRT / RRT+Ensemble}
\end{cases}
\]

递归使用该事实窗口内的真实 actions 与 delta_days；主损失保持 terminal-only：

\[
\mathcal L_m=\frac1{Bd}\sum_i\|\hat z_{i,t+k_i}^{(m)}-z_{i,t+k_i}\|_2^2.
\]

RRT 不在第一版新增 intermediate loss、random horizon 或 curriculum。context 使用起点已经可见的信息；除时间和治疗历史的确定性更新外，不在 imagined 中途注入真实的未来 progression／toxicity。

若严格数据审计后 validation 没有 H2 或 H3 窗口，完整 suite 应明确停止并报告，不允许仅对某个 variant 改用 H1 选 checkpoint。需要缩短 horizon 时，建立新的共同协议后对所有方法重跑。

训练参数继承旧协议：AdamW、lr=1e-3、weight_decay=1e-4、batch=32、epochs≤100、grad_clip=1、patience=15。所有 variant 以 validation H2/H3 均值选 checkpoint。[S1]

每个 ensemble member 独立初始化、独立 batch shuffle、独立选择 best epoch。为了对齐旧实验，可沿用 `member_seed=training_seed*1000+member_index`；它与单模型 seed 不构成同一个初始权重。报告时以 variant 内相同 seed 定义配对，不把 member 当独立临床样本。

### 7.4 训练成本与恢复

冻结 latent 下的模型无需加载 MRI encoder，不必运行多卡 DDP。默认一张空闲 GPU 顺序训练所有 member；CPU 可用于 smoke test。Ensemble 的训练／预测成本约随 member 数增加，报告实际时间与参数量。

`last.pt` 必须包含当前 variant、seed、member、epoch、模型、optimizer、early-stopping best/stale、训练数据生成器与 Python/NumPy/Torch RNG 状态、配置和数据签名。第一版只保证**完整 epoch 边界**恢复；中断于 epoch 内，回滚到上一完整 epoch。恢复不是从 best 权重重新创建 optimizer。[S12]

---

## 8. OutcomeEvaluator：先做可解释、可验证的结局接口

### 8.1 为什么第一版单独训练

此模块把 latent state 接回 CLARITY 场景中的生存预测任务，但 MVP 不直接复制其完整两路 cross-attention Actor。CLARITY 公开实现包含 Cox/BCE 结局预测；这里保留“从状态估计结局”的角色，选择更小、能处理删失的生存头，是**新设计选择，不是官方复现**。[S8]

训练流程：

```text
真实 observed states -> 训练 OutcomeHead -> 验证集选模型 -> 冻结
                                                |
D1/D2/D3/D4 预测的 imagined states ----------------+
```

同一 seed 的所有 planner/world variant 共享同一个结局头。这样不把“结局头变强”误算为 RRT 或规划器的收益。第一版不将 outcome loss 反传给 encoder 或 dynamics。

### 8.2 输入与生存起点

\[
G_\psi(z_t,c_t,h_t)\rightarrow \hat S(\tau\mid s_t).
\]

其中 \(\tau\) 从**输入状态对应的时间点**开始计时。真实状态取同一节点的 `survival_from_tp_days/event_indicator`。预测到 \(t+k\) 后的状态评价，必须配对目标节点的 residual survival 标签。

`PatientState` 不包含 survival_time；label 是数据加载器单独返回的对象。

### 8.3 建议最小生存头：分段常数风险率

使用边界 `[0,90,180,365,730]` 天；若训练数据不支持某评价时点，仍可训练模型，但该时点的正式评价输出 unavailable，不改成另一含义的指标。

网络：`concat(z,clinical,mask,history) -> LayerNorm -> Linear(128) -> SiLU -> Linear(J)`。输出每段非负风险率：

\[
\lambda_j=\mathrm{softplus}(u_j)/365.
\]

设 \(\ell_j(\tau)\) 是时间 \(\tau\) 在第 \(j\) 段内经历的时长，则：

\[
\hat S(\tau\mid s)=\exp\left[-\sum_j\lambda_j(s)\ell_j(\tau)\right].
\]

对观测生存时间 \(T\) 和 event \(E\)：

\[
\mathcal L_{surv}(s,T,E)=\sum_j\lambda_j(s)\ell_j(T)-E\log\lambda_{bin(T)}(s).
\]

超过730天的样本在730天行政删失；事件恰在边界的 bin 归属约定为右闭区间 `(b[j-1],b[j]]`，单元测试固定。删失在区间中间也保留已经观察到的时长，不把删失前未发生死亡误当为之后存活标签。

训练可使用所有合格 observed timepoints，但对同一患者按有效 landmark 数倒数加权，避免长随访患者占主导。O0/O1 使用同一标签、权重、训练预算及划分。

推荐 optimizer=AdamW、lr=1e-3、weight_decay=1e-4；最多100 epoch，patience=15，以 validation survival NLL 选模型。此处参数是 MVP 固定默认值，不进行 test-driven tuning。

### 8.4 指标与用途

- `risk365 = 1-S(365)` 用于 C-index 排序，明确报告该 risk 定义及730天截断规则。
- IPCW Brier@365 使用生存概率 `S(365)`，不是 Cox risk 或未经校准的任意 score。
- censoring 分布只从 train 标签估计；没有支持、无可比较对或 denominator=0 时写 `null + reason`。[S11]
- 首个合格 landmark／每位患者一条作为主要生存报告；所有 landmark 的结果作为补充，并报告患者数和重复观测数。
- O2 另外按 H1/H2/H3 各自目标 landmark 计算；不得把同一患者的多次 MRI 当独立患者扩大样本量。

### 8.5 规划分数不是已验证的治疗效果

MVP 取 imagined state 的 `1-S(365)` 作为预后代理分数。它是在观察性历史数据上拟合的条件关联，不是治疗干预的因果效应；当候选动作不同于历史行为时存在外推风险。

第一版 latent dynamics 没有显式模拟两次 MRI 之间的死亡、退出随访或未观测毒性；这也是本阶段只能使用预后代理代价、不能宣称治疗策略提高真实生存的原因。

本阶段不要把不同 imagined visits 的滚动一年死亡概率相加后称为“总死亡概率”，不要将任意 Cox 风险分数相加后称为“累计死亡风险”。第10节给出的规划分数只是归一化加权的**启发式状态代价**。

若结局字段审计未通过，真实数据规划的 outcome 评价必须停止并给出原因；Dynamics 实验与合成环境仍可独立运行。不能用 latent 范数或随机 risk 自动替代真实 clinical outcome。

---

## 9. Policy Agent 的可插拔设计

### 9.1 Agent 只提出候选，不拥有决策权

`PolicyAgent` 的职责是返回当前允许尝试的候选动作及可选理由。最终是否可用、如何 rollout、如何评分、是否拒绝和最终选哪一个，都由独立模块完成。

接口不直接绑定某个 LLM SDK、prompt、在线服务或 world model。

```python
class PolicyAgent(Protocol):
    def propose(self, request: PolicyRequest) -> CandidateBatch: ...

class LLMProvider(Protocol):
    def generate_json(
        self, *, model: str, messages: list[dict], schema: dict,
        temperature: float, timeout_s: float
    ) -> dict: ...
```

`PolicyRequest` 至少包含：观测截止时间、经过白名单过滤的临床摘要、已执行治疗摘要、允许候选目录、最大候选数、请求 seed、observed/imagined 标志。不能包含实际下一治疗、真实未来 latent、event、生存时间或 future clinical text。

`CandidateBatch` 至少包含：候选 ID、提议来源、去重/解析失败数量、是否 fallback、耗时、调用次数。先返回结构化 ID，再映射为内部 Action。

### 9.2 Action schema：先做离散治疗集合，不做剂量优化

```python
@dataclass(frozen=True)
class Action:
    action_id: str
    token_ids: tuple[int, ...]
    display_terms: tuple[str, ...]
    support_count: int
    known_empty: bool = False
```

MVP 的模型只编码训练集支持的治疗类别／药物集合。它不区分未建模的剂量、cycles 或开始偏移，因此 Agent 也不能生成仅剂量不同的候选并要求 world model 给出不同评分。

**空动作≠未知动作≠拒绝决策。**

- `known_empty`：数据或人工定义明确表示无 active intervention 的研究动作；不能把它称为普遍安全。
- `UNKNOWN`：动作信息缺失，不进入候选目录。
- `ABSTAIN`：系统不具备决策条件，不是治疗动作，不能自动解释成不治疗。

### 9.3 默认 CatalogPolicy：完全离线可运行

只从 train 患者、时间归属明确的真实动作集合建立目录。规范化 alias、去重，按出现次数稳定排序，默认取前6个支持度足够的动作。相同请求应返回相同候选顺序。

默认不会人为添加未经数据支持的药物组合。训练目录中没有“明确空动作”时，不自动加入它。

第一版的候选生成可以不根据 imagined latent 改变；**world model 的分支评分和重观测后的排序变化**已经能验证状态反馈。这样避免把“候选改变”与“world model／planner 改善”混在一起。

不得为了让模型获得更高 action-match 指标，强行把每个 test visit 的真实下一动作塞进候选集。真实动作不在目录时，记录 OOV／候选覆盖不足，保留在无条件评价分母中。

### 9.4 可选 LLMPolicy

后续在 `policy.py` 中增加：

```text
LLMPolicy
  -> build_sanitized_messages(PolicyRequest)
  -> LLMProvider.generate_json(...)
  -> parse/validate candidate IDs
  -> ActionCodec.decode(...)
  -> CandidateBatch
```

初版 LLM 只允许从同一个目录选择／排序 ID，不开放自由新增治疗。更换同一 provider 的模型只需修改 `model`；更换 provider 时实现一个 transport adapter 并运行接口测试，不改 world model、outcome 或 planner。

可保留 `provider="http_compatible"`、`provider="local"` 等注册入口，但首版只强制实现 CatalogPolicy 与 FakeProvider。真正 API adapter 在接入时按该厂商当时的正式文档实现，不能假设不同厂商的 JSON schema、工具调用和认证接口天然兼容。

### 9.5 离线约束与 fallback

- 默认 `allow_network=false`；即使环境中存在 API key，也不自动联网。
- `kind=llm` 但不允许联网时明确报错，或按显式配置 fallback 到 catalog，并记录来源。
- 无法解析、超时、目录外 ID：最多一次受控重试，之后显式 fallback／ABSTAIN。
- 不把非法模型输出当成空动作；不 `eval/exec` 模型文本。
- API key 只读环境变量，不进入 YAML、run.json、JSONL 或 checkpoint。
- 外部 LLM payload 不含姓名、原始病历、完整 MRI 或内部患者 ID；发送临床字段仍需符合数据使用许可。
- 默认保存请求哈希、模型/provider、返回的候选 ID及调用成本，不保存原始敏感 prompt。

### 9.6 不让 LLM“解释”无语义 latent 坐标

不得把768维向量交给 LLM后声称它看到了影像进展。imagined 节点只能提供已定义的数值摘要、模型分歧、结局代理、已知静态上下文和模拟治疗历史；需要肿瘤大小／进展文字时，应另行训练并验证 observation decoder。第一版没有这个 decoder。

### 9.7 Agent 插拔验收

同一个合法 `CandidateBatch` 无论来自 CatalogPolicy、FakeProvider 还是将来的 LLM，后续过滤、rollout 和评分结果必须一致。更换 Agent 不重训 world model，前提是 ActionCodec 和动作目录不变；扩展动作语义则是新模型协议，而不是简单换 LLM。

---

## 10. 规划器：先实现标准 MPC，再做更复杂的 imagined feedback

### 10.1 状态更新与动作执行边界

`Planner.plan(state) -> Decision` 是纯计算接口，不执行任何真实治疗。

```python
@dataclass(frozen=True)
class Decision:
    recommended_action: Action | None
    status: Literal["recommend", "abstain"]
    reason_codes: tuple[str, ...]
    imagined_plan: tuple[Action, ...]   # 仅供解释，不代表未来已执行
    score_components: dict[str, float | None]
    state_version: int
```

新观测到来时必须建立新的 `PatientState`，替换旧 imagined latent。已执行历史由外部事实提供，不得把上次推荐自动记作已执行。

### 10.2 规划时间不能偷看真实下一 MRI 日期

- 事实 dynamics 评价使用历史真实 `delta_days`，明确属于“给定事实动作与间隔”的预测。
- 真实推荐／回放使用预先声明的 planned interval。首版可取训练集正间隔中位数，并对所有候选保持一致。
- 不从当前 test 患者未来 MRI 日期反推 planned interval。
- 训练时间分布以外的候选间隔要标记外推，不能自动裁剪后假称仍在支持范围内。
- 新观测可能比计划早或晚；收到新观测立即更新时间并重新规划。

### 10.3 Beam search 的最小实现

配置：候选数最多6、H=3、beam_width=4。

`PlanNode` 只需保存：`member_states[M,D]`、共享动作历史、共享 clinical 假设、累计计划时长、路径与各项分数。所有内容仅驻内存，默认不逐节点落盘。

```text
root: 所有 member 都从本次 observed state 开始
for depth = 1..H:
    对当前 beam 中每个 node：
        Agent 返回候选 ID
        硬过滤无效候选
        对每个候选：
            各 member 用自己的 latent 独立向前一步
            按相同动作更新各分支的治疗历史与时间
            逐 member 求 outcome，计算 mean cost 与 disagreement
            累积当前分支的规划分数
    以统一分数保留最好的 beam_width 个 node
返回最佳分支的第一个 action；其余动作只作为 imagined_plan
```

所有被比较的最终分支必须到达相同 H。若没有可达 H 的有效分支，返回明确的 `no_feasible_horizon_plan`；不得把一个只走1步的低代价分支与3步分支直接比较。需要退回 H=1 时必须对所有候选统一重算并记录显式 fallback，首版默认不自动退回。

最大拓展数约 `6 + 4*6 + 4*6 = 54`；每个节点的5个 member 都需计入实际模型调用预算。批量化候选/member 前向，避免每个候选重新加载模型。

### 10.4 Ensemble 的两个不可混淆原则

**原则一：同一分支使用相同 action 序列。**

若不同 member 自己选择不同动作，那么后续方差同时包含 action 差异与模型差异，不再是 Stage 1 定义的同动作 disagreement。第一版不要这样做。

**原则二：均值可用于摘要，不用于回灌。**

Agent 将来可读取 ensemble mean／方差摘要，但每个 member 的下一 latent 始终由自身上一步 latent 推进。

在 outcome 上也先计算 `G(z_m)` 再求均值，不默认用 `G(mean(z_m))` 代替；非线性结局头下二者不等价。

### 10.5 分支代价定义

单步状态代价：

\[
\bar c_j=\frac1M\sum_m[1-\hat S_\psi(365\mid\hat s_j^{(m)})].
\]

latent disagreement：

\[
U_j=\frac1{Md}\sum_m\|\hat z_j^{(m)}-\bar z_j\|_2^2.
\]

用 validation 事实 rollout 的每个 horizon 分位数做尺度归一化：

\[
\widetilde U_j=\min\{5,U_j/(Q_{0.9}^{val}(U_j)+\epsilon)\}.
\]

若分位数退化为0且全部 disagreement为0，禁用此项并报告，不人为制造 uncertainty。该操作只是尺度统一，不是临床不确定性校准。

最终：

\[
J(\mathbf a)=\frac{\sum_{j=1}^{H}w_j[\bar c_j+\lambda_U\widetilde U_j]}{\sum_{j=1}^{H}w_j},\quad
w_j=\exp(-\widehat t_j/365).
\]

这是**预后代理状态代价 + 分歧惩罚**，不是累计生存概率。不同候选必须使用相同计划时长和 horizon，避免通过缩短评价范围取得虚假优势。

λU 首轮只比较0与一个预先声明的值，例如0.05。它是工程验证默认值，不是已证明最佳临床权重。只能基于训练／验证数据或独立合成验证场景调整；test 一经查看，不再据此选择同一轮最佳参数。

### 10.6 约束与拒绝机制

硬约束在评分前过滤，不允许“足够低的预测风险”抵消违规。

首版默认实现：schema合法、目录内 action、无重复 token、非 UNKNOWN、正 planned interval、合法数值、训练支持度达到声明条件。

临床禁忌／剂量／toxicity 规则只有在来源和适用条件经过人工确认后才能启用。不要直接把 CLARITY 的演示规则或 LLM 常识当临床真值；没有数据支持的 toxicity 头不训练，缺失 toxicity 不填0冒充无毒性。

全部候选无效、结局头未通过数据审计、state不完整或被显式设置的外推阈值拒绝时，返回 `ABSTAIN`，供人工审核。结构有效性和临床安全性分别报告。

### 10.7 关于更复杂的 imagined closed-loop policy

当前 MPC 在每次真实观测后重规划，就能运行闭环验证；在一次搜索中按不同 predicted nodes 调用 Agent只是可选增强。

第一版不让每个 member 独立调用 LLM、独立选择未来动作。后续要研究条件策略树时，需要显式区分“相同动作下的模型分歧”与“策略分支带来的结果分散”，并单独设计消融。

## 11. 回放与环境：不要制造不存在的反事实标签

### 11.1 FactualEvaluator

只用于已记录的实际治疗轨迹：

```text
真实起点 -> 输入事实 action/delta -> 递归预测 -> 对真实未来 latent 评分
```

这里允许使用该评价窗口中真实的未来治疗作为给定输入，因为测的是 conditional forecasting。报告必须注明 `conditioning=factual_actions_and_times`，不把它冒充未知未来动作下的自主预测。

### 11.2 ObservedReplay

`ObservedReplay.next_observation()` 不接受推荐动作作为状态转移参数。它读取下一次历史观测，并同时返回上一段**历史实际执行动作**以更新事实历史。

```text
展示截至 t 的信息 -> Planner 推荐 a_recommended
                         |
                         v
在 evaluator 内与 a_actual 比较并记录，不宣称 a_recommended 已执行
                         |
读取历史 x_(t+1)，更新历史为 a_actual，重新规划
```

逐步回放记录中必须分别保存 `recommended_action` 与 `actual_action` 字段，不能共用一个 `action` 字段。

若二者不同，也不能将真实下一 MRI 与对 `recommended_action` 的预测做 MSE，称作该推荐的预测误差。此时真实下一 MRI只可验证**另一次以 actual action 为输入**的事实预测。

P1 固定计划在相同短期 episode 的起点生成计划后不接收新观测；P2–P5 在每个新观测后重算。该比较包含额外信息的优势，不足以单独证明算法创新。正式对标还必须有“同样接收新观测的 CLARITY”或其他重规划基线。

### 11.3 独立合成闭环环境

`synthetic.py` 实现一个小型离散状态环境，仅用于验证算法机制，禁止使用真实药名。

建议状态为：病情代理 `b∈{0,...,5}`、负担代理 `x∈{0,...,3}`、固定亚型 `r∈{0,1}`。动作 `A0/A1/A2` 均为合成动作。示例动力学：

\[
b'=clip(b+1-e_r(a)+\xi,0,5),\qquad
x'=clip(x+v(a)-1,0,3),
\]

其中 `e_0=[0,1,2]`，`e_1=[0,2,1]`，`v=[0,1,2]`，扰动 `ξ∈{-1,0,1}` 概率为 `[0.1,0.8,0.1]`。这些数字是人为构造的测试环境，不对应真实医学效应。

真实代价可设为：

\[
c_{env}=b'/5+0.4x'/3+\mathbf1[x'=3].
\]

用固定、可复现的映射把状态编码成8维 synthetic latent；训练一个独立的 state-cost head。该 head 实现同一 `state_cost()` 接口，但不输出临床生存指标。训练／验证／测试 episode 分离，推荐500/100/200条，开发 smoke 使用更小规模。

`ToyEnv.step(action)` 才真正依据推荐改变状态。环境转移代码不能调用学得的 world model。评估器可以基于已知转移概率对有限 episode 做动态规划，计算最优期望价值；Planner 不能读取环境下一次随机扰动或 oracle value。

P0–P5 使用相同初始状态和配对的外生随机数流；动作导致不同状态属于正常。环境只在执行之后返回新状态，禁止把未来噪声给任意方法。

### 11.4 观测更新与模型缓存失效

一次 `plan()` 内可缓存重复 `(state_version, path, action, delta)` rollout。缓存驻内存，决策结束释放。新观测使 `state_version` 增加，旧 imagined-state 缓存必须失效。

真实回放的新事实历史使用 actual action；合成环境的新历史使用实际送入 `step()` 的推荐动作。系统接口预留人工审核后的 executed_action，不要求执行等于推荐。

---

## 12. 指标的精确定义

### 12.1 Dynamics

对同一 horizon 的合法测试窗口：

\[
\mathrm{MSE@k}=\frac1{N_k}\sum_i\frac1d\|\bar z_{i,t+k}-z_{i,t+k}\|^2,
\quad
\mathrm{LongMSE}=\frac{\mathrm{MSE@2}+\mathrm{MSE@3}}2.
\]

不要按 H2/H3 窗口数加权后仍使用相同的 Long MSE 名称。patient-macro 在每个 horizon 先对患者内部窗口平均，再对患者等权平均。

配对 RI：

\[
RI_s=100\frac{MSE_{base,s}-MSE_{candidate,s}}{MSE_{base,s}}.
\]

先分别计算 seed，再汇总 mean±样本std；同时保存各 seed。不要把“均值的比值”和“配对百分比的均值”混用。[S1, S4]

所有方法使用同一窗口主键 `(patient_id,start_timepoint,target_timepoint,horizon)`。缺失值不得通过删去不同方法的不同失败窗口来改善平均数。

### 12.2 Ensemble reliability

逐 horizon 分别计算：

- Spearman(U,error)，正确处理 ties；常数向量／样本不足输出 null。
- 按 U 排序取最低／最高 `floor(N/3)` 个窗口的平均误差比。
- 保留最低分歧 `floor(0.8N)` 个窗口，报告 Risk@80 与 `retained/N`。
- 同时保存 Risk@100，避免只报告相对下降隐藏绝对误差。

validation dynamics 评价可以另外计算每个 model/seed/horizon 的 U 尺度并写入 run.json；这是评分参数准备，不更新 latent normalizer。`freeze-protocol` 后这些尺度只读，test 评价不得重新拟合。

主 reliability 使用对 ensemble mean prediction 的误差；不要改成 average member error 后继续复用同一指标名。

不同模型的80%保留集合可能不同，因此跨模型的 Risk@80 比较不能直接解释成同一批患者更安全。合成／临床 planner 的推荐分支也可能分布外，事实轨迹上的相关性不能自动推广成反事实可靠性。

### 12.3 Survival

Harrell C-index 采用风险越高、预期事件越早的方向；删失处理与 ties 规则固定。优先每位患者一个预注册 landmark，避免同患者之间被当独立比较对。[S11]

IPCW Brier 在时点 τ：

\[
BS(\tau)=\frac1N\sum_i\left[
\frac{\mathbf1(T_i\le\tau,E_i=1)\hat S_i(\tau)^2}{\hat G(T_i^-)}+
\frac{\mathbf1(T_i>\tau)(1-\hat S_i(\tau))^2}{\hat G(\tau)}
\right].
\]

`G` 为 train censoring 分布估计，训练侧的 landmark 选择规则应与对应评价一致：首 landmark 任务用训练患者首 landmark，预测 Hk 任务用训练患者对应的目标 landmark 规则。不能用 test 的删失分布补充训练支持。边界 convention 与参考实现对齐并测试。生存与删失时间超出可估计支持时，不靠补 ε 得到一个看似正常的指标；输出具体原因。Brier 接收的是生存概率，不是风险 logit。[S11]

`metrics.py` 可自行实现小型、经手工例子验证的 KM/IPCW/C-index，以保持依赖简洁；开发环境可选与 scikit-survival 对拍。不能把库报错捕获后返回0或1。

### 12.4 实际治疗一致性

ActionCodec 把推荐与实际区间治疗转换到同一规范化集合。分别报告：

- action-set precision/recall/F1 与 Jaccard；
- 非空治疗子集指标、明确空治疗比例；
- 训练目录能覆盖实际动作的比例；
- 提议的候选集合包含实际动作的比例；
- 有推荐覆盖率与 ABSTAIN 比例；
- 条件于有推荐的指标及所有可评估样本上的无条件指标。

ABSTAIN 不能在主指标中自动当成正确空动作。若实际动作未知，排除 action-match 评价并报告原因。两者均明确为空时，Jaccard约定为1，并另外报告该类占比，防止空集合主导结果。

真实治疗只是历史行为，不是“最优治疗”标签。这里的高F1不能证明提高生存；低F1也不直接证明推荐更差。

### 12.5 系统和合成环境指标

所有决策记录 wall time、world model forwards、候选数、有效候选数、beam nodes、LLM calls、fallback、ABSTAIN、约束范围。修改 action 的频率是行为描述，不能单独作为“更动态所以更好”的指标。

合成环境以独立环境累计代价和期望最优价值差为主；不能用自己的预测 `J` 代替真实 env cost 作为主要收益。相同模型上的重评分仅叫 internal scoring analysis。

---

## 13. 配置设计

### 13.1 `configs/default.yaml`

下面给出第一版拟实现的完整核心配置。未启用模块也保留接口，但不得触发在线调用。

```yaml
project:
  name: CLARITY_Loop
  schema_version: cloop_v1
  device: auto
  seed: 17

data:
  encoder: brainiac
  protocol: main_v1                  # main_v1 | legacy_stage1
  input_backend: cached_latents     # 不在默认训练中加载 MRI encoder
  split_seed: 17
  train_fraction: 0.70
  validation_fraction: 0.15
  action_alignment: timestamps      # legacy_stage1 自动校验为 legacy_source
  unknown_interval_policy: exclude_action_conditioned
  min_action_support: 2
  use_progression: false
  exclude_shifted_death_labels: true
  min_latent_std: 0.000001

world:
  hidden_dim: 128
  action_embed_dim: 32
  time_embed_dim: 16
  clinical_embed_dim: 16
  history_embed_dim: 16
  use_context: true                 # legacy_stage1 固定 false
  use_history: true                 # legacy_stage1 固定 false
  delta_scale_days: 365.0
  ensemble_size: 5
  max_horizon: 3
  terminal_only: true
  teacher_forcing: false

training:
  seeds: [7, 17, 29]
  variants: [baseline, rrt, ensemble, rrt_ensemble]
  batch_size: 32
  max_epochs: 100
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  patience: 15
  checkpoint_metric: val_long_mse
  num_workers: 0
  precision: float32
  resume_at_epoch_boundary: true

outcome:
  enabled: true
  kind: piecewise_exponential
  hidden_dim: 128
  edges_days: [0, 90, 180, 365, 730]
  score_horizon_days: 365
  max_epochs: 100
  lr: 0.001
  weight_decay: 0.0001
  patience: 15
  checkpoint_metric: val_survival_nll
  patient_weighted: true
  primary_landmark: first_eligible
  freeze_before_planning: true

policy:
  kind: catalog
  max_candidates: 6
  allow_network: false
  fallback: catalog
  llm:
    provider: null
    model: null
    base_url: null
    api_key_env: CLOOP_API_KEY
    temperature: 0.0
    timeout_s: 30
    max_retries: 1
    max_calls_per_decision: 2
    ids_only: true

planner:
  horizon: 3
  beam_width: 4
  interval_source: train_median
  lambda_uncertainty: 0.05
  uncertainty_quantile: 0.90
  uncertainty_clip: 5.0
  discount_scale_days: 365.0
  require_supported_actions: true
  clinical_rules_path: null
  toxicity_model: disabled
  abstain_when_no_valid_action: true

evaluation:
  horizons: [1, 2, 3]
  selective_coverage: 0.8
  report_patient_macro: true
  prediction_records: true
  planning_methods: [frequency, fixed_plan, greedy, mpc_ensemble,
                     mpc_rrt_ensemble, mpc_rrt_ensemble_unc]
  synthetic_train_episodes: 500
  synthetic_val_episodes: 100
  synthetic_test_episodes: 200
  synthetic_episode_steps: 6

artifacts:
  default_run: brainiac_v1
  save_only_best_and_last: true
  save_predictions_jsonl: true
  save_raw_prompts: false
  save_branch_tensors: false
  allow_overwrite: false
```

`legacy_stage1` 不是悄悄覆盖参数的万能开关：配置加载器应验证其固定选项，生成完整 resolved config 并将差异写入 `run.json`。`main_v1` 不允许沿用含所有患者频次的目录。

`min_action_support` 只控制可提议目录，不应删除事实 dynamics 数据中其他已经可编码的动作；训练罕见动作与规划时是否允许它是不同问题。

### 13.2 `configs/server.yaml`

```yaml
paths:
  project_root: /home/tanyuejun/CLARITY_Loop
  source_project: /home/tanyuejun/CLARITY_HAUWM_Minimal
  clarity_root: /home/tanyuejun/CLARITY
  timeline: /data/tanyuejun/CLARITY/clinical/MU_Glioma_Post/clinical_latest.json
  timeline_alternative: /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json
  latent_dir: /data/tanyuejun/CLARITY_HAUWM_Minimal/latents/brainiac
  legacy_trajectories: /home/tanyuejun/CLARITY_HAUWM_Minimal/data/trajectories/brainiac
  mri_root: /data/tanyuejun/CLARITY/dataset/MU-Glioma-Post
  brainiac_checkpoint: /home/tanyuejun/CLARITY/BrainIAC-main/src/checkpoints/BrainIAC.ckpt
  cache_root: /data/tanyuejun/CLARITY_Loop/cache
  output_root: /home/tanyuejun/CLARITY_Loop/outputs

runtime:
  python_reference: /home/tanyuejun/miniconda3/envs/py310/bin/python
  allowed_physical_gpus: [4, 5, 6, 7]
  preferred_physical_gpu: 7
```

这些 GPU 编号来自旧仓库约束，不代表现在空闲。[S1] 默认由操作者设置 `CUDA_VISIBLE_DEVICES`，程序检查映射，不自动抢占所有可见显卡。MVP 只训练缓存 latent 上的小模型，不需要两张96GB卡才能启动。

### 13.3 配置合并与冻结

优先级：代码类型默认值 < default.yaml < server.yaml路径段 < 显式CLI参数。未知字段、类型不匹配或不允许的 protocol 组合直接报错。

首次 `prepare` 写入 resolved config、数据签名和 split。已存在 run 与新配置不一致时，不覆盖、不续训；要求使用新的 run 名称。只允许日志打印频率等不影响模型的选项在同一 run 中改变并记录。

---

## 14. 训练、评价与文件产物的实现约定

### 14.1 `run.json`：唯一实验清单

`run.json` 保存：

```text
schema_version、project Git commit / dirty flag
两份参考仓库 commit（只用于追溯）
resolved_config、Python/PyTorch/CUDA版本、设备信息
数据路径的内容签名、timeline/latent provenance、encoder provenance
patient split IDs、任务 cohort 与排除计数、action vocab/catalog
normalizer 来源、planned interval、U 归一化参数
stage states：prepared / dynamics_done / outcome_done / evaluated / protocol_frozen
各模型 best epoch、完成状态、警告、test 是否已经揭示
结果解释限制与数据质量摘要
```

没有 Git commit 的新仓库使用 `project_revision="uncommitted"` 加源码内容摘要，不伪造 commit。

`run.json` 不保存大 tensor，不保存 API key，不保存完整敏感 prompt。每次更新都采用原子覆盖。

### 14.2 `models.pt` 与 `last.pt`

`models.pt` 结构建议：

```python
{
    "schema_version": "cloop_models_v1",
    "data_signature": "...",
    "preprocessing": {
        "latent_normalizer": {},
        "clinical_codec": {},
        "action_codec": {},
    },
    "dynamics": {
        "baseline/17": {
            "member_states": [...],
            "config": {},
            "best_epochs": [...],
        },
        "rrt_ensemble/17": {...},
    },
    "outcome": {
        "clinical_only/17": {...},
        "state/17": {...},
    },
}
```

同一文件可包含多个小模型。每完成一个 member 后原子更新；未完成的 ensemble 必须标记 incomplete，不能进入正式 ensemble 评价。

`last.pt` 是**唯一恢复点**，包含当前 variant/seed/member/epoch、model、optimizer、early-stopping 状态、DataLoader generator 与 Python/NumPy/Torch RNG 状态、配置签名和数据签名。第一版只承诺完整 epoch 边界恢复。某个训练任务结束后即可删除或覆盖 `last.pt`；整个 run 完成后默认删除。

### 14.3 `metrics.json`：训练历史与聚合结果

建议结构：

```json
{
  "training": {
    "rrt/17": {
      "best_epoch": 12,
      "history": [
        {"epoch": 1, "train_loss": 0.9, "val_long_mse": 1.1},
        {"epoch": 2, "train_loss": 0.8, "val_long_mse": 1.0}
      ]
    }
  },
  "dynamics": {},
  "reliability": {},
  "outcome": {},
  "replay": {},
  "synthetic": {}
}
```

只保存标量、计数和小型数组；不要把逐患者 latent 或大预测矩阵嵌入 `metrics.json`。同一命令重复运行时覆盖对应 section，不追加重复历史。

### 14.4 `predictions.jsonl`：唯一逐样本记录文件

当 `artifacts.save_predictions_jsonl=true` 时，每行是一条带稳定 `record_id` 的 JSON，例如：

```json
{"record_id":"dyn:test:rrt:17:PID:TP1:TP3:H2","kind":"dynamics","split":"test","variant":"rrt","seed":17,"patient_id":"...","horizon":2,"mse":0.42,"disagreement":null}
{"record_id":"replay:test:mpc:17:PID:TP2","kind":"decision","split":"test","planner":"mpc_rrt_ensemble","seed":17,"patient_id":"...","step":2,"recommended_action":"A3","actual_action":"A1","status":"recommend"}
```

实现方式保持简单：文件规模小，`artifacts.py` 读取现有 JSONL -> 以 `record_id` 为 key 更新 -> 排序 -> 写临时文件 -> 原子替换。正式评价不使用“无限 append”模式，因此重复执行不会产生重复行。

不要保存完整 prompt、API key、MRI 数组或 branch latent。若只需要 aggregate 结果，可关闭 JSONL，最终 run 只有 `run.json + models.pt + metrics.json + report.md`。

### 14.5 `report.md`

`report` 命令只读取 `run.json`、`metrics.json` 和可选 `predictions.jsonl`，生成一个 `report.md`，不创建 `reports/` 子目录。报告至少包含：数据审计、完成的实验组、Dynamics、Reliability、Outcome、Replay/Synthetic、资源消耗、限制和未完成项。

### 14.6 幂等、恢复与失败行为

- `prepare`：缓存签名一致则复用；不同则要求新 protocol/run，禁止覆盖源数据。
- `train --resume`：读取 `last.pt` 并校验配置/数据签名后继续；不从 best 权重重新创建 optimizer 冒充 resume。
- 已完整完成的模型组合按签名跳过；需要重跑时使用显式 `--force-task`，只替换目标模型和对应 metrics section。
- `evaluate`：在内存中完成评价，原子更新 `metrics.json`；若启用 JSONL，按 `record_id` 替换对应记录。
- `report`：纯读已有产物，重复执行只覆盖 `report.md`。
- SIGTERM／异常：尽量在完整 epoch 边界更新 `last.pt` 和 `run.json.stage_state="interrupted"`；无完整 checkpoint 时明确失败。
- 没有模型、没有合格标签或 Agent 不可用时明确失败；正式评价绝不 fallback 到随机 risk/mock prediction。

### 14.7 依赖与打包

基础依赖：Python3.10、PyTorch、NumPy、PyYAML；测试使用 pytest。可选 scikit-survival 仅用于指标对拍；在线 provider SDK 作为 optional extra，核心模块导入时不能强制 import。

建议 `pyproject.toml`：

```toml
[project]
name = "clarity-loop"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch", "numpy", "PyYAML"]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
cloop = "cloop.cli:main"

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

安装前先检查已有 PyTorch/CUDA 环境，不用未经测试的“最新版”替换服务器 torch。`doctor` 将实际版本写入 `run.json`。

`.gitignore` 至少包含：

```text
outputs/
*.pt
*.pth
.env
__pycache__/
.pytest_cache/
```

不要全局忽略所有 `.json`，因为配置示例与小型测试 fixture 可能需要提交。

---

## 15. 命令行设计与从零执行顺序

### 15.1 首先完成仓库文件，而不是依赖旧项目的 import

在新目录按第2节创建源码和测试，实现CLI后安装。下面是实施完成后的使用方式，不是当前旧仓库已有命令。

```bash
mkdir -p /home/tanyuejun/CLARITY_Loop
cd /home/tanyuejun/CLARITY_Loop

# 在这个目录中实现本文定义的文件后再安装。
PY=/home/tanyuejun/miniconda3/envs/py310/bin/python
"$PY" -m pip install -e ".[dev]"
"$PY" -m pytest -q
"$PY" -m cloop smoke --device cpu
```

`smoke` 必须完全不读服务器医疗数据、不要求API key、不联网；自动创建内存合成数据，运行一次RRT反传、2-member ensemble、Outcome接口、Agent和MPC，以及保存/恢复测试。临时文件位于 `TemporaryDirectory`，成功后删除。

### 15.2 核查路径与数据

```bash
"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 doctor

"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 prepare
```

`doctor` 无写源数据行为。`prepare` 只写必要缓存与 `run.json`；数据审计摘要直接进入 `run.json.audit`。若动作时间语义无法确认，报告具体字段和样本数量，不默默启动正式 planning。

### 15.3 先做旧协议回归检查（可选但推荐）

```bash
"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_legacy --protocol legacy_stage1 prepare

CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_legacy --protocol legacy_stage1 \
  train --suite dynamics --variants baseline rrt --seeds 17
```

`--protocol` 为全局参数，必须在子命令前。legacy run只用于核对旧数据和代码，不与main_v1合并汇总。确认后可归档其一个目录，不增加细碎中间文件。

### 15.4 当前完整 Dynamics 实验

以下GPU7只是假设操作者确认空闲后的示例；不要直接占用其他任务使用中的显卡。

```bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 train --suite dynamics \
  --variants baseline rrt ensemble rrt_ensemble --seeds 7 17 29

"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 evaluate --suite dynamics --split validation
```

可以先只跑seed17，但完整结果报告必须清楚标注当前完成的seed集合，不将单seed自动包装为mean±std三seed实验。

### 15.5 Outcome 训练与验证

```bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 train --suite outcome --seeds 7 17 29

"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 evaluate --suite outcome --split validation
```

`train --suite outcome` 训练O0/O1；O2无需重训，仅将冻结的各dynamics输出送给同seed O1。为避免重复预测，必要的小型结果可驻内存复用，不写每个tensor一个文件。

### 15.6 先独立验证闭环环境

```bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run synthetic_v1 synthetic --seeds 7 17 29
```

该命令生成独立合成数据、训练synthetic dynamics及state-cost head、评价P0–P5；产物使用与真实数据 run 相同的扁平文件协议；不建立额外 seed/member 子目录。不能加载clinical survival模型作为synthetic oracle；也不能把synthetic结果写进clinical指标表。

### 15.7 离线 Policy 回放、冻结协议后测试

```bash
"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 evaluate --suite replay --split validation

# 开发和参数选择完成后，一次性冻结评价协议，再评估test。
"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 freeze-protocol

"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 evaluate --suite all --split test

"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 report
```

这里的 `all` 指真实数据Dynamics、Outcome和Replay，不包含Synthetic训练。已揭示test后继续修改方法必须新建实验版本并标记探索性；不能将同一test反复选择出的最好结果称为独立验证。

### 15.8 手工状态测试及训练恢复

```bash
# case文件由操作者提供当前可见状态，不含未来labels。
"$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 plan --case /path/to/current_state.json --seed 17

# 完整epoch边界恢复，不从best权重重建optimizer冒充resume。
CUDA_VISIBLE_DEVICES=7 "$PY" -m cloop \
  --config configs/default.yaml --paths configs/server.yaml \
  --run brainiac_v1 train --suite dynamics --resume
```

`plan` 输出结构化研究建议和reason codes，不执行医疗操作。下次提交新的case时，state_version必须增加，executed history来自事实而非上次recommendation。

---

## 16. 必须实现的测试与验收

### 16.1 `test_data.py`

- 患者级拆分互斥、可复现；改变患者输入顺序不改变split。
- normalization只用train；改变test latent不改变均值方差。
- 乱序TP按mri_day排序；缺失／重复时间不会被静默补齐。
- 已知interval跨两个MRI时正确裁切；同一事件多个节点重复不重复计数。
- destination存放的历史区间事件可以正确对齐，但不会进入source时刻Policy可见历史。
- 缺失action、明确空action、ABSTAIN保持三个不同状态。
- survival/event/未来progression/未来真实治疗不会出现在PolicyRequest。
- 修改未来节点及test action不会改变当前候选目录或当前state。
- `death_shifted_to_L_plus_1`按配置排除，错误数量记录正确。

### 16.2 `test_world.py`

- K=1时Baseline与相同初始化的RRT计算一致。
- RRT第二步输入等于第一步预测，不是真实中间latent。
- 终点loss的梯度能到达第一步参数；无意detach会使测试失败。
- 不同horizon的padding和mask正确，不额外推进短序列。
- M=1时U=0；构造不同member预测时U与手算一致。
- 采用独立轨迹与均值回灌的测试案例应得到不同输出；实现必须匹配前者。
- clinical/history开关对四个variant一致。
- Persistence正确复用起点，不读取未来target。

### 16.3 `test_outcome.py`

- 生存概率∈[0,1]，随τ单调不增。
- event、删失、事件落在90/180/365天边界及超过730天均有手算例子。
- positive T和event类型校验；不会把早期删失误标为365天存活。
- 换成未来landmark时，label生存起点随目标timepoint改变。
- C-index风险方向与手算一致；Brier无支持时返回null及原因。
- outcome推理不会更新encoder或dynamics，且不调用在线Agent。

### 16.4 `test_policy_planner.py`

- 删除所有API key且禁止网络，CatalogPolicy仍正常工作。
- FakeProvider返回相同候选时，与CatalogPolicy评分一致。
- 更换model字符串只改变provider请求，不改变ActionCodec或Planner。
- 非法JSON、重复ID、目录外ID、timeout均有显式处理；不静默变成空动作。
- 只有第一步被提交；imagined未来计划不写入executed history。
- 新观测后state_version与rollout缓存刷新。
- 每条分支各member执行相同动作，且各自保留latent。
- 新观测改变状态的构造案例中，MPC能够改变动作；不要求所有案例都改变。
- 所有候选非法时ABSTAIN，不返回最低惩罚的违规动作。
- 真实回放推荐≠实际动作时，不把下一真实MRI误记为推荐效果。
- 合成环境确实受推荐动作影响，而不是返回预先固定的历史下一状态。

### 16.5 `test_pipeline.py`

- CPU smoke可从无数据、无API key环境完成。
- 一次最小prepare→train→evaluate→report跑通。
- 相同签名重复 evaluate 不产生重复 JSONL 记录，metrics 对应 section 保持幂等。
- resume恢复optimizer/RNG/epoch；在可复现CPU测试中与连续训练比较。
- 正式模型加载失败时报错，不fallback到随机风险。
- 默认单 run 最终文件不超过六个，不出现 seed/member 子目录；完成后删除 `last.pt` 时通常不超过五个。

**工程验收通过≠科学假设成立。** 测试用确定性构造数据检查机制；真实数据是否优于baseline必须保留负结果。

---

## 17. 实施顺序与每阶段完成标准

| 阶段 | 实现内容 | 完成标准 |
|---|---|---|
| M0 | 项目骨架、类型、严格配置、存储 | 安装、help、CPU smoke；无在线依赖 |
| M1 | 数据导入、对齐审计、split/window | 能解释每个训练窗口的state/action/time来源 |
| M2 | OneStep/RRT/Ensemble与评价 | D0–D4可训练，独立member递归测试通过 |
| M3 | OutcomeHead与删失指标 | O0/O1训练，O2可评价，标签无未来泄漏 |
| M4 | CatalogPolicy、MPC、synthetic | 无LLM条件下完整动作反馈闭环运行 |
| M5 | 真实数据逐观测回放与报告 | 事实动作与推荐分离；P0–P5比较可追溯 |
| M6 | LLMProvider接口接入 | 仅换provider/model配置与adapter；核心模块不改 |

M6不是M2–M5的前置条件。数据标签不能支持M3时，M2与synthetic M4仍然可运行；但不能给真实数据M5填入伪造clinical score。

---

## 18. 后续正式对标 CLARITY 的扩展位置

第一版完成后，再按验证结果决定扩展，而不是一次加入全部功能。

### 18.1 强baseline

后续必须加入固定版本的官方CLARITY，至少区分：初始计划模式、每次观测重新运行模式、其原有uncertainty scoring。统一患者划分、可用输入、候选目录或LLM预算、结局标签、动作时间协议及评价单位。[S8–S10]

第一版的 `fixed_plan` 只是规划消融，不允许直接命名为 `CLARITY`。旧Stage 1的single one-step residual model也不是完整CLARITY Actor。

### 18.2 联合世界模型训练

可以后续研究 `L_dynamics + λ_survival L_survival`、治疗结构正则或encoder LoRA。但每次都需要新的run协议，区分“沿用固定representation的RRT收益”与“representation一起变化的收益”。不要将旧BrainIAC冻结模型结果和新联合训练结果混在同一受控消融。

### 18.3 真实临床反馈接口

可将`CachedObservationEncoder`替换成raw-MRI encoder；将当前显式临床mask扩展为经过验证的观测更新／状态估计器。未被建模的未来毒性、检测结果与肿瘤进展，在imagined里仍是未知，不能由LLM编造。

### 18.4 反事实与临床有效性

世界模型在观测性治疗数据上拟合，未解决混杂与动作覆盖。规划器找到更低的内部score，不足以证明真实患者寿命更长。正式医疗结论需要额外因果假设、适当的离线策略评价、专家盲评或前瞻性研究；它们不由本MVP的模拟回报自动替代。

### 18.5 新方法定位

RRT、Ensemble及MPC本身不是由本项目首创的一般技术。论文应围绕医疗纵向状态、训练与递归使用的一致性、同动作独立member分歧、真实观测更新协议，以及对治疗任务的受控证据展开，而不是把通用组件名称当创新证明。

---

## 19. 最小交付清单

实现者最终应提交：

1. 一个可安装的扁平`cloop`包，以及无数据/无网络即可运行的CPU smoke与单元测试。
2. 两个配置文件；旧服务器源路径只读，新输出位置与旧仓库隔离。
3. D0–D4、O0–O2、P0–P5的统一训练评价入口；不同证据等级分开报告。
4. 完整的CatalogPolicy；LLMPolicy契约、FakeProvider与替换provider测试。
5. 一个真正响应action的独立synthetic环境，以及不会伪装反事实的ObservedReplay。
6. 每个 run 最多六个持久产物；不使用逐样本记录；可恢复、可复算、无静默 mock fallback。

**本轮最重要的交付不是“先找到最好看的数字”，而是获得一个能把RRT+Ensemble、结局评估和观测反馈规划放在同一受控流程内验证的工作版本。**

---

## 20. 核对来源

下列链接固定到本次核对的仓库提交；服务器上的实际数据文件仍须单独审计。方法接口与新配置属于本文设计，不应理解成这些来源已经实现。

- **[S1] MyMeWM README：** 旧Stage 1协议、数据路径、冻结encoder与运行约束。  
  https://github.com/yu3jun1/MyMeWM/blob/b5e3b979e46e6ec254d72aac46fbc352b311bed6/README.md
- **[S2] MyMeWM data.py：** 患者拆分、max_available训练窗口与全horizon评价窗口。  
  https://github.com/yu3jun1/MyMeWM/blob/b5e3b979e46e6ec254d72aac46fbc352b311bed6/src/clarity_hauwm/data.py
- **[S3] MyMeWM model.py：** 单步residual dynamics、独立member递归、disagreement定义。  
  https://github.com/yu3jun1/MyMeWM/blob/b5e3b979e46e6ec254d72aac46fbc352b311bed6/src/clarity_hauwm/model.py
- **[S4] MyMeWM BrainIAC结果汇总：** 四组预测、reliability、seed与horizon实验。  
  https://github.com/yu3jun1/MyMeWM/blob/b5e3b979e46e6ec254d72aac46fbc352b311bed6/outputs/stage1/brainiac/reports/stage1.log
- **[S5] MyMeWM运行manifest：** Python路径、运行snapshot及文件哈希。  
  https://github.com/yu3jun1/MyMeWM/blob/b5e3b979e46e6ec254d72aac46fbc352b311bed6/outputs/stage1/formal_run_manifest.json
- **[S6] MRI-CORE LoRA特征provenance：** 实际timeline、MRI根目录、模型与latent来源。  
  https://github.com/yu3jun1/MyMeWM/blob/b5e3b979e46e6ec254d72aac46fbc352b311bed6/outputs/stage1/mri_core_lora/features/provenance.json
- **[S7] CLARITY extract_clinicial.py：** MRI时间修补、治疗区间摆放、生存与progression字段生成。  
  https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/dataset/extract_clinicial.py
- **[S8] CLARITY README：** 可插拔vision、Cox/BCE、离线候选fallback与ISE接口。  
  https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/README.md
- **[S9] CLARITY main.py：** SequenceWorldModel、MC-dropout、SequenceScorer及w_unc。  
  https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/main.py
- **[S10] CLARITY dataset_glioma_all_pairs_text.py：** paired数据读取、survival字段及治疗JSON。  
  https://github.com/DingTianxingjian/CLARITY/blob/dadb82241a24f5ec5e4e4dc994e3116fd4a9da04/Predictor/dataset/dataset_glioma_all_pairs_text.py
- **[S11] scikit-survival官方指标文档：** IPCW Brier及Harrell C-index的输入与删失定义；实现时核对所安装版本。  
  https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html  
  https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_censored.html
- **[S12] PyTorch官方保存/加载文档：** 通用恢复checkpoint需要optimizer等训练状态，区别于仅保存推理权重。  
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

- **[S13] MyMeWM BrainIAC dataset_stats：** 原有患者数量与各split的窗口统计。  
  https://github.com/yu3jun1/MyMeWM/blob/b5e3b979e46e6ec254d72aac46fbc352b311bed6/outputs/stage1/brainiac/reports/dataset_stats.json

若参考源代码实现，保留相应许可证与署名；不要直接复制演示prompt中的临床规则作为已验证医学约束。

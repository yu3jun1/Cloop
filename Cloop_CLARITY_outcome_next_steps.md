# Cloop：同构 CLARITY Outcome 实验的下一步诊断与最小修改建议

**适用实验：** `outputs/v3/next_stage_clarity_outcome_v1/`  
**代码依据：** Cloop 提交 `996bb1de5e22c18196d2003957cb1045510951a0`  
**文档性质：** 整理上一轮代码审查后的下一步建议，不是新模型设计，也不是新一轮实验结果。

> **只先完成两件事：核实 outcome 的训练与 checkpoint 选择是否有效；检查训练好的 outcome 是否真正利用了预测 post latent。保持“相同 outcome 架构、四组分别训练”的主实验不变。**

---

## 1. 本轮继续回答什么问题？

研究目标仍然是：

> 在使用相同 CLARITY outcome 架构、各组独立训练的条件下，RRT、Ensemble，以及二者组合带来的 latent 改善，能否转化为生存预测收益？

四组主实验继续保留：

| 组别 | 上游 dynamics | 下游 outcome |
|---|---|---|
| Baseline | 单步训练、单模型 | CLARITY 相同架构，独立训练 |
| RRT | 递归训练、单模型 | CLARITY 相同架构，独立训练 |
| Ensemble | 单步训练、多个成员 | CLARITY 相同架构，独立训练 |
| RRT+Ensemble | 递归训练、多个成员 | CLARITY 相同架构，独立训练 |

**本文件中的“固定 head”仅指对一个已经训练好的模型做输入扰动诊断，不是要求四组共用一个 outcome checkpoint。** 四组主实验的独立训练设置不变。

现有代码的主要接线符合上述目标。上一轮读取的报告尚未显示稳定的全队列生存收益；H2 的 uncertainty 筛选有局部信号，但不替代全队列结果。[S1–S4]

---

## 2. 已确认、待核实与不能提前下的结论

| 状态 | 内容 | 处理方式 |
|---|---|---|
| 已确认 | 每个 `variant/fold/seed/horizon` 独立训练 head，四组有相同初始化规则 | 保留，不重构 |
| 已确认 | Dynamics 冻结；主实验输入为真实起点和预测终点 | 保留 |
| 已确认 | Cox-risk 用于 C-index；一年 survival probability 用于 IPCW Brier | 保留 |
| 已确认 | 代码允许 epoch 0 的初始化参数成为最终 checkpoint | 先统计实际发生比例 |
| 尚未核实 | 本轮多少任务实际选择了 epoch 0；是否存在明显训练/早停失配 | 查本地 `metrics.json` 与 `models.pt` |
| 尚未核实 | Head 是否使用了预测 post latent，而不只是依赖 pre/condition | 做固定模型的输入替换诊断 |
| 不能提前认定 | 低 C-index 一定由 epoch 0、过拟合、单 token 或特征失配造成 | 根据诊断结果逐项判断 |

**证据边界：** 本文复核了训练与评价接口；没有读取你本地的 checkpoint 或特征缓存，也没有完成完整训练记录统计。`epoch 0` 是已确认的选模可能性，不是已确认的大规模发生事实。[S1]

---

## 3. 执行顺序与工作范围

| 顺序 | 操作 | 是否重新训练 | 交付结果 |
|---|---|---|---|
| A | 统计 checkpoint、查看现有训练/stop 曲线 | 否 | 哪些任务真正采用了训练后的模型，哪些需要排查 |
| B | 正常 post、打乱 post、真实 post 的同模型前向对照 | 否 | Head 是否对 post 敏感，以及这种敏感是否对预后有帮助 |
| C | 仅对 A/B 指向的问题做一次小修改 | 按需，只训练 outcome | 相同架构、相同预算下的四组新对照 |
| D | 路线锁定后独立确认 | 后续阶段 | 与开发比较分开的最终评价 |

执行 A 时统计全部任务，但优先解读预先指定的 **H2**。B 先做 H2 的四组、五折、三个种子，再查看 H1/H3；不得因为其他 horizon 更有利而临时替换主指标。

现在不做：新增 trajectory encoder、独立 value head、uncertainty 辅助 loss、LLM、MPC 或 nested cross-fitting 系统。基础患者隔离和不使用测试结果选模仍然保留。

---

## 4. 任务 A：核实训练是否产生了有效的 outcome checkpoint

### 4.1 先查现有记录，不先改代码

主要读取：

```text
outputs/v3/next_stage_clarity_outcome_v1/
├── run.json          # 配置、患者划分、来源信息
├── metrics.json      # training 中的逐任务训练记录
├── models.pt         # heads 中保存的参数；需使用本地文件
└── last.pt           # 特征缓存；任务 B 使用，需使用本地文件
```

后两个文件是代码在本地使用的实验产物，**不要假设它们一定已上传到 GitHub**。若本地缺失，先恢复原运行产物；不要把缺失当成模型无效，也不要把其他运行的文件混入本轮。

当前 `train_one_head()` 在第一次梯度更新前，就将初始化模型设为 `best_state`。如果之后 stop loss 始终更高，最终会返回这个初始化状态。训练日志即使有很多 epoch，也不代表最终选中了训练后的参数。[S1]

### 4.2 每个任务记录这些字段

| 字段 | 解释 |
|---|---|
| `available`、失败原因 | 区分任务不可用与模型表现差 |
| `best_epoch`、`epochs_ran` | 实际采用哪个 epoch，运行了多少次更新 |
| `fit_n`、`fit_events` | 该 horizon 实际用于训练的患者数与事件数 |
| `stop_n`、`stop_events` | 该 horizon 实际用于选模的患者数与事件数 |
| epoch 0 / selected / last 的 train、stop loss | 检查训练与泛化趋势，不能只看最后一轮 |
| Cox 与 BCE 分项 | 判断是排序分支还是一年概率分支出现问题 |
| `gradient_norm` | 确认有有限梯度；记录值是裁剪调用返回的总范数 |
| 两个输出的 variance | 检查是否近似常数；不能单凭方差判断模型好坏 |

每个 H 使用独立 head，因此必须按 **variant × horizon** 汇总。不要将所有 horizon 的任务合并汇总而掩盖 H2/H3 的数据量差别。

### 4.3 可直接运行的只读统计

在仓库根目录执行。下列代码只读取本地 JSON，不修改模型、不改原结果；它用于初筛，不代替完整曲线检查。

```bash
python - <<'PY'
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

path = Path('outputs/v3/next_stage_clarity_outcome_v1/metrics.json')
with path.open(encoding='utf-8') as f:
    metrics = json.load(f)
training = metrics.get('training')
if not isinstance(training, dict) or not training:
    raise SystemExit('没有有效的 training 记录，请确认文件属于本轮实验。')

def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

groups = defaultdict(list)
for key, row in sorted(training.items()):
    if not isinstance(row, dict):
        raise SystemExit(f'任务记录不是字典: {key}')
    variant = row.get('variant', key.split('/')[0])
    horizon = row.get('horizon', key.split('/')[-1])
    if not row.get('available', False):
        groups[(variant, str(horizon))].append(None)
        print('UNAVAILABLE', key, row.get('reason', '未记录原因'))
        continue
    best = row.get('best_epoch')
    groups[(variant, str(horizon))].append(best)
    history = row.get('history', [])
    indexed = {x.get('epoch'): x for x in history if isinstance(x, dict)}
    initial = indexed.get(0, {}).get('stop', {}).get('total')
    selected = indexed.get(best, {}).get('stop', {}).get('total')
    gain = initial - selected if finite_number(initial) and finite_number(selected) else None
    suspicious_grad = [
        x.get('epoch') for x in history
        if isinstance(x, dict) and x.get('epoch', 0) > 0
        and not finite_number(x.get('gradient_norm'))
    ]
    if best == 0 or best not in indexed or suspicious_grad:
        print('CHECK', key,
              {'best_epoch': best, 'epochs_ran': row.get('epochs_ran'),
               'fit_n': row.get('fit_n'), 'fit_events': row.get('fit_events'),
               'stop_n': row.get('stop_n'), 'stop_events': row.get('stop_events'),
               'stop_gain_vs_init': gain,
               'missing_or_nonfinite_gradient_epochs': suspicious_grad})

print('\n按 variant / horizon 汇总：')
for group, epochs in sorted(groups.items()):
    valid = [e for e in epochs if isinstance(e, int)]
    print(group,
          {'total_records': len(epochs), 'valid_epoch_records': len(valid),
           'epoch0': sum(e == 0 for e in valid),
           'selected_trained': sum(e > 0 for e in valid),
           'best_epoch_counts': dict(Counter(valid))})
PY
```

再检查本地 `models.pt` 中对应的 `heads[task]`：其 `best_epoch` 是否与日志一致；必要时使用现有 `_tensor_state_hash()` 将实际保存的 `state` 与 `init_state_hash`、`final_state_hash` 对照。[S1]

**“训练后参数发生变化”只证明更新发生过，不证明预后信息已经被有效学习。** 若日志与模型一致，再进入下面的判断。

### 4.4 如何解读，而不是如何强行使实验变好

| 观察到的情况 | 支持的解释 | 下一步 |
|---|---|---|
| 多数任务选择 epoch > 0，train/stop 都较初始化改善 | 流程找到了优于初始化的 checkpoint | 进入任务 B |
| 选择 epoch 0，train loss 明显下降、stop loss 上升 | 当前训练未得到 stop 集认可的改善；可能过拟合或特征分布失配 | 先看样本量与两分支曲线，再考虑小范围训练设置调整 |
| Train loss 也不下降或出现非有限梯度 | 需要排查训练更新、数值或数据 | 暂不解释为 dynamics 无下游价值 |
| Loss 改善，但风险/概率输出近似常数 | 可能主要学到总体水平而缺少个体区分 | 检查两个输出分支和任务 B，不以单个 variance 阈值定论 |
| 日志与 checkpoint 不一致 | 产物一致性问题 | 先定位缓存/恢复问题，不能直接进入结果比较 |

不得删除 epoch 0 任务后重新计算一个更好看的主表；不得仅为了确保 `best_epoch>0` 而取消初始化参考。若补充“训练后最佳 checkpoint”，它应按同一 stop 规则在 epoch ≥ 1 中选择并单独报告，不能替换历史结果。

已经保存的训练历史不包含每个 epoch 的模型参数时，不能根据日志“恢复”中间 checkpoint；必要时重跑 outcome，使用新的 run 名。

---

## 5. 任务 B：检查 head 是否真正利用预测 post latent

### 5.1 保持哪个模型不变？

对每个 `variant/fold/seed/H`，加载它自己训练好的 checkpoint，调用 `eval()`，关闭梯度。该任务的三种输入共用这个 checkpoint。

这是一组**诊断前向**，不是改动四组主实验的训练策略。

### 5.2 三种输入条件

| 条件 | Pre | Post | Condition、标签、患者集合 | 目的 |
|---|---|---|---|---|
| `normal_pred` | 原患者真实起点 | 该组正常预测终点 | 全部保持不变 | 复现原任务结果 |
| `shuffled_pred` | 原患者真实起点 | 同折、同 seed、同 H、同方法中另一患者的预测终点 | 全部保持不变 | 破坏 post 与患者的对应关系 |
| `observed_post` | 原患者真实起点 | 原患者真实目标 MRI latent | 全部保持不变 | 查看无终点预测误差时，该固定 head 的响应 |

`shuffled_pred` 建议预先固定 20 次随机重排。每次不使用 survival 标签选择排列，不跨 fold/horizon/variant 混合，不把 `pre`、condition 或标签一起重排。每患者对应一个目标的当前协议可保证组内不同记录对应不同患者。

**先复现 `normal_pred` 的原始指标。** 若不能复现，先解决输入、checkpoint 或配置不一致，不能继续用扰动结果解释模型。

### 5.3 可复用的现有接口

不需要修改 `SurvivalModule`。可在一个独立诊断脚本中复用：

```text
_rows_for_task()      → 取得该任务的 fit、stop、report rows
_new_head()          → 按原配置创建模型并加载原参数
evaluate_one_head()  → 使用现有 risk/probability 评价规则
```

缓存中的 `post_mean` 是正常 post；`true_post` 是观测参考。现有 `evaluate_one_head()` 内部会读取 `post_mean`，因此诊断时只在记录副本中替换这个字段，不要改原始缓存。[S1]

原实验使用哪个 censoring reference，诊断也用同一个；现有主评价使用该任务的 fit rows。[S1–S2]

### 5.4 输入替换参考函数

下面是新增诊断逻辑的参考，不是已存在的仓库 API。它不读写产物、不训练模型，且不原地修改传入记录。

```python
from typing import Any, Sequence
import torch


def make_post_diagnostic_rows(
    rows: Sequence[dict[str, Any]],
    mode: str,
    permutation_seed: int = 0,
) -> list[dict[str, Any]]:
    rows = list(rows)
    if not rows:
        raise ValueError('诊断队列为空')
    if mode not in {'normal_pred', 'shuffled_pred', 'observed_post'}:
        raise ValueError(f'未知模式: {mode}')
    patients = [r['patient_id'] for r in rows]
    if len(set(patients)) != len(rows):
        raise ValueError('本函数要求同一 H 队列每患者一个目标')
    scope = {(r['variant'], r['fold'], r['seed'], r['horizon']) for r in rows}
    if len(scope) != 1:
        raise ValueError('不得跨 variant/fold/seed/H 打乱')
    if mode == 'shuffled_pred' and len(rows) < 2:
        raise ValueError('少于两名患者，无法打乱')

    original_index = torch.arange(len(rows))
    permutation = original_index.clone()
    if mode == 'shuffled_pred':
        generator = torch.Generator().manual_seed(permutation_seed)
        for _ in range(1000):
            candidate = torch.randperm(len(rows), generator=generator)
            if bool((candidate != original_index).all()):
                permutation = candidate
                break
        else:
            raise RuntimeError('未生成无固定点排列，请检查输入或更换预定 seed')

    output = []
    for i, row in enumerate(rows):
        source = rows[int(permutation[i])]
        if mode == 'observed_post':
            post = row['true_post']
            donor = row['patient_id']
        else:
            post = source['post_mean']
            donor = source['patient_id']
        if post.shape != row['post_mean'].shape:
            raise ValueError('post 形状不一致')
        if not bool(torch.isfinite(post).all()):
            raise ValueError('post 含非有限值')
        new_row = dict(row)
        new_row['post_mean'] = post.detach().clone()
        # 避免沿用原预测的 MSE 描述替换后的输入。
        new_row['latent_mse'] = float(
            (post - row['true_post']).square().mean().cpu()
        )
        new_row['diagnostic_mode'] = mode
        new_row['post_donor_patient_id'] = donor
        new_row['permutation_seed'] = permutation_seed if mode == 'shuffled_pred' else None
        new_row['record_id'] = (
            f"{row['record_id']}:diag:{mode}:{permutation_seed}"
        )
        output.append(new_row)
    return output
```

扰动组只用于 post 利用诊断，不运行 uncertainty 阈值筛选，也不把扰动后的 MSE 当成某个新 dynamics 方法的成绩。

### 5.5 需要记录哪些指标？

继续使用原定义：

- C-index：原始 Cox risk score，越大表示风险越高。
- 一年 IPCW Brier：`sigmoid(survival_logit)`，越小越好。
- 若继续输出 TD-AUC，保留原风险方向；不足以计算时返回缺失及原因。

另外记录两个不依赖排序的输出变化：

\[
D_p=\frac1N\sum_i|p_i^{\mathrm{normal}}-p_i^{\mathrm{shuffled}}|,
\qquad
D_r=\frac1N\sum_i|r_i^{\mathrm{normal}}-r_i^{\mathrm{shuffled}}|.
\]

`D_r` 的尺度随 head 变化，因此主要作同一 checkpoint 内诊断，不跨不同 head 按绝对大小排名。概率和风险两个分支应分开分析：一个分支使用 post，不保证另一个也同样使用。

定义打乱后的性能退化为：

\[
\Delta C=C_{\mathrm{normal}}-C_{\mathrm{shuffled}},\qquad
\Delta B=B_{\mathrm{shuffled}}-B_{\mathrm{normal}}.
\]

正值表示打乱后更差。先在一个任务内汇总 20 次排列，再按原规则先平均 seed、后汇总 fold；排列重复不是新的独立患者样本。

### 5.6 怎样解释，怎样避免过度解释？

| 诊断现象 | 可以支持的判断 | 不能直接推出的判断 |
|---|---|---|
| 打乱后输出和指标几乎不变 | 在本次扰动与样本范围内，未观察到明显 post 依赖 | 永久证明 head 完全不使用 post |
| 打乱后指标下降，真实 post 更好 | Post 对该 evaluator 有用；改善预测输入具有进一步研究空间 | 真实 post 是严格可达上限，或 MSE 下降必然等量转化为收益 |
| 打乱后输出明显变化，但指标不稳定 | Head 对 post 敏感，但敏感性是否有益尚不清楚 | 只要有输入敏感性就表示利用了临床信息 |
| 打乱反而变好 | 正常 post 路径可能存在不利关联或泛化问题，需重复核实 | 应采用随机 post 作为治疗推荐方法 |
| 真实 post 没有变好 | 当前预测输入训练的 head 没显示观测替换收益 | MRI 不含预后信息，或换用真实 MRI 训练也一定无效 |

**两项重要限制：** 打乱 post 会制造不自然的 pre/post/condition 组合，因此这是敏感性诊断，不是因果证明；head 原本在预测 latent 上训练，换成真实 latent 也可能产生输入分布变化。因此 `observed_post` 是参考，不是严格上限。

---

## 6. 根据 A/B 结果决定是否修改：一次只解决一个问题

### 路径一：训练未优于初始化

先保持 head 架构与损失不变，确认 optimizer、梯度、数据和实际 epoch 选择。已有 train loss 明显下降但 stop 变差时，再做小范围学习率/正则化验证。

如需一个最小训练设置检查，可在**不改架构**的情况下，只比较原学习率与一个更低的候选值。候选值应在运行前写入配置，四组预算一致；依据开发/stop 表现选择后，再统一重跑四组。这里是待执行的建议，不代表已经确认学习率是本轮问题来源。

不要同时改变学习率、head 大小、输入 token、标签定义和 uncertainty 机制；否则无法定位改善来源。

### 路径二：训练有效，但未观察到有益的 post 利用

先核对 condition 和 pre 是否主导输出，以及预测 post 的尺度、变化幅度是否合理。仍然只使用现有缓存和模型做诊断。

若需要更强的正向对照，再单独用**真实 pre/post、相同 CLARITY 架构**训练一个参考模型，保持患者与标签匹配。这是诊断后的可选实验，不是现在要求重建 outcome，也不能把只在预测输入上训练的 head 的观测替换结果当成这一实验的替代。

### 路径三：训练有效、post 有用，但 RRT+Ensemble 仍未提升 survival

保留这个负结果。它说明当前改动尚未证明下游收益，不能改用更有利的 horizon、只保留部分 seed 或删除困难患者。

此时再考虑分析哪些误差方向影响 outcome，或调整 dynamics 的训练目标；这属于后续研究问题，**本轮不先添加新的 task-aware loss**。

### 路径四：训练有效，且下游配对比较出现较一致改善

保留当前架构，锁定配置，再安排独立确认。此时才继续评估 uncertainty 是否在预先确定的覆盖率下优于随机保留。

“训练有效”不是所有任务 `best_epoch>0` 就自动成立；“下游有效”也不是某个单折 C-index 上升就成立。应同时看患者级配对结果、折间方向及 Brier 是否明显退化。

---

## 7. Uncertainty 暂时保持现状，不再加训练模块

主模型诊断完成前，不新增 gate、calibration head 或 variance-to-Brier loss。

当前可继续保留的两种前向方式是：

\[
\text{mean-latent}:\quad p=\sigma\!\left(G_{\phi}^{\mathrm{logit}}(z_0,\bar z_H,c)\right),
\]

\[
\text{mean-probability}:\quad \bar p=\frac1M\sum_m\sigma\!\left(G_{\phi}^{\mathrm{logit}}(z_0,\hat z_H^{(m)},c)\right).
\]

成员概率标准差只是 dynamics 传播到该 head 后的分歧，不是临床置信区间。[S2–S3]

如果复查 H2 的 selective 结果，继续保证：所有患者先参与 uncertainty 排序；使用删失感知的 Brier；随机保留与 uncertainty 保留人数一致；同时报告实际覆盖率。评价批次中的 rank curve 是诊断，不是已部署的固定阈值策略。

不得把低 uncertainty 子集的分数直接与 baseline 全体患者的分数比较后声称整体改进。

---

## 8. 最小代码改动与结果交付

### 8.1 代码改动范围

| 文件/位置 | 本轮动作 |
|---|---|
| `clarity_survival_module.py` | 不改 |
| `clarity_downstream_head.py` | 不改架构与损失 |
| `clarity_downstream.py` | 先不改训练策略；复用模型/缓存读取与评价接口 |
| 一个独立诊断脚本 | 只实现 A 的汇总、B 的输入替换与前向；脚本名可自定 |
| 原结果目录 | 只读，不覆写 |

建议诊断结果写到独立目录，例如：

```text
outputs/v3/next_stage_clarity_outcome_diagnostics_v1/
├── run.json
├── metrics.json
├── predictions.jsonl
└── report.md
```

这是**建议新增的诊断目录，不是声称当前仓库已有这些诊断产物或命令**。不要求增加新的 artifact 框架；现有 `RunArtifacts(version="v3")` 足以使用这些文件名。

### 8.2 结果只需两张主表

**表 A：训练与选模检查**

| Variant | H | 可用任务数 | 选择 epoch 0 | 选择训练后 checkpoint | 实际 fit/stop 患者数 | 需要排查的原因 |
|---|---:|---:|---:|---:|---|---|
| 待统计 | | | | | | |

**表 B：预测 post 利用诊断**

| Variant | H | Checkpoint 类别 | Normal C-index/Brier | Shuffle ΔC/ΔB | Observed-post C-index/Brier | 概率输出变化 |
|---|---:|---|---|---|---|---|
| 待运行 | | epoch0 或 trained | | | | |

若一个任务选择了 epoch 0，仍可运行输入诊断，但必须标注其为“初始化模型的敏感性”，不能与训练后模型混写成“学到的 post 利用”。

### 8.3 完成标准

本轮完成后应能明确回答：

1. 哪些任务实际使用训练后的 checkpoint？初始化、日志和保存参数是否一致？
2. Head 对预测 post 的改变是否敏感？这种敏感是否改善预后指标？
3. 真实 post 替换提供了什么参考，是否可能受分布变化影响？
4. 下一个动作是修训练、补真实输入参考，还是接受当前无稳定下游收益的结果？

没有这些答案，不继续增加模型复杂度。

---

## 9. 结果表述与最终确认边界

本轮沿用的 `development_reuse` 会复用此前在 source validation 上选出的 world；outcome 训练特征与报告特征也分别来自 world 的样本内和样本外预测。这些是现有开发比较的已披露限制。[S1,S4]

因此，A/B 诊断适合在当前开发产物上完成，但不能据此将旧报告改称最终独立测试结果。保持正式 test 不参与诊断后的选模与反复调参。路线确定后，再按此前约定的普通 train/validation/test 确认即可；当前不要求启动新的复杂交叉拟合工程。

可采用以下结论边界：

- **训练没有找到优于初始化的 checkpoint：** 先报告训练/选模未建立有效 readout，不解释成 dynamics 已被彻底否定。
- **训练与 post 利用均成立，但四组下游无稳定差异：** 如实报告当前 latent 改善尚未证明生存收益。
- **下游出现较一致增益：** 报告相同架构、独立训练条件下的系统预后收益，等待独立确认。
- **仅 selective 子集改善：** 报告特定 horizon/覆盖率下的可靠性筛选信号，不等同于全队列改善或治疗效果。

---

## 10. 依据与状态说明

本文使用的事实来自以下固定提交的代码和已有报告；后续操作、诊断函数与判断分支是据此整理的建议，不是仓库已经验证的新结果。

- **[S1] 训练、特征、checkpoint 与评价接口：** [clarity_downstream.py](https://github.com/yu3jun1/Cloop/blob/996bb1de5e22c18196d2003957cb1045510951a0/src/cloop/v3/clarity_downstream.py)
- **[S2] 指标和 selective IPCW：** [clarity_downstream_metrics.py](https://github.com/yu3jun1/Cloop/blob/996bb1de5e22c18196d2003957cb1045510951a0/src/cloop/v3/clarity_downstream_metrics.py)
- **[S3] 同构 head 适配、Cox/BCE 与成员输出聚合：** [clarity_downstream_head.py](https://github.com/yu3jun1/Cloop/blob/996bb1de5e22c18196d2003957cb1045510951a0/src/cloop/v3/clarity_downstream_head.py)
- **[S4] 本轮开发报告：** [report.md](https://github.com/yu3jun1/Cloop/blob/996bb1de5e22c18196d2003957cb1045510951a0/outputs/v3/next_stage_clarity_outcome_v1/report.md)

**代码检查范围：** 文档内两段 Python 代码已通过语法检查；输入替换函数已用合成张量检查不修改原数据、排列可复现与异常输入处理。未执行你本地实验的完整日志审计、模型加载或真实患者前向诊断。

**最终行动顺序：先查训练，再查 post 利用，最后才决定是否小改训练。保持四组主实验和 CLARITY outcome 架构不变，不因当前结果不理想而重新设计整个系统。**

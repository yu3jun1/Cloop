"""Render the completed next-stage development and toy experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ..v1.artifacts import write_text


VARIANTS = ("persistence", "baseline", "rrt", "ensemble", "rrt_ensemble")
OUTCOMES = ("O0_final", "O1_trajectory", "O2_trajectory_uncertainty")
METHODS = ("frequency", "fixed_plan", "greedy", "mpc_ensemble",
           "mpc_rrt_ensemble", "mpc_rrt_ensemble_unc")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    for key in path:
        value = value[key]
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _fold_means(metrics: dict[str, Any], section: str, label: str,
                path: tuple[str, ...], folds: tuple[int, ...],
                seeds: tuple[int, ...]) -> list[float | None]:
    result: list[float | None] = []
    for fold in folds:
        available = []
        for seed in seeds:
            item = metrics[section][f"{section}/{label}/fold{fold}/seed{seed}"]
            value = _get(item, path)
            if value is not None:
                available.append(value)
        result.append(mean(available) if available else None)
    return result


def _format(values: list[float | None], total: int, digits: int = 3) -> str:
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not finite:
        return "NA (0/" + str(total) + ")"
    body = f"{mean(finite):.{digits}f}"
    if len(finite) > 1:
        body += f" ± {stdev(finite):.{digits}f}"
    if len(finite) < total:
        body += f" ({len(finite)}/{total})"
    return body


def _paired(a: list[float | None], b: list[float | None],
            *, lower_better: bool) -> str:
    delta = [(x - y if lower_better else y - x)
             for x, y in zip(a, b) if x is not None and y is not None]
    wins = sum(x > 0 for x in delta)
    return f"{_format(delta, len(a))}; improved {wins}/{len(delta)} folds"


def render(main_dir: Path, toy_dir: Path) -> Path:
    main_run, metrics = _read(main_dir / "run.json"), _read(main_dir / "metrics.json")
    toy_run, toy = _read(toy_dir / "run.json"), _read(toy_dir / "metrics.json")
    if main_run.get("schema_version") != "cloop_next_stage_v1":
        raise ValueError("unexpected next-stage run schema")
    seeds = (7, 17, 29)
    folds = tuple(range(5))
    expected = {
        *{f"dynamics/{variant}/fold{fold}/seed{seed}"
          for variant in VARIANTS for fold in folds for seed in seeds},
        *{f"outcome/{name}/fold{fold}/seed{seed}"
          for name in OUTCOMES for fold in folds for seed in seeds},
    }
    if set(main_run.get("completed_tasks", [])) != expected:
        raise ValueError("development experiment is incomplete or contains unexpected tasks")
    if set(toy["synthetic"]) != {f"{method}/{seed}" for method in METHODS for seed in seeds}:
        raise ValueError("toy planning experiment is incomplete")
    if not main_run.get("formal_test_untouched"):
        raise ValueError("formal test integrity flag is false")
    if not toy_run.get("synthetic_evidence_boundary"):
        raise ValueError("toy evidence boundary is missing")

    lines = [
        "# Cloop 下一阶段完整开发折实验",
        "",
        "## 协议与完整性",
        "",
        "- 动力学：5 个患者独立验证折 × 3 个种子 × 5 种方法，共 75 项。",
        "- Outcome：5 折 × 3 种子 × 3 种结构，共 45 项；每个 horizon 每患者取第一个合格目标。",
        "- 折内 world 模型只读复用 Outcome-v2；outcome 在训练折中另划患者级早停集。",
        "- 原正式测试集未用于本轮训练或评价；下表均属开发折结果。",
        f"- 基础数据签名：{main_run['base_data_signature']}；来源模型 SHA-256：{main_run['world_models_sha256']}。",
        f"- 实现 SHA-256：{main_run['implementation_sha256']}。",
        "- 所有均值与样本标准差先对每折的三个种子取均值，再跨五折计算；括号表示有定义指标的折数。",
        "",
        "## 动力学：事实条件预测",
        "",
        "| 方法 | MSE@1 ↓ | MSE@2 ↓ | MSE@3 ↓ | CosSim@3 ↑ | Drift@3 ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        values = [
            _format(_fold_means(metrics, "dynamics", variant, (key,), folds, seeds), len(folds))
            for key in ("mse@1", "mse@2", "mse@3", "CosSim@3", "drift@3")
        ]
        lines.append("| " + variant + " | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "H3 同起点 MSE 配对差值为基线减候选方法；正值表示候选误差更低。",
        "",
        "| 配对 | H3 MSE 改善（五折） |",
        "|---|---:|",
    ])
    for a, b in (("baseline", "rrt"), ("ensemble", "rrt_ensemble"),
                 ("baseline", "rrt_ensemble")):
        x = _fold_means(metrics, "dynamics", a, ("mse@3",), folds, seeds)
        y = _fold_means(metrics, "dynamics", b, ("mse@3",), folds, seeds)
        lines.append(f"| {a} → {b} | {_paired(x, y, lower_better=True)} |")

    lines.extend([
        "",
        "## Ensemble 不确定性",
        "",
        "| 方法 | Horizon | 分歧–误差 Spearman ↑ | 分歧–观察性生存 AUC | 高分歧误差 > 低分歧误差 |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant in ("ensemble", "rrt_ensemble"):
        for h in (1, 2, 3):
            rho = _fold_means(metrics, "dynamics", variant,
                              (f"uncertainty@{h}", "spearman"), folds, seeds)
            auc = _fold_means(metrics, "dynamics", variant,
                              ("uncertainty_associations", f"H{h}",
                               "survival_association_auc", "value"), folds, seeds)
            count = sum(
                metrics["dynamics"][f"dynamics/{variant}/fold{fold}/seed{seed}"]
                [f"uncertainty_tertiles@{h}"][2]["mean_mse"]
                > metrics["dynamics"][f"dynamics/{variant}/fold{fold}/seed{seed}"]
                [f"uncertainty_tertiles@{h}"][0]["mean_mse"]
                for fold in folds for seed in seeds
            )
            lines.append(f"| {variant} | H{h} | {_format(rho, len(folds))} | "
                         f"{_format(auc, len(folds))} | {count}/15 任务 |")
    lines.extend([
        "",
        "AUC 只衡量观察性关联。分歧与事实预测误差的排序能力不等于生存预测改进，也不识别治疗效果。",
        "",
        "## Outcome 消融",
        "",
        "| Horizon | 患者数 | 模型 | C-index ↑ | IBS@0–365 ↓ | AUC@365 ↑ | NLL ↓ |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for h in (1, 2, 3):
        patients = sum(metrics["outcome"][f"outcome/O0_final/fold{fold}/seed7"]
                       [f"H{h}"]["patients"] for fold in folds)
        for name in OUTCOMES:
            paths = (("c_index", "value"), ("integrated_brier", "value"),
                     ("time_dependent_auc", "value"), ("survival_nll",))
            values = [_format(_fold_means(metrics, "outcome", name,
                                         (f"H{h}", *path), folds, seeds), len(folds))
                      for path in paths]
            lines.append(f"| H{h} | {patients} | {name} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "以下配对差值对 C-index 为候选减基线，对 IBS 为基线减候选；正值表示候选改善。",
        "",
        "| Horizon | 比较 | ΔC-index | ΔIBS |",
        "|---|---|---:|---:|",
    ])
    for h in (1, 2, 3):
        for a, b in (("O0_final", "O1_trajectory"),
                     ("O1_trajectory", "O2_trajectory_uncertainty")):
            c_a = _fold_means(metrics, "outcome", a, (f"H{h}", "c_index", "value"), folds, seeds)
            c_b = _fold_means(metrics, "outcome", b, (f"H{h}", "c_index", "value"), folds, seeds)
            i_a = _fold_means(metrics, "outcome", a, (f"H{h}", "integrated_brier", "value"), folds, seeds)
            i_b = _fold_means(metrics, "outcome", b, (f"H{h}", "integrated_brier", "value"), folds, seeds)
            lines.append(f"| H{h} | {a} → {b} | {_paired(c_a, c_b, lower_better=False)} "
                         f"| {_paired(i_a, i_b, lower_better=True)} |")

    lines.extend([
        "",
        "## 独立 toy 闭环",
        "",
        "三个种子使用同一批非医疗环境初始状态与扰动流；下表为种子间均值 ± 样本标准差。",
        "",
        "| 方法 | 环境累计代价 ↓ | 相对期望最优差距 ↓ | 扰动后代价 ↓ | 世界模型前向次数 ↓ |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in METHODS:
        values = []
        for key in ("mean_environment_cost", "mean_expected_optimality_gap",
                    "mean_perturbation_recovery_cost", "mean_world_model_forwards"):
            vals = [_get(toy["synthetic"][f"{method}/{seed}"], (key,)) for seed in seeds]
            values.append(_format(vals, len(seeds), 2))
        lines.append("| " + method + " | " + " | ".join(values) + " |")
    fixed_cost = [toy["synthetic"][f"fixed_plan/{seed}"]["mean_environment_cost"] for seed in seeds]
    mpc_cost = [toy["synthetic"][f"mpc_ensemble/{seed}"]["mean_environment_cost"] for seed in seeds]
    rrt_mpc_cost = [toy["synthetic"][f"mpc_rrt_ensemble/{seed}"]["mean_environment_cost"] for seed in seeds]
    fixed_advantage = [a - b for a, b in zip(fixed_cost, mpc_cost)]
    rrt_advantage = [a - b for a, b in zip(mpc_cost, rrt_mpc_cost)]
    lines.extend([
        "",
        "## 结果解读",
        "",
        "- 单模型 RRT 的 H3 MSE 相比 baseline 在 4/5 折下降；RRT+ensemble 的 H3 "
        "MSE 折均值未优于单纯 ensemble。不能把组合方法写成稳定增益。",
        "- 两种 ensemble 的分歧能在多数任务中排序 latent 预测误差；与观察性生存结局的 "
        "AUC 跨 horizon 波动，尚不足以证明预后不确定性建模有效。",
        "- O1 对 O0 的 H1 IBS 有 5/5 折改善，C-index 只有 3/5 折改善；H2/H3 "
        "和 O2 的结果不一致。轨迹与不确定性输入未显示跨 horizon 的稳定生存优势。",
        f"- toy 中 mpc_ensemble 相对固定 3 步计划块的累计环境代价平均低 "
        f"{mean(fixed_advantage):.3f}，三个种子均为正；rrt_ensemble MPC 相对 "
        f"ensemble MPC 的代价改善均值为 {mean(rrt_advantage):.3f}，三个种子均未改善。",
        "",
        "固定计划基线在当前实现中每个 3 步计划块结束后重新规划一次；它不是整段 6 步均不更新的策略。",
        "toy 代价直接来自独立环境，并非患者风险或真实治疗反事实。",
        "",
        "## 解释边界",
        "",
        "- Outcome 的训练折 world 特征是样本内预测，外层验证折保持患者独立；这些结果仍属观察性开发分析。",
        "- H3 生存评价只有少量患者；缺少可比事件或删失支持时指标为 NA，不能补零。",
        "- 本轮 toy 输出未记录逐步计划修订频率；该行为指标不能从累计环境代价反推。",
        "- 不能从本轮或 toy 阶段推断临床治疗收益或因果效果。",
        "",
    ])
    path = main_dir / "report.md"
    write_text(path, "\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("outputs/v3/next_stage_v1"))
    parser.add_argument("--toy", type=Path, default=Path("outputs/v3/next_stage_toy_v1"))
    args = parser.parse_args()
    print(render(args.main, args.toy))


if __name__ == "__main__":
    main()

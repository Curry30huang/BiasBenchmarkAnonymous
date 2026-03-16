#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证据忠实度一致性（Evidence-strength vs Decision-accuracy）分析脚本

目标：
- 通过“支撑句检索强度”与“最终回答正确性”的关联，区分模型的“真实推理”与“猜中答案”。

数据来源：
- data/ssr_evidence_results/<model_dir>/ssr_evidence_result_*.json
  样例字段：
  - llm_result.evidence_indices: List[int]   （模型选的支撑句索引，多选）
  - ground_truth.answers: List[int]          （真值支撑句索引，多选）
  - llm_result.risk_of_bias: str
  - ground_truth.label: str

核心定义：
- 检索强度（0~1）：默认使用 Jaccard(A,G)=|A∩G|/|A∪G|
  该指标天然惩罚"选太多"（多选题预测包含正确但又额外多选时，强度下降）。
- 回答正确性：使用健壮的字符串匹配策略标准化 llm_result.risk_of_bias 与 ground_truth.label 后比较。
  标准化规则（按优先级）：包含 "high"→"high"；包含 "low"→"low"；包含 "some"/"moderate"/"unclear"→"some concerns"。

统计量：
- 盲猜率：strength < τ 但回答正确的比例（默认按低强度子集的条件比例），并给出 Wilson 置信区间。
- 推理错误率：strength ≥ τ 但回答错误的比例（默认按高强度子集的条件比例），并给出 Wilson 置信区间。
- Pearson 卡方检验：检验“回答正确性”是否显著依赖于“检索强度高/低”。p值越小，说明回答正确性越显著依赖于检索强度。
- 敏感度分析：τ ∈ [0,1] 变化，输出 χ² 或 p 值曲线数据（可选绘图）。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# loguru 是项目依赖，但为保证脚本在更“干净”的环境也可运行，这里做可选导入
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("evaluate_evidence_result")


def model_name_to_dirname(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def get_mode_suffix(mode: str = "default") -> str:
    """
    根据模式获取目录后缀

    Args:
        mode: 模式，可选值：default, cot, agent, cot_agent, tool

    Returns:
        目录后缀，如 "_cot" 或 "_cot_agent"
    """
    if mode == "default" or mode is None or mode == "":
        return ""
    elif mode == "cot":
        return "_cot"
    elif mode == "agent":
        return "_agent"
    elif mode == "cot_agent":
        return "_cot_agent"
    elif mode == "tool":
        return "_tool"
    else:
        logger.warning(f"未知模式 '{mode}'，使用默认模式")
        return ""


def _normalize_risk_level(risk_str: str) -> Optional[str]:
    """
    标准化风险等级字符串（使用健壮的字符串匹配策略）

    匹配规则（按优先级）：
    1. 如果包含 "high" 则返回 "high"
    2. 如果包含 "low" 则返回 "low"
    3. 如果包含 "some"、"moderate" 或 "unclear" 则返回 "some concerns"
    4. 如果都不匹配则返回 None（触发外部重试或报警机制）

    Args:
        risk_str: 风险等级字符串

    Returns:
        标准化后的风险等级（"low", "some concerns", "high"），如果无法识别返回None
    """
    if not risk_str:
        return None

    risk_str_lower = str(risk_str).strip().lower()

    # 按优先级匹配（先匹配最明确的，避免误判）
    # 1. 检查是否包含 "high"（优先级最高，因为 high risk 最明确）
    if "high" in risk_str_lower:
        return "high"

    # 2. 检查是否包含 "low"
    if "low" in risk_str_lower:
        return "low"

    # 3. 检查是否包含 "some"、"moderate" 或 "unclear"
    if any(keyword in risk_str_lower for keyword in ["some", "moderate", "unclear"]):
        return "some concerns"

    # 4. 如果都不匹配，返回 None（触发外部重试或报警机制）
    logger.warning(f"无法识别风险等级字符串: '{risk_str}'，返回 None")
    return None


def set_overlap_metrics(pred: Sequence[int], gold: Sequence[int]) -> Dict[str, float]:
    """
    返回一组集合重叠指标，值域均为 [0,1]（空集合时做平滑约定）。
    """
    p = set(int(i) for i in (pred or []))
    g = set(int(i) for i in (gold or []))
    inter = len(p & g)
    union = len(p | g)

    # 约定：若 pred 与 gold 都为空，则视为完全匹配（强度=1）
    if len(p) == 0 and len(g) == 0:
        precision = 1.0
        recall = 1.0
        jaccard = 1.0
    else:
        precision = inter / len(p) if len(p) > 0 else 0.0
        recall = inter / len(g) if len(g) > 0 else 0.0
        jaccard = inter / union if union > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "pred_k": float(len(p)),
        "gold_k": float(len(g)),
        "intersection_k": float(inter),
        "union_k": float(union),
    }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    当 n=0 时返回 (nan, nan)。
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    # 正态近似 z
    # alpha=0.05 -> z≈1.96
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


@dataclass(frozen=True)
class ThresholdStats:
    tau: float
    n_total: int
    n_high: int
    n_low: int
    # 2x2 计数
    high_correct: int
    high_wrong: int
    low_correct: int
    low_wrong: int
    # 比例（条件）
    blind_guess_rate: float  # P(correct | low)
    reasoning_error_rate: float  # P(wrong | high)
    reasoning_correct_rate: float  # P(correct | high)
    # 置信区间
    blind_guess_ci_low: float
    blind_guess_ci_high: float
    reasoning_error_ci_low: float
    reasoning_error_ci_high: float
    reasoning_correct_ci_low: float
    reasoning_correct_ci_high: float
    # 卡方
    chi2: float
    p_value: float


def contingency_for_tau(df: pd.DataFrame, tau: float, alpha: float) -> ThresholdStats:
    """
    df 需要列：
    - strength: float in [0,1]
    - correct: bool
    """
    if df.empty:
        return ThresholdStats(
            tau=tau,
            n_total=0,
            n_high=0,
            n_low=0,
            high_correct=0,
            high_wrong=0,
            low_correct=0,
            low_wrong=0,
            blind_guess_rate=float("nan"),
            reasoning_error_rate=float("nan"),
            reasoning_correct_rate=float("nan"),
            blind_guess_ci_low=float("nan"),
            blind_guess_ci_high=float("nan"),
            reasoning_error_ci_low=float("nan"),
            reasoning_error_ci_high=float("nan"),
            reasoning_correct_ci_low=float("nan"),
            reasoning_correct_ci_high=float("nan"),
            chi2=float("nan"),
            p_value=float("nan"),
        )

    high = df["strength"] >= tau
    low = ~high
    correct = df["correct"].astype(bool)
    wrong = ~correct

    high_correct = int((high & correct).sum())
    high_wrong = int((high & wrong).sum())
    low_correct = int((low & correct).sum())
    low_wrong = int((low & wrong).sum())

    n_high = high_correct + high_wrong
    n_low = low_correct + low_wrong
    n_total = n_high + n_low

    blind_guess_rate = (low_correct / n_low) if n_low > 0 else float("nan")
    reasoning_error_rate = (high_wrong / n_high) if n_high > 0 else float("nan")
    reasoning_correct_rate = (high_correct / n_high) if n_high > 0 else float("nan")

    bg_lo, bg_hi = wilson_ci(low_correct, n_low, alpha=alpha)
    re_lo, re_hi = wilson_ci(high_wrong, n_high, alpha=alpha)
    rc_lo, rc_hi = wilson_ci(high_correct, n_high, alpha=alpha)

    # Pearson chi-square test on 2x2
    # table rows: [high, low]; cols: [correct, wrong]
    table = np.array([[high_correct, high_wrong], [low_correct, low_wrong]], dtype=int)

    # 若有全 0 行/列，chi2_contingency 会报错；做安全兜底
    chi2 = float("nan")
    p_value = float("nan")
    if table.sum() > 0 and (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
        chi2, p_value, _, _ = chi2_contingency(table, correction=False)
        chi2 = float(chi2)
        p_value = float(p_value)

    return ThresholdStats(
        tau=float(tau),
        n_total=int(n_total),
        n_high=int(n_high),
        n_low=int(n_low),
        high_correct=int(high_correct),
        high_wrong=int(high_wrong),
        low_correct=int(low_correct),
        low_wrong=int(low_wrong),
        blind_guess_rate=float(blind_guess_rate),
        reasoning_error_rate=float(reasoning_error_rate),
        reasoning_correct_rate=float(reasoning_correct_rate),
        blind_guess_ci_low=float(bg_lo),
        blind_guess_ci_high=float(bg_hi),
        reasoning_error_ci_low=float(re_lo),
        reasoning_error_ci_high=float(re_hi),
        reasoning_correct_ci_low=float(rc_lo),
        reasoning_correct_ci_high=float(rc_hi),
        chi2=float(chi2),
        p_value=float(p_value),
    )


def load_ssr_evidence_results(model_output_dir: Path) -> pd.DataFrame:
    """
    读取 model_output_dir 下 ssr_evidence_result_*.json，返回每条样本一行的 DataFrame。
    """
    json_files = sorted(model_output_dir.glob("ssr_evidence_result_*.json"))
    if not json_files:
        raise FileNotFoundError(f"未找到结果文件：{model_output_dir}/ssr_evidence_result_*.json")

    rows: List[Dict[str, Any]] = []
    for fp in json_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"跳过无法解析的文件: {fp.name} ({e})")
            continue

        meta = data.get("metadata", {}) or {}
        llm = data.get("llm_result", {}) or {}
        gt = data.get("ground_truth", {}) or {}

        success = bool(llm.get("success", False))
        if not success:
            continue

        pred_indices = llm.get("evidence_indices", []) or []
        gold_indices = gt.get("answers", []) or []
        overlap = set_overlap_metrics(pred_indices, gold_indices)

        pred_label = _normalize_risk_level(llm.get("risk_of_bias"))
        gold_label = _normalize_risk_level(gt.get("label"))
        correct = bool(pred_label is not None and gold_label is not None and pred_label == gold_label)

        rows.append(
            {
                "id": meta.get("id", fp.stem),
                "study": meta.get("study", ""),
                "bias": meta.get("bias", ""),
                "pred_label": pred_label or "",
                "gold_label": gold_label or "",
                "correct": correct,
                "pred_indices": pred_indices,
                "gold_indices": gold_indices,
                **overlap,
                "source_file": str(fp),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"目录 {model_output_dir} 下没有可用样本（可能都 success=False 或解析失败）。")
    return df


def maybe_plot_sensitivity(df_curve: pd.DataFrame, out_png: Path, y: str) -> bool:
    """
    尝试绘图（若 matplotlib 不可用则跳过）。
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.font_manager as fm  # type: ignore
    except Exception:
        return False

    # 配置中文字体支持
    # 尝试使用系统中可用的中文字体
    chinese_fonts = [
        'PingFang SC',  # macOS
        'STHeiti',      # macOS
        'Arial Unicode MS',  # macOS/Windows
        'SimHei',       # Windows
        'Microsoft YaHei',  # Windows
        'WenQuanYi Micro Hei',  # Linux
        'Noto Sans CJK SC',  # Linux
    ]

    font_found = False
    for font_name in chinese_fonts:
        try:
            # 检查字体是否可用
            font_list = [f.name for f in fm.fontManager.ttflist]
            if font_name in font_list:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                font_found = True
                break
        except Exception:
            continue

    # 如果没有找到中文字体，使用英文标签避免警告
    if not font_found:
        logger.warning("未找到中文字体，将使用英文标签")
        xlabel = "Retrieval Strength Threshold τ"
        ylabel_map = {
            "p_value": "p-value",
            "chi2": "Chi-square Statistic"
        }
        ylabel = ylabel_map.get(y, y)
    else:
        xlabel = "检索强度阈值 τ"
        ylabel_map = {
            "p_value": "p值",
            "chi2": "卡方统计量"
        }
        ylabel = ylabel_map.get(y, y)

    plt.figure(figsize=(9, 4.5))
    plt.plot(df_curve["tau"], df_curve[y], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="SSR 证据忠实度一致性分析（检索强度 vs 决策正确性）")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="结果文件根目录（默认：项目根目录/data/ssr_evidence_results）",
    )
    parser.add_argument("--model-name", type=str, required=True, help='模型名称，如 "openai/gpt-5.1"')
    parser.add_argument(
        "--strength-metric",
        type=str,
        default="jaccard",
        choices=["jaccard", "f1", "precision", "recall"],
        help="检索强度指标（默认：jaccard）",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="置信区间显著性水平 alpha（默认 0.05 -> 95%% CI）",
    )
    parser.add_argument(
        "--tau-step",
        type=float,
        default=0.1,
        help="阈值 τ 的步长（默认 0.1）",
    )
    parser.add_argument(
        "--curve-y",
        type=str,
        default="p_value",
        choices=["p_value", "chi2"],
        help="敏感度曲线纵轴输出（默认：p_value）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认：项目根目录/data/evidence_strength/<model_dir>/）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="禁用敏感度曲线 PNG 生成（默认会自动生成）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "cot", "agent", "cot_agent", "tool"],
        help="生成模式（默认：default）。default=原版，cot=CoT，agent=Agent，cot_agent=CoT+Agent，tool 暂未实现"
    )
    parser.add_argument(
        "--tau-report",
        type=float,
        default=0.8,
        help="额外打印一个指定阈值 τ 的统计（默认 0.8；但不会影响敏感度分析）",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    results_dir = Path(args.results_dir) if args.results_dir else (project_root / "data" / "ssr_evidence_results")
    model_dirname = model_name_to_dirname(args.model_name)
    mode_suffix = get_mode_suffix(args.mode)
    model_output_dir = results_dir / f"{model_dirname}{mode_suffix}"

    # 输出目录：data/evidence_strength/<model_dir>/
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        model_dirname = model_name_to_dirname(args.model_name)
        out_dir = project_root / "data" / "evidence_strength" / model_dirname

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"结果目录: {model_output_dir}")
    logger.info(f"输出目录: {out_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"强度指标: {args.strength_metric}")

    df = load_ssr_evidence_results(model_output_dir)
    df["strength"] = df[args.strength_metric].astype(float)

    # 输出逐样本明细
    detail_csv = out_dir / "per_item.csv"
    df.sort_values(["strength", "correct"], ascending=[True, True]).to_csv(detail_csv, index=False, encoding="utf-8-sig")
    logger.info(f"逐样本明细已写入: {detail_csv}")

    # 敏感度曲线：tau ∈ [0, 0.9]，步长为 0.1
    step = float(args.tau_step)
    if step <= 0 or step > 1:
        raise ValueError("--tau-step 必须在 (0,1] 内")
    tau_max = 0.9
    taus = np.round(np.arange(0.0, tau_max + 1e-12, step), 10)
    curve: List[ThresholdStats] = [contingency_for_tau(df, float(t), alpha=float(args.alpha)) for t in taus]
    df_curve = pd.DataFrame([asdict(x) for x in curve])

    curve_csv = out_dir / "tau_sensitivity_curve.csv"
    df_curve.to_csv(curve_csv, index=False, encoding="utf-8-sig")
    logger.info(f"敏感度曲线数据已写入: {curve_csv}")

    sig = df_curve["p_value"] < 0.05
    first_sig_tau = float(df_curve.loc[sig, "tau"].iloc[0]) if sig.any() else None
    rep = contingency_for_tau(df, float(args.tau_report), alpha=float(args.alpha))

    all_tau_stats = []
    for stats in curve:
        all_tau_stats.append({
            "tau": stats.tau,
            "n_total": stats.n_total,
            "n_high": stats.n_high,
            "n_low": stats.n_low,
            "high_correct": stats.high_correct,
            "high_wrong": stats.high_wrong,
            "low_correct": stats.low_correct,
            "low_wrong": stats.low_wrong,
            "blind_guess_rate": stats.blind_guess_rate,
            "blind_guess_ci": [stats.blind_guess_ci_low, stats.blind_guess_ci_high],
            "reasoning_error_rate": stats.reasoning_error_rate,
            "reasoning_error_ci": [stats.reasoning_error_ci_low, stats.reasoning_error_ci_high],
            "reasoning_correct_rate": stats.reasoning_correct_rate,
            "reasoning_correct_ci": [stats.reasoning_correct_ci_low, stats.reasoning_correct_ci_high],
            "chi2": stats.chi2,
            "p_value": stats.p_value,
            "is_significant": bool(stats.p_value < 0.05) if not np.isnan(stats.p_value) else False,
        })

    retrieval_metrics = {
        "jaccard_mean": float(df["jaccard"].mean()) if "jaccard" in df.columns else float("nan"),
        "jaccard_median": float(df["jaccard"].median()) if "jaccard" in df.columns else float("nan"),
        "precision_mean": float(df["precision"].mean()) if "precision" in df.columns else float("nan"),
        "precision_median": float(df["precision"].median()) if "precision" in df.columns else float("nan"),
        "recall_mean": float(df["recall"].mean()) if "recall" in df.columns else float("nan"),
        "recall_median": float(df["recall"].median()) if "recall" in df.columns else float("nan"),
        "f1_mean": float(df["f1"].mean()) if "f1" in df.columns else float("nan"),
        "f1_median": float(df["f1"].median()) if "f1" in df.columns else float("nan"),
    }

    summary = {
        "model_name": args.model_name,
        "model_dir": model_output_dir.name,
        "n_items": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "strength_metric": args.strength_metric,
        "strength_mean": float(df["strength"].mean()),
        "strength_median": float(df["strength"].median()),
        "retrieval_metrics": retrieval_metrics,
        "tau_step": float(step),
        "tau_max": float(tau_max),
        "first_significant_tau_p_lt_0_05": first_sig_tau,
        "tau_report": float(args.tau_report),
        "tau_report_stats": asdict(rep),
        "all_tau_stats": all_tau_stats,
        "outputs": {"per_item_csv": str(detail_csv), "curve_csv": str(curve_csv)},
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"汇总已写入: {summary_json}")

    if not args.no_plot:
        out_png = out_dir / f"tau_curve_{args.curve_y}.png"
        if maybe_plot_sensitivity(df_curve, out_png, y=args.curve_y):
            summary["outputs"]["curve_png"] = str(out_png)
        else:
            logger.warning("未检测到 matplotlib，已跳过 PNG 绘图（但 CSV 已输出）。")

    # 控制台打印
    logger.info("=" * 80)
    logger.info(f"样本数: {len(df)}  |  准确率: {df['correct'].mean():.4f}")
    logger.info(f"强度({args.strength_metric}) 均值/中位数: {df['strength'].mean():.4f} / {df['strength'].median():.4f}")
    if "jaccard" in df.columns:
        logger.info("检索准确率（检索强度指标）:")
        logger.info(f"  Jaccard (平均/中位数): {df['jaccard'].mean():.4f} / {df['jaccard'].median():.4f}")
        if "precision" in df.columns:
            logger.info(f"  Precision (平均/中位数): {df['precision'].mean():.4f} / {df['precision'].median():.4f}")
        if "recall" in df.columns:
            logger.info(f"  Recall (平均/中位数): {df['recall'].mean():.4f} / {df['recall'].median():.4f}")
        if "f1" in df.columns:
            logger.info(f"  F1 (平均/中位数): {df['f1'].mean():.4f} / {df['f1'].median():.4f}")
    if first_sig_tau is None:
        logger.info("敏感度：在扫描的 τ 范围内未出现 p<0.05 的显著依赖。")
    else:
        logger.info(f"敏感度：首次出现显著依赖（p<0.05）的 τ ≈ {first_sig_tau:.2f}")
    if not np.isnan(rep.reasoning_correct_rate):
        logger.info(f"τ={args.tau_report:.1f} 时的关键指标:")
        logger.info(f"  推理正确率: {rep.reasoning_correct_rate:.3f} [{rep.reasoning_correct_ci_low:.3f}, {rep.reasoning_correct_ci_high:.3f}] (n_high={rep.n_high})")
        logger.info(f"  推理错误率: {rep.reasoning_error_rate:.3f} [{rep.reasoning_error_ci_low:.3f}, {rep.reasoning_error_ci_high:.3f}] (n_high={rep.n_high})")
        logger.info(f"  盲猜率: {rep.blind_guess_rate:.3f} [{rep.blind_guess_ci_low:.3f}, {rep.blind_guess_ci_high:.3f}] (n_low={rep.n_low})")
    logger.info("-" * 80)

    logger.info("所有阈值下的详细统计：")
    logger.info(
        f"{'τ':<6} {'n_total':<8} {'n_high':<8} {'n_low':<8} "
        f"{'high_correct':<12} {'high_wrong':<12} {'low_correct':<12} {'low_wrong':<12} "
        f"{'盲猜率':<12} {'推理错误率':<12} {'推理正确率':<12} {'chi2':<10} {'p-value':<10} {'显著':<6}"
    )
    for stats in curve:
        sig_mark = "是" if (not np.isnan(stats.p_value) and stats.p_value < 0.05) else "否"
        bg_rate_str = f"{stats.blind_guess_rate:.3f}" if not np.isnan(stats.blind_guess_rate) else "N/A"
        re_rate_str = f"{stats.reasoning_error_rate:.3f}" if not np.isnan(stats.reasoning_error_rate) else "N/A"
        rc_rate_str = f"{stats.reasoning_correct_rate:.3f}" if not np.isnan(stats.reasoning_correct_rate) else "N/A"
        chi2_str = f"{stats.chi2:.4f}" if not np.isnan(stats.chi2) else "N/A"
        p_str = f"{stats.p_value:.4e}" if not np.isnan(stats.p_value) else "N/A"
        logger.info(
            f"{stats.tau:<6.1f} {stats.n_total:<8} {stats.n_high:<8} {stats.n_low:<8} "
            f"{stats.high_correct:<12} {stats.high_wrong:<12} {stats.low_correct:<12} {stats.low_wrong:<12} "
            f"{bg_rate_str:<12} {re_rate_str:<12} {rc_rate_str:<12} {chi2_str:<10} {p_str:<10} {sig_mark:<6}"
        )
    logger.info("-" * 80)
    logger.info("详细置信区间：")
    for stats in curve:
        if not np.isnan(stats.blind_guess_rate):
            logger.info(
                f"τ={stats.tau:.1f}: "
                f"盲猜率={stats.blind_guess_rate:.3f} [{stats.blind_guess_ci_low:.3f}, {stats.blind_guess_ci_high:.3f}] "
                f"(n_low={stats.n_low}) | "
                f"推理错误率={stats.reasoning_error_rate:.3f} [{stats.reasoning_error_ci_low:.3f}, {stats.reasoning_error_ci_high:.3f}] "
                f"(n_high={stats.n_high}) | "
                f"推理正确率={stats.reasoning_correct_rate:.3f} [{stats.reasoning_correct_ci_low:.3f}, {stats.reasoning_correct_ci_high:.3f}] "
                f"(n_high={stats.n_high})"
            )
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

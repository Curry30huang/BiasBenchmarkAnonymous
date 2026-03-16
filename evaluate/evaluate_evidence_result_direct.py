#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证据推理正确率分析脚本（Default/Direct/Origin模式）

目标：
- 计算模型的推理正确率（准确率）
- Default模式：原始的检索模式
- Direct模式：只提供正确答案句子，不需要检索
- Origin模式：提供全量context_list句子，但不进行检索

数据来源（由 --mode 决定）：
- data/ssr_evidence_results/<model_dir>/ssr_evidence_result_*.json (default 默认模式)
- data/ssr_evidence_results_direct/<model_dir>/ssr_evidence_result_*.json (direct模式)
- data/ssr_evidence_results_origin/<model_dir>/ssr_evidence_result_*.json (origin模式)
  样例字段：
  - llm_result.risk_of_bias: str
  - ground_truth.label: str

核心定义：
- 回答正确性：使用健壮的字符串匹配策略标准化 llm_result.risk_of_bias 与 ground_truth.label 后比较。
  标准化规则（按优先级）：包含 "high"→"high"；包含 "low"→"low"；包含 "some"/"moderate"/"unclear"→"some concerns"。

统计量：
- 准确率：正确预测的比例，并给出 Wilson 置信区间。
- 按标签类别的准确率统计（low, some concerns, high）

注意：
- Direct/Origin模式不需要检索，只需要计算LLM的标签和真实标签的匹配度
- Default模式是原始的检索+推理模式
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# loguru 是项目依赖，但为保证脚本在更"干净"的环境也可运行，这里做可选导入
try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("evaluate_evidence_result_direct")


def model_name_to_dirname(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


# Direct模式不需要模式后缀，因为direct模式本身就是一个独立的模式
# 在generate_evidence.py中，direct模式下的输出目录就是 {model_dirname}，没有后缀


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


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
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

        # Direct/Origin模式不需要检查success，只需要risk_of_bias不为空即可
        risk_of_bias = llm.get("risk_of_bias")
        if not risk_of_bias or len(str(risk_of_bias).strip()) == 0:
            logger.warning(f"跳过 risk_of_bias 为空的文件: {fp.name}")
            continue

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
                "source_file": str(fp),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"目录 {model_output_dir} 下没有可用样本（可能都 risk_of_bias 为空或解析失败）。")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="SSR 证据推理正确率分析（Default/Direct/Origin模式）")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="结果文件根目录（默认：根据--mode自动选择，default->ssr_evidence_results，direct->ssr_evidence_results_direct，origin->ssr_evidence_results_origin）",
    )
    parser.add_argument("--model-name", type=str, required=True, help='模型名称，如 "openai/gpt-5.1"')
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "direct", "origin"],
        help="模式（默认：default）。default=原始检索模式，direct=只提供正确答案句子，origin=提供全量context_list但不检索"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="置信区间显著性水平 alpha（默认 0.05 -> 95%% CI）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认：项目根目录/data/evidence_strength/<model_dir>_<mode>/）",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 根据模式选择输入目录
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        if args.mode == "origin":
            results_dir = project_root / "data" / "ssr_evidence_results_origin"
        elif args.mode == "direct":
            results_dir = project_root / "data" / "ssr_evidence_results_direct"
        else:  # default
            results_dir = project_root / "data" / "ssr_evidence_results"

    model_dirname = model_name_to_dirname(args.model_name)
    # Direct/Origin模式下，模型输出目录就是 {model_dirname}，没有模式后缀
    # 因为在generate_evidence.py中，get_mode_suffix("direct"/"origin")返回""（不在已知模式列表中）
    model_output_dir = results_dir / model_dirname

    # 输出目录：data/evidence_strength/<model_dir>_<mode>/
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = project_root / "data" / "evidence_strength" / f"{model_dirname}_{args.mode}"

    out_dir.mkdir(parents=True, exist_ok=True)

    mode_name_map = {
        "default": "Default模式（原始检索）",
        "direct": "Direct模式",
        "origin": "Origin模式"
    }
    mode_name = mode_name_map.get(args.mode, args.mode)
    logger.info(f"结果目录: {model_output_dir}")
    logger.info(f"输出目录: {out_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode} ({mode_name})")

    df = load_ssr_evidence_results(model_output_dir)

    # 输出逐样本明细
    detail_csv = out_dir / "per_item.csv"
    df.sort_values(["correct", "gold_label"], ascending=[True, True]).to_csv(detail_csv, index=False, encoding="utf-8-sig")
    logger.info(f"逐样本明细已写入: {detail_csv}")

    # 计算总体准确率
    n_total = len(df)
    n_correct = int(df["correct"].sum())
    accuracy = float(df["correct"].mean())
    acc_lo, acc_hi = wilson_ci(n_correct, n_total, alpha=float(args.alpha))

    # 按标签类别统计准确率
    label_stats = []
    for label in ["low", "some concerns", "high"]:
        label_df = df[df["gold_label"] == label]
        if len(label_df) > 0:
            n_label = len(label_df)
            n_label_correct = int(label_df["correct"].sum())
            label_acc = float(label_df["correct"].mean())
            label_acc_lo, label_acc_hi = wilson_ci(n_label_correct, n_label, alpha=float(args.alpha))
            label_stats.append({
                "label": label,
                "n_total": n_label,
                "n_correct": n_label_correct,
                "accuracy": label_acc,
                "accuracy_ci_low": label_acc_lo,
                "accuracy_ci_high": label_acc_hi,
            })

    # 按预测标签统计（可选，用于分析模型预测分布）
    pred_label_stats = []
    for label in ["low", "some concerns", "high"]:
        label_df = df[df["pred_label"] == label]
        if len(label_df) > 0:
            n_label = len(label_df)
            n_label_correct = int(label_df["correct"].sum())
            label_acc = float(label_df["correct"].mean())
            pred_label_stats.append({
                "label": label,
                "n_total": n_label,
                "n_correct": n_label_correct,
                "accuracy": label_acc,
            })

    # 混淆矩阵
    confusion_matrix = pd.crosstab(
        df["gold_label"],
        df["pred_label"],
        margins=True,
        margins_name="总计"
    )

    confusion_csv = out_dir / "confusion_matrix.csv"
    confusion_matrix.to_csv(confusion_csv, encoding="utf-8-sig")
    logger.info(f"混淆矩阵已写入: {confusion_csv}")

    summary = {
        "model_name": args.model_name,
        "model_dir": model_output_dir.name,
        "n_items": int(n_total),
        "n_correct": int(n_correct),
        "accuracy": float(accuracy),
        "accuracy_ci_low": float(acc_lo),
        "accuracy_ci_high": float(acc_hi),
        "label_stats": label_stats,  # 按真实标签统计
        "pred_label_stats": pred_label_stats,  # 按预测标签统计
        "outputs": {
            "per_item_csv": str(detail_csv),
            "confusion_matrix_csv": str(confusion_csv),
        },
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"汇总已写入: {summary_json}")

    # 控制台打印关键结论
    logger.info("=" * 80)
    logger.info(f"样本数: {n_total}  |  准确率: {accuracy:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
    logger.info("-" * 80)
    logger.info("按真实标签类别的准确率统计：")
    for stat in label_stats:
        logger.info(
            f"  {stat['label']}: {stat['accuracy']:.4f} "
            f"[{stat['accuracy_ci_low']:.4f}, {stat['accuracy_ci_high']:.4f}] "
            f"(n={stat['n_total']}, 正确={stat['n_correct']})"
        )
    logger.info("-" * 80)
    logger.info("混淆矩阵：")
    logger.info("\n" + str(confusion_matrix))
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

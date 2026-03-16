#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
评估模块一致性的脚本

模块一致性主要衡量 LLM 的"直觉判断"与"逻辑推导"之间的裂痕。
使用准确率公式评价 LLM 神经推理结果与符号推理结果一致性。

需要计算三个指标（按domain区分）：
1. 逻辑保真度：比较LLM生成的domain_risk和逻辑推理得到的domain_risk的匹配度
2. 逻辑结果准确率：逻辑推理得到的domain_risk相较于ground_truth的准确率
3. LLM结果准确率：评估LLM生成的domain_risk与ground_truth的准确率

显著性验证方法：McNemar's Test (麦克内马尔检验)
- H_0: 模型"逻辑保真度高但LLM结果准确率低"的比例与"逻辑保真度低但LLM结果准确率高"的比例没有显著差异
- H_1: 模型虽然结果准确率高，但逻辑不一致（逻辑保真度低）的比例显著更高
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats
from loguru import logger

# 导入判断函数和枚举类型
from bias_assessment.rct_bias_assessment_copy import (
    rct_1_randomization_process_judgement,
    rct_2_intended_intervention_judgement,
    rct_3_missing_outcome_judgement,
    rct_4_measurement_of_the_outcome_judgement,
    rct_5_selection_of_the_reported_result_judgement,
)
from bias_assessment.evidence_entity import RCTRisk

# Domain key映射：从结果文件中的domain key到判断函数和ground truth key
DOMAIN_MAPPING = {
    "randomisation_process": {
        "judgement_func": rct_1_randomization_process_judgement,
        "ground_truth_key": "randomisation_process_judgment",
    },
    "intended_interventions": {
        "judgement_func": rct_2_intended_intervention_judgement,
        "ground_truth_key": "intended_interventions_judgment",
    },
    "missing_outcome_data": {
        "judgement_func": rct_3_missing_outcome_judgement,
        "ground_truth_key": "missing_outcome_data_judgment",
    },
    "measurement_outcome": {
        "judgement_func": rct_4_measurement_of_the_outcome_judgement,
        "ground_truth_key": "measurement_outcome_judgment",
    },
    "selection_reported_result": {
        "judgement_func": rct_5_selection_of_the_reported_result_judgement,
        "ground_truth_key": "selection_reported_result_judgment",
    },
}


def model_name_to_dirname(model_name: str) -> str:
    """
    将模型名称转换为安全的目录名（处理特殊字符）

    Args:
        model_name: 模型名称，如 "openai/gpt-5.1"

    Returns:
        安全的目录名，如 "openai_gpt-5.1"
    """
    return model_name.replace("/", "_").replace("\\", "_")


def get_mode_suffix(mode: str = "default") -> str:
    """
    根据模式获取目录后缀

    Args:
        mode: 模式，可选值：default, cot, agent, tool

    Returns:
        目录后缀，如 "_cot" 或 ""
    """
    if mode == "default" or mode is None or mode == "":
        return ""
    elif mode == "cot":
        return "_cot"
    elif mode == "agent":
        return "_agent"
    elif mode == "tool":
        return "_tool"
    else:
        logger.warning(f"未知模式 '{mode}'，使用默认模式")
        return ""


def convert_to_native_type(value: Any) -> Any:
    """
    将 numpy 类型转换为 Python 原生类型，以便 JSON 序列化

    Args:
        value: 可能是 numpy 类型的值

    Returns:
        Python 原生类型
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (list, tuple)):
        return type(value)(convert_to_native_type(item) for item in value)
    elif isinstance(value, dict):
        return {k: convert_to_native_type(v) for k, v in value.items()}
    return value


def load_result_files(result_dir: Path) -> List[Dict[str, Any]]:
    """
    加载所有结果文件

    Args:
        result_dir: 结果目录路径

    Returns:
        结果文件列表
    """
    results = []
    json_files = sorted(result_dir.glob("rct_bias_result_*.json"))

    logger.info(f"找到 {len(json_files)} 个结果文件")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            logger.warning(f"加载文件失败 {json_file.name}: {e}")

    return results


def string_to_rct_risk(risk_str: str) -> Optional[RCTRisk]:
    """
    将字符串转换为RCTRisk枚举

    Args:
        risk_str: 风险字符串

    Returns:
        RCTRisk枚举值，如果无法匹配则返回None
    """
    if not risk_str:
        return None

    risk_str = risk_str.strip()
    for risk in RCTRisk:
        if risk.value.lower() == risk_str.lower():
            return risk
    return None


def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算Wilson score置信区间（适用于比例估计）

    Args:
        successes: 成功次数
        n: 总次数
        confidence: 置信水平（默认0.95，即95%置信区间）

    Returns:
        (lower_bound, upper_bound) 置信区间的上下界
    """
    if n == 0:
        return (0.0, 0.0)

    if successes == 0:
        # 特殊情况：没有成功
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = 0.0
        denominator = 1 + (z ** 2) / n
        center = (p_hat + (z ** 2) / (2 * n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * n)) / n) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))

    if successes == n:
        # 特殊情况：全部成功
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = 1.0
        denominator = 1 + (z ** 2) / n
        center = (p_hat + (z ** 2) / (2 * n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * n)) / n) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))

    # 标准Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n

    denominator = 1 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * n)) / n) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)


def mcnemar_test_from_pairs(
    logical_correct: List[bool],
    llm_correct: List[bool]
) -> Tuple[float, float]:
    """
    从配对数据执行McNemar's Test 验证的是：模型是否在结果准确率高但逻辑不一致（逻辑保真度低）的情况下存在系统性偏差。

    Args:
        logical_correct: 逻辑保真度：LLM结果与逻辑推理结果是否匹配
        llm_correct: LLM结果准确率：LLM结果与ground_truth是否匹配

    Returns:
        (统计量, p_value) 元组
    """
    if len(logical_correct) != len(llm_correct):
        raise ValueError("配对数据长度必须相同")

    # 构建2x2列联表
    a = 0  # 逻辑保真度高且LLM结果准确率高
    b = 0  # 逻辑保真度高但LLM结果准确率低
    c = 0  # 逻辑保真度低但LLM结果准确率高
    d = 0  # 逻辑保真度低且LLM结果准确率低

    for l_correct, llm_correct_val in zip(logical_correct, llm_correct):
        if l_correct and llm_correct_val:
            a += 1
        elif l_correct and not llm_correct_val:
            b += 1
        elif not l_correct and llm_correct_val:
            c += 1
        else:
            d += 1

    # 使用McNemar's test
    # McNemar test 用于配对数据的二分类比较
    # H0: b = c (即逻辑保真度高但LLM结果准确率低的比例 = 逻辑保真度低但LLM结果准确率高的比例)

    if b + c == 0:
        # 没有不一致的对，p值为1（完全一致）
        statistic = 0.0
        p_value = 1.0
    else:
        # 使用精确二项检验（McNemar test的精确版本）
        # H0: b = c，即在不一致的对中，b 和 c 的概率相等（各为0.5）
        n_discordant = b + c
        n_minor = min(b, c)

        # 使用二项检验：H0: p = 0.5
        binom_result = stats.binomtest(n_minor, n_discordant, p=0.5, alternative='two-sided')
        p_value = binom_result.pvalue

        # 计算McNemar统计量（带连续性校正）
        # 统计量 = (|b - c| - 1)^2 / (b + c)
        statistic = ((abs(b - c) - 1) ** 2) / (b + c) if b + c > 0 else 0.0

    return float(statistic), float(p_value)


def evaluate_domain_consistency(
    result_dir: Path,
    model_name: str
) -> Dict[str, Any]:
    """
    评估模块一致性

    Args:
        result_dir: 结果目录路径
        model_name: 模型名称

    Returns:
        评估结果字典
    """
    # 加载所有结果文件
    results = load_result_files(result_dir)

    if len(results) == 0:
        logger.error("未找到任何结果文件")
        return {}

    # 统计信息：按 domain 分组
    domain_stats = defaultdict(lambda: {
        "total_cases": 0,
        "logical_fidelity_correct": 0,  # 逻辑保真度：LLM结果与逻辑推理结果匹配
        "logical_accuracy_correct": 0,   # 逻辑结果准确率：逻辑推理结果与ground_truth匹配
        "llm_accuracy_correct": 0,       # LLM结果准确率：LLM结果与ground_truth匹配
        "logical_correct_pairs": [],     # 逻辑保真度：LLM结果与逻辑推理结果匹配（配对数据，用于McNemar test）
        "llm_correct_pairs": [],         # LLM结果准确率：LLM结果与ground_truth匹配（配对数据，用于McNemar test）
        "details": []                    # 详细记录
    })

    # 遍历所有结果
    for result in results:
        llm_result = result.get("llm_result", {})
        ground_truth = result.get("ground_truth", {})
        metadata = result.get("metadata", {})
        record_id = metadata.get("id", "unknown")

        # 遍历每个 domain
        for domain_key, domain_config in DOMAIN_MAPPING.items():
            domain_data = llm_result.get(domain_key, {})
            signalling_questions = domain_data.get("signaling_questions", [])
            llm_domain_risk_str = domain_data.get("domain_risk", "")
            ground_truth_key = domain_config["ground_truth_key"]
            ground_truth_risk_str = ground_truth.get(ground_truth_key, "")

            # 检查必要数据是否存在
            if not signalling_questions:
                logger.warning(
                    f"记录 ID: {record_id}, Domain: {domain_key}, "
                    f"缺少 signaling_questions，跳过"
                )
                continue

            if not llm_domain_risk_str:
                logger.warning(
                    f"记录 ID: {record_id}, Domain: {domain_key}, "
                    f"缺少 LLM domain_risk，跳过"
                )
                continue

            if not ground_truth_risk_str:
                logger.warning(
                    f"记录 ID: {record_id}, Domain: {domain_key}, "
                    f"缺少 ground_truth，跳过"
                )
                continue

            # 使用逻辑推理函数计算逻辑domain_risk
            try:
                judgement_func = domain_config["judgement_func"]
                logical_domain_risk = judgement_func(signalling_questions)
                logical_domain_risk_str = logical_domain_risk.value
            except Exception as e:
                logger.error(
                    f"记录 ID: {record_id}, Domain: {domain_key}, "
                    f"逻辑推理失败: {e}"
                )
                continue

            # 转换为枚举类型以便比较
            llm_domain_risk = string_to_rct_risk(llm_domain_risk_str)
            logical_domain_risk_enum = logical_domain_risk
            ground_truth_risk = string_to_rct_risk(ground_truth_risk_str)

            if llm_domain_risk is None:
                logger.warning(
                    f"记录 ID: {record_id}, Domain: {domain_key}, "
                    f"无法解析LLM domain_risk: {llm_domain_risk_str}"
                )
                continue

            if ground_truth_risk is None:
                logger.warning(
                    f"记录 ID: {record_id}, Domain: {domain_key}, "
                    f"无法解析ground_truth: {ground_truth_risk_str}"
                )
                continue

            # 统计
            domain_stats[domain_key]["total_cases"] += 1

            # 1. 逻辑保真度：LLM结果与逻辑推理结果匹配
            logical_fidelity_match = (llm_domain_risk == logical_domain_risk_enum)
            if logical_fidelity_match:
                domain_stats[domain_key]["logical_fidelity_correct"] += 1

            # 2. 逻辑结果准确率：逻辑推理结果与ground_truth匹配
            logical_accuracy_match = (logical_domain_risk_enum == ground_truth_risk)
            if logical_accuracy_match:
                domain_stats[domain_key]["logical_accuracy_correct"] += 1

            # 3. LLM结果准确率：LLM结果与ground_truth匹配
            llm_accuracy_match = (llm_domain_risk == ground_truth_risk)
            if llm_accuracy_match:
                domain_stats[domain_key]["llm_accuracy_correct"] += 1

            # 保存配对数据用于McNemar test
            # 配对：逻辑保真度（LLM结果与逻辑推理结果匹配）和LLM结果准确率（LLM结果与ground_truth匹配）
            domain_stats[domain_key]["logical_correct_pairs"].append(logical_fidelity_match)
            domain_stats[domain_key]["llm_correct_pairs"].append(llm_accuracy_match)

            # 保存详细记录
            domain_stats[domain_key]["details"].append({
                "record_id": record_id,
                "llm_domain_risk": llm_domain_risk_str,
                "logical_domain_risk": logical_domain_risk_str,
                "ground_truth_risk": ground_truth_risk_str,
                "logical_fidelity_match": logical_fidelity_match,
                "logical_accuracy_match": logical_accuracy_match,
                "llm_accuracy_match": llm_accuracy_match,
            })

    # 计算每个 domain 的指标和置信区间
    evaluation_results = {}

    for domain_key, domain_stat in domain_stats.items():
        total = domain_stat["total_cases"]

        if total == 0:
            logger.warning(f"Domain {domain_key} 没有有效案例")
            continue

        # 1. 逻辑保真度
        logical_fidelity_correct = domain_stat["logical_fidelity_correct"]
        logical_fidelity = logical_fidelity_correct / total if total > 0 else 0.0
        logical_fidelity_ci = wilson_score_interval(
            logical_fidelity_correct, total, confidence=0.95
        )

        # 2. 逻辑结果准确率
        logical_accuracy_correct = domain_stat["logical_accuracy_correct"]
        logical_accuracy = logical_accuracy_correct / total if total > 0 else 0.0
        logical_accuracy_ci = wilson_score_interval(
            logical_accuracy_correct, total, confidence=0.95
        )

        # 3. LLM结果准确率
        llm_accuracy_correct = domain_stat["llm_accuracy_correct"]
        llm_accuracy = llm_accuracy_correct / total if total > 0 else 0.0
        llm_accuracy_ci = wilson_score_interval(
            llm_accuracy_correct, total, confidence=0.95
        )

        # McNemar's Test：比较逻辑保真度和LLM结果准确率
        logical_correct_pairs = domain_stat["logical_correct_pairs"]
        llm_correct_pairs = domain_stat["llm_correct_pairs"]

        if len(logical_correct_pairs) > 0 and len(llm_correct_pairs) > 0:
            try:
                mcnemar_statistic, mcnemar_p_value = mcnemar_test_from_pairs(
                    logical_correct_pairs, llm_correct_pairs
                )
            except Exception as e:
                logger.warning(f"Domain {domain_key} McNemar test失败: {e}")
                mcnemar_statistic = None
                mcnemar_p_value = None
        else:
            mcnemar_statistic = None
            mcnemar_p_value = None

        # 解释McNemar test结果
        # 比较逻辑保真度（LLM结果与逻辑推理结果匹配）和LLM结果准确率（LLM结果与ground_truth匹配）
        if mcnemar_p_value is not None:
            if mcnemar_p_value < 0.05:
                if llm_accuracy > logical_fidelity:
                    interpretation = (
                        f"模型结果准确率显著高于逻辑保真度（p={mcnemar_p_value:.6f}），"
                        f"说明模型虽然结果正确（准确率={llm_accuracy:.4f}），但逻辑不一致（保真度={logical_fidelity:.4f}）"
                    )
                else:
                    interpretation = (
                        f"逻辑保真度显著高于模型结果准确率（p={mcnemar_p_value:.6f}），"
                        f"说明模型逻辑一致（保真度={logical_fidelity:.4f}），但结果准确率较低（准确率={llm_accuracy:.4f}）"
                    )
            else:
                interpretation = (
                    f"逻辑保真度与模型结果准确率无显著差异（p={mcnemar_p_value:.6f}）"
                )
        else:
            interpretation = "无法计算McNemar test"

        # 确保所有值都是 Python 原生类型，以便 JSON 序列化
        evaluation_results[domain_key] = {
            "total_cases": int(total),
            "logical_fidelity": {
                "correct": int(logical_fidelity_correct),
                "accuracy": float(logical_fidelity),
                "ci_95": (float(logical_fidelity_ci[0]), float(logical_fidelity_ci[1])),
            },
            "logical_accuracy": {
                "correct": int(logical_accuracy_correct),
                "accuracy": float(logical_accuracy),
                "ci_95": (float(logical_accuracy_ci[0]), float(logical_accuracy_ci[1])),
            },
            "llm_accuracy": {
                "correct": int(llm_accuracy_correct),
                "accuracy": float(llm_accuracy),
                "ci_95": (float(llm_accuracy_ci[0]), float(llm_accuracy_ci[1])),
            },
            "mcnemar_test": {
                "statistic": float(mcnemar_statistic) if mcnemar_statistic is not None else None,
                "p_value": float(mcnemar_p_value) if mcnemar_p_value is not None else None,
                "is_significant": bool(mcnemar_p_value < 0.05) if mcnemar_p_value is not None else None,
                "interpretation": interpretation,
            }
        }

        logger.info(f"Domain {domain_key}:")
        logger.info(f"  总案例数: {total}")
        logger.info(f"  逻辑保真度: {logical_fidelity:.4f} (95% CI: [{logical_fidelity_ci[0]:.4f}, {logical_fidelity_ci[1]:.4f}])")
        logger.info(f"  逻辑结果准确率: {logical_accuracy:.4f} (95% CI: [{logical_accuracy_ci[0]:.4f}, {logical_accuracy_ci[1]:.4f}])")
        logger.info(f"  LLM结果准确率: {llm_accuracy:.4f} (95% CI: [{llm_accuracy_ci[0]:.4f}, {llm_accuracy_ci[1]:.4f}])")
        if mcnemar_p_value is not None:
            logger.info(f"  McNemar test: statistic={mcnemar_statistic:.4f}, p-value={mcnemar_p_value:.6f}")
            logger.info(f"  统计显著性: {'是' if mcnemar_p_value < 0.05 else '否'}")

    # 计算总体统计
    total_cases_all = sum(s["total_cases"] for s in domain_stats.values())
    total_logical_fidelity_all = sum(s["logical_fidelity_correct"] for s in domain_stats.values())
    total_logical_accuracy_all = sum(s["logical_accuracy_correct"] for s in domain_stats.values())
    total_llm_accuracy_all = sum(s["llm_accuracy_correct"] for s in domain_stats.values())

    # 收集所有domain的配对数据用于总体McNemar test
    all_logical_correct_pairs = []
    all_llm_correct_pairs = []
    for domain_stat in domain_stats.values():
        all_logical_correct_pairs.extend(domain_stat["logical_correct_pairs"])
        all_llm_correct_pairs.extend(domain_stat["llm_correct_pairs"])

    if total_cases_all > 0:
        overall_logical_fidelity = total_logical_fidelity_all / total_cases_all
        overall_logical_fidelity_ci = wilson_score_interval(
            total_logical_fidelity_all, total_cases_all, confidence=0.95
        )

        overall_logical_accuracy = total_logical_accuracy_all / total_cases_all
        overall_logical_accuracy_ci = wilson_score_interval(
            total_logical_accuracy_all, total_cases_all, confidence=0.95
        )

        overall_llm_accuracy = total_llm_accuracy_all / total_cases_all
        overall_llm_accuracy_ci = wilson_score_interval(
            total_llm_accuracy_all, total_cases_all, confidence=0.95
        )

        # 总体McNemar test
        if len(all_logical_correct_pairs) > 0 and len(all_llm_correct_pairs) > 0:
            try:
                overall_mcnemar_statistic, overall_mcnemar_p_value = mcnemar_test_from_pairs(
                    all_logical_correct_pairs, all_llm_correct_pairs
                )
            except Exception as e:
                logger.warning(f"总体McNemar test失败: {e}")
                overall_mcnemar_statistic = None
                overall_mcnemar_p_value = None
        else:
            overall_mcnemar_statistic = None
            overall_mcnemar_p_value = None

        # 解释总体McNemar test结果
        # 比较逻辑保真度（LLM结果与逻辑推理结果匹配）和LLM结果准确率（LLM结果与ground_truth匹配）
        if overall_mcnemar_p_value is not None:
            if overall_mcnemar_p_value < 0.05:
                if overall_llm_accuracy > overall_logical_fidelity:
                    overall_interpretation = (
                        f"模型结果准确率显著高于逻辑保真度（p={overall_mcnemar_p_value:.6f}），"
                        f"说明模型虽然结果正确（准确率={overall_llm_accuracy:.4f}），但逻辑不一致（保真度={overall_logical_fidelity:.4f}）"
                    )
                else:
                    overall_interpretation = (
                        f"逻辑保真度显著高于模型结果准确率（p={overall_mcnemar_p_value:.6f}），"
                        f"说明模型逻辑一致（保真度={overall_logical_fidelity:.4f}），但结果准确率较低（准确率={overall_llm_accuracy:.4f}）"
                    )
            else:
                overall_interpretation = (
                    f"逻辑保真度与模型结果准确率无显著差异（p={overall_mcnemar_p_value:.6f}）"
                )
        else:
            overall_interpretation = "无法计算McNemar test"

        evaluation_results["overall"] = {
            "total_cases": int(total_cases_all),
            "logical_fidelity": {
                "correct": int(total_logical_fidelity_all),
                "accuracy": float(overall_logical_fidelity),
                "ci_95": (float(overall_logical_fidelity_ci[0]), float(overall_logical_fidelity_ci[1])),
            },
            "logical_accuracy": {
                "correct": int(total_logical_accuracy_all),
                "accuracy": float(overall_logical_accuracy),
                "ci_95": (float(overall_logical_accuracy_ci[0]), float(overall_logical_accuracy_ci[1])),
            },
            "llm_accuracy": {
                "correct": int(total_llm_accuracy_all),
                "accuracy": float(overall_llm_accuracy),
                "ci_95": (float(overall_llm_accuracy_ci[0]), float(overall_llm_accuracy_ci[1])),
            },
            "mcnemar_test": {
                "statistic": float(overall_mcnemar_statistic) if overall_mcnemar_statistic is not None else None,
                "p_value": float(overall_mcnemar_p_value) if overall_mcnemar_p_value is not None else None,
                "is_significant": bool(overall_mcnemar_p_value < 0.05) if overall_mcnemar_p_value is not None else None,
                "interpretation": overall_interpretation,
            }
        }

        logger.info("=" * 80)
        logger.info("总体统计:")
        logger.info(f"  总案例数: {total_cases_all}")
        logger.info(f"  总体逻辑保真度: {overall_logical_fidelity:.4f} (95% CI: [{overall_logical_fidelity_ci[0]:.4f}, {overall_logical_fidelity_ci[1]:.4f}])")
        logger.info(f"  总体逻辑结果准确率: {overall_logical_accuracy:.4f} (95% CI: [{overall_logical_accuracy_ci[0]:.4f}, {overall_logical_accuracy_ci[1]:.4f}])")
        logger.info(f"  总体LLM结果准确率: {overall_llm_accuracy:.4f} (95% CI: [{overall_llm_accuracy_ci[0]:.4f}, {overall_llm_accuracy_ci[1]:.4f}])")
        if overall_mcnemar_p_value is not None:
            logger.info(f"  总体McNemar test: statistic={overall_mcnemar_statistic:.4f}, p-value={overall_mcnemar_p_value:.6f}")
            logger.info(f"  统计显著性: {'是' if overall_mcnemar_p_value < 0.05 else '否'}")
        logger.info("=" * 80)

    return {
        "model_name": model_name,
        "total_files": len(results),
        "evaluation_results": evaluation_results,
    }


def print_summary(evaluation_result: Dict[str, Any]):
    """
    打印评估结果摘要

    Args:
        evaluation_result: 评估结果字典
    """
    logger.info("=" * 80)
    logger.info("模块一致性评估结果摘要")
    logger.info("=" * 80)
    logger.info(f"模型名称: {evaluation_result.get('model_name', 'N/A')}")
    logger.info(f"总文件数: {evaluation_result.get('total_files', 0)}")
    logger.info("=" * 80)

    eval_results = evaluation_result.get("evaluation_results", {})

    # 打印表格格式的结果
    domains = [k for k in eval_results.keys() if k != "overall"]
    if not domains:
        logger.info("没有评估结果")
        return

    # 表头
    header = (
        f"{'Domain':<30} {'Cases':<10} "
        f"{'Logical Fidelity':<20} {'Logical Acc':<20} {'LLM Acc':<20} "
        f"{'McNemar p-value':<18} {'Significant':<12}"
    )
    logger.info(header)
    logger.info("-" * 150)

    # 数据行
    for domain in domains:
        result = eval_results[domain]
        logical_fidelity = result["logical_fidelity"]["accuracy"]
        logical_accuracy = result["logical_accuracy"]["accuracy"]
        llm_accuracy = result["llm_accuracy"]["accuracy"]
        mcnemar_p = result["mcnemar_test"]["p_value"]
        sig = "是" if result["mcnemar_test"]["is_significant"] else "否" if mcnemar_p is not None else "N/A"

        logical_fidelity_ci = result["logical_fidelity"]["ci_95"]
        logical_accuracy_ci = result["logical_accuracy"]["ci_95"]
        llm_accuracy_ci = result["llm_accuracy"]["ci_95"]

        # 格式化置信区间字符串
        logical_fidelity_str = f"{logical_fidelity:.4f} [{logical_fidelity_ci[0]:.4f}, {logical_fidelity_ci[1]:.4f}]"
        logical_accuracy_str = f"{logical_accuracy:.4f} [{logical_accuracy_ci[0]:.4f}, {logical_accuracy_ci[1]:.4f}]"
        llm_accuracy_str = f"{llm_accuracy:.4f} [{llm_accuracy_ci[0]:.4f}, {llm_accuracy_ci[1]:.4f}]"

        # 格式化McNemar p-value
        mcnemar_p_str = f"{mcnemar_p:.6f}" if mcnemar_p is not None else "N/A"

        row = (
            f"{domain:<30} "
            f"{result['total_cases']:<10} "
            f"{logical_fidelity_str:<20} "
            f"{logical_accuracy_str:<20} "
            f"{llm_accuracy_str:<20} "
            f"{mcnemar_p_str:<18} "
            f"{sig:<12}"
        )
        logger.info(row)

    # 打印总体结果
    if "overall" in eval_results:
        logger.info("-" * 150)
        overall = eval_results["overall"]
        logical_fidelity = overall["logical_fidelity"]["accuracy"]
        logical_accuracy = overall["logical_accuracy"]["accuracy"]
        llm_accuracy = overall["llm_accuracy"]["accuracy"]
        mcnemar_p = overall["mcnemar_test"]["p_value"]
        sig = "是" if overall["mcnemar_test"]["is_significant"] else "否" if mcnemar_p is not None else "N/A"

        logical_fidelity_ci = overall["logical_fidelity"]["ci_95"]
        logical_accuracy_ci = overall["logical_accuracy"]["ci_95"]
        llm_accuracy_ci = overall["llm_accuracy"]["ci_95"]

        # 格式化置信区间字符串
        logical_fidelity_str = f"{logical_fidelity:.4f} [{logical_fidelity_ci[0]:.4f}, {logical_fidelity_ci[1]:.4f}]"
        logical_accuracy_str = f"{logical_accuracy:.4f} [{logical_accuracy_ci[0]:.4f}, {logical_accuracy_ci[1]:.4f}]"
        llm_accuracy_str = f"{llm_accuracy:.4f} [{llm_accuracy_ci[0]:.4f}, {llm_accuracy_ci[1]:.4f}]"

        # 格式化McNemar p-value
        mcnemar_p_str = f"{mcnemar_p:.6f}" if mcnemar_p is not None else "N/A"

        row = (
            f"{'Overall':<30} "
            f"{overall['total_cases']:<10} "
            f"{logical_fidelity_str:<20} "
            f"{logical_accuracy_str:<20} "
            f"{llm_accuracy_str:<20} "
            f"{mcnemar_p_str:<18} "
            f"{sig:<12}"
        )
        logger.info(row)

    logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估模块一致性"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="结果目录路径（默认：项目根目录/data/rct_bias_results/{model_name}）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称（必需）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出JSON文件路径（默认：项目根目录/data/domain_consistency_{model_name}.json）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "cot", "agent", "tool"],
        help="生成模式（默认：default）。default=原版模式，cot=CoT模式，agent和tool模式暂未实现"
    )

    args = parser.parse_args()

    # 获取脚本所在目录和项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 设置默认路径
    if args.result_dir is None:
        model_dirname = model_name_to_dirname(args.model_name)
        mode_suffix = get_mode_suffix(args.mode)
        result_dir = project_root / "data" / "rct_bias_results" / f"{model_dirname}{mode_suffix}"
    else:
        result_dir = Path(args.result_dir)

    if args.output is None:
        # 默认输出到 data 目录，文件名包含模型名称
        model_dirname = model_name_to_dirname(args.model_name)
        output_file = project_root / "data" / f"domain_consistency_{model_dirname}.json"
    else:
        output_file = Path(args.output)

    # 检查路径是否存在
    if not result_dir.exists():
        logger.error(f"结果目录不存在: {result_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("评估模块一致性")
    logger.info("=" * 80)
    logger.info(f"结果目录: {result_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"输出文件: {output_file}")
    logger.info("=" * 80)

    # 执行评估
    try:
        evaluation_result = evaluate_domain_consistency(result_dir, args.model_name)

        # 打印摘要
        print_summary(evaluation_result)

        # 保存结果
        output_result = convert_to_native_type(evaluation_result)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, ensure_ascii=False, indent=2)

        logger.info(f"评估结果已保存到: {output_file}")

        return 0

    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.exception(e)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
评估信号问题原子一致性的脚本

原子一致性主要衡量模型在处理 "If-Then" 分支逻辑时的严谨性，
即对于所有具有前置条件的问题，是否能正确进行前置条件判断。

判断规则：
- 如果前置条件不满足 → 应该输出 NA（正确）
- 如果前置条件满足 → 应该输出非 NA（正确）
- 其他情况 → 错误

计算公式：
CAR = \frac{\sum_{i=1}^N \mathbb{I}(\text{正确判断})}{\sum_{i=1}^N \mathbb{I}(\text{具有前置条件的问题})}

其中正确判断定义为：
- 前置条件不满足且输出 NA，或
- 前置条件满足且输出非 NA

假设验证：使用二项分布检验
H_0: 模型正确判断的概率与随机判断的概率（0.5）无异。
如果 p-value < 0.05，拒绝 H_0，说明模型正确判断的概率显著不同于随机（0.5）。
结合 CAR 值判断：
- 如果 CAR 接近 1.0，说明模型显著地更倾向于正确判断（符合预期，模型遵守规则）
- 如果 CAR 接近 0，说明模型显著地更不倾向于正确判断（违背预期，模型没有遵守规则）
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats
from loguru import logger


# 定义每个 domain 的 if 条件规则
# 格式: {question_id: {precondition: [前置问题ID列表], condition: [满足条件的答案代码列表]}}
ATOMIC_CONSISTENCY_RULES = {
    "randomisation_process": {
        # Domain 1 没有 if 条件的问题
    },
    "intended_interventions": {
        "2.3": {
            "precondition": ["2.1", "2.2"],
            "condition": ["Y", "PY", "NI"],
            "description": "If Y/PY/NI to 2.1 or 2.2"
        },
        "2.4": {
            "precondition": ["2.3"],
            "condition": ["Y", "PY", "NI"],
            "description": "If Y/PY/NI to 2.3"
        },
        "2.5": {
            "precondition": ["2.4"],
            "condition": ["Y", "PY"],
            "description": "If Y/PY to 2.4"
        },
        "2.7": {
            "precondition": ["2.6"],
            "condition": ["N", "PN", "NI"],
            "description": "If N/PN/NI to 2.6"
        }
    },
    "missing_outcome_data": {
        "3.2": {
            "precondition": ["3.1"],
            "condition": ["N", "PN", "NI"],
            "description": "If N/PN/NI to 3.1"
        },
        "3.3": {
            "precondition": ["3.2"],
            "condition": ["N", "PN"],
            "description": "If N/PN to 3.2"
        },
        "3.4": {
            "precondition": ["3.3"],
            "condition": ["Y", "PY", "NI"],
            "description": "If Y/PY/NI to 3.3"
        }
    },
    "measurement_outcome": {
        "4.3": {
            "precondition": ["4.1", "4.2"],
            "condition": ["N", "PN", "NI"],
            "description": "If N/PN/NI to 4.1 and 4.2",
            "logic": "AND"  # 需要同时满足
        },
        "4.4": {
            "precondition": ["4.3"],
            "condition": ["Y", "PY", "NI"],
            "description": "If Y/PY/NI to 4.3"
        },
        "4.5": {
            "precondition": ["4.4"],
            "condition": ["Y", "PY", "NI"],
            "description": "If Y/PY/NI to 4.4"
        }
    },
    "selection_reported_result": {
        # Domain 5 没有 if 条件的问题
    }
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


def get_expected_question_ids(domain_key: str) -> List[str]:
    """
    根据domain key推断所有预期的问题ID列表

    Args:
        domain_key: domain键，如 "intended_interventions"

    Returns:
        预期的问题ID列表，如 ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"]
    """
    # 根据domain key推断domain编号
    domain_num_map = {
        "randomisation_process": 1,
        "intended_interventions": 2,
        "missing_outcome_data": 3,
        "measurement_outcome": 4,
        "selection_reported_result": 5,
    }

    domain_num = domain_num_map.get(domain_key)
    if domain_num is None:
        return []

    # 根据domain编号和规则推断所有问题ID
    # 从规则中提取所有出现的问题ID，并推断完整序列
    all_question_ids = set()
    if domain_key in ATOMIC_CONSISTENCY_RULES:
        for q_id, rule in ATOMIC_CONSISTENCY_RULES[domain_key].items():
            all_question_ids.add(q_id)
            all_question_ids.update(rule.get("precondition", []))

    if not all_question_ids:
        # 如果没有规则，返回空列表（让调用者处理）
        return []

    # 提取所有问题编号并排序
    question_nums = []
    for q_id in all_question_ids:
        try:
            # 提取问题编号，如 "2.3" -> 3
            num = int(q_id.split(".")[1])
            question_nums.append(num)
        except (ValueError, IndexError):
            continue

    if not question_nums:
        return []

    # 生成从最小到最大的完整序列
    min_num = min(question_nums)
    max_num = max(question_nums)
    expected_ids = [f"{domain_num}.{i}" for i in range(min_num, max_num + 1)]

    return expected_ids


def fill_missing_ids(signaling_questions: List[Dict[str, Any]], domain_key: str) -> List[Dict[str, Any]]:
    """
    补全信号问题列表中缺失的ID

    Args:
        signaling_questions: 信号问题列表（可能包含空ID）
        domain_key: domain键，用于推断预期的问题ID

    Returns:
        补全后的信号问题列表
    """
    if not signaling_questions:
        return signaling_questions

    # 获取预期的问题ID列表
    expected_ids = get_expected_question_ids(domain_key)
    if not expected_ids:
        # 如果无法推断，返回原列表
        return signaling_questions

    # 创建补全后的列表
    filled_questions = []
    expected_index = 0  # 预期ID的索引

    for i, q in enumerate(signaling_questions):
        q_id = q.get("id")
        q_id_str = str(q_id).strip() if q_id else ""

        if q_id_str and q_id_str in expected_ids:
            # 如果ID存在且有效，直接添加
            filled_questions.append(q.copy())
            # 更新预期索引到当前ID之后
            if q_id_str in expected_ids:
                expected_index = expected_ids.index(q_id_str) + 1
        elif not q_id_str or q_id_str == "":
            # 如果ID为空，尝试根据位置补全
            if expected_index < len(expected_ids):
                # 补全ID
                filled_q = q.copy()
                filled_q["id"] = expected_ids[expected_index]
                filled_questions.append(filled_q)
                expected_index += 1
            else:
                # 如果超出预期范围，保留原样（可能数据有问题）
                filled_questions.append(q.copy())
        else:
            # 如果ID存在但不在预期列表中，保留原样
            filled_questions.append(q.copy())

    return filled_questions


def get_answer_by_id(signaling_questions: List[Dict[str, Any]], question_id: str) -> Optional[str]:
    """
    根据问题ID获取答案代码

    Args:
        signaling_questions: 信号问题列表
        question_id: 问题ID，如 "2.1"

    Returns:
        答案代码，如果找不到则返回 None
    """
    for q in signaling_questions:
        # 兼容字符串和数字类型的id（健壮性处理）
        q_id = q.get("id")
        # 将两者都转换为字符串进行比较，以兼容JSON中可能存在的数字类型id
        if str(q_id) == str(question_id):
            return q.get("answer_code")
    return None


def check_precondition(
    signaling_questions: List[Dict[str, Any]],
    precondition: List[str],
    condition: List[str],
    logic: str = "OR",
    record_id: str = "unknown",
    current_question_id: str = "unknown",
    domain_key: str = "unknown"
) -> Tuple[bool, Optional[str]]:
    """
    检查前置条件是否满足

    Args:
        signaling_questions: 信号问题列表
        precondition: 前置问题ID列表
        condition: 满足条件的答案代码列表
        logic: 逻辑运算符，"OR" 或 "AND"（默认 "OR"）
        record_id: 记录ID，用于错误信息
        current_question_id: 当前问题ID，用于错误信息
        domain_key: domain键，用于补全缺失的ID

    Returns:
        (是否满足, 错误信息) 元组，如果没有错误则错误信息为 None
        如果找不到前置问题的答案（即使补全后），返回 (False, None) 表示前置条件不满足
    """
    if not precondition:
        return True, None

    # 先尝试补全缺失的ID
    filled_questions = fill_missing_ids(signaling_questions, domain_key)

    results = []
    missing_questions = []
    for pre_id in precondition:
        answer = get_answer_by_id(filled_questions, pre_id)
        if answer is None:
            # 如果补全后仍然找不到前置问题的答案，记录缺失的问题
            missing_questions.append(pre_id)
        else:
            results.append(answer in condition)

    # 如果有缺失的前置问题，说明LLM没有给出相关信号问题的答案
    # 根据用户要求，直接认为前置条件不满足（返回False），不计入错误
    if missing_questions:
        # 返回False表示前置条件不满足，但不抛出异常
        # 这样调用者可以将其视为错误判断
        return False, None

    if logic == "AND":
        # 所有前置问题都必须满足条件
        return all(results), None
    else:
        # OR 逻辑：至少一个前置问题满足条件
        return any(results), None


def evaluate_atomic_consistency(
    result_dir: Path,
    model_name: str
) -> Dict[str, Any]:
    """
    评估原子一致性

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

    # 统计信息：按 domain 和问题ID分组
    domain_stats = defaultdict(lambda: {
        "total_cases": 0,  # 所有具有前置条件的问题总数
        "correct": 0,       # 正确判断的数量（前置条件不满足且输出NA，或前置条件满足且输出非NA）
        "incorrect": 0,    # 错误判断的数量
        "details": []      # 详细记录
    })

    # 遍历所有结果
    for result in results:
        llm_result = result.get("llm_result", {})
        metadata = result.get("metadata", {})
        record_id = metadata.get("id", "unknown")

        # 遍历每个 domain
        for domain_key, rules in ATOMIC_CONSISTENCY_RULES.items():
            domain_data = llm_result.get(domain_key, {})
            signaling_questions = domain_data.get("signaling_questions", [])

            if not signaling_questions:
                # 如果找不到信号问题，直接报错
                study = metadata.get("study", "unknown")
                outcome = metadata.get("outcome", "unknown")
                error_msg = (
                    f"记录 ID: {record_id}, "
                    f"Study: {study}, "
                    f"Outcome: {outcome}, "
                    f"Domain: {domain_key}, "
                    f"错误: 找不到信号问题列表 (signaling_questions). "
                    f"Domain 数据键: {list(domain_data.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 先补全缺失的ID
            filled_signaling_questions = fill_missing_ids(signaling_questions, domain_key)

            # 遍历该 domain 的所有规则
            for question_id, rule in rules.items():
                precondition = rule["precondition"]
                condition = rule["condition"]
                logic = rule.get("logic", "OR")

                # 先检查是否能找到所有前置问题的答案（使用补全后的列表）
                missing_pre_ids = []
                for pre_id in precondition:
                    if get_answer_by_id(filled_signaling_questions, pre_id) is None:
                        missing_pre_ids.append(pre_id)

                # 如果补全后仍然找不到前置问题的答案，直接认为回答错误
                if missing_pre_ids:
                    study = metadata.get("study", "unknown")
                    outcome = metadata.get("outcome", "unknown")
                    available_ids = [q.get("id") for q in filled_signaling_questions]
                    logger.warning(
                        f"记录 ID: {record_id}, "
                        f"Study: {study}, "
                        f"Outcome: {outcome}, "
                        f"Domain: {domain_key}, "
                        f"当前问题: {question_id}, "
                        f"找不到前置问题的答案（已尝试补全ID）: {', '.join(missing_pre_ids)}. "
                        f"可用的信号问题ID: {available_ids}，计入错误数量"
                    )
                    # 直接认为回答错误，计入错误数量
                    domain_stats[domain_key]["total_cases"] += 1
                    domain_stats[domain_key]["incorrect"] += 1
                    domain_stats[domain_key]["details"].append({
                        "record_id": record_id,
                        "question_id": question_id,
                        "answer": "N/A",
                        "precondition": precondition,
                        "condition": condition,
                        "precondition_satisfied": False,
                        "status": "incorrect",
                        "reason": f"找不到前置问题的答案: {', '.join(missing_pre_ids)}"
                    })
                    continue

                # 检查前置条件是否满足（使用补全后的列表）
                precondition_satisfied, _ = check_precondition(
                    filled_signaling_questions,
                    precondition,
                    condition,
                    logic,
                    record_id,
                    question_id,
                    domain_key
                )

                # 获取当前问题的答案（使用补全后的列表）
                current_answer = get_answer_by_id(filled_signaling_questions, question_id)

                if current_answer is None:
                    # 如果补全后仍然找不到当前问题的答案，直接认为回答错误
                    study = metadata.get("study", "unknown")
                    outcome = metadata.get("outcome", "unknown")
                    available_ids = [q.get("id") for q in filled_signaling_questions]
                    logger.warning(
                        f"记录 ID: {record_id}, "
                        f"Study: {study}, "
                        f"Outcome: {outcome}, "
                        f"Domain: {domain_key}, "
                        f"当前问题: {question_id}, "
                        f"找不到当前问题的答案（已尝试补全ID）. "
                        f"可用的信号问题ID: {available_ids}，计入错误数量"
                    )
                    # 直接认为回答错误，计入错误数量
                    domain_stats[domain_key]["total_cases"] += 1
                    domain_stats[domain_key]["incorrect"] += 1
                    domain_stats[domain_key]["details"].append({
                        "record_id": record_id,
                        "question_id": question_id,
                        "answer": "N/A",
                        "precondition": precondition,
                        "condition": condition,
                        "precondition_satisfied": precondition_satisfied,
                        "status": "incorrect",
                        "reason": "找不到当前问题的答案"
                    })
                    continue

                # 统计所有具有前置条件的问题
                domain_stats[domain_key]["total_cases"] += 1

                # 判断是否正确进行前置条件判断：
                # - 如果前置条件不满足 → 应该输出 NA（正确）
                # - 如果前置条件满足 → 应该输出非 NA（正确）
                # - 其他情况 → 错误
                is_correct = False
                if not precondition_satisfied:
                    # 前置条件不满足，应该输出 NA
                    if current_answer == "NA":
                        is_correct = True
                else:
                    # 前置条件满足，应该输出非 NA
                    if current_answer != "NA":
                        is_correct = True

                if is_correct:
                    domain_stats[domain_key]["correct"] += 1
                    domain_stats[domain_key]["details"].append({
                        "record_id": record_id,
                        "question_id": question_id,
                        "answer": current_answer,
                        "precondition": precondition,
                        "condition": condition,
                        "precondition_satisfied": precondition_satisfied,
                        "status": "correct"
                    })
                else:
                    domain_stats[domain_key]["incorrect"] += 1
                    domain_stats[domain_key]["details"].append({
                        "record_id": record_id,
                        "question_id": question_id,
                        "answer": current_answer,
                        "precondition": precondition,
                        "condition": condition,
                        "precondition_satisfied": precondition_satisfied,
                        "status": "incorrect"
                    })

    # 计算每个 domain 的 CAR 和置信区间
    evaluation_results = {}

    for domain_key, domain_stat in domain_stats.items():
        total = domain_stat["total_cases"]
        correct = domain_stat["correct"]

        if total == 0:
            logger.warning(f"Domain {domain_key} 没有具有前置条件的问题")
            continue

        # 计算 CAR (Conditional Answer Rate) - 正确判断的比例
        car = correct / total if total > 0 else 0.0

        # 计算 95% 置信区间（使用 Wilson score interval）
        ci_lower, ci_upper = wilson_score_interval(correct, total, confidence=0.95)

        # 二项分布检验
        # H_0: 模型正确判断的概率 = 随机判断的概率
        # 如果模型是随机判断的，有2种情况（正确/错误），随机正确的概率应该是 0.5
        # 注意：scipy 1.7.0+ 中 binom_test 已被弃用，使用 binomtest 替代
        try:
            # 尝试使用新的 binomtest（scipy 1.7.0+）
            binom_result = stats.binomtest(correct, total, p=0.5, alternative='two-sided')
            p_value = binom_result.pvalue
        except AttributeError:
            # 如果 binomtest 不存在，使用旧的 binom_test（scipy < 1.7.0）
            p_value = stats.binom_test(correct, total, p=0.5, alternative='two-sided')

        # 根据 CAR 和 p-value 判断模型行为
        if p_value < 0.05:
            if car > 0.5:
                # CAR 高且显著，说明模型遵守规则
                interpretation = f"模型正确判断的概率显著高于随机（p={p_value:.6f}），且 CAR={car:.4f} 较高，说明模型遵守规则"
            else:
                # CAR 低且显著，说明模型没有遵守规则
                interpretation = f"模型正确判断的概率显著低于随机（p={p_value:.6f}），且 CAR={car:.4f} 较低，说明模型没有遵守规则"
        else:
            # 不显著，说明模型行为与随机无异
            interpretation = f"模型正确判断的概率与随机（0.5）无显著差异（p={p_value:.6f}），说明模型行为接近随机"

        # 确保所有值都是 Python 原生类型，以便 JSON 序列化
        evaluation_results[domain_key] = {
            "total_cases": int(total),
            "correct": int(correct),
            "incorrect": int(domain_stat["incorrect"]),
            "car": float(car),
            "car_ci_95": (float(ci_lower), float(ci_upper)),
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
            "interpretation": interpretation
        }

        logger.info(f"Domain {domain_key}:")
        logger.info(f"  总案例数（具有前置条件的问题数）: {total}")
        logger.info(f"  正确判断数: {correct}")
        logger.info(f"  错误判断数: {domain_stat['incorrect']}")
        logger.info(f"  CAR: {car:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        logger.info(f"  p-value: {p_value:.6f}")
        logger.info(f"  统计显著性: {'是' if p_value < 0.05 else '否'}")

    # 计算总体统计
    total_cases_all = sum(s["total_cases"] for s in domain_stats.values())
    total_correct_all = sum(s["correct"] for s in domain_stats.values())

    if total_cases_all > 0:
        overall_car = total_correct_all / total_cases_all
        overall_ci_lower, overall_ci_upper = wilson_score_interval(
            total_correct_all, total_cases_all, confidence=0.95
        )
        # 注意：scipy 1.7.0+ 中 binom_test 已被弃用，使用 binomtest 替代
        try:
            # 尝试使用新的 binomtest（scipy 1.7.0+）
            binom_result = stats.binomtest(total_correct_all, total_cases_all, p=0.5, alternative='two-sided')
            overall_p_value = binom_result.pvalue
        except AttributeError:
            # 如果 binomtest 不存在，使用旧的 binom_test（scipy < 1.7.0）
            overall_p_value = stats.binom_test(
                total_correct_all, total_cases_all, p=0.5, alternative='two-sided'
            )

        # 根据总体 CAR 和 p-value 判断模型行为
        if overall_p_value < 0.05:
            if overall_car > 0.5:
                # CAR 高且显著，说明模型遵守规则
                overall_interpretation = f"模型正确判断的概率显著高于随机（p={overall_p_value:.6f}），且 CAR={overall_car:.4f} 较高，说明模型遵守规则"
            else:
                # CAR 低且显著，说明模型没有遵守规则
                overall_interpretation = f"模型正确判断的概率显著低于随机（p={overall_p_value:.6f}），且 CAR={overall_car:.4f} 较低，说明模型没有遵守规则"
        else:
            # 不显著，说明模型行为与随机无异
            overall_interpretation = f"模型正确判断的概率与随机（0.5）无显著差异（p={overall_p_value:.6f}），说明模型行为接近随机"

        # 确保所有值都是 Python 原生类型，以便 JSON 序列化
        evaluation_results["overall"] = {
            "total_cases": int(total_cases_all),
            "correct": int(total_correct_all),
            "incorrect": int(total_cases_all - total_correct_all),
            "car": float(overall_car),
            "car_ci_95": (float(overall_ci_lower), float(overall_ci_upper)),
            "p_value": float(overall_p_value),
            "is_significant": bool(overall_p_value < 0.05),
            "interpretation": overall_interpretation
        }

        logger.info("=" * 80)
        logger.info("总体统计:")
        logger.info(f"  总案例数（具有前置条件的问题数）: {total_cases_all}")
        logger.info(f"  正确判断数: {total_correct_all}")
        logger.info(f"  错误判断数: {total_cases_all - total_correct_all}")
        logger.info(f"  总体 CAR: {overall_car:.4f} (95% CI: [{overall_ci_lower:.4f}, {overall_ci_upper:.4f}])")
        logger.info(f"  总体 p-value: {overall_p_value:.6f}")
        logger.info(f"  统计显著性: {'是' if overall_p_value < 0.05 else '否'}")
        logger.info("=" * 80)

    return {
        "model_name": model_name,
        "total_files": len(results),
        "evaluation_results": evaluation_results,
        "domain_details": dict(domain_stats)
    }


def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算Wilson score置信区间（适用于比例估计）

    这种方法在样本量小和大时都表现良好，比简单的正态近似更准确。

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


def print_summary(evaluation_result: Dict[str, Any]):
    """
    打印评估结果摘要

    Args:
        evaluation_result: 评估结果字典
    """
    logger.info("=" * 80)
    logger.info("原子一致性评估结果摘要")
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
    header = f"{'Domain':<30} {'Cases':<10} {'Correct':<10} {'CAR':<10} {'95% CI':<25} {'p-value':<12} {'Significant':<12}"
    logger.info(header)
    logger.info("-" * 120)

    # 数据行
    for domain in domains:
        result = eval_results[domain]
        car = result["car"]
        ci = result["car_ci_95"]
        p_val = result["p_value"]
        sig = "是" if result["is_significant"] else "否"
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"

        row = (
            f"{domain:<30} "
            f"{result['total_cases']:<10} "
            f"{result['correct']:<10} "
            f"{car:<10.4f} "
            f"{ci_str:<25} "
            f"{p_val:<12.6f} "
            f"{sig:<12}"
        )
        logger.info(row)

    # 打印总体结果
    if "overall" in eval_results:
        logger.info("-" * 120)
        overall = eval_results["overall"]
        car = overall["car"]
        ci = overall["car_ci_95"]
        p_val = overall["p_value"]
        sig = "是" if overall["is_significant"] else "否"
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"

        row = (
            f"{'Overall':<30} "
            f"{overall['total_cases']:<10} "
            f"{overall['correct']:<10} "
            f"{car:<10.4f} "
            f"{ci_str:<25} "
            f"{p_val:<12.6f} "
            f"{sig:<12}"
        )
        logger.info(row)

    logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估信号问题原子一致性"
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
        help="输出JSON文件路径（默认：项目根目录/data/atomic_consistency_{model_name}.json）"
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
        output_file = project_root / "data" / f"atomic_consistency_{model_dirname}.json"
    else:
        output_file = Path(args.output)

    # 检查路径是否存在
    if not result_dir.exists():
        logger.error(f"结果目录不存在: {result_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("评估信号问题原子一致性")
    logger.info("=" * 80)
    logger.info(f"结果目录: {result_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"输出文件: {output_file}")
    logger.info("=" * 80)

    # 执行评估
    try:
        evaluation_result = evaluate_atomic_consistency(result_dir, args.model_name)

        # 打印摘要
        print_summary(evaluation_result)

        # 保存结果（不包含详细记录，因为可能很大）
        output_result = {
            "model_name": evaluation_result["model_name"],
            "total_files": evaluation_result["total_files"],
            "evaluation_results": evaluation_result["evaluation_results"]
        }

        # 确保所有值都是 Python 原生类型，以便 JSON 序列化
        output_result = convert_to_native_type(output_result)

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

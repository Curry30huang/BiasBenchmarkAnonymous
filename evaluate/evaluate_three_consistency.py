#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
评估三种一致性的端到端完全正确率

该脚本统计在所有三个一致性维度（原子一致性、域一致性、聚合一致性）上完全正确的比例。

判定规则：
1. **原子一致性**：如果一个domain有分支（前置条件），那么所有分支的前置条件判断必须全部正确才认为该domain正确
2. **域一致性**：必须规则推理和LLM输出完全一致才是正确
3. **聚合一致性**：LLM的overall_risk必须与规则推理的overall_risk完全一致

只有当这三个维度都正确时，才认为该LLM输出结果完全正确。

计算公式：
完全正确率 = \frac{\sum_{i=1}^N \mathbb{I}(\text{所有三个维度都正确})}{N}
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from loguru import logger

# 导入必要的类和函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from bias_assessment.evidence_entity import RCTRisk
from bias_assessment.rct_bias_assessment_copy import (
    rct_1_randomization_process_judgement,
    rct_2_intended_intervention_judgement,
    rct_3_missing_outcome_judgement,
    rct_4_measurement_of_the_outcome_judgement,
    rct_5_selection_of_the_reported_result_judgement,
    RCT_BIAS_DOMAIN_KEYS,
    _string_to_rct_risk,
    _overall_risk_assessment
)

# 原子一致性规则（从evaluate_atomic_consistency.py复制）
ATOMIC_CONSISTENCY_RULES = {
    "randomisation_process": {},
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
            "logic": "AND"
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
    "selection_reported_result": {}
}

# Domain映射（从evaluate_domain_consistency.py复制）
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

# 聚合一致性中的domain key映射
DOMAIN_KEY_MAPPING = {
    "randomisation_process": "randomization_process_question",
    "intended_interventions": "intended_intervention",
    "missing_outcome_data": "missing_outcome",
    "measurement_outcome": "measurement_of_the_outcome",
    "selection_reported_result": "selection_of_the_reported_result"
}


def model_name_to_dirname(model_name: str) -> str:
    """将模型名称转换为安全的目录名"""
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
    """将 numpy 类型转换为 Python 原生类型，以便 JSON 序列化"""
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
    """加载所有结果文件"""
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
    """将字符串转换为RCTRisk枚举"""
    if not risk_str:
        return None

    risk_str = risk_str.strip()
    for risk in RCTRisk:
        if risk.value.lower() == risk_str.lower():
            return risk
    return None


def get_expected_question_ids(domain_key: str) -> List[str]:
    """根据domain key推断所有预期的问题ID列表"""
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

    all_question_ids = set()
    if domain_key in ATOMIC_CONSISTENCY_RULES:
        for q_id, rule in ATOMIC_CONSISTENCY_RULES[domain_key].items():
            all_question_ids.add(q_id)
            all_question_ids.update(rule.get("precondition", []))

    if not all_question_ids:
        return []

    question_nums = []
    for q_id in all_question_ids:
        try:
            num = int(q_id.split(".")[1])
            question_nums.append(num)
        except (ValueError, IndexError):
            continue

    if not question_nums:
        return []

    min_num = min(question_nums)
    max_num = max(question_nums)
    expected_ids = [f"{domain_num}.{i}" for i in range(min_num, max_num + 1)]

    return expected_ids


def fill_missing_ids(signaling_questions: List[Dict[str, Any]], domain_key: str) -> List[Dict[str, Any]]:
    """补全信号问题列表中缺失的ID"""
    if not signaling_questions:
        return signaling_questions

    expected_ids = get_expected_question_ids(domain_key)
    if not expected_ids:
        return signaling_questions

    filled_questions = []
    expected_index = 0

    for i, q in enumerate(signaling_questions):
        q_id = q.get("id")
        q_id_str = str(q_id).strip() if q_id else ""

        if q_id_str and q_id_str in expected_ids:
            filled_questions.append(q.copy())
            if q_id_str in expected_ids:
                expected_index = expected_ids.index(q_id_str) + 1
        elif not q_id_str or q_id_str == "":
            if expected_index < len(expected_ids):
                filled_q = q.copy()
                filled_q["id"] = expected_ids[expected_index]
                filled_questions.append(filled_q)
                expected_index += 1
            else:
                filled_questions.append(q.copy())
        else:
            filled_questions.append(q.copy())

    return filled_questions


def get_answer_by_id(signaling_questions: List[Dict[str, Any]], question_id: str) -> Optional[str]:
    """根据问题ID获取答案代码"""
    for q in signaling_questions:
        q_id = q.get("id")
        if str(q_id) == str(question_id):
            return q.get("answer_code")
    return None


def check_precondition(
    signaling_questions: List[Dict[str, Any]],
    precondition: List[str],
    condition: List[str],
    logic: str = "OR",
    domain_key: str = "unknown"
) -> bool:
    """检查前置条件是否满足"""
    if not precondition:
        return True

    filled_questions = fill_missing_ids(signaling_questions, domain_key)

    results = []
    for pre_id in precondition:
        answer = get_answer_by_id(filled_questions, pre_id)
        if answer is None:
            return False
        results.append(answer in condition)

    if logic == "AND":
        return all(results)
    else:
        return any(results)


def check_atomic_consistency_for_domain(
    domain_key: str,
    signaling_questions: List[Dict[str, Any]],
    record_id: str
) -> Tuple[bool, List[str]]:
    """
    检查单个domain的原子一致性

    Returns:
        (是否完全正确, 错误信息列表)
    """
    errors = []
    rules = ATOMIC_CONSISTENCY_RULES.get(domain_key, {})

    if not rules:
        # 该domain没有前置条件的问题，认为原子一致性通过
        return True, []

    filled_questions = fill_missing_ids(signaling_questions, domain_key)

    for question_id, rule in rules.items():
        precondition = rule["precondition"]
        condition = rule["condition"]
        logic = rule.get("logic", "OR")

        # 检查是否能找到所有前置问题的答案
        missing_pre_ids = []
        for pre_id in precondition:
            if get_answer_by_id(filled_questions, pre_id) is None:
                missing_pre_ids.append(pre_id)

        if missing_pre_ids:
            errors.append(
                f"Domain {domain_key}, 问题 {question_id}: "
                f"找不到前置问题的答案: {', '.join(missing_pre_ids)}"
            )
            return False, errors

        # 检查前置条件是否满足
        precondition_satisfied = check_precondition(
            filled_questions,
            precondition,
            condition,
            logic,
            domain_key
        )

        # 获取当前问题的答案
        current_answer = get_answer_by_id(filled_questions, question_id)

        if current_answer is None:
            errors.append(
                f"Domain {domain_key}, 问题 {question_id}: "
                f"找不到当前问题的答案"
            )
            return False, errors

        # 判断是否正确进行前置条件判断
        is_correct = False
        if not precondition_satisfied:
            # 前置条件不满足，应该输出 NA
            if current_answer == "NA":
                is_correct = True
        else:
            # 前置条件满足，应该输出非 NA
            if current_answer != "NA":
                is_correct = True

        if not is_correct:
            errors.append(
                f"Domain {domain_key}, 问题 {question_id}: "
                f"前置条件判断错误（前置条件满足={precondition_satisfied}, "
                f"当前答案={current_answer}）"
            )
            return False, errors

    return True, []


def check_domain_consistency_for_domain(
    domain_key: str,
    signaling_questions: List[Dict[str, Any]],
    llm_domain_risk_str: str,
    record_id: str
) -> Tuple[bool, Optional[str]]:
    """
    检查单个domain的域一致性

    Returns:
        (是否一致, 错误信息)
    """
    domain_config = DOMAIN_MAPPING.get(domain_key)
    if not domain_config:
        return False, f"未知的domain: {domain_key}"

    if not signaling_questions:
        return False, f"Domain {domain_key}: 缺少 signaling_questions"

    if not llm_domain_risk_str:
        return False, f"Domain {domain_key}: 缺少 LLM domain_risk"

    # 使用逻辑推理函数计算逻辑domain_risk
    try:
        judgement_func = domain_config["judgement_func"]
        logical_domain_risk = judgement_func(signaling_questions)
        logical_domain_risk_str = logical_domain_risk.value
    except Exception as e:
        return False, f"Domain {domain_key}: 逻辑推理失败: {e}"

    # 转换为枚举类型以便比较
    llm_domain_risk = string_to_rct_risk(llm_domain_risk_str)
    if llm_domain_risk is None:
        return False, f"Domain {domain_key}: 无法解析LLM domain_risk: {llm_domain_risk_str}"

    # 比较
    if llm_domain_risk != logical_domain_risk:
        return False, (
            f"Domain {domain_key}: LLM domain_risk ({llm_domain_risk_str}) "
            f"与逻辑推理结果 ({logical_domain_risk_str}) 不一致"
        )

    return True, None


def convert_result_domain_keys_to_standard(llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """将结果文件中的domain key转换为标准格式"""
    standard_result = {}

    for result_key, standard_key in DOMAIN_KEY_MAPPING.items():
        if result_key in llm_result:
            domain_data = llm_result[result_key]
            domain_risk_str = domain_data.get("domain_risk")
            if domain_risk_str:
                try:
                    domain_risk = _string_to_rct_risk(domain_risk_str)
                    standard_result[standard_key] = {
                        "judgement": domain_risk
                    }
                except ValueError as e:
                    logger.warning(f"无法转换domain_risk '{domain_risk_str}' 为RCTRisk: {e}")
                    continue

    return standard_result


def compute_rule_based_overall_risk(llm_result: Dict[str, Any]) -> Optional[RCTRisk]:
    """根据规则计算overall_risk"""
    try:
        standard_result = convert_result_domain_keys_to_standard(llm_result)
        overall_risk = _overall_risk_assessment(standard_result)
        return overall_risk
    except Exception as e:
        logger.warning(f"计算规则推理overall_risk失败: {e}")
        return None


def get_llm_overall_risk(result: Dict[str, Any]) -> Optional[RCTRisk]:
    """从结果文件中获取LLM给出的overall_risk"""
    llm_result = result.get("llm_result", {})
    overall_risk_str = llm_result.get("overall_risk")

    if not overall_risk_str:
        overall_risk_str = result.get("overall_risk_judgement")

    if overall_risk_str:
        try:
            return _string_to_rct_risk(overall_risk_str)
        except ValueError as e:
            logger.warning(f"无法转换LLM overall_risk '{overall_risk_str}' 为RCTRisk: {e}")
            return None

    return None


def check_aggregation_consistency(
    llm_result: Dict[str, Any],
    record_id: str
) -> Tuple[bool, Optional[str]]:
    """
    检查聚合一致性

    Returns:
        (是否一致, 错误信息)
    """
    # 获取LLM给出的overall_risk
    # 注意：这里需要从result中获取，但函数参数是llm_result，需要调整
    # 实际上我们需要完整的result，所以这个函数签名需要调整
    # 但为了保持接口一致性，我们假设llm_result包含overall_risk
    llm_overall_risk_str = llm_result.get("overall_risk")
    if not llm_overall_risk_str:
        return False, "缺少 LLM overall_risk"

    llm_overall_risk = _string_to_rct_risk(llm_overall_risk_str)
    if llm_overall_risk is None:
        return False, f"无法解析LLM overall_risk: {llm_overall_risk_str}"

    # 计算规则推理得到的overall_risk
    rule_overall_risk = compute_rule_based_overall_risk(llm_result)
    if rule_overall_risk is None:
        return False, "无法计算规则推理overall_risk"

    # 比较
    if llm_overall_risk != rule_overall_risk:
        return False, (
            f"LLM overall_risk ({llm_overall_risk_str}) "
            f"与规则推理结果 ({rule_overall_risk.value}) 不一致"
        )

    return True, None


def check_single_result(result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    检查单个结果文件是否在所有三个维度上都完全正确

    Returns:
        (是否完全正确, 详细检查结果)
    """
    llm_result = result.get("llm_result", {})
    metadata = result.get("metadata", {})
    record_id = metadata.get("id", "unknown")

    check_result = {
        "record_id": record_id,
        "atomic_consistency": {
            "all_correct": True,
            "domain_results": {}
        },
        "domain_consistency": {
            "all_correct": True,
            "domain_results": {}
        },
        "aggregation_consistency": {
            "correct": False,
            "error": None
        },
        "overall_correct": False
    }

    # 1. 检查原子一致性（按domain）
    for domain_key in ATOMIC_CONSISTENCY_RULES.keys():
        domain_data = llm_result.get(domain_key, {})
        signaling_questions = domain_data.get("signaling_questions", [])

        if not signaling_questions:
            check_result["atomic_consistency"]["domain_results"][domain_key] = {
                "correct": False,
                "errors": [f"Domain {domain_key}: 缺少 signaling_questions"]
            }
            check_result["atomic_consistency"]["all_correct"] = False
            continue

        is_correct, errors = check_atomic_consistency_for_domain(
            domain_key, signaling_questions, record_id
        )
        check_result["atomic_consistency"]["domain_results"][domain_key] = {
            "correct": is_correct,
            "errors": errors
        }
        if not is_correct:
            check_result["atomic_consistency"]["all_correct"] = False

    # 2. 检查域一致性（按domain）
    for domain_key, domain_config in DOMAIN_MAPPING.items():
        domain_data = llm_result.get(domain_key, {})
        signaling_questions = domain_data.get("signaling_questions", [])
        llm_domain_risk_str = domain_data.get("domain_risk", "")

        is_correct, error = check_domain_consistency_for_domain(
            domain_key, signaling_questions, llm_domain_risk_str, record_id
        )
        check_result["domain_consistency"]["domain_results"][domain_key] = {
            "correct": is_correct,
            "error": error
        }
        if not is_correct:
            check_result["domain_consistency"]["all_correct"] = False

    # 3. 检查聚合一致性
    # 注意：需要从完整的result中获取overall_risk
    llm_overall_risk = get_llm_overall_risk(result)
    if llm_overall_risk is None:
        check_result["aggregation_consistency"]["correct"] = False
        check_result["aggregation_consistency"]["error"] = "无法获取LLM overall_risk"
    else:
        rule_overall_risk = compute_rule_based_overall_risk(llm_result)
        if rule_overall_risk is None:
            check_result["aggregation_consistency"]["correct"] = False
            check_result["aggregation_consistency"]["error"] = "无法计算规则推理overall_risk"
        else:
            if llm_overall_risk == rule_overall_risk:
                check_result["aggregation_consistency"]["correct"] = True
            else:
                check_result["aggregation_consistency"]["correct"] = False
                check_result["aggregation_consistency"]["error"] = (
                    f"LLM overall_risk ({llm_overall_risk.value}) "
                    f"与规则推理结果 ({rule_overall_risk.value}) 不一致"
                )

    # 综合判断：三个维度都必须正确
    check_result["overall_correct"] = (
        check_result["atomic_consistency"]["all_correct"] and
        check_result["domain_consistency"]["all_correct"] and
        check_result["aggregation_consistency"]["correct"]
    )

    return check_result["overall_correct"], check_result


def evaluate_three_consistency(
    result_dir: Path,
    model_name: str
) -> Dict[str, Any]:
    """
    评估三种一致性的端到端完全正确率

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

    total_cases = 0
    fully_correct_cases = 0

    # 详细记录
    details = []

    # 按维度统计
    atomic_correct_count = 0
    domain_correct_count = 0
    aggregation_correct_count = 0

    # 遍历所有结果
    for result in results:
        metadata = result.get("metadata", {})
        record_id = metadata.get("id", "unknown")

        total_cases += 1

        # 检查单个结果
        is_correct, check_result = check_single_result(result)

        if is_correct:
            fully_correct_cases += 1

        # 统计各维度
        if check_result["atomic_consistency"]["all_correct"]:
            atomic_correct_count += 1
        if check_result["domain_consistency"]["all_correct"]:
            domain_correct_count += 1
        if check_result["aggregation_consistency"]["correct"]:
            aggregation_correct_count += 1

        # 保存详细记录（仅保存错误案例的摘要，避免数据过大）
        if not is_correct:
            details.append({
                "record_id": record_id,
                "atomic_consistency_correct": check_result["atomic_consistency"]["all_correct"],
                "domain_consistency_correct": check_result["domain_consistency"]["all_correct"],
                "aggregation_consistency_correct": check_result["aggregation_consistency"]["correct"],
                "overall_correct": False
            })

    if total_cases == 0:
        logger.error("没有有效的评估案例")
        return {}

    # 计算完全正确率
    fully_correct_rate = fully_correct_cases / total_cases if total_cases > 0 else 0.0

    # 计算各维度的正确率
    atomic_rate = atomic_correct_count / total_cases if total_cases > 0 else 0.0
    domain_rate = domain_correct_count / total_cases if total_cases > 0 else 0.0
    aggregation_rate = aggregation_correct_count / total_cases if total_cases > 0 else 0.0

    # 计算95%置信区间（使用Wilson score interval）
    fully_correct_ci = wilson_score_interval(fully_correct_cases, total_cases, confidence=0.95)
    atomic_ci = wilson_score_interval(atomic_correct_count, total_cases, confidence=0.95)
    domain_ci = wilson_score_interval(domain_correct_count, total_cases, confidence=0.95)
    aggregation_ci = wilson_score_interval(aggregation_correct_count, total_cases, confidence=0.95)

    # 确保所有值都是 Python 原生类型，以便 JSON 序列化
    evaluation_result = {
        "model_name": model_name,
        "total_files": len(results),
        "total_cases": int(total_cases),
        "fully_correct_cases": int(fully_correct_cases),
        "fully_correct_rate": float(fully_correct_rate),
        "fully_correct_ci_95": (float(fully_correct_ci[0]), float(fully_correct_ci[1])),
        "dimension_breakdown": {
            "atomic_consistency": {
                "correct_count": int(atomic_correct_count),
                "rate": float(atomic_rate),
                "ci_95": (float(atomic_ci[0]), float(atomic_ci[1]))
            },
            "domain_consistency": {
                "correct_count": int(domain_correct_count),
                "rate": float(domain_rate),
                "ci_95": (float(domain_ci[0]), float(domain_ci[1]))
            },
            "aggregation_consistency": {
                "correct_count": int(aggregation_correct_count),
                "rate": float(aggregation_rate),
                "ci_95": (float(aggregation_ci[0]), float(aggregation_ci[1]))
            }
        }
    }

    logger.info("=" * 80)
    logger.info("三种一致性端到端评估结果")
    logger.info("=" * 80)
    logger.info(f"总文件数: {len(results)}")
    logger.info(f"有效案例数: {total_cases}")
    logger.info(f"完全正确案例数: {fully_correct_cases}")
    logger.info(f"完全正确率: {fully_correct_rate:.4f} (95% CI: [{fully_correct_ci[0]:.4f}, {fully_correct_ci[1]:.4f}])")
    logger.info("")
    logger.info("各维度正确率:")
    logger.info(f"  原子一致性: {atomic_rate:.4f} (95% CI: [{atomic_ci[0]:.4f}, {atomic_ci[1]:.4f}])")
    logger.info(f"  域一致性: {domain_rate:.4f} (95% CI: [{domain_ci[0]:.4f}, {domain_ci[1]:.4f}])")
    logger.info(f"  聚合一致性: {aggregation_rate:.4f} (95% CI: [{aggregation_ci[0]:.4f}, {aggregation_ci[1]:.4f}])")
    logger.info("=" * 80)

    return {
        "model_name": model_name,
        "total_files": len(results),
        "evaluation_result": evaluation_result,
        "error_details": details  # 仅包含错误案例的摘要
    }


def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """计算Wilson score置信区间"""
    if n == 0:
        return (0.0, 0.0)

    if successes == 0:
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p_hat = 0.0
        denominator = 1 + (z ** 2) / n
        center = (p_hat + (z ** 2) / (2 * n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * n)) / n) / denominator
        return (max(0.0, center - margin), min(1.0, center + margin))

    if successes == n:
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
    """打印评估结果摘要"""
    logger.info("=" * 80)
    logger.info("三种一致性端到端评估结果摘要")
    logger.info("=" * 80)

    eval_result = evaluation_result.get("evaluation_result", {})
    logger.info(f"模型名称: {evaluation_result.get('model_name', 'N/A')}")
    logger.info(f"总文件数: {evaluation_result.get('total_files', 0)}")
    logger.info(f"有效案例数: {eval_result.get('total_cases', 0)}")
    logger.info(f"完全正确案例数: {eval_result.get('fully_correct_cases', 0)}")

    fully_correct_rate = eval_result.get('fully_correct_rate', 0.0)
    fully_correct_ci = eval_result.get('fully_correct_ci_95', (0.0, 0.0))
    logger.info(f"完全正确率: {fully_correct_rate:.4f} (95% CI: [{fully_correct_ci[0]:.4f}, {fully_correct_ci[1]:.4f}])")

    dimension_breakdown = eval_result.get('dimension_breakdown', {})
    logger.info("")
    logger.info("各维度正确率:")

    atomic = dimension_breakdown.get('atomic_consistency', {})
    atomic_ci = atomic.get('ci_95', (0.0, 0.0))
    logger.info(f"  原子一致性: {atomic.get('rate', 0.0):.4f} (95% CI: [{atomic_ci[0]:.4f}, {atomic_ci[1]:.4f}])")

    domain = dimension_breakdown.get('domain_consistency', {})
    domain_ci = domain.get('ci_95', (0.0, 0.0))
    logger.info(f"  域一致性: {domain.get('rate', 0.0):.4f} (95% CI: [{domain_ci[0]:.4f}, {domain_ci[1]:.4f}])")

    aggregation = dimension_breakdown.get('aggregation_consistency', {})
    aggregation_ci = aggregation.get('ci_95', (0.0, 0.0))
    logger.info(f"  聚合一致性: {aggregation.get('rate', 0.0):.4f} (95% CI: [{aggregation_ci[0]:.4f}, {aggregation_ci[1]:.4f}])")

    logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估三种一致性的端到端完全正确率"
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
        help="输出JSON文件路径（默认：项目根目录/data/three_consistency_{model_name}.json）"
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
        model_dirname = model_name_to_dirname(args.model_name)
        output_file = project_root / "data" / f"three_consistency_{model_dirname}.json"
    else:
        output_file = Path(args.output)

    # 检查路径是否存在
    if not result_dir.exists():
        logger.error(f"结果目录不存在: {result_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("评估三种一致性的端到端完全正确率")
    logger.info("=" * 80)
    logger.info(f"结果目录: {result_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"输出文件: {output_file}")
    logger.info("=" * 80)

    # 执行评估
    try:
        evaluation_result = evaluate_three_consistency(result_dir, args.model_name)

        # 打印摘要
        print_summary(evaluation_result)

        # 保存结果
        output_result = {
            "model_name": evaluation_result["model_name"],
            "total_files": evaluation_result["total_files"],
            "evaluation_result": evaluation_result["evaluation_result"]
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

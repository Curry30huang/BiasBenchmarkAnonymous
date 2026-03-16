#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
评估全局风险一致性（聚合一致性）的脚本

聚合一致性主要衡量模型在聚合多个 Domain 风险时，其 overall_risk 判断与规则推理得到的 overall_risk 之间的匹配程度。

计算公式：
VR (Verification Rate) = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)}{N}

其中 $\hat{y}_i$ 是模型给出的 overall_risk，$y_i$ 是规则推理得到的 overall_risk。

评估方法：

方法1：准确率 (VR) - **基础指标**
**原理**：计算完全匹配的比例，简单直观。
**优点**：
  - 简单直接，易于理解
  - 提供置信区间（Wilson score interval）
**缺点**：
  - 不考虑偶然一致性（随机匹配的概率）
  - 不区分不同类别错误的严重性

方法2：Wilcoxon Signed-Rank Test (符号秩检验) - **系统性偏差检测**
**原理**：将风险等级量化为序数（Low=1, Some Concerns=2, High=3），计算LLM结果与规则推理结果的差值。
**优点**：
  - 考虑了风险等级的序数性质，保留了方向信息
  - 可以检测系统性偏见的方向（乐观/悲观）
  - 适合配对数据（LLM vs 规则推理）
  - 可以量化偏见程度（通过中位数差异）
**意义**：如果检验结果显著（$p < 0.05$）且中位数为负，则证明模型在全局聚合时存在显著的**"乐观偏见（Optimism Bias）"**， 即它在统计学上显著地违反了"水桶原理"（低估风险）。如果中位数为正，则存在"悲观偏见"（高估风险）。
**注意**：当准确率很高（>95%）且中位数差异为0时，即使p-value显著，也不应过度解读为"系统性偏见"，因为实际偏差很小。

方法3：Cohen's Kappa (加权Kappa系数) - **一致性评估**
**原理**：评估两个评估者（LLM和规则推理）之间的一致性，考虑了偶然一致的概率。
计算公式：$\kappa = \frac{P_o - P_e}{1 - P_e}$，其中 $P_o$ 是观察一致性，$P_e$ 是期望一致性（随机一致的概率）。
**优点**：
  - 考虑了偶然一致性，比简单准确率更稳健
  - 使用加权Kappa（quadratic权重）考虑不同类别错误的严重性：
    - Low (1) → Some Concerns (2) 的惩罚较小
    - Low (1) → High (3) 的惩罚较大
  - 提供95%置信区间
  - 有标准的一致性解释标准：
    - $\kappa \geq 0.81$: 几乎完美一致
    - $0.61 \leq \kappa < 0.81$: 实质性一致
    - $0.41 \leq \kappa < 0.61$: 中等一致
    - $0.21 \leq \kappa < 0.41$: 一般一致
    - $0.00 \leq \kappa < 0.21$: 轻微一致
    - $\kappa < 0.00$: 不一致
**意义**：Kappa值更全面地评估一致性，特别是在类别分布不均匀时，能更准确地反映模型的实际表现。

**综合评估策略**：
1. 首先查看准确率（VR）：了解基本匹配情况
2. 结合Cohen's Kappa：评估一致性（考虑偶然一致性）
3. 参考Wilcoxon检验：检测是否存在系统性偏差（但需结合准确率和中位数差异综合判断）
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
from loguru import logger
from sklearn.metrics import confusion_matrix

# 导入必要的类和函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from bias_assessment.evidence_entity import RCTRisk
from bias_assessment.rct_bias_assessment_copy import (
    RCT_BIAS_DOMAIN_KEYS,
    _string_to_rct_risk,
    _overall_risk_assessment
)


# 风险等级到数值的映射（用于Wilcoxon检验）
RISK_TO_NUMERIC = {
    RCTRisk.LOW_RISK: 1,
    RCTRisk.MODERATE_RISK: 2,
    RCTRisk.HIGH_RISK: 3
}

# 结果文件中的domain key到RCT_BIAS_DOMAIN_KEYS的映射
DOMAIN_KEY_MAPPING = {
    "randomisation_process": "randomization_process_question",
    "intended_interventions": "intended_intervention",
    "missing_outcome_data": "missing_outcome",
    "measurement_outcome": "measurement_of_the_outcome",
    "selection_reported_result": "selection_of_the_reported_result"
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


def cohen_kappa_ci(
    kappa: float,
    n: int,
    llm_values: List[str],
    rule_values: List[str],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    计算Cohen's Kappa的置信区间

    使用标准误差（SE）和正态分布近似。

    Args:
        kappa: Cohen's Kappa值
        n: 样本数量
        llm_values: LLM预测值列表（用于计算观察一致性和期望一致性）
        rule_values: 规则推理值列表
        confidence: 置信水平（默认0.95，即95%置信区间）

    Returns:
        (lower_bound, upper_bound) 置信区间的上下界
    """
    if n == 0 or kappa is None:
        return (None, None)

    # 过滤有效值
    pairs = [(llm, rule) for llm, rule in zip(llm_values, rule_values)
             if llm and rule and llm.strip() and rule.strip()]

    if len(pairs) == 0:
        return (None, None)

    llm_clean, rule_clean = zip(*pairs)

    # 计算观察一致性（Po）：实际一致的比例
    po = sum(1 for llm, rule in zip(llm_clean, rule_clean) if llm == rule) / len(llm_clean)

    # 计算期望一致性（Pe）：随机一致的概率
    llm_dist = Counter(llm_clean)
    rule_dist = Counter(rule_clean)
    all_labels = set(llm_clean) | set(rule_clean)

    pe = 0.0
    for label in all_labels:
        llm_prob = llm_dist.get(label, 0) / len(llm_clean)
        rule_prob = rule_dist.get(label, 0) / len(llm_clean)
        pe += llm_prob * rule_prob

    # 计算标准误差
    # SE = sqrt((Po * (1 - Po)) / (n * (1 - Pe)^2))
    if pe >= 1.0 or n == 0:
        return (None, None)

    se = np.sqrt((po * (1 - po)) / (n * ((1 - pe) ** 2)))

    # 使用正态分布计算置信区间
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z * se

    lower = max(-1.0, kappa - margin)
    upper = min(1.0, kappa + margin)

    return (lower, upper)


def _weighted_kappa(
    rater_a: np.ndarray,
    rater_b: np.ndarray,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    weights: str = 'quadratic'
) -> float:
    """
    计算加权Kappa系数（基于原始数值差异）

    Args:
        rater_a: 评估者A的评分数组（数值）
        rater_b: 评估者B的评分数组（数值）
        min_rating: 最小评分值，如果为None则自动计算
        max_rating: 最大评分值，如果为None则自动计算
        weights: 权重类型，'linear' 或 'quadratic'

    Returns:
        加权Kappa系数
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert len(rater_a) == len(rater_b)

    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))

    # 计算混淆矩阵（使用实际的评分值作为labels）
    labels = list(range(min_rating, max_rating + 1))
    conf_mat = confusion_matrix(rater_a, rater_b, labels=labels)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    # 计算每个评分的直方图
    def histogram(ratings, min_rating, max_rating):
        hist = np.zeros(max_rating - min_rating + 1, dtype=int)
        for rating in ratings:
            if min_rating <= rating <= max_rating:
                hist[rating - min_rating] += 1
        return hist

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)

            # 计算原始数值差异（不是索引差异）
            rating_i = min_rating + i
            rating_j = min_rating + j
            diff = abs(rating_i - rating_j)

            # 根据权重类型计算权重
            if weights == 'linear':
                d = diff / (max_rating - min_rating) if (max_rating - min_rating) > 0 else 0
            else:  # quadratic
                d = (diff / (max_rating - min_rating)) ** 2.0 if (max_rating - min_rating) > 0 else 0

            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - (numerator / denominator) if denominator > 0 else 0.0


def calculate_cohen_kappa(
    llm_risks: List[RCTRisk],
    rule_risks: List[RCTRisk],
    weights: str = 'quadratic'
) -> Dict[str, Any]:
    """
    计算加权Cohen's Kappa系数及其95%置信区间

    使用加权Kappa来考虑不同类别之间错误的严重性：
    - Low (1) -> Some Concerns (2) 的惩罚较小
    - Low (1) -> High (3) 的惩罚较大

    权重类型：
    - 'linear': 线性权重 = |数值差异| / (max - min)
    - 'quadratic': 平方权重 = (数值差异)^2 / (max - min)^2

    Args:
        llm_risks: LLM预测的风险等级列表（RCTRisk枚举）
        rule_risks: 规则推理的风险等级列表（RCTRisk枚举）
        weights: 权重类型，'linear' 或 'quadratic'，默认为 'quadratic'

    Returns:
        包含Kappa值和95%置信区间的字典，格式为 {"value": float, "ci_95": (lower, upper)}，
        如果无法计算则返回 {"value": None, "ci_95": (None, None)}
    """
    if len(llm_risks) != len(rule_risks):
        return {"value": None, "ci_95": (None, None)}

    if weights not in ['linear', 'quadratic']:
        logger.warning(f"不支持的权重类型: {weights}，使用默认值 'quadratic'")
        weights = 'quadratic'

    # 转换为数值
    llm_nums = [RISK_TO_NUMERIC[risk] for risk in llm_risks]
    rule_nums = [RISK_TO_NUMERIC[risk] for risk in rule_risks]
    llm_nums = np.array(llm_nums)
    rule_nums = np.array(rule_nums)

    # 转换为字符串用于置信区间计算
    llm_strs = [risk.value for risk in llm_risks]
    rule_strs = [risk.value for risk in rule_risks]

    # 获取所有唯一的数值类别
    all_numeric_values = sorted(set(llm_nums) | set(rule_nums))

    # 检查类别分布
    llm_dist = Counter(llm_nums)
    rule_dist = Counter(rule_nums)

    # 如果只有一个类别，Kappa 无法计算（返回 None）
    if len(all_numeric_values) < 2:
        logger.warning(f"加权Cohen's Kappa 无法计算：只有 {len(all_numeric_values)} 个类别（需要至少2个类别）")
        return {"value": None, "ci_95": (None, None)}

    try:
        # 使用自定义加权Kappa函数（基于原始数值差异）
        min_rating = min(all_numeric_values)
        max_rating = max(all_numeric_values)

        kappa = _weighted_kappa(llm_nums, rule_nums, min_rating, max_rating, weights)

        # 计算95%置信区间
        kappa_ci = cohen_kappa_ci(kappa, len(llm_risks), llm_strs, rule_strs, confidence=0.95)

        return {"value": kappa, "ci_95": kappa_ci}
    except Exception as e:
        logger.warning(f"计算加权Cohen's Kappa时出错: {e}")
        return {"value": None, "ci_95": (None, None)}


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


def convert_result_domain_keys_to_standard(llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    将结果文件中的domain key转换为标准格式（用于_overall_risk_assessment函数）

    Args:
        llm_result: LLM结果字典，包含domain数据

    Returns:
        转换后的结果字典，使用标准的domain key
    """
    standard_result = {}

    for result_key, standard_key in DOMAIN_KEY_MAPPING.items():
        if result_key in llm_result:
            domain_data = llm_result[result_key]
            # 获取domain_risk并转换为judgement格式
            domain_risk_str = domain_data.get("domain_risk")
            if domain_risk_str:
                try:
                    domain_risk = _string_to_rct_risk(domain_risk_str)
                    standard_result[standard_key] = {
                        "judgement": domain_risk
                    }
                except ValueError as e:
                    logger.warning(f"无法转换domain_risk '{domain_risk_str}' 为RCTRisk: {e}")
                    # 如果转换失败，跳过该domain
                    continue

    return standard_result


def compute_rule_based_overall_risk(llm_result: Dict[str, Any]) -> Optional[RCTRisk]:
    """
    根据规则计算overall_risk

    Args:
        llm_result: LLM结果字典，包含domain数据

    Returns:
        规则推理得到的overall_risk，如果计算失败则返回None
    """
    try:
        # 转换domain key为标准格式
        standard_result = convert_result_domain_keys_to_standard(llm_result)

        # 使用_overall_risk_assessment函数计算
        overall_risk = _overall_risk_assessment(standard_result)
        return overall_risk
    except Exception as e:
        logger.warning(f"计算规则推理overall_risk失败: {e}")
        return None


def get_llm_overall_risk(result: Dict[str, Any]) -> Optional[RCTRisk]:
    """
    从结果文件中获取LLM给出的overall_risk

    Args:
        result: 结果文件字典

    Returns:
        LLM给出的overall_risk，如果找不到则返回None
    """
    # 首先尝试从llm_result中获取
    llm_result = result.get("llm_result", {})
    overall_risk_str = llm_result.get("overall_risk")

    # 如果找不到，尝试从顶层获取overall_risk_judgement
    if not overall_risk_str:
        overall_risk_str = result.get("overall_risk_judgement")

    if overall_risk_str:
        try:
            return _string_to_rct_risk(overall_risk_str)
        except ValueError as e:
            logger.warning(f"无法转换LLM overall_risk '{overall_risk_str}' 为RCTRisk: {e}")
            return None

    return None


def get_ground_truth_overall_risk(result: Dict[str, Any]) -> Optional[RCTRisk]:
    """
    从结果文件中获取ground truth的overall_risk

    Args:
        result: 结果文件字典

    Returns:
        ground truth的overall_risk，如果找不到则返回None
    """
    ground_truth = result.get("ground_truth", {})
    overall_risk_str = ground_truth.get("overall_risk")

    if overall_risk_str:
        try:
            return _string_to_rct_risk(overall_risk_str)
        except ValueError as e:
            logger.warning(f"无法转换ground truth overall_risk '{overall_risk_str}' 为RCTRisk: {e}")
            return None

    return None


def evaluate_aggregation_consistency(
    result_dir: Path,
    model_name: str
) -> Dict[str, Any]:
    """
    评估聚合一致性

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

    # 统计信息
    total_cases = 0
    correct_matches = 0
    incorrect_matches = 0

    # 新增：与ground truth比较的统计
    rule_vs_ground_truth_correct = 0  # 规则推理结果与ground truth匹配数
    llm_vs_ground_truth_correct = 0   # LLM结果与ground truth匹配数

    # 用于Wilcoxon检验的数据
    llm_risks_numeric = []
    rule_risks_numeric = []
    differences = []

    # 用于Cohen's Kappa的数据
    llm_risks_list = []
    rule_risks_list = []

    # 详细记录
    details = []

    # 遍历所有结果
    for result in results:
        metadata = result.get("metadata", {})
        record_id = metadata.get("id", "unknown")
        llm_result = result.get("llm_result", {})

        # 获取LLM给出的overall_risk
        llm_overall_risk = get_llm_overall_risk(result)

        # 计算规则推理得到的overall_risk
        rule_overall_risk = compute_rule_based_overall_risk(llm_result)

        # 获取ground truth的overall_risk
        ground_truth_overall_risk = get_ground_truth_overall_risk(result)

        # 如果任一结果缺失，跳过该记录
        if llm_overall_risk is None:
            logger.warning(f"记录 ID: {record_id}, 无法获取LLM overall_risk，跳过")
            continue

        if rule_overall_risk is None:
            logger.warning(f"记录 ID: {record_id}, 无法计算规则推理overall_risk，跳过")
            continue

        if ground_truth_overall_risk is None:
            logger.warning(f"记录 ID: {record_id}, 无法获取ground truth overall_risk，跳过")
            continue

        total_cases += 1

        # 判断是否匹配（LLM vs 规则推理）
        is_match = (llm_overall_risk == rule_overall_risk)

        if is_match:
            correct_matches += 1
        else:
            incorrect_matches += 1

        # 新增：判断规则推理结果与ground truth是否匹配
        rule_vs_gt_match = (rule_overall_risk == ground_truth_overall_risk)
        if rule_vs_gt_match:
            rule_vs_ground_truth_correct += 1

        # 新增：判断LLM结果与ground truth是否匹配
        llm_vs_gt_match = (llm_overall_risk == ground_truth_overall_risk)
        if llm_vs_gt_match:
            llm_vs_ground_truth_correct += 1

        # 转换为数值用于Wilcoxon检验
        llm_numeric = RISK_TO_NUMERIC[llm_overall_risk]
        rule_numeric = RISK_TO_NUMERIC[rule_overall_risk]
        difference = llm_numeric - rule_numeric  # 正数表示高估，负数表示低估

        llm_risks_numeric.append(llm_numeric)
        rule_risks_numeric.append(rule_numeric)
        differences.append(difference)

        # 保存用于Cohen's Kappa
        llm_risks_list.append(llm_overall_risk)
        rule_risks_list.append(rule_overall_risk)

        # 记录详细信息
        details.append({
            "record_id": record_id,
            "llm_overall_risk": llm_overall_risk.value,
            "rule_overall_risk": rule_overall_risk.value,
            "ground_truth_overall_risk": ground_truth_overall_risk.value,
            "is_match": is_match,  # LLM vs 规则推理
            "rule_vs_gt_match": rule_vs_gt_match,  # 规则推理 vs ground truth
            "llm_vs_gt_match": llm_vs_gt_match,  # LLM vs ground truth
            "difference": difference  # 正数=高估，负数=低估
        })

    if total_cases == 0:
        logger.error("没有有效的评估案例")
        return {}

    # 计算准确率（VR）
    vr = correct_matches / total_cases if total_cases > 0 else 0.0

    # 计算95%置信区间（使用Wilson score interval）
    vr_ci_lower, vr_ci_upper = wilson_score_interval(correct_matches, total_cases, confidence=0.95)

    # 新增：计算规则推理结果与ground truth的准确率
    rule_vs_gt_accuracy = rule_vs_ground_truth_correct / total_cases if total_cases > 0 else 0.0
    rule_vs_gt_ci_lower, rule_vs_gt_ci_upper = wilson_score_interval(
        rule_vs_ground_truth_correct, total_cases, confidence=0.95
    )

    # 新增：计算LLM结果与ground truth的准确率
    llm_vs_gt_accuracy = llm_vs_ground_truth_correct / total_cases if total_cases > 0 else 0.0
    llm_vs_gt_ci_lower, llm_vs_gt_ci_upper = wilson_score_interval(
        llm_vs_ground_truth_correct, total_cases, confidence=0.95
    )

    # Wilcoxon Signed-Rank Test
    # H_0: LLM结果与规则推理结果的差值的中位数为0（无系统性偏见）
    # 如果p < 0.05且中位数为负，说明存在显著的乐观偏见（低估风险）
    # 如果p < 0.05且中位数为正，说明存在显著的悲观偏见（高估风险）
    differences_array = np.array(differences)
    median_difference = np.median(differences_array)

    try:
        # Wilcoxon符号秩检验
        # 注意：scipy.stats.wilcoxon 默认是双侧检验
        wilcoxon_result = stats.wilcoxon(
            llm_risks_numeric,
            rule_risks_numeric,
            alternative='two-sided'
        )
        p_value = wilcoxon_result.pvalue
        statistic = wilcoxon_result.statistic
    except Exception as e:
        logger.error(f"Wilcoxon检验失败: {e}")
        p_value = None
        statistic = None

    # 解释结果
    if p_value is not None:
        if p_value < 0.05:
            if median_difference < 0:
                interpretation = (
                    f"模型存在显著的乐观偏见（Optimism Bias），"
                    f"中位数差异={median_difference:.2f}（负值表示低估风险），"
                    f"p={p_value:.6f}。模型在全局聚合时显著违反了'水桶原理'，倾向于低估风险。"
                )
            elif median_difference > 0:
                interpretation = (
                    f"模型存在显著的悲观偏见（Pessimism Bias），"
                    f"中位数差异={median_difference:.2f}（正值表示高估风险），"
                    f"p={p_value:.6f}。模型在全局聚合时倾向于高估风险。"
                )
            else:
                interpretation = (
                    f"模型存在显著的系统性偏见（p={p_value:.6f}），"
                    f"但中位数差异为0，需要进一步分析。"
                )
        else:
            interpretation = (
                f"模型不存在显著的系统性偏见（p={p_value:.6f}），"
                f"中位数差异={median_difference:.2f}。"
            )
    else:
        interpretation = "无法进行假设检验"

    # Cohen's Kappa 评估
    try:
        cohen_kappa_result = calculate_cohen_kappa(
            llm_risks_list,
            rule_risks_list,
            weights='quadratic'
        )
        kappa_value = cohen_kappa_result.get("value")
        kappa_ci = cohen_kappa_result.get("ci_95", (None, None))
    except Exception as e:
        logger.error(f"计算Cohen's Kappa失败: {e}")
        kappa_value = None
        kappa_ci = (None, None)

    # 解释Cohen's Kappa结果
    if kappa_value is not None:
        if kappa_value >= 0.81:
            kappa_interpretation = "几乎完美一致"
        elif kappa_value >= 0.61:
            kappa_interpretation = "实质性一致"
        elif kappa_value >= 0.41:
            kappa_interpretation = "中等一致"
        elif kappa_value >= 0.21:
            kappa_interpretation = "一般一致"
        elif kappa_value >= 0.00:
            kappa_interpretation = "轻微一致"
        else:
            kappa_interpretation = "不一致"
    else:
        kappa_interpretation = "无法计算"

    # 确保所有值都是 Python 原生类型，以便 JSON 序列化
    evaluation_result = {
        "model_name": model_name,
        "total_files": len(results),
        "total_cases": int(total_cases),
        "correct_matches": int(correct_matches),
        "incorrect_matches": int(incorrect_matches),
        "vr": float(vr),
        "vr_ci_95": (float(vr_ci_lower), float(vr_ci_upper)),
        "rule_vs_ground_truth_accuracy": {
            "correct_count": int(rule_vs_ground_truth_correct),
            "accuracy": float(rule_vs_gt_accuracy),
            "ci_95": (float(rule_vs_gt_ci_lower), float(rule_vs_gt_ci_upper))
        },
        "llm_vs_ground_truth_accuracy": {
            "correct_count": int(llm_vs_ground_truth_correct),
            "accuracy": float(llm_vs_gt_accuracy),
            "ci_95": (float(llm_vs_gt_ci_lower), float(llm_vs_gt_ci_upper))
        },
        "wilcoxon_test": {
            "statistic": float(statistic) if statistic is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
            "median_difference": float(median_difference),
            "is_significant": bool(p_value is not None and p_value < 0.05),
            "interpretation": interpretation
        },
        "cohen_kappa": {
            "value": float(kappa_value) if kappa_value is not None else None,
            "ci_95": (float(kappa_ci[0]) if kappa_ci[0] is not None else None,
                     float(kappa_ci[1]) if kappa_ci[1] is not None else None),
            "interpretation": kappa_interpretation
        }
    }

    logger.info("=" * 80)
    logger.info("聚合一致性评估结果")
    logger.info("=" * 80)
    logger.info(f"总文件数: {len(results)}")
    logger.info(f"有效案例数: {total_cases}")
    logger.info(f"正确匹配数: {correct_matches}")
    logger.info(f"错误匹配数: {incorrect_matches}")
    logger.info(f"准确率 (VR): {vr:.4f} (95% CI: [{vr_ci_lower:.4f}, {vr_ci_upper:.4f}])")
    logger.info(f"规则推理结果准确率 (vs Ground Truth): {rule_vs_gt_accuracy:.4f} (95% CI: [{rule_vs_gt_ci_lower:.4f}, {rule_vs_gt_ci_upper:.4f}])")
    logger.info(f"LLM结果准确率 (vs Ground Truth): {llm_vs_gt_accuracy:.4f} (95% CI: [{llm_vs_gt_ci_lower:.4f}, {llm_vs_gt_ci_upper:.4f}])")
    logger.info(f"Wilcoxon检验统计量: {statistic if statistic is not None else 'N/A'}")
    logger.info(f"Wilcoxon检验 p-value: {p_value if p_value is not None else 'N/A'}")
    logger.info(f"中位数差异: {median_difference:.2f} (负值=低估，正值=高估)")
    logger.info(f"统计显著性: {'是' if p_value is not None and p_value < 0.05 else '否'}")
    logger.info(f"解释: {interpretation}")
    if kappa_value is not None:
        kappa_ci_str = f"[{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]" if kappa_ci[0] is not None else "[N/A, N/A]"
        logger.info(f"Cohen's Kappa: {kappa_value:.4f} (95% CI: {kappa_ci_str})")
        logger.info(f"Kappa解释: {kappa_interpretation}")
    else:
        logger.info(f"Cohen's Kappa: 无法计算")
    logger.info("=" * 80)

    return {
        "model_name": model_name,
        "total_files": len(results),
        "evaluation_result": evaluation_result,
        "details": details  # 包含详细记录，但输出时可以选择不保存
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
    logger.info("聚合一致性评估结果摘要")
    logger.info("=" * 80)

    eval_result = evaluation_result.get("evaluation_result", {})
    logger.info(f"模型名称: {evaluation_result.get('model_name', 'N/A')}")
    logger.info(f"总文件数: {evaluation_result.get('total_files', 0)}")
    logger.info(f"有效案例数: {eval_result.get('total_cases', 0)}")
    logger.info(f"正确匹配数: {eval_result.get('correct_matches', 0)}")
    logger.info(f"错误匹配数: {eval_result.get('incorrect_matches', 0)}")

    vr = eval_result.get('vr', 0.0)
    vr_ci = eval_result.get('vr_ci_95', (0.0, 0.0))
    logger.info(f"准确率 (VR): {vr:.4f} (95% CI: [{vr_ci[0]:.4f}, {vr_ci[1]:.4f}])")

    rule_vs_gt = eval_result.get('rule_vs_ground_truth_accuracy', {})
    rule_vs_gt_acc = rule_vs_gt.get('accuracy', 0.0)
    rule_vs_gt_ci = rule_vs_gt.get('ci_95', (0.0, 0.0))
    logger.info(f"规则推理结果准确率 (vs Ground Truth): {rule_vs_gt_acc:.4f} (95% CI: [{rule_vs_gt_ci[0]:.4f}, {rule_vs_gt_ci[1]:.4f}])")

    llm_vs_gt = eval_result.get('llm_vs_ground_truth_accuracy', {})
    llm_vs_gt_acc = llm_vs_gt.get('accuracy', 0.0)
    llm_vs_gt_ci = llm_vs_gt.get('ci_95', (0.0, 0.0))
    logger.info(f"LLM结果准确率 (vs Ground Truth): {llm_vs_gt_acc:.4f} (95% CI: [{llm_vs_gt_ci[0]:.4f}, {llm_vs_gt_ci[1]:.4f}])")

    wilcoxon = eval_result.get('wilcoxon_test', {})
    logger.info(f"Wilcoxon检验 p-value: {wilcoxon.get('p_value', 'N/A')}")
    logger.info(f"中位数差异: {wilcoxon.get('median_difference', 'N/A'):.2f}")
    logger.info(f"统计显著性: {'是' if wilcoxon.get('is_significant', False) else '否'}")
    logger.info(f"解释: {wilcoxon.get('interpretation', 'N/A')}")

    cohen_kappa = eval_result.get('cohen_kappa', {})
    kappa_val = cohen_kappa.get('value')
    kappa_ci = cohen_kappa.get('ci_95', (None, None))
    if kappa_val is not None:
        kappa_ci_str = f"[{kappa_ci[0]:.4f}, {kappa_ci[1]:.4f}]" if kappa_ci[0] is not None else "[N/A, N/A]"
        logger.info(f"Cohen's Kappa: {kappa_val:.4f} (95% CI: {kappa_ci_str})")
        logger.info(f"Kappa解释: {cohen_kappa.get('interpretation', 'N/A')}")
    else:
        logger.info(f"Cohen's Kappa: 无法计算")
    logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="评估全局风险一致性（聚合一致性）"
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
        help="输出JSON文件路径（默认：项目根目录/data/aggregation_consistency_{model_name}.json）"
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
        output_file = project_root / "data" / f"aggregation_consistency_{model_dirname}.json"
    else:
        output_file = Path(args.output)

    # 检查路径是否存在
    if not result_dir.exists():
        logger.error(f"结果目录不存在: {result_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("评估全局风险一致性（聚合一致性）")
    logger.info("=" * 80)
    logger.info(f"结果目录: {result_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"输出文件: {output_file}")
    logger.info("=" * 80)

    # 执行评估
    try:
        evaluation_result = evaluate_aggregation_consistency(result_dir, args.model_name)

        # 打印摘要
        print_summary(evaluation_result)

        # 保存结果（不包含详细记录，因为可能很大）
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估RCT偏倚风险评估结果的脚本

计算以下指标：
1. 各个domain和overall_risk的准确率
2. 均偏差（Mean Deviation）：low=1, some concerns=3, high=6
3. Cohen's Kappa系数
4. 敏感度（Sensitivity）
5. 特异性（Specificity）
6. PPV（Positive Predictive Value）
7. NPV（Negative Predictive Value）
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy import stats

from loguru import logger


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


def cohen_kappa_ci(
    kappa: float,
    n: int,
    llm_values: List[str],
    gt_values: List[str],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    计算Cohen's Kappa的置信区间

    使用标准误差（SE）和正态分布近似。

    Args:
        kappa: Cohen's Kappa值
        n: 样本数量
        llm_values: LLM预测值列表（用于计算观察一致性和期望一致性）
        gt_values: 真实值列表
        confidence: 置信水平（默认0.95，即95%置信区间）

    Returns:
        (lower_bound, upper_bound) 置信区间的上下界
    """
    if n == 0 or kappa is None:
        return (None, None)

    # 计算观察一致性（Po）和期望一致性（Pe）
    from collections import Counter

    # 过滤有效值
    pairs = [(llm, gt) for llm, gt in zip(llm_values, gt_values)
             if llm and gt and llm.strip() and gt.strip()]

    if len(pairs) == 0:
        return (None, None)

    llm_clean, gt_clean = zip(*pairs)

    # 计算观察一致性（Po）：实际一致的比例
    po = sum(1 for llm, gt in zip(llm_clean, gt_clean) if llm == gt) / len(llm_clean)

    # 计算期望一致性（Pe）：随机一致的概率
    llm_dist = Counter(llm_clean)
    gt_dist = Counter(gt_clean)
    all_labels = set(llm_clean) | set(gt_clean)

    pe = 0.0
    for label in all_labels:
        llm_prob = llm_dist.get(label, 0) / len(llm_clean)
        gt_prob = gt_dist.get(label, 0) / len(llm_clean)
        pe += llm_prob * gt_prob

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


def risk_to_numeric(risk_str: str) -> Optional[int]:
    """
    将风险字符串转换为数值
    low = 1, some concerns = 3, high = 6

    Args:
        risk_str: 风险字符串

    Returns:
        数值，如果无法识别则返回None
    """
    if not risk_str:
        return None

    risk_lower = risk_str.lower().strip()

    if "low" in risk_lower and "some" not in risk_lower:
        return 1
    elif "some" in risk_lower:
        return 3
    elif "high" in risk_lower:
        return 6

    raise ValueError(f"无法识别风险值字符串: '{risk_str}'。期望包含 'low'、'some' 或 'high' 关键词。")


def risk_to_category(risk_str: str) -> Optional[str]:
    """
    将风险字符串转换为类别（用于二分类指标）

    Args:
        risk_str: 风险字符串

    Returns:
        "low" 或 "non_low"（包含some concerns和high），如果无法识别则返回None
    """
    if not risk_str:
        return None

    risk_lower = risk_str.lower().strip()

    if "low" in risk_lower and "some" not in risk_lower:
        return "low"
    elif "some" in risk_lower or "high" in risk_lower:
        return "non_low"

    raise ValueError(f"无法识别风险值字符串: '{risk_str}'。期望包含 'low'、'some' 或 'high' 关键词。")


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


def calculate_accuracy(llm_values: List[str], gt_values: List[str]) -> Dict[str, Any]:
    """
    计算准确率及其95%置信区间

    置信区间计算方法：使用 Wilson score interval。
    Wilson score interval 在样本量小和大时都表现良好，比简单的正态近似更准确，
    特别适用于比例估计的置信区间计算。

    Args:
        llm_values: LLM预测值列表
        gt_values: 真实值列表

    Returns:
        包含准确率值和95%置信区间的字典，格式为 {"value": float, "ci_95": (lower, upper)}
    """
    if len(llm_values) != len(gt_values) or len(llm_values) == 0:
        return {
            "value": 0.0,
            "ci_95": (0.0, 0.0)
        }

    correct = sum(1 for llm, gt in zip(llm_values, gt_values) if llm == gt)
    n = len(llm_values)
    accuracy = correct / n

    # 计算95%置信区间
    ci_lower, ci_upper = wilson_score_interval(correct, n, confidence=0.95)

    return {
        "value": accuracy,
        "ci_95": (ci_lower, ci_upper)
    }


def calculate_mean_deviation(llm_values: List[str], gt_values: List[str]) -> Optional[float]:
    """
    计算均偏差（Mean Deviation）
    low=1, some concerns=3, high=6

    Args:
        llm_values: LLM预测值列表
        gt_values: 真实值列表

    Returns:
        均偏差，如果无法计算则返回 None
    """
    if len(llm_values) != len(gt_values):
        return None

    deviations = []
    for llm, gt in zip(llm_values, gt_values):
        llm_num = risk_to_numeric(llm)
        gt_num = risk_to_numeric(gt)

        if llm_num is not None and gt_num is not None:
            deviation = abs(llm_num - gt_num)
            deviations.append(deviation)

    if len(deviations) == 0:
        return None

    return float(np.mean(deviations))


def _weighted_kappa(
    rater_a: np.ndarray,
    rater_b: np.ndarray,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    weights: str = 'quadratic'
) -> float:
    """
    计算加权Kappa系数（基于原始数值差异）

    参考用户提供的quadratic_weighted_kappa实现，但支持linear和quadratic两种权重类型。

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
            # 对于quadratic，使用 (diff / (max_rating - min_rating))^2
            # 对于linear，使用 diff / (max_rating - min_rating)
            if weights == 'linear':
                d = diff / (max_rating - min_rating) if (max_rating - min_rating) > 0 else 0
            else:  # quadratic
                d = (diff / (max_rating - min_rating)) ** 2.0 if (max_rating - min_rating) > 0 else 0

            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - (numerator / denominator) if denominator > 0 else 0.0


def calculate_cohen_kappa(
    llm_values: List[str],
    gt_values: List[str],
    weights: str = 'quadratic'
) -> Dict[str, Any]:
    """
    计算加权Cohen's Kappa系数及其95%置信区间

    基础 Cohen's Kappa（非加权）对所有不一致的情况一视同仁（比如 "诊断为 1 级" 和 "诊断为 2 级" 的不一致，与 "诊断为 1 级" 和 "诊断为 5 级" 的不一致，惩罚权重完全相同）；而加权 Kappa 针对「分类是有序的」场景（如医学评分：轻度 / 中度 / 重度、肿瘤分期 T1/T2/T3、疼痛评分 1-5 分），对 "相邻分类的轻微不一致" 惩罚轻，对 "跨分类的严重不一致" 惩罚重。

    使用加权Kappa来考虑不同类别之间错误的严重性：
    - low (1) -> some concerns (3) 的惩罚较小
    - low (1) -> high (6) 的惩罚较大

    权重类型：
    - 'linear': 线性权重 = |数值差异| / (max - min)
    - 'quadratic': 平方权重 = (数值差异)^2 / (max - min)^2

    置信区间计算方法：使用标准误差（SE）和正态分布近似。
    标准误差计算公式：SE = sqrt((Po * (1 - Po)) / (n * (1 - Pe)^2))，
    其中 Po 为观察一致性，Pe 为期望一致性，n 为样本数量。
    然后使用正态分布的临界值（z-score）计算95%置信区间。

    注意：当类别分布极度不均匀时（如某个类别占比超过95%），
    即使准确率很高，Kappa 也可能接近 0。这是因为 Kappa 考虑了"偶然一致"的概率，
    当某个类别占主导地位时，偶然一致的概率很高，导致 Kappa 较低。
    这是 Kappa 系数的数学特性，不是计算错误。

    Args:
        llm_values: LLM预测值列表（标准化后的风险值字符串）
        gt_values: 真实值列表（标准化后的风险值字符串）
        weights: 权重类型，'linear' 或 'quadratic'，默认为 'quadratic'

    Returns:
        包含Kappa值和95%置信区间的字典，格式为 {"value": float, "ci_95": (lower, upper)}，
        如果无法计算则返回 {"value": None, "ci_95": (None, None)}
    """
    if len(llm_values) != len(gt_values):
        return {"value": None, "ci_95": (None, None)}

    if weights not in ['linear', 'quadratic']:
        logger.warning(f"不支持的权重类型: {weights}，使用默认值 'quadratic'")
        weights = 'quadratic'

    # 过滤掉None值和空字符串，并转换为数值
    pairs = []
    for llm, gt in zip(llm_values, gt_values):
        if llm and gt and llm.strip() and gt.strip():
            llm_num = risk_to_numeric(llm)
            gt_num = risk_to_numeric(gt)
            if llm_num is not None and gt_num is not None:
                pairs.append((llm_num, gt_num))

    if len(pairs) == 0:
        return {"value": None, "ci_95": (None, None)}

    llm_nums, gt_nums = zip(*pairs)
    llm_nums = np.array(llm_nums)
    gt_nums = np.array(gt_nums)

    # 获取所有唯一的数值类别
    all_numeric_values = sorted(set(llm_nums) | set(gt_nums))

    # 检查类别分布
    from collections import Counter
    llm_dist = Counter(llm_nums)
    gt_dist = Counter(gt_nums)

    # 如果只有一个类别，Kappa 无法计算（返回 None）
    if len(all_numeric_values) < 2:
        logger.warning(f"加权Cohen's Kappa 无法计算：只有 {len(all_numeric_values)} 个类别（需要至少2个类别）")
        return {"value": None, "ci_95": (None, None)}

    # 检查类别分布是否极度不均匀
    total_samples = len(llm_nums)
    max_gt_freq = max(gt_dist.values()) if gt_dist else 0
    max_gt_ratio = max_gt_freq / total_samples if total_samples > 0 else 0

    # 如果某个类别占比超过 95%，说明分布极度不均匀
    if max_gt_ratio > 0.95:
        logger.debug(
            f"类别分布极度不均匀：最大类别占比 {max_gt_ratio:.2%}。"
            f"GT分布: {dict(gt_dist)}, LLM分布: {dict(llm_dist)}"
        )

    try:
        # 使用自定义加权Kappa函数（基于原始数值差异）
        min_rating = min(all_numeric_values)
        max_rating = max(all_numeric_values)

        kappa = _weighted_kappa(gt_nums, llm_nums, min_rating, max_rating, weights)

        # 计算95%置信区间
        kappa_ci = cohen_kappa_ci(kappa, len(llm_values), llm_values, gt_values, confidence=0.95)

        # 验证：如果准确率很高但 Kappa 很低，说明类别分布极不均匀
        accuracy = sum(1 for llm, gt in zip(llm_nums, gt_nums) if llm == gt) / len(llm_nums)
        if accuracy > 0.8 and abs(kappa) < 0.1:
            logger.debug(
                f"加权Cohen's Kappa 较低（{kappa:.4f}）但准确率较高（{accuracy:.4f}），"
                f"这是因为类别分布极不均匀（最大类别占比 {max_gt_ratio:.2%}）。"
                f"GT分布: {dict(gt_dist)}, LLM分布: {dict(llm_dist)}"
            )

        return {"value": kappa, "ci_95": kappa_ci}
    except Exception as e:
        logger.warning(f"计算加权Cohen's Kappa时出错: {e}")
        logger.debug(f"GT分布: {dict(gt_dist)}, LLM分布: {dict(llm_dist)}")
        return {"value": None, "ci_95": (None, None)}


def calculate_binary_metrics(llm_values: List[str], gt_values: List[str]) -> Dict[str, Optional[float]]:
    """
    计算二分类指标（敏感度、特异性、PPV、NPV）及其95%置信区间

    将"low risk"视为正类，"some concerns"或"high risk"视为负类。

    置信区间计算方法：所有指标（Sensitivity、Specificity、PPV、NPV）均使用 Wilson score interval。
    Wilson score interval 在样本量小和大时都表现良好，比简单的正态近似更准确，
    特别适用于比例估计的置信区间计算。

    Args:
        llm_values: LLM预测值列表
        gt_values: 真实值列表

    Returns:
        包含以下字段的字典：
        - sensitivity: 敏感度值（如果无法计算则为 None）
        - sensitivity_ci_95: 敏感度的95%置信区间，格式为 (lower, upper)
        - specificity: 特异性值（如果无法计算则为 None）
        - specificity_ci_95: 特异性的95%置信区间，格式为 (lower, upper)
        - ppv: 阳性预测值（如果无法计算则为 None）
        - ppv_ci_95: 阳性预测值的95%置信区间，格式为 (lower, upper)
        - npv: 阴性预测值（如果无法计算则为 None）
        - npv_ci_95: 阴性预测值的95%置信区间，格式为 (lower, upper)
        - tp, tn, fp, fn: 混淆矩阵的四个值
    """
    if len(llm_values) != len(gt_values):
        return {
            "sensitivity": None,
            "sensitivity_ci_95": (None, None),
            "specificity": None,
            "specificity_ci_95": (None, None),
            "ppv": None,
            "ppv_ci_95": (None, None),
            "npv": None,
            "npv_ci_95": (None, None)
        }

    # 转换为二分类标签
    llm_categories = [risk_to_category(llm) for llm in llm_values]
    gt_categories = [risk_to_category(gt) for gt in gt_values]

    # 过滤掉None值
    pairs = [(llm, gt) for llm, gt in zip(llm_categories, gt_categories)
             if llm is not None and gt is not None]

    if len(pairs) == 0:
        return {
            "sensitivity": None,
            "sensitivity_ci_95": (None, None),
            "specificity": None,
            "specificity_ci_95": (None, None),
            "ppv": None,
            "ppv_ci_95": (None, None),
            "npv": None,
            "npv_ci_95": (None, None)
        }

    llm_clean, gt_clean = zip(*pairs)

    # 计算混淆矩阵
    # TP: LLM预测为low，真实为low
    # TN: LLM预测为non_low，真实为non_low
    # FP: LLM预测为low，真实为non_low
    # FN: LLM预测为non_low，真实为low

    tp = sum(1 for llm, gt in zip(llm_clean, gt_clean) if llm == "low" and gt == "low")
    tn = sum(1 for llm, gt in zip(llm_clean, gt_clean) if llm == "non_low" and gt == "non_low")
    fp = sum(1 for llm, gt in zip(llm_clean, gt_clean) if llm == "low" and gt == "non_low")
    fn = sum(1 for llm, gt in zip(llm_clean, gt_clean) if llm == "non_low" and gt == "low")

    # 敏感度（Sensitivity, TPR）：TP / (TP + FN)
    # 在所有真实为low的样本中，LLM正确识别为low的比例
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    sensitivity_ci = wilson_score_interval(tp, tp + fn, confidence=0.95) if (tp + fn) > 0 else (None, None)

    # 特异性（Specificity, TNR）：TN / (TN + FP)
    # 在所有真实为non_low的样本中，LLM正确识别为non_low的比例
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    specificity_ci = wilson_score_interval(tn, tn + fp, confidence=0.95) if (tn + fp) > 0 else (None, None)

    # PPV（Positive Predictive Value）：TP / (TP + FP)
    # 在LLM预测为low的样本中，真正为low的比例
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    ppv_ci = wilson_score_interval(tp, tp + fp, confidence=0.95) if (tp + fp) > 0 else (None, None)

    # NPV（Negative Predictive Value）：TN / (TN + FN)
    # 在LLM预测为non_low的样本中，真正为non_low的比例
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    npv_ci = wilson_score_interval(tn, tn + fn, confidence=0.95) if (tn + fn) > 0 else (None, None)

    return {
        "sensitivity": sensitivity,
        "sensitivity_ci_95": sensitivity_ci,
        "specificity": specificity,
        "specificity_ci_95": specificity_ci,
        "ppv": ppv,
        "ppv_ci_95": ppv_ci,
        "npv": npv,
        "npv_ci_95": npv_ci,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def evaluate_rct_results(
    result_dir: Path,
    model_name: str,
    weights: str = 'quadratic'
) -> Dict[str, Any]:
    """
    评估RCT偏倚风险评估结果

    Args:
        result_dir: 结果目录路径
        model_name: 模型名称
        weights: 加权Kappa的权重类型，'linear' 或 'quadratic'，默认为 'quadratic'

    Returns:
        评估结果字典
    """
    # 加载所有结果文件
    results = load_result_files(result_dir)

    if len(results) == 0:
        logger.error("未找到任何结果文件")
        return {}

    # 定义需要评估的字段
    domains = [
        "randomisation_process_judgment",
        "intended_interventions_judgment",
        "missing_outcome_data_judgment",
        "measurement_outcome_judgment",
        "selection_reported_result_judgment",
        "overall_risk"
    ]

    evaluation_results = {}

    for domain in domains:
        logger.info(f"评估域: {domain}")

        # 提取LLM和真实值
        llm_values = []
        gt_values = []

        for result in results:
            llm_result = result.get("llm_result_normalized", {})
            gt_result = result.get("ground_truth_normalized", {})

            llm_value = llm_result.get(domain, "")
            gt_value = gt_result.get(domain, "")

            if llm_value and gt_value:
                llm_values.append(llm_value)
                gt_values.append(gt_value)

        if len(llm_values) == 0:
            logger.warning(f"域 {domain} 没有有效数据")
            continue

        logger.info(f"  有效样本数: {len(llm_values)}")

        # 计算各项指标
        accuracy = calculate_accuracy(llm_values, gt_values)
        mean_deviation = calculate_mean_deviation(llm_values, gt_values)
        cohen_kappa = calculate_cohen_kappa(llm_values, gt_values, weights=weights)
        binary_metrics = calculate_binary_metrics(llm_values, gt_values)

        evaluation_results[domain] = {
            "n_samples": len(llm_values),
            "accuracy": accuracy,
            "mean_deviation": mean_deviation,
            "cohen_kappa": cohen_kappa,
            "sensitivity": binary_metrics["sensitivity"],
            "sensitivity_ci_95": binary_metrics["sensitivity_ci_95"],
            "specificity": binary_metrics["specificity"],
            "specificity_ci_95": binary_metrics["specificity_ci_95"],
            "ppv": binary_metrics["ppv"],
            "ppv_ci_95": binary_metrics["ppv_ci_95"],
            "npv": binary_metrics["npv"],
            "npv_ci_95": binary_metrics["npv_ci_95"],
            "confusion_matrix": {
                "tp": binary_metrics.get("tp", 0),
                "tn": binary_metrics.get("tn", 0),
                "fp": binary_metrics.get("fp", 0),
                "fn": binary_metrics.get("fn", 0)
            }
        }

        # 格式化输出，包含置信区间
        acc_val = accuracy["value"]
        acc_ci = accuracy["ci_95"]
        acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
        acc_ci_str = f"({acc_ci[0]:.4f}; {acc_ci[1]:.4f})" if acc_ci[0] is not None else "(N/A; N/A)"
        logger.info(f"  准确率: {acc_str} (95% CI: {acc_ci_str})")

        mean_dev_str = f"{mean_deviation:.4f}" if mean_deviation is not None else "N/A"
        logger.info(f"  均偏差: {mean_dev_str}")

        kappa_val = cohen_kappa["value"]
        kappa_ci = cohen_kappa["ci_95"]
        kappa_str = f"{kappa_val:.4f}" if kappa_val is not None else "N/A"
        kappa_ci_str = f"({kappa_ci[0]:.4f}; {kappa_ci[1]:.4f})" if kappa_ci[0] is not None else "(N/A; N/A)"
        logger.info(f"  Cohen's Kappa: {kappa_str} (95% CI: {kappa_ci_str})")

        sens = binary_metrics['sensitivity']
        sens_ci = binary_metrics['sensitivity_ci_95']
        sens_str = f"{sens:.4f}" if sens is not None else "N/A"
        sens_ci_str = f"({sens_ci[0]:.4f}; {sens_ci[1]:.4f})" if sens_ci[0] is not None else "(N/A; N/A)"
        logger.info(f"  敏感度: {sens_str} (95% CI: {sens_ci_str})")

        spec = binary_metrics['specificity']
        spec_ci = binary_metrics['specificity_ci_95']
        spec_str = f"{spec:.4f}" if spec is not None else "N/A"
        spec_ci_str = f"({spec_ci[0]:.4f}; {spec_ci[1]:.4f})" if spec_ci[0] is not None else "(N/A; N/A)"
        logger.info(f"  特异性: {spec_str} (95% CI: {spec_ci_str})")

        ppv_val = binary_metrics['ppv']
        ppv_ci = binary_metrics['ppv_ci_95']
        ppv_str = f"{ppv_val:.4f}" if ppv_val is not None else "N/A"
        ppv_ci_str = f"({ppv_ci[0]:.4f}; {ppv_ci[1]:.4f})" if ppv_ci[0] is not None else "(N/A; N/A)"
        logger.info(f"  PPV: {ppv_str} (95% CI: {ppv_ci_str})")

        npv_val = binary_metrics['npv']
        npv_ci = binary_metrics['npv_ci_95']
        npv_str = f"{npv_val:.4f}" if npv_val is not None else "N/A"
        npv_ci_str = f"({npv_ci[0]:.4f}; {npv_ci[1]:.4f})" if npv_ci[0] is not None else "(N/A; N/A)"
        logger.info(f"  NPV: {npv_str} (95% CI: {npv_ci_str})")

    return {
        "model_name": model_name,
        "total_files": len(results),
        "evaluation_results": evaluation_results
    }


def print_summary(evaluation_result: Dict[str, Any]):
    """
    打印评估结果摘要

    Args:
        evaluation_result: 评估结果字典
    """
    logger.info("=" * 80)
    logger.info("评估结果摘要")
    logger.info("=" * 80)
    logger.info(f"模型名称: {evaluation_result.get('model_name', 'N/A')}")
    logger.info(f"总文件数: {evaluation_result.get('total_files', 0)}")
    logger.info("=" * 80)

    eval_results = evaluation_result.get("evaluation_results", {})

    # 打印表格格式的结果
    domains = list(eval_results.keys())
    if not domains:
        logger.info("没有评估结果")
        return

    # 表头
    header = f"{'Domain':<40} {'Accuracy':<10} {'Mean Dev':<10} {'Kappa':<10} {'Sens':<10} {'Spec':<10} {'PPV':<10} {'NPV':<10}"
    logger.info(header)
    logger.info("-" * 120)

    # 数据行
    for domain in domains:
        result = eval_results[domain]
        # 格式化函数：处理 None 值和字典格式
        def format_value(val, width=10):
            if val is None:
                return f"{'N/A':<{width}}"
            if isinstance(val, dict):
                # 处理新的字典格式（包含 value 和 ci_95）
                val_value = val.get("value")
                if val_value is None:
                    return f"{'N/A':<{width}}"
                return f"{val_value:<{width}.4f}"
            return f"{val:<{width}.4f}"

        # 提取值（如果是字典则提取 value）
        accuracy_val = result['accuracy']['value'] if isinstance(result['accuracy'], dict) else result['accuracy']
        kappa_val = result['cohen_kappa']['value'] if isinstance(result['cohen_kappa'], dict) else result['cohen_kappa']

        row = (
            f"{domain:<40} "
            f"{format_value(accuracy_val)} "
            f"{format_value(result['mean_deviation'])} "
            f"{format_value(kappa_val)} "
            f"{format_value(result['sensitivity'])} "
            f"{format_value(result['specificity'])} "
            f"{format_value(result['ppv'])} "
            f"{format_value(result['npv'])}"
        )
        logger.info(row)

    logger.info("=" * 80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="评估RCT偏倚风险评估结果"
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
        default="openai/gpt-5.1",
        help="模型名称（默认：openai/gpt-5.1）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出JSON文件路径（默认：项目根目录/data/evaluation_result_{model_name}.json）"
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=['linear', 'quadratic'],
        default='quadratic',
        help="加权Kappa的权重类型：'linear' 或 'quadratic'（默认：quadratic）。"
             "quadratic对类别差异的惩罚更大，更适合评估严重性差异较大的错误。"
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
        output_file = project_root / "data" / f"evaluation_result_{model_dirname}.json"
    else:
        output_file = Path(args.output)

    # 检查路径是否存在
    if not result_dir.exists():
        logger.error(f"结果目录不存在: {result_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("评估RCT偏倚风险评估结果")
    logger.info("=" * 80)
    logger.info(f"结果目录: {result_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    logger.info(f"权重类型: {args.weights}")
    logger.info(f"输出文件: {output_file}")
    logger.info("=" * 80)

    # 执行评估
    try:
        evaluation_result = evaluate_rct_results(result_dir, args.model_name, weights=args.weights)

        # 打印摘要
        print_summary(evaluation_result)

        # 保存结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

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


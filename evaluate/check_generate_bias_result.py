#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查生成的RCT偏倚风险评估结果文件

检查 llm_result 字段中五个 domain 字段是否都正确：
- signaling_questions 不为空，且数量匹配
- domain_risk 不为空

支持识别错误类型（缺少信号问题或缺少domain_risk），打印错误的元素id，
并提供可选参数控制是否删除这些错误元素的文件。
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from loguru import logger


# 各domain期望的信号问题数量
EXPECTED_SIGNALING_QUESTIONS_COUNT = {
    "randomisation_process": 3,
    "intended_interventions": 7,
    "missing_outcome_data": 4,
    "measurement_outcome": 5,
    "selection_reported_result": 3,
}

# 所有需要检查的domain列表
DOMAIN_KEYS = list(EXPECTED_SIGNALING_QUESTIONS_COUNT.keys())


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


def check_domain(
    domain_key: str,
    domain_data: Dict[str, Any],
    record_id: str,
    llm_result_raw: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    检查单个domain的数据是否正确

    Args:
        domain_key: domain键名
        domain_data: domain数据字典
        record_id: 记录ID
        llm_result_raw: 原始LLM结果（用于检查是否有可恢复的数据）

    Returns:
        (是否通过检查, 错误信息列表)
    """
    errors = []
    expected_count = EXPECTED_SIGNALING_QUESTIONS_COUNT.get(domain_key, 0)

    # 检查 signaling_questions
    signaling_questions = domain_data.get("signaling_questions")
    if signaling_questions is None:
        errors.append(f"缺少 signaling_questions 字段（为 null）")
        # 检查 llm_result_raw 中是否有可恢复的数据
        if llm_result_raw:
            raw_domain = llm_result_raw.get(domain_key, {})
            raw_response = raw_domain.get("response", {})
            if isinstance(raw_response, dict) and raw_response.get("signaling_questions"):
                errors.append(f"（提示：llm_result_raw.{domain_key}.response 中有可恢复的数据）")
    elif not isinstance(signaling_questions, list):
        errors.append(f"signaling_questions 不是列表类型（实际类型: {type(signaling_questions).__name__}）")
    elif len(signaling_questions) == 0:
        errors.append(f"signaling_questions 为空列表")
    elif len(signaling_questions) != expected_count:
        errors.append(
            f"signaling_questions 数量不匹配: "
            f"期望 {expected_count} 个，实际 {len(signaling_questions)} 个"
        )

    # 检查 domain_risk
    domain_risk = domain_data.get("domain_risk")
    if domain_risk is None:
        errors.append(f"缺少 domain_risk 字段")
        # 检查 llm_result_raw 中是否有可恢复的数据
        if llm_result_raw:
            raw_domain = llm_result_raw.get(domain_key, {})
            raw_response = raw_domain.get("response", {})
            if isinstance(raw_response, dict) and raw_response.get("domain_risk"):
                errors.append(f"（提示：llm_result_raw.{domain_key}.response 中有可恢复的数据）")
    elif not isinstance(domain_risk, str):
        errors.append(f"domain_risk 不是字符串类型")
    elif len(domain_risk.strip()) == 0:
        errors.append(f"domain_risk 为空字符串")
        # 检查 llm_result_raw 中是否有可恢复的数据
        if llm_result_raw:
            raw_domain = llm_result_raw.get(domain_key, {})
            raw_response = raw_domain.get("response", {})
            if isinstance(raw_response, dict) and raw_response.get("domain_risk"):
                errors.append(f"（提示：llm_result_raw.{domain_key}.response 中有可恢复的数据）")

    is_valid = len(errors) == 0
    return is_valid, errors


def check_single_file(
    file_path: Path
) -> Tuple[bool, Dict[str, Any]]:
    """
    检查单个结果文件

    Args:
        file_path: 结果文件路径

    Returns:
        (是否通过检查, 检查结果详情)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, {
            "error_type": "JSON解析错误",
            "error_message": str(e),
            "file_path": str(file_path)
        }
    except Exception as e:
        return False, {
            "error_type": "文件读取错误",
            "error_message": str(e),
            "file_path": str(file_path)
        }

    # 获取记录ID
    metadata = data.get("metadata", {})
    record_id = metadata.get("id", "")
    if not record_id:
        # 尝试从文件名提取ID
        filename = file_path.name
        if filename.startswith("rct_bias_result_") and filename.endswith(".json"):
            record_id = filename[len("rct_bias_result_"):-len(".json")]

    # 检查 llm_result 字段
    llm_result = data.get("llm_result")
    if llm_result is None:
        return False, {
            "record_id": record_id,
            "error_type": "缺少 llm_result 字段",
            "file_path": str(file_path),
            "domain_errors": {}
        }

    # 获取 llm_result_raw 用于检查是否有可恢复的数据
    llm_result_raw = data.get("llm_result_raw")

    # 检查各个domain
    domain_errors = {}
    all_valid = True

    for domain_key in DOMAIN_KEYS:
        domain_data = llm_result.get(domain_key)
        if domain_data is None:
            domain_errors[domain_key] = ["缺少该domain字段"]
            all_valid = False
        else:
            is_valid, errors = check_domain(domain_key, domain_data, record_id, llm_result_raw=llm_result_raw)
            if not is_valid:
                domain_errors[domain_key] = errors
                all_valid = False

    # 检查 overall_risk
    overall_risk_errors = []
    overall_risk = llm_result.get("overall_risk", "")
    if not overall_risk or (isinstance(overall_risk, str) and len(overall_risk.strip()) == 0):
        overall_risk_errors.append("overall_risk 为空字符串或缺失")
        # 检查 llm_result_raw 中是否有可恢复的数据
        if llm_result_raw:
            overall_risk_raw = llm_result_raw.get("overall_risk_raw", {})
            if overall_risk_raw and overall_risk_raw.get("raw_response"):
                overall_risk_errors.append("（提示：llm_result_raw.overall_risk_raw.raw_response 中有可恢复的数据）")
            # 也检查 overall_risk_judgement
            overall_risk_judgement = llm_result_raw.get("overall_risk_judgement")
            if overall_risk_judgement:
                overall_risk_errors.append("（提示：llm_result_raw.overall_risk_judgement 中有数据）")
        all_valid = False

    return all_valid, {
        "record_id": record_id,
        "file_path": str(file_path),
        "domain_errors": domain_errors,
        "overall_risk_errors": overall_risk_errors
    }


def check_model_results(
    model_output_dir: Path,
    delete_invalid: bool = False
) -> Dict[str, Any]:
    """
    检查指定模型目录下的所有结果文件

    Args:
        model_output_dir: 模型输出目录路径
        delete_invalid: 是否删除无效文件（默认：False）

    Returns:
        检查结果统计信息
    """
    if not model_output_dir.exists():
        logger.error(f"模型输出目录不存在: {model_output_dir}")
        return {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "deleted_files": 0,
            "errors_by_type": defaultdict(list)
        }

    # 查找所有结果文件
    pattern = "rct_bias_result_*.json"
    json_files = list(model_output_dir.glob(pattern))

    if len(json_files) == 0:
        logger.warning(f"在目录 {model_output_dir} 中未找到任何结果文件")
        return {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "deleted_files": 0,
            "errors_by_type": defaultdict(list)
        }

    logger.info(f"找到 {len(json_files)} 个结果文件")

    # 统计信息
    stats = {
        "total_files": len(json_files),
        "valid_files": 0,
        "invalid_files": 0,
        "deleted_files": 0,
        "errors_by_type": defaultdict(list)  # 按错误类型分组
    }

    # 检查每个文件
    for file_path in json_files:
        is_valid, check_result = check_single_file(file_path)

        if is_valid:
            stats["valid_files"] += 1
        else:
            stats["invalid_files"] += 1
            record_id = check_result.get("record_id", "未知ID")

            # 分析错误类型
            domain_errors = check_result.get("domain_errors", {})
            overall_risk_errors = check_result.get("overall_risk_errors", [])
            error_summary = []

            for domain_key, errors in domain_errors.items():
                for error in errors:
                    error_summary.append(f"{domain_key}: {error}")
                    # 判断错误类型
                    if "signaling_questions" in error.lower():
                        stats["errors_by_type"]["缺少或错误的信号问题"].append(record_id)
                    elif "domain_risk" in error.lower():
                        stats["errors_by_type"]["缺少或错误的domain_risk"].append(record_id)

            # 检查 overall_risk 错误
            for error in overall_risk_errors:
                error_summary.append(f"overall_risk: {error}")
                stats["errors_by_type"]["缺少或错误的overall_risk"].append(record_id)

            # 如果没有domain错误和overall_risk错误，可能是其他错误（如缺少llm_result）
            if not domain_errors and not overall_risk_errors:
                error_type = check_result.get("error_type", "未知错误")
                error_summary.append(error_type)
                stats["errors_by_type"][error_type].append(record_id)

            logger.error(
                f"✗ 文件检查失败: {file_path.name}\n"
                f"  记录ID: {record_id}\n"
                f"  错误详情: {'; '.join(error_summary)}"
            )

            # 如果需要删除无效文件
            if delete_invalid:
                try:
                    file_path.unlink()
                    stats["deleted_files"] += 1
                    logger.warning(f"  已删除无效文件: {file_path.name}")
                except Exception as e:
                    logger.error(f"  删除文件失败: {e}")

    return stats


def print_check_summary(stats: Dict[str, Any], model_name: str):
    """
    打印检查结果摘要

    Args:
        stats: 检查统计信息
        model_name: 模型名称
    """
    logger.info("=" * 80)
    logger.info(f"检查结果摘要 - {model_name}")
    logger.info("=" * 80)

    logger.info(f"总文件数: {stats['total_files']}")
    logger.info(f"有效文件: {stats['valid_files']}")
    logger.info(f"无效文件: {stats['invalid_files']}")

    if stats.get("deleted_files", 0) > 0:
        logger.info(f"已删除文件: {stats['deleted_files']}")

    # 打印错误类型统计
    errors_by_type = stats.get("errors_by_type", {})
    if errors_by_type:
        logger.info("\n错误类型统计:")
        for error_type, record_ids in errors_by_type.items():
            unique_ids = list(set(record_ids))  # 去重
            logger.info(f"  {error_type}: {len(unique_ids)} 个记录")
            if len(unique_ids) <= 20:  # 如果数量不多，打印所有ID
                logger.info(f"    记录ID: {', '.join(unique_ids)}")
            else:  # 如果数量多，只打印前20个
                logger.info(f"    记录ID (前20个): {', '.join(unique_ids[:20])} ...")

    logger.info("=" * 80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="检查生成的RCT偏倚风险评估结果文件"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="结果文件根目录路径（默认：项目根目录/data/rct_bias_results）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称（必需），如：openai/gpt-5.1"
    )
    parser.add_argument(
        "--delete-invalid",
        action="store_true",
        help="删除无效文件（默认：不删除）"
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
    if args.results_dir is None:
        results_dir = project_root / "data" / "rct_bias_results"
    else:
        results_dir = Path(args.results_dir)

    # 根据模型名称和模式计算模型输出目录
    model_dirname = model_name_to_dirname(args.model_name)
    mode_suffix = get_mode_suffix(args.mode)
    model_output_dir = results_dir / f"{model_dirname}{mode_suffix}"

    logger.info("=" * 80)
    logger.info("检查 RCT 偏倚风险评估结果文件")
    logger.info("=" * 80)
    logger.info(f"结果根目录: {results_dir}")
    logger.info(f"模型输出目录: {model_output_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    if args.delete_invalid:
        logger.warning("⚠️  删除模式: 将删除无效文件")
    logger.info("=" * 80)

    # 执行检查
    try:
        stats = check_model_results(
            model_output_dir=model_output_dir,
            delete_invalid=args.delete_invalid
        )

        # 打印摘要
        print_check_summary(stats, args.model_name)

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

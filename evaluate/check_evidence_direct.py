#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查生成的SSR证据提取结果文件（Direct/Origin模式）

检查 llm_result 字段中：
- risk_of_bias 是否不为空字符串

Direct模式和Origin模式都不需要检索，所以只需要检查risk_of_bias字段。
- Direct模式：只提供正确答案句子
- Origin模式：提供全量context_list句子，但不进行检索

支持识别错误类型，打印错误的元素id，
并提供可选参数控制是否删除这些错误元素的文件。
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from loguru import logger


def model_name_to_dirname(model_name: str) -> str:
    """
    将模型名称转换为安全的目录名（处理特殊字符）

    Args:
        model_name: 模型名称，如 "openai/gpt-5.1"

    Returns:
        安全的目录名，如 "openai_gpt-5.1"
    """
    return model_name.replace("/", "_").replace("\\", "_")


# Direct模式不需要模式后缀，因为direct模式本身就是一个独立的模式
# 在generate_evidence.py中，direct模式下的输出目录就是 {model_dirname}，没有后缀


def check_llm_result(
    llm_result: Dict[str, Any],
    record_id: str
) -> Tuple[bool, List[str]]:
    """
    检查 llm_result 字段的数据是否正确（Direct/Origin模式，只检查risk_of_bias）

    Args:
        llm_result: llm_result 数据字典
        record_id: 记录ID

    Returns:
        (是否通过检查, 错误信息列表)
    """
    errors = []

    # 只检查 risk_of_bias 字段
    risk_of_bias = llm_result.get("risk_of_bias")
    if risk_of_bias is None:
        errors.append("缺少 risk_of_bias 字段")
    elif not isinstance(risk_of_bias, str):
        errors.append(f"risk_of_bias 不是字符串类型（实际类型: {type(risk_of_bias).__name__}）")
    elif len(risk_of_bias.strip()) == 0:
        errors.append("risk_of_bias 为空字符串")

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
        if filename.startswith("ssr_evidence_result_") and filename.endswith(".json"):
            record_id = filename[len("ssr_evidence_result_"):-len(".json")]

    # 检查 llm_result 字段
    llm_result = data.get("llm_result")
    if llm_result is None:
        return False, {
            "record_id": record_id,
            "error_type": "缺少 llm_result 字段",
            "file_path": str(file_path),
            "errors": []
        }

    # 检查 llm_result 的各个字段
    is_valid, errors = check_llm_result(llm_result, record_id)

    return is_valid, {
        "record_id": record_id,
        "file_path": str(file_path),
        "errors": errors
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
    pattern = "ssr_evidence_result_*.json"
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
            errors = check_result.get("errors", [])
            error_summary = []

            for error in errors:
                error_summary.append(error)
                # 判断错误类型
                if "risk_of_bias" in error.lower():
                    stats["errors_by_type"]["risk_of_bias 字段错误"].append(record_id)
                elif "缺少" in error:
                    if "llm_result" in error:
                        stats["errors_by_type"]["缺少 llm_result 字段"].append(record_id)
                    else:
                        stats["errors_by_type"]["缺少必要字段"].append(record_id)

            # 如果没有具体错误，可能是其他错误（如JSON解析错误）
            if not errors:
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


def print_check_summary(stats: Dict[str, Any], model_name: str, mode: str = "direct"):
    """
    打印检查结果摘要

    Args:
        stats: 检查统计信息
        model_name: 模型名称
        mode: 模式（direct 或 origin）
    """
    mode_name = "Direct模式" if mode == "direct" else "Origin模式"
    logger.info("=" * 80)
    logger.info(f"检查结果摘要 - {model_name} ({mode_name})")
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
        description="检查生成的SSR证据提取结果文件（Direct/Origin模式）"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="结果文件根目录路径（默认：根据--mode自动选择，direct->ssr_evidence_results_direct，origin->ssr_evidence_results_origin）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称（必需），如：openai/gpt-5.1"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="direct",
        choices=["direct", "origin"],
        help="模式（默认：direct）。direct=只提供正确答案句子，origin=提供全量context_list但不检索"
    )
    parser.add_argument(
        "--delete-invalid",
        action="store_true",
        help="删除无效文件（默认：不删除）"
    )

    args = parser.parse_args()

    # 获取脚本所在目录和项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 设置默认路径（根据模式选择不同的目录）
    if args.results_dir is None:
        if args.mode == "origin":
            results_dir = project_root / "data" / "ssr_evidence_results_origin"
        else:  # direct
            results_dir = project_root / "data" / "ssr_evidence_results_direct"
    else:
        results_dir = Path(args.results_dir)

    # 根据模型名称计算模型输出目录
    # Direct/Origin模式下，模型输出目录就是 {model_dirname}，没有模式后缀
    # 因为在generate_evidence.py中，get_mode_suffix("direct"/"origin")返回""（不在已知模式列表中）
    model_dirname = model_name_to_dirname(args.model_name)
    model_output_dir = results_dir / model_dirname

    mode_name = "Direct模式" if args.mode == "direct" else "Origin模式"
    logger.info("=" * 80)
    logger.info(f"检查 SSR 证据提取结果文件（{mode_name}）")
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
        print_check_summary(stats, args.model_name, args.mode)

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

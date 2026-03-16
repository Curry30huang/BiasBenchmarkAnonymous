#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成SSR证据提取结果的脚本

从 ssr_dataset_merged.json 中读取数据，调用 extract_ssr_evidence 函数获取 LLM 评估结果，
存储大模型输出的证据索引、风险等级，以及标准答案。
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger

from bias_assessment.evidence_ssr import extract_ssr_evidence


def model_name_to_dirname(model_name: str) -> str:
    """
    将模型名称转换为安全的目录名（处理特殊字符）

    Args:
        model_name: 模型名称，如 "openai/gpt-5.1"

    Returns:
        安全的目录名，如 "openai_gpt-5.1"
    """
    # 将斜杠替换为下划线，其他特殊字符保持不变
    return model_name.replace("/", "_").replace("\\", "_")


def get_mode_suffix(mode: str = "default") -> str:
    """
    根据模式获取目录后缀

    Args:
        mode: 模式，可选值：default, cot, agent, cot_agent, tool, direct, origin

    Returns:
        目录后缀，如 "_cot" 或 ""
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
    elif mode == "direct":
        # direct模式使用单独的输出根目录（ssr_evidence_results_direct），这里不需要后缀
        return ""
    elif mode == "origin":
        # origin模式使用单独的输出根目录（ssr_evidence_results_origin），这里不需要后缀
        return ""
    else:
        logger.warning(f"未知模式 '{mode}'，使用默认模式")
        return ""


def get_completed_ids(model_output_dir: Path) -> set:
    """
    扫描模型输出目录，获取已完成的记录ID集合

    Args:
        model_output_dir: 模型输出目录路径

    Returns:
        已完成的记录ID集合
    """
    completed_ids = set()

    if not model_output_dir.exists():
        logger.info(f"输出目录不存在: {model_output_dir}，将创建新目录")
        return completed_ids

    # 扫描目录中的所有JSON文件
    pattern = "ssr_evidence_result_*.json"
    for file_path in model_output_dir.glob(pattern):
        # 从文件名中提取ID
        # 文件名格式: ssr_evidence_result_{id}.json
        filename = file_path.name
        if filename.startswith("ssr_evidence_result_") and filename.endswith(".json"):
            # 提取ID部分
            record_id = filename[len("ssr_evidence_result_"):-len(".json")]
            completed_ids.add(record_id)

    return completed_ids


def extract_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    从原始记录中提取标准答案

    Args:
        record: 原始JSON记录

    Returns:
        包含标准答案的字典
    """
    return {
        "label": record.get("label", ""),
        "answers": record.get("answers", []),
    }


def process_single_record(
    record: Dict[str, Any],
    idx: int,
    total: int,
    model_output_dir: Path,
    model_name: str,
    force: bool,
    completed_ids: set,
    stats_lock: Lock,
    stats: Dict[str, Any],
    mode: str = "default"
) -> Optional[str]:
    """
    处理单条记录

    Args:
        record: 原始数据记录
        idx: 当前索引（从1开始）
        total: 总记录数
        model_output_dir: 模型输出目录
        model_name: 模型名称
        force: 是否强制重新处理
        completed_ids: 已完成的记录ID集合
        stats_lock: 统计信息锁
        stats: 统计信息字典

    Returns:
        如果成功返回输出文件名，否则返回 None
    """
    record_id = record.get("id", "")
    study = record.get("study", "")
    bias = record.get("bias", "")
    question = record.get("question", "")
    context_list = record.get("context_list", [])

    logger.info(f"[{idx}/{total}] 处理: {study} (ID: {record_id})")

    # 检查必要字段
    if not record_id:
        logger.warning(f"  [{idx}] 跳过: 缺少 id")
        with stats_lock:
            stats["skipped"] += 1
        return None

    if not bias:
        logger.warning(f"  [{idx}] 跳过: 缺少 bias")
        with stats_lock:
            stats["skipped"] += 1
        return None

    if not question:
        logger.warning(f"  [{idx}] 跳过: 缺少 question")
        with stats_lock:
            stats["skipped"] += 1
        return None

    if not context_list:
        logger.warning(f"  [{idx}] 跳过: 缺少 context_list 或为空")
        with stats_lock:
            stats["skipped"] += 1
        return None

    # 根据模式决定传入的句子列表
    # - direct：只传入 answers 对应的正确句子（无需检索）
    # - origin：传入整个 context_list（无需检索）
    # - 其他模式：传入整个 context_list（需要检索）
    if mode == "direct":
        answers = record.get("answers", [])
        if not answers:
            logger.warning(f"  [{idx}] 跳过: direct 模式需要 answers 字段，但缺少或为空")
            with stats_lock:
                stats["skipped"] += 1
            return None

        # 从 context_list 中提取 answers 对应的句子
        # answers 是整数列表，表示正确的支撑句索引
        evidence_sentences = []
        for answer_idx in answers:
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(context_list):
                evidence_sentences.append(context_list[answer_idx])
            else:
                logger.warning(f"  [{idx}] 警告: answers 中的索引 {answer_idx} 超出范围 [0, {len(context_list) - 1}]，跳过")

        if not evidence_sentences:
            logger.warning(f"  [{idx}] 跳过: 无法从 context_list 中提取有效的证据句子")
            with stats_lock:
                stats["skipped"] += 1
            return None

        # 使用提取的证据句子
        sentences_to_use = evidence_sentences
        logger.info(f"  [{idx}] Direct模式：使用 {len(evidence_sentences)} 个正确的支撑句子")
    elif mode == "origin":
        # origin模式：使用整个 context_list，但提示词要求不做检索，只输出风险等级
        sentences_to_use = context_list
        logger.info(f"  [{idx}] Origin模式：使用全量句子 {len(context_list)} 条（不做检索）")
    else:
        # 其他模式：使用整个 context_list
        sentences_to_use = context_list

    # 检查等幂性：如果ID已在已完成集合中且不强制重新处理，则跳过
    if record_id in completed_ids and not force:
        logger.info(f"  [{idx}] ⊘ 跳过: 记录已处理 (ID: {record_id})")
        with stats_lock:
            stats["already_exists"] += 1
        return None

    # 生成输出文件名（基于 id）
    filename = f"ssr_evidence_result_{record_id}.json"
    output_file = model_output_dir / filename

    try:
        # 调用 LLM 进行证据提取
        logger.info(f"  [{idx}] 调用 LLM 进行证据提取（模式: {mode}）...")
        llm_result = extract_ssr_evidence(
            bias=bias,
            context_list=sentences_to_use,
            question=question,
            model_name=model_name,
            mode=mode
        )

        # 提取标准答案
        ground_truth = extract_ground_truth(record)

        # 组合结果
        combined_result = {
            # 原始元数据
            "metadata": {
                "id": record_id,
                "study": study,
                "bias": bias,
            },
            # LLM评估结果
            "llm_result": {
                "success": llm_result.get("success", False),
                "evidence_indices": llm_result.get("evidence_indices", []),
                "evidence_indices_uncertain": llm_result.get("evidence_indices_uncertain", []),
                "used_full_context_as_candidates": llm_result.get("used_full_context_as_candidates", False),
                "risk_of_bias": llm_result.get("risk_of_bias"),
                "error": llm_result.get("error"),
            },
            # 标准答案
            "ground_truth": {
                "label": ground_truth.get("label", ""),
                "answers": ground_truth.get("answers", []),
            },
            # 原始LLM结果（保留完整信息，包括raw_response等）
            "llm_result_raw": llm_result,
            # 模型信息
            "model_info": {
                "model_name": model_name,
            }
        }

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_result, f, ensure_ascii=False, indent=2)

        with stats_lock:
            stats["processed"] += 1
        logger.success(f"  [{idx}] ✓ 完成: 已保存到 {output_file.name}")
        return output_file.name

    except Exception as e:
        logger.error(f"  [{idx}] ✗ 处理失败: {e}")
        logger.exception(e)
        with stats_lock:
            stats["failed"] += 1
        return None


def generate_ssr_evidence_result(
    json_data_path: Path,
    output_dir: Path,
    model_name: str = "openai/gpt-5.1",
    limit: Optional[int] = None,
    force: bool = False,
    max_workers: int = 4,
    mode: str = "default"
) -> Dict[str, Any]:
    """
    生成 SSR 证据提取结果

    Args:
        json_data_path: ssr_dataset_merged.json 文件路径
        output_dir: 输出目录路径
        model_name: 使用的模型名称
        limit: 限制处理的记录数量（用于测试），None 表示处理所有记录
        force: 是否强制重新处理已存在的文件
        max_workers: 最大线程数（默认：4）
        mode: 生成模式（default, cot, agent, direct）

    Returns:
        包含处理统计信息的字典
    """
    # 根据模型名称和模式创建子目录
    model_dirname = model_name_to_dirname(model_name)
    mode_suffix = get_mode_suffix(mode)
    model_output_dir = output_dir / f"{model_dirname}{mode_suffix}"

    # 创建输出目录
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描已完成的记录ID
    logger.info(f"正在扫描已完成的记录: {model_output_dir}")
    completed_ids = get_completed_ids(model_output_dir)
    logger.info(f"已找到 {len(completed_ids)} 条已完成的记录")

    # 读取原始数据
    logger.info(f"正在读取原始数据: {json_data_path}")
    with open(json_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    total_records = len(original_data)
    if limit:
        original_data = original_data[:limit]
        logger.info(f"限制处理数量: {limit}/{total_records}")

    # 统计需要处理的记录
    total_to_process = len(original_data)
    already_completed_count = sum(1 for record in original_data if record.get("id", "") in completed_ids)
    remaining_count = total_to_process - already_completed_count

    logger.info(f"共 {total_to_process} 条记录")
    logger.info(f"  - 已完成: {already_completed_count} 条")
    logger.info(f"  - 待处理: {remaining_count} 条")

    # 如果所有记录都已完成且不是强制模式，直接返回
    if remaining_count == 0 and not force:
        logger.info("=" * 80)
        logger.info("所有记录都已完成，无需处理")
        logger.info("=" * 80)
        return {
            "total": total_to_process,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "already_exists": already_completed_count,
        }

    if force:
        logger.info("强制模式: 将重新处理所有记录（包括已完成的）")
    logger.info(f"使用 {max_workers} 个线程进行并行处理")

    # 统计信息（使用锁保护）
    stats = {
        "total": len(original_data),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "already_exists": 0,
    }
    stats_lock = Lock()

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_record = {
            executor.submit(
                process_single_record,
                record,
                idx + 1,
                len(original_data),
                model_output_dir,
                model_name,
                force,
                completed_ids,
                stats_lock,
                stats,
                mode
            ): (idx + 1, record)
            for idx, record in enumerate(original_data)
        }

        # 等待所有任务完成
        completed = 0
        for future in as_completed(future_to_record):
            completed += 1
            idx, record = future_to_record[future]
            try:
                result = future.result()
                if result:
                    logger.debug(f"[{idx}] 任务完成: {result}")
            except Exception as e:
                logger.error(f"[{idx}] 任务异常: {e}")

            # 定期输出进度
            if completed % 10 == 0 or completed == len(original_data):
                with stats_lock:
                    logger.info(
                        f"进度: {completed}/{len(original_data)} "
                        f"(已处理: {stats['processed']}, "
                        f"已存在: {stats['already_exists']}, "
                        f"跳过: {stats['skipped']}, "
                        f"失败: {stats['failed']})"
                    )

    return stats


def print_summary(stats: Dict[str, Any]):
    """
    打印处理结果摘要

    Args:
        stats: 处理统计信息
    """
    logger.info("=" * 80)
    logger.info("处理结果摘要")
    logger.info("=" * 80)

    logger.info(f"总记录数: {stats['total']}")
    logger.info(f"成功处理: {stats['processed']}")
    logger.info(f"已存在（跳过）: {stats['already_exists']}")
    logger.info(f"跳过: {stats['skipped']}")
    logger.info(f"失败: {stats['failed']}")
    logger.info("=" * 80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="生成 SSR 证据提取结果"
    )
    parser.add_argument(
        "--json-data",
        type=str,
        default=None,
        help="ssr_dataset_merged.json 文件路径（默认：项目根目录/data/ssr_dataset_merged.json）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录路径（默认：项目根目录/data/ssr_evidence_results / ssr_evidence_results_direct / ssr_evidence_results_origin，取决于模式）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt-5.1",
        help="使用的模型名称（默认：openai/gpt-5.1）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的记录数量（用于测试）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理已存在的文件"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="最大线程数（默认：4）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "cot", "agent", "cot_agent", "tool", "direct", "origin"],
        help="生成模式（默认：default）。default=原版模式，cot=CoT模式，agent=Agent增强模式，cot_agent=CoT+Agent（先Agent筛选句子再CoT推理），direct=直接模式，origin=全量不检索模式，tool模式暂未实现"
    )

    args = parser.parse_args()

    # 获取脚本所在目录和项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 设置默认路径
    if args.json_data is None:
        json_data_path = project_root / "data" / "ssr_dataset_merged.json"
    else:
        json_data_path = Path(args.json_data)

    if args.output_dir is None:
        # 根据模式决定默认输出目录
        if args.mode == "direct":
            output_dir = project_root / "data" / "ssr_evidence_results_direct"
        elif args.mode == "origin":
            output_dir = project_root / "data" / "ssr_evidence_results_origin"
        else:
            output_dir = project_root / "data" / "ssr_evidence_results"
    else:
        output_dir = Path(args.output_dir)

    # 检查路径是否存在
    if not json_data_path.exists():
        logger.error(f"原始数据文件不存在: {json_data_path}")
        return 1

    # 计算模型输出目录
    model_dirname = model_name_to_dirname(args.model_name)
    mode_suffix = get_mode_suffix(args.mode)
    model_output_dir = output_dir / f"{model_dirname}{mode_suffix}"

    logger.info("=" * 80)
    logger.info("生成 SSR 证据提取结果")
    logger.info("=" * 80)
    logger.info(f"原始数据文件: {json_data_path}")
    logger.info(f"输出根目录: {output_dir}")
    logger.info(f"模型输出目录: {model_output_dir}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"模式: {args.mode}")
    if args.limit:
        logger.info(f"限制处理数量: {args.limit}")
    if args.force:
        logger.info("强制模式: 将重新处理已存在的文件")
    logger.info(f"最大线程数: {args.max_workers}")
    logger.info("=" * 80)

    # 执行生成
    try:
        stats = generate_ssr_evidence_result(
            json_data_path=json_data_path,
            output_dir=output_dir,
            model_name=args.model_name,
            limit=args.limit,
            force=args.force,
            max_workers=args.max_workers,
            mode=args.mode
        )

        # 打印摘要
        print_summary(stats)

        return 0

    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.exception(e)
        return 1


def test(n: int = 10, mode: str = "default"):
    """
    测试函数：只对前n条数据进行评估
    用于快速验证程序运行是否正常

    Args:
        n: 处理的记录数量
        mode: 生成模式（default, cot, agent, direct, origin）
    """
    # 获取脚本所在目录和项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # 设置测试路径
    json_data_path = project_root / "data" / "ssr_dataset_merged.json"
    # 根据模式决定默认输出目录
    if mode == "direct":
        output_dir = project_root / "data" / "ssr_evidence_results_direct"
    elif mode == "origin":
        output_dir = project_root / "data" / "ssr_evidence_results_origin"
    else:
        output_dir = project_root / "data" / "ssr_evidence_results"
    model_name = "openai/gpt-5.1"

    # 检查路径是否存在
    if not json_data_path.exists():
        logger.error(f"原始数据文件不存在: {json_data_path}")
        return 1

    # 计算模型输出目录（考虑模式后缀）
    model_dirname = model_name_to_dirname(model_name)
    mode_suffix = get_mode_suffix(mode)
    model_output_dir = output_dir / f"{model_dirname}{mode_suffix}"

    logger.info("=" * 80)
    logger.info(f"测试模式：生成 SSR 证据提取结果（仅前{n}条数据）")
    logger.info("=" * 80)
    logger.info(f"原始数据文件: {json_data_path}")
    logger.info(f"输出根目录: {output_dir}")
    logger.info(f"模型输出目录: {model_output_dir}")
    logger.info(f"模型名称: {model_name}")
    logger.info(f"模式: {mode}")
    logger.info(f"限制处理数量: {n}")
    logger.info(f"强制模式: 否（跳过已存在的文件）")
    logger.info(f"最大线程数: 4")
    logger.info("=" * 80)

    # 执行生成（只处理前n条）
    try:
        stats = generate_ssr_evidence_result(
            json_data_path=json_data_path,
            output_dir=output_dir,
            model_name=model_name,
            limit=n,
            force=False,  # 测试时不强制重新处理
            max_workers=4,  # 测试时使用较少的线程数
            mode=mode
        )

        # 打印摘要
        print_summary(stats)

        logger.info("=" * 80)
        logger.info("测试完成！")
        logger.info("=" * 80)

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

    # 如果直接运行脚本，可以使用test()函数进行快速测试
    # 取消下面的注释来运行测试函数
    # sys.exit(test(n=40))

    # 否则运行主函数（支持命令行参数）
    sys.exit(main())

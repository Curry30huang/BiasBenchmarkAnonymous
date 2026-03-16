from loguru import logger
from pathlib import Path
from utils.openai_llm import OpenAICompatibleClient
from typing import Dict, Any, List, Optional
from bias_assessment.rct_bias_assessment import (
    _load_model_config,
    _create_llm_client,
    _load_prompt_template,
    _parse_json_response,
    _clean_control_characters,
    _fix_json_unescaped_backslashes
)
# 延迟导入以避免循环导入
# from bias_assessment.evidence_ssr_agent import extract_ssr_evidence_agent

# 加载模型配置文件
MODEL_CONFIG_PATH = Path(__file__).parent.parent / "model_config.json"

# Prompt模板文件路径
SSR_EVIDENCE_EXTRACTION_FILE_PATH = Path(__file__).parent / "prompts" / "ssr_evidence_extraction.txt"
SSR_EVIDENCE_EXTRACTION_COT_FILE_PATH = Path(__file__).parent / "prompts" / "ssr_evidence_extraction_cot.txt"
SSR_EVIDENCE_EXTRACTION_DIRECT_FILE_PATH = Path(__file__).parent / "prompts" / "ssr_evidence_extraction_direct.txt"
SSR_EVIDENCE_EXTRACTION_ORIGIN_FILE_PATH = Path(__file__).parent / "prompts" / "ssr_evidence_extraction_origin.txt"


def _get_prompt_file_path(mode: str = "default") -> Path:
    """
    根据模式获取prompt文件路径

    Args:
        mode: 模式，可选值：default, cot, agent, tool, direct, origin（目前支持default、cot、agent、direct、origin）

    Returns:
        prompt文件路径
    """
    if mode == "cot":
        return SSR_EVIDENCE_EXTRACTION_COT_FILE_PATH
    elif mode == "direct":
        return SSR_EVIDENCE_EXTRACTION_DIRECT_FILE_PATH
    elif mode == "origin":
        return SSR_EVIDENCE_EXTRACTION_ORIGIN_FILE_PATH
    elif mode == "agent":
        # agent模式使用特殊的处理逻辑，不直接使用prompt文件
        return SSR_EVIDENCE_EXTRACTION_FILE_PATH
    elif mode == "cot_agent":
        # cot_agent 模式：Agent 筛选 + CoT 推理，逻辑在 evidence_ssr_cot_agent 中，此处仅占位
        return SSR_EVIDENCE_EXTRACTION_FILE_PATH
    elif mode == "default" or mode is None or mode == "":
        return SSR_EVIDENCE_EXTRACTION_FILE_PATH
    else:
        # tool模式暂未实现，使用默认
        logger.warning(f"模式 '{mode}' 暂未实现，使用默认模式")
        return SSR_EVIDENCE_EXTRACTION_FILE_PATH

def _format_context_list(context_list: List[str], with_index: bool = True) -> str:
    """
    格式化上下文句子列表为字符串，每个句子前加上索引

    Args:
        context_list: 上下文句子列表
        with_index: 是否添加索引（direct模式不需要索引）

    Returns:
        格式化后的字符串
    """
    formatted_lines = []
    for idx, sentence in enumerate(context_list):
        if with_index:
            formatted_lines.append(f"[{idx}] {sentence}")
        else:
            formatted_lines.append(sentence)
    return "\n".join(formatted_lines)


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

    risk_str_lower = risk_str.strip().lower()

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


def _validate_evidence_indices(indices: Any, context_list_length: int) -> List[int]:
    """
    验证并规范化证据索引

    Args:
        indices: 证据索引（可能是列表、整数或其他类型）
        context_list_length: 上下文列表的长度

    Returns:
        规范化后的索引列表
    """
    if indices is None:
        return []

    # 如果是单个整数，转换为列表
    if isinstance(indices, int):
        indices = [indices]

    # 如果不是列表，尝试转换
    if not isinstance(indices, list):
        try:
            indices = list(indices)
        except (TypeError, ValueError):
            logger.warning(f"无法将证据索引转换为列表: {indices}")
            return []

    # 验证每个索引
    valid_indices = []
    for idx in indices:
        try:
            idx_int = int(idx)
            if 0 <= idx_int < context_list_length:
                valid_indices.append(idx_int)
            else:
                logger.warning(f"证据索引 {idx_int} 超出范围 [0, {context_list_length - 1}]，跳过")
        except (ValueError, TypeError):
            logger.warning(f"无效的证据索引: {idx}，跳过")

    # 去重并排序
    valid_indices = sorted(list(set(valid_indices)))
    return valid_indices


def extract_ssr_evidence(
    bias: str,
    context_list: List[str],
    question: str,
    model_name: str = "openai/gpt-5.1",
    client: Optional[OpenAICompatibleClient] = None,
    max_retries: int = 3,
    mode: str = "default"
) -> Dict[str, Any]:
    """
    提取SSR证据：判断能够准确回答question的证据支撑句子的下标索引，并判断偏倚风险

    Args:
        bias: 偏倚类型（如 "blinding of outcome assessment (detection bias) all outcomes"）
        context_list: 上下文句子列表
        question: 问题描述（包含low risk、high risk、unclear risk的说明）
        model_name: 使用的模型名称，默认为 "openai/gpt-5.1"
        client: 可选的LLM客户端，如果提供则复用，否则创建新客户端
        max_retries: 最大重试次数，默认为3
        mode: 模式，可选值：default, cot, agent, direct, origin

    Returns:
        包含以下字段的字典：
        - success: 是否成功
        - evidence_indices: 支撑句子索引列表（整数列表）
        - risk_of_bias: 偏倚风险等级（"low", "some concerns", "high"）
        - raw_response: 原始LLM响应
        - parsed_response: 解析后的JSON响应（如果成功）
        - error: 错误信息（如果失败）
        - agent_metadata: Agent处理过程的元数据（仅agent模式）
    """
    # 如果是 agent 模式，调用 agent 版本的函数
    if mode == "agent":
        logger.info("使用Agent增强模式进行SSR证据提取")
        from bias_assessment.evidence_ssr_agent import extract_ssr_evidence_agent
        return extract_ssr_evidence_agent(
            bias=bias,
            context_list=context_list,
            question=question,
            model_name=model_name,
            client=client,
            max_retries=max_retries
        )
    # CoT + Agent 模式：先 Agent 筛选，再对候选句子做 CoT 推理
    if mode == "cot_agent":
        logger.info("使用 CoT+Agent 模式进行SSR证据提取")
        from bias_assessment.evidence_ssr_cot_agent import extract_ssr_evidence_cot_agent
        return extract_ssr_evidence_cot_agent(
            bias=bias,
            context_list=context_list,
            question=question,
            model_name=model_name,
            client=client,
            max_retries=max_retries
        )

    # 验证输入
    if not context_list:
        logger.error("context_list 不能为空")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "error": "context_list 不能为空"
        }

    if not question or not question.strip():
        logger.error("question 不能为空")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "error": "question 不能为空"
        }

    # 根据模式获取prompt文件路径
    prompt_file_path = _get_prompt_file_path(mode)

    # 加载prompt模板
    try:
        prompt_template = _load_prompt_template(prompt_file_path)
    except Exception as e:
        logger.error(f"加载SSR证据提取prompt模板失败: {e}")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "error": f"加载prompt模板失败: {e}"
        }

    # 格式化上下文列表
    # - direct: 不需要索引（句子已预先筛选为正确证据）
    # - origin: 使用全量context_list，但不做检索（仍保留索引，方便阅读与对齐默认格式）
    is_direct_mode = (mode == "direct")
    formatted_context = _format_context_list(context_list, with_index=not is_direct_mode)

    # 构建prompt，替换占位符
    prompt = prompt_template.replace("{bias}", bias if bias else "Not specified")
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{context_list}", formatted_context)

    # 如果没有提供客户端，创建新客户端（使用上下文管理器）
    if client is None:
        client = _create_llm_client(model_name)
        use_context_manager = True
    else:
        use_context_manager = False

    # 重试机制
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"开始调用LLM进行SSR证据提取（尝试 {attempt}/{max_retries}）...")

            # 如果使用上下文管理器，确保连接池正确释放
            if use_context_manager:
                with client:
                    # 调用LLM
                    response = client.simple_chat(
                        message=prompt,
                        temperature=0,
                    )

                    # 解析JSON响应
                    parsed_response = _parse_json_response(response, "ssr_evidence_extraction")

                    # 处理解析结果
                    return _process_ssr_response(
                        response,
                        parsed_response,
                        context_list,
                        attempt,
                        max_retries,
                        use_context_manager,
                        mode
                    )
            else:
                # 复用提供的客户端
                # 调用LLM
                response = client.simple_chat(
                    message=prompt,
                    temperature=0,
                )

                # 解析JSON响应
                parsed_response = _parse_json_response(response, "ssr_evidence_extraction")

                # 处理解析结果
                result = _process_ssr_response(
                    response,
                    parsed_response,
                    context_list,
                    attempt,
                    max_retries,
                    use_context_manager,
                    mode
                )

                # 如果成功，直接返回
                if result.get("success"):
                    return result

                # 如果达到最大重试次数，返回最后一次的结果
                if attempt == max_retries:
                    return result

                # 否则继续重试
                logger.info(f"SSR证据提取失败，将在 0.5 秒后重试...")
                import time
                time.sleep(0.5)
                continue

        except Exception as e:
            logger.error(f"SSR证据提取调用LLM时发生错误（尝试 {attempt}/{max_retries}）: {e}")
            if attempt < max_retries:
                logger.info(f"将在 0.5 秒后重试...")
                import time
                time.sleep(0.5)  # 等待0.5秒后重试
                continue
            else:
                logger.error("SSR证据提取所有重试都失败")
                return {
                    "success": False,
                    "evidence_indices": [],
                    "risk_of_bias": None,
                    "raw_response": None,
                    "parsed_response": None,
                    "error": f"调用LLM失败: {e}"
                }

    return {
        "success": False,
        "evidence_indices": [],
        "risk_of_bias": None,
        "raw_response": None,
        "parsed_response": None,
        "error": "所有重试都失败"
    }


def _process_ssr_response(
    response: str,
    parsed_response: Optional[Any],
    context_list: List[str],
    attempt: int,
    max_retries: int,
    use_context_manager: bool,
    mode: str = "default"
) -> Dict[str, Any]:
    """
    处理SSR响应，提取证据索引和风险等级

    Args:
        response: 原始LLM响应
        parsed_response: 解析后的JSON响应
        context_list: 上下文句子列表（用于验证索引）
        attempt: 当前尝试次数
        max_retries: 最大重试次数
        use_context_manager: 是否使用上下文管理器
        mode: 模式，可选值：default, cot, agent, direct, origin

    Returns:
        处理后的结果字典
    """
    if parsed_response is None:
        logger.warning(f"SSR证据提取响应解析失败（尝试 {attempt}/{max_retries}）")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": response,
            "parsed_response": None,
            "error": "响应解析失败"
        }

    # 处理数组格式的响应（如果LLM返回的是数组，取第一个元素）
    if isinstance(parsed_response, list):
        if len(parsed_response) > 0:
            parsed_response = parsed_response[0]
            logger.info("SSR证据提取响应是数组格式，提取第一个元素")
        else:
            logger.warning(f"SSR证据提取响应是空数组（尝试 {attempt}/{max_retries}）")
            return {
                "success": False,
                "evidence_indices": [],
                "risk_of_bias": None,
                "raw_response": response,
                "parsed_response": parsed_response,
                "error": "响应是空数组"
            }

    # 确保response是字典格式
    if not isinstance(parsed_response, dict):
        logger.warning(f"SSR证据提取响应不是字典格式（类型: {type(parsed_response)}）（尝试 {attempt}/{max_retries}）")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": response,
            "parsed_response": parsed_response,
            "error": f"响应不是字典格式（类型: {type(parsed_response)}）"
        }

    # 提取evidence_indices
    # - direct/origin 模式不需要（direct=只提供正确句子；origin=提供全量句子但不做检索）
    is_no_retrieval_mode = mode in ("direct", "origin")
    if is_no_retrieval_mode:
        # no-retrieval 模式下，不需要提取evidence_indices
        evidence_indices = []
        logger.info(f"{mode.capitalize()}模式：跳过证据索引提取，只判断风险等级")
    else:
        evidence_indices_raw = parsed_response.get("evidence_indices", [])
        evidence_indices = _validate_evidence_indices(evidence_indices_raw, len(context_list))
        logger.info(f"提取到 {len(evidence_indices)} 个有效证据索引: {evidence_indices}")

    # 提取risk_of_bias
    risk_of_bias_raw = parsed_response.get("risk_of_bias", "")
    risk_of_bias = _normalize_risk_level(risk_of_bias_raw)

    if risk_of_bias is None:
        logger.warning(f"无法识别风险等级: '{risk_of_bias_raw}'（尝试 {attempt}/{max_retries}）")
        return {
            "success": False,
            "evidence_indices": evidence_indices,
            "risk_of_bias": None,
            "raw_response": response,
            "parsed_response": parsed_response,
            "error": f"无法识别风险等级: '{risk_of_bias_raw}'"
        }

    logger.info(f"SSR证据提取成功，风险等级: {risk_of_bias}")

    return {
        "success": True,
        "evidence_indices": evidence_indices,
        "risk_of_bias": risk_of_bias,
        "raw_response": response,
        "parsed_response": parsed_response
    }

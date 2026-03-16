"""
CoT + Agent 模式：证据检索阶段用 Agent 筛选句子，筛选后用 CoT 提示词做规范推理与思考。

流程：
1. 调用 evidence_ssr_agent 的 Agent 逻辑（置信度分级 + DVR 反思）得到候选证据索引。
2. 用候选句子组成短列表，使用 CoT 提示词调用强模型进行推理，得到 evidence_indices（对候选列表的 0-based）+ risk_of_bias + reasoning。
3. 将 CoT 的 evidence_indices 映射回原始 context_list 的索引，并返回与 extract_ssr_evidence 一致的字典。
"""
from loguru import logger
from pathlib import Path
from typing import Dict, Any, List, Optional

from bias_assessment.rct_bias_assessment import (
    _load_model_config,
    _create_llm_client,
    _load_prompt_template,
    _parse_json_response,
)
# 不从 evidence_ssr 顶层导入，在函数内按需导入，避免 evidence_ssr <-> evidence_ssr_cot_agent 循环引用

# CoT+Agent 第二阶段使用的 CoT 提示词
SSR_COT_AGENT_PROMPT_PATH = Path(__file__).parent / "prompts" / "ssr_evidence_extraction_cot_agent.txt"


def extract_ssr_evidence_cot_agent(
    bias: str,
    context_list: List[str],
    question: str,
    model_name: str = "openai/gpt-5.1",
    client: Optional[Any] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    CoT + Agent 模式：先 Agent 筛选句子，再对候选句子做 CoT 推理。

    Args:
        bias: 偏倚类型
        context_list: 上下文句子列表（原始）
        question: 问题描述
        model_name: 强模型名称（Agent 第二阶段 + CoT 阶段均使用）
        client: 可选的 LLM 客户端
        max_retries: 最大重试次数

    Returns:
        与 extract_ssr_evidence 一致的字典，包含：
        - success
        - evidence_indices: 正常确定的证据句子索引（CoT 选中的）
        - evidence_indices_uncertain: 仅当无确定证据时使用的 Uncertain 候选索引，与 evidence_indices 互斥区分
        - used_full_context_as_candidates: 是否因 Agent 无候选而使用了全量 context 作为候选（供 check 与下游识别）
        - risk_of_bias, raw_response, parsed_response
        - agent_metadata（Agent 阶段元数据）, cot_reasoning, error
    """
    # 延迟导入，避免与 evidence_ssr 的循环引用（无论谁先被 import 都安全）
    from bias_assessment.evidence_ssr import (
        _format_context_list,
        _normalize_risk_level,
        _validate_evidence_indices,
    )

    if not context_list or not question or not question.strip():
        err = "context_list 或 question 为空"
        logger.error(err)
        return {
            "success": False,
            "evidence_indices": [],
            "evidence_indices_uncertain": [],
            "used_full_context_as_candidates": False,
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "agent_metadata": None,
            "cot_reasoning": None,
            "error": err,
        }

    # Step 1: Agent 筛选句子
    from bias_assessment.evidence_ssr_agent import extract_ssr_evidence_agent

    logger.info("CoT+Agent 模式：阶段 1 使用 Agent 筛选证据句子...")
    agent_result = extract_ssr_evidence_agent(
        bias=bias,
        context_list=context_list,
        question=question,
        model_name=model_name,
        client=client,
        max_retries=max_retries,
        sentences_only=True,  # 只做阶段1+2 检索句子，不调第三阶段 LLM，由后续 CoT 生成最终答案
    )
    agent_indices = agent_result.get("evidence_indices") or []
    agent_metadata = agent_result.get("agent_metadata") or {}
    used_full_context_as_candidates = False

    # 若 Agent 未返回任何候选，则用全量句子作为候选（退化为“全量 + CoT”），并标记供 check 与下游识别
    if not agent_indices:
        logger.warning("Agent 未返回任何候选句子，CoT 阶段使用全量 context_list 作为候选")
        agent_indices = list(range(len(context_list)))
        used_full_context_as_candidates = True

    # 去重排序，构建候选列表及 候选下标 -> 原始下标 的映射
    agent_indices = sorted(set(agent_indices))
    filtered_sentences = [context_list[i] for i in agent_indices if 0 <= i < len(context_list)]
    # 映射: 候选列表中的位置 j -> 原始 context 中的索引 agent_indices[j]
    index_map = [agent_indices[j] for j in range(len(filtered_sentences))]

    if not filtered_sentences:
        logger.error("筛选后候选句子为空，无法进行 CoT 阶段")
        return {
            "success": False,
            "evidence_indices": [],
            "evidence_indices_uncertain": [],
            "used_full_context_as_candidates": used_full_context_as_candidates,
            "risk_of_bias": agent_result.get("risk_of_bias"),
            "raw_response": agent_result.get("raw_response"),
            "parsed_response": agent_result.get("parsed_response"),
            "agent_metadata": agent_metadata,
            "cot_reasoning": None,
            "error": "筛选后候选句子为空",
        }

    # Step 2: 加载 CoT 提示词并调用强模型
    try:
        prompt_template = _load_prompt_template(SSR_COT_AGENT_PROMPT_PATH)
    except Exception as e:
        logger.error(f"加载 CoT+Agent 提示词失败: {e}")
        return {
            "success": False,
            "evidence_indices": [],
            "evidence_indices_uncertain": [],
            "used_full_context_as_candidates": used_full_context_as_candidates,
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "agent_metadata": agent_metadata,
            "cot_reasoning": None,
            "error": f"加载提示词失败: {e}",
        }

    formatted_candidates = _format_context_list(filtered_sentences, with_index=True)
    prompt = prompt_template.replace("{bias}", bias if bias else "Not specified")
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{context_list}", formatted_candidates)

    use_context_manager = client is None
    if client is None:
        strong_client = _create_llm_client(model_name)
    else:
        strong_client = client

    cot_response = None
    cot_parsed = None
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"CoT+Agent 模式：阶段 2 CoT 推理（尝试 {attempt}/{max_retries}）...")
            if use_context_manager:
                with strong_client:
                    cot_response = strong_client.simple_chat(message=prompt, temperature=0.1)
            else:
                cot_response = strong_client.simple_chat(message=prompt, temperature=0.1)

            cot_parsed = _parse_json_response(cot_response, "ssr_cot_agent")
            if cot_parsed is not None:
                break
            last_error = "CoT 响应解析失败"
        except Exception as e:
            last_error = str(e)
            logger.warning(f"CoT 调用异常（尝试 {attempt}/{max_retries}）: {e}")
            if attempt < max_retries:
                import time
                time.sleep(0.5)

    if cot_parsed is None:
        logger.error("CoT 阶段解析失败，使用 Agent 候选作为 uncertain 结果")
        return {
            "success": agent_result.get("success", False),
            "evidence_indices": [],
            "evidence_indices_uncertain": list(agent_indices),
            "used_full_context_as_candidates": used_full_context_as_candidates,
            "risk_of_bias": agent_result.get("risk_of_bias"),
            "raw_response": cot_response or agent_result.get("raw_response"),
            "parsed_response": cot_parsed or agent_result.get("parsed_response"),
            "agent_metadata": agent_metadata,
            "cot_reasoning": None,
            "error": last_error or "CoT 解析失败",
        }

    # 处理数组包装
    if isinstance(cot_parsed, list) and len(cot_parsed) > 0:
        cot_parsed = cot_parsed[0]
    if not isinstance(cot_parsed, dict):
        return {
            "success": False,
            "evidence_indices": [],
            "evidence_indices_uncertain": list(agent_indices),
            "used_full_context_as_candidates": used_full_context_as_candidates,
            "risk_of_bias": agent_result.get("risk_of_bias"),
            "raw_response": cot_response,
            "parsed_response": cot_parsed,
            "agent_metadata": agent_metadata,
            "cot_reasoning": None,
            "error": "CoT 返回格式不是字典",
        }

    # CoT 的 evidence_indices 是相对于候选列表的 0-based
    cot_indices_raw = cot_parsed.get("evidence_indices", [])
    n_candidates = len(filtered_sentences)
    cot_indices_valid = _validate_evidence_indices(cot_indices_raw, n_candidates)
    # 映射回原始 context 的索引
    original_indices = sorted(set(index_map[j] for j in cot_indices_valid if j < len(index_map)))

    # 若 CoT 未选出任何证据但候选来自 Agent（含 Uncertain fallback），放入 evidence_indices_uncertain，与正常证据区分
    evidence_indices_uncertain = []
    if not original_indices and agent_indices:
        logger.warning(
            "CoT 未选出证据但 Agent 提供了候选（Uncertain fallback），将候选索引写入 evidence_indices_uncertain"
        )
        evidence_indices_uncertain = list(agent_indices)

    risk_raw = cot_parsed.get("risk_of_bias", "")
    risk_of_bias = _normalize_risk_level(risk_raw)
    if risk_of_bias is None:
        risk_of_bias = agent_result.get("risk_of_bias")

    cot_reasoning = cot_parsed.get("reasoning") if isinstance(cot_parsed.get("reasoning"), str) else None

    success = risk_of_bias is not None
    return {
        "success": success,
        "evidence_indices": original_indices,
        "evidence_indices_uncertain": evidence_indices_uncertain,
        "used_full_context_as_candidates": used_full_context_as_candidates,
        "risk_of_bias": risk_of_bias,
        "raw_response": cot_response,
        "parsed_response": cot_parsed,
        "agent_metadata": agent_metadata,
        "cot_reasoning": cot_reasoning,
        "error": None if success else "risk_of_bias 无法识别",
    }

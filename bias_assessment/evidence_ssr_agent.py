from loguru import logger
from pathlib import Path
from utils.openai_llm import OpenAICompatibleClient, create_client
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from bias_assessment.rct_bias_assessment import (
    _load_model_config,
    _create_llm_client,
    _load_prompt_template,
    _parse_json_response,
)
from bias_assessment.evidence_ssr import (
    _format_context_list,
    _normalize_risk_level,
    _validate_evidence_indices,
    SSR_EVIDENCE_EXTRACTION_FILE_PATH,
)

# 加载模型配置文件
MODEL_CONFIG_PATH = Path(__file__).parent.parent / "model_config.json"

# Prompt模板文件路径
SSR_CONFIDENCE_GRADING_FILE_PATH = Path(__file__).parent / "prompts" / "ssr_confidence_grading.txt"
SSR_DVR_REFLECTION_FILE_PATH = Path(__file__).parent / "prompts" / "ssr_dvr_reflection.txt"

# 轻量级模型（第一阶段使用）
LIGHTWEIGHT_MODEL = "qwen/qwen3-32b"

# 置信度阈值
CORRECT_THRESHOLD = 0.85
INCORRECT_THRESHOLD = 0.6


def _grade_sentence_confidence(
    bias: str,
    question: str,
    sentence_index: int,
    sentence: str,
    lightweight_client: OpenAICompatibleClient,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    第一阶段：对单个句子进行置信度分级（使用轻量级模型）

    Args:
        bias: 偏倚类型
        question: 问题描述
        sentence_index: 句子索引
        sentence: 句子内容
        lightweight_client: 轻量级模型客户端
        max_retries: 最大重试次数

    Returns:
        包含以下字段的字典：
        - success: 是否成功
        - sentence_index: 句子索引
        - confidence_score: 置信度得分 (0-1)
        - confidence_grade: 置信度等级 ("Correct", "Incorrect", "Uncertain")
        - explanation: 解释说明
        - raw_response: 原始响应
        - error: 错误信息（如果失败）
    """
    # 加载prompt模板
    try:
        prompt_template = _load_prompt_template(SSR_CONFIDENCE_GRADING_FILE_PATH)
    except Exception as e:
        logger.error(f"加载置信度分级prompt模板失败: {e}")
        return {
            "success": False,
            "sentence_index": sentence_index,
            "confidence_score": None,
            "confidence_grade": None,
            "explanation": None,
            "raw_response": None,
            "error": f"加载prompt模板失败: {e}"
        }

    # 构建prompt
    prompt = prompt_template.replace("{bias}", bias if bias else "Not specified")
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{sentence_index}", str(sentence_index))
    prompt = prompt.replace("{sentence}", sentence)

    # 重试机制
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"对句子 {sentence_index} 进行置信度分级（尝试 {attempt}/{max_retries}）...")

            # 调用轻量级模型
            response = lightweight_client.simple_chat(
                message=prompt,
                temperature=0,
            )

            # 解析JSON响应
            parsed_response = _parse_json_response(response, "ssr_confidence_grading")

            if parsed_response is None:
                logger.warning(f"句子 {sentence_index} 置信度分级响应解析失败（尝试 {attempt}/{max_retries}）")
                if attempt < max_retries:
                    continue
                return {
                    "success": False,
                    "sentence_index": sentence_index,
                    "confidence_score": None,
                    "confidence_grade": None,
                    "explanation": None,
                    "raw_response": response,
                    "error": "响应解析失败"
                }

            # 处理数组格式
            if isinstance(parsed_response, list):
                if len(parsed_response) > 0:
                    parsed_response = parsed_response[0]
                else:
                    logger.warning(f"句子 {sentence_index} 置信度分级响应是空数组")
                    if attempt < max_retries:
                        continue
                    return {
                        "success": False,
                        "sentence_index": sentence_index,
                        "confidence_score": None,
                        "confidence_grade": None,
                        "explanation": None,
                        "raw_response": response,
                        "error": "响应是空数组"
                    }

            # 验证响应格式
            if not isinstance(parsed_response, dict):
                logger.warning(f"句子 {sentence_index} 置信度分级响应不是字典格式")
                if attempt < max_retries:
                    continue
                return {
                    "success": False,
                    "sentence_index": sentence_index,
                    "confidence_score": None,
                    "confidence_grade": None,
                    "explanation": None,
                    "raw_response": response,
                    "error": f"响应不是字典格式（类型: {type(parsed_response)}）"
                }

            # 提取字段
            confidence_score = parsed_response.get("confidence_score")
            confidence_grade = parsed_response.get("confidence_grade", "").strip()
            explanation = parsed_response.get("explanation", "")

            # 验证置信度得分
            if confidence_score is None:
                logger.warning(f"句子 {sentence_index} 置信度得分缺失")
                if attempt < max_retries:
                    continue
                return {
                    "success": False,
                    "sentence_index": sentence_index,
                    "confidence_score": None,
                    "confidence_grade": None,
                    "explanation": explanation,
                    "raw_response": response,
                    "error": "置信度得分缺失"
                }

            # 验证置信度等级
            if confidence_grade not in ["Correct", "Incorrect", "Uncertain"]:
                logger.warning(f"句子 {sentence_index} 置信度等级无效: '{confidence_grade}'")
                # 根据得分自动推断等级
                try:
                    score = float(confidence_score)
                    if score > CORRECT_THRESHOLD:
                        confidence_grade = "Correct"
                    elif score < INCORRECT_THRESHOLD:
                        confidence_grade = "Incorrect"
                    else:
                        confidence_grade = "Uncertain"
                    logger.info(f"根据得分 {score} 自动推断置信度等级为: {confidence_grade}")
                except (ValueError, TypeError):
                    if attempt < max_retries:
                        continue
                    return {
                        "success": False,
                        "sentence_index": sentence_index,
                        "confidence_score": confidence_score,
                        "confidence_grade": None,
                        "explanation": explanation,
                        "raw_response": response,
                        "error": f"置信度等级无效且无法推断: '{confidence_grade}'"
                    }

            logger.info(f"句子 {sentence_index} 置信度分级成功: {confidence_grade} (得分: {confidence_score})")

            return {
                "success": True,
                "sentence_index": sentence_index,
                "confidence_score": float(confidence_score),
                "confidence_grade": confidence_grade,
                "explanation": explanation,
                "raw_response": response
            }

        except Exception as e:
            logger.error(f"句子 {sentence_index} 置信度分级调用LLM时发生错误（尝试 {attempt}/{max_retries}）: {e}")
            if attempt < max_retries:
                import time
                time.sleep(0.5)
                continue
            else:
                return {
                    "success": False,
                    "sentence_index": sentence_index,
                    "confidence_score": None,
                    "confidence_grade": None,
                    "explanation": None,
                    "raw_response": None,
                    "error": f"调用LLM失败: {e}"
                }

    return {
        "success": False,
        "sentence_index": sentence_index,
        "confidence_score": None,
        "confidence_grade": None,
        "explanation": None,
        "raw_response": None,
        "error": "所有重试都失败"
    }


def _reflect_uncertain_sentence(
    bias: str,
    question: str,
    sentence_index: int,
    sentence: str,
    strong_model_client: OpenAICompatibleClient,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    第二阶段：对不确定句子进行DVR反思（使用强推理模型）

    Args:
        bias: 偏倚类型
        question: 问题描述
        sentence_index: 句子索引
        sentence: 句子内容
        strong_model_client: 强推理模型客户端
        max_retries: 最大重试次数

    Returns:
        包含以下字段的字典：
        - success: 是否成功
        - sentence_index: 句子索引
        - final_confidence_grade: 最终置信度等级
        - confidence_score: 置信度得分
        - divide: 分解的约束列表
        - verify: 验证矩阵
        - refine: 精炼结果
        - raw_response: 原始响应
        - error: 错误信息（如果失败）
    """
    # 加载prompt模板
    try:
        prompt_template = _load_prompt_template(SSR_DVR_REFLECTION_FILE_PATH)
    except Exception as e:
        logger.error(f"加载DVR反思prompt模板失败: {e}")
        return {
            "success": False,
            "sentence_index": sentence_index,
            "final_confidence_grade": None,
            "confidence_score": None,
            "divide": None,
            "verify": None,
            "refine": None,
            "raw_response": None,
            "error": f"加载prompt模板失败: {e}"
        }

    # 构建prompt
    prompt = prompt_template.replace("{bias}", bias if bias else "Not specified")
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{sentence_index}", str(sentence_index))
    prompt = prompt.replace("{sentence}", sentence)

    # 重试机制
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"对句子 {sentence_index} 进行DVR反思（尝试 {attempt}/{max_retries}）...")

            # 调用强推理模型
            response = strong_model_client.simple_chat(
                message=prompt,
                temperature=0,
            )

            # 解析JSON响应
            parsed_response = _parse_json_response(response, "ssr_dvr_reflection")

            if parsed_response is None:
                logger.warning(f"句子 {sentence_index} DVR反思响应解析失败（尝试 {attempt}/{max_retries}）")
                if attempt < max_retries:
                    continue
                return {
                    "success": False,
                    "sentence_index": sentence_index,
                    "final_confidence_grade": None,
                    "confidence_score": None,
                    "divide": None,
                    "verify": None,
                    "refine": None,
                    "raw_response": response,
                    "error": "响应解析失败"
                }

            # 处理数组格式
            if isinstance(parsed_response, list):
                if len(parsed_response) > 0:
                    parsed_response = parsed_response[0]
                else:
                    logger.warning(f"句子 {sentence_index} DVR反思响应是空数组")
                    if attempt < max_retries:
                        continue
                    return {
                        "success": False,
                        "sentence_index": sentence_index,
                        "final_confidence_grade": None,
                        "confidence_score": None,
                        "divide": None,
                        "verify": None,
                        "refine": None,
                        "raw_response": response,
                        "error": "响应是空数组"
                    }

            # 验证响应格式
            if not isinstance(parsed_response, dict):
                logger.warning(f"句子 {sentence_index} DVR反思响应不是字典格式")
                if attempt < max_retries:
                    continue
                return {
                    "success": False,
                    "sentence_index": sentence_index,
                    "final_confidence_grade": None,
                    "confidence_score": None,
                    "divide": None,
                    "verify": None,
                    "refine": None,
                    "raw_response": response,
                    "error": f"响应不是字典格式（类型: {type(parsed_response)}）"
                }

            # 提取refine字段
            refine = parsed_response.get("refine", {})
            if not isinstance(refine, dict):
                refine = {}

            final_confidence_grade = refine.get("final_confidence_grade", "").strip()
            confidence_score = refine.get("confidence_score")

            # 验证最终置信度等级
            if final_confidence_grade not in ["Correct", "Incorrect", "Uncertain"]:
                logger.warning(f"句子 {sentence_index} DVR反思最终置信度等级无效: '{final_confidence_grade}'")
                # 如果置信度得分存在，根据得分推断
                if confidence_score is not None:
                    try:
                        score = float(confidence_score)
                        if score > CORRECT_THRESHOLD:
                            final_confidence_grade = "Correct"
                        elif score < INCORRECT_THRESHOLD:
                            final_confidence_grade = "Incorrect"
                        else:
                            final_confidence_grade = "Uncertain"
                        logger.info(f"根据得分 {score} 自动推断最终置信度等级为: {final_confidence_grade}")
                    except (ValueError, TypeError):
                        if attempt < max_retries:
                            continue
                        return {
                            "success": False,
                            "sentence_index": sentence_index,
                            "final_confidence_grade": None,
                            "confidence_score": confidence_score,
                            "divide": parsed_response.get("divide"),
                            "verify": parsed_response.get("verify"),
                            "refine": refine,
                            "raw_response": response,
                            "error": f"最终置信度等级无效且无法推断: '{final_confidence_grade}'"
                        }
                else:
                    if attempt < max_retries:
                        continue
                    return {
                        "success": False,
                        "sentence_index": sentence_index,
                        "final_confidence_grade": None,
                        "confidence_score": None,
                        "divide": parsed_response.get("divide"),
                        "verify": parsed_response.get("verify"),
                        "refine": refine,
                        "raw_response": response,
                        "error": f"最终置信度等级无效: '{final_confidence_grade}'"
                    }

            logger.info(f"句子 {sentence_index} DVR反思成功: {final_confidence_grade} (得分: {confidence_score})")

            return {
                "success": True,
                "sentence_index": sentence_index,
                "final_confidence_grade": final_confidence_grade,
                "confidence_score": float(confidence_score) if confidence_score is not None else None,
                "divide": parsed_response.get("divide"),
                "verify": parsed_response.get("verify"),
                "refine": refine,
                "raw_response": response
            }

        except Exception as e:
            logger.error(f"句子 {sentence_index} DVR反思调用LLM时发生错误（尝试 {attempt}/{max_retries}）: {e}")
            if attempt < max_retries:
                import time
                time.sleep(0.5)
                continue
            else:
                return {
                    "success": False,
                    "sentence_index": sentence_index,
                    "final_confidence_grade": None,
                    "confidence_score": None,
                    "divide": None,
                    "verify": None,
                    "refine": None,
                    "raw_response": None,
                    "error": f"调用LLM失败: {e}"
                }

    return {
        "success": False,
        "sentence_index": sentence_index,
        "final_confidence_grade": None,
        "confidence_score": None,
        "divide": None,
        "verify": None,
        "refine": None,
        "raw_response": None,
        "error": "所有重试都失败"
    }


def extract_ssr_evidence_agent(
    bias: str,
    context_list: List[str],
    question: str,
    model_name: str = "openai/gpt-5.1",
    client: Optional[OpenAICompatibleClient] = None,
    max_retries: int = 3,
    sentences_only: bool = False,
) -> Dict[str, Any]:
    """
    Agent增强版本的SSR证据提取：使用CRAG三级置信度逻辑和DVR反思机制

    流程：
    1. 第一阶段：使用轻量级模型对所有句子进行置信度分级
    2. 第二阶段：对Uncertain句子使用强推理模型进行DVR反思
    3. 合并结果；若 sentences_only=False，再调用强模型生成 risk_of_bias（第三阶段）

    Args:
        bias: 偏倚类型
        context_list: 上下文句子列表
        question: 问题描述
        model_name: 强推理模型名称（第二阶段使用）
        client: 可选的强推理模型客户端
        max_retries: 最大重试次数
        sentences_only: 若为 True，只做阶段1+2 得到证据句子索引，不调用第三阶段 LLM（省 Token）

    Returns:
        包含以下字段的字典：
        - success: 是否成功
        - evidence_indices: 支撑句子索引列表
        - risk_of_bias: 偏倚风险等级（sentences_only=True 时为 None）
        - raw_response: 原始响应（包含所有阶段的响应）
        - parsed_response: 解析后的响应
        - agent_metadata: Agent处理过程的元数据
        - error: 错误信息（如果失败）
    """
    # 验证输入
    if not context_list:
        logger.error("context_list 不能为空")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "agent_metadata": None,
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
            "agent_metadata": None,
            "error": "question 不能为空"
        }

    # 计算第一阶段的最大线程数（用于配置连接池）
    stage1_max_workers = min(len(context_list), 10)

    # 创建轻量级模型客户端（第一阶段）
    # 配置连接池大小以支持并发请求：pool_connections 和 pool_maxsize 应该 >= max_workers
    # 额外增加一些缓冲，避免连接竞争
    pool_size = max(stage1_max_workers + 5, 10)  # 至少10个连接，但根据线程数动态调整
    try:
        model_config = _load_model_config()
        lightweight_config = model_config.get('models', {}).get(LIGHTWEIGHT_MODEL)
        if not lightweight_config:
            raise ValueError(f"未在 model_config.json 中找到模型配置: {LIGHTWEIGHT_MODEL}")

        lightweight_client = create_client(
            api_key=lightweight_config['api_key'],
            base_url=lightweight_config['base_url'],
            model=lightweight_config['model'],
            pool_connections=pool_size,
            pool_maxsize=pool_size
        )
        logger.info(f"创建轻量级模型客户端，连接池大小: {pool_size} (支持 {stage1_max_workers} 个并发线程)")
    except Exception as e:
        logger.error(f"创建轻量级模型客户端失败: {e}")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "agent_metadata": None,
            "error": f"创建轻量级模型客户端失败: {e}"
        }

    # 创建或复用强推理模型客户端（第二阶段）
    # 注意：第二阶段的线程数取决于Uncertain句子的数量，这里先估算
    # 如果客户端是复用的，我们假设它的连接池已经足够大
    if client is None:
        # 估算第二阶段的最大线程数（最多10个）
        estimated_stage2_workers = 10
        pool_size = max(estimated_stage2_workers + 5, 10)

        model_config = _load_model_config()
        strong_config = model_config.get('models', {}).get(model_name)
        if not strong_config:
            raise ValueError(f"未在 model_config.json 中找到模型配置: {model_name}")

        strong_model_client = create_client(
            api_key=strong_config['api_key'],
            base_url=strong_config['base_url'],
            model=strong_config['model'],
            pool_connections=pool_size,
            pool_maxsize=pool_size
        )
        logger.info(f"创建强推理模型客户端，连接池大小: {pool_size} (支持最多 {estimated_stage2_workers} 个并发线程)")
        use_context_manager = True
    else:
        # 复用提供的客户端（假设连接池已经足够大）
        strong_model_client = client
        use_context_manager = False
        logger.info("复用提供的强推理模型客户端（假设连接池已配置）")

    agent_metadata = {
        "stage1_grading": [],
        "stage2_reflection": [],
        "final_evidence_indices": []
    }

    try:
        # ========== 第一阶段：置信度分级（并行处理） ==========
        logger.info(f"开始第一阶段：对 {len(context_list)} 个句子进行置信度分级（并行处理）...")

        correct_indices = []
        incorrect_indices = []
        uncertain_indices = []

        # 使用线程池并行处理所有句子的置信度分级
        # requests.Session 是线程安全的，可以共享使用
        with lightweight_client:
            # 根据句子数量动态调整线程数，但不超过合理上限
            max_workers = min(len(context_list), 10)  # 最多10个线程

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_index = {
                    executor.submit(
                        _grade_sentence_confidence,
                        bias=bias,
                        question=question,
                        sentence_index=idx,
                        sentence=sentence,
                        lightweight_client=lightweight_client,
                        max_retries=max_retries
                    ): idx
                    for idx, sentence in enumerate(context_list)
                }

                # 收集结果（保持顺序）
                grading_results = [None] * len(context_list)
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        grading_result = future.result()
                        grading_results[idx] = grading_result

                        if not grading_result.get("success"):
                            logger.warning(f"句子 {idx} 置信度分级失败，标记为Uncertain")
                            uncertain_indices.append(idx)
                        else:
                            confidence_grade = grading_result.get("confidence_grade")
                            if confidence_grade == "Correct":
                                correct_indices.append(idx)
                            elif confidence_grade == "Incorrect":
                                incorrect_indices.append(idx)
                            else:  # Uncertain
                                uncertain_indices.append(idx)
                    except Exception as e:
                        logger.error(f"句子 {idx} 置信度分级任务异常: {e}")
                        # 创建失败结果
                        failed_result = {
                            "success": False,
                            "sentence_index": idx,
                            "confidence_score": None,
                            "confidence_grade": None,
                            "explanation": None,
                            "raw_response": None,
                            "error": f"任务异常: {e}"
                        }
                        grading_results[idx] = failed_result
                        uncertain_indices.append(idx)

                # 按索引顺序添加到 metadata（确保顺序正确）
                agent_metadata["stage1_grading"] = grading_results

        logger.info(f"第一阶段完成: Correct={len(correct_indices)}, Incorrect={len(incorrect_indices)}, Uncertain={len(uncertain_indices)}")

        # ========== 第二阶段：DVR反思（仅对Uncertain句子，并行处理） ==========
        if uncertain_indices:
            logger.info(f"开始第二阶段：对 {len(uncertain_indices)} 个Uncertain句子进行DVR反思（并行处理）...")

            # 根据Uncertain句子数量动态调整线程数，但不超过合理上限
            max_workers = min(len(uncertain_indices), 10)  # 最多10个线程

            if use_context_manager:
                with strong_model_client:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # 提交所有任务
                        future_to_index = {
                            executor.submit(
                                _reflect_uncertain_sentence,
                                bias=bias,
                                question=question,
                                sentence_index=idx,
                                sentence=context_list[idx],
                                strong_model_client=strong_model_client,
                                max_retries=max_retries
                            ): idx
                            for idx in uncertain_indices
                        }

                        # 收集结果
                        for future in as_completed(future_to_index):
                            idx = future_to_index[future]
                            try:
                                reflection_result = future.result()
                                agent_metadata["stage2_reflection"].append(reflection_result)

                                if reflection_result.get("success"):
                                    final_grade = reflection_result.get("final_confidence_grade")
                                    if final_grade == "Correct":
                                        correct_indices.append(idx)
                                        logger.info(f"句子 {idx} 通过DVR反思，升级为Correct")
                                    elif final_grade == "Incorrect":
                                        incorrect_indices.append(idx)
                                        logger.info(f"句子 {idx} 通过DVR反思，降级为Incorrect")
                                    # 如果仍然是Uncertain，不添加到correct_indices
                                else:
                                    logger.warning(f"句子 {idx} DVR反思失败，保持Uncertain状态")
                            except Exception as e:
                                logger.error(f"句子 {idx} DVR反思任务异常: {e}")
                                # 创建失败结果
                                failed_result = {
                                    "success": False,
                                    "sentence_index": idx,
                                    "final_confidence_grade": None,
                                    "confidence_score": None,
                                    "divide": None,
                                    "verify": None,
                                    "refine": None,
                                    "raw_response": None,
                                    "error": f"任务异常: {e}"
                                }
                                agent_metadata["stage2_reflection"].append(failed_result)
            else:
                # 复用提供的客户端
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_index = {
                        executor.submit(
                            _reflect_uncertain_sentence,
                            bias=bias,
                            question=question,
                            sentence_index=idx,
                            sentence=context_list[idx],
                            strong_model_client=strong_model_client,
                            max_retries=max_retries
                        ): idx
                        for idx in uncertain_indices
                    }

                    # 收集结果
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            reflection_result = future.result()
                            agent_metadata["stage2_reflection"].append(reflection_result)

                            if reflection_result.get("success"):
                                final_grade = reflection_result.get("final_confidence_grade")
                                if final_grade == "Correct":
                                    correct_indices.append(idx)
                                    logger.info(f"句子 {idx} 通过DVR反思，升级为Correct")
                                elif final_grade == "Incorrect":
                                    incorrect_indices.append(idx)
                                    logger.info(f"句子 {idx} 通过DVR反思，降级为Incorrect")
                            else:
                                logger.warning(f"句子 {idx} DVR反思失败，保持Uncertain状态")
                        except Exception as e:
                            logger.error(f"句子 {idx} DVR反思任务异常: {e}")
                            # 创建失败结果
                            failed_result = {
                                "success": False,
                                "sentence_index": idx,
                                "final_confidence_grade": None,
                                "confidence_score": None,
                                "divide": None,
                                "verify": None,
                                "refine": None,
                                "raw_response": None,
                                "error": f"任务异常: {e}"
                            }
                            agent_metadata["stage2_reflection"].append(failed_result)

            logger.info(f"第二阶段完成: 最终Correct={len(correct_indices)}")

        # ========== 生成最终答案 ==========
        # 使用Correct句子作为证据
        final_evidence_indices = sorted(list(set(correct_indices)))
        agent_metadata["final_evidence_indices"] = final_evidence_indices

        logger.info(f"最终证据索引: {final_evidence_indices}")

        # 如果没有Correct句子，但有Uncertain句子，使用Uncertain句子（标记为潜在相关）
        if not final_evidence_indices:
            # 找出仍然为Uncertain的句子
            remaining_uncertain = [
                idx for idx in uncertain_indices
                if idx not in [r.get("sentence_index") for r in agent_metadata["stage2_reflection"] if r.get("success") and r.get("final_confidence_grade") != "Uncertain"]
            ]
            if remaining_uncertain:
                logger.warning(f"没有Correct句子，使用 {len(remaining_uncertain)} 个Uncertain句子作为潜在相关证据")
                final_evidence_indices = remaining_uncertain
                agent_metadata["final_evidence_indices"] = final_evidence_indices

        # 第三阶段（可选）：基于证据句子调用 LLM 判断风险等级；sentences_only=True 时跳过以省 Token
        risk_of_bias = None
        if not sentences_only and final_evidence_indices:
            # 格式化证据句子（只包含Correct句子）
            evidence_sentences = [context_list[idx] for idx in final_evidence_indices]
            formatted_evidence = _format_context_list(evidence_sentences)

            try:
                prompt_template = _load_prompt_template(SSR_EVIDENCE_EXTRACTION_FILE_PATH)
                final_prompt = prompt_template.replace("{bias}", bias if bias else "Not specified")
                final_prompt = final_prompt.replace("{question}", question)
                final_prompt = final_prompt.replace("{context_list}", formatted_evidence)

                try:
                    if use_context_manager:
                        with strong_model_client:
                            final_response = strong_model_client.simple_chat(
                                message=final_prompt,
                                temperature=0,
                            )
                    else:
                        final_response = strong_model_client.simple_chat(
                            message=final_prompt,
                            temperature=0,
                        )

                    final_parsed = _parse_json_response(final_response, "final_risk_assessment")
                    if isinstance(final_parsed, list) and len(final_parsed) > 0:
                        final_parsed = final_parsed[0]

                    if isinstance(final_parsed, dict):
                        risk_of_bias_raw = final_parsed.get("risk_of_bias", "")
                        risk_of_bias = _normalize_risk_level(risk_of_bias_raw)
                    else:
                        risk_of_bias = None
                except Exception as e:
                    logger.error(f"最终风险等级判断调用LLM失败: {e}")
                    risk_of_bias = None
            except Exception as e:
                logger.error(f"加载最终风险等级判断prompt模板失败: {e}")
                risk_of_bias = None

        # 构建返回结果
        result = {
            "success": True,
            "evidence_indices": final_evidence_indices,
            "risk_of_bias": risk_of_bias,
            "raw_response": {
                "stage1_grading": [r.get("raw_response") for r in agent_metadata["stage1_grading"]],
                "stage2_reflection": [r.get("raw_response") for r in agent_metadata["stage2_reflection"]]
            },
            "parsed_response": {
                "stage1_grading": agent_metadata["stage1_grading"],
                "stage2_reflection": agent_metadata["stage2_reflection"]
            },
            "agent_metadata": agent_metadata
        }

        if not sentences_only and risk_of_bias is None:
            result["success"] = False
            result["error"] = "无法确定风险等级"
        else:
            if sentences_only:
                logger.info(f"Agent增强SSR句子检索完成（仅句子），证据索引: {final_evidence_indices}")
            else:
                logger.info(f"Agent增强SSR证据提取成功，风险等级: {risk_of_bias}")

        return result

    except Exception as e:
        logger.error(f"Agent增强SSR证据提取过程中发生错误: {e}")
        return {
            "success": False,
            "evidence_indices": [],
            "risk_of_bias": None,
            "raw_response": None,
            "parsed_response": None,
            "agent_metadata": agent_metadata,
            "error": f"处理过程失败: {e}"
        }

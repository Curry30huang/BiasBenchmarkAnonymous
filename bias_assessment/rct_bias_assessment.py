import json
from loguru import logger
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from utils.openai_llm import OpenAICompatibleClient, create_client
from bias_assessment.evidence_entity import RCTRisk
from typing import Dict, Any, Tuple, Optional

# 加载模型配置文件
MODEL_CONFIG_PATH = Path(__file__).parent.parent / "model_config.json"

# Prompt 通用模板文件路径(不携带outcome)
PROMPT_NORMAL_FILE_PATH = Path(__file__).parent / "prompts" / "rct_domain_signal_normal.txt"
PROMPT_NORMAL_COT_FILE_PATH = Path(__file__).parent / "prompts" / "rct_domain_signal_normal_cot.txt"
RANDOMIZATION_PROCESS_QUESTION_FILE_PATH = Path(__file__).parent / "prompts" / "1randomization_process_question.txt"
INTENDED_INTERVENTION_FILE_PATH = Path(__file__).parent / "prompts" / "2intended_intervention.txt"
MISSING_OUTCOME_FILE_PATH = Path(__file__).parent / "prompts" / "3missing_outcome.txt"
MEASUREMENT_OF_THE_OUTCOME_FILE_PATH = Path(__file__).parent / "prompts" / "4measurement_of_the_outcome.txt"
SELECTION_OF_THE_REPORTED_RESULT_FILE_PATH = Path(__file__).parent / "prompts" / "5selection_of_the_reported_result.txt"
OVERALL_RISK_ASSESSMENT_FILE_PATH = Path(__file__).parent / "prompts" / "overall_risk_assessment.txt"
OVERALL_RISK_ASSESSMENT_COT_FILE_PATH = Path(__file__).parent / "prompts" / "overall_risk_assessment_cot.txt"


def _get_prompt_file_path(base_path: Path, mode: str = "default") -> Path:
    """
    根据模式获取prompt文件路径

    Args:
        base_path: 基础文件路径（默认模式的路径）
        mode: 模式，可选值：default, cot, agent, tool（目前只支持default和cot）

    Returns:
        prompt文件路径
    """
    if mode == "cot":
        # 将 .txt 替换为 _cot.txt
        if base_path.suffix == ".txt":
            return base_path.parent / f"{base_path.stem}_cot{base_path.suffix}"
        else:
            return base_path.parent / f"{base_path.name}_cot"
    elif mode == "default" or mode is None or mode == "":
        return base_path
    else:
        # agent和tool模式暂未实现，使用默认
        logger.warning(f"模式 '{mode}' 暂未实现，使用默认模式")
        return base_path

# RCT偏倚风险评估的五个domain key（与数据集前缀统一）
RCT_BIAS_DOMAIN_KEYS = [
    "randomisation_process",
    "intended_interventions",
    "missing_outcome_data",
    "measurement_outcome",
    "selection_reported_result"
]

# Bias domain显示名称映射
BIAS_DOMAIN_NAMES = {
    "randomisation_process": "Bias arising from the randomisation process",
    "intended_interventions": "Bias due to deviations from intended interventions",
    "missing_outcome_data": "Bias due to missing outcome data",
    "measurement_outcome": "Bias in measurement of the outcome",
    "selection_reported_result": "Bias in selection of the reported result"
}

def _clean_control_characters(text: str) -> str:
    """
    清理JSON字符串中的控制字符

    JSON标准不允许在字符串值中使用未转义的控制字符（ASCII 0-31），
    除了允许的\n, \r, \t等。此函数会清理这些字符。

    Args:
        text: 需要清理的文本

    Returns:
        清理后的文本
    """
    # 保留允许的控制字符：\n (10), \r (13), \t (9)
    # 移除其他控制字符（ASCII 0-31，除了9, 10, 13）
    # 保留已转义的\n, \r, \t等
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        code = ord(char)

        # 如果是反斜杠，检查是否是转义序列
        if char == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            # 如果是允许的转义字符，保留
            if next_char in ['n', 'r', 't', '\\', '"', '/', 'u', 'b', 'f']:
                result.append(char)
                result.append(next_char)
                i += 2
                continue

        # 如果是控制字符（ASCII 0-31），但不是允许的\n, \r, \t
        if 0 <= code <= 31 and code not in [9, 10, 13]:
            # 移除这个控制字符
            i += 1
            continue

        # 保留其他字符
        result.append(char)
        i += 1

    return ''.join(result)


def _fix_json_unescaped_backslashes(text: str) -> str:
    """
    修复JSON字符串值中的未转义反斜杠（更简单但更可靠的方法）

    在JSON字符串值中，反斜杠必须转义为 \\。此函数会修复字符串值内的未转义反斜杠。

    策略：在字符串值内部，将反斜杠后跟非转义字符的情况，转义为 \\。

    Args:
        text: 需要修复的JSON文本

    Returns:
        修复后的文本
    """
    result = []
    i = 0
    in_string = False
    escape_count = 0  # 连续反斜杠计数

    while i < len(text):
        char = text[i]

        # 检测字符串边界（考虑转义的引号）
        if char == '"':
            # 计算前面连续反斜杠的数量
            escape_count = 0
            j = i - 1
            while j >= 0 and text[j] == '\\':
                escape_count += 1
                j -= 1

            # 如果反斜杠数量是偶数，这是字符串边界
            if escape_count % 2 == 0:
                in_string = not in_string
                escape_count = 0
            result.append(char)
            i += 1
            continue

        # 在字符串内部处理反斜杠
        if in_string and char == '\\':
            # 检查下一个字符
            if i + 1 < len(text):
                next_char = text[i + 1]
                # 有效的JSON转义字符
                valid_escapes = ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
                if next_char in valid_escapes:
                    # 已经是有效的转义序列，保留
                    result.append(char)
                    result.append(next_char)
                    i += 2
                else:
                    # 无效的转义序列（如 \mathrm），需要转义反斜杠
                    result.append('\\\\')
                    i += 1
            else:
                # 反斜杠在末尾，转义它
                result.append('\\\\')
                i += 1
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def _load_model_config() -> dict:
    """从 model_config.json 加载配置"""
    try:
        with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"模型配置文件不存在: {MODEL_CONFIG_PATH}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"模型配置文件格式错误: {e}")
        raise


def _create_llm_client(model_name: str) -> OpenAICompatibleClient:
    """创建LLM客户端"""
    # 从 model_config.json 中加载模型配置
    model_config = _load_model_config()
    models = model_config.get('models', {})
    if model_name not in models:
        raise ValueError(f"未在 model_config.json 中找到模型配置: {model_name}")
    config = models[model_name]
    return create_client(
        api_key=config['api_key'],
        base_url=config['base_url'],
        model=config['model']
    )

def _load_prompt_template(file_path: Path) -> str:
    """
    加载prompt模板
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def _assemble_prompts(outcome: str, mode: str = "default") -> Dict[str, str]:
    """
    组装五个prompt基础模板

    将normal_prompt中的{signal_questions}替换为对应的信号问题，
    将{example_id_1}和{example_id_2}替换为固定的问题ID，
    将{bias_domain}替换为对应的domain名称

    Args:
        outcome: 目标结局指标
        mode: 模式，可选值：default, cot, agent, tool（目前只支持default和cot）
    """
    # 根据模式获取prompt文件路径
    normal_prompt_path = _get_prompt_file_path(PROMPT_NORMAL_FILE_PATH, mode)

    # 首先加载通用模板
    normal_prompt = _load_prompt_template(normal_prompt_path)
    # 替换outcome
    normal_prompt = normal_prompt.replace("{outcome}", outcome)

    # 定义prompt文件路径、对应的key和固定的问题ID
    prompt_configs = [
        (RCT_BIAS_DOMAIN_KEYS[0], RANDOMIZATION_PROCESS_QUESTION_FILE_PATH, "1.1", "1.2"),
        (RCT_BIAS_DOMAIN_KEYS[1], INTENDED_INTERVENTION_FILE_PATH, "2.1", "2.2"),
        (RCT_BIAS_DOMAIN_KEYS[2], MISSING_OUTCOME_FILE_PATH, "3.1", "3.2"),
        (RCT_BIAS_DOMAIN_KEYS[3], MEASUREMENT_OF_THE_OUTCOME_FILE_PATH, "4.1", "4.2"),
        (RCT_BIAS_DOMAIN_KEYS[4], SELECTION_OF_THE_REPORTED_RESULT_FILE_PATH, "5.1", "5.2"),
    ]

    prompts = {}

    for key, file_path, first_id, second_id in prompt_configs:
        # 加载信号问题内容
        signal_question_content = _load_prompt_template(file_path)

        # 替换normal_prompt中的占位符
        assembled_prompt = normal_prompt.replace("{signal_questions}", signal_question_content)
        assembled_prompt = assembled_prompt.replace("{example_id_1}", first_id)
        assembled_prompt = assembled_prompt.replace("{example_id_2}", second_id)
        # 替换bias_domain
        bias_domain_name = BIAS_DOMAIN_NAMES.get(key, key)
        assembled_prompt = assembled_prompt.replace("{bias_domain}", bias_domain_name)

        prompts[key] = assembled_prompt

    return prompts


def _parse_json_response(response_text: str, key: str = "") -> Optional[Any]:
    """
    解析JSON响应，提供多层防护措施

    1. 首先尝试直接解析JSON
    2. 如果失败，尝试修复未转义的反斜杠（处理LaTeX公式等）
    3. 如果还失败，尝试清理控制字符后重新解析
    4. 如果还失败，尝试从markdown代码块中提取JSON
    5. 如果还失败，使用括号匹配算法提取[]数组（正确处理嵌套结构）
    6. 如果都失败，返回None

    Args:
        response_text: LLM的响应文本
        key: prompt的key，用于日志记录

    Returns:
        解析后的JSON对象，如果解析失败返回None
    """
    # 若 LLM 返回 None 或非字符串（如 API 异常、空响应），直接返回 None，避免 .lstrip 报错
    if response_text is None or not isinstance(response_text, str):
        if key:
            logger.warning(f"{key} 响应为空或非字符串（类型: {type(response_text)}），跳过解析")
        return None
    # 预处理：清理响应文本，去除BOM字符和前后空白
    response_text = response_text.lstrip('\ufeff').strip()

    # 方法1: 直接尝试解析JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        error_msg = str(e)
        # 方法1.1: 如果是无效转义序列错误（如 \mathrm），尝试修复反斜杠
        if 'Invalid \\escape' in error_msg or 'Invalid escape' in error_msg.lower():
            try:
                response_text_fixed = _fix_json_unescaped_backslashes(response_text)
                return json.loads(response_text_fixed)
            except json.JSONDecodeError:
                pass

        # 方法1.2: 如果错误是控制字符相关，尝试清理后重新解析
        if 'control character' in error_msg.lower() or 'Invalid' in error_msg:
            response_text_cleaned = _clean_control_characters(response_text)
            try:
                return json.loads(response_text_cleaned)
            except json.JSONDecodeError:
                # 清理后仍然失败，尝试修复反斜杠
                try:
                    response_text_fixed = _fix_json_unescaped_backslashes(response_text_cleaned)
                    return json.loads(response_text_fixed)
                except json.JSONDecodeError:
                    pass

        # 方法1.3: 对于其他错误，尝试修复反斜杠和清理控制字符
        try:
            response_text_fixed = _fix_json_unescaped_backslashes(response_text)
            response_text_fixed = _clean_control_characters(response_text_fixed)
            return json.loads(response_text_fixed)
        except json.JSONDecodeError:
            pass

    # 方法2: 尝试从markdown代码块中提取JSON
    # 2a: 按代码块边界截取（支持多行、嵌套），避免正则 \{.*?\} 在嵌套时截断
    for opener in ("```json", "```"):
        if opener in response_text:
            start = response_text.find(opener) + len(opener)
            # 跳过开标签后的换行等
            start = response_text.find("\n", start)
            if start == -1:
                start = response_text.find(opener) + len(opener)
            else:
                start += 1
            end = response_text.find("```", start)
            if end != -1:
                json_str = response_text[start:end].strip()
                if json_str:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        err = str(e)
                        if "Invalid \\escape" in err or "Invalid escape" in err.lower():
                            try:
                                json_str = _fix_json_unescaped_backslashes(json_str)
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                pass
                        if "control character" in err.lower() or "Invalid" in err:
                            try:
                                json_str = _clean_control_characters(json_str)
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                pass
                    # 若 2a 解析失败，继续尝试下面的正则
            break  # 只尝试第一个匹配的 opener

    # 2b: 正则匹配（兼容旧逻辑，对简单无嵌套可能有效）
    markdown_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # ```json ... ```
        r'```\s*(\{.*?\})\s*```',      # ``` ... ```
    ]

    for pattern in markdown_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                error_msg = str(e)
                # 尝试修复未转义的反斜杠
                if 'Invalid \\escape' in error_msg or 'Invalid escape' in error_msg.lower():
                    try:
                        json_str_fixed = _fix_json_unescaped_backslashes(json_str)
                        return json.loads(json_str_fixed)
                    except json.JSONDecodeError:
                        pass
                # 如果错误是控制字符相关，尝试清理后重新解析
                if 'control character' in error_msg.lower() or 'Invalid' in error_msg:
                    json_str_cleaned = _clean_control_characters(json_str)
                    try:
                        return json.loads(json_str_cleaned)
                    except json.JSONDecodeError:
                        # 清理后仍然失败，尝试修复反斜杠
                        try:
                            json_str_fixed = _fix_json_unescaped_backslashes(json_str_cleaned)
                            return json.loads(json_str_fixed)
                        except json.JSONDecodeError:
                            continue
                continue

    # 方法3: 使用括号匹配算法提取{}或[]之间的内容（对象或数组格式）
    # 查找最外层的 {} 或 [] 对，正确处理嵌套的JSON结构
    # 先尝试清理控制字符，然后再进行括号匹配
    response_text_cleaned_for_brackets = _clean_control_characters(response_text)

    # 先尝试提取对象 {}
    brace_start = response_text_cleaned_for_brackets.find('{')
    if brace_start != -1:
        # 使用计数器匹配最外层的 }
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(brace_start, len(response_text_cleaned_for_brackets)):
            char = response_text_cleaned_for_brackets[i]

            # 处理转义字符
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # 处理字符串内的内容
            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            # 只在字符串外处理括号
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 找到匹配的 }
                        json_str = response_text_cleaned_for_brackets[brace_start:i+1]
                        # 清理前后空白字符
                        json_str = json_str.strip()
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            error_msg = str(e)
                            # 尝试修复未转义的反斜杠
                            if 'Invalid \\escape' in error_msg or 'Invalid escape' in error_msg.lower():
                                try:
                                    json_str_fixed = _fix_json_unescaped_backslashes(json_str)
                                    return json.loads(json_str_fixed)
                                except json.JSONDecodeError:
                                    pass

                            # 如果错误是控制字符相关，尝试清理后重新解析
                            if 'control character' in error_msg.lower() or 'Invalid' in error_msg:
                                # 清理控制字符后重试
                                json_str_cleaned = _clean_control_characters(json_str)
                                try:
                                    return json.loads(json_str_cleaned)
                                except json.JSONDecodeError:
                                    # 清理后仍然失败，尝试修复反斜杠
                                    try:
                                        json_str_fixed = _fix_json_unescaped_backslashes(json_str_cleaned)
                                        return json.loads(json_str_fixed)
                                    except json.JSONDecodeError:
                                        pass

                            # 记录详细的错误信息以便调试
                            logger.warning(
                                f"{key} 方法3提取的JSON解析失败: {error_msg}。"
                                f"提取的JSON长度: {len(json_str)}, "
                                f"前200字符: {json_str[:200]}"
                            )
                            # 尝试清理可能的BOM和不可见字符
                            json_str_cleaned = json_str.lstrip('\ufeff').strip()
                            if json_str_cleaned != json_str:
                                try:
                                    return json.loads(json_str_cleaned)
                                except json.JSONDecodeError:
                                    # 尝试修复反斜杠
                                    try:
                                        json_str_fixed = _fix_json_unescaped_backslashes(json_str_cleaned)
                                        return json.loads(json_str_fixed)
                                    except json.JSONDecodeError:
                                        pass
                            break

    # 如果对象提取失败，尝试提取数组 []
    bracket_start = response_text_cleaned_for_brackets.find('[')
    if bracket_start != -1:
        # 使用计数器匹配最外层的 ]
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(bracket_start, len(response_text_cleaned_for_brackets)):
            char = response_text_cleaned_for_brackets[i]

            # 处理转义字符
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # 处理字符串内的内容
            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            # 只在字符串外处理括号
            if not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        # 找到匹配的 ]
                        json_str = response_text_cleaned_for_brackets[bracket_start:i+1]
                        # 清理前后空白字符
                        json_str = json_str.strip()
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            error_msg = str(e)
                            # 尝试修复未转义的反斜杠
                            if 'Invalid \\escape' in error_msg or 'Invalid escape' in error_msg.lower():
                                try:
                                    json_str_fixed = _fix_json_unescaped_backslashes(json_str)
                                    return json.loads(json_str_fixed)
                                except json.JSONDecodeError:
                                    pass

                            # 如果错误是控制字符相关，尝试清理后重新解析
                            if 'control character' in error_msg.lower() or 'Invalid' in error_msg:
                                # 清理控制字符后重试
                                json_str_cleaned = _clean_control_characters(json_str)
                                try:
                                    return json.loads(json_str_cleaned)
                                except json.JSONDecodeError:
                                    # 清理后仍然失败，尝试修复反斜杠
                                    try:
                                        json_str_fixed = _fix_json_unescaped_backslashes(json_str_cleaned)
                                        return json.loads(json_str_fixed)
                                    except json.JSONDecodeError:
                                        pass

                            # 记录详细的错误信息以便调试
                            logger.warning(
                                f"{key} 方法3提取的JSON解析失败: {error_msg}。"
                                f"提取的JSON长度: {len(json_str)}, "
                                f"前200字符: {json_str[:200]}"
                            )
                            # 尝试清理可能的BOM和不可见字符
                            json_str_cleaned = json_str.lstrip('\ufeff').strip()
                            if json_str_cleaned != json_str:
                                try:
                                    return json.loads(json_str_cleaned)
                                except json.JSONDecodeError:
                                    # 尝试修复反斜杠
                                    try:
                                        json_str_fixed = _fix_json_unescaped_backslashes(json_str_cleaned)
                                        return json.loads(json_str_fixed)
                                    except json.JSONDecodeError:
                                        pass
                            break

    # 所有方法都失败
    logger.error(f"{key} JSON解析失败，已尝试所有方法。响应文本长度: {len(response_text)}")

    # 尝试最后一次：修复反斜杠并清理所有控制字符后解析
    final_cleaned = _fix_json_unescaped_backslashes(response_text)
    final_cleaned = _clean_control_characters(final_cleaned)
    try:
        return json.loads(final_cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"{key} 最终清理后仍解析失败: {e}")
        if len(response_text) < 1000:  # 如果响应较短，打印出来便于调试
            logger.error(f"原始响应内容: {response_text}")
            logger.error(f"清理后响应内容: {final_cleaned}")
        else:
            # 如果响应较长，打印前500和后200字符
            logger.error(f"原始响应前500字符: {response_text[:500]}")
            logger.error(f"原始响应后200字符: {response_text[-200:]}")
            logger.error(f"清理后响应前500字符: {final_cleaned[:500]}")
            logger.error(f"清理后响应后200字符: {final_cleaned[-200:]}")

    return None

def _process_single_prompt(
    key: str,
    prompt_template: str,
    context: str,
    model_name: str = "openai/gpt-5.1",
    client: Optional[OpenAICompatibleClient] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    处理单个prompt并返回结果

    Args:
        key: prompt的key
        prompt_template: prompt模板
        context: 上下文内容
        model_name: 模型名称
        client: 可选的LLM客户端，如果提供则复用，否则创建新客户端

    Returns:
        (key, result_dict) 元组
    """
    # 如果没有提供客户端，创建新客户端（使用上下文管理器）
    if client is None:
        client = _create_llm_client(model_name)
        use_context_manager = True
    else:
        use_context_manager = False

    try:
        # 如果使用上下文管理器，确保连接池正确释放
        if use_context_manager:
            with client:
                # 使用 replace 替换 {context}，避免 JSON 中的花括号导致 format() 报错
                prompt = prompt_template.replace("{context}", context)

                # 调用LLM
                logger.info(f"开始处理 {key}...")
                response = client.simple_chat(
                    message=prompt,
                    temperature=0,
                )

                # 尝试解析JSON响应（多层防护）
                parsed_response = _parse_json_response(response, key)

                if parsed_response is not None:
                    result = {
                        "success": True,
                        "success_answer": True,
                        "success_parse": True,
                        "response": parsed_response,
                        "raw_response": response
                    }
                else:
                    # 解析失败，返回原始响应
                    logger.warning(f"{key} 的响应无法解析为JSON，返回原始响应")
                    logger.warning(f"原始响应: {response}")
                    result = {
                        "success": False,
                        "success_parse": False,
                        "response": None,
                        "raw_response": response
                    }

                logger.info(f"{key} 处理完成")
                return key, result
        else:
            # 复用提供的客户端
            # 使用 replace 替换 {context}，避免 JSON 中的花括号导致 format() 报错
            prompt = prompt_template.replace("{context}", context)

            # 调用LLM
            logger.info(f"开始处理 {key}...")
            response = client.simple_chat(
                message=prompt,
                temperature=0,
            )

            # 尝试解析JSON响应（多层防护）
            parsed_response = _parse_json_response(response, key)

            if parsed_response is not None:
                result = {
                    "success": True,
                    "success_answer": True,
                    "success_parse": True,
                    "response": parsed_response,
                    "raw_response": response
                }
            else:
                # 解析失败，返回原始响应
                logger.warning(f"{key} 的响应无法解析为JSON，返回原始响应")
                logger.warning(f"原始响应: {response}")
                result = {
                    "success": False,
                    "success_parse": False,
                    "response": None,
                    "raw_response": response
                }

            logger.info(f"{key} 处理完成")
            return key, result

    except Exception as e:
        logger.error(f"处理 {key} 时发生错误: {e}")
        return key, {
            "success": False,
            "success_answer": False,
            "success_parse": False,
            "error": str(e),
            "response": None
        }


# 重试时携带错误信息的 prompt 后缀（英文，便于模型理解与兼容）
RETRY_ERROR_FOOTER = (
    "\n\nRequired: output a single JSON object (not an array) with exactly two keys: "
    '"signaling_questions" (array of objects with id, answer_code, evidence_quote) and '
    '"domain_risk" (string: exactly one of "Low risk of bias", "Some concerns", "High risk of bias"). '
    "Do not wrap the JSON in markdown code blocks."
)


def _domain_result_ok(result: Dict[str, Any]) -> bool:
    """
    判断某 domain 的最终结果是否正常：解析成功、有 response、有 domain_risk、且无提取异常。
    任一不满足则视为异常，可触发重试。
    """
    if not result:
        return False
    if result.get("extraction_error"):
        return False
    if not result.get("success_parse"):
        return False
    if result.get("response") is None:
        return False
    if not result.get("domain_risk"):
        return False
    return True


def _build_retry_error_description(result: Dict[str, Any]) -> str:
    """
    根据 result 状态生成统一格式的重试错误说明（兼容各类异常）。
    格式固定为 "What went wrong: ..." 便于模型理解。
    """
    if result.get("extraction_error"):
        return (
            "What went wrong: An error occurred while extracting your response. "
            "Ensure you output valid JSON only, with keys 'signaling_questions' and 'domain_risk'."
        )
    if not result.get("success_parse") or result.get("response") is None:
        return (
            "What went wrong: The response could not be parsed as valid JSON, or was wrapped in markdown incorrectly. "
            "Output a single JSON object only, with keys 'signaling_questions' and 'domain_risk'."
        )
    resp = result.get("response")
    if isinstance(resp, list):
        return (
            "What went wrong: You returned a JSON array. "
            "Required: a single JSON object with keys 'signaling_questions' (array) and 'domain_risk' (string)."
        )
    if not isinstance(resp, dict):
        return (
            "What went wrong: Your response was not a JSON object. "
            "Required: a single JSON object with keys 'signaling_questions' and 'domain_risk'."
        )
    domain_risk_val = (resp.get("domain_risk") or "").strip() if isinstance(resp.get("domain_risk"), str) else ""
    if not domain_risk_val:
        return (
            "What went wrong: Your JSON object was missing the key 'domain_risk'. "
            "It must be exactly one of: 'Low risk of bias', 'Some concerns', 'High risk of bias'."
        )
    if result.get("domain_risk_error"):
        return (
            "What went wrong: The value of 'domain_risk' was not valid. "
            "It must be exactly one of: 'Low risk of bias', 'Some concerns', 'High risk of bias'."
        )
    return (
        "What went wrong: The response did not meet the required format. "
        "Output a single JSON object with keys 'signaling_questions' and 'domain_risk'."
    )


def _process_single_prompt_with_error_feedback(
    key: str,
    prompt_template: str,
    context: str,
    model_name: str,
    client: OpenAICompatibleClient,
    error_description: str,
    previous_raw_response: str,
    max_previous_response_chars: int = 2500,
) -> Tuple[str, Dict[str, Any]]:
    """
    带错误反馈的 domain 重试：将上一轮错误描述与原始响应拼入 prompt，只调用一次 LLM。
    """
    base_prompt = prompt_template.replace("{context}", context)
    excerpt = (previous_raw_response or "").strip()
    if len(excerpt) > max_previous_response_chars:
        excerpt = excerpt[:max_previous_response_chars] + "\n... (truncated)"
    retry_block = (
        "\n\n[RETRY - Your previous response was invalid.]\n"
        f"{error_description}\n\n"
        "Previous response (for reference):\n"
        f"{excerpt}\n"
        f"{RETRY_ERROR_FOOTER}"
    )
    prompt = base_prompt + retry_block
    logger.info(f"{key} 重试（携带错误反馈）...")
    try:
        response = client.simple_chat(message=prompt, temperature=0)
        parsed = _parse_json_response(response, key)
        if parsed is not None:
            return key, {
                "success": True,
                "success_answer": True,
                "success_parse": True,
                "response": parsed,
                "raw_response": response,
            }
        return key, {
            "success": False,
            "success_parse": False,
            "response": None,
            "raw_response": response,
        }
    except Exception as e:
        logger.error(f"{key} 重试时发生错误: {e}")
        return key, {
            "success": False,
            "success_parse": False,
            "error": str(e),
            "response": None,
            "raw_response": previous_raw_response,
        }


def _extract_domain_result(domain_key: str, result: Dict[str, Any]) -> None:
    """
    从 result["response"] 中提取 signaling_questions 和 domain_risk，并写回 result。
    处理数组/字典格式及 domain_risk 转换；不打印原始响应（由上层在需要时打印）。
    """
    if not result.get("success_parse") or not result.get("response"):
        result["signaling_questions"] = None
        result["domain_risk"] = None
        return
    response_obj = result.get("response")
    if isinstance(response_obj, list):
        if len(response_obj) > 0:
            response_obj = response_obj[0]
            logger.info(f"{domain_key} 响应是数组格式，提取第一个元素")
        else:
            result["signaling_questions"] = None
            result["domain_risk"] = None
            return
    if not isinstance(response_obj, dict):
        result["signaling_questions"] = None
        result["domain_risk"] = None
        return
    sq = response_obj.get("signaling_questions")
    if sq is not None and not isinstance(sq, list):
        sq = []
    result["signaling_questions"] = sq or []
    logger.info(f"{domain_key} 提取到 {len(result['signaling_questions'])} 个信号问题答案")
    domain_risk_str = (response_obj.get("domain_risk") or "").strip() if isinstance(response_obj.get("domain_risk"), str) else ""
    result.pop("domain_risk_error", None)
    if domain_risk_str:
        try:
            result["domain_risk"] = _string_to_rct_risk(domain_risk_str)
            logger.info(f"{domain_key} 大模型判断的domain_risk: {result['domain_risk'].value}")
        except ValueError as e:
            result["domain_risk"] = None
            result["domain_risk_error"] = str(e)
    else:
        result["domain_risk"] = None
    return


def _string_to_rct_risk(risk_str: str) -> RCTRisk:
    """
    将字符串转换为RCTRisk枚举

    Args:
        risk_str: 风险字符串

    Returns:
        RCTRisk枚举值

    Raises:
        ValueError: 如果字符串无法匹配任何枚举值
    """
    risk_str = risk_str.strip()
    for risk in RCTRisk:
        if risk.value.lower() == risk_str.lower():
            return risk
    raise ValueError(f"无法将字符串 '{risk_str}' 转换为 RCTRisk 枚举")

def _overall_risk_assessment(
    rct_bias_result: dict,
    outcome: str,
    model_name: str = "openai/gpt-5.1",
    max_retries: int = 3,
    client: Optional[OpenAICompatibleClient] = None,
    mode: str = "default"
) -> Dict[str, Any]:
    """
    最终overall 风险评定（通过LLM判断）

    根据五个domain的偏倚风险评估结果，构建prompt询问大模型，获取整体评估结果。

    Args:
        rct_bias_result: assess_rct_bias返回的结果字典，包含五个domain的domain_risk结果
        outcome: 目标结局指标
        model_name: 使用的模型名称，默认为 "openai/gpt-5.1"
        max_retries: 最大重试次数，默认为3
        client: 可选的LLM客户端，如果提供则复用，否则创建新客户端

    Returns:
        包含以下字段的字典：
        - overall_risk: 整体偏倚风险评估结果（RCTRisk枚举类型），如果失败返回None
        - success: 是否成功
        - raw_response: 原始LLM响应
        - parsed_response: 解析后的响应
        - error: 错误信息（如果有）
    """
    # 收集所有domain的domain_risk结果（字符串格式）
    domain_risks = {}
    for domain_key in RCT_BIAS_DOMAIN_KEYS:
        domain_result = rct_bias_result.get(domain_key, {})
        domain_risk = domain_result.get("domain_risk")
        if domain_risk:
            # 如果domain_risk是枚举，转换为字符串
            if isinstance(domain_risk, RCTRisk):
                domain_risk_str = domain_risk.value
            elif isinstance(domain_risk, str):
                domain_risk_str = domain_risk
            else:
                domain_risk_str = str(domain_risk)
            domain_risks[domain_key] = domain_risk_str
        else:
            # 如果没有domain_risk，使用 "Not assessed"
            domain_risks[domain_key] = "Not assessed"
            logger.warning(f"Domain {domain_key} 没有domain_risk结果，使用 'Not assessed'")

    # 初始化返回结果
    result = {
        "overall_risk": None,
        "success": False,
        "raw_response": None,
        "parsed_response": None,
        "error": None
    }

    # 根据模式获取prompt文件路径
    overall_prompt_path = _get_prompt_file_path(OVERALL_RISK_ASSESSMENT_FILE_PATH, mode)

    # 加载prompt模板
    try:
        prompt_template = _load_prompt_template(overall_prompt_path)
    except Exception as e:
        logger.error(f"加载overall_risk_assessment prompt模板失败: {e}")
        result["error"] = f"加载prompt模板失败: {e}"
        return result

    # 构建prompt，替换占位符
    prompt = prompt_template.replace("{outcome}", outcome)
    prompt = prompt.replace("{randomisation_process_risk}", domain_risks.get("randomisation_process", "Not assessed"))
    prompt = prompt.replace("{intended_interventions_risk}", domain_risks.get("intended_interventions", "Not assessed"))
    prompt = prompt.replace("{missing_outcome_data_risk}", domain_risks.get("missing_outcome_data", "Not assessed"))
    prompt = prompt.replace("{measurement_outcome_risk}", domain_risks.get("measurement_outcome", "Not assessed"))
    prompt = prompt.replace("{selection_reported_result_risk}", domain_risks.get("selection_reported_result", "Not assessed"))

    # 如果没有提供客户端，创建新客户端（使用上下文管理器）
    if client is None:
        client = _create_llm_client(model_name)
        use_context_manager = True
    else:
        use_context_manager = False

    # 重试机制
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"开始调用LLM进行overall_risk评估（尝试 {attempt}/{max_retries}）...")

            # 如果使用上下文管理器，确保连接池正确释放
            if use_context_manager:
                with client:
                    # 调用LLM
                    response = client.simple_chat(
                        message=prompt,
                        temperature=0,
                    )

                    # 解析JSON响应
                    parsed_response = _parse_json_response(response, "overall_risk_assessment")

                    if parsed_response is None:
                        logger.warning(f"overall_risk_assessment 响应解析失败（尝试 {attempt}/{max_retries}）")
                        if attempt < max_retries:
                            continue
                        else:
                            logger.error("overall_risk_assessment 所有重试都失败，无法解析响应")
                            return None

                    # 处理数组格式的响应
                    if isinstance(parsed_response, list):
                        if len(parsed_response) > 0:
                            parsed_response = parsed_response[0]
                            logger.info("overall_risk_assessment 响应是数组格式，提取第一个元素")
                        else:
                            logger.warning(f"overall_risk_assessment 响应是空数组（尝试 {attempt}/{max_retries}）")
                            result["raw_response"] = response
                            result["error"] = "响应是空数组"
                            if attempt < max_retries:
                                continue
                            else:
                                return result

                    # 存储原始响应和解析后的响应
                    result["raw_response"] = response
                    result["parsed_response"] = parsed_response

                    # 提取overall_risk
                    if isinstance(parsed_response, dict):
                        overall_risk_str = parsed_response.get("overall_risk", "")
                        if overall_risk_str:
                            try:
                                overall_risk = _string_to_rct_risk(overall_risk_str)
                                logger.info(f"overall_risk_assessment 成功，结果: {overall_risk.value}")
                                result["overall_risk"] = overall_risk
                                result["success"] = True
                                return result
                            except ValueError as e:
                                logger.warning(f"overall_risk字符串 '{overall_risk_str}' 无法转换为枚举: {e}（尝试 {attempt}/{max_retries}）")
                                result["error"] = f"无法转换overall_risk: {e}"
                                if attempt < max_retries:
                                    continue
                                else:
                                    logger.error("overall_risk_assessment 所有重试都失败，无法转换overall_risk")
                                    return result
                        else:
                            logger.warning(f"overall_risk_assessment 响应中没有overall_risk字段（尝试 {attempt}/{max_retries}）")
                            result["error"] = "响应中缺少overall_risk字段"
                            if attempt < max_retries:
                                continue
                            else:
                                logger.error("overall_risk_assessment 所有重试都失败，响应中缺少overall_risk字段")
                                return result
                    else:
                        logger.warning(f"overall_risk_assessment 响应不是字典格式（尝试 {attempt}/{max_retries}）")
                        result["error"] = f"响应不是字典格式（类型: {type(parsed_response)}）"
                        if attempt < max_retries:
                            continue
                        else:
                            logger.error("overall_risk_assessment 所有重试都失败，响应格式不正确")
                            return result
            else:
                # 复用提供的客户端
                # 调用LLM
                response = client.simple_chat(
                    message=prompt,
                    temperature=0,
                )

                # 解析JSON响应
                parsed_response = _parse_json_response(response, "overall_risk_assessment")

                if parsed_response is None:
                    logger.warning(f"overall_risk_assessment 响应解析失败（尝试 {attempt}/{max_retries}）")
                    result["raw_response"] = response
                    result["error"] = "响应解析失败"
                    if attempt < max_retries:
                        continue
                    else:
                        logger.error("overall_risk_assessment 所有重试都失败，无法解析响应")
                        return result

                # 处理数组格式的响应
                if isinstance(parsed_response, list):
                    if len(parsed_response) > 0:
                        parsed_response = parsed_response[0]
                        logger.info("overall_risk_assessment 响应是数组格式，提取第一个元素")
                    else:
                        logger.warning(f"overall_risk_assessment 响应是空数组（尝试 {attempt}/{max_retries}）")
                        result["parsed_response"] = parsed_response
                        result["error"] = "响应是空数组"
                        if attempt < max_retries:
                            continue
                        else:
                            return result

                # 存储解析后的响应
                result["parsed_response"] = parsed_response

                # 提取overall_risk
                if isinstance(parsed_response, dict):
                    overall_risk_str = parsed_response.get("overall_risk", "")
                    if overall_risk_str:
                        try:
                            overall_risk = _string_to_rct_risk(overall_risk_str)
                            logger.info(f"overall_risk_assessment 成功，结果: {overall_risk.value}")
                            result["overall_risk"] = overall_risk
                            result["success"] = True
                            return result
                        except ValueError as e:
                            logger.warning(f"overall_risk字符串 '{overall_risk_str}' 无法转换为枚举: {e}（尝试 {attempt}/{max_retries}）")
                            result["error"] = f"无法转换overall_risk: {e}"
                            if attempt < max_retries:
                                continue
                            else:
                                logger.error("overall_risk_assessment 所有重试都失败，无法转换overall_risk")
                                return result
                    else:
                        logger.warning(f"overall_risk_assessment 响应中没有overall_risk字段（尝试 {attempt}/{max_retries}）")
                        result["error"] = "响应中缺少overall_risk字段"
                        if attempt < max_retries:
                            continue
                        else:
                            logger.error("overall_risk_assessment 所有重试都失败，响应中缺少overall_risk字段")
                            return result
                else:
                    logger.warning(f"overall_risk_assessment 响应不是字典格式（尝试 {attempt}/{max_retries}）")
                    result["error"] = f"响应不是字典格式（类型: {type(parsed_response)}）"
                    if attempt < max_retries:
                        continue
                    else:
                        logger.error("overall_risk_assessment 所有重试都失败，响应格式不正确")
                        return result

        except Exception as e:
            logger.error(f"overall_risk_assessment 调用LLM时发生错误（尝试 {attempt}/{max_retries}）: {e}")
            if attempt < max_retries:
                logger.info(f"将在 {attempt + 1} 秒后重试...")
                import time
                time.sleep(1)  # 等待1秒后重试
                continue
            else:
                logger.error("overall_risk_assessment 所有重试都失败")
                result["error"] = f"所有重试都失败: {e}"
                return result

    result["error"] = "所有重试都失败"
    return result

def assess_rct_bias(context: str, outcome: str, model_name: str = "openai/gpt-5.1", mode: str = "default") -> Dict[str, Any]:
    """
    RCT 偏倚风险评估（并行处理五个prompt）

    Args:
        context: 论文上下文内容
        outcome: 目标结局指标
        model_name: 使用的模型名称，默认为 "openai/gpt-5.1"
        mode: 模式，可选值：default, cot, agent, tool（目前只支持default和cot）

    Returns:
        包含五个评估域结果的字典，每个domain包含：
        - signaling_questions: 大模型回答的子问题答案列表
        - domain_risk: 大模型判断的domain风险等级（RCTRisk枚举）
        - overall_risk_judgement: 整体风险评估结果（RCTRisk枚举）
    """
    # 组装所有prompt
    prompts = _assemble_prompts(outcome, mode=mode)

    # 使用线程池并行处理
    results = {}

    # 创建一个共享的客户端（requests.Session是线程安全的，可以共享使用）
    # 这样可以实现连接池的复用，提高性能
    shared_client = _create_llm_client(model_name)

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有任务，共享同一个客户端（线程安全）
            future_to_key = {
                executor.submit(
                    _process_single_prompt,
                    key,
                    prompt_template,
                    context,
                    model_name,
                    shared_client  # 传递共享的客户端，实现连接池复用
                ): key
                for key, prompt_template in prompts.items()
            }

            # 收集结果
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result_key, result = future.result()
                    results[result_key] = result
                except Exception as e:
                    logger.error(f"获取 {key} 的结果时发生错误: {e}")
                    results[key] = {
                        "success": False,
                        "error": str(e),
                        "response": None
                    }

        logger.info(f"所有prompt处理完成，成功: {sum(1 for r in results.values() if r.get('success'))}/5")

        # 为每个domain提取大模型返回的结果
        for domain_key, result in results.items():
            # 若首次解析失败但存在 raw_response，尝试从 markdown/括号 等再次解析（直接恢复，避免打解析失败）
            response_obj = result.get("response")
            if not response_obj and result.get("raw_response"):
                reparsed = _parse_json_response(result["raw_response"], domain_key)
                if reparsed is not None:
                    result["response"] = reparsed
                    result["success_parse"] = True
                    response_obj = reparsed
                    logger.info(f"{domain_key} 从 raw_response 再次解析成功，继续提取结果")
            if result.get("success_parse") and response_obj is not None:
                try:
                    _extract_domain_result(domain_key, result)
                except Exception as e:
                    logger.error(f"提取 {domain_key} 结果时发生错误: {e}")
                    result["signaling_questions"] = None
                    result["domain_risk"] = None
                    result["extraction_error"] = str(e)
        else:
            logger.warning(f"{domain_key} 的响应解析失败，无法提取结果")
            result["signaling_questions"] = None
            result["domain_risk"] = None
            # 输出原始响应以便调试
            raw_response = result.get("raw_response")
            if raw_response:
                logger.warning(f"{domain_key} 的原始LLM响应（解析失败）:")
                if len(raw_response) < 2000:
                    logger.warning(f"{raw_response}")
                else:
                    logger.warning(f"前1000字符: {raw_response[:1000]}")
                    logger.warning(f"后500字符: {raw_response[-500:]}")

        # 尝试从 response 中恢复缺失的 domain_risk（修复功能）
        for domain_key in RCT_BIAS_DOMAIN_KEYS:
            domain_result = results.get(domain_key, {})
            domain_risk = domain_result.get("domain_risk")
            if not domain_risk:
                # 尝试从 response 中恢复
                response = domain_result.get("response")
                if response:
                    # response 可能是字典或列表（列表时取第一个元素）
                    response_obj = None
                    if isinstance(response, dict):
                        response_obj = response
                    elif isinstance(response, list) and len(response) > 0:
                        response_obj = response[0] if isinstance(response[0], dict) else None

                    if response_obj and isinstance(response_obj, dict):
                        domain_risk_str = response_obj.get("domain_risk", "")
                        if domain_risk_str:
                            try:
                                domain_risk = _string_to_rct_risk(domain_risk_str)
                                domain_result["domain_risk"] = domain_risk
                                logger.info(f"{domain_key} 从 response 中恢复 domain_risk: {domain_risk.value}")
                            except ValueError as e:
                                logger.warning(f"{domain_key} 从 response 中提取的 domain_risk '{domain_risk_str}' 无法转换为枚举: {e}")

        # 对“最终判断异常”的 domain 做一次重试（仅一次，防止反复调用）：解析失败、格式错误、缺 domain_risk、提取异常等任一出现即触发
        domains_to_retry = [k for k in RCT_BIAS_DOMAIN_KEYS if not _domain_result_ok(results.get(k, {}))]
        if domains_to_retry:
            logger.info(f"以下 domain 最终结果异常，将携带错误信息重试一次（仅此一次）: {domains_to_retry}")
        for domain_key in domains_to_retry:
            result = results.get(domain_key, {})
            raw = (result.get("raw_response") or "").strip()
            error_description = _build_retry_error_description(result)
            _, retry_result = _process_single_prompt_with_error_feedback(
                domain_key,
                prompts[domain_key],
                context,
                model_name,
                shared_client,
                error_description,
                raw,
            )
            results[domain_key] = retry_result
            _extract_domain_result(domain_key, results[domain_key])
            if _domain_result_ok(results[domain_key]):
                logger.info(f"{domain_key} 重试后结果正常")

        # 检查是否所有domain都有domain_risk，如果有缺失则跳过overall_risk_assessment
        all_domains_have_risk = True
        missing_domains = []
        for domain_key in RCT_BIAS_DOMAIN_KEYS:
            domain_result = results.get(domain_key, {})
            domain_risk = domain_result.get("domain_risk")
            if not domain_risk:
                all_domains_have_risk = False
                missing_domains.append(domain_key)

        if all_domains_have_risk:
            # 评估 overall risk（通过LLM），复用共享客户端
            try:
                overall_risk_result = _overall_risk_assessment(results, outcome, model_name, client=shared_client, mode=mode)
                # overall_risk_result 现在是一个字典，包含 overall_risk, success, raw_response 等
                if isinstance(overall_risk_result, dict):
                    results["overall_risk_judgement"] = overall_risk_result.get("overall_risk")
                    # 确保存储 overall_risk_raw，包含所有大模型输出的内容
                    results["overall_risk_raw"] = {
                        "success": overall_risk_result.get("success", False),
                        "raw_response": overall_risk_result.get("raw_response"),
                        "parsed_response": overall_risk_result.get("parsed_response"),
                        "error": overall_risk_result.get("error"),
                    }
                else:
                    # 兼容旧格式（如果返回的是枚举）
                    results["overall_risk_judgement"] = overall_risk_result
                    # 即使返回的是枚举，也尝试存储原始响应（如果存在）
                    if hasattr(overall_risk_result, "__dict__"):
                        # 如果返回的对象有属性，尝试提取
                        pass
            except Exception as e:
                logger.error(f"评估overall_risk时发生错误: {e}")
                results["overall_risk_judgement"] = None
                results["overall_risk_error"] = str(e)
                # 即使出现异常，也记录错误信息到 overall_risk_raw
                results["overall_risk_raw"] = {
                    "success": False,
                    "raw_response": None,
                    "parsed_response": None,
                    "error": str(e),
                }
        else:
            logger.warning(f"以下domain缺少domain_risk，跳过overall_risk_assessment: {missing_domains}")
            results["overall_risk_judgement"] = None
            results["overall_risk_skipped"] = True
            results["missing_domains"] = missing_domains
            # 记录跳过原因到 overall_risk_raw
            results["overall_risk_raw"] = {
                "success": False,
                "raw_response": None,
                "parsed_response": None,
                "error": f"跳过overall_risk_assessment: 以下domain缺少domain_risk - {missing_domains}",
                "skipped": True,
                "missing_domains": missing_domains,
            }
    finally:
        # 确保共享客户端正确关闭
        shared_client.close()

    return results
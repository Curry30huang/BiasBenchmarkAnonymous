import json
from loguru import logger
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from utils.openai_llm import OpenAICompatibleClient, create_client
from bias_assessment.evidence_entity import RCTRisk
from typing import Dict, Any, List, Tuple, Union, Optional

# 加载模型配置文件
MODEL_CONFIG_PATH = Path(__file__).parent.parent / "model_config.json"

# Prompt 通用模板文件路径(不携带outcome)
PROMPT_NORMAL_FILE_PATH = Path(__file__).parent / "prompts" / "rct_domain_signal_normal.txt"
RANDOMIZATION_PROCESS_QUESTION_FILE_PATH = Path(__file__).parent / "prompts" / "1randomization_process_question.txt"
INTENDED_INTERVENTION_FILE_PATH = Path(__file__).parent / "prompts" / "2intended_intervention.txt"
MISSING_OUTCOME_FILE_PATH = Path(__file__).parent / "prompts" / "3missing_outcome.txt"
MEASUREMENT_OF_THE_OUTCOME_FILE_PATH = Path(__file__).parent / "prompts" / "4measurement_of_the_outcome.txt"
SELECTION_OF_THE_REPORTED_RESULT_FILE_PATH = Path(__file__).parent / "prompts" / "5selection_of_the_reported_result.txt"

# RCT偏倚风险评估的五个domain key
RCT_BIAS_DOMAIN_KEYS = [
    "randomization_process_question",
    "intended_intervention",
    "missing_outcome",
    "measurement_of_the_outcome",
    "selection_of_the_reported_result"
]

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


def _fix_json_string_escapes(text: str) -> str:
    """
    修复JSON字符串中的未转义反斜杠

    在JSON字符串值中，反斜杠必须转义为 \\，但LLM可能返回包含未转义反斜杠的文本
    （例如LaTeX公式中的 \mathrm{mmHg}）。

    此函数会智能地修复字符串值内的反斜杠，同时避免破坏已经正确转义的序列。

    Args:
        text: 需要修复的JSON文本

    Returns:
        修复后的文本
    """
    result = []
    i = 0
    in_string = False
    escape_next = False

    while i < len(text):
        char = text[i]

        # 处理转义状态
        if escape_next:
            escape_next = False
            result.append(char)
            i += 1
            continue

        if char == '\\':
            escape_next = True
            result.append(char)
            i += 1
            continue

        # 检测字符串的开始和结束
        if char == '"' and (i == 0 or text[i-1] != '\\' or (i >= 2 and text[i-2] == '\\')):
            # 检查是否是转义的引号（偶数个连续反斜杠后跟引号）
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == '\\':
                backslash_count += 1
                j -= 1
            # 如果反斜杠数量是偶数，则这个引号是字符串边界
            if backslash_count % 2 == 0:
                in_string = not in_string
            result.append(char)
            i += 1
            continue

        # 在字符串内部，检查未转义的反斜杠
        if in_string and char == '\\':
            # 检查下一个字符是否是有效的转义字符
            if i + 1 < len(text):
                next_char = text[i + 1]
                # 如果是有效的转义字符，保留原样
                if next_char in ['n', 'r', 't', '\\', '"', '/', 'u', 'b', 'f']:
                    result.append(char)
                    result.append(next_char)
                    i += 2
                    continue
                else:
                    # 如果不是有效的转义字符（如 \mathrm），需要转义反斜杠
                    result.append('\\\\')
                    i += 1
                    continue

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

def _assemble_prompts(outcome: str) -> Dict[str, str]:
    """
    组装五个prompt基础模板

    将normal_prompt中的{signal_questions}替换为对应的信号问题，
    将{example_id_1}和{example_id_2}替换为固定的问题ID
    """
    # 首先加载通用模板
    normal_prompt = _load_prompt_template(PROMPT_NORMAL_FILE_PATH)
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
    # 预处理：清理响应文本
    # 去除BOM字符和前后空白
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
    # 匹配 ```json ... ``` 或 ``` ... ```
    markdown_patterns = [
        r'```json\s*(\[.*?\]|\{.*?\})\s*```',  # ```json ... ```
        r'```\s*(\[.*?\]|\{.*?\})\s*```',      # ``` ... ```
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

    # 方法3: 使用括号匹配算法提取[]之间的内容（数组格式）
    # 查找最外层的 [] 对，正确处理嵌套的JSON结构
    # 先尝试清理控制字符，然后再进行括号匹配
    response_text_cleaned_for_brackets = _clean_control_characters(response_text)
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
    model_name: str = "openai/gpt-5.1"
) -> Tuple[str, Dict[str, Any]]:
    """
    处理单个prompt并返回结果

    Args:
        key: prompt的key
        prompt_template: prompt模板
        context: 上下文内容
        model_name: 模型名称

    Returns:
        (key, result_dict) 元组
    """
    try:
        # 使用上下文管理器确保连接池正确释放
        with _create_llm_client(model_name) as client:
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

def rct_1_randomization_process_judgement(answers: list[dict]) -> RCTRisk:
    """
    answers: list[dict]，包含字段 id，answer_code，evidence_quote，其中id为信号问题的id，answer_code为答案代码(NA/Y/PY/PN/N/NI)，evidence_quote为证据引用。
    返回值为偏倚风险评估结果，RCTRisk枚举类型。
    """

    # 判断答案是否合法（兼容字符串和数字类型的id）
    for answer in answers:
        # 将id转换为字符串进行比较，以兼容JSON中可能存在的数字类型id
        answer_id_str = str(answer['id'])
        if answer_id_str not in ['1.1', '1.2', '1.3']:
            raise ValueError(f"Invalid answer id: {answer['id']}")
        if answer['answer_code'] not in ['NA', 'Y', 'PY', 'PN', 'N', 'NI']:
            raise ValueError(f"Invalid answer code: {answer['answer_code']}")

    # 创建答案字典，方便查找（将id统一转换为字符串作为键，以兼容数字类型id）
    answer_dict = {str(ans['id']): ans['answer_code'] for ans in answers}

    # 获取各个问题的答案
    q11 = answer_dict.get('1.1', 'NI')  # 分配序列是否随机
    q12 = answer_dict.get('1.2', 'NI')  # 分配序列是否被充分隐藏
    q13 = answer_dict.get('1.3', 'NI')  # 基线差异是否表明随机化过程存在问题

    # 辅助函数：判断答案是否为"是"（Y或PY）
    def is_yes(code):
        return code in ['Y', 'PY']

    # 辅助函数：判断答案是否为"否"（N或PN）
    def is_no(code):
        return code in ['N', 'PN']

    # 根据表格规则进行判断（按优先级：High risk > Low risk > Some concerns）
    # 表格规则：
    # 1. 1.1=Y/PY/NI AND 1.2=Y/PY AND 1.3=NI/N/PN → Low
    # 2. 1.1=Y/PY AND 1.2=Y/PY AND 1.3=Y/PY → Some concerns
    # 3. 1.1=N/PN/NI AND 1.2=Y/PY AND 1.3=Y/PY → Some concerns
    # 4. 1.1=Any AND 1.2=NI AND 1.3=N/PN/NI → Some concerns
    # 5. 1.1=Any AND 1.2=NI AND 1.3=Y/PY → High
    # 6. 1.1=Any AND 1.2=N/PN AND 1.3=Any → High

    # ========== High risk of bias 判断 ==========
    # 规则6：1.1 = Any AND 1.2 = N/PN AND 1.3 = Any
    if is_no(q12):
        return RCTRisk.HIGH_RISK

    # 规则5：1.1 = Any AND 1.2 = NI AND 1.3 = Y/PY
    if q12 == 'NI' and is_yes(q13):
        return RCTRisk.HIGH_RISK

    # ========== Low risk of bias 判断 ==========
    # 规则1：1.1 = Y/PY/NI AND 1.2 = Y/PY AND 1.3 = NI/N/PN
    # 1.1 = Y/PY/NI 表示 (is_yes(q11) or q11 == 'NI')
    # 1.3 = NI/N/PN 表示 (q13 == 'NI' or is_no(q13))
    if (is_yes(q11) or q11 == 'NI') and is_yes(q12) and (q13 == 'NI' or is_no(q13)):
        return RCTRisk.LOW_RISK

    # ========== Some concerns 判断 ==========
    # 规则2：1.1 = Y/PY AND 1.2 = Y/PY AND 1.3 = Y/PY
    if is_yes(q11) and is_yes(q12) and is_yes(q13):
        return RCTRisk.MODERATE_RISK

    # 规则3：1.1 = N/PN/NI AND 1.2 = Y/PY AND 1.3 = Y/PY
    if (is_no(q11) or q11 == 'NI') and is_yes(q12) and is_yes(q13):
        return RCTRisk.MODERATE_RISK

    # 规则4：1.1 = Any AND 1.2 = NI AND 1.3 = N/PN/NI
    if q12 == 'NI' and (is_no(q13) or q13 == 'NI'):
        return RCTRisk.MODERATE_RISK

    # 如果以上条件都不满足，默认返回 Some concerns（保守估计，避免遗漏风险）
    return RCTRisk.MODERATE_RISK

def rct_2_intended_intervention_judgement(answers: list[dict]) -> RCTRisk:
    """
    answers: list[dict]，包含字段 id，answer_code，evidence_quote，其中id为信号问题的id，answer_code为答案代码(NA/Y/PY/PN/N/NI)，evidence_quote为证据引用。
    返回值为偏倚风险评估结果，RCTRisk枚举类型。
    """

    # 判断答案是否合法（兼容字符串和数字类型的id）
    for answer in answers:
        # 将id转换为字符串进行比较，以兼容JSON中可能存在的数字类型id
        answer_id_str = str(answer['id'])
        if answer_id_str not in ['2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7']:
            raise ValueError(f"Invalid answer id: {answer['id']}")
        if answer['answer_code'] not in ['NA', 'Y', 'PY', 'PN', 'N', 'NI']:
            raise ValueError(f"Invalid answer code: {answer['answer_code']}")

    # 创建答案字典，方便查找（将id统一转换为字符串作为键，以兼容数字类型id）
    answer_dict = {str(ans['id']): ans['answer_code'] for ans in answers}

    # 获取各个问题的答案
    q21 = answer_dict.get('2.1', 'NI')  # 参与者是否知道分组
    q22 = answer_dict.get('2.2', 'NI')  # 护理者和干预实施者是否知道分组
    q23 = answer_dict.get('2.3', 'NI')  # 是否有因试验背景导致的偏离
    q24 = answer_dict.get('2.4', 'NI')  # 这些偏离是否可能影响结果
    q25 = answer_dict.get('2.5', 'NI')  # 偏离是否在组间平衡
    q26 = answer_dict.get('2.6', 'NI')  # 是否使用了适当的分析
    q27 = answer_dict.get('2.7', 'NI')  # 分析失败是否有实质性影响

    # 辅助函数：判断答案是否为"是"（Y或PY）
    def is_yes(code):
        return code in ['Y', 'PY']

    # 辅助函数：判断答案是否为"否"（N或PN）
    def is_no(code):
        return code in ['N', 'PN']

    # ========== 第一部分判断 (2.1-2.5) ==========
    # 判断2.1和2.2是否都是N/PN
    both_unaware = is_no(q21) and is_no(q22)

    # 判断2.1或2.2中至少有一个是Y/PY/NI（即不是Both N/PN的情况）
    either_aware = not both_unaware

    # 判断2.2的值
    q22_is_no = is_no(q22)
    q22_is_ni = q22 == 'NI'
    q22_is_yes = is_yes(q22)

    # 判断2.3的值
    q23_is_na = q23 == 'NA'
    q23_is_no = is_no(q23)
    q23_is_yes_or_ni = is_yes(q23) or q23 == 'NI'

    # 判断2.4的值
    q24_is_na = q24 == 'NA'
    q24_is_yes = is_yes(q24)
    q24_is_yes_or_ni = is_yes(q24) or q24 == 'NI'

    # 判断2.5的值
    q25_is_na = q25 == 'NA'
    q25_is_no_or_ni = is_no(q25) or q25 == 'NI'

    # 根据表格规则判断第一部分的风险等级（按优先级：High > Low > Some concerns）
    part1_risk = None

    # High risk: 规则6
    # Either 2.1 or 2.2 Y/PY/NI, 2.2=Y/PY, 2.3=Y/PY/NI, 2.4=Y/PY/NI, 2.5=N/PN/NI
    if (either_aware and q22_is_yes and q23_is_yes_or_ni and
        q24_is_yes_or_ni and q25_is_no_or_ni):
        part1_risk = RCTRisk.HIGH_RISK

    # Low risk: 规则1和规则2
    # 规则1: Both 2.1 & 2.2 N/PN, 2.3=NA, 2.4=NA, 2.5=NA
    elif both_unaware and q23_is_na and q24_is_na and q25_is_na:
        part1_risk = RCTRisk.LOW_RISK

    # 规则2: Either 2.1 or 2.2 Y/PY/NI, 2.2=N/PN, 2.3=NA, 2.4=NA, 2.5=NA
    elif either_aware and q22_is_no and q23_is_na and q24_is_na and q25_is_na:
        part1_risk = RCTRisk.LOW_RISK

    # Some concerns: 规则3、4、5
    # 规则3: Either 2.1 or 2.2 Y/PY/NI, 2.2=NI, 2.3=NA, 2.4=NA, 2.5=NA
    elif either_aware and q22_is_ni and q23_is_na and q24_is_na and q25_is_na:
        part1_risk = RCTRisk.MODERATE_RISK

    # 规则4: Either 2.1 or 2.2 Y/PY/NI, 2.2=Y/PY, 2.3=N/PN, 2.4=NA, 2.5=NA
    elif either_aware and q22_is_yes and q23_is_no and q24_is_na and q25_is_na:
        part1_risk = RCTRisk.MODERATE_RISK

    # 规则5: Either 2.1 or 2.2 Y/PY/NI, 2.2=Y/PY, 2.3=Y/PY/NI, 2.4=Y/PY, 2.5=NA
    elif either_aware and q22_is_yes and q23_is_yes_or_ni and q24_is_yes and q25_is_na:
        part1_risk = RCTRisk.MODERATE_RISK

    # 如果以上条件都不满足，默认返回 Some concerns
    else:
        part1_risk = RCTRisk.MODERATE_RISK

    # ========== 第二部分判断 (2.6-2.7) ==========
    # 判断2.6的值
    q26_is_yes = is_yes(q26)
    q26_is_no_or_ni = is_no(q26) or q26 == 'NI'

    # 判断2.7的值
    q27_is_na = q27 == 'NA'
    q27_is_no = is_no(q27)
    q27_is_yes_or_ni = is_yes(q27) or q27 == 'NI'

    # 根据表格规则判断第二部分的风险等级（按优先级：High > Low > Some concerns）
    part2_risk = None

    # High risk: 规则3
    # 2.6=N/PN/NI, 2.7=Y/PY/NI
    if q26_is_no_or_ni and q27_is_yes_or_ni:
        part2_risk = RCTRisk.HIGH_RISK

    # Low risk: 规则1
    # 2.6=Y/PY, 2.7=NA
    elif q26_is_yes and q27_is_na:
        part2_risk = RCTRisk.LOW_RISK

    # Some concerns: 规则2
    # 2.6=N/PN/NI, 2.7=N/PN
    elif q26_is_no_or_ni and q27_is_no:
        part2_risk = RCTRisk.MODERATE_RISK

    # 如果以上条件都不满足，默认返回 Some concerns
    else:
        part2_risk = RCTRisk.MODERATE_RISK

    # ========== 综合两部分结果 ==========
    # 规则1: 'Low' risk of bias in Part 1 AND 'Low' risk of bias in Part 2 → Low
    if part1_risk == RCTRisk.LOW_RISK and part2_risk == RCTRisk.LOW_RISK:
        return RCTRisk.LOW_RISK

    # 规则3: 'High' risk of bias in either Part 1 OR in Part 2 → High
    if part1_risk == RCTRisk.HIGH_RISK or part2_risk == RCTRisk.HIGH_RISK:
        return RCTRisk.HIGH_RISK

    # 规则2: 'Some concerns' in either Part 1 OR in Part 2, AND NOT 'High' risk in either part → Some concerns
    # （由于已经排除了High的情况，这里只需要检查是否有Some concerns）
    if part1_risk == RCTRisk.MODERATE_RISK or part2_risk == RCTRisk.MODERATE_RISK:
        return RCTRisk.MODERATE_RISK

    # 默认返回 Some concerns
    return RCTRisk.MODERATE_RISK

def rct_3_missing_outcome_judgement(answers: list[dict]) -> RCTRisk:
    """
    answers: list[dict]，包含字段 id，answer_code，evidence_quote，其中id为信号问题的id，answer_code为答案代码(NA/Y/PY/PN/N/NI)，evidence_quote为证据引用。
    返回值为偏倚风险评估结果，RCTRisk枚举类型。
    """

    # 判断答案是否合法（兼容字符串和数字类型的id）
    for answer in answers:
        # 将id转换为字符串进行比较，以兼容JSON中可能存在的数字类型id
        answer_id_str = str(answer['id'])
        if answer_id_str not in ['3.1', '3.2', '3.3', '3.4']:
            raise ValueError(f"Invalid answer id: {answer['id']}")
        if answer['answer_code'] not in ['NA', 'Y', 'PY', 'PN', 'N', 'NI']:
            raise ValueError(f"Invalid answer code: {answer['answer_code']}")

    # 创建答案字典，方便查找（将id统一转换为字符串作为键，以兼容数字类型id）
    answer_dict = {str(ans['id']): ans['answer_code'] for ans in answers}

    # 获取各个问题的答案
    q31 = answer_dict.get('3.1', 'NI')  # 数据是否可用于所有或几乎所有随机参与者
    q32 = answer_dict.get('3.2', 'NI')  # 是否有证据表明结果没有被缺失数据偏倚
    q33 = answer_dict.get('3.3', 'NI')  # 缺失是否可能依赖于真实值
    q34 = answer_dict.get('3.4', 'NI')  # 缺失是否很可能依赖于真实值

    # 辅助函数：判断答案是否为"是"（Y或PY）
    def is_yes(code):
        return code in ['Y', 'PY']

    # 辅助函数：判断答案是否为"否"（N或PN）
    def is_no(code):
        return code in ['N', 'PN']

    # 根据表格规则进行判断（按优先级：Low risk > High risk > Some concerns）
    # 表格规则：
    # 1. 3.1=Y/PY, 3.2=NA, 3.3=NA, 3.4=NA → Low
    # 2. 3.1=N/PN/NI, 3.2=Y/PY, 3.3=NA, 3.4=NA → Low
    # 3. 3.1=N/PN/NI, 3.2=N/PN, 3.3=N/PN, 3.4=NA → Low
    # 4. 3.1=N/PN/NI, 3.2=N/PN, 3.3=Y/PY/NI, 3.4=N/PN → Some concerns
    # 5. 3.1=N/PN/NI, 3.2=N/PN, 3.3=Y/PY/NI, 3.4=Y/PY/NI → High

    # 判断3.1的值
    q31_is_yes = is_yes(q31)
    q31_is_no_or_ni = is_no(q31) or q31 == 'NI'

    # 判断3.2的值
    q32_is_na = q32 == 'NA'
    q32_is_yes = is_yes(q32)
    q32_is_no = is_no(q32)

    # 判断3.3的值
    q33_is_na = q33 == 'NA'
    q33_is_no = is_no(q33)
    q33_is_yes_or_ni = is_yes(q33) or q33 == 'NI'

    # 判断3.4的值
    q34_is_na = q34 == 'NA'
    q34_is_no = is_no(q34)
    q34_is_yes_or_ni = is_yes(q34) or q34 == 'NI'

    # ========== Low risk of bias 判断 ==========
    # 规则1：3.1=Y/PY, 3.2=NA, 3.3=NA, 3.4=NA
    if q31_is_yes and q32_is_na and q33_is_na and q34_is_na:
        return RCTRisk.LOW_RISK

    # 规则2：3.1=N/PN/NI, 3.2=Y/PY, 3.3=NA, 3.4=NA
    if q31_is_no_or_ni and q32_is_yes and q33_is_na and q34_is_na:
        return RCTRisk.LOW_RISK

    # 规则3：3.1=N/PN/NI, 3.2=N/PN, 3.3=N/PN, 3.4=NA
    if q31_is_no_or_ni and q32_is_no and q33_is_no and q34_is_na:
        return RCTRisk.LOW_RISK

    # ========== High risk of bias 判断 ==========
    # 规则5：3.1=N/PN/NI, 3.2=N/PN, 3.3=Y/PY/NI, 3.4=Y/PY/NI
    if q31_is_no_or_ni and q32_is_no and q33_is_yes_or_ni and q34_is_yes_or_ni:
        return RCTRisk.HIGH_RISK

    # ========== Some concerns 判断 ==========
    # 规则4：3.1=N/PN/NI, 3.2=N/PN, 3.3=Y/PY/NI, 3.4=N/PN
    if q31_is_no_or_ni and q32_is_no and q33_is_yes_or_ni and q34_is_no:
        return RCTRisk.MODERATE_RISK

    # 如果以上条件都不满足，默认返回 Some concerns（保守估计，避免遗漏风险）
    return RCTRisk.MODERATE_RISK


def rct_4_measurement_of_the_outcome_judgement(answers: list[dict]) -> RCTRisk:
    """
    answers: list[dict]，包含字段 id，answer_code，evidence_quote，其中id为信号问题的id，answer_code为答案代码(NA/Y/PY/PN/N/NI)，evidence_quote为证据引用。
    返回值为偏倚风险评估结果，RCTRisk枚举类型。
    """

    # 判断答案是否合法（兼容字符串和数字类型的id）
    for answer in answers:
        # 将id转换为字符串进行比较，以兼容JSON中可能存在的数字类型id
        answer_id_str = str(answer['id'])
        if answer_id_str not in ['4.1', '4.2', '4.3', '4.4', '4.5']:
            raise ValueError(f"Invalid answer id: {answer['id']}")
        if answer['answer_code'] not in ['NA', 'Y', 'PY', 'PN', 'N', 'NI']:
            raise ValueError(f"Invalid answer code: {answer['answer_code']}")

    # 创建答案字典，方便查找（将id统一转换为字符串作为键，以兼容数字类型id）
    answer_dict = {str(ans['id']): ans['answer_code'] for ans in answers}

    # 获取各个问题的答案
    q41 = answer_dict.get('4.1', 'NI')  # 测量方法是否不适当
    q42 = answer_dict.get('4.2', 'NI')  # 测量或确定结果是否可能在干预组之间不同
    q43 = answer_dict.get('4.3', 'NI')  # 结果评估者是否知道参与者接受的干预
    q44 = answer_dict.get('4.4', 'NI')  # 结果评估是否可能受到对接受的干预的了解的影响
    q45 = answer_dict.get('4.5', 'NI')  # 结果评估是否很可能受到对接受的干预的了解的影响

    # 辅助函数：判断答案是否为"是"（Y或PY）
    def is_yes(code):
        return code in ['Y', 'PY']

    # 辅助函数：判断答案是否为"否"（N或PN）
    def is_no(code):
        return code in ['N', 'PN']

    # 根据表格规则进行判断（按优先级：High risk > Low risk > Some concerns）
    # 表格规则：
    # 1. 4.1=N/PN/NI, 4.2=N/PN, 4.3=N/PN, 4.4=NA, 4.5=NA → Low
    # 2. 4.1=N/PN/NI, 4.2=N/PN, 4.3=Y/PY/NI, 4.4=N/PN, 4.5=NA → Low
    # 3. 4.1=N/PN/NI, 4.2=N/PN, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=N/PN → Some concerns
    # 4. 4.1=N/PN/NI, 4.2=N/PN, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=Y/PY/NI → High
    # 5. 4.1=N/PN/NI, 4.2=NI, 4.3=N/PN, 4.4=NA, 4.5=NA → Some concerns
    # 6. 4.1=N/PN/NI, 4.2=NI, 4.3=Y/PY/NI, 4.4=N/PN, 4.5=NA → Some concerns
    # 7. 4.1=N/PN/NI, 4.2=NI, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=N/PN → Some concerns
    # 8. 4.1=N/PN/NI, 4.2=NI, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=Y/PY/NI → High
    # 9. 4.1=Y/PY, 4.2=Any, 4.3=Any, 4.4=Any, 4.5=Any → High
    # 10. 4.1=Any, 4.2=Y/PY, 4.3=Any, 4.4=Any, 4.5=Any → High

    # 判断4.1的值
    q41_is_yes = is_yes(q41)
    q41_is_no_or_ni = is_no(q41) or q41 == 'NI'

    # 判断4.2的值
    q42_is_yes = is_yes(q42)
    q42_is_no = is_no(q42)
    q42_is_ni = q42 == 'NI'

    # 判断4.3的值
    q43_is_no = is_no(q43)
    q43_is_yes_or_ni = is_yes(q43) or q43 == 'NI'

    # 判断4.4的值
    q44_is_na = q44 == 'NA'
    q44_is_no = is_no(q44)
    q44_is_yes_or_ni = is_yes(q44) or q44 == 'NI'

    # 判断4.5的值
    q45_is_na = q45 == 'NA'
    q45_is_no = is_no(q45)
    q45_is_yes_or_ni = is_yes(q45) or q45 == 'NI'

    # ========== High risk of bias 判断 ==========
    # 规则9：4.1=Y/PY, 4.2=Any, 4.3=Any, 4.4=Any, 4.5=Any
    if q41_is_yes:
        return RCTRisk.HIGH_RISK

    # 规则10：4.1=Any, 4.2=Y/PY, 4.3=Any, 4.4=Any, 4.5=Any
    if q42_is_yes:
        return RCTRisk.HIGH_RISK

    # 规则4：4.1=N/PN/NI, 4.2=N/PN, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=Y/PY/NI
    if (q41_is_no_or_ni and q42_is_no and q43_is_yes_or_ni and
        q44_is_yes_or_ni and q45_is_yes_or_ni):
        return RCTRisk.HIGH_RISK

    # 规则8：4.1=N/PN/NI, 4.2=NI, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=Y/PY/NI
    if (q41_is_no_or_ni and q42_is_ni and q43_is_yes_or_ni and
        q44_is_yes_or_ni and q45_is_yes_or_ni):
        return RCTRisk.HIGH_RISK

    # ========== Low risk of bias 判断 ==========
    # 规则1：4.1=N/PN/NI, 4.2=N/PN, 4.3=N/PN, 4.4=NA, 4.5=NA
    if (q41_is_no_or_ni and q42_is_no and q43_is_no and
        q44_is_na and q45_is_na):
        return RCTRisk.LOW_RISK

    # 规则2：4.1=N/PN/NI, 4.2=N/PN, 4.3=Y/PY/NI, 4.4=N/PN, 4.5=NA
    if (q41_is_no_or_ni and q42_is_no and q43_is_yes_or_ni and
        q44_is_no and q45_is_na):
        return RCTRisk.LOW_RISK

    # ========== Some concerns 判断 ==========
    # 规则3：4.1=N/PN/NI, 4.2=N/PN, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=N/PN
    if (q41_is_no_or_ni and q42_is_no and q43_is_yes_or_ni and
        q44_is_yes_or_ni and q45_is_no):
        return RCTRisk.MODERATE_RISK

    # 规则5：4.1=N/PN/NI, 4.2=NI, 4.3=N/PN, 4.4=NA, 4.5=NA
    if (q41_is_no_or_ni and q42_is_ni and q43_is_no and
        q44_is_na and q45_is_na):
        return RCTRisk.MODERATE_RISK

    # 规则6：4.1=N/PN/NI, 4.2=NI, 4.3=Y/PY/NI, 4.4=N/PN, 4.5=NA
    if (q41_is_no_or_ni and q42_is_ni and q43_is_yes_or_ni and
        q44_is_no and q45_is_na):
        return RCTRisk.MODERATE_RISK

    # 规则7：4.1=N/PN/NI, 4.2=NI, 4.3=Y/PY/NI, 4.4=Y/PY/NI, 4.5=N/PN
    if (q41_is_no_or_ni and q42_is_ni and q43_is_yes_or_ni and
        q44_is_yes_or_ni and q45_is_no):
        return RCTRisk.MODERATE_RISK

    # 如果以上条件都不满足，默认返回 Some concerns（保守估计，避免遗漏风险）
    return RCTRisk.MODERATE_RISK


def rct_5_selection_of_the_reported_result_judgement(answers: list[dict]) -> RCTRisk:
    """
    answers: list[dict]，包含字段 id，answer_code，evidence_quote，其中id为信号问题的id，answer_code为答案代码(NA/Y/PY/PN/N/NI)，evidence_quote为证据引用。
    返回值为偏倚风险评估结果，RCTRisk枚举类型。
    """

    # 判断答案是否合法（兼容字符串和数字类型的id）
    for answer in answers:
        # 将id转换为字符串进行比较，以兼容JSON中可能存在的数字类型id
        answer_id_str = str(answer['id'])
        if answer_id_str not in ['5.1', '5.2', '5.3']:
            raise ValueError(f"Invalid answer id: {answer['id']}")
        if answer['answer_code'] not in ['NA', 'Y', 'PY', 'PN', 'N', 'NI']:
            raise ValueError(f"Invalid answer code: {answer['answer_code']}")

    # 创建答案字典，方便查找（将id统一转换为字符串作为键，以兼容数字类型id）
    answer_dict = {str(ans['id']): ans['answer_code'] for ans in answers}

    # 获取各个问题的答案
    q51 = answer_dict.get('5.1', 'NI')  # 数据是否按照预先指定的计划分析
    q52 = answer_dict.get('5.2', 'NI')  # 结果是否可能从多个合格的结果测量中选择
    q53 = answer_dict.get('5.3', 'NI')  # 结果是否可能从多个合格的数据分析中选择

    # 辅助函数：判断答案是否为"是"（Y或PY）
    def is_yes(code):
        return code in ['Y', 'PY']

    # 辅助函数：判断答案是否为"否"（N或PN）
    def is_no(code):
        return code in ['N', 'PN']

    # 根据表格规则进行判断（按优先级：High risk > Low risk > Some concerns）
    # 表格规则：
    # 1. 5.1=Y/PY, 5.2=N/PN, 5.3=N/PN → Low
    # 2. 5.1=N/PN/NI, 5.2=N/PN, 5.3=N/PN → Some concerns
    # 3. Any answer, 5.2=N/PN, 5.3=NI → Some concerns
    # 4. Any answer, 5.2=NI, 5.3=N/PN → Some concerns
    # 5. Any answer, 5.2=NI, 5.3=NI → Some concerns
    # 6. Any answer, Either 5.2 or 5.3 Y/PY → High

    # 判断5.1的值
    q51_is_yes = is_yes(q51)
    q51_is_no_or_ni = is_no(q51) or q51 == 'NI'

    # 判断5.2的值
    q52_is_yes = is_yes(q52)
    q52_is_no = is_no(q52)
    q52_is_ni = q52 == 'NI'

    # 判断5.3的值
    q53_is_yes = is_yes(q53)
    q53_is_no = is_no(q53)
    q53_is_ni = q53 == 'NI'

    # ========== High risk of bias 判断 ==========
    # 规则6：Any answer, Either 5.2 or 5.3 Y/PY
    if q52_is_yes or q53_is_yes:
        return RCTRisk.HIGH_RISK

    # ========== Low risk of bias 判断 ==========
    # 规则1：5.1=Y/PY, 5.2=N/PN, 5.3=N/PN
    if q51_is_yes and q52_is_no and q53_is_no:
        return RCTRisk.LOW_RISK

    # ========== Some concerns 判断 ==========
    # 规则2：5.1=N/PN/NI, 5.2=N/PN, 5.3=N/PN
    if q51_is_no_or_ni and q52_is_no and q53_is_no:
        return RCTRisk.MODERATE_RISK

    # 规则3：Any answer, 5.2=N/PN, 5.3=NI
    if q52_is_no and q53_is_ni:
        return RCTRisk.MODERATE_RISK

    # 规则4：Any answer, 5.2=NI, 5.3=N/PN
    if q52_is_ni and q53_is_no:
        return RCTRisk.MODERATE_RISK

    # 规则5：Any answer, 5.2=NI, 5.3=NI
    if q52_is_ni and q53_is_ni:
        return RCTRisk.MODERATE_RISK

    # 如果以上条件都不满足，默认返回 Some concerns（保守估计，避免遗漏风险）
    return RCTRisk.MODERATE_RISK

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

def _overall_risk_assessment(rct_bias_result: dict) -> RCTRisk:
    """
    最终overall 风险评定

    根据五个domain的偏倚风险评估结果计算整体评估：
    1. 如果五个domain都是Low risk of bias，那么整体返回Low risk of bias
    2. 如果至少有一个High risk of bias，那么整体返回High risk of bias
    3. 如果至少有一个Some concerns但不存在High risk of bias，那么整体返回Some concerns

    Args:
        rct_bias_result: assess_rct_bias返回的结果字典，包含五个domain的judgement结果

    Returns:
        整体偏倚风险评估结果，RCTRisk枚举类型
    """
    # 收集所有domain的judgement结果
    judgements = []
    for domain_key in RCT_BIAS_DOMAIN_KEYS:
        domain_result = rct_bias_result.get(domain_key, {})
        judgement = domain_result.get("judgement")
        if judgement:
            # 如果judgement是字符串，转换为枚举
            if isinstance(judgement, str):
                judgement = _string_to_rct_risk(judgement)
            judgements.append(judgement)
        else:
            logger.warning(f"Domain {domain_key} 没有judgement结果，跳过该domain")
            raise ValueError(f"Domain {domain_key} 没有judgement结果，跳过该domain")

    # 如果没有有效的judgement结果，返回Some concerns（保守估计）
    if not judgements:
        logger.warning("所有domain都没有有效的judgement结果，返回Some concerns")
        raise ValueError("所有domain都没有有效的judgement结果，返回Some concerns")

    # 检查是否有High risk of bias
    has_high_risk = any(judgement == RCTRisk.HIGH_RISK for judgement in judgements)

    # 检查是否所有都是Low risk of bias
    all_low_risk = all(judgement == RCTRisk.LOW_RISK for judgement in judgements)

    # 检查是否有Some concerns
    has_some_concerns = any(judgement == RCTRisk.MODERATE_RISK for judgement in judgements)

    # 根据规则进行判断
    if has_high_risk:
        # 如果至少有一个High risk of bias，整体返回High risk of bias
        # logger.info(f"检测到High risk of bias，整体评估为High risk of bias")
        return RCTRisk.HIGH_RISK
    elif all_low_risk:
        # 如果所有都是Low risk of bias，整体返回Low risk of bias
        # logger.info(f"所有domain都是Low risk of bias，整体评估为Low risk of bias")
        return RCTRisk.LOW_RISK
    elif has_some_concerns:
        # 如果至少有一个Some concerns但不存在High risk of bias，整体返回Some concerns
        # logger.info(f"检测到Some concerns且无High risk of bias，整体评估为Some concerns")
        return RCTRisk.MODERATE_RISK
    else:
        # 默认返回Some concerns（保守估计）
        logger.warning(f"无法确定整体评估结果，默认返回Some concerns。Judgements: {judgements}")
        return RCTRisk.MODERATE_RISK

def assess_rct_bias(context: str, outcome: str, model_name: str = "openai/gpt-5.1") -> Dict[str, Any]:
    """
    RCT 偏倚风险评估（并行处理五个prompt）

    Args:
        context: 论文上下文内容
        model_name: 使用的模型名称，默认为 "openai/gpt-5.1"

    Returns:
        包含五个评估域结果的字典
    """
    # 组装所有prompt
    prompts = _assemble_prompts(outcome)

    # 使用线程池并行处理
    results = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有任务
        future_to_key = {
            executor.submit(
                _process_single_prompt,
                key,
                prompt_template,
                context,
                model_name
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

    # 根据不同的domain调用相应的判断函数
    domain_judgement_map = {
        RCT_BIAS_DOMAIN_KEYS[0]: rct_1_randomization_process_judgement,
        RCT_BIAS_DOMAIN_KEYS[1]: rct_2_intended_intervention_judgement,
        RCT_BIAS_DOMAIN_KEYS[2]: rct_3_missing_outcome_judgement,
        RCT_BIAS_DOMAIN_KEYS[3]: rct_4_measurement_of_the_outcome_judgement,
        RCT_BIAS_DOMAIN_KEYS[4]: rct_5_selection_of_the_reported_result_judgement,
    }

    # 为每个domain计算偏倚风险评估结果
    for domain_key, result in results.items():
        if result.get("success_parse") and result.get("response"):
            try:
                # 提取answers列表
                answers = result.get("response")

                # 确保answers是列表格式
                if isinstance(answers, list):
                    # 调用对应的判断函数
                    judgement_func = domain_judgement_map.get(domain_key)
                    if judgement_func:
                        judgement_result = judgement_func(answers)
                        result["judgement"] = judgement_result
                        logger.info(f"{domain_key} 偏倚风险评估结果: {judgement_result.value}")
                    else:
                        logger.warning(f"未找到 {domain_key} 对应的判断函数")
                        result["judgement"] = None
                else:
                    logger.warning(f"{domain_key} 的响应不是列表格式，无法进行判断")
                    result["judgement"] = None
            except Exception as e:
                logger.error(f"计算 {domain_key} 偏倚风险评估结果时发生错误: {e}")
                result["judgement"] = None
                result["judgement_error"] = str(e)
        else:
            logger.warning(f"{domain_key} 的响应解析失败，无法进行偏倚风险评估")
            result["judgement"] = None

    # 评估 overall risk
    overall_risk = _overall_risk_assessment(results)
    results["overall_risk_judgement"] = overall_risk

    return results
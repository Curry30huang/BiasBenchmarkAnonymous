"""
使用 openai 标准调用 API 接口
兼容大部分模型的OpenAI格式接口，支持自定义URL和模型名称
"""

import os
import json
import time
import requests
from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class MessageRole(Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """消息数据类"""
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatCompletionRequest:
    """聊天完成请求数据类"""
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": self.stream
        }

        # 只添加非None的字段
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.stop is not None:
            data["stop"] = self.stop

        return data


@dataclass
class ChatCompletionResponse:
    """聊天完成响应数据类"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
    system_fingerprint: Optional[str] = None  # 阿里百炼等平台可能返回的额外字段
    provider: Optional[str] = None  # OpenRouter等平台可能返回的额外字段

    def get_content(self) -> str:
        """获取响应内容"""
        if self.choices and len(self.choices) > 0:
            return self.choices[0]["message"]["content"]
        return ""


class OpenAICompatibleClient:
    """OpenAI兼容的API客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        pool_connections: int = 10,
        pool_maxsize: int = 10
    ):
        """
        初始化OpenAI兼容客户端

        Args:
            api_key: API密钥，如果为None则从环境变量OPENAI_API_KEY获取
            base_url: API基础URL，如果为None则从环境变量OPENAI_BASE_URL获取
            model: 默认模型名称，如果为None则从环境变量OPENAI_MODEL获取
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟时间（秒）
            pool_connections: 连接池中缓存的连接数（每个主机）
            pool_maxsize: 连接池的最大连接数（每个主机）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError("API密钥未提供，请设置api_key参数或OPENAI_API_KEY环境变量")

        # 确保base_url以/结尾
        if not self.base_url.endswith("/"):
            self.base_url += "/"

        # 创建带连接池的Session
        # 使用HTTPAdapter配置连接池参数
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        # 配置HTTP适配器，包含连接池设置
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )

        # 为HTTP和HTTPS协议注册适配器
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(
            f"初始化OpenAI兼容客户端: {self.base_url}, 模型: {self.model}, "
            f"连接池: {pool_connections}/{pool_maxsize}"
        )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[requests.Response, Iterator[str]]:
        """
        发送HTTP请求

        Args:
            method: HTTP方法
            endpoint: API端点
            data: 请求数据
            stream: 是否流式响应

        Returns:
            HTTP响应或流式响应迭代器
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    # 使用session.get()以利用连接池
                    response = self.session.get(url, headers=headers, timeout=self.timeout)
                elif method.upper() == "POST":
                    # 使用session.post()以利用连接池
                    response = self.session.post(
                        url,
                        headers=headers,
                        json=data,
                        timeout=self.timeout,
                        stream=stream
                    )
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")

                response.raise_for_status()

                if stream:
                    return self._handle_stream_response(response)
                else:
                    return response

            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                else:
                    raise e

        # 这行代码不应该被执行到，但为了类型检查
        raise RuntimeError("请求失败")

    def _handle_stream_response(self, response: requests.Response) -> Iterator[str]:
        """处理流式响应"""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 移除 'data: ' 前缀
                    if data.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except json.JSONDecodeError:
                        continue

    def chat_completion(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = 1.0,
        frequency_penalty: Optional[float] = 0.0,
        presence_penalty: Optional[float] = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Union[ChatCompletionResponse, Iterator[str]]:
        """
        发送聊天完成请求

        Args:
            messages: 消息列表
            model: 模型名称，如果为None则使用默认模型
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p参数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            stop: 停止词
            stream: 是否流式响应

        Returns:
            聊天完成响应或流式响应迭代器
        """
        model = model or self.model

        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream
        )

        logger.info(f"发送聊天完成请求: 模型={model}, 消息数={len(messages)}, 流式={stream}")

        response = self._make_request("POST", "chat/completions", request.to_dict(), stream=stream)

        if stream:
            return response  # type: ignore
        else:
            # 确保response是requests.Response类型
            if isinstance(response, requests.Response):
                response_data = response.json()
                # 过滤掉ChatCompletionResponse不支持的字段
                filtered_data = {
                    'id': response_data.get('id', ''),
                    'object': response_data.get('object', ''),
                    'created': response_data.get('created', 0),
                    'model': response_data.get('model', ''),
                    'choices': response_data.get('choices', []),
                    'usage': response_data.get('usage'),
                    'system_fingerprint': response_data.get('system_fingerprint'),
                    'provider': response_data.get('provider')
                }
                return ChatCompletionResponse(**filtered_data)
            else:
                raise RuntimeError("非流式响应应该是requests.Response类型")

    def simple_chat(
        self,
        message: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        简单的聊天接口

        Args:
            message: 用户消息
            system_message: 系统消息
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            响应内容
        """
        messages = []

        if system_message:
            messages.append(Message(role=MessageRole.SYSTEM.value, content=system_message))

        messages.append(Message(role=MessageRole.USER.value, content=message))

        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # 确保response是ChatCompletionResponse类型
        if isinstance(response, ChatCompletionResponse):
            return response.get_content()
        else:
            raise RuntimeError("simple_chat应该返回ChatCompletionResponse类型")

    def stream_chat(
        self,
        message: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        流式聊天接口

        Args:
            message: 用户消息
            system_message: 系统消息
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            流式响应迭代器
        """
        messages = []

        if system_message:
            messages.append(Message(role=MessageRole.SYSTEM.value, content=system_message))

        messages.append(Message(role=MessageRole.USER.value, content=message))

        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        # 确保response是Iterator[str]类型
        if hasattr(response, '__iter__') and not isinstance(response, ChatCompletionResponse):
            return response  # type: ignore
        else:
            raise RuntimeError("stream_chat应该返回Iterator[str]类型")

    def get_models(self) -> List[Dict[str, Any]]:
        """
        获取可用模型列表

        Returns:
            模型列表
        """
        try:
            response = self._make_request("GET", "models")
            # 确保response是requests.Response类型
            if isinstance(response, requests.Response):
                data = response.json()
                return data.get("data", [])
            else:
                raise RuntimeError("get_models应该返回requests.Response类型")
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    def close(self):
        """
        关闭Session，释放连接池资源

        建议在不再使用客户端时调用此方法，特别是在长时间运行的程序中
        """
        if hasattr(self, 'session') and self.session:
            self.session.close()
            logger.info("已关闭HTTP连接池")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动关闭连接池"""
        self.close()
        return False


# 便捷函数
def create_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> OpenAICompatibleClient:
    """
    创建OpenAI兼容客户端的便捷函数

    Args:
        api_key: API密钥
        base_url: API基础URL
        model: 默认模型名称
        **kwargs: 其他参数

    Returns:
        OpenAI兼容客户端实例
    """
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        **kwargs
    )


def quick_chat(
    message: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    system_message: Optional[str] = None,
    **kwargs
) -> str:
    """
    快速聊天接口

    Args:
        message: 用户消息
        api_key: API密钥
        base_url: API基础URL
        model: 模型名称
        system_message: 系统消息
        **kwargs: 其他参数

    Returns:
        响应内容
    """
    client = create_client(api_key=api_key, base_url=base_url, model=model)
    return client.simple_chat(
        message=message,
        system_message=system_message,
        **kwargs
    )


if __name__ == "__main__":
    # 使用示例
    try:
        # 方式1：使用上下文管理器（推荐，自动管理连接池）
        with create_client(
            base_url="https://api.openai.com/v1",  # 或其他兼容的API地址
            model="gpt-3.5-turbo",
            pool_connections=10,  # 连接池配置
            pool_maxsize=20       # 最大连接数
        ) as client:
            # 简单聊天
            response = client.simple_chat("你好，请介绍一下你自己")
            print(f"响应: {response}")

            # 带系统消息的聊天
            response = client.simple_chat(
                message="请帮我分析这段医学文献",
                system_message="你是一个专业的医学文献分析专家"
            )
            print(f"分析结果: {response}")

            # 流式聊天
            print("流式响应:")
            for chunk in client.stream_chat("请写一首关于春天的诗"):
                print(chunk, end="", flush=True)
            print()
        # 上下文管理器会自动调用 close() 关闭连接池

        # 方式2：手动管理（适用于长时间运行的程序）
        client = create_client(
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo"
        )
        try:
            # 多次请求会复用连接池中的连接，提升性能
            for i in range(5):
                response = client.simple_chat(f"这是第 {i+1} 次请求")
                print(f"响应 {i+1}: {response[:50]}...")
        finally:
            client.close()  # 手动关闭连接池

    except Exception as e:
        logger.error(f"示例运行失败: {e}")
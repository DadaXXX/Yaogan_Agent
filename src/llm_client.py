"""统一的云端 LLM 客户端，支持 DeepSeek / OpenAI 等兼容接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求"""
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """LLM 聊天响应的统一封装"""
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def chat(self, messages: list, tools: Optional[list] = None) -> ChatResponse:
        """调用 LLM，支持 function calling"""
        ...


class DeepSeekClient(LLMClient):
    """DeepSeek API 客户端（兼容 OpenAI 格式）"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    def chat(self, messages: list, tools: Optional[list] = None) -> ChatResponse:
        kwargs = dict(model=self.model, messages=messages, timeout=60.0)
        if tools:
            kwargs["tools"] = tools

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ChatResponse(content=msg.content, tool_calls=tool_calls)


class OpenAIClient(LLMClient):
    """OpenAI API 客户端"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    def chat(self, messages: list, tools: Optional[list] = None) -> ChatResponse:
        kwargs = dict(model=self.model, messages=messages, timeout=60.0)
        if tools:
            kwargs["tools"] = tools

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ChatResponse(content=msg.content, tool_calls=tool_calls)


def create_llm_client(provider: str, api_key: str, model: str, base_url: Optional[str] = None) -> LLMClient:
    """工厂方法：根据 provider 创建对应的 LLM 客户端"""
    if provider == "deepseek":
        return DeepSeekClient(api_key=api_key, model=model, base_url=base_url or "https://api.deepseek.com")
    elif provider == "openai":
        return OpenAIClient(api_key=api_key, model=model, base_url=base_url)
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}，可选: deepseek, openai")

"""工具注册中心 — 管理工具定义、Schema 生成和调度执行。"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """单个工具的定义"""
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]


class ToolRegistry:
    """工具注册中心，负责注册、生成 schemas 和分发执行。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已注册")
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        """返回 OpenAI/DeepSeek 格式的 function calling tools 列表"""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return schemas

    def execute(self, tool_name: str, arguments: dict) -> str:
        """执行工具并返回结果文本"""
        tool = self._tools.get(tool_name)
        if not tool:
            return f"错误：未知工具 '{tool_name}'"
        try:
            result = tool.fn(**arguments)
            return str(result)
        except Exception as e:
            return f"工具 '{tool_name}' 执行失败: {e}"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

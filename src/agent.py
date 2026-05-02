"""Agent 核心循环 — 协调 LLM 与工具的交互。"""

from src.config import Config
from src.conversation import ConversationManager
from src.llm_client import LLMClient
from src.tools.registry import ToolRegistry


class Agent:
    """遥感 Agent —— 接收用户问题，自主决策调用工具，生成分析回答。"""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        conversation: ConversationManager,
        config: Config,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.conversation = conversation
        self.max_tool_rounds = config.max_tool_rounds

    def run(self, user_input: str) -> str:
        """主循环：接收用户输入，返回最终回答。"""
        self.conversation.add_user_message(user_input)

        for turn in range(self.max_tool_rounds):
            messages = self.conversation.get_messages()
            tool_schemas = self.tools.get_schemas()
            use_tools = tool_schemas if turn < self.max_tool_rounds - 1 else None

            response = self.llm.chat(messages, tools=use_tools)

            if not response.tool_calls:
                # LLM 返回文本 → 最终回答
                self.conversation.add_assistant_message(response.content or "")
                return response.content or ""

            # LLM 请求调用工具
            self.conversation.add_assistant_tool_call(
                content=response.content or "",
                tool_calls=response.tool_calls,
            )

            for tc in response.tool_calls:
                result = self.tools.execute(tc.name, tc.arguments)
                self.conversation.add_tool_result(tc.id, result)

        # 超过最大轮数，强制要求 LLM 总结
        final_response = self.llm.chat(
            self.conversation.get_messages(),
            tools=None,
        )
        answer = final_response.content or "已达到最大工具调用轮数，请简化问题或分步询问。"
        self.conversation.add_assistant_message(answer)
        return answer

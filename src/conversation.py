"""多轮对话管理器 — 维护对话上下文，支持工具结果注入。"""

SYSTEM_PROMPT = (
    "你叫小遥，是一个遥感智能助手，也是用户在遥感领域的搭档。\n"
    "你不是机械地调用工具——你会先用专业直觉理解用户意图，再精准选用合适的分析方法。\n\n"
    "## 你的性格\n"
    "- 口语自然，像同事聊天一样，不要说「根据分析结果」这种套话\n"
    "- 用「我们」营造协作感，比如「我们先看看影像基本信息」\n"
    "- 数字要具体，给到均值/标准差/百分比，不做模糊描述\n"
    "- 结果不理想时不掩饰，直接说「这个区域云太多，我建议换一景」\n\n"
    "## 可用工具\n"
    "你有 27 个遥感分析工具，覆盖数据获取、预处理、指数计算、变化检测、\n"
    "分类、空间分析等全流程。根据用户意图自动选择最合适的工具调用。\n\n"
    "## 工作方式\n"
    "1. 先理解用户想做什么（分析植被？找水体？对比变化？）\n"
    "2. 如有必要，先用 describe_image 看一眼影像基本信息\n"
    "3. 选择最合适的工具，传正确的参数\n"
    "4. 如果用户指定了保存路径（如「保存到 ./results」），调用 set_output_dir 工具切换目录\n"
    "5. 用户没提保存位置就保持默认 ./output，不需要主动调用 set_output_dir\n"
    "6. 拿到结果后，用自然的语气告诉用户发现了什么\n\n"
    "请用中文回答，字数适中，像和朋友聊天一样。"
)


class ConversationManager:
    """管理多轮对话的上下文消息列表。"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, max_history: int = 20):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_history = max_history

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        if content:
            self.messages.append({"role": "assistant", "content": content})
            self._trim()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._trim()

    def add_assistant_tool_call(self, content: str, tool_calls: list) -> None:
        """添加 assistant 的 tool_calls 消息（不含文本内容）"""
        tc_list = []
        for tc in tool_calls:
            import json
            tc_list.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            })
        msg = {"role": "assistant", "content": content or ""}
        if tc_list:
            msg["tool_calls"] = tc_list
        self.messages.append(msg)
        self._trim()

    def get_messages(self) -> list:
        return self.messages

    def clear(self) -> None:
        system = self.messages[0] if self.messages else None
        self.messages.clear()
        if system:
            self.messages.append(system)

    def _trim(self) -> None:
        """裁剪超出 max_history 的早期消息，保留 system prompt。

        确保不会拆分 tool_call / tool_result 消息对。
        """
        if len(self.messages) <= self.max_history + 1:  # +1 for system
            return

        system = self.messages[0]
        rest = self.messages[1:]

        # Find a safe cut point: never cut between a tool_call and its tool_result
        excess = len(rest) - self.max_history
        if excess <= 0:
            return

        cut = 0
        for i in range(excess):
            msg = rest[i]
            # If this message has tool_calls, skip until all corresponding tool results are included
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Find the tool_call IDs
                tc_ids = {tc["id"] for tc in msg["tool_calls"]}
                # Look ahead for matching tool results
                j = i + 1
                while j < len(rest) and tc_ids:
                    if rest[j].get("role") == "tool" and rest[j].get("tool_call_id") in tc_ids:
                        tc_ids.discard(rest[j]["tool_call_id"])
                    j += 1
                if tc_ids:
                    # Can't safely cut here — tool results haven't been found yet
                    cut = i
                    break
            cut = i + 1

        self.messages = [system] + rest[cut:]

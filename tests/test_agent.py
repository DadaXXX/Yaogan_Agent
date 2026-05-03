"""Tests for src/agent.py — agent loop with mocked LLM."""

from unittest.mock import MagicMock, patch

from src.agent import Agent
from src.config import Config
from src.conversation import ConversationManager
from src.tools.registry import Tool, ToolRegistry


def _make_mock_llm(responses):
    """Create a mock LLM client that returns responses in sequence."""
    from src.llm_client import ChatResponse, ToolCall

    llm = MagicMock()
    call_count = {"n": 0}

    def chat_fn(messages, tools=None):
        resp = responses[call_count["n"]]
        call_count["n"] += 1
        return resp

    llm.chat = chat_fn
    return llm


def test_agent_returns_text_response():
    """Agent returns LLM text when no tool calls are made."""
    from src.llm_client import ChatResponse

    llm = _make_mock_llm([
        ChatResponse(content="Hello!", tool_calls=[]),
    ])
    registry = ToolRegistry()
    conv = ConversationManager()
    config = Config(api_key="test", max_tool_rounds=3)

    agent = Agent(llm, registry, conv, config)
    result = agent.run("Hi")
    assert result == "Hello!"


def test_agent_executes_tool_then_responds():
    """Agent executes a tool call, then returns the final response."""
    from src.llm_client import ChatResponse, ToolCall

    registry = ToolRegistry()
    registry.register(Tool(
        name="echo",
        description="echo",
        parameters={},
        fn=lambda msg="": f"echoed: {msg}",
    ))

    llm = _make_mock_llm([
        ChatResponse(content="", tool_calls=[
            ToolCall(id="tc1", name="echo", arguments={"msg": "hi"}),
        ]),
        ChatResponse(content="Tool said: echoed: hi", tool_calls=[]),
    ])

    conv = ConversationManager()
    config = Config(api_key="test", max_tool_rounds=3)
    agent = Agent(llm, registry, conv, config)
    result = agent.run("echo hi")
    assert "echoed: hi" in result


def test_agent_max_tool_rounds():
    """Agent stops after max_tool_rounds and returns fallback."""
    from src.llm_client import ChatResponse, ToolCall

    registry = ToolRegistry()
    registry.register(Tool(
        name="noop",
        description="noop",
        parameters={},
        fn=lambda: "ok",
    ))

    # Always return a tool call — agent should stop after max rounds
    def always_tool(messages, tools=None):
        if tools:
            return ChatResponse(content="", tool_calls=[
                ToolCall(id="tc1", name="noop", arguments={}),
            ])
        return ChatResponse(content="Final answer after max rounds", tool_calls=[])

    llm = MagicMock()
    llm.chat = always_tool

    conv = ConversationManager()
    config = Config(api_key="test", max_tool_rounds=2)
    agent = Agent(llm, registry, conv, config)
    result = agent.run("do something")
    assert "Final answer" in result or "最大工具调用轮数" in result

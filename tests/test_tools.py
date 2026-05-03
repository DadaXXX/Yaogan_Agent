"""Tests for tools — registry, safe expression evaluator, path traversal."""

import ast
import numpy as np
import pytest

from src.tools.registry import Tool, ToolRegistry
from src.tools.remote_sensing import _safe_eval
from src.tools._utils import safe_path


# ── ToolRegistry ─────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            fn=lambda: "ok",
        ))
        assert "test_tool" in reg.list_tools()

    def test_duplicate_register_raises(self):
        reg = ToolRegistry()
        tool = Tool(name="dup", description="", parameters={}, fn=lambda: "ok")
        reg.register(tool)
        with pytest.raises(ValueError, match="已注册"):
            reg.register(tool)

    def test_execute_calls_fn(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="add",
            description="add",
            parameters={},
            fn=lambda a=0, b=0: str(a + b),
        ))
        result = reg.execute("add", {"a": 3, "b": 4})
        assert result == "7"

    def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.execute("nonexistent", {})
        assert "未知工具" in result

    def test_execute_handles_exception(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="fail",
            description="fails",
            parameters={},
            fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        ))
        result = reg.execute("fail", {})
        assert "执行失败" in result

    def test_get_schemas(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="echo",
            description="echo input",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            fn=lambda msg="": msg,
        ))
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"


# ── Safe expression evaluator ────────────────────────────

class TestSafeEval:
    def _eval(self, expr: str, variables: dict = None):
        tree = ast.parse(expr, mode="eval")
        return _safe_eval(tree, variables or {})

    def test_basic_arithmetic(self):
        assert self._eval("1 + 2") == 3
        assert self._eval("10 - 3") == 7
        assert self._eval("4 * 5") == 20
        assert self._eval("10 / 4") == 2.5

    def test_band_variables(self):
        b1 = np.array([1.0, 2.0, 3.0])
        b2 = np.array([4.0, 5.0, 6.0])
        result = self._eval("(b2 - b1) / (b2 + b1)", {"b1": b1, "b2": b2})
        expected = (b2 - b1) / (b2 + b1)
        np.testing.assert_allclose(result, expected)

    def test_safe_functions(self):
        b1 = np.array([1.0, 4.0, 9.0])
        result = self._eval("sqrt(b1)", {"b1": b1})
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_rejects_import(self):
        with pytest.raises(ValueError, match="不支持"):
            self._eval("__import__('os').system('id')")

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError):
            self._eval("b1.__class__", {"b1": np.array([1.0])})

    def test_rejects_function_def(self):
        with pytest.raises(ValueError):
            self._eval("lambda: 1")

    def test_rejects_dunder_methods(self):
        with pytest.raises(ValueError):
            self._eval("b1.__array__.__class__.__bases__[0].__subclasses__()",
                       {"b1": np.array([1.0])})

    def test_unknown_variable(self):
        with pytest.raises(ValueError, match="未知变量"):
            self._eval("x + 1", {})

    def test_unsupported_function(self):
        with pytest.raises(ValueError, match="不支持的函数"):
            self._eval("exec('print(1)')")


# ── Path traversal prevention ────────────────────────────

class TestSafePath:
    def test_allows_cwd_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "test.tif"
        f.touch()
        result = safe_path("test.tif", str(tmp_path / "output"))
        assert result == f.resolve()

    def test_allows_output_dir(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        f = out / "result.tif"
        f.touch()
        result = safe_path(str(f), str(out))
        assert result == f.resolve()

    def test_rejects_traversal(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        with pytest.raises(ValueError, match="不在允许的目录内"):
            safe_path("../../etc/passwd", str(out))

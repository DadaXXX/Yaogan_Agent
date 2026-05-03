"""Tests for src/config.py — configuration loading and validation."""

import os
import pytest


def test_load_config_with_api_key(monkeypatch):
    """Config loads successfully when RS_API_KEY is set."""
    monkeypatch.setenv("RS_API_KEY", "test-key-123")
    monkeypatch.setenv("RS_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("RS_LLM_MODEL", "deepseek-chat")

    # Re-import to pick up env changes
    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import load_config

    config = load_config()
    assert config.api_key == "test-key-123"
    assert config.llm_provider == "deepseek"
    assert config.llm_model == "deepseek-chat"


def test_load_config_missing_api_key_exits(monkeypatch):
    """Config raises SystemExit when RS_API_KEY is empty."""
    monkeypatch.delenv("RS_API_KEY", raising=False)

    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import load_config

    with pytest.raises(SystemExit):
        load_config()


def test_config_default_values(monkeypatch):
    """Config has sensible defaults for optional fields."""
    monkeypatch.setenv("RS_API_KEY", "test-key")

    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import load_config

    config = load_config()
    assert config.max_tool_rounds == 5
    assert config.output_dir == "./output"
    assert config.sam_model_type == "vit_b"


def test_config_server_api_key(monkeypatch):
    """Server API key loads from env."""
    monkeypatch.setenv("RS_API_KEY", "test-key")
    monkeypatch.setenv("RS_SERVER_API_KEY", "server-secret")

    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import load_config

    config = load_config()
    assert config.server_api_key == "server-secret"

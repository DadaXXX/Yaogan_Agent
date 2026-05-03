"""Tests for app.py — FastAPI endpoints, auth, session management."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked config."""
    monkeypatch.setenv("RS_API_KEY", "test-key")
    monkeypatch.setenv("RS_SERVER_API_KEY", "")

    import importlib
    import src.config
    importlib.reload(src.config)

    # Patch load_config before importing app
    with patch("src.config.load_config") as mock_load:
        from src.config import Config
        mock_load.return_value = Config(
            api_key="test-key",
            server_api_key="",
            output_dir="/tmp/rs-test-output",
        )
        import app as app_module
        importlib.reload(app_module)

        from fastapi.testclient import TestClient
        yield TestClient(app_module.app)


@pytest.fixture
def auth_client(monkeypatch):
    """Create a test client with API key auth enabled."""
    monkeypatch.setenv("RS_API_KEY", "test-key")
    monkeypatch.setenv("RS_SERVER_API_KEY", "secret-key")

    import importlib
    import src.config
    importlib.reload(src.config)

    with patch("src.config.load_config") as mock_load:
        from src.config import Config
        mock_load.return_value = Config(
            api_key="test-key",
            server_api_key="secret-key",
            output_dir="/tmp/rs-test-output",
        )
        import app as app_module
        importlib.reload(app_module)

        from fastapi.testclient import TestClient
        yield TestClient(app_module.app)


def test_health_endpoint(client):
    """Health check returns ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "sessions" in data


def test_tools_endpoint(client):
    """Tools endpoint returns tool list."""
    resp = client.get("/api/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert len(data["tools"]) > 0


def test_root_serves_html(client):
    """Root endpoint serves the web UI."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "小遥" in resp.text


def test_auth_blocks_without_key(auth_client):
    """API returns 401 when server API key is set and request has no key."""
    resp = auth_client.post("/api/chat", json={"message": "test", "session_id": "s1"})
    assert resp.status_code == 401


def test_auth_allows_with_key(auth_client):
    """API allows request with correct X-API-Key header."""
    with patch("app._create_toolkits") as mock_toolkits, \
         patch("app.llm") as mock_llm:
        from src.llm_client import ChatResponse
        from src.tools.registry import ToolRegistry

        reg = ToolRegistry()
        mock_toolkits.return_value = reg
        mock_llm.chat = MagicMock(return_value=ChatResponse(content="ok", tool_calls=[]))

        resp = auth_client.post(
            "/api/chat",
            json={"message": "test", "session_id": "s1"},
            headers={"X-API-Key": "secret-key"},
        )
        assert resp.status_code == 200


def test_auth_health_skipped(auth_client):
    """Health endpoint skips auth."""
    resp = auth_client.get("/api/health")
    assert resp.status_code == 200


def test_session_eviction(monkeypatch):
    """Sessions are evicted after TTL."""
    import app as app_module
    import time

    # Clear sessions
    app_module._sessions.clear()

    # Create a session
    conv = app_module._get_session("test-session")
    assert "test-session" in app_module._sessions

    # Simulate time passing
    entry = app_module._sessions["test-session"]
    entry.last_access = time.time() - app_module.SESSION_TTL - 1

    # Next access should evict and create new
    conv2 = app_module._get_session("another-session")
    assert "test-session" not in app_module._sessions
    assert "another-session" in app_module._sessions

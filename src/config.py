"""全局配置管理 — 从环境变量加载 API Key 等敏感信息。"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Config:
    # ── LLM 配置 ──
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    api_key: str = ""
    api_base_url: str = "https://api.deepseek.com"

    # ── Agent 配置 ──
    max_tool_rounds: int = 5

    # ── SAM 配置 ──
    sam_model_type: str = "vit_b"
    sam_cache_dir: str = ""

    # ── 输出保存配置 ──
    output_dir: str = "./output"

    # ── 服务端配置 ──
    server_api_key: str = ""

    # ── GEE 配置 ──
    gee_project_id: str = ""


def load_config() -> Config:
    """从环境变量加载配置，API Key 为空时打印错误并退出"""
    config = Config(
        llm_provider=os.getenv("RS_LLM_PROVIDER", "deepseek"),
        llm_model=os.getenv("RS_LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("RS_API_KEY", ""),
        api_base_url=os.getenv("RS_API_BASE_URL", "https://api.deepseek.com"),
        max_tool_rounds=int(os.getenv("RS_MAX_TOOL_ROUNDS", "5")),
        sam_model_type=os.getenv("RS_SAM_MODEL", "vit_b"),
        sam_cache_dir=os.getenv("RS_SAM_CACHE_DIR", ""),
        output_dir=os.getenv("RS_OUTPUT_DIR", "./output"),
        server_api_key=os.getenv("RS_SERVER_API_KEY", ""),
        gee_project_id=os.getenv("RS_GEE_PROJECT_ID", ""),
    )
    if not config.api_key:
        print(
            "错误: 未设置 API Key。\n"
            "请设置环境变量 RS_API_KEY，或在 .env 文件中配置:\n"
            "  RS_API_KEY=your_api_key_here\n"
            "参见 .env.example 了解所有可用配置。"
        )
        sys.exit(1)
    return config

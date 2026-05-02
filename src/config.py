"""全局配置管理 — 从环境变量加载 API Key 等敏感信息。"""

import os
from dataclasses import dataclass, field
from pathlib import Path


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


def load_config() -> Config:
    """从环境变量加载配置，未设置时用默认值"""
    return Config(
        llm_provider=os.getenv("RS_LLM_PROVIDER", "deepseek"),
        llm_model=os.getenv("RS_LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("RS_API_KEY", "sk-1cf3f74c91e84111976ddfa4e0355220"),
        api_base_url=os.getenv("RS_API_BASE_URL", "https://api.deepseek.com"),
        max_tool_rounds=int(os.getenv("RS_MAX_TOOL_ROUNDS", "5")),
        sam_model_type=os.getenv("RS_SAM_MODEL", "vit_b"),
        sam_cache_dir=os.getenv("RS_SAM_CACHE_DIR", ""),
        output_dir=os.getenv("RS_OUTPUT_DIR", "./output"),
    )

"""遥感分析助手 — FastAPI 云端服务入口。

启动: uvicorn app:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import load_config
from src.agent import Agent
from src.conversation import ConversationManager
from src.llm_client import create_llm_client
from src.tools.remote_sensing import RemoteSensingToolkit
from src.tools.preprocessing import PreprocessingToolkit
from src.tools.analysis import AnalysisToolkit
from src.tools.classification import ClassificationToolkit
from src.tools.geo_utils import GeoToolkit
from src.tools.gee_client import GEEClient
from main import build_registry, ensure_output_dir_tool

app = FastAPI(title="遥感分析助手", version="3.0")

# 全局单例
config = load_config()
toolkit = RemoteSensingToolkit(output_dir=config.output_dir)
geo = GeoToolkit(output_dir=config.output_dir)
pre = PreprocessingToolkit(output_dir=config.output_dir)
analysis = AnalysisToolkit(output_dir=config.output_dir)
classify = ClassificationToolkit(output_dir=config.output_dir)
gee = GEEClient(download_dir=config.output_dir)
_all_modules = [toolkit, geo, pre, analysis, classify, gee]

def _set_output_dir(path: str) -> str:
    from pathlib import Path
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for m in _all_modules:
        m.output_dir = str(p)
    return f"输出目录已切换至: {p}"

registry = build_registry(toolkit, geo, pre, analysis, classify, gee)
ensure_output_dir_tool(registry, _set_output_dir)
llm = create_llm_client(config.llm_provider, config.api_key, config.llm_model, config.api_base_url)

# 每个 session 独立的对话管理器
conversations: dict[str, ConversationManager] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    if session_id not in conversations:
        conversations[session_id] = ConversationManager()
    conv = conversations[session_id]
    agent = Agent(llm, registry, conv, config)
    reply = agent.run(req.message)
    return ChatResponse(reply=reply)


@app.get("/api/tools")
async def list_tools():
    return {"tools": registry.list_tools()}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    web_dir = Path(__file__).parent / "web"
    return FileResponse(web_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

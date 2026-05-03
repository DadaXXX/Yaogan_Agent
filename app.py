"""遥感分析助手 — FastAPI 云端服务入口。

启动: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

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

app = FastAPI(title="遥感分析助手", version="3.1")

config = load_config()

# ── CORS (only after auth is in place) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth middleware ──────────────────────────────────────

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API key auth for external callers. Web UI (same-origin) is exempt."""
    if config.server_api_key:
        path = request.url.path

        # Always allow health check, static files, and web UI page
        if path in ("/api/health", "/") or path.startswith("/static") or path.startswith("/output"):
            return await call_next(request)

        # Allow same-origin requests (browser web UI) — they send a Referer header
        referer = request.headers.get("referer", "")
        host = request.headers.get("host", "")
        if host and host in referer:
            return await call_next(request)

        # External API callers must provide X-API-Key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key != config.server_api_key:
            raise HTTPException(status_code=401, detail="无效或缺失的 API Key")
    return await call_next(request)


# ── Session management with LRU eviction ────────────────

MAX_SESSIONS = 100
SESSION_TTL = 3600  # 1 hour


class SessionEntry:
    __slots__ = ("conversation", "last_access")

    def __init__(self, conversation: ConversationManager):
        self.conversation = conversation
        self.last_access = time.time()


_sessions: OrderedDict[str, SessionEntry] = OrderedDict()


def _get_session(session_id: str) -> ConversationManager:
    """Get or create a conversation session with LRU eviction."""
    _evict_expired()

    if session_id in _sessions:
        entry = _sessions[session_id]
        entry.last_access = time.time()
        _sessions.move_to_end(session_id)
        return entry.conversation

    # Evict oldest if at capacity
    while len(_sessions) >= MAX_SESSIONS:
        _sessions.popitem(last=False)

    conv = ConversationManager()
    _sessions[session_id] = SessionEntry(conv)
    return conv


def _evict_expired():
    """Remove sessions idle for more than SESSION_TTL seconds."""
    now = time.time()
    expired = [sid for sid, entry in _sessions.items()
               if now - entry.last_access > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]


# ── Toolkit factory (per-request isolation) ─────────────

def _create_toolkits(output_dir: str):
    """Create fresh toolkit instances for a single request."""
    toolkit = RemoteSensingToolkit(output_dir=output_dir)
    geo = GeoToolkit(output_dir=output_dir)
    pre = PreprocessingToolkit(output_dir=output_dir)
    analysis = AnalysisToolkit(output_dir=output_dir)
    classify = ClassificationToolkit(output_dir=output_dir)
    gee = GEEClient(project_id=config.gee_project_id, download_dir=output_dir)
    all_modules = [toolkit, geo, pre, analysis, classify, gee]

    def _set_output_dir(path: str) -> str:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        for m in all_modules:
            m.output_dir = str(p)
        return f"输出目录已切换至: {p}"

    registry = build_registry(toolkit, geo, pre, analysis, classify, gee)
    ensure_output_dir_tool(registry, _set_output_dir)
    return registry


# ── LLM client (shared, with timeout) ──────────────────

llm = create_llm_client(config.llm_provider, config.api_key, config.llm_model, config.api_base_url)


# ── API models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    images: list[dict] = []
    tools_used: list[str] = []


# ── Endpoints ───────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    conv = _get_session(session_id)

    # Create per-request toolkits
    registry = _create_toolkits(config.output_dir)
    agent = Agent(llm, registry, conv, config)

    # Record files before agent run
    out_dir = Path(config.output_dir)
    before_files = set()
    if out_dir.exists():
        before_files = {f.name for f in out_dir.glob("*.png")}

    reply = await asyncio.to_thread(agent.run, req.message)

    # Only include images newly generated by this request
    images = []
    if out_dir.exists():
        for f in sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name not in before_files:
                images.append({"url": f"/output/{f.name}", "label": f.stem})

    return ChatResponse(reply=reply, images=images, tools_used=registry.list_tools())


@app.get("/api/tools")
async def list_tools():
    # Return tool list from a fresh registry
    registry = _create_toolkits(config.output_dir)
    return {"tools": registry.list_tools()}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(_sessions),
        "max_sessions": MAX_SESSIONS,
    }


@app.post("/api/upload")
async def upload_file(file: bytes, filename: str):
    """Upload a file (GeoTIFF, image, CSV, GeoJSON) for use in analysis."""
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / filename
    target.write_bytes(file)
    return {"path": str(target), "size": len(file)}


# ── GEE Data Download API ──────────────────────────────

class GEESearchRequest(BaseModel):
    collection: str
    lon: float
    lat: float
    buffer_km: float = 20
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    cloud_cover: float = 10
    max_results: int = 50


class GEEDownloadRequest(BaseModel):
    collection: str
    image_ids: list[str]
    lon: float
    lat: float
    buffer_km: float = 20
    bands: str = ""
    scale: str = "10"
    cloud_mask: bool = False
    add_ndvi: bool = False


@app.post("/api/gee/search")
async def gee_search(req: GEESearchRequest):
    """Search GEE images, return structured JSON."""
    gee = GEEClient(project_id=config.gee_project_id, download_dir=config.output_dir)
    result = await asyncio.to_thread(
        gee.search_images_json,
        collection=req.collection,
        lon=req.lon,
        lat=req.lat,
        buffer_km=req.buffer_km,
        start_date=req.start_date,
        end_date=req.end_date,
        cloud_cover=req.cloud_cover,
        max_results=req.max_results,
    )
    if "error" in result:
        err = result["error"]
        if "serviceUsageConsumer" in err or "permission" in err.lower():
            err = (
                "GCP 项目权限不足:\n"
                "① 启用 API: https://console.developers.google.com/apis/api/earthengine.googleapis.com\n"
                "② 添加权限: https://console.cloud.google.com/iam-admin/iam → 找到你的邮箱 → 编辑 → 添加角色「Service Usage Consumer」→ 保存\n"
                "③ 重新认证: 在终端运行 earthengine authenticate --force\n"
                "完成后重试"
            )
        raise HTTPException(status_code=400, detail=err)
    return result


@app.post("/api/gee/download")
async def gee_download(req: GEEDownloadRequest):
    """Submit batch download tasks to Google Drive."""
    gee = GEEClient(project_id=config.gee_project_id, download_dir=config.output_dir)
    tasks = await asyncio.to_thread(
        gee.batch_download,
        collection=req.collection,
        image_ids=req.image_ids,
        lon=req.lon,
        lat=req.lat,
        buffer_km=req.buffer_km,
        bands=req.bands,
        scale=req.scale,
        cloud_mask=req.cloud_mask,
        add_ndvi=req.add_ndvi,
    )
    if tasks and "error" in tasks[0]:
        raise HTTPException(status_code=400, detail=tasks[0]["error"])
    return {
        "tasks": tasks,
        "message": f"已提交 {len(tasks)} 个导出任务到 Google Drive",
    }


@app.post("/api/gee/upload-shp")
async def gee_upload_shp(files: list[UploadFile] = File(...)):
    """Upload shapefile components (.shp/.dbf/.shx/.prj), return GeoJSON."""
    import tempfile
    import shutil

    # Save uploaded files to a temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="shp_"))
    saved_names = []

    for f in files:
        fname = f.filename or "unknown"
        target = tmp_dir / fname
        content = await f.read()
        target.write_bytes(content)
        saved_names.append(fname)

    # Find the .shp file
    shp_files = [n for n in saved_names if n.lower().endswith(".shp")]
    if not shp_files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="未找到 .shp 文件")

    shp_path = tmp_dir / shp_files[0]

    try:
        import fiona
        with fiona.open(shp_path) as src:
            features = []
            for feat in src:
                features.append({
                    "type": "Feature",
                    "geometry": feat["geometry"],
                    "properties": feat.get("properties", {}),
                })

            # Calculate bounds from all geometries
            bounds = src.bounds  # (minx, miny, maxx, maxy)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        return {
            "geojson": geojson,
            "bounds": list(bounds),
            "feature_count": len(features),
            "file": shp_files[0],
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="缺少 fiona 库，无法读取 SHP 文件")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SHP 读取失败: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/gee/tasks")
async def gee_tasks():
    """Query GEE export task status."""
    gee = GEEClient(project_id=config.gee_project_id, download_dir=config.output_dir)
    tasks = await asyncio.to_thread(gee.get_tasks)
    return {"tasks": tasks}


class GEEAuthRequest(BaseModel):
    project_id: str = ""


class GEECompositeRequest(BaseModel):
    collection: str
    lon: float
    lat: float
    buffer_km: float = 20
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    cloud_cover: float = 10
    period_days: int = 16
    bands: str = ""
    scale: str = "10"
    cloud_mask: bool = False
    add_ndvi: bool = False


@app.post("/api/gee/composite")
async def gee_composite(req: GEECompositeRequest):
    """Composite images by time period and export."""
    gee = GEEClient(project_id=config.gee_project_id, download_dir=config.output_dir)
    tasks = await asyncio.to_thread(
        gee.composite_by_period,
        collection=req.collection,
        lon=req.lon,
        lat=req.lat,
        buffer_km=req.buffer_km,
        start_date=req.start_date,
        end_date=req.end_date,
        cloud_cover=req.cloud_cover,
        period_days=req.period_days,
        bands=req.bands,
        scale=req.scale,
        cloud_mask=req.cloud_mask,
        add_ndvi=req.add_ndvi,
    )
    if tasks and "error" in tasks[0]:
        raise HTTPException(status_code=400, detail=tasks[0]["error"])
    return {
        "tasks": tasks,
        "message": f"已提交 {len(tasks)} 个合成导出任务",
    }


class GEETileRequest(BaseModel):
    collection: str
    image_id: str


@app.post("/api/gee/tile")
async def gee_tile(req: GEETileRequest):
    """Get tile URL for displaying an image on the map."""
    gee = GEEClient(project_id=config.gee_project_id, download_dir=config.output_dir)
    result = await asyncio.to_thread(
        gee.get_image_tile,
        collection=req.collection,
        image_id=req.image_id,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/gee/auth")
async def gee_auth(req: GEEAuthRequest):
    """Trigger GEE authentication."""
    gee = GEEClient(project_id=req.project_id, download_dir=config.output_dir)
    result = await asyncio.to_thread(gee.authenticate, req.project_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/gee/auth-status")
async def gee_auth_status():
    """Check GEE authentication status."""
    gee = GEEClient(project_id=config.gee_project_id, download_dir=config.output_dir)
    result = await asyncio.to_thread(gee.check_auth)
    return result


@app.get("/")
async def index():
    web_dir = Path(__file__).parent / "web"
    return FileResponse(web_dir / "index.html")


# ── Static file mounts ──────────────────────────────────

web_static = Path(__file__).parent / "web" / "static"
if web_static.exists():
    app.mount("/static", StaticFiles(directory=str(web_static)), name="static")

output_dir = Path(config.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

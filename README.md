# 遥感智能分析助手 (Yaogan Agent)

LLM + Function Calling 驱动的遥感影像智能分析系统。用户用自然语言描述需求，系统自主调用专业遥感工具完成分析。集成 Google Earth Engine，支持在线搜索、预览、批量下载和时间合成功能。

## 系统架构

```
用户输入（自然语言 / Web UI / API）
      │
      ▼
┌──────────────────┐
│  FastAPI Server  │  Web 服务 + API 认证 + 会话管理
│  (app.py)        │  GEE 搜索/下载/合成/瓦片 API
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Agent (主循环)  │  协调 LLM 与工具的交互
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────────────┐
│ LLM    │ │ ToolRegistry (27 个工具) │
│ Client │ ├──────────────────────────┤
│(Deep-  │ │ RemoteSensingToolkit     │ NDVI/SAR/分割/分类/波段运算
│Seek /  │ │ PreprocessingToolkit     │ 云掩膜/归一化/直方图匹配/重采样
│ OpenAI)│ │ AnalysisToolkit          │ 变化检测/PCA/光谱剖面/统计
└────────┘ │ ClassificationToolkit    │ RF/SVM 监督分类/精度评价
           │ GeoToolkit               │ 地理裁剪/重投影/分区统计/ROI
           │ GEEClient                │ GEE 搜索/下载/合成/瓦片
           └──────────────────────────┘

┌──────────────────────────────────────┐
│  Web UI                              │
│  ├─ 对话分析 (聊天 + 工具调用)       │
│  ├─ GEE 数据下载 (地图 + 搜索 + 下载)│
│  ├─ 输出影像                         │
│  └─ 设置 (GEE 认证)                  │
└──────────────────────────────────────┘
```

## 功能概览

### 对话分析

用自然语言与助手交互，自动调用遥感工具：

- "帮我计算这张影像的 NDVI"
- "对两期影像做变化检测"
- "用 KMeans 分类，分 5 类"
- "下载北京附近的 Sentinel-2 数据"

### GEE 数据下载

集成 Leaflet 地图的可视化数据获取面板：

- **地图选点**：点击地图选择经纬度
- **矩形/多边形框选**：用绘图工具选择区域
- **SHP 文件上传**：上传 .shp/.dbf/.shx/.prj 显示选区
- **影像搜索**：按数据集、日期、云量筛选
- **影像预览**：搜索结果在地图上显示覆盖范围，点击 🗺️ 加载影像瓦片
- **批量下载**：选择多景影像一键导出到 Google Drive
- **时间合成**：按 8 天/16 天/月/季度合成为中值影像后批量导出
- **去云处理**：下载时自动去除云和云影
- **添加 NDVI**：下载时自动计算并添加 NDVI 波段
- **取消搜索**：长时间搜索可随时取消
- **任务进度**：查看 GEE 导出任务状态，链接到 GEE 平台

### 支持的数据集

| 数据集 | 分辨率 | 时间范围 |
|--------|--------|----------|
| Sentinel-2 SR | 10m | 2017-至今 |
| Sentinel-2 TOA | 10m | 2015-至今 |
| Landsat 8 SR | 30m | 2013-至今 |
| Landsat 9 SR | 30m | 2021-至今 |
| MODIS NDVI | 250m | 2000-至今 |
| MODIS LST | 1km | 2000-至今 |
| SRTM DEM | 30m | 2000 |
| ALOS DEM | 30m | 2006-2011 |
| Dynamic World | 10m | 2015-至今 |

## 支持的分析工具 (27 个)

| 模块 | 工具 | 功能 |
|------|------|------|
| **遥感分析** | `describe_image` | 影像基本信息（形状、类型、值范围） |
| | `compute_ndvi` | NDVI 植被指数，自动识别红/近红外波段 |
| | `compute_sar_backscatter` | SAR 背散射统计 |
| | `segment_otsu` | Otsu 大津法阈值分割 |
| | `segment_sam` | Meta SAM 语义分割 |
| | `classify_landcover` | KMeans 无监督土地覆被分类 |
| | `band_math` | 波段算术运算（安全表达式求值） |
| | `compute_index` | 通用遥感指数 (NDVI/LSWI/NDWI/MNDWI/EVI/SAVI/NBR/VARI) |
| | `clip_raster` | 像素坐标裁剪 |
| | `extract_bands` | 波段提取 |
| **预处理** | `cloud_mask` | QA 波段云掩膜（Sentinel-2/Landsat） |
| | `normalize_image` | 影像归一化 (minmax/zscore) |
| | `histogram_match` | 直方图匹配（多时相分析） |
| | `resample_raster` | 重采样到目标分辨率 |
| **分析** | `change_detection` | 变化检测 (difference/ratio/cva) |
| | `pca_transform` | PCA 降维变换 |
| | `spectral_profile` | 指定像素光谱剖面 |
| | `compute_statistics` | 详细统计（含直方图特征） |
| **分类** | `train_classifier` | RF/SVM 监督分类器训练 |
| | `predict_classify` | 分类预测 |
| | `accuracy_assessment` | 精度评价 (OA/Kappa/F1/混淆矩阵) |
| **地理空间** | `clip_by_geo` | 经纬度裁剪 |
| | `reproject_raster` | 坐标重投影 |
| | `zonal_statistics` | 矢量分区统计 |
| | `extract_roi` | 按矢量边界提取 ROI |
| **GEE** | `search_gee_images` | GEE 影像搜索 |
| | `download_gee_image` | GEE 影像下载到 Google Drive |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入以下必要配置：

```env
# LLM API Key (必填)
RS_API_KEY=your_api_key_here

# GEE 项目 ID (使用 GEE 数据下载功能时必填)
RS_GEE_PROJECT_ID=your_gee_project_id
```

### 3. GEE 认证

首次使用 GEE 功能前需要认证：

```bash
earthengine authenticate --force
earthengine set_project your_gee_project_id
```

测试认证：

```bash
python -c "import ee; ee.Initialize(project='your_gee_project_id'); print('OK')"
```

### 4. 启动

**CLI 模式：**

```bash
python main.py
```

**Web 服务模式：**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000

**Docker 部署：**

```bash
docker-compose up -d
```

## Web UI 说明

### 对话分析

左侧默认面板。输入自然语言问题，助手自动调用工具分析。

支持的命令：
- `/clear` — 清除对话历史
- `/help` — 显示帮助

### GEE 数据下载

左侧导航点击「GEE 数据下载」进入。

**操作流程：**

1. 在地图上点击选择位置，或用工具栏绘制矩形/多边形选区
2. 选择数据集、时间范围、云量等参数
3. 点击「搜索影像」
4. 搜索结果列表中：
   - 点击 🗺️ 在地图上加载影像预览
   - 勾选要下载的影像
   - 点击「隐藏范围」去除覆盖框
5. 点击「批量下载」提交到 Google Drive
6. 或选择合成周期（如每 16 天），点击「按周期合成下载」

**SHP 文件上传：**

点击「上传 SHP」，选择 .shp/.dbf/.shx/.prj 文件，选区会显示在地图上。

### 设置

配置 GEE 认证信息。填入 GCP 项目 ID，点击「认证 GEE」。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RS_API_KEY` | LLM API Key (必填) | - |
| `RS_LLM_PROVIDER` | 提供商 (deepseek/openai) | deepseek |
| `RS_LLM_MODEL` | 模型名 | deepseek-chat |
| `RS_API_BASE_URL` | API 地址 | https://api.deepseek.com |
| `RS_MAX_TOOL_ROUNDS` | 最大工具调用轮数 | 5 |
| `RS_OUTPUT_DIR` | 输出目录 | ./output |
| `RS_SERVER_API_KEY` | Web API 认证密钥 (可选) | - |
| `RS_GEE_PROJECT_ID` | GEE GCP 项目 ID | - |
| `RS_SAM_MODEL` | SAM 模型类型 (vit_b/vit_l/vit_h) | vit_b |

## Web API

### POST /api/chat

对话接口，Agent 自动调用工具回答问题。

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我计算这张影像的 NDVI", "session_id": "test"}'
```

### POST /api/gee/search

搜索 GEE 影像。

```bash
curl -X POST http://localhost:8000/api/gee/search \
  -H "Content-Type: application/json" \
  -d '{"collection": "sentinel2", "lon": 116.40, "lat": 39.90, "start_date": "2024-01-01", "end_date": "2024-12-31"}'
```

### POST /api/gee/download

批量下载影像到 Google Drive。

```bash
curl -X POST http://localhost:8000/api/gee/download \
  -H "Content-Type: application/json" \
  -d '{"collection": "sentinel2", "image_ids": ["S2A_..."], "lon": 116.40, "lat": 39.90, "bands": "B4,B3,B2,B8"}'
```

### POST /api/gee/composite

按时间周期合成后批量导出。

```bash
curl -X POST http://localhost:8000/api/gee/composite \
  -H "Content-Type: application/json" \
  -d '{"collection": "modis_ndvi", "lon": 116.40, "lat": 39.90, "start_date": "2024-01-01", "end_date": "2024-12-31", "period_days": 16}'
```

### POST /api/gee/tile

获取影像瓦片 URL，用于在地图上显示。

```bash
curl -X POST http://localhost:8000/api/gee/tile \
  -H "Content-Type: application/json" \
  -d '{"collection": "sentinel2", "image_id": "S2A_MSIL2A_20240315..."}'
```

### POST /api/gee/upload-shp

上传 SHP 文件包，返回 GeoJSON。

```bash
curl -X POST http://localhost:8000/api/gee/upload-shp \
  -F "files=@region.shp" -F "files=@region.dbf" -F "files=@region.shx"
```

### GET /api/gee/tasks

查询 GEE 导出任务状态。

### GET /api/health

健康检查。

## 项目结构

```
Yaogan_Agent/
├── app.py                 # FastAPI 服务端
├── main.py                # CLI 入口 + 工具注册
├── .env                   # 环境变量 (不提交)
├── .env.example           # 环境变量模板
├── Dockerfile             # Docker 构建文件
├── docker-compose.yml     # Docker Compose
├── requirements.txt       # Python 依赖
├── pyproject.toml         # 项目元数据
├── src/
│   ├── config.py          # 配置管理
│   ├── agent.py           # Agent 核心循环
│   ├── conversation.py    # 对话管理
│   ├── llm_client.py      # LLM 客户端 (DeepSeek/OpenAI)
│   └── tools/
│       ├── _utils.py      # 工具函数 (路径安全检查)
│       ├── registry.py    # 工具注册中心
│       ├── remote_sensing.py    # 遥感分析工具
│       ├── preprocessing.py     # 预处理工具
│       ├── analysis.py          # 分析工具
│       ├── classification.py    # 分类工具
│       ├── geo_utils.py         # 地理空间工具
│       └── gee_client.py        # GEE 客户端
├── web/
│   ├── index.html         # Web UI 页面
│   └── static/
│       ├── style.css      # 样式
│       ├── app.js         # 对话逻辑
│       └── gee.js         # GEE 面板逻辑
└── tests/
    ├── test_config.py     # 配置测试
    ├── test_tools.py      # 工具测试
    ├── test_agent.py      # Agent 测试
    └── test_app.py        # API 测试
```

## 技术栈

- **LLM**: DeepSeek / OpenAI (Function Calling)
- **后端**: FastAPI + uvicorn + asyncio
- **前端**: 原生 HTML/CSS/JS + Leaflet 地图 + Leaflet Draw
- **遥感处理**: rasterio, numpy, opencv-python, scikit-learn
- **语义分割**: Meta SAM (Segment Anything Model)
- **数据获取**: Google Earth Engine API
- **分类**: Random Forest / SVM (scikit-learn)
- **部署**: Docker + Docker Compose

## 安全特性

- API Key 不硬编码，通过环境变量加载
- `band_math` 使用安全 AST 表达式求值（防止代码注入）
- 路径遍历防护（所有文件操作校验路径）
- 模型序列化使用 joblib（替代 pickle）
- Web API 支持密钥认证
- XSS 防护（前端使用 textContent）
- 对话裁剪保持 tool_call/tool_result 配对完整性
- LLM 调用 60 秒超时
- 会话 LRU 淘汰（最多 100 个，1 小时过期）

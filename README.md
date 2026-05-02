# 本地遥感大模型框架设计

这是一个面向遥感领域的**大模型 Agent 框架**，将大语言模型（LLM）与专业遥感图像处理工具相结合，实现对遥感影像的智能分析。用户只需用自然语言描述需求，系统就能自主调用工具完成计算并给出定量分析结论。

## 核心思想

**LLM + Function Calling + 领域工具 = 智能遥感分析助手**

- 利用 LLM 的语义理解和推理能力解析用户问题
- 通过 Function Calling（工具调用）机制让 LLM 自主决定调用哪些遥感工具
- 工具执行结果返回给 LLM，由其整合为最终分析报告

## 系统架构

```
用户输入（自然语言）
      │
      ▼
┌──────────────────┐
│  Conversation    │  对话历史管理
│  Manager         │  维护 system prompt + 多轮上下文
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Agent (主循环)  │  协调 LLM ↔ 工具
│                  │  最多 max_tool_rounds 轮交互
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────────┐
│ LLM    │ │ ToolRegistry         │
│ Client │ │ 工具注册中心          │
│(Deep-  │ │ 管理工具定义 & 调度   │
│ Seek / │ └──────────┬───────────┘
│ OpenAI)│            │
└────────┘            ▼
              ┌──────────────────────┐
              │ RemoteSensingToolkit │
              │ 遥感分析工具集        │
              │ • describe_image     │
              │ • compute_ndvi       │
              │ • compute_sar_backscatter │
              │ • segment_otsu       │
              │ • segment_sam        │
              │ • classify_landcover │
              └──────────────────────┘
```

## 工作流程

```
                    ┌──────────────┐
                    │  用户输入问题  │
                    │ "计算 NDVI"  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ Agent.run()  │
                    │ 添加用户消息  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ LLM 推理一轮  │
                    │ 返回工具调用  │
                    │ (tool_calls) │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ ToolRegistry │
                    │ .execute()   │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ 遥感工具执行   │
                    │ 返回文本结果  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ 结果注入对话  │
                    │ 继续下一轮    │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ LLM 整合结果  │
                    │ 生成最终回答  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ 输出分析报告  │
                    └──────────────┘
```

### NDVI 运算示例 — 详细执行流程

假设用户输入：**"请计算这张影像的 NDVI 并分析植被覆盖状况"**

#### 第 1 轮：LLM 决定调用 `describe_image`

```
LLM 收到消息:
  [system] 你是遥感专家，可用工具: describe_image, compute_ndvi, ...
  [user]   请计算这张影像的 NDVI 并分析植被覆盖状况

LLM 输出:
  → tool_call: describe_image(image_path="input.tif")

Agent 执行:
  → RemoteSensingToolkit.describe_image("input.tif")
  → 返回:"形状 (1000×800×4), 数据类型: uint16, 最小值: 0, 最大值: 10000, ..."

结果注入对话:
  [tool] 形状 (1000×800×4), 数据类型: uint16, 最小值: 0, 最大值: 10000, ...
```

#### 第 2 轮：LLM 决定调用 `compute_ndvi`

```
LLM 看到影像信息后:
  → 4 波段影像 (R,G,B,NIR)，可以计算 NDVI
  → tool_call: compute_ndvi(image_path="input.tif")

Agent 执行 compute_ndvi 内部逻辑:
  1. 读取影像 → 4 波段数组 (1000, 800, 4)
  2. 提取: NIR = 第 4 波段, Red = 第 3 波段
  3. NDVI = (NIR - Red) / (NIR + Red + 1e-6)
  4. 统计各区间占比: 水体(<0), 裸土(0~0.2), 稀疏(0.2~0.4), 茂密(>0.4)

  → 返回:"
  NDVI 统计:
    均值: 0.45
    标准差: 0.12
    水体/阴影 (NDVI<0): 5.2%
    裸土/建筑 (0~0.2): 15.3%
    稀疏植被 (0.2~0.4): 30.1%
    茂密植被 (>0.4): 49.4%
  "
```

#### 第 3 轮：LLM 输出最终分析报告

```
LLM 整合所有工具结果，生成自然语言回答:

  "该影像为 4 波段遥感影像（含近红外）。NDVI 均值为 0.45，表明整体植被覆盖较好。
  其中茂密植被（NDVI>0.4）占比 49.4%，稀疏植被占 30.1%，裸土/建筑占 15.3%，
  水体/阴影占 5.2%。该区域以植被覆盖为主，生态环境良好。"
```

## 模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| **Config** | [src/config.py](src/config.py) | 从环境变量加载 LLM 配置（API Key、模型名、提供商） |
| **LLMClient** | [src/llm_client.py](src/llm_client.py) | 统一 LLM API 接口，支持 DeepSeek / OpenAI（兼容 OpenAI 格式都可接入） |
| **ConversationManager** | [src/conversation.py](src/conversation.py) | 维护 system prompt + 多轮对话上下文，支持 tool_call/tool_result 注入 |
| **Agent** | [src/agent.py](src/agent.py) | 核心协调循环：LLM 推理 → 工具调用 → 结果反馈 → 最终回答 |
| **ToolRegistry** | [src/tools/registry.py](src/tools/registry.py) | 工具注册、OpenAI 格式 schema 生成、按名调度执行 |
| **RemoteSensingToolkit** | [src/tools/remote_sensing.py](src/tools/remote_sensing.py) | 遥感分析算法实现：NDVI、SAR背散射、Otsu分割、SAM分割、KMeans分类 |

## 支持的遥感工具

| 工具 | 功能 |
|------|------|
| `describe_image` | 读取影像基本信息（形状、数据类型、值范围、均值、标准差） |
| `compute_ndvi` | 计算归一化植被指数，分区间统计（水体/裸土/稀疏/茂密） |
| `compute_sar_backscatter` | SAR 背散射统计（均值/标准差/最大最小值） |
| `segment_otsu` | Otsu 大津法阈值分割 |
| `segment_sam` | Meta SAM 模型语义分割（需额外安装） |
| `classify_landcover` | KMeans 无监督土地覆被分类 |

## 本地部署建议

1. **LLM 选择**: 本框架通过 API 调用云端模型（DeepSeek / OpenAI），也支持替换为本地模型（需自行实现 LLMClient 子类）
2. **计算资源**: 遥感工具自身计算量小（CPU 即可）；若使用 SAM 分割需要 GPU
3. **数据处理依赖**: `rasterio`、`numpy`、`opencv-python`

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   ```bash
   set RS_API_KEY=your_api_key_here
   set RS_LLM_PROVIDER=deepseek    # 或 openai
   set RS_LLM_MODEL=deepseek-chat  # 或 gpt-4o
   ```

3. 运行示例：
   ```bash
   python main.py --image path/to/image.tif --question "请帮我计算 NDVI 并分析植被状况。" --model-path ./models/your-llm
   ```

## 后续扩展方向

- 接入深度分割模型（例如 SegFormer、SAM）
- 增加高光谱解谱与分类模块
- 设计本地 API 服务（FastAPI）供 GUI 或前端调用
- 支持更多 LLM 提供商（本地 ollama、claude 等）
- 添加流式输出支持，提升交互体验
